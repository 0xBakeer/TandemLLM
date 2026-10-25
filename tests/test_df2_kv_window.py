"""SPD-177: the draft's context K/V committed for a window of rows (engine/drafters/dflash2.py `kv_window_for`),
CPU only, with the pool's bookkeeping fallback (no driver: pages counted, nothing mapped).

  * below the threshold (the window plus one page of rows) the window buffer commits exactly the pages a
    plain `PagedBuffer` commits, at every row count: the path of today;
  * past it, the pages wholly below `rows - window - page` are given back (their addresses alias one
    page) and the account follows; the rows a draft reads stay committed;
  * a resume or a restore at an earlier row (`_fit_window`) lowers the floor and the window is mapped
    again; `trim` below the floor lowers it too; a snapshot past the threshold holds only the rows
    from the floor and restores into the same rows;
  * the window is the family's choice: a DFlash2 drafter is built without it unless its family asks
    (only the 27B's builds do), and an explicit QWEN38_DF2_KV_WINDOW wins either way.

Run: python tests/test_df2_kv_window.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine import kvpool  # noqa: E402
from engine.drafters import dflash2 as D  # noqa: E402

MiB = kvpool.MiB
SHAPE = (5, 8, 65536, 128)            # the draft's layers, kv heads, rows, head dim: 256 B a row


def _pool():
    return kvpool.KVPool(hard_floor=0, margin_rows=0, meminfo=lambda: 1 << 50)


def _pair(pool):
    W = D._window_buffer(kvpool)
    w = W(SHAPE, 2, torch.bfloat16, "cpu", owner="w", pool=pool, window=2048)
    p = kvpool.PagedBuffer(SHAPE, 2, torch.bfloat16, "cpu", owner="p", pool=pool)
    return w, p


def test_below_the_threshold_it_is_the_plain_buffer():
    pool = _pool()
    w, p = _pair(pool)
    assert w.page_rows == 8192 and w.keep == 2048 + 8192
    for rows in (1, 100, 2048, 8191, 8192, 8193, 12000, 16384, 18431):
        assert w.need_bytes(rows) == p.need_bytes(rows), rows
        assert w.ensure(rows) == p.ensure(rows), rows
        assert w.floor == 0 and set(w.handles) == set(p.handles), rows
    assert pool.owner_bytes("w") == pool.owner_bytes("p")


def test_past_it_the_old_pages_go_and_the_window_stays():
    pool = _pool()
    w, _ = _pair(pool)
    w.ensure(65536)
    assert w.floor == (65536 - 10240) // 8192 * 8192 == 49152
    held = set(w.handles)
    lo_page = 49152 * 256 // (2 * MiB)              # the first page of region 0 at the floor
    assert lo_page in held and lo_page - 1 not in held and (lo_page - 1) in w.alias
    # the rows a draft reads: the window behind the newest row is committed in every region
    need = set(w._pages_for(65536 - 2048, 65536))
    assert need <= held
    assert pool.owner_bytes("w") == len(held) * 2 * MiB == 2 * 40 * 2 * MiB


def test_a_resume_earlier_maps_its_window_again():
    pool = _pool()
    w, _ = _pair(pool)
    w.ensure(65536)
    d = types.SimpleNamespace(_ckb=w, _cvb=None, _cs=None, _partial=False)
    D.DFlash2Drafter._fit_window(d, 20000)
    assert w.floor == (20000 - 10240) // 8192 * 8192 == 8192
    w.ensure(20000)
    assert set(w._pages_for(20000 - 2048, 20000)) <= set(w.handles)
    assert not (set(w.handles) & w.alias)
    w.trim(1000)
    assert w.floor == 0 and not w.alias


def _drafter(pool):
    W = D._window_buffer(kvpool)
    d = D.DFlash2Drafter.__new__(D.DFlash2Drafter)
    d._psync, d._partial, d.module, d.ctx_len = [], False, object(), 0
    d._ckb = W(SHAPE, 2, torch.bfloat16, "cpu", owner="w", pool=pool, window=2048)
    d._cvb = W(SHAPE, 2, torch.bfloat16, "cpu", owner="w", pool=pool, window=2048)
    d._ck, d._cv = d._ckb.tensor, d._cvb.tensor
    d._cs = torch.zeros(SHAPE[2], dtype=torch.uint8)
    return d


def _write(d, a, b, val):
    for buf in (d._ckb, d._cvb):
        buf.ensure(b)
    d._ck[:, :, a:b] = val
    d._cv[:, :, a:b] = val
    d._cs[a:b] = 1
    d.ctx_len = b


def test_snapshot_holds_the_rows_from_the_floor():
    d = _drafter(_pool())
    _write(d, 0, 40000, 1.5)
    snap = d.state_snapshot()
    assert snap[0] == "dflash2w" and snap[2] == d._ckb.floor == 24576 and snap[3].shape[2] == 40000 - 24576
    d._ck.zero_()
    d._cs.zero_()
    d.state_restore(snap)
    assert d.ctx_len == 40000 and bool((d._ck[:, :, 30000:40000] == 1.5).all())
    assert bool((d._cs[24576:40000] == 1).all()) and not bool(d._cs[:24576].any())


def test_views_below_the_threshold_are_todays_rows():
    d = _drafter(_pool())
    _write(d, 0, 12000, 2.0)
    views = d.kv_views()
    assert [ax for _, ax in views] == [2, 2, 0] and views[0][0] is d._ck and views[1][0] is d._cv
    assert d._ckb.floor == 0 and bool((d._cs[:12000] == 1).all())


def test_an_anchor_copied_from_below_the_floor_resumes_as_not_real():
    """A cache copies rows [0, 8K) through kv_views while a 40K session holds the floor at 24,576; a
    second request resumes from that copy: its stamps say the rows are not real, so the draft's
    window starts at the first real row (none here) instead of reading garbage."""
    d = _drafter(_pool())
    _write(d, 0, 40000, 1.0)
    views = d.kv_views()
    saved = [t.narrow(ax, 0, 8192).clone() for t, ax in views]        # the anchor's copy
    assert not bool(saved[2].any()), "the copy carries 'not real' for rows below the floor"
    hi = [t.narrow(ax, 0, 30000).clone() for t, ax in views]          # an anchor above the floor
    assert bool((hi[2][24576:30000] == 1).all()) and not bool(hi[2][:24576].any())
    # a second request: reset, resume at 8K from the copy (the drafter is told before and after)
    d.reset()
    for buf in (d._ckb, d._cvb):
        buf.trim(8192)
    d.state_resume(8192)
    for (t, ax), b in zip(d.kv_views(), saved):
        t.narrow(ax, 0, 8192).copy_(b)
    d.state_resume(8192)
    assert d._ckb.floor == 0 and not bool(d._cs[:8192].any()), "no row of the window counts as real"
    # and from the copy above the floor: the window is real again
    d.reset()
    for buf in (d._ckb, d._cvb):
        buf.ensure(30000)                              # the engine's paged set commits the rows
    d.state_resume(30000)
    for (t, ax), b in zip(d.kv_views(), hi):
        t.narrow(ax, 0, 30000).copy_(b)
    d.state_resume(30000)
    assert d._ckb.floor == (30000 - 10240) // 8192 * 8192 == 16384
    assert bool((d._cs[30000 - 2048:30000] == 1).all()), "the window rows came back real"
    assert not bool(d._cs[16384:24576].any()), "rows the first session had given back stay not real"



def test_a_resume_below_the_window_through_the_paged_set_commits_its_rows():
    """SPD-211: the server's resume goes through the paged set (`kvpool.ensure_for`), whose fast path
    compares the set's row count. A resume at 28,672 after a 37,387-token session moves the floor from
    24,576 down to 16,384; the set's count has to come down with it, or the pages of rows 16,384..24,575
    stay unmapped and the next sync writes into them (an illegal memory access on the board)."""
    out = D.check_window("cpu")
    assert "floor 24,576" in out[0] and "floor 16,384" in out[0], out
    assert out[1].startswith("rows 16,384..28,928 committed"), out


def test_a_trim_below_the_floor_keeps_the_set_count_honest():
    pool = kvpool.KVPool(hard_floor=0, margin_rows=0, meminfo=lambda: 1 << 50)
    ps = kvpool.PagedSet("s", pool=pool)
    W = D._window_buffer(kvpool)
    w = ps.add(W(SHAPE, 2, torch.bfloat16, "cpu", pool=pool, window=2048))
    ps.ensure(40000)
    assert w.floor == 24576 and ps.rows == 40960
    w.set_floor(8192)                                   # a resume far back, outside the set's own calls
    assert ps.rows == w.rows == 8192, (ps.rows, w.rows)
    ps.ensure(20000)
    assert not [p for p in w._pages_for(w.floor, 20000) if p not in w.handles]
    ps.trim(5000)
    assert w.floor == 0 and ps.rows == w.rows

def test_cover_maps_the_pages_a_gather_reads_and_nothing_else():
    """SPD-223: a draft graph's gather reads [pos0 - window + 1, + span). A floor lowered without its ensure (the
    SPD-211 shape: pages uncovered stay unmapped until the next ensure) is mapped by `cover`, below the floor onto
    the shared page, at or above it fresh; a second cover maps nothing, and rows already mapped cost nothing."""
    pool = _pool()
    w, _ = _pair(pool)
    w.ensure(65536)
    assert w.cover(65536 - 2048, 65536) == 0                    # the window behind the newest row: mapped
    w.set_floor(16384)                                          # lowered: pages 16384..49151 unmapped
    lo, hi = 20000, 20000 + 2048
    miss = [q for q in w._pages_for(lo, hi) if q not in w.handles and q not in w.alias]
    assert miss, "the lowered floor left the gathered rows unmapped"
    assert w.cover(lo, hi) == len(miss)
    assert all(q in w.handles or q in w.alias for q in w._pages_for(lo, hi))
    assert w.cover(lo, hi) == 0
    return f"{len(miss)} unmapped page(s) under a lowered floor mapped once, then nothing"


def test_the_window_is_the_27bs_choice_and_off_for_any_other_build():
    import ast
    import inspect
    from engine import settings
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert settings.KNOBS["DF2_KV_WINDOW"][0] == "0", "the shared default stays off"
    assert inspect.signature(D.DFlash2Drafter.__init__).parameters["kv_window"].default is False
    assert "kv_window" not in inspect.signature(__import__("engine.drafters.dspark", fromlist=["x"])
                                                .DSparkDrafter.__init__).parameters
    env = os.environ.pop("QWEN38_DF2_KV_WINDOW", None)
    try:
        assert D.kv_window_for(False) is False and D.kv_window_for(True) is True
        for v, want in (("0", False), ("1", True)):
            os.environ["QWEN38_DF2_KV_WINDOW"] = v
            assert D.kv_window_for(False) is want and D.kv_window_for(True) is want, v
    finally:
        os.environ.pop("QWEN38_DF2_KV_WINDOW", None)
        if env is not None:
            os.environ["QWEN38_DF2_KV_WINDOW"] = env
    # who asks: every `kv_window=True` in the engine is a build in the 27B's package
    asks = []
    for dp, _, fs in os.walk(os.path.join(root, "engine")):
        for f in fs:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dp, f)
            for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == "kv_window" and not (isinstance(kw.value, ast.Constant)
                                                          and kw.value.value is False):
                            asks.append(os.path.relpath(path, root))
    assert asks and all(a.startswith(os.path.join("engine", "models", "qwen3_5") + os.sep)
                        for a in asks), asks
    own = os.path.join("engine", "models", "qwen3_5") + os.sep
    for dp, _, fs in os.walk(os.path.join(root, "engine", "models")):
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), root)
            if f.endswith(".py") and not rel.startswith(own) and "__pycache__" not in rel:
                assert "kv_window" not in open(os.path.join(dp, f), encoding="utf-8").read(), rel


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"{n} passed")
