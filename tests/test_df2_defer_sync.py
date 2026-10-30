"""SPD-161, QWEN38_DF2_DEFER_SYNC: a parallel slot's drafter sync waits for the packed draft, on the CPU.

A stub module whose context projection and K/V are row-wise functions stands in for the checkpoint:
  1. `flush_syncs` over several drafters (one checkpoint, several pending syncs each) writes every
     drafter's cache exactly as syncing each block at once did;
  2. a deferred sync sets `ctx_len` at once and leaves the cache untouched until a reader flushes it:
     the snapshot, `kv_views`, a resume and the lone graph run each see the rows; `reset` drops them;
  3. without the packed draft hook (no side slots) or past DEFER_SYNC_ROWS the sync runs at once.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine.drafters import dflash2 as D  # noqa: E402

L, HKV, HD, MAXLEN, IN = 2, 2, 4, 128, 6


class StubModule:
    def __init__(self, w):
        self.w = w
        self.dtype = torch.float32

    def project_context(self, x):
        return x * 2.0 + 1.0

    def context_kv(self, h, positions):
        out = []
        for i in range(L):
            k = (h[:, :HKV * HD] + i + positions[:, None].float()).view(-1, HKV, HD).transpose(0, 1)
            v = (h[:, :HKV * HD] - i).view(-1, HKV, HD).transpose(0, 1)
            out.append((k, v))
        return out


def drafter(w):
    d = D.DFlash2Drafter.__new__(D.DFlash2Drafter)
    d.module = StubModule(w)
    d._build = lambda: d.module
    d.eng = types.SimpleNamespace(device="cpu")
    d._ck = torch.zeros(L, HKV, MAXLEN, HD)
    d._cv = torch.zeros(L, HKV, MAXLEN, HD)
    d._psync, d.ctx_len, d._tap_i, d._tap_rows = [], 0, 0, []
    d._want = {0: 0}
    d._cs = d._ckb = d._cvb = None             # SPD-177's window off: no row stamps
    d.kv_window = False
    return d


def sync(d, n, first, seed):
    g = torch.Generator().manual_seed(seed)
    d._tap_rows = [torch.randn(96, IN * 2, generator=g)]
    D.DFlash2Drafter.sync(d, list(range(n)), None, first)


def _with(defer, meet):
    keep = (D.DEFER_SYNC, D.DRAFT_MEET)
    D.DEFER_SYNC, D.DRAFT_MEET = defer, meet
    return keep


def test_flush_syncs_writes_what_syncing_at_once_writes():
    w = {}
    plan = [[(3, 0), (2, 3)], [(4, 10)], [(1, 5), (5, 6)]]
    now = [drafter(w) for _ in plan]
    later = [drafter(w) for _ in plan]
    keep = _with(False, None)
    try:
        for e, (d, steps) in enumerate(zip(now, plan)):
            for j, (n, first) in enumerate(steps):
                sync(d, n, first, 100 * e + j)
    finally:
        D.DEFER_SYNC, D.DRAFT_MEET = keep
    keep = _with(True, object())
    try:
        for e, (d, steps) in enumerate(zip(later, plan)):
            for j, (n, first) in enumerate(steps):
                sync(d, n, first, 100 * e + j)
            assert d.ctx_len == steps[-1][1] + steps[-1][0], "ctx_len is set at once"
            assert not d._ck.any(), "nothing written before the flush"
        D.flush_syncs(later)
    finally:
        D.DEFER_SYNC, D.DRAFT_MEET = keep
    for a, b in zip(now, later):
        assert torch.equal(a._ck, b._ck) and torch.equal(a._cv, b._cv) and a.ctx_len == b.ctx_len
        assert b._psync == []
    return "3 drafters, 5 pending syncs, one flush: the caches equal syncing each block at once"


def test_every_reader_flushes_first_and_reset_drops():
    w = {}
    keep = _with(True, object())
    try:
        for reader in ("snapshot", "views", "resume", "graph"):
            d, ref = drafter(w), drafter(w)
            sync(d, 3, 0, 7)
            D.DEFER_SYNC = False
            sync(ref, 3, 0, 7)
            D.DEFER_SYNC = True
            if reader == "snapshot":
                snap = D.DFlash2Drafter.state_snapshot(d)
                assert torch.equal(snap[2], ref._ck[:, :, :3])
            elif reader == "views":
                D.DFlash2Drafter.kv_views(d)
            elif reader == "resume":
                D.DFlash2Drafter.state_resume(d, 3)
            else:
                # SPD-223: a graph is reused while every tensor it baked is in place (`DraftGraph.pointers`;
                # this fake drafter has no engine, so its pointers are pinned)
                from engine.drafters import draft_graph as G
                keep_ptrs = G.DraftGraph.pointers
                G.DraftGraph.pointers = staticmethod(lambda _d: ("fixed",))
                try:
                    d._graph = types.SimpleNamespace(ptrs=("fixed",), run=lambda a, p: "ran")
                    assert D.DFlash2Drafter.graph_run(d, 1, 3) == "ran"
                finally:
                    G.DraftGraph.pointers = keep_ptrs
            assert torch.equal(d._ck, ref._ck), reader
        d = drafter(w)
        sync(d, 3, 0, 7)
        D.DFlash2Drafter.reset(d)
        assert d._psync == [] and d.ctx_len == 0
    finally:
        D.DEFER_SYNC, D.DRAFT_MEET = keep
    return "snapshot, kv_views, resume and the graph run see the rows; reset drops them"


def test_no_packed_draft_or_a_long_sync_runs_at_once():
    w = {}
    keep = _with(True, None)
    try:
        d = drafter(w)
        sync(d, 3, 0, 1)
        assert d._psync == [] and d._ck.any(), "no side slots: at once"
        D.DRAFT_MEET = object()
        d = drafter(w)
        d._tap_rows = [torch.randn(D.DEFER_SYNC_ROWS + 1, IN * 2)]
        D.DFlash2Drafter.sync(d, list(range(D.DEFER_SYNC_ROWS + 1)), None, 0)
        assert d._psync == [] and d._ck.any(), "a prompt's sync: at once"
    finally:
        D.DEFER_SYNC, D.DRAFT_MEET = keep
    return "no packed draft hook, or more than DEFER_SYNC_ROWS rows: the sync runs at once"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:64s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
