"""ENG-157: the cache tiers know no model -- they keep a sequence through the engine's adapter.

A third architecture made up here (`Toy`: two recurrent sections of different dtypes and three
position-indexed views of different shapes, dtypes and token axes) runs the tiers with no edit in
engine/cache.py, disk_cache.py or spill.py:

  * a snapshot (`capture` / `restore`) round-trips every section and every view;
  * the resident prefix anchors on the grid and resumes the next turn there, equal to a cold run;
  * the disk tier writes its anchor (format 2, the state's section names) and chunks, and after a
    restart restores them into a fresh engine, equal to a cold run;
  * the session spill saves the conversation and reads it back in place, equal to before;
and the 27B's anchor files written before the sections were named (format 1, "S" and "conv") are
still read by the new code, bit for bit.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache, disk_cache, spill  # noqa: E402
from engine.models.adapter import CostModel  # noqa: E402
from engine.models.state import SectionState  # noqa: E402
from test_cache import tokens  # noqa: E402

CH = 16
BIG = 1 << 30


class _ToyState(SectionState):
    def __init__(self, eng):
        self.eng = eng

    def sections(self):
        return {"toy.sum": self.eng.s1, "toy.count": self.eng.s2}

    def views(self):
        e = self.eng
        return [("a", e.va, 1), ("b", e.vb, 0), ("c", e.vc, 2)]


class Toy:
    """A recurrent model of no known family: each token folds into a fp32 sum and an int64 count;
    its rows (three views) depend on the token and the state before it, so a row and the state
    after a position depend only on the tokens up to it."""

    def __init__(self, max_len: int = 512):
        self.s1 = torch.zeros(3, 5)
        self.s2 = torch.zeros(2, dtype=torch.long)
        self.va = torch.zeros(2, max_len, 4, dtype=torch.bfloat16)
        self.vb = torch.zeros(max_len, 3)
        self.vc = torch.zeros(1, 2, max_len, dtype=torch.float16)
        self.kv = types.SimpleNamespace(length=0, max_len=max_len, pages=None)
        self.cfg = types.SimpleNamespace(vocab_size=128)
        self.hidden_post_norm = None

    # the adapter's cold verbs
    def cache_state(self):
        return _ToyState(self)

    def cost(self) -> CostModel:
        sec = sum(t.numel() * t.element_size() for t in (self.s1, self.s2))
        row = sum(t.numel() // t.shape[ax] * t.element_size() for _, t, ax in self.cache_state().views())
        return CostModel(section_bytes=sec, view_bytes_per_token=row, parts={"recurrent": sec})

    def settle(self):
        pass

    def discard_pending(self):
        pass

    def trim_pending(self, n):
        return n == self.kv.length

    def resume_at(self, n):
        self.kv.length = n

    def reset(self):
        self.s1.zero_()
        self.s2.zero_()
        self.kv.length = 0

    def window_ok(self, i):
        return False

    @torch.no_grad()
    def forward(self, ids, start, last_only=True):
        assert start == self.kv.length
        ids = ids.view(-1).tolist()
        rows = []
        for j, t in enumerate(ids):
            p = start + j
            self.va[:, p] = (self.s1[:2, :4] * 0.25 + t).to(torch.bfloat16)
            self.vb[p] = self.s1[0, :3] - t
            self.vc[0, :, p] = torch.tensor([t, int(self.s2[0]) % 97], dtype=torch.float16)
            self.s1.mul_(0.5).add_(torch.arange(15.0).view(3, 5) * (t % 7))
            self.s2 += torch.tensor([1, t])
            rows.append(self.s1.flatten()[:8].clone())
        self.kv.length = start + len(ids)
        self.hidden_post_norm = torch.stack(rows).unsqueeze(0)
        lg = torch.zeros(1, 1, self.cfg.vocab_size)
        lg[0, 0, int(self.s2[1]) % self.cfg.vocab_size] = 1.0 + float(self.s1.sum())
        return lg


def _cold(ids):
    eng = Toy()
    lg, _, _ = cache.prefill(eng, None, ids, "cpu", chunk=CH)
    return eng, lg


def _same(eng, ref, n, lg=None, ref_lg=None):
    assert eng.kv.length == ref.kv.length == n
    assert torch.equal(eng.s1, ref.s1) and torch.equal(eng.s2, ref.s2)
    for (_, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
        assert torch.equal(a.narrow(ax, 0, n), b.narrow(ax, 0, n))
    if lg is not None:
        assert torch.equal(lg, ref_lg)


def test_a_snapshot_round_trips_every_section_and_view():
    eng = Toy()
    ids = tokens(70, seed=1)
    cache.prefill(eng, None, ids, "cpu", chunk=CH)
    snap = cache.capture(eng)
    ref = Toy()
    cache.prefill(ref, None, ids, "cpu", chunk=CH)
    eng.forward(torch.tensor(tokens(9, seed=2)), start=70)
    cache.restore(eng, snap)
    _same(eng, ref, 70)
    assert snap.nbytes == eng.cost().section_bytes + 70 * eng.cost().view_bytes_per_token
    return f"2 sections, 3 views, {snap.nbytes} bytes at 70 tokens"


def test_the_resident_prefix_resumes_a_new_model_exactly():
    eng, res = Toy(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG)
    turn1 = tokens(100, seed=3)
    cache.prefill(eng, None, turn1, "cpu", chunk=CH, resident=res)
    assert sorted(res.anchors)[-1] == 96 and len(res.anchors[96]) == 2
    turn2 = turn1 + tokens(30, seed=4)
    lg, reused, _ = cache.prefill(eng, None, turn2, "cpu", chunk=CH, resident=res)
    assert reused == 96, reused
    ref, ref_lg = _cold(turn2)
    _same(eng, ref, len(turn2), lg, ref_lg)
    return "anchors of 2 sections on the grid; turn 2 resumed at 96, equal to a cold run"


def _tier(d):
    return disk_cache.DiskTier(d, chunk=CH, fingerprint="toy", settings={"admit_after": 1, "budget_gb": 1},
                               free_bytes=lambda p: 10 ** 15, head_anchors=2, min_tokens=CH).open()


def test_the_disk_tier_round_trips_a_new_model_after_a_restart():
    ids = tokens(150, seed=5)
    with tempfile.TemporaryDirectory() as d:
        t = _tier(d)
        cache.prefill(Toy(), None, ids, "cpu", chunk=CH, disk=t)
        assert t.flush()
        t.close()
        anc = os.listdir(os.path.join(d, "anchors"))
        assert len(anc) == 1
        meta = t._read_header(os.path.join(d, "anchors", anc[0]), "anchor")
        assert meta["format"] == 2 and [s["name"] for s in meta["sections"]] == ["toy.sum", "toy.count", "chain"]
        t2 = _tier(d)                                  # a restart
        prompt = ids + tokens(20, seed=6)
        eng = Toy()
        info = {}
        lg, reused, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, disk=t2, info=info)
        assert (info.get("kind"), reused) == ("disk", 144), (info, reused)
        ref, ref_lg = _cold(prompt)
        _same(eng, ref, len(prompt), lg, ref_lg)
        t2.close()
    return "anchor format 2 with the state's names; restored after a restart, equal to a cold run"


def test_the_session_spill_round_trips_a_new_model():
    d = tempfile.mkdtemp(prefix="toy-spill-")
    try:
        t = _tier(d)
        sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="toy", tier=t,
                                settings={"budget_gb": 1.0, "min_tokens": 64, "drain_min_tokens": CH, "codec": "off"},
                                free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
        eng, res = Toy(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
        ids = tokens(130, seed=7)
        cache.prefill(eng, None, ids, "cpu", chunk=CH, resident=res)
        ref = Toy()
        cache.prefill(ref, None, ids, "cpu", chunk=CH)
        anchors0 = {b: tuple(x.clone() for x in a) for b, a in res.anchors.items()}
        info = sp.save_resident(res, eng, None, "evict")
        assert info is not None, sp.stats
        res.clear()
        for _, v, _ in eng.cache_state().views():
            v.zero_()
        got = sp.restore(res, eng, None, sp.entries[info["eid"]])
        assert got is not None, sp.stats
        for (_, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
            assert torch.equal(a.narrow(ax, 0, 130), b.narrow(ax, 0, 130))
        for b in sorted(anchors0)[-4:]:
            assert all(torch.equal(x, y) for x, y in zip(res.anchors[b], anchors0[b]))
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return "rows of 3 views and the newest anchors back in place, bit for bit"


def test_a_27b_anchor_file_of_format_1_is_still_read():
    from test_cache import fresh
    from test_disk_cache import Pos, exact, run, tier
    ids = tokens(150, seed=8)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, settings={"admit_after": 1})
        run(fresh(), ids, t, Pos())
        t.close()
        p = os.path.join(d, "anchors", os.listdir(os.path.join(d, "anchors"))[0])
        meta = t._read_header(p, "anchor")
        meta.pop("tokens", None)
        # what the 27B wrote before ENG-157: format 1, its two sections named "S" and "conv"
        meta["format"] = 1
        for s in meta["sections"]:
            s["name"] = {"gdn.S": "S", "gdn.conv": "conv"}.get(s["name"], s["name"])
        raw = json.dumps(meta, separators=(",", ":")).encode()
        with open(p, "r+b") as f:
            f.seek(8)
            f.write(len(raw).to_bytes(4, "little") + raw + b"\0" * (disk_cache.ALIGN - 12 - len(raw)))
        t2 = tier(d, settings={"admit_after": 1})
        assert t2.report()["entries"] == 1
        prompt = ids + tokens(20, seed=9)
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, prompt, t2, dr)
        assert (kind, reused) == ("disk", 144), (kind, reused)
        exact(eng, lg, prompt, dr)
        t2.close()
    return "a format-1 anchor (S, conv) restored by the new code, equal to a cold prefill"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<62} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<62} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
