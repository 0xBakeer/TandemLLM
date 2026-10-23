"""SPD-187, QWEN38_ROUND_LOG: the round logger's lines on the CPU (no events: a pass's ms is null there).

A joint round, a lone round and a slot's block: each one JSON line with the fields the planner reads, the
slot's per-node q from its router arm before it learns, ranks by drafter score, depths from the tree,
the accepted path; a drafter with no `_node_q` logs q as null.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tree import DraftTree  # noqa: E402
from server import roundlog  # noqa: E402


class Req:
    def __init__(self, slot, rows, start):
        self.slot, self.rows, self.start = slot, rows, start


class Arm:
    def _node_q(self, i, tree, depth, rank):
        return 0.5 ** depth


class D2:
    _cand_host = [[11, 12], [13, 14]]
    _logp_host = [[[-0.1, -2.3], [-0.7, -0.7]], [[-0.2, -1.9], [-1.5, -0.3]]]


class LenRouter:
    last_key = "l"
    small, large = None, Arm()
    large.mtp = D2()


def test_round_slot_and_draft_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rounds.jsonl")
        log = roundlog.RoundLog(path)
        rid = log.round([Req(0, 8, 300), Req(3, 6, 512)], None, cap=8, pre_rows=[12, 6], graphed=True)
        rid2 = log.round([Req(1, 4, 100)], None)
        tree = DraftTree([5, 6, 7, 8], [-1, 0, 1, 0], scores=[1.0, 0.9, 0.4, 0.7], source=["root", "mtp", "mtp", "sib"])
        import torch
        lg = torch.randn(4, 100)
        top = roundlog.RoundLog.topk_async(lg)
        log.slot(rid, 3, 2, 512, tree, [0, 1, 2], 6, LenRouter(), top=top, drawn=[6, 7, 9])
        log.slot(rid2, 1, 1, 100, tree, [0], 4, object())
        log.draft([Req(0, 16, 300), Req(3, 16, 512)], None)
        log.fh.close()
        lines = [json.loads(x) for x in open(path)]
    import torch
    r, r2, s, s2, dr = lines
    assert (r["k"], r["rid"], r["E"], r["slots"], r["rows"], r["pre_rows"], r["cap"], r["ctx"], r["graphed"]) == \
        ("round", 1, 2, [0, 3], [8, 6], [12, 6], 8, [300, 512], True)
    assert r["pass_ms"] is None and r2["E"] == 1 and r2["rid"] == 2 and r2["period_ms"] is not None
    assert s["k"] == "slot" and s["rid"] == 1 and s["slot"] == 3 and s["n"] == 4 and s["accepted"] == 2
    assert s["depth"] == [0, 1, 2, 1] and s["rank"] == [0, 1, 3, 2] and s["pre_n"] == 6
    assert s["q"] == [None, 0.5, 0.25, 0.5], s["q"]
    assert s2["q"] is None, "a drafter without a calibrated q"
    ref = torch.log_softmax(lg, -1).topk(64, dim=-1)
    assert s["top_ids"] == ref.indices.tolist() and len(s["top_lp"][0]) == 64
    assert max(abs(a - b) for a, b in zip(s["top_lp"][2], ref.values[2].tolist())) < 1e-3
    assert s["drawn"] == [6, 7, 9] and s["cand"] == [[11, 12], [13, 14]]
    assert s["lat_lp"] == [[-0.1, -2.3], [-0.2, -1.9]], "the rows along the lattice's greedy walk"
    assert s2["cand"] is None and s2["top_ids"] if "top_ids" in s2 else True
    assert dr["k"] == "draft" and dr["E"] == 2 and dr["ms"] is None and "pass_ms" not in dr
    return "round, lone round, slot (q from the proposing arm, ranks, depths, path) and draft lines"


def test_the_lattice_and_the_sampling_profile_with_round_log_lattice():
    """QWEN38_ROUND_LOG_LATTICE: the arm's whole lattice scores [L][k][k] and the request's seed and profile."""
    import torch
    from engine.sample import Sampler
    real = roundlog.LATTICE
    roundlog.LATTICE = True
    try:
        d2 = D2()
        sc = torch.tensor([[[0.5, -1.25], [2.0, 0.0]], [[1.0, 1.5], [-0.5, 0.25]]])
        d2._lattice = (torch.tensor([[11, 12], [13, 14]]), sc)
        arm = LenRouter()
        arm.large.mtp = d2
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rounds.jsonl")
            log = roundlog.RoundLog(path)
            rid = log.round([Req(0, 4, 50)], None)
            tree = DraftTree([5, 6], [-1, 0], scores=[1.0, 0.9], source=["root", "mtp"])
            sm = Sampler(temperature=1.0, top_p=0.95, top_k=20, seed=1042)
            log.slot(rid, 0, 1, 50, tree, [0, 1], 2, arm, drawn=[6, 9], sampler=sm)
            log.slot(rid, 0, 1, 52, tree, [0], 2, arm, drawn=[3])
            log.fh.close()
            _, s, s2 = [json.loads(x) for x in open(path)]
    finally:
        roundlog.LATTICE = real
    assert s["lat_scores"] == sc.tolist(), s["lat_scores"]
    assert s["samp"] == {"seed": 1042, "t": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}, s["samp"]
    assert s2["samp"] is None, "a greedy request"
    return "the whole lattice and the sampling profile, greedy logged as null"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:48s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
