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


class LenRouter:
    last_key = "l"
    small, large = None, Arm()


def test_round_slot_and_draft_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rounds.jsonl")
        log = roundlog.RoundLog(path)
        rid = log.round([Req(0, 8, 300), Req(3, 6, 512)], None, cap=8, pre_rows=[12, 6], graphed=True)
        rid2 = log.round([Req(1, 4, 100)], None)
        tree = DraftTree([5, 6, 7, 8], [-1, 0, 1, 0], scores=[1.0, 0.9, 0.4, 0.7], source=["root", "mtp", "mtp", "sib"])
        log.slot(rid, 3, 2, 512, tree, [0, 1, 2], 6, LenRouter())
        log.slot(rid2, 1, 1, 100, tree, [0], 4, object())
        log.draft([Req(0, 16, 300), Req(3, 16, 512)], None)
        log.fh.close()
        lines = [json.loads(x) for x in open(path)]
    r, r2, s, s2, dr = lines
    assert (r["k"], r["rid"], r["E"], r["slots"], r["rows"], r["pre_rows"], r["cap"], r["ctx"], r["graphed"]) == \
        ("round", 1, 2, [0, 3], [8, 6], [12, 6], 8, [300, 512], True)
    assert r["pass_ms"] is None and r2["E"] == 1 and r2["rid"] == 2 and r2["period_ms"] is not None
    assert s["k"] == "slot" and s["rid"] == 1 and s["slot"] == 3 and s["n"] == 4 and s["accepted"] == 2
    assert s["depth"] == [0, 1, 2, 1] and s["rank"] == [0, 1, 3, 2] and s["pre_n"] == 6
    assert s["q"] == [None, 0.5, 0.25, 0.5], s["q"]
    assert s2["q"] is None, "a drafter without a calibrated q"
    assert dr["k"] == "draft" and dr["E"] == 2 and dr["ms"] is None and "pass_ms" not in dr
    return "round, lone round, slot (q from the proposing arm, ranks, depths, path) and draft lines"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:48s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
