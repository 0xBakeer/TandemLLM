"""SPD-249, no GPU: the joint pass's measured shape and the entry layouts of the new launches.

  * `ExpertUnion` counts each layer's distinct routed experts of a sampled pass (the shared slot left out),
    samples every N-th pass, waits for a pending copy, and reports by lane count; off reports None;
  * the serving's `joint_stats` hands that report to /health and the runner's line;
  * `rows_entry` carries the count tensor as its first destination row (field 8), `gdn_rows_entry` its limit
    (field 15), `gdn_pre_entry` the 11 fields of a state's conv and gates call;
  * `_applied_deferred` keeps a count the kernel's rule keeps (at most the limit, room for the block's path in the
    33 parked rows) and zeroes the others, and the state stays among the parked ones.
"""
from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.union import ExpertUnion  # noqa: E402


def _ids(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor([r + [999] for r in rows], dtype=torch.int32)       # the shared slot last


def test_the_union_of_a_sampled_pass():
    u = ExpertUnion(layers=3, experts=8, every=2, device="cpu")
    m = u.begin()                                                 # pass 1: sampled
    assert m is not None
    u.note(m, 0, _ids([[1, 2], [2, 3]]))
    u.note(m, 1, _ids([[5, 5], [5, 5]]))
    u.note(m, 2, _ids([[0, 7], [1, 6], [2, 5]]))
    u.end(m, rows=3, lanes=2)
    assert u.begin() is None                                      # pass 2: not sampled
    m = u.begin()                                                 # pass 3: sampled (the copy is read first)
    for li in range(3):
        u.note(m, li, _ids([[0, 1]]))
    u.end(m, rows=1, lanes=1)
    u.harvest()                                                   # the runner's next pass reads the copy
    rep = u.report()
    assert rep["passes"] == 3 and set(rep["by_lanes"]) == {1, 2}, rep
    two = rep["by_lanes"][2]
    assert two["u_layer"] == [3.0, 1.0, 6.0] and two["u_mean"] == round(10 / 3, 1) and two["rows_mean"] == 3.0, two
    assert two["u_min"] == 1.0 and two["u_max"] == 6.0
    assert rep["by_lanes"][1]["u_layer"] == [2.0, 2.0, 2.0]
    return "layers 3 / 1 / 6 distinct of 2 lanes, 2 a layer of 1; every 2nd pass; the shared slot not counted"


def test_off_counts_nothing():
    u = ExpertUnion(layers=2, experts=4, every=0, device="cpu")
    assert u.begin() is None and u.report() is None
    return "every = 0: no mask, report None"


def test_a_pending_copy_holds_the_next_sample():
    u = ExpertUnion(layers=1, experts=4, every=1, device="cpu")
    m = u.begin()
    u.note(m, 0, _ids([[0, 1]]))
    u.end(m, rows=1, lanes=1)
    u._pending = u._pending[:1] + (types.SimpleNamespace(query=lambda: False),) + u._pending[2:]
    assert u.begin() is None                                      # the last copy is not done: no new sample

    def no_cuda():
        raise AssertionError("report() queried an event")
    u._pending = u._pending[:1] + (types.SimpleNamespace(query=no_cuda),) + u._pending[2:]
    assert u.report()["by_lanes"] == {}                           # a report never touches CUDA (another thread)
    u._pending = u._pending[:1] + (None,) + u._pending[2:]
    assert u.begin() is not None and u.report()["by_lanes"][1]["u_layer"] == [2.0]
    return "a sample waits for the last copy; a report makes no CUDA call; the next sample reads it"


def test_joint_stats_reports_the_union():
    from engine.models.flashnext.serving import FlashNextServing
    s = object.__new__(FlashNextServing)
    s.fn = types.SimpleNamespace(union=ExpertUnion(1, 4, 1, device="cpu"))
    m = s.fn.union.begin()
    s.fn.union.note(m, 0, _ids([[3, 2]]))
    s.fn.union.end(m, rows=1, lanes=1)
    s.fn.union.harvest()
    js = s.joint_stats()
    assert js["experts_union"]["by_lanes"][1]["u_layer"] == [2.0], js
    s.fn = types.SimpleNamespace(union=ExpertUnion(1, 4, 0, device="cpu"))
    assert s.joint_stats() is None
    return "joint_stats = {experts_union: the report}; None when off"


def test_entry_layouts():
    from tools import fn_seq_kernels as SK
    src = torch.zeros(6, 8, dtype=torch.bfloat16)
    dst = torch.zeros(33, 8, dtype=torch.bfloat16)
    idx = torch.arange(2)
    off = torch.zeros(4, dtype=torch.int32)
    e0 = SK.rows_entry(src, dst, idx, 2)
    e1 = SK.rows_entry(src, dst, idx, 2, off=off[1:2])
    assert len(e0) == len(e1) == SK.COMMIT_FIELDS and e0[8] == 0 and e1[8] == off[1:2].data_ptr() and e0[:8] == e1[:8]
    qkv = torch.zeros(3, 16, dtype=torch.bfloat16)
    g = torch.zeros(3, 2)
    S = torch.zeros(1, 2, 4, 4)
    ctl = torch.zeros(3, 2, dtype=torch.int32)
    out = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
    r0 = SK.gdn_rows_entry(qkv, g, g, S, S, ctl, out, False)
    r1 = SK.gdn_rows_entry(qkv, g, g, S, S, ctl, out, False, limit=8)
    assert r0[15] == 0 and r1[15] == 8 and r0[:15] == r1[:15]
    cs = torch.zeros(1, 4, 3, dtype=torch.bfloat16)
    srcs = torch.zeros(3, 3, dtype=torch.int32)
    y = torch.zeros(3, 4, dtype=torch.bfloat16)
    ab = torch.zeros(3, 4, dtype=torch.bfloat16)
    p = SK.gdn_pre_entry(qkv, cs, srcs, y, ab, g, g)
    assert len(p) == SK.COMMIT_FIELDS and p[10] == 3 and p[0] == qkv.data_ptr() and p[4] == y.data_ptr() and p[11:] == [0] * 5
    return "rows_entry field 8 = the count's address, gdn_rows_entry field 15 = the limit, gdn_pre_entry 11 fields"


def test_the_deferred_counts_follow_the_kernels_rule():
    from engine.models.flashnext import engine as E
    eng = object.__new__(E.FlashNextEngine)
    eng._pending = {}
    counts = {}

    def pend_of(st):
        return ({}, counts[id(st)])
    eng._pend_of = pend_of
    real = E.GDN_DEFER
    E.GDN_DEFER = 8
    try:
        cx = []
        for P, M in ((0, 3), (5, 3), (8, 3), (9, 3), (30, 3), (31, 3)):
            st = types.SimpleNamespace()
            counts[id(st)] = torch.full((36,), P, dtype=torch.int32)
            cx.append({"st": st, "M": M})
        eng._applied_deferred(cx)
        got = [int(counts[id(x["st"])][0]) for x in cx]
    finally:
        E.GDN_DEFER = real
    # kept: 1..8 rows with room for the path (P + M <= 33); zeroed: over the limit
    assert got == [0, 5, 8, 0, 0, 0], got
    assert all(id(x["st"]) in eng._pending for x in cx)
    return "counts 0/5/8/9/30/31 with 3-row blocks at limit 8 -> 0/5/8/0/0/0; every state still parked"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
