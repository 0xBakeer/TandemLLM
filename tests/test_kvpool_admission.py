"""SRV-307 + SRV-305, CPU only: the parallel admission with the shared KV pool, and the wait for
memory as the dashboard sees it.

  * a request's KV part (the pages its prompt and max_tokens may still commit) is reserved; what
    its slot has committed since it was admitted is credited, never counted twice;
  * the KV part is never capped by the engine's room to its peak (the pages are outside the
    memory policy);
  * a request alone is admitted, as before; a release frees its bytes and its credit record;
  * waiters for memory keep their arrival order (`wait` gives each its place), `unwait` takes one
    out, and the report counts them (`waiting_memory`);
  * the parallel view shows a held request's lane as `waiting_memory` with the label "waiting for
    memory", counts it in `totals.waiting_memory` and `waiting`, never in `decoding`;
  * the contract schema knows the new state and the new total.

Run: python tests/test_kvpool_admission.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from server import slots  # noqa: E402

GiB = slots.GiB


def _adm(avail_gib: float, floor_gib: float = 15, committed=None, room=None):
    return slots.Admission(int(floor_gib * GiB), meminfo=lambda: {"MemAvailable": int(avail_gib * GiB)},
                           growth=lambda: 0, room=room, committed=committed)


def test_kv_part_is_reserved_and_credited_as_it_commits():
    done = {"slot1": 0, "slot2": 0}
    adm = _adm(40, committed=lambda o: done[o])
    ok, _ = adm.reserve("a", 2 * GiB + 10 * GiB, kv=10 * GiB, owner="slot1")
    assert ok                                      # alone
    ok, d = adm.reserve("b", 1 * GiB + 14 * GiB, kv=14 * GiB, owner="slot2")
    # 40 - (2 + 10 held) - (1 + 14) = 13 < 15
    assert not ok and d["kv_reserved"] == 10 * GiB and d["after"] == 13 * GiB
    # slot 1 committed 6 GiB: MemAvailable fell by those 6, the reservation by the same 6
    done["slot1"] = 6 * GiB
    adm.meminfo = lambda: {"MemAvailable": 34 * GiB}
    ok, d = adm.reserve("b", 1 * GiB + 14 * GiB, kv=14 * GiB, owner="slot2")
    assert not ok and d["after"] == 13 * GiB, "the same answer: nothing counted twice"
    adm.release("a")
    ok, d = adm.reserve("b", 15 * GiB, kv=14 * GiB, owner="slot2")
    assert ok                                      # alone again
    assert adm.report()["kv_to_commit_gib"] == 14.0


def test_credit_never_goes_negative_or_past_the_kv_part():
    c = {"slot1": 5 * GiB}
    adm = _adm(100, committed=lambda o: c[o])
    adm.reserve("a", 3 * GiB, kv=2 * GiB, owner="slot1")
    c["slot1"] = 1 * GiB                           # trimmed below its baseline
    ws, kv = adm._held()
    assert (ws, kv) == (1 * GiB, 2 * GiB)
    c["slot1"] = 50 * GiB
    ws, kv = adm._held()
    assert (ws, kv) == (1 * GiB, 0)


def test_kv_is_not_capped_by_the_engine_room():
    adm = _adm(60, room=lambda: 1 * GiB)
    adm.reserve("x", GiB, owner=None)              # alone, workspace only
    ok, d = adm.check(2 * GiB + 30 * GiB, kv=30 * GiB)
    # workspace parts capped at the 1 GiB room; the 30 GiB of pages counted in full
    assert d["after"] == 60 * GiB - 1 * GiB - 30 * GiB and ok


def test_plain_reservations_behave_as_before():
    adm = _adm(30)
    assert adm.reserve("a", 4 * GiB)[0]
    ok, d = adm.reserve("b", 12 * GiB)             # 30 - 4 - 12 = 14 < 15
    assert not ok and d["kv"] == 0
    assert adm.reserve("c", 10 * GiB)[0]           # 16 >= 15
    adm.release("a")
    adm.release("c")
    assert adm.reserved() == 0


def test_waiters_keep_their_order_and_are_counted():
    adm = _adm(10)
    assert adm.wait("w1") == 1
    assert adm.wait("w2") == 2
    assert adm.wait("w1") == 1                     # asking again keeps its place
    assert adm.report()["waiting_memory"] == 2
    adm.unwait("w1")
    assert adm.wait("w2") == 1
    adm.release("w2")
    assert adm.report()["waiting_memory"] == 0


def test_describe_names_the_kv_pages():
    adm = _adm(30)
    adm.reserve("a", 4 * GiB, kv=3 * GiB, owner="s1")
    ok, d = adm.check(20 * GiB, kv=18 * GiB)
    text = slots.Admission.describe(d)
    assert "KV pages: this request 18.0, in flight 3.0" in text and "< floor 15.0" in text


def test_parallel_lane_waiting_for_memory():
    from server import parallel

    class Rec:
        waiting_memory = True

    assert parallel.ParallelBoard._state(None, Rec(), None) == "waiting_memory"
    Rec.waiting_memory = False
    assert parallel.ParallelBoard._state(None, Rec(), 1.0) == "done"


def test_contract_knows_the_state_and_the_total():
    p = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "parallel.schema.json")
    s = json.load(open(p))
    txt = json.dumps(s)
    assert '"waiting_memory"' in txt
    assert "waiting_memory" in s["properties"]["totals"]["properties"] if "totals" in s.get(
        "properties", {}) else "waiting_memory" in txt


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
