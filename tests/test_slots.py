"""Parallel requests, phase A: the scheduler's parts on their own (server/slots.py), CPU only.

  * the GPU turn: exclusive, handed over only to a waiter that is a quantum behind in GPU time,
    never handed over by a holder that is alone, and `held_by` true only for the holder;
  * the joint round's barrier: two decoding slots meet in one round run by the last submitter,
    a slot that left decode does not keep the other waiting, and a slot alone never waits;
  * the row shares keep a joint pass at or under 32 rows, and a block that cannot share runs
    in a pass of its own;
  * the admission counts bytes: alone always admitted, otherwise MemAvailable less the caches'
    growth, the other requests' reservations and this one's worst case against the floor; a
    reservation released frees its bytes; the check is atomic;
  * the side slots: the smallest free one whose window holds the request, none when every one is
    busy or too short;
  * the per-request keys a side slot keeps cover everything the handler publishes.

Run: python tests/test_slots.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import slots  # noqa: E402

GiB = slots.GiB


# ------------------------------------------------------------------ the turn
def test_turn_is_exclusive_and_alone_never_hands_over():
    t = slots.Turn(quantum_s=0.0)
    t.acquire(0)
    assert t.held_by(0) and not t.held_by(1)
    assert t.pause(0) is False, "alone, a holder keeps the turn"
    got = []

    def other():
        t.acquire(1)
        got.append(time.monotonic())
        t.release(1)

    th = threading.Thread(target=other)
    th.start()
    time.sleep(0.05)
    assert not got, "the turn is exclusive"
    t.release(0)
    th.join(2)
    assert got, "released, the waiter gets it"


def test_pause_hands_over_to_a_waiter_behind_and_comes_back():
    clock = [0.0]
    t = slots.Turn(quantum_s=0.05, clock=lambda: clock[0])
    t.acquire(0)
    order = []

    def other():
        t.acquire(1)
        order.append("1 runs")
        t.release(1)

    th = threading.Thread(target=other)
    th.start()
    while not t.waiting():
        time.sleep(0.001)
    clock[0] = 0.01                         # slot 0 is only 10 ms ahead: under the quantum
    assert t.pause(0) is False
    clock[0] = 0.2                          # 200 ms ahead: the waiter goes first
    assert t.pause(0) is True
    order.append("0 back")
    th.join(2)
    assert order == ["1 runs", "0 back"], order
    assert t.held_by(0)
    t.release(0)
    assert t.stats["handoffs"] == 1


def test_pause_by_a_thread_that_does_not_hold_it_is_a_no_op():
    t = slots.Turn()
    assert t.pause(3) is False
    t.acquire(0)
    assert t.pause(1) is False
    assert t.held_by(0)
    t.release(0)


def test_try_acquire_takes_a_free_turn_and_never_waits():
    """SPD-249: `try_acquire` holds a free turn (True); with another holder, or a waiter owed it next, it takes
    nothing and returns at once (False)."""
    t = slots.Turn(quantum_s=0.0)
    assert t.try_acquire("runner") and t.held_by("runner")
    t.release("runner")
    t.acquire(1)
    t0 = time.monotonic()
    assert not t.try_acquire("runner") and time.monotonic() - t0 < 0.05
    assert t.held_by(1) and t.waiting() == 0
    got = []
    th = threading.Thread(target=lambda: (t.acquire(2), got.append(2), t.release(2)))
    th.start()
    while not t.waiting():
        time.sleep(0.001)
    t.release(1)
    th.join(2)
    assert got == [2] and t.owner is None
    assert t.try_acquire("runner")
    t.release("runner")
    return "a free turn taken; held or owed: nothing taken, no wait"


# ------------------------------------------------------------------ the barrier
class _Req:
    pass


def _pool(n=2, timeout=0.005):
    ss = [slots.Slot(i, 1024, values=None if i == 0 else {}) for i in range(n)]
    turn = slots.Turn(quantum_s=0.0)
    meet = slots.Meeting(32, timeout)
    return slots.Pool(ss, turn=turn, meeting=meet), turn, meet


def test_two_decoding_slots_meet_in_one_round_run_by_the_last():
    pool, turn, meet = _pool(2)
    for s in pool.slots:
        s.phase = "decode"
    runs = []

    def run(m):
        with m.cv:
            pend = list(m.pending.values())
            m.pending.clear()
            runs.append(sorted(r.slot for r in pend))
            for r in pend:
                r.out, r.ms = f"logits-{r.slot}", False
                m.transit.add(r.slot)
            m.cv.notify_all()

    res = {}

    def slot(i):
        turn.acquire(i)
        req = slots.VerifyReq(i, None, None, [-1, 0, 1], 0)
        slots.barrier(meet, pool, turn, i, req, run)
        res[i] = (req.out, turn.held_by(i))
        turn.release(i)

    ths = [threading.Thread(target=slot, args=(i,)) for i in (0, 1)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(5)
    assert runs == [[0, 1]], f"one joint round for both: {runs}"
    assert res[0][0] == "logits-0" and res[1][0] == "logits-1"
    assert res[0][1] and res[1][1], "each slot holds the turn again when the barrier returns"
    assert meet.stats["timeouts"] == 0


def test_a_prefill_holding_the_turn_does_not_starve_a_split_round():
    """SRV-311: slots 1 and 2 decode, slot 1 waits in this round, slot 2 is in another meeting
    (the packed draft) and never submits here, and slot 3 holds the turn for a long prefill
    (not decoding). The waiter times out and runs its round once it gets the turn, instead of
    waiting for the prefill to end."""
    pool, turn, meet = _pool(4)
    pool.slots[1].phase = pool.slots[2].phase = "decode"
    runs = []

    def run(m):
        with m.cv:
            pend = list(m.pending.values())
            m.pending.clear()
            runs.append(sorted(r.slot for r in pend))
            for r in pend:
                r.out, r.ms = f"logits-{r.slot}", False
            m.cv.notify_all()

    stop = threading.Event()

    def prefill():
        turn.acquire(3)
        while not stop.is_set():
            time.sleep(0.02)              # a prefill chunk
            turn.pause(3)                 # offered between chunks
        turn.release(3)

    pf = threading.Thread(target=prefill)
    pf.start()
    time.sleep(0.05)
    res = {}

    def slot1():
        turn.acquire(1)
        req = slots.VerifyReq(1, None, None, [-1, 0, 1], 0)
        t0 = time.monotonic()
        slots.barrier(meet, pool, turn, 1, req, run)
        res["s"] = time.monotonic() - t0
        res["out"] = req.out
        turn.release(1)

    th = threading.Thread(target=slot1)
    th.start()
    th.join(3)
    stop.set()
    pf.join(3)
    assert res.get("out") == "logits-1", res
    assert res["s"] < 1.0, f"waited {res['s']:.2f} s behind the prefill"
    assert runs == [[1]] and meet.stats["timeouts"] == 1


def test_a_slot_that_left_decode_does_not_keep_the_other_waiting():
    pool, turn, meet = _pool(2, timeout=0.002)
    pool.slots[0].phase = "decode"
    pool.slots[1].phase = "decode"
    done = []

    def run(m):
        with m.cv:
            pend = list(m.pending.values())
            m.pending.clear()
            for r in pend:
                r.out = "x"
            done.append(len(pend))
            m.cv.notify_all()

    turn.acquire(0)
    req = slots.VerifyReq(0, None, None, [-1, 0], 0)

    def leave():
        time.sleep(0.05)
        pool.slots[1].phase = ""               # slot 1's request ended without a block

    th = threading.Thread(target=leave)
    th.start()
    t0 = time.monotonic()
    slots.barrier(meet, pool, turn, 0, req, run)
    th.join()
    assert req.out == "x" and done == [1]
    assert time.monotonic() - t0 < 2.0
    turn.release(0)


def test_the_turn_idle_runs_the_round_after_the_timeout():
    pool, turn, meet = _pool(2, timeout=0.002)
    pool.slots[0].phase = pool.slots[1].phase = "decode"   # slot 1 decodes but never submits

    def run(m):
        with m.cv:
            for r in m.pending.values():
                r.out = "y"
            m.pending.clear()
            m.cv.notify_all()

    turn.acquire(0)
    req = slots.VerifyReq(0, None, None, [-1, 0], 0)
    slots.barrier(meet, pool, turn, 0, req, run)
    assert req.out == "y" and meet.stats["timeouts"] == 1
    assert turn.held_by(0)
    turn.release(0)


def test_shares_keep_a_joint_pass_under_32_rows():
    assert slots.share(1) is None, "alone, a slot keeps the router's own budget"
    for n in range(2, 9):
        s = slots.share(n)
        assert s is not None and s >= 2 and s * n <= 32, (n, s)
    assert slots.share(2) == 16 and slots.share(4) == 8


def test_spd163_a_familys_policy_sets_its_shares_and_auto_n():
    assert slots.share(2, 64, {2: 32}) == 32 and slots.share(4, 64, {2: 32}) == 8
    assert slots.share(2, 32, {2: 32}) == 16, "never past the row cap"
    m = slots.Meeting(64, 0.002, cut="auto", auto_n=2, shares={2: 32})
    if not slots.AUTO_JOINT_N_SET:
        assert m.auto_n == 2 and m.joint_for(2) and not m.joint_for(1)
    assert slots.Meeting(64, 0.002, cut="auto").auto_n == slots.AUTO_JOINT_N, "no policy: as before"
    assert m.shares == {2: 32} and slots.Meeting(32).shares == {}


def test_batches_cut_at_max_rows_and_keep_unbatchable_blocks_alone():
    meet = slots.Meeting(32, 0.002)
    meet.pending = {0: slots.VerifyReq(0, None, None, list(range(16)), 0),
                    1: slots.VerifyReq(1, None, None, list(range(16)), 0),
                    2: slots.VerifyReq(2, None, None, list(range(8)), 0, batchable=False),
                    3: slots.VerifyReq(3, None, None, list(range(8)), 0)}
    groups = sorted([r.slot for r in g] for g in meet.batches())
    assert groups == [[0, 1], [2], [3]], groups
    assert all(sum(len(meet.pending[i].parents) for i in g) <= 32 for g in groups)


def test_more_lanes_than_one_pass_holds_verify_in_more_passes():
    """SRV-707: past 16 decoding lanes a share is 2 rows and the round runs in passes of at most
    32 rows; a joint cut groups at most 16 trees a pass (anchor and one node each)."""
    for n in (16, 17, 24, 32, 48, 64):
        s = slots.share(n)
        assert s == 2, (n, s)
        meet = slots.Meeting(32, 0.002)
        meet.pending = {i: slots.VerifyReq(i, None, None, list(range(s)), 0) for i in range(n)}
        groups = meet.batches()
        assert sorted(r.slot for g in groups for r in g) == list(range(n))
        assert all(sum(r.rows for r in g) <= 32 for g in groups), n
        assert len(groups) == -(-n // 16), (n, len(groups))
        jm = slots.Meeting(32, 0.002, cut="joint")
        # a joint cut takes the blocks that came with their tree (SPD-155)
        jm.pending = {i: slots.VerifyReq(i, None, None, list(range(10)), 0, tree=object())
                      for i in range(n)}
        jm.pending[n] = slots.VerifyReq(n, None, None, list(range(4)), 0, batchable=False)
        jg = jm.batches()
        assert all(len(g) <= jm.per_pass() for g in jg) and jm.per_pass() == 16
        assert sorted(r.slot for g in jg for r in g) == list(range(n + 1))
        assert [r.slot for r in jg[-1]] == [n], "an unbatchable block keeps a pass of its own"
    one = slots.Meeting(32, 0.002, cut="joint")
    one.pending = {i: slots.VerifyReq(i, None, None, list(range(10)), 0) for i in range(3)}
    assert [len(g) for g in one.batches()] == [3], "up to 16 trees stay one joint pass"


# ------------------------------------------------------------------ admission
def test_srv775_alone_is_checked_against_the_floor():
    """SRV-775: a request with nothing else in flight is checked as any other (earlier it went unchecked, and a lone
    side-slot turn took MemAvailable to 5.0 GiB); `alone` stays a count, `lone_checked` says the check ran."""
    avail = [16 * GiB]
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": avail[0]})
    ok, d = adm.reserve("a", 5 * GiB)                  # 16 - 5 = 11 >= 10
    assert ok and d.get("alone") and d["after"] == 11 * GiB, d
    assert adm.stats["alone"] == 1 and adm.stats["lone_checked"] == 1
    adm.release("a")
    assert adm.reserved() == 0
    avail[0] = 3 * GiB
    ok, d = adm.reserve("b", 5 * GiB)                  # 3 - 5 < 10, alone or not
    assert not ok and d.get("alone") and "b" not in adm.held, d
    assert adm.stats["lone_checked"] == 2 and slots.Admission.describe(d).startswith("alone: MemAvailable 3.0")
    return "alone: 16 - 5 = 11 >= 10 admitted, 3 - 5 refused"


def test_srv775_the_room_caps_the_growth_and_the_prefills_in_flight_an_over_peak_admit_counts_in_full():
    """SRV-775: the engine's room to its peak caps the caches' growth, the workspaces in flight and this request's, as
    before: the memory gate, on every slot, keeps each prefill it admits under the peak. Hold 3's c8 round
    (2026-11-02: MemAvailable 46.9 GiB, cache growth 13.4, eight cold prompts of 2.75 GiB of workspace and 0.22 of KV
    pages): counted in full past the room, the eighth found 46.9 - 13.4 - 20.8 - 3.0 = 9.7 GiB < 10 and waited 12 s;
    under the room (13.5) all eight start. A lone request the gate admitted over the peak (`over_peak`) counts its
    workspace in full past the room: at room 0, g2's 9.91 GiB and 0.5 of KV pages are refused at MemAvailable 20
    (20 - 9.91 - 0.5 < 10) and admitted at 21. A workspace leaves the reservation when its prefill has run, its KV
    part kept until the release."""
    ws, kv = int(2.75 * GiB), int(0.22 * GiB)
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(46.9 * GiB)},
                          growth=lambda: int(13.4 * GiB), room=lambda: int(13.5 * GiB))
    for i in range(8):
        ok, d = adm.reserve(f"c{i}", ws + kv, kv=kv)
        assert ok, (i, slots.Admission.describe(d))
    # the eighth: 13.4 + 7 x 2.75 + 2.75 = 35.4 under the room, capped at 13.5; the KV pages past it
    assert d["room"] == int(13.5 * GiB) and d["under"] == int(13.4 * GiB) + 8 * ws, d
    assert d["after"] == int(46.9 * GiB) - int(13.5 * GiB) - 8 * kv and d["in_flight_ws"] == 7 * ws, d
    text = slots.Admission.describe(d)
    assert text.startswith("MemAvailable 46.9 GiB - the engine's room to its peak 13.5 (under the room: cache "
                           "growth 13.4, workspaces in flight ") and "this request's" in text, text
    full = int(46.9 * GiB) - int(13.4 * GiB) - 8 * (ws + kv)      # every workspace in full: the 12 s wait
    assert full < adm.floor and abs(full / GiB - 9.74) < 0.01, full / GiB
    assert adm.ws_in_flight() == 8 * ws
    adm.prefilled("c0")
    assert adm.held["c0"] == kv and adm.ws_in_flight() == 7 * ws, adm.held
    g2, g2kv = int(9.91 * GiB), GiB // 2
    for avail, want in ((20.0, False), (21.0, True)):
        def lone():
            return slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(avail * GiB)},
                                   growth=lambda: int(6.6 * GiB), room=lambda: 0)
        ok, d = lone().reserve("g2", g2 + g2kv, kv=g2kv, over_peak=True)
        assert ok is want and d.get("alone") and d["over_peak"], (avail, d)
        assert d["after"] == int(avail * GiB) - g2 - g2kv, d
        assert "this request over the peak 9.9" in slots.Admission.describe(d), slots.Admission.describe(d)
        ok, d = lone().reserve("g2", g2 + g2kv, kv=g2kv)                   # under the peak: the room caps it
        assert ok and d["after"] == int(avail * GiB) - g2kv, d
    rep = adm.report()
    assert rep["ws_in_flight_gib"] == round(7 * ws / GiB, 3) and rep["lone_checked"] == 1, rep
    return "c8 at hold 3's figures: 8 of 8 under the room (full count 9.7 < 10); over the peak alone: 20 refused, 21 in"


def test_srv775_an_over_peak_admit_is_credited_the_allocators_free_blocks():
    """SRV-775 (2026-11-02, the lone chat that waited 15 minutes): a prefill admitted over the peak takes the
    allocator's free default-pool blocks before any new memory from the box, and MemAvailable already leaves those
    out with the engine's whole reserve. The saved case: floor 10, MemAvailable 13.8, room 0, cache growth 13.6, a
    workspace of 3.9 and 0.2 of KV pages. Charged in full it found 9.7 < 10 and waited; with 2.3 GiB of free blocks
    it is 1.6 new and admitted at 12.0. A credit past the workspace leaves it 0 new; a reading that fails gives no
    credit, as before; a request under the peak is never credited."""
    ws, kv = int(3.9 * GiB), int(0.2 * GiB)
    avail = int(13.8 * GiB)

    def adm(spare):
        return slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": avail}, growth=lambda: int(13.6 * GiB),
                               room=lambda: 0, spare=spare)

    def boom():
        raise RuntimeError("no allocator reading")

    rows = []
    for spare, want_ok, want_new in ((lambda: int(2.3 * GiB), True, ws - int(2.3 * GiB)), (lambda: 0, False, ws),
                                     (lambda: 10 * GiB, True, 0), (boom, False, ws), (None, False, ws)):
        a = adm(spare)
        ok, d = a.reserve("c", ws + kv, kv=kv, over_peak=True)
        assert ok is want_ok and d["new_ws"] == want_new, (d, want_ok, want_new)
        assert d["after"] == avail - want_new - kv, d
        assert d["spare"] == ws - want_new and ("c" in a.held) is want_ok, d
        rows.append(f"{d['after'] / GiB:.1f}")
    ok, d = adm(lambda: int(2.3 * GiB)).reserve("c", ws + kv, kv=kv, over_peak=True)
    text = slots.Admission.describe(d)
    assert ("this request over the peak 1.6 new (2.3 of its 3.9 in the allocator's free blocks) = 12.0 GiB "
            "(>= floor 10.0)") in text, text
    _, d = adm(lambda: 0).reserve("c", ws + kv, kv=kv, over_peak=True)
    text = slots.Admission.describe(d)
    assert "this request over the peak 3.9 = 9.7 GiB (< floor 10.0)" in text and "free blocks" not in text, text
    asked = []
    a = adm(lambda: asked.append(1) or int(2.3 * GiB))
    ok, d = a.reserve("u", ws + kv, kv=kv)                      # under the peak: the room caps it, no reading
    assert ok and asked == [] and "spare" not in d and d["after"] == avail - kv, d
    # no room cap (no policy): the over-peak line names the credit too
    a = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": avail}, spare=lambda: int(2.3 * GiB))
    ok, d = a.reserve("n", ws + kv, kv=kv, over_peak=True)
    assert ok and d["after"] == avail - (ws - int(2.3 * GiB)) - kv, d
    assert "this request 1.8 new (2.3 of its 3.9 in the allocator's free blocks)" in slots.Admission.describe(d), \
        slots.Admission.describe(d)
    return "after " + ", ".join(rows) + " GiB (spare 2.3, 0, 10, raising, none)"


def test_srv787_a_cold_128k_turn_is_not_charged_the_room_and_its_workspace():
    """SRV-787 (2026-11-05): a turn the gate admitted over the peak was charged the engine's room to its peak AND its
    workspace, though the policy grows no cache while that workspace is in flight. 22:41, the 128k turn that waited
    900 s with the engine idle: floor 10, MemAvailable 25.0, room 8.5, cache growth 13.6, a workspace of 9.47 with
    1.44 in free blocks, 0.355 of KV pages and 0.457 in flight. The sum said 7.7 < 10; the larger of the two says
    15.7 on today's growth, 16.2 on the growth the policy allows (`_par_growth`). 23:00 (82k on slot 5, 9.3 < 10 and
    34 s of wait) and 23:03 (9.9 < 10) start at once; the cold 128k turn of 23:03 (4.1 GiB of KV pages to commit)
    still waits, at 9.6 instead of 5.5. Every figure as its log line printed it."""
    rows = []

    def case(avail, room, growth, ws, spare, kv, kv_in=0.0):
        a = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(avail * GiB)},
                            growth=lambda: int(growth * GiB), room=lambda: int(room * GiB),
                            spare=lambda: int(spare * GiB))
        if kv_in:
            assert a.reserve("decoding", int(kv_in * GiB), kv=int(kv_in * GiB))[0]   # its KV pages only
        w, s, k = int(ws * GiB), int(spare * GiB), int(kv * GiB)
        ok, d = a.reserve("turn", w + k, kv=k, over_peak=True)
        summed = int(avail * GiB) - min(int(room * GiB), int(growth * GiB)) - (w - s) - k - int(kv_in * GiB)
        assert d["after"] == int(avail * GiB) - max(min(int(room * GiB), int(growth * GiB)), w - s) - k \
            - int(kv_in * GiB), d
        assert d["after"] != summed and ok is (d["after"] >= 10 * GiB) and ("turn" in a.held) is ok, d
        rows.append(f"{summed / GiB:.1f} -> {d['after'] / GiB:.1f}")
        return ok, d, summed
    ok, d, summed = case(25.0, 8.5, 13.6, 9.47, 1.44, 0.355, 0.457)               # 22:41, today's growth
    assert f"{summed / GiB:.1f}" == "7.7" and ok and f"{d['after'] / GiB:.1f}" == "15.7", d
    text = slots.Admission.describe(d)
    assert text.startswith("MemAvailable 25.0 GiB - the larger of the engine's room to its peak 8.5 (under the room: "
                           "cache growth 13.6, workspaces in flight 0.0) and this request over the peak 8.0 new (1.4 "
                           "of its 9.5 in the allocator's free blocks) = 15.7 GiB (>= floor 10.0)"), text
    ok, d, _ = case(25.0, 8.5, 1.7, 9.47, 1.44, 0.355, 0.457)                     # 22:41, the policy's growth
    assert ok and f"{d['after'] / GiB:.1f}" == "16.2", d
    text = slots.Admission.describe(d)
    assert ("the larger of cache growth 1.7 + workspaces in flight 0.0 and this request over the peak 8.0 new "
            "(1.4 of its 9.5 in the allocator's free blocks) = 16.2 GiB (>= floor 10.0)") in text, text
    ok, d, summed = case(21.1, 4.1, 13.6, 8.46, 1.4, 0.6)                         # 23:00, slot 5
    assert f"{summed / GiB:.1f}" == "9.3" and ok and f"{d['after'] / GiB:.1f}" == "13.4", d
    assert slots.Admission.describe(d).startswith("alone: MemAvailable 21.1 GiB - the larger of"), d
    ok, d, summed = case(21.82, 4.06, 13.6, 8.62, 1.36, 0.64)                      # 23:03
    assert f"{summed / GiB:.1f}" == "9.9" and ok and f"{d['after'] / GiB:.1f}" == "13.9", d
    ok, d, summed = case(21.8, 4.1, 13.6, 9.47, 1.4, 4.1)                          # 23:03, the cold 128k turn
    assert f"{summed / GiB:.1f}" == "5.5" and not ok and f"{d['after'] / GiB:.1f}" == "9.6", d
    return "summed -> the larger: " + ", ".join(rows)


def test_srv787_under_the_peak_and_without_a_policy_the_count_is_as_before():
    """SRV-787 changes only an over-peak admit under a policy: a prefill under the peak is capped by the room with the
    growth and the workspaces in flight, as before, and with no room (no policy) the over-peak workspace adds to the
    growth, as before. Another prefill's workspace in flight counts with the over-peak one."""
    def adm(room):
        return slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": 30 * GiB}, growth=lambda: 4 * GiB,
                               room=room)
    ok, d = adm(lambda: 6 * GiB).reserve("u", 5 * GiB)                            # under: min(6, 4 + 5)
    assert ok and d["after"] == 24 * GiB, d
    ok, d = adm(None).reserve("n", 5 * GiB, over_peak=True)                       # no policy: 4 + 5
    assert ok and d["after"] == 21 * GiB, d
    a = adm(lambda: 6 * GiB)
    assert a.reserve("other", 3 * GiB)[0]                                         # a workspace in flight
    ok, d = a.reserve("o", 5 * GiB, over_peak=True)                               # max(min(6, 4 + 3), 3 + 5)
    assert ok and d["after"] == 22 * GiB, d
    assert "+ workspaces in flight 3.0 = 22.0 GiB" in slots.Admission.describe(d), slots.Admission.describe(d)
    return "under 24, no policy 21, beside a workspace in flight 22"


def test_srv775_describe_names_a_peak_wait():
    d = {"peak_wait": True, "need": int(76.8 * GiB), "peak": int(75.8 * GiB), "ws": int(2.8 * GiB),
         "reserved": int(74.0 * GiB), "over": int(1.0 * GiB)}
    text = slots.Admission.describe(d)
    assert text == ("the engine's peak: reserved 74.0 + this request's workspace 2.8 > peak 75.8 (1.0 GiB over "
                    "after shedding)"), text
    return text


def test_admission_counts_the_requests_in_flight_and_the_growth():
    avail = [40 * GiB]
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": avail[0]},
                          growth=lambda: 5 * GiB)
    assert adm.reserve("a", 8 * GiB)[0]
    ok, d = adm.reserve("b", 8 * GiB)       # 40 - 5 - 8 - 8 = 19 >= 15
    assert ok, d
    ok, d = adm.reserve("c", 8 * GiB)       # 40 - 5 - 16 - 8 = 11 < 15
    assert not ok and d["after"] == 11 * GiB, d
    assert "c" not in adm.held
    adm.release("a")
    ok, d = adm.reserve("c", 8 * GiB)       # 40 - 5 - 8 - 8 = 19
    assert ok, d
    assert slots.Admission.describe(d).endswith("(>= floor 15.0)")


def test_admission_without_meminfo_admits_and_says_so():
    adm = slots.Admission(15 * GiB, meminfo=lambda: None)
    adm.reserve("a", 1)
    ok, d = adm.reserve("b", 100 * GiB)
    assert ok and d["avail"] is None
    assert "not checked" in slots.Admission.describe(d)


def test_admission_check_and_reserve_are_atomic():
    """Two requests racing for the last bytes: exactly one gets them."""
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": 30 * GiB})
    adm.reserve("first", 1 * GiB)            # not alone any more: the floor applies
    wins = []
    barrier = threading.Barrier(8)

    def go(k):
        barrier.wait()
        if adm.reserve(k, 12 * GiB)[0]:
            wins.append(k)

    ths = [threading.Thread(target=go, args=(f"r{i}",)) for i in range(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    # 30 - 1 - 12 = 17 >= 10 for the first; 30 - 13 - 12 = 5 < 10 for any second
    assert len(wins) == 1, wins


def test_slot_bytes_follow_the_byte_math():
    assert slots.slot_bytes(65536, constant=0) == 65536 * 106_752
    assert slots.slot_bytes(0) == 7 * GiB + GiB // 2


# ------------------------------------------------------------------ side slots
def test_take_side_gives_the_smallest_free_slot_that_fits():
    ss = [slots.Slot(0, 262144), slots.Slot(1, 65536, values={}), slots.Slot(2, 16384, values={})]
    pool = slots.Pool(ss)
    s = pool.take_side(10000, owner="a")
    assert s is ss[2] and s.owner == "a" and s.busy
    s2 = pool.take_side(10000, owner="b")
    assert s2 is ss[1], "the next free one that fits"
    assert pool.take_side(100, owner="c") is None, "every side slot busy"
    pool.release_side(s)
    assert pool.take_side(20000) is None, "16k is too short for 20k, and 64k is busy"
    assert pool.take_side(16000) is ss[2]
    assert pool.stats["side_grants"] == 3 and pool.stats["peak_running"] >= 2


def test_the_overlay_covers_what_the_generation_publishes():
    for k in ("engine", "drafter", "max_len", "resident", "mem", "last_ctx", "last_prompt_n",
              "hidden_spans", "blocks", "last_prefill", "lane", "disk_cache", "spill"):
        assert k in slots.OVERLAY, k



def test_a_joint_pass_never_gets_a_tree_deeper_than_its_kernels():
    """SPD-223: a 32-deep chain (the lookup's deep chain) is cut to the family's joint depth (16 levels,
    the anchor included); a shallow tree and a missing cap leave the tree as it is."""
    from engine.tree import DraftTree
    chain = DraftTree(list(range(32)), [-1] + list(range(31)), [1.0] * 32)
    cut = slots.depth_cut(chain, 16)
    assert len(cut.tokens) == 16 and max(cut.depths()) == 15 and cut.tokens == list(range(16))
    wide = DraftTree([5, 6, 7, 8], [-1, 0, 0, 1], [1.0, .5, .4, .3])
    assert slots.depth_cut(wide, 16) is wide and slots.depth_cut(chain, None) is chain

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
