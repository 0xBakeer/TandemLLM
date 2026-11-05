"""SRV-763 (QA SRV-764): the server's estimate of when the router's own question is answered
(`app._own_eta_ms`, the router's `own_eta`), on an idle box and after a burst.

Live 2026-10-31 (fd627ac3): on an idle box the package question (about 2,400 rows) was skipped as
`qwen_busy` at an estimate of 13.5 s and took 2.2 to 3.1 s when asked: the router's class question
had just ended and still held slot 0 for its end's bookkeeping, and that slot counted as a 4,096-row
prefill chunk on each of the question's 3 turns (3 x 3.8 s). After a burst of chats the estimate's
prefill rate, the median of the last 16 requests, read 150 to 190 tok/s: every request in it had
shared the GPU turn with 7 others. Requests that prefilled alone read 969 tok/s (median of 58), shared
ones 178 (146) in the ledger from 13:45 to 18:10.

  1  a request's slot in its end's bookkeeping (or held with no prompt marked) charges no turn; slot 0
     held by TandemLLM's own job still charges a chunk; QWEN38_OWN_UNITS=busy is the old rule
                                                  test_1_a_slot_in_its_end_charges_no_turn
  2  a slot that prefills charges the chunk it is on   test_2_a_prefilling_slot_charges_its_chunk
  3  the estimate reads the rate of the requests that prefilled alone; none yet: the peek's rate;
     QWEN38_OWN_RATE=all is the old rule           test_3_the_solo_rate
  4  a request's prompt had the turn alone when no other holder acquired it from its lock to its
     first token; without side slots, always; only those are noted
                                                  test_4_alone_is_no_other_acquire
  5  the prefill marks its slot and the chunk hook counts its rows down; a phase set already stays
                                                  test_5_the_prefill_mark
  6  the record calls its first-token hook once    test_6_the_first_token_hook
  7  a slot granted to a request that has not reached its prefill (the memory admission, the vision
     tower) charges its prompt's rows up to a chunk a turn, until the prefill starts or the request
     ends; on slot 0 too                          test_7_a_granted_slot_charges_its_prompt_until_it_prefills
  8  the burst term (requests in the router or waiting for a slot) reads the peek's rate, as before;
     QWEN38_OWN_BURST_RATE=solo reads the solo rate; with no burst it is 0
                                                  test_8_the_burst_term_keeps_the_peeks_rate

Run: python tests/test_own_eta_solo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_conductor_router import setup  # noqa: E402

from server import app, cache_peek, usage  # noqa: E402
from server import round_runner as RR  # noqa: E402
from server import slots as S  # noqa: E402

PACKAGE_Q = ("Which of these instructions fits this request best?", "z" * 2370)
RATE = 1000.0                                     # one row a ms
PEEK = 178.0                                      # a peek's rate after a burst (any slower one)


def _rows(texts) -> int:
    return app.OWN_TEMPLATE_ROWS + sum(len(t) for t in texts)     # QTok: a char a token


class _Env:
    """A pool of 3 slots, the rate pinned, the solo rates empty; restored on exit."""

    def __init__(self, peek=RATE, n=3):
        self.peek, self.n = peek, n

    def __enter__(self):
        setup()
        self.saved = (app._peek_rate, RR.RUNNER, list(cache_peek.SOLO_RATES._q), app.OWN_RATE,
                      app.OWN_UNITS, app.OWN_BURST_RATE, app.PLANNING["n"])
        cache_peek.SOLO_RATES._q.clear()
        app._peek_rate = lambda: self.peek
        app.STATE["prefix_chunk"] = 4096
        app.STATE["lane"] = None
        RR.RUNNER = None
        self.pool = app.STATE["pool"] = S.Pool([S.Slot(i, 65_536) for i in range(self.n)])
        return self

    def __exit__(self, *exc):
        if app.LOCK.locked():
            app.HOLDER["who"] = None
            app.LOCK.release()
        for s in self.pool.slots:
            s.phase, s.prefill_left, s.pending_rows = "", None, None
            if s.lock.locked():
                s.lock.release()
        app._peek_rate, RR.RUNNER, q, app.OWN_RATE, app.OWN_UNITS, app.OWN_BURST_RATE, \
            app.PLANNING["n"] = self.saved
        cache_peek.SOLO_RATES._q.clear()
        cache_peek.SOLO_RATES._q.extend(q)
        app.STATE.pop("pool", None)
        return False


def test_1_a_slot_in_its_end_charges_no_turn():
    with _Env() as env:
        own = 1e3 * _rows(PACKAGE_Q) / RATE
        assert app._own_eta_ms(PACKAGE_Q) == own, "an idle box: the prefill alone"
        # the class question ended and its request still holds slot 0 (its end's bookkeeping)
        assert app.LOCK.acquire(timeout=5)
        app.HOLDER["who"] = {"own": True, "rank": app.RANK_TURN, "t_hold": 0.0}
        assert app._turn_units_ms(RATE) == []
        assert app._own_eta_ms(PACKAGE_Q) == own, "a free side slot takes it; slot 0 does no GPU work"
        # a side slot held with no prompt marked (its request ends): no unit either
        env.pool.slots[1].lock.acquire()
        assert app._turn_units_ms(RATE) == [] and app._own_eta_ms(PACKAGE_Q) == own
        # the old rule: each busy slot that does not decode is a whole chunk a turn
        app.OWN_UNITS = "busy"
        chunk = 1e3 * 4096 / RATE
        assert app._turn_units_ms(RATE) == [chunk, chunk]
        old = app._own_eta_ms(PACKAGE_Q)
        assert abs(old - (own + (1 + app.OWN_ANSWER_TURNS) * 2 * chunk)) < 1e-6, old
        app.OWN_UNITS = "prefill"
        # TandemLLM's own job on slot 0 (the prepare or memory job), no request: still a chunk
        app.HOLDER["who"] = {"own": False, "rank": app.RANK_PREP, "t_hold": 0.0}
        assert app._turn_units_ms(RATE) == [chunk]
        app.HOLDER["who"] = None                     # a lock no request took
        assert app._turn_units_ms(RATE) == [chunk]


def test_2_a_prefilling_slot_charges_its_chunk():
    with _Env() as env:
        s1, s2 = env.pool.slots[1], env.pool.slots[2]
        s1.lock.acquire()
        s1.phase, s1.prefill_left = "prefill", 300
        assert app._turn_units_ms(RATE) == [300.0]
        s1.prefill_left = 10_000
        assert app._turn_units_ms(RATE) == [4096.0], "at most a chunk before its next safe point"
        s2.lock.acquire()
        s2.phase = "solo"
        RR.RUNNER = None
        assert app._turn_units_ms(RATE) == [app.OWN_ROUND_MS, 4096.0]
        turns = 1 + app.OWN_ANSWER_TURNS
        want = 1e3 * _rows(PACKAGE_Q) / RATE + turns * (app.OWN_ROUND_MS + 4096.0)
        assert abs(app._own_eta_ms(PACKAGE_Q) - want) < 1e-6


def test_3_the_solo_rate():
    with _Env(peek=PEEK):
        assert app._own_rate() == PEEK, "no request prefilled alone yet: the peek's rate"
        for _ in range(3):
            cache_peek.SOLO_RATES.note(2000, 2000.0)              # 1,000 tok/s alone
        assert app._own_rate() == 1000.0
        assert app._own_eta_ms(PACKAGE_Q) == 1e3 * _rows(PACKAGE_Q) / 1000.0
        app.OWN_RATE = "all"
        assert app._own_rate() == PEEK
        assert abs(app._own_eta_ms(PACKAGE_Q) - 1e3 * _rows(PACKAGE_Q) / PEEK) < 1e-6
        # a short prompt says nothing of a rate, as for the peek's
        app.OWN_RATE = "solo"
        cache_peek.SOLO_RATES._q.clear()
        cache_peek.SOLO_RATES.note(400, 100.0)
        assert app._own_rate() == PEEK


def test_4_alone_is_no_other_acquire():
    turn = S.Turn()
    pool = S.Pool([S.Slot(0, 1024), S.Slot(1, 1024)], turn=turn)
    turn.acquire(1)
    rec = usage.RequestRecord("r", "chat", True)
    app._alone_watch(rec, pool)
    rec.on_first()
    assert rec.prefill_alone is True
    rec = usage.RequestRecord("r", "chat", True)
    app._alone_watch(rec, pool)
    turn.release(1)
    turn.acquire(2)                                  # another slot ran on the GPU meanwhile
    rec.on_first()
    assert rec.prefill_alone is False
    turn.release(2)
    rec = usage.RequestRecord("r", "chat", True)
    app._alone_watch(rec, None)
    assert rec.prefill_alone is True and rec.on_first is None, "one slot: the lock serializes"
    # only an alone request's rate is noted for the estimate; every one for the peek
    saved = (list(cache_peek.SOLO_RATES._q), list(cache_peek.RATES._q))
    try:
        cache_peek.SOLO_RATES._q.clear()
        cache_peek.RATES._q.clear()
        for alone, ms in ((True, 2000.0), (False, 11_000.0)):
            r = usage.RequestRecord("r", "chat", False)
            r.prompt_n, r.prefill_alone = 2000, alone
            r.t_lock, r.t_first = 1.0, 1.0 + ms / 1e3
            app._account(r)
        assert [round(x) for x in cache_peek.SOLO_RATES._q] == [1000]
        assert [round(x) for x in cache_peek.RATES._q] == [1000, 182]
    finally:
        cache_peek.SOLO_RATES._q.clear()
        cache_peek.SOLO_RATES._q.extend(saved[0])
        cache_peek.RATES._q.clear()
        cache_peek.RATES._q.extend(saved[1])


def test_5_the_prefill_mark():
    with _Env() as env:
        s0 = env.pool.slots[0]
        hook = app._par_chunks(None)
        with app._prefill_mark(5000):
            assert (s0.phase, s0.prefill_left) == ("prefill", 5000)
            hook(4096, 5000)
            assert s0.prefill_left == 904
        assert (s0.phase, s0.prefill_left) == ("", None)
        s0.phase = "decode"
        with app._prefill_mark(5000):
            assert (s0.phase, s0.prefill_left) == ("decode", None)
        assert s0.phase == "decode"
        s0.phase = ""
        try:
            with app._prefill_mark(10):
                raise RuntimeError("the prefill failed")
        except RuntimeError:
            pass
        assert (s0.phase, s0.prefill_left) == ("", None)
    app.STATE.pop("pool", None)
    with app._prefill_mark(10):                      # no side slots: nothing to mark
        pass


def test_6_the_first_token_hook():
    rec = usage.RequestRecord("r", "chat", True)
    seen = []
    rec.on_first = lambda: seen.append(rec.t_first)
    assert list(rec.track(iter([1, 2, 3]))) == [1, 2, 3]
    assert len(seen) == 1 and seen[0] == rec.t_first


def test_7_a_granted_slot_charges_its_prompt_until_it_prefills():
    with _Env() as env:
        s1 = env.pool.slots[1]
        assert s1.lock.acquire(blocking=False)
        app._pending_mark(env.pool, s1, 300)             # granted, its prefill not started
        assert app._turn_units_ms(RATE) == [300.0]
        app._pending_mark(env.pool, s1, 7262)
        assert app._turn_units_ms(RATE) == [4096.0], "at most a chunk before its next safe point"
        turns = 1 + app.OWN_ANSWER_TURNS
        want = 1e3 * _rows(PACKAGE_Q) / RATE + turns * 4096.0
        assert abs(app._own_eta_ms(PACKAGE_Q) - want) < 1e-6
        app._pending_mark(env.pool, s1, 0)
        assert s1.pending_rows is None and app._turn_units_ms(RATE) == []
        # slot 0: a client request holds it, waiting for memory; then it prefills, then it ends
        assert app.LOCK.acquire(timeout=5)
        app.HOLDER["who"] = {"own": False, "rank": app.RANK_TURN, "t_hold": 0.0}
        app._pending_mark(env.pool, None, 5000)
        s0 = env.pool.slots[0]
        assert s0.pending_rows == 5000 and app._turn_units_ms(RATE) == [4096.0]
        with app._prefill_mark(5000):
            assert s0.pending_rows is None and app._turn_units_ms(RATE) == [4096.0]
        assert app._turn_units_ms(RATE) == [], "its end's bookkeeping charges nothing"
        app._pending_mark(env.pool, None, 900)           # a request that never prefills
        app._par_end(env.pool, None, "c1")
        assert s0.pending_rows is None and app._turn_units_ms(RATE) == []
        # a side slot given back clears it too
        app._pending_mark(env.pool, s1, 900)
        env.pool.release_side(s1)
        assert s1.pending_rows is None
        # the old rule charges every busy slot a whole chunk, marked or not
        assert s1.lock.acquire(blocking=False)
        app._pending_mark(env.pool, s1, 300)
        app.OWN_UNITS = "busy"
        assert app._turn_units_ms(RATE) == [4096.0, 4096.0]


def test_8_the_burst_term_keeps_the_peeks_rate():
    with _Env(peek=PEEK):
        for _ in range(3):
            cache_peek.SOLO_RATES.note(2000, 2000.0)              # 1,000 tok/s alone
        rows = _rows(PACKAGE_Q)
        own = 1e3 * rows / 1000.0
        assert app._own_eta_ms(PACKAGE_Q) == own, "no burst: no burst term"
        app.PLANNING["n"] = 8                    # the asker and 7 other chats in the router
        want = own + 1e3 * rows * 7 / PEEK
        assert abs(app._own_eta_ms(PACKAGE_Q) - want) < 1e-6, "the burst at the peek's rate"
        app.OWN_BURST_RATE = "solo"
        want = own + 1e3 * rows * 7 / 1000.0
        assert abs(app._own_eta_ms(PACKAGE_Q) - want) < 1e-6
        # no peek's rate yet: the solo rate
        app.OWN_BURST_RATE = "peek"
        app._peek_rate = lambda: None
        assert abs(app._own_eta_ms(PACKAGE_Q) - want) < 1e-6


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
