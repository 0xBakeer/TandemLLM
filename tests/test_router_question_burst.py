"""SRV-769: the router's questions never wait for the prefill burst they belong to.

Live 2026-11-01 (Flash-Next, its joint prefill on, an idle box): every new Chat-tab message waited
about 8 s. The turn is on its way to a slot (`Burst.received`) from the handler's start until it is
bound; the router's class question is a request on this server too, bound first, and a member with
a job holds while a request is on its way (`join`), up to QSE_PREFILL_BURST_WAIT_S: it held for the
turn that waited for its answer, until the router's timeout (9.4-9.7 s), and the router failed open.

The test matrix (SRV-770):

  1  before the fix (`QSE_PREFILL_BURST_ASKING=in`, no `ask`): the question holds for its own parent
     until the parent is bound (the router's timeout); the hold names it (`held_own`)
                                              test_the_question_held_for_its_own_parent
  2  `ask`: the question's parent is not on its way while the question is: it starts at once,
     no hold; the parent is on its way again after `answered`
                                              test_a_question_never_holds_for_its_parent
  3  a real burst still gathers: four turns at once, each asking its question, the questions in
     one joint forward (each held for the other turns, never for its own)
                                              test_the_questions_of_a_burst_still_gather
  4  raw requests (no parent) gather as before: eight in one joint forward
                                              test_raw_requests_gather_as_before
  5  the incoming count cannot leak: every order of received, name, ask, answered and settle
     (nested questions, a parent bound while its question runs, unknown parents) ends at 0
                                              test_the_incoming_count_cannot_leak
  7  a member holding for a turn keeps holding when the turn's question arrives (`question`: the
     count never dips between the question on its way and its turn out), and the question joins
     that member's forward
                                              test_the_question_joins_the_forward_that_held_for_its_turn
  8  the server on the fake engine (`--parallel 2`, the conductor's class question on its own port,
     `--backends`): the question finds its turn (`asked`), nothing left on its way; `in` asks nothing
                                              test_the_class_question_finds_its_turn_on_the_fake_engine

CPU. Run: python tests/test_router_question_burst.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server.prefill_burst import Burst  # noqa: E402

NOP = dict(release=lambda: None, acquire=lambda: None)


def _question(b, slot, parent, run, ask=True):
    """The router's question of turn `parent`: on its way and asked (SRV-769, server/app.py), bound on
    `slot`, at its prefill with a job. Returns (join's result, seconds in join, the way `ask` returned)."""
    way, asked = b.question(parent) if ask else (b.received(parent=parent), None)
    b.name(way, "q-" + parent)
    assert b.arrive(slot, idle=True, way=way)
    b.settle(way)
    t0 = time.monotonic()
    got = b.join(slot, "job-" + parent, run, rows=474, **NOP)
    return got, time.monotonic() - t0, asked


def test_the_question_held_for_its_own_parent():
    b = Burst(wait_s=5)
    turn = b.received()
    b.name(turn, "P")
    # the router's timeout: the turn fails open and is bound, which ends the hold
    threading.Timer(0.3, lambda: b.settle(turn)).start()
    got, took, _ = _question(b, 1, "P", lambda js: list(js), ask=False)
    assert got == (False, None) and 0.25 < took < 2.0, (got, took)
    assert b.stats["held"] == 1 and b.stats["held_own"] == 1 and b.stats["asked"] == 0
    h = b.report()["holds"][-1]
    assert h["request"] == "q-P" and h["parent"] == "P" and h["for"] == ["P"] and h["own_parent"], h
    assert b.incoming == 0 and not b.is_open()


def test_a_question_never_holds_for_its_parent():
    b = Burst(wait_s=5)
    turn = b.received()
    b.name(turn, "P")
    got, took, asked = _question(b, 1, "P", lambda js: list(js))
    assert got == (False, None) and took < 0.05, (got, took)
    assert b.stats["held"] == 0 and b.stats["held_own"] == 0 and b.stats["asked"] == 1
    assert b.incoming == 0 and b.report()["asking"] == 1
    b.answered(asked)
    assert b.incoming == 1, "the turn is on its way again"
    b.settle(turn)
    assert b.incoming == 0 and b.report()["asking"] == 0 and not b.named


def test_the_questions_of_a_burst_still_gather():
    b = Burst(wait_s=5)
    turns = {}
    for i in range(4):
        turns[i] = b.received()
        b.name(turns[i], f"P{i}")
    runs, got, lock = [], {}, threading.Lock()

    def run(jobs):
        runs.append(sorted(jobs))
        return [j.upper() for j in jobs]

    def chat(i, delay):
        time.sleep(delay)
        r, took, asked = _question(b, i, f"P{i}", run)
        b.answered(asked)
        b.settle(turns[i])                       # the turn bound after its answer
        with lock:
            got[i] = r

    th = [threading.Thread(target=chat, args=(i, 0.02 * i)) for i in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join(3)
    assert runs == [[f"job-P{i}" for i in range(4)]], runs
    assert got == {i: (True, f"JOB-P{i}") for i in range(4)}, got
    assert b.stats["held"] >= 1 and b.stats["held_own"] == 0 and b.stats["timeouts"] == 0, b.stats
    assert all(not h["own_parent"] and h["parent"] not in h["for"] for h in b.holds), list(b.holds)
    assert b.incoming == 0 and not b.is_open() and not b.named


def test_raw_requests_gather_as_before():
    b = Burst(wait_s=5)
    toks = [b.received() for _ in range(8)]
    runs, got, lock = [], {}, threading.Lock()

    def run(jobs):
        runs.append(sorted(jobs))
        return [j * 10 for j in jobs]

    def request(i, delay):
        time.sleep(delay)
        b.name(toks[i], f"cmpl-{i}")
        assert b.arrive(i, idle=True, way=toks[i])
        b.settle(toks[i])
        r = b.join(i, i + 1, run, rows=512, **NOP)
        with lock:
            got[i] = r

    th = [threading.Thread(target=request, args=(i, 0.0 if i == 0 else 0.03 + 0.002 * i)) for i in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join(3)
    assert runs == [list(range(1, 9))], runs
    assert got == {i: (True, (i + 1) * 10) for i in range(8)}
    assert b.stats["held"] >= 1 and b.stats["held_own"] == 0 and b.stats["asked"] == 0
    assert b.incoming == 0 and not b.is_open()


def test_the_incoming_count_cannot_leak():
    for seed in range(300):
        rnd = random.Random(seed)
        b = Burst()
        live, asks = [], []
        for _ in range(rnd.randint(5, 40)):
            op = rnd.random()
            if op < 0.25 or not live:
                w = b.received()
                b.name(w, f"r{len(live)}-{seed}")
                live.append(w)
            elif op < 0.5:
                w = rnd.choice(live)
                asks.append(b.ask(w.name if rnd.random() < 0.9 else "unknown"))
            elif op < 0.7 and asks:
                b.answered(asks.pop(rnd.randrange(len(asks))))
            elif op < 0.8 and asks:
                b.answered(rnd.choice(asks))         # an extra answered: no effect past zero
            else:
                b.settle(rnd.choice(live))
            assert b.incoming == sum(1 for w in live if not w.settled and w.asking == 0), seed
        for a in asks:
            b.answered(a)
        for w in live:
            b.settle(w)
        assert b.incoming == 0 and not b.ways and not b.named, seed


def test_the_question_joins_the_forward_that_held_for_its_turn():
    b = Burst(wait_s=5)
    turn = b.received()
    b.name(turn, "P")
    seen, notify = [], b.cv.notify_all

    def watch():
        seen.append(b.incoming)
        notify()

    b.cv.notify_all = watch
    raw = b.received()
    b.name(raw, "R")
    assert b.arrive(0, idle=True, way=raw)
    b.settle(raw)
    runs, got = [], {}

    def run(jobs):
        runs.append(sorted(jobs))
        return [j.upper() for j in jobs]

    th = threading.Thread(target=lambda: got.__setitem__(0, b.join(0, "job-R", run, rows=512, **NOP)))
    th.start()
    t0 = time.monotonic()
    while b.stats["held"] == 0 and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    assert b.stats["held"] == 1 and b.holds[-1]["for"] == ["P"], list(b.holds)
    del seen[:]
    q, asked = b.question("P")
    b.name(q, "q-P")
    time.sleep(0.2)                              # a member that saw nobody on its way would go alone
    assert not runs and b.incoming == 1
    assert b.arrive(1, idle=False, way=q), "the burst is still open"
    assert seen and min(seen) >= 1, ("the count dipped between the question and its turn", seen)
    b.settle(q)
    got[1] = b.join(1, "job-P", run, rows=474, **NOP)
    th.join(3)
    assert runs == [["job-P", "job-R"]], runs
    assert got == {0: (True, "JOB-R"), 1: (True, "JOB-P")}, got
    assert b.stats["joint"] == 1 and b.stats["held_own"] == 0 and b.stats["timeouts"] == 0, b.stats
    b.answered(asked)
    b.settle(turn)
    assert b.incoming == 0 and b.report()["asking"] == 0 and not b.named and not b.is_open()


def test_the_class_question_finds_its_turn_on_the_fake_engine():
    import subprocess
    from test_conductor_e2e import LAUNCH, ROOT, _call, _env, _stream, _wait
    for mode in ("out", "in"):
        d = tempfile.mkdtemp(prefix="srv769-e2e-")
        env = dict(_env(), QSE_PREFILL_BURST="joint", QSE_PREFILL_BURST_ASKING=mode)
        srv = subprocess.Popen([sys.executable, "-c", LAUNCH, "--fake-engine", "--port", "0",
                                "--served-model", "qwen38-spark-engine",
                                "--usage-ledger", os.path.join(d, "ledger.sqlite3"),
                                "--fake-tps", "2000", "--parallel", "2", "--side-max-len", "8192",
                                "--conductor-config", os.path.join(d, "conductor"),
                                "--backends", os.path.join(d, "config.sqlite3")], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            port = _wait(None, srv)
            st, _, got = _call(port, "GET", "/v1/conductor/router")
            r = got["doc"]
            r["classifier"]["stages"] = [{"kind": "decider", "enabled": False, "accept": 0.8,
                                          "timeout_ms": 400},
                                         {"kind": "qwen", "enabled": True, "accept": 0.45,
                                          "timeout_ms": 8000}]
            st, _, w = _call(port, "PUT", "/v1/conductor/router",
                             {"doc": r, "note": "qwen first", "base_version": got["version"]})
            assert st == 200, w
            msgs = [{"role": "user", "content": "Fix the off-by-one in server/slots.py take_side "
                                                "and add a unit test for it."}]
            _stream(port, msgs, headers={"X-Tandem-Profile": "coding", "x-session-id": f"srv769-{mode}"})
            st, _, h = _call(port, "GET", "/health")
            b = h["parallel"]["prefill_burst"]
            assert b["incoming"] == 0 and b["asking"] == 0 and b["held_own"] == 0, (mode, b)
            if mode == "out":
                assert b["asked"] >= 1, ("the class question found its turn", b)
            else:
                assert b["asked"] == 0, (mode, b)
        finally:
            srv.kill()
            srv.wait(10)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
