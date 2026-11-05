"""VIS-119: the router's questions before a new user message, one after the other, each a prefill
of its own (2026-10-31, live fd627ac3, an idle box): the class question 0.63-1.37 s, the package
question 2.18-3.11 s, the library question in shadow 1.32 s, all inside the turn's queue time; the
same session and text again (class sticky) still queued 1.41 s for the library question alone.
Asked at once they do not hide each other's time: the GPU serves the prefills one after the
other (two class questions at once 1.32-1.44 s against 0.67-0.69 s alone, class and package
together 2.84-2.96 s against 3.0-3.4 s in series, with the class answer at 2.84 s instead of
0.65 s; 2026-11-01, through live :8000), so the fix asks fewer questions on the request's path.

The test matrix:

  1  `QWEN38_PACKAGE_HOLD=answer`: a follow-up the class question keeps in a class whose package
     answer holds (`class`) holds an answer of none too: no package question (`how: sticky_none`);
     `pick` (the default) asks again, as FX15 did
                                            test_a_none_holds_for_the_class, test_pick_asks_again
  2  under `answer`: a turn with no answer (busy, a timeout) holds nothing; a new class asks
     again; a class with `message` asks on every message
                                            test_no_answer_and_a_new_class_ask_again
  3  `QWEN38_LIBRARY_HOLD=class`: the library question on such a class: asked once per class and
     session, its answer held (picks and none) for the next message of the class, nothing added
     twice; `message` (the default) asks again; a class with `message` asks again
                                            test_the_library_answer_holds_for_the_class
  4  a shadow library question that goes to Qwen on this server's own port is not asked before
     the queue: the row says `deferred`, the plan carries the job; after the turn, once the server
     is idle, it is asked once, its answer lands in the session's picks, the row and the journal
     on disk                                test_a_shadow_question_waits_for_the_turn
  5  a busy server holds the job; no idle moment within the wait drops it, journaled
                                            test_a_busy_server_holds_then_drops
  6  `QWEN38_SHADOW_ASK=inline` asks before the queue as before; `off` asks nothing; `on`, a
     decider first, the test box (dry) and a raw request ask in place or not at all, as before
                                            test_modes_and_what_stays_in_place
  7  `Router.own_port_first`: the first stage the cascade would ask, resolved through the
     proxy's stage override as the cascade resolves it (`--backends`): None (the server's own
     requests) keeps the configured adapter, a relayed backend's adapter is not this server's
     queue, False skips the stage           test_own_port_first
  8  the worker: a full queue drops at once, no idle signal waits the idle time from the job's
     arrival, a failing job does not stop the next and its row says `failed`
                                                              test_the_worker
  9  `Conductor.finish` hands the plan's jobs to the worker after the row is written; the
     server's idle time is None while anything runs, waits, plans or holds the lock, and starts
     at the last request's or side-lane job's end
                                            test_finish_and_the_servers_idle_time
 10  the real server on the fake engine, started with `--backends` (the proxy in the process, as
     live runs): the row says `deferred` at the turn's end, and the question goes to the
     server's own port once it is idle (2 s after the stream ended), through the proxy's auth
     (the stage got a 200: `qwen_unknown_label`), its answer in the same row
                                            test_end_to_end_on_the_fake_engine
 11  a turn that ends in tool calls drops its jobs unasked (`deferred.dropped: tool_calls`): the
     client's next step follows at once    test_a_tool_calls_turn_drops_its_jobs
 12  the defaults: SHADOW_ASK idle, PACKAGE_HOLD pick, LIBRARY_HOLD message, in the registry, the
     standalone table and the modules      test_the_defaults
 13  a turn that failed open (SRV-763: the class question could not be answered in time) asks no
     library question, after the turn neither                test_a_busy_turn_defers_nothing
 14  the same message again before its job ran (a regenerate): the first job asks, the second
     takes its answer from the session (`cached`)  test_a_repeat_inside_the_idle_time_asks_once

CPU. Run: python tests/test_router_questions_serial.py
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_class_core import PAGODA, make  # noqa: E402
from test_conductor_router import COMPOSITE, TOOLS, turn  # noqa: E402
from test_libsearch import SKILLS  # noqa: E402
from test_library_router import NEW_TASK, decider, lib_cond, lib_notes  # noqa: E402
from test_package_sticky import Script, _make, _session  # noqa: E402
from test_snapshots import S, served  # noqa: E402

from server import app  # noqa: E402
from server.conductor import libsearch  # noqa: E402
try:
    from server.conductor import deferred  # noqa: E402
except ImportError:            # a tree before VIS-119: each test that needs it fails alone
    deferred = None
from server.conductor import router as router_mod  # noqa: E402
from server.conductor.classify import Answer  # noqa: E402
from server.conductor.decisions import Journal  # noqa: E402


class _Knob:
    """A module constant set for a block, put back after it."""

    def __init__(self, mod, name, value):
        self.mod, self.name, self.value = mod, name, value

    def __enter__(self):
        self.old = getattr(self.mod, self.name)
        setattr(self.mod, self.name, self.value)

    def __exit__(self, *a):
        setattr(self.mod, self.name, self.old)


def _pkg_asked(q) -> int:
    """The three_d package questions the Qwen fake got."""
    return sum(1 for _, _, o in q.calls if "photo-to-3d" in o)


def _lib_asked(fake) -> int:
    """The library questions a stage got (its options: library skills and none)."""
    return sum(1 for _, o in fake.calls if "none" in o and set(o) - {"none"} <= set(SKILLS))


class Flaky:
    """The Qwen stage of `test_package_sticky.Script` whose package answers come from `errors`
    first (an error string fails that call, None answers as the script does)."""

    def __init__(self, script, errors=()):
        self.script, self.errors, self.calls = script, list(errors), []

    def choice(self, state, prompt, options, max_chars=None):
        self.calls.append((state, prompt, dict(options)))
        if "none" in options and self.errors:
            err = self.errors.pop(0)
            if err is not None:
                return Answer("qwen", error=err)
        label, probs = self.script(state, options, prompt)
        return Answer("qwen", label, probs[label], dict(probs), 1.0, model="fake")


def _rec():
    return types.SimpleNamespace(status=200, finish_reason="stop", prompt_tokens=100,
                                 completion_tokens=10, tool_calls=0, tool_names=[],
                                 reasoning_tokens=0, ttft_ms=50.0, total_ms=100.0,
                                 states=None, thoughts=None, voice=None)


# ------------------------------------------------------------------ 1
def test_a_none_holds_for_the_class():
    with _Knob(router_mod, "PACKAGE_HOLD", "answer"):
        s = Script(pkg=("none", 0.9))
        cond, q = _make(s)
        plans = _session(cond, "s-v119-1", [PAGODA, "Now build it with voxels.",
                                            "Add a koi pond."])
    first = plans[0].row["extra"]["package_choice"]
    assert first["chosen"] is None and first["how"] == "none", first
    for p in plans[1:]:
        ch = p.row["extra"]["package_choice"]
        assert p.row["class_id"] == "three_d" and ch["chosen"] is None, ch
        assert ch["how"] == "sticky_none", ch
    assert _pkg_asked(q) == 1, "the package question ran again after a none in the same class"


def test_pick_asks_again():
    with _Knob(router_mod, "PACKAGE_HOLD", "pick"):
        s = Script(pkg=("none", 0.9))
        cond, q = _make(s)
        _session(cond, "s-v119-1b", [PAGODA, "Now build it with voxels."])
    assert _pkg_asked(q) == 2
    # a pick holds under both
    for hold in ("pick", "answer"):
        with _Knob(router_mod, "PACKAGE_HOLD", hold):
            cond, q = _make(Script())
            plans = _session(cond, f"s-v119-1c-{hold}", [PAGODA, "Make the pagoda taller."])
        assert plans[1].row["extra"]["package_choice"]["how"] == "sticky" and _pkg_asked(q) == 1


# ------------------------------------------------------------------ 2
def test_no_answer_and_a_new_class_ask_again():
    with _Knob(router_mod, "PACKAGE_HOLD", "answer"):
        _no_answer_and_a_new_class()


def _no_answer_and_a_new_class():
    # a timeout: nothing holds, the next message of the class asks again (and its none holds)
    s = Script(pkg=("none", 0.9))
    cond, _ = _make(s)
    q = Flaky(s, errors=["qwen_timeout"])
    cond.router.factory = lambda kind, role, timeout: q if kind == "qwen" else None
    cond.router._adapters.clear()
    plans = _session(cond, "s-v119-2", [PAGODA, "Now build it with voxels.", "Add a koi pond."])
    hows = [p.row["extra"]["package_choice"]["how"] for p in plans]
    assert hows == ["no_answer", "none", "sticky_none"], hows
    assert _pkg_asked(q) == 2
    # a none held on three_d; the session goes to chat and back: three_d is new again, asked
    s = Script(pkg=("none", 0.9))
    cond, q = _make(s)
    _session(cond, "s-v119-2b", [PAGODA])
    msgs = [{"role": "user", "content": PAGODA}, {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Unrelated: why is Canberra the capital of Australia?"}]
    s.cls[0] = "chat"
    plan, _, _ = turn(cond, {"model": COMPOSITE, "tools": TOOLS, "messages": msgs},
                      headers={"x-session-id": "s-v119-2b"})
    assert plan.row["class_id"] == "chat"
    s.cls[0] = "three_d"
    msgs += [{"role": "assistant", "content": "A compromise."},
             {"role": "user", "content": "Back to the garden: make it winter."}]
    n = _pkg_asked(q)
    plan, _, _ = turn(cond, {"model": COMPOSITE, "tools": TOOLS, "messages": msgs},
                      headers={"x-session-id": "s-v119-2b"})
    ch = plan.row["extra"]["package_choice"]
    assert ch["how"] == "none" and _pkg_asked(q) == n + 1, ch
    # a class without `package_sticky` (`message`, the coding classes) asks on every message
    cond, q = make(qwen_fn=Script(pkg=("none", 0.9)))
    _session(cond, "s-v119-2c", [PAGODA, "Now build it with voxels.", "Add a koi pond."])
    assert _pkg_asked(q) == 3


# ------------------------------------------------------------------ 3
def _sticky_code_edit(cond, mode="class"):
    _, r = cond.store.get("router")
    r = copy.deepcopy(r)
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"]["package_sticky"] = mode
    cond.store.put("router", None, r, note="code_edit holds its answers", author="t")


def _two_messages(cond, sid="ses-snap"):
    s = S()
    p1, _ = served(cond, list(s.msgs), sid=sid)
    msgs = s.step(("bash", {"command": "ls"}, "a")) + [
        {"role": "assistant", "content": "The game runs."}, {"role": "user", "content": NEW_TASK}]
    p2, _ = served(cond, msgs, sid=sid)
    return p1, p2


def test_the_library_answer_holds_for_the_class():
    with _Knob(libsearch, "LIBRARY_HOLD", "class"):
        _library_holds()
    # the default `message`: asked again
    cond, fakes = lib_cond("on", picks=("browser-game", "pandas-cleaning"))
    _sticky_code_edit(cond)
    p1, p2 = _two_messages(cond)
    assert "held" not in p2.row["extra"]["library"] and _lib_asked(fakes["decider"]) == 2


def _library_holds():
    cond, fakes = lib_cond("on", picks=("browser-game", "pandas-cleaning"))
    _sticky_code_edit(cond)
    p1, p2 = _two_messages(cond)
    assert lib_notes(p1) == ["browser-game"] and _lib_asked(fakes["decider"]) == 1
    lib = p2.row["extra"]["library"]
    assert lib.get("held") == "class" and lib["chosen"] == ["browser-game"], lib
    assert lib_notes(p2) == [], "the held skill is live: not added twice"
    assert _lib_asked(fakes["decider"]) == 1, "the library question ran again in the class"
    # a none holds as well
    cond, fakes = lib_cond("on", picks=())
    _sticky_code_edit(cond)
    p1, p2 = _two_messages(cond)
    assert p1.row["extra"]["library"]["how"] == "none" and lib_notes(p1) == []
    assert p2.row["extra"]["library"].get("held") == "class" and _lib_asked(fakes["decider"]) == 1
    # `message`, and a class with `message`: asked again, the new pick goes in (R1 test 4)
    for knob, sticky in (("message", "class"), ("class", "message")):
        with _Knob(libsearch, "LIBRARY_HOLD", knob):
            cond, fakes = lib_cond("on", picks=("browser-game", "pandas-cleaning"))
            _sticky_code_edit(cond, sticky)
            p1, p2 = _two_messages(cond)
        assert "held" not in p2.row["extra"]["library"]
        assert lib_notes(p2) == ["pandas-cleaning"] and _lib_asked(fakes["decider"]) == 2, \
            (knob, sticky)


# ------------------------------------------------------------------ 4
def _qwen_first(cond, fakes, picks=("browser-game",)):
    """The live router's stages: the decider off, Qwen on this server's own port first."""
    _, r = cond.store.get("router")
    r = copy.deepcopy(r)
    r["classifier"]["stages"] = [{"kind": "decider", "enabled": False, "accept": 0.8,
                                  "timeout_ms": 400},
                                 {"kind": "qwen", "enabled": True, "accept": 0.45,
                                  "timeout_ms": 8000}]
    cond.store.put("router", None, r, note="qwen first", author="t")
    fakes["qwen"].fn = decider(picks)
    fakes["qwen"].local = True              # the server's own port, as `Router._make` builds it
    cond.router._adapters.clear()


def _shadow(picks=("browser-game",), journal=None):
    cond, fakes = lib_cond("shadow", picks=picks)
    _qwen_first(cond, fakes, picks)
    if journal is not None:
        cond.journal = cond.router.journal = journal
    cond.router.deferred = deferred.Deferred(signal=lambda: cond.router.own_idle, settle_s=0.0,
                                             poll_s=0.01)
    return cond, fakes


def test_a_shadow_question_waits_for_the_turn():
    j = Journal(os.path.join(tempfile.mkdtemp(prefix="v119-j-"), "decisions.sqlite3"))
    cond, fakes = _shadow(journal=j)
    idle = {"s": None}
    cond.router.own_idle = lambda: idle["s"]
    plan, ids = served(cond, list(S().msgs))
    lib = plan.row["extra"]["library"]
    assert lib["how"] == "deferred" and lib["chosen"] == [], lib
    assert _lib_asked(fakes["qwen"]) == 0, "the shadow question ran before the queue"
    assert len(plan.deferred) == 1 and "library" not in (plan.row.get("shadow") or {})
    # the prompt is the one `off` serves
    off, _ = lib_cond("off")
    _qwen_first(off, _)
    _, ids_off = served(off, list(S().msgs))
    assert ids == ids_off
    cond.finish(plan, _rec())
    assert plan.deferred == [], "the jobs went to the worker"
    time.sleep(0.1)
    assert _lib_asked(fakes["qwen"]) == 0, "asked while the server was busy"
    idle["s"] = 30.0
    assert cond.router.deferred.flush(5.0)
    assert _lib_asked(fakes["qwen"]) == 1
    lib = plan.row["extra"]["library"]
    assert lib["chosen"] == ["browser-game"] and lib["how"] == "qwen", lib
    assert lib["deferred"]["waited_ms"] >= 100 and lib["stages"], lib
    assert plan.row["shadow"]["library"] == {"chosen": ["browser-game"], "trigger": "message"}
    # the journal row on disk has the answer (the update came after the row's write)
    rid = plan.row["id"]
    j.flush()
    with j._lock:
        j.ring.pop(rid, None)
    on_disk = j.get(rid)
    assert on_disk["extra"]["library"]["chosen"] == ["browser-game"], on_disk["extra"]["library"]
    assert on_disk["shadow"]["library"]["chosen"] == ["browser-game"]
    assert on_disk["outcome"]["status"] == 200, "the turn's own fields stay"
    # the session holds the answer: the same message again asks nothing
    p2, _ = served(cond, list(S().msgs))
    assert p2.row["extra"]["library"].get("cached") and _lib_asked(fakes["qwen"]) == 1
    assert p2.deferred == []


# ------------------------------------------------------------------ 5
def test_a_busy_server_holds_then_drops():
    cond, fakes = _shadow()
    cond.router.deferred.max_wait_s = 0.3
    cond.router.own_idle = lambda: None                    # always busy
    plan, _ = served(cond, list(S().msgs))
    cond.finish(plan, _rec())
    assert cond.router.deferred.flush(5.0)
    assert _lib_asked(fakes["qwen"]) == 0
    assert plan.row["extra"]["library"]["deferred"] == {"dropped": "busy"}
    assert cond.router.deferred.stats["dropped_busy"] == 1
    # the next message asks (nothing was held), after the turn again
    cond.router.own_idle = lambda: 10.0
    p2, _ = served(cond, list(S().msgs))
    assert p2.row["extra"]["library"]["how"] == "deferred"
    cond.finish(p2, _rec())
    assert cond.router.deferred.flush(5.0) and _lib_asked(fakes["qwen"]) == 1


# ------------------------------------------------------------------ 6
def test_modes_and_what_stays_in_place():
    with _Knob(deferred, "SHADOW_ASK", "inline"):
        cond, fakes = _shadow()
        plan, _ = served(cond, list(S().msgs))
    lib = plan.row["extra"]["library"]
    assert lib["how"] == "qwen" and lib["chosen"] == ["browser-game"] and not plan.deferred
    assert _lib_asked(fakes["qwen"]) == 1
    with _Knob(deferred, "SHADOW_ASK", "off"):
        cond, fakes = _shadow()
        plan, _ = served(cond, list(S().msgs))
    assert plan.row["extra"]["library"]["how"] == "shadow_off" and not plan.deferred
    assert _lib_asked(fakes["qwen"]) == 0
    # `on` changes the prompt: asked in place
    cond, fakes = lib_cond("on")
    _qwen_first(cond, fakes)
    plan, _ = served(cond, list(S().msgs))
    assert lib_notes(plan) == ["browser-game"] and not plan.deferred
    # a decider first (tens of ms): in place (R1's shadow test keeps its row)
    cond, fakes = lib_cond("shadow")
    plan, _ = served(cond, list(S().msgs))
    assert plan.row["extra"]["library"]["how"] == "decider" and not plan.deferred
    # the test box (dry): in place
    cond, fakes = _shadow()
    out = cond.router.explain(S().msgs[-1]["content"], "coding",
                              messages=list(S().msgs))
    assert out["extra"]["library"]["chosen"] == ["browser-game"], out["extra"]["library"]
    # a raw request: no plan, no job
    plan = cond.plan({"model": COMPOSITE, "messages": list(S().msgs), "tools": TOOLS},
                     {"X-Tandem": "raw"}, client_kind="opencode", request_id="raw", chat=True)
    assert not plan.active and plan.deferred == []


# ------------------------------------------------------------------ 7
def test_own_port_first():
    cond, fakes = _shadow()
    _, r = cond.store.get("router")
    _, roles = cond.store.get("roles")
    rt = cond.router
    # the Qwen stage as `Router._make` builds it: local when the role's endpoint is `self`
    rt.factory = lambda kind, role, timeout: types.SimpleNamespace(
        local=role.get("endpoint") == "self") if kind == "qwen" else fakes[kind]
    own = dict(roles, classifier=dict(roles["classifier"], endpoint="self"))
    assert rt.own_port_first(r, own)
    assert not rt.own_port_first(r, dict(roles, classifier=dict(roles["classifier"],
                                                                endpoint="http://other:8000")))
    assert not rt.own_port_first(r, dict(own, classifier=dict(own["classifier"], mode="off")))
    r2 = copy.deepcopy(r)
    r2["classifier"]["stages"][0]["enabled"] = True        # the decider first
    assert not rt.own_port_first(r2, own)
    # the decider role off: it answers role_off at once, Qwen is the first asked
    assert rt.own_port_first(r2, dict(own, decider=dict(own["decider"], mode="off")))
    # an adapter's own `local` decides, whatever the role's endpoint says
    local = types.SimpleNamespace(local=True)
    rt.factory = lambda kind, role, timeout: local if kind == "qwen" else fakes[kind]
    assert rt.own_port_first(r, dict(roles, classifier=dict(roles["classifier"],
                                                            endpoint="http://x")))
    rt.factory = lambda kind, role, timeout: types.SimpleNamespace(
        local=role.get("endpoint") == "self") if kind == "qwen" else fakes[kind]
    # the proxy's override (`--backends`, live): None for this server's own requests, the Chat
    # tab's among them, keeps the configured stage: asked after the turn
    rt.stage_override = lambda st, roles: None
    assert rt.own_port_first(r, own)
    assert not rt.own_port_first(r, dict(roles, classifier=dict(roles["classifier"],
                                                                endpoint="http://other:8000")))
    # a request relayed to a backend: its adapter (no `local`) is not this server's queue
    relayed = types.SimpleNamespace(choice=lambda *a, **k: None)
    rt.stage_override = lambda st, roles: relayed if st["kind"] == "qwen" else None
    assert not rt.own_port_first(r, own)
    # False skips the stage, as the cascade skips it: the decider switched off there, Qwen next
    rt.stage_override = lambda st, roles: False if st["kind"] == "decider" else None
    assert rt.own_port_first(r2, own)
    rt.stage_override = lambda st, roles: False            # the harness's class question off
    assert not rt.own_port_first(r2, own)


# ------------------------------------------------------------------ 8
def test_the_worker():
    ran, dropped = [], []

    def job(i, fail=False):
        def run(w):
            if fail:
                raise RuntimeError("boom")
            ran.append((i, w))
        return {"run": run, "drop": lambda why: dropped.append((i, why))}

    gate = threading.Event()
    w = deferred.Deferred(signal=lambda: (lambda: 10.0 if gate.is_set() else None),
                          settle_s=0.0, poll_s=0.01)
    plan = types.SimpleNamespace(deferred=[job(i) for i in range(deferred.QUEUE_MAX + 3)])
    queued = w.submit(plan)
    # the thread may have taken the first job off the queue already: one more fits then
    assert queued >= deferred.QUEUE_MAX and plan.deferred == []
    assert w.stats["dropped_full"] == deferred.QUEUE_MAX + 3 - queued
    assert [why for _, why in dropped] == ["full"] * w.stats["dropped_full"]
    gate.set()
    assert w.flush(5.0) and len(ran) == queued
    # no idle signal: the idle time counts from the job's arrival
    w2 = deferred.Deferred(signal=lambda: None, settle_s=0.2, poll_s=0.01)
    t0 = time.monotonic()
    w2.submit(types.SimpleNamespace(deferred=[job("late")]))
    assert w2.flush(5.0) and time.monotonic() - t0 >= 0.2
    assert ran[-1][0] == "late" and ran[-1][1] >= 200
    # a failing job: counted, its row says so, the next runs
    w3 = deferred.Deferred(signal=lambda: None, settle_s=0.0, poll_s=0.01)
    w3.submit(types.SimpleNamespace(deferred=[job("bad", fail=True), job("good")]))
    assert w3.flush(5.0) and w3.stats["failed"] == 1 and ran[-1][0] == "good"
    assert dropped[-1] == ("bad", "failed"), dropped
    assert deferred.mode() in ("idle", "inline", "off")
    with _Knob(deferred, "SHADOW_ASK", "sometimes"):
        assert deferred.mode() == "idle"


# ------------------------------------------------------------------ 9
def test_finish_and_the_servers_idle_time():
    cond, _ = _shadow()
    got = []
    cond.router.deferred = types.SimpleNamespace(submit=lambda p: got.append(list(p.deferred)))
    plan, _ = served(cond, list(S().msgs))
    assert len(plan.deferred) == 1
    cond.finish(plan, _rec())
    assert got and got[0] == plan.deferred and plan.row.get("outcome", {}).get("status") == 200
    # the server's idle time
    saved = {k: app.INFLIGHT[k] for k in ("waiting", "running", "lane")}
    lane_end = app.STATE.pop("lane_end_mono", None)
    try:
        for k in saved:
            app.INFLIGHT[k] = 0
        app.WAITERS.clear()
        app.PLANNING["n"] = 0
        app.STATE["last_end_mono"] = time.monotonic() - 7.0
        assert app._own_idle_s() >= 7.0
        # a side-lane job that just ended used the GPU: the idle time starts at its end
        app.STATE["lane_end_mono"] = time.monotonic() - 1.0
        assert 1.0 <= app._own_idle_s() < 7.0
        app.STATE["lane_end_mono"] = time.monotonic() - 9.0
        assert app._own_idle_s() < 9.0
        for k in ("waiting", "running", "lane"):
            app.INFLIGHT[k] = 1
            assert app._own_idle_s() is None, k
            app.INFLIGHT[k] = 0
        app.PLANNING["n"] = 1
        assert app._own_idle_s() is None
        app.PLANNING["n"] = 0
        app.WAITERS.append({"own": True})
        assert app._own_idle_s() is None
        app.WAITERS.clear()
        assert app.LOCK.acquire(timeout=1)
        try:
            assert app._own_idle_s() is None
        finally:
            app.LOCK.release()
        assert app._own_idle_s() >= 7.0
    finally:
        app.INFLIGHT.update(saved)
        app.WAITERS.clear()
        app.PLANNING["n"] = 0
        app.STATE.pop("lane_end_mono", None)
        if lane_end is not None:
            app.STATE["lane_end_mono"] = lane_end


# ------------------------------------------------------------------ 10
def test_end_to_end_on_the_fake_engine():
    """The real server on the fake engine (CPU, its own port, config and ledger): a coding turn
    whose library question is in shadow ends with `how: deferred` in its row; once the server
    is idle the question goes to the server's own port and its answer lands in the same row (the
    fake writes no option id: `no_answer`, the stage's `qwen_unknown_label`)."""
    import subprocess
    from test_conductor_e2e import LAUNCH, ROOT, _call, _env, _stream, _wait
    d = tempfile.mkdtemp(prefix="v119-e2e-")
    cdir = os.path.join(d, "conductor")
    ledger = os.path.join(d, "ledger.sqlite3")
    # its own proxy config in the temp dir (`--backends`, as live starts): the proxy's stage
    # override is set, and answers None for the engine's own requests
    env = dict(_env(), QWEN38_SHADOW_ASK_IDLE_S="2")
    srv = subprocess.Popen([sys.executable, "-c", LAUNCH, "--fake-engine", "--port", "0",
                            "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
                            "--fake-tps", "2000", "--conductor-config", cdir,
                            "--backends", os.path.join(d, "config.sqlite3")], cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        port = _wait(None, srv)
        st, _, cfg = _call(port, "GET", "/v1/config")
        assert st == 200 and cfg["engine"]["embedded"], "the proxy is on in the process"
        lib = os.path.join(ROOT, "packages", "library")
        names = [n for n in ("browser-game", "canvas-2d")
                 if os.path.exists(os.path.join(lib, n, "SKILL.md"))]
        assert names, "the shipped library"
        st, _, got = _call(port, "GET", "/v1/conductor/packages")
        doc = got["doc"]
        doc["skill_dirs"] = list(doc.get("skill_dirs") or []) + [lib]
        for n in names:
            doc["packages"][n] = {"title": n, "kind": "skill", "source": f"skill:{n}",
                                  "description": f"{n.replace('-', ' ')}: a 2D game on a canvas",
                                  "body": "", "max_tokens": 500, "enabled": True,
                                  "library": True}
        st, _, w = _call(port, "PUT", "/v1/conductor/packages",
                         {"doc": doc, "note": "the library", "base_version": got["version"]})
        assert st == 200, w
        st, _, got = _call(port, "GET", "/v1/conductor/router")
        r = got["doc"]
        r["library"] = {"index": os.path.join(lib, "INDEX.md"), "class_filter": False}
        r["classifier"]["stages"] = [{"kind": "decider", "enabled": False, "accept": 0.8,
                                      "timeout_ms": 400},
                                     {"kind": "qwen", "enabled": True, "accept": 0.45,
                                      "timeout_ms": 8000}]
        st, _, w = _call(port, "PUT", "/v1/conductor/router",
                         {"doc": r, "note": "qwen first", "base_version": got["version"]})
        assert st == 200, w
        st, _, got = _call(port, "GET", "/v1/conductor/profiles/coding")
        p = got["doc"]
        p["library"] = "shadow"
        st, _, w = _call(port, "PUT", "/v1/conductor/profiles/coding",
                         {"doc": p, "note": "library shadow", "base_version": got["version"]})
        assert st == 200, w
        msgs = [{"role": "user", "content": "Build a 2D browser game on a canvas: a game loop, "
                                            "keyboard input, tile collisions and a HUD."}]
        t0 = time.monotonic()
        _stream(port, msgs, headers={"X-Tandem-Profile": "coding", "x-session-id": "v119"},
                tools=True)
        t_done = time.monotonic()
        st, _, got = _call(port, "GET", "/v1/conductor/decisions?session=c:v119")
        row = got["decisions"][0]
        info = row["extra"]["library"]
        assert info["mode"] == "shadow" and info["candidates"], info
        # not asked before the queue: `deferred` at the turn's end, or its answer already landed
        assert info["how"] == "deferred" or "deferred" in info, info
        if time.monotonic() - t0 < 1.5:
            # well inside the idle time: the question has not been asked yet
            assert info["how"] == "deferred" and "deferred" not in info, info
        end = time.monotonic() + 30
        while time.monotonic() < end:
            st, _, one = _call(port, "GET", f"/v1/conductor/decisions/{row['id']}")
            info = one["decision"]["extra"]["library"]
            if "deferred" in info:
                break
            time.sleep(0.25)
        # asked once the server had been idle for its 2 s: the server's idle clock starts at the
        # stream's end, a little before the job's own clock (`waited_ms`) starts at `finish`
        assert time.monotonic() - t_done >= 1.8, info
        assert info.get("deferred", {}).get("waited_ms", 0) >= 1500, info
        # the question reached the server's own port through the proxy and got an answer (200):
        # the fake writes no option id, so the stage's error is the label, not the call's
        assert info["how"] == "no_answer" and info["stages"], info
        assert all(s[0]["stage"] == "qwen" and s[0].get("error") == "qwen_unknown_label"
                   for s in info["stages"]), info["stages"]
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()



# ------------------------------------------------------------------ 11
def test_a_tool_calls_turn_drops_its_jobs():
    cond, fakes = _shadow()
    cond.router.own_idle = lambda: 30.0
    plan, _ = served(cond, list(S().msgs))
    assert len(plan.deferred) == 1
    rec = _rec()
    rec.finish_reason, rec.tool_calls = "tool_calls", 1
    cond.finish(plan, rec)
    assert plan.deferred == [] and cond.router.deferred.flush(5.0)
    assert cond.router.deferred.stats["queued"] == 0
    assert cond.router.deferred.stats["dropped_tool_calls"] == 1
    assert plan.row["extra"]["library"]["deferred"] == {"dropped": "tool_calls"}
    time.sleep(0.05)
    assert _lib_asked(fakes["qwen"]) == 0, "asked between the turn's tool steps"
    # a turn that stops: submitted and asked
    p2, _ = served(cond, list(S().msgs))
    cond.finish(p2, _rec())
    assert cond.router.deferred.flush(5.0) and _lib_asked(fakes["qwen"]) == 1


# ------------------------------------------------------------------ 12
def test_the_defaults():
    from engine.settings import KNOBS
    from server import knobs
    want = {"SHADOW_ASK": "idle", "SHADOW_ASK_IDLE_S": "5", "PACKAGE_HOLD": "pick",
            "LIBRARY_HOLD": "message"}
    for k, v in want.items():
        assert KNOBS[k][0] == v and knobs.DEFAULTS[k] == v, (k, KNOBS[k], knobs.DEFAULTS.get(k))
    tex = open(os.path.join(os.path.dirname(HERE), "manual", "generated", "engine-knobs.tex"),
               encoding="utf-8").read()
    for k, v in want.items():
        name = ("QWEN38_" + k).replace("_", r"\_\allowbreak{}")
        assert f"\\texttt{{{name}}} & \\texttt{{{v}}}" in tex, k
    env = {k: os.environ.get("QWEN38_" + k) for k in want}
    if not any(env.values()):
        assert router_mod.PACKAGE_HOLD == "pick" and libsearch.LIBRARY_HOLD == "message"
        assert deferred.SHADOW_ASK == "idle" and deferred.SHADOW_ASK_IDLE_S == 5.0


# ------------------------------------------------------------------ 13
def test_a_busy_turn_defers_nothing():
    cond, fakes = _shadow()
    cond.router.own_eta = lambda texts, max_new: 20_000.0     # the class question: qwen_busy
    plan, _ = served(cond, list(S().msgs))
    assert plan.fail_open, plan.row.get("stages")
    lib = plan.row["extra"]["library"]
    assert lib["how"] == "fail_open" and plan.deferred == [], lib
    assert _lib_asked(fakes["qwen"]) == 0


# ------------------------------------------------------------------ 14
def test_a_repeat_inside_the_idle_time_asks_once():
    """The same message again before its job ran (a regenerate): both turns queue a job, the
    first asks, the second takes its answer from the session."""
    cond, fakes = _shadow()
    idle = {"s": None}
    cond.router.own_idle = lambda: idle["s"]
    p1, _ = served(cond, list(S().msgs))
    p2, _ = served(cond, list(S().msgs))
    assert len(p1.deferred) == 1 and len(p2.deferred) == 1
    cond.finish(p1, _rec())
    cond.finish(p2, _rec())
    idle["s"] = 30.0
    assert cond.router.deferred.flush(5.0)
    assert _lib_asked(fakes["qwen"]) == 1
    a, b = p1.row["extra"]["library"], p2.row["extra"]["library"]
    assert a["how"] == "qwen" and a["chosen"] == ["browser-game"] and not a.get("cached"), a
    assert b["how"] == "qwen" and b["chosen"] == ["browser-game"] and b.get("cached"), b
    assert "waited_ms" in b["deferred"] and "stages" not in b, b
    assert p2.row["shadow"]["library"] == {"chosen": ["browser-game"], "trigger": "message"}


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
