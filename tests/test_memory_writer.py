"""The background memory writer (2026-10-05, P-M2, server/conductor/memwriter.py and
`app.memory_once`), on the random 4-layer model as the preparer's tests (CPU torch).

  1  idle and due (the session's end): the writer's hidden turn runs in scratch rows, its blocks
     are written as `qwen-writer` provisional notes, and the conversation's rows, anchors and end
     anchor are as they were (the next turn resumes at the end and writes the same tokens)
                                                        test_the_writer_writes_and_leaves_the_rows
  2  a request that waits stops it; nothing is written and nothing moves
                                                        test_a_waiting_request_stops_the_writer
  3  not due: too few new tokens since the last pass, mode off, no session seen
                                                        test_not_due
  4  the gate: Laya's no skips the turn (journaled), a mark (compaction, give_up) makes it due
     after `due_idle_s`                                 test_gate_and_marks
  5  shadow: the turn runs, the ops are counted, nothing is stored
                                                        test_shadow_stores_nothing
  6  a held box lock (a hold's): the look is "hold", kept in the stats and printed once; with the
     lock free the session is written                  test_a_held_box_lock_is_reported
  7  a compaction whose record is not in place (a short compaction prompt served as a guest, on the
     box 2026-10-05): its mark stays and the idle look writes the session with trigger compaction
                                                        test_a_compaction_without_its_record_waits

Run: python tests/test_memory_writer.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_compaction_continue as C  # noqa: E402  (sets the CPU environment first)

from server import app, compaction  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import _state  # noqa: E402

BLOCK = ('<tandem_memory op="add" scope="project" kind="gotcha">\ntitle: the checker wants node 20'
         '\n\nThe build fails on node 18 with an import attribute error; use node 20.\n'
         '</tandem_memory>')


class _Tok(C._Tok):
    """The preparer's tokenizer, whose decode of the writer's output is `out` (the random model
    writes no block of its own)."""

    out = BLOCK

    def decode(self, ids, skip_special_tokens=False):
        return self.out


def _setup(mode="on", gate=False, idle_s=0.0, min_new=1, decider=None, after="idle"):
    res = C._prep_state()
    app.STATE["compaction"]["prepare"] = "off"
    cond = Conductor(tempfile.mkdtemp(prefix="cond-mw-"), state=app.STATE, journal=False,
                     factory=lambda kind, role, timeout: decider)
    v, st = cond.store.get("settings")
    st.setdefault("memory", {})["notes"] = {"writer": {
        "mode": mode, "gate": gate, "idle_s": max(10.0, idle_s), "due_idle_s": 0.0,
        "min_new_tokens": min_new, "max_tokens": 64, "after_compaction": after}}
    cond.store.put("settings", None, st, note="test", author="test", base=v)
    if decider is not None:
        v, roles = cond.store.get("roles")
        roles["decider"] = dict(roles.get("decider") or {}, mode="on", endpoint="self")
        cond.store.put("roles", None, roles, note="test", author="test", base=v)
    app.STATE["conductor"] = cond
    ctx = {"mode": "on", "user": "owner", "project": "/home/user/app", "session": "c:ses",
           "from_user": True}
    cond.memory.writer.seen("ses", ctx, "Remember that the build needs node 20.")
    return res, cond


def _idle(seconds=1000.0):
    app.STATE["last_end_mono"] = app.time.monotonic() - seconds


def test_the_writer_writes_and_leaves_the_rows():
    res, cond = _setup()
    prompt = tokens(90, seed=81)
    out1, _ = C._turn(prompt, 30)
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    app.STATE["tok"] = _Tok()
    _idle()
    got = app.memory_once()
    assert got == "wrote", got
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    ns = cond.notes.notes("owner")
    assert [(n.title, n.writer, n.status, n.scope) for n in ns] == [
        ("the checker wants node 20", "qwen-writer", "provisional", "project")], ns
    run = cond.memory.writer.runs[0]
    assert run["trigger"] == "idle" and run["ops"][0]["op"] == "add" and run["tokens"] > 0, run
    assert app.memory_once() == "not_due"                 # done at this record
    tool = tokens(17, seed=82)
    nxt, pre = C._turn(prompt + out1 + tool, 10)
    assert pre["kind"] == "end", pre
    _state(end=True)
    C._turn(prompt, 30)
    assert nxt == C._turn(prompt + out1 + tool, 10)[0]


def test_a_waiting_request_stops_the_writer():
    res, cond = _setup()
    prompt = tokens(90, seed=83)
    out1, _ = C._turn(prompt, 30)
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    app.STATE["tok"] = _Tok()
    _idle()
    calls = {"n": 0}
    real = app._waiting

    def waiting():
        calls["n"] += 1
        return calls["n"] > 6
    app._waiting = waiting
    try:
        got = app.memory_once()
    finally:
        app._waiting = real
    assert got == "stopped", got
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    assert cond.notes.notes("owner") == [] and not app.LOCK.locked()
    assert cond.memory.writer.runs[0]["skipped"] == "stopped"


def test_not_due():
    res, cond = _setup(min_new=10_000)
    C._turn(tokens(90, seed=84), 30)
    _idle()
    assert app.memory_once() == "not_due"
    res, cond = _setup(mode="off")
    C._turn(tokens(90, seed=85), 30)
    _idle()
    assert app.memory_once() == "off"
    res, cond = _setup()
    C._turn(tokens(90, seed=86), 30, session="other")
    _idle()
    assert app.memory_once() == "not_due"                # a session the memory never saw


class _Decider:
    def __init__(self, label):
        self.label = label

    def choice(self, text, question, options, max_chars=None):
        from server.conductor.classify import Answer
        return Answer("decider", self.label, 0.95, {})

    def state_chars(self):
        return 4000


def test_gate_and_marks():
    res, cond = _setup(gate=True, min_new=10_000, decider=_Decider("no"))
    C._turn(tokens(90, seed=87), 30)
    _idle(1.0)                                           # not the session's end yet
    assert app.memory_once() == "not_due"
    cond.memory.writer.mark("ses", "compaction")
    assert app.memory_once() == "gate_no"
    assert cond.memory.writer.runs[0]["skipped"] == "gate_no"
    assert app.memory_once() == "not_due"                # the mark was used
    res, cond = _setup(gate=True, min_new=10_000, decider=_Decider("yes"))
    C._turn(tokens(90, seed=88), 30)
    app.STATE["tok"] = _Tok()
    _idle(1.0)
    cond.memory.writer.mark("ses", "give_up")
    assert app.memory_once() == "wrote"
    assert cond.memory.writer.runs[0]["trigger"] == "give_up"


def test_shadow_stores_nothing():
    res, cond = _setup(mode="shadow")
    C._turn(tokens(90, seed=89), 30)
    app.STATE["tok"] = _Tok()
    _idle()
    assert app.memory_once() == "wrote"
    run = cond.memory.writer.runs[0]
    assert run["mode"] == "shadow" and run["ops"] == [{"op": "add", "kind": "gotcha",
                                                       "scope": "project", "stored": False}], run
    assert cond.notes.notes("owner") == []


def test_a_held_box_lock_is_reported():
    import contextlib
    import fcntl
    import io
    res, cond = _setup()
    C._turn(tokens(90, seed=90), 30)
    app.STATE["tok"] = _Tok()
    _idle()
    fd = os.open(os.environ["BOX_LOCK"], os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(3):
                got = app.memory_once()
                app._writer_look(got)
        assert got == "hold", got
        assert cond.memory.writer.stats["last_look"] == "hold"
        assert out.getvalue().count("[memory] writer look: hold") == 1, out.getvalue()
        assert cond.notes.notes("owner") == [] and not cond.memory.writer.runs
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    got = app.memory_once()
    app._writer_look(got)
    assert got == "wrote" and cond.memory.writer.stats["last_look"] == "wrote", got
    assert [n.writer for n in cond.notes.notes("owner")] == ["qwen-writer"]


def test_a_compaction_without_its_record_waits():
    import types
    res, cond = _setup(min_new=10_000, after="turn")
    C._turn(tokens(90, seed=91), 30)
    app.STATE["tok"] = _Tok()
    plan = types.SimpleNamespace(compaction=True)
    real = app._writer_record
    app._writer_record = lambda res, conv: None          # the guest's rows: no record in place
    try:
        app._writer_after_compaction(plan, "ses")
    finally:
        app._writer_record = real
    assert not cond.memory.writer.runs and cond.notes.notes("owner") == []
    assert cond.memory.writer.state("ses")["due"] == "compaction"
    _idle(1.0)                                           # due_idle_s 0: the next idle look
    assert app.memory_once() == "wrote"
    assert cond.memory.writer.runs[0]["trigger"] == "compaction"
    assert [n.writer for n in cond.notes.notes("owner")] == ["qwen-writer"]
    # with the record in place the writer runs at once, before the next request
    res, cond = _setup(min_new=10_000, after="turn")
    C._turn(tokens(90, seed=92), 30)
    app.STATE["tok"] = _Tok()
    app._writer_after_compaction(plan, "ses")
    assert cond.memory.writer.runs[0]["trigger"] == "compaction", cond.memory.writer.runs


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
