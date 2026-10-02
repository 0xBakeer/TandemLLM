"""Laya as the test judge in shadow (J1 step 7, Fable's plan of 2026-10-02 sections 1.3 and 4):
asked only on the cases Tier 0 narrowed, after the request, with a 100 ms deadline, journaled
beside the Tier 0 twin; it acts on nothing.

The test matrix:

  1  the narrowed cases: a run's output without an exit code or a runner's counts (`failure`),
     a `.tandem` log tail (`build`); a result with an exit code, a read, a write: no question;
     one question of each kind a request, digests within their size
                                                        test_the_questions
  2  judge off: no question, no thread; shadow: the questions ride the plan, the served prompt
     and the plan are those of off                      test_off_and_shadow_change_nothing
  3  asked on the judge's own thread after the request (Conductor.finish): one `kind: outcome`
     row (task judge) per question with the answer, its probabilities, the twin and the
     agreement; the deadline is the config's; a role that is off, a timeout or an error is
     `unknown`, journaled, and nothing fails          test_asked_after_the_request
  4  tools/judge_report.py: per question asked, answered, confident, coverage, twin agreement,
     labelled precision and the gate (0.95 and 0.3) from a ledger file
                                                        test_the_report

Run: python tests/test_judge.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

from test_conductor_router import QTok  # noqa: E402
from test_snapshots import S, fresh, plan, served  # noqa: E402

from server import app  # noqa: E402
from server.conductor.classify import Answer  # noqa: E402
from server.conductor.decisions import Journal  # noqa: E402

import judge_report  # noqa: E402


def bash(cmd, out):
    return ("bash", {"command": cmd}, out)


class FakeDecider:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def choice(self, state, question, options):
        self.calls.append((state, question, dict(options)))
        return self.fn(state, options)


def test_the_questions():
    cond = fresh({"judge": "shadow"})
    s = S()
    plan(cond, s.msgs)
    p = plan(cond, s.step(bash("node run.js", "x" * 5000 + "\nTypeError: y is undefined"),
                          bash("node other.js", "fine"),
                          bash("tail -n 60 ./.tandem/build.log", "compiled\nerror: cannot find module x"),
                          bash("make", "exit code 2"),
                          ("read", {"filePath": "/w/a.log"}, "Error: something")))
    qs = {q["qid"]: q for q in p.judge}
    assert sorted(qs) == ["build", "failure"], p.judge
    assert qs["failure"]["digest"] == "fine" and qs["failure"]["twin"] == "clean", \
        "the newest narrowed output of a request"
    assert qs["build"]["twin"] == "failed" and qs["build"]["deadline_ms"] == 100
    assert set(qs["failure"]["options"]) == {"failure", "clean", "unclear"}
    p = plan(cond, s.step(bash("node run.js", "y" * 9000 + "\nTypeError: z")))
    q = p.judge[0]
    assert q["twin"] == "failure" and len(q["digest"]) <= 2000 and q["digest"].endswith(
        "TypeError: z")
    p = plan(cond, s.step(bash("make", "exit code 0"),
                          ("write", {"filePath": "/w/a", "content": "x"}, "Wrote file.")))
    assert p.judge == [], "an exit code is a fact: nothing to ask"
    assert "_narrow" not in (p.snapshot or {})


def test_off_and_shadow_change_nothing():
    s = S()
    turns = [list(s.msgs), s.step(bash("node run.js", "Error: x"))]
    a, b = fresh(None), fresh({"judge": "shadow"})
    for msgs in turns:
        pa, ia = served(a, msgs)
        pb, ib = served(b, msgs)
        assert ia == ib and pa.injected == pb.injected and pb.extras == []
        assert pa.judge is None
    assert pb.judge and pb.judge[0]["qid"] == "failure"


def test_asked_after_the_request():
    cond = fresh({"judge": "shadow"})
    _, roles = cond.store.get("roles")
    roles["decider"]["mode"] = "on"
    roles["decider"]["endpoint"] = "self"
    cond.store.put("roles", None, roles, note="decider on", author="t")
    rows = []
    real = cond.journal.note

    def note(kind, task, **kw):
        rows.append(dict(kw, kind=kind, task=task))
        return real(kind, task, **kw)
    cond.journal.note = note
    asked = []

    def factory(endpoint, model, timeout_ms):
        asked.append((endpoint, model, timeout_ms))
        return FakeDecider(lambda st, o: Answer("decider", "failure", 0.91,
                                                {"failure": 0.91, "clean": 0.06,
                                                 "unclear": 0.03}, 23.0, model="laya-english"))
    cond.judge.factory = factory
    s = S()
    plan(cond, s.msgs)
    p = plan(cond, s.step(bash("node run.js", "TypeError: y")))
    rec = types.SimpleNamespace(status="ok", finish_reason="stop", tool_calls=0, tool_names=[],
                                completion_tokens=1, reasoning_tokens=0, prompt_tokens=10,
                                ttft_ms=1.0, total_ms=2.0, states=None)
    cond.finish(p, rec)
    cond.judge.flush()
    j = [r for r in rows if r["task"] == "judge"]
    assert len(j) == 1 and j[0]["decision"] == "failure", rows
    d = j[0]["detail"]
    assert d["question"] == "failure" and d["twin"] == "failure" and d["agree"] is True
    assert d["confidence"] == 0.91 and d["model"] == "laya-english" and d["ms"] == 23.0
    assert asked == [("self", "auto", 100)], asked
    # a timeout: unknown, journaled
    cond.judge.factory = lambda e, m, t: FakeDecider(
        lambda st, o: Answer("decider", ms=100.0, error="decider_timeout"))
    p = plan(cond, s.step(bash("node run.js", "fine")))
    cond.finish(p, rec)
    cond.judge.flush()
    j = [r for r in rows if r["task"] == "judge"]
    assert j[-1]["decision"] == "unknown" and j[-1]["detail"]["error"] == "decider_timeout"
    # the role off: unknown, and no client is made
    roles["decider"]["mode"] = "off"
    cond.store.put("roles", None, roles, note="decider off", author="t")
    cond.judge.factory = lambda *a: (_ for _ in ()).throw(AssertionError("asked"))
    p = plan(cond, s.step(bash("node run.js", "fine again")))
    cond.finish(p, rec)
    cond.judge.flush()
    j = [r for r in rows if r["task"] == "judge"]
    assert j[-1]["decision"] == "unknown" and j[-1]["detail"]["error"] == "role_off"
    assert cond.judge.stats["asked"] == 3 and cond.judge.stats["answered"] == 1


def test_the_report():
    d = tempfile.mkdtemp(prefix="judge-")
    path = os.path.join(d, "ledger.sqlite3")
    j = Journal(path)
    rows = []
    for i in range(10):
        conf = 0.95 if i < 6 else 0.5
        rows.append(j.note("outcome", "judge", decision="failure" if i % 2 == 0 else "clean",
                           detail={"question": "failure", "twin": "failure", "confidence": conf,
                                   "agree": i % 2 == 0, "ms": 20.0 + i}))
    rows.append(j.note("outcome", "judge", decision="unknown",
                       detail={"question": "build", "error": "decider_timeout"}))
    j.flush()
    time.sleep(0.2)
    j.close()
    conn = sqlite3.connect(path)
    n = conn.execute("SELECT COUNT(*) FROM conductor_events WHERE kind='outcome'").fetchone()[0]
    conn.close()
    assert n == 11, n
    got = judge_report.rows(path)
    rep = judge_report.report(got)
    f = rep["failure"]
    assert (f["asked"], f["answered"], f["confident"], f["coverage"]) == (10, 10, 6, 0.6), f
    assert f["twin_agree"] == 0.5 and f["gate"] == "needs labels" and f["ms_p50"] == 25.0
    labels = {r["id"]: "right" for r in rows[:6]}
    rep = judge_report.report(got, labels=labels)
    assert rep["failure"]["precision"] == 1.0 and rep["failure"]["gate"] == "may act"
    labels[rows[0]["id"]] = "wrong"
    assert judge_report.report(got, labels=labels)["failure"]["gate"] == "stays shadow"
    assert rep["build"]["errors"] == {"decider_timeout": 1} and rep["build"]["gate"] == \
        "needs labels"
    assert judge_report.main(["--ledger", path]) == 0


if __name__ == "__main__":
    app.STATE.setdefault("tok", QTok())
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
