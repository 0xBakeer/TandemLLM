"""OPS-43 / OPS-44: the eval harness, tools/tandem_eval.py.

The test matrix is OPS-43's Gherkin plus the schema check:

  1  both arms on the same build and night       test_a_night_end_to_end_records_both_arms
  2  alternation and isolation                    test_the_schedule_*, test_the_store_is_cleared_*
  3  checkers stay pristine                       test_pristine_*
  4  the indicative label                         test_n_below_ten_is_indicative (and the Compare
                                                  view's fixture in dashboard/tests/conductor.test.ts)
  5  the raw row                                  test_the_raw_row_*
  -  report.json validates                        test_the_report_validates_*, the example file,
                                                  and POST/GET /v1/conductor/evals over HTTP

The end-to-end night runs the real harness against a fake `opencode` (a script that answers
`run` and `export` like opencode 1.18.33 and solves the task only on the TandemLLM arm) and a fake
engine API in this process; the HTTP test starts the server with the fake engine.

Run: python tests/test_tandem_eval.py [name-filter]
"""

from __future__ import annotations

import copy
import json
import os
import stat
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools import contract_check, tandem_eval as te  # noqa: E402

EXAMPLE = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "tandem-eval.example.json")
BUILD = {"version": "0.3.2-test", "code_sha": "abc123def456"}

FAKE_OPENCODE = r'''#!/usr/bin/env python3
"""A stand-in for opencode 1.18.33's `run --format json` and `export` (tests/test_tandem_eval.py).
The TandemLLM arm writes the right answer and runs the checker; the raw arm writes a wrong one,
runs the checker once and stops (the driver would continue)."""
import json, os, subprocess, sys, time, secrets
STORE = os.environ["FAKE_OC_STORE"]
args = sys.argv[1:]
if args[0] == "export":
    print("Exporting session: " + args[1], file=sys.stderr)
    print(open(os.path.join(STORE, args[1] + ".json")).read())
    sys.exit(0)
cfg = json.load(open("opencode.json"))
headers = cfg["provider"]["tandem-eval"]["options"]["headers"]
raw = headers.get("X-Tandem") == "raw"
with open(os.path.join(STORE, "calls.jsonl"), "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "headers": headers, "argv": args,
                        "key": os.environ.get("TANDEM_EVAL_KEY")}) + "\n")
sid = "ses_" + secrets.token_hex(6)
t0 = int(time.time() * 1000)
print(json.dumps({"type": "step_start", "sessionID": sid}))
if os.environ.get("FAKE_OC_TAMPER"):
    open("check.py", "a").write("\n# changed by the model\n")
open("answer.txt", "w").write("41" if raw else "42")
out = subprocess.run([sys.executable, "check.py"], capture_output=True, text=True).stdout
t1 = int(time.time() * 1000) + 60000
parts = [{"type": "tool", "tool": "bash", "callID": "c1",
          "state": {"status": "completed", "input": {"command": "python3 check.py"},
                    "output": out, "time": {"start": t1 - 100, "end": t1}}}]
msgs = [{"info": {"role": "user", "time": {"created": t0}}, "parts": []},
        {"info": {"role": "assistant", "finish": "tool-calls", "time": {"created": t0, "completed": t1},
                  "tokens": {"input": 1000, "output": 200, "reasoning": 800,
                             "cache": {"read": 3000, "write": 0}}}, "parts": parts},
        {"info": {"role": "assistant", "finish": "stop", "time": {"created": t1, "completed": t1 + 1000},
                  "tokens": {"input": 10, "output": 0, "reasoning": 50, "cache": {"read": 4000}}},
         "parts": [] if raw else [{"type": "text", "text": "Done: RESULT: PASS"}]}]
json.dump({"info": {"id": sid}, "messages": msgs}, open(os.path.join(STORE, sid + ".json"), "w"))
print(json.dumps({"type": "step_finish", "sessionID": sid}))
'''

CHECK = '''import os, sys
print("taskcheck answer")
a = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer.txt")).read().strip() \\
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer.txt")) else ""
if a == "42":
    print("RESULT: PASS")
else:
    print(f"RESULT: FAIL <- (a) answer off by {abs(42 - int(a or 0))} > 0")
    sys.exit(1)
'''


class FakeEngine:
    """The engine API the harness reads and writes, in this process."""

    def __init__(self, guard=True):
        self.calls: list[tuple[str, str]] = []
        self.published = []
        self.guard = guard
        eng = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                eng.calls.append(("GET", self.path))
                if self.path.startswith("/v1/conductor/evals"):
                    reps = [{"kind": "speed_guard", "file": "guard.json", "build": BUILD,
                             "verdict": "pass"}] if eng.guard else []
                    return self._send(200, {"contract_version": "1.0", "reports": reps,
                                            "engine": BUILD})
                if self.path.startswith("/v1/dashboard/requests"):
                    import time
                    ts = te.now_iso(time.time())
                    return self._send(200, {"requests": [
                        {"ts": ts, "client": {"kind": "opencode"}, "decode_tps": 44.0,
                         "prefill_tps": 1500.0, "completion_tokens": 1000, "reasoning_tokens": 800,
                         "prompt_tokens": 4000, "cached_tokens": 3000, "finish_reason": "tool_calls"}],
                        "next_before": None})
                if self.path.startswith("/v1/conductor/events"):
                    return self._send(200, {"events": [
                        {"mode": "act", "decision": "PROMPT_TAIL_NOTE", "task": "S1"},
                        {"mode": "shadow", "decision": "REGENERATE_CALL", "task": "C1"},
                        {"mode": "act", "decision": "PASS", "task": "SP3"}], "monitor": True})
                return self._send(404, {"error": {"message": "no"}})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                eng.calls.append(("POST", self.path))
                if self.headers.get("Authorization") != "Bearer adm":
                    return self._send(401, {"error": {"message": "token"}})
                if self.path == "/v1/cache/clear":
                    return self._send(200, {"cleared": True})
                if self.path == "/v1/conductor/evals":
                    eng.published.append(body["report"])
                    return self._send(201, {"file": "suite-x.json"})
                return self._send(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def _setup(tmp: str, n_suites=1) -> tuple[str, str, str]:
    """A suites file with a tiny task, the fake opencode, its store."""
    tpl = os.path.join(tmp, "tpl")
    os.makedirs(tpl)
    with open(os.path.join(tpl, "check.py"), "w") as f:
        f.write(CHECK)
    with open(os.path.join(tpl, "SPEC.md"), "w") as f:
        f.write("Write 42 into answer.txt.\n")
    checker = os.path.join(tmp, "skill-checker")
    with open(checker, "w") as f:
        f.write("#!/bin/sh\n")
    suites = {"version": 1, "opencode": {"permission": {"question": "deny"}}, "suites": [
        {"id": f"answer{i}", "title": f"answer {i}", "kind": "opencode", "template": tpl,
         "timeout_min": 2, "prompt": "Read SPEC.md and do it. Run python3 check.py.",
         "continue": "Continue.",
         "gates": {"list": [{"id": "pass", "args": "check\\.py", "final": True}]},
         "final_check": [sys.executable, "check.py"],
         "pristine": ["run:check.py", checker]} for i in range(n_suites)]}
    path = os.path.join(tmp, "suites.json")
    with open(path, "w") as f:
        json.dump(suites, f)
    oc = os.path.join(tmp, "opencode")
    with open(oc, "w") as f:
        f.write(FAKE_OPENCODE)
    os.chmod(oc, os.stat(oc).st_mode | stat.S_IEXEC)
    store = os.path.join(tmp, "store")
    os.makedirs(store)
    return path, oc, store


def _night(tmp, eng, n=2, extra=(), n_suites=1, env=None):
    path, oc, store = _setup(tmp, n_suites)
    old = dict(os.environ)
    os.environ.update({"FAKE_OC_STORE": store, "TANDEM_EVAL_KEY": "k-secret", "QSE_ADMIN_TOKEN":
                       "adm", **(env or {})})
    try:
        rc = te.main(["--suites", path, "--n", str(n), "--base", eng.url + "/v1",
                      "--work", os.path.join(tmp, "work"), "--out", os.path.join(tmp, "out"),
                      "--night", "2026-10-01", "--opencode", oc, *extra])
    finally:
        os.environ.clear()
        os.environ.update(old)
    with open(os.path.join(tmp, "out", "2026-10-01", "report.json")) as f:
        rep = json.load(f)
    calls = [json.loads(x) for x in open(os.path.join(store, "calls.jsonl"))]
    return rc, rep, calls


# ------------------------------------------------------------------ 1: both arms, one night

def test_a_night_end_to_end_records_both_arms():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rc, rep, calls = _night(tmp, eng, n=2, extra=("--publish",))
            assert rc == 0, rep
            assert not te.validate(rep), te.validate(rep)[:5]
            assert rep["build"] == BUILD and rep["night"] == "2026-10-01"
            arms = {(r["suite_id"], r["arm"]): r for r in rep["rows"]}
            raw, tan = arms[("answer0", "raw")], arms[("answer0", "tandem")]
            assert (raw["n"], tan["n"]) == (2, 2)
            assert (raw["pass_rate"], tan["pass_rate"]) == (0.0, 1.0)
            assert tan["minutes_to_pass"] is not None and raw["minutes_to_pass"] is None
            assert raw["continuations"] == 2 and tan["continuations"] == 0
            assert raw["empty_stops"] == 2 and tan["empty_stops"] == 0
            assert tan["interventions"] == 2 and tan["decode_tps"] == 44.0
            assert tan["cache_hit"] == round(7000 / 8010, 2)
            assert raw["build"] == tan["build"] == BUILD["code_sha"]
            c = rep["comparison"][0]
            assert c["verdict"] == "better" and c["pass_rate_not_lower"] and c["minutes_not_higher"]
            # every run records its hour and its session; the report went to the box
            assert all(r["started_at"].startswith("20") and r["session_id"] for r in rep["runs"])
            assert len(eng.published) == 1 and eng.published[0]["kind"] == "suite"
            # the arms reached opencode as headers, the key from the environment, never a file
            assert [c_["headers"] for c_ in calls] == [{"X-Tandem": "raw"},
                                                       {"X-Tandem-Profile": "overnight"}] * 2
            assert all(c_["key"] == "k-secret" for c_ in calls)
            for r in rep["runs"]:
                txt = open(os.path.join(r["dir"], "opencode.json")).read()
                assert "k-secret" not in txt and "{env:TANDEM_EVAL_KEY}" in txt
            md = open(os.path.join(tmp, "out", "2026-10-01", "report.md")).read()
            assert "| answer 0 | raw | 2 | 0.00 |" in md and "Indicative" in md
    finally:
        eng.close()


# ------------------------------------------------------------------ 2: alternation, isolation

def test_the_schedule_alternates_arms_with_unique_tags():
    suites = [{"id": "a"}, {"id": "b"}]
    plan = te.schedule(suites, 3, "raw", "2026-10-01")
    for sid in ("a", "b"):
        assert [p["arm"] for p in plan if p["suite"] == sid] == ["raw", "tandem"] * 3
    assert len({p["tag"] for p in plan}) == len(plan) == 12
    assert all(p["tag"].startswith(f"te-2026-10-01-{p['suite']}-{p['arm']}-r{p['rep']}-")
               for p in plan)
    # a colliding token draws again
    toks = iter(["aa", "aa", "bb", "cc", "dd"])
    plan = te.schedule([{"id": "a"}], 1, "tandem", "n", token=lambda: next(toks))
    assert [p["arm"] for p in plan] == ["tandem", "raw"] and len({p["tag"] for p in plan}) == 2


def test_the_store_is_cleared_between_arms_and_recorded():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, rep, calls = _night(tmp, eng, n=2, n_suites=2)
        clears = [c for c in eng.calls if c == ("POST", "/v1/cache/clear")]
        # raw a0, tandem a0, raw a1, tandem a1, ...: every run changes the arm
        assert len(clears) == len(rep["runs"]) == 8
        assert all(r["store_clear"]["status"] == 200 for r in rep["runs"])
        assert rep["isolation"]["unique_tags"] and len(rep["isolation"]["store_clears"]) == 8
        assert "suffix store" in rep["isolation"]["suffix_store"]
        # every run in its own folder, which carries its tag (opencode's system prompt names it)
        dirs = {c["cwd"] for c in calls}
        assert len(dirs) == 8 and all(os.path.basename(d).startswith("te-") for d in dirs)
    finally:
        eng.close()
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, rep, _ = _night(tmp, eng, n=1, extra=("--no-clear-store",))
        assert not [c for c in eng.calls if c[1] == "/v1/cache/clear"]
        assert all(r["store_clear"] is None for r in rep["runs"])
    finally:
        eng.close()


# ------------------------------------------------------------------ 3: pristine checkers

def test_pristine_a_changed_checker_is_caught_and_the_final_check_uses_the_template():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, rep, _ = _night(tmp, eng, n=1, env={"FAKE_OC_TAMPER": "1"})
            for r in rep["runs"]:
                assert not r["pristine"]["ok"] and r["pristine"]["changed"] == ["run:check.py"]
                assert r["pristine"]["restored"] == ["check.py"]
                assert open(os.path.join(r["dir"], "check.py")).read() == CHECK
            assert not any(row["pristine"] for row in rep["rows"])
            assert "Pristine: changed in" in open(os.path.join(tmp, "out", "2026-10-01",
                                                               "report.md")).read()
    finally:
        eng.close()


def test_pristine_hashes_see_outside_files_and_the_template():
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _setup(tmp)
        s = te.load_suites(path)[0]
        before = te.pristine_hashes(s)
        run = os.path.join(tmp, "run")
        te.prepare(s, run)
        assert te.pristine_check(s, before, run) == {"ok": True, "changed": []}
        with open(s["pristine"][1], "a") as f:
            f.write("# edited\n")
        with open(os.path.join(s["template"], "SPEC.md"), "a") as f:
            f.write("more\n")
        got = te.pristine_check(s, before, run)
        assert not got["ok"] and set(got["changed"]) == {s["pristine"][1], "template:"}


# ------------------------------------------------------------------ 4: indicative

def test_n_below_ten_is_indicative():
    runs = [_run("raw", passed=False), _run("tandem", passed=True, minutes=12.0)] * 2
    s = [{"id": "x", "title": "x"}]
    rep = te.build_report(_meta(), runs, s, {"status": "found", "source": "g", "row": {}})
    assert all(r["indicative"] for r in rep["rows"]) and rep["comparison"][0]["indicative"]
    assert rep["verdict"]["indicative"] and "indicative" in rep["verdict"]["why"]
    runs10 = [_run("raw", passed=True, minutes=20.0), _run("tandem", passed=True, minutes=12.0)] * 10
    rep = te.build_report(_meta(), runs10, s, {"status": "found", "source": "g", "row": {}})
    assert not any(r["indicative"] for r in rep["rows"]) and rep["verdict"]["better"] is True
    assert not rep["verdict"]["indicative"]


def _run(arm, passed, minutes=None, rep=0):
    return {"suite": "x", "arm": arm, "rep": rep, "tag": f"t-{arm}-{id(object())}",
            "started_at": "2026-10-01T01:00:00Z", "ended_at": "2026-10-01T01:30:00Z",
            "error": None, "store_clear": None, "dir": None, "session_id": "ses_1",
            "exit_code": 0, "timed_out": False, "wall_minutes": 30.0,
            "gates": {"total": 1, "passed": int(passed),
                      "by_gate": [{"id": "pass", "passed": passed, "minutes": minutes}]},
            "passed": passed, "minutes_to_pass": minutes, "final_check": None,
            "tokens": {"input": 1, "output": 2, "reasoning": 3, "cache_read": 4},
            "cache_hit": 0.8, "gen_tps": 40.0, "turns": 3, "tool_calls": 2, "tool_errors": 0,
            "empty_stops": 0, "last_finish": "stop",
            "continuations": {"needed": 0, "sent": 0, "driver": "watch"},
            "engine": None, "conductor": None, "pristine": {"ok": True, "changed": []}}


def _meta():
    return {"night": "2026-10-01", "build": dict(BUILD), "base": "http://x/v1",
            "profile": "overnight", "driver": {"mode": "watch", "max_continues": 0}}


def test_the_comparison_names_better_worse_and_mixed():
    s = [{"id": "x", "title": "x"}]
    row = {"status": "found", "source": "g", "row": {"verdict": "pass"}}
    cases = [
        ([_run("raw", True, 20.0), _run("tandem", True, 12.0)], "better", True),
        ([_run("raw", True, 12.0), _run("tandem", True, 12.0)], "equal", True),
        ([_run("raw", True, 12.0), _run("tandem", False)], "worse", False),
        ([_run("raw", True, 12.0), _run("tandem", True, 20.0)], "mixed", False),
        ([_run("raw", False), _run("tandem", False)], "equal", True),
    ]
    for runs, want, better in cases:
        rep = te.build_report(_meta(), runs, s, row)
        assert rep["comparison"][0]["verdict"] == want, (want, rep["comparison"])
        assert rep["verdict"]["better"] is better, (want, rep["verdict"])
    rep = te.build_report(_meta(), [_run("raw", True, 1.0)], s, row)
    assert rep["comparison"][0]["verdict"] == "incomplete" and rep["verdict"]["better"] is None


# ------------------------------------------------------------------ 5: the raw row

def test_the_raw_row_is_found_attached_or_missing():
    s = [{"id": "x", "title": "x"}]
    runs = [_run("raw", True, 20.0), _run("tandem", True, 12.0)]
    eng = FakeEngine(guard=True)
    try:
        api = te.Api(eng.url)
        got = te.raw_row(api, BUILD, None)
        assert got["status"] == "found" and got["source"] == "guard.json"
        assert te.raw_row(api, {"version": "other", "code_sha": "zzz"}, None)["status"] == \
            "missing"
    finally:
        eng.close()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "row.json")
        with open(p, "w") as f:
            json.dump({"verdict": "pass", "views": {"off": {"mean": 50.1}}}, f)
        got = te.raw_row(None, BUILD, p)
        assert got["status"] == "attached" and got["row"]["views"]["off"]["mean"] == 50.1
    rep = te.build_report(_meta(), runs, s, {"status": "missing", "source": None, "row": None})
    assert rep["verdict"]["better"] is None and "never-worse row" in rep["verdict"]["why"]
    rep = te.build_report(_meta(), runs, s, {"status": "found", "source": "g",
                                             "row": {"verdict": "worse"}})
    assert rep["verdict"]["better"] is False


# ------------------------------------------------------------------ the schema

def test_the_report_validates_and_the_example_is_current():
    s = [{"id": "x", "title": "x"}]
    rep = te.build_report(_meta(), [_run("raw", True, 20.0), _run("tandem", True, 12.0)], s,
                          {"status": "found", "source": "g", "row": {}})
    assert te.validate(rep) == []
    broken = copy.deepcopy(rep)
    broken["rows"][0]["arm"] = "TandemLLM"
    del broken["verdict"]
    errs = te.validate(broken)
    assert any("arm" in e for e in errs) and any("verdict" in e for e in errs)
    with open(EXAMPLE) as f:
        ex = json.load(f)
    assert te.validate(ex) == []
    assert {r["arm"] for r in ex["rows"]} == {"raw", "tandem"}
    from server.conductor import routes
    assert routes.eval_errors(ex) == [] and routes.eval_errors({"kind": "x"})


def test_the_suites_file_loads_and_its_prompts_resolve():
    suites = te.load_suites(te.SUITES, include_disabled=True)
    ids = [s["id"] for s in suites]
    assert ids == ["minicalc", "csvq", "spiral", "photo3d", "pagoda", "protocol"]
    assert [s["id"] for s in te.load_suites(te.SUITES)] == ["minicalc", "csvq", "spiral",
                                                            "photo3d", "protocol"]
    for s in suites:
        if s["kind"] == "opencode" and s.get("prompt"):
            p, c = te.prompt_of(s, "/RUN")
            assert "check.py" in p and c
    # the coding tasks' checkers fail an empty folder and name their checker
    for tid in ("minicalc", "csvq"):
        s = next(x for x in suites if x["id"] == tid)
        with tempfile.TemporaryDirectory() as tmp:
            te.prepare(s, os.path.join(tmp, "r"))
            got = te.final_check(s["final_check"], os.path.join(tmp, "r"))
            assert got["verdict"] == "FAIL", got
    # a prompt file's markers are lines of their own
    with tempfile.TemporaryDirectory() as tmp:
        pf = os.path.join(tmp, "P.md")
        with open(pf, "w") as f:
            f.write("Paste after the line `PROMPT START`.\n\nPROMPT START\n\nWork in ~/x now.\n"
                    "CONTINUE\n\nGo on in ~/x.\n")
        s = {"prompt_file": pf, "prompt_start": "PROMPT START", "prompt_end": "CONTINUE",
             "rewrite": ["~/x"]}
        assert te.prompt_of(s, "/R") == ("Work in /R now.", "Go on in /R.")


def test_reading_an_opencode_export():
    t0 = 1_790_000_000.0
    ms = t0 * 1000

    def tool(cmd, out, end_min, status="completed"):
        return {"type": "tool", "tool": "bash", "state": {"status": status,
                                                          "input": {"command": cmd},
                                                          "output": out,
                                                          "time": {"end": ms + end_min * 60000}}}
    sk = "sketchcheck  shots/{s}.png   (style: spiral)\nRESULT: {r}"
    exp = {"messages": [
        {"info": {"role": "assistant", "finish": "tool-calls",
                  "time": {"created": ms, "completed": ms + 120000},
                  "tokens": {"input": 100, "output": 10, "reasoning": 90, "cache": {"read": 900}}},
         "parts": [tool("sketchcheck shots/s1-tone.png ref/mona.jpg",
                        sk.format(s="s1-tone", r="FAIL  <- band 0.03 > 0.02"), 3),
                   tool("sketchcheck shots/s3-ribbon.png", sk.format(s="s3-ribbon", r="FAIL  <- "
                                                                     "tone_corr 0.9 < 0.95"), 5),
                   tool("sketchcheck shots/s3-ribbon.png", sk.format(s="s3-ribbon", r="PASS"), 9),
                   tool("ls", "boom", 9.5, status="error")]},
        {"info": {"role": "assistant", "finish": "stop",
                  "time": {"created": ms + 600000, "completed": ms + 660000},
                  "tokens": {"input": 5, "output": 0, "reasoning": 0}}, "parts": []}]}
    suite = next(s for s in te.load_suites(te.SUITES) if s["id"] == "spiral")
    got = te.read_export(exp, suite, t0)
    assert [(g["id"], g["passed"], g["minutes"]) for g in got["gates"]] == \
        [("tone", True, 3.0), ("ribbon", True, 9.0), ("animated", False, None)]
    assert got["empty_stops"] == 1 and got["tool_errors"] == 1 and got["tool_calls"] == 4
    assert got["tokens"] == {"input": 105, "output": 10, "reasoning": 90, "cache_read": 900}
    assert got["wall_minutes"] == 11.0 and got["gen_tps"] == round(100 / 180, 2)
    assert got["cache_hit"] == round(900 / 1005, 4)
    # ordinal gates: the k-th PASS is gate k
    pg = "pagodacheck  .\nRESULT: {r}"
    exp2 = {"messages": [{"info": {"role": "assistant", "time": {}}, "parts": [
        tool("pagodacheck .", pg.format(r="PASS"), 30), tool("pagodacheck .", pg.format(
            r="FAIL  <- (a) x 3.5 > 1.0"), 40), tool("pagodacheck .", pg.format(r="PASS"), 55)]}]}
    g = te.gate_progress(te.tool_parts(exp2), {"ordinal": True, "total": 9}, ms)
    assert [x["minutes"] for x in g[:3]] == [30.0, 55.0, None] and len(g) == 9


def test_the_box_lock_refuses_while_a_hold_runs():
    lk = te.BoxLock("dgx", wait_s=120)
    cmd = lk.argv()
    assert cmd[:3] == ["ssh", "-o", "BatchMode=yes"] and cmd[3] == "dgx"
    assert "pgrep -fa '[h]old.sh' && exit 3" in cmd[4] and "flock -w 120 ~/.qwen38-box.flock" \
        in cmd[4]
    assert te.night_of(1_790_820_000.0) in ("2026-09-30", "2026-10-01")   # local time


# ------------------------------------------------------------------ over HTTP (fake engine)

def test_the_report_is_kept_for_the_compare_screen_over_http():
    """POST /v1/conductor/evals keeps a report in the conductor's evals/; GET lists it."""
    from test_conductor_e2e import ADMIN, PROXY, _call, _check, _port, _server, _wait
    port = _port()
    cdir = tempfile.mkdtemp(prefix="qse-eval-cond-")
    proc, _ = _server(port, conductor_dir=cdir)
    try:
        _wait(port, proc)
        with open(EXAMPLE) as f:
            ex = json.load(f)
        st, _, got = _call(port, "POST", "/v1/conductor/evals", {"report": ex}, headers=PROXY,
                           token=None)
        assert st in (401, 403, 404), got                     # a write needs the admin token
        st, _, got = _call(port, "POST", "/v1/conductor/evals", {"report": {"kind": "x"}},
                           token=ADMIN)
        assert st == 422 and "kind" in got["error"]["message"]
        st, _, got = _call(port, "POST", "/v1/conductor/evals", {"report": ex}, token=ADMIN)
        assert st == 201, got
        _check(got, "eval-write")
        st, _, got = _call(port, "GET", "/v1/conductor/evals")
        assert st == 200
        _check(got, "evals")
        suite = [r for r in got["reports"] if r.get("kind") == "suite"]
        assert len(suite) == 1 and suite[0]["night"] == ex["night"]
        assert contract_check.validate(suite[0], json.load(open(te.SCHEMA))) == []
    finally:
        proc.terminate()
        proc.wait(10)


def test_a_run_sees_its_own_folder_as_pwd():
    """opencode reads $PWD for its project directory; the harness's own PWD must not leak in."""
    import tempfile
    from tools import tandem_eval as t
    d = tempfile.mkdtemp(prefix="te-pwd-")
    log = os.path.join(d, "log")
    code, timed_out, lines = t.run_process(["sh", "-c", 'echo "$PWD"'], d,
                                           dict(os.environ, PWD="/somewhere/else"), 10, log)
    assert code == 0 and not timed_out and lines == [os.path.abspath(d)], lines


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.path.insert(0, HERE)
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and only in name:
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
