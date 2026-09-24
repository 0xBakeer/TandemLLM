"""The fake engine (VIS-18's fixture) as the e2e runs it: `server/app.py --fake-engine` in a
process of its own, with triton unimportable, a temp ledger and a test admin token.

What the dashboard's e2e depends on: the server starts with no model and no GPU; the live usage
check passes; every dashboard endpoint validates against the contract; the rows reach the ledger;
the steering words give an error and a tool call; the production ledger is refused; SIGTERM drains
and exits 0.

Run: python tests/test_fake_engine.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import contract_check, usage_check  # noqa: E402

ADMIN = "e2e-" + "0" * 40
LAUNCH = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
          "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', "
          "run_name='__main__')")


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start(port, ledger, home=None, extra=()):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home or os.environ["HOME"],
           "QSE_ADMIN_TOKEN": ADMIN, "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.Popen([sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port),
                             "--served-model", "qwen38-spark-engine", "--usage-ledger", ledger,
                             "--fake-tps", "400", *extra],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def _get(port, path, token=ADMIN):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _chat(port, text, **extra):
    body = dict({"messages": [{"role": "user", "content": text}], "max_tokens": 200}, **extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_the_fake_engine_end_to_end():
    port = _port()
    ledger = os.path.join(tempfile.mkdtemp(prefix="qse-e2e-"), "ledger.sqlite3")
    proc = _start(port, ledger)
    try:
        end = time.time() + 60
        while time.time() < end:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except OSError:
                if proc.poll() is not None:
                    raise AssertionError(proc.stdout.read())
                time.sleep(0.3)
        args = type("A", (), {"base": f"http://127.0.0.1:{port}", "model": "qwen38-spark-engine",
                              "token": None, "prompt": "Name three rivers.", "max_tokens": 64,
                              "timeout": 30.0})()
        bad = {c[0]: r["errors"] for c in usage_check.CASES
               if (r := usage_check.run_case(args, *c))["errors"]}
        assert not bad, bad
        try:
            _chat(port, "FAKE_ERROR please")
            raise AssertionError("FAKE_ERROR did not fail")
        except urllib.error.HTTPError as exc:
            assert exc.code == 500
        body = _chat(port, "use the tool now",
                     tools=[{"type": "function", "function": {"name": "web_search"}}])
        assert body["choices"][0]["finish_reason"] == "tool_calls", body["choices"][0]
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "web_search"
        for ep, name in (("summary", "summary"), ("usage?bucket=hour", "usage"),
                         ("requests?limit=50", "requests"), ("system", "system"),
                         ("logs?follow=0&backlog=100", "logs")):
            got = _get(port, f"/v1/dashboard/{ep}")
            errs = contract_check.check(got, name)
            assert not errs, (ep, errs[:5])
        rows = _get(port, "/v1/dashboard/requests?limit=50")["requests"]
        assert len(rows) >= 8 and {"error", "tool_calls"} <= {r["finish_reason"] for r in rows}
        try:
            _get(port, "/v1/dashboard/summary", token=None)
            raise AssertionError("the dashboard answered without a token")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
        out = proc.stdout.read()
        assert "FAKE ENGINE" in out and "[ledger] on" in out and ADMIN not in out
        n = sqlite3.connect(ledger).execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert n == len(rows), (n, len(rows))
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_fake_engine_refuses_the_production_ledger():
    home = tempfile.mkdtemp(prefix="qse-home-")
    proc = _start(_port(), "~/.qwen38-spark-engine/usage/ledger.sqlite3", home=home)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode != 0 and "production ledger" in out, out[-500:]
    assert not os.path.exists(os.path.join(home, ".qwen38-spark-engine", "usage"))


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
