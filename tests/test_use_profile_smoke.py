"""SPD-167: ops/use-profile.sh's smoke turn retries once and logs the whole answer before a rollback.

A first activation of the ThinkingCap community profile rolled itself back after one empty
smoke answer, cause not found. A first activation warms its caches and graphs; one empty turn
is no reason to roll back. The function is taken from the script as written and run against a fake
server that answers empty first and then "ready"."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fn(name: str) -> str:
    text = (ROOT / "ops" / "use-profile.sh").read_text()
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", text, re.S | re.M)
    assert m, name
    return m.group(0)


def _server(answers: list):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"weights": {"name": "community/x"}})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            msg = answers.pop(0) if answers else {"content": "ready"}
            self._send({"choices": [{"message": msg}]})
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run(answers: list) -> subprocess.CompletedProcess:
    srv = _server(answers)
    port = srv.server_address[1]
    script = (f'say() {{ echo "SAY $*"; }}; PY=python3; PORT={port}; SMOKE_URL=http://127.0.0.1:{port}/v1; '
              f'SMOKE_MODEL=x; SMOKE_TIMEOUT_S=20; SMOKE_RETRY_S=0\n{_fn("smoke")}\nsmoke community/x')
    try:
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    finally:
        srv.shutdown()


def test_one_empty_answer_is_retried():
    r = _run([{"content": ""}])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "gave no answer" in r.stdout and "[smoke] ready" in r.stdout


def test_two_empty_answers_fail_with_the_body_in_the_log():
    r = _run([{"content": ""}, {"content": ""}])
    assert r.returncode == 1
    assert r.stdout.count("gave no answer") == 2 and '"choices"' in r.stdout


def test_an_answer_in_the_reasoning_field_counts():
    r = _run([{"content": "", "reasoning_content": "ready"}])
    assert r.returncode == 0 and "(in reasoning) ready" in r.stdout


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
