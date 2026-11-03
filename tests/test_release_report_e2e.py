"""tools/release_report against the fake engine (no model, no GPU): two fake models with other windows
and lanes get other plans; a busy server is refused; a server whose answers lack `X-Tandem: raw` is
refused at the probe and gives INVALID rows; two sections run end to end into data.json and a page, and
nothing calls the chat endpoint. Needs the engine's CPU packages (torch); skips otherwise.

Run: python tests/test_release_report_e2e.py
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

FAKE_LAUNCH = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
               "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', run_name='__main__')")


def _ready() -> str | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "no torch (the fake engine's server needs CPU torch)"
    return None


class Fake:
    """server/app.py --fake-engine with `lanes` and `max_len`."""

    def __init__(self, lanes: int, max_len: int, tps: float = 400.0):
        from ports import free_port
        self.port = free_port()
        self.home = tempfile.mkdtemp(prefix="rr-fake-")
        self.base = f"http://127.0.0.1:{self.port}"
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": self.home, "PYTHONPATH": ROOT,
               "CUDA_VISIBLE_DEVICES": "", "QSE_ADMIN_TOKEN": "rr-" + "0" * 40}
        self.log = open(os.path.join(self.home, "server.log"), "w")
        self.p = subprocess.Popen([sys.executable, "-c", FAKE_LAUNCH, "--fake-engine", "--port", str(self.port),
                                   "--served-model", "TandemLLM", "--usage-ledger", os.path.join(self.home, "l.sqlite3"),
                                   "--fake-tps", f"{tps:g}", "--parallel", str(lanes), "--max-len", str(max_len),
                                   "--host", "127.0.0.1"], cwd=ROOT, env=env, stdout=self.log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(240):
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return
            except OSError:
                pass
            if self.p.poll() is not None:
                break
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("the fake server did not start: " + open(os.path.join(self.home, "server.log")).read()[-800:])

    def text(self) -> str:
        self.log.flush()
        return open(os.path.join(self.home, "server.log")).read()

    def stop(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
        shutil.rmtree(self.home, ignore_errors=True)


def report(*args, url: str) -> tuple[int, str, str]:
    out = tempfile.mkdtemp(prefix="rr-out-")
    r = subprocess.run([sys.executable, "-m", "tools.release_report", *args, "--url", url, "--hw", "fake"]
                       + ([] if args and args[0] == "plan" else ["--out", out]),
                       cwd=ROOT, capture_output=True, text=True, timeout=900, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return r.returncode, r.stdout + r.stderr, out


def test_two_fake_models_get_two_plans():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    a, b = Fake(4, 32768), Fake(2, 8192)
    try:
        ra, oa, _ = report("plan", url=a.base)
        rb, ob, _ = report("plan", url=b.base)
    finally:
        a.stop()
        b.stop()
    assert ra == 0 and rb == 0, oa[-800:] + ob[-800:]
    la = next(x for x in oa.splitlines() if "a/b lengths" in x)
    lb = next(x for x in ob.splitlines() if "a/b lengths" in x)
    assert "16k" in la and "16k" not in lb and "4k" in lb, (la, lb)
    assert "2k x [1, 2, 3, 4]" in oa and "2k x [1, 2]" in ob, (oa[-600:], ob[-600:])
    assert "slot limit 4" in oa and "slot limit 2" in ob
    assert "estimate (minutes)" in oa


def test_a_busy_server_is_refused():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    f = Fake(2, 16384, tps=10)
    try:
        body = json.dumps({"model": "TandemLLM", "prompt": "Write a long text. FAKE_SLOW FAKE_LONG", "max_tokens": 2000,
                           "stream": True}).encode()
        req = urllib.request.Request(f.base + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        holder = threading.Thread(target=lambda: urllib.request.urlopen(req, timeout=120).read(200), daemon=True)
        holder.start()
        time.sleep(2.0)
        rc, out, _ = report("run", "--only", "a", url=f.base)
    finally:
        f.stop()
    assert rc == 3 and "busy" in out and "running" in out, out[-800:]


class StripRaw(http.server.BaseHTTPRequestHandler):
    """A server in front of the fake engine that answers without `X-Tandem` (the chat layer's answer)."""
    target = ""

    def _relay(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(self.target + self.path, data=body, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            self.send_response(r.status)
            for k, v in r.headers.items():
                if k.lower() not in ("x-tandem", "transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in iter(lambda: r.read1(65536), b""):
                self.wfile.write(chunk)
                self.wfile.flush()

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def log_message(self, *a):
        pass


def test_answers_without_the_raw_header_are_refused_and_invalid():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    f = Fake(2, 16384)
    StripRaw.target = f.base
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StripRaw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        rc, out, _ = report("run", "--only", "a", url=url)
        assert rc == 3 and "not served raw" in out, out[-800:]
        # the rows themselves: a section measured through it is INVALID, never a number
        from pathlib import Path
        from tools.gate import hw
        from tools.release_report import client as cmod, plan
        from tools.release_report.prompts import Prose
        from tools.release_report.sections import Ctx, lengths
        h = cmod.health(url)
        pl = plan.make(h, {}, hw.profile("fake"))
        pl.lengths, pl.runs = [1024], {1024: 1}
        cl = cmod.ReportClient(url, "TandemLLM")
        rows = [r for r in lengths.measure(Ctx(url, cl, h, pl, Prose("repo"), Path(tempfile.mkdtemp()), print))
                if not r.get("skipped")]
        assert rows and all(not r["valid"] and "not the bare engine" in r["why"] for r in rows), rows
    finally:
        srv.shutdown()
        f.stop()


def test_two_sections_end_to_end_into_data_and_a_page():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    f = Fake(4, 32768)
    try:
        rc, out, d = report("run", "--only", "a", "c", url=f.base)
        log = f.text()
    finally:
        f.stop()
    assert rc == 0, out[-1500:]
    data = json.load(open(os.path.join(d, "data.json")))
    secs = {r["section"] for r in data["rows"]}
    assert secs == {"a", "c"}, secs
    a = [r for r in data["rows"] if r["section"] == "a" and not r.get("skipped")]
    assert a and all(r["valid"] and r["v"]["decode_tok_s"] > 0 and r["ids"] for r in a), a[:2]
    c = [r for r in data["rows"] if r["section"] == "c" and r["key"].startswith("2k/")]
    assert {r["v"]["n"] for r in c} == {1, 2, 3, 4} and all(r["valid"] for r in c), [(r["key"], r["why"]) for r in c]
    assert all(r["cond"]["proof"]["max_running"] == r["v"]["n"] for r in c)
    run = data["runs"][-1]
    assert run["mode"] == "url" and run["plan"]["lanes"] == 4 and run["knobs"]["max_running"] == 4
    assert os.path.isdir(os.path.join(d, "raw")) and os.listdir(os.path.join(d, "raw"))
    assert "chatcmpl-" not in log and "/v1/chat/completions" not in log     # no chat-endpoint request reached it
    r = subprocess.run([sys.executable, "-m", "tools.release_report", "render", os.path.join(d, "data.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    page = open(os.path.join(d, "report.html")).read()
    assert "TandemLLM bare engine: raw /v1/completions, no conductor or chat layer." in page
    home = os.path.expanduser("~")
    assert home not in page and "127.0.0.1" not in page


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
