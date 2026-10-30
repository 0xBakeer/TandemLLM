"""SRV-671: a lost CUDA context makes /health fail, refuses requests in words, and the Chat tab says
so; fault injection on the CPU (the random 4-layer model behind the real HTTP server).

THE INCIDENT (2026-10-21 about 15:00, Ornith 1.5 122B, 0.13.0). A device-side assert (SRV-669)
left the process's CUDA context failing every call. For 30 minutes /health said ok, every chat
request answered 500 with the driver's text, the Chat tab's Now line read "Waiting for the engine
(CUDA error: device-side assert triggered Search for `cudaErrorAssert' in ...", and the watchdog
counted no failure (logs/watchdog.fails stayed 0).

  1  `server/gpu_health`: a CUDA error with a probe that fails marks the engine failed, once; an
     error the context survives (the probe answers) marks nothing; a plain error marks nothing;
     `/health`'s probe is rate-limited and marks a failure nobody's request saw; the words drop
     the driver's documentation pointer; the probe skips while a graph capture holds the lock
  2  through the server: a request whose generation raises the sticky error answers 503 with the
     words (`engine_failed`), never the driver's text; /health then answers 500 `failed` with the
     short reason under `failed`; `qse_up` is 0; the next request is refused at once (503,
     Retry-After) and never reaches the engine
  3  a failure a background job hit: /health's own probe finds it
  4  chatd: the refusal's `type` reaches `EngineError.kind`; the Now line is the engine's sentence
     for `engine_failed`, and an older engine's raw driver text is named in words

The watchdog's side (500 `failed` restarts after two checks, no loading grace) is in
tests/test_ops_scripts.py.

Run: python tests/test_gpu_health.py
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_app_loop as L  # noqa: E402  (sets the CPU environment before the engine imports)

from chatd.engine import Engine, EngineError  # noqa: E402
from chatd.runs import engine_words  # noqa: E402
from server import app, gpu_health  # noqa: E402

RAW = ("CUDA error: device-side assert triggered\nSearch for `cudaErrorAssert' in "
       "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more "
       "information.\nCUDA kernel errors might be async")
#: torch.AcceleratorError by name (torch 2.8+), without a GPU
AcceleratorError = type("AcceleratorError", (RuntimeError,), {})


class _Device:
    """The probe's answers: None while the context is fine, the sticky error once it is not."""

    def __init__(self):
        self.err, self.calls = None, 0

    def probe(self, device=None):
        self.calls += 1
        return self.err


def _inject():
    """A CUDA device the process has opened, whose probe this test decides."""
    gpu_health.reset()
    dev = _Device()
    real = (gpu_health.probe, gpu_health._on_cuda)
    gpu_health.probe = dev.probe
    gpu_health._on_cuda = lambda device=None: True
    return dev, real


def _restore(real):
    gpu_health.probe, gpu_health._on_cuda = real
    gpu_health.reset()


# ------------------------------------------------------------------ 1. the module

def test_only_a_context_that_stopped_answering_is_a_failure():
    dev, real = _inject()
    try:
        assert gpu_health.note(RuntimeError("the engine fell over"), "request") is None
        assert dev.calls == 0                                  # not a CUDA error: no probe
        # a CUDA error the context survives (a launch with a bad grid): the probe answers
        assert gpu_health.note(AcceleratorError("CUDA error: invalid argument"), "request") is None
        assert gpu_health.FAULT is None and dev.calls == 1
        dev.err = "CUDA error: device-side assert triggered"
        f = gpu_health.note(AcceleratorError(RAW), "memory writer")
        assert f is not None and f["error"] == "CUDA error: device-side assert triggered", f
        assert f["where"] == "memory writer"
        again = gpu_health.note(AcceleratorError("CUDA error: an illegal memory access"), "request")
        assert again is f and f["where"] == "memory writer"    # the first error is kept
        w = gpu_health.words()
        assert "needs a restart" in w and "device-side assert triggered" in w
        assert "Search for" not in w and "https://" not in w, w
        # /health's probe, rate-limited: one probe in PROBE_EVERY_S
        gpu_health.reset()
        dev.err, dev.calls = None, 0
        assert gpu_health.check("cuda", now=100.0) is None and dev.calls == 1
        assert gpu_health.check("cuda", now=101.0) is None and dev.calls == 1
        dev.err = "CUDA error: device-side assert triggered"
        assert gpu_health.check("cuda", now=103.0) is None and dev.calls == 1
        f = gpu_health.check("cuda", now=105.5)
        assert f is not None and f["where"] == "health probe" and dev.calls == 2
        # a CPU engine is never probed
        gpu_health.reset()
        gpu_health._on_cuda = real[1]
        assert gpu_health.check("cpu") is None and gpu_health.note(AcceleratorError(RAW)) is None
    finally:
        _restore(real)


def test_the_probe_waits_for_no_capture():
    """The probe is a CUDA call off the engine's thread: while a graph capture holds
    `engine.capture_lock.LOCK` it is skipped (no answer, no failure), as the disk tier's writer
    waits for a capture."""
    from engine import capture_lock
    gpu_health.reset()
    held, go = threading.Event(), threading.Event()

    def capture():
        with capture_lock.LOCK:
            held.set()
            go.wait(10)
    th = threading.Thread(target=capture)
    th.start()
    held.wait(10)
    try:
        assert gpu_health.probe() is None
    finally:
        go.set()
        th.join(10)
    # with the lock free the probe asks the device: on this CPU run there is none to answer
    assert gpu_health.probe() is not None


# ------------------------------------------------------------------ 2-3. through the server

def _server():
    L.serve()
    httpd = app.EngineServer(("127.0.0.1", 0), app.Handler)
    th = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    return httpd


def _call(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    raw = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=raw, headers={"Content-Type": "application/json"} if raw else {})
    r = c.getresponse()
    data = r.read().decode()
    c.close()
    return r.status, dict(r.getheaders()), data


def test_a_lost_context_fails_health_and_refuses_requests_in_words():
    import contextlib
    import io
    dev, real = _inject()
    httpd = _server()
    port = httpd.server_address[1]
    gen = app.generate_stream
    calls = []

    def broken(*a, **k):
        calls.append(1)
        dev.err = "CUDA error: device-side assert triggered"     # the context is gone now
        raise AcceleratorError(RAW)
        yield                                                    # noqa: unreachable (a generator)
    try:
        st, _, body = _call(port, "GET", "/health")
        assert st == 200 and json.loads(body)["status"] == "ok", body
        app.generate_stream = broken
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            st, hdr, body = _call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4})
        assert st == 503, (st, body)
        err = json.loads(body)["error"]
        assert err["type"] == "engine_failed" and "needs a restart" in err["message"], err
        assert "Search for" not in body and "cudaErrorAssert" not in body, body
        assert "[gpu] the CUDA context failed" in out.getvalue(), out.getvalue()
        st, _, body = _call(port, "GET", "/health")
        h = json.loads(body)
        assert st == 500 and h["status"] == "failed", (st, body)
        assert h["failed"] == "CUDA error: device-side assert triggered", h
        assert "needs a restart" in h["error"] and h["gpu"]["where"] == "request", h
        st, _, body = _call(port, "GET", "/metrics/up")
        assert st == 200 and body.strip().endswith("qse_up 0"), body
        n = len(calls)
        st, hdr, body = _call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4})
        assert st == 503 and hdr.get("Retry-After") == str(gpu_health.RETRY_AFTER_S), (st, hdr)
        assert json.loads(body)["error"]["type"] == "engine_failed"
        assert len(calls) == n, "a failed engine must not run a request"
        # a streamed request is refused the same way, before any event
        st, _, body = _call(port, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert st == 503 and "needs a restart" in body, (st, body)
    finally:
        app.generate_stream = gen
        httpd.shutdown()
        httpd.server_close()
        _restore(real)


def test_a_failure_only_a_background_job_saw_is_found_by_health():
    dev, real = _inject()
    httpd = _server()
    port = httpd.server_address[1]
    try:
        st, _, body = _call(port, "GET", "/health")
        assert st == 200, body
        dev.err = "CUDA error: device-side assert triggered"   # the memory writer's prefill, say
        gpu_health._LAST["t"] = 0.0                            # the rate limit's window passed
        st, _, body = _call(port, "GET", "/health")
        h = json.loads(body)
        assert st == 500 and h["status"] == "failed" and h["gpu"]["where"] == "health probe", h
    finally:
        httpd.shutdown()
        httpd.server_close()
        _restore(real)


# ------------------------------------------------------------------ 4. chatd

def test_the_chat_tab_says_it_in_words():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Refuse(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            raw = json.dumps({"error": {"message": gpu_health.words(
                {"error": "CUDA error: device-side assert triggered"}), "type": "engine_failed",
                "code": 503}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "60")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        try:
            Engine(f"http://127.0.0.1:{srv.server_address[1]}").stream(b"{}", {})
        except EngineError as exc:
            got = exc
        else:
            raise AssertionError("no refusal")
    finally:
        srv.shutdown()
        srv.server_close()
    assert got.status == 503 and got.retryable and got.kind == "engine_failed", vars(got)
    assert got.retry_after == 60.0
    words = engine_words(got)
    assert words.startswith("The engine needs a restart: its GPU stopped working"), words
    # an older engine's 500 with the driver's text, as the Chat tab showed it on 2026-10-21
    old = EngineError(500, RAW, True)
    assert engine_words(old) == "its GPU stopped working; the engine needs a restart"
    # anything else: its first line
    assert engine_words(EngineError(502, "bad gateway\nmore", True)) == "bad gateway"
    return f"Now line: {words}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:66s} ok  {got or ''}")
            passed += 1
    print(f"{passed} passed")
