"""SRV-777: Stop on a model call whose response has not started.

An engine sends no headers while a request waits for memory (or in its line), so `Engine.stream` blocks in
`getresponse()` and the run has no `Call` to close. The connection is handed to the run as soon as the request is
sent (`opened`): Stop shuts it, the blocked read returns at once with a retryable `EngineError`, and the engine
reads the close. A Stop that came before the connection existed shuts it when it opens.

Run: python tests/test_chatd_stop_pending.py
"""

from __future__ import annotations

import http.client
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import engine as engine_mod  # noqa: E402
from chatd.runs import Run  # noqa: E402


class _Conn:
    """An `http.client.HTTPConnection` as `shut` sees it, recording what was done to it."""

    def __init__(self, done: list):
        self.done = done
        self.sock = self

    def shutdown(self, how):
        self.done.append(("shutdown", how))

    def close(self):
        self.done.append("close")


def _bare_run() -> Run:
    r = Run.__new__(Run)
    r.stop_mode, r.call, r.child, r.pending = None, None, None, None
    r.wake, r.cancel, r.waiting = threading.Event(), threading.Event(), {}
    return r


def test_stop_shuts_the_connection_whose_response_has_not_started():
    done: list = []
    r = _bare_run()
    r._opened(_Conn(done))
    assert done == [] and r.pending is not None, "no Stop yet: the connection is kept, untouched"
    r.stop()
    assert done == [("shutdown", socket.SHUT_RDWR), "close"] and r.stop_mode == "stop", done
    assert r.cancel.is_set() and r.wake.is_set()
    return "pending connection shut on Stop"


def test_a_stop_before_the_connection_existed_shuts_it_when_it_opens():
    done: list = []
    r = _bare_run()
    r.stop()
    assert done == [] and r.stop_mode == "stop"
    r._opened(_Conn(done))
    assert done == [("shutdown", socket.SHUT_RDWR), "close"], done
    return "opened after Stop: shut at once"


def test_the_engines_silence_ends_when_the_connection_is_shut():
    """A server that reads the request and sends nothing (an engine waiting for memory): `stream` hands the
    connection over before it waits; `shut` from another thread ends the wait at once with a retryable error, and
    the server reads the close."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    seen = {}

    def server():
        c, _ = srv.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += c.recv(65536)
        seen["request"] = data.split(b"\r\n", 1)[0]
        c.settimeout(10)
        try:
            seen["eof"] = c.recv(1) == b""
        except OSError:
            seen["eof"] = True
        c.close()
    th = threading.Thread(target=server, daemon=True)
    th.start()
    got = {}
    opened = threading.Event()

    def call():
        try:
            engine_mod.Engine(f"http://127.0.0.1:{port}").stream(
                b"{}", {}, opened=lambda conn: (got.update(conn=conn), opened.set()))
        except engine_mod.EngineError as exc:
            got["error"] = exc
            got["t"] = time.monotonic()
    ct = threading.Thread(target=call, daemon=True)
    ct.start()
    assert opened.wait(5), "the connection was never handed over"
    time.sleep(0.3)                                # the call blocks on the response's headers
    assert "error" not in got
    t0 = time.monotonic()
    engine_mod.shut(got["conn"])
    ct.join(5)
    th.join(5)
    srv.close()
    assert isinstance(got.get("error"), engine_mod.EngineError) and got["error"].retryable, got
    assert got["t"] - t0 < 1.0, got["t"] - t0
    assert seen.get("request", b"").startswith(b"POST /v1/chat/completions") and seen.get("eof"), seen
    return f"ended {1e3 * (got['t'] - t0):.0f} ms after the shut; the server read the close"


class _ShutMidway(http.client.HTTPConnection):
    """A connection whose `getresponse()` meets a Stop after its state check and before it reads `self.sock`:
    http.client looks up `response_class` first, so the shut lands right there."""

    def connect(self):
        self.sock, self.peer = socket.socketpair()

    @property
    def response_class(self):
        engine_mod.shut(self)
        return http.client.HTTPResponse


class _BrokenResponse(http.client.HTTPConnection):
    """A connection whose response class itself fails while the socket is still there: a fault, not a shut."""

    def connect(self):
        self.sock, self.peer = socket.socketpair()

    @property
    def response_class(self):
        def broken(sock, method=None):
            raise AttributeError("not a shut")
        return broken


def test_a_shut_inside_getresponse_is_the_same_retryable_error():
    """Stop's `shut` can land inside `getresponse()` after its state check and before it reads `conn.sock`; `close()`
    has set that to None and http.client's response reads None (AttributeError, not an OSError). `stream` answers it
    as any shut connection: a retryable `EngineError`, so the run ends "stopped". An AttributeError with the socket
    still there is not hidden."""
    eng = engine_mod.Engine("http://127.0.0.1:9")
    eng._conn = lambda timeout: _ShutMidway(eng.host, eng.port, timeout=timeout)
    try:
        eng.stream(b"{}", {})
    except engine_mod.EngineError as exc:
        assert exc.retryable and exc.status == 0, (exc.status, exc.retryable)
        got = str(exc)
    else:
        raise AssertionError("a shut inside getresponse returned a call")
    eng._conn = lambda timeout: _BrokenResponse(eng.host, eng.port, timeout=timeout)
    try:
        eng.stream(b"{}", {})
    except AttributeError as exc:
        assert str(exc) == "not a shut", exc
    else:
        raise AssertionError("an AttributeError with the socket still there was hidden")
    return f"shut midway: EngineError ({got}); a fault with the socket set: AttributeError"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:66s} ok   {fn()}")
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:66s} FAIL ({exc})")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
