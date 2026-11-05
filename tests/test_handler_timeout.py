"""A connection has a socket timeout: a stalled request cannot hold a handler thread forever.

The handler used to run with no timeout at all (BaseRequestHandler.timeout was None): a client
that sent `POST` headers, a Content-Length and one byte, then stopped, pinned one handler thread
of the ThreadingHTTPServer per connection, forever, with zero requests served. The timeout is
now `Handler.timeout` seconds per socket op (60 s in production; the tests set it directly):

  a stalled body read     -> 408, the connection closes, the thread is free
  an idle keep-alive peer -> the stdlib's own timeout path: one log line, closed, no traceback
  a client that keeps
    sending               -> served, keep-alive and all -- the clock restarts at every op
  a reader that stops
    reading the answer    -> the write times out: abandoned, closed, no traceback

Run: python tests/test_handler_timeout.py
"""

from __future__ import annotations

import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import _setup  # noqa: E402  (first: the CPU environment, via test_app_loop)
from server import app  # noqa: E402

TIMEOUT_S = 0.3           # the production 60 s, at test speed
WAIT_S = 10.0             # how long a test waits for what the timeout must have done
_GET = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"


def _connect():
    return socket.create_connection(("127.0.0.1", _server.server_address[1]), timeout=WAIT_S)


def test_the_handler_ships_a_bounded_default_timeout():
    # the shipped value, captured before any test re-times the class: None was the whole
    # bug -- socketserver.setup() only calls settimeout() when the attribute is not None
    assert _AT_IMPORT is not None, "Handler.timeout ships None: no connection ever expires"
    assert 0 < _AT_IMPORT <= 300.0, _AT_IMPORT


def _serve():
    global _server
    _setup()
    app.Handler.timeout = TIMEOUT_S
    _server = app.EngineServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=_server.serve_forever, daemon=True).start()


_server = None
_AT_IMPORT = app.Handler.timeout       # the shipped value, captured AT IMPORT: a test run
                                       # in any order (plain pytest) reads it before _serve


def _get_health(s):
    """One GET /health on this connection, the response drained whole; the status line back."""
    s.sendall(_GET)
    s.settimeout(WAIT_S)
    buf = b""
    while b"\r\n\r\n" not in buf:
        got = s.recv(4096)
        assert got, f"closed before the headers: {buf!r}"
        buf += got
    head, _, rest = buf.partition(b"\r\n\r\n")
    n = int(dict(l.split(": ", 1) for l in head.decode().split("\r\n")[1:]
                 if ": " in l).get("Content-Length", 0))
    while len(rest) < n:
        got = s.recv(4096)
        assert got, "closed mid-body"
        rest += got
    return head.split(b"\r\n", 1)[0].decode()


def test_a_body_stalled_mid_flight_is_a_408_and_the_thread_dies():
    _serve()
    try:
        s = _connect()
        try:
            s.sendall(b"POST /v1/dashboard/session HTTP/1.1\r\nHost: x\r\nContent-Type: "
                      b"application/json\r\nContent-Length: 5000\r\n\r\n{}")   # 4998 to come
            # before the fix this recv never returns: the read waits for bytes that never
            # arrive, and the handler thread is never released
            s.settimeout(WAIT_S)
            buf = b""
            while b"\r\n" not in buf:
                got = s.recv(4096)
                if not got:
                    raise AssertionError(f"connection closed, no status line: {buf!r}")
                buf += got
            status = buf.split(b"\r\n", 1)[0].decode()
            assert status.startswith("HTTP/1.1 408"), status
            while s.recv(4096):                    # and then the close: the thread is gone
                pass
        finally:
            s.close()
    finally:
        _server.shutdown()


def test_an_idle_keep_alive_connection_is_closed_by_the_timeout():
    _serve()
    try:
        s = _connect()
        try:
            assert _get_health(s).startswith("HTTP/1.1 200")   # served, keep-alive stays on
            # the wait for the NEXT request line is the stdlib's own TimeoutError path:
            # one log line, close_connection, no traceback. Before the timeout existed this
            # recv blocked forever -- a healthy connection, held forever by nobody.
            s.settimeout(TIMEOUT_S * 10)
            try:
                got = s.recv(4096)
            except TimeoutError:
                raise AssertionError("the idle keep-alive connection is still open: no timeout")
            assert got == b"", f"expected the timeout to close it, read {got!r}"
        finally:
            s.close()
    finally:
        _server.shutdown()


def test_a_client_that_keeps_sending_is_served_keep_alive_and_all():
    _serve()
    try:
        s = _connect()
        try:
            for _ in range(3):                    # three full requests on one connection:
                assert _get_health(s).startswith("HTTP/1.1 200")   # none of them is stale
        finally:
            s.close()
    finally:
        _server.shutdown()


def test_a_forward_that_lifts_the_timeout_leaves_the_next_request_guarded():
    # the reported sequence: GET /health, GET /v1/chat-app/capabilities (the chat forward
    # runs chat_forward._relay, which sets the socket timeout to None while it streams),
    # then a stalled POST body on the SAME keep-alive connection. With the timeout never
    # re-armed the 408 path can never run: the handler thread waits on bytes that never
    # come, 10x the timeout and counting.
    import http.server

    class FakeChatd(http.server.BaseHTTPRequestHandler):
        # HTTP/1.0 on purpose: the real answer here is small and the relay drains to the
        # close — a keep-alive upstream would hold this relay thread, not our concern
        def do_GET(self):                                   # capabilities: a sized 200
            raw = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeChatd)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    os.environ["QSE_CHAT_URL"] = f"http://127.0.0.1:{fake.server_address[1]}"
    os.environ["QSE_CHAT_FORWARD_KEY"] = "test-key"
    _serve()
    try:
        s = _connect()
        try:
            assert _get_health(s).startswith("HTTP/1.1 200")
            s.sendall(b"GET /v1/chat-app/capabilities HTTP/1.1\r\nHost: x\r\n\r\n")
            s.settimeout(WAIT_S)
            buf = b""
            while b"\r\n\r\n" not in buf:
                got = s.recv(4096)
                assert got, "closed before the capabilities answer"
                buf += got
            head, _, rest = buf.partition(b"\r\n\r\n")
            n = int(dict(l.split(": ", 1) for l in head.decode().split("\r\n")[1:]
                         if ": " in l).get("Content-Length", 0))
            while len(rest) < n:
                rest += s.recv(4096)
            assert b"200" in head.split(b"\r\n", 1)[0], head
            # the next request on this connection: a stalled body must 408 again
            s.sendall(b"POST /v1/dashboard/session HTTP/1.1\r\nHost: x\r\nContent-Type: "
                      b"application/json\r\nContent-Length: 5000\r\n\r\n{}")   # 4998 to come
            buf = b""
            while b"\r\n" not in buf:
                got = s.recv(4096)
                if not got:
                    raise AssertionError(f"closed, no status line: {buf!r}")
                buf += got
            status = buf.split(b"\r\n", 1)[0].decode()
            assert status.startswith("HTTP/1.1 408"), \
                f"the timeout was never re-armed after the forward: {status}"
        finally:
            s.close()
    finally:
        _server.shutdown()
        fake.shutdown()
        os.environ.pop("QSE_CHAT_URL", None)
        os.environ.pop("QSE_CHAT_FORWARD_KEY", None)


class _Stalled:
    """A socket writer whose reader stopped reading: every write times out."""

    def __init__(self):
        self.writes = 0

    def write(self, data):
        self.writes += 1
        raise TimeoutError("timed out")

    def flush(self):
        pass


def test_a_write_that_times_out_is_a_client_gone_and_closes_the_connection():
    from test_app_loop import Req
    req = Req("/health", {})
    raw = _Stalled()
    req.wfile = app._GoneOnTimeout(raw, req)
    req.close_connection = False
    # _send_bytes treats it as a reader that left: no exception out, the connection closes
    req._send_bytes(200, b"{}")
    assert req.close_connection is True
    assert raw.writes == 1
    # a later write fails at once instead of waiting out a second timeout
    try:
        req.wfile.write(b"data: [DONE]\n\n")
    except app.ClientGone:
        pass
    else:
        raise AssertionError("a write after the timeout went through")
    assert raw.writes == 1, "the second write waited on the socket again"
    req.wfile.flush()                   # the stdlib's flush after the handler: no error


class _SmallSendBuf(app.Handler):
    """The server side of the connection with a small send buffer: a reader that stops reading
    fills it, and the write that follows waits out the timeout."""

    def setup(self):
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        super().setup()


def test_a_reader_that_stops_reading_mid_stream_is_abandoned():
    import time
    from test_app_loop import serve
    serve(max_len=4096)
    app.Handler.timeout = TIMEOUT_S
    tracebacks = []
    real_gen, real_exc = app.generate_stream, app.logbuf.print_exc

    class Server(app.EngineServer):
        def handle_error(self, request, client_address):
            tracebacks.append("handle_error")

    app.generate_stream = lambda *a, **k: iter([ord("a")] * 4000)
    app.logbuf.print_exc = lambda *a, **k: tracebacks.append("print_exc")
    server = Server(("127.0.0.1", 0), _SmallSendBuf)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    before = dict(app.INFLIGHT)
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        s.settimeout(WAIT_S)
        s.connect(("127.0.0.1", server.server_address[1]))
        body = b'{"prompt": "ab", "max_tokens": 4000, "stream": true}'
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: "
                  b"application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        # and then nothing is read: the buffers fill, the stream's next write times out
        t0 = time.monotonic()
        while app.INFLIGHT["abandoned"] == before["abandoned"]:
            assert time.monotonic() - t0 < WAIT_S, f"never abandoned: {dict(app.INFLIGHT)}"
            time.sleep(0.02)
        got = b""
        while True:                                 # what reached the buffers, then the close
            part = s.recv(65536)
            if not part:
                break
            got += part
        assert got.startswith(b"HTTP/1.1 200"), got[:80]
        assert b"[DONE]" not in got and b'"error"' not in got, got[-300:]
        assert app.INFLIGHT["errors"] == before["errors"], "a reader that left is not an error"
        assert tracebacks == [], tracebacks
    finally:
        s.close()
        server.shutdown()
        app.generate_stream, app.logbuf.print_exc = real_gen, real_exc


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
