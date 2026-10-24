"""The pre-auth body is capped, and a negative Content-Length is a 400, not a hung thread.

`_read()` trusted the client's `Content-Length`: `4294967296` got that many bytes buffered
in full before any auth answered, and `-1` became `rfile.read(-1)` -- read-until-EOF on a
keep-alive connection, a permanently hung handler thread per connection, zero requests
needed. The size is now validated before the first byte is read: a number, within
`MAX_BODY_BYTES`, or a 400/413 and the connection closes.

Run: python tests/test_body_limits.py
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import _setup  # noqa: E402  (first: the CPU environment, via test_app_loop)
from server import app  # noqa: E402

CAP = 64 * 1024


def _status_line(s):
    buf = b""
    while b"\r\n" not in buf:
        got = s.recv(4096)
        if not got:
            raise AssertionError(f"connection closed, no status line: {buf!r}")
        buf += got
    return buf.split(b"\r\n", 1)[0].decode()


def _post(url_body, content_length):
    s = socket.create_connection(("127.0.0.1", _server.server_address[1]), timeout=5)
    try:
        head = (f"POST /v1/dashboard/session HTTP/1.1\r\nHost: x\r\nContent-Type: "
                f"application/json\r\nContent-Length: {content_length}\r\n\r\n").encode()
        s.sendall(head + url_body)
        return _status_line(s)
    finally:
        s.close()


def _serve():
    global _server
    _setup()
    app.http_body.MAX_BODY_BYTES = CAP             # a real budget, a testable number
    _server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=_server.serve_forever, daemon=True).start()


_server = None


def test_negative_content_length_is_a_400():
    _serve()
    try:
        for bad in ("-1", "-3"):
            status = _post(b"", bad)
            assert status.startswith("HTTP/1.1 400"), f"Content-Length: {bad} -> {status}"
    finally:
        _server.shutdown()


def test_a_nonnumeric_content_length_is_a_400():
    _serve()
    try:
        status = _post(b"", "not-a-number")
        assert status.startswith("HTTP/1.1 400"), status
    finally:
        _server.shutdown()


def test_an_oversized_content_length_is_a_413_before_the_body_arrives():
    _serve()
    try:
        # the promise alone is refused: the body never has to be sent
        status = _post(b"", "999999999")
        assert status.startswith("HTTP/1.1 413"), status
    finally:
        _server.shutdown()


def test_a_normal_body_is_served_as_before():
    _serve()
    try:
        body = b'{"token": "wrong"}'
        status = _post(body, str(len(body)))
        assert status.startswith("HTTP/1.1 401"), status   # the usual wrong-token answer
    finally:
        _server.shutdown()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
