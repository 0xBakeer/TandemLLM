"""A chunked request body is consumed: two of them down one keep-alive socket get two
clean answers.

The handler read only `Content-Length` bodies. A `Transfer-Encoding: chunked` request was
answered with its chunks still on the wire, and the next request the connection carried was
parsed starting from the middle of that body -- the second response on a keep-alive
connection came out as the server's confusion (a 501 off a chunk-size line). The handler now
decodes the chunked framing itself, and closes any connection whose body it did not read.

Run: python tests/test_chunked_keepalive.py
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


def _chunked_req(path, payload):
    body = b""
    for i in range(0, len(payload), 7):             # deliberately more than one chunk
        part = payload[i:i + 7]
        body += b"%x\r\n%s\r\n" % (len(part), part)
    body += b"0\r\n\r\n"
    head = (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            f"Transfer-Encoding: chunked\r\n\r\n").encode()
    return head + body


def _read_response(s, buf):
    """One HTTP/1.1 response off the socket: status line, headers, the Content-Length body."""
    while b"\r\n\r\n" not in buf:
        got = s.recv(4096)
        if not got:
            raise AssertionError(f"connection closed mid-headers: {buf!r}")
        buf += got
    head, _, buf = buf.partition(b"\r\n\r\n")
    status = head.split(b" ", 2)[1].decode()
    n = 0
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            n = int(line.split(b":", 1)[1])
    while len(buf) < n:
        got = s.recv(4096)
        if not got:
            raise AssertionError("connection closed mid-body")
        buf += got
    return status, head, buf[:n], buf[n:]


def test_two_chunked_requests_on_one_connection_are_two_clean_answers():
    _setup()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=10)
        try:
            buf = b""
            for i in range(2):                       # two wrong logins: two 401s, one socket
                s.sendall(_chunked_req("/v1/dashboard/session",
                                       b'{"token": "wrong-' + str(i).encode() + b'"}'))
                status, head, body, buf = _read_response(s, buf)
                assert status == "401", f"request {i}: HTTP {status} {head[:120]!r} {body[:80]!r}"
            # and the connection is still a usable HTTP connection
            s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            status, head, body, buf = _read_response(s, buf)
            assert status == "200", f"after the chunked pair: HTTP {status} {head[:120]!r}"
        finally:
            s.close()
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
