"""The chunk size line is hex digits only, and the trailer loop has a bound.

`int(x, 16)` accepted `0x10`, `1_0` and `+5` as chunk sizes, and the 64-byte `readline`
cap split a longer line: seventy zeros and a `1` parsed as the last chunk and the rest
of the line became a trailer. A trailer stream had no bound at all. The parser now
accepts hex digits only, answers 400 when a size line does not end in `\\n` (it was too
long to read whole), and caps the trailers at the stdlib's 100 header lines.

Run: python tests/test_http_body.py
"""

from __future__ import annotations

import io
import os
import sys
from email.message import Message

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.http_body import BodyError, read_body  # noqa: E402


def _headers(**kw):
    m = Message()                                  # what BaseHTTPRequestHandler carries
    for k, v in kw.items():
        m[k] = v
    return m


def _body(raw: bytes, **hdrs) -> bytes:
    return read_body(io.BytesIO(raw), _headers(**hdrs))


def _refused(raw: bytes, code: int, **hdrs):
    try:
        _body(raw, **hdrs)
    except BodyError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code} ({exc.message})"
        return exc
    raise AssertionError(f"expected BodyError {code}, got a body")


def run():
    # a good chunked body still decodes, extensions and all
    ok = b"5;ext=x\r\nhello\r\n6\r\n world\r\n0\r\nX-t: 1\r\n\r\n"
    assert _body(ok, **{"Transfer-Encoding": "chunked"}) == b"hello world"
    # int(x, 16) leniency: the 0x prefix, an underscore, a plus sign
    for bad in (b"0x1\r\nh\r\n0\r\n\r\n", b"1_0\r\n" + b"x" * 16 + b"\r\n0\r\n\r\n",
                b"+1\r\nh\r\n0\r\n\r\n", b"0X1\r\nh\r\n0\r\n\r\n", b" 1 \x00\r\nh\r\n0\r\n\r\n"):
        _refused(bad, 400, **{"Transfer-Encoding": "chunked"})
    # the split size line: 70 zeros and a 1 was the last chunk, the rest a trailer
    _refused(b"0" * 70 + b"1\r\n5\r\nhello\r\n0\r\n\r\n", 400,
             **{"Transfer-Encoding": "chunked"})
    # an empty size line is not a size either
    _refused(b"\r\n", 400, **{"Transfer-Encoding": "chunked"})
    # the trailers are capped: 150 lines and no blank line is a 400, not a hang
    _refused(b"0\r\n" + b"x: 1\r\n" * 150, 400, **{"Transfer-Encoding": "chunked"})
    # 99 trailer lines then the blank line still passes the cap
    assert _body(b"0\r\n" + b"x: 1\r\n" * 99 + b"\r\n",
                 **{"Transfer-Encoding": "chunked"}) == b""
    # the Content-Length spine, unchanged: absent is no body, garbage and negatives 400
    assert _body(b"", **{"Content-Length": ""}) == b""
    _refused(b"", 400, **{"Content-Length": "abc"})
    _refused(b"", 400, **{"Content-Length": "-1"})
    print("ok")


if __name__ == "__main__":
    run()
