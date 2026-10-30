"""The chunk size line is hex digits only, and the trailer loop has a bound.

`int(x, 16)` accepted `0x10`, `1_0` and `+5` as chunk sizes, and the 64-byte `readline`
cap split a longer line: seventy zeros and a `1` parsed as the last chunk and the rest
of the line became a trailer. A trailer stream had no bound at all. The parser now
accepts hex digits only, answers 400 when a size line does not end in `\\n` (it was too
long to read whole), and caps the trailers at the stdlib's 100 header lines.

A Content-Length is ASCII digits only: `int()` also took `+5`, `1_0`, ` 5 ` and `٥` (an
Arabic-Indic five), and the value sizes a pre-auth allocation. A trailer line that does
not end in `\\n` is a 400 too: the 1024-byte `readline` cap split it, and the lone `\\n`
counted as the blank line while the real one stayed on the wire for the next request.

Run: python tests/test_http_body.py
"""

from __future__ import annotations

import io
import os
import sys
from email.message import Message

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.http_body import BodyError, conflicting_framing, read_body  # noqa: E402


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


def test_a_good_chunked_body_decodes():
    # extensions and all
    ok = b"5;ext=x\r\nhello\r\n6\r\n world\r\n0\r\nX-t: 1\r\n\r\n"
    assert _body(ok, **{"Transfer-Encoding": "chunked"}) == b"hello world"


def test_a_chunk_size_is_hex_digits_only():
    # int(x, 16) leniency: the 0x prefix, an underscore, a plus sign
    for bad in (b"0x1\r\nh\r\n0\r\n\r\n", b"1_0\r\n" + b"x" * 16 + b"\r\n0\r\n\r\n",
                b"+1\r\nh\r\n0\r\n\r\n", b"0X1\r\nh\r\n0\r\n\r\n", b" 1 \x00\r\nh\r\n0\r\n\r\n"):
        _refused(bad, 400, **{"Transfer-Encoding": "chunked"})


def test_a_split_size_line_is_refused():
    # 70 zeros and a 1 was the last chunk, the rest a trailer
    _refused(b"0" * 70 + b"1\r\n5\r\nhello\r\n0\r\n\r\n", 400,
             **{"Transfer-Encoding": "chunked"})


def test_an_empty_size_line_is_refused():
    _refused(b"\r\n", 400, **{"Transfer-Encoding": "chunked"})


def test_the_trailers_are_capped():
    # 150 lines and no blank line is a 400, not a hang
    _refused(b"0\r\n" + b"x: 1\r\n" * 150, 400, **{"Transfer-Encoding": "chunked"})


def test_ninety_nine_trailer_lines_still_pass():
    assert _body(b"0\r\n" + b"x: 1\r\n" * 99 + b"\r\n",
                 **{"Transfer-Encoding": "chunked"}) == b""


def test_a_trailer_line_longer_than_the_cap_is_refused():
    # the 1024-byte readline cap splits it: the lone "\n" would count as the blank
    # line and the real one would stay on the wire for the next request
    _refused(b"0\r\n" + b"x" * 1023 + b"\r\n" + b"\r\n", 400,
             **{"Transfer-Encoding": "chunked"})


def test_the_content_length_spine_is_unchanged():
    # absent is no body, garbage and negatives 400
    assert _body(b"", **{"Content-Length": ""}) == b""
    _refused(b"", 400, **{"Content-Length": "abc"})
    _refused(b"", 400, **{"Content-Length": "-1"})


def test_a_content_length_is_ascii_digits_only():
    # int() also took "+5", "1_0", " 5 " and an Arabic-Indic five
    for bad in ("+5", "1_0", " 5 ", "٥", "0x5", "5.0", "5,0", ""):
        if bad == "":
            continue                      # an absent header is no body, covered above
        _refused(b"", 400, **{"Content-Length": bad})
    # the plain digits still work
    assert _body(b"hello", **{"Content-Length": "5"}) == b"hello"


def test_chunked_with_a_content_length_is_a_conflict():
    # RFC 9112 section 6.1: the two framings disagree about where the body ends
    h = _headers(**{"Transfer-Encoding": "chunked", "Content-Length": "5"})
    assert conflicting_framing(h) is True
    # either alone is not a conflict
    assert conflicting_framing(_headers(**{"Transfer-Encoding": "chunked"})) is False
    assert conflicting_framing(_headers(**{"Content-Length": "5"})) is False
    assert conflicting_framing(_headers()) is False


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
