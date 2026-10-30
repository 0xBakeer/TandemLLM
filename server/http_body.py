"""The request-body framing both servers share (HTTP, issue #9).

One Content-Length spine and one chunked parser, stdlib only: `server/app.py` and
`proxy/app.py` used to carry the same framing line for line, and a fix had to land
twice. `read_body(rfile, headers)` answers the body whole or not at all; anything
unreadable or too big is a BodyError whose connection must close (a promise left
half-read is the next request parsed from the middle of a body).

`MAX_BODY_BYTES` is looked up per request through this module global, so a test or
an embedder can set it; QSE_MAX_BODY_MB overrides at import.
"""

from __future__ import annotations

import os
import re

MAX_BODY_BYTES = int(os.environ.get("QSE_MAX_BODY_MB", 512)) * (1 << 20)


class BodyError(Exception):
    """The request's framing is unreadable or too big: a 400/413 with this code and message.
    The body behind it was not read, so the connection closes (handle_one_request)."""

    def __init__(self, code: int, type_: str, message: str):
        super().__init__(message)
        self.code, self.type, self.message = code, type_, message


def read_body(rfile, headers) -> bytes:
    """The request's body, consumed whole or not at all. An answer written with the
    body still unread is a desync one request away: every path out of here has read
    to the end of the framing, and BodyError closes the connection."""
    if "chunked" in (headers.get("Transfer-Encoding") or "").lower():
        return _read_chunked(rfile)              # Transfer-Encoding outranks Content-Length
    n = _content_length(headers)                 # validated before a single byte is read
    return rfile.read(n)


def _content_length(headers) -> int:
    """The promised body size, validated before a single byte of it is read. This header
    is the client's and it sizes a pre-auth allocation: 4294967296 was once a
    multi-GB read on a handler thread, and -1 was rfile.read(-1) -- read-until-EOF, a
    permanently hung thread on a keep-alive connection."""
    raw = headers.get("Content-Length")
    if raw is None or not raw.strip():
        return 0                                 # an absent header is no body, as before
    try:
        n = int(raw)
    except ValueError:
        raise BodyError(400, "invalid_request_error",
                        "Content-Length: not a number") from None
    if n < 0:
        raise BodyError(400, "invalid_request_error", "Content-Length: negative")
    if n > MAX_BODY_BYTES:
        raise BodyError(413, "too_large",
                        f"a request body is at most {MAX_BODY_BYTES} bytes")
    return n


# A chunk size is hex digits, nothing else. int(x, 16) also accepted 0x10, 1_0 and +5;
# a size line longer than the readline cap would then be re-read as further chunks, so
# seventy zeros and a '1' parsed as the last chunk and the rest of the line became a
# trailer. An unterminated (too long) size line is a 400, as the stdlib does.
_CHUNK_SIZE = re.compile(rb"[0-9a-fA-F]+")


def _read_chunked(rfile) -> bytes:
    """A chunked body, RFC 9112 7.1: size line, that many bytes, CRLF, until the zero
    chunk, then the trailers to their empty line. A framing error mid-stream leaves the
    stream at an unknown offset, and only a closed connection says so."""
    out = bytearray()
    while True:
        line = rfile.readline(64)
        if line and not line.endswith(b"\n"):
            raise BodyError(400, "invalid_request_error",
                            "transfer-encoding: chunk size line too long")
        token = line.split(b";", 1)[0].strip()   # extensions cut; hex digits only
        if not _CHUNK_SIZE.fullmatch(token):
            raise BodyError(400, "invalid_request_error", "transfer-encoding: bad chunk size")
        size = int(token, 16)
        if size == 0:
            for _ in range(100):                 # the trailers, to a blank line; capped
                t = rfile.readline(1024)         # at the stdlib's 100 header lines
                if not t or t in (b"\r\n", b"\n"):
                    break
            else:
                raise BodyError(400, "invalid_request_error",
                                "transfer-encoding: too many trailer lines")
            break
        if len(out) + size > MAX_BODY_BYTES:
            raise BodyError(413, "too_large", f"a request body is at most "
                                              f"{MAX_BODY_BYTES} bytes")
        chunk = rfile.read(size + 2)
        if len(chunk) != size + 2 or not chunk.endswith(b"\r\n"):
            raise BodyError(400, "invalid_request_error", "transfer-encoding: truncated "
                                                          "chunk")
        out += chunk[:size]
    return bytes(out)
