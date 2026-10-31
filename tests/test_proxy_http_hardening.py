"""The proxy's pre-auth body read has a spine, and the login throttle cannot be dodged.

`_read_raw` trusted the client's `Content-Length`: `-1` became `rfile.read(-1)` --
read-until-EOF, a permanently hung handler thread per keep-alive connection -- and any
client-sized number buffered that much before any auth would answer. The shared module
`server/http_body.py` validates the header first: a number, not negative, within the
body cap, else a 400/413 and a closed connection. And the throttle key is
the TCP peer, never the `X-Real-IP` header (the client's to write: a script rotating it
filed every failure in its own bucket and the 429 never fired). The auth table's overflow
used to `clear()` -- a flood of one-shot keys wiped every active ban.

    a negative Content-Length          400 at once, not a hung thread
    a Content-Length over the cap      413 before the body arrives
    rotating X-Real-IP failures        the peer's bucket still trips the 429
    a 10k-key flood of failures        an active ban survives the eviction

Run: python tests/test_proxy_http_hardening.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from proxy_util import Proxy  # noqa: E402

ADMIN = "adm-" + "a1b2c3d4" * 6
CAP = 64 * 1024


def _status_line(s: socket.socket) -> str:
    buf = b""
    while b"\r\n" not in buf:
        got = s.recv(4096)
        if not got:
            raise AssertionError(f"connection closed, no status line: {buf!r}")
        buf += got
    return buf.split(b"\r\n", 1)[0].decode()


def _raw_post(port: int, path: str, content_length: str, body: bytes = b"",
              headers: dict | None = None) -> str:
    """One request with the header written by hand: http.client refuses the absurd ones."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        head = (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                f"Content-Length: {content_length}\r\n"
                + "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
                + "\r\n").encode()
        s.sendall(head + body)
        return _status_line(s)
    finally:
        s.close()


def test_negative_content_length_is_a_400_not_a_hang():
    with Proxy(env={"QSE_ADMIN_TOKEN": ADMIN}, login="on") as px:
        for bad in ("-1", "-3"):
            status = _raw_post(px.port, "/v1/dashboard/session", bad)
            assert status.startswith("HTTP/1.1 400"), f"Content-Length: {bad} -> {status}"


def test_a_nonnumeric_content_length_is_a_400():
    with Proxy(env={"QSE_ADMIN_TOKEN": ADMIN}, login="on") as px:
        status = _raw_post(px.port, "/v1/dashboard/session", "not-a-number")
        assert status.startswith("HTTP/1.1 400"), status


def test_an_oversized_content_length_is_a_413_before_the_body_arrives():
    from proxy import app as proxy_app
    saved = proxy_app.http_body.MAX_BODY_BYTES
    proxy_app.http_body.MAX_BODY_BYTES = CAP
    try:
        with Proxy(env={"QSE_ADMIN_TOKEN": ADMIN}, login="on") as px:
            # the body promised but never sent: a 413 must not wait for it
            status = _raw_post(px.port, "/v1/dashboard/session", str(CAP + 1))
            assert status.startswith("HTTP/1.1 413"), status
            # a body within the cap still reads
            status = _raw_post(px.port, "/v1/dashboard/session", "2", b"{}")
            assert status.startswith("HTTP/1.1 401"), status
    finally:
        proxy_app.http_body.MAX_BODY_BYTES = saved


def test_a_chunked_body_over_the_cap_is_a_413():
    from proxy import app as proxy_app
    saved = proxy_app.http_body.MAX_BODY_BYTES
    proxy_app.http_body.MAX_BODY_BYTES = CAP
    try:
        with Proxy(env={"QSE_ADMIN_TOKEN": ADMIN}, login="on") as px:
            s = socket.create_connection(("127.0.0.1", px.port), timeout=5)
            try:
                head = (b"POST /v1/dashboard/session HTTP/1.1\r\nHost: x\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Transfer-Encoding: chunked\r\n\r\n")
                s.sendall(head)                       # the size line alone busts the cap
                s.sendall(b"%x\r\n" % (CAP + 1))
                assert _status_line(s).startswith("HTTP/1.1 413")
            finally:
                s.close()
    finally:
        proxy_app.http_body.MAX_BODY_BYTES = saved


def test_five_wrong_tokens_from_one_client_trip_the_throttle():
    """Live socket, no X-Real-IP anywhere: every attempt is keyed as the peer it came
    from, so the 6th failure from ONE client is the 429. The old key — believing the
    header always — filed every failure in its own bucket and the 429 never fired; the
    matrix of WHO may send the header is pinned per-rule in the login-key test."""
    with Proxy(env={"QSE_ADMIN_TOKEN": ADMIN}, login="on") as px:
        statuses = []
        for i in range(8):
            code, _, _ = px.post("/v1/dashboard/session", {"token": "wrong"})
            statuses.append(code)
        assert statuses[:5] == [401] * 5, statuses
        assert statuses[5] == 429, f"the throttle never fired: {statuses}"


def test_a_key_flood_does_not_wipe_an_active_ban():
    """The old `self._fails.clear()` at the 10k cap handed an attacker a free pass: the
    flood itself unbanned the peer the throttle had just caught."""
    from server import auth as auth_mod
    t0 = time.time()
    a = auth_mod.Auth(admin_token=ADMIN, login=True, clock=lambda: t0)
    a.login_failed("victim")
    a.login_failed("victim")
    a.login_failed("victim")
    a.login_failed("victim")
    a.login_failed("victim")                       # banned: five fails inside the window
    assert a.login_blocked("victim")
    for i in range(auth_mod.LOGIN_MAX_KEYS + 500):  # the flood, over the cap
        a.login_failed(f"203.0.113.{i}")
    assert a.login_blocked("victim"), "the flood wiped the active ban"
    assert len(a._fails) <= auth_mod.LOGIN_MAX_KEYS, "the cap no longer bounds the table"



def test_the_login_key_follows_the_proxy_trust():
    """Loopback is trusted by default (the nginx-on-this-host shape: every real client
    shares the 127.0.0.1 peer); TANDEM_TRUSTED_PROXY adds a LAN proxy; from any other
    peer X-Real-IP is noise."""
    from server import auth as auth_mod
    plain = auth_mod.Auth(admin_token=ADMIN, login=True)
    hdrs = {"X-Real-IP": "198.51.100.7"}
    assert plain.login_key("203.0.113.9", hdrs) == "203.0.113.9"
    assert plain.login_key("127.0.0.1", hdrs) == "198.51.100.7"     # loopback, no opt-in
    assert plain.login_key("::1", hdrs) == "198.51.100.7"
    assert plain.login_key("::ffff:127.0.0.1", hdrs) == "198.51.100.7"
    trusted = auth_mod.Auth(admin_token=ADMIN, login=True, trusted_proxies=["192.0.2.2"])
    assert trusted.login_key("192.0.2.2", hdrs) == "198.51.100.7"
    assert trusted.login_key("203.0.113.9", hdrs) == "203.0.113.9"   # not the proxy: noise
    assert trusted.login_key("192.0.2.2", {}) == "192.0.2.2"        # proxy, no header


def test_the_overflow_sweep_lands_at_ninety_percent():
    """One sweep per 10 % of the table: after a flood the table sits at the low
    mark, not one under the cap, so the next thousand logins sweep nothing."""
    from server import auth as auth_mod
    a = auth_mod.Auth(admin_token=ADMIN, login=True, clock=lambda: 0.0)
    for i in range(auth_mod.LOGIN_MAX_KEYS + 1):
        a.login_failed(f"203.0.113.{i}")
    mark = auth_mod.LOGIN_MAX_KEYS - auth_mod.LOGIN_MAX_KEYS // 10
    assert len(a._fails) == mark, f"swept to {len(a._fails)}, not the {mark} low mark"

def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"PASS ({len(tests)} tests)")


if __name__ == "__main__":
    _main()