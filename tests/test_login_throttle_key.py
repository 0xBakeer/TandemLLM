"""The login throttle keys on the TCP peer, never on a client header.

`X-Real-IP` used to key it, and that header belongs to the client: a script rotating the value
per attempt kept every failure in its own bucket and the throttle never fired. There is no
trusted-proxy opt-in that would make the header believable (`--trust-loopback` reads its
absence as a signal of trust, never its value), so the header is not read at all: the peer is
the key, and a spoofed header changes nothing.

Run: python tests/test_login_throttle_key.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import ADMIN, _setup, call  # noqa: E402


def _fail(n, spoof=None):
    for i in range(n):
        headers = {"X-Real-IP": spoof(i)} if spoof else {}
        code, _, _ = call("POST", "/v1/dashboard/session", headers, {"token": f"wrong-{i}"})
        assert code == 401, (i, code)


def test_rotating_x_real_ip_does_not_evade_the_throttle():
    _setup()
    _fail(5, spoof=lambda i: f"203.0.113.{i}")             # one "address" per attempt
    code, head, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "203.0.113.99"},
                         {"token": ADMIN})                 # the right token, the sixth try
    assert code == 429 and "Retry-After" in head, f"spoofed X-Real-IP evaded the throttle: {code}"


def test_the_peer_still_gets_blocked_and_a_second_peer_does_not():
    _setup()
    _fail(5)                                               # five from the default peer
    code, _, _ = call("POST", "/v1/dashboard/session", {}, {"token": ADMIN})
    assert code == 429, "the peer's own sixth attempt is blocked"
    code, _, _ = call("POST", "/v1/dashboard/session", {}, {"token": ADMIN}, peer="10.9.8.7")
    assert code == 204, "another peer is untouched"


def test_the_throttle_state_is_not_shared_with_a_spoofed_key():
    """The header must be inert in both directions: five failures that a client filed under
    a guessed header value leave every peer's own bucket empty."""
    _setup()
    _fail(5, spoof=lambda i: "198.51.100.1")
    code, _, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "198.51.100.1"},
                      {"token": ADMIN}, peer="10.9.8.7")    # the header hits nothing
    assert code == 204, "the header alone is not a throttle key"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
