"""The login throttle keys on the TCP peer unless the peer is a configured proxy.

`X-Real-IP` used to key it, and that header belongs to the client: a script rotating
the value per attempt kept every failure in its own bucket and the throttle never
fired. Keying on the TCP peer always, though, puts every user behind a same-box
nginx into one bucket: five wrong passwords from anyone lock everyone out. Policy:
believe the header ONLY from a trusted proxy — loopback by default, plus
Auth.trusted_proxies naming a proxy's own address — and key every other peer on
itself. Never both at once: a spoofable value must not merge with real IPs.

Run: python tests/test_login_throttle_key.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import ADMIN, _setup, call  # noqa: E402


def _fail(n, peer="127.0.0.1", spoof=None):
    for i in range(n):
        headers = {"X-Real-IP": spoof(i)} if spoof else {}
        code, _, _ = call("POST", "/v1/dashboard/session", headers, {"token": f"wrong-{i}"},
                          peer=peer)
        assert code == 401, (i, code)


def test_a_remote_peer_with_a_spoofed_header_does_not_evade_the_throttle():
    """The header stays inert from a peer that is not a configured proxy: five
    failures filed under rotating fakes all land in the peer's own bucket."""
    _setup()
    _fail(5, peer="10.9.8.7", spoof=lambda i: f"203.0.113.{i}")   # one fake per attempt
    code, head, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "203.0.113.99"},
                         {"token": ADMIN}, peer="10.9.8.7")       # right token, 6th try
    assert code == 429 and "Retry-After" in head, f"spoofed X-Real-IP evaded the throttle: {code}"


def test_the_peer_still_gets_blocked_and_a_second_peer_does_not():
    """A no-header attempt from any peer — trusted proxy included, the header is
    then absent — keys on the peer itself."""
    _setup()
    _fail(5)                                               # five from the default peer
    code, _, _ = call("POST", "/v1/dashboard/session", {}, {"token": ADMIN})
    assert code == 429, "the peer's own sixth attempt is blocked"
    code, _, _ = call("POST", "/v1/dashboard/session", {}, {"token": ADMIN}, peer="10.9.8.7")
    assert code == 204, "another peer is untouched"


def test_the_throttle_is_per_real_client_behind_the_default_loopback_proxy():
    """The nginx shape: five failures from one real client's IP must not lock out
    another client's correct login."""
    _setup()
    _fail(5, spoof=lambda i: "203.0.113.7")                 # five from real client A
    code, head, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "203.0.113.7"},
                         {"token": ADMIN})                 # A's sixth: throttled
    assert code == 429, f"the same real client's loop is not throttled: {code}"
    code, _, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "198.51.100.1"},
                      {"token": ADMIN})                    # another client: free
    assert code == 204, "one user's five wrong passwords locked out another"


def test_the_throttle_state_is_not_shared_with_a_spoofed_key():
    """The header must be inert from an untrusted peer in both directions: five
    failures that a remote client filed under a guessed header value leave the
    peers' own buckets empty."""
    _setup()
    _fail(5, peer="10.9.8.6", spoof=lambda i: "198.51.100.1")
    code, _, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "198.51.100.1"},
                      {"token": ADMIN}, peer="10.9.8.7")    # the header hits nothing here
    assert code == 204, "the header alone is not a throttle key"
    code, _, _ = call("POST", "/v1/dashboard/session", {}, {"token": ADMIN},
                      peer="10.9.8.6")                     # ...and 10.9.8.6 keyed itself
    assert code == 429, "the remote peer's own failures were not counted"


def test_the_opt_in_list_adds_a_lan_proxy():
    """A named proxy is trusted: its X-Real-IP becomes the key, per real client.
    The list is Auth.trusted_proxies -- from_env fills it from TANDEM_TRUSTED_PROXY,
    a test hands it to the constructor."""
    _setup(trusted_proxies=["10.9.8.7"])
    _fail(5, peer="10.9.8.7", spoof=lambda i: "203.0.113.99")
    code, _, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "203.0.113.99"},
                      {"token": ADMIN}, peer="10.9.8.7")
    assert code == 429, f"the opt-in proxy's client was not throttled: {code}"
    code, _, _ = call("POST", "/v1/dashboard/session", {"X-Real-IP": "198.51.100.2"},
                      {"token": ADMIN}, peer="10.9.8.7")
    assert code == 204, "a different real client behind the opt-in proxy was locked"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n = ok = 0
    for name, fn in tests:
        n += 1
        try:
            fn()
            print(f"  {name:64s} ok")
            ok += 1
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {name:64s} FAIL {type(exc).__name__}: {exc}")
    print(f"{ok}/{n}")
    sys.exit(0 if ok == n else 1)
