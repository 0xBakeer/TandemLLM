"""_splice's pumps are idle-bounded (chatd.egress): a tunnel where neither side ever
speaks (half-open) ends in about `idle` instead of standing forever; traffic still
flows while it flows. socketpair + tiny timeouts, CPU.

Run: python tests/test_chatd_egress_splice.py
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.egress import _splice


def test_a_quiet_tunnel_ends_on_the_idle_timeout():
    a, b = socket.socketpair()                       # both ends stay open, nothing sent
    done = threading.Event()
    threading.Thread(target=lambda: (_splice(a, b, idle=0.2), done.set()),
                      daemon=True).start()
    t0 = time.monotonic()
    assert done.wait(3.0), "the splice of a half-open tunnel never ended"
    el = time.monotonic() - t0
    assert 0.15 <= el < 3.0, f"ended in {el:.2f} s, not ~idle"


def test_a_one_way_flow_keeps_the_tunnel_past_the_idle_clock():
    # a download: the client sends nothing at all while upstream trickles bytes. The
    # shared clock must keep the tunnel up; the per-direction clock used to cut this.
    cl_a, cl_b = socket.socketpair()                 # client side of the splice
    sv_a, sv_b = socket.socketpair()                 # upstream side of the splice
    done = threading.Event()

    def run():
        _splice(cl_a, sv_a, idle=0.4)
        done.set()
    threading.Thread(target=run, daemon=True).start()
    cl_b.settimeout(2.0)
    t0 = time.monotonic()
    for _ in range(4):                               # ~0.8 s of one-way flow, idle 0.4
        sv_b.sendall(b"x" * 64)
        assert cl_b.recv(64) == b"x" * 64            # reached the silent client
        time.sleep(0.2)
    assert time.monotonic() - t0 > 0.8, f"flow only lasted {time.monotonic() - t0:.2f} s"
    assert not done.is_set(), "the splice died inside a live one-way download"
    sv_b.close()                                     # the flow stops: idle now applies
    assert done.wait(2.5), "the splice did not end when both directions went quiet"


def test_each_direction_reaches_its_far_end():
    # four socket ends: client <-> a |splice| b <-> server, a|b spliced together
    cl_a, cl_b = socket.socketpair()                 # cl_b is the client's end
    sv_a, sv_b = socket.socketpair()                 # sv_b is the server's end
    threading.Thread(target=lambda: _splice(cl_a, sv_a, idle=5.0), daemon=True).start()
    cl_b.sendall(b"GET / HTTP/1.1\r\n")
    sv_b.settimeout(3)
    assert sv_b.recv(100) == b"GET / HTTP/1.1\r\n"   # client -> server
    sv_b.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
    cl_b.settimeout(3)
    assert cl_b.recv(100) == b"HTTP/1.1 200 OK\r\n\r\n"   # server -> client
    sv_b.close()                                     # the server ends: pumps end
    time.sleep(0.2)
    cl_a.close()
    cl_b.close()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n = ok = 0
    for name, fn in tests:
        n += 1
        try:
            fn()
            print(f"  {name:56s} ok")
            ok += 1
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {name:56s} FAIL {type(exc).__name__}: {exc}")
    print(f"{ok}/{n}")
    sys.exit(0 if ok == n else 1)
