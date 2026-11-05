"""demux's first frame is bounded (chatd.sandbox, the SRV-359 stdio-MCP stall): a hijacked
exec stream that answers its 101 and then never writes returns in about first_timeout
instead of holding the thread + socket + caller's slot forever; and once the first frame
has arrived, a long quiet on the stream is NOT cut. socketpair only, CPU, tiny timeouts.

Run: python tests/test_demux_first_timeout.py
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.sandbox import demux


def frame(stream: int, payload: bytes) -> bytes:
    return bytes([stream]) + b"\0\0\0" + struct.pack(">I", len(payload)) + payload


def test_silence_after_the_hijack_ends_the_wait():
    a, b = socket.socketpair()
    done = threading.Event()

    def worker():
        demux(a, b"", lambda s, d: None, first_timeout=0.2)
        done.set()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t0 = time.monotonic()
    assert done.wait(2.0), "demux did not return on a silent hijacked socket"
    el = time.monotonic() - t0
    assert 0.15 <= el < 2.0, f"returned in {el:.2f} s, not ~first_timeout"
    a.close()
    b.close()


def test_first_frame_in_hand_is_not_cut_by_the_timeout():
    got = []
    a, b = socket.socketpair()
    b.sendall(frame(1, b"hello"))
    b.close()                                    # EOF after the frame: a clean end
    demux(a, b"", lambda s, d: got.append((s, d)), first_timeout=30.0)
    assert got == [(1, b"hello")], got
    a.close()


def test_quiet_after_the_first_frame_survives():
    a, b = socket.socketpair()
    got = []
    done = threading.Event()

    def worker():
        demux(a, b"", lambda s, d: got.append((s, d)), first_timeout=0.3)
        done.set()
    threading.Thread(target=worker, daemon=True).start()
    b.sendall(frame(1, b"one"))
    time.sleep(0.8)                              # longer than the first-frame window
    assert not done.is_set(), "the timeout cut a stream that had already talked"
    b.sendall(frame(1, b"two"))
    b.close()
    assert done.wait(2.0)
    assert got == [(1, b"one"), (1, b"two")], got
    a.close()


def test_partial_first_frame_still_dies_at_the_deadline():
    a, b = socket.socketpair()
    b.sendall(b"\x01\0")                          # a header that never completes
    done = threading.Event()

    def worker():
        demux(a, b"", lambda s, d: None, first_timeout=0.2)
        done.set()
    threading.Thread(target=worker, daemon=True).start()
    assert done.wait(2.0), "a never-completing first frame did not end"
    a.close()
    b.close()


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
