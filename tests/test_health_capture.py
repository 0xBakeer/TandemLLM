"""SPD-249: /health never waits on the capture lock.

An idle graph capture holds `engine.capture_lock.LOCK` for up to seconds; a /health that waited on it would miss the
watchdog's checks (two failed ones restart the engine). /health reads host-side values only; its one device look
(`server/gpu_health.probe`) skips while a capture holds the lock.

  * the fake engine with a capture held (QSE_FAKE_CAPTURE_HOLD_S): every /health answers in under 50 ms;
  * `gpu_health.check` with the lock held by another thread returns at once (no probe, no wait).
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_fake_engine import _port, _start  # noqa: E402


def test_health_answers_fast_while_a_capture_holds_the_lock():
    port = _port()
    d = tempfile.mkdtemp(prefix="qse-healthcap-")
    mark = os.path.join(d, "held")
    proc = _start(port, os.path.join(d, "ledger.sqlite3"),
                  env_extra={"QSE_FAKE_CAPTURE_HOLD_S": "3", "QSE_FAKE_CAPTURE_MARK": mark})
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:                                      # noqa: BLE001
                if proc.poll() is not None or time.time() - t0 > 60:
                    raise AssertionError("the fake server did not come up")
                time.sleep(0.2)
        while not os.path.exists(mark):
            assert time.time() - t0 < 60, "the capture hold never started"
            time.sleep(0.05)
        os.remove(mark)
        while not os.path.exists(mark):                    # a fresh hold: 3 s of it ahead
            assert time.time() - t0 < 60
            time.sleep(0.01)
        worst = 0.0
        for _ in range(20):
            s = time.perf_counter()
            body = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            worst = max(worst, time.perf_counter() - s)
            assert b'"status"' in body
            time.sleep(0.05)
        assert worst < 0.050, f"/health took {worst * 1e3:.1f} ms while a capture held the lock"
    finally:
        proc.kill()
        proc.wait()
    return f"20 /health while the lock was held: worst {worst * 1e3:.1f} ms"


def test_the_gpu_probe_skips_while_a_capture_holds_the_lock():
    from engine import capture_lock
    from server import gpu_health
    gpu_health.reset()
    real = gpu_health._on_cuda
    gpu_health._on_cuda = lambda device: True
    held = threading.Event()
    done = threading.Event()

    def capture():
        with capture_lock.LOCK:
            held.set()
            done.wait(5)

    t = threading.Thread(target=capture)
    t.start()
    held.wait()
    try:
        s = time.perf_counter()
        got = gpu_health.check("cuda", now=1e9)
        took = time.perf_counter() - s
    finally:
        done.set()
        t.join()
        gpu_health._on_cuda = real
        gpu_health.reset()
    assert got is None and took < 0.050, (got, took)
    return f"no probe and no wait: {took * 1e3:.2f} ms"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<60} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<60} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
