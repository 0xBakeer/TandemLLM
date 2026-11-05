"""SPD-249: the GPU's busy window, for host work that need not run while the GPU waits for the host.

A round runner thread issues a joint pass and then waits for it: tens of ms in which its thread holds no
interpreter lock and needs none. Host work of other threads that runs outside that wait (the stream writers, a
metrics sample, a status body) takes the lock from the runner while the GPU waits for the runner. A family calls
`busy()` between a pass's issue and its wait; the runner's `HOOK` (its thread only) then does its own deferred work
and opens a window (`opened()`), and closes it when the wait returns (`closed()`). `gate(timeout)` holds a thread's
non-critical work until the next window opens (or `timeout` s pass); while no runner holds windows (`hold(False)`)
it returns at once. HOOK None: `busy()` does nothing (the default, and every family that never calls it).

No CUDA call is made here, and nothing a pass computes depends on any of it."""

from __future__ import annotations

import threading

#: the round runner's work at a window's start (`Runner._busy`, its own thread), or None
HOOK = None

_cv = threading.Condition()
_seq = 0                 # windows opened so far
_open = False            # a window is open now
_held = False            # a runner holds windows (its joint rounds run): `gate` waits for the next one
_stats = {"windows": 0, "gated": 0, "gate_timeouts": 0, "gate_wait_ms": 0.0}


def busy() -> None:
    """A family's joint pass is queued on the GPU and its thread is about to wait for it."""
    h = HOOK
    if h is not None:
        h()


def hold(on: bool) -> None:
    """A runner starts (True) or stops (False) holding windows; stopping lets every waiting thread go."""
    global _held, _open
    with _cv:
        _held = bool(on)
        if not _held:
            _open = False
        _cv.notify_all()


def opened() -> None:
    global _seq, _open
    with _cv:
        _seq += 1
        _open = True
        _stats["windows"] += 1
        _cv.notify_all()


def closed() -> None:
    global _open
    with _cv:
        _open = False


def is_open() -> bool:
    return _open


def gate(timeout: float) -> bool:
    """Wait until the next window opens (True), or `timeout` s (False). At once (True) while no runner holds
    windows. A window already open does not count: work started late in one would run past its end."""
    import time
    with _cv:
        if not _held:
            return True
        n = _seq
        t0 = time.perf_counter()
        ok = _cv.wait_for(lambda: _seq != n or not _held, timeout)
        _stats["gated"] += 1
        _stats["gate_wait_ms"] += 1e3 * (time.perf_counter() - t0)
        if not ok:
            _stats["gate_timeouts"] += 1
        return ok


def report() -> dict:
    with _cv:
        return dict(_stats, held=_held, gate_wait_ms=round(_stats["gate_wait_ms"], 1))
