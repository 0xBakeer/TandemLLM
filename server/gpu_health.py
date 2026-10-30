"""SRV-671: a CUDA context the engine can no longer use, told apart from an error it survives.

After a device-side assert every CUDA call of the process fails with the same sticky error until
the process ends. Live 2026-10-21 about 15:00 (Ornith 1.5 122B, 0.13.0): an embedding gather out
of bounds (SRV-669) left the context so; `/health` said ok for 30 minutes, every chat request
answered 500 with the driver's text, and the watchdog restarted nothing.

  `note(exc, where)`  where a request or a background job failed: a CUDA error is checked with
                      `probe`, and a failing probe marks the engine failed. An error the context
                      survives (an out-of-memory, a launch with a bad grid) marks nothing.
  `check(device)`     `/health`'s look, a probe at most every PROBE_EVERY_S: a failed probe marks
                      the engine failed whoever's error it was (a background job's included).
  `probe(device)`     `torch.cuda.mem_get_info`: it raises the sticky error and does not wait for
                      the queued kernels (0.2 ms on the Spark with 400 GEMMs queued). It holds
                      `engine.capture_lock.LOCK` as every CUDA call off the engine's thread does
                      (a call during a graph capture in global mode would invalidate it), and a
                      capture that holds the lock skips the probe.

Once failed the process stays failed: only a new process has a new context. `/health` answers 500
with `status: failed` (the watchdog restarts on it, ops/watchdog.sh), a completion answers 503 with
the words below and a Retry-After, `qse_up` is 0.
"""

from __future__ import annotations

import threading
import time

#: the driver's words for an error that leaves the context unusable
STICKY = ("device-side assert", "illegal memory access", "unspecified launch failure",
          "misaligned address", "illegal instruction", "uncorrectable ECC", "launch timed out",
          "cudaErrorAssert", "cudaErrorIllegalAddress", "cudaErrorLaunchFailure")
#: how often `/health` asks the device itself (s)
PROBE_EVERY_S = 5.0
#: the Retry-After of a completion refused while the engine waits for its restart (s)
RETRY_AFTER_S = 60

FAULT: dict | None = None
_LOCK = threading.Lock()
_LAST = {"t": 0.0}


def first_line(text: str) -> str:
    """The error's first line, without the driver's pointer to its documentation."""
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def cuda_error(exc: BaseException) -> bool:
    """An error a CUDA call raised (the context may or may not have survived it)."""
    text = str(exc)
    return (type(exc).__name__ == "AcceleratorError" or "CUDA error" in text
            or any(m in text for m in STICKY))


def _on_cuda(device) -> bool:
    """A CUDA device whose context this process has opened (never opened here for a probe)."""
    if device is not None and "cuda" not in str(device):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:                                              # noqa: BLE001
        return False


def probe(device=None) -> str | None:
    """None when the context answers (or a capture holds the lock: no answer now), else the
    error's first line."""
    from engine import capture_lock
    if not capture_lock.LOCK.acquire(blocking=False):
        return None
    try:
        import torch
        torch.cuda.mem_get_info(device if device is not None and "cuda:" in str(device) else None)
    except Exception as exc:                                       # noqa: BLE001
        return first_line(str(exc)) or type(exc).__name__
    finally:
        capture_lock.LOCK.release()
    return None


def mark(error: str, where: str) -> dict:
    """The engine failed: the first error is kept and said once in the log."""
    global FAULT
    with _LOCK:
        if FAULT is None:
            FAULT = {"error": first_line(error) or "CUDA error", "where": where,
                     "since": round(time.time(), 3)}
            print(f"[gpu] the CUDA context failed ({FAULT['error']}, seen in {where}): the engine "
                  f"needs a restart; /health answers 500", flush=True)
        return FAULT


def note(exc: BaseException, where: str = "request", device=None) -> dict | None:
    """A failure seen by a request or a job: the engine is marked failed when it was a CUDA error
    and the context no longer answers."""
    if FAULT is not None:
        return FAULT
    if not cuda_error(exc) or not _on_cuda(device):
        return None
    got = probe(device)
    if got is None or not any(m in got for m in STICKY):
        return None
    return mark(first_line(str(exc)) or got, where)


def check(device=None, now: float | None = None) -> dict | None:
    """`/health`'s look: the fault, else a probe at most every PROBE_EVERY_S."""
    if FAULT is not None:
        return FAULT
    if not _on_cuda(device):
        return None
    now = time.monotonic() if now is None else now
    with _LOCK:
        if now - _LAST["t"] < PROBE_EVERY_S:
            return None
        _LAST["t"] = now
    got = probe(device)
    # only an error that leaves the context unusable: anything else this probe could meet (a
    # call refused during another thread's capture) is not a failure of the engine
    if got is None or not any(m in got for m in STICKY):
        return None
    return mark(got, "health probe")


def words(fault: dict | None = None) -> str:
    """What a client and the Chat tab say."""
    f = fault or FAULT or {}
    err = f.get("error") or "CUDA error"
    return (f"The engine needs a restart: its GPU stopped working ({err}). No request can be "
            f"served until it restarts.")


def reset() -> None:
    """For the tests: a fresh process's state."""
    global FAULT
    with _LOCK:
        FAULT = None
        _LAST["t"] = 0.0
