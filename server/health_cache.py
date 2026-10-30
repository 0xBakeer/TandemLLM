"""SPD-249: `/health` at a price the round runner does not feel.

Every `/health` builds a ~21 kB body from a dozen reports (caches, memory, roles, the slot pool, the
round runner, the joint graphs) in the server's Python, and that Python holds the interpreter lock
the round runner's host work needs. On one DGX Spark (2026-10-30, 8 requests at once on
/v1/completions under a /health poll every 0.25 s, two servers each) the engine gave 125.2 and
125.8 tok/s with every call building the body, 131.5 and 131.3 with a TTL of 1 s (the profile
that sets the TTL names the A/B).

Two things here:
  parts   every build times its parts (`Parts`): `/health` `health_cost` says what the last full
          build cost by part, how many calls built it and how many reused it.
  reuse   with a TTL (`QSE_HEALTH_TTL_MS`, 0: off, every call builds as before) the full body is
          built at most once a TTL; a call inside it answers the last body with the fields a
          watchdog decides on built fresh (status, fault, uptime, in flight, engine busy). One
          builder at a time, and nobody waits for it: a call that finds another thread building
          answers the last body (the first build of the process is the only one waited for).
No CUDA call is made here; what the parts call is what `/health` called before.
"""

from __future__ import annotations

import json
import os
import threading
import time

#: SPD-249: `/health`'s full body is rebuilt at most once in this many ms (0: every call builds it)
TTL_MS = float(os.environ.get("QSE_HEALTH_TTL_MS", "0") or 0)


class Parts:
    """`part(name, fn, *args)`: `fn(*args)` with its wall ms added to `ms[name]`."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def __call__(self, name: str, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t0) * 1e3


def _inner(d: dict) -> str:
    """A dict's JSON without its braces ('' for an empty dict)."""
    return json.dumps(d, ensure_ascii=False)[1:-1]


class HealthCache:
    """The full `/health` body, built by `build(part)` at most once a TTL (see the module)."""

    def __init__(self, ttl_ms: float | None = None, clock=time.monotonic) -> None:
        self.ttl_ms = TTL_MS if ttl_ms is None else float(ttl_ms)
        self._clock = clock
        self._lock = threading.Lock()
        self._body: str | None = None          # the last full body's JSON without its braces
        self._t = 0.0
        self.builds = 0
        self.reused = 0
        self.last_ms: dict[str, float] = {}
        self.build_ms_sum = 0.0
        self.build_ms_max = 0.0

    def _build(self, build, fresh_keys) -> None:
        part = Parts()
        t0 = time.perf_counter()
        heavy = build(part)
        for k in fresh_keys:
            heavy.pop(k, None)
        body = part("encode", _inner, heavy)
        total = (time.perf_counter() - t0) * 1e3
        self._body, self._t = body, self._clock()
        self.builds += 1
        self.build_ms_sum += total
        self.build_ms_max = max(self.build_ms_max, total)
        self.last_ms = dict({k: round(v, 3) for k, v in part.ms.items()}, total=round(total, 3))

    def report(self) -> dict:
        n = max(1, self.builds)
        return {"ttl_ms": self.ttl_ms, "builds": self.builds, "reused": self.reused,
                "build_ms_mean": round(self.build_ms_sum / n, 3), "build_ms_max": round(self.build_ms_max, 3),
                "last_build_ms": self.last_ms}

    def answer(self, fresh: dict, build) -> bytes:
        """The JSON bytes of `fresh` (built by the caller on every call) and the full body
        (`build(part)` -> dict; its keys that `fresh` also has are dropped)."""
        if self.ttl_ms <= 0:
            self._build(build, fresh)
        else:
            stale = self._body is None or (self._clock() - self._t) * 1e3 >= self.ttl_ms
            if stale and self._lock.acquire(blocking=self._body is None):
                try:
                    if self._body is None or (self._clock() - self._t) * 1e3 >= self.ttl_ms:
                        self._build(build, fresh)
                    else:
                        self.reused += 1
                finally:
                    self._lock.release()
            else:
                self.reused += 1
        head = _inner(dict(fresh, health_cost=self.report()))
        body = self._body or ""
        return ("{" + head + (", " + body if body else "") + "}").encode()
