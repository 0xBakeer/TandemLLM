"""Shadow questions after the turn (VIS-119).

A decision type in shadow journals what it would do and changes nothing in the request. The
library's package question in shadow (R1, server/conductor/libsearch.py) still asked Qwen through
this server's own port before the request queued: on an idle box a Chat-tab message waited 1.41 s
in the queue, 1.32 s of it that question (2026-10-31), for an answer the request never reads. Two
questions asked at once do not hide each other's time either: the GPU serves their prefills one
after the other (two class questions at once 1.32-1.44 s against 0.67-0.69 s alone, 2026-11-01).

`QWEN38_SHADOW_ASK`:
  * `idle` (the default): the question is a job of the request's plan (`Plan.deferred`); when the
    request has ended (`Conductor.finish`) it goes to this module's thread, which asks it once the
    server has been idle for `QWEN38_SHADOW_ASK_IDLE_S` seconds (no request running, queued or in
    the router, the engine lock free: the router's `own_idle`). One question at a time. Its answer
    goes into the session's picks and the request's journal row (`extra.library`, with
    `deferred.waited_ms`), as an update of the row. A job that finds no idle moment within
    MAX_WAIT_S, or comes while QUEUE_MAX jobs wait, is dropped (`deferred.dropped`), as is one
    whose turn ended in tool calls (`tool_calls`: the client's next step follows at once) and
    one whose question failed (`failed`). In agent clients the turn that answers a new message
    mostly ends in tool calls, so their library questions are mostly dropped;
  * `inline`: asked before the queue, as before;
  * `off`: never asked (`how: shadow_off`).
A question a decider answers first (in process, tens of milliseconds), a plan decided under the
engine lock and the dashboard's test box keep asking in place. Nothing here reaches a request.
"""

from __future__ import annotations

import queue
import threading
import time

from server.knobs import SETTINGS as _S

#: `idle`, `inline` or `off` (the module's header)
SHADOW_ASK = (_S.get("SHADOW_ASK") or "idle").strip().lower()
#: the server's idle time a job waits for, s
SHADOW_ASK_IDLE_S = float(_S.get("SHADOW_ASK_IDLE_S") or 5.0)
#: the jobs waiting for an idle moment; a job beyond them is dropped (`full`)
QUEUE_MAX = 16
#: how long a job waits for an idle moment before it is dropped (`busy`), s
MAX_WAIT_S = 600.0
#: how often the thread looks at the server's idle time, s
POLL_S = 0.25


def mode() -> str:
    """`SHADOW_ASK` (anything but `inline` and `off` is `idle`)."""
    return SHADOW_ASK if SHADOW_ASK in ("idle", "inline", "off") else "idle"


class Deferred:
    """The thread that asks the shadow questions after their turns. `signal()` gives the server's
    idle callback (`Router.own_idle`: seconds idle, None while busy) or None when the server set
    none (a test, a tool), in which case a job waits its idle time from when it came."""

    def __init__(self, signal=lambda: None, *, settle_s=None, max_wait_s: float = MAX_WAIT_S,
                 poll_s: float = POLL_S, clock=time.monotonic, sleep=time.sleep):
        self.signal = signal
        self.settle_s = settle_s                   # None: SHADOW_ASK_IDLE_S
        self.max_wait_s, self.poll_s = float(max_wait_s), float(poll_s)
        self.clock, self.sleep = clock, sleep
        self._q: queue.Queue = queue.Queue(QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.stats = {"queued": 0, "asked": 0, "dropped_full": 0, "dropped_busy": 0,
                      "dropped_tool_calls": 0, "failed": 0}

    def submit(self, plan) -> int:
        """Queue the plan's jobs (`plan.deferred`); returns how many were queued. A job the queue
        has no room for is dropped at once."""
        jobs = list(getattr(plan, "deferred", None) or ())
        if not jobs:
            return 0
        plan.deferred = []
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="conductor-shadow-ask",
                                                daemon=True)
                self._thread.start()
        n = 0
        for job in jobs:
            job.setdefault("t", self.clock())
            try:
                self._q.put_nowait(job)
                self.stats["queued"] += 1
                n += 1
            except queue.Full:
                self.stats["dropped_full"] += 1
                self._drop(job, "full")
        return n

    def drop(self, plan, why: str) -> int:
        """Drop the plan's jobs unasked (`why` in their rows); returns how many."""
        jobs = list(getattr(plan, "deferred", None) or ())
        plan.deferred = []
        for job in jobs:
            self.stats[f"dropped_{why}"] = self.stats.get(f"dropped_{why}", 0) + 1
            self._drop(job, why)
        return len(jobs)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued job ran or was dropped (tests); True when none is left."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                waited = self.wait(job["t"])
                if waited is None:
                    self.stats["dropped_busy"] += 1
                    self._drop(job, "busy")
                else:
                    self.stats["asked"] += 1
                    job["run"](waited)
            except Exception as exc:                               # noqa: BLE001
                self.stats["failed"] += 1
                print(f"[conductor] a shadow question after its turn failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                self._drop(job, "failed")        # the row says so, not `deferred` for good
            finally:
                self._q.task_done()

    def wait(self, t0: float) -> float | None:
        """Block until the server has been idle for the settle time; the ms waited since the job
        came, or None when no idle moment came within `max_wait_s`."""
        settle = max(0.0, SHADOW_ASK_IDLE_S if self.settle_s is None else float(self.settle_s))
        while True:
            now = self.clock()
            fn = self.signal()
            if fn is None:
                ready = now - t0 >= settle
            else:
                try:
                    idle = fn()
                except Exception:                                  # noqa: BLE001
                    idle = None
                ready = idle is not None and idle >= settle
            if ready:
                return (now - t0) * 1e3
            if now - t0 >= self.max_wait_s:
                return None
            self.sleep(self.poll_s)

    @staticmethod
    def _drop(job, why: str) -> None:
        drop = job.get("drop")
        if drop is not None:
            try:
                drop(why)
            except Exception as exc:                               # noqa: BLE001
                print(f"[conductor] a dropped shadow question's row failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
