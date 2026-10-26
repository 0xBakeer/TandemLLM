"""SPD-245: the prefills of requests that arrive together run before any of them decodes.

Without it, eight requests sent at once prefill one after another, and the lanes that have their
first token decode beside the prefills still waiting: those rounds run at one to seven lanes, each
pass of the weights shared by fewer tokens, and every later prefill waits behind them for its turn.
A burst opens when a request is bound to a slot while no slot decodes (`arrive(idle=True)`); every
request bound while it is open joins it. A member that has prefilled gives the GPU turn back and
waits until every member has prefilled (`prefilled`), then all of them decode from the first round
on, in joint rounds of all the lanes.

Only the order of the work moves: each prefill is the forward its request runs alone, so the logits,
the states and every token are the ones without the burst. A member waits at most `wait_s` (then it
goes on and the wait counts as a timeout); a member that ends before its prefill leaves (`leave`).

Plain Python and CPU-testable (tests/test_prefill_burst.py); server/app.py builds one when
`QSE_PREFILL_BURST` is on and the server runs parallel slots.
"""

from __future__ import annotations

import threading
import time


class Burst:
    def __init__(self, wait_s: float = 10.0, clock=time.monotonic):
        self.wait_s = float(wait_s)
        self.clock = clock
        self.cv = threading.Condition()
        self.pending: set = set()          # members that have not prefilled yet
        self.done: set = set()             # members that prefilled and wait for the others
        self.gen = 0                       # bumped when a burst closes
        self.size = 0                      # members of the open burst
        self.stats = {"bursts": 0, "members": 0, "max_members": 0, "timeouts": 0, "wait_ms": 0.0}

    def is_open(self) -> bool:
        return bool(self.pending or self.done)

    def arrive(self, who, idle: bool) -> bool:
        """Request on slot `who`, before its prefill: a member of the open burst, or of a new one when
        `idle` (no slot decodes). False: not a member (it runs as before)."""
        with self.cv:
            if not (self.is_open() or idle) or who in self.pending or who in self.done:
                return False
            if not self.is_open():
                self.size = 0
            self.pending.add(who)
            self.size += 1
            return True

    def member(self, who) -> bool:
        with self.cv:
            return who in self.pending

    def _close(self) -> None:
        self.stats["bursts"] += 1
        self.stats["members"] += self.size
        self.stats["max_members"] = max(self.stats["max_members"], self.size)
        self.done.clear()
        self.gen += 1
        self.cv.notify_all()

    def leave(self, who) -> None:
        """A member that ends without `prefilled` (an error, a refusal): the others stop waiting for it."""
        with self.cv:
            if who in self.pending:
                self.pending.discard(who)
                if not self.pending and self.done:
                    self._close()

    def prefilled(self, who, release=None, acquire=None) -> bool:
        """Member `who` has prefilled. The last one closes the burst and goes on; the others call
        `release()` (the GPU turn back), wait for the close or `wait_s`, then `acquire()`. True when it
        waited for the burst."""
        with self.cv:
            if who not in self.pending:
                return False
            self.pending.discard(who)
            if not self.pending:
                if self.done:
                    self._close()
                else:
                    self.size = 0          # a burst of one: nothing to wait for
                return False
            self.done.add(who)
            gen = self.gen
        if release is not None:
            release()
        t0 = self.clock()
        try:
            with self.cv:
                end = t0 + self.wait_s
                while self.gen == gen:
                    left = end - self.clock()
                    if left <= 0:
                        self.done.discard(who)
                        self.stats["timeouts"] += 1
                        if not self.pending and self.done:
                            self._close()
                        elif not self.pending and not self.done:
                            self.size = 0
                        break
                    self.cv.wait(min(left, 0.5))
                self.stats["wait_ms"] += (self.clock() - t0) * 1e3
        finally:
            if acquire is not None:
                acquire()
        return True

    def report(self) -> dict:
        with self.cv:
            return {"open": self.is_open(), "pending": sorted(map(str, self.pending)),
                    "waiting": sorted(map(str, self.done)), "wait_s": self.wait_s,
                    **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()}}
