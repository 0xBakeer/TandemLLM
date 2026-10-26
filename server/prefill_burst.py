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

With `join` (QSE_PREFILL_BURST=joint) the members' prompts are not prefilled one after another
but in one forward: each member hands its prompt over at its prefill (`join(who, job)`) and gives the
turn back; the last one runs `run(jobs)` (engine/model.py `prefill_joint`: every row bit for bit
its own forward's, the weights read once a layer for all of them) and hands each member its result.
A member whose prompt the joint forward cannot take joins with no job and prefills alone.

Requests on their way to a slot (`received` at the handler's start, `settle` once bound or queued):
eight requests sent at once reach their slots one by one, and the first one bound on an idle engine
used to reach its prefill before the second was bound, find itself the burst's only member and prefill
alone (live 2026-10-26: one of the eight in both C8 rounds of a pair, its first token at 0.6 s, the
other seven at 4 s). A member that would run the joint forward alone while another request is still on
its way waits for it (the turn given back), up to `wait_s`; with none on the way it starts at once, so
a lone request never waits.

Plain Python and CPU-testable (tests/test_prefill_burst.py); server/app.py builds one when
`QSE_PREFILL_BURST` is wait or joint and the server runs parallel slots.
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
        self.jobs: dict = {}               # join: member -> its job, until the last one takes them
        self.results: dict = {}            # join: member -> (ok, result or exception)
        self.incoming = 0                  # requests received and not yet bound to a slot (or queued)
        self.stats = {"bursts": 0, "members": 0, "max_members": 0, "timeouts": 0, "wait_ms": 0.0,
                      "joint": 0, "joint_rows": 0, "held": 0}

    def received(self) -> list:
        """A request on its way to a slot; `settle` the token once it is bound (after `arrive`) or
        queued, and at its end whatever happened (idempotent)."""
        with self.cv:
            self.incoming += 1
            return [False]

    def settle(self, tok) -> None:
        if not tok:
            return
        with self.cv:
            if not tok[0]:
                tok[0] = True
                self.incoming -= 1
                self.cv.notify_all()

    def is_open(self) -> bool:
        return bool(self.pending or self.done or self.jobs)

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
            self.cv.notify_all()
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
                elif not self.pending and self.jobs:
                    # the joint forward's members wait for a leader that will not come: they go alone
                    for m in self.jobs:
                        self.results[m] = (False, None)
                    self.jobs = {}
                    self._close()

    def join(self, who, job, run, release=None, acquire=None, rows: int = 0):
        """Member `who` at its prefill, holding the GPU turn. `job`: what `run(jobs)` takes for it (a list
        of jobs in arrival order -> their results), or None (it prefills alone). The last member to come
        runs the jobs, holding the turn; the others give the turn back (`release`) and take it again
        (`acquire`) once their result is in. A member that would be the last while a request is still on
        its way to a slot (`received`) waits like the others, and runs the jobs itself if that request
        joins nothing (bound elsewhere, queued, gone). Returns (True, result) for a job the joint forward
        ran, (False, None) when the member prefills alone (no job, a burst of one, or `wait_s` passed
        before the last member came)."""
        with self.cv:
            if who not in self.pending:
                return False, None
            self.pending.discard(who)
            self.cv.notify_all()
            if job is not None:
                self.jobs[who] = job
                self.stats["joint_rows"] += int(rows)
            if self.pending or (self.incoming > 0 and (job is not None or self.jobs)):
                if job is None:
                    return False, None             # alone; the waiting members stay for the others
                lead = False
                if not self.pending:
                    self.stats["held"] += 1        # one more is on its way: wait for it
            else:
                lead, batch = True, self.jobs
                self.jobs = {}
        if lead:
            return self._lead(who, batch, run)
        if release is not None:
            release()
        t0 = self.clock()
        batch = None
        try:
            with self.cv:
                while who not in self.results:
                    if who in self.jobs and not self.pending and self.incoming == 0:
                        batch = self.jobs              # nobody else is coming: this member runs them
                        self.jobs = {}
                        break
                    if who in self.jobs and self.clock() - t0 >= self.wait_s:
                        del self.jobs[who]             # not taken yet: it prefills alone
                        self.stats["timeouts"] += 1
                        self.results[who] = (False, None)
                        break
                    self.cv.wait(0.05 if who in self.jobs else 0.5)
                if batch is None:
                    ok, r = self.results.pop(who)
                self.stats["wait_ms"] += (self.clock() - t0) * 1e3
        finally:
            if acquire is not None:
                acquire()
        if batch is not None:
            return self._lead(who, batch, run)
        if ok is None:
            raise r
        return (True, r) if ok else (False, None)

    def _lead(self, who, batch: dict, run):
        """The last member: the joint forward over `batch` (member -> job), holding the turn."""
        if len(batch) <= 1:
            with self.cv:
                for m in batch:
                    if m != who:
                        self.results[m] = (False, None)
                self._close()
            return False, None
        try:
            out = run(list(batch.values()))
        except BaseException as e:                 # every member hears of it, the leader raises
            with self.cv:
                for m in batch:
                    if m != who:
                        self.results[m] = (None, e)
                self._close()
            raise
        with self.cv:
            for m, r in zip(batch, out):
                if m != who:
                    self.results[m] = (True, r)
            self.stats["joint"] += 1
            self._close()
        return (True, out[list(batch).index(who)]) if who in batch else (False, None)

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
            return {"open": self.is_open(), "pending": sorted(map(str, self.pending)), "incoming": self.incoming,
                    "waiting": sorted(map(str, self.done)), "wait_s": self.wait_s,
                    **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()}}
