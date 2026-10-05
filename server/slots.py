"""Parallel requests, phase A: sequence slots, a fair GPU turn, lockstep verify rounds, and
admission by bytes.

`--parallel N` (`QSE_PARALLEL`, off by default) builds N - 1 side slots beside the engine. Off,
nothing here is built and the server is the single-sequence server it always was.

  * A `Slot` is one sequence's complete state: an engine (KV, recurrent state, verify graphs) and a
    drafter stack over the ONE copy of the weights every slot shares. Slot 0 is the server's own
    engine behind `LOCK`, with the resident prefix, the disk tier, the spill and the side lane; it
    reads and writes `STATE` as it always did. A side slot has a shorter window (`--side-max-len`),
    no resident prefix, and its own values of the per-request keys (`OVERLAY`): a request bound to
    it sees them through `STATE` (server/app.py `SlotState`), so the 6,000-line handler runs on it
    unchanged.
  * `Turn` decides which slot may run GPU work. A request holds it while it runs and offers it back
    between two blocks and between two prefill chunks; it hands it over only to a waiting slot that
    has used a quantum less GPU time than it has (virtual time, the idea of Linux's CFS). Alone, a
    request never hands over.
  * `Meeting` is the round's barrier: while two or more slots decode, each submits its tree block
    and waits, and one pass verifies them all (engine/model.py `forward_tree_multi`), the weights
    read once. A slot alone never submits and keeps its graphed `forward_tree`.
  * `Admission` counts bytes: before a request on a side slot prefills, its worst case (the
    prefill's workspace plus one state-store entry) is added to what the requests in flight have
    reserved, and the request waits while MemAvailable minus all of that would fall under the
    floor. An out-of-memory on this board freezes the box, so it is never caught, only avoided.

Everything here is plain Python and CPU-testable (tests/test_slots.py).
"""

from __future__ import annotations

import threading
import time

GiB = 1 << 30
MiB = 1 << 20

#: the per-request keys a side slot keeps for itself (server/app.py `SlotState`): what the
#: generation publishes for the handler (`last_ctx`, `blocks`, ...) and the slot-0-only machinery,
#: which a side slot does not have (None)
OVERLAY = ("engine", "drafter", "max_len", "resident", "disk_cache", "spill", "lane",
           "end_align", "mem", "probe", "last_ctx", "last_prompt_n", "hidden_spans", "blocks",
           "last_prefill", "spill_restored")


# ------------------------------------------------------------------ memory
def read_meminfo(path: str = "/proc/meminfo") -> dict | None:
    """`/proc/meminfo` in BYTES, or None where there is none (a Mac, a test)."""
    try:
        out = {}
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts:
                    out[k] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
        return out
    except OSError:
        return None


def slot_bytes(max_len: int, *, per_token: int = 106_752, constant: int = 7 * GiB + GiB // 2) -> int:
    """What one side slot costs before it serves anything: 64 KiB of engine KV, 2 x 20 KiB of
    draft KV (two drafter arms) and 256 B of rotary table a token of window, plus a constant for
    the recurrent state, the verify scratch and fold buffers, the graph pools and the draft
    buffers: 6.9 to 7.5 GiB allocated measured on 2026-10-05 for an 8,192-token slot (MemAvailable
    fell 5.9 to 9.4 GiB a slot), most of it the pools of the 121 verify graphs a slot captures."""
    return int(max_len) * int(per_token) + int(constant)


class Admission:
    """Byte-counting admission for parallel requests.

    `reserve(key, need)` admits a request whose worst case is `need` bytes when

        MemAvailable - cache growth left - what the requests in flight reserved - need >= floor

    and records the reservation until `release(key)`. A reservation counts in full until it is
    released, also the part that has already been allocated (and so already left MemAvailable):
    the check is conservative by up to one request's worst case, never optimistic. `growth()` is
    what the long-lived caches may still take (their budgets minus what they hold). Without
    /proc/meminfo (a Mac, a test without `meminfo`) every request is admitted and says so."""

    def __init__(self, floor_bytes: int, *, meminfo=read_meminfo, growth=None):
        self.floor = int(floor_bytes)
        self.meminfo = meminfo
        self.growth = growth or (lambda: 0)
        self.lock = threading.Lock()
        self._admit = threading.Lock()
        self.held: dict = {}
        self.stats = {"checks": 0, "admitted": 0, "waits": 0, "refused": 0, "min_after": None}

    def reserved(self) -> int:
        with self.lock:
            return sum(self.held.values())

    def check(self, need: int, count: bool = True) -> tuple[bool, dict]:
        mi = self.meminfo() if self.meminfo is not None else None
        grow = int(self.growth())
        with self.lock:
            held = sum(self.held.values())
        d = {"floor": self.floor, "grow_left": grow, "reserved": held, "need": int(need)}
        if not mi or "MemAvailable" not in mi or self.floor <= 0:
            d.update(avail=None, after=None, ok=True)
            return True, d
        avail = int(mi["MemAvailable"])
        after = avail - grow - held - int(need)
        d.update(avail=avail, after=after, ok=after >= self.floor)
        if count:
            self.stats["checks"] += 1
            m = self.stats["min_after"]
            self.stats["min_after"] = after if m is None else min(m, after)
        return d["ok"], d

    def reserve(self, key, need: int) -> tuple[bool, dict]:
        """Check and record in one step, so two requests cannot both take the last bytes. A
        request with nothing else in flight is admitted without the check: alone, it is the
        single-sequence server's request, which the engine's own memory policy (engine/cache.py
        `Budget`) admits as it always did."""
        with self._admit:
            with self.lock:
                alone = not self.held
            if alone:
                with self.lock:
                    self.held[key] = int(need)
                self.stats["admitted"] += 1
                self.stats["alone"] = self.stats.get("alone", 0) + 1
                return True, {"alone": True, "need": int(need), "ok": True, "avail": None}
            ok, d = self.check(need)
            if ok:
                with self.lock:
                    self.held[key] = int(need)
                self.stats["admitted"] += 1
        return ok, d

    def release(self, key) -> None:
        with self.lock:
            self.held.pop(key, None)

    @staticmethod
    def describe(d: dict) -> str:
        if d.get("avail") is None:
            return "no /proc/meminfo: not checked"
        g = lambda b: f"{b / GiB:.1f}"                                    # noqa: E731
        return (f"MemAvailable {g(d['avail'])} GiB - cache growth {g(d['grow_left'])} - "
                f"in flight {g(d['reserved'])} - this request {g(d['need'])} = "
                f"{g(d['after'])} GiB ({'>=' if d['ok'] else '<'} floor {g(d['floor'])})")

    def report(self) -> dict:
        with self.lock:
            held = dict(self.held)
        return {"floor_gib": round(self.floor / GiB, 2), "in_flight": len(held),
                "reserved_gib": round(sum(held.values()) / GiB, 3), **self.stats}


# ------------------------------------------------------------------ the turn
class Turn:
    """A fair hand-over of the GPU between slots, by virtual time.

    Every slot has a virtual time: the GPU time it has used. `pause(who)` is called by the holder
    before each unit of work (a decode block, a prefill chunk); it charges the time since the last
    pause, and when a waiting slot is behind the holder by more than one quantum, it hands the turn
    to the waiter furthest behind and queues. A 1.5 s prefill chunk puts its slot 1.5 s ahead, and a
    decoding slot runs about 1.5 s of blocks before the prefill runs again. A slot that starts a
    request (`reset`) starts at the smallest virtual time of the slots running, so a slot that was
    idle has no credit to spend. Alone, a holder never hands over.

    `settle()` runs in the holder's thread before it hands the turn to another slot (the server
    passes `torch.cuda.synchronize`): the next slot starts on an idle device, so its own timings --
    the length router prices widths from measured draft and verify times -- do not include the last
    slot's queued work.
    """

    def __init__(self, quantum_s: float = 0.05, clock=time.monotonic, settle=None):
        self.quantum = float(quantum_s)
        self.clock = clock
        self.settle = settle
        self._cv = threading.Condition()
        self._owner = None
        self._grant = None                 # the waiter the turn was handed to
        self._waiting: list = []           # arrival order
        self._v: dict = {}                 # virtual time, seconds of GPU used
        self._mark = 0.0
        self.stats = {"acquires": 0, "handoffs": 0, "waits": 0}

    @property
    def owner(self):
        return self._owner

    def waiting(self) -> int:
        return len(self._waiting)

    def held_by(self, who) -> bool:
        return self._owner == who

    def _floor(self, but=None) -> float:
        active = [self._v[w] for w in [self._owner, *self._waiting]
                  if w is not None and w != but and w in self._v]
        return min(active) if active else max(self._v.values(), default=0.0)

    def reset(self, who) -> None:
        with self._cv:
            self._v[who] = self._floor(but=who)

    def _charge(self, now: float) -> None:
        if self._owner is not None:
            self._v[self._owner] = self._v.get(self._owner, 0.0) + max(0.0, now - self._mark)
        self._mark = now

    def _behind(self):
        if not self._waiting:
            return None
        return min(self._waiting, key=lambda w: (self._v.get(w, 0.0), self._waiting.index(w)))

    def _wait_for_turn(self, who) -> None:
        self._waiting.append(who)
        try:
            while not (self._owner is None and (self._grant == who or (
                    self._grant is None and self._behind() == who))):
                self._cv.wait()
        finally:
            self._waiting.remove(who)
        self._grant = None
        self._owner = who
        self._mark = self.clock()
        self.stats["acquires"] += 1

    def acquire(self, who) -> None:
        with self._cv:
            if who not in self._v:
                self._v[who] = self._floor(but=who)
            if self._owner is None and not self._waiting and self._grant is None:
                self._owner = who
                self._mark = self.clock()
                self.stats["acquires"] += 1
                return
            self.stats["waits"] += 1
            self._wait_for_turn(who)

    def release(self, who) -> None:
        if self.settle is not None and self._waiting:
            self.settle()
        with self._cv:
            if self._owner != who:
                raise RuntimeError(f"turn released by {who!r}, held by {self._owner!r}")
            self._charge(self.clock())
            self._owner = None
            self._grant = self._behind()
            self._cv.notify_all()

    def pause(self, who) -> bool:
        """Offer the turn back before a unit of work; True if it was handed over (and is back)."""
        with self._cv:
            if self._owner != who:
                return False
            self._charge(self.clock())
            nxt = self._behind()
            if nxt is None or self._v.get(nxt, 0.0) + self.quantum >= self._v[who]:
                return False
        if self.settle is not None:
            self.settle()
        with self._cv:
            self._charge(self.clock())
            nxt = self._behind()
            if nxt is None:
                return False
            self.stats["handoffs"] += 1
            self._owner = None
            self._grant = nxt
            self._cv.notify_all()
            self._wait_for_turn(who)
            return True

    def report(self) -> dict:
        with self._cv:
            return {"quantum_ms": self.quantum * 1e3, "owner": self._owner,
                    "waiting": list(self._waiting),
                    "vtime_s": {str(k): round(v, 3) for k, v in self._v.items()}, **self.stats}


# ------------------------------------------------------------------ slots
class Slot:
    """One sequence. `values` holds a side slot's own values of the `OVERLAY` keys; slot 0 has
    none (it is `STATE`). `phase` is "decode" while its request is in the block loop (the barrier
    waits only for those slots), else ""."""

    def __init__(self, index: int, max_len: int, values: dict | None = None, lock=None):
        self.index = int(index)
        self.max_len = int(max_len)
        self.values = values
        self.lock = lock if lock is not None else threading.Lock()
        self.owner = None
        self.served = 0
        self.phase = ""
        self.t_start = None

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def report(self) -> dict:
        eng = (self.values or {}).get("engine")
        return {"index": self.index, "max_len": self.max_len, "busy": self.busy,
                "owner": self.owner, "served": self.served, "phase": self.phase,
                "kv_length": getattr(getattr(eng, "kv", None), "length", None)}


class VerifyReq:
    """One slot's tree block, submitted to the round. The slot that runs the round fills `out`
    and `ms`; the submitter reads them when it holds the turn again."""

    __slots__ = ("slot", "eng", "tokens", "parents", "start", "rows", "batchable", "out", "ms",
                 "error", "joint")

    def __init__(self, slot: int, eng, tokens, parents, start: int, batchable: bool = True):
        self.slot, self.eng, self.tokens, self.parents, self.start = slot, eng, tokens, parents, start
        self.rows = len(parents)
        self.batchable = bool(batchable)
        self.out = None
        self.ms = None
        self.error: BaseException | None = None
        self.joint = 1                     # how many blocks shared the pass this one ran in


#: Tree nodes a slot may verify in a joint round, by how many slots decode: the joint pass stays at
#: or under the 32 rows where every projection kernel is row-invariant (docs/kernels.md). A slot
#: alone keeps the router's own budget.
SHARES = {2: 16, 3: 10, 4: 8}


def share(n_decoding: int, max_rows: int = 32) -> int | None:
    if n_decoding <= 1:
        return None
    s = SHARES.get(n_decoding, max(2, max_rows // n_decoding))
    return max(2, min(s, max_rows // n_decoding))


class Meeting:
    """The round's barrier. A decoding slot submits its verify block and waits; the round runs when
    every decoding slot has submitted -- run by the last one, which holds the GPU turn -- or, by a
    waiter, when the turn has been idle for `timeout_s` with blocks pending (a slot that stopped
    decoding for a moment: a forced close, a single-token step). After the first joint round the
    slots run in lockstep, because they all get their logits from the same pass. `cv` guards
    `pending` and the results."""

    def __init__(self, max_rows: int = 32, timeout_s: float = 0.002):
        self.max_rows = int(max_rows)
        self.timeout = float(timeout_s)
        self.cv = threading.Condition()
        self.pending: dict[int, VerifyReq] = {}
        # slots whose round ran and which have not taken the turn back yet: on their way (a woken
        # thread waits for the GIL up to its switch interval), not idle
        self.transit: set = set()
        self.stats = {"rounds": 0, "joint_rounds": 0, "joint_blocks": 0, "submitted": 0,
                      "timeouts": 0, "single_rounds": 0, "alone": 0, "trimmed": 0,
                      "rows_hist": {}, "joint_hist": {},
                      # a slot's round, submit to submit, by how many slots decoded: {n: [ms, rounds]}
                      "round_ms": {},
                      # with `timing` on: a joint pass's GPU time (synchronized), {blocks: [ms, passes]}
                      "pass_ms": {}}
        self.timing = False
        self._last_submit: dict = {}

    def batches(self) -> list:
        """The pending blocks cut into passes: the batchable ones together while they fit in
        `max_rows` rows, every other block in a pass of its own, in arrival order."""
        out, cur, rows = [], [], 0
        for r in self.pending.values():
            if not r.batchable:
                out.append([r])
                continue
            if cur and rows + r.rows > self.max_rows:
                out.append(cur)
                cur, rows = [], 0
            cur.append(r)
            rows += r.rows
        if cur:
            out.append(cur)
        return out

    def report(self) -> dict:
        with self.cv:
            return {"max_rows": self.max_rows, "timeout_ms": self.timeout * 1e3,
                    "pending": sorted(self.pending), **self.stats}


class Pool:
    """The slots: slot 0 (the server's own engine, behind `LOCK`) and the side slots. Requests
    queue for slot 0 the way they always did (server/app.py `_take`); a request that may use a side
    slot also tries `take_side` while it waits."""

    def __init__(self, slots: list[Slot], turn: Turn | None = None,
                 meeting: Meeting | None = None, admission: Admission | None = None):
        if not slots:
            raise ValueError("a pool needs at least slot 0")
        self.slots = list(slots)
        self.turn = turn
        self.meeting = meeting
        self.admission = admission
        self._lk = threading.Lock()
        self.stats = {"side_grants": 0, "slot0_grants": 0, "max_running": 0}

    @property
    def n(self) -> int:
        return len(self.slots)

    def side(self) -> list[Slot]:
        return self.slots[1:]

    def take_side(self, need: int, owner=None) -> Slot | None:
        """A free side slot whose window holds `need` rows, the smallest that does; None."""
        for s in sorted(self.side(), key=lambda s: s.max_len):
            if need <= s.max_len and not s.busy and s.lock.acquire(blocking=False):
                s.owner = owner
                s.t_start = time.monotonic()
                with self._lk:
                    self.stats["side_grants"] += 1
                    self.stats["max_running"] = max(self.stats["max_running"], self.running())
                return s
        return None

    def release_side(self, slot: Slot) -> None:
        slot.owner = None
        slot.served += 1
        slot.phase = ""
        slot.t_start = None
        slot.lock.release()

    def running(self) -> int:
        return sum(1 for s in self.slots if s.busy)

    def decoding(self) -> list[int]:
        return [s.index for s in self.slots if s.phase == "decode"]

    def report(self) -> dict:
        return {"n": self.n, "running": self.running(), "decoding": self.decoding(),
                "slots": [s.report() for s in self.slots], **self.stats,
                "turn": self.turn.report() if self.turn is not None else None,
                "meeting": self.meeting.report() if self.meeting is not None else None,
                "admission": self.admission.report() if self.admission is not None else None}


# ------------------------------------------------------------------ the barrier
def barrier(meet: Meeting, pool: Pool, turn: Turn, slot_index: int, req: VerifyReq, run) -> None:
    """Submit `req` to `meet` and return once its round has run.

    The slot whose submission completes the set of decoding slots runs the round (`run(meet)`,
    holding the GPU turn); the others release the turn and wait. A waiter runs the round itself
    when the set becomes complete because a slot left decode, or when the turn has been idle for
    the barrier's timeout with blocks pending and no missing slot on its way back to the turn."""
    with meet.cv:
        meet.pending[slot_index] = req
        meet.stats["submitted"] += 1
        dec = pool.decoding()
        now = time.monotonic()
        last = meet._last_submit.get(slot_index)
        meet._last_submit[slot_index] = now
        if last is not None and now - last < 5.0:
            r = meet.stats["round_ms"].setdefault(str(len(dec)), [0.0, 0])
            r[0] += (now - last) * 1e3
            r[1] += 1
        complete = all(i in meet.pending for i in dec)
    if not complete:
        turn.release(slot_index)
        idle_since = None
        try:
            with meet.cv:
                while req.out is None and req.error is None:
                    meet.cv.wait(meet.timeout / 2)
                    if req.out is not None or req.error is not None:
                        break
                    if all(i in meet.pending for i in pool.decoding()):
                        break                           # a slot left decode: the set is complete
                    now = time.monotonic()
                    missing = set(pool.decoding()) - set(meet.pending)
                    if turn.owner is None and not turn.waiting() and not (missing & meet.transit):
                        idle_since = idle_since or now
                        if now - idle_since >= meet.timeout:
                            meet.stats["timeouts"] += 1
                            break
                    else:
                        idle_since = None
        finally:
            turn.acquire(slot_index)
    if req.out is None and req.error is None:
        run(meet)
    with meet.cv:
        meet.transit.discard(slot_index)
    if req.error is not None:
        raise req.error


# ------------------------------------------------------------------ the request's slot
_CUR = threading.local()


def bind(slot: Slot | None) -> None:
    """The slot the current thread's request runs on (None: slot 0 / outside a request)."""
    _CUR.slot = slot


def current() -> Slot | None:
    return getattr(_CUR, "slot", None)


def values() -> dict | None:
    """The current thread's side-slot values, None on slot 0 (the fast path of `SlotState`)."""
    s = getattr(_CUR, "slot", None)
    return s.values if s is not None else None
