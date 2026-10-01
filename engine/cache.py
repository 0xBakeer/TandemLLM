"""Serving-time caches. Four of them, and not one may change a token.

This engine holds ONE sequence, and every request so far has paid for its whole prompt from
scratch: `eng.reset()`, then a forward over every prompt token. On a multi-turn chat that is the
same two thousand tokens re-read on every turn, and on a fleet of requests sharing a system prompt
it is the same system prompt re-read per request. The board has ~90 GB of unified memory doing
nothing. These four caches spend it.

  1. `StateStore` -- the engine's whole state at a token boundary, kept in RAM and keyed by the
     token prefix that produced it. It serves BOTH the session cache (a conversation's state after
     its last turn) and the prefix cache (checkpoints every `chunk` tokens of a prefill), because
     those two are the same object with two different sets of boundaries.
  2. `ResponseCache` -- the tokens a (prompt, params) request produced. Greedy is deterministic, so
     the second identical request is a dictionary lookup.
  3. `PersistentSuffixStore` -- an append-only token log of what this engine has read and written,
     on the box, with a suffix array over it, handed to the lookup drafter as a corpus.
  4. `SuffixStoreSet` -- several such stores behind one `lookup`, so the private corpus dir and the
     engine's own history are both askable in the same 0.2 ms.
  5. `ResidentPrefix` -- the KV the last prefill left in the engine's own
     buffer, reused in place, with the recurrent state cloned at chunk boundaries. It is what
     serves a growing agent conversation past the store's per-entry cap.

`Budget` (OPS-47) holds them all to one memory policy: a ceiling for the engine between requests,
a declared peak during a prefill, a cap on each cache, and an order in which they are given back.

WHAT MAKES A RESTORE EXACT, which is the whole design and the only interesting part.

A restored state is not *nearly* the state a cold prefill would have left, it is the same bytes: the
KV rows, the 48 recurrent states, the 48 convolution states and the drafter's own KV are cloned out
and copied back. So a request that resumes at token L and forwards tokens L.. computes exactly what
it would have computed had it forwarded 0..L first -- PROVIDED the cold run would have forwarded
0..L as its own step. That proviso is real and it is why `prefill` chunks:

  * `forward(tokens[0:n])` in one call and `forward(tokens[0:L])` then `forward(tokens[L:n])` are
    not the same arithmetic. The attention of the second form reads the first form's rows out of
    the cache rather than recomputing them, SDPA takes a different backend when it is handed a mask
    instead of `is_causal`, and the chunked delta rule blocks the sequence differently.
  * So the cold path and the warm path use the SAME chunking. With the prefix cache on, every
    prefill is a sequence of `chunk`-token forwards, warm or cold. A resume that lands ON that
    chunk grid -- which every PREFIX-cache checkpoint does, because that is where they are taken
    -- then splits the remaining tokens exactly as a cold prefill would, and the two runs are
    bit-identical by construction rather than by luck. `tests/test_cache.py` asserts that.
  * A SESSION boundary is different and the difference is not a detail. It falls wherever the
    previous turn stopped, off the grid, so the tail splits differently; and the state a turn ends
    in was written by speculative verify blocks and a rank-k rollback, not by prefill chunks. It is
    the state that really produced the previous turn, which is arguably the more faithful thing to
    continue from, but it is NOT the state a cold re-read of the conversation would compute. So the
    session cache is held to the gate the whole engine is held to -- the same argmax -- and not to
    bitwise equality. `tools/cache_gate.py` measures both.
  * Against the ONE-SHOT prefill this engine did before, chunking is a different arithmetic order,
    in the same sense that every fused kernel in `engine/model.py` is. Same gate: identical argmax.

A hash is a hint here, never an answer: every hit re-checks the stored token prefix element for
element before the state is restored. A 64-bit collision would otherwise answer a request with
another request's state, which is the one bug in this file that no test downstream would catch.

PRIVACY. Everything here lives in RAM on the box, except `PersistentSuffixStore`, which writes
token ids -- never text -- under a directory outside this repository. Nothing in this file writes
to the repository, and nothing it holds reaches a log line.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections import OrderedDict, deque

import torch

# 64-bit polynomial rolling hash. The multiplier is the FNV-1a prime; what matters is only that
# `h(tokens[:i])` is available for every i after one pass, so probing a candidate boundary costs
# nothing and probing thirty costs nothing thirty times.
_P = 0x100000001B3
_MASK = (1 << 64) - 1


def prefix_hashes(tokens) -> list[int]:
    """`out[i]` is the hash of `tokens[:i]`, so `out` is one longer than `tokens`."""
    h = 0xCBF29CE484222325
    out = [h]
    for t in tokens:
        h = ((h ^ (int(t) + 1)) * _P) & _MASK
        out.append(h)
    return out


def extend_hashes(hashes: list[int], tokens) -> list[int]:
    """Continue a hash list with more tokens, in place. Returns it for chaining."""
    h = hashes[-1]
    for t in tokens:
        h = ((h ^ (int(t) + 1)) * _P) & _MASK
        hashes.append(h)
    return hashes


class _PrefixHashMemo:
    """`prefix_hashes` for the prompt of each prefill, reusing the previous prompt's list over the
    head the two share: a growing conversation hashes only its new tokens (a full pass is ~20 ms
    at 190k tokens). The list returned is the one a full pass computes."""

    def __init__(self):
        self.keys: list = []
        self.hashes: list[int] = prefix_hashes([])

    def __call__(self, keys) -> list[int]:
        keys = list(keys)
        c = _common_prefix(keys, self.keys)
        h = self.hashes[:c + 1]
        extend_hashes(h, keys[c:])
        self.keys, self.hashes = keys, h
        return h


_hash_memo = _PrefixHashMemo()


# --------------------------------------------------------------------------- the state snapshot
class StateSnapshot:
    """Everything the engine and its drafter need to be back at token `length`.

    The KV is sliced to `length` before it is cloned, because the buffer is `max_len` long and
    cloning all of it would cost the same for a 100-token prefix as for an 8,000-token one. The
    recurrent state is not sliceable -- it is a fixed [48, 1, H, Dk, Dv] fp32 tensor and it is the
    bulk of a snapshot on any prompt shorter than a few thousand tokens.
    """

    __slots__ = ("length", "k", "v", "S", "conv", "drafter", "ks", "vs", "_bytes")

    def __init__(self, length: int, k, v, S, conv, drafter=None, ks=None, vs=None):
        self.length = int(length)
        self.k, self.v, self.S, self.conv, self.drafter = k, v, S, conv, drafter
        # the e4m3 cache's scales (QWEN38_KV_FP8); None for a bf16 cache
        self.ks, self.vs = ks, vs
        self._bytes = (_nbytes(k) + _nbytes(v) + _nbytes(S) + _nbytes(conv)
                       + _nbytes(ks) + _nbytes(vs) + _tree_bytes(drafter))

    @property
    def nbytes(self) -> int:
        return self._bytes

    def parts(self) -> dict:
        return {"kv": _nbytes(self.k) + _nbytes(self.v), "recurrent": _nbytes(self.S),
                "conv": _nbytes(self.conv), "drafter": _tree_bytes(self.drafter)}


def _nbytes(x) -> int:
    return 0 if x is None else x.numel() * x.element_size()


def _tree_bytes(obj) -> int:
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return _nbytes(obj)
    if isinstance(obj, (tuple, list)):
        return sum(_tree_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tree_bytes(x) for x in obj.values())
    return 0


#: One DFlash2 arm's draft KV, measured on the board (engine/drafters/dflash2.py `state_snapshot`).
#: The fallback for a snapshottable drafter that cannot say its own figure.
ARM_BYTES_PER_TOKEN = 20_000


def _drafter_bytes_per_token(drafter) -> int:
    """What the drafter's `state_snapshot` clones per committed position.

    Asked of the drafter rather than assumed: the served length router snapshots BOTH arms until
    the latch releases one, so a constant for one arm undercounted the served stack by
    a whole arm's 20 kB a token.
    """
    fn = getattr(drafter, "snapshot_bytes_per_token", None)
    return int(fn()) if fn is not None else ARM_BYTES_PER_TOKEN


def _snapshot_estimate(eng, length: int, drafter=None) -> int:
    """Bytes `capture` will clone at this length, WITHOUT cloning. The recurrent state is fixed;
    the KV is per token; the drafter's cache adds its own per-token figure WHEN it is
    snapshottable."""
    per_token = 2 * len(eng.cfg.attention_layers) * eng.cfg.num_key_value_heads * eng.cfg.head_dim * 2
    if _snapshottable(drafter):
        per_token += _drafter_bytes_per_token(drafter)
    # A conservative bound, not an exact figure: state and KV get their own margins, because the
    # point is "definitely not bigger than the cap", not "exactly this".
    return 2 * eng.state.nbytes + int(1.25 * per_token * length)


def capture(eng, drafter=None, max_bytes: int = 0) -> "StateSnapshot | None":
    """Clone the engine's state at its current length. Cheap in time, dear in bytes.

    `max_bytes > 0` refuses BEFORE cloning when the snapshot would exceed it -- the caller's
    store cap. Cloning first and declining in `put` still allocates and frees the whole snapshot
    every checkpoint, and that churn is what left the allocator holding ~31 GB of reserved
    segments after a 30k prefill on 2026-09-19; the check belongs here.
    """
    if max_bytes and _snapshot_estimate(eng, eng.kv.length, drafter) > max_bytes:
        return None
    settle = getattr(eng, "_settle", None)
    if settle is not None:
        settle()                      # a chain accepted in full whose walked state is not in yet
    n = eng.kv.length
    return StateSnapshot(
        n,
        eng.kv.k[:, :, :, :n, :].clone(),
        eng.kv.v[:, :, :, :n, :].clone(),
        eng.state.S.clone(),
        eng.state.conv.clone(),
        drafter.state_snapshot() if _snapshottable(drafter) else None,
        eng.kv.ks[..., :n].clone() if getattr(eng.kv, "fp8", False) else None,
        eng.kv.vs[..., :n].clone() if getattr(eng.kv, "fp8", False) else None,
    )


def restore(eng, snap: StateSnapshot, drafter=None) -> None:
    """Put `snap` back. After this the engine is exactly where `capture` was called."""
    n = snap.length
    eng.kv.k[:, :, :, :n, :].copy_(snap.k)
    eng.kv.v[:, :, :, :n, :].copy_(snap.v)
    if snap.ks is not None:
        eng.kv.ks[..., :n].copy_(snap.ks)
        eng.kv.vs[..., :n].copy_(snap.vs)
    eng.state.S.copy_(snap.S)
    eng.state.conv.copy_(snap.conv)
    eng._pending_walk = False
    eng._pend = None                  # a commit pending on the state just overwritten
    eng.kv.length = n
    eng.state.primed = n > 0
    eng._trace = None
    eng.tree = None
    eng.trace = None
    if snap.drafter is not None and _snapshottable(drafter):
        drafter.state_restore(snap.drafter)


def _snapshottable(drafter) -> bool:
    return drafter is not None and hasattr(drafter, "state_snapshot")


def drafter_is_cacheable(drafter) -> bool:
    """Can this drafter survive a prefix that was restored rather than forwarded?

    Three answers. A drafter with `state_snapshot` carries its own cache across, exactly. A drafter
    with no `sync` -- the lookup drafters -- is rebuilt from the token ids by `prime`, which the
    server calls anyway. A drafter that wants hidden states and cannot snapshot them would be
    handed a cache with a hole in it, and a hole in a position-indexed draft cache is permanent: it
    would decline for ever and the engine would decode at one token a block. So the cache turns
    itself off rather than trade a warm prefill for a cold decode.
    """
    if drafter is None or _snapshottable(drafter):
        return True
    return not hasattr(drafter, "sync")


# --------------------------------------------------------------------------- the memory budget
GIB = 1 << 30

#: A prefill's transient (the allocator's peak minus what it held before the prefill), as the
#: engine measured it on 2026-10-01: 1.15 GiB on short prompts, 4.87 GiB at 173,855 tokens. It is
#: the attention's scores and the chunk's activations, and it grows with the context the last
#: chunk attends over, so the estimate is the larger of the short-prompt floor and a straight line
#: through zero and the long reading (7.4 GiB at 262,144). `Budget.observe` raises the slope when a
#: prefill takes more than the line said.
WORKSPACE_FLOOR = int(1.15 * GIB)
WORKSPACE_PER_TOKEN = int(4.87 * GIB / 173_855)


class MemoryRefused(RuntimeError):
    """A prefill that would take the engine over its declared peak, with every cache already
    given back. The server answers it with a 503 and a Retry-After instead of running into the
    driver's out-of-memory wedge."""

    def __init__(self, need: int, peak: int, n: int):
        self.need, self.peak, self.n = int(need), int(peak), int(n)
        super().__init__(f"a {n}-token prefill needs {need / GIB:.1f} GiB with every cache "
                         f"given back; the engine's peak is {peak / GIB:.1f} GiB")


def _torch_memory() -> tuple[int, int]:
    import torch as _t
    return int(_t.cuda.memory_allocated()), int(_t.cuda.memory_reserved())


def _torch_empty() -> None:
    import torch as _t
    _t.cuda.empty_cache()


def _torch_peak(reset: bool = False) -> int:
    import torch as _t
    if reset:
        _t.cuda.reset_peak_memory_stats()
        return 0
    return int(_t.cuda.max_memory_allocated())


class Budget:
    """One memory policy for the serving caches (OPS-47, Khaled's decisions of 2026-10-01).

    THE NUMBERS. At load, with empty caches, the engine at a 262,144-token window holds 57 GiB.
    What grew from there to 75 and 83 GiB on 2026-10-01 was three things: the caches (14.5 GiB at
    their caps), a long prefill's workspace (4.9 GiB at 172k tokens) and the allocator's reserve,
    which kept every freed block (0.4, 2.5, 7.0 GiB over the day). So the policy has three limits:

      * `ceiling` -- what the engine holds between requests, allocated plus reserve. A cache grows
        only while `allocated + its bytes` stays under `ceiling - slack` (`admit`), and at the start
        of a request anything over it is given back (`gate`).
      * `peak` -- what the engine may hold during a prefill: the ceiling plus the workspace of the
        longest prompt the window allows. The residency planner counts this, not today's free
        memory, before a role may load beside the engine.
      * `slack` -- the reserve the allocator may keep between requests; over it, `done` hands the
        free blocks back to the driver once the request has finished.

    THE ORDER things are given back in, cheapest to lose first: `LEVELS`. The state store served 0
    hits on 2026-10-01; the anchors in the body of a prompt serve only a divergence in its middle
    (none in a night's ledger); the guest stash saves one session from a short request; the parked
    session saves a re-read of a whole other session; the pinned head saved 253 s in a day.

    WHY IT CANNOT CHANGE A TOKEN. Every limit here decides only where a prefill starts: a store
    entry, a resident anchor, the pin, the park, or token 0, each checked against the prompt's
    own tokens before any state is restored. Giving a cache back is the cold path the engine ran
    before the cache existed. A 503 changes no answer either: it is a request that was not served.

    `probe` returns (allocated, reserved) in bytes -- the CUDA caching allocator's, less what the
    decision models hold in the same allocator (`exclude`), since they are roles with claims of
    their own and not the engine. `empty` is `torch.cuda.empty_cache`. Both are arguments so the
    tests can run the policy on a fake allocator.
    """

    LEVELS = ("store", "anchors", "stash", "park", "pin")

    def __init__(self, ceiling_bytes: int, peak_bytes: int, slack_bytes: int, *,
                 probe=None, empty=None, exclude=None, peak_probe=None,
                 ws_floor: int = WORKSPACE_FLOOR, ws_per_token: int = WORKSPACE_PER_TOKEN,
                 on_event=None):
        self.ceiling = int(ceiling_bytes)
        self.peak = int(peak_bytes)
        self.slack = int(slack_bytes)
        self._probe = probe or _torch_memory
        self._empty = empty or _torch_empty
        self._peak = peak_probe or _torch_peak
        self.lifetime_peak = 0             # the allocator's peak since start, across the resets
        self._mark: int | None = None
        self.exclude = exclude
        self.ws_floor = int(ws_floor)
        self.ws_per_token = int(ws_per_token)
        self.on_event = on_event
        self.shedders: dict = {}
        self.ws_now = 0                    # the running prefill's workspace, until it is done
        self.stats = {"gates": 0, "refused": 0, "admitted": 0, "admit_refused": 0,
                      "empties": 0, "empty_ms": 0.0, "empty_freed": 0, "last_empty_ms": 0.0,
                      "shed": {lv: 0 for lv in self.LEVELS},
                      "shed_bytes": {lv: 0 for lv in self.LEVELS},
                      "observed": None, "last_gate": None, "prefill_peak": 0,
                      "last_prefill_peak": None}
        self.events: deque = deque(maxlen=50)
        self.lock = threading.Lock()

    @property
    def on(self) -> bool:
        return self.ceiling > 0 and self.peak > 0

    def register(self, level: str, fn) -> None:
        """`fn(safe) -> bytes given back`. `safe=True` is a call from inside a cache's own
        operation: only what nothing is using may go (a free buffer, the store), never the rows
        or anchors of the prompt being served."""
        if level not in self.LEVELS:
            raise ValueError(f"unknown level {level!r}: {self.LEVELS}")
        self.shedders[level] = fn

    def memory(self) -> tuple[int, int]:
        alloc, res = self._probe()
        x = int(self.exclude()) if self.exclude is not None else 0
        return max(0, alloc - x), max(0, res - x)

    def workspace(self, n: int) -> int:
        return max(self.ws_floor, self.ws_per_token * int(n))

    def _event(self, kind: str, **kw) -> dict:
        ev = dict(kind=kind, ts=round(time.time(), 3), **kw)
        self.events.append(ev)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:                                      # noqa: BLE001
                pass
        return ev

    def shed(self, level: str, why: str, safe: bool = False) -> int:
        fn = self.shedders.get(level)
        if fn is None:
            return 0
        freed = int(fn(safe) or 0)
        if freed:
            self.stats["shed"][level] += 1
            self.stats["shed_bytes"][level] += freed
            self._event("shed", level=level, why=why, freed=freed, safe=safe)
        return freed

    def empty(self, why: str) -> int:
        """Hand the allocator's free blocks back to the driver. Between requests only: the verify
        and draft graphs keep pools of their own, which this does not touch."""
        _, before = self.memory()
        t0 = time.perf_counter()
        self._empty()
        ms = (time.perf_counter() - t0) * 1e3
        _, after = self.memory()
        freed = max(0, before - after)
        self.stats["empties"] += 1
        self.stats["empty_ms"] += ms
        self.stats["last_empty_ms"] = round(ms, 2)
        self.stats["empty_freed"] += freed
        self._event("empty", why=why, freed=freed, ms=round(ms, 2))
        return freed

    # --- the request's start ---------------------------------------------------------------
    def gate(self, n: int) -> dict:
        """Before a prefill of an `n`-token prompt: back under the ceiling, room for the
        workspace under the peak, or `MemoryRefused`. Called with the engine lock held and
        before the response has started, so a refusal is still an HTTP status."""
        if not self.on:
            return {}
        with self.lock:
            self.stats["gates"] += 1
            shed: list = []
            alloc, res = self.memory()
            if res > self.ceiling and res > alloc:
                self.empty("ceiling")
                alloc, res = self.memory()
            for lv in self.LEVELS:
                if res <= self.ceiling:
                    break
                if self.shed(lv, "ceiling"):
                    shed.append(lv)
                    self.empty("ceiling")
                    alloc, res = self.memory()
            ws = self.workspace(n)
            if res + ws > self.peak and res > alloc:
                self.empty("peak")
                alloc, res = self.memory()
            for lv in self.LEVELS:
                if res + ws <= self.peak:
                    break
                if self.shed(lv, "peak"):
                    shed.append(lv)
                    self.empty("peak")
                    alloc, res = self.memory()
            g = {"n": int(n), "workspace": ws, "allocated": alloc, "reserved": res,
                 "shed": shed, "ts": round(time.time(), 3)}
            self.stats["last_gate"] = g
            if res + ws > self.peak:
                self.stats["refused"] += 1
                self._event("refused", n=int(n), need=res + ws, peak=self.peak)
                raise MemoryRefused(res + ws, self.peak, n)
            self.ws_now = ws
            return g

    def admit(self, level: str, nbytes: int) -> bool:
        """May a cache allocate `nbytes` more? Under `ceiling - slack` allocated and, while a
        prefill is still to run, with its workspace still under the peak. When not, the levels
        below `level` give back what nothing is using, once, and the question is asked again."""
        if not self.on or nbytes <= 0:
            return True
        lower = self.LEVELS[:self.LEVELS.index(level)] if level in self.LEVELS else ()
        for attempt in range(2):
            alloc, res = self.memory()
            if (alloc + nbytes <= self.ceiling - self.slack
                    and max(res, alloc + nbytes) + self.ws_now <= self.peak):
                self.stats["admitted"] += 1
                return True
            if attempt == 0:
                if not any([self.shed(lv, f"admit:{level}", safe=True) for lv in lower]):
                    break
        self.stats["admit_refused"] += 1
        self._event("admit_refused", level=level, nbytes=int(nbytes))
        return False

    def mark(self) -> None:
        """A prefill's forwards start here: the allocator's peak counts from now."""
        try:
            self.lifetime_peak = max(self.lifetime_peak, self._peak())
            self._peak(reset=True)
            self._mark = self._probe()[0]
        except Exception:                                          # noqa: BLE001
            self._mark = None

    def lifetime(self) -> int:
        """`max_memory_allocated` since the process started, whatever `mark` reset."""
        try:
            return max(self.lifetime_peak, self._peak())
        except Exception:                                          # noqa: BLE001
            return self.lifetime_peak

    def prefilled(self, n: int) -> None:
        """The prefill has run: its workspace is free again, and what it really took corrects the
        estimate upward when it was low. The workspace is the allocator's peak over the forwards
        less what is allocated once they are done -- not less what was allocated before them:
        the anchors and the store snapshot a prefill takes on its way are still allocated at its
        end, and counting them as workspace taught the estimate 113 kB a token from a 48k prompt
        (27.6 GiB at 262k) in the 2026-10-01 gate. Only a long prefill (64k tokens and more)
        teaches the slope, where the context's part dominates the fixed one."""
        self.ws_now = 0
        workspace = None
        if self._mark is not None:
            try:
                peak = self._peak()
                self.lifetime_peak = max(self.lifetime_peak, peak)
                workspace = peak - max(self._mark, self._probe()[0])
                x = int(self.exclude()) if self.exclude is not None else 0
                self.stats["last_prefill_peak"] = {"n": int(n), "peak": peak - x}
                self.stats["prefill_peak"] = max(self.stats["prefill_peak"], peak - x)
            except Exception:                                      # noqa: BLE001
                workspace = None
        self._mark = None
        if workspace is None or n < 65_536:
            return
        self.stats["observed"] = {"n": int(n), "workspace": int(workspace),
                                  "estimate": self.workspace(n)}
        per = int(workspace / n)
        if per > self.ws_per_token:
            self._event("estimate_raised", n=int(n), workspace=int(workspace),
                        per_token=per, was=self.ws_per_token)
            self.ws_per_token = per

    def done(self) -> int:
        """After a request: the reserve over its slack goes back to the driver."""
        self.ws_now = 0
        if not self.on:
            return 0
        with self.lock:
            alloc, res = self.memory()
            if res - alloc > self.slack:
                return self.empty("idle")
        return 0

    def report(self) -> dict:
        alloc, res = self.memory()
        return {"ceiling": self.ceiling, "peak": self.peak, "slack": self.slack,
                "allocated": alloc, "reserved": res, "reserve": res - alloc,
                "ws_floor": self.ws_floor, "ws_per_token": self.ws_per_token,
                "events": list(self.events)[-20:],
                **{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.stats.items()}}


# --------------------------------------------------------------------------- the state store
class _Entry:
    __slots__ = ("tokens", "snap", "conv_id", "created", "hits", "kind")

    def __init__(self, tokens, snap, conv_id, kind="prefix"):
        self.tokens = tokens
        self.snap = snap
        self.conv_id = conv_id
        self.created = time.time()
        self.hits = 0
        # where the snapshot came from: the end of a turn ("session") or a prefill chunk boundary
        # ("prefix"). One store holds both and a lookup cannot tell them apart otherwise.
        self.kind = kind


class StateStore:
    """Token-prefix -> engine state, under a byte budget, least-recently-used out first.

    Keyed by `(length, hash)` and verified by the tokens themselves. `lengths` is the set of
    boundaries anything has been stored at, which is what makes a lookup cheap: the candidates for
    a new prompt are the stored lengths below it, and each one is one dictionary probe.
    """

    def __init__(self, budget_bytes: int, chunk: int = 256, max_entry_bytes: int = 0,
                 per_prefill: int = 0):
        self.budget = int(budget_bytes)
        # How many chunk boundaries of ONE prefill are stored: its longest ones that fit the
        # per-entry cap. Every boundary (0) nests: a cold prefill used to put 13 snapshots of its
        # own prefix (1,024 to 13,312 tokens, ~11 GB), which is more than the budget, so each cold
        # prefill evicted everything else -- 214 puts and 206 evictions in the 2026-09-30 night.
        self.per_prefill = max(0, int(per_prefill))
        # One snapshot may not exceed this, HOWEVER much budget is free. At a 256k window a
        # boundary-length snapshot is 154 MB + 85.5 kB x tokens -- several GB -- and `put` clones
        # it BEFORE evictions run, so the allocator sees store + clone at once. That transient
        # overshoot on a 121 GiB board (53 GB engine + 24 GB store + 40 GB page cache) is the
        # 2026-09-19 wedge: a long-prompt prefill pushed MemAvailable to zero and the GPU driver
        # locked. Default: a quarter of the budget.
        self.max_entry = int(max_entry_bytes) or 0   # 0 = no per-entry cap (the server sets one)
        self.chunk = int(chunk)
        self._d: OrderedDict = OrderedDict()
        self._lengths: dict[int, int] = {}       # length -> how many entries sit at it
        self.bytes = 0
        self.stats = {"puts": 0, "hits": 0, "misses": 0, "evictions": 0,
                      "tokens_reused": 0, "tokens_forwarded": 0, "rejected_collisions": 0,
                      "hits_session": 0, "hits_prefix": 0}
        # the kind of the entry the latest `find` restored, for the request's `cache_source`
        self.last_kind: str | None = None
        self.lock = threading.Lock()

    # --- writing ---------------------------------------------------------------------------
    def put(self, tokens, snap: StateSnapshot, conv_id: str | None = None,
            hashes: list[int] | None = None, kind: str = "prefix") -> None:
        n = snap.length
        if self.max_entry and snap.nbytes > self.max_entry:
            self.stats["declined_big"] = self.stats.get("declined_big", 0) + 1
            return
        if n == 0 or n > len(tokens):
            # The engine forwarded more tokens than the caller collected, which is every generation
            # that ended inside a block: an abandoned stream, or a token budget that ran out part
            # way through an accepted path. There is no honest snapshot to store for it. The KV
            # could be truncated but the RECURRENT state cannot -- it has already absorbed those
            # tokens and there is no inverse -- so an entry keyed by the shorter prefix would
            # restore a state that has seen text the key does not mention.
            #
            # Counted rather than silent, because a store that declines every put looks exactly
            # like a store that is switched off, and phase 9 spent an hour on the difference.
            self.stats["declined_short"] = self.stats.get("declined_short", 0) + 1
            return
        key = (n, (hashes[n] if hashes is not None else prefix_hashes(tokens[:n])[n]))
        with self.lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= old.snap.nbytes
                self._lengths[n] -= 1
                if not self._lengths[n]:
                    del self._lengths[n]
            self._d[key] = _Entry(tuple(tokens[:n]), snap, conv_id, kind)
            self._lengths[n] = self._lengths.get(n, 0) + 1
            self.bytes += snap.nbytes
            self.stats["puts"] += 1
            self._evict_locked()

    def _evict_locked(self, room: int = 0) -> None:
        while self.bytes + room > self.budget and self._d:
            key, entry = self._d.popitem(last=False)
            self.bytes -= entry.snap.nbytes
            n = key[0]
            self._lengths[n] -= 1
            if not self._lengths[n]:
                del self._lengths[n]
            self.stats["evictions"] += 1

    def reserve(self, nbytes: int) -> bool:
        """Make room for a snapshot of `nbytes` BEFORE it is cloned (OPS-47): evict the oldest
        entries until the store plus it fits the budget. `put` evicting after the clone meant
        the allocator held the full store and the new snapshot at once, every checkpoint. False,
        and nothing evicted, when the snapshot is over the entry cap or the whole budget."""
        if (self.max_entry and nbytes > self.max_entry) or nbytes > self.budget:
            return False
        with self.lock:
            self._evict_locked(nbytes)
        return True

    def drop_session(self, conv_id: str | None) -> int:
        """Forget every entry of one conversation -- its compaction has started, and the history
        those entries are keyed by will not be sent again. Returns the bytes given back."""
        if not conv_id:
            return 0
        with self.lock:
            keys = [k for k, e in self._d.items() if e.conv_id == conv_id]
            freed = 0
            for key in keys:
                entry = self._d.pop(key)
                freed += entry.snap.nbytes
                self._lengths[key[0]] -= 1
                if not self._lengths[key[0]]:
                    del self._lengths[key[0]]
            self.bytes -= freed
            if keys:
                self.stats["dropped_session"] = self.stats.get("dropped_session", 0) + len(keys)
            return freed

    def shed(self, safe: bool = False) -> int:
        """`Budget`'s first level: everything, the cheapest cache to lose. Returns the bytes."""
        with self.lock:
            freed = self.bytes
        self.clear()
        return freed

    def set_limits(self, budget_bytes: int | None = None, max_entry_bytes: int | None = None):
        """A new budget or entry cap, from the next request on; over the budget goes now."""
        with self.lock:
            if budget_bytes is not None:
                self.budget = int(budget_bytes)
            if max_entry_bytes is not None:
                self.max_entry = int(max_entry_bytes)
            self._evict_locked()

    def boundaries(self, eng, drafter, start: int, n: int, chunk: int) -> set | None:
        """The chunk boundaries of a prefill over [start, n) to snapshot, or None for every one
        (`per_prefill` 0, where `capture` refuses the ones over the cap as it goes)."""
        if not self.per_prefill:
            return None
        lines = range((start // chunk + 1) * chunk, n, chunk)
        fit = [L for L in lines
               if not self.max_entry or _snapshot_estimate(eng, L, drafter) <= self.max_entry]
        if len(fit) < len(lines):
            self.stats["skipped_big"] = self.stats.get("skipped_big", 0) + 1
        return set(fit[-self.per_prefill:])

    # --- reading ---------------------------------------------------------------------------
    def find(self, tokens, hashes: list[int], max_len: int, min_len: int = 0):
        """The longest stored prefix of `tokens` no longer than `max_len`, or None.

        `max_len` is normally `len(tokens) - 1`: a request needs at least one token to forward,
        because the logits it answers with are the ones that token's own forward produces and no
        snapshot holds them. `min_len`: only an entry LONGER than this counts -- the resident
        prefix already offers that much without a copy, so a shorter entry is not a hit.
        """
        with self.lock:
            cands = sorted((L for L in self._lengths if min_len < L <= max_len), reverse=True)
            for L in cands:
                key = (L, hashes[L])
                entry = self._d.get(key)
                if entry is None:
                    continue
                if entry.tokens != tuple(tokens[:L]):
                    # a 64-bit collision, or two different token streams at one boundary. Either
                    # way this is not the state that prefix would have produced.
                    self.stats["rejected_collisions"] += 1
                    continue
                self._d.move_to_end(key)
                entry.hits += 1
                self.stats["hits"] += 1
                self.stats[f"hits_{entry.kind}"] = self.stats.get(f"hits_{entry.kind}", 0) + 1
                self.last_kind = entry.kind
                return L, entry
            if not min_len:
                # a request the resident prefix serves is not a store miss
                self.stats["misses"] += 1
            self.last_kind = None
            return None

    def report(self) -> dict:
        with self.lock:
            per = {}
            for e in self._d.values():
                for k, v in e.snap.parts().items():
                    per[k] = per.get(k, 0) + v
            return {"entries": len(self._d), "bytes": self.bytes, "budget": self.budget,
                    "boundaries": sorted(self._lengths), "bytes_by_part": per,
                    "chunk": self.chunk, **self.stats}

    def clear(self) -> None:
        with self.lock:
            self._d.clear()
            self._lengths.clear()
            self.bytes = 0


# --------------------------------------------------------------------------- the resident prefix
#: Rows past a generation's final `kv.length` that its rejected verify rows or a drafter's block
#: may have written. The widest block the served stack writes is 33 rows (a 32-node tree plus its
#: anchor); 512 is the margin a guest's stash is checked against.
GUEST_SLACK_ROWS = 512


def resident_capable(drafter) -> bool:
    """Can this drafter pick up at a position whose draft KV it already holds in place?

    The same three answers as `drafter_is_cacheable`, with `state_resume` in place of
    `state_snapshot`: a drafter with no `sync` is rebuilt from token ids by `prime`; one with
    `state_resume` is told where its own position-indexed cache is valid up to; any other would
    be left with a hole, so the resident prefix turns itself off for it.
    """
    if drafter is None:
        return True
    fn = getattr(drafter, "can_resume", None)
    if fn is not None:
        return bool(fn())
    if hasattr(drafter, "state_resume"):
        return True
    return not hasattr(drafter, "sync")


def _kv_views(eng, drafter) -> list:
    """Every position-indexed buffer a guest request can overwrite, as (tensor, axis)."""
    views = [(eng.kv.k, 3), (eng.kv.v, 3)]
    if getattr(eng.kv, "fp8", False):
        views += [(eng.kv.ks, 3), (eng.kv.vs, 3)]
    fn = getattr(drafter, "kv_views", None) if drafter is not None else None
    if fn is not None:
        views += [(t, ax) for t, ax in fn() if t is not None]
    return views


class ResidentPrefix:
    """The live KV buffer used as the prefix cache, with the recurrent state at a few boundaries.

    THE PROBLEM IT SOLVES. An agent client -- opencode -- sends one conversation
    that only ever grows: every turn's prompt is the previous prompt plus the previous answer
    plus a tool result. `StateStore` keeps whole snapshots, and a snapshot is 147 MiB of
    recurrent state plus 104 kB a token of target and draft KV, so under its per-entry cap
    (a quarter of CACHE_GB) nothing past 13,312 tokens was ever stored: at 190k tokens every
    turn re-read the whole conversation, ~310 s a turn, `skipped_big` 13,039 by the evening.

    THE OBSERVATION. The KV the previous request prefilled is still sitting in the engine's own
    buffer, row for row, and a KV row depends only on the tokens up to it. What cannot be
    recovered from the buffer is the recurrent state: the 48 GDN layers' S and convolution tail
    have absorbed every token and there is no inverse. So this keeps the KV where it is -- no
    copy -- and clones only S and conv (146.8 MiB) at chunk-grid boundaries of the prefill,
    "anchors". A request whose prompt shares the first `c` tokens of the resident prompt resumes
    at the largest anchor `B <= c`: S and conv are copied back, `kv.length = B`, and the drafter
    is told its own position-indexed cache is valid up to `B` (`state_resume`).

    WHY IT IS LOSSLESS, not merely close. An anchor is taken right after the prefill chunk that
    ends at `B`, the rows below `B` were written by that same grid of prefill chunks, and the
    resumed request forwards `B..n` on the same grid a cold prefill would use. Every forward after
    the resume therefore sees the same inputs a cold prefill's does, and `tests/test_resident.py`
    checks the logits, S, conv and KV rows for bit equality. Anchors are taken ONLY from prefill
    chunks, never from rows a verify block wrote: the generated answer is always re-read as
    prompt on the next turn, which is what a cold prefill of that turn does too. A request that
    started from a SESSION snapshot (written by verify blocks) leaves no anchors at all.

    GUESTS. A short unrelated request between two turns (opencode's title call, an Open WebUI
    chat) would overwrite the rows the conversation needs. A request shorter than `guest_frac` of
    the resident prompt and `guest_max` tokens is a guest: before its prefill the rows it can
    reach are copied aside (up to `stash_bytes`), and they are copied back at the start of the
    next request. If the guest ran further than the stash, the resident prefix is cut to what
    the stash covered.

    MEMORY. `budget_bytes` bounds the anchors (27 at the default 4 GiB) and `stash_bytes` the
    guest stash (2 GiB = ~20k rows at 104 kB a row). Both are allocated once and reused: an
    evicted anchor's buffers take the next anchor, so a 190k prefill that passes 185 boundaries
    allocates no more than the budget.

    A LONG FOREIGN REQUEST (`pin_bytes`). A request too long to be a guest replaces the resident
    prompt. opencode's compaction request is one: no tools, no system text, the history as one
    user message, 110k-132k tokens that share nothing with the conversation. The next build
    request starts with the same reasoning header, tool block and system text as every request
    before the compaction (at least 20,480 tokens), and found nothing: 7 of 7 re-read 52k-58k
    tokens from zero on the 2026-09-30 night. So before such a request cuts the rows, the head of
    the old prompt is PINNED: its KV rows and the anchors inside it are copied aside, as long a
    head as `pin_bytes` holds. A later request that shares more with the pinned head than with
    the rows in place gets them copied back and resumes at the largest pinned anchor it shares.
    The rows and anchors are the bytes a grid prefill wrote, so the resume is as exact as any
    other anchor's.

    ANOTHER SESSION (`park_bytes`, OPS-47). Two opencode sessions used in turn (A, then B, then A)
    share their head and nothing after it, and each switch re-read the other session past the
    head: 81,199 tokens in 125 s on the 2026-09-30 night. So when a request of ANOTHER session
    (`x-session-id`, or no id and a prompt that parts from the rows) replaces the resident prompt,
    the rows it is about to overwrite -- from where the two part up to the newest anchor that fits
    -- are copied aside with the newest `park_anchors` anchors, which move rather than copy. A
    later request that shares more with the parked prompt than with the rows in place gets them
    back: the parked rows and the rows in their place are SWAPPED, so B is parked while A comes
    back, and A, B, A, B is a copy each time instead of a re-read. The head below the parked rows
    has to be in place (it is the head both sessions share); the rows and anchors are a grid
    prefill's bytes, so the resume is as exact as the pin's.

    THE MEMORY POLICY (`mem`, a `Budget`). Every buffer this allocates after the first is asked for
    first: an anchor that is refused reuses an evicted one, a stash that is refused leaves the
    guest to overwrite the rows (the prompt is cut there), a pin or a park that is refused is not
    taken. `shed_*` give each part back when the budget says so, in its order.
    """

    def __init__(self, budget_bytes: int, chunk: int, *, tail: int = 4, stash_bytes: int = 0,
                 guest_max: int = 16384, guest_frac: float = 0.25,
                 pin_bytes: int = 0, pin_anchors: int = 8, park_bytes: int = 0,
                 park_anchors: int = 4, mem: "Budget | None" = None):
        self.budget = int(budget_bytes)
        self.chunk = int(chunk)
        self.tail = max(1, int(tail))
        self.stash_budget = int(stash_bytes)
        self.guest_max = int(guest_max)
        self.guest_frac = float(guest_frac)
        self.tokens: list[int] = []        # the prompt the live rows [0, valid) were prefilled from
        self.valid = 0
        self.anchors: dict[int, tuple] = {}
        self._free: list[tuple] = []       # an evicted anchor's buffers, for the next one
        self._allocated = 0                # anchor buffers alive, in use or free
        self._anchor_bytes = 0
        self._stash: dict | None = None    # the rows a guest may overwrite, copied aside
        self._stash_bufs: list | None = None
        self._stash_rows = 0
        self.capturing = False             # this request's prefill may add anchors
        # the pinned head of an earlier prompt: rows [0, pin_rows) of every view, and anchors
        self.pin_budget = int(pin_bytes)
        self.pin_anchor_max = max(1, int(pin_anchors))
        self.pin_tokens: list[int] = []
        self.pin_rows = 0
        self.pin_anchors: dict[int, tuple] = {}
        self._pin_bufs: list | None = None
        self._pin_views: list | None = None
        self._pin_spare: list[tuple] = []  # anchor-shaped buffers the pin owns
        # the parked session: rows [park_base, park_base + park_rows) of `park_tokens`, and anchors
        self.park_budget = int(park_bytes)
        self.park_anchor_max = max(1, int(park_anchors))
        self.park_tokens: list[int] = []
        self.park_base = 0
        self.park_rows = 0
        self.park_owner: str | None = None
        self.park_anchors: dict[int, tuple] = {}
        self._park_bufs: list | None = None
        self._park_views: list | None = None
        self.owner: str | None = None      # the session (x-session-id) the rows belong to
        self.mem = mem                     # the memory policy (Budget), None for no limits
        self.switches: deque = deque(maxlen=20)
        self.stats = {"hits": 0, "misses": 0, "tokens_reused": 0, "anchors_taken": 0,
                      "anchors_evicted": 0, "guests": 0, "stash_restored": 0, "stash_short": 0,
                      "invalidated": 0,
                      "pins": 0, "pin_hits": 0, "pin_tokens_reused": 0, "pin_refused": 0,
                      "parks": 0, "park_hits": 0, "park_swaps": 0, "park_tokens_reused": 0,
                      "park_refused": 0, "park_dropped": 0, "switches": 0,
                      "refused_anchor": 0, "refused_stash": 0, "refused_pin": 0,
                      "refused_park": 0}
        self.lock = threading.Lock()

    # --- lookups ---------------------------------------------------------------------------
    def common(self, ids) -> int:
        """How many leading tokens `ids` shares with the prompt the resident rows hold."""
        m = min(len(ids), self.valid)
        if m == 0:
            return 0
        if list(ids[:m]) == self.tokens[:m]:
            return m
        lo, hi = 0, m            # tokens[:lo] equal, tokens[:hi] not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if list(ids[:mid]) == self.tokens[:mid]:
                lo = mid
            else:
                hi = mid
        return lo

    def best(self, c: int, max_len: int) -> int:
        """The largest anchor at or below both `c` and `max_len`; 0 for none."""
        lim = min(c, max_len)
        return max((b for b in self.anchors if b <= lim), default=0)

    # --- the request's start ---------------------------------------------------------------
    def settle_guest(self, eng) -> None:
        """Put back what the last guest request may have overwritten. Called before anything
        else reads the resident rows."""
        st = self._stash
        if st is None:
            return
        self._stash = None
        s, r = st["start"], st["rows"]
        for (t, ax), buf in zip(st["views"], self._stash_bufs):
            t.narrow(ax, s, r).copy_(buf.narrow(ax, 0, r))
        self.stats["stash_restored"] += 1
        reach = int(eng.kv.length) + GUEST_SLACK_ROWS
        if reach > s + r and self.valid > s + r:
            # the guest wrote past what was saved: the rows beyond it are not the resident's any
            # more, and neither is any anchor above them
            self.stats["stash_short"] += 1
            self._cut(s + r)

    def is_guest(self, n: int, c: int) -> bool:
        return (self.valid > 0 and c < self.valid and n <= self.guest_max
                and n < self.guest_frac * self.valid and self.stash_budget > 0)

    def resume(self, eng, drafter, b: int) -> None:
        """Bring the engine to `b` from the rows in place and the anchor at `b`."""
        S, conv = self.anchors[b]
        eng.state.S.copy_(S)
        eng.state.conv.copy_(conv)
        eng._pending_walk = False
        eng._pend = None
        eng.kv.length = b
        eng.state.primed = b > 0
        eng._trace = None
        eng.tree = None
        eng.trace = None
        if drafter is not None and hasattr(drafter, "state_resume"):
            drafter.state_resume(b)
        self.stats["hits"] += 1
        self.stats["tokens_reused"] += b

    def begin(self, eng, drafter, ids, start: int, c: int, kind: str | None,
              session: str | None = None, compacting: bool = False) -> None:
        """What the rows will hold once this request's prefill has run.

        `start` is where the prefill resumes (a resident anchor, a store entry, or 0) and `c` how
        much of the resident prompt `ids` shares. A guest keeps the resident prompt and stashes the
        rows it can reach; anything else becomes the resident prompt from here on. `session` is
        the request's session key; `compacting` says it is that session's compaction, whose old
        rows nobody asks for again (they are not parked).
        """
        self.capturing = False
        if kind == "session":
            # rows written by verify blocks: nothing anchored after them is a cold prefill's state
            self.clear()
            self.owner = session
            return
        n = len(ids)
        if self.is_guest(n, c):
            self._stash_for(eng, drafter, min(start, (c // self.chunk) * self.chunk))
            return
        if self.valid and c < self.valid:
            self.stats["invalidated"] += 1
        keep = min(start, c)
        parked = self.stats["parks"]
        if (self.park_budget and self.valid > keep and not compacting
                and (session is None or session != self.owner)):
            self._park(eng, drafter, keep)
        # A parked prompt keeps its rows above `keep`, and its head below `keep` stays in place
        # (the new prompt shares it): pinning that head too held the same session twice -- 3.8 GiB
        # of pin beside 4 GiB of park in the 2026-10-01 gate. The pin stays the compaction's (and
        # any replacement the park did not take).
        if self.pin_budget and self.valid > keep and self.stats["parks"] == parked:
            self._pin(eng, drafter, keep)
        self._cut(keep)
        self.tokens = list(ids)
        self.valid = start
        self.owner = session
        self.capturing = self.chunk > 0

    def advanced(self, i: int) -> None:
        """The prefill has written rows up to `i`."""
        if self.capturing:
            self.valid = i

    # --- anchors ---------------------------------------------------------------------------
    def anchor(self, eng, b: int) -> None:
        """Clone the recurrent state at grid boundary `b`, reusing an evicted anchor's buffers."""
        if not self.capturing or b in self.anchors:
            return
        if not self._anchor_bytes:
            self._anchor_bytes = eng.state.nbytes
        cap = max(self.tail + 1, self.budget // max(1, self._anchor_bytes))
        if len(self.anchors) >= cap:
            self._evict_one()
        if not self._free and self.mem is not None and not self.mem.admit(
                "anchors", self._anchor_bytes):
            # over the budget: the next anchor takes an evicted one's buffers, or is not taken
            self.stats["refused_anchor"] += 1
            if not self.anchors:
                return
            self._evict_one()
        if self._free:
            S, conv = self._free.pop()
            S.copy_(eng.state.S)
            conv.copy_(eng.state.conv)
        else:
            S, conv = eng.state.S.clone(), eng.state.conv.clone()
            self._allocated += 1
        self.anchors[b] = (S, conv)
        self.stats["anchors_taken"] += 1

    def _buffers(self, eng) -> tuple | None:
        """A free (S, conv) pair for the next anchor, evicting one if the budget is full; None
        when the memory budget refuses a new one and no anchor is left to evict."""
        if not self._anchor_bytes:
            self._anchor_bytes = eng.state.nbytes
        cap = max(self.tail + 1, self.budget // max(1, self._anchor_bytes))
        if len(self.anchors) >= cap:
            self._evict_one()
        if not self._free and self.mem is not None and not self.mem.admit(
                "anchors", self._anchor_bytes):
            self.stats["refused_anchor"] += 1
            if not self.anchors:
                return None
            self._evict_one()
        if self._free:
            return self._free.pop()
        self._allocated += 1
        return torch.empty_like(eng.state.S), torch.empty_like(eng.state.conv)

    def head_line(self, b: int) -> bool:
        """With a pin: one of the `pin_anchor_max` anchors every `4 * chunk` tokens at the head of
        the prompt, which eviction keeps. The head is where an agent conversation parts from
        itself -- its tool block and system text, a compaction, a new day's date line -- and the
        smallest-gap rule alone thins it out as the conversation grows: at 200k tokens the lowest
        surviving anchor sat above a 5k tool block, and the pin had nothing to offer."""
        step = 4 * self.chunk
        return bool(self.pin_budget) and b % step == 0 and b <= step * self.pin_anchor_max

    def _evict_one(self) -> None:
        """Drop the anchor whose removal leaves the smallest gap, never one of the newest
        `tail` (the next turn resumes near the end) nor a head line; an early divergence
        anywhere."""
        bs = sorted(self.anchors)
        body = [(j, b) for j, b in enumerate(bs[:-self.tail] if len(bs) > self.tail else [])
                if not self.head_line(b)]
        if not body:
            victim = next((b for b in bs if not self.head_line(b)), bs[0])
        else:
            best, victim = None, body[0][1]
            for j, b in body:
                lo = bs[j - 1] if j > 0 else 0
                hi = bs[j + 1]
                gap = hi - lo
                if best is None or gap < best:
                    best, victim = gap, b
        self._free.append(self.anchors.pop(victim))
        self.stats["anchors_evicted"] += 1

    def _cut(self, keep: int) -> None:
        """The rows at and above `keep` are not the resident's any more."""
        for b in [b for b in self.anchors if b > keep]:
            self._free.append(self.anchors.pop(b))
        self.valid = min(self.valid, keep)
        self.tokens = self.tokens[:self.valid]

    # --- guests ----------------------------------------------------------------------------
    def _stash_for(self, eng, drafter, s: int) -> None:
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        cap = self.stash_budget // max(1, per_row)
        rows = max(0, min(self.valid - s, cap, eng.kv.max_len - s))
        if rows <= 0:
            self._cut(s)
            return
        if (self._stash_bufs is None or self._stash_rows < rows
                or len(self._stash_bufs) != len(views)):
            size = min(cap, max(rows, self.valid))
            self._stash_bufs, self._stash_rows = None, 0
            if self.mem is not None and not self.mem.admit("stash", size * per_row):
                # no room for the stash: the guest overwrites the rows, which stop being the
                # resident's -- the next turn re-reads them, as it did before the stash existed
                self.stats["refused_stash"] += 1
                self._cut(s)
                return
            self._stash_bufs = [torch.empty(*(t.shape[:ax] + (size,) + t.shape[ax + 1:]),
                                            dtype=t.dtype, device=t.device) for t, ax in views]
            self._stash_rows = size
        for (t, ax), buf in zip(views, self._stash_bufs):
            buf.narrow(ax, 0, rows).copy_(t.narrow(ax, s, rows))
        self._stash = {"start": s, "rows": rows, "views": views}
        self.stats["guests"] += 1
        if rows < self.valid - s:
            # the stash could not hold every row a guest might reach; what it did not save is
            # decided after the guest, from how far it actually wrote
            pass

    # --- the pinned head ---------------------------------------------------------------------
    def _pin(self, eng, drafter, keep: int) -> None:
        """Copy aside the longest anchored head of the resident prompt that `pin_bytes` holds,
        when it is longer than the `keep` rows the next prefill leaves in place."""
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        ab = self._anchor_bytes or eng.state.nbytes
        heads = sorted(b for b in self.anchors if b <= self.valid)
        best = 0
        for j, b in enumerate(heads):
            k = min(j + 1, self.pin_anchor_max)
            if b * per_row + k * ab <= self.pin_budget:
                best = b
        if best < keep + self.chunk:
            return
        if self.pin_rows and self.pin_tokens[:best] == self.tokens[:best]:
            return                         # the same head is already pinned
        # anchors spread over the head, the longest always among them: where the next request
        # parts from it is not known yet, and it resumes at the largest pinned anchor below that.
        # The head lines are that spread (eviction keeps them); without them, the anchors
        # nearest to an even spread.
        below = [b for b in heads if b <= best]
        k = self.pin_anchor_max
        lines = [b for b in below if self.head_line(b) and b != best]
        if lines:
            keep_b = sorted(lines[-(k - 1):] + [best]) if k > 1 else [best]
        else:
            keep_b = sorted({max(b for b in below if b <= max(below[0], best * j // k))
                             for j in range(1, k + 1)})
        if (self._pin_bufs is None or len(self._pin_bufs) != len(views)
                or self._pin_bufs[0].shape[views[0][1]] < best):
            size = max(best, self._pin_bufs[0].shape[views[0][1]] if self._pin_bufs else 0)
            size = min(size, max(best, (self.pin_budget - len(keep_b) * ab) // max(1, per_row)))
            spare = max(0, len(keep_b) - len(self._pin_spare) - len(self.pin_anchors))
            if self._pin_bufs is not None:
                # the old head's rows go: the new head is a longer copy of it or another prompt's
                self._drop_pin(rows_only=True)
            if self.mem is not None and not self.mem.admit("pin", size * per_row + spare * ab):
                self.stats["refused_pin"] += 1
                return
            self._pin_bufs = [torch.empty(*(t.shape[:ax] + (size,) + t.shape[ax + 1:]),
                                          dtype=t.dtype, device=t.device) for t, ax in views]
        for (t, ax), buf in zip(views, self._pin_bufs):
            buf.narrow(ax, 0, best).copy_(t.narrow(ax, 0, best))
        self._pin_spare += list(self.pin_anchors.values())
        self.pin_anchors = {}
        for b in keep_b:
            if self._pin_spare:
                S, conv = self._pin_spare.pop()
            else:
                S, conv = torch.empty_like(eng.state.S), torch.empty_like(eng.state.conv)
            S.copy_(self.anchors[b][0])
            conv.copy_(self.anchors[b][1])
            self.pin_anchors[b] = (S, conv)
        self._pin_spare = []               # what an earlier, longer head needed is not kept
        self.pin_tokens = list(self.tokens[:best])
        self.pin_rows = best
        self._pin_views = [(t, ax) for t, ax in views]
        self.stats["pins"] += 1

    def pin_offer(self, ids, c: int, max_len: int) -> int:
        """The pinned anchor a prompt could resume at, when it beats the rows in place; else 0."""
        if not self.pin_rows or not self.pin_anchors:
            return 0
        m = min(len(ids), self.pin_rows)
        if list(ids[:m]) == self.pin_tokens[:m]:
            cp = m
        else:
            lo, hi = 0, m
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if list(ids[:mid]) == self.pin_tokens[:mid]:
                    lo = mid
                else:
                    hi = mid
            cp = lo
        bp = max((b for b in self.pin_anchors if b <= min(cp, max_len)), default=0)
        return bp if bp > self.best(c, max_len) else 0

    def unpin(self, eng, drafter) -> bool:
        """Make the pinned head the resident prompt: its rows copied back, its anchors the live
        ones. False, and nothing changed, if the buffers it was copied from are not the ones in
        service any more (a drafter that reallocated its cache)."""
        views = _kv_views(eng, drafter)
        if (self._pin_views is None or len(views) != len(self._pin_views)
                or any(a is not b or x != y
                       for (a, x), (b, y) in zip(views, self._pin_views))):
            self.stats["pin_refused"] += 1
            return False
        self._stash = None
        for b in list(self.anchors):
            self._free.append(self.anchors.pop(b))
        r = self.pin_rows
        for (t, ax), buf in zip(views, self._pin_bufs):
            t.narrow(ax, 0, r).copy_(buf.narrow(ax, 0, r))
        for b, (S0, c0) in sorted(self.pin_anchors.items()):
            S, conv = self._buffers(eng)
            S.copy_(S0)
            conv.copy_(c0)
            self.anchors[b] = (S, conv)
        self.tokens = list(self.pin_tokens)
        self.valid = r
        self.stats["pin_hits"] += 1
        return True

    def _drop_pin(self, rows_only: bool = False) -> int:
        freed = sum(b.numel() * b.element_size() for b in (self._pin_bufs or []))
        self._pin_bufs, self._pin_views = None, None
        self.pin_tokens, self.pin_rows = [], 0
        if not rows_only:
            freed += (len(self.pin_anchors) + len(self._pin_spare)) * self._anchor_bytes
            self.pin_anchors, self._pin_spare = {}, []
        return freed

    # --- the parked session ------------------------------------------------------------------
    def _park(self, eng, drafter, keep: int) -> None:
        """Copy aside the rows of the prompt being replaced, from `keep` (where the new prompt
        parts from it) to its newest anchor that `park_bytes` holds; its newest anchors move with
        them. The previous parked session, if any, is given up."""
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        ab = self._anchor_bytes or eng.state.nbytes
        cap_rows = max(0, (self.park_budget - self.park_anchor_max * ab) // max(1, per_row))
        above = sorted(b for b in self.anchors if keep < b <= min(self.valid, keep + cap_rows))
        rows = (above[-1] - keep) if above else 0
        if rows < self.chunk:
            return
        self._drop_park_rows()
        have = self._park_bufs[0].shape[views[0][1]] if (
            self._park_bufs is not None and len(self._park_bufs) == len(views)) else 0
        if have < rows:
            # whole 4-line steps, so a session that grows by a turn does not reallocate each time
            size = min(cap_rows, -(-rows // (4 * self.chunk)) * 4 * self.chunk)
            self._park_bufs = None
            if self.mem is not None and not self.mem.admit("park", size * per_row):
                self.stats["refused_park"] += 1
                return
            self._park_bufs = [torch.empty(*(t.shape[:ax] + (size,) + t.shape[ax + 1:]),
                                           dtype=t.dtype, device=t.device) for t, ax in views]
        for (t, ax), buf in zip(views, self._park_bufs):
            buf.narrow(ax, 0, rows).copy_(t.narrow(ax, keep, rows))
        for b in above[-self.park_anchor_max:]:
            self.park_anchors[b] = self.anchors.pop(b)
            self._allocated -= 1
        self.park_tokens = list(self.tokens[:above[-1]])
        self.park_base, self.park_rows, self.park_owner = keep, rows, self.owner
        self._park_views = [(t, ax) for t, ax in views]
        self.stats["parks"] += 1

    def _drop_park_rows(self) -> None:
        """Forget the parked prompt but keep its row buffers: its anchors join the live pool's
        free list, as far as the pool's cap allows."""
        for b in list(self.park_anchors):
            self._free.append(self.park_anchors.pop(b))
            self._allocated += 1
        self.park_tokens, self.park_base, self.park_rows, self.park_owner = [], 0, 0, None
        self._trim_pool()

    def _trim_pool(self) -> None:
        """The live pool back under its cap: anchors evicted, then free buffers let go. Anchors
        that come back from the park can take it over the cap for a moment."""
        if not self._anchor_bytes:
            return
        cap = max(self.tail + 1, self.budget // self._anchor_bytes)
        while len(self.anchors) > cap:
            self._evict_one()
        while self._allocated > cap and self._free:
            self._free.pop()
            self._allocated -= 1

    def park_offer(self, ids, c: int, max_len: int) -> int:
        """The parked anchor a prompt could resume at, when it beats the rows in place; else 0.
        The head below the parked rows must be the rows in place: `ids` shares it with them."""
        if not self.park_rows or not self.park_anchors or c < self.park_base:
            return 0
        top = self.park_base + self.park_rows
        m = min(len(ids), top)
        cp = _common_prefix(list(ids[:m]), self.park_tokens[:m])
        if cp <= self.park_base:
            return 0
        bk = max((b for b in self.park_anchors if b <= min(cp, max_len)), default=0)
        return bk if bk > self.best(c, max_len) else 0

    def unpark(self, eng, drafter) -> bool:
        """Make the parked prompt the resident one, and park the resident one in its place: the
        two sets of rows are swapped a block at a time, and so are the anchors. False, and
        nothing changed, if the buffers the rows came from are not the ones in service any more."""
        views = _kv_views(eng, drafter)
        if (self._park_views is None or len(views) != len(self._park_views)
                or any(a is not b or x != y
                       for (a, x), (b, y) in zip(views, self._park_views))):
            self.stats["park_refused"] += 1
            return False
        self._stash = None
        base, ra = self.park_base, self.park_rows
        cap = self._park_bufs[0].shape[views[0][1]]
        above = sorted(b for b in self.anchors if base < b <= min(self.valid, base + cap))
        # the prompt in place is parked in turn unless it is the parked one's own session
        rb = (above[-1] - base) if above and (self.owner is None
                                              or self.owner != self.park_owner) else 0
        if rb < self.chunk:
            rb = 0
        step = 4 * self.chunk
        for (t, ax), buf in zip(views, self._park_bufs):
            both = min(ra, rb)
            for j in range(0, both, step):
                w = min(step, both - j)
                tmp = t.narrow(ax, base + j, w).clone()
                t.narrow(ax, base + j, w).copy_(buf.narrow(ax, j, w))
                buf.narrow(ax, j, w).copy_(tmp)
            if rb > ra:
                buf.narrow(ax, ra, rb - ra).copy_(t.narrow(ax, base + ra, rb - ra))
            elif ra > rb:
                t.narrow(ax, base + rb, ra - rb).copy_(buf.narrow(ax, rb, ra - rb))
        back, back_tokens, back_owner = self.park_anchors, self.park_tokens, self.park_owner
        self.park_anchors = {}
        if rb:
            for b in above[-self.park_anchor_max:]:
                self.park_anchors[b] = self.anchors.pop(b)
                self._allocated -= 1
            self.park_tokens = list(self.tokens[:base + rb])
            self.park_rows, self.park_owner = rb, self.owner
            self.stats["park_swaps"] += 1
        else:
            self.park_tokens, self.park_rows, self.park_owner = [], 0, None
        for b in [b for b in self.anchors if b > base]:
            self._free.append(self.anchors.pop(b))
        for b, pair in back.items():
            self.anchors[b] = pair
            self._allocated += 1
        self.tokens = list(back_tokens)
        self.valid = base + ra
        self.owner = back_owner
        self._trim_pool()
        self.stats["park_hits"] += 1
        return True

    def drop_session(self, session: str | None) -> int:
        """A compaction of `session` started: its parked rows go. Returns the bytes."""
        if session is not None and self.park_owner == session and self.park_rows:
            self.stats["park_dropped"] += 1
            return self.shed_park()
        return 0

    # --- giving memory back (Budget's levels) ------------------------------------------------
    def shed_anchors(self, safe: bool = False) -> int:
        """The free anchor buffers, and unless `safe`, every anchor but the newest `tail` and the
        head lines: a divergence in the middle of a prompt then resumes lower, or from zero."""
        if not safe:
            keep = set(sorted(self.anchors)[-self.tail:]) if self.anchors else set()
            for b in [b for b in self.anchors if b not in keep and not self.head_line(b)]:
                self._free.append(self.anchors.pop(b))
                self.stats["anchors_evicted"] += 1
        n = len(self._free)
        self._free = []
        self._allocated -= n
        return n * self._anchor_bytes

    def shed_stash(self, safe: bool = False) -> int:
        """The guest stash's buffers. A stash still waiting to be put back is the resident
        prompt's rows: unless `safe`, the prompt is cut where the guest began and they go too."""
        freed = 0
        if self._stash is not None:
            if safe:
                return 0
            self._cut(self._stash["start"])
            self._stash = None
            # the anchors the cut freed are not kept as free buffers either
            freed += len(self._free) * self._anchor_bytes
            self._allocated -= len(self._free)
            self._free = []
        freed += sum(b.numel() * b.element_size() for b in (self._stash_bufs or []))
        self._stash_bufs, self._stash_rows = None, 0
        return freed

    def shed_park(self, safe: bool = False) -> int:
        freed = sum(b.numel() * b.element_size() for b in (self._park_bufs or []))
        freed += len(self.park_anchors) * self._anchor_bytes
        self.park_anchors = {}
        self._park_bufs, self._park_views = None, None
        self.park_tokens, self.park_base, self.park_rows, self.park_owner = [], 0, 0, None
        return freed

    def shed_pin(self, safe: bool = False) -> int:
        return self._drop_pin()

    def set_limits(self, *, budget: int | None = None, stash: int | None = None,
                   guest_max: int | None = None, pin: int | None = None,
                   park: int | None = None) -> None:
        """New caps from the next request on; what is over a lower cap goes now."""
        if budget is not None:
            self.budget = int(budget)
            if self._anchor_bytes:
                cap = max(self.tail + 1, self.budget // self._anchor_bytes)
                while len(self.anchors) > cap:
                    self._evict_one()
                extra = max(0, self._allocated - cap)
                take = min(extra, len(self._free))
                del self._free[len(self._free) - take:]
                self._allocated -= take
        if guest_max is not None:
            self.guest_max = int(guest_max)
        if stash is not None:
            self.stash_budget = int(stash)
            if self._stash is None and self._stash_bufs is not None and sum(
                    b.numel() * b.element_size() for b in self._stash_bufs) > self.stash_budget:
                self.shed_stash()
        if pin is not None:
            self.pin_budget = int(pin)
            if self.pin_bytes > self.pin_budget:
                self._drop_pin()
        if park is not None:
            self.park_budget = int(park)
            if self.park_bytes > self.park_budget:
                self.shed_park()

    # --- bookkeeping -----------------------------------------------------------------------
    def clear(self) -> None:
        for b in list(self.anchors):
            self._free.append(self.anchors.pop(b))
        self.tokens, self.valid = [], 0
        self._stash = None
        self.capturing = False

    @property
    def pin_bytes(self) -> int:
        pin = sum(b.numel() * b.element_size() for b in (self._pin_bufs or []))
        return pin + (len(self.pin_anchors) + len(self._pin_spare)) * self._anchor_bytes

    @property
    def park_bytes(self) -> int:
        park = sum(b.numel() * b.element_size() for b in (self._park_bufs or []))
        return park + len(self.park_anchors) * self._anchor_bytes

    @property
    def stash_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in (self._stash_bufs or []))

    @property
    def anchor_bytes(self) -> int:
        """The live pool: anchors in use and free ones, not the pin's or the park's."""
        return self._allocated * self._anchor_bytes

    @property
    def bytes(self) -> int:
        return self.anchor_bytes + self.stash_bytes + self.pin_bytes + self.park_bytes

    def report(self) -> dict:
        return {"valid": self.valid, "anchors": sorted(self.anchors),
                "anchor_bytes": self._anchor_bytes, "anchor_buffers": self._allocated,
                "budget": self.budget, "stash_budget": self.stash_budget,
                "stash_rows": self._stash_rows, "stash_pending": self._stash is not None,
                "bytes": self.bytes, "chunk": self.chunk, "tail": self.tail,
                "pin_budget": self.pin_budget, "pin_rows": self.pin_rows,
                "pin_bytes": self.pin_bytes,
                "pin_anchor_list": sorted(self.pin_anchors),
                "anchor_pool_bytes": self.anchor_bytes, "stash_bytes": self.stash_bytes,
                "guest_max": self.guest_max, "owner": _short(self.owner),
                "park_budget": self.park_budget, "park_bytes": self.park_bytes,
                "park_base": self.park_base, "park_rows": self.park_rows,
                "park_owner": _short(self.park_owner),
                "park_anchor_list": sorted(self.park_anchors),
                "last_switch": self.switches[-1] if self.switches else None,
                "switch_list": list(self.switches), **self.stats}


def _short(session: str | None) -> str | None:
    """A session key as the dashboard shows it: enough to tell two apart, not the whole id."""
    return None if session is None else (session if len(session) <= 12 else session[:12] + "…")


# --------------------------------------------------------------------------- the prefill itself
def prefill_chunk(prefix_on: bool, prefix_chunk: int, max_rows: int) -> int:
    """The chunk a server's prefill runs at.

    With the prefix cache on it is the cache's grid, as it always was. With it off the engine used
    to forward the whole prompt in ONE call, and on 2026-09-23 a 131,072-token prompt sent that way
    to a test server with a 262,144-token window is the prime suspect for the box running out of
    unified memory and wedging (NVRM NV_ERR_NO_MEMORY at 12:38-13:00): every activation of
    a forward is proportional to its rows. `max_rows` bounds it; 0 restores the single call.
    """
    return prefix_chunk if prefix_on else max(0, int(max_rows))


def prefill(eng, drafter, ids: list[int], device, *, store: StateStore | None = None,
            chunk: int = 0, conv_id: str | None = None, checkpoint: bool = False,
            resident: "ResidentPrefix | None" = None, on_chunk=None, info: dict | None = None,
            key_ids: list[int] | None = None, mem: "Budget | None" = None,
            compacting: bool = False):
    """Bring the engine to `len(ids)` and return the logits of the last position.

    Returns `(logits, reused, forwarded)`. `reused` is how many prompt tokens came out of the
    store or the resident prefix and `forwarded` how many were actually run through the 64
    layers -- the two numbers the TTFT of a warm request is made of.

    `chunk = 0` forwards the whole remainder in one call, which is what the engine did before this
    file existed and what a run with no prefix cache still does.

    `resident` (the live buffer as a cache, `ResidentPrefix`) is asked first; a store entry is
    used only when it is LONGER than the resident anchor, because a restore copies the KV and a
    resident resume does not. `on_chunk(done, total)` is called after every chunk -- the server's
    hook for noticing a client that left (it raises, and the prefill stops there with every
    anchor it took so far still good). `info["kind"]` says where the state came from.

    `key_ids`: what the caches compare instead of `ids` -- the prompt with each image's
    placeholder rows replaced by an id derived from the image's content (engine/vision.py). The
    placeholders alone are the same for every image; the rows the engine writes for them are not.
    None for text, where the key is the prompt itself.

    `conv_id` is the session key (the client's conversation id or opencode's `x-session-id`): the
    store files its entries under it and the resident prefix knows whose rows it holds, which is
    what parks a session another one replaces and brings it back (`ResidentPrefix.unpark`).
    `compacting`: this request is that session's compaction, whose old rows are not parked. `mem`
    (`Budget`): every snapshot the store would clone is asked for first, and the forwards'
    allocator peak is measured against the workspace estimate.
    """
    n = len(ids)
    if n == 0:
        raise ValueError("a prefill needs at least one token")
    t_in = time.perf_counter()
    keys = ids if key_ids is None else key_ids
    if len(keys) != n:
        raise ValueError(f"key_ids has {len(keys)} entries for a {n}-token prompt")
    hashes = _hash_memo(keys)
    res = resident if (resident is not None and chunk > 0 and resident_capable(drafter)) else None
    start, kind, c, b = 0, None, 0, 0
    if res is not None:
        res.settle_guest(eng)
        c = res.common(keys)
        b = res.best(c, n - 1)
        was_owner, was_valid, was_c = res.owner, res.valid, c
    bp = res.pin_offer(keys, c, n - 1) if res is not None else 0
    bk = res.park_offer(keys, c, n - 1) if res is not None else 0
    hit, via = None, None
    if store is not None:
        hit = store.find(keys, hashes, max_len=n - 1, min_len=max(b, bp, bk))
    if hit is None and (bp or bk):
        # the parked session or the pinned head beats the rows in place and every store entry
        if bk >= bp and res.unpark(eng, drafter):
            via = "park"
        elif bp and res.unpin(eng, drafter):
            via = "pin"
        if via is not None:
            c = res.common(keys)
            b = res.best(c, n - 1)
            res.stats[f"{via}_tokens_reused"] += b
        elif store is not None:
            hit = store.find(keys, hashes, max_len=n - 1, min_len=b)
    if hit is not None:
        L, entry = hit
        start, kind = L, entry.kind
    elif b > 0:
        start, kind = b, "resident"
    if res is not None:
        # decided before any row is written: a guest's stash must hold the rows as they are now
        res.begin(eng, drafter, keys, start, c, kind, session=conv_id, compacting=compacting)
    if hit is not None:
        restore(eng, hit[1].snap, drafter)
    elif kind == "resident":
        res.resume(eng, drafter, start)
    else:
        eng.reset()
    if res is not None and kind != "resident":
        res.stats["misses"] += 1
    if info is not None:
        info["kind"] = kind
        # what the live view needs to turn the chunk hook's `done` into progress and a
        # rate -- once per prefill, before the loop; the loop itself gains nothing
        info["start"], info["chunk"], info["t0"] = start, int(chunk), time.perf_counter()
    logits = None
    i = start
    ckpt = (store.boundaries(eng, drafter, start, n, chunk)
            if (store is not None and checkpoint and chunk > 0) else None)
    if mem is not None:
        mem.mark()
    while i < n:
        if chunk > 0:
            # The grid is anchored at absolute position 0, not at the resume point: a resume that
            # lands off the grid -- which is what a session boundary is -- takes a SHORT first
            # step and is back on the grid afterwards. Anchoring it at `start` instead would put
            # every checkpoint of that prefill off the grid too, and the next request resuming
            # from one of those would inherit the misalignment for ever. This way the only forward
            # that differs from a cold run's is the first one after a session resume.
            t = min(chunk - (i % chunk), n - i)
        else:
            t = n - i
        blk = torch.tensor(ids[i:i + t], device=device, dtype=torch.long)
        logits = eng.forward(blk, start=i, last_only=(i + t >= n))
        if drafter is not None and hasattr(drafter, "sync"):
            drafter.sync(ids[i:i + t], eng.hidden_post_norm[0], i)
        i += t
        if res is not None:
            res.advanced(i)
            if i < n and i % chunk == 0:
                res.anchor(eng, i)
        if (store is not None and checkpoint and i < n and chunk > 0 and i % chunk == 0
                and (ckpt is None or i in ckpt)):
            # room is made BEFORE the clone (OPS-47): evicting after it held store + snapshot
            est = _snapshot_estimate(eng, i, drafter)
            snap = None
            if not store.reserve(est):
                store.stats["skipped_big"] = store.stats.get("skipped_big", 0) + 1
            elif mem is not None and not mem.admit("store", est):
                store.stats["skipped_mem"] = store.stats.get("skipped_mem", 0) + 1
            else:
                snap = capture(eng, drafter, max_bytes=store.max_entry)
                if snap is not None:
                    store.put(keys, snap, conv_id, hashes)
                else:
                    store.stats["skipped_big"] = store.stats.get("skipped_big", 0) + 1
        if on_chunk is not None and i < n:
            on_chunk(i, n)
    if mem is not None:
        mem.prefilled(n)
    if store is not None:
        store.stats["tokens_reused"] += start
        store.stats["tokens_forwarded"] += n - start
    if res is not None and was_valid and res._stash is None and (
            (conv_id is not None and conv_id != was_owner)
            or (conv_id is None and was_c < was_valid)):
        # another session's rows were in place (or, without session ids, the prompt parted from
        # them): what this switch cost, for the dashboard
        res.stats["switches"] += 1
        res.switches.append({"ts": round(time.time(), 3), "from": _short(was_owner),
                             "to": _short(conv_id), "tokens": n, "reused": start,
                             "reread": n - start, "via": via or kind or "none",
                             "ms": round((time.perf_counter() - t_in) * 1e3, 1)})
    return logits, start, n - start


# --------------------------------------------------------------------------- response cache
class ResponseCache:
    """(prompt, params) -> the token ids the engine wrote. Off unless asked for.

    Greedy decoding is a function, so this is memoisation and not a heuristic. It is nevertheless
    opt-in, because a server that answers from memory answers a changed MODEL from memory too, and
    because the one thing a benchmark must never measure is a dictionary.
    """

    def __init__(self, budget_bytes: int = 64 << 20, ttl_s: float = 3600.0):
        self.budget = int(budget_bytes)
        self.ttl = float(ttl_s)
        self._d: OrderedDict = OrderedDict()
        self.bytes = 0
        self.stats = {"hits": 0, "misses": 0, "expired": 0, "evictions": 0, "puts": 0}
        self.lock = threading.Lock()

    @staticmethod
    def key(ids, **params) -> tuple:
        h = prefix_hashes(ids)[-1]
        return (len(ids), h, tuple(sorted((str(k), repr(v)) for k, v in params.items())))

    def get(self, key):
        with self.lock:
            rec = self._d.get(key)
            if rec is None:
                self.stats["misses"] += 1
                return None
            ids, tokens, born = rec
            if self.ttl > 0 and time.time() - born > self.ttl:
                del self._d[key]
                self.bytes -= 8 * len(ids) + 8 * len(tokens)
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            self._d.move_to_end(key)
            self.stats["hits"] += 1
            return ids

    def put(self, key, ids, prompt_ids) -> None:
        if not ids:
            return
        cost = 8 * len(ids) + 8 * len(prompt_ids)
        with self.lock:
            if key in self._d:
                old = self._d.pop(key)
                self.bytes -= 8 * len(old[0]) + 8 * len(old[1])
            self._d[key] = (list(ids), list(prompt_ids), time.time())
            self.bytes += cost
            self.stats["puts"] += 1
            while self.bytes > self.budget and self._d:
                _, old = self._d.popitem(last=False)
                self.bytes -= 8 * len(old[0]) + 8 * len(old[1])
                self.stats["evictions"] += 1

    def report(self) -> dict:
        with self.lock:
            return {"entries": len(self._d), "bytes": self.bytes, "budget": self.budget,
                    "ttl_s": self.ttl, **self.stats}

    def clear(self) -> None:
        with self.lock:
            self._d.clear()
            self.bytes = 0


# --------------------------------------------------------------------------- suffix stores
DOC_SEP = 1 << 30


def _common_prefix(a: list, b: list, step: int = 4096) -> int:
    """How many leading elements two lists share: slice compares a block at a time, then the
    block that differs element by element, so the cost is one pass over the shared part."""
    m = min(len(a), len(b))
    i = 0
    while i < m:
        j = min(m, i + step)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return m


class SuffixStoreSet:
    """Several `CorpusSuffixStore`-shaped stores behind one `lookup`.

    A position is an opaque handle to the drafter -- it only ever hands one back to
    `continuation` -- so a store index is packed into its high bits and unpacked on the way out.
    The longest match wins; where two stores match at the same order, both vote, which is what the
    drafter's counting wants.
    """

    STRIDE = 1 << 40

    def __init__(self, stores):
        self.stores = [s for s in stores if s is not None]
        self.max_order = max((getattr(s, "max_order", 8) for s in self.stores), default=8)

    def __bool__(self) -> bool:
        return bool(self.stores)

    def lookup(self, context, min_order: int, max_samples: int = 64):
        best, pos = 0, []
        for i, s in enumerate(self.stores):
            n, p = s.lookup(context, min_order, max_samples=max_samples)
            if not p or n < best:
                continue
            if n > best:
                best, pos = n, []
            pos.extend(i * self.STRIDE + q for q in p)
        return best, pos

    def continuation(self, pos: int, k: int):
        i, p = divmod(pos, self.STRIDE)
        return self.stores[i].continuation(p, k)


class PersistentSuffixStore:
    """An append-only token log of what this engine has read and written, with an index over it.

    The lookup drafter's local index dies with the request and its corpus store is a file somebody
    built by hand. This is the third thing: what the engine itself saw, kept across restarts, so
    that the second time it is asked about a subject it drafts the phrasing it used the first time
    at 0.2 ms a lookup instead of 4 ms of prediction head.

    On disk: `tokens.bin`, int32 little-endian, documents separated by `DOC_SEP`; `meta.json`. No
    text, ever -- the same rule `tools/build_corpus.py` states, for the same reason.

    The suffix array is rebuilt in a background thread when enough has accumulated, and the drafter
    is handed the new store by reference assignment. Until then the new tokens are simply not
    findable, which is the right failure: a drafter that cannot find something declines.
    """

    def __init__(self, path: str, *, max_tokens: int = 48_000_000,
                 rebuild_every: int = 100_000, max_order: int = 8, readonly: bool = False,
                 tokenizer_sha: str | None = None):
        self.path = os.path.expanduser(path)
        self.tokenizer_sha = tokenizer_sha        # recorded on create, checked on open
        # A store that is read but never written: the instrument for measuring a fixed benchmark
        # against real traffic's store without the benchmark writing itself into it.
        self.readonly = bool(readonly)
        self.max_tokens = int(max_tokens)
        self.rebuild_every = int(rebuild_every)
        self.max_order = int(max_order)
        self.store = None                  # a CorpusSuffixStore, swapped in atomically
        self.pending = 0
        self.n_tokens = 0
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.stats = {"appended": 0, "rebuilds": 0, "trims": 0, "rebuild_ms": 0.0,
                      "rebuild_ms_total": 0.0, "rebuild_last_ts": 0.0, "turns": 0,
                      "turn_tokens": 0, "turn_shared": 0}
        # the prompts of the last few turns, to append only what a new turn adds to one of them
        self._recent: list[list[int]] = []
        self.recent_max = 4
        # bumped by every lookup that matched: the metrics wrapper reads it once a block to count
        # the blocks the store had continuations for, never once a lookup
        self.matched = 0

    # --- disk ------------------------------------------------------------------------------
    @property
    def _bin(self) -> str:
        return os.path.join(self.path, "tokens.bin")

    def open(self) -> "PersistentSuffixStore":
        """Create the directory if needed and build the index over whatever is already there."""
        os.makedirs(self.path, exist_ok=True)
        meta = os.path.join(self.path, "meta.json")
        if not os.path.exists(meta):
            m = {"max_order": self.max_order, "doc_sep": DOC_SEP,
                 "what": "token ids written and read by this engine; no text"}
            if self.tokenizer_sha:
                m["tokenizer_sha256"] = self.tokenizer_sha
            with open(meta, "w") as f:
                json.dump(m, f)
        else:
            from engine.tokfp import KEY, check_store
            with open(meta) as f:
                m = json.load(f)
            check_store(m, self.tokenizer_sha, self.path)
            if self.tokenizer_sha and KEY not in m and not self.readonly:
                # written before the fingerprint existed, by this engine and this model: from now
                # on it is recorded, so another tokenizer's store can be told apart
                m[KEY] = self.tokenizer_sha
                with open(meta + ".tmp", "w") as f:
                    json.dump(m, f)
                os.replace(meta + ".tmp", meta)
        os.chmod(self.path, 0o700)
        self.rebuild(background=False)
        return self

    def append(self, tokens) -> None:
        """Add one document. Returns immediately; the index catches up on its own."""
        if not tokens or self.readonly:
            return
        buf = struct.pack(f"<{len(tokens) + 1}i", *(int(t) for t in tokens), DOC_SEP)
        with self.lock:
            with open(self._bin, "ab") as f:
                f.write(buf)
            self.pending += len(tokens) + 1
            self.stats["appended"] += len(tokens)
            due = self.pending >= self.rebuild_every
        if due:
            self.rebuild(background=True)

    def append_turn(self, prompt, out) -> int:
        """Add one request: its prompt and its output, minus what an earlier turn already added.

        An agent client sends one conversation that only grows, and appending the whole prompt
        after every turn wrote about 200 copies of it: 397.5M tokens appended in the 2026-09-30
        night for about 4.8M new ones, a suffix-array rebuild after every turn (607, 17-28 s
        each), and 16 trims that dropped every other session's content. A prompt that starts with
        the prompt of one of the last `recent_max` turns appends only from where they part, with
        the `max_order` tokens before that point in front, so every n-gram across the join is
        still in the store. Returns the number of tokens appended."""
        prompt = list(prompt)
        shared, which = 0, -1
        for j, doc in enumerate(self._recent):
            c = _common_prefix(prompt, doc)
            if c > shared:
                shared, which = c, j
        start = max(0, shared - self.max_order) if shared else 0
        doc = prompt[start:] + list(out)
        self.append(doc)
        if which >= 0:
            self._recent.pop(which)
        self._recent.append(prompt)
        del self._recent[:-self.recent_max]
        self.stats["turns"] += 1
        self.stats["turn_tokens"] += len(prompt) + len(out)
        self.stats["turn_shared"] += shared
        return len(doc) if not self.readonly else 0

    def _read_tokens(self):
        import numpy as np
        if not os.path.exists(self._bin):
            return np.zeros(0, dtype=np.int64)
        a = np.fromfile(self._bin, dtype="<i4").astype(np.int64)
        if a.shape[0] > self.max_tokens:
            # The cap forgets the oldest half rather than the oldest token, so trimming is a rare
            # rewrite and not one per request. The cut is moved forward to the next document
            # boundary, so no continuation is left starting in the middle of a document.
            cut = a.shape[0] - self.max_tokens // 2
            nxt = np.nonzero(a[cut:] >= DOC_SEP)[0]
            cut = cut + int(nxt[0]) + 1 if nxt.size else cut
            a = a[cut:]
            a.astype("<i4").tofile(self._bin)
            self.stats["trims"] += 1
        return a

    def rebuild(self, background: bool = True) -> None:
        if background:
            with self.lock:
                if self._thread is not None and self._thread.is_alive():
                    return
                self._thread = threading.Thread(target=self._rebuild_now, daemon=True,
                                                name="suffix-store-rebuild")
                self._thread.start()
            return
        self._rebuild_now()

    def _rebuild_now(self) -> None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from engine.drafters.ngram import CorpusSuffixStore
        from tools.build_corpus import build_suffix_array
        t0 = time.perf_counter()
        toks = self._read_tokens()
        if toks.shape[0] < self.max_order:
            return
        sa = build_suffix_array(toks, max_order=self.max_order)
        store = CorpusSuffixStore(toks, sa, max_order=self.max_order,
                                  meta={"doc_sep": DOC_SEP, "max_order": self.max_order})
        ms = (time.perf_counter() - t0) * 1e3
        with self.lock:
            self.store = store
            self.n_tokens = int(toks.shape[0])
            self.pending = 0
            self.stats["rebuilds"] += 1
            self.stats["rebuild_ms"] = ms
            self.stats["rebuild_ms_total"] += ms
            self.stats["rebuild_last_ts"] = time.time()

    # --- the store interface the drafter speaks ---------------------------------------------
    def lookup(self, context, min_order: int, max_samples: int = 64):
        s = self.store
        if s is None:
            return 0, []
        n, pos = s.lookup(context, min_order, max_samples=max_samples)
        if pos:
            self.matched += 1
        return n, pos

    def continuation(self, pos: int, k: int):
        s = self.store
        return s.continuation(pos, k) if s is not None else []

    def report(self) -> dict:
        with self.lock:
            return {"path": self.path, "tokens": self.n_tokens, "pending": self.pending,
                    "max_tokens": self.max_tokens, "indexed": self.store is not None,
                    "matched": self.matched, **self.stats}
