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
from collections import OrderedDict

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
    the latch releases one (ENG-104), so a constant for one arm undercounted the served stack by
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
    eng._pend = None                  # a commit pending on the state just overwritten (SPD-37)
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
        # ("prefix"). One store holds both and a lookup cannot tell them apart otherwise (SRV-9).
        self.kind = kind


class StateStore:
    """Token-prefix -> engine state, under a byte budget, least-recently-used out first.

    Keyed by `(length, hash)` and verified by the tokens themselves. `lengths` is the set of
    boundaries anything has been stored at, which is what makes a lookup cheap: the candidates for
    a new prompt are the stored lengths below it, and each one is one dictionary probe.
    """

    def __init__(self, budget_bytes: int, chunk: int = 256, max_entry_bytes: int = 0):
        self.budget = int(budget_bytes)
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

    def _evict_locked(self) -> None:
        while self.bytes > self.budget and self._d:
            key, entry = self._d.popitem(last=False)
            self.bytes -= entry.snap.nbytes
            n = key[0]
            self._lengths[n] -= 1
            if not self._lengths[n]:
                del self._lengths[n]
            self.stats["evictions"] += 1

    # --- reading ---------------------------------------------------------------------------
    def find(self, tokens, hashes: list[int], max_len: int):
        """The longest stored prefix of `tokens` no longer than `max_len`, or None.

        `max_len` is normally `len(tokens) - 1`: a request needs at least one token to forward,
        because the logits it answers with are the ones that token's own forward produces and no
        snapshot holds them.
        """
        with self.lock:
            cands = sorted((L for L in self._lengths if L <= max_len), reverse=True)
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


# --------------------------------------------------------------------------- the prefill itself
def prefill_chunk(prefix_on: bool, prefix_chunk: int, max_rows: int) -> int:
    """The chunk a server's prefill runs at.

    With the prefix cache on it is the cache's grid, as it always was. With it off the engine used
    to forward the whole prompt in ONE call, and on 2026-09-23 a 131,072-token prompt sent that way
    to a test server with a 262,144-token window is the prime suspect for the box running out of
    unified memory and wedging (NVRM NV_ERR_NO_MEMORY at 12:38-13:00, SPD-18): every activation of
    a forward is proportional to its rows. `max_rows` bounds it; 0 restores the single call.
    """
    return prefix_chunk if prefix_on else max(0, int(max_rows))


def prefill(eng, drafter, ids: list[int], device, *, store: StateStore | None = None,
            chunk: int = 0, conv_id: str | None = None, checkpoint: bool = False):
    """Bring the engine to `len(ids)` and return the logits of the last position.

    Returns `(logits, reused, forwarded)`. `reused` is how many prompt tokens came out of the
    store and `forwarded` how many were actually run through the 64 layers -- the two numbers the
    TTFT of a warm request is made of.

    `chunk = 0` forwards the whole remainder in one call, which is what the engine did before this
    file existed and what a run with no prefix cache still does.
    """
    n = len(ids)
    if n == 0:
        raise ValueError("a prefill needs at least one token")
    hashes = prefix_hashes(ids)
    start = 0
    if store is not None:
        hit = store.find(ids, hashes, max_len=n - 1)
        if hit is not None:
            L, entry = hit
            restore(eng, entry.snap, drafter)
            start = L
    if start == 0:
        eng.reset()
    logits = None
    i = start
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
        if store is not None and checkpoint and i < n and chunk > 0 and i % chunk == 0:
            snap = capture(eng, drafter, max_bytes=store.max_entry)
            if snap is not None:
                store.put(ids, snap, conv_id, hashes)
            else:
                store.stats["skipped_big"] = store.stats.get("skipped_big", 0) + 1
    if store is not None:
        store.stats["tokens_reused"] += start
        store.stats["tokens_forwarded"] += n - start
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
                 rebuild_every: int = 100_000, max_order: int = 8, readonly: bool = False):
        self.path = os.path.expanduser(path)
        # A store that is read but never written: the instrument for measuring a fixed benchmark
        # against real traffic's store without the benchmark writing itself into it (SPD-17).
        self.readonly = bool(readonly)
        self.max_tokens = int(max_tokens)
        self.rebuild_every = int(rebuild_every)
        self.max_order = int(max_order)
        self.store = None                  # a CorpusSuffixStore, swapped in atomically
        self.pending = 0
        self.n_tokens = 0
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.stats = {"appended": 0, "rebuilds": 0, "trims": 0, "rebuild_ms": 0.0}
        # bumped by every lookup that matched: the metrics wrapper reads it once a block to count
        # the blocks the store had continuations for (SRV-9), never once a lookup
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
            with open(meta, "w") as f:
                json.dump({"max_order": self.max_order, "doc_sep": DOC_SEP,
                           "what": "token ids written and read by this engine; no text"}, f)
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
        with self.lock:
            self.store = store
            self.n_tokens = int(toks.shape[0])
            self.pending = 0
            self.stats["rebuilds"] += 1
            self.stats["rebuild_ms"] = (time.perf_counter() - t0) * 1e3

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
