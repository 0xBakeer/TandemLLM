"""Kolibri-1 behind the engine's OpenAI server.

    python server/app.py --kolibri --port 8001 --served-model Kolibri-1 \\
        --kolibri-set ./Kolibri-1-NVFP4 --kolibri-tokenizer ./Kolibri-1-NVFP4 --max-len 65536

`load(app, a)` is called from `server/app.py`'s `main()` in place of the Qwen loader, the way
`server/fake_engine.load` is: the handler, streaming, usage, ledger, metrics, logs and dashboard
API are the server's own code. What is replaced:

* `app.generate_stream`: the decode loop for `engine.kolibri.model.KolibriEngine` (one token a
  forward; no drafter yet). Greedy, sampling, penalties, logit bias, log-probabilities, stop
  patterns and the thinking budget go through the same objects the Qwen loop uses.
* `app.build_prompt`: the server's own, through Kolibri's tokenizer and template
  (`engine/kolibri/chat.py`), returning `in_think = "model"` when the template leaves the model to
  open `<think>` itself (server/stream.py `MODEL_OPENS`).
* The prefix cache: `RingPrefix` below, the Kolibri form of the resident prefix, with its guest
  stash (a short unrelated request between two turns does not cost the conversation its rows).

THE PREFIX CACHE. The KV of the last request stays in the engine. A new request shares `c` tokens
with what the cache holds. The full layers keep every row, so any cut is exact for them; the
sliding layers keep a 640-row ring, which can be cut back only 128 rows (`KolibriKV.back`). So:

* `c` within 128 rows of the end: truncate to `c` and prefill the rest;
* otherwise: restore the ring from the largest ANCHOR at or below `c` and prefill from there.

An anchor is the ring (52 MB) cloned at a prefill boundary: every multiple of `chunk` from 0, and
the end of every prompt (where the previous turn's answer begins, which is where the next turn's
prompt leaves the cache: the template drops the answer's reasoning from history). The prefill runs
on the same grid of `chunk` from 0 whether it starts cold or warm, so a resumed prefill splits the
rest exactly as a cold one would. A resume at `s` drops every anchor above `s` (the full-layer rows
above `s` are about to be rewritten). Anchors are kept up to `anchor_bytes`; eviction drops the
anchor whose removal leaves the smallest gap and never the newest four.

The rows a decode step wrote are reused as they are (they hold the answer the client is sending
back); against a cold re-read that is a different arithmetic order for those rows, the same as
the Qwen engine's session cache, which is held to the same argmax rather than to equal bits.
"""

from __future__ import annotations

import os
import time
import types

import torch

from engine.kolibri.spec import SpecRows


# ------------------------------------------------------------------------------ the prefix cache
class _Parked:
    """A conversation set aside: every full-layer row it had, its ring at its end, and the anchors
    (and end logits) above the point where the request that displaced it diverged. The anchors at or
    below that point stay in the live table, shared: they belong to whichever conversation holds
    the same tokens up to them."""

    __slots__ = ("tokens", "length", "prompt_end", "rows", "ring", "anchors", "end_logits",
                 "nbytes", "t")

    def __init__(self, tokens, length, prompt_end, rows, ring, anchors, end_logits, nbytes):
        self.tokens, self.length, self.prompt_end = tokens, length, prompt_end
        self.rows, self.ring, self.anchors, self.end_logits = rows, ring, anchors, end_logits
        self.nbytes = nbytes
        self.t = time.monotonic()


class RingPrefix:
    def __init__(self, chunk: int = 2048, anchor_bytes: int = 2 << 30, tail: int = 4,
                 stash_bytes: int = 6 << 30, park_min: int = 64, park_anchors: int = 8,
                 guest_min: int | None = None):
        self.chunk = int(chunk)
        self.stash_bytes = int(stash_bytes)
        self.park_min = int(park_min if guest_min is None else guest_min)
        self.park_anchors = max(1, int(park_anchors))
        self.parked: list[_Parked] = []          # oldest first
        self.prompt_end = 0                      # where the live conversation's last prompt ended
        self.budget = int(anchor_bytes)
        self.tail = max(1, int(tail))
        self.tokens: list[int] = []        # what rows [0, kv.length) hold
        self.anchors: dict[int, tuple] = {}
        # the last row's logits at a prompt's end anchor: the same prompt again needs no forward
        self.end_logits: dict[int, torch.Tensor] = {}
        self._free: list[tuple] = []
        self._each = 0
        self.stats = {"requests": 0, "cold": 0, "truncated": 0, "anchor_hits": 0,
                      "tokens_reused": 0, "tokens_forwarded": 0, "anchors_taken": 0,
                      "anchors_evicted": 0, "parked": 0, "unparked": 0, "park_evicted": 0,
                      "park_skipped": 0, "park_ms": 0.0, "unpark_ms": 0.0}

    @staticmethod
    def common(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        if a[:n] == b[:n]:
            return n
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid
        return lo

    def resume(self, eng, ids: list[int]) -> tuple[int, str | None]:
        """Bring the engine to the longest reusable prefix of `ids` (at most len - 1, so the last
        prompt token is always forwarded). Returns (start, kind)."""
        kv = eng.kv
        self.stats["requests"] += 1
        n = len(ids)
        full = self.common(ids, self.tokens[:kv.length])
        # a parked conversation that this request continues further than the live one: swap it in
        best, bc = None, full
        for e in self.parked:
            ce = self.common(ids, e.tokens)
            if ce > bc:
                best, bc = e, ce
        if best is not None:
            self._unpark(eng, best)
            full = self.common(ids, self.tokens[:kv.length])
        if full == n and n in self.anchors and n in self.end_logits:
            # the same prompt as one before it (a retry, or n > 1 choices): its end anchor and the
            # logits of its last row, no forward at all
            kv.restore(self.anchors[n])
            self._drop_above(n)
            self.tokens = list(ids)
            self.prompt_end = n
            self.stats["anchor_hits"] += 1
            self.stats["tokens_reused"] += n
            return n, "anchor"
        c = min(full, n - 1)
        start, kind = 0, None
        if c > 0 and c >= kv.length - kv.back:
            eng.truncate(c)
            start, kind = c, "truncate"
            self.stats["truncated"] += 1
        else:
            b = max((p for p in self.anchors if p <= c), default=0)
            if self._diverges(kv, c):
                self._park(eng, b)
            if b > 0:
                kv.restore(self.anchors[b])
                start, kind = b, "anchor"
                self.stats["anchor_hits"] += 1
            else:
                eng.reset()
                self.stats["cold"] += 1
        self._drop_above(start)
        self.tokens = list(ids[:start])
        self.prompt_end = n
        self.stats["tokens_reused"] += start
        return start, kind

    # --- parked conversations ------------------------------------------------------------------
    # A request that leaves the live conversation inside its last prompt (an unrelated chat, Open
    # WebUI's title and tag calls, opencode's title call, a second opencode session) would
    # overwrite the conversation's rows and drop its anchors. Instead the conversation is PARKED
    # first: its full-layer rows [0, L) are copied aside (20 KB a row in BF16), with its ring and its
    # anchors above the point of divergence. The request then runs as the live conversation and
    # keeps its own reuse on its next turns. Every request is matched against the live
    # conversation and every parked one; when a parked one shares more, it is swapped back in (the
    # live one is parked in turn), so either side of a switch resumes where it stopped. A turn of
    # the same conversation (the request keeps all of the previous prompt; only the answer
    # differs) is not a divergence. Parked conversations share `stash_bytes`, oldest evicted first;
    # one that cannot fit alone is not parked.
    def _diverges(self, kv, c: int) -> bool:
        L = kv.length
        return (self.stash_bytes > 0 and L >= self.park_min and c < L - kv.back
                and c < min(self.prompt_end, L))

    def _park_cost(self, kv, L: int, n_anchors: int) -> int:
        ring = kv.ring.nbytes
        return L * kv.bytes_per_token + ring * (1 + n_anchors)

    def _park(self, eng, keep: int) -> bool:
        """Set the live conversation aside; anchors above `keep` go with it."""
        kv = eng.kv
        L = kv.length
        if L <= 0:
            return False
        own = sorted(p for p in self.anchors if keep < p <= L)
        drop = own[:-self.park_anchors]
        own = own[-self.park_anchors:]
        cost = self._park_cost(kv, L, len(own))
        if cost > self.stash_bytes:
            self.stats["park_skipped"] += 1
            return False
        while self.parked and sum(e.nbytes for e in self.parked) + cost > self.stash_bytes:
            self._evict_parked()
        t0 = time.perf_counter()
        views = [kv.k, kv.v] + ([kv.ks, kv.vs] if kv.fp8 else [])
        rows = [t.narrow(3, 0, L).clone() for t in views]
        for p in drop:
            self._recycle(self.anchors.pop(p))
            self.end_logits.pop(p, None)
        anchors = {p: self.anchors.pop(p) for p in own}
        logits = {p: self.end_logits.pop(p) for p in own if p in self.end_logits}
        self.parked.append(_Parked(list(self.tokens[:L]), L, min(self.prompt_end, L), rows,
                                   kv.snapshot(), anchors, logits, cost))
        self.stats["parked"] += 1
        self.stats["park_ms"] += (time.perf_counter() - t0) * 1e3
        return True

    def _evict_parked(self) -> None:
        e = self.parked.pop(0)
        for snap in list(e.anchors.values()) + [e.ring]:
            self._recycle(snap)
        self.stats["park_evicted"] += 1

    def _recycle(self, snap: tuple) -> None:
        """A ring snapshot no longer needed: its buffers serve the next anchor (a few are kept;
        the rest go back to the allocator)."""
        if len(self._free) < 8:
            self._free.append(snap)

    def _unpark(self, eng, e: _Parked) -> None:
        """Swap parked conversation `e` in; the live one is parked when it is worth keeping."""
        kv = eng.kv
        t0 = time.perf_counter()
        self.parked.remove(e)
        shared = self.common(self.tokens[:kv.length], e.tokens)
        if not (kv.length >= self.park_min and kv.length > shared and self._park(eng, shared)):
            self._drop_above(shared)
        # anchors at or below `shared` are the live table's and hold e's tokens too
        views = [kv.k, kv.v] + ([kv.ks, kv.vs] if kv.fp8 else [])
        for t, r in zip(views, e.rows):
            t.narrow(3, 0, e.length).copy_(r)
        kv.restore(e.ring)
        self._recycle(e.ring)
        for p, snap in e.anchors.items():
            if p in self.anchors:
                self._recycle(snap)
            else:
                self.anchors[p] = snap
        for p, lg in e.end_logits.items():
            self.end_logits.setdefault(p, lg)
        self.tokens = e.tokens
        self.prompt_end = e.prompt_end
        self.stats["unparked"] += 1
        self.stats["unpark_ms"] += (time.perf_counter() - t0) * 1e3

    def _drop_above(self, s: int) -> None:
        """Rows above `s` are about to be rewritten: anchors above it no longer match them."""
        for p in [p for p in self.anchors if p > s]:
            self._free.append(self.anchors.pop(p))
            self.end_logits.pop(p, None)

    def anchor(self, eng) -> None:
        """The ring at the engine's current length, kept."""
        kv = eng.kv
        n = kv.length
        if n <= 0 or n in self.anchors:
            return
        if not self._each:
            self._each = kv.ring.nbytes
        cap = max(self.tail + 1, self.budget // max(1, self._each))
        while len(self.anchors) >= cap:
            self._evict_one()
        if self._free:
            snap = kv.snapshot_into(self._free.pop())
        else:
            snap = kv.snapshot()
        self.anchors[n] = snap
        self.stats["anchors_taken"] += 1

    def _evict_one(self) -> None:
        bs = sorted(self.anchors)
        body = bs[:-self.tail] if len(bs) > self.tail else bs[:1]
        best, victim = None, body[0]
        for j, b in enumerate(body):
            lo = bs[j - 1] if j > 0 else 0
            hi = bs[j + 1] if j + 1 < len(bs) else b
            if best is None or hi - lo < best:
                best, victim = hi - lo, b
        self._free.append(self.anchors.pop(victim))
        self.end_logits.pop(victim, None)
        self.stats["anchors_evicted"] += 1

    def prefill(self, eng, ids: list[int], start: int, on_chunk=None) -> torch.Tensor:
        """Forward ids[start:] on the grid of `chunk` from 0, anchoring at every grid boundary and
        at the end; returns the last row's fp32 logits [V]."""
        n = len(ids)
        i = start
        logits = None
        if start == n:
            # a copy: penalties and the logit bias edit the row they are given in place
            logits = self.end_logits[n].clone()
        while i < n:
            t = min(self.chunk - (i % self.chunk), n - i)
            logits = eng.prefill(ids[i:i + t], start=i)
            i += t
            self.tokens = list(ids[:i])
            if i % self.chunk == 0 or i == n:
                self.anchor(eng)
                if i == n and n in self.anchors:
                    self.end_logits[n] = logits.detach().clone()
            if on_chunk is not None and i < n:
                # the host runs ahead of the GPU: without a sync every chunk's callback fires
                # within milliseconds and the prefill watch (heartbeat comments, client-gone
                # check) sees no progress for the whole prefill
                if logits.is_cuda:
                    torch.cuda.synchronize(logits.device)
                on_chunk(i, n)
        self.stats["tokens_forwarded"] += n - start
        return logits

    def committed(self, ctx: list[int], length: int) -> None:
        """After a generation: rows [0, length) hold ctx[:length]."""
        self.tokens = list(ctx[:length])

    def report(self) -> dict:
        return {"chunk": self.chunk, "anchors": sorted(self.anchors),
                "anchor_bytes": self._each, "budget": self.budget, "valid": len(self.tokens),
                "parked_conversations": [{"tokens": e.length, "bytes": e.nbytes,
                                          "anchors": sorted(e.anchors),
                                          "idle_s": round(time.monotonic() - e.t, 1)}
                                         for e in list(self.parked)],
                "park_budget": self.stash_bytes,
                **{k: (round(v, 1) if isinstance(v, float) else v)
                   for k, v in self.stats.items()}}


# ------------------------------------------------------------------------------ across a restart
# A planned stop (SIGTERM) writes the live conversation to disk once the work in
# flight is done: its full-layer rows [0, L), the ring at L, the newest anchors (the next turn
# resumes at the last prompt's end anchor) with their end logits, the tokens. The next start reads
# it back after the warm-up, so the client's next turn reuses it as if the server had not stopped.
# A checksum of every tensor is taken on the GPU before the write and again after the read: equal
# sums mean the state is back bit for bit. The file names the set, the tokenizer and the KV layout
# it was written for; any other engine ignores it. It is removed once read (a crash later must not
# bring back an older state silently).
SESSION_VERSION = 1
PERSIST_PARKED_ANCHORS = 2        # a parked conversation's newest anchors written at a drain
PERSIST_BYTES = 4 << 30           # at most this much written at a drain (the live one first)


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
    except OSError:
        return ""
    return h.hexdigest()[:16]


def session_fingerprint(eng, set_dir: str, tok_dir: str) -> dict:
    kv = eng.kv
    return {"version": SESSION_VERSION, "set": os.path.realpath(set_dir),
            "manifest": _sha256(os.path.join(set_dir, "manifest.json")),
            "tokenizer": _sha256(os.path.join(tok_dir, "tokenizer.json")),
            "kv_dtype": str(kv.k.dtype), "fp8_kv": bool(kv.fp8), "ring": int(kv.ring.R),
            "full_layers": int(kv.k.shape[0]), "heads": int(kv.k.shape[2]),
            "head_dim": int(kv.k.shape[-1]) if kv.k.dim() == 5 else 0}


def tensor_sum(t: torch.Tensor) -> list[int]:
    """Two integer checksums of a tensor's bytes, on its own device (integer sums do not depend
    on the reduction order, so the same bytes give the same pair on any run)."""
    t = t.contiguous()
    nb = t.numel() * t.element_size()
    b = t.view(torch.uint8).reshape(-1)
    if nb % 4 == 0:
        b = b.view(torch.int32)
    s1 = s2 = 0
    step = 1 << 24
    for i in range(0, b.numel(), step):
        x = b[i:i + step].to(torch.int64)
        w = torch.arange(i, i + x.numel(), device=x.device, dtype=torch.int64) % 65521 + 1
        s1 += int(x.sum())
        s2 += int((x * w).sum())
    return [s1 & ((1 << 64) - 1), s2 & ((1 << 64) - 1)]


def _live_tensors(pc: "RingPrefix", eng, keep_anchors: int = 4) -> list[tuple[str, torch.Tensor]]:
    kv = eng.kv
    L = kv.length
    out = [("k", kv.k.narrow(3, 0, L)), ("v", kv.v.narrow(3, 0, L))]
    if kv.fp8:
        out += [("ks", kv.ks.narrow(3, 0, L)), ("vs", kv.vs.narrow(3, 0, L))]
    out += [(f"ring.{i}", t) for i, t in enumerate(kv.ring.tensors())]
    for p in sorted(p for p in pc.anchors if p <= L)[-keep_anchors:]:
        out += [(f"anchor.{p}.{i}", t) for i, t in enumerate(pc.anchors[p][1:])]
        if p in pc.end_logits:
            out.append((f"logits.{p}", pc.end_logits[p]))
    return out


def _parked_tensors(e: "_Parked", keep_anchors: int | None = None) -> list[tuple[str, torch.Tensor]]:
    names = ["k", "v", "ks", "vs"][:len(e.rows)]
    out = list(zip(names, e.rows))
    out += [(f"ring.{i}", t) for i, t in enumerate(e.ring[1:])]
    keep = sorted(e.anchors)
    if keep_anchors is not None:
        keep = keep[-keep_anchors:] if keep_anchors > 0 else []
    for p in keep:
        out += [(f"anchor.{p}.{i}", t) for i, t in enumerate(e.anchors[p][1:])]
        if p in e.end_logits:
            out.append((f"logits.{p}", e.end_logits[p]))
    return out


def _write_conv(path: str, fp: dict, tokens: list[int], L: int, prompt_end: int,
                items: list[tuple[str, torch.Tensor]]) -> dict:
    import json
    t0 = time.perf_counter()
    sums = {name: tensor_sum(t) for name, t in items}
    t_sum = time.perf_counter()
    entries, off = [], 0
    for name, t in items:
        nb = t.numel() * t.element_size()
        entries.append({"name": name, "dtype": str(t.dtype).replace("torch.", ""),
                        "shape": list(t.shape), "offset": off, "nbytes": nb})
        off += nb
    head = json.dumps({"fingerprint": fp, "tokens": tokens, "length": L,
                       "prompt_end": prompt_end, "entries": entries, "sums": sums,
                       "written": time.time()}).encode()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        for name, t in items:
            t.contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tofile(f)
    # no fsync: a planned restart reads the file back from the page cache, and a write the
    # kernel had not flushed before a power cut fails its checksum on the next start (and is
    # dropped there); an fsync per file would hold a large drain to the disk's write rate
    os.replace(tmp, path)
    t1 = time.perf_counter()
    return {"tokens": L, "bytes": off, "ms": round((t1 - t0) * 1e3, 1),
            "sum_ms": round((t_sum - t0) * 1e3, 1)}


def _read_conv(path: str, fp: dict, dev, max_len: int):
    """(header, {name: tensor on dev}) or (reason it was skipped, None)."""
    import json
    import numpy as np
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
        base = 8 + n
        if head.get("fingerprint") != fp:
            return "fingerprint", None
        L = int(head["length"])
        if L > max_len or len(head["tokens"]) != L:
            return f"length {L} against max_len {max_len}", None
        got = {}
        for e in head["entries"]:
            f.seek(base + e["offset"])
            raw = np.fromfile(f, dtype=np.uint8, count=e["nbytes"])
            t = torch.from_numpy(raw).to(dev)
            got[e["name"]] = t.view(getattr(torch, e["dtype"])).reshape(e["shape"])
    return head, got


def _anchors_of(got: dict, nring: int) -> tuple[dict, dict]:
    anchors, logits = {}, {}
    for name in got:
        if name.startswith("anchor.") and name.endswith(".0"):
            p = int(name.split(".")[1])
            anchors[p] = (p,) + tuple(got[f"anchor.{p}.{i}"] for i in range(nring))
        elif name.startswith("logits."):
            logits[int(name.split(".")[1])] = got[name]
    return anchors, logits


def _session_files(path: str) -> list[str]:
    d = os.path.dirname(path) or "."
    try:
        names = sorted((n for n in os.listdir(d) if n.startswith("parked.") and n.endswith(".bin")
                        and n.split(".")[1].isdigit()), key=lambda n: int(n.split(".")[1]))
    except OSError:
        names = []
    return [os.path.join(d, n) for n in names]


def save_session(pc: "RingPrefix", eng, path: str, fp: dict) -> dict | None:
    """Write the live conversation to `path` and every parked one beside it (`parked.N.bin`, oldest
    first), each atomically. Returns what was written, or None when nothing was held."""
    kv = eng.kv
    L = kv.length
    for old in _session_files(path):
        os.remove(old)
    t0 = time.perf_counter()
    out = {"live": None, "parked": []}
    with torch.inference_mode():
        if L > 0 and len(pc.tokens) >= L:
            out["live"] = _write_conv(path, fp, pc.tokens[:L], L, min(pc.prompt_end, L),
                                      _live_tensors(pc, eng))
        d = os.path.dirname(path) or "."
        # newest first, up to PERSIST_BYTES in all (a stop's grace period is short); the file
        # index keeps the LRU order
        room = PERSIST_BYTES - (out["live"]["bytes"] if out["live"] else 0)
        for i, e in reversed(list(enumerate(list(pc.parked)))):
            items = _parked_tensors(e, PERSIST_PARKED_ANCHORS)
            nb = sum(t.numel() * t.element_size() for _, t in items)
            if nb > room:
                out.setdefault("not_written", []).append(e.length)
                continue
            room -= nb
            out["parked"].insert(0, _write_conv(os.path.join(d, f"parked.{i}.bin"), fp,
                                                e.tokens, e.length, e.prompt_end, items))
    if out["live"] is None and not out["parked"]:
        return None
    out["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    out["bytes"] = sum(r["bytes"] for r in [out["live"]] + out["parked"] if r)
    return out


def load_session(pc: "RingPrefix", eng, path: str, fp: dict) -> dict | None:
    """Read what `save_session` wrote: the live conversation into the engine, the parked ones into
    the prefix cache's parked list. Returns what was read (`exact`: every checksum equal after the
    read), or None when there is nothing. Files are removed once read."""
    kv = eng.kv
    files = ([path] if os.path.isfile(path) else []) + _session_files(path)
    if not files:
        return None
    t0 = time.perf_counter()
    nring = len(kv.ring.tensors())
    out = {"live": None, "parked": [], "skipped": []}
    exact = True
    with torch.inference_mode():
        if os.path.isfile(path):
            head, got = _read_conv(path, fp, kv.k.device, kv.max_len)
            if got is None:
                out["skipped"].append(head)
            else:
                L = int(head["length"])
                views = [("k", kv.k), ("v", kv.v)] + ([("ks", kv.ks), ("vs", kv.vs)] if kv.fp8
                                                      else [])
                for name, t in views:
                    t.narrow(3, 0, L).copy_(got[name])
                kv.restore((L,) + tuple(got[f"ring.{i}"] for i in range(nring)))
                pc.tokens = [int(x) for x in head["tokens"]]
                pc.prompt_end = int(head.get("prompt_end", L))
                pc.anchors, pc.end_logits = _anchors_of(got, nring)
                ok = {name: tensor_sum(t) for name, t in _live_tensors(pc, eng)} == head["sums"]
                exact &= ok
                out["live"] = {"tokens": L, "exact": ok, "anchors": sorted(pc.anchors),
                               "age_s": round(time.time() - head["written"], 1)}
                if not ok:                       # torn or stale: never serve from it
                    eng.reset()
                    pc.tokens, pc.anchors, pc.end_logits, pc.prompt_end = [], {}, {}, 0
        pc.parked = []
        for f in _session_files(path):
            head, got = _read_conv(f, fp, kv.k.device, kv.max_len)
            if got is None:
                out["skipped"].append(head)
                continue
            L = int(head["length"])
            rows = [got[n] for n in ("k", "v", "ks", "vs") if n in got]
            anchors, logits = _anchors_of(got, nring)
            ring = (L,) + tuple(got[f"ring.{i}"] for i in range(nring))
            e = _Parked([int(x) for x in head["tokens"]], L, int(head["prompt_end"]), rows, ring,
                        anchors, logits, pc._park_cost(kv, L, len(anchors)))
            ok = {name: tensor_sum(t) for name, t in _parked_tensors(e)} == head["sums"]
            exact &= ok
            if ok:
                pc.parked.append(e)
            out["parked"].append({"tokens": L, "exact": ok})
    out["exact"] = exact
    out["bytes"] = sum(os.path.getsize(f) for f in files)
    out["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
    return out


# ------------------------------------------------------------------------------ what the server reads
class Served:
    """The attributes `server/app.py` reads off an engine outside the decode loop."""

    def __init__(self, eng, prefix: RingPrefix):
        self.eng = eng
        self.prefix = prefix
        c = eng.cfg
        full = [i for i, t in enumerate(c.layer_types) if t != "sliding_attention"]
        self.cfg = types.SimpleNamespace(vocab_size=c.vocab, attention_layers=full,
                                         num_key_value_heads=c.nkv, head_dim=c.hd)
        # cache_stats() reads a recurrent state's size; Kolibri's per-model state is the ring
        self.state = types.SimpleNamespace(S=torch.zeros(1), conv=torch.zeros(1))
        self.w = types.SimpleNamespace(nvfp4_source="kolibri", fp8_head_source=None,
                                       report=lambda: "kolibri")
        self.max_len = eng.max_len

    @property
    def kv(self):
        return self.eng.kv


# ------------------------------------------------------------------------------ the decode loop
def make_generate_stream(app):
    STATE = app.STATE

    def generate_stream(prompt, max_new, eos, think=None, conv_id=None, deadline=None, pen=None,
                        pstop=None, sampler=None, lpr=None, on_prefill=None, mm=None):
        served = STATE["engine"]
        eng, pc = served.eng, served.prefix
        ctx = [int(t) for t in prompt.tolist()]
        if pen is not None:
            pen.seed(ctx)
        STATE["last_ctx"] = ctx
        bs = STATE["blocks"] = app.BlockStats()
        if think is not None:
            think.start(ctx)
        where = getattr(on_prefill, "info", None)
        if where is None:
            where = {}
        t_pre = time.perf_counter()
        with torch.inference_mode():
            start, kind = pc.resume(eng, ctx)
            where.update(kind=kind, start=start, chunk=pc.chunk, t0=t_pre)
            row = pc.prefill(eng, ctx, start, on_chunk=on_prefill)
        STATE["last_prefill"] = {"reused": start, "forwarded": len(ctx) - start,
                                 "ms": (time.perf_counter() - t_pre) * 1e3,
                                 "kind": kind if start else None}
        n_out = 0
        sampling = sampler is not None and sampler.on
        # the next row: a decode step or a row of a verified draft tree (engine/kolibri/spec.py);
        # every row is the decode step's own bits, so the picks below do not see the difference
        rows = SpecRows(eng, STATE.get("kolibri_verifier"), STATE.get("kolibri_spec"))
        rows.start(ctx)
        STATE["kolibri_rows"] = rows
        try:
            while True:
                # `row` is an inference tensor: penalties, the grammar and the logit bias edit it
                # in place, which torch allows only inside inference mode
                with torch.inference_mode():
                    if pen is not None:
                        pen.mask = bool(think is not None and think.inside)
                        pen.apply_single(row)
                    tok = sample_row(sampler, row, len(ctx)) if sampling else int(row.argmax())
                    if lpr is not None:
                        lpr.rows(row[None], [tok])
                ctx.append(tok)
                n_out += 1
                if n_out == 1:
                    bs.first()
                else:
                    bs.block()
                if pen is not None:
                    with torch.inference_mode():
                        pen.commit([tok])
                if pstop is not None and pstop.observe([tok]):
                    yield tok
                    return
                yield tok
                if tok in eos or n_out >= max_new:
                    return
                if deadline is not None and deadline.expired():
                    return
                if think is not None:
                    think.observe([tok])
                    if think.hit:
                        # close the block with the budget's phrase, forwarded like any token
                        print(f"[think] closed the reasoning block: reason="
                              f"{think.reason or 'budget'} at {think.n} tokens", flush=True)
                        think.t_forced = time.perf_counter()
                        closing = list(think.close_ids)
                        rows.settle()
                        with torch.inference_mode():
                            row = eng.decode(tok)
                            for t in closing[:-1]:
                                row = eng.decode(t)
                        ctx.extend(closing)
                        if pen is not None:
                            pen.commit(closing)
                        think.observe(closing)
                        if lpr is not None:
                            lpr.forced(closing)
                        for t in closing:
                            n_out += 1
                            yield t
                            if n_out >= max_new:
                                return
                        if pstop is not None and pstop.observe(closing):
                            return
                        tok = closing[-1]
                with torch.inference_mode():
                    row = rows.next(tok, ctx)
        finally:
            with torch.inference_mode():
                rows.settle()
            # rows [0, kv.length) hold ctx[:kv.length]: the last token is decided, not forwarded
            pc.committed(ctx, eng.kv.length)

    return generate_stream



def sample_row(sampler, row, index: int) -> int:
    """`sampler(row, index)` with the same distribution, cheaper when top_k is set: the release's
    defaults (top_k 128, top_p 0.97) leave at most k tokens with mass, so the temperature, min_p and
    top-p filters run over the k largest logits instead of sorting all 128,000. A seeded (coupled) request keeps the
    sampler's own position-keyed draw."""
    import math
    if getattr(sampler, "coupled", False) or sampler.top_k <= 0:
        return sampler(row, index=index)
    lg = row.float()
    if sampler.temperature > 0.0:
        lg = lg / sampler.temperature
    vals, idx = torch.topk(lg, min(sampler.top_k, lg.shape[-1]))
    if sampler.min_p > 0.0:
        vals = vals.masked_fill(vals < vals[0] + math.log(sampler.min_p), float("-inf"))
    if sampler.top_p < 1.0:
        cum = torch.cumsum(torch.softmax(vals, -1), -1)
        drop = cum > sampler.top_p
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        vals = vals.masked_fill(drop, float("-inf"))
    p = torch.softmax(vals, -1)
    j = torch.multinomial(p, 1, generator=sampler._rng(p.device))
    return int(idx[j])

def make_build_prompt(app, original):
    """The server's build_prompt, with `in_think = "model"` when Kolibri's template leaves the
    model to open the reasoning block itself."""
    from engine.kolibri.chat import thinking_mode

    def build_prompt(body):
        ids, kind, in_think = original(body)
        if kind == "chat":
            text = app.STATE["tok"].decode(ids.tolist()[-8:], skip_special_tokens=False)
            in_think = thinking_mode(text)
        return ids, kind, in_think

    return build_prompt


# ------------------------------------------------------------------------------ speculation
#: the self-check's prompt: long enough that the sliding layers' 640-slot ring has wrapped
SELFCHECK_TEXT = ("The ring keeps the last six hundred and forty rows of every sliding layer. "
                  "A verified row must equal the decoded row bit for bit, or greedy changes. ")


def setup_spec(app, a, eng, tok, tdir) -> None:
    """The verifier (after its self-check on the board), the lookup source and StairCut; or the
    reason speculation stays off, in `STATE["kolibri_spec_off"]`."""
    from engine.kolibri import spec as S
    from engine.kolibri.verify import make_verifier
    from engine.tokfp import fingerprint
    STATE = app.STATE
    STATE.update(kolibri_verifier=None, kolibri_spec=None, kolibri_spec_off=None)
    if getattr(a, "kolibri_spec", "on") != "on":
        STATE["kolibri_spec_off"] = "--kolibri-spec off"
        print("[kolibri-spec] off (--kolibri-spec off)", flush=True)
        return
    t0 = time.time()
    ids = tok(SELFCHECK_TEXT * 40, return_tensors="pt").input_ids[0].tolist()[:720]
    why = []
    v = make_verifier(eng, log=lambda m: (why.append(m), print(m, flush=True)), selfcheck_ids=ids)
    if v is None:
        STATE["kolibri_spec_off"] = why[-1] if why else "verifier unavailable"
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stair = os.path.expanduser(a.kolibri_stair or os.path.join(root, "ops", "kolibri-stair.json"))
    prices = S.PriceTable.load(stair) if os.path.isfile(stair) else S.default_prices()
    sha = fingerprint(tdir)
    store = None
    sdir = os.path.expanduser(getattr(a, "kolibri_suffix_store", "") or "")
    if sdir:
        from engine.cache import PersistentSuffixStore
        store = PersistentSuffixStore(sdir, tokenizer_sha=sha).open()
        STATE["suffix_store"] = store
        STATE["suffix_scope"] = "all"
    corpus = os.path.expanduser(getattr(a, "kolibri_corpus", "") or "")
    look = S.LookupSource(corpus=corpus if os.path.isdir(corpus) else "", tokenizer_sha=sha,
                          store=store)
    STATE["kolibri_verifier"] = v
    STATE["kolibri_spec"] = S.KolibriSpec([look], S.StairCut(prices, max_rows=v.max_rows),
                                          max_rows=v.max_rows)
    took = v.capture(S.SIZES)
    print(f"[kolibri-spec] on: prices {stair if os.path.isfile(stair) else 'study estimate'}; "
          f"corpus {corpus if look.d.corpus is not None else 'none'}; session store "
          f"{sdir or 'off'}; {len(S.SIZES)} verify graphs in {took:.1f}s; "
          f"{time.time() - t0:.1f}s in all", flush=True)


# ------------------------------------------------------------------------------ load
def load(app, a) -> None:
    """Load Kolibri-1, fill `app.STATE`, replace `generate_stream` and `build_prompt`."""
    from engine.kolibri import chat
    from engine.kolibri.model import KolibriEngine
    t0 = time.time()
    if not a.kolibri_set:
        raise SystemExit("--kolibri needs --kolibri-set DIR (the downloaded Kolibri-1 set)")
    set_dir = os.path.expanduser(a.kolibri_set)
    fp8_dir = os.path.expanduser(a.kolibri_fp8) if a.kolibri_fp8 else None
    tdir = chat.tokenizer_dir(*(os.path.expanduser(d) for d in
                                (a.kolibri_tokenizer or "", fp8_dir or "", set_dir) if d))
    tok = chat.load_tokenizer(tdir)
    gdir = fp8_dir if fp8_dir and os.path.isfile(os.path.join(fp8_dir, "generation_config.json")) \
        else tdir
    stops = chat.stop_ids(gdir)
    eng = KolibriEngine.load(set_dir, fp8_dir, device="cuda", max_len=int(a.max_len),
                             graphs=a.kolibri_graphs == "on")
    prefix = RingPrefix(chunk=int(a.kolibri_chunk), anchor_bytes=int(a.kolibri_anchor_gb * (1 << 30)),
                        stash_bytes=int(a.kolibri_stash_gb * (1 << 30)),
                        park_min=int(getattr(a, "kolibri_park_min", 64)))
    served = Served(eng, prefix)
    samp = chat.sampling_defaults(gdir) if a.kolibri_sampling == "release" else {}
    temperature = samp.get("temperature", a.temperature)
    top_p = samp.get("top_p", a.top_p)
    top_k = samp.get("top_k", a.top_k)
    app.STATE.update(
        engine=served, tok=tok, drafter=None, k=0, device="cuda", tree=False,
        sampled_tree=False, relax=app.Relax(1.0, 1), model=a.served_model,
        started=int(time.time()), verbose=a.verbose, max_len=int(a.max_len),
        default_max_tokens=int(a.default_max_tokens), reasoning_format=a.reasoning_format,
        request_timeout=float(a.request_timeout), max_queue=int(a.max_queue),
        queue_timeout=float(a.queue_timeout), draining=False, cfg_eos=stops,
        think_budget=a.think_budget, think_stall=bool(a.kolibri_think_stall == "on"),
        reasoning_effort=a.reasoning_effort,
        pen_spec=app.PenaltySpec(a.rep_penalty, a.presence_penalty, a.frequency_penalty,
                                 a.no_repeat_ngram),
        temperature=temperature, top_p=top_p, top_k=top_k,
        pattern_stop=(tuple(int(x) for x in a.pattern_stop.split(":"))
                      if a.pattern_stop else None),
        state_store=None, session_cache=False, prefix_cache=True, prefix_chunk=prefix.chunk,
        resident=None, response_cache=None, suffix_store=None, suffix_scope="all",
        kolibri_prefix=prefix, usage_default=(a.usage_default == "on"),
        # the prefill watch's SSE comments for a long streamed prefill (`--prefill-heartbeat-s`)
        prefill_heartbeat=float(getattr(a, "prefill_heartbeat_s", 0.0) or 0.0))
    setup_spec(app, a, eng, tok, tdir)
    app.generate_stream = make_generate_stream(app)
    app.build_prompt = make_build_prompt(app, app.build_prompt)
    print(f"[kolibri] tokenizer {tdir}; stop ids {stops}; sampling default "
          f"T={temperature} top_p={top_p} top_k={top_k}; think stall "
          f"{app.STATE['think_stall']}; prefix chunk {prefix.chunk}, anchors "
          f"{a.kolibri_anchor_gb:g} GiB; KV {eng.kv.nbytes() / 1e9:.2f} GB at max_len "
          f"{a.max_len}; loaded in {time.time() - t0:.1f}s", flush=True)
    # one warm request: Triton compiles and the decode graph is captured here, not in the first
    # client's request
    with torch.no_grad():
        ids = tok("warm up the kernels", return_tensors="pt").input_ids[0].cuda()
        list(app.generate_stream(ids, 4, set()))
    eng.reset()
    prefix.tokens, prefix.anchors, prefix.end_logits = [], {}, {}
    prefix.parked, prefix.prompt_end = [], 0
    print(f"[kolibri] warm-up done in {time.time() - t0:.1f}s", flush=True)
    # the session a planned stop wrote, and the hook that writes the next one
    sdir = os.path.expanduser(getattr(a, "kolibri_session_dir", "") or "")
    if sdir and getattr(a, "kolibri_persist", "on") == "on":
        path = os.path.join(sdir, "live.bin")
        fp = session_fingerprint(eng, set_dir, tdir)
        try:
            r = load_session(prefix, eng, path, fp)
        except Exception as exc:                 # a bad file costs the session, never the start
            r = {"error": repr(exc)}
            eng.reset()
            prefix.tokens, prefix.anchors, prefix.end_logits = [], {}, {}
        if r is not None:
            app.STATE["session_restored"] = r
            print(f"[kolibri] session read back: {r}", flush=True)

        def on_drained():
            with app.LOCK:
                r = save_session(prefix, eng, path, fp)
            app.STATE["session_saved"] = r
            print(f"[kolibri] session written: {r}", flush=True)

        app.STATE["on_drained"] = on_drained
