"""The round logger (SPD-187, QWEN38_ROUND_LOG=<path>): one JSON line per verify round, per slot block and
per packed draft, for the round planner's prices and payoffs (engine/methods/roundplan.py).

Off unless the environment names a file; on, it costs a few CUDA events and a dictionary per round, and
never synchronizes: a pass is timed by two events around its call, read when they are done (on a later
round, after the slots' walks have synced). Lines:

  {"k": "round", "rid", "t", "E", "slots", "rows", "pre_rows", "cap", "ctx", "graphed", "pass_ms",
   "period_ms"}                  one joint (or lone, E = 1) verify pass; `pre_rows` the trees before the
                                 round's cut, `period_ms` the time since the previous round began
  {"k": "slot", "rid", "slot", "E", "ctx", "n", "parents", "source", "score", "q", "depth", "rank",
   "path", "accepted", "pre_n"}  a slot's block after its walk: per node its source, drafter score,
                                 calibrated q (the slot's router, before it learns from this block),
                                 depth and rank, and the accepted path (a node's label: on the path;
                                 its parent on the path: it was reached)
  {"k": "draft", "t", "E", "slots", "ms"}   one packed draft pass

A slot line also carries, for the drafter studies (~/Projects/spd-e5/fable-teach-drafter-1023.md):
`top_ids` / `top_lp` per verified node the target's top-64 ids and log-probabilities (the verify's row
after penalties, temperature 1; copied behind the walk's own synchronize, `topk_async`), `drawn` the
tokens the walk emitted (the target's draw at each path node), `cand` / `lat_lp` the proposing DFlash2
arm's lattice candidates and log-probabilities, and with QWEN38_ROUND_LOG_HIDDEN=1 `hidden` the anchor's
post-norm hidden state.
"""

from __future__ import annotations

import json
import threading
import time

import torch

from engine.settings import SETTINGS as _S

PATH = _S.get("ROUND_LOG") or ""
#: SPD-187: the anchor's post-norm hidden in each slot line too (fp16, base64; about 10 KB a block)
HIDDEN = _S.get("ROUND_LOG_HIDDEN") == "1"
#: SPD-201: each slot line also carries the proposing arm's whole lattice (`lat_scores` [L][k][k], the raw
#: pair scores the walk reads: a depth's log q given its predecessor is their log-softmax at the request's
#: temperature) and the request's sampling profile and seed (`samp`), for the keyed-noise ranking study
LATTICE = _S.get("ROUND_LOG_LATTICE") == "1"
#: the target's top-k a verified node logs (ids and log-probabilities of the verify's own row)
TOPK = 64


class RoundLog:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.fh = open(path, "a", buffering=1)
        self.rid = 0
        self.pending: list = []           # (record, start event, end event) not yet done
        self.t_last = None

    # ---- timing without a synchronize ------------------------------------------------------
    def begin(self):
        """Two events and a record for a pass about to be issued; `end(...)` after its call."""
        if not torch.cuda.is_available():
            return None, None
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        return e0, e1

    def round(self, group, ev, cap=None, pre_rows=None, graphed=None, plan=None) -> int:
        """Log one verify pass over `group` (server/slots.py VerifyReq) timed by `ev`; returns its id."""
        now = time.perf_counter()
        with self.lock:
            self.rid += 1
            rid = self.rid
            period = None if self.t_last is None else (now - self.t_last) * 1e3
            self.t_last = now
            rec = {"k": "round", "rid": rid, "t": round(time.time(), 4), "E": len(group),
                   "slots": [r.slot for r in group], "rows": [r.rows for r in group],
                   "pre_rows": pre_rows, "cap": cap, "ctx": [int(r.start) for r in group],
                   "graphed": graphed, "pass_ms": None,
                   "period_ms": None if period is None else round(period, 3)}
            if plan is not None:
                # SPD-187: the round planner's choice and its own prediction (shadow mode: not acted on)
                rec["plan"] = {"T": plan.get("T"), "gain": _r(plan.get("gain")), "ms": _r(plan.get("ms")),
                               "key": list(plan.get("key") or ()), "version": plan.get("version"),
                               "ctx": plan.get("ctx")}
            if ev is not None and ev[0] is not None:
                ev[1].record()
                self.pending.append((rec, ev[0], ev[1]))
            else:
                self._write(rec)
            self._drain()
        return rid

    def draft(self, part, ev) -> None:
        with self.lock:
            rec = {"k": "draft", "t": round(time.time(), 4), "E": len(part),
                   "slots": [r.slot for r in part], "ms": None, "pass_ms": None}
            if ev is not None and ev[0] is not None:
                ev[1].record()
                self.pending.append((rec, ev[0], ev[1]))
            else:
                self._write(rec)
            self._drain()

    def _drain(self, force: bool = False) -> None:
        keep = []
        for rec, e0, e1 in self.pending:
            if force or e1.query():
                ms = round(e0.elapsed_time(e1), 3)
                rec["pass_ms" if rec["k"] == "round" else "ms"] = ms
                self._write(rec)
            else:
                keep.append((rec, e0, e1))
        self.pending = keep

    def _write(self, rec: dict) -> None:
        if rec.get("k") == "draft":
            rec.pop("pass_ms", None)
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ---- a slot's block after its walk -----------------------------------------------------
    @staticmethod
    def topk_async(rows, hidden=None):
        """The target's top-k of each verified row, launched now and copied into pinned memory behind the
        stream; read by `slot` after the walk has synchronized (no synchronize of its own)."""
        if rows is None or not torch.is_tensor(rows):
            return None
        with torch.no_grad():
            x = rows.float()
            vals, idx = torch.topk(x, min(TOPK, x.shape[-1]), dim=-1)
            lp = vals - torch.logsumexp(x, dim=-1, keepdim=True)
            pin = rows.is_cuda
            hv = torch.empty(lp.shape, dtype=torch.float32, pin_memory=pin)
            hi = torch.empty(idx.shape, dtype=torch.int32, pin_memory=pin)
            hv.copy_(lp, non_blocking=pin)
            hi.copy_(idx.to(torch.int32), non_blocking=pin)
            hh = None
            if HIDDEN and hidden is not None:
                hh = torch.empty(hidden.shape[-1], dtype=torch.float16, pin_memory=pin)
                hh.copy_(hidden.reshape(-1, hidden.shape[-1])[0].to(torch.float16), non_blocking=pin)
        return hv, hi, hh

    def slot(self, rid, slot: int, E: int, ctx: int, tree, path: list, pre_n: int | None,
             drafter=None, top=None, drawn=None, sampler=None) -> None:
        if tree is None:
            return
        n = len(tree.tokens)
        dep = tree.depths()
        order = sorted(range(1, n), key=lambda i: -tree.scores[i])
        rank = [0] * n
        for r, i in enumerate(order, start=1):
            rank[i] = r
        q = node_q(drafter, tree, dep, rank)
        rec = {"k": "slot", "rid": rid, "slot": slot, "E": E, "ctx": int(ctx), "n": n,
               "parents": list(tree.parents), "source": list(tree.source),
               "score": [round(float(x), 5) for x in tree.scores], "q": q, "depth": dep, "rank": rank,
               "path": list(path), "accepted": len(path) - 1, "pre_n": pre_n,
               "drawn": list(drawn) if drawn is not None else None}
        if top is not None:
            hv, hi, hh = top
            rec["top_ids"] = hi[:n].tolist()
            rec["top_lp"] = [[round(v, 3) for v in r] for r in hv[:n].tolist()]
            if hh is not None:
                import base64
                rec["hidden"] = base64.b64encode(hh.numpy().tobytes()).decode()
        cand, lat = lattice(drafter)
        rec["cand"], rec["lat_lp"] = cand, lat
        if LATTICE:
            rec["lat_scores"] = lattice_scores(drafter)
            rec["pred_proj"] = pred_proj(drafter)
            sm = sampler if sampler is not None and getattr(sampler, "on", False) else None
            rec["samp"] = (None if sm is None else
                           {"seed": sm.seed, "t": sm.temperature, "top_p": sm.top_p, "top_k": sm.top_k,
                            "min_p": sm.min_p})
        with self.lock:
            self._write(rec)
            self._drain()


def _r(x):
    return None if x is None else round(float(x), 3)


class PassTimer:
    """SPD-187: a pass timed by two CUDA events, its ms handed to a callback once the events are done
    (`drain`, called on a later round): no synchronize."""

    pending: list = []
    lock = threading.Lock()

    @staticmethod
    def begin():
        if not torch.cuda.is_available():
            return None
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        return e0, e1

    @classmethod
    def end(cls, ev, cb) -> None:
        if ev is None:
            return
        ev[1].record()
        with cls.lock:
            cls.pending.append((ev[0], ev[1], cb))

    @classmethod
    def drain(cls) -> None:
        with cls.lock:
            keep, done = [], []
            for e0, e1, cb in cls.pending:
                (done if e1.query() else keep).append((e0, e1, cb))
            cls.pending = keep
        for e0, e1, cb in done:
            cb(e0.elapsed_time(e1))


def tree_q(drafter, tree):
    """Per node the proposing arm's calibrated q (`node_q`), for the round planner."""
    if tree is None:
        return None
    n = len(tree.tokens)
    dep = tree.depths()
    order = sorted(range(1, n), key=lambda i: -tree.scores[i])
    rank = [0] * n
    for r, i in enumerate(order, start=1):
        rank[i] = r
    return node_q(drafter, tree, dep, rank)


def _arm(drafter):
    key = getattr(drafter, "last_key", None)
    if key is not None and hasattr(drafter, "small"):
        return drafter.small if key == "s" else drafter.large
    return drafter


def lattice(drafter):
    """The proposing arm's DFlash2 lattice of this block, host-side already (`walk_host_logp`): its
    candidates [L][k] and their log-probabilities, or (None, None)."""
    arm = _arm(drafter)
    d2 = getattr(arm, "mtp", arm)
    cand = getattr(d2, "_cand_host", None)
    lat = getattr(d2, "_logp_host", None)
    if isinstance(lat, list) and lat and isinstance(lat[0], list) and lat[0] and isinstance(lat[0][0], list):
        # [L][parent][candidate]: the rows along the lattice's own greedy walk (L x k numbers, not L x k x k)
        rows, j = [], 0
        for slot_lp in lat:
            row = slot_lp[j] if j < len(slot_lp) else slot_lp[0]
            rows.append([round(float(v), 3) for v in row])
            j = max(range(len(row)), key=row.__getitem__)
        lat = rows
    elif isinstance(lat, list):
        lat = [[round(float(v), 3) for v in r] if isinstance(r, list) else round(float(r), 3) for r in lat]
    return cand, lat


def lattice_scores(drafter):
    """The proposing arm's whole lattice scores of this block ([L][k][k], 3 decimals), or None. A copy to
    the host: only while QWEN38_ROUND_LOG_LATTICE is on (logging runs)."""
    arm = _arm(drafter)
    d2 = getattr(arm, "mtp", arm)
    lt = getattr(d2, "_lattice", None)
    if not lt or lt[1] is None:
        return None
    sc = lt[1].float().cpu()
    return [[[round(float(v), 3) for v in row] for row in sl] for sl in sc.tolist()] if sc.dim() == 3 else \
        [[round(float(v), 3) for v in row] for row in sc.tolist()]


def pred_proj(drafter):
    """The proposing arm's selector-projected block hidden of this block ([L, r] fp16, base64), or None (a
    lone draft, or no packed draft this round)."""
    arm = _arm(drafter)
    d2 = getattr(arm, "mtp", arm)
    pj = getattr(d2, "_pred_proj", None)
    if pj is None:
        return None
    import base64
    t = pj.detach().to(torch.float16).cpu().contiguous()
    return {"shape": list(t.shape), "b64": base64.b64encode(t.numpy().tobytes()).decode()}


def node_q(drafter, tree, dep, rank):
    """Per node the calibrated acceptance probability of the router arm that proposed `tree`
    (engine/router.py `_node_q`), or None when the drafter has none."""
    arm = _arm(drafter)
    fn = getattr(arm, "_node_q", None)
    if fn is None:
        return None
    try:
        return [None] + [round(float(fn(i, tree, dep[i], rank[i])), 5) for i in range(1, len(tree.tokens))]
    except Exception:                                                   # noqa: BLE001
        return None


LOG = RoundLog(PATH) if PATH else None
