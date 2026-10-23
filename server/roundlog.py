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
"""

from __future__ import annotations

import json
import threading
import time

import torch

from engine.settings import SETTINGS as _S

PATH = _S.get("ROUND_LOG") or ""


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

    def round(self, group, ev, cap=None, pre_rows=None, graphed=None) -> int:
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
    def slot(self, rid, slot: int, E: int, ctx: int, tree, path: list, pre_n: int | None,
             drafter=None) -> None:
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
               "path": list(path), "accepted": len(path) - 1, "pre_n": pre_n}
        with self.lock:
            self._write(rec)
            self._drain()


def node_q(drafter, tree, dep, rank):
    """Per node the calibrated acceptance probability of the router arm that proposed `tree`
    (engine/router.py `_node_q`), or None when the drafter has none."""
    arm = drafter
    key = getattr(drafter, "last_key", None)
    if key is not None and hasattr(drafter, "small"):
        arm = drafter.small if key == "s" else drafter.large
    fn = getattr(arm, "_node_q", None)
    if fn is None:
        return None
    try:
        return [None] + [round(float(fn(i, tree, dep[i], rank[i])), 5) for i in range(1, len(tree.tokens))]
    except Exception:                                                   # noqa: BLE001
        return None


LOG = RoundLog(PATH) if PATH else None
