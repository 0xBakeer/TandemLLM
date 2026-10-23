"""The round planner (SPD-187): StairCut for a whole joint verify round, written once for every model.

A joint round verifies E slots' trees in one pass. Its cost is set by the pass's rows -- under the joint
graphs by the row bucket E x T every block is padded to -- and its payoff by what the slots' kept nodes are
worth. Each round the planner takes the bucket T (rows a block) that maximises

    sum over slots s of (1 + G_s(min(T, n_s)))   /   (V(kind, E x T, ctx) + D(E) + tails(E))

where G_s(r) is slot s's gain curve -- the cumulative calibrated q of its tree admitted best-first with
ancestors, r rows -- V the joint pass's price (`PriceModel`, a prior file followed online), D the packed
draft and tails the rest of a round, both learned per E. Each slot then keeps its own best-first prefix of
T nodes (`split`): under the graphs the split inside a bucket costs nothing.

Determinism. The planner decides on a frozen snapshot of its prices: a plan is a pure function of the
round's quantised key and the snapshot's version. A new snapshot is cut every `every` rounds only when a
decision it would make differs and a price behind it moved by `move` or more; within a snapshot a key's
bucket changes only when the challenger's score beats the incumbent's by `hold` (5 %).

The cut chooses which rows are verified, never their arithmetic: every kept row is computed as it is in
any other pass (row-invariant kernels), so greedy output and seeded draws do not depend on the plan.
"""

from __future__ import annotations

import copy
import hashlib
import json

from engine.methods.staircut import PriceModel

#: a slot's gain curve, bucketed by G_s(8): the slot classes of the round's key
CURVE_BUCKETS = ((5.0, "copy"), (3.0, "strong"), (1.5, "fair"), (0.0, "weak"))


def gain_curve(q: list | None, parents: list, scores: list, t_max: int) -> list:
    """[G(1), ..., G(t_max)]: the cumulative q of the tree admitted best-first by `scores` with its
    ancestors (one pass, a path skipped when it would pass t_max), G(r) the gain of the admissions
    that fit in r rows (the anchor is row 1 and gains nothing); flat past the tree. `q[i]` the node's
    calibrated acceptance probability (None: its score)."""
    n = len(parents)
    val = [0.0] + [float(q[i] if q is not None and q[i] is not None else scores[i]) for i in range(1, n)]
    order = sorted(range(1, n), key=lambda i: -scores[i])
    keep = [False] * n
    keep[0] = True
    rows, gained = 1, 0.0
    out = [0.0] * (t_max + 1)
    for i in order:
        if keep[i]:
            continue
        path, j = [], i
        while not keep[j]:
            path.append(j)
            j = parents[j]
        if rows + len(path) > t_max:
            continue
        for k in path:
            keep[k] = True
            gained += val[k]
        rows += len(path)
        out[rows] = gained
    # a row count between two admissions keeps the gain of the last one that fit
    best = 0.0
    for r in range(1, t_max + 1):
        best = max(best, out[r])
        out[r] = best
    return out[1:]


def curve_class(curve: list) -> str:
    g8 = curve[min(7, len(curve) - 1)] if curve else 0.0
    return next(name for lo, name in CURVE_BUCKETS if g8 >= lo)


class RoundPlanner:
    """Plans the row bucket of a joint round. `prices`: a PriceModel whose kind "joint{E}" prices E x T
    rows by context class (the eager pass "joint{E}-eager"). `buckets`: the rows a block may take. `cap`:
    the most rows a pass may take (`cap(E)`, the family's row-invariant range)."""

    def __init__(self, prices: PriceModel | None, buckets=(4, 8, 16, 32), cap=None, hold: float = 0.05,
                 every: int = 128, move: float = 0.10):
        self.live = prices or PriceModel()
        self.buckets = tuple(sorted(buckets))
        self.cap = cap or (lambda E: 64)
        self.hold, self.every, self.move = hold, every, move
        self.D: dict = {}             # E -> learned packed-draft ms
        self.tails: dict = {}         # E -> learned rest of a round, ms
        self.version = 0
        self.snap = self._freeze()
        self.incumbent: dict = {}     # (version, key) -> T
        self.rounds = 0

    # ---- learning (live) --------------------------------------------------------------------
    def observe_pass(self, E: int, rows: int, ms: float, ctx_class: int, eager: bool = False) -> None:
        self.live.observe(rows, ms, f"joint{E}" + ("-eager" if eager else ""), ctx_class)

    def _ema(self, d: dict, E: int, v: float, a: float = 0.1) -> None:
        d[E] = v if E not in d else (1 - a) * d[E] + a * v

    def observe_draft(self, E: int, ms: float) -> None:
        self._ema(self.D, E, ms)

    def observe_tail(self, E: int, ms: float) -> None:
        self._ema(self.tails, E, max(0.0, ms))

    # ---- snapshots -------------------------------------------------------------------------
    def _freeze(self) -> dict:
        return {"prices": copy.deepcopy((self.live.table, self.live.V, self.live.delta, self.live.step)),
                "D": dict(self.D), "tails": dict(self.tails)}

    def _snap_prices(self) -> PriceModel:
        table, V, delta, step = self.snap["prices"]
        pm = PriceModel.__new__(PriceModel)
        pm.table, pm.V, pm.delta, pm.step, pm.ema = table, V, delta, step, self.live.ema
        return pm

    def tick(self, keys_seen=()) -> bool:
        """Once a round: every `every` rounds, cut a new snapshot when a decision would change on it and
        a price it reads moved by `move` or more. Returns whether the version moved."""
        self.rounds += 1
        if self.rounds % self.every:
            return False
        cand = self._freeze()
        old, new = self._snap_prices(), None
        moved = False
        for E, ctx, curves in keys_seen:
            keep = self.snap
            self.snap = cand
            new = self._snap_prices()
            t_new = self._best(E, ctx, curves, None)[0]
            self.snap = keep
            t_old = self._best(E, ctx, curves, None)[0]
            if t_new != t_old:
                k = f"joint{E}"
                a, b = self.snap["tails"].get(E, 0.0), cand["tails"].get(E, 0.0)
                if a and abs(b - a) / a >= self.move:       # the round's rest moved
                    moved = True
                    break
                for T in (t_old, t_new):         # a price behind either decision
                    a, b = old.prices(E * T, k, ctx)[-1], new.prices(E * T, k, ctx)[-1]
                    if a and abs(b - a) / a >= self.move:
                        moved = True
                if moved:
                    break
        if moved:
            self.snap = cand
            self.version += 1
        return moved

    # ---- the plan --------------------------------------------------------------------------
    def key(self, E: int, ctx: int, curves: list) -> tuple:
        return (E, ctx, tuple(sorted(curve_class(c) for c in curves)))

    def cost(self, E: int, T: int, ctx: int, pm: PriceModel | None = None, eager: bool = False) -> float:
        pm = pm or self._snap_prices()
        v = pm.prices(E * T, f"joint{E}" + ("-eager" if eager else ""), ctx)[-1]
        return v + self.snap["D"].get(E, 0.0) + self.snap["tails"].get(E, 0.0)

    def _best(self, E: int, ctx: int, curves: list, sizes: list | None):
        pm = self._snap_prices()
        best = None
        for T in self.buckets:
            if E * T > self.cap(E) or T < 2:
                continue
            gain = sum(1.0 + c[min(T, len(c)) - 1] for c in curves)
            ms = self.cost(E, T, ctx, pm)
            s = gain / ms if ms > 0 else 0.0
            if best is None or s > best[1]:
                best = (T, s, gain, ms)
        return best

    def plan(self, E: int, ctx: int, curves: list, sizes: list | None = None) -> dict:
        """The bucket for this round: {"T", "score", "gain", "ms", "key", "version"}; the incumbent of
        the key keeps its bucket unless the best beats it by `hold`."""
        k = self.key(E, ctx, curves)
        best = self._best(E, ctx, curves, sizes)
        if best is None:
            return {"T": None, "key": k, "version": self.version}
        inc = self.incumbent.get((self.version, k))
        if inc is not None and inc != best[0] and E * inc <= self.cap(E):
            gain = sum(1.0 + c[min(inc, len(c)) - 1] for c in curves)
            ms = self.cost(E, inc, ctx)
            s_inc = gain / ms if ms > 0 else 0.0
            if best[1] < s_inc * (1.0 + self.hold):
                best = (inc, s_inc, gain, ms)
        self.incumbent[(self.version, k)] = best[0]
        return {"T": best[0], "score": best[1], "gain": best[2], "ms": best[3], "key": k,
                "version": self.version}

    def digest(self) -> str:
        """A hash of the snapshot the plans are made on (determinism checks)."""
        doc = json.dumps(self.snap, sort_keys=True, default=str)
        return hashlib.sha256(doc.encode()).hexdigest()[:16]


def split(curves_nodes: list, T: int) -> list:
    """Each slot's kept node set for bucket T: its own best-first prefix of T rows with ancestors
    (`curves_nodes`: per slot (parents, scores)). Ancestor-closed by construction."""
    out = []
    for parents, scores in curves_nodes:
        order = sorted(range(1, len(parents)), key=lambda i: -scores[i])
        keep = {0}
        for i in order:
            path, j = [], i
            while j not in keep:
                path.append(j)
                j = parents[j]
            if len(keep) + len(path) <= T:
                keep.update(path)
        out.append(keep)
    return out
