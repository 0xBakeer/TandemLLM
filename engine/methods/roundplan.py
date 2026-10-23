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


def admission(parents: list, scores: list, t_max: int | None = None) -> list:
    """The best-first admission of a tree by `scores` with ancestors: [(node, row)], row the node's row
    in the cut (the anchor is row 1), a path skipped when it would pass `t_max`."""
    n = len(parents)
    order = sorted(range(1, n), key=lambda i: -scores[i])
    keep = [False] * n
    keep[0] = True
    rows, out = 1, []
    for i in order:
        if keep[i]:
            continue
        path, j = [], i
        while not keep[j]:
            path.append(j)
            j = parents[j]
        if t_max is not None and rows + len(path) > t_max:
            continue
        for k in reversed(path):
            keep[k] = True
            rows += 1
            out.append((k, rows))
    return out


def gain_curve(q: list | None, parents: list, scores: list, t_max: int, cal: list | None = None) -> list:
    """[G(1), ..., G(t_max)]: the cumulative q of the tree admitted best-first by `scores` with its
    ancestors (one pass, a path skipped when it would pass t_max), G(r) the gain of the admissions
    that fit in r rows (the anchor is row 1 and gains nothing); flat past the tree. `q[i]` the node's
    calibrated acceptance probability (None: its score); `cal[r]` (SPD-187) scales the q of the node
    admitted at row r (the planner's calibration by row: realised hits over predicted q)."""
    n = len(parents)
    val = [0.0] + [float(q[i] if q is not None and q[i] is not None else scores[i]) for i in range(1, n)]
    if cal is not None:
        for k, r in admission(parents, scores, t_max):
            val[k] *= cal[min(r, len(cal) - 1)]
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
        # SPD-187, learned online (always): the cost of a bucket as the round period measured at it ((E, T,
        # ctx) -> [ms, rounds], an EMA with an outlier gate); the payoff from `accept` (an AcceptModel: the
        # probability each node is accepted, prequential); `explore`: an allowed bucket with fewer than `nmin`
        # measured rounds is planned first (the warm-up). The hold zone: `ref_T(E)` is the reference cut's
        # bucket; when the plan's rolling tokens/ms at an E falls under the reference's estimate by more than
        # `guard`, the reference cut is taken for `held_rounds` rounds (the plan still computed: shadow)
        self.P: dict = {}
        self.chosen: dict = {}        # (E, T) -> rounds planned at that bucket (the report)
        self.explore, self.nmin = False, 8
        self.accept = None
        self.ref_T = None
        self.guard, self.held_rounds = 0.05, 512
        self.rate: dict = {}          # E -> [plan tokens/ms EMA, reference tokens/ms EMA, planned rounds]
        self.held: dict = {}          # E -> rounds left on the reference cut
        self.cur = None               # the round being accounted: E, T, ctx, tokens, reference tokens
        self.state_path = None        # where `state()` is saved every `save_every` rounds (a thread)
        self.save_every = 512
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

    def observe_period(self, E: int, T: int, ctx_class: int, ms: float) -> None:
        """SPD-187: a round at bucket T took `ms` start to start (pass, draft, walks, tails)."""
        p = self.P.setdefault((E, T, ctx_class), [0.0, 0])
        if p[1] >= 4 and ms > 3.0 * p[0]:
            return                              # a round that waited on something else (a prefill, a join)
        p[0] = 0.9 * p[0] + 0.1 * ms if p[1] >= 4 else (p[0] * p[1] + ms) / (p[1] + 1)
        p[1] += 1
        if self.explore and p[1] == self.nmin and all(self._measured(E, t, ctx_class) for t in self._allowed(E)):
            self.snap = self._freeze()          # the warm-up of this (E, ctx) is done: plan on its periods
            self.version += 1

    # ---- the round's accounting (the hold zone) -----------------------------------------------
    def begin_round(self, E: int, T: int, ctx: int) -> None:
        self.cur = {"E": E, "T": T, "ctx": ctx, "tok": 0.0, "ref": 0.0}

    def add_block(self, tokens: float, ref_tokens: float) -> None:
        """A slot's block of the current round: the tokens it wrote, and what the reference cut would have
        written (exact where the reference rows were verified, the acceptance model's estimate past them)."""
        if self.cur is not None:
            self.cur["tok"] += tokens
            self.cur["ref"] += ref_tokens

    def end_round(self, period_ms: float) -> None:
        c, self.cur = self.cur, None
        if c is None or self.ref_T is None or period_ms <= 0:
            return
        E = c["E"]
        ref_ms = (self.P.get((E, self.ref_T(E), c["ctx"])) or [None, 0])
        if ref_ms[1] < self.nmin:
            return
        r = self.rate.setdefault(E, [0.0, 0.0, 0])
        a = 0.02 if r[2] >= 50 else 1.0 / (r[2] + 1)
        r[0] = (1 - a) * r[0] + a * c["tok"] / period_ms
        r[1] = (1 - a) * r[1] + a * c["ref"] / ref_ms[0]
        r[2] += 1
        if self.held.get(E):
            self.held[E] -= 1
        elif r[2] >= 64 and r[0] < (1.0 - self.guard) * r[1]:
            self.held[E] = self.held_rounds     # the plan lost to the reference: its cut, learning goes on
            r[2] = 0

    # ---- state ---------------------------------------------------------------------------------
    def state(self) -> dict:
        return {"format": 1, "P": [[list(k), v] for k, v in self.P.items()], "version": self.version,
                "accept": self.accept.state() if self.accept is not None else None,
                "rate": [[k, v] for k, v in self.rate.items()]}

    def save_async(self) -> None:
        """The state to `state_path` (JSON, written beside and renamed), off the round's thread."""
        import os
        import threading
        st, path = self.state(), self.state_path

        def run():
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(st, f)
                os.replace(tmp, path)
            except OSError:
                pass
        threading.Thread(target=run, name="roundplan-save", daemon=True).start()

    def load(self, st: dict) -> None:
        """Warm start: the measured periods (counted as half measured, so the warm-up re-checks them) and the
        acceptance model (its sums halved)."""
        if st.get("format") != 1:
            return
        for k, v in st.get("P", []):
            self.P[tuple(k)] = [v[0], min(v[1], max(1, self.nmin // 2))]
        if self.accept is not None and st.get("accept"):
            self.accept.load(st["accept"])

    def _allowed(self, E: int) -> list:
        return [T for T in self.buckets if T >= 2 and E * T <= self.cap(E)]

    def _measured(self, E: int, T: int, ctx: int) -> bool:
        p = self.P.get((E, T, ctx))
        return p is not None and p[1] >= self.nmin

    # ---- snapshots -------------------------------------------------------------------------
    def _freeze(self) -> dict:
        snap = {"prices": copy.deepcopy((self.live.table, self.live.V, self.live.delta, self.live.step)),
                "D": dict(self.D), "tails": dict(self.tails)}
        if getattr(self, "explore", False):
            snap["P"] = {k: v[0] for k, v in self.P.items() if v[1] >= self.nmin}
        return snap

    def _snap_prices(self) -> PriceModel:
        table, V, delta, step = self.snap["prices"]
        pm = PriceModel.__new__(PriceModel)
        pm.table, pm.V, pm.delta, pm.step, pm.ema = table, V, delta, step, self.live.ema
        return pm

    def tick(self, keys_seen=()) -> bool:
        """Once a round: every `every` rounds, cut a new snapshot when a decision would change on it and
        a price it reads moved by `move` or more. Returns whether the version moved."""
        self.rounds += 1
        if self.state_path and self.rounds % self.save_every == 0:
            self.save_async()
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
                    a, b = self.snap.get("P", {}).get((E, T, ctx)), cand.get("P", {}).get((E, T, ctx))
                    if b is not None and (a is None or abs(b - a) / a >= self.move):
                        moved = True             # SPD-187: a measured round period behind it moved
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
        measured = self.snap.get("P", {}).get((E, T, ctx))
        if measured is not None:
            return measured                     # SPD-187: the round period measured at this bucket
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
        the key keeps its bucket unless the best beats it by `hold`. With `explore`, an allowed bucket not
        yet measured `nmin` rounds at this (E, ctx) is taken first, the fewest-measured, the smaller on a tie
        ({"explore": True})."""
        k = self.key(E, ctx, curves)
        if self.explore:
            todo = [T for T in self._allowed(E) if not self._measured(E, T, ctx)]
            if todo:
                T = min(todo, key=lambda t: ((self.P.get((E, t, ctx)) or [0, 0])[1], t))
                return {"T": T, "key": k, "version": self.version, "explore": True}
        if self.held.get(E) and self.ref_T is not None:
            shadow = self._best(E, ctx, curves, sizes)
            return {"T": self.ref_T(E), "key": k, "version": self.version, "held": True,
                    "shadow_T": shadow[0] if shadow else None}
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
        self.chosen[(E, best[0])] = self.chosen.get((E, best[0]), 0) + 1
        return {"T": best[0], "score": best[1], "gain": best[2], "ms": best[3], "key": k,
                "version": self.version}

    def report(self) -> dict:
        """/health: the version, the measured round period by (E, T, ctx) and the rounds at each bucket."""
        return {"version": self.version, "explore": self.explore,
                "periods": {f"E{E} T{T} ctx{c}": [round(v[0], 1), v[1]] for (E, T, c), v in sorted(self.P.items())},
                "rows_chosen": {f"E{E} rows{E * T}": n for (E, T), n in sorted(self.chosen.items())},
                "rate": {f"E{E}": {"plan_tok_ms": round(v[0], 4), "ref_tok_ms": round(v[1], 4), "rounds": v[2]}
                         for E, v in sorted(self.rate.items())},
                "held": {f"E{E}": n for E, n in self.held.items() if n},
                "accept": self.accept.errors() if self.accept is not None else None,
                "accept_history": self.accept.history[-16:] if self.accept is not None else None}

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
