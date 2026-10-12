"""StairCut with its price model and its calibration, written once for every model (ENG-245).

Each round keeps the part of the draft that maximises expected committed tokens per millisecond,

    (1 + sum of q over the kept nodes) / (V(rows) + draft),

where q of a node is the chance the walk reaches and takes it (`ProbDeciles`: the draft head's
probability mapped to the measured rate of acceptance) and V(rows) what a verify block of that many
rows costs with its commit (`PriceModel`: a prior from the model's price file, followed online).
The cut decides only which rows are verified; every kept row is the target's to accept or not, so a
cut moves speed and never text.

Strategies: `StairCut.cut` is `chain_siblings` -- a spine (the head's greedy chain) and, with
`siblings`, one leaf a depth (the head's second token there, a child of the spine node above it) --
moved here from Flash-Next's package as it was. With a lookup continuation (`look`, ENG-229) the
round is one tree of both sources -- the spine, and the lookup's chain where it leaves the spine (or
runs on past its end) -- admitted best-first by q; where the two agree, a spine node's q is that of
either being right. The 27B's best-first admission over any tree (engine/router.py) joins as a
further strategy in ENG-257.
"""

from __future__ import annotations

import json
import os


class PriceModel:
    """The verify-and-commit price of a block of `rows` rows, in ms, by kind of block ("chain",
    "tree") and context class: a prior table from a price file, followed online (EMA 0.1 a count);
    the counts not measured yet follow the prior shifted by the learned offset (a long context moves
    every count), and a price never falls with more rows (a cut must not buy a row for nothing because
    one count was measured at a busier moment than the next).

    The file (ops/prices/<model>/<profile>.<device>.json, and the 27B's ops/stair-tables-*.json):

        {"<ctx class>": {"<kind>": {"<rows>": ms, ...}, ...}, ..., "step_ms": 3.7, "measured": "..."}

    A context class not in the file takes the nearest lower one (the lowest, below all); rows between
    two measured counts interpolate, rows past the last add `step_ms` a row (else the last two counts'
    slope). Without a file every count starts at 1 ms, the same for all: the cut drafts to the full
    depth until the first rounds price it."""

    def __init__(self, table: dict | None = None, step_ms: float | None = None, ema: float = 0.1):
        self.table = {int(c): {str(k): {int(r): float(ms) for r, ms in rows.items()} for k, rows in kinds.items()}
                      for c, kinds in (table or {}).items()}
        self.step, self.ema = step_ms, ema
        self.V: dict = {}              # (kind, class) -> {rows: learned ms}
        self.delta: dict = {}          # (kind, class) -> learned shift of the prior

    @classmethod
    def load(cls, path: str, log=print) -> "PriceModel":
        if not os.path.isfile(path):
            log(f"[prices] no price file {path}: the cut learns its prices online")
            return cls()
        with open(path) as f:
            doc = json.load(f)
        table = {k: v for k, v in doc.items() if k.lstrip("-").isdigit()}
        for c, kinds in table.items():
            for kind, rows in kinds.items():
                for r, ms in rows.items():
                    if int(r) < 1 or not 0.0 < float(ms) < 10_000.0:
                        raise ValueError(f"price file {path}: [{c}][{kind}][{r}] = {ms!r} is not a row count and ms")
        return cls(table, doc.get("step_ms"))

    def _rows(self, kind: str, ctx_class: int) -> dict | None:
        if not self.table:
            return None
        below = [c for c in self.table if c <= ctx_class]
        return self.table[max(below) if below else min(self.table)].get(kind)

    def prior(self, rows: int, kind: str = "chain", ctx_class: int = 0) -> float | None:
        tbl = self._rows(kind, ctx_class)
        if not tbl:
            return None
        if rows in tbl:
            return tbl[rows]
        ks = sorted(tbl)
        if rows > ks[-1]:
            step = self.step if self.step is not None else (
                (tbl[ks[-1]] - tbl[ks[-2]]) / (ks[-1] - ks[-2]) if len(ks) > 1 else 0.0)
            return tbl[ks[-1]] + step * (rows - ks[-1])
        if rows < ks[0]:
            return tbl[ks[0]]
        lo = max(k for k in ks if k < rows)
        hi = min(k for k in ks if k > rows)
        return tbl[lo] + (tbl[hi] - tbl[lo]) * (rows - lo) / (hi - lo)

    def prices(self, n: int, kind: str = "chain", ctx_class: int = 0) -> list[float]:
        """[0, price of 1 row, ..., price of n rows]."""
        V = self.V.get((kind, ctx_class), {})
        d = self.delta.get((kind, ctx_class), 0.0)
        out, p = [0.0], 0.0
        for r in range(1, n + 1):
            v = V.get(r)
            if v is None:
                pr = self.prior(r, kind, ctx_class)
                v = pr + d if pr is not None else 1.0
            p = max(p, v)
            out.append(p)
        return out

    def price(self, rows: int, kind: str = "chain", ctx_class: int = 0) -> float:
        return self.prices(rows, kind, ctx_class)[rows]

    def observe(self, rows: int, ms: float, kind: str = "chain", ctx_class: int = 0) -> None:
        key, a = (kind, ctx_class), self.ema
        V = self.V.setdefault(key, {})
        V[rows] = ms if rows not in V else (1 - a) * V[rows] + a * ms
        pr = self.prior(rows, kind, ctx_class)
        if pr is not None:
            self.delta[key] = (1 - a) * self.delta.get(key, 0.0) + a * (ms - pr)

    def learned(self, kind: str = "chain", ctx_class: int = 0) -> tuple[dict, float]:
        """(learned ms by row count, learned offset) of one kind and class, for reports."""
        return dict(self.V.get((kind, ctx_class), {})), self.delta.get((kind, ctx_class), 0.0)


class ProbDeciles:
    """The draft head's probability mapped to the measured rate of acceptance, per kind of node
    ("spine", "sib") and tenth of probability: it starts at the probability itself (a prior worth
    `prior_n` rounds a bin), learns from every verified round, and halves old evidence past `cap`
    rounds a bin, so it follows the text."""

    def __init__(self, kinds=("spine", "sib", "look"), prior_n: float = 4.0, cap: float = 400.0):
        self.prior_n, self.cap = prior_n, cap
        self.cal = {k: [[0.0, 0.0] for _ in range(10)] for k in kinds}

    def q(self, kind: str, p: float) -> float:
        hits, n = self.cal[kind][min(int(p * 10), 9)]
        return (hits + self.prior_n * p) / (n + self.prior_n)

    def learn(self, kind: str, p: float, hit: bool) -> None:
        b = self.cal[kind][min(int(p * 10), 9)]
        b[0] += float(hit)
        b[1] += 1.0
        if b[1] > self.cap:
            b[0] *= 0.5
            b[1] *= 0.5


class StairCut:
    """The cut over a draft head's chain (and, with `siblings`, one second-choice leaf a depth). q of a
    spine node: the product of the calibrated first-token probabilities down to it; q of a sibling at
    depth k: the chance the walk reaches depth k, rejects the spine token there and takes the sibling.
    `kind`: the price table's kind for its blocks. `frozen`: observe nothing (a warm-up run)."""

    def __init__(self, prices: PriceModel, calibrator: ProbDeciles | None = None, siblings: bool = True,
                 max_rows: int = 32, kind: str = "chain"):
        self.prices, self.cal = prices, calibrator or ProbDeciles()
        self.siblings, self.max_rows, self.kind = siblings, max_rows, kind
        self.draft_ms = None
        self.last = None
        self.frozen = False

    def cut(self, anchor: int, tok: list, prob: list, look: tuple | None = None) -> tuple[list[int], list[int]]:
        """tok / prob: per depth the (first, second) draft token and its probability; `look`: a lookup
        continuation (tokens, confidence) or None. Returns the block in DFS pre-order: (tokens,
        parents), row 0 the anchor."""
        if look is not None and look[0]:
            return self._merged(anchor, tok, prob, list(look[0]), float(look[1]))
        D = len(tok)
        q = self.cal.q
        reach, spine_q, sib_q = 1.0, [], []
        for k in range(D):
            c1 = q("spine", prob[k][0])
            sib_q.append(reach * (1 - c1) * q("sib", prob[k][1]) if self.siblings else 0.0)
            reach *= c1
            spine_q.append(reach)
        draft = self.draft_ms or 0.0
        price = self.prices.prices(min(self.max_rows, 2 * D + 1), self.kind)
        best = (1.0 / (price[1] + draft), 0, ())
        for n in range(1, D + 1):
            gain = 1.0 + sum(spine_q[:n])
            rows = 1 + n
            if rows > self.max_rows:
                break
            score = gain / (price[rows] + draft)
            if score > best[0]:
                best = (score, n, ())
            # siblings at depths 1..n + 1 (a sibling's parent is the spine node above it), best first
            sibs = []
            for k in sorted(range(min(n + 1, D)), key=lambda k: -sib_q[k]):
                if rows + 1 > self.max_rows or sib_q[k] <= 0.0:
                    break
                g2, r2 = gain + sib_q[k], rows + 1
                s2 = g2 / (price[r2] + draft)
                if s2 <= score:
                    break
                gain, rows, score = g2, r2, s2
                sibs.append(k)
                if score > best[0]:
                    best = (score, n, tuple(sibs))
        _, n, sibs = best
        # DFS pre-order: a depth's sibling leaf before the spine child that continues the chain
        tokens, parents, kind = [anchor], [-1], [None]
        spine_row = 0
        for k in range(n + 1):
            if k in sibs:
                tokens.append(tok[k][1])
                parents.append(spine_row)
                kind.append(("sib", k))
            if k < n:
                tokens.append(tok[k][0])
                parents.append(spine_row)
                kind.append(("spine", k))
                spine_row = len(tokens) - 1
        self.last = (kind, prob, len(tokens))
        return tokens, parents

    def _merged(self, anchor: int, tok: list, prob: list, look: list, conf: float) -> tuple[list[int], list[int]]:
        """ENG-229: the spine and the lookup's chain in one block. The lookup shares the spine's nodes
        while its tokens are the spine's (j of them; a shared node's q is 1 - both wrong), then leaves
        it under the spine node at depth j (or runs on past the spine's end): its first own node is
        taken when the spine's is not and the lookup is right, each next one when the lookup is right
        again. Nodes are admitted in order of q -- a parent's q is never below its child's, so a node
        comes after its parent -- for as long as (1 + sum q) / (V(rows) + draft) rises."""
        q = self.cal.q
        pl = q("look", conf)
        D = len(tok)
        j = 0
        while j < min(D, len(look)) and look[j] == tok[j][0]:
            j += 1
        # node: [token, parent node (-1: the anchor), q, (kind, k), depth]
        nodes, reach, up = [], 1.0, -1
        for k in range(D):
            c1 = q("spine", prob[k][0])
            c = 1.0 - (1.0 - c1) * (1.0 - pl) if k < j else c1
            nodes.append([tok[k][0], up, reach * c, ("spine", k) if k >= j else ("both", k), k])
            if k == j and j < len(look):
                lq = reach * (1.0 - c1) * pl           # the lookup's own token where the spine's is wrong
            reach *= c
            up = len(nodes) - 1
        if j == D and j < len(look):
            lq = reach * pl                             # the lookup runs on past the spine's end
        if j < len(look):
            lup = j - 1                                 # the spine node above (-1: the anchor)
            for i, t in enumerate(look[j:]):
                nodes.append([t, lup, lq, ("look", j + i), j + i])
                lup, lq = len(nodes) - 1, lq * pl
        order = sorted(range(len(nodes)), key=lambda i: (-nodes[i][2], nodes[i][4]))
        draft = self.draft_ms or 0.0
        price = self.prices.prices(self.max_rows, self.kind)
        best, gain, taken, kept = (1.0 / (price[1] + draft), 0), 1.0, set(), []
        for i in order:
            if len(kept) + 2 > self.max_rows:
                break
            if nodes[i][1] >= 0 and nodes[i][1] not in taken:
                continue
            taken.add(i)
            kept.append(i)
            gain += nodes[i][2]
            score = gain / (price[len(kept) + 1] + draft)
            if score > best[0]:
                best = (score, len(kept))
        keep = set(kept[:best[1]])
        tokens, parents, kind = [anchor], [-1], [None]
        row: dict = {}

        def emit(par: int, prow: int) -> None:
            # the lookup's branch before the spine's continuation, as `chain_siblings` puts a sibling
            for i in sorted((i for i in keep if nodes[i][1] == par), key=lambda i: nodes[i][3][0] != "look"):
                tokens.append(nodes[i][0])
                parents.append(prow)
                kind.append(nodes[i][3])
                row[i] = len(tokens) - 1
                emit(i, row[i])
        emit(-1, 0)
        self.last = ("merged", kind, prob, len(tokens), conf, parents)
        return tokens, parents

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        """The last block's accepted path (its rows), what its verify and commit cost and what the draft
        cost: the calibration learns from every node the walk reached, the prices from the round."""
        if self.frozen:
            return
        if self.last[0] == "merged":
            self._observe_merged(path, round_ms, draft_ms)
            return
        kind, prob, rows = self.last
        on = {kind[r] for r in path[1:]}
        for r in range(1, rows):
            t, k = kind[r]
            if not all(("spine", i) in on for i in range(k)):
                continue
            if t == "spine":
                self.cal.learn("spine", prob[k][0], ("spine", k) in on)
            elif ("spine", k) not in on:
                self.cal.learn("sib", prob[k][1], ("sib", k) in on)
        self.prices.observe(rows, round_ms, self.kind)
        self.draft_ms = draft_ms if self.draft_ms is None else 0.9 * self.draft_ms + 0.1 * draft_ms

    def _observe_merged(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        """A merged block: a node the walk reached (its parent on the path) teaches its source -- a
        spine node the spine's bin, a lookup node the lookup's, a shared node both."""
        _, kind, prob, rows, conf, parents = self.last
        on = set(path)
        for r in range(1, rows):
            if parents[r] not in on:
                continue
            t, k = kind[r]
            hit = r in on
            if t in ("spine", "both"):
                self.cal.learn("spine", prob[k][0], hit)
            if t in ("look", "both"):
                self.cal.learn("look", conf, hit)
        self.prices.observe(rows, round_ms, self.kind)
        self.draft_ms = draft_ms if self.draft_ms is None else 0.9 * self.draft_ms + 0.1 * draft_ms
