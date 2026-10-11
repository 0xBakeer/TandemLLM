"""ENG-245: StairCut, its price model and its calibration in one place (engine/methods/staircut.py).

  * the shared cut decides every round as Flash-Next's own StairCut did before the move (a frozen copy
    below, `_OldStairCut`, from engine/models/flashnext/spec.py at 9f748434): random drafts, walks and
    round times, with and without siblings, and with `FN_TRACE=<fn_draft_trace JSON>` the recorded
    traces (prose, code, chat) -- the same tokens and parents, the same learned prices and calibration;
  * the cut itself: a confident chain kept, an unsure one dropped, a paying sibling first, calibration
    learned from the walk, prices never cheaper for more rows, the offset learned from a long context;
  * the price model: Flash-Next's file is the prior StairCut had in code; a missing file is logged once
    and prices every count alike; the nearest lower context class; interpolation between measured
    counts; the 27B's stair tables (ops/stair-tables-nvfp4.json) load in the same format;
  * no price constant left in Flash-Next's package.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods.staircut import PriceModel, ProbDeciles, StairCut  # noqa: E402
from engine.methods.walk import accept_tree  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FN_PRICES = os.path.join(ROOT, "ops", "prices", "qwen3.8-flash-next", "flashnext-3.05.dgx-spark.json")


class _OldStairCut:
    """StairCut on Flash-Next's MTP drafts: each round keeps the part of the draft that maximises
    expected committed tokens per millisecond,

        (1 + sum of q over the kept nodes) / (V(rows) + draft),

    over a spine (the MTP's greedy chain) and, with `siblings`, one leaf per depth (the MTP's second
    token at that depth, a child of the spine node above it). q of a spine node is the chance the walk
    reaches and takes it: the product of the calibrated first-token probabilities down to it; q of a
    sibling at depth k is the chance the walk reaches depth k, rejects the spine token there and takes
    the sibling. Calibration maps the draft head's probability to the measured rate of acceptance, per
    tenth of probability, starting from the probability itself (a prior of 4 rounds a bin) and learned
    online from every verified round. V(rows) is the measured verify-and-commit time a row count costs,
    started from the 2026-10-11 bench (Flash-Next 3.05, 1k) and followed online (EMA 0.1); the counts
    not measured yet follow the prior shifted by the learned offset (a long context moves every
    count), and the prices never fall with more rows. A verify of N rows reads the union of the
    experts its rows route to, so the price is a ramp, not a flat stair.
    """

    V0 = {1: 24.0, 2: 27.9, 3: 30.6, 4: 34.6, 5: 37.4, 6: 41.3}   # verify + commit, ms (bench 2026-10-11)
    STEP = 3.7                                                    # ms a row past the table

    def __init__(self, siblings: bool = True, max_rows: int = 32):
        self.siblings, self.max_rows = siblings, max_rows
        self.V: dict = {}              # learned verify + commit ms by row count (EMA 0.1)
        self.delta = 0.0               # learned shift of the prior table (the context moves it)
        self.draft_ms = None
        self.cal = {"spine": [[0.0, 0.0] for _ in range(10)], "sib": [[0.0, 0.0] for _ in range(10)]}
        self.last = None
        self.frozen = False            # observe nothing (a warm-up run of a measurement)

    def prior(self, rows: int) -> float:
        top = max(self.V0)
        return self.V0[rows] if rows in self.V0 else self.V0[top] + self.STEP * (rows - top)

    def prices(self, n: int) -> list[float]:
        """The verify + commit price of 1..n rows: learned where measured, else the prior table
        shifted by the learned offset; never cheaper for more rows (a cut must not buy a row for
        nothing because one count was measured at a busier moment than the next)."""
        out, p = [0.0], 0.0
        for r in range(1, n + 1):
            p = max(p, self.V.get(r, self.prior(r) + self.delta))
            out.append(p)
        return out

    def price(self, rows: int) -> float:
        return self.prices(rows)[rows]

    def c(self, kind: str, p: float) -> float:
        b = min(int(p * 10), 9)
        hits, n = self.cal[kind][b]
        return (hits + 4 * p) / (n + 4)

    def cut(self, anchor: int, tok: list, prob: list) -> tuple[list[int], list[int]]:
        """tok / prob: per depth the (first, second) draft token and its probability."""
        D = len(tok)
        reach, spine_q, sib_q = 1.0, [], []
        for k in range(D):
            c1 = self.c("spine", prob[k][0])
            sib_q.append(reach * (1 - c1) * self.c("sib", prob[k][1]) if self.siblings else 0.0)
            reach *= c1
            spine_q.append(reach)
        draft = self.draft_ms or 0.0
        price = self.prices(min(self.max_rows, 2 * D + 1))
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

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        if self.frozen:
            return
        kind, prob, rows = self.last
        on = {kind[r] for r in path[1:]}
        for r in range(1, rows):
            t, k = kind[r]
            reached = all(("spine", i) in on for i in range(k))
            if not reached:
                continue
            if t == "spine":
                self._learn("spine", prob[k][0], ("spine", k) in on)
            elif ("spine", k) not in on:
                self._learn("sib", prob[k][1], ("sib", k) in on)
        verify = round_ms
        self.V[rows] = verify if rows not in self.V else 0.9 * self.V[rows] + 0.1 * verify
        self.delta = 0.9 * self.delta + 0.1 * (verify - self.prior(rows))
        self.draft_ms = draft_ms if self.draft_ms is None else 0.9 * self.draft_ms + 0.1 * draft_ms

    def _learn(self, kind: str, p: float, hit: bool) -> None:
        b = self.cal[kind][min(int(p * 10), 9)]
        b[0] += float(hit)
        b[1] += 1.0
        if b[1] > 400:                         # follow the text: halve old evidence
            b[0] *= 0.5
            b[1] *= 0.5


def _new(siblings: bool) -> StairCut:
    return StairCut(PriceModel.load(FN_PRICES), siblings=siblings)


def _replay(rounds, siblings: bool) -> int:
    """rounds: (tok, prob, accepted depth, round ms, draft ms) per round; both cuts see the same."""
    a, b = _OldStairCut(siblings=siblings), _new(siblings)
    for tok, prob, acc, ms, dms in rounds:
        ta, pa = a.cut(1, tok, prob)
        tb, pb = b.cut(1, tok, prob)
        assert (ta, pa) == (tb, pb), (ta, pa, tb, pb)
        # the target: the spine token to depth `acc`, then the sibling there on odd depths, else neither
        picks, node = [-1] * len(ta), 0
        for k in range(len(tok)):
            if k >= acc:
                picks[node] = tok[k][1] if acc % 2 else -1
                break
            picks[node] = tok[k][0]
            node = next((c for c in range(len(ta)) if pa[c] == node and ta[c] == tok[k][0]), None)
            if node is None:
                break
        path, _ = accept_tree(DraftTree(tokens=ta, parents=pa), picks)
        a.observe(path, ms(len(ta)), dms)
        b.observe(path, ms(len(tb)), dms)
    assert a.V == b.prices.V[("chain", 0)] and abs(a.delta - b.prices.delta[("chain", 0)]) == 0.0
    assert a.cal == b.cal.cal and a.draft_ms == b.draft_ms
    return len(rounds)


def test_the_shared_cut_decides_as_the_old_one_on_random_drafts():
    rng = random.Random(7)
    n = 0
    for siblings in (False, True):
        for _ in range(20):
            rounds = []
            for _ in range(150):
                D = rng.choice((3, 4, 5, 6))
                p1 = [rng.random() ** rng.choice((0.2, 1, 3)) for _ in range(D)]
                tok = [[10 + k, 20 + k] for k in range(D)]
                prob = [[x, rng.random() * (1 - x)] for x in p1]
                acc = rng.randint(0, D)
                base = rng.uniform(20, 45)
                rounds.append((tok, prob, acc, lambda r, base=base: base + 3.1 * r, rng.uniform(0.5, 3.0)))
            n += _replay(rounds, siblings)
    return f"{n} rounds, chain and siblings: the same blocks, prices and calibration"


def test_the_shared_cut_decides_as_the_old_one_on_recorded_traces():
    path = os.environ.get("FN_TRACE")
    if not path:
        return "skipped (FN_TRACE names a tools/fn_draft_trace.py JSON)"
    T = json.load(open(path))["traces"]
    n = 0
    for name, tr in T.items():
        out, rounds, i = tr["out"], [], 0
        while i < len(tr["drafts"]):
            d = tr["drafts"][i]
            acc = 0
            while acc < len(d["tok"]) and i + 1 + acc < len(out) and d["tok"][acc][0] == out[i + 1 + acc]:
                acc += 1
            rounds.append((d["tok"], d["prob"], acc, lambda r: 22.9 + 3.4 * r, 1.6))
            i += acc + 1
        for siblings in (False, True):
            n += _replay(rounds, siblings)
    return f"{len(T)} recorded traces ({', '.join(T)}), {n} rounds: the same blocks"


def test_staircut():
    sc = _new(True)
    tok = [[10 + k, 20 + k] for k in range(5)]
    toks, par = sc.cut(1, tok, [[0.99, 0.005]] * 5)
    assert toks == [1, 10, 11, 12, 13, 14] and par == [-1, 0, 1, 2, 3, 4], (toks, par)
    toks, par = sc.cut(1, tok, [[0.05, 0.02]] * 5)
    assert toks == [1], toks
    toks, par = sc.cut(1, tok, [[0.3, 0.65]] + [[0.97, 0.01]] * 4)
    DraftTree(tokens=toks, parents=par).check()
    assert 20 in toks and toks.index(20) < toks.index(10), (toks, par)          # sibling before the spine
    # learning: a spine token at p 0.95 that is always rejected pulls its bin down
    before = sc.cal.q("spine", 0.95)
    for _ in range(50):
        sc.cut(1, tok, [[0.95, 0.01]] * 5)
        sc.observe([0], 30.0, 2.0)
    assert sc.cal.q("spine", 0.95) < 0.2 < before
    # prices: a count measured slow does not make the next count cheaper, and the offset moves the rest
    sc2 = _new(False)
    sc2.prices.V[("chain", 0)] = {3: 60.0}
    p = sc2.prices.prices(6)
    assert all(p[r + 1] >= p[r] for r in range(1, 6)) and p[4] >= 60.0, p
    sc3 = _new(True)
    for _ in range(80):
        sc3.cut(1, tok, [[0.99, 0.0]] * 5)
        sc3.last = (sc3.last[0], sc3.last[1], 2)
        sc3.observe([0, 1], 40.0, 2.0)               # every 2-row block cost 40 ms: a long context
    delta = sc3.prices.learned()[1]
    assert delta > 8.0 and sc3.prices.price(5) > sc3.prices.prior(5) + 8.0
    toks, par = sc3.cut(1, tok, [[0.99, 0.0]] * 5)
    assert 20 not in toks, toks                       # a sibling of q 0 is never bought
    return "confident chain kept, unsure dropped, a paying sibling first, calibration learned, prices monotone"


def test_price_model():
    pm = PriceModel.load(FN_PRICES)
    old = _OldStairCut()
    assert [pm.prior(r) for r in range(1, 12)] == [old.prior(r) for r in range(1, 12)]
    logs = []
    none = PriceModel.load(os.path.join(ROOT, "ops", "prices", "no-such-model", "x.dgx-spark.json"), log=logs.append)
    assert len(logs) == 1 and "no-such-model" in logs[0] and none.prices(4) == [0.0, 1.0, 1.0, 1.0, 1.0]
    toks, _ = StairCut(none, siblings=False).cut(1, [[10 + k, 20 + k] for k in range(4)], [[0.6, 0.1]] * 4)
    assert len(toks) == 5                             # every count alike: the full depth until priced
    t = PriceModel({"1024": {"chain": {"2": 10.0, "8": 22.0}}, "16384": {"chain": {"2": 20.0, "8": 32.0}}})
    assert t.prior(5) == 16.0 and t.prior(1) == 10.0 and t.prior(10) == 26.0          # interpolated, below, slope
    assert t.prior(5, ctx_class=20000) == 26.0 and t.prior(5, ctx_class=512) == 16.0  # nearest lower, lowest
    assert t.prior(5, kind="tree") is None
    stair = PriceModel.load(os.path.join(ROOT, "ops", "stair-tables-nvfp4.json"))
    doc = json.load(open(os.path.join(ROOT, "ops", "stair-tables-nvfp4.json")))
    assert stair.prior(16, "tree", 16384) == doc["16384"]["tree"]["16"]
    assert stair.prior(16, "tree", 20000) == doc["16384"]["tree"]["16"]
    return "Flash-Next's file = the old prior; missing file logged once; classes, interpolation; the 27B's stair tables"


def test_no_price_constant_in_the_flashnext_package():
    hits = [p for p in glob.glob(os.path.join(ROOT, "engine", "models", "flashnext", "*.py"))
            if "V0 =" in open(p).read() or "STEP =" in open(p).read()]
    assert not hits and not os.path.exists(os.path.join(ROOT, "engine", "models", "flashnext", "spec.py")), hits
    assert isinstance(ProbDeciles().q("spine", 0.5), float)
    return "no V0 / STEP in engine/models/flashnext/, spec.py gone"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<62} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<62} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
