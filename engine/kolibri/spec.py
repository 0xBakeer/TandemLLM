"""Speculative decoding for Kolibri-1: draft sources, StairCut, and the row source the loop reads.

THE LOOP. The decode loop asks for the next logits row after each token it picks. `SpecRows`
answers either with one decode step or with a row of a verified block: when a draft source
proposes a tree for the current token, the whole tree is verified in one pass
(`engine/kolibri/verify.py`), and the loop then walks it. Each picked token selects the child that
carries it, and that child's row is the next row; a token with no such child ends the walk, the
path walked so far is committed to the cache, and the next proposal starts from that token.

The loop picks every token from a row that is bit-equal to the decode step's row for the same
token after the same tokens. That is the verifier's contract, checked at load. So the text cannot
depend on the drafts: greedy gives the same tokens, a seeded sampler the same draws, penalties and
the grammar see the same history. A draft only decides which rows are computed together.

STAIRCUT. A verify of R rows costs more than one decode step and less than R of them; on
Kolibri's MoE the curve is a ramp (every row brings its own experts), not Qwen's flat stretch. The
price table holds the measured verify time per row count at a few context lengths
(`tools/kolibri_spec_check.py prices`). For each proposed tree the cut admits nodes best first
(with their ancestors) and keeps the prefix with the most expected tokens per millisecond:

    value(n) = (expected accepted nodes + 1) / (verify_ms(n + 1, ctx) + draft_ms + overhead_ms)

and fires only when the best value beats one decode step, 1 / decode_ms(ctx), by `margin`.
Expected accepted nodes = the sum of the kept nodes' path probabilities, after calibration.

CALIBRATION. A lookup tree's node scores are vote shares, and a vote share is not an acceptance
rate. Each node's probability is rebuilt as the product along its path of `rho * share`, where
`rho` is a per-level continuation rate learned per (source, match length) from what the verify
accepted (a Beta count, decayed so it follows the text).

DRAFT SOURCES. Anything with this shape can feed the cut, alone or merged with the others:

    name: str
    cost_ms: float                      # what one proposal costs (a lookup: ~0.1)
    prime(ctx), observe(tokens)         # the sequence so far, then each new token
    propose(ctx, budget) -> DraftTree | None   # anchor ctx[-1], scores = path probabilities
    feedback(tree, path)                # the nodes the verify accepted (indices into `tree`)

`LookupSource` is the suffix lookup (prompt, output, the session store, a corpus in Kolibri's ids).
A block drafter (DSpark-style, `tools/kd_train.py` trains one) plugs in as a
second source: its tree is merged with the lookup's (`DraftTree.merge`, shared prefixes are shared
rows) and the cut prices the merge with its `cost_ms`. A drafter that reads the target's hidden
states takes `Verifier.hidden` (the final-normed rows of the last verify) and the engine's decode
output; the residual after given layers comes from `KolibriEngine.set_taps` (decode graph, prefill
chunks) and `Verifier.taps` (verify rows).
"""

from __future__ import annotations

import json
import os
import time

import torch

from engine.tree import DraftTree

#: row counts the verify graphs are captured for; a cut ends on one of them
SIZES = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32)


# ------------------------------------------------------------------------------ prices
class PriceTable:
    """Measured ms per verify row count and per decode step, by context length.

    File (ops/kolibri-stair.json):
        {"decode": {"1024": 14.1, ...},
         "chain": {"1024": {"2": 18.0, ...}, ...}, "tree": {...},
         "overhead_ms": 0.4, "measured": "..."}
    Rows and context are interpolated linearly; past the last row count the last slope continues;
    outside the context range the nearest class is used.
    """

    def __init__(self, doc: dict):
        self.doc = doc
        self.decode = {int(k): float(v) for k, v in doc["decode"].items()}
        self.curves = {kind: {int(c): {int(r): float(ms) for r, ms in rows.items()}
                              for c, rows in doc.get(kind, {}).items()}
                       for kind in ("chain", "tree")}
        if not self.curves["tree"]:
            self.curves["tree"] = self.curves["chain"]
        self.overhead_ms = float(doc.get("overhead_ms", 0.4))
        self.ctxs = sorted(self.decode)

    @classmethod
    def load(cls, path: str) -> "PriceTable":
        with open(os.path.expanduser(path)) as f:
            return cls(json.load(f))

    @staticmethod
    def _interp(table: dict[int, float], x: float) -> float:
        ks = sorted(table)
        if x <= ks[0]:
            return table[ks[0]]
        if x >= ks[-1]:
            if len(ks) == 1:
                return table[ks[0]]
            slope = (table[ks[-1]] - table[ks[-2]]) / (ks[-1] - ks[-2])
            return table[ks[-1]] + slope * (x - ks[-1])
        for lo, hi in zip(ks, ks[1:]):
            if lo <= x <= hi:
                f = (x - lo) / (hi - lo)
                return table[lo] + f * (table[hi] - table[lo])
        return table[ks[-1]]

    def _by_ctx(self, per_ctx: dict[int, float], ctx: int) -> float:
        cs = sorted(per_ctx)
        if ctx <= cs[0]:
            return per_ctx[cs[0]]
        if ctx >= cs[-1]:
            return per_ctx[cs[-1]]
        for lo, hi in zip(cs, cs[1:]):
            if lo <= ctx <= hi:
                f = (ctx - lo) / (hi - lo)
                return per_ctx[lo] + f * (per_ctx[hi] - per_ctx[lo])
        return per_ctx[cs[-1]]

    def decode_ms(self, ctx: int) -> float:
        return self._by_ctx(self.decode, ctx)

    def verify_ms(self, rows: int, ctx: int, tree: bool = False) -> float:
        curve = self.curves["tree" if tree else "chain"]
        return self._by_ctx({c: self._interp(r, rows) for c, r in curve.items()}, ctx)


def default_prices() -> PriceTable:
    """Before a measurement: the shape a byte count predicts (a verify of R rows reads 1, 1.2, 1.53,
    2.06, 2.88, 4.04 times one token's bytes at 1, 2, 4, 8, 16, 32 rows) on a 14 ms step."""
    ramp = {1: 1.0, 2: 1.2, 4: 1.53, 8: 2.06, 16: 2.88, 32: 4.04}
    out = {"decode": {}, "chain": {}, "overhead_ms": 0.5, "measured": "study estimate, not measured"}
    for c, ms in ((1024, 14.1), (8192, 14.8), (32768, 17.1)):
        out["decode"][str(c)] = ms
        out["chain"][str(c)] = {str(r): ms * f for r, f in ramp.items()}
    return PriceTable(out)


# ------------------------------------------------------------------------------ calibration
class Rho:
    """Per-level continuation rate of a lookup line per (source, match length): Beta counts.

    Every proposal's top line is scored against the tokens that really followed, whether the cut
    fired it or not (a bucket the cut never fires would otherwise never learn). Priors: a long
    match inside the request is usually a copy; a corpus match is a guess."""

    PRIORS = {"local": {3: (2.0, 2.0), 4: (2.5, 1.5), 5: (3.0, 1.0)},
              "both": {3: (2.0, 2.0), 4: (2.5, 1.5), 5: (3.0, 1.0)},
              "corpus": {5: (1.0, 2.0)}}

    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.c: dict[tuple, list] = {}

    def key(self, src: str, mlen: int) -> tuple:
        return (src, min(int(mlen), 8))

    def prior(self, src: str, mlen: int) -> tuple:
        table = self.PRIORS.get(src, self.PRIORS["corpus"])
        best = None
        for m in sorted(table):
            if mlen >= m:
                best = table[m]
        return best or table[min(table)]

    def __call__(self, src: str, mlen: int) -> float:
        s, f = self.c.get(self.key(src, mlen), (0.0, 0.0))
        a, b = self.prior(src, mlen)
        return (s + a) / (s + f + a + b)

    def update(self, src: str, mlen: int, accepted: int, stopped: bool) -> None:
        k = self.key(src, mlen)
        e = self.c.setdefault(k, [0.0, 0.0])
        e[0] = e[0] * self.decay + accepted
        e[1] = e[1] * self.decay + (1.0 if stopped else 0.0)

    def report(self) -> dict:
        return {f"{s}/{m}": round(self(s, m), 3) for (s, m) in sorted(self.c)}


# ------------------------------------------------------------------------------ the lookup source
class _Bounded:
    """A suffix store asked at most as deep as can still match.

    The n-gram ending at token i has the (n-1)-gram ending at i-1 as its prefix, so the longest
    match at i is at most the longest at i-1 plus one. On fresh text that bound is below the
    corpus's minimum order most of the time, and the binary searches are skipped. A bound that
    is stale (a new request, a store rebuilt in the background) only costs a missed proposal."""

    def __init__(self, store):
        self.store = store
        self.max_order = getattr(store, "max_order", 8)
        self.reset()

    def reset(self):
        self.last_n = None
        self.bound = self.max_order
        self.skipped = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def lookup(self, context, min_order: int, max_samples: int = 64):
        n = len(context)
        cap = self.max_order
        if self.last_n is not None and 0 < n - self.last_n <= 64:
            cap = min(cap, self.bound + (n - self.last_n))
        self.last_n = n
        if cap < min_order:
            self.bound = cap
            self.skipped += 1
            return 0, []
        found, pos = self.store.lookup(context[-cap:], min_order, max_samples=max_samples)
        self.bound = found if pos else min_order - 1
        return found, pos


class LookupSource:
    """The suffix lookup: the request's own tokens (prompt + output so far), the engine's session
    store, and a static corpus, all in Kolibri's ids (`engine/drafters/ngram.py`)."""

    name = "lookup"
    cost_ms = 0.15

    def __init__(self, corpus: str | None = None, tokenizer_sha: str | None = None,
                 min_order: int = 3, min_corpus_order: int = 6, max_depth: int = 31,
                 branch_top_k: int = 2, store=None, rho: Rho | None = None):
        from engine.drafters.ngram import NgramDrafter
        self.d = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=tokenizer_sha,
                              min_order=min_order, min_corpus_order=min_corpus_order,
                              max_depth=max_depth, branch_top_k=branch_top_k)
        if store is not None:
            self.d.add_store(store)
        self._bounded = []
        c = self.d.corpus
        if c is not None and hasattr(c, "stores"):
            c.stores = [_Bounded(x) for x in c.stores]
            self._bounded = list(c.stores)
        elif c is not None:
            self.d.corpus = _Bounded(c)
            self._bounded = [self.d.corpus]
        self.rho = rho or Rho()
        self._lines: list = []            # (start, line, src, mlen) not yet scored

    def prime(self, ctx: list[int]) -> None:
        self.d.prime(ctx)
        self._lines = []
        for b in self._bounded:
            b.reset()

    def observe_ctx(self, ctx: list[int]) -> None:
        """The index follows `ctx` (which only grows during a request); earlier lines are scored
        against what followed them."""
        loc = self.d.local
        if len(ctx) > len(loc.tokens):
            loc.extend(ctx[len(loc.tokens):])
        keep = []
        for start, line, src, mlen in self._lines:
            got = ctx[start:start + len(line)]
            m = 0
            while m < len(got) and got[m] == line[m]:
                m += 1
            if m < len(got):                      # a token the line did not have
                self.rho.update(src, mlen, m, True)
            elif m == len(line):                  # the whole line came true
                self.rho.update(src, mlen, m, False)
            else:
                keep.append((start, line, src, mlen))
        self._lines = keep[-64:]

    def propose(self, ctx: list[int], budget: int) -> DraftTree | None:
        self.observe_ctx(ctx)
        mlen, cands = self.d.candidates(ctx, budget)
        if not cands:
            return None
        src = self.d.last_source
        if self.d.last_top:
            self._lines.append((len(ctx), list(self.d.last_top[:budget]), src, mlen))
        r = self.rho(src, mlen)
        tree = self.d.build_tree(ctx[-1], cands, budget, rho=r)
        if tree.n_draft == 0:
            return None
        return tree

    def feedback(self, tree: DraftTree, path: list[int]) -> None:
        """The verify's outcome; the calibration learns from every line in `observe_ctx`."""


# ------------------------------------------------------------------------------ StairCut
class StairCut:
    """Cut a tree to the node count with the most expected tokens per ms on the measured curve."""

    def __init__(self, prices: PriceTable, max_rows: int = 32, margin: float = 0.03,
                 sizes=SIZES):
        self.prices = prices
        self.max_rows = int(max_rows)
        self.margin = float(margin)
        self.sizes = tuple(s for s in sizes if s <= self.max_rows)

    def cut(self, tree: DraftTree, ctx_len: int, draft_ms: float = 0.0):
        """(tree cut to its best size, value in tokens/s, plain value) or (None, 0, plain)."""
        plain = 1000.0 / self.prices.decode_ms(ctx_len)
        if tree is None or tree.n_draft == 0:
            return None, 0.0, plain
        order = sorted(range(1, len(tree.tokens)), key=lambda i: -tree.scores[i])
        keep, gained = {0}, 0.0
        best_keep, best_v = None, 0.0
        has_branch = False
        for i in order:
            add = [n for n in tree.path(i) if n not in keep]
            if len(keep) + len(add) > self.max_rows:
                continue
            keep.update(add)
            gained += sum(tree.scores[n] for n in add)
            rows = len(keep)
            if rows not in self.sizes:
                continue
            branched = has_branch or _branched(tree, keep)
            has_branch = branched
            ms = (self.prices.verify_ms(rows, ctx_len, tree=branched) + draft_ms
                  + self.prices.overhead_ms)
            v = (gained + 1.0) * 1000.0 / ms
            if v > best_v:
                best_v, best_keep = v, set(keep)
        if best_keep is None or best_v <= plain * (1.0 + self.margin):
            return None, best_v, plain
        return tree.subset(best_keep), best_v, plain


def _branched(tree: DraftTree, keep: set) -> bool:
    seen = set()
    for i in keep:
        if i == 0:
            continue
        p = tree.parents[i]
        if p in seen:
            return True
        seen.add(p)
    return False


class KolibriSpec:
    """The draft sources and the cut: one proposal per call, or None (decode one token)."""

    def __init__(self, sources: list, cut: StairCut, max_rows: int = 32):
        self.sources = list(sources)
        self.cut = cut
        self.max_rows = int(max_rows)
        self.stats = {"proposals": 0, "fired": 0, "declined": 0, "rows": 0, "accepted": 0,
                      "propose_ms": 0.0, "verify_rounds": 0}
        self._fired: list = []

    def prime(self, ctx: list[int]) -> None:
        for s in self.sources:
            s.prime(ctx)

    def propose(self, ctx: list[int], room: int) -> DraftTree | None:
        """A cut tree anchored at ctx[-1], at most `room` rows, or None."""
        t0 = time.perf_counter()
        self.stats["proposals"] += 1
        budget = min(self.max_rows, room) - 1
        if budget < 1:
            return None
        tree, cost, fired = None, 0.0, []
        for s in self.sources:
            t = s.propose(ctx, budget)
            if t is None or t.n_draft == 0:
                continue
            fired.append(s)
            cost += s.cost_ms
            tree = t if tree is None else tree.merge(t)
        self.cut.max_rows = min(self.max_rows, room)
        best, _, _ = self.cut.cut(tree, len(ctx), cost) if tree is not None else (None, 0, 0)
        self.stats["propose_ms"] += (time.perf_counter() - t0) * 1e3
        if best is None:
            self.stats["declined"] += 1
            self._fired = []
            return None
        self.stats["fired"] += 1
        self.stats["rows"] += len(best.tokens)
        self._fired = fired
        return best

    def feedback(self, tree: DraftTree, path: list[int]) -> None:
        self.stats["accepted"] += len(path) - 1
        self.stats["verify_rounds"] += 1
        for s in self._fired:
            s.feedback(tree, path)

    def report(self) -> dict:
        st = dict(self.stats)
        st["propose_ms"] = round(st["propose_ms"], 1)
        if st["verify_rounds"]:
            st["accepted_per_round"] = round(st["accepted"] / st["verify_rounds"], 2)
            st["rows_per_round"] = round(st["rows"] / st["verify_rounds"], 2)
        for s in self.sources:
            rho = getattr(s, "rho", None)
            if rho is not None:
                st[f"{s.name}_rho"] = rho.report()
        return st


# ------------------------------------------------------------------------------ the loop's rows
class SpecRows:
    """The next logits row for the decode loop: a decode step, or a row of a verified tree.

    `start(ctx, row)` after the prefill (ctx = the prompt); then `next(tok, ctx)` with ctx already
    ending in `tok`, the token the loop just picked; `settle()` before any direct `eng.decode` the
    loop makes itself and at the end. Rows [0, kv.length) always hold ctx[:kv.length] after a
    settle.
    """

    def __init__(self, eng, verifier=None, spec: KolibriSpec | None = None):
        self.eng = eng
        self.v = verifier
        self.spec = spec if verifier is not None else None
        self.pend = None                 # (tree, logits, kids, path)
        self.stats = {"tokens": 0, "from_verify": 0, "decodes": 0}

    @property
    def on(self) -> bool:
        return self.spec is not None

    def start(self, ctx: list[int]) -> None:
        self.pend = None
        if self.spec is not None:
            self.spec.prime(ctx)

    def next(self, tok: int, ctx: list[int]) -> torch.Tensor:
        self.stats["tokens"] += 1
        if self.pend is not None:
            tree, lg, kids, path = self.pend
            child = kids.get((path[-1], tok))
            if child is not None:
                path.append(child)
                self.stats["from_verify"] += 1
                return lg[child]
            self.settle()
        eng = self.eng
        if self.spec is not None:
            room = min(self.v.max_rows, eng.max_len - eng.kv.length)
            tree = self.spec.propose(ctx, room) if room >= 2 else None
            if tree is not None:
                assert tree.tokens[0] == tok
                lg = self.v.verify(tree.tokens, tree.parents)
                kids = {(p, tree.tokens[i]): i for i, p in enumerate(tree.parents) if i}
                self.pend = (tree, lg, kids, [0])
                return lg[0]
        self.stats["decodes"] += 1
        return eng.decode(tok)

    def settle(self) -> None:
        if self.pend is None:
            return
        tree, _, _, path = self.pend
        self.pend = None
        self.v.commit(path)
        self.spec.feedback(tree, path)

    def report(self) -> dict:
        out = dict(self.stats)
        if self.spec is not None:
            out.update(self.spec.report())
        return out
