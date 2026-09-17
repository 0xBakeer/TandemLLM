"""Choosing a drafter, and a draft length, once per step.

Two drafters with opposite cost structures are available. A suffix memory proposes long blocks for
nothing but only where the context repeats itself, which on this board is 5-10 % of steps. The
checkpoint's prediction head proposes on every step but reads 3.0 GB per token it proposes -- about
17 ms, against a verify step of 151-180 ms -- so a draft it gets wrong is not free.

The choice is an expected-value calculation, not a heuristic, and every term in it is something the
engine has already measured:

    value(drafter, m)  =  E[accepted | drafter, m] / (verify(m + 1) + cost(drafter, m))

`verify` is the measured curve in notes/SPEED-LEDGER.md; `cost` is zero for the suffix memory and
linear in m for the prediction head; `E[accepted]` is estimated from that drafter's own recent
history, and for the suffix memory it is conditioned on the length of the match it found, which is
the one signal available before committing.

The router deliberately sees nothing about what the text says. It sees match lengths, recent
acceptance and block economics.
"""

from __future__ import annotations

from engine.drafters import Drafter

# verify(B) in seconds, measured on this board; see notes/SPEED-LEDGER.md, "verify cost against
# block length". Linear between the measured points, flat-extrapolated past the ends.
VERIFY_MS = {1: 151.0, 2: 161.4, 4: 164.9, 6: 167.4, 8: 169.7, 12: 173.6, 16: 179.4, 32: 209.8}
MTP_MS_PER_TOKEN = 16.6      # measured: 846 ms of drafting over 17 three-token proposals
ROLLBACK_MS = 22.0           # measured, near-flat in the accepted prefix length


def verify_ms(b: int) -> float:
    keys = sorted(VERIFY_MS)
    if b <= keys[0]:
        return VERIFY_MS[keys[0]]
    if b >= keys[-1]:
        return VERIFY_MS[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return VERIFY_MS[lo] + f * (VERIFY_MS[hi] - VERIFY_MS[lo])
    return VERIFY_MS[keys[-1]]


class _Rate:
    """Exponentially weighted acceptance, kept per drafter."""

    def __init__(self, init: float, alpha: float = 0.15):
        self.value = init
        self.alpha = alpha

    def update(self, accepted: int, proposed: int) -> None:
        if proposed:
            self.value += self.alpha * (accepted / proposed - self.value)


class RouterDrafter(Drafter):
    """Runs the suffix memory first and falls back to the prediction head.

    The suffix memory is tried first because asking it costs a dictionary lookup. It is used when
    the expected value of its proposal beats the head's, which given its zero cost means: when the
    match it found is long enough that its measured acceptance at that match length pays for the
    extra verify width and the rollback it risks.
    """

    name = "router"

    def __init__(self, engram, mtp, mtp_depth: int = 3, min_engram_order: int = 4):
        self.engram = engram
        self.mtp = mtp
        self.mtp_depth = mtp_depth
        self.min_engram_order = min_engram_order
        self.rate_engram = _Rate(0.5)
        self.rate_mtp = _Rate(0.6)
        self.last = None
        self.last_n = 0
        self.stats = {"engram": 0, "mtp": 0, "engram_tokens": 0, "mtp_tokens": 0}

    def reset(self) -> None:
        self.engram.reset()
        self.mtp.reset()
        self.last = None

    def prime(self, tokens: list[int]) -> None:
        self.engram.prime(tokens)

    def sync(self, tokens, hidden, first_pos) -> None:
        self.mtp.sync(tokens, hidden, first_pos)

    def observe(self, tokens: list[int]) -> None:
        # `tokens` is the accepted block: the drafted prefix plus the model's own token.
        accepted = max(0, len(tokens) - 1)
        if self.last == "engram":
            self.rate_engram.update(accepted, self.last_n)
        elif self.last == "mtp":
            self.rate_mtp.update(accepted, self.last_n)
        self.engram.observe(tokens)

    def _value(self, rate: float, m: int, cost_ms: float) -> float:
        expected = rate * m
        risk = ROLLBACK_MS * (1.0 - rate ** m)
        return expected / (verify_ms(m + 1) + cost_ms + risk)

    def propose(self, context: list[int], k: int) -> list[int]:
        draft = self.engram.propose(context, k)
        if draft and len(draft) >= 1:
            order = self.engram.last_order
            if order >= self.min_engram_order:
                v_engram = self._value(self.rate_engram.value, len(draft), 0.0)
                m = min(self.mtp_depth, k)
                v_mtp = self._value(self.rate_mtp.value, m, MTP_MS_PER_TOKEN * m)
                if v_engram >= v_mtp:
                    self.last, self.last_n = "engram", len(draft)
                    self.stats["engram"] += 1
                    self.stats["engram_tokens"] += len(draft)
                    return draft
        m = min(self.mtp_depth, k)
        if m <= 0:
            return []
        draft = self.mtp.propose(context, m)
        self.last, self.last_n = "mtp", len(draft)
        self.stats["mtp"] += 1
        self.stats["mtp_tokens"] += len(draft)
        return draft


# ---------------------------------------------------------------------------------------------
# Second version, 2026-09-17. The first router chose a drafter; this one prices them.
# ---------------------------------------------------------------------------------------------


class MergedRouter(Drafter):
    """Spend one block's node budget across a lookup tree and the block drafter's chain.

    The first router's measured defect was not its shape but its evidence: it estimated the suffix
    memory's acceptance from single-digit sample counts, and on the steps where it guessed wrong it
    spent a whole verify on a lookup that did not pay (SPEED-LEDGER 12:25, -0.8 tok/s on prose,
    -2.2 on edit). Two things fix that, and neither is a heuristic.

    **The estimate comes from the drafter, not from a counter.** `NgramDrafter` returns a tree whose
    scores are path probabilities, so the expected accepted length of its proposal is a number it
    computes per step from the counts it actually found. All the router has to learn is one scalar:
    how optimistic those probabilities have been lately. A single calibration factor needs far less
    evidence than a per-drafter acceptance rate.

    **Nothing is winner-take-all once the verify path can take a tree.** Merging costs the union of
    the two proposals -- shared prefixes are shared nodes -- and at 1.896 ms per node a wrong lookup
    costs a few milliseconds instead of a step. Until `forward_tree` exists the router still has to
    pick a chain, and then it picks the one with the higher expected value under the measured curve;
    `propose_tree` is the same decision with the merge left in, and is what the tree verify will
    call.
    """

    name = "merged"

    def __init__(self, ngram, mtp, mtp_depth: int = 3, node_budget: int = 16,
                 mtp_ms_per_token: float = MTP_MS_PER_TOKEN, rollback_ms: float = ROLLBACK_MS,
                 alpha: float = 0.15):
        self.ngram = ngram
        self.mtp = mtp
        self.mtp_depth = mtp_depth
        self.node_budget = node_budget
        self.mtp_ms_per_token = mtp_ms_per_token
        self.rollback_ms = rollback_ms
        # how many of the tokens it expected did the lookup drafter actually get, lately
        self.calib = _Rate(1.0, alpha)
        self.rate_mtp = _Rate(0.6, alpha)
        self.last = None
        self.last_n = 0
        self.last_expected = 0.0
        self.stats = {"ngram": 0, "mtp": 0, "merged": 0,
                      "ngram_tokens": 0, "mtp_tokens": 0, "declined": 0}

    # --- state ---------------------------------------------------------------------------------

    def reset(self) -> None:
        self.ngram.reset()
        self.mtp.reset()
        self.last = None

    def prime(self, tokens: list[int]) -> None:
        self.ngram.prime(tokens)

    def sync(self, tokens, hidden, first_pos) -> None:
        self.mtp.sync(tokens, hidden, first_pos)

    def observe(self, tokens: list[int]) -> None:
        accepted = max(0, len(tokens) - 1)
        if self.last == "ngram" and self.last_expected > 0:
            self.calib.update(accepted, self.last_expected)
        elif self.last == "mtp" and self.last_n:
            self.rate_mtp.update(accepted, self.last_n)
        self.ngram.observe(tokens)

    # --- pricing -------------------------------------------------------------------------------

    def _value(self, expected: float, nodes: int, cost_ms: float, p_reject: float) -> float:
        """Tokens per second if every step looked like this one."""
        ms = verify_ms(nodes + 1) + cost_ms + self.rollback_ms * p_reject
        return (expected + 1.0) / (ms / 1000.0)

    def _mtp_value(self) -> float:
        rate = self.rate_mtp.value
        m = self.mtp_depth
        expected = rate * m
        return self._value(expected, m, self.mtp_ms_per_token * m, 1.0 - rate ** m)

    # --- proposing -----------------------------------------------------------------------------

    def propose_tree(self, context: list[int], k: int):
        """The block this router wants verified, as a tree. Needs a tree-capable verify path."""
        from engine.tree import DraftTree

        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        chain = self.mtp.propose(context, self.mtp_depth) if self.mtp_depth > 0 else []
        mtp_tree = DraftTree.chain(context[-1], chain, source="mtp") if chain else None
        if tree is None:
            self.last, self.last_n = ("mtp", len(chain)) if chain else (None, 0)
            self.stats["mtp" if chain else "declined"] += 1
            return mtp_tree
        expected = tree.expected_accepted() * self.calib.value
        if mtp_tree is None:
            self.last, self.last_expected = "ngram", max(expected, 1e-6)
            self.stats["ngram"] += 1
            return tree
        merged = tree.merge(mtp_tree).prune(self.node_budget)
        self.last, self.last_expected = "merged", max(merged.expected_accepted() * self.calib.value,
                                                      1e-6)
        self.stats["merged"] += 1
        return merged

    def propose(self, context: list[int], k: int) -> list[int]:
        """Chain interface, for the verify path that exists today: price both, take the better."""
        v_mtp = self._mtp_value()
        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        if tree is not None and tree.n_draft:
            expected = tree.expected_accepted() * self.calib.value
            # a chain-only verify can only take one branch, so price the best one
            best = self._best_branch(tree, k)
            n = len(best)
            p_reject = min(1.0, max(0.0, 1.0 - expected / max(n, 1)))
            if self._value(min(expected, n), n, 0.0, p_reject) >= v_mtp:
                self.last, self.last_n, self.last_expected = "ngram", n, max(expected, 1e-6)
                self.stats["ngram"] += 1
                self.stats["ngram_tokens"] += n
                return best
        m = min(self.mtp_depth, k)
        if m <= 0:
            self.stats["declined"] += 1
            self.last, self.last_n = None, 0
            return []
        draft = self.mtp.propose(context, m)
        self.last, self.last_n = "mtp", len(draft)
        self.stats["mtp"] += 1
        self.stats["mtp_tokens"] += len(draft)
        return draft

    @staticmethod
    def _best_branch(tree, k: int) -> list[int]:
        node, out = 0, []
        while len(out) < k:
            kids = tree.children_of(node)
            if not kids:
                break
            node = max(kids, key=lambda c: tree.scores[c])
            out.append(tree.tokens[node])
        return out
