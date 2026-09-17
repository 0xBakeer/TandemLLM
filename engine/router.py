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

# The same two numbers on the NVFP4 weight set with the trimmed 32k draft head, read off the decode
# loop's own instrumentation in tools/bench_ngram.py at 10:01 rather than off a microbenchmark:
# 3.4 ms per drafted token (2188 ms over 215 three-token drafts), and 6.2 ms per rollback on all
# five workloads. Getting the rollback wrong by 4x is not a rounding error in a router -- it is the
# term that decides how long a chain is worth proposing, and at 23 ms the router shortens chains
# that pay at 6.
MTP_MS_PER_TOKEN_TRIMMED = 3.4
ROLLBACK_MS_NVFP4 = 6.2


# The same curve after the MLPs went to four bits (SPEED-LEDGER 14:40). It is not a line, and
# reading it as one cost 5 % of the prose row at 10:37. The FP4 kernel does sixteen rows of
# tensor-core work whether one row is asked for or sixteen, so B = 1 runs a different path
# (M = 1 GEMV, 126.12 ms) and everything from B = 2 to B = 4 costs what B = 4 costs. Interpolating
# between 126.12 and 145.02 tells a router that a two-token block is 13 ms cheaper than a
# four-token one, and it is not cheaper at all; the router then shortens chains that pay.
#
# The entry for 2 is not a separate measurement. It is the kernel's own tiling asserted as a cost,
# and tools/profile_block.py should be run at B = 2 and B = 3 to confirm it.
VERIFY_MS_NVFP4 = {1: 126.12, 2: 145.02, 4: 145.02, 8: 153.52, 16: 175.10}


def verify_ms(b: int, table: dict[int, float] | None = None) -> float:
    table = table or VERIFY_MS
    keys = sorted(table)
    if b <= keys[0]:
        return table[keys[0]]
    if b >= keys[-1]:
        slope = (table[keys[-1]] - table[keys[-2]]) / (keys[-1] - keys[-2])
        return table[keys[-1]] + slope * (b - keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return table[lo] + f * (table[hi] - table[lo])
    return table[keys[-1]]


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
        self.last, self.last_n, self.last_depth = "mtp", len(draft), m
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
                 alpha: float = 0.15, verify_ms_table: dict[int, float] | None = None,
                 depth_margin: float = 0.05, head_fixed_ms: float = 0.0,
                 adaptive_depth: bool = True):
        self.ngram = ngram
        self.mtp = mtp
        self.mtp_depth = mtp_depth
        self.node_budget = node_budget
        self.mtp_ms_per_token = mtp_ms_per_token
        self.rollback_ms = rollback_ms
        self.depth_margin = depth_margin
        # A block drafter costs the same whether its block is read in full or not: one non-causal
        # pass over its own layers and one pass over the vocabulary head. So its price is fixed per
        # call rather than per proposed token, and shortening its block saves nothing. The chained
        # prediction head is the other case: it pays per token, and shortening it is a real saving.
        self.head_fixed_ms = head_fixed_ms
        self.adaptive_depth = adaptive_depth
        # Accepted tokens per call, kept directly rather than derived from a per-token rate. A
        # chained drafter's acceptance is prefix-geometric; a block drafter's is not -- measured,
        # its block of 7 at 46 % acceptance yields 3.2 tokens, which is 0.46 * 7 and not the 0.85
        # a geometric model would predict.
        self.mean_accepted = _Rate(0.5, alpha)
        # the verify curve the pricing runs on; FP8 by default, swapped for the NVFP4 one when the
        # MLPs are four bits, because four bits made width more expensive (SPEED-LEDGER 14:40)
        self.verify_table = dict(verify_ms_table) if verify_ms_table else dict(VERIFY_MS)
        keys = sorted(self.verify_table)
        self.prune_base_ms = self.verify_table[keys[0]]
        self.prune_per_node_ms = ((self.verify_table[keys[-1]] - self.verify_table[keys[0]])
                                  / max(keys[-1] - keys[0], 1))
        # how many of the tokens it expected did the lookup drafter actually get, lately
        self.calib = _Rate(1.0, alpha)
        self.rate_mtp = _Rate(0.6, alpha)
        self.last = None
        self.last_n = 0
        self.last_expected = 0.0
        self.last_depth = 0          # the head chain length this step actually paid for
        self.last_tree = None        # what the lookup drafter offered, chosen or not
        self.stats = {"ngram": 0, "mtp": 0, "merged": 0, "ngram_tokens": 0, "mtp_tokens": 0,
                      "declined": 0, "depth_hist": {}}

    # --- state ---------------------------------------------------------------------------------

    def reset(self) -> None:
        self.ngram.reset()
        self.mtp.reset()
        self.last = None
        self.last_tree = None

    def prime(self, tokens: list[int]) -> None:
        self.ngram.prime(tokens)

    def sync(self, tokens, hidden, first_pos) -> None:
        self.mtp.sync(tokens, hidden, first_pos)

    def observe(self, tokens: list[int]) -> None:
        """Learn from the block, including from the drafter that did not get to write it.

        The calibration factor is the only thing standing between the lookup drafter's own estimate
        and reality, and updating it only when the drafter is chosen is a trap: on a `quote`
        workload the drafter alone runs at 43.19 tok/s against the block drafter's 30.50, and the
        router picked it **zero times** out of nineteen blocks, so the factor never moved off 1.00
        and the drafter never got a turn. A policy that only learns about what it chose will keep
        choosing it.

        There is no need for exploration here. The tree was built before the choice was made, the
        tokens the target actually wrote are in this call, and `accepted_against` is a walk down a
        tree of at most sixteen nodes. The counterfactual is free, so it is taken on every step.
        """
        accepted = max(0, len(tokens) - 1)
        if self.last_tree is not None and self.last_expected > 0:
            would_have = self.last_tree.accepted_against(list(tokens))
            self.calib.update(would_have, self.last_expected)
        if self.last == "mtp" and self.last_n:
            self.rate_mtp.update(accepted, self.last_n)
            self.mean_accepted.update(accepted, self.last_n)
        self.last_tree = None
        self.ngram.observe(tokens)

    # --- pricing -------------------------------------------------------------------------------

    def _verify_ms(self, b: int) -> float:
        return verify_ms(b, self.verify_table)

    def _value(self, expected: float, nodes: int, cost_ms: float, p_reject: float) -> float:
        """Tokens per second if every step looked like this one."""
        ms = self._verify_ms(nodes + 1) + cost_ms + self.rollback_ms * p_reject
        return (expected + 1.0) / (ms / 1000.0)

    def _mtp_expected(self, d: int) -> float:
        """Expected accepted tokens from a chain of `d`, under a per-token acceptance rate.

        A chain is accepted prefix-first, so the i-th token only counts if the i-1 before it were
        accepted too: the expectation is the geometric sum, not `rate * d`. Getting this wrong is
        what makes a router propose long chains on text where it should propose short ones.
        """
        rate = self.rate_mtp.value
        return sum(rate ** i for i in range(1, d + 1))

    def _mtp_value(self, d: int) -> float:
        rate = self.rate_mtp.value
        cost = self.mtp_ms_per_token * d + self.head_fixed_ms
        if self.adaptive_depth:
            return self._value(self._mtp_expected(d), d, cost, 1.0 - rate ** d)
        expected = min(self.mean_accepted.value * d, float(d))
        return self._value(expected, d, cost, 1.0 if expected < d else 0.0)

    def _best_mtp_depth(self) -> tuple[int, float]:
        """The chain length worth proposing right now, priced rather than fixed.

        On German the head's acceptance is low enough that a three-token chain is *slower than not
        drafting at all*: it spends 50 ms of drafting and 22 ms of rollback to buy a third of a
        token (measured in the simulator over the recorded traces). A one-token chain on the same
        text still pays. So the depth is chosen per step.

        The router does not have the option of proposing nothing while it holds a head. The head
        conditions its next draft on the target's hidden state at the last committed position, and
        a step that verifies no block never computes that row (engine/spec.py). Depth one is the
        floor, and where even depth one does not pay, the honest reading is that the drafter is
        wrong for the text rather than that the router should switch off.
        """
        values = [(d, self._mtp_value(d)) for d in range(1, self.mtp_depth + 1)]
        best_v = max(v for _, v in values)
        # Ties go to the longer chain. The differences this is deciding between are a few per cent,
        # and the cost model's own terms are not known to a few per cent: the verify curve is flat
        # from B = 2 to B = 4 because the FP4 kernel tiles sixteen rows regardless, so tokens two
        # and three of a chain are close to free and their upside is real. Shortening on a margin
        # this thin is what cost 5 % of the prose row at 10:37.
        if not self.adaptive_depth:
            return self.mtp_depth, self._mtp_value(self.mtp_depth)
        best_d = max(d for d, v in values if v >= best_v * (1.0 - self.depth_margin))
        return best_d, next(v for d, v in values if d == best_d)

    # --- proposing -----------------------------------------------------------------------------

    def propose_tree(self, context: list[int], k: int):
        """The block this router wants verified, as a tree. Needs a tree-capable verify path."""
        from engine.tree import DraftTree

        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        depth, _ = self._best_mtp_depth()
        self.stats["depth_hist"][depth] = self.stats["depth_hist"].get(depth, 0) + 1
        self.last_depth = depth
        chain = self.mtp.propose(context, depth) if depth > 0 else []
        mtp_tree = DraftTree.chain(context[-1], chain, source="mtp") if chain else None
        if tree is None:
            self.last, self.last_n = ("mtp", len(chain)) if chain else (None, 0)
            self.stats["mtp" if chain else "declined"] += 1
            return mtp_tree
        if mtp_tree is None:
            self.last, self.last_n = "ngram", tree.n_draft
            self.stats["ngram"] += 1
            return tree
        merged = tree.merge(mtp_tree).prune(self.node_budget,
                                            per_node_ms=self.prune_per_node_ms,
                                            base_ms=self.prune_base_ms)
        self.last, self.last_n = "merged", merged.n_draft
        self.stats["merged"] += 1
        return merged

    def propose(self, context: list[int], k: int) -> list[int]:
        """Chain interface, for the verify path that exists today: price both, take the better."""
        depth, v_mtp = self._best_mtp_depth()
        self.stats["depth_hist"][depth] = self.stats["depth_hist"].get(depth, 0) + 1
        self.last_depth = 0
        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        self.last_tree = tree
        self.last_expected = tree.expected_accepted() if tree is not None else 0.0
        if tree is not None and tree.n_draft:
            expected = self.last_expected * self.calib.value
            # a chain-only verify can only take one branch, so price the best one
            best = self._best_branch(tree, k)
            n = len(best)
            p_reject = min(1.0, max(0.0, 1.0 - expected / max(n, 1)))
            if self._value(min(expected, n), n, 0.0, p_reject) >= v_mtp:
                self.last_depth = 0
                self.last, self.last_n = "ngram", n
                self.stats["ngram"] += 1
                self.stats["ngram_tokens"] += n
                return best
        m = min(depth, k)
        if m <= 0:
            self.stats["declined"] += 1
            self.last, self.last_n = None, 0
            return []
        draft = self.mtp.propose(context, m)
        self.last, self.last_n, self.last_depth = "mtp", len(draft), m
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
