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

# The same board, the same weights, measured through `forward_tree` instead (tools/verify_tree.py
# --curve, SPEED-LEDGER 11:47). Keyed by nodes INCLUDING the anchor, which is what a tree calls
# `n_draft + 1`. Two things in it decide the shape of every tree this router builds.
#
# It is a STAIRCASE with its step at sixteen, not a line: 2 nodes cost 140.2 ms and 16 cost 164.4,
# which is 17 % more step for eight times the width, and then 24 costs 225.1 -- the FP4 kernel's
# second tile of sixteen rows, paid in full. So the node budget is 16, and 17 is expensive.
#
# And the SHAPE is free: the same sixteen nodes cost 164.37 as a chain, 165.39 branching two ways
# and 165.41 branching three. The ancestor mask, the convolution's gather and the ancestor-masked
# UT factorisation are elementwise work on [16, 16] matrices against 19.4 GB of weight reads.
TREE_MS_NVFP4 = {2: 140.16, 4: 143.79, 8: 150.01, 12: 156.60, 16: 164.37, 24: 225.10, 32: 225.10}
# The commit is paid on EVERY tree block rather than only on a rejection, because a tree has no
# single successor state: 6.0 to 7.9 ms, and flat in the tree size.
TREE_COMMIT_MS = 6.6


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

    def state_snapshot(self):
        """The neural sub-drafter's cache. The lookup drafter is rebuilt from tokens by `prime`."""
        return ("router", self.mtp.state_snapshot())

    def snapshot_bytes_per_token(self) -> int:
        from engine.cache import _drafter_bytes_per_token
        return _drafter_bytes_per_token(self.mtp)

    def state_restore(self, snap) -> None:
        kind, sub = snap
        if kind != "router":
            raise ValueError(f"not a router snapshot: {kind!r}")
        self.mtp.state_restore(sub)

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
    wants_rows = True          # it holds a head that reads the engine's tap; see engine/spec.py

    def __init__(self, ngram, mtp, mtp_depth: int = 3, node_budget: int = 16,
                 mtp_ms_per_token: float = MTP_MS_PER_TOKEN, rollback_ms: float = ROLLBACK_MS,
                 alpha: float = 0.15, verify_ms_table: dict[int, float] | None = None,
                 depth_margin: float = 0.05, head_fixed_ms: float = 0.0,
                 adaptive_depth: bool = True, tree_ms_table: dict[int, float] | None = None,
                 commit_ms: float = TREE_COMMIT_MS, head_prior: float = 4.3,
                 always_head: bool = False):
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
        # How many nodes the head may put in the tree before the merge. Deliberately the whole
        # budget rather than a share of it: the merge shares prefixes and the prune afterwards is
        # what decides, so starving either source before they meet only hides candidates from the
        # marginal rule.
        self.head_budget = node_budget
        # Tree blocks are priced off their own measured curve, not off the chain one. They are not
        # the same measurement: a chain of eight costs 153.5 ms through `forward_block` and a tree
        # of eight costs 150.0 through `forward_tree`, and that gap is larger than the differences
        # this router decides on.
        self.tree_table = dict(tree_ms_table) if tree_ms_table else dict(TREE_MS_NVFP4)
        tk = sorted(self.tree_table)
        self.tree_base_ms = self.tree_table[tk[0]]
        self.tree_per_node_ms = ((self.tree_table[16] - self.tree_table[tk[0]]) / (16 - tk[0])
                                 if 16 in self.tree_table else 1.73)
        self.commit_ms = commit_ms
        # Merging needs both proposals, and having both means paying for both. A block drafter is
        # 35 ms against a 140-164 ms step, so on a workload where the lookup drafter fires on 94 %
        # of blocks -- `quote` -- paying it every step is a fifth of the budget spent on a proposal
        # that loses. So the head is skipped where the lookup tree alone already beats what the
        # head has been worth lately. `always_head` turns that off, for measuring it.
        self.always_head = always_head
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
        self.calib_head = _Rate(1.0, alpha)
        # What the head's block has actually been accepting lately, in tokens. It starts at the
        # measured figure for the block drafter (SPEED-LEDGER 10:28: 4.30 over five workloads) so
        # that the head is asked until evidence says not to. That is the opposite of the 11:03
        # trap rather than a repeat of it: the optimism sits on the option that is NOT the one
        # whose counterfactual is free, and the lookup drafter's tree is built and scored against
        # the target every step whether or not the head runs.
        self.head_accepted = _Rate(head_prior, alpha)
        self.last_head_tree = None
        self.rate_mtp = _Rate(0.6, alpha)
        self.last = None
        self.last_n = 0
        self.last_expected = 0.0
        self.last_depth = 0          # the head chain length this step actually paid for
        self.last_tree = None        # what the lookup drafter offered, chosen or not
        # ENG-102: the head arm's per-token q rows when the request samples (None for the lookup
        # arm, whose proposal is deterministic and takes the p(d) accept).
        self.last_q = None
        self.stats = {"ngram": 0, "mtp": 0, "merged": 0, "ngram_tokens": 0, "mtp_tokens": 0,
                      "declined": 0, "depth_hist": {}}

    # --- state ---------------------------------------------------------------------------------

    def reset(self) -> None:
        self.ngram.reset()
        self.mtp.reset()
        self.last = None
        self.last_tree = None
        self.last_q = None

    def set_sampling(self, sampler) -> None:
        """The lookup arm never samples; the head arm does when the request samples (ENG-102)."""
        if hasattr(self.mtp, "set_sampling"):
            self.mtp.set_sampling(sampler)

    def prime(self, tokens: list[int]) -> None:
        self.ngram.prime(tokens)

    def state_snapshot(self):
        """The neural sub-drafter's cache. The lookup drafter is rebuilt from tokens by `prime`."""
        return ("merged", self.mtp.state_snapshot())

    def snapshot_bytes_per_token(self) -> int:
        from engine.cache import _drafter_bytes_per_token
        return _drafter_bytes_per_token(self.mtp)

    def state_restore(self, snap) -> None:
        kind, sub = snap
        if kind != "merged":
            raise ValueError(f"not a merged snapshot: {kind!r}")
        self.mtp.state_restore(sub)

    def sync(self, tokens, hidden, first_pos, rows=None) -> None:
        if getattr(self.mtp, "wants_rows", False):
            self.mtp.sync(tokens, hidden, first_pos, rows=rows)
        else:
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
        # The head's tree gets the same treatment, for the same reason: its node scores are a
        # softmax of a selector score that was never calibrated against anything, so what the
        # router needs from it is one scalar saying how optimistic it has been lately. It is free
        # here, and it is never conditioned on the head having been chosen.
        if self.last_head_tree is not None:
            got = self.last_head_tree.accepted_against(list(tokens))
            exp_head = self.last_head_tree.expected_accepted()
            if exp_head > 0:
                self.calib_head.update(got, exp_head)
            # And what it was worth in tokens, which is the number the skip decision is made on.
            # Only counted on steps where the head actually ran: on a step where it was skipped
            # there is no tree to score, and pretending otherwise would be the 11:03 trap with the
            # sign flipped.
            self.head_accepted.update(got, 1)
        self.last_head_tree = None
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

    def _calibrated_expected(self, tree) -> float:
        """Expected accepted length of a tree whose nodes came from two sources.

        Each source's node scores are its own kind of optimism -- counts out of a suffix store on
        one side, a softmax of a selector score on the other -- so each is scaled by the factor
        learned for it. Summing them is the identity used everywhere else here: a node is accepted
        only if its whole path is, so the expectation of a tree is the sum of its nodes' path
        probabilities.
        """
        total = 0.0
        for i in range(1, len(tree.tokens)):
            src = tree.source[i]
            if src.startswith("ngram"):
                total += tree.scores[i] * self.calib.value
            elif src.startswith("df2"):
                total += tree.scores[i] * self.calib_head.value
            else:
                total += tree.scores[i]     # already priced by the caller, e.g. the mtp chain
        return total

    def _tree_value(self, tree, cost_ms: float) -> float:
        """Tokens per second if every step verified this tree.

        The commit is unconditional, unlike the chain rollback: a tree has no single successor
        state, so the accepted path's state is always reconstructed. A tree block is therefore
        about 6 ms dearer than a chain block that accepted everything and about 6 ms cheaper than
        one that did not.
        """
        if tree is None or tree.n_draft == 0:
            return 0.0
        expected = min(self._calibrated_expected(tree), float(tree.n_draft))
        ms = verify_ms(tree.n_draft + 1, self.tree_table) + cost_ms + self.commit_ms
        return (expected + 1.0) / (ms / 1000.0)

    @staticmethod
    def _chain_of(tree):
        """The head's own best line out of its tree: its greedy walk, as a chain."""
        from engine.tree import DraftTree
        if tree is None or tree.n_draft == 0:
            return None
        node, toks, scores = 0, [], []
        while True:
            kids = [c for c in range(node + 1, len(tree.tokens)) if tree.parents[c] == node]
            if not kids:
                break
            node = max(kids, key=lambda c: tree.scores[c])
            toks.append(tree.tokens[node])
            scores.append(tree.scores[node])
        if not toks:
            return None
        return DraftTree.chain(tree.tokens[0], toks, scores=scores, source="df2-greedy")

    def _chain_value(self, tree, cost_ms: float) -> float:
        """The same pricing on the chain curve, with the rollback paid only on a rejection."""
        expected = min(self._calibrated_expected(tree), float(tree.n_draft))
        p_reject = min(1.0, max(0.0, 1.0 - expected / max(tree.n_draft, 1)))
        ms = verify_ms(tree.n_draft + 1, self.verify_table) + cost_ms + self.rollback_ms * p_reject
        return (expected + 1.0) / (ms / 1000.0)

    def _head_tree(self, context: list[int], depth: int):
        """The head's proposal as a tree: its own if it builds one, otherwise its chain."""
        from engine.tree import DraftTree

        if hasattr(self.mtp, "propose_tree"):
            return self.mtp.propose_tree(context, self.head_budget)
        chain = self.mtp.propose(context, depth) if depth > 0 else []
        if not chain:
            return None
        # NOT scores of 1.0. A chain is accepted prefix-first, so the i-th token counts only if the
        # i-1 above it did; `DraftTree.chain`'s default says every token is certain, and the 11:03
        # entry is what that does to a router -- an option that claims eight accepted tokens before
        # a block has been verified cannot be outbid by anything. The prior here is the head's own
        # measured per-token rate, compounded, which is `_mtp_expected` written per node.
        rate = self.rate_mtp.value
        return DraftTree.chain(context[-1], chain,
                               scores=[rate ** (i + 1) for i in range(len(chain))], source="mtp")

    def propose_tree(self, context: list[int], k: int):
        """The block this router wants verified, as a tree.

        Three shapes are priced against the measured tree curve and the widest one usually wins,
        because that curve is nearly flat below sixteen: the verify reads 19.4 GB of weights whether
        it carries two rows or sixteen, so a marginal node costs about 1.7 ms against a 140-164 ms
        step. That is the thesis of this track -- a sixteen-node tree verifies for what a chain of
        sixteen costs, and a chain of sixteen can only accept along one line.

          * the head's own tree -- the block drafter's lattice, its greedy path plus the best
            alternatives around it -- which costs one fixed call whatever its width;
          * the lookup drafter's tree, which costs a dictionary lookup;
          * the two grafted together, which costs the UNION of their nodes, since shared prefixes
            are shared nodes. The 11:18 measurement says that union is worth more than either part
            on `edit`, where the lookup supplies the verbatim stretches and the head the bridges.

        Whichever wins is pruned to the node budget, best-first under the marginal rule.
        """
        self.last_q = None                     # q-aware accept is a chain mechanism (ENG-102 v1)
        depth, _ = self._best_mtp_depth()
        self.stats["depth_hist"][depth] = self.stats["depth_hist"].get(depth, 0) + 1
        self.last_depth = depth
        # The lookup drafter is asked first because asking it is a dictionary lookup, 0.20 ms.
        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        self.last_tree = tree
        self.last_expected = tree.expected_accepted() if tree is not None else 0.0
        head_cost = self.head_fixed_ms + self.mtp_ms_per_token * depth
        v_ngram = self._tree_value(tree, 0.0)
        prior_ms = (verify_ms(self.node_budget + 1, self.tree_table) + head_cost + self.commit_ms)
        v_head_prior = (self.head_accepted.value + 1.0) / (prior_ms / 1000.0)
        if not self.always_head and tree is not None and v_ngram > v_head_prior:
            head_tree = None
            self.stats["head_skipped"] = self.stats.get("head_skipped", 0) + 1
        else:
            head_tree = self._head_tree(context, depth)
        self.last_head_tree = head_tree

        best, v_best, label = head_tree, self._tree_value(head_tree, head_cost), "mtp"
        # The chain is one of the options, priced on the CHAIN curve, because a chain-shaped block
        # is 12.5 ms cheaper than a tree of the same width -- it has a kernel the tree does not
        # (SPEED-LEDGER 13:49). `forward_tree` sends a chain-shaped tree to `forward_block`, so
        # proposing a line is all this has to do to take the cheaper price.
        chain_tree = self._chain_of(head_tree)
        if chain_tree is not None:
            v_chain = self._chain_value(chain_tree, head_cost)
            if v_chain > v_best:
                best, v_best, label = chain_tree, v_chain, "chain"
        if v_ngram > v_best:
            best, v_best, label = tree, v_ngram, "ngram"
        if head_tree is not None and tree is not None:
            merged = head_tree.merge(tree)
            if merged.n_draft > self.node_budget:
                merged = merged.prune(self.node_budget, per_node_ms=self.tree_per_node_ms,
                                      base_ms=self.tree_base_ms)
            if self._tree_value(merged, head_cost) > v_best:
                best, label = merged, "merged"
        if best is None:
            self.last, self.last_n = None, 0
            self.stats["declined"] += 1
            return None
        if best.n_draft > self.node_budget:
            best = best.prune(self.node_budget, per_node_ms=self.tree_per_node_ms,
                              base_ms=self.tree_base_ms)
        self.last, self.last_n = label, best.n_draft
        self.stats[label] = self.stats.get(label, 0) + 1
        self.stats["ngram_tokens" if label == "ngram" else "mtp_tokens"] += best.n_draft
        self.stats["nodes"] = self.stats.get("nodes", 0) + best.n_draft
        return best

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
                self.last_q = None                    # a lookup proposal is deterministic
                self.stats["ngram"] += 1
                self.stats["ngram_tokens"] += n
                return best
        m = min(depth, k)
        if m <= 0:
            self.stats["declined"] += 1
            self.last, self.last_n = None, 0
            self.last_q = None
            return []
        draft = self.mtp.propose(context, m)
        self.last, self.last_n, self.last_depth = "mtp", len(draft), m
        self.last_q = getattr(self.mtp, "last_q", None)   # the head's q rows when sampling
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
