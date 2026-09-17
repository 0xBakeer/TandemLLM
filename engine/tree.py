"""A draft tree, and the one layout property that lets a linear-attention model verify it.

A chain drafter proposes `t1 t2 t3`; a tree drafter proposes `t1 -> {t2a, t2b}`, and on this board
that is nearly free. The measured verify curve is `V(N) = 149.1 ms + 1.896 ms * N` (SPEED-LEDGER,
09:55), so a node is worth adding when it has about a 5 % chance of extending the accepted prefix.
That is a much lower bar than on a server, where a verify pass is compute-bound and width costs
real time.

**The layout rule.** Nodes are stored in DFS pre-order, so every node's index is greater than its
parent's and each subtree occupies a contiguous range. That is not a convenience. For the 48 Gated
DeltaNet layers, tree verification is a *mask change* to the UT factorisation the chunked kernel
already runs, and the forward substitution in `solve_tril` is only valid if the ancestor-masked
matrix is strictly lower triangular -- which holds exactly when every ancestor has a lower index
than its descendant. DFS pre-order guarantees it; breadth-first order does not.

This module is the contract two drafters have to agree on before their candidates can share one
verify call: the lookup drafter's tree (`engine/drafters/ngram.py`) and the block drafter's lattice.
`merge` is the graft operation -- one tree, two sources, rather than a router choosing between them.

Nothing here touches the GPU or torch, so it is all testable on a laptop; the mask this produces is
consumed by the verify path, which is not written yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Verify cost on this board, from SPEED-LEDGER 09:55 (FP8 weights). `N` counts drafted nodes, not
# the anchor. Overridden by the caller for other weight sets.
VERIFY_BASE_MS = 149.1
VERIFY_PER_NODE_MS = 1.896


@dataclass
class DraftTree:
    """Candidate continuations of one anchor token, in DFS pre-order.

    `tokens[0]` is the anchor -- the last token the model has already committed to -- and carries
    parent -1. Every other node is a drafted token. `len(tree) - 1` is what the cost model calls N.
    """

    tokens: list[int]
    parents: list[int]
    scores: list[float] = field(default_factory=list)
    source: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.scores:
            self.scores = [1.0] + [0.0] * (len(self.tokens) - 1)
        if not self.source:
            self.source = ["root"] + ["?"] * (len(self.tokens) - 1)

    # --- basics -------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_draft(self) -> int:
        """Drafted nodes, excluding the anchor. This is `N` in the verify cost model."""
        return len(self.tokens) - 1

    def check(self) -> None:
        """Assert the invariants the verify path depends on. Cheap enough to run in tests only."""
        assert self.parents[0] == -1, "node 0 must be the anchor"
        assert len(self.parents) == len(self.tokens) == len(self.scores) == len(self.source)
        for i in range(1, len(self.tokens)):
            assert 0 <= self.parents[i] < i, f"node {i} parent {self.parents[i]} breaks pre-order"
        # contiguity of subtrees, which is what "DFS pre-order" adds over "parent < child"
        for i in range(len(self.tokens)):
            end = i + 1
            while end < len(self.tokens) and self._is_descendant(end, i):
                end += 1
            for j in range(end, len(self.tokens)):
                assert not self._is_descendant(j, i), f"subtree of {i} is not contiguous at {j}"

    def _is_descendant(self, node: int, of: int) -> bool:
        while node > of:
            node = self.parents[node]
        return node == of

    def depths(self) -> list[int]:
        """Depth of each node; the anchor is 0. These are the position offsets for RoPE."""
        d = [0] * len(self.tokens)
        for i in range(1, len(self.tokens)):
            d[i] = d[self.parents[i]] + 1
        return d

    def path(self, node: int) -> list[int]:
        """Node indices from the anchor down to `node`, inclusive."""
        out = []
        while node >= 0:
            out.append(node)
            node = self.parents[node]
        return list(reversed(out))

    def ancestor_mask(self) -> list[list[bool]]:
        """`m[i][j]` is True when j is i or an ancestor of i: what each node may attend to.

        Strictly lower triangular plus the diagonal, which is the property `solve_tril` needs.
        """
        n = len(self.tokens)
        m = [[False] * n for _ in range(n)]
        for i in range(n):
            j = i
            while j >= 0:
                m[i][j] = True
                j = self.parents[j]
        return m

    def conv_windows(self, width: int) -> list[list[int]]:
        """For each node, the `width` columns its depthwise causal convolution reads.

        Indices address `cat([conv_state, block], -1)`: columns `0 .. width-2` are the convolution
        state (oldest first, so column `width-2` is the token immediately before the block's anchor)
        and column `width-1+j` is node j. Each row is oldest first with the node itself last, which
        is the order `F.conv1d` applies the kernel's taps in.

        On a chain this reproduces the ordinary sliding window exactly. On a tree it follows the
        node's own ancestors, which in DFS pre-order are not adjacent columns -- that is the whole
        difference, and it is why the convolution needs a gather rather than a shift.
        """
        out: list[list[int]] = []
        for i in range(len(self.tokens)):
            chain: list[int] = []
            cur = i
            while cur >= 0 and len(chain) < width:
                chain.append(width - 1 + cur)
                cur = self.parents[cur]
            for j in range(width - len(chain)):
                chain.append(width - 2 - j)
            out.append(list(reversed(chain)))
        return out

    def leaves(self) -> list[int]:
        has_child = [False] * len(self.tokens)
        for p in self.parents[1:]:
            has_child[p] = True
        return [i for i, h in enumerate(has_child) if not h]

    # --- construction -------------------------------------------------------------------------

    @staticmethod
    def chain(anchor: int, tokens: list[int], scores: list[float] | None = None,
              source: str = "chain") -> "DraftTree":
        n = len(tokens)
        scores = list(scores) if scores else [1.0] * n
        return DraftTree(tokens=[anchor] + list(tokens),
                         parents=[-1] + list(range(n)),
                         scores=[1.0] + scores,
                         source=["root"] + [source] * n)

    @staticmethod
    def from_sequences(anchor: int, seqs: list[tuple[list[int], float]],
                       source: str = "tree") -> "DraftTree":
        """Build the prefix tree of a set of scored continuations.

        A node's score is the largest score of any sequence passing through it, which for the
        frequency scores this engine uses is the path probability of its most likely completion.
        """
        b = TreeBuilder(anchor)
        for toks, score in seqs:
            node = 0
            for t in toks:
                node = b.add(node, t, score, source)
        return b.build()

    # --- operations ---------------------------------------------------------------------------

    def expected_accepted(self) -> float:
        """Expected accepted draft tokens, under the scores read as path probabilities.

        A drafted token is accepted only if every token above it on its path is, so the expected
        accepted length of a tree is the sum of its nodes' path probabilities. This is the quantity
        the firing policy and the node budget are both decided by.
        """
        return sum(self.scores[1:])

    def value(self, draft_ms: float = 0.0, base_ms: float = VERIFY_BASE_MS,
              per_node_ms: float = VERIFY_PER_NODE_MS) -> float:
        """Expected tokens per second if every step looked like this one."""
        gained = self.expected_accepted() + 1.0        # the model's own token is always accepted
        cost = (base_ms + per_node_ms * self.n_draft + draft_ms) / 1000.0
        return gained / cost if cost > 0 else 0.0

    def prune(self, budget: int, per_node_ms: float = VERIFY_PER_NODE_MS,
              base_ms: float = VERIFY_BASE_MS, draft_ms: float = 0.0) -> "DraftTree":
        """Keep at most `budget` nodes, best-first, and only while each one pays for itself.

        The marginal rule is the one in RESEARCH section 0.2: with throughput `tau / (V + d)`, a node
        earns its 1.896 ms when it adds more than `tau / (V + d) * per_node_ms` expected tokens.
        Nodes are admitted in descending score with their ancestors, which keeps DFS pre-order
        reachable and never admits a node whose parent was rejected.
        """
        if self.n_draft <= 0:
            return self
        order = sorted(range(1, len(self.tokens)), key=lambda i: -self.scores[i])
        keep = {0}
        gained = 1.0
        cost_ms = base_ms + draft_ms
        for i in order:
            if len(keep) - 1 >= budget:
                break
            chain = [n for n in self.path(i) if n not in keep]
            add = sum(self.scores[n] for n in chain)
            if len(keep) - 1 + len(chain) > budget:
                continue
            threshold = (gained / max(cost_ms, 1e-6)) * per_node_ms * len(chain)
            if add < threshold:
                continue
            keep.update(chain)
            gained += add
            cost_ms += per_node_ms * len(chain)
        return self.subset(keep)

    def subset(self, keep: set[int]) -> "DraftTree":
        """The tree restricted to `keep`, re-indexed in DFS pre-order. `keep` must be ancestor-closed."""
        children: dict[int, list[int]] = {}
        for i in range(1, len(self.tokens)):
            if i in keep:
                children.setdefault(self.parents[i], []).append(i)
        tokens: list[int] = []
        parents: list[int] = []
        scores: list[float] = []
        source: list[str] = []
        remap: dict[int, int] = {}

        def walk(old: int, parent_new: int) -> None:
            new = len(tokens)
            remap[old] = new
            tokens.append(self.tokens[old])
            parents.append(parent_new)
            scores.append(self.scores[old])
            source.append(self.source[old])
            for c in sorted(children.get(old, []), key=lambda c: -self.scores[c]):
                walk(c, new)

        walk(0, -1)
        return DraftTree(tokens, parents, scores, source)

    def merge(self, other: "DraftTree", prefer: str = "max") -> "DraftTree":
        """Graft `other` onto this tree. Both must share the same anchor.

        Where the two sources propose the same token at the same place the node is shared, so a
        merged tree costs strictly less than the sum of its parts. This is the operation that
        replaces the router: instead of choosing a drafter per step, both put candidates in the
        same verify call and the node budget decides how much each gets.
        """
        assert self.tokens[0] == other.tokens[0], "trees must share an anchor"
        b = TreeBuilder(self.tokens[0])

        def graft(tree: "DraftTree", src_node: int, dst_node: int) -> None:
            for c in tree.children_of(src_node):
                n = b.add(dst_node, tree.tokens[c], tree.scores[c], tree.source[c], prefer=prefer)
                graft(tree, c, n)

        graft(self, 0, 0)
        graft(other, 0, 0)
        return b.build()

    def children_of(self, node: int) -> list[int]:
        return [i for i in range(1, len(self.tokens)) if self.parents[i] == node]

    # --- offline evaluation --------------------------------------------------------------------

    def accepted_against(self, continuation: list[int]) -> int:
        """How many drafted tokens a greedy verify would accept, given what the target really wrote.

        Greedy verification accepts the unique root-to-node path whose tokens match the target's own
        continuation, and stops at the first place the tree has no matching child. This makes offline
        evaluation exact rather than an estimate -- see tools/sim_draft.py.
        """
        node, n = 0, 0
        for want in continuation:
            nxt = next((c for c in self.children_of(node) if self.tokens[c] == want), None)
            if nxt is None:
                break
            node, n = nxt, n + 1
        return n


class TreeBuilder:
    """Accumulates nodes in any order and emits them in DFS pre-order."""

    def __init__(self, anchor: int):
        self.tokens = [anchor]
        self.parents = [-1]
        self.scores = [1.0]
        self.source = ["root"]
        self.kids: list[dict[int, int]] = [{}]

    def add(self, parent: int, token: int, score: float, source: str = "?",
            prefer: str = "max") -> int:
        existing = self.kids[parent].get(token)
        if existing is not None:
            if prefer == "max" and score > self.scores[existing]:
                self.scores[existing] = score
            elif prefer == "sum":
                self.scores[existing] = min(1.0, self.scores[existing] + score)
            return existing
        idx = len(self.tokens)
        self.tokens.append(token)
        self.parents.append(parent)
        self.scores.append(score)
        self.source.append(source)
        self.kids.append({})
        self.kids[parent][token] = idx
        return idx

    def build(self) -> DraftTree:
        tokens: list[int] = []
        parents: list[int] = []
        scores: list[float] = []
        source: list[str] = []

        def walk(old: int, parent_new: int) -> None:
            new = len(tokens)
            tokens.append(self.tokens[old])
            parents.append(parent_new)
            scores.append(self.scores[old])
            source.append(self.source[old])
            for _, c in sorted(self.kids[old].items(), key=lambda kv: -self.scores[kv[1]]):
                walk(c, new)

        walk(0, -1)
        return DraftTree(tokens, parents, scores, source)


def lattice_tree(anchor: int, cand: list[list[int]], logp: list[list[list[float]]],
                 greedy: list[int], budget: int, source: str = "df2") -> DraftTree:
    """A tree out of a block drafter's lattice: the greedy path, then the best nodes around it.

    `cand[l][c]` is the c-th candidate token at slot l, `logp[l][p][c]` is the log probability of
    candidate c given that slot l-1 took candidate p (for slot 0 every predecessor row is the
    anchor, so row 0 is the one that means anything), and `greedy[l]` is the candidate index the
    released greedy walk took at slot l.

    Two decisions, both from measurements already in the ledger.

    **The greedy path goes in first, whatever its score.** The 10:28 entry measured the exact
    maximiser of the selector's own objective -- Viterbi -- accepting 3.22 tokens a block against
    greedy's 4.30. What a verify pays for is the expected accepted PREFIX, in which slot 0
    multiplies every later term, and a path maximiser will happily trade slot 0 away for a better
    total. Greedy's slot 0 is the head's own top-1, the most reliable single signal in the lattice.

    **The rest is best-first on path probability.** A node's path probability is its parent's times
    a conditional, so priorities fall monotonically down any path; popping in descending order
    therefore yields the highest-probability ancestor-closed set of nodes of that size, which is
    exactly the set that maximises the expected accepted length for a given budget.

    Pure Python and no torch, so the simulator scores the tree the engine will actually build --
    `tools/sim_draft.py` and `engine/drafters/dflash2.py` call this same function.
    """
    import heapq
    import math

    b = TreeBuilder(anchor)
    heap: list[tuple] = []
    tie = 0
    n = 0
    L = len(cand)
    k = len(cand[0]) if L else 0
    row, parent, lp = 0, 0, 0.0
    for slot, c in enumerate(greedy[:L]):
        if n >= budget:
            break
        base = lp
        lp += logp[slot][row][c]
        here, parent = parent, b.add(parent, cand[slot][c], math.exp(lp), f"{source}-greedy")
        n += 1
        for c2 in range(k):
            if c2 != c:
                tie += 1
                heapq.heappush(heap, (-(base + logp[slot][row][c2]), tie, here, slot, c2))
        row = c
    while heap and n < budget:
        negp, _, parent_id, slot, c = heapq.heappop(heap)
        lp = -negp
        nid = b.add(parent_id, cand[slot][c], math.exp(lp), f"{source}-tree")
        n += 1
        if slot + 1 < L:
            for c2 in range(k):
                tie += 1
                heapq.heappush(heap, (-(lp + logp[slot + 1][c][c2]), tie, nid, slot + 1, c2))
    return b.build()
