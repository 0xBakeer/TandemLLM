"""SPD-201 host cut (c): `engine.tree.lattice_tree` as one C++ call with the GIL released.

The packed draft hands each slot its lattice (candidates [L, k], log-probabilities [L, k, k]); the tree a slot
verifies is built from it best-first on path probability, the greedy path first (engine/tree.py
`lattice_tree`, whose docstring has the why). In Python that is a few hundred heap operations a slot over
lists the lattice was copied into, on the round's critical path once per slot. Here the same statements run in
C++ over the host tensors: the same float64 sums (`lp + logp[...]`, the float32 entries widened as Python
widens them), the same `exp`, the same heap order ((-p, tie) with the same tie counter), the same
de-duplication of a token under one parent and the same DFS pre-order emission (children by descending score,
first added first among equals). `check()` holds it to the Python function on random lattices.

Built on first use from csrc/lattice/lattice.cpp beside this module (engine/native_build.py; CPU only), cached by
source hash.
"""

from __future__ import annotations

import torch

from engine import native_build as nb

_SRC = nb.source(__file__, "lattice.cpp")
_CPP = nb.source(__file__, "lattice.h")
_MOD = None
_NAMES = {0: "root", 1: "greedy", 2: "tree"}


def _module():
    global _MOD
    if _MOD is None:
        tag = nb.tag(_CPP, _SRC)
        _MOD = nb.load(f"qwen38_lattice_{tag}", cpp_sources=[_SRC], functions=["lattice_tree_cpp"],
                       extra_cflags=["-O2"], verbose=False)
    return _MOD


def lattice_tree(anchor: int, cand, logp, greedy: list[int], budget: int, source: str = "df2"):
    """`engine.tree.lattice_tree` from host tensors (`cand` int64 [L, k], `logp` [L, k, k]) or lists."""
    from engine.tree import DraftTree
    if not torch.is_tensor(cand):
        cand = torch.tensor(cand, dtype=torch.long)
    if not torch.is_tensor(logp):
        logp = torch.tensor(logp, dtype=torch.float64)
    t, p, s, z = _module().lattice_tree_cpp(int(anchor), cand.long().cpu(), logp.cpu(), [int(g) for g in greedy],
                                            int(budget))
    src = [("root" if v == 0 else f"{source}-{_NAMES[v]}") for v in z.tolist()]
    return DraftTree(t.tolist(), p.tolist(), s.tolist(), src)


def check(n: int = 300, seed: int = 0) -> str:
    """The C++ build against engine/tree.py's on random lattices: tokens, parents, scores, sources equal."""
    import random
    from engine.tree import lattice_tree as ref
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        L, k = rng.randint(1, 7), rng.choice((2, 4, 8, 16))
        cand = torch.stack([torch.randperm(1000, generator=g)[:k] for _ in range(L)])
        if rng.random() < 0.2:
            cand[:, 1] = cand[:, 0]                     # a repeated candidate: the builder's merge
        logp = torch.log_softmax(torch.randn(L, k, k, generator=g) * 2, dim=-1)       # float32, as served
        greedy = [rng.randrange(k) for _ in range(L)]
        budget = rng.randint(0, 32)
        a = ref(7, cand.tolist(), logp.tolist(), greedy, budget)
        b = lattice_tree(7, cand, logp, greedy, budget)
        assert (a.tokens, a.parents, a.scores, a.source) == (b.tokens, b.parents, b.scores, b.source), (i, a, b)
    return f"{n} random lattices: the C++ tree is the Python tree (tokens, parents, scores, sources)"
