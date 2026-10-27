"""The block drafter's host walk and lattice tree in C++ (SPD-201 host cut, QWEN38_TREE_NATIVE).

Between the packed draft and the verify pass the GPU waits for the host to turn every slot's lattice into a
draft tree: the walk blob and the [L, k, k] log-probabilities into Python lists (`tolist`), the greedy walk,
then `engine/tree.py` `lattice_tree` over them. Here the same steps read the pinned host copies in place:
`walk` is `DFlash2Module._walk_blob` and `lattice` is `lattice_tree` + `TreeBuilder.build` -- the same
double-precision sums in the same order, the same heap order (path probability, then push order), the same
stable child order -- so the tree is the Python one, node for node (`check`). Built on first use from
engine/csrc/native_tree/tree_host.cpp (engine/native_build.py), cached by the source's hash.
"""

from __future__ import annotations

from engine import native_build as nb
from engine.settings import SETTINGS as _S  # noqa: E402

ON = _S.get("TREE_NATIVE") == "1"

_CPP = nb.source(__file__, "tree_host.cpp")

_MOD = None
_READY = None


def ready() -> bool:
    """ON, and the extension built (the first call builds it). A host without a C++ toolchain logs once and
    runs the Python walk and tree, which build the same tree."""
    global _READY
    if _READY is None:
        if not ON:
            return False
        try:
            module()
            _READY = True
        except Exception as e:                 # noqa: BLE001 - no compiler, no headers: the Python path
            print(f"[drafter] the native tree did not build ({type(e).__name__}: {e}); the Python walk and tree "
                  f"serve (the same tree)", flush=True)
            _READY = False
    return _READY


def module():
    global _MOD
    if _MOD is None:
        tag = nb.tag(_CPP)
        _MOD = nb.load(f"tree_host_{tag}", cpp_sources=[_CPP], functions=["walk", "lattice"],
                       extra_cflags=["-O2"], verbose=False)
    return _MOD


_KIND = ("root", "df2-greedy", "df2-tree")


def lattice_tree(blob, logp, L: int, k: int, anchor: int, chain: list, budget: int):
    """`engine/tree.py` `lattice_tree` over a block's host blob and log-probabilities, as a DraftTree."""
    from engine.tree import DraftTree
    t, p, s, kd = module().lattice(blob, logp, int(L), int(k), int(anchor), [int(x) for x in chain],
                                   int(budget))
    return DraftTree(list(t), list(p), list(s), [_KIND[x] for x in kd])


def check(n: int = 400, seed: int = 0) -> list[str]:
    """The C++ walk and tree against `_walk_blob` and `lattice_tree` on random lattices, node for node
    (duplicate candidates and tied probabilities included). CPU only."""
    import random

    import torch

    from engine.drafters.dflash2 import DFlash2Module
    from engine.tree import lattice_tree as py_tree
    rnd = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    same = 0
    for it in range(n):
        L, k = rnd.choice([(7, 16), (15, 16), (15, 8), (3, 4)])
        vocab = rnd.choice([5, 50, 100000])
        cand = torch.randint(0, vocab, (L, k), generator=g)
        local = torch.randint(0, k, ((L - 1) * k,), generator=g)
        blob = torch.cat([torch.randint(0, k, (1,), generator=g), local, cand.reshape(-1)]).long()
        raw = torch.randn(L, k, k, generator=g)
        if it % 5 == 0:
            raw = raw.round()                          # ties
        logp = torch.log_softmax(raw / 0.7, dim=-1).float()
        toks_py, cand_py = DFlash2Module._walk_blob(blob.tolist(), L, k)
        toks_c = list(module().walk(blob, L, k))
        assert toks_py == toks_c, (it, toks_py, toks_c)
        chain = toks_py[:rnd.randint(1, L)]
        greedy = [cand_py[l].index(chain[l]) if chain[l] in cand_py[l] else 0
                  for l in range(min(len(cand_py), len(chain)))]
        anchor = rnd.randint(0, vocab)
        budget = rnd.choice([1, 4, 8, 16, 32, 64])
        a = py_tree(anchor, cand_py, logp.tolist(), greedy, budget)
        b = lattice_tree(blob, logp, L, k, anchor, chain, budget)
        assert (a.tokens, a.parents, a.scores, a.source) == (b.tokens, b.parents, b.scores, b.source), it
        same += 1
    return [f"{same}/{n} random lattices: walk and tree identical to the Python (tokens, parents, scores, "
            f"sources)"]
