"""The block drafter's host walk and lattice tree in C++ (SPD-201 host cut, QWEN38_TREE_NATIVE).

Between the packed draft and the verify pass the GPU waits for the host to turn every slot's lattice into a
draft tree: the walk blob and the [L, k, k] log-probabilities into Python lists (`tolist`), the greedy walk,
then `engine/tree.py` `lattice_tree` over them. Here the same steps read the pinned host copies in place:
`walk` is `DFlash2Module._walk_blob` and `lattice` is `lattice_tree` + `TreeBuilder.build` -- the same
double-precision sums in the same order, the same heap order (path probability, then push order), the same
stable child order -- so the tree is the Python one, node for node (`check`). Built on first use with
`torch.utils.cpp_extension.load_inline`, cached by the source's hash.
"""

from __future__ import annotations

import hashlib
import os
import sys

from engine.settings import SETTINGS as _S  # noqa: E402

ON = _S.get("TREE_NATIVE") == "1"

_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <queue>
#include <vector>
#include <algorithm>

// blob: [1 + (L-1)*k + L*k] int64 -- the first slot's argmax, every later slot's argmax a predecessor, the
// candidates [L][k] (dflash2 `_blob`)
std::vector<int64_t> walk(torch::Tensor blob, int64_t L, int64_t k) {
    const int64_t* b = blob.data_ptr<int64_t>();
    const int64_t* local = b + 1;
    const int64_t* cand = b + 1 + (L - 1) * k;
    std::vector<int64_t> toks;
    toks.reserve(L);
    int64_t idx = b[0];
    toks.push_back(cand[idx]);
    for (int64_t e = 0; e < L - 1; ++e) {
        idx = local[e * k + idx];
        toks.push_back(cand[(e + 1) * k + idx]);
    }
    return toks;
}

struct Item { double negp; int64_t tie; int64_t parent; int64_t slot; int64_t c; };
struct Later {
    bool operator()(const Item& a, const Item& b) const {
        if (a.negp != b.negp) return a.negp > b.negp;
        return a.tie > b.tie;
    }
};

struct Builder {
    std::vector<int64_t> tokens, parents, kind;
    std::vector<double> scores;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> kids;   // (token, node) in insertion order
    explicit Builder(int64_t anchor) {
        tokens.push_back(anchor); parents.push_back(-1); scores.push_back(1.0); kind.push_back(0);
        kids.emplace_back();
    }
    int64_t add(int64_t parent, int64_t token, double score, int64_t k) {
        for (auto& kv : kids[parent]) {
            if (kv.first == token) {
                if (score > scores[kv.second]) scores[kv.second] = score;
                return kv.second;
            }
        }
        int64_t idx = (int64_t)tokens.size();
        tokens.push_back(token); parents.push_back(parent); scores.push_back(score); kind.push_back(k);
        kids.emplace_back();
        kids[parent].push_back({token, idx});
        return idx;
    }
};

static void dfs(const Builder& b, int64_t old, int64_t parent_new, std::vector<int64_t>& t,
                std::vector<int64_t>& p, std::vector<double>& s, std::vector<int64_t>& kd) {
    int64_t nw = (int64_t)t.size();
    t.push_back(b.tokens[old]); p.push_back(parent_new); s.push_back(b.scores[old]); kd.push_back(b.kind[old]);
    std::vector<std::pair<int64_t, int64_t>> ks = b.kids[old];
    std::stable_sort(ks.begin(), ks.end(), [&](const std::pair<int64_t, int64_t>& x,
                                               const std::pair<int64_t, int64_t>& y) {
        return -b.scores[x.second] < -b.scores[y.second];
    });
    for (auto& kv : ks) dfs(b, kv.second, nw, t, p, s, kd);
}

// `lattice_tree(anchor, cand, logp, greedy, budget)` with `greedy` from `chain` as dflash2 takes it:
// candidates from the blob, logp [L, k, k] float32
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<double>, std::vector<int64_t>>
lattice(torch::Tensor blob, torch::Tensor logp, int64_t L, int64_t k, int64_t anchor,
        std::vector<int64_t> chain, int64_t budget) {
    const int64_t* cand = blob.data_ptr<int64_t>() + 1 + (L - 1) * k;
    const float* lp3 = logp.data_ptr<float>();
    auto LP = [&](int64_t slot, int64_t row, int64_t c) { return (double)lp3[(slot * k + row) * k + c]; };
    int64_t G = std::min<int64_t>(L, (int64_t)chain.size());
    std::vector<int64_t> greedy(G);
    for (int64_t l = 0; l < G; ++l) {
        int64_t at = 0;
        for (int64_t c = 0; c < k; ++c) if (cand[l * k + c] == chain[l]) { at = c; break; }
        greedy[l] = at;
    }
    Builder b(anchor);
    std::priority_queue<Item, std::vector<Item>, Later> heap;
    int64_t tie = 0, n = 0, row = 0, parent = 0;
    double lp = 0.0;
    for (int64_t slot = 0; slot < G && slot < L; ++slot) {
        if (n >= budget) break;
        int64_t c = greedy[slot];
        double base = lp;
        lp += LP(slot, row, c);
        int64_t here = parent;
        parent = b.add(parent, cand[slot * k + c], std::exp(lp), 1);
        n += 1;
        for (int64_t c2 = 0; c2 < k; ++c2) {
            if (c2 != c) { tie += 1; heap.push({-(base + LP(slot, row, c2)), tie, here, slot, c2}); }
        }
        row = c;
    }
    while (!heap.empty() && n < budget) {
        Item it = heap.top(); heap.pop();
        double l2 = -it.negp;
        int64_t nid = b.add(it.parent, cand[it.slot * k + it.c], std::exp(l2), 2);
        n += 1;
        if (it.slot + 1 < L) {
            for (int64_t c2 = 0; c2 < k; ++c2) {
                tie += 1; heap.push({-(l2 + LP(it.slot + 1, it.c, c2)), tie, nid, it.slot + 1, c2});
            }
        }
    }
    std::vector<int64_t> t, p, kd; std::vector<double> s;
    dfs(b, 0, -1, t, p, s, kd);
    return {t, p, s, kd};
}
"""

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
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        tag = hashlib.sha1(_CPP.encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"tree_host_{tag}", cpp_sources=[_CPP], functions=["walk", "lattice"],
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
