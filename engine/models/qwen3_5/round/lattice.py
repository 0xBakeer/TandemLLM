"""SPD-201 host cut (c): `engine.tree.lattice_tree` as one C++ call with the GIL released.

The packed draft hands each slot its lattice (candidates [L, k], log-probabilities [L, k, k]); the tree a slot
verifies is built from it best-first on path probability, the greedy path first (engine/tree.py
`lattice_tree`, whose docstring has the why). In Python that is a few hundred heap operations a slot over
lists the lattice was copied into, on the round's critical path once per slot. Here the same statements run in
C++ over the host tensors: the same float64 sums (`lp + logp[...]`, the float32 entries widened as Python
widens them), the same `exp`, the same heap order ((-p, tie) with the same tie counter), the same
de-duplication of a token under one parent and the same DFS pre-order emission (children by descending score,
first added first among equals). `check()` holds it to the Python function on random lattices.

Built on first use with torch.utils.cpp_extension.load_inline (CPU only), cached by source hash.
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch

_SRC = r"""
#include <torch/extension.h>
#include <cmath>
#include <queue>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace {
struct Item { double negp; long long tie; int parent; int slot; int c; };
struct Cmp { bool operator()(const Item& a, const Item& b) const {
    if (a.negp != b.negp) return a.negp > b.negp;     // a min-heap on (negp, tie), as heapq
    return a.tie > b.tie; } };
}

std::vector<torch::Tensor> lattice_tree_cpp(int64_t anchor, torch::Tensor cand, torch::Tensor logp,
                                            std::vector<int64_t> greedy, int64_t budget) {
    TORCH_CHECK(cand.dim() == 2 && cand.dtype() == torch::kLong && !cand.is_cuda(), "cand: int64 [L, k] on the host");
    TORCH_CHECK(logp.dim() == 3 && !logp.is_cuda(), "logp: [L, k, k] on the host");
    auto lp_d = logp.to(torch::kDouble).contiguous();   // Python widens each float32 entry to a float exactly
    auto cd = cand.contiguous();
    const int L = cd.size(0), k = cd.size(1);
    const int64_t* C = cd.data_ptr<int64_t>();
    const double* P = lp_d.data_ptr<double>();
    std::vector<int64_t> tok{anchor};
    std::vector<int> par{-1};
    std::vector<double> sc{1.0};
    std::vector<int8_t> src{0};
    std::vector<std::vector<std::pair<int64_t, int>>> kids(1);   // per node: (token, node) in insertion order
    {
        pybind11::gil_scoped_release nogil;
        auto add = [&](int parent, int64_t token, double score, int8_t s) -> int {
            for (auto& kv : kids[parent]) {
                if (kv.first == token) {
                    if (score > sc[kv.second]) sc[kv.second] = score;
                    return kv.second;
                }
            }
            int idx = (int)tok.size();
            tok.push_back(token); par.push_back(parent); sc.push_back(score); src.push_back(s);
            kids.emplace_back();
            kids[parent].emplace_back(token, idx);
            return idx;
        };
        std::priority_queue<Item, std::vector<Item>, Cmp> heap;
        long long tie = 0;
        int n = 0, row = 0, parent = 0;
        double lp = 0.0;
        const int G = std::min<int>((int)greedy.size(), L);
        for (int slot = 0; slot < G; ++slot) {
            if (n >= budget) break;
            const int c = (int)greedy[slot];
            const double base = lp;
            lp += P[((size_t)slot * k + row) * k + c];
            const int here = parent;
            parent = add(parent, C[(size_t)slot * k + c], std::exp(lp), 1);
            ++n;
            for (int c2 = 0; c2 < k; ++c2) {
                if (c2 != c) {
                    ++tie;
                    heap.push(Item{-(base + P[((size_t)slot * k + row) * k + c2]), tie, here, slot, c2});
                }
            }
            row = c;
        }
        while (!heap.empty() && n < budget) {
            Item it = heap.top(); heap.pop();
            const double lpp = -it.negp;
            const int nid = add(it.parent, C[(size_t)it.slot * k + it.c], std::exp(lpp), 2);
            ++n;
            if (it.slot + 1 < L) {
                for (int c2 = 0; c2 < k; ++c2) {
                    ++tie;
                    heap.push(Item{-(lpp + P[((size_t)(it.slot + 1) * k + it.c) * k + c2]), tie, nid, it.slot + 1, c2});
                }
            }
        }
    }
    // DFS pre-order, children by descending score, the first added first among equals
    const int N = (int)tok.size();
    auto t_tok = torch::empty({N}, torch::kLong), t_par = torch::empty({N}, torch::kLong);
    auto t_sc = torch::empty({N}, torch::kDouble), t_src = torch::empty({N}, torch::kInt8);
    int64_t* ot = t_tok.data_ptr<int64_t>(); int64_t* op = t_par.data_ptr<int64_t>();
    double* os = t_sc.data_ptr<double>(); int8_t* oz = t_src.data_ptr<int8_t>();
    int out = 0;
    std::vector<std::pair<int, int>> stack{{0, -1}};
    while (!stack.empty()) {
        auto [old, pnew] = stack.back(); stack.pop_back();
        const int nw = out++;
        ot[nw] = tok[old]; op[nw] = pnew; os[nw] = sc[old]; oz[nw] = src[old];
        std::vector<int> ch;
        for (auto& kv : kids[old]) ch.push_back(kv.second);
        std::stable_sort(ch.begin(), ch.end(), [&](int a, int b) { return sc[a] > sc[b]; });
        for (int i = (int)ch.size() - 1; i >= 0; --i) stack.push_back({ch[i], nw});
    }
    return {t_tok, t_par, t_sc, t_src};
}
"""
_CPP = ("std::vector<torch::Tensor> lattice_tree_cpp(int64_t anchor, torch::Tensor cand, torch::Tensor logp, "
        "std::vector<int64_t> greedy, int64_t budget);")
_MOD = None
_NAMES = {0: "root", 1: "greedy", 2: "tree"}


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        tag = hashlib.sha1((_CPP + _SRC).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"qwen38_lattice_{tag}", cpp_sources=[_SRC], functions=["lattice_tree_cpp"],
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
