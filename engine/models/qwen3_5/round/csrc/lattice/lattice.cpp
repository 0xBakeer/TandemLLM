
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
