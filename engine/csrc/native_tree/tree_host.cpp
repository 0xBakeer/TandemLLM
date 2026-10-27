
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
