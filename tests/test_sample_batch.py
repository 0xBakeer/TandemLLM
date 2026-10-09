"""The batched sampled walk and the fast top-p filter (engine/sample.py BATCH, TOPP_K) on the CPU.

The batched walk must emit exactly the target's sampled distribution, as the one-draw-a-node walk
does: test_sample.py's statistical gates rerun with BATCH on, plus a q-aware tree (a sampled spine
child carrying its q beside a deterministic sibling), whose first emitted token must follow p
whatever q is. The fast filter must keep the sort's set on rows whose nucleus fits its k and fall
back to the sort on rows whose nucleus is wider.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import test_sample as TS  # noqa: E402
from engine import sample as S  # noqa: E402


def _batched(fn):
    old = S.BATCH
    S.BATCH = True
    try:
        return fn()
    finally:
        S.BATCH = old


def test_batched_tree_walk_matches_direct_sampling():
    _batched(TS.test_tree_walk_matches_direct_sampling)
    return "chain-shaped tree: first and second token follow their rows"


def test_batched_walk_is_exact_on_a_branched_tree():
    _batched(TS.test_tree_walk_is_exact_on_a_branched_tree)
    return "branched tree: every prefix within 0.012 of the chain's probability"


def test_batched_q_aware_tree_follows_p():
    V = 6
    g = torch.Generator().manual_seed(5)
    p0 = (torch.randn(V, generator=g) * 1.2).softmax(-1)
    q_row = (torch.randn(V, generator=g) * 2.0).softmax(-1)       # a miscalibrated proposal
    p1 = (torch.randn(V, generator=g)).softmax(-1)
    s = S.Sampler(temperature=1.0)
    dists = torch.stack([p0, p1, p1])
    counts = [0] * V
    n = 40000
    top = int(p0.argmax())

    def run():
        for _ in range(n):
            # the drafter samples its spine token from q every round; the sibling is
            # deterministic (p's argmax, or the next token when the spine took it)
            spine = int(torch.multinomial(q_row, 1, generator=g))
            sib = top if top != spine else (top + 1) % V
            path, new = s.tree_walk(dists, [0, spine, sib], [-1, 0, 0], q=[None, q_row, None])
            counts[new[0]] += 1
    _batched(run)
    for t in range(V):
        assert abs(counts[t] / n - float(p0[t])) < 0.012, (t, counts[t] / n, float(p0[t]))
    return "q-aware spine + sibling: first token follows p within 0.012"


def test_fast_top_p_keeps_the_sorts_set():
    g = torch.Generator().manual_seed(11)
    rows = torch.randn(200, 50, generator=g) * 3.0
    rows[0] = 0.0                                                  # flat: nucleus wider than k
    s = S.Sampler(temperature=0.7, top_p=0.9)
    old = S.TOPP_K
    try:
        S.TOPP_K = 0
        ref = s.probs_rows(rows)
        S.TOPP_K = 8
        got = s.probs_rows(rows)
    finally:
        S.TOPP_K = old
    assert torch.equal(ref > 0, got > 0), "kept sets differ"
    assert torch.allclose(ref, got, atol=0, rtol=0) or (ref - got).abs().max() < 1e-6
    return "200 rows (one wider than k = 8): same kept set, same probabilities"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:48s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
