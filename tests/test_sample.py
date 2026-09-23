"""The sampler, unit-tested on a CPU (ENG-19).

What matters for the engine is that `temperature = 0` is the argmax it always was, and that a
sampled row follows the requested distribution: temperature scales, top-k truncates, top-p keeps
the smallest prefix whose mass exceeds the threshold, and a seed reproduces a request exactly.
None of that needs a model -- it is a function of one logits row.

The speculation half gets its own statistical gate: `chain_pick`/`tree_walk` must emit tokens
distributed exactly like direct sampling, because that is the property the pipeline sells
("verify_lossless, but for sampling"). With a deterministic drafter the accept rule is p(d) and
the residual is the draw itself, so the tests compare empirical histograms against the rows.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine.sample import Sampler  # noqa: E402


def draw(s, row, n):
    return [s(row) for _ in range(n)]


def hist(tokens, k):
    n = len(tokens)
    return [tokens.count(i) / n for i in range(k)]


def _rows(n=5):
    # Correlated rows over a 4-token vocabulary, the way a block's rows look: later rows differ
    # from the first, so a test cannot pass by reusing row 0 everywhere.
    g = torch.Generator().manual_seed(0)
    return torch.stack([torch.randn(4, generator=g) * 2.0 for _ in range(n)])


def test_chain_rejection_matches_direct_sampling():
    rows = _rows()
    dists = rows.softmax(-1)                      # what probs_rows(lg) hands chain_pick
    draft = rows.argmax(-1).tolist()[:2]          # a good drafter proposes the argmax chain
    n_trials, kvocab = 20000, rows.shape[1]

    s_spec = Sampler(temperature=1.0, seed=1234)
    s_direct = Sampler(temperature=1.0, seed=1234)
    direct, first, second = [], [], []
    for _ in range(n_trials):
        direct.append(s_direct(rows[0]))
        n_acc, x = s_spec.chain_pick(dists, draft)
        first.append(draft[0] if n_acc >= 1 else x)
        if n_acc >= 1:
            second.append(draft[1] if n_acc >= 2 else x)
    p0, p1 = dists[0], dists[1]
    for got, want, label in ((hist(first, kvocab), p0, "first token"),
                             (hist(second, kvocab), p1, "second token")):
        for t in range(kvocab):
            assert abs(got[t] - want[t]) < 0.02, (
                f"{label} {t}: empirical {got[t]:.4f}, row says {want[t]:.4f}")
    for t in range(kvocab):
        assert abs(hist(direct, kvocab)[t] - p0[t]) < 0.02, "the baseline itself is off"


def test_chain_rejection_accept_rate_is_the_draft_probability():
    # With q = delta_d the textbook accept probability is p(d); the empirical rate must match.
    rows = _rows()
    dists = rows.softmax(-1)
    draft = rows.argmax(-1).tolist()[:2]
    s = Sampler(temperature=1.0, seed=99)
    n_trials = 20000
    accepted = sum(1 for _ in range(n_trials) if s.chain_pick(dists, draft)[0] >= 1)
    want = float(dists[0][draft[0]])
    assert abs(accepted / n_trials - want) < 0.02, f"accept {accepted/n_trials:.4f} vs p(d) {want:.4f}"


def test_tree_walk_matches_direct_sampling():
    # A chain-shaped tree -- [anchor, a, b], each one child -- so the descent is deterministic
    # and `new[1]` is unambiguously a draw from node a's own row. A three-child root would mix
    # three rows into the second token and prove nothing about any single one.
    rows = _rows()
    dists = rows.softmax(-1)
    tokens = [0, 1, 2]
    parents = [-1, 0, 1]
    n_trials, kvocab = 20000, rows.shape[1]
    s_spec = Sampler(temperature=1.0, seed=321)
    s_direct = Sampler(temperature=1.0, seed=321)
    first, deep = [], []
    for _ in range(n_trials):
        s_direct(rows[0])
        path, new = s_spec.tree_walk(dists, tokens, parents)
        first.append(new[0])
        if len(path) >= 2:
            deep.append(new[1])
    p0, p1 = dists[0], dists[1]
    for t in range(kvocab):
        assert abs(hist(first, kvocab)[t] - p0[t]) < 0.02, (
            f"first token {t}: empirical {hist(first, kvocab)[t]:.4f}, row says {p0[t]:.4f}")
    assert deep, "the walk must sometimes descend below the root or the test proves nothing"
    for t in range(kvocab):
        assert abs(hist(deep, kvocab)[t] - p1[t]) < 0.02, (
            f"token below the root {t}: empirical {hist(deep, kvocab)[t]:.4f}, row says {p1[t]:.4f}")


def test_chain_pick_reproduces_with_a_seed():
    rows = _rows()
    a = Sampler(temperature=0.8, top_p=0.9, seed=7)
    b = Sampler(temperature=0.8, top_p=0.9, seed=7)
    da, db = a.probs_rows(rows), b.probs_rows(rows)
    draft = rows.argmax(-1).tolist()[:2]
    assert [a.chain_pick(da, draft) for _ in range(30)] == \
           [b.chain_pick(db, draft) for _ in range(30)]


def test_probs_rows_matches_single_row_filtering():
    rows = _rows()
    s = Sampler(temperature=0.7, top_p=0.85, top_k=3, seed=1)
    got = s.probs_rows(rows)
    for i in range(rows.shape[0]):
        want = s._filter(rows[i].float())
        assert torch.allclose(got[i], want, atol=1e-6), f"row {i} differs"


def test_q_aware_accept_follows_p_when_q_differs():
    # ENG-102: a drafter that samples its proposal carries q; the accept is min(1, p(d)/q(d))
    # with the residual (p - q)+ on rejection. The emitted tokens must follow p, not q, however
    # different the two are -- that is the whole theorem, and it is what the residual draw buys.
    lab = Sampler(temperature=1.0, seed=5)          # drafts d ~ q
    acc = Sampler(temperature=1.0, seed=6)          # the verify's accept draws
    g = torch.Generator().manual_seed(3)
    p_row = torch.randn(6, generator=g).softmax(-1)
    q_row = torch.randn(6, generator=g).softmax(-1)
    dists = torch.stack([p_row, p_row])             # row 0 verifies the draft, row 1 the bonus
    n_trials = 20000
    hist = [0] * 6
    for _ in range(n_trials):
        d = lab.pick(q_row)                         # exactly what the drafter would sample
        n_acc, x = acc.chain_accept(dists, [d], [q_row])
        hist[d if n_acc >= 1 else x] += 1
    for t in range(6):
        assert abs(hist[t] / n_trials - float(p_row[t])) < 0.02, (
            f"token {t}: empirical {hist[t] / n_trials:.4f}, p says {float(p_row[t]):.4f}")


def test_q_aware_survives_a_miscalibrated_q():
    # q uniform over the vocabulary (the worst a drafter can do) must still land on p.
    lab = Sampler(temperature=1.0, seed=15)
    acc = Sampler(temperature=1.0, seed=16)
    g = torch.Generator().manual_seed(9)
    p_row = torch.randn(6, generator=g).softmax(-1)
    q_row = torch.full((6,), 1 / 6)
    dists = torch.stack([p_row, p_row])
    n_trials = 20000
    hist = [0] * 6
    for _ in range(n_trials):
        d = lab.pick(q_row)
        n_acc, x = acc.chain_accept(dists, [d], [q_row])
        hist[d if n_acc >= 1 else x] += 1
    for t in range(6):
        assert abs(hist[t] / n_trials - float(p_row[t])) < 0.02


def test_q_equal_p_accepts_every_draft():
    # The limiting case the speed comes from: a perfect proposal distribution accepts always.
    acc = Sampler(temperature=1.0, seed=21)
    lab = Sampler(temperature=1.0, seed=22)
    g = torch.Generator().manual_seed(4)
    p_row = torch.randn(6, generator=g).softmax(-1)
    dists = torch.stack([p_row, p_row])
    for _ in range(500):
        d = lab.pick(p_row)
        assert acc.chain_accept(dists, [d], [p_row])[0] == 1


def test_q_aware_none_rows_match_the_deterministic_shortcut():
    # A deterministic arm (no q) must behave exactly like chain_pick: same RNG consumption.
    dists = torch.stack([_rows()[0].softmax(-1)] * 2)
    a = Sampler(temperature=1.0, seed=9)
    b = Sampler(temperature=1.0, seed=9)
    assert [a.chain_pick(dists, [2]) for _ in range(30)] == \
           [b.chain_accept(dists, [2], [None]) for _ in range(30)]


def test_q_aware_reproduces_with_a_seed():
    g = torch.Generator().manual_seed(11)
    p_row = torch.randn(6, generator=g).softmax(-1)
    q_row = torch.randn(6, generator=g).softmax(-1)
    dists = torch.stack([p_row, p_row])
    a = Sampler(temperature=1.0, seed=31)
    b = Sampler(temperature=1.0, seed=31)
    lab = Sampler(temperature=1.0, seed=32)
    drafts = [lab.pick(q_row) for _ in range(30)]
    assert [a.chain_accept(dists, [d], [q_row]) for d in drafts] == \
           [b.chain_accept(dists, [d], [q_row]) for d in drafts]


def test_temperature_zero_is_the_argmax():
    row = torch.tensor([1.0, 5.0, -2.0, 4.0])
    s = Sampler(temperature=0.0)
    assert not s.on
    assert all(t == 1 for t in draw(s, row, 20)), "greedy is untouched"


def test_temperature_zero_is_greedy_whatever_top_p_and_top_k_say():
    # OpenAI and vLLM both read `temperature: 0` as greedy, and a client that sends a top-p beside
    # it means "greedy". The engine read it as sampling ON with no temperature division, so
    # `{"temperature": 0, "top_p": 0.95}` sampled the RAW logits -- and skipped the response cache,
    # which only runs when the sampler is off.
    row = torch.tensor([1.0, 5.0, -2.0, 4.0])
    for s in (Sampler(temperature=0.0, top_p=0.95), Sampler(temperature=0.0, top_k=40),
              Sampler(temperature=0.0, top_p=0.9, top_k=20)):
        assert not s.on, "temperature 0 is greedy, and a greedy answer is cacheable"
        assert all(t == 1 for t in draw(s, row, 20))


def test_sampling_follows_a_uniform_distribution():
    row = torch.zeros(16)
    s = Sampler(temperature=1.0, seed=11)
    n = 8000
    counts = [0] * 16
    for t in draw(s, row, n):
        counts[t] += 1
    for c in counts:
        assert abs(c / n - 1 / 16) < 0.02, f"uniform row, empirical {c/n:.4f}"


def test_a_seed_reproduces_a_request_exactly():
    row = torch.linspace(-1.0, 1.0, 32)
    a = draw(Sampler(temperature=0.8, top_p=0.9, seed=7), row, 50)
    b = draw(Sampler(temperature=0.8, top_p=0.9, seed=7), row, 50)
    assert a == b, "the same seed and row must give the same tokens"


def test_top_k_truncates():
    row = torch.tensor([0.0, 10.0, 9.0, -10.0])
    s = Sampler(temperature=1.0, top_k=2, seed=3)
    seen = set(draw(s, row, 500))
    assert seen <= {1, 2}, f"only the top two may appear, saw {seen}"


def test_top_p_keeps_the_mass_prefix():
    # softmax([3, 2, 0]) ~= [0.665, 0.245, 0.090]; top_p 0.8 keeps the first two
    row = torch.tensor([3.0, 2.0, 0.0])
    s = Sampler(temperature=1.0, top_p=0.8, seed=5)
    seen = set(draw(s, row, 500))
    assert seen <= {0, 1}, f"the tail token is out, saw {seen}"
    assert 0 in seen and 1 in seen, "both kept tokens must appear"


def test_temperature_sharpens():
    # logits/0.1 = [0, 20, 10]: the 10-nat gap makes the best token near-certain
    row = torch.tensor([0.0, 2.0, 1.0])
    s = Sampler(temperature=0.1, seed=13)
    counts = [0, 0, 0]
    for t in draw(s, row, 500):
        counts[t] += 1
    assert counts[1] > 490, f"sharpened sampling favors the argmax, got {counts}"


def test_validation():
    for bad in (lambda: Sampler(temperature=-1.0), lambda: Sampler(top_p=0.0),
                lambda: Sampler(top_p=1.5), lambda: Sampler(top_k=-2)):
        try:
            bad()
            raise AssertionError("out-of-range sampler parameter must raise")
        except ValueError:
            pass


def test_the_speculative_helpers_track_no_gradients():
    """ENG-104: `chain_pick` and `tree_walk` are inference-only and must say so themselves, not
    rely on a caller's `no_grad` -- a test or a notebook calling them bare built a graph."""
    seen = []

    class Spy(Sampler):
        def pick(self, row):
            seen.append(torch.is_grad_enabled())
            return super().pick(row)

    s = Spy(temperature=1.0, seed=3)
    rows = torch.softmax(torch.randn(4, 8, requires_grad=True), dim=-1)
    assert torch.is_grad_enabled()
    s.chain_pick(rows, [1, 2, 3])
    s.tree_walk(rows, [0, 1, 2, 3], [-1, 0, 1, 2])
    assert seen and not any(seen), f"grad was on inside a helper: {seen}"

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:48s} ok")
            passed += 1
    print(f"{passed} passed")
