"""The sampler, unit-tested on a CPU (ENG-19 half one).

What matters for the engine is that `temperature = 0` is the argmax it always was, and that a
sampled row follows the requested distribution: temperature scales, top-k truncates, top-p keeps
the smallest prefix whose mass exceeds the threshold, and a seed reproduces a request exactly.
None of that needs a model -- it is a function of one logits row.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine.sample import Sampler  # noqa: E402


def draw(s, row, n):
    return [s(row) for _ in range(n)]


def test_temperature_zero_is_the_argmax():
    row = torch.tensor([1.0, 5.0, -2.0, 4.0])
    s = Sampler(temperature=0.0)
    assert not s.on
    assert all(t == 1 for t in draw(s, row, 20)), "greedy is untouched"


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


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:48s} ok")
            passed += 1
    print(f"{passed} passed")
