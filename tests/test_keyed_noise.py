"""SPD-201, QSE/QWEN38_SAMPLE_KEYED_KERNEL: a round's keyed draws from one kernel (engine/keyed_noise.py).

On CUDA (skipped elsewhere):
  1. the kernel's rows are `torch.rand(V, generator=g)` after `g.manual_seed(key)`, bit for bit, for
     vocabulary sizes of one and several of ATen's grid-stride iterations, and its offset step is the
     generator's. The launch policy is ATen's: this test pins it for the torch it runs under;
  2. `round_walks` over seeded slots (random trees, seeds, positions, two sampling profiles) with the
     kernel and without: the same draws, paths, new tokens, and each slot's generator left in the same
     state (seed and offset).

Run: python tests/test_keyed_noise.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import keyed_noise as KN  # noqa: E402
from engine import sample as S  # noqa: E402


def _cuda():
    return torch.cuda.is_available() and KN.HAVE_TRITON


def test_the_kernel_is_torch_rand_bit_for_bit():
    if not _cuda():
        return "skipped: no CUDA device"
    lines = KN.check(n_keys=48, sizes=(1, 5, 256, 1000, 151936, 248320, 300000, 1_000_000))
    return f"torch {torch.__version__}: " + "; ".join(lines)


def _tree(rng, n):
    parents, path = [-1], [0]
    for i in range(1, n):
        path = path[:rng.randint(1, len(path))]
        parents.append(path[-1])
        path.append(i)
    return parents


def _specs(seed_round, V, dev):
    rng = random.Random(seed_round)
    specs = []
    for e in range(8):
        n = rng.randint(1, 16)
        parents = _tree(rng, n)
        prof = (1.0, 0.95, 20) if e % 3 else (0.7, 0.9, 0)
        sm = S.Sampler(temperature=prof[0], top_p=prof[1], top_k=prof[2], seed=1000 + 37 * e + seed_round)
        g = torch.Generator(device=dev)
        g.manual_seed(5)
        rows = torch.randn(n, V, generator=g, device=dev) * 3
        # tokens on the tree: mostly the rows' top tokens, so walks go deep
        top = rows.argmax(-1).tolist()
        tokens = [rng.randrange(V)] + [top[parents[c]] if rng.random() < 0.7 else rng.randrange(V)
                                       for c in range(1, n)]
        for c in range(1, n):
            # most children carry a token their parent's row favours: the walks go deep
            if rng.random() < 0.8:
                rows[parents[c], tokens[c]] += 14.0
        specs.append({"sampler": sm, "rows": rows, "tokens": tokens, "parents": parents,
                      "start": rng.randint(0, 5000), "q": None, "picks": None})
    return specs


def test_round_walks_with_the_kernel_walk_and_leave_the_generators_alike():
    if not _cuda():
        return "skipped: no CUDA device"
    dev = torch.device("cuda")
    V = 248320
    deep = 0
    for r in range(40):
        outs, states = [], []
        for on in (False, True):
            S.KEYED_KERNEL = on
            specs = _specs(r, V, dev)
            outs.append(S.round_walks(specs))
            states.append([(sp["sampler"].generator.initial_seed(), sp["sampler"].generator.get_offset())
                           for sp in specs])
        S.KEYED_KERNEL = False
        assert outs[0] == outs[1], (r, outs)
        assert states[0] == states[1], (r, states)
        deep += sum(len(w[1][0]) > 2 for w in outs[0] if w is not None)
    return f"40 rounds x 8 seeded slots: same paths and tokens, generators alike ({deep} walks past depth 1)"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
