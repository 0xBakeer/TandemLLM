"""SPD-249, no GPU: the host arrays a verify block stages are the same with the tree plans kept by shape.

  * `_tree_plan(parents)` gives the depth, paths, GDN tree control and slots and the conv / PLE window sources the
    per-call functions give, and its ancestor cells write exactly the cells (and values) of `stage_rows`' loop,
    over random DFS pre-order trees of 1..33 rows, on a buffer with stale values in it;
  * a shape asked again is the kept plan; past the cap the plans start over;
  * `NGramHash.ids_one` with its integer lists kept equals the torch `ids` over random contexts with EOS in them.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.ple import NGramHash  # noqa: E402
from tools import fn_seq_kernels as SK  # noqa: E402


def _tree(rng: random.Random, m: int) -> list[int]:
    """A DFS pre-order parents list of m rows (row 0 the root)."""
    parents, stack = [-1], [0]
    for i in range(1, m):
        while len(stack) > 1 and rng.random() < 0.35:
            stack.pop()
        parents.append(stack[-1])
        stack.append(i)
    return parents


def _eng() -> FlashNextEngine:
    return object.__new__(FlashNextEngine)          # `_tree_plan` reads only the class and its own cache


def test_the_plan_is_the_per_call_arrays():
    rng = random.Random(7)
    eng = _eng()
    n = 0
    for _ in range(400):
        m = rng.randint(1, 33)
        parents = _tree(rng, m)
        tp = eng._tree_plan(parents)
        depth, paths = FlashNextEngine._tree_paths(parents)
        assert tp["depth"].tolist() == depth and tp["paths"] == paths and tp["dmax"] == max(depth)
        ctl, slots = SK.tree_ctl(parents)
        assert tp["slots"] == slots and np.array_equal(tp["ctl"], np.asarray(ctl))
        assert np.array_equal(tp["csrc"], np.asarray(SK.window_src(parents, (3, 2, 1), 3)))
        assert np.array_equal(tp["psrc"], np.asarray(SK.window_src(parents, (9, 6, 3), 9)))
        L = rng.randint(0, 200000)
        stale = np.random.default_rng(n).integers(-5, 10 ** 6, size=(33, 34), dtype=np.int64)
        a, b = stale.copy(), stale.copy()
        for i, pa in enumerate(paths):
            a[i, :len(pa)] = L + np.asarray(pa)
        b[tp["ri"], tp["ci"]] = L + tp["pv"]
        assert np.array_equal(a, b)
        n += 1
    return f"{n} random trees of 1..33 rows: depth, paths, ctl, slots, windows and ancestor cells equal"


def test_a_shape_is_kept_and_the_cap_starts_over():
    eng = _eng()
    p = [-1, 0, 1, 0]
    assert eng._tree_plan(p) is eng._tree_plan(list(p))
    eng.TREE_PLANS_MAX = 3
    for k in range(5):
        eng._tree_plan([-1] + list(range(k + 1)))
    assert len(eng._tplans) <= 3
    return "same dict for a shape asked again; at most the cap kept"


def test_ids_one_with_kept_lists_is_the_torch_ids():
    rng = random.Random(3)
    eos = 7
    sizes = [rng.randint(10 ** 5, 10 ** 7) for _ in range(16)]
    offs = list(np.cumsum([0] + sizes[:-1]))
    h = NGramHash(vocab_size=200000, ngram_size=3, heads_per_ngram=8, eos=eos,
                  multipliers=[rng.randint(1, 2 ** 62) for _ in range(3)], sizes=sizes, offsets=[int(x) for x in offs])
    n = 0
    for _ in range(300):
        ctx = [rng.choice([eos, rng.randint(0, 199999)]) for _ in range(2)]
        tok = rng.choice([eos, rng.randint(0, 199999)])
        want = h.ids(torch.tensor([[tok]]), torch.tensor([ctx]))[0, 0].tolist()
        assert h.ids_one(tok, ctx) == want, (ctx, tok)
        n += 1
    assert "_lists" in h.__dict__
    return f"{n} contexts (EOS among them): ids_one == ids"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
