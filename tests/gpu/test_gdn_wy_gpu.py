"""SPD-38 on the board: the WY recurrence against a float64 walk and against the sequential kernels.

`tools/gdn_wy_kernels.py` computes the verify block's recurrence with every row at once. It is the
same mathematics as `_block_step` / `_tree_step` in another order, so the yardstick is a float64
walk of the same inputs: the per-row updates (the commit's factors) and a chain's walked state must be
as close to it as the sequential kernels are, the output as close as its bf16 rounding allows; the
state carried over 1,024 tokens must stay within 1e-6 of the sequential kernel's; and a pending
commit (SPD-37) applied in the recurrence must be the commit kernel's bits.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from engine.tree import DraftTree  # noqa: E402
from tools import gdn_commit_kernels as CK  # noqa: E402
from tools import gdn_verify_kernels as VK  # noqa: E402
from tools import gdn_wy_kernels as WK  # noqa: E402


def _tree(n, rng, depth_cap=15):
    parents, path = [-1], [0]
    for i in range(1, n):
        path = path[:min(rng.randint(1, len(path)), depth_cap)]
        parents.append(path[-1])
        path.append(i)
    t = DraftTree(tokens=[0] * n, parents=parents)
    t.check()
    return t


def test_wy_is_as_close_to_float64_as_the_walk():
    rng = random.Random(3)
    worst = {}
    cases = [(n, None, 0.0) for n in (2, 3, 8, 9, 16, 17, 24, 32)]
    cases += [(n, _tree(n, rng), 0.0) for n in (4, 9, 16, 24, 32)]
    cases += [(16, None, 3.0), (32, _tree(32, rng), 3.0)]
    for (n, tree, corr), fused in [(c, f) for c in cases for f in (False, True)]:
        r = WK.compare(n, tree, seed=n, corr=corr, fused=fused)
        assert r["u wy"] < 1e-6, (n, tree is not None, fused, r)
        assert r.get("S wy", 0.0) < 2e-6, (n, r)
        assert r["o wy"] <= max(1.2 * r["o seq"], 5e-3), (n, r)
        assert r["conv"] == 0.0, (n, r)                     # a chain's conv state: exact
        if not fused:
            assert r["kk"] < 1e-6 and r["gc"] < 1e-6, (n, r)
        for k, v in r.items():
            worst[k] = max(worst.get(k, 0.0), v)
    return "chains 2..32, 5 trees, correlated keys, conv+gates apart and fused: " + "  ".join(
        f"{k} {v:.1e}" for k, v in worst.items()
        if k in ("u seq", "u wy", "S wy", "o wy", "conv rows differ"))


def test_the_state_over_1024_tokens():
    d = WK.drift(blocks=64, T=16)
    assert d < 1e-6, d
    return f"64 chain blocks of 16, WY against sequential: {d:.1e}"


def test_a_pending_commit_is_the_commit_kernel_bit_for_bit():
    """(a) commit kernel, then the WY verify; (b) the WY verify with the commit pending: the state it
    writes back, its outputs and its factors are the same bits (commits of 2+ rows; a one-row commit
    is the commit kernel's P = 1 specialisation, SPD-37's documented ulp)."""
    kw = WK.KW
    gen = torch.Generator(device="cuda").manual_seed(5)
    rng = random.Random(8)
    n_prev = 16
    checked = 0
    cases = ((2, None), (7, None), (16, None), (5, _tree(9, rng)), (16, _tree(24, rng)))
    for (rows_n, nxt_tree), fused in [(c, f) for c in cases for f in (False, True)]:
        prev = WK._inputs(n_prev, gen)
        scratch = torch.empty_like(prev["state"])
        _, (pk, pu, pg), _ = VK.verify_mixer(**prev, **kw, out_state=scratch, wy=True)
        entry = prev["state"]
        rows = list(range(rows_n))
        nt = len(nxt_tree.parents) if nxt_tree is not None else 16
        nxt = WK._inputs(nt, gen)
        ta = WK._tree_args(nxt_tree) if nxt_tree is not None else {}
        Sa = entry.clone()
        CK.fused_commit(Sa[None], Sa[None], pk, pu, pg, rows)
        committed = Sa.clone()
        ia = {k: v.clone() for k, v in nxt.items() if k != "state"}
        oa, fa, _ = VK.verify_mixer(**ia, state=Sa, **ta, **kw, wy=True, fused=fused,
                                    out_state=torch.empty_like(Sa) if nxt_tree is None else None)
        Sb = entry.clone()
        ib = {k: v.clone() for k, v in nxt.items() if k != "state"}
        pend = (pk[0], pu[0], pg[0], torch.tensor(rows, dtype=torch.int32, device="cuda"),
                torch.tensor([len(rows)], dtype=torch.int32, device="cuda"))
        ob, fb, _ = VK.verify_mixer(**ib, state=Sb, **ta, **kw, wy=True, fused=fused,
                                    pend=pend, store_state=False)
        assert torch.equal(Sb, committed), (rows_n, (Sb - committed).abs().max().item())
        assert torch.equal(oa, ob) and all(torch.equal(x, y) for x, y in zip(fa, fb)), rows_n
        assert torch.equal(ia["conv_state"], ib["conv_state"]), rows_n
        checked += 1
    return (f"{checked} commits of 2..16 rows into chain and tree verifies, conv+gates apart and "
            f"fused: state, outputs, factors and conv state bit-identical")


def test_a_tree_without_its_mask_is_refused():
    x = WK._inputs(4, torch.Generator(device="cuda").manual_seed(1))
    t = DraftTree(tokens=[0] * 4, parents=[-1, 0, 0, 1])
    ta = WK._tree_args(t)
    ta.pop("anc")
    try:
        VK.verify_mixer(**x, **WK.KW, **ta, wy=True)
    except ValueError as e:
        assert "ancestor mask" in str(e)
    else:
        raise AssertionError("a tree without its ancestor mask must be refused")
    return "refused with a ValueError"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
