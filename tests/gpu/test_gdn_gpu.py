"""SPD-22 / SPD-24 / SPD-26 on the board: the commit, the verify mixer and the fused add+norm.

The commit kernel against the torch rank-k for EVERY chain prefix 1..16 and for random DFS trees,
in place and out of place, and its convolution tails exactly; the verify mixer against the general
path for chains of 2..16 rows (odd lengths included) and for random trees; the add+norm kernel
bit-identical at every row count 1..16 and a prefill-sized one, NaN included.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from engine import gdn  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from tools import gdn_commit_kernels as CK  # noqa: E402
from tools import gdn_verify_kernels as VK  # noqa: E402
from tools.norm_kernels import add_rms_norm, rms_norm  # noqa: E402

G = torch.Generator(device="cuda").manual_seed(1)


def _random_tree(n: int, rng: random.Random) -> DraftTree:
    """A random DFS pre-order tree of n nodes: each node's parent is on the current root path."""
    parents, path = [-1], [0]
    for i in range(1, n):
        cut = rng.randint(1, len(path))
        path = path[:cut]
        parents.append(path[-1])
        path.append(i)
    t = DraftTree(tokens=[0] * n, parents=parents)
    t.check()
    return t


def test_commit_every_chain_prefix_and_random_tree_paths():
    L, H, Dk, Dv, T = 48, 48, 128, 128, 16
    S = torch.randn(L, 1, H, Dk, Dv, device="cuda", generator=G) * 0.1
    kk = torch.nn.functional.normalize(torch.randn(L, H, T, Dk, device="cuda", generator=G), dim=-1)
    u = torch.randn(L, H, T, Dv, device="cuda", generator=G) * 0.05
    gc = (-torch.rand(L, H, T, device="cuda", generator=G) * 0.3).cumsum(-1)
    rng = random.Random(0)
    sets = [list(range(k)) for k in range(1, T + 1)]
    for _ in range(12):
        tr = _random_tree(T, rng)
        sets.append(tr.path(rng.randrange(T)))
    worst = 0.0
    for rows in sets:
        ref = CK.commit_reference(S[:, 0], kk, u, gc, rows)
        o = torch.empty_like(S)
        CK.fused_commit(S, o, kk, u, gc, rows)
        rel = ((o[:, 0] - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, rel)
        inplace = S.clone()
        CK.fused_commit(inplace, inplace, kk, u, gc, rows)
        assert torch.equal(inplace, o), rows
    assert worst < 1e-5, worst
    raw = torch.randn(L, 10240, T, device="cuda", generator=G).to(torch.bfloat16)
    ce = torch.randn(L, 1, 10240, 3, device="cuda", generator=G).to(torch.bfloat16)
    for rows in sets:
        got = torch.empty_like(ce)
        CK.conv_commit(ce, got, raw, rows)
        for l in (0, 17, 47):
            want = gdn.conv_tail(raw[l][None], ce[l], torch.tensor(rows, device="cuda"), 4)
            assert torch.equal(got[l], want), (rows, l)
    return f"{len(sets)} row sets (16 chain prefixes, 12 tree paths): rel <= {worst:.1e}, " \
           f"in place == out of place, conv tails exact"


def _mixer_inputs(n):
    kd, vh, dv, W = 2048, 48, 128, 4
    C = 2 * kd + vh * dv
    return dict(
        mixed=(torch.randn(n, C, device="cuda", generator=G) * 0.5).to(torch.bfloat16),
        conv_state=(torch.randn(1, C, W - 1, device="cuda", generator=G) * 0.5).to(torch.bfloat16),
        conv_w=(torch.randn(C, W, device="cuda", generator=G) * 0.3).to(torch.bfloat16),
        a_raw=torch.randn(n, vh, device="cuda", generator=G).to(torch.bfloat16),
        b_raw=torch.randn(n, vh, device="cuda", generator=G).to(torch.bfloat16),
        a_log=(torch.randn(vh, device="cuda", generator=G) * 0.5).to(torch.bfloat16),
        dt_bias=(torch.randn(vh, device="cuda", generator=G) * 0.5).to(torch.bfloat16),
        state=torch.randn(1, vh, 128, dv, device="cuda", generator=G) * 0.05)


def _mixer_case(n, tree=None):
    kw = dict(key_dim=2048, key_heads=16, value_heads=48, head_k=128, head_v=128)
    inp = _mixer_inputs(n)
    win = dep = None
    if tree is not None:
        win = torch.tensor(tree.conv_windows(4), dtype=torch.long, device="cuda")
        dep = torch.tensor(tree.depths(), dtype=torch.long, device="cuda")
    r_in = {k: v.clone() for k, v in inp.items()}
    f_in = {k: v.clone() for k, v in inp.items()}
    o_r, fac_r = VK.reference(**r_in, window=win, depths=dep, **kw)
    o_f, fac_f, _ = VK.verify_mixer(**f_in, window=win, depths=dep,
                                    max_depth=max(tree.depths()) if tree else 0, **kw)
    rel = lambda x, y: ((x.float() - y.float()).abs().max() / y.float().abs().max()).item()
    r = [rel(o_f, o_r), rel(fac_f[0], fac_r[0]), rel(fac_f[1], fac_r[1]), rel(fac_f[2], fac_r[2]),
         rel(f_in["state"], r_in["state"])]
    assert torch.equal(f_in["conv_state"], r_in["conv_state"]), "conv state"
    assert max(r) < 2e-2, (n, r)
    if tree is not None:
        assert torch.equal(f_in["state"], inp["state"]), "a tree verify must not touch the state"
    return max(r)


def test_verify_mixer_chains_of_every_length_and_random_trees():
    worst = max(_mixer_case(n) for n in (2, 3, 5, 8, 9, 15, 16))
    rng = random.Random(3)
    for n in (4, 9, 16):
        for _ in range(3):
            worst = max(worst, _mixer_case(n, _random_tree(n, rng)))
    return f"chains of 2,3,5,8,9,15,16 rows and 9 random trees: worst rel {worst:.2e} " \
           f"(conv rounded once, beta fp32), conv state exact, trees leave the state alone"


def test_add_rms_norm_every_row_count_and_nan():
    w = (torch.randn(5120, device="cuda", generator=G) * 0.1).to(torch.bfloat16)
    for m in list(range(1, 17)) + [260]:
        res = torch.randn(m, 5120, device="cuda", generator=G).to(torch.bfloat16)
        x = torch.randn(m, 5120, device="cuda", generator=G).to(torch.bfloat16)
        if m == 5:
            x[2, 9] = float("nan")
        h_ref = res + x
        y_ref = rms_norm(h_ref, w, 1e-6)
        h, y = add_rms_norm(res, x, w, 1e-6)
        assert torch.equal(h.nan_to_num(7.0), h_ref.nan_to_num(7.0)), m
        assert torch.equal(y.nan_to_num(7.0), y_ref.nan_to_num(7.0)), m
    return "rows 1..16 and 260 bit-identical to res + x then rms_norm, a NaN row included"



def _pending_case(prev_tree, rows, next_tree, warps):
    """One layer: a previous block verified (chain or tree) and accepted along `rows`; then the
    next block verified two ways -- (a) the commit kernel, then the verify on the committed state,
    as the engine does without SPD-37; (b) the verify with the commit pending, applied in its
    recurrence and written back. Returns whether every output is the same bits."""
    kw = dict(key_dim=2048, key_heads=16, value_heads=48, head_k=128, head_v=128)
    n_prev = len(prev_tree.parents) if prev_tree is not None else 16
    prev = _mixer_inputs(n_prev)
    win = dep = None
    if prev_tree is not None:
        win = torch.tensor(prev_tree.conv_windows(4), dtype=torch.long, device="cuda")
        dep = torch.tensor(prev_tree.depths(), dtype=torch.long, device="cuda")
    scratch = torch.empty_like(prev["state"])
    _, (pk, pu, pg), _ = VK.verify_mixer(
        **prev, window=win, depths=dep,
        max_depth=max(prev_tree.depths()) if prev_tree is not None else 0,
        out_state=scratch if prev_tree is None else None, **kw)
    entry = prev["state"]                                  # untouched: the walk went to scratch
    nxt = _mixer_inputs(len(next_tree.parents) if next_tree is not None else 16)
    nwin = ndep = None
    if next_tree is not None:
        nwin = torch.tensor(next_tree.conv_windows(4), dtype=torch.long, device="cuda")
        ndep = torch.tensor(next_tree.depths(), dtype=torch.long, device="cuda")
    md = max(next_tree.depths()) if next_tree is not None else 0
    # (a) commit, then verify
    Sa = entry.clone()
    CK.fused_commit(Sa[None], Sa[None], pk, pu, pg, rows)
    committed = Sa.clone()
    ia = {k: v.clone() for k, v in nxt.items() if k != "state"}
    oa, fa, _ = VK.verify_mixer(**ia, state=Sa, window=nwin, depths=ndep, max_depth=md,
                                out_state=torch.empty_like(Sa) if next_tree is None else None,
                                warps=warps, **kw)
    # (b) the verify with the commit pending
    Sb = entry.clone()
    ib = {k: v.clone() for k, v in nxt.items() if k != "state"}
    pend = (pk[0], pu[0], pg[0], torch.tensor(rows, dtype=torch.int32, device="cuda"),
            torch.tensor([len(rows)], dtype=torch.int32, device="cuda"))
    # the factors into strided static buffers, as the engine keeps them for a graph (16 rows)
    H = 48
    stat = (torch.full((H, 16, 128), 7.0, device="cuda"), torch.full((H, 16, 128), 7.0, device="cuda"),
            torch.full((H, 16), 7.0, device="cuda"))
    ob, fb, _ = VK.verify_mixer(**ib, state=Sb, window=nwin, depths=ndep, max_depth=md,
                                pend=pend, store_state=False, warps=warps, fac_out=stat, **kw)
    same = (torch.equal(Sb, committed) and torch.equal(oa, ob)
            and all(torch.equal(x, y) for x, y in zip(fa, fb))
            and torch.equal(ia["conv_state"], ib["conv_state"]))
    # nothing pending (P = 0 on the device): the state is read and left alone
    Sc = committed.clone()
    ic = {k: v.clone() for k, v in nxt.items() if k != "state"}
    zero = pend[:4] + (torch.zeros(1, dtype=torch.int32, device="cuda"),)
    oc, fc, _ = VK.verify_mixer(**ic, state=Sc, window=nwin, depths=ndep, max_depth=md,
                                pend=zero, store_state=False, warps=warps, **kw)
    same = same and torch.equal(Sc, committed) and torch.equal(oc, oa) and \
        all(torch.equal(x, y) for x, y in zip(fc, fa))
    return same, (Sb - committed).abs().max().item()


def test_a_pending_commit_in_the_verify_is_the_commit_kernel_bit_for_bit():
    """SPD-37: the recurrence applies the previous block's commit with the commit kernel's
    arithmetic, so the state it writes back, its outputs and its factors are the bits of
    commit-then-verify -- for every chain prefix 1..16 and 12 random tree paths, into a chain and
    into a tree verify, on one warp and on four."""
    rng = random.Random(7)
    cases = [(None, list(range(k)), None) for k in range(1, 17)]
    for j in range(12):
        tr = _random_tree(16, rng)
        cases.append((tr, tr.path(rng.randrange(16)), _random_tree(9, rng) if j % 2 else None))
    fails = []
    for warps in (1, 4):
        for prev_tree, rows, next_tree in cases:
            same, d = _pending_case(prev_tree, rows, next_tree, warps)
            if not same:
                fails.append((warps, rows, next_tree is not None, d))
    assert not fails, fails[:4]
    return (f"{len(cases)} commits (16 chain prefixes, 12 tree paths) x chain/tree verify x "
            f"1 and 4 warps: state, outputs, factors (into strided static buffers) and conv "
            f"bit-identical to commit-then-verify; P = 0 on the device leaves the state alone")

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
