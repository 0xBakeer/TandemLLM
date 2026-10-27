"""Parallel requests: the packed draft, on a CPU with a tiny random drafter module.

`forward_block_multi` runs several slots' blocks through ONE module with each 4-bit projection as
one call over all their rows; attention, the convolutions and the norms stay per block. Each
block's output must equal `forward_block` on that block alone, bit for bit where the platform's
matmul computes a row the same whatever rows share the call (probed; the box's CPU BLAS does, and
so does the served skinny kernel up to 32 rows), within 1e-5 otherwise.

Run: python tests/test_packed_draft.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.drafters import dflash2 as D  # noqa: E402
from engine.drafters.dflash2 import DFlash2Config, DFlash2Module  # noqa: E402

V, H, K = 40, 32, 4


def tiny(block: int = 8, seed: int = 0):
    """A two-layer random block drafter (sliding attention, the convolutions, the selector)."""
    raw = {"hidden_size": H, "intermediate_size": 64, "num_hidden_layers": 2,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": V,
           "rms_norm_eps": 1e-6, "layer_types": ["sliding_attention"] * 2, "sliding_window": 2048,
           "dflash_config": {"block_size": block, "conv_kernel_size": 2, "conv_group_size": 8,
                             "mask_token_id": V - 1, "selector_rank": 8, "selector_top_k": K,
                             "target_layer_ids": [1, 2, 3, 4, 5]}}
    cfg = DFlash2Config(raw)
    g = torch.Generator().manual_seed(seed)
    w = {}
    for name, shape in cfg.expected_tensors().items():
        if name.endswith("norm.weight"):
            w[name] = torch.ones(shape)
        else:
            w[name] = torch.randn(shape, generator=g) * 0.3
    embed = torch.randn(V, H, generator=g)
    head = torch.randn(V, H, generator=g)
    return DFlash2Module(cfg, w), embed, head


class Q:
    """Stands in for an NVFP4 weight: not a torch.Tensor, so `_forward_block_steps` hands its
    projections out to be batched."""

    def __init__(self, w):
        self.w = w
        self.N = w.shape[0]


def _quantised(block=8, seed=0):
    m, embed, _ = tiny(block=block, seed=seed)
    for k in list(m.w):
        if any(k.endswith(f"{p}.weight") for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                                                   "gate_proj", "up_proj", "down_proj")):
            m.w[k] = Q(m.w[k])
    return m, embed


def _lin(x, w):
    return torch.nn.functional.linear(x, w.w if isinstance(w, Q) else w)


def _item(m, embed, pos0, ctx, g):
    cfg = m.cfg
    bs = cfg.block_size
    ids = torch.randint(0, cfg.vocab_size - 1, (bs,), generator=g)
    noise = embed[ids]
    positions = torch.arange(pos0, pos0 + bs)
    ctx_kv = [(torch.randn(cfg.num_key_value_heads, ctx, cfg.head_dim, generator=g),
               torch.randn(cfg.num_key_value_heads, ctx, cfg.head_dim, generator=g))
              for _ in range(cfg.num_hidden_layers)]
    return (noise, positions, ctx_kv, torch.arange(pos0 - ctx, pos0), None)


def _row_independent() -> bool:
    torch.manual_seed(0)
    w, x = torch.randn(64, 32), torch.randn(24, 32)
    full = torch.nn.functional.linear(x, w)
    return all(torch.equal(torch.nn.functional.linear(x[:m], w), full[:m]) for m in (8, 16))


EXACT = _row_independent()


def _eq(a, b):
    return torch.equal(a, b) if EXACT else torch.allclose(a, b, rtol=0, atol=1e-5)


def test_packed_blocks_equal_blocks_alone():
    real = D._lin
    D._lin = _lin
    try:
        for n in (2, 3):
            m, embed = _quantised()
            g = torch.Generator().manual_seed(n)
            items = [_item(m, embed, 40 + 9 * j, 20 + 5 * j, g) for j in range(n)]
            packed = D.forward_block_multi(m, items)
            for j, it in enumerate(items):
                alone = m.forward_block(it[0], it[1], it[2], it[3])
                assert _eq(packed[j], alone), (n, j, EXACT)
    finally:
        D._lin = real


def test_the_step_form_equals_forward_block_alone():
    """One block through the generator equals `forward_block` (the lone draft path) bit for bit."""
    real = D._lin
    D._lin = _lin
    try:
        m, embed = _quantised(block=16, seed=3)
        it = _item(m, embed, 100, 60, torch.Generator().manual_seed(9))
        assert torch.equal(D.forward_block_multi(m, [it])[0],
                           m.forward_block(it[0], it[1], it[2], it[3]))
    finally:
        D._lin = real


def test_the_folded_attention_is_the_expanded_one():
    """SPD-203, DF2_PACK_FOLD: the query groups folded into their key/value head's rows give the
    expanded attention's blocks (within float rounding: the drafter's bits may move), with the window mask."""
    real, fold = D._lin, D.PACK_FOLD
    D._lin = _lin
    try:
        m, embed = _quantised(block=16, seed=4)
        g = torch.Generator().manual_seed(11)
        items = [_item(m, embed, 200 + 7 * j, 30 + 9 * j, g) for j in range(3)]
        bs = m.cfg.block_size
        masked = []
        for it in items:
            noise, positions, ctx_kv, ctx_pos, _ = it
            allpos = torch.cat([ctx_pos, positions])
            mask = (positions[:, None] - allpos[None, :]) < 40          # a window that cuts the context
            masked.append((noise, positions, ctx_kv, ctx_pos, (mask, mask)))
        out = {}
        for f in (False, True):
            D.PACK_FOLD = f
            out[f] = D.forward_block_multi(m, items + masked)
        worst = max(float((a - b).abs().max()) for a, b in zip(out[False], out[True]))
        assert worst < 1e-5, worst
    finally:
        D._lin, D.PACK_FOLD = real, fold
    return f"6 blocks (3 masked): folded = expanded within {worst:.1e}"


def test_the_window_with_a_free_tail_is_the_cat():
    """SPD-203, DF2_PACK_TAIL: a window gathered with the block's rows free at its tail, the block's keys
    and values written there, gives the blocks `torch.cat` gives, bit for bit."""
    real = D._lin
    D._lin = _lin
    try:
        m, embed = _quantised(block=16, seed=5)
        g = torch.Generator().manual_seed(13)
        items = [_item(m, embed, 300 + 5 * j, 25 + 4 * j, g) for j in range(3)]
        bs = m.cfg.block_size
        tails = []
        for noise, positions, ctx_kv, ctx_pos, masks in items:
            kv = []
            for ck, cv in ctx_kv:
                n0 = ck.shape[1]
                fk = torch.full((ck.shape[0], n0 + bs, ck.shape[2]), float("nan"))
                fv = torch.full_like(fk, float("nan"))
                fk[:, :n0], fv[:, :n0] = ck, cv
                kv.append((fk, fv, n0))
            tails.append((noise, positions, kv, ctx_pos, masks))
        a, b = D.forward_block_multi(m, items), D.forward_block_multi(m, tails)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    finally:
        D._lin = real
    return "3 blocks: the free-tail window = the cat, bit for bit"


def test_the_batched_pass_is_the_blocks_alone():
    """SPD-203, DF2_PACK_BATCH: the batched block pass (every row-wise statement once over all rows,
    attention once per layer over windows padded to the longest, padded columns masked, the query
    groups folded) gives each block what it computes alone, within float rounding (the drafter's bits
    may move: the attention's tiling and the padded width)."""
    real = D._lin
    D._lin = _lin
    try:
        m, embed = _quantised(block=16, seed=6)
        cfg = m.cfg
        bs, W = cfg.block_size, int(cfg.sliding_window)
        g = torch.Generator().manual_seed(17)
        ctxs = (25, 40, 33)
        items = [_item(m, embed, 400 + 11 * j, c, g) for j, c in enumerate(ctxs)]
        E, L = len(items), max(ctxs)
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        ctx_k, ctx_v = [], []
        for i in range(cfg.num_hidden_layers):
            kb = torch.full((E, nkv, L + bs, hd), float("nan"))
            vb = torch.full_like(kb, float("nan"))
            for e, it in enumerate(items):
                ck, cv = it[2][i]
                kb[e].zero_()
                vb[e].zero_()
                kb[e, :, :ck.shape[1]] = ck
                vb[e, :, :cv.shape[1]] = cv
            ctx_k.append(kb)
            ctx_v.append(vb)
        positions = torch.stack([it[1] for it in items])                 # [E, bs]
        allpos = torch.stack([torch.cat([it[3], torch.zeros(L - it[3].shape[0], dtype=torch.long), it[1]])
                              for it in items])
        seen = torch.stack([torch.cat([torch.ones(c, dtype=torch.bool), torch.zeros(L - c, dtype=torch.bool),
                                       torch.ones(bs, dtype=torch.bool)]) for c in ctxs])
        mask = ((positions[:, :, None] - allpos[:, None, :]) < W) & seen[:, None, :]
        noise = torch.cat([it[0] for it in items])
        out = D.forward_block_batched(m, noise, positions.reshape(-1), ctx_k, ctx_v, mask).view(E, bs, -1)
        worst = max(float((out[e] - m.forward_block(it[0], it[1], it[2], it[3])).abs().max())
                    for e, it in enumerate(items))
        assert worst < 1e-5, worst
    finally:
        D._lin = real
    return f"3 blocks over windows of 25/40/33: batched = alone within {worst:.1e}"


def test_bf16_weights_are_not_handed_out():
    """A plain tensor weight runs inside the block (cuBLAS may pick another algorithm by M): the
    generator yields nothing for an all-bf16 module."""
    m, embed, _ = tiny()
    it = _item(m, embed, 30, 10, torch.Generator().manual_seed(1))
    # SPD-161's DF2_PACK_KP (on by default since SPD-201) hands the convolutions' kernel projection out by
    # design (the drafter's rows may move by an ulp); this is the contract without it
    kp, D.PACK_KP = D.PACK_KP, False
    try:
        gen = m._forward_block_steps(*it)
        try:
            next(gen)
        except StopIteration as fin:
            assert torch.equal(fin.value, m.forward_block(it[0], it[1], it[2], it[3]))
        else:
            raise AssertionError("a bf16 module must not hand out projections")
    finally:
        D.PACK_KP = kp


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
