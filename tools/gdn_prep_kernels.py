"""A GDN prefill chunk's preparation as one kernel (SPD-175): the causal depthwise convolution over the
qkv projection, its SiLU, and the l2 normalisation of q and k, from the projection's rows straight into
the three tensors `tools.gdn_prefill_kernels.fused_chunk_prefill` reads.

The path it replaces (engine/model.py `linear_attention`, `engine/gdn.py` conv_update / conv_prefill
and l2norm) writes the transposed projection, concatenates the convolution state in front of it, runs
cuDNN's depthwise convolution, a SiLU, then widens q and k to the value heads and normalises them in
bf16, and copies each of q, k and v to a contiguous tensor: about 0.8 s of an 8k prefill in a dozen
kernels. Here every element is read once and written once.

NUMERICS. The convolution sums its four taps in fp32, oldest first, and rounds to bf16; the SiLU runs
in fp32 on that and rounds to bf16 (the reference's two roundings). The l2 normalisation keeps the
reference's bf16 cast points (the squares, their sum, the sum plus eps, the reciprocal square root,
the product), but the order inside the convolution and the sum of squares is this kernel's: the bits
are not the reference path's, so the kernel ships behind QWEN38_GDN_PREP and the prefill quality gate
(tools/prefill_gate.py) decides.

q and k come out at the key heads (16), not widened: the scan reads key head h // rep itself
(the NOREP layout).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _prep(Y, CS, W, Q, K, V, T, s_y, NQK: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
              HAS_STATE: tl.constexpr, EPS: tl.constexpr):
        """One (row block, 128-channel head) of the qkv projection: conv + SiLU, l2 norm for q, k."""
        tb = tl.program_id(0)
        hh = tl.program_id(1)
        t = tb * BT + tl.arange(0, BT)
        ch = hh * D + tl.arange(0, D)
        mt = t < T
        acc = tl.zeros((BT, D), dtype=tl.float32)
        for j in tl.static_range(4):
            r = t - 3 + j
            inr = mt & (r >= 0)
            x = tl.load(Y + r[:, None] * s_y + ch[None, :], mask=inr[:, None], other=0.0).to(tl.float32)
            if HAS_STATE:
                ins = mt & (r < 0)
                xs = tl.load(CS + ch[None, :] * 3 + (r[:, None] + 3), mask=ins[:, None],
                             other=0.0).to(tl.float32)
                x = x + xs
            w = tl.load(W + ch * 4 + j).to(tl.float32)
            acc = acc + x * w[None, :]
        c = acc.to(tl.bfloat16).to(tl.float32)
        s = (c / (1.0 + tl.exp(-c))).to(tl.bfloat16)
        if hh < 2 * NQK:
            sf = s.to(tl.float32)
            sq = (sf * sf).to(tl.bfloat16).to(tl.float32)
            ss = tl.sum(sq, axis=1).to(tl.bfloat16).to(tl.float32)
            den = (ss + EPS).to(tl.bfloat16).to(tl.float32)
            rs = tl.rsqrt(den).to(tl.bfloat16).to(tl.float32)
            o = (sf * rs[:, None]).to(tl.bfloat16)
            if hh < NQK:
                tl.store(Q + t[:, None] * (NQK * D) + (hh * D + tl.arange(0, D))[None, :], o,
                         mask=mt[:, None])
            else:
                tl.store(K + t[:, None] * (NQK * D) + ((hh - NQK) * D + tl.arange(0, D))[None, :], o,
                         mask=mt[:, None])
        else:
            nv = tl.num_programs(1) - 2 * NQK
            tl.store(V + t[:, None] * (nv * D) + ((hh - 2 * NQK) * D + tl.arange(0, D))[None, :], s,
                     mask=mt[:, None])


def gdn_prep(qkv: torch.Tensor, conv_state: torch.Tensor | None, weight: torch.Tensor, n_qk: int,
             n_v: int, d: int = 128, eps: float = 1e-6, bt: int = 32):
    """qkv [T, C] bf16 (row stride free, columns contiguous; C = (2 n_qk + n_v) d), conv_state
    [C, 3] or None (a fresh sequence), weight [C, 4]. Returns q [1, T, n_qk, d], k likewise
    (l2-normalised), v [1, T, n_v, d], and the new convolution state [C, 3]."""
    T, C = qkv.shape
    assert C == (2 * n_qk + n_v) * d and qkv.stride(1) == 1 and weight.shape == (C, 4)
    q = torch.empty(T, n_qk, d, dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty_like(q)
    v = torch.empty(T, n_v, d, dtype=torch.bfloat16, device=qkv.device)
    cs = conv_state.reshape(C, 3).contiguous() if conv_state is not None else qkv
    w = weight.reshape(C, 4).contiguous()
    _prep[(triton.cdiv(T, bt), 2 * n_qk + n_v)](qkv, cs, w, q, k, v, T, qkv.stride(0),
                                                 NQK=n_qk, D=d, BT=bt, HAS_STATE=conv_state is not None,
                                                 EPS=eps, num_warps=4)
    # the new state: the last three inputs, the old state's tail in front of a short chunk
    if T >= 3:
        new = qkv[T - 3:T].t()
    else:
        old = (conv_state.reshape(C, 3) if conv_state is not None
               else torch.zeros(C, 3, dtype=qkv.dtype, device=qkv.device))
        new = torch.cat([old, qkv.t()], dim=1)[:, -3:]
    return (q.view(1, T, n_qk, d), k.view(1, T, n_qk, d), v.view(1, T, n_v, d),
            new.contiguous())


def chunk_prefill_normed(q, k, v, g, beta, state=None, chunk_size: int = 64, bv: int | None = None,
                         warps: int | None = None):
    """`tools.gdn_prefill_kernels.fused_chunk_prefill` for q and k that come l2-normalised and contiguous
    (`gdn_prep`): its two kernels, its launch, its knobs; nothing in that file changes."""
    from tools import gdn_prefill_kernels as G
    B, T, Hk, Dk = q.shape
    H, Dv = v.shape[2], v.shape[-1]
    rep = H // Hk
    bv = bv or G.BV
    warps = warps or G.WARPS_SCAN
    if B != 1 or chunk_size != G.CHUNK or H % Hk or Dv % bv:
        raise ValueError("chunk_prefill_normed: one sequence, the kernel's chunk and blocking")
    qq, kk = q.reshape(T, Hk, Dk), k.reshape(T, Hk, Dk)
    vv = v.reshape(T, H, Dv).contiguous()
    gg = g.reshape(T, H).contiguous()
    bb = beta.reshape(T, H).contiguous()
    nc = (T + chunk_size - 1) // chunk_size
    ub = torch.empty(H, T, Dv, dtype=torch.float32, device=q.device)
    kcd = torch.empty(H, T, Dk, dtype=torch.float32, device=q.device)
    out = torch.empty(T, H, Dv, dtype=q.dtype, device=q.device)
    S = (torch.zeros(1, H, Dk, Dv, dtype=torch.float32, device=q.device)
         if state is None else state.clone().float().reshape(1, H, Dk, Dv))
    prec, prec_scan, prec_a = G.PREC, G.PREC_SCAN or G.PREC, G.PREC_A
    G._intra[(H, nc)](kk, vv, gg, bb, ub, kcd, T, kk.stride(0), kk.stride(1), vv.stride(0), vv.stride(1),
                      gg.stride(0), gg.stride(1), C=chunk_size, DK=Dk, DV=Dv, STEPS=G.STEPS, PREC=prec,
                      SERIES=G.SERIES, REP=rep, num_warps=G.WARPS_INTRA)
    G._scan[(H, Dv // bv)](qq, kk, gg, ub, kcd, out, S, T, nc, qq.stride(0), qq.stride(1), gg.stride(0),
                             gg.stride(1), out.stride(0), out.stride(1), C=chunk_size, DK=Dk, DV=Dv, BV=bv,
                             SCALE=Dk ** -0.5, PREC=prec_scan, PREC_A=prec_a, REP=rep,
                             num_warps=warps, num_stages=G.STAGES_SCAN)
    return out.reshape(B, T, H, Dv), S.reshape(B, H, Dk, Dv)


def reference(qkv, conv_state, weight, n_qk, n_v, d=128, eps=1e-6):
    """The engine's path on the same inputs (engine/gdn.py), for the check."""
    from engine import gdn
    T, C = qkv.shape
    x = qkv.t().unsqueeze(0)
    if conv_state is not None:
        st = conv_state.reshape(1, C, 3).clone()
        mixed = gdn.conv_update(x, st, weight)
        new = st[0]
    else:
        mixed, new = gdn.conv_prefill(x, weight)
        new = new[0]
    mixed = mixed.transpose(1, 2)
    q, k, v = mixed.split([n_qk * d, n_qk * d, n_v * d], dim=-1)
    q = gdn.l2norm(q.reshape(1, T, n_qk, d), dim=-1, eps=eps)
    k = gdn.l2norm(k.reshape(1, T, n_qk, d), dim=-1, eps=eps)
    return q, k, v.reshape(1, T, n_v, d), new


def check(T: int = 1024, n_qk: int = 16, n_v: int = 48, d: int = 128, state: bool = True, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    C = (2 * n_qk + n_v) * d
    y = (torch.randn(T, C + 6144, device="cuda") * 0.7).bfloat16()[:, :C]
    cs = (torch.randn(C, 3, device="cuda") * 0.7).bfloat16() if state else None
    w = (torch.randn(C, 4, device="cuda") * 0.5).bfloat16()
    got = gdn_prep(y, cs, w, n_qk, n_v, d)
    ref = reference(y, cs, w, n_qk, n_v, d)
    out = {}
    for name, a, b in zip(("q", "k", "v", "state"), got, ref):
        a, b = a.float(), b.float()
        out[name] = {"equal_frac": float((a == b).float().mean()), "max_abs": float((a - b).abs().max()),
                     "scale": float(b.abs().max())}
    return out


if __name__ == "__main__":
    import json
    for T in (1, 2, 64, 1024, 1350):
        for st in (False, True):
            print(T, st, json.dumps(check(T=T, state=st)))
