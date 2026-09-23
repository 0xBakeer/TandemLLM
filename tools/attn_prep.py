"""An attention layer's query/key glue in one launch: the q and k norms and the partial rotary.

Between the q/k/v projections and the attention kernel every one of the sixteen attention layers ran
the norms and the rotary as torch operations: a copy of the query half out of the interleaved
q|gate projection, two RMS norms, two gathers of the rotary table, and for each of q and k a slice,
a multiply, a `rotate_half` (chunk, negate, concatenate), a multiply, an add and a concatenate --
about seventeen launches a layer for 27 KB of data at sixteen rows (SPD-40, the block budget's
"attention glue").

Here it is one program per (row, head): the row's 256 values are normalised as
`tools/norm_kernels.py::_rms_norm` normalises them (same block, same warps, so the same reduction),
rounded to bf16, and the first 64 dimensions rotated the way `Qwen38Engine.apply_rope` rotates them
in bf16 -- `x * cos` rounded, `rotate_half(x) * sin` rounded, their sum rounded. The partner value a
dimension needs for `rotate_half` is recomputed from its own input with the same `rstd`, which is
the same number the reference's normalised tensor holds there. So the output is the reference's,
bit for bit (`check()`), written straight into the [heads, rows, 256] layout attention reads.
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
    def _attn_prep(Q, s_qt, s_qh, K, s_kt, s_kh, WQ, WK, COS, SIN, POS, QO, KO, T,
                   HQ: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr):
        t = tl.program_id(0)
        j = tl.program_id(1)
        cols = tl.arange(0, D)
        is_q = j < HQ
        if is_q:
            base = Q + t * s_qt + j * s_qh
            W = WQ
            out = QO + (j * T + t) * D
        else:
            base = K + t * s_kt + (j - HQ) * s_kh
            W = WK
            out = KO + ((j - HQ) * T + t) * D
        x = tl.load(base + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x) / D + EPS)
        w = tl.load(W + cols).to(tl.float32)
        y = (x * rstd * (1.0 + w)).to(tl.bfloat16)
        # rotate_half over the first R dims: [-x2, x1], the partner of i is i ^ (R / 2)
        rot = cols < R
        pc = tl.where(rot, cols ^ (R // 2), cols)
        xp = tl.load(base + pc).to(tl.float32)
        wp = tl.load(W + pc).to(tl.float32)
        yp = (xp * rstd * (1.0 + wp)).to(tl.bfloat16).to(tl.float32)
        yp = tl.where(cols < R // 2, -yp, yp)
        pos = tl.load(POS + t)
        c = tl.load(COS + pos * R + cols, mask=rot, other=0.0).to(tl.float32)
        s = tl.load(SIN + pos * R + cols, mask=rot, other=0.0).to(tl.float32)
        a = (y.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
        b = (yp * s).to(tl.bfloat16).to(tl.float32)
        r = (a + b).to(tl.bfloat16)
        tl.store(out + cols, tl.where(rot, r, y))


def attn_prep(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
              cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor,
              eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """q [T, Hq, D] and k [T, Hk, D] (any row / head strides, unit last stride), the rotary tables
    [P, R] and the rows' positions [T] -> (q [1, Hq, T, D], k [1, Hk, T, D]) bf16, contiguous:
    `apply_rope(rms_norm(q).transpose, rms_norm(k).transpose, cos[positions], sin[positions])`."""
    T, Hq, D = q.shape
    Hk = k.shape[1]
    R = cos.shape[-1]
    assert q.stride(2) == 1 and k.stride(2) == 1 and cos.is_contiguous() and sin.is_contiguous()
    qo = torch.empty(1, Hq, T, D, dtype=q.dtype, device=q.device)
    ko = torch.empty(1, Hk, T, D, dtype=k.dtype, device=k.device)
    _attn_prep[(T, Hq + Hk)](q, q.stride(0), q.stride(1), k, k.stride(0), k.stride(1), wq, wk,
                             cos, sin, positions, qo, ko, T,
                             HQ=Hq, D=D, R=R, EPS=eps, num_warps=4)
    return qo, ko


def reference(q, k, wq, wk, cos, sin, positions, eps):
    """The engine's own path: the fused RMS norm, a transpose, `apply_rope` in bf16."""
    from tools.norm_kernels import rms_norm
    R = cos.shape[-1]
    qn = rms_norm(q[None], wq, eps).transpose(1, 2)
    kn = rms_norm(k[None], wk, eps).transpose(1, 2)
    c = cos[positions][None, None]
    s = sin[positions][None, None]

    def rot(x):
        xr, xp = x[..., :R], x[..., R:]
        a, b = xr.chunk(2, dim=-1)
        return torch.cat([xr * c + torch.cat([-b, a], dim=-1) * s, xp], dim=-1)

    return rot(qn), rot(kn)


def where_differs(a, b) -> str:
    """Which output, how many elements, which rows / heads / dims, and by how much."""
    out = []
    for name, x, y in (("q", a[0], b[0]), ("k", a[1], b[1])):
        d = (x.float() - y.float()).abs()
        bad = (d > 0).nonzero()
        if bad.numel():
            h, t, c = bad[:, 1], bad[:, 2], bad[:, 3]
            out.append(f"{name}: {bad.shape[0]} of {d.numel()} differ, max {d.max().item():.3e}, "
                       f"dims {sorted(set(c.tolist()))[:12]}, rows {sorted(set(t.tolist()))[:8]}, "
                       f"heads {sorted(set(h.tolist()))[:8]}")
    return "; ".join(out) or "equal"


def check(seed: int = 0) -> list[str]:
    """Against the engine's path at 1, 8 and 16 rows, chain and tree positions, and a query that is
    a strided slice of the interleaved q|gate projection, as the engine hands it over."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    Hq, Hk, D, R, P = 24, 4, 256, 64, 4096
    inv = 1.0 / (1e7 ** (torch.arange(0, R, 2, dtype=torch.float32, device="cuda") / R))
    fr = torch.arange(P, dtype=torch.float32, device="cuda")[:, None] * inv[None]
    emb = torch.cat([fr, fr], dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    wq = (torch.randn(D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    wk = (torch.randn(D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    out = []
    for T, pos in ((1, [3001]), (8, list(range(700, 708))), (16, list(range(40, 56))),
                   (16, [900 + d for d in (0, 1, 1, 2, 3, 2, 3, 4, 1, 2, 5, 6, 7, 8, 9, 10)])):
        qg = torch.randn(T, Hq, 2 * D, device="cuda", generator=g).to(torch.bfloat16)
        q = qg[..., :D]                                   # a strided view, as in the engine
        k = torch.randn(T, Hk, D, device="cuda", generator=g).to(torch.bfloat16)
        p = torch.tensor(pos, device="cuda")
        a = attn_prep(q, k, wq, wk, cos, sin, p, 1e-6)
        b = reference(q, k, wq, wk, cos, sin, p, 1e-6)
        same = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        d = max((a[0].float() - b[0].float()).abs().max().item(),
                (a[1].float() - b[1].float()).abs().max().item())
        out.append(f"T={T}: {'bit-identical' if same else where_differs(a, b)}")
    return out


if __name__ == "__main__":
    print("\n".join(check()))
