"""The EXL3 prefill slab product with its output rotation in the epilogue (SPD-175).

`tools/exl3_linear.py` `_slab_matmul` runs a prefill projection as the library GEMM of the rotated
input against a decoded fp16 slab, kept in fp32, and then `rot_out_kernel`: a 128-point Walsh-Hadamard
over each 128-column block of every row, the 1/sqrt(128) scale and the per-column scale svh, rounded
to bf16. At 8k that is the fp32 product written and read back once per projection (about 1 GB a
layer) and a sixth of the prefill's non-GEMM time.

Here the product and the rotation are one Triton kernel. Its main loop is the plain tl.dot loop over
K in ascending order with an fp32 accumulator, which gives the library GEMM's bits on every slab
shape and row count measured on GB10 (2026-10-23, 5 tilings x 6 shapes x M in 700..8192, all equal);
its epilogue is `wht128`'s butterfly in the same stage order (bit 0 first), the same fp32 operations
in the same order, and the same rounding. So the output is the shipped path's, bit for bit, and the
row's bits do not depend on M (the K order never does).

The 27B's own module: only engine/model.py imports it (behind QWEN38_SLAB_TRITON and the window knobs),
and it reads tools/exl3_linear.py without changing it, so no other family's rows or fingerprint move.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

RSQRT128 = tl.constexpr(0.08838834764831845) if HAVE_TRITON else 0.08838834764831845


if HAVE_TRITON:

    @triton.jit
    def _wht_stage(x, R: tl.constexpr, H: tl.constexpr):
        """One butterfly stage over [R * 2 * H] values viewed as (R, 2, H): low + high, low - high."""
        x = tl.reshape(x, (R, 2, H))
        x = tl.permute(x, (0, 2, 1))
        a, b = tl.split(x)
        x = tl.join(a + b, a - b)
        return tl.permute(x, (0, 2, 1))

    @triton.jit
    def _wht128_rows(x, BM: tl.constexpr, BN: tl.constexpr):
        """The unnormalised natural-order Walsh-Hadamard over each 128-column block of x [BM, BN]:
        stage s pairs column i with i ^ 2^s, low + high and low - high, bit 0 first (`wht128`)."""
        R: tl.constexpr = BM * (BN // 128)
        x = tl.reshape(x, (R * 64, 2))
        a, b = tl.split(x)
        x = tl.join(a + b, a - b)
        x = _wht_stage(x, R * 32, 2)
        x = _wht_stage(x, R * 16, 4)
        x = _wht_stage(x, R * 8, 8)
        x = _wht_stage(x, R * 4, 16)
        x = _wht_stage(x, R * 2, 32)
        x = _wht_stage(x, R, 64)
        return tl.reshape(x, (BM, BN))

    @triton.jit
    def _slab_rot(A, B, SV, C, M, N, K, s_am, s_bk, s_cm,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                  OUT_F32: tl.constexpr):
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (GROUP * npn)
        fm = gid * GROUP
        gs = min(npm - fm, GROUP)
        pm = fm + (pid % gs)
        pn = (pid % (GROUP * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm[:, None] < M
        a = A + rm[:, None] * s_am + rk[None, :]
        b = B + rk[:, None] * s_bk + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(a, mask=mm, other=0.0)
            y = tl.load(b)
            acc = tl.dot(x, y, acc)
            a += BK
            b += BK * s_bk
        v = _wht128_rows(acc, BM, BN)
        sv = tl.load(SV + rn).to(tl.float32)
        r = v * RSQRT128 * sv[None, :]
        if OUT_F32:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], r, mask=mm)
        else:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], r.to(tl.bfloat16), mask=mm)


if HAVE_TRITON:

    @triton.jit
    def _add_norm_rot(R, X, W, H, Y, SUH, XH, M, N, EPS: tl.constexpr, BLOCK: tl.constexpr,
                      G: tl.constexpr):
        """`_add_rms_norm` (tools/norm_kernels.py) and then `rot_in_kernel` of its bf16 output for each
        of the G members' sign vectors: the same operations in the same order, one launch."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        r = tl.load(R + row * N + cols, mask=m, other=0.0).to(tl.float32)
        a = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
        h = (r + a).to(H.dtype.element_ty)
        tl.store(H + row * N + cols, h, mask=m)
        x = h.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x) / N + EPS)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = (x * rstd * (1.0 + w)).to(Y.dtype.element_ty)
        tl.store(Y + row * N + cols, y, mask=m)
        yf = y.to(tl.float32)
        for g in tl.static_range(G):
            su = tl.load(SUH + g * N + cols, mask=m, other=0.0).to(tl.float32)
            v = tl.reshape(yf * su, (BLOCK // 128, 128))
            v = _wht128_rows(v, BLOCK // 128, 128)
            o = tl.reshape((v * RSQRT128).to(tl.float16), (BLOCK,))
            tl.store(XH + (g * M + row) * N + cols, o, mask=m)


def add_norm_rot(res: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, suh: torch.Tensor,
                 out: tuple) -> torch.Tensor:
    """`tools.norm_kernels.add_rms_norm(res, x, weight, eps, out=out)` and the rotated input of a
    projection whose input signs are suh [G, N] (`tools.exl3_linear.rotate_in` of the norm's output):
    returns xh [G, M, N] fp16."""
    n = x.shape[-1]
    r = res.reshape(-1, n)
    a = x.reshape(-1, n).contiguous()
    h, y = (t.view(-1, n) for t in out)
    M = r.shape[0]
    G = suh.shape[0]
    xh = torch.empty(G, M, n, dtype=torch.float16, device=x.device)
    blk = 1
    while blk < n:
        blk *= 2
    _add_norm_rot[(M,)](r, a, weight, h, y, suh, xh, M, n, EPS=eps, BLOCK=blk, G=G,
                        num_warps=8 if n >= 4096 else 4)
    return xh


#: (BM, BN, BK, warps, stages). The tiling never changes a row's bits (every tiling gives the library
#: GEMM's), so it is picked for speed alone: 128 x 256 x 64 is the fastest or within 4 % on every 27B slab
#: shape at 700-8,192 rows (2026-10-23 sweep, raw slab-rot.json), except the narrow 1,024-column members
#: and the shortest calls, where 128 x 128 x 32 wins
WIDE = (128, 256, 64, 8, 3)
NARROW = (128, 128, 32, 4, 4)


def pick(M: int, N: int) -> tuple:
    return WIDE if (N % 256 == 0 and N >= 2048 and M > 768) else NARROW


def slab_rot(xh: torch.Tensor, wi: torch.Tensor, svh: torch.Tensor, out: torch.Tensor,
             tile: tuple | None = None) -> torch.Tensor:
    """out[M, n] = rot_out(xh[M, K] @ wi[K, n]) with the column scales svh[n]; xh, wi fp16
    (contiguous rows), out bf16 or fp32 (any row stride, unit column stride). n % 128 == 0."""
    M, K = xh.shape
    N = wi.shape[1]
    assert N % 128 == 0 and xh.stride(1) == 1 and wi.stride(1) == 1 and out.stride(1) == 1
    bm, bn, bk, w, st = tile or pick(M, N)
    assert K % bk == 0 and N % bn == 0
    grid = (triton.cdiv(M, bm) * (N // bn),)
    _slab_rot[grid](xh, wi, svh, out, M, N, K, xh.stride(0), wi.stride(0), out.stride(0),
                    BM=bm, BN=bn, BK=bk, GROUP=8, OUT_F32=out.dtype == torch.float32,
                    num_warps=w, num_stages=st)
    return out


# ------------------------------------------------------------------- the 27B's prefill entry points
# Only engine/model.py imports this module (the 27B's arithmetic manifest); the shared EXL3 path in
# tools/exl3_linear.py is read, never changed, so no other family's rows or fingerprint move.

def slab_matmul(x: torch.Tensor | None, w, *, xh: torch.Tensor | None = None, M: int | None = None,
                triton_path: bool = True) -> torch.Tensor:
    """`tools.exl3_linear.exl3_matmul` for a call on the slab path (more than EXL3_CHUNK_UNTIL rows): the
    rotated input (made here unless `xh` is given), then per slab the Triton product + rot_out
    (`triton_path`), or the library one."""
    from tools import exl3_linear as X
    if xh is None:
        M = x.shape[0]
        if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
            x = x.contiguous()
        xh = X.rotate_in(x, w)
    if not triton_path:
        return X.exl3_matmul(None, w, xh=xh, M=M, dev=xh.device)
    out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=xh.device)
    segs = X._slabs(w) or [(g, n0, n1, None) for g, n0, n1 in X.segments(w, X.SLAB_N)]
    for g, n0, n1, wi in segs:
        slab_rot(xh[g], wi if wi is not None else w.inner(n0, n1), w.svh[n0:n1], out[:, n0:n1])
    return out


def mlp_slab(x: torch.Tensor, gu, wd, *, xh: torch.Tensor | None = None,
             triton_path: bool = True) -> torch.Tensor | None:
    """`tools.exl3_linear.exl3_mlp_slab` with the rotated input given (`xh`, from `add_norm_rot`) and,
    with `triton_path`, gate|up through `slab_rot` into bf16, exl3_rot_in_silu (the bridge's bits) and the
    down projection through `slab_matmul`. None when the shapes do not pair (the caller's usual path)."""
    from tools import exl3_linear as X
    F = wd.K
    if not (gu.sizes and len(gu.sizes) == 2 and gu.sizes[0] == gu.sizes[1] == F
            and wd.suh.shape[0] == 1 and x.shape[0] > X.CHUNK_UNTIL):
        return None
    if X._mm_out_fp32() is False:
        return None
    mod = X._module()
    M = x.shape[0]
    if xh is None:
        xh = X.rotate_in(x, gu)
    segs = X._slabs(gu) or [(g, n0, n1, None) for g, n0, n1 in X.segments(gu, X.SLAB_N)]
    xd = torch.empty(1, M, F, dtype=torch.float16, device=x.device)
    if triton_path:
        y = torch.empty(M, 2 * F, dtype=torch.bfloat16, device=x.device)
        for g, n0, n1, wi in segs:
            slab_rot(xh[g], wi if wi is not None else gu.inner(n0, n1), gu.svh[n0:n1], y[:, n0:n1])
        mod.exl3_rot_in_silu(y, wd.suh, xd)
        del y
        return slab_matmul(None, wd, xh=xd, M=M)
    gs = [t for t in segs if t[0] == 0]
    us = [t for t in segs if t[0] == 1]
    if len(gs) != len(us):
        return None
    for (_, a0, a1, wg), (_, b0, b1, wu) in zip(gs, us):
        if b0 - F != a0 or b1 - a1 != b0 - a0:
            return None
        yg = X._mm_fp32(xh[0], wg if wg is not None else gu.inner(a0, a1))
        yu = X._mm_fp32(xh[1], wu if wu is not None else gu.inner(b0, b1))
        mod.exl3_mlp_bridge(yg, yu, gu.svh[a0:a1], gu.svh[b0:b1], wd.suh[0, a0:a1].contiguous(),
                            xd[0, :, a0:a1])
        del yg, yu
    return X.exl3_matmul(None, wd, xh=xd, M=M, dev=x.device)


def add_rms_norm_out(res: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, out: tuple):
    """`tools.norm_kernels.add_rms_norm` into the given contiguous (h, y) (h may be `res`: a row is read
    before it is written): its kernel, its launch."""
    from tools import norm_kernels as N
    n = x.shape[-1]
    r = res.reshape(-1, n)
    a = x.reshape(-1, n).contiguous()
    h, y = (t.view(-1, n) for t in out)
    N._add_rms_norm[(r.shape[0],)](r, a, weight, h, y, n, EPS=eps, BLOCK=N._block(n), PDL=N._pdl(),
                                   num_warps=8 if n >= 4096 else 4)
    return out


def rms_norm_gated_out(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float,
                       out: torch.Tensor) -> torch.Tensor:
    """`tools.norm_kernels.rms_norm_gated` into `out`: its kernel, its launch."""
    from tools import norm_kernels as N
    n = x.shape[-1]
    flat = x.reshape(-1, n).contiguous()
    g = gate.reshape(-1, n).contiguous()
    y = out.view(-1, n)
    N._rms_norm_gated[(flat.shape[0],)](flat, g, weight, y, n, EPS=eps, BLOCK=N._block(n), PDL=N._pdl(),
                                        num_warps=8 if n >= 4096 else 4)
    return out
