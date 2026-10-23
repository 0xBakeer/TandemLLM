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
