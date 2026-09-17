"""y = x @ W^T with W left in the checkpoint's stored format.

The checkpoint stores every quantised projection as fp8 e4m3 codes [N, K] plus a bf16 scale table
[N/128, K/128] -- one scale per 128x128 block of the weight -- and the dequantised weight is
`code.to(f32) * scale[n // 128, k // 128]`. Dequantising into memory would double both the resident
footprint and the bytes read per decode step, and the decode step of this model is nothing but
weight reads, so the weights stay as codes and the scale is applied inside the GEMM.

Tiling follows the scale table: BLOCK_N = BLOCK_K = 128 makes every tile step cover exactly one
scale block, so the scale is a single scalar per step, applied to the fp32 accumulator rather than
to the weights. That is both cheaper and closer to the reference than scaling bf16 weights: e4m3 ->
bf16 is exact (four significand bits into eight), the products accumulate in fp32, and one fp32
multiply at the end carries the block scale.

Where the block scale is applied is a real choice, not a detail:

  * `SCALE_W = True` folds it into the weight in bf16 first, which is the checkpoint's own
    definition of the dequantised weight and therefore reproduces a stock bf16 module.
  * `SCALE_W = False` leaves the weight as exact bf16 codes and scales the fp32 accumulator, which
    is arithmetically better -- it drops a bf16 rounding of every weight -- and is what a serving
    stack does with a block-scaled fp8 GEMM.

They differ by about 0.2 % per projection, which is invisible in one layer and is not invisible
after sixty-four of them. The default is the reference-matching one; the other is a measured option.

Small N with a small M leaves the board under-occupied -- `out_proj` is N = 5120, which is 40 tiles
-- so the kernel can split the K loop across programs and reduce in a second pass. SPLIT_K = 1 is
the plain path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BLOCK = 128  # the checkpoint's quantisation block, in both N and K


@triton.jit
def _fp8_gemm_kernel(X, W, S, Y,
                     M, N, K,
                     stride_xm, stride_wn, stride_sn, stride_ym,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SCALE_W: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # BLOCK_N == 128 == the scale block, so the whole n-tile shares one scale row.
    s_row = pid_n
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :])
        s = tl.load(S + s_row * stride_sn + k0 // BLOCK_K)
        if SCALE_W:
            wb = w.to(tl.bfloat16) * s.to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
        else:
            acc += s.to(tl.float32) * tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None])


@triton.jit
def _fp8_gemm_splitk_kernel(X, W, S, Y,
                            M, N, K,
                            stride_xm, stride_wn, stride_sn, stride_yk, stride_ym,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            SPLIT_K: tl.constexpr, SCALE_W: tl.constexpr):
    """Same product, K split across SPLIT_K programs, each writing its own fp32 partial plane."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    s_row = pid_n
    n_steps = K // BLOCK_K
    for step in range(pid_k, n_steps, SPLIT_K):
        kk = step * BLOCK_K + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :])
        s = tl.load(S + s_row * stride_sn + step)
        if SCALE_W:
            wb = w.to(tl.bfloat16) * s.to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
        else:
            acc += s.to(tl.float32) * tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(Y + pid_k * stride_yk + rm[:, None] * stride_ym + rn[None, :], acc, mask=m_mask[:, None])


@triton.jit
def _fp8_dequant_kernel(W, S, Y, N, K, stride_wn, stride_sn, stride_yn,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    w = tl.load(W + rn[:, None] * stride_wn + rk[None, :])
    s = tl.load(S + pid_n * stride_sn + pid_k).to(tl.float32)
    tl.store(Y + rn[:, None] * stride_yn + rk[None, :], (w.to(tl.float32) * s).to(tl.bfloat16))


class FP8Block:
    """A projection weight as the checkpoint stores it: fp8 codes plus a bf16 128x128 scale table."""

    __slots__ = ("w", "s", "N", "K", "_bf16")  # `w` is cleared by tools that expand a weight once

    def __init__(self, codes: torch.Tensor, scale_inv: torch.Tensor):
        assert codes.dtype == torch.float8_e4m3fn and codes.dim() == 2, (codes.dtype, codes.shape)
        self.w = codes
        self.s = scale_inv.to(torch.bfloat16).contiguous()
        self.N, self.K = codes.shape
        assert self.N % BLOCK == 0 and self.K % BLOCK == 0, (self.N, self.K)
        assert tuple(self.s.shape) == (self.N // BLOCK, self.K // BLOCK), (self.s.shape, codes.shape)
        self._bf16 = None

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 2

    def dequant(self) -> torch.Tensor:
        """The bf16 weight, as the reference implementation would hold it."""
        if self.w.device.type != "cuda":
            s = self.s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            return (self.w.to(torch.float32) * s).to(torch.bfloat16)
        y = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.w.device)
        _fp8_dequant_kernel[(self.N // BLOCK, self.K // BLOCK)](
            self.w, self.s, y, self.N, self.K,
            self.w.stride(0), self.s.stride(0), y.stride(0),
            BLOCK_N=BLOCK, BLOCK_K=BLOCK, num_warps=4, num_stages=2)
        return y

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            self._bf16 = self.dequant()
        return self._bf16


SCALE_ON_WEIGHT = os.environ.get("QWEN38_SCALE_ON", "weight") == "weight"


def fp8_matmul(x: torch.Tensor, w: FP8Block, *, block_m: int | None = None,
               split_k: int = 1, num_warps: int = 4, num_stages: int = 3,
               out: torch.Tensor | None = None, scale_on_weight: bool | None = None) -> torch.Tensor:
    """y[M, N] = x[M, K] @ W[N, K]^T, W in the stored format. x is bf16."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == w.K, (x.shape, w.shape)
    M = x.shape[0]
    x = x.contiguous()
    if x.device.type != "cuda":
        return torch.nn.functional.linear(x, w.dequant())
    if block_m is None:
        block_m = 16 if M <= 16 else (64 if M <= 128 else 128)
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    scale_w = SCALE_ON_WEIGHT if scale_on_weight is None else scale_on_weight
    if block_m >= 128:
        num_stages = min(num_stages, 2)  # a 128x128 tile pair per stage is the shared-memory limit
    if split_k > 1:
        parts = torch.empty(split_k, M, w.N, dtype=torch.float32, device=x.device)
        _fp8_gemm_splitk_kernel[(w.N // BLOCK, triton.cdiv(M, block_m), split_k)](
            x, w.w, w.s, parts, M, w.N, w.K,
            x.stride(0), w.w.stride(0), w.s.stride(0), parts.stride(0), parts.stride(1),
            BLOCK_M=block_m, BLOCK_N=BLOCK, BLOCK_K=BLOCK, SPLIT_K=split_k, SCALE_W=scale_w,
            num_warps=num_warps, num_stages=num_stages)
        out.copy_(parts.sum(0).to(torch.bfloat16))
        return out
    _fp8_gemm_kernel[(w.N // BLOCK, triton.cdiv(M, block_m))](
        x, w.w, w.s, out, M, w.N, w.K,
        x.stride(0), w.w.stride(0), w.s.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_N=BLOCK, BLOCK_K=BLOCK, SCALE_W=scale_w,
        num_warps=num_warps, num_stages=num_stages)
    return out
