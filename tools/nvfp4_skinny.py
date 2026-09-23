"""The W4A16 product for 1..32 rows as a CUDA kernel that feeds the tensor cores from registers.

`tools/nvfp4_linear_v2.py` reads the verify's 13.7 GB of NVFP4 projections at about 169 GB/s in
the engine and 194 GB/s cold, against a practical ceiling of about 235 GB/s on this board. Triton
decodes each weight tile into a blocked register layout and then has to move it into the tensor
core's operand layout, which it does through shared memory, for every 128-wide K step of every
tile. This kernel never does that, because of one observation about the stored format:

    `cvt.rn.f16x2.e2m1x2` turns ONE packed byte into ONE f16x2 register holding (k, k+1), the low
    half being the even element -- which is exactly how an `mma.m16n8k16` operand register holds
    two consecutive K elements.

So a thread can load 16 contiguous bytes of a weight row with one vector load and decode each byte
straight into an operand register. The only thing that has to agree is which logical K each of the
instruction's sixteen K slots stands for, and a dot product does not care about the order of its
terms, so the slots are assigned to match the bytes a thread already holds:

    lane (g, t) of a warp holds bytes 16t .. 16t+15 of a 64-byte row chunk (128 logical K).
    In the chunk's j-th k16 instruction (j = 0..7) its B registers are bytes 16t+2j and 16t+2j+1,
    i.e. logical K 32t+4j .. 32t+4j+3; its A registers are the ACTIVATION at the same four K,
    which are four consecutive bf16 of the activation row -- also a contiguous load.

The weights are used as the B operand (8 weight rows per instruction) and the activations as A
(16 rows per instruction), so a verify block of up to 16 rows is one instruction per weight tile
and 17..32 rows are two. The stored layout does not change: the prefill paths, the v1/v2 kernels
and the dequantiser read the same bytes they always have, and nothing is duplicated in memory.

Scales. Group j of a row covers logical K 16j..16j+15, eight bytes. Lane t's sixteen bytes are
groups 2t and 2t+1 of the chunk, instructions 0..3 use the first and 4..7 the second, and both
scales arrive in one 16-bit load, decoded by `cvt.rn.f16x2.e4m3x2`. The scale multiplies the
decoded weight in fp16 before the product, exactly as v2 does (exact: e2m1 x e4m3 needs six
significand bits and fp16 has eleven), and the per-tensor scale multiplies the fp32 accumulator.

Determinism. The reduction order over K is a function of the weight's shape only -- the tile table
below is keyed by (N, K) -- never of the row count, so a row's result is the same bits whether it
is verified alone or in a block of sixteen, which is what the losslessness gate compares. K is split
across the warps of a CTA and summed in warp order through shared memory; there are no atomics.

Built on first use with `torch.utils.cpp_extension.load_inline` for sm_121a (the e2m1 converter is
an arch-specific instruction), cached under ~/.cache/torch_extensions by source hash.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QWEN38_NVFP4_SKINNY=1 routes every NVFP4 projection of 1..32 rows here instead of to v2.
SKINNY = os.environ.get("QWEN38_NVFP4_SKINNY", "0") == "1"
SKINNY_MAX = 32

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint4 ld_w(const uint8_t* p) {
    // weights are read once per step: keep them out of L1 so the activation stays there
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ uint32_t ld_s(const uint8_t* p) {
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_x(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

// four packed bytes -> four f16x2 registers, byte i -> register i, low half = low nibble
__device__ __forceinline__ void dec4(uint32_t w, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                     uint32_t& r3) {
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(w));
}

// two e4m3 scales (low byte first) -> f16x2
__device__ __forceinline__ uint32_t dec_s(uint32_t s) {
    uint32_t r;
    unsigned short h = (unsigned short)s;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// bf16x2 -> f16x2, round to nearest (the same conversion as `x.to(float16)` through fp32)
__device__ __forceinline__ uint32_t bf2h(uint32_t u) {
    float lo = __uint_as_float(u << 16), hi = __uint_as_float(u & 0xffff0000u);
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Y[M, N] = (X[M, K] @ W[N, K]^T) * s2, M <= 16 * MT.
// grid.x = ceil(N / (8 NT)); block = 32 WK: WK warps share the CTA's 8 NT weight rows and split
// its 128-wide K steps into contiguous slices, summed in warp order at the end.
template <int NT, int MT, int WK, int PF>
__global__ void __launch_bounds__(32 * WK)
skinny_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
              const uint8_t* __restrict__ S, const float* __restrict__ S2V, float s2,
              __nv_bfloat16* __restrict__ Y, int M, int N, int KQ,
              int ldx, int ldw, int lds, int ldy) {
    extern __shared__ float red[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * (8 * NT);
    const int per = (KQ + WK - 1) / WK;
    const int q0 = warp * per, q1 = min(KQ, q0 + per);

    // this lane's weight rows (clamped: a row past N is read and never stored)
    const uint8_t* wrow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int r = min(n0 + 8 * i + g, N - 1);
        wrow[i] = W + (size_t)r * ldw + 16 * t;
        srow[i] = S + (size_t)r * lds + 2 * t;
    }
    // this lane's activation rows: g and g + 8 of each 16-row tile
    const __nv_bfloat16* xrow[2 * MT];
    bool xon[2 * MT];
#pragma unroll
    for (int m = 0; m < 2 * MT; ++m) {
        int r = (m >> 1) * 16 + (m & 1) * 8 + g;
        xon[m] = r < M;
        xrow[m] = X + (size_t)(xon[m] ? r : 0) * ldx + 32 * t;
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][i][c] = 0.f;

    uint4 wb[1 + PF][NT];
    uint32_t sb[1 + PF][NT];
    uint4 xb[1 + PF][2 * MT][4];

    auto load = [&](int buf, int q) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            wb[buf][i] = ld_w(wrow[i] + (size_t)q * 64);
            sb[buf][i] = ld_s(srow[i] + (size_t)q * 8);
        }
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m)
#pragma unroll
            for (int v = 0; v < 4; ++v)
                xb[buf][m][v] = xon[m] ? ld_x(xrow[m] + (size_t)q * 128 + 8 * v)
                                       : make_uint4(0, 0, 0, 0);
    };

    auto compute = [&](int buf) {
        // activation operands: 16 f16x2 per row, pair p = logical K 32t + 2p, 2p + 1
        uint32_t xa[2 * MT][16];
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                xa[m][4 * v + 0] = bf2h(xb[buf][m][v].x);
                xa[m][4 * v + 1] = bf2h(xb[buf][m][v].y);
                xa[m][4 * v + 2] = bf2h(xb[buf][m][v].z);
                xa[m][4 * v + 3] = bf2h(xb[buf][m][v].w);
            }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t sc = dec_s(sb[buf][i]);
            uint32_t s_lo = __byte_perm(sc, sc, 0x1010), s_hi = __byte_perm(sc, sc, 0x3232);
            uint32_t wr[4] = {wb[buf][i].x, wb[buf][i].y, wb[buf][i].z, wb[buf][i].w};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                uint32_t d0, d1, d2, d3;
                dec4(wr[r], d0, d1, d2, d3);
                const uint32_t s = r < 2 ? s_lo : s_hi;       // bytes 0..7 group 2t, 8..15 2t+1
                d0 = hmul2(d0, s); d1 = hmul2(d1, s); d2 = hmul2(d2, s); d3 = hmul2(d3, s);
                // register r holds bytes 4r..4r+3 = instructions j = 2r (bytes 4r, 4r+1) and 2r+1
#pragma unroll
                for (int a = 0; a < MT; ++a) {
                    const int j0 = 2 * r, j1 = 2 * r + 1;
                    mma(acc[a][i], xa[2 * a][2 * j0], xa[2 * a + 1][2 * j0],
                        xa[2 * a][2 * j0 + 1], xa[2 * a + 1][2 * j0 + 1], d0, d1);
                    mma(acc[a][i], xa[2 * a][2 * j1], xa[2 * a + 1][2 * j1],
                        xa[2 * a][2 * j1 + 1], xa[2 * a + 1][2 * j1 + 1], d2, d3);
                }
            }
        }
    };

    if (PF) {
        // two steps an iteration so every buffer index is a constant: a register array indexed
        // at run time is a local-memory array. Steps are still consumed in order, so PF = 0 and
        // PF = 1 sum K in the same order and give the same bits.
        int q = q0;
        if (q < q1) load(0, q);
        for (; q + 1 < q1; q += 2) {
            load(1, q + 1);
            compute(0);
            if (q + 2 < q1) load(0, q + 2);
            compute(1);
        }
        if (q < q1) compute(0);
    } else {
        for (int q = q0; q < q1; ++q) {
            load(0, q);
            compute(0);
        }
    }

    constexpr int R = MT * NT * 4;
    if (WK > 1) {
        if (warp > 0) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        red[((warp - 1) * R + (a * NT + i) * 4 + c) * 32 + lane] = acc[a][i][c];
        }
        __syncthreads();
        if (warp > 0) return;
        for (int w = 1; w < WK; ++w)
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        acc[a][i][c] += red[((w - 1) * R + (a * NT + i) * 4 + c) * 32 + lane];
    }

    // D[m][n]: c0, c1 = row g, columns 2t, 2t+1; c2, c3 = row g + 8
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int n = n0 + 8 * i + 2 * t;
            if (n >= N) continue;
            const float sa = S2V ? S2V[n] : s2, sb2 = S2V ? S2V[min(n + 1, N - 1)] : s2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = a * 16 + h * 8 + g;
                if (m >= M) continue;
                __nv_bfloat16* y = Y + (size_t)m * ldy + n;
                const __nv_bfloat16 v0 = __float2bfloat16_rn(acc[a][i][2 * h] * sa);
                if (n + 1 < N) {
                    const __nv_bfloat16 v1 = __float2bfloat16_rn(acc[a][i][2 * h + 1] * sb2);
                    __nv_bfloat162 p; p.x = v0; p.y = v1;
                    *reinterpret_cast<__nv_bfloat162*>(y) = p;
                } else {
                    *y = v0;
                }
            }
        }
}

template <int NT, int MT, int WK, int PF>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
            const float* s2v, float s2, torch::Tensor& y) {
    const int M = x.size(0), N = w.size(0), K = w.size(1) * 2;
    const int smem = WK > 1 ? (WK - 1) * MT * NT * 4 * 32 * 4 : 0;
    auto kern = skinny_kernel<NT, MT, WK, PF>;
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    attr = true;
    dim3 grid((N + 8 * NT - 1) / (8 * NT));
    kern<<<grid, 32 * WK, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(s.data_ptr()), s2v, s2,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K / 128,
        x.stride(0), w.stride(0), s.stride(0), y.stride(0));
}

#define WKS(NT, MT, PF)                                                                   \
    switch (wk) {                                                                         \
        case 1: launch<NT, MT, 1, PF>(x, w, s, p, s2, y); return;                        \
        case 2: launch<NT, MT, 2, PF>(x, w, s, p, s2, y); return;                        \
        case 4: launch<NT, MT, 4, PF>(x, w, s, p, s2, y); return;                        \
        case 8: launch<NT, MT, 8, PF>(x, w, s, p, s2, y); return;                        \
    }

}  // namespace

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, double s2,
            torch::Tensor y, int64_t nt, int64_t wk, int64_t pf) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x");
    TORCH_CHECK(w.scalar_type() == at::kByte && w.stride(1) == 1 && s.stride(1) == 1, "w/s");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) * 2, "K");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "rows");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 16 == 0, "alignment");
    const float* p = s2v.numel() ? s2v.data_ptr<float>() : nullptr;
    const int mt = x.size(0) > 16 ? 2 : 1;
    // Seventeen rows and up take eight weight rows a warp: sixteen would not fit in registers
    // twice over. The K split (`wk`) is what fixes a row's summation order, and it is unchanged.
    if (mt == 2 && nt == 16) nt = 8;
    if (mt == 1 && pf == 1) {
        if (nt == 4) { WKS(4, 1, 1) } else if (nt == 8) { WKS(8, 1, 1) } else { WKS(16, 1, 1) }
    } else if (mt == 1) {
        if (nt == 4) { WKS(4, 1, 0) } else if (nt == 8) { WKS(8, 1, 0) } else { WKS(16, 1, 0) }
    } else if (pf == 1) {
        if (nt == 4) { WKS(4, 2, 1) } else { WKS(8, 2, 1) }
    } else {
        if (nt == 4) { WKS(4, 2, 0) } else { WKS(8, 2, 0) }
    }
    TORCH_CHECK(false, "no instance for nt=", nt, " wk=", wk, " pf=", pf, " mt=", mt);
}
"""

_CPP = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, "
        "double s2, torch::Tensor y, int64_t nt, int64_t wk, int64_t pf);")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        _MOD = load_inline(name="qwen38_nvfp4_skinny", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                           functions=["skinny"],
                           # the e2m1 converter is an arch-specific instruction: sm_121a and
                           # nothing else (an explicit arch flag also stops torch adding its own)
                           extra_cuda_cflags=["-O3", "-lineinfo",
                                              "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _MOD


# (N, K) -> tile. Keyed by shape only, never by row count: the K split is the reduction order.
# Filled from the cold sweep and the in-engine A/B (notes/SPEED-LEDGER.md, 2026-09-23 kernels).
_CONFIG: dict[tuple[int, int], dict] = {}
_FALLBACK = {"nt": 8, "wk": 4, "pf": 1}
# A measured table from a file, for the in-engine A/B before a table is written into this one:
# {"17408x5120": {"nt": 8, "wk": 4, "pf": 1}, ...}
if os.environ.get("QWEN38_SKINNY_TILES"):
    import json as _json
    for _k, _v in _json.load(open(os.environ["QWEN38_SKINNY_TILES"])).items():
        _n, _kk = (int(v) for v in _k.split("x"))
        _CONFIG[(_n, _kk)] = {"nt": int(_v["nt"]), "wk": int(_v["wk"]), "pf": int(_v["pf"])}


def pick(N: int, K: int) -> dict:
    return _CONFIG.get((N, K), _FALLBACK)


def set_config(N: int, K: int, cfg: dict) -> None:
    _CONFIG[(N, K)] = dict(cfg)


_EMPTY: dict = {}


def nvfp4_matmul_skinny(x: torch.Tensor, w, *, out: torch.Tensor | None = None,
                        nt: int | None = None, wk: int | None = None,
                        pf: int | None = None) -> torch.Tensor:
    """y[M, N] = x[M, K] @ W[N, K]^T for M <= 32; W an NVFP4Block or an NVFP4Group."""
    M = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and 1 <= M <= SKINNY_MAX, x.shape
    x = x.contiguous()
    if x.data_ptr() % 16:
        # the activation is read with 16-byte loads; a view at an odd offset is copied once
        x = x.clone()
    cfg = pick(w.N, w.K)
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    s2v = getattr(w, "s2v", None)
    if s2v is None:
        s2v = _EMPTY.get(x.device)
        if s2v is None:
            s2v = _EMPTY[x.device] = torch.empty(0, dtype=torch.float32, device=x.device)
    _module().skinny(x, w.w, w.s.view(torch.uint8), s2v, float(w.s2), out,
                     cfg["nt"] if nt is None else nt, cfg["wk"] if wk is None else wk,
                     cfg["pf"] if pf is None else pf)
    return out


def use_skinny(M: int) -> bool:
    return SKINNY and 1 <= M <= SKINNY_MAX


# ------------------------------------------------------------------ gate and bench
def _exact(w) -> torch.Tensor:
    """The weight in fp32 with no rounding at all: e2m1 x e4m3 x s2, the value every kernel means."""
    from tools.nvfp4_linear import FP4_GRID, GROUP
    grid = FP4_GRID.to(w.w.device)
    lo, hi = (w.w & 0x0F).long(), (w.w >> 4).long()
    vals = torch.empty(w.N, w.K, dtype=torch.float32, device=w.w.device)
    vals[:, 0::2] = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
    vals[:, 1::2] = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
    return vals * w.s.float().repeat_interleave(GROUP, 1) * w.s2


def check(shapes=((17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
                  (12288, 5120), (1024, 5120)),
          rows=(1, 2, 7, 8, 9, 14, 16, 17, 24, 32), seed: int = 0) -> list[str]:
    """Against the exact product and against v2, on random NVFP4 weights.

    The bar is v2's own: `max|skinny - v2|` at or below `max|v2 - exact|` (two fp32 orders of the
    same fp16 products), and row independence -- a row computed alone equals the same row inside a
    block, bit for bit -- at every row count, for every tile in the sweep.
    """
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
    g = torch.Generator(device="cuda").manual_seed(seed)
    out, worst = [], 0.0
    for (N, K) in shapes:
        ref = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float32) * 0.02
        w = quantize_to_nvfp4(ref)
        wd = _exact(w)
        x = torch.randn(max(rows), K, device="cuda", generator=g).to(torch.bfloat16)
        for cfg in ({"nt": 4, "wk": 1, "pf": 0}, {"nt": 8, "wk": 4, "pf": 1},
                    {"nt": 16, "wk": 2, "pf": 0}, {"nt": 8, "wk": 8, "pf": 1}):
            single = None
            for M in rows:
                xm = x[:M]
                y = nvfp4_matmul_skinny(xm, w, **cfg).float()
                y2 = nvfp4_matmul_v2(xm, w).float()
                exact = xm.float().to(torch.float16).float() @ wd.T
                d_s = (y - exact).abs().max().item()
                d_v2 = (y2 - exact).abs().max().item()
                d_sv = (y - y2).abs().max().item()
                ratio = d_sv / max(d_v2, 1e-12)
                worst = max(worst, ratio)
                if single is None:
                    single = y[:1].clone()
                indep = (y[:1] - single).abs().max().item()
                assert indep == 0.0, f"row independence {N}x{K} {cfg} M={M}: {indep}"
                assert d_s <= 2 * d_v2 + 1e-6, f"{N}x{K} {cfg} M={M}: {d_s} vs v2 {d_v2}"
            out.append(f"{N}x{K} {cfg}: last |skinny-exact| {d_s:.3e}  |v2-exact| {d_v2:.3e}  "
                       f"|skinny-v2| {d_sv:.3e}")
        del w, wd, ref
    out.append(f"worst |skinny-v2| / |v2-exact| = {worst:.3f}; row independence exact everywhere")
    return out


def bench_cold(N: int, K: int, gb: float = 2.5, rows=(1, 8, 16, 32), tiles=None,
               reps: int = 5) -> list[str]:
    """A chain of distinct weights bigger than the board's caches, so every byte is a DRAM read.

    Same method as tools/cold_bw.py: the chain is `n` independent copies, each projection timed
    over the whole chain. A ranking, never a budget -- the engine's own verify decides.
    """
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
    one = quantize_to_nvfp4(torch.randn(N, K, device="cuda") * 0.02)
    per = one.nbytes
    n = max(4, int(gb * 1e9 / per))
    ws = []
    for _ in range(n):
        b = type(one).__new__(type(one))
        b.w, b.s, b.s2, b.N, b.K, b._bf16 = one.w.clone(), one.s.clone(), one.s2, one.N, one.K, None
        ws.append(b)
    del one
    tiles = tiles or [{"nt": 8, "wk": 4, "pf": 1}]
    lines = [f"### {N}x{K}  {per / 1e6:.1f} MB x {n} = {n * per / 1e9:.2f} GB"]
    for M in rows:
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        cands = [("v2", lambda b: nvfp4_matmul_v2(x, b))]
        for c in tiles:
            cands.append((f"nt{c['nt']}:wk{c['wk']}:pf{c['pf']}",
                          lambda b, c=c: nvfp4_matmul_skinny(x, b, **c)))
        for name, fn in cands:
            for b in ws[:2]:
                fn(b)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(reps):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for b in ws:
                    fn(b)
                t1.record()
                t1.synchronize()
                best = min(best, t0.elapsed_time(t1))
            lines.append(f"  M={M:3d} {name:>18} {best:8.2f} ms  {n * per / best / 1e6:7.1f} GB/s")
            RESULTS.setdefault(f"{N}x{K}", {}).setdefault(name, {})[M] = n * per / best / 1e6
    return lines


RESULTS: dict = {}


def best_tiles(rows=(8, 16)) -> dict:
    """Per shape, the tile with the best mean GB/s over the verify's row counts."""
    out = {}
    for shape, by_tile in RESULTS.items():
        scored = [(sum(v.get(m, 0.0) for m in rows) / len(rows), name)
                  for name, v in by_tile.items() if name != "v2"]
        if not scored:
            continue
        gbps, name = max(scored)
        nt, wk, pf = (int(p[2:]) for p in name.split(":"))
        out[shape] = {"nt": nt, "wk": wk, "pf": pf, "gbps": round(gbps, 1),
                      "v2_gbps": round(sum(by_tile["v2"].get(m, 0.0) for m in rows) / len(rows), 1)}
    return out


def read_ceiling(gb: float = 2.5, reps: int = 5) -> str:
    """What a plain streaming read reaches here: a float sum over a buffer bigger than the caches."""
    buf = torch.empty(int(gb * 1e9 / 4), dtype=torch.float32, device="cuda").uniform_()
    buf.sum()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        buf.sum()
        t1.record()
        t1.synchronize()
        best = min(best, t0.elapsed_time(t1))
    n = buf.numel() * 4
    del buf
    return f"read ceiling: torch sum over {n / 1e9:.2f} GB in {best:.2f} ms = {n / best / 1e6:.1f} GB/s"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", default="", help="N:K[,N:K...]")
    ap.add_argument("--rows", default="1,8,16")
    ap.add_argument("--tiles", default="nt8:wk4:pf1")
    ap.add_argument("--gb", type=float, default=2.5)
    ap.add_argument("--write-tiles", default="", help="write the best tile per shape here")
    a = ap.parse_args()
    if a.bench:
        print(read_ceiling(a.gb), flush=True)
    if a.check:
        for line in check():
            print(line, flush=True)
    tiles = [dict(zip(("nt", "wk", "pf"), (int(p[2:]) for p in t.split(":"))))
             for t in a.tiles.split(",")]
    for shp in filter(None, a.bench.split(",")):
        N, K = (int(v) for v in shp.split(":"))
        for line in bench_cold(N, K, a.gb, [int(r) for r in a.rows.split(",")], tiles):
            print(line, flush=True)
    if a.write_tiles:
        import json
        best = best_tiles()
        json.dump(best, open(a.write_tiles, "w"), indent=1)
        for shape, c in best.items():
            print(f"best {shape}: nt{c['nt']}:wk{c['wk']}:pf{c['pf']} {c['gbps']} GB/s "
                  f"(v2 {c['v2_gbps']})", flush=True)
