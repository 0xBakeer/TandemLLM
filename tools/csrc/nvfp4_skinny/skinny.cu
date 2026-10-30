
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#ifndef XSTUB
#define XSTUB 0
#endif
#ifndef SSTUB
#define SSTUB 0
#endif
#ifndef SRUN
#define SRUN 0
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint4 ld_w(const uint8_t* p) {
    // weights are read once per step: keep them out of L1 so the activation stays there.
    // LDW_HINT (QWEN38_SKINNY_LDW): 1 marks them first to go from L2 as well (a 15 GB
    // stream through a 24 MiB L2 has nothing to reuse; what should stay is the activations and the
    // recurrent state), 2 asks the L2 for 256-byte lines on the miss, 3 both. Same bytes, same bits.
    uint4 r;
#if LDW_HINT == 1 || LDW_HINT == 3
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
#if LDW_HINT == 1
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#elif LDW_HINT == 2
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#elif LDW_HINT == 3
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#else
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#endif
    return r;
}

__device__ __forceinline__ uint32_t ld_s(const uint8_t* p) {
#if SSTUB
    return 0x3838u;                       // timing only: two e4m3 1.0 scales, no load
#else
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
#endif
}

__device__ __forceinline__ uint4 ld_x(const void* p) {
#if XSTUB
    return make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);  // timing only
#else
    return __ldg(reinterpret_cast<const uint4*>(p));
#endif
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
// PF: 0 no prefetch; 1 the next step's weights, scales and activation rows in registers; 2 the
// weights and scales only (the activation is an L1 hit, and its double buffer is 32 registers a
// row). MINB asks the compiler for registers that fit that many CTAs on an SM: at 166 registers
// and 256 threads one CTA fits, and a grid of 160 CTAs runs as 3.3 waves.
// IL: 0, warp w sums the contiguous steps [w per, (w + 1) per); 1, the steps w, w + WK, w + 2 WK, ..
// so the CTA's warps read neighbouring 64-byte chunks of each row at the same time. Either
// way the order is a function of (K, WK, IL) only, never of M or NT: an 8-row N tile (NT = 1) sums
// a row exactly as a 16-row one does, and a row is the same bits alone or in a block.
// KR (17..32 rows): the products in the order r (weight register) -> i (N tile) -> a (row
// tile) instead of i -> r -> a, with each register's activation (one 16-byte vector a row) loaded right
// before its products. Weight register r of a K step pairs with exactly one activation vector per row:
// a lane holds 2 MT vectors at a time instead of 4 x 2 MT (the wide tile spills at MT 2 holding them
// all). Every acc[a][i] still receives (r0 j0, r0 j1, r1 j0, ...) with the same values: the same bits.
// SPW (47, 2026-09-26): one warp sums SPW consecutive slices of the K split, each from zero and
// kept apart (in shared memory), and warp 0 adds them in slice order, an empty slice adding 0.f -- the
// served reduction, op for op. A CTA then needs ceil(nonempty slices / SPW) warps (7 for K = 5120,
// whose 16 slices are 13 of 3 steps, one of 1 and two empty) and two CTAs fit an SM, so one streams
// while the other reduces. SPW = 1 is the kernel as it was.
template <int NT, int MT, int WK, int PF, int MINB, int IL = 0, int KR = 0, int SPW = 1>
__global__ void __launch_bounds__(32 * WK / SPW, MINB)
skinny_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
              const uint8_t* __restrict__ S, const float* __restrict__ S2V, float s2,
              __nv_bfloat16* __restrict__ Y, int M, int N, int KQ,
              int ldx, int ldw, int lds, int ldy, int pdl) {
    extern __shared__ float red[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * (8 * NT);
    const int per = (KQ + WK - 1) / WK;
    const int qs = IL ? WK : 1;
    const int q0 = IL ? warp : warp * SPW * per, q1 = IL ? KQ : min(KQ, q0 + SPW * per);

    // this lane's weight rows (clamped: a row past N is read and never stored)
    const uint8_t* wrow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int r = min(n0 + 8 * i + g, N - 1);
        wrow[i] = W + (size_t)r * ldw + 16 * t;
#if SRUN
        srow[i] = S + ((size_t)(r >> 4) * KQ * 16 + (r & 15)) * 8 + 2 * t;
#else
        srow[i] = S + (size_t)r * lds + 2 * t;
#endif
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

    constexpr int NB = PF ? 2 : 1;             // weight buffers
    constexpr int XB = PF == 1 ? 2 : 1;        // activation buffers
    uint4 wb[NB][NT];
    uint32_t sb[NB][NT];
    uint4 xb[XB][2 * MT][4];

    auto load_x = [&](int buf, int q) {
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m)
#pragma unroll
            for (int v = 0; v < 4; ++v)
                xb[buf][m][v] = xon[m] ? ld_x(xrow[m] + (size_t)q * 128 + 8 * v)
                                       : make_uint4(0, 0, 0, 0);
    };
    auto load = [&](int buf, int q) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            wb[buf][i] = ld_w(wrow[i] + (size_t)q * 64);
            sb[buf][i] = ld_s(srow[i] + (size_t)q * (SRUN ? 128 : 8));
        }
        if (PF != 2 && !KR) load_x(XB == 2 ? buf : 0, q);
    };

    auto compute = [&](int buf, int q) {
        if (KR) {
            uint32_t s_lo[NT], s_hi[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t sc = dec_s(sb[buf][i]);
                s_lo[i] = __byte_perm(sc, sc, 0x1010);
                s_hi[i] = __byte_perm(sc, sc, 0x3232);
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                uint32_t xr[2 * MT][4];
#pragma unroll
                for (int m = 0; m < 2 * MT; ++m) {
                    const uint4 u = xon[m] ? ld_x(xrow[m] + (size_t)q * 128 + 8 * r)
                                           : make_uint4(0, 0, 0, 0);
                    xr[m][0] = bf2h(u.x); xr[m][1] = bf2h(u.y);
                    xr[m][2] = bf2h(u.z); xr[m][3] = bf2h(u.w);
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const uint32_t w = r == 0 ? wb[buf][i].x : r == 1 ? wb[buf][i].y
                                     : r == 2 ? wb[buf][i].z : wb[buf][i].w;
                    uint32_t d0, d1, d2, d3;
                    dec4(w, d0, d1, d2, d3);
                    const uint32_t s = r < 2 ? s_lo[i] : s_hi[i];
                    d0 = hmul2(d0, s); d1 = hmul2(d1, s); d2 = hmul2(d2, s); d3 = hmul2(d3, s);
#pragma unroll
                    for (int a = 0; a < MT; ++a) {
                        mma(acc[a][i], xr[2 * a][0], xr[2 * a + 1][0], xr[2 * a][1],
                            xr[2 * a + 1][1], d0, d1);
                        mma(acc[a][i], xr[2 * a][2], xr[2 * a + 1][2], xr[2 * a][3],
                            xr[2 * a + 1][3], d2, d3);
                    }
                }
            }
            return;
        }
        if (PF == 2) load_x(0, q);
        const int xbuf = XB == 2 ? buf : 0;
        // activation operands: 16 f16x2 per row, pair p = logical K 32t + 2p, 2p + 1
        uint32_t xa[2 * MT][16];
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                xa[m][4 * v + 0] = bf2h(xb[xbuf][m][v].x);
                xa[m][4 * v + 1] = bf2h(xb[xbuf][m][v].y);
                xa[m][4 * v + 2] = bf2h(xb[xbuf][m][v].z);
                xa[m][4 * v + 3] = bf2h(xb[xbuf][m][v].w);
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

    // SPW > 1: a slice ends after step q -> its partial goes to shared memory and the next starts at 0
    constexpr int R = MT * NT * 4;
    auto slice_done = [&](int q) {
        if constexpr (SPW > 1) {
            if ((q + 1) % per == 0 || q + 1 == q1) {
                const int sl = q / per;
#pragma unroll
                for (int a = 0; a < MT; ++a)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            red[(sl * R + (a * NT + i) * 4 + c) * 32 + lane] = acc[a][i][c];
                            acc[a][i][c] = 0.f;
                        }
            }
        }
    };

    //, programmatic dependent launch: this grid may start while the kernel before it runs.
    // Nothing of the activation may be read until that kernel is done; the weights and scales are
    // not its output, so the first step's weights (PF = 2 loads no activation with them) are in
    // flight before the wait. Without the launch attribute the wait returns at once.
    if (PF != 2 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (PF) {
        // two steps an iteration so every buffer index is a constant: a register array indexed
        // at run time is a local-memory array. Steps are still consumed in order, so PF = 0 and
        // PF = 1 sum K in the same order and give the same bits.
        int q = q0;
        if (q < q1) load(0, q);
        if (PF == 2 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
        for (; q + qs < q1; q += 2 * qs) {
            load(1, q + qs);
            compute(0, q);
            slice_done(q);
            if (q + 2 * qs < q1) load(0, q + 2 * qs);
            compute(1, q + qs);
            slice_done(q + qs);
        }
        if (q < q1) { compute(0, q); slice_done(q); }
    } else {
        for (int q = q0; q < q1; q += qs) {
            load(0, q);
            compute(0, q);
            slice_done(q);
        }
    }

    // the next projection may begin its own weight prologue while this one reduces and stores
    if (pdl) asm volatile("griddepcontrol.launch_dependents;");
    if constexpr (SPW > 1) {
        __syncthreads();
        if (warp > 0) return;
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    acc[a][i][c] = red[((a * NT + i) * 4 + c) * 32 + lane];
        for (int w = 1; w < WK; ++w) {
            const bool on = w * per < KQ;
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        acc[a][i][c] += on ? red[(w * R + (a * NT + i) * 4 + c) * 32 + lane] : 0.f;
        }
    } else if (WK > 1) {
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

template <int NT, int MT, int WK, int PF, int MINB, int IL = 0, int KR = 0, int SPW = 1>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
            const float* s2v, float s2, torch::Tensor& y, int pdl) {
    const int M = x.size(0), N = w.size(0), K = w.size(1) * 2;
    const int smem = SPW > 1 ? WK * MT * NT * 4 * 32 * 4
                             : (WK > 1 ? (WK - 1) * MT * NT * 4 * 32 * 4 : 0);
    // SPW > 1: the warps the non-empty slices need, SPW a warp
    const int kq_ = K / 128, per_ = (kq_ + WK - 1) / WK;
    const int warps = SPW > 1 ? ((kq_ + per_ - 1) / per_ + SPW - 1) / SPW : WK;
    // the K-split partials live in shared memory: 99 KB an SM on this board
    TORCH_CHECK(smem <= 99 * 1024, "skinny tile nt=", NT, " wk=", WK, " mt=", MT, " needs ",
                smem, " bytes of shared memory for its K-split partials; the SM has 101376");
    auto kern = skinny_kernel<NT, MT, WK, PF, MINB, IL, KR, SPW>;
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    attr = true;
    dim3 grid((N + 8 * NT - 1) / (8 * NT));
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const uint8_t* wp = w.data_ptr<uint8_t>();
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(s.data_ptr());
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    const int kq = K / 128, lx = x.stride(0), lw = w.stride(0), ls = s.stride(0), ly = y.stride(0);
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = dim3(32 * warps);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = at::cuda::getCurrentCUDAStream();
        cudaLaunchAttribute la[1];
        la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = la;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kern, xp, wp, sp, s2v, s2, yp, M, N, kq, lx, lw, ls, ly, pdl);
    } else {
        kern<<<grid, 32 * warps, smem, at::cuda::getCurrentCUDAStream()>>>(
            xp, wp, sp, s2v, s2, yp, M, N, kq, lx, lw, ls, ly, pdl);
    }
}

#define WK4(NT, MT, PF, MINB)                                                             \
    switch (wk) {                                                                         \
        case 1: launch<NT, MT, 1, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
        case 2: launch<NT, MT, 2, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
        case 4: launch<NT, MT, 4, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
        case 8: launch<NT, MT, 8, PF, MINB>(x, w, s, p, s2, y, pdl); return;                  \
    }
// sixteen K-split warps (512 threads) only with one CTA an SM in mind
#define WK5(NT, MT, PF)                                                                   \
    if (wk == 16) { launch<NT, MT, 16, PF, 1>(x, w, s, p, s2, y, pdl); return; }             \
    WK4(NT, MT, PF, 1)
// the interleaved K split, for the few tiles the down / out / o sweep asks for
#define IL1(NT, MT, PF)                                                                   \
    if (wk == 8) { launch<NT, MT, 8, PF, 1, 1>(x, w, s, p, s2, y, pdl); return; }            \
    if (wk == 16) { launch<NT, MT, 16, PF, 1, 1>(x, w, s, p, s2, y, pdl); return; }


//, the slice-serial tile (`ser`). The K split above (WK slices of contiguous 128-wide steps,
// each summed from zero, then added in slice order) is a row's summation order; nothing in it says
// that slice w must be summed by warp w. Here every warp walks ALL the slices of its own 8 NT weight
// rows in order, keeping the slice's partial (from zero, the same mma sequence per accumulator as the
// served kernel) and a running total (the first slice's partial, then + each next one, empty slices
// adding 0.f as the served reduction does). Every row is therefore the same bits as the served tile
// of the same WK. What that buys: the W warps of a CTA now read the SAME activation step at the same
// time, so the CTA stages it once in shared memory (cp.async, NB buffers, zero-filled past M) and the
// lanes read their A operands from there -- no activation registers in flight, a 1/(8 NT W)-th of the
// activation traffic per weight row instead of all of it, and the weights double-buffered in registers.
// Row layout in shared memory: a row's 16 pieces of 16 bytes (8 bf16) as [r][t] (piece kk = 4t + r is
// the lane t's register-r vector), rows XSTRIDE bytes apart so a quarter-warp's 16-byte loads hit 32
// distinct banks.
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int NT, int MT, int W, int NB, int MINB, int G = 1>
__global__ void __launch_bounds__(32 * W, MINB)
skinny_ser_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ Wt,
                  const uint8_t* __restrict__ S, const float* __restrict__ S2V, float s2,
                  __nv_bfloat16* __restrict__ Y, int M, int N, int KQ, int WK,
                  int ldx, int ldw, int lds, int ldy) {
    constexpr int ROWS = 16 * MT;
    constexpr int XSTRIDE = 320;
    constexpr int STEPB = ROWS * XSTRIDE;      // one K step's staged activation
    constexpr int CHUNK = G * STEPB;           // G steps a barrier
    static_assert(G == 1 || G % 2 == 0, "G steps a chunk: 1 or even (the weights' buffer is g's parity)");
    extern __shared__ __align__(16) uint8_t xs[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * (8 * NT * W) + warp * (8 * NT);
    const int per = (KQ + WK - 1) / WK;

    const uint8_t* wrow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int r = min(n0 + 8 * i + g, N - 1);
        wrow[i] = Wt + (size_t)r * ldw + 16 * t;
#if SRUN
        srow[i] = S + ((size_t)(r >> 4) * KQ * 16 + (r & 15)) * 8 + 2 * t;
#else
        srow[i] = S + (size_t)r * lds + 2 * t;
#endif
    }
    // this lane's rows inside a staged chunk: g and g + 8 of each 16-row tile
    int xoff[2 * MT];
#pragma unroll
    for (int m = 0; m < 2 * MT; ++m) xoff[m] = ((m >> 1) * 16 + (m & 1) * 8 + g) * XSTRIDE + 16 * t;

    // chunk c = K steps c G .. c G + G - 1 (those below KQ), one commit group a chunk
    auto issue = [&](int c) {
        uint8_t* dst = xs + (c % NB) * CHUNK;
        for (int p = threadIdx.x; p < G * ROWS * 16; p += 32 * W) {
            const int gs = p / (ROWS * 16), rp = p - gs * (ROWS * 16);
            const int q = c * G + gs;
            if (q >= KQ) break;
            const int row = rp >> 4, kk = rp & 15;
            const bool on = row < M;
            const __nv_bfloat16* src = X + (size_t)(on ? row : 0) * ldx + (size_t)q * 128 + 8 * kk;
            const uint32_t d = smem_u32(dst + gs * STEPB + row * XSTRIDE + ((kk & 3) * 4 + (kk >> 2)) * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                         :: "r"(d), "l"(src), "r"(on ? 16 : 0) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };

    float tot[MT][NT][4], part[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) { tot[a][i][c] = 0.f; part[a][i][c] = 0.f; }

    uint4 wb[2][NT];
    uint32_t sb[2][NT];
    auto load = [&](int buf, int q) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            wb[buf][i] = ld_w(wrow[i] + (size_t)q * 64);
            sb[buf][i] = ld_s(srow[i] + (size_t)q * (SRUN ? 128 : 8));
        }
    };
    // the served kernel's per-accumulator order: register r (j0 then j1), for r = 0..3
    auto compute = [&](int buf, const uint8_t* xb) {
        uint32_t s_lo[NT], s_hi[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t sc = dec_s(sb[buf][i]);
            s_lo[i] = __byte_perm(sc, sc, 0x1010);
            s_hi[i] = __byte_perm(sc, sc, 0x3232);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            uint32_t xr[2 * MT][4];
#pragma unroll
            for (int m = 0; m < 2 * MT; ++m) {
                const uint4 u = *reinterpret_cast<const uint4*>(xb + xoff[m] + 64 * r);
                xr[m][0] = bf2h(u.x); xr[m][1] = bf2h(u.y);
                xr[m][2] = bf2h(u.z); xr[m][3] = bf2h(u.w);
            }
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                const uint32_t w = r == 0 ? wb[buf][i].x : r == 1 ? wb[buf][i].y
                                 : r == 2 ? wb[buf][i].z : wb[buf][i].w;
                uint32_t d0, d1, d2, d3;
                dec4(w, d0, d1, d2, d3);
                const uint32_t s = r < 2 ? s_lo[i] : s_hi[i];
                d0 = hmul2(d0, s); d1 = hmul2(d1, s); d2 = hmul2(d2, s); d3 = hmul2(d3, s);
#pragma unroll
                for (int a = 0; a < MT; ++a) {
                    mma(part[a][i], xr[2 * a][0], xr[2 * a + 1][0], xr[2 * a][1],
                        xr[2 * a + 1][1], d0, d1);
                    mma(part[a][i], xr[2 * a][2], xr[2 * a + 1][2], xr[2 * a][3],
                        xr[2 * a + 1][3], d2, d3);
                }
            }
        }
    };
    int w = 0, wend = min(KQ, per);
    auto fold = [&](int q) {
        // after step q: a slice ends here -> its partial into the total, in slice order
        while (w < WK && q + 1 == wend) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        tot[a][i][c] = w == 0 ? part[a][i][c] : tot[a][i][c] + part[a][i][c];
                        part[a][i][c] = 0.f;
                    }
            ++w;
            wend = min(KQ, (w + 1) * per);
            if (w * per >= KQ) break;     // the rest are empty slices, added below
        }
    };
#pragma unroll
    for (int c = 0; c < NB - 1; ++c) issue(c);
    load(0, 0);
    if constexpr (G == 1) {
        auto step = [&](int buf, int q) {
            asm volatile("cp.async.wait_group %0;" :: "n"(NB - 2) : "memory");
            __syncthreads();              // every lane's pieces of step q landed; step q-1 is done
            issue(q + NB - 1);
            compute(buf, xs + (q % NB) * CHUNK);
            fold(q);
        };
        int q = 0;
        for (; q + 1 < KQ; q += 2) {
            load(1, q + 1);
            step(0, q);
            if (q + 2 < KQ) load(0, q + 2);
            step(1, q + 1);
        }
        if (q < KQ) step(0, q);
    } else {
        // one barrier every G steps; the steps inside a chunk run in order as above
        const int nch = (KQ + G - 1) / G;
        for (int c = 0; c < nch; ++c) {
            asm volatile("cp.async.wait_group %0;" :: "n"(NB - 2) : "memory");
            __syncthreads();
            issue(c + NB - 1);
            const uint8_t* xc = xs + (c % NB) * CHUNK;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int q = c * G + g;
                if (q < KQ) {
                    if (q + 1 < KQ) load((g + 1) & 1, q + 1);
                    compute(g & 1, xc + g * STEPB);
                    fold(q);
                }
            }
        }
    }
    // empty slices (KQ not a multiple of WK): the served reduction adds their zero partials too
    for (; w < WK; ++w)
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) tot[a][i][c] = w == 0 ? 0.f : tot[a][i][c] + 0.f;
    asm volatile("cp.async.wait_group 0;" ::: "memory");

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
                const __nv_bfloat16 v0 = __float2bfloat16_rn(tot[a][i][2 * h] * sa);
                if (n + 1 < N) {
                    const __nv_bfloat16 v1 = __float2bfloat16_rn(tot[a][i][2 * h + 1] * sb2);
                    __nv_bfloat162 p; p.x = v0; p.y = v1;
                    *reinterpret_cast<__nv_bfloat162*>(y) = p;
                } else {
                    *y = v0;
                }
            }
        }
}

template <int NT, int MT, int W, int NB, int MINB, int G = 1>
void launch_ser(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
                const float* s2v, float s2, torch::Tensor& y, int wk) {
    const int M = x.size(0), N = w.size(0), K = w.size(1) * 2;
    const int smem = NB * G * 16 * MT * 320;
    auto kern = skinny_ser_kernel<NT, MT, W, NB, MINB, G>;
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    attr = true;
    dim3 grid((N + 8 * NT * W - 1) / (8 * NT * W));
    const int kq = K / 128, lx = x.stride(0), lw = w.stride(0), ls = s.stride(0), ly = y.stride(0);
    kern<<<grid, 32 * W, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(s.data_ptr()), s2v, s2,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, kq, wk, lx, lw, ls, ly);
}

}  // namespace

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, double s2,
            torch::Tensor y, int64_t nt, int64_t wk, int64_t pf, int64_t minb, int64_t il,
            int64_t pdl_, int64_t kr, int64_t spw) {
    const int pdl = (int)pdl_;
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x");
    TORCH_CHECK(w.scalar_type() == at::kByte && w.stride(1) == 1 && s.stride(1) == 1, "w/s");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) * 2, "K");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "rows");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 16 == 0, "alignment");
    const float* p = s2v.numel() ? s2v.data_ptr<float>() : nullptr;
    const int mt = x.size(0) > 16 ? 2 : 1;
    // SPW (slices a warp): the instances the 2026-09-26 sweep asks for
    if (spw > 1) {
        TORCH_CHECK(il == 0 && wk == 16, "spw needs the contiguous 16-slice split");
#define SPWI(NT, MT, PF, MINB, KR, SPW)                                                   \
        if (nt == NT && mt == MT && pf == PF && minb == MINB && kr == KR && spw == SPW) {  \
            launch<NT, MT, 16, PF, MINB, 0, KR, SPW>(x, w, s, p, s2, y, pdl); return; }
        SPWI(2, 1, 2, 2, 0, 2) SPWI(2, 1, 2, 4, 0, 4) SPWI(4, 1, 2, 2, 0, 2) SPWI(1, 1, 2, 2, 0, 2)
        SPWI(2, 1, 2, 1, 0, 2)
        SPWI(4, 2, 0, 2, 1, 2) SPWI(2, 2, 2, 2, 0, 2) SPWI(2, 2, 0, 2, 1, 2) SPWI(4, 2, 0, 1, 1, 2)
#undef SPWI
        TORCH_CHECK(false, "no spw instance for nt=", nt, " mt=", mt, " pf=", pf, " minb=", minb,
                    " kr=", kr, " spw=", spw);
    }
    // Seventeen rows and up take eight weight rows a warp: sixteen would not fit in registers
    // twice over. The K split (`wk`) is what fixes a row's summation order, and it is unchanged.
    if (mt == 2 && nt == 16) nt = 8;
    // the register-sequential order for 17..32 rows (up to 16 rows it is not instantiated)
    if (kr && mt == 2) {
        TORCH_CHECK((pf == 0 || pf == 2) && minb == 1 && il == 0 && (wk == 8 || wk == 16) &&
                    (nt == 2 || nt == 4), "no kr instance for nt=", nt, " wk=", wk, " pf=", pf,
                    " minb=", minb, " il=", il);
#define KR1(NT, PF)                                                                       \
        if (wk == 16) { launch<NT, 2, 16, PF, 1, 0, 1>(x, w, s, p, s2, y, pdl); return; }    \
        launch<NT, 2, 8, PF, 1, 0, 1>(x, w, s, p, s2, y, pdl); return;
        if (nt == 2) { if (pf == 0) { KR1(2, 0) } KR1(2, 2) }
        if (pf == 0) { KR1(4, 0) } KR1(4, 2)
    }
    // every (nt, pf, minb, wk) the sweep asks for; minb = 2 only for up to 16 rows
#define PFS1(NT)                                                                          \
    if (minb == 2) {                                                                      \
        if (pf == 0) { WK4(NT, 1, 0, 2) } else if (pf == 1) { WK4(NT, 1, 1, 2) }          \
        else { WK4(NT, 1, 2, 2) }                                                         \
    } else {                                                                              \
        if (pf == 0) { WK5(NT, 1, 0) } else if (pf == 1) { WK5(NT, 1, 1) }                \
        else { WK5(NT, 1, 2) }                                                            \
    }
#define PFS2(NT)                                                                          \
    if (pf == 0) { WK5(NT, 2, 0) } else if (pf == 1) { WK5(NT, 2, 1) } else { WK5(NT, 2, 2) }
    if (il) {
        if (minb == 1 && (nt == 1 || nt == 2) && (pf == 1 || pf == 2)) {
            if (mt == 1) {
                if (nt == 1) { if (pf == 1) { IL1(1, 1, 1) } else { IL1(1, 1, 2) } }
                else { if (pf == 1) { IL1(2, 1, 1) } else { IL1(2, 1, 2) } }
            } else {
                if (nt == 1) { if (pf == 1) { IL1(1, 2, 1) } else { IL1(1, 2, 2) } }
                else { if (pf == 1) { IL1(2, 2, 1) } else { IL1(2, 2, 2) } }
            }
        }
        TORCH_CHECK(false, "no interleaved instance for nt=", nt, " wk=", wk, " pf=", pf,
                    " minb=", minb, " mt=", mt);
    }
    // an 8-row N tile at 512 threads: two CTAs an SM if the registers allow it
    if (mt == 1 && nt == 1 && minb == 2 && wk == 16) {
        if (pf == 1) { launch<1, 1, 16, 1, 2>(x, w, s, p, s2, y, pdl); return; }
        if (pf == 2) { launch<1, 1, 16, 2, 2>(x, w, s, p, s2, y, pdl); return; }
    }
    if (mt == 1) {
        if (nt == 1) { PFS1(1) } else if (nt == 2) { PFS1(2) } else if (nt == 4) { PFS1(4) }
        else if (nt == 8) { PFS1(8) } else { PFS1(16) }
    } else {
        if (nt == 1) { PFS2(1) } else if (nt == 2) { PFS2(2) } else if (nt == 4) { PFS2(4) }
        else { PFS2(8) }
    }
    TORCH_CHECK(false, "no instance for nt=", nt, " wk=", wk, " pf=", pf, " minb=", minb,
                " mt=", mt);
}

// the slice-serial tile. `wk` is the served K split (the summation order), `w` the warps a CTA
// (each 8 NT weight rows over the whole K), `nb` the staged activation buffers.
void skinny_ser(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor s2v, double s2,
                torch::Tensor y, int64_t nt, int64_t wk, int64_t wn, int64_t nb, int64_t g) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x");
    TORCH_CHECK(w.scalar_type() == at::kByte && w.stride(1) == 1 && s.stride(1) == 1, "w/s");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) * 2, "K");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "rows");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 16 == 0 && x.data_ptr() != nullptr &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "alignment");
    const float* p = s2v.numel() ? s2v.data_ptr<float>() : nullptr;
    const float f = (float)s2;
    const int mt = x.size(0) > 16 ? 2 : 1, k = (int)wk;
#define SER(NT, MT, W, NB, MINB)                                                          \
    if (nt == NT && mt == MT && wn == W && nb == NB && g == 1) {                          \
        launch_ser<NT, MT, W, NB, MINB>(x, w, s, p, f, y, k); return; }
#define SERG(NT, MT, W, NB, MINB, G)                                                      \
    if (nt == NT && mt == MT && wn == W && nb == NB && g == G) {                          \
        launch_ser<NT, MT, W, NB, MINB, G>(x, w, s, p, f, y, k); return; }
    SER(1, 1, 4, 2, 4) SER(1, 1, 8, 2, 2) SER(2, 1, 4, 2, 4) SER(2, 1, 8, 2, 2)
    SER(2, 1, 4, 3, 4) SER(2, 1, 8, 3, 2) SER(1, 1, 16, 2, 1) SER(2, 1, 16, 3, 1)
    SER(1, 2, 4, 2, 4) SER(1, 2, 8, 2, 2) SER(1, 2, 16, 2, 1)
    SER(2, 2, 4, 2, 4) SER(2, 2, 8, 2, 2) SER(2, 2, 16, 2, 1)
    SER(2, 2, 4, 3, 4) SER(2, 2, 8, 3, 2) SER(2, 2, 16, 3, 1)
    SER(4, 2, 4, 2, 2) SER(4, 2, 8, 2, 1) SER(4, 2, 4, 3, 2) SER(4, 2, 8, 3, 1)
    SER(1, 2, 8, 3, 2) SER(1, 2, 16, 3, 1) SER(1, 2, 8, 4, 2)
    // G steps a barrier (MT 1: 5 KB a step; MT 2: 10 KB a step)
    SERG(2, 1, 8, 2, 2, 2) SERG(2, 1, 8, 2, 1, 4) SERG(1, 1, 16, 2, 1, 4) SERG(2, 1, 16, 2, 1, 2)
    SERG(2, 2, 8, 2, 2, 2) SERG(1, 2, 16, 2, 1, 2) SERG(1, 2, 16, 2, 1, 4) SERG(2, 2, 16, 2, 1, 2)
    SERG(4, 2, 4, 2, 2, 2)
#undef SER
#undef SERG
    TORCH_CHECK(false, "no slice-serial instance for nt=", nt, " mt=", mt, " w=", wn, " nb=", nb,
                " g=", g);
}
