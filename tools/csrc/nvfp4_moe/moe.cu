// NVFP4 routed experts (NVIDIA FP4-Local's MoE) for 1..64 rows, W4A16 on the tensor cores with bf16 activations.
//
// Weights as stored, stacked by expert: gate|up [E, 2I, H/2] (two E2M1 codes a byte, low nibble = even column), their
// e4m3 block scales [E, 2I, H/16] (one per 16 values) and fp32 globals [E, 2]; down [E, H, I/2], [E, H, I/16], [E].
//
// A weight byte is decoded straight into a bf16x2 B-operand register: cvt.rn.f16x2.e2m1x2 (exact), times the block
// scale in f16 (exact: E2M1 x E4M3 has at most 6 significant bits and stays inside the f16 normal range), then the f16
// values' fp32 widening keeps them exactly and their top 16 bits are the bf16 (exact for 6 significant bits). The
// activations stay bf16 (W4A16). A lane (g, t) of a warp holds bytes 16t..16t+15 of a 64-byte row chunk (128 logical
// K): in the chunk's j-th k16 instruction its B registers are bytes 2j and 2j+1 (logical K 32t+4j .. 32t+4j+3) and its
// A registers the activation at the same four K, two consecutive bf16 pairs of the row. The global scale multiplies
// the fp32 sum.
//
// Plan (moe_plan): the (row, slot) pairs of routed experts sorted by (expert, pair) and cut into groups of at most 16
// pairs of one expert; a group is one 16-row A tile, its expert's weights read once for it. The shared expert (id E)
// is not here (the caller's MXFP8 path).
//
// Row invariance: a group's rows never meet inside an mma (each output element is its own row's dot product), and
// the K split of a CTA's warps depends on the weight's shape only, the slices summed in warp order. A pair's result is
// the same bits whichever pairs share its group, its call or its launch. No atomics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define PMAX 1024                                         // pairs a plan holds (64 rows x 16 slots)

__device__ __forceinline__ uint4 ldg16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

// a pair of e4m3 block scales (low byte first) -> f16x2 of each, broadcast to both halves
__device__ __forceinline__ void dec_s(uint32_t two, uint32_t& lo, uint32_t& hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)two));
    lo = __byte_perm(r, r, 0x1010);
    hi = __byte_perm(r, r, 0x3232);
}

// four bytes of E2M1 code pairs x their block scale (f16x2) -> four bf16x2 (byte i -> d[i], low half = low nibble)
__device__ __forceinline__ void dec4(uint32_t w, uint32_t s, uint32_t* d) {
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(w));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t h;
        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(h) : "r"(d[i]), "r"(s));
        const float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        d[i] = __byte_perm(__float_as_uint(f.x), __float_as_uint(f.y), 0x7632);
    }
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// ------------------------------------------------------------------------------------------- plan
// one CTA of PMAX threads. ids [P] (flat (row, slot) expert ids; E = the shared expert, skipped). Out: order [P]
// (pairs by (expert, pair)), and per group g < ng: its first sorted position, pair count (<= 16) and expert.
__global__ void __launch_bounds__(PMAX) moe_plan_kernel(const int* __restrict__ ids, int P, int E,
        int* __restrict__ order, int* __restrict__ gstart, int* __restrict__ gcount, int* __restrict__ gexp,
        int* __restrict__ ng) {
    __shared__ int se[PMAX], sc[PMAX];
    const int i = threadIdx.x;
    const int e = i < P ? ids[i] : 0x7fffffff;
    if (i < P) {
        int r = 0;
        for (int j = 0; j < P; ++j) {
            const int f = ids[j];
            r += (f < e) || (f == e && j < i);
        }
        se[r] = e;
        order[r] = i;
    }
    __syncthreads();
    int lead = 0, s = i;
    if (i < P && se[i] < E) {
        while (s > 0 && se[s - 1] == se[i]) --s;
        lead = ((i - s) & 15) == 0;
    }
    sc[i] = lead;
    __syncthreads();
    for (int o = 1; o < PMAX; o <<= 1) {                  // inclusive scan of the leader flags
        const int v = i >= o ? sc[i - o] : 0;
        __syncthreads();
        sc[i] += v;
        __syncthreads();
    }
    if (lead) {
        const int k = sc[i] - 1;
        int end = i + 1;
        while (end < P && end < i + 16 && se[end] == se[i]) ++end;
        gstart[k] = i;
        gcount[k] = end - i;
        gexp[k] = se[i];
    }
    if (i == PMAX - 1) *ng = sc[i];
}

// ------------------------------------------------------------------------------------------- products
// one K step (128 logical K) of a lane: weight bytes wb (16), scale pair sp, the activation rows' 32 values xa/xb
// (rows g and g + 8 of the tile) -> acc
__device__ __forceinline__ void step_mma(float* acc, const uint4& wb, uint32_t sp, const uint4* xa, const uint4* xb) {
    uint32_t s_lo, s_hi;
    dec_s(sp, s_lo, s_hi);
    const uint32_t wv[4] = {wb.x, wb.y, wb.z, wb.w};
    const uint32_t* A0 = reinterpret_cast<const uint32_t*>(xa);       // 16 bf16 pairs of row g
    const uint32_t* A1 = reinterpret_cast<const uint32_t*>(xb);       // of row g + 8
#pragma unroll
    for (int r = 0; r < 4; ++r) {                                     // bytes 4r..4r+3 -> instructions 2r, 2r+1
        uint32_t d[4];
        dec4(wv[r], r < 2 ? s_lo : s_hi, d);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = 2 * r + h;
            mma_bf16(acc, A0[2 * j], A1[2 * j], A0[2 * j + 1], A1[2 * j + 1], d[2 * h], d[2 * h + 1]);
        }
    }
}

__device__ __forceinline__ void load_x(uint4* v, const __nv_bfloat16* row, bool on) {
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = on ? __ldg(reinterpret_cast<const uint4*>(row) + u) : make_uint4(0, 0, 0, 0);
}

// the WK warps' partial sums -> warp 0, in warp order. part: [WK - 1][NA][32] floats. Returns true in warp 0.
template <int WK, int NA>
__device__ __forceinline__ bool reduce_warps(float* acc, float* part, int warp, int lane) {
    if (WK == 1) return true;
    if (warp > 0)
#pragma unroll
        for (int i = 0; i < NA; ++i) part[((warp - 1) * NA + i) * 32 + lane] = acc[i];
    __syncthreads();
    if (warp == 0)
        for (int v = 1; v < WK; ++v)
#pragma unroll
            for (int i = 0; i < NA; ++i) acc[i] += part[((v - 1) * NA + i) * 32 + lane];
    return warp == 0;
}

// gate|up and the activation: A[pair, c] = bf16(bf16(silu(g)) u), g = bf16(gate . x s2g), u = bf16(up . x s2u).
// grid (I / (8 NT), GY): a CTA owns intermediate columns c0..c0+8NT-1 (gate rows c0.., up rows I + c0..), groups y,
// y + GY, ..; a K step loads its 2 NT weight vectors before the activation and the products (more bytes in flight)
template <int WK, int NT, int MT>
__global__ void __launch_bounds__(32 * WK) moe_gu_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int S,
        const int* __restrict__ order, const int* __restrict__ gstart, const int* __restrict__ gcount,
        const int* __restrict__ gexp, const int* __restrict__ ngp, const uint8_t* __restrict__ W,
        const uint8_t* __restrict__ SC, const float* __restrict__ S2, __nv_bfloat16* __restrict__ A, int I, int H) {
    extern __shared__ float part[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int c0 = blockIdx.x * 8 * NT, KQ = H / 128, per = (KQ + WK - 1) / WK;
    const int q0 = warp * per, q1 = min(KQ, q0 + per);
    const int ng = *ngp;
    for (int gi = blockIdx.y; gi < ng; gi += gridDim.y) {
        const int e = gexp[gi], s0 = gstart[gi], cnt = gcount[gi];
        // row tile a: the group's pairs 16a + g and 16a + g + 8
        int pr[2 * MT];
        const __nv_bfloat16* xr[2 * MT];
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m) {
            const int i = 16 * (m >> 1) + 8 * (m & 1) + g;
            pr[m] = i < cnt ? order[s0 + i] : -1;
            xr[m] = X + (size_t)(pr[m] >= 0 ? pr[m] / S : 0) * ldx + 32 * t;
        }
        // tile n: gate rows c0 + 8n + g (2n), up rows I + c0 + 8n + g (2n + 1)
        const uint8_t* wr[2 * NT];
        const uint8_t* sr[2 * NT];
#pragma unroll
        for (int n = 0; n < 2 * NT; ++n) {
            const size_t r = (size_t)e * 2 * I + (n & 1) * I + c0 + 8 * (n >> 1) + g;
            wr[n] = W + r * (H / 2) + 16 * t;
            sr[n] = SC + r * (H / 16) + 2 * t;
        }
        constexpr int NA = MT * 2 * NT * 4;                  // acc[a][n] at (a 2NT + n) 4
        float acc[NA];
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[i] = 0.f;
        for (int q = q0; q < q1; ++q) {
            uint4 b[2 * NT];
            uint32_t sp[2 * NT];
#pragma unroll
            for (int n = 0; n < 2 * NT; ++n) {
                b[n] = ldg16(wr[n] + (size_t)q * 64);
                sp[n] = *reinterpret_cast<const uint16_t*>(sr[n] + q * 8);
            }
#pragma unroll
            for (int a = 0; a < MT; ++a) {
                if (16 * a >= cnt) break;                     // a tile past the group's rows (warp-uniform)
                uint4 xa[4], xb[4];
                load_x(xa, xr[2 * a] + (size_t)q * 128, pr[2 * a] >= 0);
                load_x(xb, xr[2 * a + 1] + (size_t)q * 128, pr[2 * a + 1] >= 0);
#pragma unroll
                for (int n = 0; n < 2 * NT; ++n) step_mma(acc + (a * 2 * NT + n) * 4, b[n], sp[n], xa, xb);
            }
        }
        if (reduce_warps<WK, NA>(acc, part, warp, lane)) {
            const float s2g = S2[2 * e], s2u = S2[2 * e + 1];
            // tile n: gate 2n, up 2n + 1: (row g, col 2t), (row g, col 2t+1), (row g+8, ..)
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int p = pr[2 * a + (k >> 1)];
                        if (p < 0) continue;
                        const float gv = bfr(acc[(a * 2 * NT + 2 * n) * 4 + k] * s2g);
                        const float uv = bfr(acc[(a * 2 * NT + 2 * n + 1) * 4 + k] * s2u);
                        const float av = bfr(bfr(gv / (1.0f + expf(-gv))) * uv);
                        A[(size_t)p * I + c0 + 8 * n + 2 * t + (k & 1)] = __float2bfloat16_rn(av);
                    }
        }
        if (WK > 1) __syncthreads();                       // part is the next group's
    }
}

// down: Y[pair, n] = (A[pair] . down[e][n]) s2d, fp32. grid (H / (8 NT), GY).
template <int WK, int NT, int MT>
__global__ void __launch_bounds__(32 * WK) moe_down_kernel(const __nv_bfloat16* __restrict__ A,
        const int* __restrict__ order, const int* __restrict__ gstart, const int* __restrict__ gcount,
        const int* __restrict__ gexp, const int* __restrict__ ngp, const uint8_t* __restrict__ W,
        const uint8_t* __restrict__ SC, const float* __restrict__ S2, float* __restrict__ Y, int I, int H) {
    extern __shared__ float part[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 8 * NT, KQ = I / 128, per = (KQ + WK - 1) / WK;
    const int q0 = warp * per, q1 = min(KQ, q0 + per);
    const int ng = *ngp;
    for (int gi = blockIdx.y; gi < ng; gi += gridDim.y) {
        const int e = gexp[gi], s0 = gstart[gi], cnt = gcount[gi];
        int pr[2 * MT];
        const __nv_bfloat16* xr[2 * MT];
#pragma unroll
        for (int m = 0; m < 2 * MT; ++m) {
            const int i = 16 * (m >> 1) + 8 * (m & 1) + g;
            pr[m] = i < cnt ? order[s0 + i] : -1;
            xr[m] = A + (size_t)(pr[m] >= 0 ? pr[m] : 0) * I + 32 * t;
        }
        constexpr int NA = MT * NT * 4;
        float acc[NA];
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[i] = 0.f;
        for (int q = q0; q < q1; ++q) {
            uint4 b[NT];
            uint32_t sp[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const size_t r = (size_t)e * H + n0 + 8 * n + g;
                b[n] = ldg16(W + r * (I / 2) + 16 * t + (size_t)q * 64);
                sp[n] = *reinterpret_cast<const uint16_t*>(SC + r * (I / 16) + 2 * t + q * 8);
            }
#pragma unroll
            for (int a = 0; a < MT; ++a) {
                if (16 * a >= cnt) break;
                uint4 xa[4], xb[4];
                load_x(xa, xr[2 * a] + (size_t)q * 128, pr[2 * a] >= 0);
                load_x(xb, xr[2 * a + 1] + (size_t)q * 128, pr[2 * a + 1] >= 0);
#pragma unroll
                for (int n = 0; n < NT; ++n) step_mma(acc + (a * NT + n) * 4, b[n], sp[n], xa, xb);
            }
        }
        if (reduce_warps<WK, NA>(acc, part, warp, lane)) {
            const float s2 = S2[e];
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int p = pr[2 * a + (k >> 1)];
                        if (p >= 0) Y[(size_t)p * H + n0 + 8 * n + 2 * t + (k & 1)] = acc[(a * NT + n) * 4 + k] * s2;
                    }
        }
        if (WK > 1) __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------- host
void nvfp4_moe_plan(torch::Tensor ids, int64_t E, torch::Tensor order, torch::Tensor gstart, torch::Tensor gcount,
                    torch::Tensor gexp, torch::Tensor ng) {
    const int P = ids.numel();
    TORCH_CHECK(P <= PMAX && ids.dtype() == torch::kInt && ids.is_contiguous(), "nvfp4_moe_plan: <= 1024 int32 ids");
    moe_plan_kernel<<<1, PMAX, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int>(), P, (int)E, order.data_ptr<int>(), gstart.data_ptr<int>(), gcount.data_ptr<int>(),
        gexp.data_ptr<int>(), ng.data_ptr<int>());
}

void nvfp4_moe_gu(torch::Tensor x, int64_t S, torch::Tensor order, torch::Tensor gstart, torch::Tensor gcount,
                  torch::Tensor gexp, torch::Tensor ng, torch::Tensor w, torch::Tensor sc, torch::Tensor s2,
                  torch::Tensor a, int64_t gy, int64_t wk, int64_t nt, int64_t mt) {
    const int H = x.size(1), I = a.size(1);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0, "x bf16 rows");
    TORCH_CHECK(w.size(1) == 2 * I && w.size(2) * 2 == H && H % 128 == 0 && I % 8 == 0, "gate|up [E, 2I, H/2]");
    auto st = at::cuda::getCurrentCUDAStream();
#define L(WK, NT, MT)                                                                                               \
    if (wk == WK && nt == NT && mt == MT) {                                                                         \
        TORCH_CHECK(I % (8 * NT) == 0, "I % 8 NT");                                                                 \
        moe_gu_kernel<WK, NT, MT><<<dim3(I / (8 * NT), (int)gy), 32 * WK, (WK - 1) * MT * 2 * NT * 4 * 32 * 4, st>>>( \
            (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (int)S, order.data_ptr<int>(),                    \
            gstart.data_ptr<int>(), gcount.data_ptr<int>(), gexp.data_ptr<int>(), ng.data_ptr<int>(),              \
            w.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), s2.data_ptr<float>(), (__nv_bfloat16*)a.data_ptr(), I, H); \
        return;                                                                                                     \
    }
    L(4, 1, 1) L(2, 2, 1) L(4, 2, 1) L(4, 1, 2) L(4, 1, 4) L(2, 2, 4) L(4, 2, 2) L(2, 1, 4) L(1, 1, 4) L(1, 2, 4)
#undef L
    TORCH_CHECK(false, "nvfp4_moe_gu: (wk, nt, mt) not built");
}

void nvfp4_moe_down(torch::Tensor a, torch::Tensor order, torch::Tensor gstart, torch::Tensor gcount,
                    torch::Tensor gexp, torch::Tensor ng, torch::Tensor w, torch::Tensor sc, torch::Tensor s2,
                    torch::Tensor y, int64_t gy, int64_t wk, int64_t nt, int64_t mt) {
    const int I = a.size(1), H = y.size(1);
    TORCH_CHECK(w.size(1) == H && w.size(2) * 2 == I && I % 128 == 0, "down [E, H, I/2]");
    TORCH_CHECK(y.dtype() == torch::kFloat && y.is_contiguous(), "y fp32 [P, H]");
    auto st = at::cuda::getCurrentCUDAStream();
#define L(WK, NT, MT)                                                                                               \
    if (wk == WK && nt == NT && mt == MT) {                                                                         \
        TORCH_CHECK(H % (8 * NT) == 0, "H % 8 NT");                                                                 \
        moe_down_kernel<WK, NT, MT><<<dim3(H / (8 * NT), (int)gy), 32 * WK, (WK - 1) * MT * NT * 4 * 32 * 4, st>>>( \
            (const __nv_bfloat16*)a.data_ptr(), order.data_ptr<int>(), gstart.data_ptr<int>(), gcount.data_ptr<int>(), \
            gexp.data_ptr<int>(), ng.data_ptr<int>(), w.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(),               \
            s2.data_ptr<float>(), y.data_ptr<float>(), I, H);                                                       \
        return;                                                                                                     \
    }
    L(1, 2, 1) L(1, 1, 1) L(5, 2, 1) L(1, 2, 2) L(1, 2, 4) L(1, 4, 4) L(1, 1, 4) L(5, 2, 4)
#undef L
    TORCH_CHECK(false, "nvfp4_moe_down: (wk, nt, mt) not built");
}
