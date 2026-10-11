"""Flash-Next kernels on the board (ENG-223, ENG-141, ENG-143, ENG-149): CUDA, row-invariant.

Row invariance is the property speculation needs: a row computed alone (the one-token walk) and the
same row inside a verify block of up to 32 rows give the same bits. Every kernel here computes a row
(or a (row, expert) pair) with an instruction sequence that does not depend on the other rows: no
reduction runs across rows, no tile shape depends on M. The `_rows` twins of the convention
(docs/verify-twins.md) are the same entry points called with M > 1.

  gemv          y[M, N] = x[M, K] . W[N, K]^T in fp32, W int8 rows with an fp32 scale (the
                hyper-connection mixers, `pack.Int8Rows`) or fp16 (router, PLE projections, the GDN
                a/b projections). A group of LPR lanes owns one output row; each lane sums its
                strided K slice in K order, the group adds the lanes by a fixed xor tree, then the
                scale. 16-byte weight loads.
  mix_up        the mixer's second half in one launch: a = bf16(silu(bf16(d) / hc)) (the down
                projection d from `gemv`), the up projection of a, and the stream mean
                x[j] = bf16(sum_s bf16(bf16(sigmoid(bf16(up_s . a))) * n[s H + j]) / hc) -- one warp
                a column j, its four lane groups the four streams.
  router        softmax over the bf16-rounded logits of E experts in fp32, the top k with the lower
                index first on ties, weights renormalised in slot order and rounded to bf16 (as the
                reference's `.to(logits.dtype)`), and one more slot for the shared expert: id E,
                weight bf16(sigmoid(bf16(logit E))).
  moe_gu        gate and up of every (row, slot) pair: a CTA owns one pair, one matrix and one
                128-column block. It rotates the pair's input with the expert's own suh into shared
                memory (exactly `exl3_linear.rot_in_kernel`'s arithmetic), runs the trellis GEMV over
                its K range split in eight contiguous warp slices (one row of the m16n8k16 tile is
                real), adds the warps in warp order, and applies the output rotation and svh -> bf16.
  moe_down      the down projection of every pair: the CTA forms bf16(bf16(silu(g)) * u) of its
                pair and rotates it with the expert's suh (`rot_in_silu_kernel`'s arithmetic), the
                same GEMV body, fp32 out.
  moe_combine   out[m] = bf16(sum over the row's slots in ascending expert id of w * y): routed
                experts in id order, then the shared expert (id E), in fp32, rounded once.

The routed and the shared experts are one bank (`MoEBank`): expert E is the shared expert, so the
same kernels read it, at its own K. Only the experts a call selects are read: a pair's CTAs take
their expert's pointers from the bank by the id in the device buffer, nothing else. Pairs are
launched in ascending expert id (`order`), so rows that share an expert read its tiles back to back
(from L2 the second time); the arithmetic of a pair does not depend on where it is launched.

The trellis decode, the Hadamard and the mma helpers are tools/exl3_linear.py's (`_HELPERS`),
which ports the bit extraction and the codebook from exllamav3 (MIT, Copyright (c) 2025
Turboderp; the notice travels with that source). Built on first use for sm_121a.
"""

from __future__ import annotations

import hashlib
import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tools import exl3_linear as X  # noqa: E402

# the helper half of exl3_linear's CUDA source: includes, the licence, the trellis decode, mma,
# cp.async and the 128-point Hadamard; it ends inside the anonymous namespace
_HELPERS = X._CUDA[: X._CUDA.index("constexpr float RSQRT128")]

_CUDA = r"""
constexpr float RSQRT128 = 0.08838834764831845f;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + ::expf(-v)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + ::expf(-v)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// ------------------------------------------------------------------------------------- gemv
// 8 bf16 activations from x (16 bytes) as fp32
__device__ __forceinline__ void load_x8(const __nv_bfloat16* p, float* v) {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    v[0] = bf_lo(r.x); v[1] = bf_hi(r.x); v[2] = bf_lo(r.y); v[3] = bf_hi(r.y);
    v[4] = bf_lo(r.z); v[5] = bf_hi(r.z); v[6] = bf_lo(r.w); v[7] = bf_hi(r.w);
}

// W int8 (VEC 16) or fp16 (VEC 8) rows; lanes l of a group of LPR take K chunks l, l + LPR, ..
template <typename TW> struct WVec;
template <> struct WVec<int8_t> {
    static constexpr int VEC = 16;
    static __device__ __forceinline__ void load(const int8_t* p, float* w) {
        const int4 r = *reinterpret_cast<const int4*>(p);
        const int32_t q[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int b = 0; b < 4; ++b) w[4 * i + b] = (float)(int8_t)((q[i] >> (8 * b)) & 0xff);
    }
};
template <> struct WVec<__half> {
    static constexpr int VEC = 8;
    static __device__ __forceinline__ void load(const __half* p, float* w) {
        const uint4 r = *reinterpret_cast<const uint4*>(p);
        const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
        for (int i = 0; i < 4; ++i) { w[2 * i] = __low2float(h[i]); w[2 * i + 1] = __high2float(h[i]); }
    }
};

// the dot of one weight row with MC activation rows, by a group of LPR lanes; every lane ends with
// the full sums (fixed xor tree)
template <typename TW, int LPR, int MC>
__device__ __forceinline__ void row_dot(const TW* __restrict__ wr, const __nv_bfloat16* __restrict__ X,
                                        int ldx, int m0, int mc, int K, int sl, float* acc) {
    constexpr int VEC = WVec<TW>::VEC;
#pragma unroll
    for (int m = 0; m < MC; ++m) acc[m] = 0.f;
    for (int k = sl * VEC; k < K; k += LPR * VEC) {
        float w[VEC];
        WVec<TW>::load(wr + k, w);
#pragma unroll
        for (int m = 0; m < MC; ++m) {
            if (m < mc) {
                float x[VEC];
                load_x8(X + (size_t)(m0 + m) * ldx + k, x);
                if (VEC == 16) load_x8(X + (size_t)(m0 + m) * ldx + k + 8, x + 8);
                float s = acc[m];
#pragma unroll
                for (int e = 0; e < VEC; ++e) s = fmaf(x[e], w[e], s);
                acc[m] = s;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MC; ++m)
#pragma unroll
        for (int o = LPR / 2; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
}

template <typename TW, int LPR, int MC>
__global__ void __launch_bounds__(256) gemv_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const TW* __restrict__ W, const float* __restrict__ S, float* __restrict__ Y, int ldy,
        int M, int N, int K) {
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / LPR, sl = lane % LPR;
    const int n = (blockIdx.x * 8 + warp) * RPW + sub;
    const bool live = n < N;
    const TW* wr = W + (size_t)(live ? n : 0) * K;
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[MC];
        row_dot<TW, LPR, MC>(wr, X, ldx, m0, mc, live ? K : 0, sl, acc);
        if (live && sl == 0) {
            const float sc = S ? S[n] : 1.0f;
#pragma unroll
            for (int m = 0; m < MC; ++m) if (m < mc) Y[(size_t)(m0 + m) * ldy + n] = acc[m] * sc;
        }
    }
}

// ----------------------------------------------------------------------------------- mix_up
// A[M, R] = bf16(silu(bf16(D) / HC)) staged in shared memory (bf16), then per column j (one warp,
// four groups of 8 lanes = the four streams): u_s = bf16(up row (s H + j) . a), x = mean of
// bf16(bf16(sigmoid(u_s)) * n_s).
template <typename TW, int MC>
__global__ void __launch_bounds__(256) mix_up_kernel(const float* __restrict__ D, int ldd,
        const __nv_bfloat16* __restrict__ Nn, int ldn, const TW* __restrict__ W, const float* __restrict__ S,
        __nv_bfloat16* __restrict__ Xo, int ldxo, int M, int H, int R, float inv_hc) {
    extern __shared__ __nv_bfloat16 a_s[];                 // [M, R]
    for (int e = threadIdx.x; e < M * R; e += blockDim.x) {
        const int m = e / R, r = e % R;
        a_s[m * R + r] = __float2bfloat16_rn(silu(bfr(bfr(D[(size_t)m * ldd + r]) * inv_hc)));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int s = lane >> 3, sl = lane & 7;
    const int j = blockIdx.x * 8 + warp;
    if (j >= H) return;
    const int n = s * H + j;
    const TW* wr = W + (size_t)n * R;
    const float sc = S ? S[n] : 1.0f;
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[MC];
        row_dot<TW, 8, MC>(wr, a_s, R, m0, mc, R, sl, acc);
#pragma unroll
        for (int m = 0; m < MC; ++m) {
            float p = 0.f;
            if (m < mc) {
                const float u = bfr(sigm(bfr(acc[m] * sc)));
                p = bfr(u * __bfloat162float(Nn[(size_t)(m0 + m) * ldn + n]));
            }
            // the four streams' products, summed in stream order by lane 0
            const float p1 = __shfl_sync(0xffffffffu, p, 8), p2 = __shfl_sync(0xffffffffu, p, 16),
                        p3 = __shfl_sync(0xffffffffu, p, 24);
            if (lane == 0 && m < mc)
                Xo[(size_t)(m0 + m) * ldxo + j] = __float2bfloat16_rn((((p + p1) + p2) + p3) * inv_hc);
        }
    }
}

// ----------------------------------------------------------------------------------- router
// one warp a row; L [M, ldl] fp32 logits (E routed + 1 shared gate), out ids/wts [M, k + 1]
template <int PER>
__global__ void __launch_bounds__(32) router_kernel(const float* __restrict__ L, int ldl, int E, int k,
        int* __restrict__ IDS, float* __restrict__ WTS) {
    const int m = blockIdx.x, lane = threadIdx.x;
    const float* l = L + (size_t)m * ldl;
    float v[PER];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        v[i] = e < E ? bfr(l[e]) : -INFINITY;
        mx = fmaxf(mx, v[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) { v[i] = (lane + 32 * i < E) ? ::expf(v[i] - mx) : 0.f; sum += v[i]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
#pragma unroll
    for (int i = 0; i < PER; ++i) v[i] = (lane + 32 * i < E) ? v[i] / sum : -1.f;
    float tot = 0.f, wsel = 0.f;
    int idsel = 0;
    for (int s = 0; s < k; ++s) {
        float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int e = lane + 32 * i;
            if (v[i] > bv || (v[i] == bv && e < bi)) { bv = v[i]; bi = e; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
#pragma unroll
        for (int i = 0; i < PER; ++i) if (lane + 32 * i == bi) v[i] = -2.f;
        tot += bv;
        if (lane == s) { wsel = bv; idsel = bi; }
    }
    if (lane < k) {
        IDS[(size_t)m * (k + 1) + lane] = idsel;
        WTS[(size_t)m * (k + 1) + lane] = bfr(wsel / tot);
    }
    if (lane == k) {
        IDS[(size_t)m * (k + 1) + k] = E;
        WTS[(size_t)m * (k + 1) + k] = bfr(sigm(bfr(l[E])));
    }
}

// --------------------------------------------------------------------------------- MoE GEMV
// One row of one expert matrix over one 128-column block: XH (fp16 [Kin], shared memory) @ Wi, the
// K tiles split over the CTA's 8 warps in contiguous slices, the warps added in warp order into
// RED[128]. Trellis tiles of the 8 columns' tiles stream through cp.async stages per warp.
template <int KB, int ST>
__device__ __forceinline__ void moe_row_gemm(const __half* __restrict__ xh, const uint32_t* __restrict__ T,
        int ldt, int cb, int Kin, uint32_t* __restrict__ stage_all, float* __restrict__ red) {
    constexpr int TW = 8 * KB;
    constexpr int NTW = 8;
    constexpr int SW = NTW * TW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    uint32_t* stage = stage_all + (size_t)warp * ST * SW;
    const int KT = Kin / 16;
    const int per = (KT + 7) / 8;
    const int k0 = min(KT, warp * per), k1 = min(KT, k0 + per);
    const int nt0 = cb * 8;
    float acc[NTW][2][4];
#pragma unroll
    for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[j][h][c] = 0.f;
    auto issue = [&](int kt, int buf) {
        if (kt < k1) {
            const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
            uint4* dst = reinterpret_cast<uint4*>(stage + buf * SW);
            for (int c = lane; c < SW / 4; c += 32) cp_async16(dst + c, src + c);
        }
        cp_commit();
    };
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) issue(k0 + s, s);
    for (int kt = k0; kt < k1; ++kt) {
        const int i = kt - k0;
        issue(kt + ST - 1, (i + ST - 1) % ST);
        cp_wait<ST - 1>();
        __syncwarp();
        const uint32_t* buf = stage + (i % ST) * SW;
        // A: row 0 only (lanes g == 0); rows 1..15 are zeros
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(xh + kt * 16);
        const uint32_t a0 = g == 0 ? xw[t] : 0u, a2 = g == 0 ? xw[t + 4] : 0u;
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
            FragB f0, f1;
            dq_tile<KB>(buf + j * TW, lane * 8, f0, f1);
            mma(acc[j][0], a0, 0u, a2, 0u, f0.r[0], f0.r[1]);
            mma(acc[j][1], a0, 0u, a2, 0u, f1.r[0], f1.r[1]);
        }
        __syncwarp();
    }
    cp_wait<0>();
    for (int step = 0; step < 8; ++step) {
        if (warp == step && g == 0) {
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = j * 16 + h * 8 + 2 * t;
                    if (step == 0) { red[col] = acc[j][h][0]; red[col + 1] = acc[j][h][1]; }
                    else { red[col] += acc[j][h][0]; red[col + 1] += acc[j][h][1]; }
                }
        }
        __syncthreads();
    }
}

template <int ST>
__device__ __forceinline__ void moe_row_gemm_k(int kb, const __half* xh, const uint32_t* T, int ldt,
        int cb, int Kin, uint32_t* stage, float* red) {
    switch (kb) {
        case 2: moe_row_gemm<2, ST>(xh, T, ldt, cb, Kin, stage, red); break;
        case 3: moe_row_gemm<3, ST>(xh, T, ldt, cb, Kin, stage, red); break;
        case 4: moe_row_gemm<4, ST>(xh, T, ldt, cb, Kin, stage, red); break;
        case 5: moe_row_gemm<5, ST>(xh, T, ldt, cb, Kin, stage, red); break;
        case 6: moe_row_gemm<6, ST>(xh, T, ldt, cb, Kin, stage, red); break;
        case 8: moe_row_gemm<8, ST>(xh, T, ldt, cb, Kin, stage, red); break;
        default: break;
    }
}

struct MoEArgs {
    const uint32_t* const* tr;    // [E + 1, 3] trellis pointers (gate, up, down)
    const int* kb;                 // [E + 1, 3] bits
    const __half* const* su;       // [E + 1, 3] suh pointers
    const __half* const* sv;       // [E + 1, 3] svh pointers
};

// grid (2 * I/128, P); blockIdx.y -> pair ORDER[y]; Y [P, 2, I] bf16
template <int ST>
__global__ void __launch_bounds__(256) moe_gu_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const int* __restrict__ IDS, const int* __restrict__ ORDER, int S, MoEArgs a,
        __nv_bfloat16* __restrict__ Y, int H, int I) {
    extern __shared__ uint32_t smem[];
    __half* xh = reinterpret_cast<__half*>(smem);                          // [H]
    float* red = reinterpret_cast<float*>(smem + H / 2);                   // [128]
    uint32_t* stage = smem + H / 2 + 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int p = ORDER[blockIdx.y];
    const int m = p / S;
    const int e = IDS[p];
    const int mat = blockIdx.x & 1, cb = blockIdx.x >> 1;
    const __half* suh = a.su[e * 3 + mat];
    // the pair's rotated input: rot_in_kernel's arithmetic, one warp a 128 block
    for (int blk = warp; blk * 128 < H; blk += 8) {
        const int k = blk * 128 + 4 * lane;
        const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)m * ldx + k);
        const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
        float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
        float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
        v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
        wht128(v0, v1, v2, v3, lane);
        __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
        __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xh + k) = o;
    }
    __syncthreads();
    moe_row_gemm_k<ST>(a.kb[e * 3 + mat], xh, a.tr[e * 3 + mat], I / 16, cb, H, stage, red);
    if (warp == 0) {
        float v0 = red[4 * lane], v1 = red[4 * lane + 1], v2 = red[4 * lane + 2], v3 = red[4 * lane + 3];
        wht128(v0, v1, v2, v3, lane);
        const int n = cb * 128 + 4 * lane;
        const __half* svh = a.sv[e * 3 + mat];
        const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        __nv_bfloat162 o01 = __floats2bfloat162_rn(v0 * RSQRT128 * __low2float(s01), v1 * RSQRT128 * __high2float(s01));
        __nv_bfloat162 o23 = __floats2bfloat162_rn(v2 * RSQRT128 * __low2float(s23), v3 * RSQRT128 * __high2float(s23));
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(Y + ((size_t)p * 2 + mat) * I + n) = o;
    }
}

// grid (H/128, P); YD [P, H] fp32
template <int ST>
__global__ void __launch_bounds__(256) moe_down_kernel(const __nv_bfloat16* __restrict__ YGU,
        const int* __restrict__ IDS, const int* __restrict__ ORDER, MoEArgs a, float* __restrict__ YD,
        int H, int I) {
    extern __shared__ uint32_t smem[];
    __half* xh = reinterpret_cast<__half*>(smem);                          // [I]
    float* red = reinterpret_cast<float*>(smem + I / 2);
    uint32_t* stage = smem + I / 2 + 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int p = ORDER[blockIdx.y];
    const int e = IDS[p];
    const int cb = blockIdx.x;
    const __half* suh = a.su[e * 3 + 2];
    const __nv_bfloat16* yg = YGU + (size_t)p * 2 * I;
    const __nv_bfloat16* yu = yg + I;
    // rot_in_silu_kernel's arithmetic
    for (int blk = warp; blk * 128 < I; blk += 8) {
        const int k = blk * 128 + 4 * lane;
        const uint2 gr = *reinterpret_cast<const uint2*>(yg + k);
        const uint2 ur = *reinterpret_cast<const uint2*>(yu + k);
        const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
        float gv[4] = {bf_lo(gr.x), bf_hi(gr.x), bf_lo(gr.y), bf_hi(gr.y)};
        float uv[4] = {bf_lo(ur.x), bf_hi(ur.x), bf_lo(ur.y), bf_hi(ur.y)};
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = bfr(bfr(gv[q] / (1.0f + ::expf(-gv[q]))) * uv[q]);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
        float v0 = v[0] * __low2float(s01), v1 = v[1] * __high2float(s01);
        float v2 = v[2] * __low2float(s23), v3 = v[3] * __high2float(s23);
        wht128(v0, v1, v2, v3, lane);
        __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
        __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xh + k) = o;
    }
    __syncthreads();
    moe_row_gemm_k<ST>(a.kb[e * 3 + 2], xh, a.tr[e * 3 + 2], H / 16, cb, I, stage, red);
    if (warp == 0) {
        float v0 = red[4 * lane], v1 = red[4 * lane + 1], v2 = red[4 * lane + 2], v3 = red[4 * lane + 3];
        wht128(v0, v1, v2, v3, lane);
        const int n = cb * 128 + 4 * lane;
        const __half* svh = a.sv[e * 3 + 2];
        const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        *reinterpret_cast<float4*>(YD + (size_t)p * H + n) = make_float4(
            v0 * RSQRT128 * __low2float(s01), v1 * RSQRT128 * __high2float(s01),
            v2 * RSQRT128 * __low2float(s23), v3 * RSQRT128 * __high2float(s23));
    }
}

// grid (H/256, M), 256 threads: out[m, c] = bf16(sum over slots in ascending expert id of w * y)
__global__ void __launch_bounds__(256) moe_combine_kernel(const float* __restrict__ YD,
        const int* __restrict__ IDS, const float* __restrict__ WTS, int S, __nv_bfloat16* __restrict__ OUT,
        int ldo, int H) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    __shared__ int ord[64];
    if (threadIdx.x < S) {
        const int e = IDS[m * S + threadIdx.x];
        int r = 0;
        for (int s = 0; s < S; ++s) r += IDS[m * S + s] < e;
        ord[r] = threadIdx.x;
    }
    __syncthreads();
    if (c >= H) return;
    float acc = 0.f;
    for (int r = 0; r < S; ++r) {
        const int s = ord[r];
        acc = fmaf(WTS[m * S + s], YD[(size_t)(m * S + s) * H + c], acc);
    }
    OUT[(size_t)m * ldo + c] = __float2bfloat16_rn(acc);
}

}  // namespace

// --------------------------------------------------------------------------------- host side
static int lpr_for(int K) { return K >= 2048 ? 32 : (K >= 512 ? 16 : 8); }

#define GEMV_LAUNCH(TW, LPR, MC)                                                                   \
    gemv_kernel<TW, LPR, MC><<<dim3((N + 8 * (32 / LPR) - 1) / (8 * (32 / LPR))), 256, 0, st>>>(   \
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const TW*)w.data_ptr(),             \
        s_ptr, (float*)y.data_ptr(), (int)y.stride(0), M, N, K)

template <typename TW>
static void gemv_dispatch(const torch::Tensor& x, const torch::Tensor& w, const float* s_ptr,
                          torch::Tensor& y, int M, int N, int K, cudaStream_t st) {
    const int lpr = lpr_for(K);
    if (M == 1) {
        if (lpr == 32) GEMV_LAUNCH(TW, 32, 1); else if (lpr == 16) GEMV_LAUNCH(TW, 16, 1); else GEMV_LAUNCH(TW, 8, 1);
    } else {
        if (lpr == 32) GEMV_LAUNCH(TW, 32, 8); else if (lpr == 16) GEMV_LAUNCH(TW, 16, 8); else GEMV_LAUNCH(TW, 8, 8);
    }
}

void fn_gemv(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> s, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0, "x bf16 [M, K]");
    TORCH_CHECK(w.dim() == 2 && w.is_contiguous(), "w [N, K] contiguous");
    TORCH_CHECK(y.dtype() == torch::kFloat && y.stride(1) == 1, "y fp32 [M, N]");
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N, "shapes");
    const float* sp = nullptr;
    if (w.dtype() == torch::kInt8) {
        TORCH_CHECK(s.has_value() && s->dtype() == torch::kFloat && s->numel() == N, "int8 rows need fp32 scales [N]");
        TORCH_CHECK(K % 16 == 0, "K % 16");
        sp = (const float*)s->data_ptr();
        gemv_dispatch<int8_t>(x, w, sp, y, M, N, K, st);
    } else {
        TORCH_CHECK(w.dtype() == torch::kHalf && K % 8 == 0, "w int8 or fp16, K % 8");
        gemv_dispatch<__half>(x, w, nullptr, y, M, N, K, st);
    }
}

void fn_mix_up(torch::Tensor d, torch::Tensor n, torch::Tensor w, c10::optional<torch::Tensor> s,
               torch::Tensor x, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(d.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = d.size(0), R = w.size(1), H = x.size(1);
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1 && d.size(1) >= R, "d fp32 [M, >=R]");
    TORCH_CHECK(n.dtype() == torch::kBFloat16 && n.stride(1) == 1 && n.size(1) == hc * H, "n bf16 [M, hc H]");
    TORCH_CHECK(hc == 4 && w.size(0) == 4 * H && w.is_contiguous(), "w [4 H, R], hc 4");
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.size(0) == M, "x bf16 [M, H]");
    const size_t smem = (size_t)M * R * 2;
    dim3 grid((H + 7) / 8);
    const float inv = 1.0f / (float)hc;
    if (w.dtype() == torch::kInt8) {
        TORCH_CHECK(s.has_value() && s->numel() == 4 * H && R % 16 == 0, "int8 scales, R % 16");
        auto k = M == 1 ? mix_up_kernel<int8_t, 1> : mix_up_kernel<int8_t, 8>;
        k<<<grid, 256, smem, st>>>((const float*)d.data_ptr(), (int)d.stride(0), (const __nv_bfloat16*)n.data_ptr(),
            (int)n.stride(0), (const int8_t*)w.data_ptr(), (const float*)s->data_ptr(),
            (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, R, inv);
    } else {
        TORCH_CHECK(w.dtype() == torch::kHalf && R % 8 == 0, "fp16 up weight, R % 8");
        auto k = M == 1 ? mix_up_kernel<__half, 1> : mix_up_kernel<__half, 8>;
        k<<<grid, 256, smem, st>>>((const float*)d.data_ptr(), (int)d.stride(0), (const __nv_bfloat16*)n.data_ptr(),
            (int)n.stride(0), (const __half*)w.data_ptr(), nullptr,
            (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, R, inv);
    }
}

void fn_router(torch::Tensor logits, int64_t E, int64_t k, torch::Tensor ids, torch::Tensor wts) {
    const at::cuda::OptionalCUDAGuard guard(logits.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = logits.size(0);
    TORCH_CHECK(logits.dtype() == torch::kFloat && logits.stride(1) == 1 && logits.size(1) >= E + 1, "logits fp32 [M, E + 1]");
    TORCH_CHECK(ids.dtype() == torch::kInt && wts.dtype() == torch::kFloat && ids.is_contiguous() && wts.is_contiguous(), "ids int32, wts fp32");
    TORCH_CHECK(ids.numel() == M * (k + 1) && wts.numel() == M * (k + 1) && k < 32, "slots");
    TORCH_CHECK(E == 512, "router kernel is built for 512 experts");
    router_kernel<16><<<M, 32, 0, st>>>((const float*)logits.data_ptr(), (int)logits.stride(0), (int)E, (int)k,
        (int*)ids.data_ptr(), (float*)wts.data_ptr());
}

static MoEArgs moe_args(const torch::Tensor& tr, const torch::Tensor& kb, const torch::Tensor& su,
                        const torch::Tensor& sv) {
    MoEArgs a;
    a.tr = (const uint32_t* const*)tr.data_ptr();
    a.kb = (const int*)kb.data_ptr();
    a.su = (const __half* const*)su.data_ptr();
    a.sv = (const __half* const*)sv.data_ptr();
    return a;
}

constexpr int MOE_ST = 3;

// the input row (fp16), the reduction row, and 8 warps x MOE_ST stages x 8 tiles of the bank's
// widest trellis tile (8 kbmax words)
static size_t moe_smem(int kin, int kbmax) {
    return (size_t)(kin / 2 + 128) * 4 + (size_t)8 * MOE_ST * 8 * 8 * kbmax * 4;
}

void fn_moe_gu(torch::Tensor x, torch::Tensor ids, torch::Tensor order, torch::Tensor tr, torch::Tensor kb,
               torch::Tensor su, torch::Tensor sv, torch::Tensor y, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), H = x.size(1), P = ids.numel(), S = P / M, I = y.size(2);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, H]");
    TORCH_CHECK(ids.dtype() == torch::kInt && order.dtype() == torch::kInt && order.numel() == P && P == M * S, "ids/order int32 [M S]");
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && y.size(0) == P && y.size(1) == 2, "y bf16 [P, 2, I]");
    TORCH_CHECK(H % 128 == 0 && I % 128 == 0, "H, I multiples of 128");
    const size_t smem = moe_smem(H, (int)kbmax);
    auto kern = moe_gu_kernel<MOE_ST>;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); attr = true; }
    kern<<<dim3(2 * (I / 128), P), 256, smem, st>>>((const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const int*)ids.data_ptr(), (const int*)order.data_ptr(), S, moe_args(tr, kb, su, sv),
        (__nv_bfloat16*)y.data_ptr(), H, I);
}

void fn_moe_down(torch::Tensor ygu, torch::Tensor ids, torch::Tensor order, torch::Tensor tr, torch::Tensor kb,
                 torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(ygu.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = ygu.size(0), I = ygu.size(2), H = yd.size(1);
    TORCH_CHECK(ygu.dtype() == torch::kBFloat16 && ygu.is_contiguous() && ygu.size(1) == 2, "ygu bf16 [P, 2, I]");
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == P, "yd fp32 [P, H]");
    TORCH_CHECK(ids.numel() == P && order.numel() == P, "pairs");
    const size_t smem = moe_smem(I, (int)kbmax);
    auto kern = moe_down_kernel<MOE_ST>;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); attr = true; }
    kern<<<dim3(H / 128, P), 256, smem, st>>>((const __nv_bfloat16*)ygu.data_ptr(), (const int*)ids.data_ptr(),
        (const int*)order.data_ptr(), moe_args(tr, kb, su, sv), (float*)yd.data_ptr(), H, I);
}

void fn_moe_combine(torch::Tensor yd, torch::Tensor ids, torch::Tensor wts, torch::Tensor out) {
    const at::cuda::OptionalCUDAGuard guard(yd.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = out.size(0), H = out.size(1), S = ids.numel() / M;
    TORCH_CHECK(S <= 64 && yd.size(0) == M * S && yd.size(1) == H, "combine shapes");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.stride(1) == 1, "out bf16 [M, H]");
    moe_combine_kernel<<<dim3((H + 255) / 256, M), 256, 0, st>>>((const float*)yd.data_ptr(),
        (const int*)ids.data_ptr(), (const float*)wts.data_ptr(), S, (__nv_bfloat16*)out.data_ptr(),
        (int)out.stride(0), H);
}
"""

_CPP = (
    "void fn_gemv(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> s, torch::Tensor y);\n"
    "void fn_mix_up(torch::Tensor d, torch::Tensor n, torch::Tensor w, c10::optional<torch::Tensor> s, "
    "torch::Tensor x, int64_t hc);\n"
    "void fn_router(torch::Tensor logits, int64_t E, int64_t k, torch::Tensor ids, torch::Tensor wts);\n"
    "void fn_moe_gu(torch::Tensor x, torch::Tensor ids, torch::Tensor order, torch::Tensor tr, "
    "torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor y, int64_t kbmax);\n"
    "void fn_moe_down(torch::Tensor ygu, torch::Tensor ids, torch::Tensor order, torch::Tensor tr, "
    "torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax);\n"
    "void fn_moe_combine(torch::Tensor yd, torch::Tensor ids, torch::Tensor wts, torch::Tensor out);\n")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        src = "#define EXL3_ST 3\n" + _HELPERS + _CUDA
        tag = hashlib.sha1((_CPP + src).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"fn_kernels_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                           functions=["fn_gemv", "fn_mix_up", "fn_router", "fn_moe_gu", "fn_moe_down",
                                      "fn_moe_combine"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _MOD


# ----------------------------------------------------------------------------------- weights
class MoEBank:
    """The experts of one layer as the kernels read them: E routed experts and the shared expert as
    expert E. Each projection is a trellis [Kin/16, N/16, 16 K] with suh [Kin] and svh [N] anywhere
    on the device; the bank holds their pointers ([E + 1, 3] for gate, up, down), their K and the
    tensors themselves (`keep`, so the pointers stay valid)."""

    def __init__(self, experts: list, device):
        """experts: [(gate, up, down)] of (trellis, suh, svh) device tensors, the shared expert last."""
        tr, kb, su, sv = [], [], [], []
        keep = []
        for proj in experts:
            for t, s_u, s_v in proj:
                assert t.dtype == torch.int16 and t.is_contiguous() and t.data_ptr() % 16 == 0
                assert s_u.dtype == torch.float16 and s_v.dtype == torch.float16
                tr.append(t.data_ptr())
                kb.append(t.shape[2] // 16)
                su.append(s_u.data_ptr())
                sv.append(s_v.data_ptr())
                keep += [t, s_u, s_v]
        self.keep = keep
        self.n = len(experts)                       # E + 1
        self.tr = torch.tensor(tr, dtype=torch.int64, device=device)
        self.kb = torch.tensor(kb, dtype=torch.int32, device=device)
        self.su = torch.tensor(su, dtype=torch.int64, device=device)
        self.sv = torch.tensor(sv, dtype=torch.int64, device=device)
        g = experts[0][0]
        self.H = g[0].shape[0] * 16
        self.I = g[0].shape[1] * 16
        self.ks = sorted(set(kb))
        if not set(self.ks) <= {2, 3, 4, 5, 6, 8}:
            raise ValueError(f"expert K {self.ks}: the MoE kernels take K 2..6 and 8")

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keep)

    def block(self, e: int, j: int):
        """Projection j (0 gate, 1 up, 2 down) of expert e as an EXL3Block (views; the reference
        checks dequantise it)."""
        t, s_u, s_v = self.keep[9 * e + 3 * j: 9 * e + 3 * j + 3]
        return X.EXL3Block(t, s_u, s_v)


def gemv(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] bf16 @ w^T -> fp32 [M, N]; w an fp16 tensor [N, K] or int8 rows (`pack.Int8Rows`)."""
    if hasattr(w, "q"):
        q, s = w.q, w.s
    else:
        q, s = w, None
    if out is None:
        out = torch.empty(x.shape[0], q.shape[0], dtype=torch.float32, device=x.device)
    _module().fn_gemv(x, q, s, out)
    return out


def mix_up(d: torch.Tensor, n: torch.Tensor, w_up, hc: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The mixer's second half: d [M, >=R] fp32 (the down projection), n [M, hc H] bf16 (the normed
    streams), w_up [hc H, R] fp16 or Int8Rows -> x [M, H] bf16."""
    q, s = (w_up.q, w_up.s) if hasattr(w_up, "q") else (w_up, None)
    H = q.shape[0] // hc
    if out is None:
        out = torch.empty(d.shape[0], H, dtype=torch.bfloat16, device=d.device)
    # a launch stages its rows' activations in shared memory: at most MIX_ROWS rows a launch
    for r0 in range(0, d.shape[0], MIX_ROWS):
        _module().fn_mix_up(d[r0:r0 + MIX_ROWS], n[r0:r0 + MIX_ROWS], q, s, out[r0:r0 + MIX_ROWS], hc)
    return out


MIX_ROWS = 64


def router(logits: torch.Tensor, E: int, k: int, ids=None, wts=None):
    """logits fp32 [M, E + 1] (the shared gate last) -> ids int32 [M, k + 1], wts fp32 [M, k + 1]."""
    M = logits.shape[0]
    if ids is None:
        ids = torch.empty(M, k + 1, dtype=torch.int32, device=logits.device)
        wts = torch.empty(M, k + 1, dtype=torch.float32, device=logits.device)
    _module().fn_router(logits, E, k, ids, wts)
    return ids, wts


def moe(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, bank: MoEBank, order: torch.Tensor | None = None,
        bufs: dict | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """The experts of every (row, slot) pair and their combine: x [M, H] bf16, ids/wts [M, S]
    (S = top-k + the shared slot) -> [M, H] bf16. `order` (int32 [M S]) is the launch order of the
    pairs; ascending expert id by default (`expert_order`). `bufs` holds preallocated
    intermediates for a graph."""
    M, S = ids.shape
    P = M * S
    mod = _module()
    if order is None:
        order = expert_order(ids)
    b = bufs if bufs is not None else {}
    ygu = b.get("ygu")
    if ygu is None or ygu.shape[0] != P:
        ygu = torch.empty(P, 2, bank.I, dtype=torch.bfloat16, device=x.device)
        yd = torch.empty(P, bank.H, dtype=torch.float32, device=x.device)
        if bufs is not None:
            bufs["ygu"], bufs["yd"] = ygu, yd
    else:
        yd = b["yd"]
    flat = ids.reshape(-1)
    mod.fn_moe_gu(x, flat, order, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1])
    mod.fn_moe_down(ygu, flat, order, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
    if out is None:
        out = torch.empty(M, bank.H, dtype=torch.bfloat16, device=x.device)
    mod.fn_moe_combine(yd, flat, wts.reshape(-1), out)
    return out


def expert_order(ids: torch.Tensor) -> torch.Tensor:
    """The pairs in ascending expert id (stable), as int32; fixed shape, no host sync."""
    return torch.sort(ids.reshape(-1), stable=True).indices.to(torch.int32)


# ------------------------------------------------------------------------------- references
def moe_reference(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, experts: list) -> torch.Tensor:
    """The kernels' arithmetic in torch, per pair, fp32 GEMV in place of the tensor-core sums:
    experts[e] = (gate, up, down) EXL3Blocks. Equal to the kernels within fp32 summation order
    (the trellis decode, the rotations and every bf16 rounding are the same)."""
    from tools.exl3_linear import hadamard128
    M, S = ids.shape
    H = x.shape[1]
    out = torch.zeros(M, H, dtype=torch.float32, device=x.device)
    Hd = hadamard128(x.device)

    def rot(v, suh):
        n = v.shape[-1]
        return ((v.float() * suh.float()).view(-1, n // 128, 128) @ Hd).view(-1, n).to(torch.float16).float()

    def gemm(xh, w):
        yp = xh @ w.inner().float()
        n = yp.shape[-1]
        return (yp.view(-1, n // 128, 128) @ Hd).view(-1, n) * w.svh.float()

    for m in range(M):
        es = ids[m].tolist()
        for s in sorted(range(S), key=lambda s: es[s]):
            g, u, d = experts[es[s]]
            xr = x[m:m + 1]
            yg = gemm(rot(xr, g.suh[0]), g).to(torch.bfloat16).float()
            yu = gemm(rot(xr, u.suh[0]), u).to(torch.bfloat16).float()
            a = (F_silu(yg).to(torch.bfloat16).float() * yu).to(torch.bfloat16)
            yd = gemm(rot(a, d.suh[0]), d)
            out[m] = out[m] + float(wts[m, s]) * yd[0]
    return out.to(torch.bfloat16)


def F_silu(v: torch.Tensor) -> torch.Tensor:
    return v / (1.0 + torch.exp(-v))


def router_reference(logits: torch.Tensor, E: int, k: int):
    """The router kernel's arithmetic in torch (ids int32, wts fp32)."""
    lb = logits[:, :E].to(torch.bfloat16).float()
    p = torch.softmax(lb, dim=-1)
    v, i = torch.sort(p, dim=-1, descending=True, stable=True)
    v, i = v[:, :k], i[:, :k]
    tot = v[:, 0].clone()
    for j in range(1, k):
        tot = tot + v[:, j]
    w = (v / tot[:, None]).to(torch.bfloat16).float()
    sg = torch.sigmoid(logits[:, E].to(torch.bfloat16).float()).to(torch.bfloat16).float()
    ids = torch.cat([i, torch.full_like(i[:, :1], E)], 1).to(torch.int32)
    return ids, torch.cat([w, sg[:, None]], 1)
