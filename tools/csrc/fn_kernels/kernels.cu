
constexpr float RSQRT128 = 0.08838834764831845f;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + ::expf(-v)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + ::expf(-v)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Programmatic dependent launch: a kernel launched with the attribute may start while its
// predecessor runs; `pdl_wait` is where it needs the predecessor's output (it returns when the whole
// predecessor grid has completed and flushed; a no-op without the attribute). Every kernel here
// triggers right after its wait, so its successor's CTAs can launch and prefetch their weights while
// it runs. Neither changes what a thread computes.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

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
// U chunks a pass: their weight loads are issued together (more bytes in flight a lane); each
// lane still adds its chunks in K order, so the sums are the one-chunk loop's bit for bit
// GS (int8 with a scale a group of G weights, G a multiple of VEC): a chunk's products summed on their
// own, then added times its group's scale (one FMA a chunk); null: the row's scale is the caller's,
// after the sums
template <typename TW, int LPR, int MC, int U = 4>
__device__ __forceinline__ void row_dot(const TW* __restrict__ wr, const __nv_bfloat16* __restrict__ X,
                                        int ldx, int m0, int mc, int K, int sl, float* acc,
                                        const float* __restrict__ GS = nullptr, int G = 0) {
    constexpr int VEC = WVec<TW>::VEC;
    constexpr int STEP = LPR * VEC;
#pragma unroll
    for (int m = 0; m < MC; ++m) acc[m] = 0.f;
    for (int k0 = sl * VEC; k0 < K; k0 += U * STEP) {
        float w[U][VEC], gs[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (k0 + u * STEP < K) {
                WVec<TW>::load(wr + k0 + u * STEP, w[u]);
                gs[u] = GS ? GS[(k0 + u * STEP) / G] : 1.0f;
            }
#pragma unroll
        for (int m = 0; m < MC; ++m) {
            if (m < mc) {
                float s = acc[m];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int k = k0 + u * STEP;
                    if (k < K) {
                        float x[VEC];
                        load_x8(X + (size_t)(m0 + m) * ldx + k, x);
                        if (VEC == 16) load_x8(X + (size_t)(m0 + m) * ldx + k + 8, x + 8);
                        if (GS) {
                            float t = 0.f;
#pragma unroll
                            for (int e = 0; e < VEC; ++e) t = fmaf(x[e], w[u][e], t);
                            s = fmaf(gs[u], t, s);
                        } else {
#pragma unroll
                            for (int e = 0; e < VEC; ++e) s = fmaf(x[e], w[u][e], s);
                        }
                    }
                }
                acc[m] = s;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MC; ++m)
#pragma unroll
        for (int o = LPR / 2; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
}

// SPD-231: `row_dot` for NB weight rows at once (rows wr + nb ldw, nb < nlive): each activation chunk is
// loaded once for the NB rows. Every (row, activation row) sum is `row_dot`'s own: the lane's chunks in K
// order, a chunk's products in element order (summed on their own, then times the group's scale, with
// GS), the same xor tree -- U changes nothing of that, and here a pass is one chunk.
template <typename TW, int LPR, int MC, int NB>
__device__ __forceinline__ void row_dot_nb(const TW* __restrict__ wr, size_t ldw, int nlive,
                                           const __nv_bfloat16* __restrict__ X, int ldx, int m0, int mc, int K,
                                           int sl, float (&acc)[NB][MC], const float* __restrict__ GS = nullptr,
                                           size_t ldg = 0, int G = 0) {
    constexpr int VEC = WVec<TW>::VEC;
    constexpr int STEP = LPR * VEC;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int m = 0; m < MC; ++m) acc[nb][m] = 0.f;
    for (int k0 = sl * VEC; k0 < K; k0 += STEP) {
        float w[NB][VEC], gs[NB];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
            if (nb < nlive) {
                WVec<TW>::load(wr + nb * ldw + k0, w[nb]);
                gs[nb] = GS ? GS[nb * ldg + k0 / G] : 1.0f;
            }
#pragma unroll
        for (int m = 0; m < MC; ++m) {
            if (m < mc) {
                float x[VEC];
                load_x8(X + (size_t)(m0 + m) * ldx + k0, x);
                if (VEC == 16) load_x8(X + (size_t)(m0 + m) * ldx + k0 + 8, x + 8);
#pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
                    if (nb < nlive) {
                        if (GS) {
                            float t = 0.f;
#pragma unroll
                            for (int e = 0; e < VEC; ++e) t = fmaf(x[e], w[nb][e], t);
                            acc[nb][m] = fmaf(gs[nb], t, acc[nb][m]);
                        } else {
#pragma unroll
                            for (int e = 0; e < VEC; ++e) acc[nb][m] = fmaf(x[e], w[nb][e], acc[nb][m]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int m = 0; m < MC; ++m)
#pragma unroll
            for (int o = LPR / 2; o > 0; o >>= 1) acc[nb][m] += __shfl_xor_sync(0xffffffffu, acc[nb][m], o);
}

__device__ __forceinline__ void store_out(float* y, float v) { *y = v; }
__device__ __forceinline__ void store_out(__nv_bfloat16* y, float v) { *y = __float2bfloat16_rn(v); }

template <typename TW, int LPR, int MC, typename TY>
__global__ void __launch_bounds__(256) gemv_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const TW* __restrict__ W, const float* __restrict__ S, TY* __restrict__ Y, int ldy,
        int M, int N, int K, int G) {
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / LPR, sl = lane % LPR;
    const int n = (blockIdx.x * (blockDim.x >> 5) + warp) * RPW + sub;
    const bool live = n < N;
    const TW* wr = W + (size_t)(live ? n : 0) * K;
    if (live)                                          // the weights do not depend on the predecessor
        for (int off = sl * 128; off < K * (int)sizeof(TW); off += LPR * 128)
            prefetch_l2(reinterpret_cast<const char*>(wr) + off);
    pdl_wait();
    pdl_trigger();
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[MC];
        row_dot<TW, LPR, MC>(wr, X, ldx, m0, mc, live ? K : 0, sl, acc, G ? S + (size_t)(live ? n : 0) * (K / G) : nullptr, G);
        if (live && sl == 0) {
            const float sc = S && !G ? S[n] : 1.0f;
#pragma unroll
            for (int m = 0; m < MC; ++m) if (m < mc) store_out(Y + (size_t)(m0 + m) * ldy + n, acc[m] * sc);
        }
    }
}

// The same dot products for a weight of few rows (`gemv_split`): one CTA a row, its K split into KS
// contiguous slices, warp w the slice w with gemv_kernel's lane-strided sums, the slices added in warp
// order; KS is a function of the weight's shape, so a row's bits do not depend on M.
template <typename TW, int MC, typename TY, int MINB = 1>
__global__ void __launch_bounds__(256, MINB) gemv_split_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const TW* __restrict__ W, const float* __restrict__ S, TY* __restrict__ Y, int ldy, int M, int N, int K, int G) {
    __shared__ float red[8][MC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, ks = blockDim.x >> 5;
    const int n = blockIdx.x, kc = K / ks, k0 = warp * kc;
    const TW* wr = W + (size_t)n * K + k0;
    for (int off = lane * 128; off < kc * (int)sizeof(TW); off += 32 * 128)
        prefetch_l2(reinterpret_cast<const char*>(wr) + off);
    pdl_wait();
    pdl_trigger();
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[MC];
        row_dot<TW, 32, MC>(wr, X + k0, ldx, m0, mc, kc, lane, acc, G ? S + (size_t)n * (K / G) + k0 / G : nullptr, G);
        if (lane == 0)
#pragma unroll
            for (int m = 0; m < MC; ++m) red[warp][m] = acc[m];
        __syncthreads();
        if (threadIdx.x < mc) {
            float t = red[0][threadIdx.x];
            for (int w = 1; w < ks; ++w) t += red[w][threadIdx.x];
            store_out(Y + (size_t)(m0 + threadIdx.x) * ldy + n, t * (S && !G ? S[n] : 1.0f));
        }
        __syncthreads();
    }
}

// SPD-231: `gemv_split_kernel` for M > 1 with NB weight rows a CTA: the same KS warps, slices, lanes and
// warp-order slice sums for every row (its bits), each activation chunk loaded once for the NB rows (the
// one-row-a-CTA kernel read all of x again for each of N rows: 212 MB of L2 a call at 32 rows).
template <typename TW, int MC, int NB, typename TY>
__global__ void __launch_bounds__(256) gemv_split_nb_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const TW* __restrict__ W, const float* __restrict__ S, TY* __restrict__ Y, int ldy, int M, int N, int K, int G) {
    __shared__ float red[8][NB][MC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, ks = blockDim.x >> 5;
    const int n0 = blockIdx.x * NB, nlive = min(NB, N - n0), kc = K / ks, k0 = warp * kc;
    const TW* wr = W + (size_t)n0 * K + k0;
    for (int nb = 0; nb < nlive; ++nb)
        for (int off = lane * 128; off < kc * (int)sizeof(TW); off += 32 * 128)
            prefetch_l2(reinterpret_cast<const char*>(wr + (size_t)nb * K) + off);
    pdl_wait();
    pdl_trigger();
    const float* gs = G ? S + (size_t)n0 * (K / G) + k0 / G : nullptr;
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[NB][MC];
        row_dot_nb<TW, 32, MC, NB>(wr, (size_t)K, nlive, X + k0, ldx, m0, mc, kc, lane, acc, gs, (size_t)(G ? K / G : 0), G);
        if (lane == 0)
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                for (int m = 0; m < MC; ++m) red[warp][nb][m] = acc[nb][m];
        __syncthreads();
        if (threadIdx.x < NB * MC) {
            const int nb = threadIdx.x / MC, m = threadIdx.x % MC;
            if (nb < nlive && m < mc) {
                float t = red[0][nb][m];
                for (int w = 1; w < ks; ++w) t += red[w][nb][m];
                store_out(Y + (size_t)(m0 + m) * ldy + n0 + nb, t * (S && !G ? S[n0 + nb] : 1.0f));
            }
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------------- mix_up
// A[M, R] = bf16(silu(bf16(D) / HC)) staged in shared memory (bf16), then per column j (one warp,
// four groups of 8 lanes = the four streams): u_s = bf16(up row (s H + j) . a), x = mean of
// bf16(bf16(sigmoid(u_s)) * n_s).
template <typename TW, int MC, int MINB = 1>
__global__ void __launch_bounds__(256, MINB) mix_up_kernel(const float* __restrict__ D, int ldd,
        const __nv_bfloat16* __restrict__ Nn, int ldn, const TW* __restrict__ W, const float* __restrict__ S,
        __nv_bfloat16* __restrict__ Xo, int ldxo, int M, int H, int R, float inv_hc, int G) {
    extern __shared__ __nv_bfloat16 a_s[];                 // [M, R]
    {
        const int j0 = blockIdx.x * 8 + (threadIdx.x >> 5), s0 = (threadIdx.x & 31) >> 3, l0 = threadIdx.x & 7;
        if (j0 < H)
            for (int off = l0 * 128; off < R * (int)sizeof(TW); off += 8 * 128)
                prefetch_l2(reinterpret_cast<const char*>(W + (size_t)(s0 * H + j0) * R) + off);
    }
    pdl_wait();
    pdl_trigger();
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
    const float sc = S && !G ? S[n] : 1.0f;
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[MC];
        row_dot<TW, 8, MC>(wr, a_s, R, m0, mc, R, sl, acc, G ? S + (size_t)n * (R / G) : nullptr, G);
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

// SPD-231: mix_up for M > 1. A = bf16(silu(bf16(D) / HC)) [M, R] written once (`mix_a_kernel`, mix_up_kernel's
// staging arithmetic), then a group of 8 lanes owns one column j and its 4 stream rows s H + j in-lane:
// each A chunk is loaded once for the 4 rows. Per (row, stream) the sums are mix_up_kernel's (the same 8
// lanes, chunks and xor tree), the products bf16(bf16(sigmoid(bf16(u))) n) likewise, and their sum
// (((p0 + p1) + p2) + p3) / HC in mix_up_kernel's order. 4 columns a warp, 16 a CTA of 4 warps.
__global__ void __launch_bounds__(256) mix_a_kernel(const float* __restrict__ D, int ldd, __nv_bfloat16* __restrict__ A,
                                                    int M, int R, float inv_hc) {
    pdl_wait();
    pdl_trigger();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < M * R) {
        const int m = e / R, r = e % R;
        A[e] = __float2bfloat16_rn(silu(bfr(bfr(D[(size_t)m * ldd + r]) * inv_hc)));
    }
}

template <typename TW, int MC>
__global__ void __launch_bounds__(128) mix_up_nb_kernel(const __nv_bfloat16* __restrict__ A,
        const __nv_bfloat16* __restrict__ Nn, int ldn, const TW* __restrict__ W, const float* __restrict__ S,
        __nv_bfloat16* __restrict__ Xo, int ldxo, int M, int H, int R, float inv_hc, int G) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sl = lane & 7, grp = lane >> 3;
    const int j = (blockIdx.x * 4 + warp) * 4 + grp;
    const bool live = j < H;
    const int jj = live ? j : 0;
    if (live)
        for (int s = 0; s < 4; ++s)
            for (int off = sl * 128; off < R * (int)sizeof(TW); off += 8 * 128)
                prefetch_l2(reinterpret_cast<const char*>(W + (size_t)(s * H + jj) * R) + off);
    pdl_wait();
    pdl_trigger();
    const TW* wr = W + (size_t)jj * R;                    // stream s's row: wr + s H R
    const float* gs = G ? S + (size_t)jj * (R / G) : nullptr;
    for (int m0 = 0; m0 < M; m0 += MC) {
        const int mc = min(MC, M - m0);
        float acc[4][MC];
        row_dot_nb<TW, 8, MC, 4>(wr, (size_t)H * R, live ? 4 : 0, A, R, m0, mc, R, sl, acc, gs,
                                 (size_t)H * (G ? R / G : 0), G);
        if (live && sl == 0) {
#pragma unroll
            for (int m = 0; m < MC; ++m) {
                if (m < mc) {
                    float p[4];
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const int n = s * H + j;
                        const float sc = S && !G ? S[n] : 1.0f;
                        const float u = bfr(sigm(bfr(acc[s][m] * sc)));
                        p[s] = bfr(u * __bfloat162float(Nn[(size_t)(m0 + m) * ldn + n]));
                    }
                    Xo[(size_t)(m0 + m) * ldxo + j] = __float2bfloat16_rn((((p[0] + p[1]) + p[2]) + p[3]) * inv_hc);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------------- mixers on tensor cores
// SPD-241 (QWEN38_FN_MIXER_MMA): the mixers' int8 group-scaled GEMVs on mma.sync m16n8k16 bf16 at every M (the
// decode step's one row included, so a row's bits never depend on its call's rows). A warp owns 8 weight rows
// (B's n) and up to 16 activation rows a tile (A's m; rows past M read as zero). Inside each 64-wide K chunk the
// logical k of a kstep maps to physical k so a lane's fragment is 16 contiguous bytes of its weight row: lane
// (g, t) at kstep s holds physical k = 16 t + 4 s + {0, 1} (b0 / a0) and + {2, 3} (b1 / a2), for A and B alike.
// int8 is exact in bf16 and products are exact in fp32; the order of a row's sums is fixed: the tensor core's
// sum of a kstep, ksteps in order into a group's fresh fragment, then acc = fma(scale, group sum, acc) in group
// order (and, for the GEMV, its K slices added in slice order by the column's last slice CTA).
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// two int8 of a word (bytes b, b + 1) as a bf16 pair (exact)
__device__ __forceinline__ uint32_t i8x2_bf16(uint32_t w, int b) {
    const float lo = (float)(int8_t)((w >> (8 * b)) & 0xff), hi = (float)(int8_t)((w >> (8 * b + 8)) & 0xff);
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// a K range of one 16-row tile: CH chunks of 64 from the lane's registers WV (16 bytes a chunk) and the tile's
// rows in shared memory (XS + row * LDS, the range's first column at 0); GC chunks a scale group; SC(c) the
// scales of the lane's two C columns for chunk group c. acc[4] += the tile's sums (c0/c1 row g, c2/c3 row g + 8).
template <int CH, int GC>
__device__ __forceinline__ void mma_tile(const uint4* WV, const __nv_bfloat16* XS, int lds, int r0, int M,
                                         const float* sc0, const float* sc1, float* acc) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bool l0 = r0 + g < M, l1 = r0 + g + 8 < M;
    const __nv_bfloat16* x0 = XS + (size_t)(r0 + g) * lds + 16 * t;
    const __nv_bfloat16* x1 = XS + (size_t)(r0 + g + 8) * lds + 16 * t;
#pragma unroll
    for (int q = 0; q < CH / GC; ++q) {
        float part[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int cc = 0; cc < GC; ++cc) {
            const int c = q * GC + cc;
            const uint32_t w[4] = {WV[c].x, WV[c].y, WV[c].z, WV[c].w};
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const uint2 xa = l0 ? *reinterpret_cast<const uint2*>(x0 + 64 * c + 4 * st) : make_uint2(0u, 0u);
                const uint2 xb = l1 ? *reinterpret_cast<const uint2*>(x1 + 64 * c + 4 * st) : make_uint2(0u, 0u);
                mma_bf16(part, xa.x, xb.x, xa.y, xb.y, i8x2_bf16(w[st], 0), i8x2_bf16(w[st], 2));
            }
        }
        acc[0] = fmaf(sc0[q], part[0], acc[0]);
        acc[1] = fmaf(sc1[q], part[1], acc[1]);
        acc[2] = fmaf(sc0[q], part[2], acc[2]);
        acc[3] = fmaf(sc1[q], part[3], acc[3]);
    }
}

#define MIX_KSLICE 512
#define MIX_LDS (MIX_KSLICE + 8)
#define MIX_SLICES 24          // the most K slices a tensor-core GEMV adds (K <= 12,288)
// down|inject GEMV: X bf16 [M, K] (M <= 64), W int8 [N, K], S fp32 [N, K / G] (G a multiple of 64 dividing the
// slice) -> P fp32 [K / MIX_KSLICE][M][N] partial sums; the last of a column's slice CTAs to finish (CNT, a
// ticket a column, back to 0 after) adds the column's partials in slice order into Y fp32 [M, ldy]. grid (ceil(N / 32), K / MIX_KSLICE), 128 threads: warp w the 8 rows from
// 32 blockIdx.x + 8 w; the CTA's slice of X staged once in shared memory.
template <int GC>
__global__ void __launch_bounds__(128) mixer_mma_gemv_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const int8_t* __restrict__ W, const float* __restrict__ S, float* __restrict__ P, int M, int N, int K, int G,
        float* __restrict__ Y, int ldy, int* __restrict__ CNT) {
    extern __shared__ __nv_bfloat16 xs[];                  // [M rounded to 16][MIX_LDS]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 32 + warp * 8, k0 = blockIdx.y * MIX_KSLICE;
    const int n = n0 + g;
    uint4 wv[MIX_KSLICE / 64];
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int c = 0; c < MIX_KSLICE / 64; ++c)        // the weights and scales do not depend on the predecessor
        wv[c] = n < N ? *reinterpret_cast<const uint4*>(W + (size_t)n * K + k0 + 64 * c + 16 * t) : z4;
    const int ng = K / G, q0 = k0 / G;
    constexpr int NQ = MIX_KSLICE / 64 / GC;
    float sc0[NQ], sc1[NQ];
    const int na = n0 + 2 * t, nb = na + 1;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        sc0[q] = na < N ? S[(size_t)na * ng + q0 + q] : 0.f;
        sc1[q] = nb < N ? S[(size_t)nb * ng + q0 + q] : 0.f;
    }
    pdl_wait();
    pdl_trigger();
    for (int e = threadIdx.x; e < M * (MIX_KSLICE / 8); e += blockDim.x) {
        const int m = e / (MIX_KSLICE / 8), k8 = e % (MIX_KSLICE / 8);
        *reinterpret_cast<uint4*>(xs + (size_t)m * MIX_LDS + 8 * k8) =
            *reinterpret_cast<const uint4*>(X + (size_t)m * ldx + k0 + 8 * k8);
    }
    __syncthreads();
    float* p = P + (size_t)blockIdx.y * M * N;
    if (n0 < N) {
        for (int r0 = 0; r0 < M; r0 += 16) {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            mma_tile<MIX_KSLICE / 64, GC>(wv, xs, MIX_LDS, r0, M, sc0, sc1, acc);
            const int m0 = r0 + g, m1 = m0 + 8;
            if (m0 < M && na < N) p[(size_t)m0 * N + na] = acc[0];
            if (m0 < M && nb < N) p[(size_t)m0 * N + nb] = acc[1];
            if (m1 < M && na < N) p[(size_t)m1 * N + na] = acc[2];
            if (m1 < M && nb < N) p[(size_t)m1 * N + nb] = acc[3];
        }
    }
    // the column's last slice adds every slice's partials in slice order
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(CNT + blockIdx.x, 1) == (int)gridDim.y - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int KS = gridDim.y, c0 = blockIdx.x * 32, nc = min(32, N - c0);
    for (int e = threadIdx.x; e < M * nc; e += blockDim.x) {
        const int m = e / nc, nn = c0 + e % nc;
        const size_t o = (size_t)m * N + nn;
        float a[MIX_SLICES];                            // every slice's partial in flight, then added in order
#pragma unroll
        for (int sl = 0; sl < MIX_SLICES; ++sl) a[sl] = sl < KS ? __ldcg(P + (size_t)sl * M * N + o) : 0.f;
        float v = a[0];
#pragma unroll
        for (int sl = 1; sl < MIX_SLICES; ++sl) if (sl < KS) v += a[sl];
        Y[(size_t)m * ldy + nn] = v;
    }
    if (threadIdx.x == 0) CNT[blockIdx.x] = 0;
}

#define MIXUP_R 320
#define MIXUP_LDS (MIXUP_R + 8)
// mix_up: A bf16 [M, R] (`mix_a_kernel`'s), Nn bf16 [M, 4 H], W int8 [4 H, R] (scales a group of 64: [4 H, R / 64])
// -> Xo bf16 [M, H]. grid (H / 8), 128 threads: warp s the stream s's 8 rows s H + j0.., the whole K; the
// epilogue is mix_up_kernel's: u = bf16(sigmoid(bf16(sum))), p = bf16(u n), x = bf16((((p0 + p1) + p2) + p3) / hc).
__global__ void __launch_bounds__(128) mixer_mma_mixup_kernel(const __nv_bfloat16* __restrict__ A,
        const __nv_bfloat16* __restrict__ Nn, int ldn, const int8_t* __restrict__ W, const float* __restrict__ S,
        __nv_bfloat16* __restrict__ Xo, int ldxo, int M, int H, float inv_hc) {
    extern __shared__ __nv_bfloat16 sm[];
    __nv_bfloat16* as = sm;                                         // [M rounded to 16][MIXUP_LDS]
    const int mr = (M + 15) / 16 * 16;
    float* ps = reinterpret_cast<float*>(sm + (size_t)mr * MIXUP_LDS);   // [4][M][8]
    const int lane = threadIdx.x & 31, s = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int j0 = blockIdx.x * 8, n = s * H + j0 + g;
    uint4 wv[MIXUP_R / 64];
#pragma unroll
    for (int c = 0; c < MIXUP_R / 64; ++c)                // the weights and scales do not depend on the predecessor
        wv[c] = *reinterpret_cast<const uint4*>(W + (size_t)n * MIXUP_R + 64 * c + 16 * t);
    constexpr int NQ = MIXUP_R / 64;
    float sc0[NQ], sc1[NQ];
    const int na = s * H + j0 + 2 * t, nb = na + 1;
#pragma unroll
    for (int q = 0; q < NQ; ++q) { sc0[q] = S[(size_t)na * NQ + q]; sc1[q] = S[(size_t)nb * NQ + q]; }
    pdl_wait();
    pdl_trigger();
    for (int e = threadIdx.x; e < M * (MIXUP_R / 8); e += blockDim.x) {
        const int m = e / (MIXUP_R / 8), k8 = e % (MIXUP_R / 8);
        *reinterpret_cast<uint4*>(as + (size_t)m * MIXUP_LDS + 8 * k8) =
            *reinterpret_cast<const uint4*>(A + (size_t)m * MIXUP_R + 8 * k8);
    }
    __syncthreads();
    for (int r0 = 0; r0 < M; r0 += 16) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        mma_tile<MIXUP_R / 64, 1>(wv, as, MIXUP_LDS, r0, M, sc0, sc1, acc);
        const int m0 = r0 + g, m1 = m0 + 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = i < 2 ? m0 : m1, jj = 2 * t + (i & 1);
            if (m < M) {
                const float u = bfr(sigm(bfr(acc[i])));
                ps[((size_t)s * M + m) * 8 + jj] = bfr(u * __bfloat162float(Nn[(size_t)m * ldn + s * H + j0 + jj]));
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < M * 8; e += blockDim.x) {
        const int m = e / 8, jj = e % 8;
        const float p0 = ps[(0 * M + m) * 8 + jj], p1 = ps[(1 * M + m) * 8 + jj];
        const float p2 = ps[(2 * M + m) * 8 + jj], p3 = ps[(3 * M + m) * 8 + jj];
        Xo[(size_t)m * ldxo + j0 + jj] = __float2bfloat16_rn((((p0 + p1) + p2) + p3) * inv_hc);
    }
}

// ----------------------------------------------------------------------------------- router
// one warp a row; L [M, ldl] fp32 logits (E routed + 1 shared gate), out ids/wts [M, k + 1]
template <int PER>
__global__ void __launch_bounds__(32) router_kernel(const float* __restrict__ L, int ldl, int E, int k,
        int* __restrict__ IDS, float* __restrict__ WTS) {
    pdl_wait();
    pdl_trigger();
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
// K tiles split over the CTA's 8 warps in contiguous slices, each warp's 8 column tiles through
// cp.async stages, the warp partials added in warp order. `issue_first` streams the first stages
// (the trellis does not depend on the input, so a kernel calls it before it has the input);
// `run` waits for XH and finishes. RED [8][128].
template <int KB, int ST, int WARPS>
struct RowGemm {
    static constexpr int TW = tile_u32(KB), NTW = 8, SW = NTW * TW;
    const uint32_t* T; int ldt, nt0, k0, k1, lane, warp;
    uint32_t* stage;

    __device__ __forceinline__ RowGemm(const uint32_t* T_, int ldt_, int cb, int Kin, uint32_t* stage_all)
        : RowGemm(T_, ldt_, cb, 0, Kin / 16, stage_all) {}
    // the K tiles [ka, kb) split over the CTA's warps in contiguous slices
    __device__ __forceinline__ RowGemm(const uint32_t* T_, int ldt_, int cb, int ka, int kb, uint32_t* stage_all)
        : T(T_), ldt(ldt_), nt0(cb * 8) {
        lane = threadIdx.x & 31; warp = threadIdx.x >> 5;
        stage = stage_all + (size_t)warp * ST * SW;
        const int per = (kb - ka + WARPS - 1) / WARPS;
        k0 = min(kb, ka + warp * per); k1 = min(kb, k0 + per);
    }
    __device__ __forceinline__ void issue(int kt, int buf) {
        if (kt < k1) {
            const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
            uint4* dst = reinterpret_cast<uint4*>(stage + buf * SW);
            for (int c = lane; c < SW / 4; c += 32) cp_async16(dst + c, src + c);
        }
        cp_commit();
    }
    __device__ __forceinline__ void issue_first() {
#pragma unroll
        for (int s = 0; s < ST - 1; ++s) issue(k0 + s, s);
    }
    __device__ __forceinline__ void run(const __half* __restrict__ xh, float* __restrict__ red) {
        const int g = lane >> 2, t = lane & 3;
        float acc[NTW][2][4];
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[j][h][c] = 0.f;
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
        if (g == 0) {
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = j * 16 + h * 8 + 2 * t;
                    red[warp * 128 + col] = acc[j][h][0];
                    red[warp * 128 + col + 1] = acc[j][h][1];
                }
        }
        __syncthreads();
    }
};

// warp 0's four columns: the warp partials added in warp order, then the output rotation
template <int WARPS>
__device__ __forceinline__ void red_rot(const float* red, int lane, float& v0, float& v1, float& v2, float& v3) {
    const int c = 4 * lane;
    v0 = red[c]; v1 = red[c + 1]; v2 = red[c + 2]; v3 = red[c + 3];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) {
        v0 += red[w * 128 + c]; v1 += red[w * 128 + c + 1]; v2 += red[w * 128 + c + 2]; v3 += red[w * 128 + c + 3];
    }
    wht128(v0, v1, v2, v3, lane);
}

struct MoEArgs {
    const uint32_t* const* tr;    // [E + 1, 3] trellis pointers (gate, up, down)
    const int* kb;                 // [E + 1, 3] bits
    const __half* const* su;       // [E + 1, 3] suh pointers
    const __half* const* sv;       // [E + 1, 3] svh pointers
};

template <int KB, int MOE_ST, int WARPS>
__device__ __forceinline__ void moe_gu_body(const __nv_bfloat16* __restrict__ X, int ldx, int m, int p, int e,
        int mat, int cb, MoEArgs a, __nv_bfloat16* __restrict__ Y, int H, int I, uint32_t* smem) {
    __half* xh = reinterpret_cast<__half*>(smem);                          // [H]
    float* red = reinterpret_cast<float*>(smem + H / 2);                   // [8][128]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemm<KB, MOE_ST, WARPS> rg(a.tr[e * 3 + mat], I / 16, cb, H, smem + H / 2 + 8 * 128);
    rg.issue_first();
    // the pair's rotated input (rot_in_kernel's arithmetic), while the first stages stream
    const __half* suh = a.su[e * 3 + mat];
    for (int blk = warp; blk * 128 < H; blk += WARPS) {
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
    rg.run(xh, red);
    if (warp == 0) {
        float v0, v1, v2, v3;
        red_rot<WARPS>(red, lane, v0, v1, v2, v3);
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

// grid (2 * I/128, P); blockIdx.y -> pair ORDER[y] (ORDER null: y); Y [P, 2, I] bf16
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) moe_gu_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const int* __restrict__ IDS, const int* __restrict__ ORDER, int S, MoEArgs a,
        __nv_bfloat16* __restrict__ Y, int H, int I) {
    extern __shared__ uint32_t smem[];
    pdl_wait();                                        // the router's ids, the mixer's x
    pdl_trigger();
    const int p = ORDER ? ORDER[blockIdx.y] : blockIdx.y;
    const int m = p / S;
    const int e = IDS[p];
    const int mat = blockIdx.x & 1, cb = blockIdx.x >> 1;
    switch (a.kb[e * 3 + mat]) {
        case 2: moe_gu_body<2, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 3: moe_gu_body<3, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 4: moe_gu_body<4, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 5: moe_gu_body<5, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 6: moe_gu_body<6, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 8: moe_gu_body<8, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 7: moe_gu_body<7, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 15: moe_gu_body<15, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 25: moe_gu_body<25, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        case 35: moe_gu_body<35, ST, WARPS>(X, ldx, m, p, e, mat, cb, a, Y, H, I, smem); break;
        default: break;
    }
}

template <int KB, int MOE_ST, int WARPS>
__device__ __forceinline__ void moe_down_body(const __nv_bfloat16* __restrict__ YGU, int p, int e, int cb,
        MoEArgs a, float* __restrict__ YD, int H, int I, uint32_t* smem) {
    __half* xh = reinterpret_cast<__half*>(smem);                          // [I]
    float* red = reinterpret_cast<float*>(smem + I / 2);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemm<KB, MOE_ST, WARPS> rg(a.tr[e * 3 + 2], H / 16, cb, I, smem + I / 2 + 8 * 128);
    rg.issue_first();                                  // the weights stream before gate|up is done
    pdl_wait();
    pdl_trigger();
    pdl_trigger();
    const __half* suh = a.su[e * 3 + 2];
    const __nv_bfloat16* yg = YGU + (size_t)p * 2 * I;
    const __nv_bfloat16* yu = yg + I;
    // rot_in_silu_kernel's arithmetic
    for (int blk = warp; blk * 128 < I; blk += WARPS) {
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
    rg.run(xh, red);
    if (warp == 0) {
        float v0, v1, v2, v3;
        red_rot<WARPS>(red, lane, v0, v1, v2, v3);
        const int n = cb * 128 + 4 * lane;
        const __half* svh = a.sv[e * 3 + 2];
        const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        *reinterpret_cast<float4*>(YD + (size_t)p * H + n) = make_float4(
            v0 * RSQRT128 * __low2float(s01), v1 * RSQRT128 * __high2float(s01),
            v2 * RSQRT128 * __low2float(s23), v3 * RSQRT128 * __high2float(s23));
    }
}

// grid (H/128, P); YD [P, H] fp32. The ids were written two launches back (the router), before
// gate|up waited on them, so the pointers and the first trellis stages are read before the wait.
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) moe_down_kernel(const __nv_bfloat16* __restrict__ YGU,
        const int* __restrict__ IDS, const int* __restrict__ ORDER, MoEArgs a, float* __restrict__ YD,
        int H, int I) {
    extern __shared__ uint32_t smem[];
    const int p = ORDER ? ORDER[blockIdx.y] : blockIdx.y;
    const int e = IDS[p];
    const int cb = blockIdx.x;
    switch (a.kb[e * 3 + 2]) {
        case 2: moe_down_body<2, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 3: moe_down_body<3, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 4: moe_down_body<4, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 5: moe_down_body<5, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 6: moe_down_body<6, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 8: moe_down_body<8, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 7: moe_down_body<7, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 15: moe_down_body<15, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 25: moe_down_body<25, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        case 35: moe_down_body<35, ST, WARPS>(YGU, p, e, cb, a, YD, H, I, smem); break;
        default: pdl_wait(); pdl_trigger(); break;
    }
}

// ENG-241: the experts by group. A pair's CTA above streams and decodes its expert's trellis tiles
// for one row; when several rows of a block (or of a joint pass) take the same expert, a group of up
// to MOE_GR of them shares one CTA: the tiles are decoded once and multiplied into every row of the
// group, each row in its own lane group of the mma's A operand (row 0 above: lanes g == 0). A row's
// sums are the same products in the same order, its K slices the same warps', its warp partials
// added in the same warp order, so a row's bits do not depend on its group (the load-time self-check
// compares them with the decode step's).
#define MOE_GR 8

// groups G [P][2 + MOE_GR] (expert, rows, the pairs), NG [1]: the pairs in launch order (ORDER, or 0..P-1),
// a group the run of one expert, split every MOE_GR pairs. One thread; P is at most a few hundred.
__global__ void moe_groups_kernel(const int* __restrict__ IDS, const int* __restrict__ ORDER, int P,
        int* __restrict__ G, int* __restrict__ NG) {
    pdl_wait();
    pdl_trigger();
    if (threadIdx.x != 0) return;
    int ng = 0, cur = -1;
    int* g = G;
    for (int i = 0; i < P; ++i) {
        const int p = ORDER ? ORDER[i] : i;
        const int e = IDS[p];
        if (ng == 0 || e != cur || g[1] == MOE_GR) {
            g = G + (size_t)ng * (2 + MOE_GR);
            g[0] = e; g[1] = 0;
            ++ng; cur = e;
        }
        g[2 + g[1]] = p;
        g[1] += 1;
    }
    NG[0] = ng;
}

// RowGemm over the rows of a group: row r of XH [MOE_GR][Kin] (fp16, shared) in lane group g == r.
// RED [WARPS][MOE_GR][128] may overlay XH: every warp is past its reads before any writes.
template <int KB, int ST, int WARPS>
struct RowGemmG : RowGemm<KB, ST, WARPS> {
    using B = RowGemm<KB, ST, WARPS>;
    __device__ __forceinline__ RowGemmG(const uint32_t* T_, int ldt_, int cb, int Kin, uint32_t* stage_all)
        : B(T_, ldt_, cb, Kin, stage_all) {}
    __device__ __forceinline__ void run_rows(const __half* __restrict__ xh, int Kin, int R, float* __restrict__ red) {
        const int g = this->lane >> 2, t = this->lane & 3;
        float acc[B::NTW][2][4];
#pragma unroll
        for (int j = 0; j < B::NTW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[j][h][c] = 0.f;
        const bool live = g < R;
        for (int kt = this->k0; kt < this->k1; ++kt) {
            const int i = kt - this->k0;
            this->issue(kt + ST - 1, (i + ST - 1) % ST);
            cp_wait<ST - 1>();
            __syncwarp();
            const uint32_t* buf = this->stage + (i % ST) * B::SW;
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(xh + (size_t)(live ? g : 0) * Kin + kt * 16);
            const uint32_t a0 = live ? xw[t] : 0u, a2 = live ? xw[t + 4] : 0u;
#pragma unroll
            for (int j = 0; j < B::NTW; ++j) {
                FragB f0, f1;
                dq_tile<KB>(buf + j * B::TW, this->lane * 8, f0, f1);
                mma(acc[j][0], a0, 0u, a2, 0u, f0.r[0], f0.r[1]);
                mma(acc[j][1], a0, 0u, a2, 0u, f1.r[0], f1.r[1]);
            }
            __syncwarp();
        }
        cp_wait<0>();
        __syncthreads();                               // every warp is done with XH
        if (live) {
#pragma unroll
            for (int j = 0; j < B::NTW; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = j * 16 + h * 8 + 2 * t;
                    red[((size_t)this->warp * MOE_GR + g) * 128 + col] = acc[j][h][0];
                    red[((size_t)this->warp * MOE_GR + g) * 128 + col + 1] = acc[j][h][1];
                }
        }
        __syncthreads();
    }
};

// row r's four columns of a lane: the warp partials in warp order, then the output rotation (red_rot's)
template <int WARPS>
__device__ __forceinline__ void red_rot_row(const float* red, int r, int lane, float& v0, float& v1, float& v2, float& v3) {
    const int c = 4 * lane;
    const float* q = red + (size_t)r * 128;
    v0 = q[c]; v1 = q[c + 1]; v2 = q[c + 2]; v3 = q[c + 3];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) {
        const float* qw = q + (size_t)w * MOE_GR * 128;
        v0 += qw[c]; v1 += qw[c + 1]; v2 += qw[c + 2]; v3 += qw[c + 3];
    }
    wht128(v0, v1, v2, v3, lane);
}

template <int KB, int MOE_ST, int WARPS>
__device__ __forceinline__ void moe_gu_group_body(const __nv_bfloat16* __restrict__ X, int ldx, int S,
        const int* __restrict__ grp, int mat, int cb, MoEArgs a, __nv_bfloat16* __restrict__ Y, int H, int I,
        uint32_t* smem) {
    const int e = grp[0], R = grp[1];
    __half* xh = reinterpret_cast<__half*>(smem);                          // [MOE_GR][H]
    float* red = reinterpret_cast<float*>(smem);                           // over xh, after the sums
    uint32_t* stages = smem + (size_t)MOE_GR * H / 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemmG<KB, MOE_ST, WARPS> rg(a.tr[e * 3 + mat], I / 16, cb, H, stages);
    rg.issue_first();
    const __half* suh = a.su[e * 3 + mat];
    for (int r = 0; r < R; ++r) {
        const int m = grp[2 + r] / S;
        for (int blk = warp; blk * 128 < H; blk += WARPS) {
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
            *reinterpret_cast<uint2*>(xh + (size_t)r * H + k) = o;
        }
    }
    __syncthreads();
    rg.run_rows(xh, H, R, red);
    const int n = cb * 128 + 4 * lane;
    const __half* svh = a.sv[e * 3 + mat];
    const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
    for (int r = warp; r < R; r += WARPS) {
        float v0, v1, v2, v3;
        red_rot_row<WARPS>(red, r, lane, v0, v1, v2, v3);
        __nv_bfloat162 o01 = __floats2bfloat162_rn(v0 * RSQRT128 * __low2float(s01), v1 * RSQRT128 * __high2float(s01));
        __nv_bfloat162 o23 = __floats2bfloat162_rn(v2 * RSQRT128 * __low2float(s23), v3 * RSQRT128 * __high2float(s23));
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(Y + ((size_t)grp[2 + r] * 2 + mat) * I + n) = o;
    }
}

// grid (2 * I/128, P): blockIdx.y -> group y (past NG: nothing); Y [P, 2, I] bf16
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) moe_gu_group_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const int* __restrict__ G, const int* __restrict__ NG, int S, MoEArgs a,
        __nv_bfloat16* __restrict__ Y, int H, int I) {
    extern __shared__ uint32_t smem[];
    pdl_wait();                                        // the groups, the mixer's x
    pdl_trigger();
    if ((int)blockIdx.y >= NG[0]) return;
    const int* grp = G + (size_t)blockIdx.y * (2 + MOE_GR);
    const int mat = blockIdx.x & 1, cb = blockIdx.x >> 1;
    switch (a.kb[grp[0] * 3 + mat]) {
        case 2: moe_gu_group_body<2, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 3: moe_gu_group_body<3, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 4: moe_gu_group_body<4, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 5: moe_gu_group_body<5, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 6: moe_gu_group_body<6, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 8: moe_gu_group_body<8, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 7: moe_gu_group_body<7, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 15: moe_gu_group_body<15, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 25: moe_gu_group_body<25, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        case 35: moe_gu_group_body<35, ST, WARPS>(X, ldx, S, grp, mat, cb, a, Y, H, I, smem); break;
        default: break;
    }
}

template <int KB, int MOE_ST, int WARPS>
__device__ __forceinline__ void moe_down_group_body(const __nv_bfloat16* __restrict__ YGU, const int* __restrict__ grp,
        int cb, MoEArgs a, float* __restrict__ YD, int H, int I, uint32_t* smem) {
    const int e = grp[0], R = grp[1];
    __half* xh = reinterpret_cast<__half*>(smem);                          // [MOE_GR][I]
    float* red = reinterpret_cast<float*>(smem + (size_t)MOE_GR * I / 2);   // [WARPS][MOE_GR][128]
    uint32_t* stages = smem + (size_t)MOE_GR * I / 2 + (size_t)WARPS * MOE_GR * 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemmG<KB, MOE_ST, WARPS> rg(a.tr[e * 3 + 2], H / 16, cb, I, stages);
    rg.issue_first();
    const __half* suh = a.su[e * 3 + 2];
    for (int r = 0; r < R; ++r) {
        const __nv_bfloat16* yg = YGU + (size_t)grp[2 + r] * 2 * I;
        const __nv_bfloat16* yu = yg + I;
        for (int blk = warp; blk * 128 < I; blk += WARPS) {
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
            *reinterpret_cast<uint2*>(xh + (size_t)r * I + k) = o;
        }
    }
    __syncthreads();
    rg.run_rows(xh, I, R, red);
    const int n = cb * 128 + 4 * lane;
    const __half* svh = a.sv[e * 3 + 2];
    const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
    for (int r = warp; r < R; r += WARPS) {
        float v0, v1, v2, v3;
        red_rot_row<WARPS>(red, r, lane, v0, v1, v2, v3);
        *reinterpret_cast<float4*>(YD + (size_t)grp[2 + r] * H + n) = make_float4(
            v0 * RSQRT128 * __low2float(s01), v1 * RSQRT128 * __high2float(s01),
            v2 * RSQRT128 * __low2float(s23), v3 * RSQRT128 * __high2float(s23));
    }
}

// grid (H/128, P): blockIdx.y -> group y; YD [P, H] fp32
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) moe_down_group_kernel(const __nv_bfloat16* __restrict__ YGU,
        const int* __restrict__ G, const int* __restrict__ NG, MoEArgs a, float* __restrict__ YD, int H, int I) {
    extern __shared__ uint32_t smem[];
    pdl_wait();
    pdl_trigger();
    if ((int)blockIdx.y >= NG[0]) return;
    const int* grp = G + (size_t)blockIdx.y * (2 + MOE_GR);
    const int cb = blockIdx.x;
    switch (a.kb[grp[0] * 3 + 2]) {
        case 2: moe_down_group_body<2, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 3: moe_down_group_body<3, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 4: moe_down_group_body<4, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 5: moe_down_group_body<5, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 6: moe_down_group_body<6, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 8: moe_down_group_body<8, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 7: moe_down_group_body<7, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 15: moe_down_group_body<15, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 25: moe_down_group_body<25, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        case 35: moe_down_group_body<35, ST, WARPS>(YGU, grp, cb, a, YD, H, I, smem); break;
        default: break;
    }
}

// grid (H/256, M), 256 threads: out[m, c] = bf16(sum over slots in ascending expert id of w * y)
__global__ void __launch_bounds__(256) moe_combine_kernel(const float* __restrict__ YD,
        const int* __restrict__ IDS, const float* __restrict__ WTS, int S, __nv_bfloat16* __restrict__ OUT,
        int ldo, int H) {
    pdl_wait();
    pdl_trigger();
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

// ------------------------------------------------------------------------------ norms, inject
// a fixed-order block sum (up to 1024 threads): thread-strided, the warp's xor tree, the warps in order
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) t += sh[w];
    return t;
}

template <typename TN> __device__ __forceinline__ float wf(const TN* w, int i);
template <> __device__ __forceinline__ float wf(const __half* w, int i) { return __half2float(w[i]); }
template <> __device__ __forceinline__ float wf(const __nv_bfloat16* w, int i) { return __bfloat162float(w[i]); }

// the reference's grouped RMS norm with weight 1 + w over one (row, group): bf16(x r (1 + w)),
// r = rsqrt(sum(x^2) / G + eps), x in fp32. grid (M, groups), 1024 threads, G <= 4096.
template <typename TN>
__global__ void __launch_bounds__(1024) hc_norm_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const TN* __restrict__ Wn, __nv_bfloat16* __restrict__ Y, int ldy, int G, float eps) {
    __shared__ float sh[32];
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.x, g = blockIdx.y;
    const __nv_bfloat16* x = X + (size_t)m * ldx + g * G;
    float v[4];
    float ss = 0.f;
    int c = 0;
    for (int i = threadIdx.x; i < G; i += blockDim.x, ++c) { v[c] = __bfloat162float(x[i]); ss = fmaf(v[c], v[c], ss); }
    const float r = rsqrtf(block_sum(ss, sh) / (float)G + eps);
    c = 0;
    for (int i = threadIdx.x; i < G; i += blockDim.x, ++c)
        Y[(size_t)m * ldy + g * G + i] = __float2bfloat16_rn(v[c] * r * (1.0f + wf(Wn, g * G + i)));
}

// the inject of one (row, stream) and the next site's norm of that stream (one group = one stream):
// inj = 2 bf16(sigmoid(bf16(bf16(d) / hc))), h = bf16(h0 + bf16(out inj)), n = norm(h) (1 + w)
template <typename TN>
__global__ void __launch_bounds__(1024) inject_norm_kernel(const __nv_bfloat16* __restrict__ H0, int ldh,
        const __nv_bfloat16* __restrict__ OUT, int ldo, const float* __restrict__ D, int ldd, int R,
        const TN* __restrict__ Wn, __nv_bfloat16* __restrict__ HN, __nv_bfloat16* __restrict__ NN,
        int G, float inv_hc, float eps) {
    __shared__ float sh[32];
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.x, s = blockIdx.y;
    const float inj = 2.0f * bfr(sigm(bfr(bfr(D[(size_t)m * ldd + R + s]) * inv_hc)));
    const __nv_bfloat16* h0 = H0 + (size_t)m * ldh + s * G;
    const __nv_bfloat16* o = OUT + (size_t)m * ldo;
    float v[4];
    float ss = 0.f;
    int c = 0;
    for (int i = threadIdx.x; i < G; i += blockDim.x, ++c) {
        v[c] = bfr(__bfloat162float(h0[i]) + bfr(__bfloat162float(o[i]) * inj));
        HN[(size_t)m * ldh + s * G + i] = __float2bfloat16_rn(v[c]);
        ss = fmaf(v[c], v[c], ss);
    }
    if (Wn == nullptr) return;
    const float r = rsqrtf(block_sum(ss, sh) / (float)G + eps);
    c = 0;
    for (int i = threadIdx.x; i < G; i += blockDim.x, ++c)
        NN[(size_t)m * ldh + s * G + i] = __float2bfloat16_rn(v[c] * r * (1.0f + wf(Wn, s * G + i)));
}

// The MoE combine folded into the inject of the MLP site: the CTA of (row m, stream s) forms the
// row's expert sum for every column (moe_combine_kernel's arithmetic: slots in ascending expert id,
// fp32 fma, one rounding) into shared memory, then inject_norm's arithmetic for its stream. The
// four streams' CTAs each form the whole sum (from L2); the bits are moe_combine's.
template <typename TN>
__global__ void __launch_bounds__(1024) combine_inject_norm_kernel(const __nv_bfloat16* __restrict__ H0, int ldh,
        const float* __restrict__ YD, const int* __restrict__ IDS, const float* __restrict__ WTS, int S,
        const float* __restrict__ D, int ldd, int R, const TN* __restrict__ Wn, __nv_bfloat16* __restrict__ HN,
        __nv_bfloat16* __restrict__ NN, int G, float inv_hc, float eps) {
    __shared__ float sh[32];
    __shared__ int ord[64];
    __shared__ __nv_bfloat16 o[4096];
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.x, s = blockIdx.y;
    if (threadIdx.x < S) {
        const int e = IDS[m * S + threadIdx.x];
        int r = 0;
        for (int q = 0; q < S; ++q) r += IDS[m * S + q] < e;
        ord[r] = threadIdx.x;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < G; c += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < S; ++r) {
            const int q = ord[r];
            acc = fmaf(WTS[m * S + q], YD[(size_t)(m * S + q) * G + c], acc);
        }
        o[c] = __float2bfloat16_rn(acc);
    }
    __syncthreads();
    const float inj = 2.0f * bfr(sigm(bfr(bfr(D[(size_t)m * ldd + R + s]) * inv_hc)));
    const __nv_bfloat16* h0 = H0 + (size_t)m * ldh + s * G;
    float v[4];
    float ss = 0.f;
    int c = 0;
    for (int i = threadIdx.x; i < G; i += blockDim.x, ++c) {
        v[c] = bfr(__bfloat162float(h0[i]) + bfr(__bfloat162float(o[i]) * inj));
        HN[(size_t)m * ldh + s * G + i] = __float2bfloat16_rn(v[c]);
        ss = fmaf(v[c], v[c], ss);
    }
    if (Wn == nullptr) return;
    const float r = rsqrtf(block_sum(ss, sh) / (float)G + eps);
    c = 0;
    for (int i = threadIdx.x; i < G; i += blockDim.x, ++c)
        NN[(size_t)m * ldh + s * G + i] = __float2bfloat16_rn(v[c] * r * (1.0f + wf(Wn, s * G + i)));
}

// the GDN output norm of one head row (D = 128, a warp): bf16(bf16(w bf16(o r)) act(z)), w plain
__global__ void __launch_bounds__(256) gated_norm_kernel(const __nv_bfloat16* __restrict__ O,
        const __nv_bfloat16* __restrict__ Z, const __nv_bfloat16* __restrict__ Wn, __nv_bfloat16* __restrict__ Y,
        int rows, float eps, int sig) {
    pdl_wait();
    pdl_trigger();
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    float v[4], ss = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[e] = __bfloat162float(O[(size_t)r * 128 + lane * 4 + e]); ss = fmaf(v[e], v[e], ss); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rr = rsqrtf(ss / 128.0f + eps);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane * 4 + e;
        const float hn = bfr(__bfloat162float(Wn[i]) * bfr(v[e] * rr));
        const float z = __bfloat162float(Z[(size_t)r * 128 + i]);
        const float a = sig ? sigm(z) : z / (1.0f + ::expf(-z));
        Y[(size_t)r * 128 + i] = __float2bfloat16_rn(hn * a);
    }
}

// ------------------------------------------------------------------------------- dense EXL3
// One row of an EXL3 projection (or a group: members side by side in N, each with its own suh)
// over one 128-column block and one cluster rank's slice of K: the CTA rotates the row's input
// over its K slice with the block's member suh (rot_in_kernel's arithmetic) while its first trellis
// stages stream, runs RowGemm, and rank 0 adds the ranks' warp partials (rank-major, warp order)
// through distributed shared memory, then the output rotation and svh. grid.x = ((cb M) + row) S
// + rank, cluster (S): the ranks of one (block, row) are one cluster, the rows of one block are
// adjacent clusters (the second reads the trellis from L2).
template <int KB, int ST, int WARPS, bool OF32>
__device__ __forceinline__ void dense_body(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint32_t* __restrict__ T, int ldt, const __half* __restrict__ SUH, const __half* __restrict__ SVH,
        void* __restrict__ Yv, int ldy, int M, int Kin, int N, int b1, int b2, int S, uint32_t* smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int rank = blockIdx.x % S, t2 = blockIdx.x / S;
    const int row = t2 % M, cb = t2 / M;
    const int member = (cb >= b1) + (cb >= b2);
    const int KT = Kin / 16, per_r = KT / S;
    const int ka = rank * per_r, kb = ka + per_r;
    __half* xh = reinterpret_cast<__half*>(smem);                          // [per_r 16]
    float* red = reinterpret_cast<float*>(smem + per_r * 8);               // [WARPS][128]
    RowGemm<KB, ST, WARPS> rg(T, ldt, cb, ka, kb, smem + per_r * 8 + WARPS * 128);
    rg.issue_first();                                  // the weights do not depend on the predecessor
    pdl_wait();
    pdl_trigger();
    const __half* suh = SUH + (size_t)member * Kin;
    for (int blk = ka / 8 + warp; blk < kb / 8; blk += WARPS) {
        const int k = blk * 128 + 4 * lane;
        const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)row * ldx + k);
        const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
        float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
        float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
        v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
        wht128(v0, v1, v2, v3, lane);
        __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
        __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xh + (k - ka * 16)) = o;
    }
    __syncthreads();
    rg.run(xh - ka * 16, red);
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    if (S > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        if (rank == 0 && warp == 0) {
            const int c = 4 * lane;
            for (int r = 0; r < S; ++r) {
                const float* rr = r == 0 ? red : cluster.map_shared_rank(red, r);
                for (int w = 0; w < WARPS; ++w) {
                    if (r == 0 && w == 0) { v0 = rr[c]; v1 = rr[c + 1]; v2 = rr[c + 2]; v3 = rr[c + 3]; }
                    else { v0 += rr[w * 128 + c]; v1 += rr[w * 128 + c + 1]; v2 += rr[w * 128 + c + 2]; v3 += rr[w * 128 + c + 3]; }
                }
            }
            wht128(v0, v1, v2, v3, lane);
        }
        cluster.sync();
        if (rank != 0) return;
    } else if (warp == 0) {
        red_rot<WARPS>(red, lane, v0, v1, v2, v3);
    }
    if (warp == 0) {
        const int n = cb * 128 + 4 * lane;
        const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
        v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
        if constexpr (OF32) {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(Yv) + (size_t)row * ldy + n) = make_float4(v0, v1, v2, v3);
        } else {
            __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
            uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(Yv) + (size_t)row * ldy + n) = o;
        }
    }
}

template <int KB, int ST, int WARPS, bool OF32>
__global__ void __launch_bounds__(256) dense_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint32_t* __restrict__ T, int ldt, const __half* __restrict__ SUH, const __half* __restrict__ SVH,
        void* __restrict__ Yv, int ldy, int M, int Kin, int N, int b1, int b2, int S) {
    extern __shared__ uint32_t smem[];
    dense_body<KB, ST, WARPS, OF32>(X, ldx, T, ldt, SUH, SVH, Yv, ldy, M, Kin, N, b1, b2, S, smem);
}

// SPD-125: the same projection for M > 1 rows with each trellis tile decoded once for 16 rows. First
// every (member, row)'s input rotated once (dense_rot_kernel: dense_body's arithmetic, into XH [G, M,
// Kin] fp16), then one CTA a (128-column block, group of 16 rows, rank): rows 0..7 of the group in the
// mma's lane groups of a0/a2, rows 8..15 in a1/a3. A row's sums are the products the one-row CTA forms
// (its K slices the same warps', in the same K order; an mma's output element reads only its own
// row of A), its warp partials added in the same warp and rank order, then the same output rotation
// and svh, so every output byte is the one-row call's.

// grid (Kin/128/4, M, G), 4 warps: one warp a 128-block of one (member, row)
__global__ void __launch_bounds__(128) dense_rot_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Kin) {
    pdl_wait();
    pdl_trigger();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int row = blockIdx.y, member = blockIdx.z;
    const int k = blk * 128 + 4 * lane;
    const __half* suh = SUH + (size_t)member * Kin;
    const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)row * ldx + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
    float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
    float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + ((size_t)member * M + row) * Kin + k) = o;
}

// grid.x = ((cb MG) + group) S + rank, cluster (S); RED [WARPS][R][128] overlays the stages after the sums
template <int KB, int ST, int WARPS, bool OF32>
__global__ void __launch_bounds__(256) dense_rows_kernel(const __half* __restrict__ XH,
        const uint32_t* __restrict__ T, int ldt, const __half* __restrict__ SVH,
        void* __restrict__ Yv, int ldy, int M, int Kin, int N, int b1, int b2, int S) {
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int MG = (M + 15) / 16;
    const int rank = blockIdx.x % S, t2 = blockIdx.x / S;
    const int grp = t2 % MG, cb = t2 / MG;
    const int member = (cb >= b1) + (cb >= b2);
    const int KT = Kin / 16, per_r = KT / S;
    const int ka = rank * per_r, kb = ka + per_r;
    const int r0 = grp * 16, R = min(16, M - r0);
    RowGemm<KB, ST, WARPS> rg(T, ldt, cb, ka, kb, smem);
    rg.issue_first();                                  // the weights do not depend on the predecessor
    pdl_wait();
    pdl_trigger();
    // the A operands of K tile kt from XH (rows g and 8 + g of the group, K 2t.. and 2t + 8..), a tile ahead
    const __half* xh = XH + ((size_t)member * M + r0) * Kin;
    const bool lo = g < R, hi = g + 8 < R;
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(xh + (size_t)(lo ? g : 0) * Kin + 2 * t);
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(xh + (size_t)(hi ? g + 8 : 0) * Kin + 2 * t);
    uint32_t xn[4] = {0u, 0u, 0u, 0u};
    auto load_x = [&](int kt) {
        xn[0] = lo ? __ldg(pa + kt * 8) : 0u;
        xn[1] = hi ? __ldg(pb + kt * 8) : 0u;
        xn[2] = lo ? __ldg(pa + kt * 8 + 4) : 0u;
        xn[3] = hi ? __ldg(pb + kt * 8 + 4) : 0u;
    };
    float acc[8][2][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[j][h][c] = 0.f;
    if (rg.k0 < rg.k1) load_x(rg.k0);
    for (int kt = rg.k0; kt < rg.k1; ++kt) {
        const int i = kt - rg.k0;
        const uint32_t a0 = xn[0], a1 = xn[1], a2 = xn[2], a3 = xn[3];
        if (kt + 1 < rg.k1) load_x(kt + 1);
        rg.issue(kt + ST - 1, (i + ST - 1) % ST);
        cp_wait<ST - 1>();
        __syncwarp();
        const uint32_t* buf = rg.stage + (i % ST) * rg.SW;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            FragB f0, f1;
            dq_tile<KB>(buf + j * rg.TW, lane * 8, f0, f1);
            mma(acc[j][0], a0, a1, a2, a3, f0.r[0], f0.r[1]);
            mma(acc[j][1], a0, a1, a2, a3, f1.r[0], f1.r[1]);
        }
        __syncwarp();
    }
    cp_wait<0>();
    __syncthreads();                                   // every warp is done with the stages
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = j * 16 + h * 8 + 2 * t;
            if (lo) {
                red[((size_t)warp * R + g) * 128 + col] = acc[j][h][0];
                red[((size_t)warp * R + g) * 128 + col + 1] = acc[j][h][1];
            }
            if (hi) {
                red[((size_t)warp * R + g + 8) * 128 + col] = acc[j][h][2];
                red[((size_t)warp * R + g + 8) * 128 + col + 1] = acc[j][h][3];
            }
        }
    __syncthreads();
    cg::cluster_group cluster = cg::this_cluster();
    if (S > 1) cluster.sync();
    {
        const int c = 4 * lane, n = cb * 128 + 4 * lane;
        const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        // one warp of the cluster a row: the ranks' warp partials, rank-major and in warp order (the one-row
        // CTA's sums), a rank's WARPS partials loaded before they are added
        for (int r = rank * WARPS + warp; r < R; r += S * WARPS) {
            float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
            for (int q = 0; q < S; ++q) {
                const float* rr = S == 1 ? red : cluster.map_shared_rank(red, q);
                float4 p[WARPS];
#pragma unroll
                for (int w = 0; w < WARPS; ++w) p[w] = *reinterpret_cast<const float4*>(rr + ((size_t)w * R + r) * 128 + c);
#pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    if (q == 0 && w == 0) { v0 = p[w].x; v1 = p[w].y; v2 = p[w].z; v3 = p[w].w; }
                    else { v0 += p[w].x; v1 += p[w].y; v2 += p[w].z; v3 += p[w].w; }
                }
            }
            wht128(v0, v1, v2, v3, lane);
            v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
            v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
            const size_t m = (size_t)(r0 + r);
            if constexpr (OF32) {
                *reinterpret_cast<float4*>(reinterpret_cast<float*>(Yv) + m * ldy + n) = make_float4(v0, v1, v2, v3);
            } else {
                __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
                uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
                *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(Yv) + m * ldy + n) = o;
            }
        }
    }
    if (S > 1) cluster.sync();                         // every rank keeps its RED until the others have read it
}

// ------------------------------------------------------------------------- attention, a row's inputs
// grid (Hq + 2 Hk + 1, M), D threads: CTA h < Hq of row m normalises query head h (1 + w) and rotates
// its first ROT dims at the row's position POS[m] (the reference's rope: bf16(bf16(x c) + bf16(rot(x) s)));
// Hq..Hq+Hk-1 the same for the key heads, written into the cache at the row's slot SLOT[m] (its
// position for a decode step; its place in the block for a verify); then the value heads into the
// cache; the next CTA the indexer key; then (QSA, ENG-147) one CTA an indexer query head, normed
// (1 + w) over ihd and rotated like a query head. QKV row: per q head [q (D) | gate (D)], then k, then v.
__global__ void __launch_bounds__(256) attn_prep_kernel(const __nv_bfloat16* __restrict__ QKV, int ldq,
        const __nv_bfloat16* __restrict__ IQK, int ldi, const int64_t* __restrict__ POS, const int64_t* __restrict__ SLOT,
        const __nv_bfloat16* __restrict__ QW, const __nv_bfloat16* __restrict__ KW,
        const __nv_bfloat16* __restrict__ COS, const __nv_bfloat16* __restrict__ SIN,
        __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC, __nv_bfloat16* __restrict__ VC,
        __nv_bfloat16* __restrict__ IKC, int Hq, int Hk, int D, int ROT, int ioff, int ihd, int cap, float eps,
        const __nv_bfloat16* __restrict__ QLN, __nv_bfloat16* __restrict__ IQ, int nhi) {
    __shared__ float sh[8];
    __shared__ float nv[256];
    pdl_wait();
    pdl_trigger();
    const int b = blockIdx.x, t = threadIdx.x, m = blockIdx.y;
    const int64_t pos = POS[m], slot = SLOT[m];
    const __nv_bfloat16* qkv = QKV + (size_t)m * ldq;
    if (b > Hq + 2 * Hk) {
        const int ih = b - Hq - 2 * Hk - 1;
        const float x = t < ihd ? __bfloat162float(IQK[(size_t)m * ldi + ih * ihd + t]) : 0.f;
        const float r = rsqrtf(block_sum(x * x, sh) / (float)ihd + eps);
        const float n = t < ihd ? bfr(x * r * (1.0f + __bfloat162float(QLN[t]))) : 0.f;
        nv[t] = n;
        __syncthreads();
        if (t >= ihd) return;
        float o = n;
        if (t < ROT) {
            const float c = __bfloat162float(COS[(size_t)pos * ROT + t]), sn = __bfloat162float(SIN[(size_t)pos * ROT + t]);
            const float rh = t < ROT / 2 ? -nv[t + ROT / 2] : nv[t - ROT / 2];
            o = bfr(bfr(n * c) + bfr(rh * sn));
        }
        IQ[((size_t)m * nhi + ih) * ihd + t] = __float2bfloat16_rn(o);
        return;
    }
    if (b >= Hq + Hk) {
        if (b < Hq + 2 * Hk) {
            const int vh = b - Hq - Hk;
            VC[((size_t)vh * cap + slot) * D + t] = qkv[(size_t)Hq * 2 * D + (size_t)Hk * D + vh * D + t];
        } else if (IKC != nullptr && t < ihd) {
            IKC[(size_t)slot * ihd + t] = IQK[(size_t)m * ldi + ioff + t];
        }
        return;
    }
    const bool isq = b < Hq;
    const float x = __bfloat162float(isq ? qkv[(size_t)b * 2 * D + t] : qkv[(size_t)Hq * 2 * D + (b - Hq) * D + t]);
    const float r = rsqrtf(block_sum(x * x, sh) / (float)D + eps);
    const float n = bfr(x * r * (1.0f + __bfloat162float((isq ? QW : KW)[t])));
    nv[t] = n;
    __syncthreads();
    float o = n;
    if (t < ROT) {
        const float c = __bfloat162float(COS[(size_t)pos * ROT + t]), sn = __bfloat162float(SIN[(size_t)pos * ROT + t]);
        const float rh = t < ROT / 2 ? -nv[t + ROT / 2] : nv[t - ROT / 2];
        o = bfr(bfr(n * c) + bfr(rh * sn));
    }
    if (isq) Q[((size_t)m * Hq + b) * D + t] = __float2bfloat16_rn(o);
    else KC[((size_t)(b - Hq) * cap + slot) * D + t] = __float2bfloat16_rn(o);
}

// y[m][h D + d] = bf16(o[m][h D + d] * bf16(sigmoid(gate))), the gate beside each query head in row m of QKV
__global__ void __launch_bounds__(256) gate_mul_kernel(const __nv_bfloat16* __restrict__ O,
        const __nv_bfloat16* __restrict__ QKV, int ldq, __nv_bfloat16* __restrict__ Y, int n, int per, int D) {
    pdl_wait();
    pdl_trigger();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int m = i / per, r = i % per;
    const int h = r / D, d = r % D;
    const float g = bfr(sigm(__bfloat162float(QKV[(size_t)m * ldq + (size_t)h * 2 * D + D + d])));
    Y[i] = __float2bfloat16_rn(__bfloat162float(O[i]) * g);
}

}  // namespace

// --------------------------------------------------------------------------------- host side
static int g_pdl = 1;      // fn_set_pdl: launch with programmatic stream serialization (QWEN38_FN_PDL)
static int g_gemv_split = 1;   // fn_set_gemv_split: few-row weights on gemv_split_kernel (QWEN38_FN_GEMV_SPLIT)
static int g_gemv_nb = 0;      // fn_set_gemv_nb: SPD-231, M > 1 on gemv_split_nb / mix_up_nb (QWEN38_FN_GEMV_NB)
// fn_set_gemv_mc: SPD-239, the rows a pass and the CTAs an SM of gemv_split / mix_up for M > 1 (QWEN38_FN_GEMV_MC):
// 0 = 8 rows, no bound; 1 = 4 rows, 2 CTAs; 2 = 4 rows, 3 CTAs; 3 = 2 rows, 4 CTAs; 4 = 8 rows, 2 CTAs; 5 = 2 rows,
// 3 CTAs. A row's sums are the same chunks in the same order whatever the rows a pass (its bits)
static int g_gemv_mc = 0;
static int g_mixer_mma = 0;    // fn_set_mixer_mma: SPD-241, the mixers' int8 GEMV and mix_up on tensor cores (QWEN38_FN_MIXER_MMA)

// the tensor-core GEMV's tickets: an int a column block, zero between launches (each launch's last CTA of a
// column sets its ticket back), one buffer a device
static int* mix_tickets(c10::Device dev, int n) {
    static torch::Tensor t;
    if (!t.defined() || t.numel() < n || t.device() != dev)
        t = torch::zeros({std::max(n, 256)}, torch::TensorOptions().dtype(torch::kInt).device(dev));
    return (int*)t.data_ptr();
}

// a kernel's dynamic shared memory past 48 KiB, set once
template <typename... KArgs>
static void mix_smem_once(void (*k)(KArgs...)) {
    static std::vector<const void*> done;
    for (const void* p : done) if (p == (const void*)k) return;
    TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) == cudaSuccess,
                "mixer kernel shared memory");
    done.push_back((const void*)k);
}

template <typename... KArgs, typename... Args>
static void launch(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    TORCH_CHECK(e == cudaSuccess, "fn_kernels launch: ", cudaGetErrorString(e));
}

void fn_set_pdl(int64_t on) { g_pdl = on ? 1 : 0; }
void fn_set_gemv_split(int64_t on) { g_gemv_split = on ? 1 : 0; }
void fn_set_gemv_nb(int64_t on) { g_gemv_nb = on ? 1 : 0; }
void fn_set_gemv_mc(int64_t v) { g_gemv_mc = (int)v; }
void fn_set_mixer_mma(int64_t on) { g_mixer_mma = on ? 1 : 0; }

template <typename... KArgs, typename... Args>
static void launch_cluster(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, int S, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    at[1].id = cudaLaunchAttributeClusterDimension;
    at[1].val.clusterDim.x = S; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = S > 1 ? 2 : 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    TORCH_CHECK(e == cudaSuccess, "fn_dense launch: ", cudaGetErrorString(e));
}

template <int KB, int ST, int WARPS, bool OF32>
static void dense_launch(const torch::Tensor& x, const torch::Tensor& t, const torch::Tensor& suh,
                         const torch::Tensor& svh, torch::Tensor& y, int M, int Kin, int N, int b1, int b2,
                         int S, cudaStream_t st) {
    constexpr int TW = tile_u32(KB);
    const int per_r = Kin / 16 / S;
    const size_t smem = (size_t)(per_r * 8 + WARPS * 128) * 4 + (size_t)WARPS * ST * 8 * TW * 4;
    auto kern = dense_kernel<KB, ST, WARPS, OF32>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        attr = true;
    }
    launch_cluster(kern, dim3((N / 128) * M * S), dim3(32 * WARPS), smem, S, st,
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const uint32_t*)t.data_ptr(),
        (int)(t.stride(0) / t.stride(1)), (const __half*)suh.data_ptr(), (const __half*)svh.data_ptr(),
        (void*)y.data_ptr(), (int)y.stride(0), M, Kin, N, b1, b2, S);
}

#define DENSE_W(KB, OF32)                                                                         \
    if (warps == 4) dense_launch<KB, 3, 4, OF32>(x, t, suh, svh, y, M, Kin, N, b1, b2, S, st);    \
    else dense_launch<KB, 3, 8, OF32>(x, t, suh, svh, y, M, Kin, N, b1, b2, S, st);
#define DENSE_KB(OF32)                                                                            \
    switch (kb) {                                                                                 \
        case 2: { DENSE_W(2, OF32) break; } case 3: { DENSE_W(3, OF32) break; }                     \
        case 4: { DENSE_W(4, OF32) break; } case 5: { DENSE_W(5, OF32) break; }                     \
        case 6: { DENSE_W(6, OF32) break; } case 8: { DENSE_W(8, OF32) break; }                     \
        case 15: { DENSE_W(15, OF32) break; } case 25: { DENSE_W(25, OF32) break; }                 \
        case 35: { DENSE_W(35, OF32) break; }                                                     \
        default: TORCH_CHECK(false, "fn_dense: K 2..6 and 8, 1.5, 2.5 and 3.5");                  \
    }

void fn_dense(torch::Tensor x, torch::Tensor t, torch::Tensor suh, torch::Tensor svh, torch::Tensor y,
              int64_t b1, int64_t b2, int64_t S, int64_t warps) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3 && t.stride(2) == 1 && t.stride(1) == t.size(2),
                "trellis [Kin/16, N/16, 16 K] with dense tiles (a group member may be a view)");
    // a tile 16 K int16 wide, or 16 K + 8 for a half-integer rate (the kernels' 10 K + 5)
    const int kb = t.size(2) % 16 ? 10 * (int)(t.size(2) / 16) + 5 : (int)(t.size(2) / 16);
    const int M = x.size(0), Kin = x.size(1), N = t.size(1) * 16;
    TORCH_CHECK(t.size(0) * 16 == Kin && N % 128 == 0 && Kin % 128 == 0, "shapes");
    TORCH_CHECK(S >= 1 && S <= 8 && (Kin / 16) % (8 * S) == 0, "a rank's K slice must be whole 128 blocks");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.is_contiguous() && suh.size(-1) == Kin, "suh fp16 [G, Kin]");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh");
    TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "y [M, N]");
    TORCH_CHECK(M >= 1 && M <= 32 && (warps == 4 || warps == 8), "rows 1..32, warps 4 or 8");
    if (y.dtype() == torch::kFloat) { DENSE_KB(true) } else { TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32"); DENSE_KB(false) }
}

template <int KB, int ST, int WARPS, bool OF32>
static void dense_rows_launch(const torch::Tensor& xh, const torch::Tensor& t, const torch::Tensor& svh,
                              torch::Tensor& y, int M, int Kin, int N, int b1, int b2, int S, cudaStream_t st) {
    constexpr int SW = 8 * tile_u32(KB);
    const size_t smem = std::max((size_t)WARPS * ST * SW * 4, (size_t)WARPS * std::min(16, M) * 128 * 4);
    auto kern = dense_rows_kernel<KB, ST, WARPS, OF32>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        attr = true;
    }
    launch_cluster(kern, dim3((N / 128) * ((M + 15) / 16) * S), dim3(32 * WARPS), smem, S, st,
        (const __half*)xh.data_ptr(), (const uint32_t*)t.data_ptr(), (int)(t.stride(0) / t.stride(1)),
        (const __half*)svh.data_ptr(), (void*)y.data_ptr(), (int)y.stride(0), M, Kin, N, b1, b2, S);
}

#define DENSE_ROWS_W(KB, OF32)                                                                    \
    if (warps == 4) dense_rows_launch<KB, 3, 4, OF32>(xh, t, svh, y, M, Kin, N, b1, b2, S, st);   \
    else dense_rows_launch<KB, 3, 8, OF32>(xh, t, svh, y, M, Kin, N, b1, b2, S, st);
#define DENSE_ROWS_KB(OF32)                                                                       \
    switch (kb) {                                                                                 \
        case 2: { DENSE_ROWS_W(2, OF32) break; } case 3: { DENSE_ROWS_W(3, OF32) break; }           \
        case 4: { DENSE_ROWS_W(4, OF32) break; } case 5: { DENSE_ROWS_W(5, OF32) break; }           \
        case 6: { DENSE_ROWS_W(6, OF32) break; } case 8: { DENSE_ROWS_W(8, OF32) break; }           \
        case 15: { DENSE_ROWS_W(15, OF32) break; } case 25: { DENSE_ROWS_W(25, OF32) break; }       \
        case 35: { DENSE_ROWS_W(35, OF32) break; }                                                \
        default: TORCH_CHECK(false, "fn_dense_rows: K 2..6 and 8, 1.5, 2.5 and 3.5");             \
    }

// xh: fp16 [G, M, Kin] workspace for the rotated inputs (G = suh's members)
void fn_dense_rows(torch::Tensor x, torch::Tensor t, torch::Tensor suh, torch::Tensor svh, torch::Tensor xh,
                   torch::Tensor y, int64_t b1, int64_t b2, int64_t S, int64_t warps) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3 && t.stride(2) == 1 && t.stride(1) == t.size(2),
                "trellis [Kin/16, N/16, 16 K] with dense tiles (a group member may be a view)");
    const int kb = t.size(2) % 16 ? 10 * (int)(t.size(2) / 16) + 5 : (int)(t.size(2) / 16);
    const int M = x.size(0), Kin = x.size(1), N = t.size(1) * 16;
    const int G = suh.dim() == 2 ? suh.size(0) : 1;
    TORCH_CHECK(t.size(0) * 16 == Kin && N % 128 == 0 && Kin % 128 == 0, "shapes");
    TORCH_CHECK(S >= 1 && S <= 8 && (Kin / 16) % (8 * S) == 0, "a rank's K slice must be whole 128 blocks");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.is_contiguous() && suh.size(-1) == Kin, "suh fp16 [G, Kin]");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.numel() >= (int64_t)G * M * Kin, "xh fp16 [G, M, Kin]");
    TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "y [M, N]");
    TORCH_CHECK(M >= 1 && M <= 32 && (warps == 4 || warps == 8), "rows 1..32, warps 4 or 8");
    launch(dense_rot_kernel, dim3((Kin + 511) / 512, M, G), dim3(128), 0, st, (const __nv_bfloat16*)x.data_ptr(),
        (int)x.stride(0), (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Kin);
    if (y.dtype() == torch::kFloat) { DENSE_ROWS_KB(true) } else { TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32"); DENSE_ROWS_KB(false) }
}

void fn_attn_prep(torch::Tensor qkv, c10::optional<torch::Tensor> iqk, torch::Tensor pos, torch::Tensor slot,
                  torch::Tensor qw, torch::Tensor kw, torch::Tensor cos, torch::Tensor sin, torch::Tensor q, torch::Tensor kc,
                  torch::Tensor vc, c10::optional<torch::Tensor> ikc, int64_t hq, int64_t hk, int64_t ioff, double eps,
                  c10::optional<torch::Tensor> qln, c10::optional<torch::Tensor> iq) {
    const at::cuda::OptionalCUDAGuard guard(qkv.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int D = q.size(2), ROT = cos.size(1), cap = kc.size(1), M = qkv.size(0);
    TORCH_CHECK(D == 256 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(1) == hq * 2 * D + 2 * hk * D, "qkv [M, Hq 2D + 2 Hk D], D 256");
    TORCH_CHECK(q.is_contiguous() && q.size(0) == M && kc.is_contiguous() && vc.is_contiguous() && kc.size(0) == hk && kc.size(2) == D, "q [M, Hq, D], caches [Hk, cap, D]");
    TORCH_CHECK(pos.dtype() == torch::kLong && slot.dtype() == torch::kLong && pos.numel() >= M && slot.numel() >= M, "pos, slot int64 [M]");
    TORCH_CHECK(cos.dtype() == torch::kBFloat16 && ROT % 2 == 0 && ROT <= D, "cos/sin bf16");
    const int ihd = ikc.has_value() ? ikc->size(1) : 0;
    TORCH_CHECK(ihd <= 256 && (!iqk.has_value() || (iqk->dim() == 2 && iqk->stride(1) == 1)), "indexer head, iqk [M, n]");
    const int nhi = iq.has_value() ? iq->size(1) : 0;
    TORCH_CHECK(!iq.has_value() || (qln.has_value() && iq->is_contiguous() && iq->size(0) == M && iq->size(2) == ihd
                && iqk.has_value()), "iq [M, heads, ihd] with q_layernorm and iqk");
    launch(attn_prep_kernel, dim3(hq + 2 * hk + 1 + nhi, M), dim3(D), 0, st, (const __nv_bfloat16*)qkv.data_ptr(), (int)qkv.stride(0),
        iqk.has_value() ? (const __nv_bfloat16*)iqk->data_ptr() : (const __nv_bfloat16*)nullptr, iqk.has_value() ? (int)iqk->stride(0) : 0,
        (const int64_t*)pos.data_ptr(), (const int64_t*)slot.data_ptr(),
        (const __nv_bfloat16*)qw.data_ptr(), (const __nv_bfloat16*)kw.data_ptr(), (const __nv_bfloat16*)cos.data_ptr(),
        (const __nv_bfloat16*)sin.data_ptr(), (__nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)kc.data_ptr(),
        (__nv_bfloat16*)vc.data_ptr(), ikc.has_value() ? (__nv_bfloat16*)ikc->data_ptr() : (__nv_bfloat16*)nullptr,
        (int)hq, (int)hk, D, ROT, (int)ioff, ihd, cap, (float)eps,
        qln.has_value() ? (const __nv_bfloat16*)qln->data_ptr() : (const __nv_bfloat16*)nullptr,
        iq.has_value() ? (__nv_bfloat16*)iq->data_ptr() : (__nv_bfloat16*)nullptr, nhi);
}

void fn_gate_mul(torch::Tensor o, torch::Tensor qkv, torch::Tensor y, int64_t D) {
    const at::cuda::OptionalCUDAGuard guard(o.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(o.is_contiguous() && y.is_contiguous() && qkv.dim() == 2 && qkv.stride(1) == 1, "o, y contiguous; qkv [M, n]");
    const int n = o.numel(), M = qkv.size(0);
    TORCH_CHECK(n % M == 0, "rows");
    launch(gate_mul_kernel, dim3((n + 255) / 256), dim3(256), 0, st, (const __nv_bfloat16*)o.data_ptr(),
        (const __nv_bfloat16*)qkv.data_ptr(), (int)qkv.stride(0), (__nv_bfloat16*)y.data_ptr(), n, n / M, (int)D);
}

static int lpr_for(int K) { return K >= 2048 ? 32 : (K >= 512 ? 16 : 8); }

// warps a CTA: as many as keep at least two CTAs an SM (a row group is one warp's, so the sums do
// not depend on it)
static int gemv_warps(int N, int LPR) {
    const int warps_needed = (N + 32 / LPR - 1) / (32 / LPR);
    return warps_needed >= 8 * 96 ? 8 : (warps_needed >= 4 * 96 ? 4 : 2);
}

#define GEMV_LAUNCH(TW, LPR, MC)                                                                   \
    launch(gemv_kernel<TW, LPR, MC, TY>,                                                           \
        dim3((N + gemv_warps(N, LPR) * (32 / LPR) - 1) / (gemv_warps(N, LPR) * (32 / LPR))),       \
        dim3(32 * gemv_warps(N, LPR)), 0, st,                                                      \
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const TW*)w.data_ptr(),             \
        s_ptr, (TY*)y.data_ptr(), (int)y.stride(0), M, N, K, G)

// K slices of a gemv_split launch for a weight of N rows and K columns (1: gemv_kernel): as many warps
// as the board holds at once (48 SMs x 64) while a slice keeps at least 256 columns
static int gemv_split_ks(int N, int K, int vec) {
    if (N > 640) return 1;
    int ks = 1;
    while (ks < 8 && N * ks * 2 <= 3072 && K % (ks * 2 * vec) == 0 && K / (ks * 2) >= 256) ks *= 2;
    return ks;
}

template <typename TW, typename TY>
static void gemv_dispatch(const torch::Tensor& x, const torch::Tensor& w, const float* s_ptr,
                          torch::Tensor& y, int M, int N, int K, cudaStream_t st, int G = 0) {
    if (g_mixer_mma && std::is_same<TW, int8_t>::value && std::is_same<TY, float>::value && (G == 64 || G == 128)
        && K % MIX_KSLICE == 0 && K / MIX_KSLICE <= MIX_SLICES && x.stride(0) % 8 == 0) {
        // SPD-241: the tensor-core GEMV at every M (64 rows a launch: a row's sums never depend on the call)
        const int KS = K / MIX_KSLICE, NB = (N + 31) / 32;
        auto P = torch::empty({KS, std::min(M, 64), N}, x.options().dtype(torch::kFloat));
        auto cnt = mix_tickets(x.device(), NB);
        auto k = G == 64 ? mixer_mma_gemv_kernel<1> : mixer_mma_gemv_kernel<2>;
        mix_smem_once(k);
        for (int r0 = 0; r0 < M; r0 += 64) {
            const int mm = std::min(64, M - r0);
            launch(k, dim3(NB, KS), dim3(128), (size_t)mm * MIX_LDS * 2, st,
                   (const __nv_bfloat16*)x.data_ptr() + (size_t)r0 * x.stride(0), (int)x.stride(0),
                   (const int8_t*)w.data_ptr(), s_ptr, (float*)P.data_ptr(), mm, N, K, G,
                   (float*)y.data_ptr() + (size_t)r0 * y.stride(0), (int)y.stride(0), cnt);
        }
        return;
    }
    int ks = g_gemv_split ? gemv_split_ks(N, K, WVec<TW>::VEC) : 1;
    while (G && ks > 1 && (K / ks) % G) ks /= 2;           // a slice holds whole scale groups
    // SPD-231: int8 weights of at least 256 rows (fp16 ones and narrow ones were slower on the board)
    if (ks > 1 && g_gemv_nb && M > 1 && N >= 256 && std::is_same<TW, int8_t>::value) {
        launch(gemv_split_nb_kernel<TW, 8, 4, TY>, dim3((N + 3) / 4), dim3(32 * ks), 0, st,
               (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const TW*)w.data_ptr(), s_ptr,
               (TY*)y.data_ptr(), (int)y.stride(0), M, N, K, G);
        return;
    }
    if (ks > 1) {
        auto kern = M == 1 ? gemv_split_kernel<TW, 1, TY> : gemv_split_kernel<TW, 8, TY>;
        if (M > 1 && g_gemv_mc == 1) kern = gemv_split_kernel<TW, 4, TY, 2>;
        if (M > 1 && g_gemv_mc == 2) kern = gemv_split_kernel<TW, 4, TY, 3>;
        if (M > 1 && g_gemv_mc == 3) kern = gemv_split_kernel<TW, 2, TY, 4>;
        if (M > 1 && g_gemv_mc == 4) kern = gemv_split_kernel<TW, 8, TY, 2>;
        if (M > 1 && g_gemv_mc == 5) kern = gemv_split_kernel<TW, 2, TY, 3>;
        launch(kern, dim3(N), dim3(32 * ks), 0, st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
               (const TW*)w.data_ptr(), s_ptr, (TY*)y.data_ptr(), (int)y.stride(0), M, N, K, G);
        return;
    }
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
    TORCH_CHECK((y.dtype() == torch::kFloat || y.dtype() == torch::kBFloat16) && y.stride(1) == 1, "y fp32 or bf16 [M, N]");
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N, "shapes");
    const bool f32 = y.dtype() == torch::kFloat;
    if (w.dtype() == torch::kInt8) {
        // fp32 scales: one a row [N], or one a group of G weights [N, K / G] (G a multiple of 16)
        TORCH_CHECK(s.has_value() && s->dtype() == torch::kFloat && s->is_contiguous() && s->numel() % N == 0,
                    "int8 rows need fp32 scales [N] or [N, K / G]");
        TORCH_CHECK(K % 16 == 0, "K % 16");
        const int ng = s->numel() / N, G = ng == 1 ? 0 : K / ng;
        TORCH_CHECK(!G || (K % ng == 0 && G % 16 == 0), "scale groups: K / groups a multiple of 16");
        const float* sp = (const float*)s->data_ptr();
        if (f32) gemv_dispatch<int8_t, float>(x, w, sp, y, M, N, K, st, G);
        else gemv_dispatch<int8_t, __nv_bfloat16>(x, w, sp, y, M, N, K, st, G);
    } else {
        TORCH_CHECK(w.dtype() == torch::kHalf && K % 8 == 0, "w int8 or fp16, K % 8");
        if (f32) gemv_dispatch<__half, float>(x, w, nullptr, y, M, N, K, st);
        else gemv_dispatch<__half, __nv_bfloat16>(x, w, nullptr, y, M, N, K, st);
    }
}

void fn_hc_norm(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t G, double eps) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), N = x.size(1);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && y.dtype() == torch::kBFloat16 && y.stride(1) == 1, "bf16 rows");
    TORCH_CHECK(N % G == 0 && G <= 1024 * 4 && w.numel() == N && w.is_contiguous(), "groups");
    dim3 grid(M, N / G);
    if (w.dtype() == torch::kHalf)
        launch(hc_norm_kernel<__half>, grid, dim3(1024), 0, st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
            (const __half*)w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), (int)G, (float)eps);
    else {
        TORCH_CHECK(w.dtype() == torch::kBFloat16, "norm weight fp16 or bf16");
        launch(hc_norm_kernel<__nv_bfloat16>, grid, dim3(1024), 0, st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
            (const __nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), (int)G, (float)eps);
    }
}

void fn_inject_norm(torch::Tensor h0, torch::Tensor out, torch::Tensor d, int64_t R, c10::optional<torch::Tensor> w,
                    torch::Tensor hn, torch::Tensor nn, int64_t hc, double eps) {
    const at::cuda::OptionalCUDAGuard guard(h0.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = h0.size(0), N = h0.size(1), G = N / hc;
    TORCH_CHECK(h0.dtype() == torch::kBFloat16 && h0.is_contiguous() && hn.is_contiguous() && nn.is_contiguous(), "h0, hn, nn contiguous bf16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.stride(1) == 1 && out.size(1) == G, "out bf16 [M, G]");
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1 && d.size(1) >= R + hc, "d fp32 [M, R + hc]");
    TORCH_CHECK(G <= 1024 * 4, "group");
    dim3 grid(M, hc);
    const float inv = 1.0f / (float)hc;
    if (!w.has_value() || w->dtype() == torch::kHalf)
        launch(inject_norm_kernel<__half>, grid, dim3(1024), 0, st, (const __nv_bfloat16*)h0.data_ptr(), N,
            (const __nv_bfloat16*)out.data_ptr(), (int)out.stride(0), (const float*)d.data_ptr(), (int)d.stride(0), (int)R,
            w.has_value() ? (const __half*)w->data_ptr() : (const __half*)nullptr, (__nv_bfloat16*)hn.data_ptr(),
            (__nv_bfloat16*)nn.data_ptr(), G, inv, (float)eps);
    else
        launch(inject_norm_kernel<__nv_bfloat16>, grid, dim3(1024), 0, st, (const __nv_bfloat16*)h0.data_ptr(), N,
            (const __nv_bfloat16*)out.data_ptr(), (int)out.stride(0), (const float*)d.data_ptr(), (int)d.stride(0), (int)R,
            (const __nv_bfloat16*)w->data_ptr(), (__nv_bfloat16*)hn.data_ptr(), (__nv_bfloat16*)nn.data_ptr(), G, inv, (float)eps);
}

void fn_combine_inject_norm(torch::Tensor h0, torch::Tensor yd, torch::Tensor ids, torch::Tensor wts, torch::Tensor d,
                           int64_t R, c10::optional<torch::Tensor> w, torch::Tensor hn, torch::Tensor nn, int64_t hc, double eps) {
    const at::cuda::OptionalCUDAGuard guard(h0.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = h0.size(0), N = h0.size(1), G = N / hc, S = ids.numel() / M;
    TORCH_CHECK(h0.dtype() == torch::kBFloat16 && h0.is_contiguous() && hn.is_contiguous() && nn.is_contiguous(), "bf16 rows");
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == M * S && yd.size(1) == G, "yd [M S, G]");
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1 && d.size(1) >= R + hc, "d fp32 [M, R + hc]");
    TORCH_CHECK(G <= 4096 && S <= 64, "group, slots");
    dim3 grid(M, hc);
    const float inv = 1.0f / (float)hc;
    if (!w.has_value() || w->dtype() == torch::kHalf)
        launch(combine_inject_norm_kernel<__half>, grid, dim3(1024), 0, st, (const __nv_bfloat16*)h0.data_ptr(), N,
            (const float*)yd.data_ptr(), (const int*)ids.data_ptr(), (const float*)wts.data_ptr(), S,
            (const float*)d.data_ptr(), (int)d.stride(0), (int)R,
            w.has_value() ? (const __half*)w->data_ptr() : (const __half*)nullptr, (__nv_bfloat16*)hn.data_ptr(),
            (__nv_bfloat16*)nn.data_ptr(), G, inv, (float)eps);
    else
        launch(combine_inject_norm_kernel<__nv_bfloat16>, grid, dim3(1024), 0, st, (const __nv_bfloat16*)h0.data_ptr(), N,
            (const float*)yd.data_ptr(), (const int*)ids.data_ptr(), (const float*)wts.data_ptr(), S,
            (const float*)d.data_ptr(), (int)d.stride(0), (int)R, (const __nv_bfloat16*)w->data_ptr(),
            (__nv_bfloat16*)hn.data_ptr(), (__nv_bfloat16*)nn.data_ptr(), G, inv, (float)eps);
}

void fn_gated_norm(torch::Tensor o, torch::Tensor z, torch::Tensor w, torch::Tensor y, double eps, bool sig) {
    const at::cuda::OptionalCUDAGuard guard(o.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(o.dtype() == torch::kBFloat16 && z.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16, "bf16");
    TORCH_CHECK(o.is_contiguous() && z.is_contiguous() && y.is_contiguous() && w.numel() == 128 && o.size(-1) == 128, "head rows of 128");
    const int rows = o.numel() / 128;
    launch(gated_norm_kernel, dim3((rows + 7) / 8), dim3(256), 0, st, (const __nv_bfloat16*)o.data_ptr(),
        (const __nv_bfloat16*)z.data_ptr(), (const __nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), rows,
        (float)eps, sig ? 1 : 0);
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
    if (g_mixer_mma && w.dtype() == torch::kInt8 && R == MIXUP_R && H % 8 == 0 && M <= 64 && s.has_value()
        && s->numel() == (int64_t)4 * H * (R / 64) && n.stride(0) % 8 == 0) {
        // SPD-241: A once, then the tensor-core mix_up (every M; the wrapper hands it at most 64 rows)
        auto a = torch::empty({M, R}, n.options());
        launch(mix_a_kernel, dim3((M * R + 255) / 256), dim3(256), 0, st, (const float*)d.data_ptr(), (int)d.stride(0),
               (__nv_bfloat16*)a.data_ptr(), M, R, inv);
        mix_smem_once(mixer_mma_mixup_kernel);
        const int mr = (M + 15) / 16 * 16;
        launch(mixer_mma_mixup_kernel, dim3(H / 8), dim3(128), (size_t)mr * MIXUP_LDS * 2 + (size_t)4 * M * 8 * 4, st,
               (const __nv_bfloat16*)a.data_ptr(), (const __nv_bfloat16*)n.data_ptr(), (int)n.stride(0),
               (const int8_t*)w.data_ptr(), (const float*)s->data_ptr(), (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
               M, H, inv);
        return;
    }
    if (g_gemv_nb && M > 1 && w.dtype() == torch::kInt8) {
        // SPD-231: A once, then 16 columns a CTA with the streams in-lane (int8 weights; fp16 ones were slower)
        auto a = torch::empty({M, R}, n.options());
        launch(mix_a_kernel, dim3((M * R + 255) / 256), dim3(256), 0, st, (const float*)d.data_ptr(), (int)d.stride(0),
               (__nv_bfloat16*)a.data_ptr(), M, R, inv);
        if (w.dtype() == torch::kInt8) {
            TORCH_CHECK(s.has_value() && s->is_contiguous() && s->numel() % (4 * H) == 0 && R % 16 == 0, "int8 scales, R % 16");
            const int ng = s->numel() / (4 * H), G = ng == 1 ? 0 : R / ng;
            TORCH_CHECK(!G || (R % ng == 0 && G % 16 == 0), "scale groups: R / groups a multiple of 16");
            launch(mix_up_nb_kernel<int8_t, 8>, dim3((H + 15) / 16), dim3(128), 0, st, (const __nv_bfloat16*)a.data_ptr(),
                (const __nv_bfloat16*)n.data_ptr(), (int)n.stride(0), (const int8_t*)w.data_ptr(), (const float*)s->data_ptr(),
                (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, R, inv, G);
        } else {
            TORCH_CHECK(w.dtype() == torch::kHalf && R % 8 == 0, "fp16 up weight, R % 8");
            launch(mix_up_nb_kernel<__half, 8>, dim3((H + 15) / 16), dim3(128), 0, st, (const __nv_bfloat16*)a.data_ptr(),
                (const __nv_bfloat16*)n.data_ptr(), (int)n.stride(0), (const __half*)w.data_ptr(), (const float*)nullptr,
                (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, R, inv, 0);
        }
        return;
    }
    if (w.dtype() == torch::kInt8) {
        TORCH_CHECK(s.has_value() && s->is_contiguous() && s->numel() % (4 * H) == 0 && R % 16 == 0, "int8 scales, R % 16");
        const int ng = s->numel() / (4 * H), G = ng == 1 ? 0 : R / ng;
        TORCH_CHECK(!G || (R % ng == 0 && G % 16 == 0), "scale groups: R / groups a multiple of 16");
        auto k = M == 1 ? mix_up_kernel<int8_t, 1> : mix_up_kernel<int8_t, 8>;
        if (M > 1 && g_gemv_mc == 1) k = mix_up_kernel<int8_t, 4, 2>;
        if (M > 1 && g_gemv_mc == 2) k = mix_up_kernel<int8_t, 4, 3>;
        if (M > 1 && g_gemv_mc == 3) k = mix_up_kernel<int8_t, 2, 4>;
        if (M > 1 && g_gemv_mc == 4) k = mix_up_kernel<int8_t, 8, 2>;
        if (M > 1 && g_gemv_mc == 5) k = mix_up_kernel<int8_t, 2, 3>;
        launch(k, grid, dim3(256), smem, st, (const float*)d.data_ptr(), (int)d.stride(0), (const __nv_bfloat16*)n.data_ptr(),
            (int)n.stride(0), (const int8_t*)w.data_ptr(), (const float*)s->data_ptr(),
            (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, R, inv, G);
    } else {
        TORCH_CHECK(w.dtype() == torch::kHalf && R % 8 == 0, "fp16 up weight, R % 8");
        auto k = M == 1 ? mix_up_kernel<__half, 1> : mix_up_kernel<__half, 8>;
        launch(k, grid, dim3(256), smem, st, (const float*)d.data_ptr(), (int)d.stride(0), (const __nv_bfloat16*)n.data_ptr(),
            (int)n.stride(0), (const __half*)w.data_ptr(), (const float*)nullptr,
            (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, R, inv, 0);
    }
}

void fn_router(torch::Tensor logits, int64_t E, int64_t k, torch::Tensor ids, torch::Tensor wts) {
    const at::cuda::OptionalCUDAGuard guard(logits.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = logits.size(0);
    TORCH_CHECK(logits.dtype() == torch::kFloat && logits.stride(1) == 1 && logits.size(1) >= E + 1, "logits fp32 [M, E + 1]");
    TORCH_CHECK(ids.dtype() == torch::kInt && wts.dtype() == torch::kFloat && ids.is_contiguous() && wts.is_contiguous(), "ids int32, wts fp32");
    TORCH_CHECK(ids.numel() == M * (k + 1) && wts.numel() == M * (k + 1) && k < 32, "slots");
    TORCH_CHECK(E <= 512, "router kernel is built for up to 512 experts");
    if (E <= 256)
        launch(router_kernel<8>, dim3(M), dim3(32), 0, st, (const float*)logits.data_ptr(), (int)logits.stride(0), (int)E,
            (int)k, (int*)ids.data_ptr(), (float*)wts.data_ptr());
    else
        launch(router_kernel<16>, dim3(M), dim3(32), 0, st, (const float*)logits.data_ptr(), (int)logits.stride(0), (int)E,
            (int)k, (int*)ids.data_ptr(), (float*)wts.data_ptr());
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

// cp.async stages a warp keeps (fn_set_moe_st; QWEN38_FN_MOE_ST): fewer stages, less shared
// memory, more CTAs resident on an SM. The arithmetic does not depend on it.
static int g_moe_st = 2;
// warps a MoE CTA splits its K range over (fn_set_moe_warps; QWEN38_FN_MOE_WARPS): sets the order
// of the partial sums, so a build keeps one value for every row count
static int g_moe_warps = 8;

void fn_set_moe_st(int64_t st) { TORCH_CHECK(st >= 2 && st <= 4, "MoE stages 2..4"); g_moe_st = (int)st; }
void fn_set_moe_warps(int64_t w) { TORCH_CHECK(w == 4 || w == 8, "MoE warps 4 or 8"); g_moe_warps = (int)w; }

// the input row (fp16), the eight warps' partial rows, and 8 warps x stages x 8 tiles of the bank's
// widest trellis tile (tile_u32(kbmax) words)
static size_t moe_smem(int kin, int kbmax) {
    return (size_t)(kin / 2 + 8 * 128) * 4 + (size_t)g_moe_warps * g_moe_st * 8 * tile_u32(kbmax) * 4;
}

#define MOE_PICK(KERN) (g_moe_warps == 4                                                            \
    ? (g_moe_st == 2 ? KERN<2, 4> : (g_moe_st == 3 ? KERN<3, 4> : KERN<4, 4>))                         \
    : (g_moe_st == 2 ? KERN<2, 8> : (g_moe_st == 3 ? KERN<3, 8> : KERN<4, 8>)))

void fn_moe_gu(torch::Tensor x, torch::Tensor ids, torch::Tensor order, torch::Tensor tr, torch::Tensor kb,
               torch::Tensor su, torch::Tensor sv, torch::Tensor y, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), H = x.size(1), P = ids.numel(), S = P / M, I = y.size(2);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, H]");
    TORCH_CHECK(ids.dtype() == torch::kInt && order.dtype() == torch::kInt && (order.numel() == P || order.numel() == 0) && P == M * S, "ids/order int32 [M S] (order empty: launch order)");
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && y.size(0) == P && y.size(1) == 2, "y bf16 [P, 2, I]");
    TORCH_CHECK(H % 128 == 0 && I % 128 == 0, "H, I multiples of 128");
    const size_t smem = moe_smem(H, (int)kbmax);
    auto kern = MOE_PICK(moe_gu_kernel);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    launch(kern, dim3(2 * (I / 128), P), dim3(32 * g_moe_warps), smem, st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const int*)ids.data_ptr(), order.numel() ? (const int*)order.data_ptr() : (const int*)nullptr, S,
        moe_args(tr, kb, su, sv), (__nv_bfloat16*)y.data_ptr(), H, I);
}

void fn_moe_down(torch::Tensor ygu, torch::Tensor ids, torch::Tensor order, torch::Tensor tr, torch::Tensor kb,
                 torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(ygu.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = ygu.size(0), I = ygu.size(2), H = yd.size(1);
    TORCH_CHECK(ygu.dtype() == torch::kBFloat16 && ygu.is_contiguous() && ygu.size(1) == 2, "ygu bf16 [P, 2, I]");
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == P, "yd fp32 [P, H]");
    TORCH_CHECK(ids.numel() == P && (order.numel() == P || order.numel() == 0), "pairs");
    const size_t smem = moe_smem(I, (int)kbmax);
    auto kern = MOE_PICK(moe_down_kernel);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    launch(kern, dim3(H / 128, P), dim3(32 * g_moe_warps), smem, st, (const __nv_bfloat16*)ygu.data_ptr(), (const int*)ids.data_ptr(),
        order.numel() ? (const int*)order.data_ptr() : (const int*)nullptr, moe_args(tr, kb, su, sv),
        (float*)yd.data_ptr(), H, I);
}

// the grouped experts' shared memory: the group's rows (fp16) and, for down, the warp partials beside
// them (gate|up lays them over the rows), then the stages
static size_t moe_group_smem(int kin, int kbmax, bool red_apart) {
    const size_t rows = (size_t)MOE_GR * kin * 2, red = (size_t)g_moe_warps * MOE_GR * 128 * 4;
    return (red_apart ? rows + red : (rows > red ? rows : red)) + (size_t)g_moe_warps * g_moe_st * 8 * tile_u32(kbmax) * 4;
}

void fn_moe_groups(torch::Tensor ids, torch::Tensor order, torch::Tensor g, torch::Tensor ng) {
    const at::cuda::OptionalCUDAGuard guard(ids.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = ids.numel();
    TORCH_CHECK(ids.dtype() == torch::kInt && (order.numel() == P || order.numel() == 0), "ids/order int32");
    TORCH_CHECK(g.dtype() == torch::kInt && g.size(0) >= P && g.size(1) == 2 + MOE_GR && ng.numel() == 1, "groups");
    launch(moe_groups_kernel, dim3(1), dim3(32), 0, st, (const int*)ids.data_ptr(),
        order.numel() ? (const int*)order.data_ptr() : (const int*)nullptr, P, (int*)g.data_ptr(), (int*)ng.data_ptr());
}

void fn_moe_gu_group(torch::Tensor x, torch::Tensor g, torch::Tensor ng, int64_t S, torch::Tensor tr, torch::Tensor kb,
                     torch::Tensor su, torch::Tensor sv, torch::Tensor y, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int H = x.size(1), P = y.size(0), I = y.size(2);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, H]");
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && y.size(1) == 2, "y bf16 [P, 2, I]");
    const size_t smem = moe_group_smem(H, (int)kbmax, false);
    auto kern = MOE_PICK(moe_gu_group_kernel);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    TORCH_CHECK(smem <= 96 * 1024, "grouped gate|up shared memory");
    launch(kern, dim3(2 * (I / 128), P), dim3(32 * g_moe_warps), smem, st, (const __nv_bfloat16*)x.data_ptr(),
        (int)x.stride(0), (const int*)g.data_ptr(), (const int*)ng.data_ptr(), (int)S, moe_args(tr, kb, su, sv),
        (__nv_bfloat16*)y.data_ptr(), H, I);
}

void fn_moe_down_group(torch::Tensor ygu, torch::Tensor g, torch::Tensor ng, torch::Tensor tr, torch::Tensor kb,
                       torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(ygu.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = ygu.size(0), I = ygu.size(2), H = yd.size(1);
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == P, "yd fp32 [P, H]");
    const size_t smem = moe_group_smem(I, (int)kbmax, true);
    auto kern = MOE_PICK(moe_down_group_kernel);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    TORCH_CHECK(smem <= 96 * 1024, "grouped down shared memory");
    launch(kern, dim3(H / 128, P), dim3(32 * g_moe_warps), smem, st, (const __nv_bfloat16*)ygu.data_ptr(),
        (const int*)g.data_ptr(), (const int*)ng.data_ptr(), moe_args(tr, kb, su, sv), (float*)yd.data_ptr(), H, I);
}

void fn_moe_combine(torch::Tensor yd, torch::Tensor ids, torch::Tensor wts, torch::Tensor out) {
    const at::cuda::OptionalCUDAGuard guard(yd.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = out.size(0), H = out.size(1), S = ids.numel() / M;
    TORCH_CHECK(S <= 64 && yd.size(0) == M * S && yd.size(1) == H, "combine shapes");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.stride(1) == 1, "out bf16 [M, H]");
    launch(moe_combine_kernel, dim3((H + 255) / 256, M), dim3(256), 0, st, (const float*)yd.data_ptr(),
        (const int*)ids.data_ptr(), (const float*)wts.data_ptr(), S, (__nv_bfloat16*)out.data_ptr(),
        (int)out.stride(0), H);
}
