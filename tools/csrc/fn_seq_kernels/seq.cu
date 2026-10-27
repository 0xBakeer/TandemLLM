
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------ commits (SPD-239)
// The copies of many states' commits in one launch: E [ne][16] int64, one entry a CTA column (grid (ne, 8)).
// kind 0, rows: dst row i (i < n) = src row idx[i], `words` 4-byte words a row (strides in words):
//   [0, src, src_stride, dst, dst_stride, words, idx (int64 [n] on the device), n]
// kind 1, a bf16 window [C, w] (w <= 9) in place from its own columns and a block's rows: column j becomes
//   cols[j] < w ? the window's column cols[j] : row cols[j] - w of src (a row's C channels at src + row * stride);
//   a thread owns a channel, so every read of the channel is before its writes:
//   [1, window, src, src_stride (elements), C, w, 0, cols[0..w-1]]
__global__ void __launch_bounds__(256) commit_multi_kernel(const long long* __restrict__ E) {
    pdl_wait();
    pdl_trigger();
    const long long* e = E + (size_t)blockIdx.x * 16;
    const int step = gridDim.y * blockDim.x, first = blockIdx.y * blockDim.x + threadIdx.x;
    if (e[0] == 0) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(e[1]);
        uint32_t* dst = reinterpret_cast<uint32_t*>(e[3]);
        const long long ss = e[2], ds = e[4];
        const int words = (int)e[5], n = (int)e[7];
        const long long* idx = reinterpret_cast<const long long*>(e[6]);
        for (int i = 0; i < n; ++i) {
            const uint32_t* s = src + idx[i] * ss;
            uint32_t* d = dst + (long long)i * ds;
            for (int w = first; w < words; w += step) d[w] = s[w];
        }
        return;
    }
    __nv_bfloat16* win = reinterpret_cast<__nv_bfloat16*>(e[1]);
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(e[2]);
    const long long ss = e[3];
    const int C = (int)e[4], w = (int)e[5];
    for (int c = first; c < C; c += step) {
        __nv_bfloat16 old[9], nv[9];
#pragma unroll
        for (int j = 0; j < 9; ++j) if (j < w) old[j] = win[(size_t)c * w + j];
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            if (j < w) {
                const int col = (int)e[7 + j];
                nv[j] = col < w ? old[col] : src[(size_t)(col - w) * ss + c];
            }
        }
#pragma unroll
        for (int j = 0; j < 9; ++j) if (j < w) win[(size_t)c * w + j] = nv[j];
    }
}

// ------------------------------------------------------------------------------ GDN conv + gates
// grid (nC + 1, M), 256 threads. CTA x < nC: 256 channels of row m. The window of row m is
// [tap3, tap2, tap1, x_m]; SRC[m][j] (j = 0, 1, 2 for distance 3, 2, 1) is a row of the block
// (>= 0) or state column -1 - SRC (< 0; column 2 is the newest). acc = fma over the taps in order,
// y = bf16(silu(acc)). SHIFT (one row, decode): the state becomes [col1, col2, x] (all reads first).
// CTA x == nC: the gates of row m's H heads.
__global__ void __launch_bounds__(256) gdn_pre_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        __nv_bfloat16* __restrict__ CS, const __nv_bfloat16* __restrict__ W, const int* __restrict__ SRC,
        __nv_bfloat16* __restrict__ Y, int ldy, int C, int nC,
        const __nv_bfloat16* __restrict__ AB, int ldab, const float* __restrict__ ALOG,
        const float* __restrict__ DTB, float* __restrict__ G, float* __restrict__ BETA, int H, int shift) {
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.y;
    if ((int)blockIdx.x == nC) {
        const int h = threadIdx.x;
        if (h >= H) return;
        const float a = b2f(AB[(size_t)m * ldab + h]), b = b2f(AB[(size_t)m * ldab + H + h]);
        const float x = a + DTB[h];
        const float sp = x > 20.0f ? x : ::logf(1.0f + ::expf(x));
        G[(size_t)m * H + h] = -::expf(ALOG[h]) * sp;
        BETA[(size_t)m * H + h] = 1.0f / (1.0f + ::expf(-b));
        return;
    }
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const float x = b2f(X[(size_t)m * ldx + c]);
    float tap[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int s = SRC[m * 3 + j];
        tap[j] = s >= 0 ? b2f(X[(size_t)s * ldx + c]) : b2f(CS[(size_t)c * 3 + (-1 - s)]);
    }
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) acc = fmaf(tap[j], b2f(W[(size_t)c * 4 + j]), acc);
    acc = fmaf(x, b2f(W[(size_t)c * 4 + 3]), acc);
    Y[(size_t)m * ldy + c] = __float2bfloat16_rn(acc / (1.0f + ::expf(-acc)));
    if (shift) {
        const __nv_bfloat16 s1 = CS[(size_t)c * 3 + 1], s2 = CS[(size_t)c * 3 + 2];
        CS[(size_t)c * 3 + 0] = s1;
        CS[(size_t)c * 3 + 1] = s2;
        CS[(size_t)c * 3 + 2] = X[(size_t)m * ldx + c];
    }
}

// ------------------------------------------------------------------------------ GDN recurrence
// One row of the rule on the thread's 16 values of the tile (lane l: key rows l + 32 j; warp w:
// columns c0 .. c0 + 3):
//   q, k <- l2-normalised (q also times 1/sqrt(128)), key head hk
//   s <- s exp(g); kv = s^T k; delta = (v - kv) beta; s += k (x) delta; out = s^T q
// OUT null: no output (a parked commit row).
__device__ __forceinline__ void gdn_row(float (&s)[4][4], const __nv_bfloat16* __restrict__ row, int koff, int voff,
        float g, float beta, int h, int hk, int lane, int c0, __nv_bfloat16* __restrict__ out) {
    float q[4], k[4], qq = 0.f, kk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = b2f(row[hk * 128 + lane + 32 * j]);
        k[j] = b2f(row[koff + hk * 128 + lane + 32 * j]);
        qq = fmaf(q[j], q[j], qq);
        kk = fmaf(k[j], k[j], kk);
    }
    const float rq = rsqrtf(warp_sum(qq) + 1e-6f) * 0.08838834764831845f;
    const float rk = rsqrtf(warp_sum(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[j] *= rq; k[j] *= rk; }
    const float eg = ::expf(g);
    float o4[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { s[j][c] *= eg; p = fmaf(s[j][c], k[j], p); }
        const float kv = warp_sum(p);
        const float delta = (b2f(row[voff + h * 128 + c0 + c]) - kv) * beta;
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { s[j][c] = fmaf(k[j], delta, s[j][c]); o = fmaf(s[j][c], q[j], o); }
        o4[c] = warp_sum(o);
    }
    if (out != nullptr && lane == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) out[c0 + c] = __float2bfloat16_rn(o4[c]);
    }
}

__device__ __forceinline__ void tile_load(float (&s)[4][4], const float* __restrict__ src, int lane, int c0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(src + (size_t)(lane + 32 * j) * 128 + c0);
        s[j][0] = t.x; s[j][1] = t.y; s[j][2] = t.z; s[j][3] = t.w;
    }
}

__device__ __forceinline__ void tile_store(const float (&s)[4][4], float* __restrict__ dst, int lane, int c0) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<float4*>(dst + (size_t)(lane + 32 * j) * 128 + c0) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}

// grid (H, 128 / 16), 128 threads: CTA (h, vb) owns columns vb 16 .. vb 16 + 15 of head h's state.
// First the parked rows of the last commit (PRE, *PN of them, the commit folded into this call): the
// entry tile walked over them and written back (the committed state). Then the block's rows: CTL[m] =
// (load, save): load -1 continue from the previous row's tile, -2 the entry state S, >= 0 scratch
// slot; save >= 0 leaves the tile in that slot after the row. WRITE: the last row's tile goes back
// to S (a decode step).
__device__ __forceinline__ void gdn_rows_body(const __nv_bfloat16* __restrict__ QKV, int ldq,
        int koff, int voff, const float* __restrict__ G, const float* __restrict__ BETA,
        float* __restrict__ S, float* __restrict__ STK, const int* __restrict__ CTL, int M,
        __nv_bfloat16* __restrict__ OUT, int H, int REP, int write,
        const __nv_bfloat16* __restrict__ PRE, int ldp, const float* __restrict__ PG, const float* __restrict__ PB,
        const int* __restrict__ PN) {
    const int h = blockIdx.x, vb = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c0 = vb * 16 + warp * 4;
    const int hk = h / REP;
    const size_t hs = (size_t)h * 128 * 128;
    float s[4][4];
    const int P = PN != nullptr ? *PN : 0;
    if (P > 0) {
        tile_load(s, S + hs, lane, c0);
        for (int p = 0; p < P; ++p)
            gdn_row(s, PRE + (size_t)p * ldp, koff, voff, PG[(size_t)p * H + h], PB[(size_t)p * H + h], h, hk, lane, c0, nullptr);
        tile_store(s, S + hs, lane, c0);
    }
    for (int m = 0; m < M; ++m) {
        const int ld = CTL[2 * m], sv = CTL[2 * m + 1];
        if (ld != -1 || m == 0)
            tile_load(s, (ld >= 0) ? STK + (size_t)ld * H * 128 * 128 + hs : S + hs, lane, c0);
        gdn_row(s, QKV + (size_t)m * ldq, koff, voff, G[(size_t)m * H + h], BETA[(size_t)m * H + h], h, hk, lane, c0,
                OUT + ((size_t)m * H + h) * 128);
        if (sv >= 0) tile_store(s, STK + (size_t)sv * H * 128 * 128 + hs, lane, c0);
    }
    if (write && M > 0) tile_store(s, S + hs, lane, c0);
}

__global__ void __launch_bounds__(128) gdn_rows_kernel(const __nv_bfloat16* __restrict__ QKV, int ldq,
        int koff, int voff, const float* __restrict__ G, const float* __restrict__ BETA,
        float* __restrict__ S, float* __restrict__ STK, const int* __restrict__ CTL, int M,
        __nv_bfloat16* __restrict__ OUT, int H, int REP, int write,
        const __nv_bfloat16* __restrict__ PRE, int ldp, const float* __restrict__ PG, const float* __restrict__ PB,
        const int* __restrict__ PN) {
    pdl_wait();
    pdl_trigger();
    gdn_rows_body(QKV, ldq, koff, voff, G, BETA, S, STK, CTL, M, OUT, H, REP, write, PRE, ldp, PG, PB, PN);
}

// SPD-239: `gdn_rows_kernel` for several states in one launch, grid (H, 8, states): state z's call from its
// entry E[z] = [qkv, ldq, g, beta, S, scratch (its own slots), ctl, M, out, pre, ldp, pg, pb, pn, write, 0]
// (pre / pg / pb / pn 0: no parked rows), each CTA the one-state kernel's own arithmetic
__global__ void __launch_bounds__(128) gdn_rows_multi_kernel(const long long* __restrict__ E, int koff, int voff,
                                                             int H, int REP) {
    pdl_wait();
    pdl_trigger();
    const long long* e = E + (size_t)blockIdx.z * 16;
    gdn_rows_body(reinterpret_cast<const __nv_bfloat16*>(e[0]), (int)e[1], koff, voff,
                  reinterpret_cast<const float*>(e[2]), reinterpret_cast<const float*>(e[3]),
                  reinterpret_cast<float*>(e[4]), reinterpret_cast<float*>(e[5]), reinterpret_cast<const int*>(e[6]),
                  (int)e[7], reinterpret_cast<__nv_bfloat16*>(e[8]), H, REP, (int)e[14],
                  reinterpret_cast<const __nv_bfloat16*>(e[9]), (int)e[10], reinterpret_cast<const float*>(e[11]),
                  reinterpret_cast<const float*>(e[12]), reinterpret_cast<const int*>(e[13]));
}

// --------------------------------------------------------------------------------------- PLE
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) t += sh[w];
    return t;
}

// grid (M, hc), 1024 threads: KEY, QRY [M, hc H] (normed), VAL [M, H] -> GV [M, hc H]
__global__ void __launch_bounds__(1024) ple_gate_kernel(const __nv_bfloat16* __restrict__ KEY,
        const __nv_bfloat16* __restrict__ QRY, const __nv_bfloat16* __restrict__ VAL,
        __nv_bfloat16* __restrict__ GV, int Hd, float sqrt_h) {
    __shared__ float sh[32];
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.x, s = blockIdx.y, hc = gridDim.y;
    const size_t o = ((size_t)m * hc + s) * Hd;
    float acc = 0.f;
    for (int i = threadIdx.x; i < Hd; i += blockDim.x) acc += bfr(b2f(KEY[o + i]) * b2f(QRY[o + i]));
    const float g = bfr(bfr(block_sum(acc, sh)) / sqrt_h);
    float r = bfr(::sqrtf(fmaxf(fabsf(g), 1e-6f)));
    r = g > 0.f ? r : (g < 0.f ? -r : 0.f);
    const float sg = bfr(1.0f / (1.0f + ::expf(-r)));
    for (int i = threadIdx.x; i < Hd; i += blockDim.x)
        GV[o + i] = __float2bfloat16_rn(sg * b2f(VAL[(size_t)m * Hd + i]));
}

// grid (C / 256, M): y = bf16(gv + bf16(silu(bf16(acc)))), acc = fma over taps at distance 9, 6, 3
// (SRC[m][0..2]: a row of the block, or -1 - state column; column 8 is the newest) and the row's
// own normed value, the weights rounded to bf16 by the caller (the reference casts them)
__global__ void __launch_bounds__(256) ple_conv_kernel(const __nv_bfloat16* __restrict__ GVN,
        const __nv_bfloat16* __restrict__ GV, const __nv_bfloat16* __restrict__ PS, int slen,
        const __nv_bfloat16* __restrict__ W, const int* __restrict__ SRC, __nv_bfloat16* __restrict__ Y, int C) {
    pdl_wait();
    pdl_trigger();
    const int m = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int s = SRC[m * 3 + j];
        const float v = s >= 0 ? b2f(GVN[(size_t)s * C + c]) : b2f(PS[(size_t)c * slen + (-1 - s)]);
        acc = fmaf(v, b2f(W[(size_t)c * 4 + j]), acc);
    }
    acc = fmaf(b2f(GVN[(size_t)m * C + c]), b2f(W[(size_t)c * 4 + 3]), acc);
    const float a = bfr(acc);
    const float sl = bfr(a / (1.0f + ::expf(-a)));
    Y[(size_t)m * C + c] = __float2bfloat16_rn(b2f(GV[(size_t)m * C + c]) + sl);
}

// ------------------------------------------------------------------------------------- QSA
// One pooled indexer key per CTA (128 threads = the indexer head dim): the mean of the block's 4 raw
// keys (fp32, in key order, / 4, rounded to bf16), the RMS norm (1 + w), the rope at the block's first
// position (bf16(bf16(n c) + bf16(rot(n) s)) over the first ROT dims). SLOTS[b][0..3]: the cache slots
// of the block's keys; POSB[b]: its first position.
__device__ __forceinline__ float qsa_pool_body(const __nv_bfloat16* __restrict__ IKC,
        const int* __restrict__ SLOTS, const int* __restrict__ POSB, const __nv_bfloat16* __restrict__ KLN,
        const __nv_bfloat16* __restrict__ COS, const __nv_bfloat16* __restrict__ SIN, int b, int ROT, float eps) {
    __shared__ float sh[4];
    __shared__ float nv[128];
    const int t = threadIdx.x;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += b2f(IKC[(size_t)SLOTS[b * 4 + i] * 128 + t]);
    const float v = bfr(acc / 4.0f);
    const float r = rsqrtf(block_sum(v * v, sh) / 128.0f + eps);
    const float n = bfr(v * r * (1.0f + b2f(KLN[t])));
    nv[t] = n;
    __syncthreads();
    float o = n;
    if (t < ROT) {
        const int pos = POSB[b];
        const float c = b2f(COS[(size_t)pos * ROT + t]), sn = b2f(SIN[(size_t)pos * ROT + t]);
        const float rh = t < ROT / 2 ? -nv[t + ROT / 2] : nv[t - ROT / 2];
        o = bfr(bfr(n * c) + bfr(rh * sn));
    }
    return o;
}

__global__ void __launch_bounds__(128) qsa_pool_kernel(const __nv_bfloat16* __restrict__ IKC,
        const int* __restrict__ SLOTS, const int* __restrict__ POSB, const __nv_bfloat16* __restrict__ KLN,
        const __nv_bfloat16* __restrict__ COS, const __nv_bfloat16* __restrict__ SIN, __nv_bfloat16* __restrict__ OUT,
        int ROT, float eps) {
    pdl_wait();
    pdl_trigger();
    const int b = blockIdx.x;
    const float o = qsa_pool_body(IKC, SLOTS, POSB, KLN, COS, SIN, b, ROT, eps);
    OUT[(size_t)b * 128 + threadIdx.x] = __float2bfloat16_rn(o);
}

// SPD-239: `qsa_pool` of many (state, layer) pairs into their pools in one launch, grid (NB, pairs): pair z's
// entry E[z] = [ikc, slots, posb, kln, pool, rows (int64 [NB], the pool row of each block)]; a CTA computes
// qsa_pool_kernel's block and writes it at its pool row (what `pool.index_copy_(0, rows, qsa_pool(...))` writes)
__global__ void __launch_bounds__(128) qsa_pool_multi_kernel(const long long* __restrict__ E,
        const __nv_bfloat16* __restrict__ COS, const __nv_bfloat16* __restrict__ SIN, int ROT, float eps) {
    pdl_wait();
    pdl_trigger();
    const long long* e = E + (size_t)blockIdx.y * 16;
    const int b = blockIdx.x;
    const float o = qsa_pool_body(reinterpret_cast<const __nv_bfloat16*>(e[0]), reinterpret_cast<const int*>(e[1]),
                                  reinterpret_cast<const int*>(e[2]), reinterpret_cast<const __nv_bfloat16*>(e[3]),
                                  COS, SIN, b, ROT, eps);
    const long long row = reinterpret_cast<const long long*>(e[5])[b];
    reinterpret_cast<__nv_bfloat16*>(e[4])[row * 128 + threadIdx.x] = __float2bfloat16_rn(o);
}

// The indexer's score of block b for row t: sum over the NH query heads in order of
// relu(q_h . pooled_b) (fp32, lane-strided then a fixed xor tree), / sqrt(128). grid (M, NB / 8), 8
// warps a CTA, a warp a block. QCTL[t] = (complete blocks, first own block): blocks below the
// first own block read the shared pool, the rest the row's own pooled blocks; blocks past the
// complete ones score -inf.
__global__ void __launch_bounds__(256) qsa_scores_kernel(const __nv_bfloat16* __restrict__ IQ,
        const __nv_bfloat16* __restrict__ POOL, const __nv_bfloat16* __restrict__ OWN, int NO,
        const int* __restrict__ QCTL, float* __restrict__ SC, int NB, int NH) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const int b = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (b >= NB) return;
    const int nblk = QCTL[t * 4], own0 = QCTL[t * 4 + 1];
    if (b >= nblk) {
        if (lane == 0) SC[(size_t)t * NB + b] = -INFINITY;
        return;
    }
    const __nv_bfloat16* pk = b < own0 ? POOL + (size_t)b * 128 : OWN + ((size_t)t * NO + (b - own0)) * 128;
    float k[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) k[e] = b2f(pk[lane * 4 + e]);
    float tot = 0.f;
    for (int h = 0; h < NH; ++h) {
        const __nv_bfloat16* q = IQ + ((size_t)t * NH + h) * 128 + lane * 4;
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) d = fmaf(b2f(q[e]), k[e], d);
        tot += fmaxf(warp_sum(d), 0.f);
    }
    if (lane == 0) SC[(size_t)t * NB + b] = tot / sqrtf(128.0f);
}

// The selection of one row (a CTA, 1,024 threads): keys (~ord(score) << 32 | block) over the row's NB
// scores bitonic-sorted ascending in shared memory -- score descending, the lower block first on ties
// (a stable descending sort's order) -- then the first TOPK blocks sorted ascending, 4 positions each,
// then the row's tail. NB a power of two, <= 8,192.
__device__ __forceinline__ uint32_t ord_f(float f) {
    const uint32_t u = __float_as_uint(f + 0.0f);          // -0.0 ties with +0.0, as in a float compare
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(1024) qsa_topk_kernel(const float* __restrict__ SC, int NB,
        const int* __restrict__ TAIL, int* __restrict__ SEL, int nsel, int TOPK) {
    extern __shared__ unsigned long long keys[];
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < NB; i += blockDim.x)
        keys[i] = ((unsigned long long)(~ord_f(SC[(size_t)t * NB + i])) << 32) | (unsigned)i;
    __syncthreads();
    for (int k = 2; k <= NB; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < NB; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned long long a = keys[i], b = keys[ixj];
                    if (((i & k) == 0) == (a > b)) { keys[i] = b; keys[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    // the kept blocks, ascending: keys[0 .. TOPK) hold them; sort their block numbers
    unsigned* blk = reinterpret_cast<unsigned*>(keys + TOPK);
    for (int i = threadIdx.x; i < TOPK; i += blockDim.x) blk[i] = (unsigned)(keys[i] & 0xffffffffu);
    __syncthreads();
    for (int k = 2; k <= TOPK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < TOPK; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned a = blk[i], b = blk[ixj];
                    if (((i & k) == 0) == (a > b)) { blk[i] = b; blk[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    int* out = SEL + (size_t)t * nsel;
    for (int i = threadIdx.x; i < 4 * TOPK; i += blockDim.x) out[i] = (int)blk[i >> 2] * 4 + (i & 3);
    if (threadIdx.x < 3) out[4 * TOPK + threadIdx.x] = TAIL[t * 3 + threadIdx.x];
}

}  // namespace

static int g_pdl = 1;

template <typename... KArgs, typename... Args>
static void launch(void (*k)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    TORCH_CHECK(e == cudaSuccess, "fn_seq_kernels launch: ", cudaGetErrorString(e));
}

void seq_set_pdl(int64_t on) { g_pdl = on ? 1 : 0; }

void seq_gdn_rows_multi(torch::Tensor E, int64_t koff, int64_t voff, int64_t H, int64_t rep) {
    const at::cuda::OptionalCUDAGuard guard(E.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(E.dtype() == torch::kLong && E.is_contiguous() && E.dim() == 2 && E.size(1) == 16, "entries int64 [n, 16]");
    if (E.size(0) == 0) return;
    launch(gdn_rows_multi_kernel, dim3((unsigned)H, 8, (unsigned)E.size(0)), dim3(128), st, (const long long*)E.data_ptr(),
           (int)koff, (int)voff, (int)H, (int)rep);
}

void seq_commit_multi(torch::Tensor E) {
    const at::cuda::OptionalCUDAGuard guard(E.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(E.dtype() == torch::kLong && E.is_contiguous() && E.dim() == 2 && E.size(1) == 16, "entries int64 [n, 16]");
    if (E.size(0) == 0) return;
    launch(commit_multi_kernel, dim3((unsigned)E.size(0), 8), dim3(256), st, (const long long*)E.data_ptr());
}

void seq_gdn_pre(torch::Tensor x, torch::Tensor cs, torch::Tensor w, torch::Tensor src, torch::Tensor y,
                 torch::Tensor ab, torch::Tensor alog, torch::Tensor dtb, torch::Tensor g, torch::Tensor beta, bool shift) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), C = y.size(1), H = alog.numel();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.size(1) >= C, "x bf16 [M, >= C]");
    TORCH_CHECK(cs.dtype() == torch::kBFloat16 && cs.is_contiguous() && cs.numel() == (int64_t)C * 3, "conv state bf16 [C, 3]");
    TORCH_CHECK(w.dtype() == torch::kBFloat16 && w.is_contiguous() && w.numel() == (int64_t)C * 4, "conv weight bf16 [C, 4]");
    TORCH_CHECK(src.dtype() == torch::kInt && src.is_contiguous() && src.numel() == M * 3, "src int32 [M, 3]");
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && y.size(0) == M, "y bf16 [M, C]");
    TORCH_CHECK(ab.dtype() == torch::kBFloat16 && ab.stride(1) == 1 && ab.size(1) == 2 * H, "ab bf16 [M, 2 H]");
    TORCH_CHECK(alog.dtype() == torch::kFloat && dtb.dtype() == torch::kFloat && H <= 256, "A_log, dt_bias fp32 [H]");
    TORCH_CHECK(!shift || M == 1, "the state shifts in a one-row call only");
    const int nC = (C + 255) / 256;
    launch(gdn_pre_kernel, dim3(nC + 1, M), dim3(256), st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (__nv_bfloat16*)cs.data_ptr(), (const __nv_bfloat16*)w.data_ptr(), (const int*)src.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), C, nC, (const __nv_bfloat16*)ab.data_ptr(), (int)ab.stride(0),
        (const float*)alog.data_ptr(), (const float*)dtb.data_ptr(), (float*)g.data_ptr(), (float*)beta.data_ptr(),
        H, shift ? 1 : 0);
}

void seq_gdn_rows(torch::Tensor qkv, int64_t koff, int64_t voff, torch::Tensor g, torch::Tensor beta, torch::Tensor S,
                  torch::Tensor stk, torch::Tensor ctl, torch::Tensor out, int64_t rep, bool write,
                  c10::optional<torch::Tensor> pre, c10::optional<torch::Tensor> pg, c10::optional<torch::Tensor> pb,
                  c10::optional<torch::Tensor> pn) {
    const at::cuda::OptionalCUDAGuard guard(qkv.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = qkv.size(0), H = S.size(-3);
    TORCH_CHECK(qkv.dtype() == torch::kBFloat16 && qkv.stride(1) == 1, "qkv bf16 [M, C]");
    TORCH_CHECK(S.dtype() == torch::kFloat && S.is_contiguous() && S.size(-1) == 128 && S.size(-2) == 128, "S fp32 [H, 128, 128]");
    TORCH_CHECK(stk.dtype() == torch::kFloat && stk.is_contiguous(), "scratch fp32");
    TORCH_CHECK(ctl.dtype() == torch::kInt && ctl.is_contiguous() && ctl.numel() == 2 * M, "ctl int32 [M, 2]");
    TORCH_CHECK(g.dtype() == torch::kFloat && beta.dtype() == torch::kFloat && g.numel() == (int64_t)M * H, "g, beta fp32 [M, H]");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)M * H * 128, "out bf16 [M, H, 128]");
    const bool has_pre = pn.has_value();
    TORCH_CHECK(!has_pre || (pre.has_value() && pg.has_value() && pb.has_value() && pre->dtype() == torch::kBFloat16
                && pre->stride(1) == 1 && pg->dtype() == torch::kFloat && pb->dtype() == torch::kFloat
                && pn->dtype() == torch::kInt), "parked rows: pre bf16 [P, C], pg / pb fp32 [P, H], pn int32 [1]");
    launch(gdn_rows_kernel, dim3(H, 8), dim3(128), st, (const __nv_bfloat16*)qkv.data_ptr(), (int)qkv.stride(0),
        (int)koff, (int)voff, (const float*)g.data_ptr(), (const float*)beta.data_ptr(), (float*)S.data_ptr(),
        (float*)stk.data_ptr(), (const int*)ctl.data_ptr(), M, (__nv_bfloat16*)out.data_ptr(), H, (int)rep, write ? 1 : 0,
        has_pre ? (const __nv_bfloat16*)pre->data_ptr() : (const __nv_bfloat16*)nullptr, has_pre ? (int)pre->stride(0) : 0,
        has_pre ? (const float*)pg->data_ptr() : (const float*)nullptr, has_pre ? (const float*)pb->data_ptr() : (const float*)nullptr,
        has_pre ? (const int*)pn->data_ptr() : (const int*)nullptr);
}

void seq_ple_gate(torch::Tensor key, torch::Tensor qry, torch::Tensor val, torch::Tensor gv, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(key.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = val.size(0), Hd = val.size(1);
    TORCH_CHECK(key.is_contiguous() && qry.is_contiguous() && val.is_contiguous() && gv.is_contiguous(), "contiguous");
    TORCH_CHECK(key.numel() == (int64_t)M * hc * Hd && qry.numel() == key.numel() && gv.numel() == key.numel(), "shapes");
    launch(ple_gate_kernel, dim3(M, hc), dim3(1024), st, (const __nv_bfloat16*)key.data_ptr(),
        (const __nv_bfloat16*)qry.data_ptr(), (const __nv_bfloat16*)val.data_ptr(), (__nv_bfloat16*)gv.data_ptr(),
        Hd, (float)std::sqrt((double)Hd));
}

void seq_ple_conv(torch::Tensor gvn, torch::Tensor gv, torch::Tensor ps, torch::Tensor w, torch::Tensor src, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(gvn.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = gvn.size(0), C = gvn.size(1), slen = ps.size(-1);
    TORCH_CHECK(gvn.is_contiguous() && gv.is_contiguous() && y.is_contiguous() && ps.is_contiguous() && w.is_contiguous(), "contiguous");
    TORCH_CHECK(ps.numel() == (int64_t)C * slen && w.numel() == (int64_t)C * 4 && w.dtype() == torch::kBFloat16, "state [C, slen], w bf16 [C, 4]");
    TORCH_CHECK(src.dtype() == torch::kInt && src.numel() == M * 3, "src int32 [M, 3]");
    launch(ple_conv_kernel, dim3((C + 255) / 256, M), dim3(256), st, (const __nv_bfloat16*)gvn.data_ptr(),
        (const __nv_bfloat16*)gv.data_ptr(), (const __nv_bfloat16*)ps.data_ptr(), slen, (const __nv_bfloat16*)w.data_ptr(),
        (const int*)src.data_ptr(), (__nv_bfloat16*)y.data_ptr(), C);
}

void seq_qsa_pool(torch::Tensor ikc, torch::Tensor slots, torch::Tensor posb, torch::Tensor kln, torch::Tensor cos,
                  torch::Tensor sin, torch::Tensor out, double eps) {
    const at::cuda::OptionalCUDAGuard guard(ikc.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int NB = posb.numel();
    TORCH_CHECK(ikc.dtype() == torch::kBFloat16 && ikc.is_contiguous() && ikc.size(1) == 128, "indexer keys bf16 [cap, 128]");
    TORCH_CHECK(slots.dtype() == torch::kInt && slots.is_contiguous() && slots.numel() == NB * 4, "slots int32 [NB, 4]");
    TORCH_CHECK(posb.dtype() == torch::kInt && posb.is_contiguous(), "posb int32 [NB]");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)NB * 128, "out bf16 [NB, 128]");
    TORCH_CHECK(cos.dtype() == torch::kBFloat16 && cos.size(1) <= 128, "cos/sin bf16 [L, ROT]");
    if (NB == 0) return;
    launch(qsa_pool_kernel, dim3(NB), dim3(128), st, (const __nv_bfloat16*)ikc.data_ptr(), (const int*)slots.data_ptr(),
        (const int*)posb.data_ptr(), (const __nv_bfloat16*)kln.data_ptr(), (const __nv_bfloat16*)cos.data_ptr(),
        (const __nv_bfloat16*)sin.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)cos.size(1), (float)eps);
}

void seq_qsa_pool_multi(torch::Tensor E, int64_t NB, torch::Tensor cos, torch::Tensor sin, double eps) {
    const at::cuda::OptionalCUDAGuard guard(E.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(E.dtype() == torch::kLong && E.is_contiguous() && E.dim() == 2 && E.size(1) == 16, "entries int64 [n, 16]");
    TORCH_CHECK(cos.dtype() == torch::kBFloat16 && cos.size(1) <= 128, "cos/sin bf16 [L, ROT]");
    if (E.size(0) == 0 || NB == 0) return;
    launch(qsa_pool_multi_kernel, dim3((unsigned)NB, (unsigned)E.size(0)), dim3(128), st, (const long long*)E.data_ptr(),
        (const __nv_bfloat16*)cos.data_ptr(), (const __nv_bfloat16*)sin.data_ptr(), (int)cos.size(1), (float)eps);
}

void seq_qsa_topk(torch::Tensor sc, torch::Tensor tail, torch::Tensor sel, int64_t topk) {
    const at::cuda::OptionalCUDAGuard guard(sc.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = sc.size(0), NB = sc.size(1);
    TORCH_CHECK(sc.dtype() == torch::kFloat && sc.is_contiguous() && NB <= 8192 && (NB & (NB - 1)) == 0 && NB >= topk,
                "scores fp32 [M, NB], NB a power of two <= 8192");
    TORCH_CHECK(tail.dtype() == torch::kInt && tail.is_contiguous() && sel.dtype() == torch::kInt && sel.is_contiguous()
                && sel.size(1) >= 4 * topk + 3, "tail int32 [M, 3], sel int32 [M, >= 4 topk + 3]");
    const size_t smem = (size_t)NB * 8 + (size_t)topk * 4;
    cudaFuncSetAttribute(qsa_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(M); cfg.blockDim = dim3(1024); cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, qsa_topk_kernel, (const float*)sc.data_ptr(), NB, (const int*)tail.data_ptr(),
                                       (int*)sel.data_ptr(), (int)sel.size(1), (int)topk);
    TORCH_CHECK(e == cudaSuccess, "qsa_topk launch: ", cudaGetErrorString(e));
}

void seq_qsa_scores(torch::Tensor iq, torch::Tensor pool, torch::Tensor own, torch::Tensor qctl, torch::Tensor sc) {
    const at::cuda::OptionalCUDAGuard guard(iq.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = iq.size(0), NH = iq.size(1), NB = sc.size(1), NO = own.size(1);
    TORCH_CHECK(iq.dtype() == torch::kBFloat16 && iq.is_contiguous() && iq.size(2) == 128, "iq bf16 [M, NH, 128]");
    TORCH_CHECK(pool.is_contiguous() && own.is_contiguous() && own.size(0) == M, "pool [*, 128], own [M, NO, 128]");
    TORCH_CHECK(qctl.dtype() == torch::kInt && qctl.is_contiguous() && qctl.numel() == M * 4, "qctl int32 [M, 4]");
    TORCH_CHECK(sc.dtype() == torch::kFloat && sc.is_contiguous() && sc.size(0) == M, "sc fp32 [M, NB]");
    launch(qsa_scores_kernel, dim3(M, (NB + 7) / 8), dim3(256), st, (const __nv_bfloat16*)iq.data_ptr(),
        (const __nv_bfloat16*)pool.data_ptr(), (const __nv_bfloat16*)own.data_ptr(), NO, (const int*)qctl.data_ptr(),
        (float*)sc.data_ptr(), NB, NH);
}
