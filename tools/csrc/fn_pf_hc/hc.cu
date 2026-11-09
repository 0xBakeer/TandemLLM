#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// SPD-249 P2: a prompt chunk's hyper-connection mixer in two launches a site, each output the bits of the path it
// replaces (engine/models/flashnext/engine.py `_mix_pre` on fn_prefill_kernels' fp32 `mm_rows`):
//
//   down   D = n @ (q s)^T: each output one fp32 fma chain over k = 0 .. K-1 from 0, the library fp32 GEMM's sum
//          (tests/gpu/test_fn_pf_hc_gpu.py holds it to the library at every row count of a chunk); the weight
//          float(q) * s, `Int8Rows.dequant`'s one rounding; n bf16 read as it is (its fp32 form is exact). The
//          epilogue writes D and A = bf16(silu(bf16(D) * (1 / hc))) for the first R columns (torch's
//          `silu(d.to(bf16).float() / hc).to(bf16)`: a division by a scalar is a multiply by its fp32 reciprocal,
//          silu x / (1 + exp(-x)) in fp32).
//   up     u = A @ (q s)^T, the same fma chain over the R columns, never stored: the epilogue is mix_tail's,
//          x = bf16((((0 + p0) + p1) + p2) + p3) * (1 / hc)) with p_s = bf16(bf16(sigmoid(bf16(u_s))) n_s).
//
// One SIMT tile body: a CTA owns BM rows and BN weight rows (outputs), BK-deep k steps through two shared-memory
// buffers in fp32 (activations converted, weights dequantised on the way in), a thread TM x TN outputs. The tile
// shape never changes a sum (each output's chain is its own, k ascending), so any shape gives the same bits.

namespace {

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

constexpr int MODE_DOWN = 0, MODE_UP = 1;

// MODE_DOWN: weight row of tile column c = n0 + c; a thread's columns tx TN .. + TN.
// MODE_UP: the tile's BN = 4 BJ columns are 4 streams of BJ: column c = s BJ + jl reads weight row s H + j0 + jl;
// a thread's TN = 4 TJ columns are its TJ j's in each stream (the epilogue needs all four streams of a j).
template <int BM, int BN, int TM, int TN, int BK, int MODE>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) hc_tile_kernel(
        const __nv_bfloat16* __restrict__ X, int ldx, const int8_t* __restrict__ Q, const float* __restrict__ S,
        int ngs, int G, int M, int N, int K, float inv_hc,
        float* __restrict__ D, int ldd, __nv_bfloat16* __restrict__ A, int lda, int R,
        const __nv_bfloat16* __restrict__ Nn, int ldn, __nv_bfloat16* __restrict__ Xo, int ldxo, int H) {
    constexpr int NTX = BN / TN, NTY = BM / TM, NT = NTX * NTY;
    constexpr int BMP = BM + 4, BNP = BN + 4;
    constexpr int AV = BM * BK / 8, BV = BN * BK / 16;          // 16-byte vectors a tile
    constexpr int LA = (AV + NT - 1) / NT, LB = (BV + NT - 1) / NT;
    constexpr int BJ = BN / 4, TJ = TN / 4;
    __shared__ __align__(16) float As[2][BK][BMP];
    __shared__ __align__(16) float Bs[2][BK][BNP];
    const int tid = threadIdx.x, tx = tid % NTX, ty = tid / NTX;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * (MODE == MODE_UP ? BJ : BN);   // MODE_UP: j0

    uint4 ra[LA];
    uint4 rb[LB];
    float rs[LB];
    auto wrow = [&](int c) -> int {                              // the weight row of tile column c, -1 past N
        if (MODE == MODE_UP) return (c / BJ) * H + n0 + (c % BJ);
        return n0 + c < N ? n0 + c : -1;
    };
    auto load = [&](int k0) {
#pragma unroll
        for (int l = 0; l < LA; ++l) {
            const int v = tid + l * NT;
            ra[l] = make_uint4(0u, 0u, 0u, 0u);
            if (v < AV) {
                const int r = v / (BK / 8), kc = (v % (BK / 8)) * 8;
                if (m0 + r < M) ra[l] = *reinterpret_cast<const uint4*>(X + (size_t)(m0 + r) * ldx + k0 + kc);
            }
        }
#pragma unroll
        for (int l = 0; l < LB; ++l) {
            const int v = tid + l * NT;
            rb[l] = make_uint4(0u, 0u, 0u, 0u);
            rs[l] = 0.f;
            if (v < BV) {
                const int c = v / (BK / 16), kc = (v % (BK / 16)) * 16;
                const int w = wrow(c);
                if (w >= 0) {
                    rb[l] = *reinterpret_cast<const uint4*>(Q + (size_t)w * K + k0 + kc);
                    rs[l] = S[(size_t)w * ngs + (k0 + kc) / G];
                }
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int l = 0; l < LA; ++l) {
            const int v = tid + l * NT;
            if (v < AV) {
                const int r = v / (BK / 8), kc = (v % (BK / 8)) * 8;
                const uint4 q = ra[l];
                As[buf][kc + 0][r] = bf_lo(q.x); As[buf][kc + 1][r] = bf_hi(q.x);
                As[buf][kc + 2][r] = bf_lo(q.y); As[buf][kc + 3][r] = bf_hi(q.y);
                As[buf][kc + 4][r] = bf_lo(q.z); As[buf][kc + 5][r] = bf_hi(q.z);
                As[buf][kc + 6][r] = bf_lo(q.w); As[buf][kc + 7][r] = bf_hi(q.w);
            }
        }
#pragma unroll
        for (int l = 0; l < LB; ++l) {
            const int v = tid + l * NT;
            if (v < BV) {
                const int c = v / (BK / 16), kc = (v % (BK / 16)) * 16;
                const int8_t* q = reinterpret_cast<const int8_t*>(&rb[l]);
                const float s = rs[l];
#pragma unroll
                for (int e = 0; e < 16; ++e) Bs[buf][kc + e][c] = (float)q[e] * s;   // Int8Rows.dequant's product
            }
        }
    };

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int nk = K / BK;
    load(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < nk; ++t) {
        const int buf = t & 1;
        if (t + 1 < nk) load((t + 1) * BK);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
            if constexpr (TM % 4 == 0) {
#pragma unroll
                for (int i = 0; i < TM; i += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(&As[buf][kk][ty * TM + i]);
                    a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
                }
            } else {
#pragma unroll
                for (int i = 0; i < TM; ++i) a[i] = As[buf][kk][ty * TM + i];
            }
            if constexpr (MODE == MODE_UP) {
#pragma unroll
                for (int s = 0; s < 4; ++s)
#pragma unroll
                    for (int jj = 0; jj < TJ; ++jj) b[s * TJ + jj] = Bs[buf][kk][s * BJ + tx * TJ + jj];
            } else if constexpr (TN % 4 == 0) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * TN + j]);
                    b[j] = v.x; b[j + 1] = v.y; b[j + 2] = v.z; b[j + 3] = v.w;
                }
            } else {
#pragma unroll
                for (int j = 0; j < TN; ++j) b[j] = Bs[buf][kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (t + 1 < nk) store(buf ^ 1);
        __syncthreads();
    }

    if constexpr (MODE == MODE_DOWN) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int m = m0 + ty * TM + i;
            if (m >= M) continue;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int n = n0 + tx * TN + j;
                if (n >= N) continue;
                const float v = acc[i][j];
                D[(size_t)m * ldd + n] = v;
                if (n < R) {
                    const float z = bfr(v) * inv_hc;
                    A[(size_t)m * lda + n] = __float2bfloat16_rn(z / (1.0f + ::expf(-z)));
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int m = m0 + ty * TM + i;
            if (m >= M) continue;
#pragma unroll
            for (int jj = 0; jj < TJ; ++jj) {
                const int j = n0 + tx * TJ + jj;
                const __nv_bfloat16* nn = Nn + (size_t)m * ldn + j;
                float x = 0.f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float sg = bfr(1.0f / (1.0f + ::expf(-bfr(acc[i][s * TJ + jj]))));
                    x += bfr(sg * __bfloat162float(nn[(size_t)s * H]));
                }
                Xo[(size_t)m * ldxo + j] = __float2bfloat16_rn(x * inv_hc);
            }
        }
    }
}


// inject_norm (tools/csrc/fn_kernels/kernels.cu `inject_norm_kernel`, its launch of 1,024 threads a (row, stream)) bit
// for bit, a row's four streams in one CTA of 4 x 128 threads with 16-byte loads and stores. The kernel it replaces
// gives thread t of 1,024 the elements t, t + 1024, t + 2048 (.. < G) of its stream, sums their squares in that order
// (fp32 fma from 0), adds a warp's 32 partials in the xor tree (offsets 16, 8, 4, 2, 1, lane 0's value) and the 32
// warps' sums in order. Here a thread holds those 1,024 virtual threads' partials 8 at a time (virtual thread 8 r + e
// is thread r's register e: the same elements in the same order), rebuilds lane 0's tree of virtual warp r / 4 from
// its group of 4 threads (offset 16 = thread r ^ 2, 8 = r ^ 1, then 4, 2, 1 inside thread r's registers), and adds the
// warps in order: the same sum, the same r, the same bits for h and n.
template <typename TN> __device__ __forceinline__ float wload(const TN* w, int i);
template <> __device__ __forceinline__ float wload(const __half* w, int i) { return __half2float(w[i]); }
template <> __device__ __forceinline__ float wload(const __nv_bfloat16* w, int i) { return __bfloat162float(w[i]); }

constexpr int INJ_T = 128;   // threads a stream

template <typename TN, int NSEG>
__global__ void __launch_bounds__(4 * INJ_T) pf_inject_norm_kernel(const __nv_bfloat16* __restrict__ H0, int ldh,
        const __nv_bfloat16* __restrict__ OUT, int ldo, const float* __restrict__ D, int ldd, int R,
        const TN* __restrict__ Wn, __nv_bfloat16* __restrict__ HN, __nv_bfloat16* __restrict__ NN, float* __restrict__ N32,
        int G, float inv_hc, float eps) {
    __shared__ float sh[4][32];
    const int m = blockIdx.x, s = threadIdx.x / INJ_T, r = threadIdx.x % INJ_T;
    const float sg = 1.0f / (1.0f + ::expf(-bfr(bfr(D[(size_t)m * ldd + R + s]) * inv_hc)));
    const float inj = 2.0f * bfr(sg);
    const __nv_bfloat16* h0 = H0 + (size_t)m * ldh + s * G;
    const __nv_bfloat16* o = OUT + (size_t)m * ldo;
    float v[NSEG][8];
    float ss[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) ss[e] = 0.f;
#pragma unroll
    for (int c = 0; c < NSEG; ++c) {
        const int i = c * 1024 + 8 * r;
        if (i < G) {
            const uint4 hq = *reinterpret_cast<const uint4*>(h0 + i);
            const uint4 oq = *reinterpret_cast<const uint4*>(o + i);
            const uint32_t hw[4] = {hq.x, hq.y, hq.z, hq.w}, ow[4] = {oq.x, oq.y, oq.z, oq.w};
            uint32_t pk[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float a0 = bfr(bf_lo(hw[e]) + bfr(bf_lo(ow[e]) * inj));
                const float a1 = bfr(bf_hi(hw[e]) + bfr(bf_hi(ow[e]) * inj));
                v[c][2 * e] = a0; v[c][2 * e + 1] = a1;
                ss[2 * e] = fmaf(a0, a0, ss[2 * e]);
                ss[2 * e + 1] = fmaf(a1, a1, ss[2 * e + 1]);
                const __nv_bfloat162 b2 = __floats2bfloat162_rn(a0, a1);
                pk[e] = *reinterpret_cast<const uint32_t*>(&b2);
            }
            *reinterpret_cast<uint4*>(HN + (size_t)m * ldh + s * G + i) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
    }
    if (Wn == nullptr) return;
    // lane 0's xor tree of virtual warp r / 4: offset 16 (thread r ^ 2), offset 8 (thread r ^ 1), then 4, 2, 1
    float a[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] = ss[e] + __shfl_xor_sync(0xffffffffu, ss[e], 2);
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] = a[e] + __shfl_xor_sync(0xffffffffu, a[e], 1);
    const float c0 = a[0] + a[4], c1 = a[1] + a[5], c2 = a[2] + a[6], c3 = a[3] + a[7];
    const float d0 = c0 + c2, d1 = c1 + c3;
    if ((r & 3) == 0) sh[s][r >> 2] = d0 + d1;
    __syncthreads();
    float t = sh[s][0];
    for (int w = 1; w < 32; ++w) t += sh[s][w];
    const float rr = rsqrtf(t / (float)G + eps);
#pragma unroll
    for (int c = 0; c < NSEG; ++c) {
        const int i = c * 1024 + 8 * r;
        if (i < G) {
            uint32_t pk[4];
            float f[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float n0 = v[c][2 * e] * rr * (1.0f + wload(Wn, s * G + i + 2 * e));
                const float n1 = v[c][2 * e + 1] * rr * (1.0f + wload(Wn, s * G + i + 2 * e + 1));
                const __nv_bfloat162 b2 = __floats2bfloat162_rn(n0, n1);
                pk[e] = *reinterpret_cast<const uint32_t*>(&b2);
                f[2 * e] = __low2float(b2); f[2 * e + 1] = __high2float(b2);
            }
            *reinterpret_cast<uint4*>(NN + (size_t)m * ldh + s * G + i) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            if (N32 != nullptr) {      // n's fp32 form (`n.float()`, exact): the down GEMM's input
                float* q = N32 + (size_t)m * ldh + s * G + i;
                *reinterpret_cast<float4*>(q) = make_float4(f[0], f[1], f[2], f[3]);
                *reinterpret_cast<float4*>(q + 4) = make_float4(f[4], f[5], f[6], f[7]);
            }
        }
    }
}

template <int BM, int BN, int TM, int TN, int BK, int MODE>
void launch(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& s, int N, int Ncols, float inv_hc,
            float* D, int ldd, __nv_bfloat16* A, int lda, int R, const __nv_bfloat16* Nn, int ldn,
            __nv_bfloat16* Xo, int ldxo, int H, cudaStream_t st) {
    const int M = x.size(0), K = x.size(1);
    const int ngs = s.dim() == 1 ? 1 : s.size(1), G = K / ngs;
    TORCH_CHECK(K % BK == 0 && G % BK == 0, "K ", K, " and its scale groups of ", G, ": multiples of ", BK);
    const dim3 grid((M + BM - 1) / BM, (Ncols + (MODE == MODE_UP ? BN / 4 : BN) - 1) / (MODE == MODE_UP ? BN / 4 : BN));
    hc_tile_kernel<BM, BN, TM, TN, BK, MODE><<<grid, (BM / TM) * (BN / TN), 0, st>>>(
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const int8_t*)q.data_ptr(), (const float*)s.data_ptr(),
        ngs, G, M, N, K, inv_hc, D, ldd, A, lda, R, Nn, ldn, Xo, ldxo, H);
}

void check_w(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& s) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0
                && ((uintptr_t)x.data_ptr() % 16) == 0, "x bf16 [M, K], rows 16-byte aligned");
    TORCH_CHECK(q.dtype() == torch::kChar && q.is_contiguous() && q.size(1) == x.size(1) && x.size(1) % 16 == 0
                && ((uintptr_t)q.data_ptr() % 16) == 0, "q int8 [N, K]");
    TORCH_CHECK(s.dtype() == torch::kFloat && s.is_contiguous() && s.size(0) == q.size(0)
                && (s.dim() == 1 || (s.dim() == 2 && x.size(1) % s.size(1) == 0)), "s fp32 [N] or [N, groups]");
}

}  // namespace

// D [M, N] fp32 and A [M, R] bf16 of the down|inject weight; `tile` picks the CTA shape (the bits are every shape's)
void pf_hc_down(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor d, torch::Tensor a, int64_t hc,
                int64_t tile) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    check_w(x, q, s);
    const int M = x.size(0), N = q.size(0), R = a.size(1);
    TORCH_CHECK(d.dtype() == torch::kFloat && d.is_contiguous() && d.size(0) == M && d.size(1) == N, "d fp32 [M, N]");
    TORCH_CHECK(a.dtype() == torch::kBFloat16 && a.is_contiguous() && a.size(0) == M && R <= N, "a bf16 [M, R]");
    if (M == 0) return;
    float* D = (float*)d.data_ptr();
    __nv_bfloat16* A = (__nv_bfloat16*)a.data_ptr();
    const float ih = 1.0f / (float)hc;
    switch (tile) {
        case 0: launch<64, 108, 8, 9, 16, MODE_DOWN>(x, q, s, N, N, ih, D, N, A, R, R, nullptr, 0, nullptr, 0, 0, st); break;
        case 1: launch<128, 108, 8, 9, 16, MODE_DOWN>(x, q, s, N, N, ih, D, N, A, R, R, nullptr, 0, nullptr, 0, 0, st); break;
        case 2: launch<128, 128, 8, 8, 16, MODE_DOWN>(x, q, s, N, N, ih, D, N, A, R, R, nullptr, 0, nullptr, 0, 0, st); break;
        case 3: launch<64, 64, 8, 8, 16, MODE_DOWN>(x, q, s, N, N, ih, D, N, A, R, R, nullptr, 0, nullptr, 0, 0, st); break;
        case 4: launch<64, 108, 8, 9, 32, MODE_DOWN>(x, q, s, N, N, ih, D, N, A, R, R, nullptr, 0, nullptr, 0, 0, st); break;
        case 5: launch<32, 108, 4, 9, 32, MODE_DOWN>(x, q, s, N, N, ih, D, N, A, R, R, nullptr, 0, nullptr, 0, 0, st); break;
        default: TORCH_CHECK(false, "pf_hc_down: tile ", tile);
    }
}

// x [M, H] bf16 = mix_tail(a @ (q s)^T, n) for the up weight [hc H, R] (hc 4)
void pf_hc_up_tail(torch::Tensor a, torch::Tensor q, torch::Tensor s, torch::Tensor n, torch::Tensor x, int64_t hc,
                   int64_t tile) {
    const at::cuda::OptionalCUDAGuard guard(a.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    check_w(a, q, s);
    const int M = a.size(0), H = x.size(1);
    TORCH_CHECK(hc == 4 && q.size(0) == 4 * H, "q [4 H, R]");
    TORCH_CHECK(n.dtype() == torch::kBFloat16 && n.stride(1) == 1 && n.size(0) == M && n.size(1) == 4 * H, "n bf16 [M, 4 H]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.size(0) == M, "x bf16 [M, H]");
    if (M == 0) return;
    const __nv_bfloat16* Nn = (const __nv_bfloat16*)n.data_ptr();
    __nv_bfloat16* Xo = (__nv_bfloat16*)x.data_ptr();
    const int ldn = n.stride(0), ldx = x.stride(0), N = q.size(0);
    const float ih = 1.0f / (float)hc;
    switch (tile) {
        case 0: TORCH_CHECK(H % 32 == 0, "H"); launch<128, 128, 8, 8, 16, MODE_UP>(a, q, s, N, H, ih, nullptr, 0, nullptr, 0, 0, Nn, ldn, Xo, ldx, H, st); break;
        case 1: TORCH_CHECK(H % 32 == 0, "H"); launch<64, 128, 8, 8, 16, MODE_UP>(a, q, s, N, H, ih, nullptr, 0, nullptr, 0, 0, Nn, ldn, Xo, ldx, H, st); break;
        case 2: TORCH_CHECK(H % 64 == 0, "H"); launch<128, 256, 8, 16, 16, MODE_UP>(a, q, s, N, H, ih, nullptr, 0, nullptr, 0, 0, Nn, ldn, Xo, ldx, H, st); break;
        case 3: TORCH_CHECK(H % 32 == 0, "H"); launch<128, 128, 8, 8, 32, MODE_UP>(a, q, s, N, H, ih, nullptr, 0, nullptr, 0, 0, Nn, ldn, Xo, ldx, H, st); break;
        case 4: TORCH_CHECK(H % 16 == 0, "H"); launch<64, 64, 8, 4, 16, MODE_UP>(a, q, s, N, H, ih, nullptr, 0, nullptr, 0, 0, Nn, ldn, Xo, ldx, H, st); break;
        default: TORCH_CHECK(false, "pf_hc_up_tail: tile ", tile);
    }
}

// inject_norm's (h, n) for hc 4 streams of G (G a multiple of 8, at most 4,096); w None: h only (n is h)
void pf_inject_norm(torch::Tensor h0, torch::Tensor out, torch::Tensor d, int64_t R, c10::optional<torch::Tensor> w,
                    torch::Tensor hn, torch::Tensor nn, c10::optional<torch::Tensor> n32, int64_t hc, double eps) {
    const at::cuda::OptionalCUDAGuard guard(h0.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = h0.size(0), N = h0.size(1), G = N / hc;
    TORCH_CHECK(hc == 4 && N % 4 == 0 && G % 8 == 0 && G <= 4096, "4 streams of G, G a multiple of 8 up to 4,096");
    TORCH_CHECK(h0.dtype() == torch::kBFloat16 && h0.is_contiguous() && hn.is_contiguous() && nn.is_contiguous()
                && hn.dtype() == torch::kBFloat16 && nn.dtype() == torch::kBFloat16, "h0, hn, nn contiguous bf16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.stride(1) == 1 && out.size(1) == G && out.stride(0) % 8 == 0
                && ((uintptr_t)out.data_ptr() % 16) == 0, "out bf16 [M, G], 16-byte rows");
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1 && d.size(1) >= R + hc, "d fp32 [M, R + hc]");
    TORCH_CHECK(!n32.has_value() || (w.has_value() && n32->dtype() == torch::kFloat && n32->is_contiguous()
                && n32->size(0) == M && n32->size(1) == N), "n32 fp32 [M, hc G] (with a norm weight)");
    if (M == 0) return;
    float* N32 = n32.has_value() ? (float*)n32->data_ptr() : nullptr;
    const float inv = 1.0f / (float)hc;
    const int nseg = (G + 1023) / 1024;
    const bool half = !w.has_value() || w->dtype() == torch::kHalf;
    const void* wp = w.has_value() ? w->data_ptr() : nullptr;
#define PF_INJ(NS) \
    if (half) pf_inject_norm_kernel<__half, NS><<<M, 4 * INJ_T, 0, st>>>((const __nv_bfloat16*)h0.data_ptr(), N, \
        (const __nv_bfloat16*)out.data_ptr(), (int)out.stride(0), (const float*)d.data_ptr(), (int)d.stride(0), (int)R, \
        (const __half*)wp, (__nv_bfloat16*)hn.data_ptr(), (__nv_bfloat16*)nn.data_ptr(), N32, G, inv, (float)eps); \
    else pf_inject_norm_kernel<__nv_bfloat16, NS><<<M, 4 * INJ_T, 0, st>>>((const __nv_bfloat16*)h0.data_ptr(), N, \
        (const __nv_bfloat16*)out.data_ptr(), (int)out.stride(0), (const float*)d.data_ptr(), (int)d.stride(0), (int)R, \
        (const __nv_bfloat16*)wp, (__nv_bfloat16*)hn.data_ptr(), (__nv_bfloat16*)nn.data_ptr(), N32, G, inv, (float)eps);
    switch (nseg) {
        case 1: PF_INJ(1) break;
        case 2: PF_INJ(2) break;
        case 3: PF_INJ(3) break;
        default: PF_INJ(4) break;
    }
#undef PF_INJ
}

namespace {
// a = float(bf16(silu(bf16(d) * (1 / hc)))) for d's first R columns: torch's `silu(d.to(bf16).float() / hc).to(bf16)`
// and `mm_rows`' `.float()` of it in one pass (a division by a scalar is torch's multiply by its fp32 reciprocal)
__global__ void __launch_bounds__(256) pf_mix_a_kernel(const float* __restrict__ D, int ldd, float* __restrict__ A, int R,
                                                       float inv_hc, long total) {
    const long e = (long)blockIdx.x * 256 + threadIdx.x;
    if (e >= total) return;
    const long m = e / R, r = e % R;
    const float z = bfr(D[m * ldd + r]) * inv_hc;
    A[e] = bfr(z / (1.0f + ::expf(-z)));
}
}  // namespace

void pf_mix_a(torch::Tensor d, torch::Tensor a, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(d.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1, "d fp32 [M, >= R]");
    TORCH_CHECK(a.dtype() == torch::kFloat && a.is_contiguous() && a.size(0) == d.size(0) && a.size(1) <= d.size(1), "a fp32 [M, R]");
    const long total = a.numel();
    if (total == 0) return;
    pf_mix_a_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>((const float*)d.data_ptr(), (int)d.stride(0),
        (float*)a.data_ptr(), (int)a.size(1), 1.0f / (float)hc, total);
}

namespace {
// y[m, n] = one fp32 fma chain over k = 0 .. K-1 from 0 of x[m, k] w[n, k] (x [M, K], w [N, K] fp32 contiguous):
// the reference order the up mixer's fused launch keeps; 16 x 16 outputs a CTA, K through shared memory in 16s
__global__ void __launch_bounds__(256) pf_chain_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                                       float* __restrict__ Y, int M, int N, int K) {
    __shared__ float xs[16][17], ws[16][17];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int m = blockIdx.y * 16 + ty, n = blockIdx.x * 16 + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int xm = blockIdx.y * 16 + ty, wn = blockIdx.x * 16 + ty;
        xs[ty][tx] = (xm < M && k0 + tx < K) ? X[(size_t)xm * K + k0 + tx] : 0.f;
        ws[ty][tx] = (wn < N && k0 + tx < K) ? W[(size_t)wn * K + k0 + tx] : 0.f;
        __syncthreads();
        const int kn = min(16, K - k0);
        for (int k = 0; k < kn; ++k) acc = fmaf(xs[ty][k], ws[tx][k], acc);
        __syncthreads();
    }
    if (m < M && n < N) Y[(size_t)m * N + n] = acc;
}
}  // namespace

void pf_chain(torch::Tensor x, torch::Tensor w, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kFloat && w.dtype() == torch::kFloat && y.dtype() == torch::kFloat && x.is_contiguous()
                && w.is_contiguous() && y.is_contiguous() && x.size(1) == w.size(1), "x [M, K], w [N, K], y [M, N] fp32");
    const int M = x.size(0), N = w.size(0), K = x.size(1);
    if (M == 0 || N == 0) return;
    pf_chain_kernel<<<dim3((N + 15) / 16, (M + 15) / 16), 256, 0, st>>>((const float*)x.data_ptr(), (const float*)w.data_ptr(),
        (float*)y.data_ptr(), M, N, K);
}
