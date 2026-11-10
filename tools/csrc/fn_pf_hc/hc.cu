#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// SPD-249 P2: a prompt chunk's hyper-connection inject and the mixer's activation in fewer passes, each output the
// bits of the launches it replaces (engine/models/flashnext/engine.py `_inject_pre`, `_mix_pre`). No GEMM here: the
// mixer's products stay with fn_prefill_kernels' `mm_rows`, whatever its mode.

namespace {

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }


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

}  // namespace

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
// a = bf16(silu(bf16(d) * (1 / hc))) for d's first R columns, stored as fp32 (TO float: the fp32 GEMM's input) or as
// bf16 (TO __nv_bfloat16): torch's `silu(d.to(bf16).float() / hc).to(bf16)` (a division by a scalar is torch's multiply
// by its fp32 reciprocal, silu x / (1 + exp(-x)) in fp32) and, for fp32, `mm_rows`' `.float()` of it, in one pass
template <typename TO>
__global__ void __launch_bounds__(256) pf_mix_a_kernel(const float* __restrict__ D, int ldd, TO* __restrict__ A, int R,
                                                       float inv_hc, long total) {
    const long e = (long)blockIdx.x * 256 + threadIdx.x;
    if (e >= total) return;
    const long m = e / R, r = e % R;
    const float z = bfr(D[m * ldd + r]) * inv_hc;
    const __nv_bfloat16 b = __float2bfloat16_rn(z / (1.0f + ::expf(-z)));
    if constexpr (sizeof(TO) == 4) A[e] = __bfloat162float(b);
    else A[e] = b;
}
}  // namespace

void pf_mix_a(torch::Tensor d, torch::Tensor a, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(d.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1, "d fp32 [M, >= R]");
    TORCH_CHECK((a.dtype() == torch::kFloat || a.dtype() == torch::kBFloat16) && a.is_contiguous()
                && a.size(0) == d.size(0) && a.size(1) <= d.size(1), "a fp32 or bf16 [M, R]");
    const long total = a.numel();
    if (total == 0) return;
    const unsigned grid = (unsigned)((total + 255) / 256);
    const float inv = 1.0f / (float)hc;
    if (a.dtype() == torch::kFloat)
        pf_mix_a_kernel<float><<<grid, 256, 0, st>>>((const float*)d.data_ptr(), (int)d.stride(0), (float*)a.data_ptr(),
                                                     (int)a.size(1), inv, total);
    else
        pf_mix_a_kernel<__nv_bfloat16><<<grid, 256, 0, st>>>((const float*)d.data_ptr(), (int)d.stride(0),
                                                             (__nv_bfloat16*)a.data_ptr(), (int)a.size(1), inv, total);
}
