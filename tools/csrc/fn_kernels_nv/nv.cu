
// ------------------------------------------------------------------------- NVIDIA FP4-Local's weight types
// Appended to fn_kernels' own source (tools/fn_kernels_nv.py builds the two texts as one module), so its kernels run
// unchanged on the two weight types NVIDIA FP4-Local stores and the EXL3 packs never have: bf16 GEMV weights (the
// head beyond the skinny kernels' rows; exact in fp32) and fp32 norm weights (the hyper-connection norms). The shared
// module and its build keep their bytes.
namespace {   // fn_kernels' own (its kernels and helpers live in it)
template <> struct WVec<__nv_bfloat16> {
    static constexpr int VEC = 8;
    static __device__ __forceinline__ void load(const __nv_bfloat16* p, float* w) {
        const uint4 r = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
        for (int i = 0; i < 4; ++i) { w[2 * i] = __low2float(h[i]); w[2 * i + 1] = __high2float(h[i]); }
    }
};
template <> __device__ __forceinline__ float wf(const float* w, int i) { return w[i]; }
}  // namespace

void nv_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0, "x bf16 [M, K]");
    TORCH_CHECK(w.dtype() == torch::kBFloat16 && w.dim() == 2 && w.is_contiguous(), "w bf16 [N, K] contiguous");
    TORCH_CHECK((y.dtype() == torch::kFloat || y.dtype() == torch::kBFloat16) && y.stride(1) == 1, "y fp32 or bf16 [M, N]");
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N && K % 8 == 0, "shapes, K % 8");
    if (y.dtype() == torch::kFloat) gemv_dispatch<__nv_bfloat16, float>(x, w, nullptr, y, M, N, K, st);
    else gemv_dispatch<__nv_bfloat16, __nv_bfloat16>(x, w, nullptr, y, M, N, K, st);
}

void nv_hc_norm(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t G, double eps) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), N = x.size(1);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && y.dtype() == torch::kBFloat16 && y.stride(1) == 1, "bf16 rows");
    TORCH_CHECK(N % G == 0 && G <= 1024 * 4 && w.numel() == N && w.is_contiguous() && w.dtype() == torch::kFloat, "groups, fp32 w");
    launch(hc_norm_kernel<float>, dim3(M, N / G), dim3(1024), 0, st, (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const float*)w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), (int)G, (float)eps);
}

void nv_inject_norm(torch::Tensor h0, torch::Tensor out, torch::Tensor d, int64_t R, torch::Tensor w,
                    torch::Tensor hn, torch::Tensor nn, int64_t hc, double eps) {
    const at::cuda::OptionalCUDAGuard guard(h0.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = h0.size(0), N = h0.size(1), G = N / hc;
    TORCH_CHECK(h0.dtype() == torch::kBFloat16 && h0.is_contiguous() && hn.is_contiguous() && nn.is_contiguous(), "h0, hn, nn contiguous bf16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.stride(1) == 1 && out.size(1) == G, "out bf16 [M, G]");
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1 && d.size(1) >= R + hc, "d fp32 [M, R + hc]");
    TORCH_CHECK(G <= 1024 * 4 && w.dtype() == torch::kFloat, "group, fp32 w");
    launch(inject_norm_kernel<float>, dim3(M, hc), dim3(1024), 0, st, (const __nv_bfloat16*)h0.data_ptr(), N,
        (const __nv_bfloat16*)out.data_ptr(), (int)out.stride(0), (const float*)d.data_ptr(), (int)d.stride(0), (int)R,
        (const float*)w.data_ptr(), (__nv_bfloat16*)hn.data_ptr(), (__nv_bfloat16*)nn.data_ptr(), G, 1.0f / (float)hc, (float)eps);
}

void nv_combine_inject_norm(torch::Tensor h0, torch::Tensor yd, torch::Tensor ids, torch::Tensor wts, torch::Tensor d,
                            int64_t R, torch::Tensor w, torch::Tensor hn, torch::Tensor nn, int64_t hc, double eps) {
    const at::cuda::OptionalCUDAGuard guard(h0.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = h0.size(0), N = h0.size(1), G = N / hc, S = ids.numel() / M;
    TORCH_CHECK(h0.dtype() == torch::kBFloat16 && h0.is_contiguous() && hn.is_contiguous() && nn.is_contiguous(), "bf16 rows");
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == M * S && yd.size(1) == G, "yd [M S, G]");
    TORCH_CHECK(d.dtype() == torch::kFloat && d.stride(1) == 1 && d.size(1) >= R + hc, "d fp32 [M, R + hc]");
    TORCH_CHECK(G <= 4096 && S <= 64 && w.dtype() == torch::kFloat, "group, slots, fp32 w");
    launch(combine_inject_norm_kernel<float>, dim3(M, hc), dim3(1024), 0, st, (const __nv_bfloat16*)h0.data_ptr(), N,
        (const float*)yd.data_ptr(), (const int*)ids.data_ptr(), (const float*)wts.data_ptr(), S,
        (const float*)d.data_ptr(), (int)d.stride(0), (int)R, (const float*)w.data_ptr(),
        (__nv_bfloat16*)hn.data_ptr(), (__nv_bfloat16*)nn.data_ptr(), G, 1.0f / (float)hc, (float)eps);
}
