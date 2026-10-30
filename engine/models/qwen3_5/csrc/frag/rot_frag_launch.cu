void exl3_rot_in_frag(torch::Tensor x, torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x bf16 [M, Kin]");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.dim() == 2 && suh.is_contiguous(), "suh fp16 [G, Kin]");
    const int M = x.size(0), Kin = x.size(1), G = suh.size(0), Mp = (M + 15) / 16 * 16;
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3 && xh.size(0) == G
                && xh.size(1) == Mp && xh.size(2) == Kin, "xh fp16 [G, M to 16, Kin]");
    TORCH_CHECK(Kin % 128 == 0 && suh.size(1) == Kin, "Kin");
    dim3 grid((Kin / 128 + 3) / 4, Mp, G);
    rot_in_frag_kernel<<<grid, 128, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Mp, Kin);
}

void exl3_rot_in_silu_frag(torch::Tensor y, torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(y.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.dim() == 2 && y.stride(1) == 1, "y bf16 [M, 2 Kin]");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.dim() == 2 && suh.size(0) == 1 && suh.is_contiguous(), "suh fp16 [1, Kin]");
    const int M = y.size(0), Kin = suh.size(1), Mp = (M + 15) / 16 * 16;
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.numel() == (int64_t)Mp * Kin, "xh fp16 [1, M to 16, Kin]");
    TORCH_CHECK(Kin % 128 == 0 && y.size(1) == 2 * Kin, "shapes");
    TORCH_CHECK(y.stride(0) % 4 == 0 && ((uintptr_t)y.data_ptr()) % 8 == 0, "y alignment");
    dim3 grid((Kin / 128 + 3) / 4, Mp, 1);
    rot_in_silu_frag_kernel<<<grid, 128, 0, st>>>((const __nv_bfloat16*)y.data_ptr(), (int)y.stride(0),
        (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Kin);
}
