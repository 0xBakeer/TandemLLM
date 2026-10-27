
#define EXL3_TALL_WK_CASES(KB, MT, NTW, WN, OF32, XSM, MINB)                                     \
    switch (wk) {                                                                                \
        case 4: launch_gemm<KB, MT, NTW, WN, 4, EXL3_ST, OF32, XSM, MINB>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
        case 8: launch_gemm<KB, MT, NTW, WN, 8, EXL3_ST, OF32, XSM, MINB>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
        default: TORCH_CHECK(false, "exl3 tall gemm: wk must be 4 or 8, got ", wk);              \
    }
#define EXL3_TALL_CFG_CASES(KB, OF32)                                                            \
    switch (cfg) {                                                                               \
        case 5: EXL3_TALL_WK_CASES(KB, 4, 2, 4, OF32, false, 1)                                  \
        case 6: EXL3_TALL_WK_CASES(KB, 4, 4, 2, OF32, false, 1)                                  \
        default: TORCH_CHECK(false, "exl3 tall gemm: bad cfg ", cfg);                            \
    }
#define EXL3_TALL_KB_CASES(OF32)                                                                 \
    switch (kb) {                                                                                \
        case 2: EXL3_TALL_CFG_CASES(2, OF32)                                                     \
        case 3: EXL3_TALL_CFG_CASES(3, OF32)                                                     \
        case 4: EXL3_TALL_CFG_CASES(4, OF32)                                                     \
        case 5: EXL3_TALL_CFG_CASES(5, OF32)                                                     \
        case 6: EXL3_TALL_CFG_CASES(6, OF32)                                                     \
        case 7: EXL3_TALL_CFG_CASES(7, OF32)                                                     \
        case 8: EXL3_TALL_CFG_CASES(8, OF32)                                                     \
        default: TORCH_CHECK(false, "exl3 tall gemm: K bits must be 2..8, got ", kb);            \
    }

void exl3_gemm_tall(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y,
                    int64_t kb, int64_t cfg, int64_t wk, int64_t S, int64_t b1, int64_t b2) {
    const at::cuda::OptionalCUDAGuard guard(xh.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3, "xh: fp16 [G, M, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3, "trellis: int16 [Kin/16, N/16, 16K]");
    TORCH_CHECK(t.size(2) == 16 * kb && t.stride(2) == 1 && t.stride(1) == 16 * kb, "trellis tiles must be dense");
    const int M = xh.size(1), Kin = xh.size(2), N = t.size(1) * 16;
    const int ldt = t.stride(0) / t.stride(1);
    TORCH_CHECK(t.size(0) * 16 == Kin && N % 128 == 0 && Kin % 128 == 0, "shapes");
    TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "y [M, N]");
    TORCH_CHECK(M >= 1, "rows");
    TORCH_CHECK(S >= 1 && S <= 16, "cluster size");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh fp16 [N]");
    if (y.dtype() == torch::kFloat) { EXL3_TALL_KB_CASES(true) }
    else { TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32"); EXL3_TALL_KB_CASES(false) }
}
