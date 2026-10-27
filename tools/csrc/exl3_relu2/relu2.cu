
// MoEArgs here: [E, 2] pointers (up, down)
template <int KB, int ST, int WARPS>
__device__ __forceinline__ void up_body(const __nv_bfloat16* __restrict__ Xin, int ldx, int m, int p, int e, int cb,
        MoEArgs a, __nv_bfloat16* __restrict__ A, int H, int I, uint32_t* smem) {
    __half* xh = reinterpret_cast<__half*>(smem);                          // [H]
    float* red = reinterpret_cast<float*>(smem + H / 2);                   // [WARPS][128]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemm<KB, ST, WARPS> rg(a.tr[e * 2], I / 16, cb, H, smem + H / 2 + 8 * 128);
    rg.issue_first();
    const __half* suh = a.su[e * 2];
    for (int blk = warp; blk * 128 < H; blk += WARPS) {
        const int k = blk * 128 + 4 * lane;
        const uint2 xr = *reinterpret_cast<const uint2*>(Xin + (size_t)m * ldx + k);
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
        float v[4];
        red_rot<WARPS>(red, lane, v[0], v[1], v[2], v[3]);
        const int n = cb * 128 + 4 * lane;
        const __half* svh = a.sv[e * 2];
        const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        const float s[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
        float r[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float u = fmaxf(bfr(v[q] * RSQRT128 * s[q]), 0.0f);
            r[q] = u * u;
        }
        __nv_bfloat162 o01 = __floats2bfloat162_rn(r[0], r[1]);
        __nv_bfloat162 o23 = __floats2bfloat162_rn(r[2], r[3]);
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(A + (size_t)p * I + n) = o;
    }
}

// grid (I/128, P); A [P, I] bf16
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) relu2_up_kernel(const __nv_bfloat16* __restrict__ Xin, int ldx,
        const int* __restrict__ IDS, int S, MoEArgs a, __nv_bfloat16* __restrict__ A, int H, int I) {
    extern __shared__ uint32_t smem[];
    const int p = blockIdx.y, m = p / S, e = IDS[p], cb = blockIdx.x;
    switch (a.kb[e * 2]) {
        case 2: up_body<2, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 3: up_body<3, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 4: up_body<4, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 5: up_body<5, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 6: up_body<6, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 7: up_body<7, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 8: up_body<8, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 15: up_body<15, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 25: up_body<25, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        case 35: up_body<35, ST, WARPS>(Xin, ldx, m, p, e, cb, a, A, H, I, smem); break;
        default: break;
    }
}

template <int KB, int ST, int WARPS>
__device__ __forceinline__ void down_body(const __nv_bfloat16* __restrict__ A, int p, int e, int cb, MoEArgs a,
        float* __restrict__ YD, int H, int I, uint32_t* smem) {
    __half* xh = reinterpret_cast<__half*>(smem);                          // [I]
    float* red = reinterpret_cast<float*>(smem + I / 2);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemm<KB, ST, WARPS> rg(a.tr[e * 2 + 1], H / 16, cb, I, smem + I / 2 + 8 * 128);
    rg.issue_first();
    const __half* suh = a.su[e * 2 + 1];
    const __nv_bfloat16* ar = A + (size_t)p * I;
    for (int blk = warp; blk * 128 < I; blk += WARPS) {
        const int k = blk * 128 + 4 * lane;
        const uint2 xr = *reinterpret_cast<const uint2*>(ar + k);
        const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
        float v0 = bf_lo(xr.x), v1 = bf_hi(xr.x), v2 = bf_lo(xr.y), v3 = bf_hi(xr.y);
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
        const __half* svh = a.sv[e * 2 + 1];
        const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
        *reinterpret_cast<float4*>(YD + (size_t)p * H + n) = make_float4(
            bfr(v0 * RSQRT128 * __low2float(s01)), bfr(v1 * RSQRT128 * __high2float(s01)),
            bfr(v2 * RSQRT128 * __low2float(s23)), bfr(v3 * RSQRT128 * __high2float(s23)));
    }
}

// grid (H/128, P); YD [P, H] fp32 (bf16 values)
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) relu2_down_kernel(const __nv_bfloat16* __restrict__ A, const int* __restrict__ IDS,
        MoEArgs a, float* __restrict__ YD, int H, int I) {
    extern __shared__ uint32_t smem[];
    const int p = blockIdx.y, e = IDS[p], cb = blockIdx.x;
    switch (a.kb[e * 2 + 1]) {
        case 2: down_body<2, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 3: down_body<3, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 4: down_body<4, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 5: down_body<5, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 6: down_body<6, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 7: down_body<7, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 8: down_body<8, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 15: down_body<15, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 25: down_body<25, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        case 35: down_body<35, ST, WARPS>(A, p, e, cb, a, YD, H, I, smem); break;
        default: break;
    }
}

// SPD-125: a verify block's pairs of one expert in groups of up to MOE_GR (moe_groups_kernel: the pairs in
// ascending expert id, a group the run of one expert), the expert's tiles decoded once a group, row r of the
// group in the mma's lane group g == r (RowGemmG). A pair's input rotation, products, K slices, warp order,
// output rotation and epilogue are up_body's / down_body's, so its bits do not depend on its group. RED
// [WARPS][MOE_GR][128] overlays the rows and the stages once the sums are done.
template <int KB, int ST, int WARPS>
__device__ __forceinline__ void up_group_body(const __nv_bfloat16* __restrict__ Xin, int ldx, int S, const int* __restrict__ grp,
        int cb, MoEArgs a, __nv_bfloat16* __restrict__ A, int H, int I, uint32_t* smem) {
    const int e = grp[0], R = grp[1];
    __half* xh = reinterpret_cast<__half*>(smem);                          // [MOE_GR][H]
    float* red = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemmG<KB, ST, WARPS> rg(a.tr[e * 2], I / 16, cb, H, smem + (size_t)MOE_GR * H / 2);
    rg.issue_first();
    const __half* suh = a.su[e * 2];
    for (int r = 0; r < R; ++r) {
        const int m = grp[2 + r] / S;
        for (int blk = warp; blk * 128 < H; blk += WARPS) {
            const int k = blk * 128 + 4 * lane;
            const uint2 xr = *reinterpret_cast<const uint2*>(Xin + (size_t)m * ldx + k);
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
    const __half* svh = a.sv[e * 2];
    const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
    const float s[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
    for (int r = warp; r < R; r += WARPS) {
        float v[4];
        red_rot_row<WARPS>(red, r, lane, v[0], v[1], v[2], v[3]);
        float q2[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float u = fmaxf(bfr(v[q] * RSQRT128 * s[q]), 0.0f);
            q2[q] = u * u;
        }
        __nv_bfloat162 o01 = __floats2bfloat162_rn(q2[0], q2[1]);
        __nv_bfloat162 o23 = __floats2bfloat162_rn(q2[2], q2[3]);
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(A + (size_t)grp[2 + r] * I + n) = o;
    }
}

// grid (I/128, P): blockIdx.y -> group y (past NG: nothing); A [P, I] bf16
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) relu2_up_group_kernel(const __nv_bfloat16* __restrict__ Xin, int ldx,
        const int* __restrict__ G, const int* __restrict__ NG, int S, MoEArgs a, __nv_bfloat16* __restrict__ A, int H, int I) {
    extern __shared__ uint32_t smem[];
    if ((int)blockIdx.y >= NG[0]) return;
    const int* grp = G + (size_t)blockIdx.y * (2 + MOE_GR);
    const int cb = blockIdx.x;
    switch (a.kb[grp[0] * 2]) {
        case 2: up_group_body<2, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 3: up_group_body<3, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 4: up_group_body<4, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 5: up_group_body<5, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 6: up_group_body<6, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 7: up_group_body<7, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 8: up_group_body<8, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 15: up_group_body<15, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 25: up_group_body<25, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        case 35: up_group_body<35, ST, WARPS>(Xin, ldx, S, grp, cb, a, A, H, I, smem); break;
        default: break;
    }
}

template <int KB, int ST, int WARPS>
__device__ __forceinline__ void down_group_body(const __nv_bfloat16* __restrict__ A, const int* __restrict__ grp, int cb,
        MoEArgs a, float* __restrict__ YD, int H, int I, uint32_t* smem) {
    const int e = grp[0], R = grp[1];
    __half* xh = reinterpret_cast<__half*>(smem);                          // [MOE_GR][I]
    float* red = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    RowGemmG<KB, ST, WARPS> rg(a.tr[e * 2 + 1], H / 16, cb, I, smem + (size_t)MOE_GR * I / 2);
    rg.issue_first();
    const __half* suh = a.su[e * 2 + 1];
    for (int r = 0; r < R; ++r) {
        const __nv_bfloat16* ar = A + (size_t)grp[2 + r] * I;
        for (int blk = warp; blk * 128 < I; blk += WARPS) {
            const int k = blk * 128 + 4 * lane;
            const uint2 xr = *reinterpret_cast<const uint2*>(ar + k);
            const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
            float v0 = bf_lo(xr.x), v1 = bf_hi(xr.x), v2 = bf_lo(xr.y), v3 = bf_hi(xr.y);
            const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
            v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
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
    const __half* svh = a.sv[e * 2 + 1];
    const uint2 sv = *reinterpret_cast<const uint2*>(svh + n);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
    for (int r = warp; r < R; r += WARPS) {
        float v0, v1, v2, v3;
        red_rot_row<WARPS>(red, r, lane, v0, v1, v2, v3);
        *reinterpret_cast<float4*>(YD + (size_t)grp[2 + r] * H + n) = make_float4(
            bfr(v0 * RSQRT128 * __low2float(s01)), bfr(v1 * RSQRT128 * __high2float(s01)),
            bfr(v2 * RSQRT128 * __low2float(s23)), bfr(v3 * RSQRT128 * __high2float(s23)));
    }
}

// grid (H/128, P): blockIdx.y -> group y; YD [P, H] fp32 (bf16 values)
template <int ST, int WARPS>
__global__ void __launch_bounds__(256) relu2_down_group_kernel(const __nv_bfloat16* __restrict__ A, const int* __restrict__ G,
        const int* __restrict__ NG, MoEArgs a, float* __restrict__ YD, int H, int I) {
    extern __shared__ uint32_t smem[];
    if ((int)blockIdx.y >= NG[0]) return;
    const int* grp = G + (size_t)blockIdx.y * (2 + MOE_GR);
    const int cb = blockIdx.x;
    switch (a.kb[grp[0] * 2 + 1]) {
        case 2: down_group_body<2, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 3: down_group_body<3, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 4: down_group_body<4, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 5: down_group_body<5, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 6: down_group_body<6, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 7: down_group_body<7, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 8: down_group_body<8, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 15: down_group_body<15, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 25: down_group_body<25, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        case 35: down_group_body<35, ST, WARPS>(A, grp, cb, a, YD, H, I, smem); break;
        default: break;
    }
}

}  // namespace

constexpr int ST = 2, WARPS = 8;

static MoEArgs args_of(const torch::Tensor& tr, const torch::Tensor& kb, const torch::Tensor& su, const torch::Tensor& sv) {
    MoEArgs a;
    a.tr = (const uint32_t* const*)tr.data_ptr();
    a.kb = (const int*)kb.data_ptr();
    a.su = (const __half* const*)su.data_ptr();
    a.sv = (const __half* const*)sv.data_ptr();
    return a;
}

// the input row (fp16), the warps' partial rows, and WARPS x ST x 8 tiles of the widest trellis tile
static size_t smem_of(int kin, int kbmax) {
    return (size_t)(kin / 2 + 8 * 128) * 4 + (size_t)WARPS * ST * 8 * tile_u32(kbmax) * 4;
}

void nh_relu2_up(torch::Tensor x, torch::Tensor ids, torch::Tensor tr, torch::Tensor kb, torch::Tensor su,
                 torch::Tensor sv, torch::Tensor a, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), H = x.size(1), P = ids.numel(), S = P / M, I = a.size(1);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, X]");
    TORCH_CHECK(ids.dtype() == torch::kInt && P == M * S, "ids int32 [M S]");
    TORCH_CHECK(a.dtype() == torch::kBFloat16 && a.is_contiguous() && a.size(0) == P, "a bf16 [P, I]");
    TORCH_CHECK(H % 128 == 0 && I % 128 == 0, "X, I multiples of 128");
    const size_t smem = smem_of(H, (int)kbmax);
    cudaFuncSetAttribute(relu2_up_kernel<ST, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    relu2_up_kernel<ST, WARPS><<<dim3(I / 128, P), dim3(32 * WARPS), smem, st>>>(
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const int*)ids.data_ptr(), S, args_of(tr, kb, su, sv),
        (__nv_bfloat16*)a.data_ptr(), H, I);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nh_relu2_down(torch::Tensor a, torch::Tensor ids, torch::Tensor tr, torch::Tensor kb, torch::Tensor su,
                   torch::Tensor sv, torch::Tensor yd, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(a.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = a.size(0), I = a.size(1), H = yd.size(1);
    TORCH_CHECK(a.dtype() == torch::kBFloat16 && a.is_contiguous(), "a bf16 [P, I]");
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == P && ids.numel() == P, "yd fp32 [P, X]");
    const size_t smem = smem_of(I, (int)kbmax);
    cudaFuncSetAttribute(relu2_down_kernel<ST, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    relu2_down_kernel<ST, WARPS><<<dim3(H / 128, P), dim3(32 * WARPS), smem, st>>>(
        (const __nv_bfloat16*)a.data_ptr(), (const int*)ids.data_ptr(), args_of(tr, kb, su, sv),
        (float*)yd.data_ptr(), H, I);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the group's rows (fp16) and the stages, or RED over both, whichever is larger
static size_t group_smem_of(int kin, int kbmax) {
    const size_t rows = (size_t)MOE_GR * kin * 2 + (size_t)WARPS * ST * 8 * tile_u32(kbmax) * 4;
    const size_t red = (size_t)WARPS * MOE_GR * 128 * 4;
    return rows > red ? rows : red;
}

// g [P, 2 + MOE_GR] int32, ng [1] int32: the pairs in `order` (ascending expert id) cut into groups
void nh_relu2_groups(torch::Tensor ids, torch::Tensor order, torch::Tensor g, torch::Tensor ng) {
    const at::cuda::OptionalCUDAGuard guard(ids.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = ids.numel();
    TORCH_CHECK(ids.dtype() == torch::kInt && order.dtype() == torch::kInt && order.numel() == P, "ids/order int32 [P]");
    TORCH_CHECK(g.dtype() == torch::kInt && g.size(0) >= P && g.size(1) == 2 + MOE_GR && ng.numel() == 1, "groups");
    moe_groups_kernel<<<1, 32, 0, st>>>((const int*)ids.data_ptr(), (const int*)order.data_ptr(), P, (int*)g.data_ptr(),
                                        (int*)ng.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nh_relu2_up_group(torch::Tensor x, torch::Tensor g, torch::Tensor ng, int64_t S, torch::Tensor tr, torch::Tensor kb,
                       torch::Tensor su, torch::Tensor sv, torch::Tensor a, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int H = x.size(1), P = a.size(0), I = a.size(1);
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, X]");
    TORCH_CHECK(a.dtype() == torch::kBFloat16 && a.is_contiguous() && g.size(0) >= P, "a bf16 [P, I]");
    TORCH_CHECK(H % 128 == 0 && I % 128 == 0, "X, I multiples of 128");
    const size_t smem = group_smem_of(H, (int)kbmax);
    TORCH_CHECK(smem <= 96 * 1024, "grouped up shared memory");
    cudaFuncSetAttribute(relu2_up_group_kernel<ST, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    relu2_up_group_kernel<ST, WARPS><<<dim3(I / 128, P), dim3(32 * WARPS), smem, st>>>(
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const int*)g.data_ptr(), (const int*)ng.data_ptr(), (int)S,
        args_of(tr, kb, su, sv), (__nv_bfloat16*)a.data_ptr(), H, I);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nh_relu2_down_group(torch::Tensor a, torch::Tensor g, torch::Tensor ng, torch::Tensor tr, torch::Tensor kb,
                         torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(a.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int P = a.size(0), I = a.size(1), H = yd.size(1);
    TORCH_CHECK(a.dtype() == torch::kBFloat16 && a.is_contiguous(), "a bf16 [P, I]");
    TORCH_CHECK(yd.dtype() == torch::kFloat && yd.is_contiguous() && yd.size(0) == P, "yd fp32 [P, X]");
    const size_t smem = group_smem_of(I, (int)kbmax);
    TORCH_CHECK(smem <= 96 * 1024, "grouped down shared memory");
    cudaFuncSetAttribute(relu2_down_group_kernel<ST, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    relu2_down_group_kernel<ST, WARPS><<<dim3(H / 128, P), dim3(32 * WARPS), smem, st>>>(
        (const __nv_bfloat16*)a.data_ptr(), (const int*)g.data_ptr(), (const int*)ng.data_ptr(), args_of(tr, kb, su, sv),
        (float*)yd.data_ptr(), H, I);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
