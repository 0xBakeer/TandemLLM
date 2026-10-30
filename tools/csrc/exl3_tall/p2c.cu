
template <int KB, int WK, int ST, bool OF32>
__global__ void __launch_bounds__(256, 3) gemm_p2c_kernel(
        const __half* __restrict__ XH, const uint32_t* __restrict__ T, const __half* __restrict__ SVH,
        void* __restrict__ Yv, int ldy, int Mtot, int Kin, int N, int ldt, int b1, int b2, int S) {
    constexpr int TW = 8 * KB, RQ = 68, G = 2, NT = 4;      // G slice groups a phase, NT column tiles
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sg = warp >> 2, rg = warp & 3;
    const int g = lane >> 2, t = lane & 3;
    const int rank = blockIdx.x % S, half = (blockIdx.x / S) & 1, chunk = blockIdx.x / (2 * S);
    const int cb = blockIdx.y;
    const int KT = Kin / 16;
    const int member = (cb >= b1) + (cb >= b2);
    const int M = min(64, Mtot - chunk * 64);
    const __half* xh = XH + (size_t)member * Mtot * Kin + (size_t)chunk * 64 * Kin;
    Yv = OF32 ? (void*)(reinterpret_cast<float*>(Yv) + (size_t)chunk * 64 * ldy)
              : (void*)(reinterpret_cast<__nv_bfloat16*>(Yv) + (size_t)chunk * 64 * ldy);
    uint32_t* raw = smem + (size_t)sg * ST * NT * TW;
    uint32_t* dec = smem + (size_t)G * ST * NT * TW + (size_t)sg * 2 * NT * 128;
    float* red = reinterpret_cast<float*>(smem + (size_t)G * ST * NT * TW + (size_t)G * 2 * NT * 128);

    const int per_r = (KT + S - 1) / S;
    const int r0 = min(KT, rank * per_r), r1 = min(KT, r0 + per_r);
    const int per_w = (r1 - r0 + WK - 1) / WK;
    const int nt0 = cb * 8 + half * NT;
    const int gtid = rg * 32 + lane;

    for (int ph = 0; ph < WK / G; ++ph) {
        const int sl = ph * G + sg;
        const int k0 = min(r1, r0 + sl * per_w), k1 = min(r1, k0 + per_w);
        float acc[NT][2][4];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[j][h][c] = 0.f;
        auto issue = [&](int kt, int buf) {
            if (kt < k1) {
                const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
                uint4* dst = reinterpret_cast<uint4*>(raw + buf * NT * TW);
                for (int c = gtid; c < NT * TW / 4; c += 128) cp_async16(dst + c, src + c);
            }
            cp_commit();
        };
        uint32_t xn[4];
        auto load_x = [&](int kt) {
            const int kk = kt * 16 + 2 * t;
            const int ra = 16 * rg + g, rb = ra + 8;
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(xh + (size_t)ra * Kin + kk);
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(xh + (size_t)rb * Kin + kk);
            xn[0] = ra < M ? __ldg(pa) : 0u;
            xn[1] = rb < M ? __ldg(pb) : 0u;
            xn[2] = ra < M ? __ldg(pa + 4) : 0u;
            xn[3] = rb < M ? __ldg(pb + 4) : 0u;
        };
#pragma unroll
        for (int s = 0; s < ST - 1; ++s) issue(k0 + s, s);
        if (k0 < k1) load_x(k0);
        for (int kt = k0; kt < k1; ++kt) {
            const int i = kt - k0;
            uint32_t xa[4] = {xn[0], xn[1], xn[2], xn[3]};
            if (kt + 1 < k1) load_x(kt + 1);
            issue(kt + ST - 1, (i + ST - 1) % ST);
            cp_wait<ST - 1>();
            bar_named(1 + sg, 128);
            uint32_t* db = dec + (i & 1) * NT * 128;
            {
                const int j = rg;                                 // this warp decodes 1 of the 4 tiles
                FragB f0, f1;
                dq_tile<KB>(raw + (i % ST) * NT * TW + j * TW, lane * 8, f0, f1);
                *reinterpret_cast<uint4*>(db + j * 128 + lane * 4) = make_uint4(f0.r[0], f0.r[1], f1.r[0], f1.r[1]);
            }
            bar_named(1 + sg, 128);
            if (16 * rg < M) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const uint4 v = *reinterpret_cast<const uint4*>(db + j * 128 + lane * 4);
                    mma(acc[j][0], xa[0], xa[1], xa[2], xa[3], v.x, v.y);
                    mma(acc[j][1], xa[0], xa[1], xa[2], xa[3], v.z, v.w);
                }
            }
        }
        cp_wait<0>();
        for (int step = 0; step < G; ++step) {
            if (sg == step) {
                const int s = ph * G + step;
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = j * 16 + h * 8 + 2 * t;
                        float* r_lo = red + (16 * rg + g) * RQ + col;
                        float* r_hi = red + (16 * rg + 8 + g) * RQ + col;
                        if (s == 0) {
                            r_lo[0] = acc[j][h][0]; r_lo[1] = acc[j][h][1];
                            r_hi[0] = acc[j][h][2]; r_hi[1] = acc[j][h][3];
                        } else {
                            r_lo[0] += acc[j][h][0]; r_lo[1] += acc[j][h][1];
                            r_hi[0] += acc[j][h][2]; r_hi[1] += acc[j][h][3];
                        }
                    }
            }
            __syncthreads();
        }
    }

    const int rows = min(M, 64);
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    if (S > 1 && rank == 0) {
        for (int r = 1; r < S; ++r) {
            const float* rem = cluster.map_shared_rank(red, half * S + r);
            for (int e = threadIdx.x; e < rows * 64; e += 256) {
                const int m = e >> 6, c = e & 63;
                red[m * RQ + c] += rem[m * RQ + c];
            }
            __syncthreads();
        }
    }
    cluster.sync();
    if (half == 0 && rank == 0) {
        const float* other = cluster.map_shared_rank(red, S);
        for (int m = warp; m < rows; m += 8) {
            const float* rr = (lane < 16 ? red : other) + m * RQ + 4 * (lane & 15);
            float v0 = rr[0], v1 = rr[1], v2 = rr[2], v3 = rr[3];
            wht128(v0, v1, v2, v3, lane);
            const int n = cb * 128 + 4 * lane;
            const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
            const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
            v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
            v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
            if constexpr (OF32) {
                float4* y = reinterpret_cast<float4*>(reinterpret_cast<float*>(Yv) + (size_t)m * ldy + n);
                *y = make_float4(v0, v1, v2, v3);
            } else {
                __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
                uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
                *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(Yv) + (size_t)m * ldy + n) = o;
            }
        }
    }
    cluster.sync();                                   // the other half's buffer stays until it is read
}

template <int KB, int WK, bool OF32>
void launch_p2c(const at::Tensor& xh, const at::Tensor& t, const at::Tensor& svh, at::Tensor& y,
                int M, int Kin, int N, int ldt, int S, int b1, int b2, cudaStream_t stream) {
    constexpr int TW = 8 * KB, ST = EXL3_P2_ST, G = 2, NT = 4;
    const size_t smem = ((size_t)G * ST * NT * TW + (size_t)G * 2 * NT * 128) * 4 + (size_t)64 * 68 * 4;
    auto kern = gemm_p2c_kernel<KB, WK, ST, OF32>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        attr = true;
    }
    cudaLaunchConfig_t cfg = {};
    const int R = (M + 63) / 64;
    cfg.gridDim = dim3(2 * S * R, N / 128, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2 * S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern,
        (const __half*)xh.data_ptr(), (const uint32_t*)t.data_ptr(), (const __half*)svh.data_ptr(),
        (void*)y.data_ptr(), (int)y.stride(0), M, Kin, N, ldt, b1, b2, S);
    TORCH_CHECK(e == cudaSuccess, "exl3 p2c launch: ", cudaGetErrorString(e));
}

#define P2C_WK(KB, OF32) switch (wk) { \
    case 4: launch_p2c<KB, 4, OF32>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
    case 8: launch_p2c<KB, 8, OF32>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
    default: TORCH_CHECK(false, "p2c: wk must be 4 or 8, got ", wk); }
#define P2C_KB(OF32) switch (kb) { \
    case 3: P2C_WK(3, OF32) \
    case 4: P2C_WK(4, OF32) \
    case 5: P2C_WK(5, OF32) \
    case 6: P2C_WK(6, OF32) \
    default: TORCH_CHECK(false, "p2c: K bits 3..6, got ", kb); }

void exl3_gemm_p2c(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y,
                   int64_t kb, int64_t wk, int64_t S, int64_t b1, int64_t b2) {
    const at::cuda::OptionalCUDAGuard guard(xh.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3, "xh: fp16 [G, M, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3, "trellis: int16 [Kin/16, N/16, 16K]");
    TORCH_CHECK(t.size(2) == 16 * kb && t.stride(2) == 1 && t.stride(1) == 16 * kb, "trellis tiles must be dense");
    const int M = xh.size(1), Kin = xh.size(2), N = t.size(1) * 16;
    const int ldt = t.stride(0) / t.stride(1);
    TORCH_CHECK(t.size(0) * 16 == Kin && N % 128 == 0 && Kin % 128 == 0, "shapes");
    TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "y [M, N]");
    TORCH_CHECK(M >= 1 && S >= 1 && 2 * S <= 16, "rows / cluster size");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh fp16 [N]");
    if (y.dtype() == torch::kFloat) { P2C_KB(true) }
    else { TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32"); P2C_KB(false) }
}
