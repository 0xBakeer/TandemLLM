// bf16 weights times bf16 activations, 1..32 rows, on the tensor cores: Y[M, N] = X[M, K] @ W[N, K]^T, fp32 out.
// NVIDIA FP4-Local keeps its mixers (down|inject, K = 10,240) and router in bf16; fn_gemv streams them on the CUDA
// cores at ~140 GB/s. Same shape of kernel as tools/csrc/mxfp8_skinny (its header): the weight is the B operand (8
// rows an mma), a lane (g, t) holds 16 contiguous values (32 bytes) of weight row g in each 64-wide K step and the
// activation at the same logical K; a CTA owns 8 NT weight rows, its WK warps split the K steps into contiguous
// slices that depend on (K, WK) only, summed in warp order. Row-invariant, no atomics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 ldw(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MT, int WK, int NT>
__global__ void __launch_bounds__(32 * WK)
bf16_skinny_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ y,
                   int M, int N, int kq, int lx, int lw, int ly) {
    extern __shared__ float part[];                       // [WK - 1][MT][NT][4][32]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 8 * NT;
    const int per = (kq + WK - 1) / WK, k0 = warp * per, k1 = min(kq, k0 + per);
    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[a][n][i] = 0.f;
    const __nv_bfloat16* wr[NT];
    bool wok[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int row = n0 + 8 * n + g;
        wok[n] = row < N;
        wr[n] = w + (size_t)(wok[n] ? row : 0) * lw + 16 * t;
    }
    for (int kk = k0; kk < k1; ++kk) {
        uint32_t B[NT][8];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const uint4 lo = wok[n] ? ldw(wr[n] + (size_t)kk * 64) : make_uint4(0, 0, 0, 0);
            const uint4 hi = wok[n] ? ldw(wr[n] + (size_t)kk * 64 + 8) : make_uint4(0, 0, 0, 0);
            B[n][0] = lo.x; B[n][1] = lo.y; B[n][2] = lo.z; B[n][3] = lo.w;
            B[n][4] = hi.x; B[n][5] = hi.y; B[n][6] = hi.z; B[n][7] = hi.w;
        }
        const int kx = kk * 64 + 16 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int r0 = a * 16 + g, r1 = r0 + 8;
            uint4 x0a = make_uint4(0, 0, 0, 0), x0b = x0a, x1a = x0a, x1b = x0a;
            if (r0 < M) { x0a = __ldg(reinterpret_cast<const uint4*>(x + (size_t)r0 * lx + kx));
                          x0b = __ldg(reinterpret_cast<const uint4*>(x + (size_t)r0 * lx + kx + 8)); }
            if (r1 < M) { x1a = __ldg(reinterpret_cast<const uint4*>(x + (size_t)r1 * lx + kx));
                          x1b = __ldg(reinterpret_cast<const uint4*>(x + (size_t)r1 * lx + kx + 8)); }
            const uint32_t X0[8] = {x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w};
            const uint32_t X1[8] = {x1a.x, x1a.y, x1a.z, x1a.w, x1b.x, x1b.y, x1b.z, x1b.w};
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    mma_bf16(acc[a][n], X0[2 * j], X1[2 * j], X0[2 * j + 1], X1[2 * j + 1], B[n][2 * j], B[n][2 * j + 1]);
        }
    }
    if (WK > 1) {
        if (warp > 0) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int i = 0; i < 4; ++i) part[((((warp - 1) * MT + a) * NT + n) * 4 + i) * 32 + lane] = acc[a][n][i];
        }
        __syncthreads();
        if (warp > 0) return;
        for (int v = 1; v < WK; ++v)
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[a][n][i] += part[((((v - 1) * MT + a) * NT + n) * 4 + i) * 32 + lane];
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = n0 + 8 * n + 2 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int r0 = a * 16 + g, r1 = r0 + 8;
            if (r0 < M) {
                if (c < N) y[(size_t)r0 * ly + c] = acc[a][n][0];
                if (c + 1 < N) y[(size_t)r0 * ly + c + 1] = acc[a][n][1];
            }
            if (r1 < M) {
                if (c < N) y[(size_t)r1 * ly + c] = acc[a][n][2];
                if (c + 1 < N) y[(size_t)r1 * ly + c + 1] = acc[a][n][3];
            }
        }
    }
}

template <int MT, int WK, int NT>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
    const int M = x.size(0), N = w.size(0), K = w.size(1);
    const int smem = WK > 1 ? (WK - 1) * MT * NT * 4 * 32 * 4 : 0;
    auto kern = bf16_skinny_kernel<MT, WK, NT>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<(N + 8 * NT - 1) / (8 * NT), 32 * WK, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        y.data_ptr<float>(), M, N, K / 64, x.stride(0), w.stride(0), y.stride(0));
}

void bf16_skinny(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t nt) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kFloat,
                "bf16_skinny: x, w bf16, y fp32");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1), "bf16_skinny: K % 64");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "bf16_skinny: 1..32 rows");
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1 && x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0,
                "bf16_skinny: row-major, 16-byte aligned rows");
    const bool two = x.size(0) > 16;
#define L(WK, NT)                                     \
    if (wk == WK && nt == NT) {                       \
        if (two) launch<2, WK, NT>(x, w, y);          \
        else launch<1, WK, NT>(x, w, y);              \
        return;                                       \
    }
    L(1, 1) L(2, 1) L(4, 1) L(8, 1) L(16, 1) L(32, 1)
    L(4, 2) L(8, 2) L(16, 2) L(32, 2)
#undef L
    TORCH_CHECK(false, "bf16_skinny: (wk, nt) not built");
}
