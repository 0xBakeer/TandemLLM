// MXFP8 (OCP MX: e4m3 values, one E8M0 scale per 32) times bf16 activations, 1..32 rows, on the tensor cores.
// Y[M, N] = X[M, K] @ dequant(W)[N, K]^T, bf16 out, fp32 accumulate.
//
// The weight is the B operand (8 weight rows an mma), the activation the A operand (16 rows an mma). A lane (g, t)
// of a warp loads 16 contiguous bytes (16 e4m3 values) of weight row g in each 64-wide K step, and in the step's
// j-th k16 instruction (j = 0..3) its B registers hold values 4j, 4j+1 (k slots 2t, 2t+1) and 4j+2, 4j+3 (k slots
// 2t+8, 2t+9); its A registers hold the activation at the same logical K, four consecutive bf16 of the row. A dot
// product does not care which slot a term sits in, so this is the product over K, term for term.
//
// Exactness of the weight: e4m3 -> f16 (cvt.rn.f16x2.e4m3x2, exact) -> fp32 (exact) x 2^(s - 127) (exact, a power of
// two) -> bf16 (exact for every normal result: e4m3 has 3 significand bits, bf16 7). The bf16 weight is the bits
// `(e4m3.float() * 2**(s-127)).to(bfloat16)` gives, the reference's dequantised weight.
//
// Row invariance: a CTA owns 8 weight rows; its WK warps split the K steps into contiguous slices that depend on
// (K, WK) only, each summed from zero, and warp 0 adds the slices in warp order. Nothing depends on M: a row is the
// same bits alone or in a block of 32. No atomics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Programmatic dependent launch (SM90+; on GB10 the step is memory-bound, so the pipe should never drain between
// kernels): a CTA streams its weights -- which never depend on the kernel before -- into L2 and its first registers,
// then waits for the predecessor's output (the activations), then lets its own successor launch. The arithmetic is
// the same with or without it.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
static int g_pdl = 1;
template <typename... KArgs, typename... Args>
static void launch_pdl(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    TORCH_CHECK(e == cudaSuccess, "launch: ", cudaGetErrorString(e));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

// two e4m3 bytes (low first) x the group scale -> bf16x2 (low half = low byte)
__device__ __forceinline__ uint32_t dec2(uint32_t two, float sc) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)two));
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    float2 f = __half22float2(hh);
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ float e8m0(uint32_t s) {
    return s ? __uint_as_float(s << 23) : __uint_as_float(0x00400000u);   // 2^(s - 127); s = 0 -> 2^-127
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// PF: the next K step's weight vectors and scales are loaded before this step's products (registers double-buffered;
// the arithmetic, and so the bits, are the same)
template <int MT, int WK, int NT, int PF = 0>
__global__ void __launch_bounds__(32 * WK)
mxfp8_skinny_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                    __nv_bfloat16* __restrict__ y, int M, int N, int kq, int lx, int lw, int ls, int ly) {
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
    const uint8_t* wr[NT];
    const uint8_t* sr[NT];
    bool wok[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int row = n0 + 8 * n + g;
        wok[n] = row < N;
        wr[n] = w + (size_t)(wok[n] ? row : 0) * lw + 16 * t;
        sr[n] = s + (size_t)(wok[n] ? row : 0) * ls + (t >> 1);
    }
    // this warp's K slice of its rows into L2 (a 128-byte line is two K steps), before the wait
    for (int kk = k0 + (t & 1); kk < k1 && t < 2; kk += 2)
#pragma unroll
        for (int n = 0; n < NT; ++n)
            if (wok[n]) prefetch_l2(wr[n] - 16 * t + (size_t)kk * 64);
    uint4 nw[NT];
    uint32_t ns[NT];
    if (PF && k0 < k1) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            nw[n] = wok[n] ? ldg16(wr[n] + (size_t)k0 * 64) : make_uint4(0, 0, 0, 0);
            ns[n] = wok[n] ? sr[n][k0 * 2] : 0u;
        }
    }
    pdl_wait();                                            // the activations are the predecessor's
    pdl_trigger();
    for (int kk = k0; kk < k1; ++kk) {
        uint32_t B[NT][8];
        uint4 cw[NT];
        uint32_t cs[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (PF) {
                cw[n] = nw[n];
                cs[n] = ns[n];
                if (kk + 1 < k1) {
                    nw[n] = wok[n] ? ldg16(wr[n] + (size_t)(kk + 1) * 64) : make_uint4(0, 0, 0, 0);
                    ns[n] = wok[n] ? sr[n][(kk + 1) * 2] : 0u;
                }
            } else {
                cw[n] = wok[n] ? ldg16(wr[n] + (size_t)kk * 64) : make_uint4(0, 0, 0, 0);
                cs[n] = wok[n] ? sr[n][kk * 2] : 0u;
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const uint4 wb = cw[n];
            const float sc = wok[n] ? e8m0(cs[n]) : 0.f;
            const uint32_t wv[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                B[n][2 * q] = dec2(wv[q] & 0xffffu, sc);
                B[n][2 * q + 1] = dec2(wv[q] >> 16, sc);
            }
        }
        const int kx = kk * 64 + 16 * t;                  // the lane's 16 logical K of this step
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int r0 = a * 16 + g, r1 = r0 + 8;
            uint4 x0a = make_uint4(0, 0, 0, 0), x0b = x0a, x1a = x0a, x1b = x0a;
            if (r0 < M) { x0a = ldg16(x + (size_t)r0 * lx + kx); x0b = ldg16(x + (size_t)r0 * lx + kx + 8); }
            if (r1 < M) { x1a = ldg16(x + (size_t)r1 * lx + kx); x1b = ldg16(x + (size_t)r1 * lx + kx + 8); }
            const uint32_t X0[8] = {x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w};
            const uint32_t X1[8] = {x1a.x, x1a.y, x1a.z, x1a.w, x1b.x, x1b.y, x1b.z, x1b.w};
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int j = 0; j < 4; ++j)               // values 4j, 4j+1 | 4j+2, 4j+3
                    mma_bf16(acc[a][n], X0[2 * j], X1[2 * j], X0[2 * j + 1], X1[2 * j + 1], B[n][2 * j], B[n][2 * j + 1]);
        }
    }
    // warp order: warp 0's slice, then + warp 1's, ... (a function of K and WK only)
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
    // c0 (row g, col 2t), c1 (row g, col 2t+1), c2 (row g+8, col 2t), c3 (row g+8, col 2t+1)
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = n0 + 8 * n + 2 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int r0 = a * 16 + g, r1 = r0 + 8;
            if (r0 < M) {
                if (c < N) y[(size_t)r0 * ly + c] = __float2bfloat16_rn(acc[a][n][0]);
                if (c + 1 < N) y[(size_t)r0 * ly + c + 1] = __float2bfloat16_rn(acc[a][n][1]);
            }
            if (r1 < M) {
                if (c < N) y[(size_t)r1 * ly + c] = __float2bfloat16_rn(acc[a][n][2]);
                if (c + 1 < N) y[(size_t)r1 * ly + c + 1] = __float2bfloat16_rn(acc[a][n][3]);
            }
        }
    }
}

template <int MT, int WK, int NT, int PF>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, torch::Tensor& y) {
    const int M = x.size(0), N = w.size(0), K = w.size(1);
    const int smem = WK > 1 ? (WK - 1) * MT * NT * 4 * 32 * 4 : 0;
    auto kern = mxfp8_skinny_kernel<MT, WK, NT, PF>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid((N + 8 * NT - 1) / (8 * NT));
    launch_pdl(kern, grid, dim3(32 * WK), smem, at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K / 64, (int)x.stride(0), (int)w.stride(0),
        (int)s.stride(0), (int)y.stride(0));
}

void mxfp8_skinny(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor y, int64_t wk, int64_t nt,
                  int64_t pf) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kByte && s.scalar_type() == at::kByte,
                "mxfp8_skinny: x bf16, w uint8 (e4m3 bytes), s uint8 (E8M0)");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) && s.size(1) * 32 == w.size(1), "mxfp8_skinny: K");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 64, "mxfp8_skinny: 1..64 rows");
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && s.stride(1) == 1 && x.stride(0) % 8 == 0 && w.stride(0) % 16 == 0,
                "mxfp8_skinny: row-major, 16-byte aligned rows");
    const int mt = (int)(x.size(0) + 15) / 16;            // 16-row tiles: a row's bits do not depend on them
#define L(WK, NT, PF)                                 \
    if (wk == WK && nt == NT && pf == PF) {           \
        if (mt == 4) launch<4, WK, NT, PF>(x, w, s, y);        \
        else if (mt == 3) launch<3, WK, NT, PF>(x, w, s, y);   \
        else if (mt == 2) launch<2, WK, NT, PF>(x, w, s, y);   \
        else launch<1, WK, NT, PF>(x, w, s, y);       \
        return;                                       \
    }
    L(1, 1, 0) L(2, 1, 0) L(4, 1, 0) L(8, 1, 0) L(16, 1, 0)
    L(1, 2, 0) L(2, 2, 0) L(4, 2, 0) L(8, 2, 0) L(16, 2, 0)
    L(1, 4, 0) L(2, 4, 0) L(4, 4, 0) L(8, 4, 0) L(16, 4, 0)
    L(1, 1, 1) L(2, 1, 1) L(4, 1, 1) L(8, 1, 1) L(16, 1, 1) L(1, 2, 1) L(2, 2, 1) L(4, 2, 1) L(8, 2, 1) L(16, 2, 1)
    L(8, 4, 1) L(16, 4, 1)
#undef L
    TORCH_CHECK(false, "mxfp8_skinny: (wk, nt) not built");
}
