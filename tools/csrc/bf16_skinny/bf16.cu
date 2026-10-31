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

// PF: the next K step's weight vectors loaded before this step's products (same bits)
template <int MT, int WK, int NT, int PF = 0>
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
    // this warp's K slice of its rows into L2 (one 128-byte line a K step), before the wait
    for (int kk = k0; kk < k1 && t == 0; ++kk)
#pragma unroll
        for (int n = 0; n < NT; ++n)
            if (wok[n]) prefetch_l2(wr[n] + (size_t)kk * 64);
    uint4 nlo[NT], nhi[NT];
    if (PF && k0 < k1) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            nlo[n] = wok[n] ? ldw(wr[n] + (size_t)k0 * 64) : make_uint4(0, 0, 0, 0);
            nhi[n] = wok[n] ? ldw(wr[n] + (size_t)k0 * 64 + 8) : make_uint4(0, 0, 0, 0);
        }
    }
    pdl_wait();
    pdl_trigger();
    for (int kk = k0; kk < k1; ++kk) {
        uint32_t B[NT][8];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            uint4 lo, hi;
            if (PF) {
                lo = nlo[n]; hi = nhi[n];
                if (kk + 1 < k1) {
                    nlo[n] = wok[n] ? ldw(wr[n] + (size_t)(kk + 1) * 64) : make_uint4(0, 0, 0, 0);
                    nhi[n] = wok[n] ? ldw(wr[n] + (size_t)(kk + 1) * 64 + 8) : make_uint4(0, 0, 0, 0);
                }
            } else {
                lo = wok[n] ? ldw(wr[n] + (size_t)kk * 64) : make_uint4(0, 0, 0, 0);
                hi = wok[n] ? ldw(wr[n] + (size_t)kk * 64 + 8) : make_uint4(0, 0, 0, 0);
            }
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

// Cluster split-K (SM90+ thread-block clusters, distributed shared memory): a weight with few rows (the mixers'
// down|inject: 324 rows = 41 CTAs of 8 rows for 48 SMs) streams its K in CS chunks, one CTA a chunk, the CS CTAs of a
// row tile one cluster. Each CTA sums its chunk (warps in warp order); then the cluster's rank 0 adds the other ranks'
// sums from their shared memory in rank order. One launch, no atomics; the order depends on (K, WK, CS) only, so a
// row is the same bits alone or in a block.
#include <cooperative_groups.h>
template <int MT, int WK, int CS>
__global__ void __launch_bounds__(32 * WK)
bf16_cluster_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ y,
                    int M, int N, int kq, int lx, int lw, int ly) {
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    extern __shared__ float part[];                       // [WK - 1][MT][4][32], then the CTA's sum [MT][4][32]
    float* fin = part + (WK - 1) * MT * 4 * 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int rank = (int)cl.block_rank(), n0 = (blockIdx.x / CS) * 8;
    const int cper = (kq + CS - 1) / CS, c0 = rank * cper, c1 = min(kq, c0 + cper);
    const int per = (c1 - c0 + WK - 1) / WK, k0 = min(c1, c0 + warp * per), k1 = min(c1, k0 + per);
    const int row = n0 + g;
    const bool wok = row < N;
    const __nv_bfloat16* wr = w + (size_t)(wok ? row : 0) * lw + 16 * t;
    for (int kk = k0; kk < k1 && t == 0; ++kk)
        if (wok) prefetch_l2(wr + (size_t)kk * 64);
    float acc[MT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[a][i] = 0.f;
    uint4 nlo = make_uint4(0, 0, 0, 0), nhi = nlo;
    if (k0 < k1 && wok) { nlo = ldw(wr + (size_t)k0 * 64); nhi = ldw(wr + (size_t)k0 * 64 + 8); }
    pdl_wait();
    pdl_trigger();
    for (int kk = k0; kk < k1; ++kk) {
        const uint4 lo = nlo, hi = nhi;
        if (kk + 1 < k1 && wok) { nlo = ldw(wr + (size_t)(kk + 1) * 64); nhi = ldw(wr + (size_t)(kk + 1) * 64 + 8); }
        const uint32_t B[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
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
            for (int j = 0; j < 4; ++j)
                mma_bf16(acc[a], X0[2 * j], X1[2 * j], X0[2 * j + 1], X1[2 * j + 1], B[2 * j], B[2 * j + 1]);
        }
    }
    // the CTA's sum: warps in warp order, into fin
    if (warp > 0)
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int i = 0; i < 4; ++i) part[(((warp - 1) * MT + a) * 4 + i) * 32 + lane] = acc[a][i];
    __syncthreads();
    if (warp == 0) {
        for (int v = 1; v < WK; ++v)
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[a][i] += part[(((v - 1) * MT + a) * 4 + i) * 32 + lane];
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int i = 0; i < 4; ++i) fin[(a * 4 + i) * 32 + lane] = acc[a][i];
    }
    cl.sync();                                            // every rank's sum is in its shared memory
    if (rank == 0 && warp == 0) {
        for (int r = 1; r < CS; ++r) {
            const float* rf = cl.map_shared_rank(fin, r);
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[a][i] += rf[(a * 4 + i) * 32 + lane];
        }
        const int c = n0 + 2 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int r0 = a * 16 + g, r1 = r0 + 8;
            if (r0 < M) {
                if (c < N) y[(size_t)r0 * ly + c] = acc[a][0];
                if (c + 1 < N) y[(size_t)r0 * ly + c + 1] = acc[a][1];
            }
            if (r1 < M) {
                if (c < N) y[(size_t)r1 * ly + c] = acc[a][2];
                if (c + 1 < N) y[(size_t)r1 * ly + c + 1] = acc[a][3];
            }
        }
    }
    cl.sync();                                            // rank 0 has read the others' shared memory
}

template <int MT, int WK, int CS>
void launch_cluster(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
    const int M = x.size(0), N = w.size(0), K = w.size(1);
    const size_t smem = (size_t)WK * MT * 4 * 32 * 4;
    auto kern = bf16_cluster_kernel<MT, WK, CS>;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((N + 7) / 8 * CS); cfg.blockDim = dim3(32 * WK); cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    at[1].id = cudaLaunchAttributeClusterDimension;
    at[1].val.clusterDim.x = CS; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 2;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), y.data_ptr<float>(), M, N, K / 64, (int)x.stride(0),
        (int)w.stride(0), (int)y.stride(0));
    TORCH_CHECK(e == cudaSuccess, "bf16_cluster launch: ", cudaGetErrorString(e));
}

template <int MT, int WK, int NT, int PF>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
    const int M = x.size(0), N = w.size(0), K = w.size(1);
    const int smem = WK > 1 ? (WK - 1) * MT * NT * 4 * 32 * 4 : 0;
    auto kern = bf16_skinny_kernel<MT, WK, NT, PF>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    launch_pdl(kern, dim3((N + 8 * NT - 1) / (8 * NT)), dim3(32 * WK), smem, at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        y.data_ptr<float>(), M, N, K / 64, (int)x.stride(0), (int)w.stride(0), (int)y.stride(0));
}

// The hyper-connection mixer's second half for NVIDIA FP4-Local's bf16 up weight [4 H, 320], every row count on the tensor
// cores (fn_kernels' mix_up_kernel streams it on the CUDA cores: at bandwidth for one row, compute-bound at 32). A =
// bf16(silu(bf16(bf16(D) / hc))) [M, 320] is staged in shared memory by every CTA; u_s = W[s H + j] . A[m] (fp32, the
// mma's order: lane (g, t) holds 16 contiguous weights of row g in each 64-wide K chunk, the activations at the same
// logical K), then mix_up_kernel's epilogue: p_s = bf16(bf16(sigmoid(bf16(u_s))) n_s), x = bf16((((p0 + p1) + p2) + p3) / hc).
// grid H / 8, 128 threads: warp s = stream s's 8 rows s H + j0... A row's bits never depend on the others: row-invariant.
#define MXU_R 320
#define MXU_LDS (MXU_R + 4)            // a row 162 words apart: a half-warp's 8-byte A loads hit 32 distinct banks
__device__ __forceinline__ float bfr_(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float silu_(float v) { return v / (1.0f + ::expf(-v)); }

template <int JG>
__global__ void __launch_bounds__(128 * JG) mixup_bf16_kernel(const float* __restrict__ D, int ldd,
        const __nv_bfloat16* __restrict__ Nn, int ldn, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ Xo, int ldxo, int M, int H, float inv_hc) {
    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16* as = reinterpret_cast<__nv_bfloat16*>(sm_raw);   // [mr][MXU_LDS]
    const int mr = (M + 15) / 16 * 16;
    float* ps = reinterpret_cast<float*>(sm_raw + (size_t)mr * MXU_LDS * 2);   // [JG][4][M][8]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, s = warp & 3, jg = warp >> 2, g = lane >> 2, t = lane & 3;
    const int j0 = (blockIdx.x * JG + jg) * 8;
    float* pj = ps + (size_t)jg * 4 * M * 8;
    constexpr int NQ = MXU_R / 64;
    uint4 wv[2 * NQ];
    const __nv_bfloat16* wr = W + (size_t)(s * H + j0 + g) * MXU_R + 16 * t;
#pragma unroll
    for (int c = 0; c < NQ; ++c) {                        // the weights never depend on the predecessor
        wv[2 * c] = ldw(wr + 64 * c);
        wv[2 * c + 1] = ldw(wr + 64 * c + 8);
    }
    pdl_wait();
    pdl_trigger();
    for (int e = threadIdx.x; e < mr * MXU_R; e += blockDim.x) {
        const int m = e / MXU_R, k = e % MXU_R;
        as[m * MXU_LDS + k] = m < M ? __float2bfloat16_rn(silu_(bfr_(bfr_(D[(size_t)m * ldd + k]) * inv_hc)))
                                    : __float2bfloat16_rn(0.f);
    }
    __syncthreads();
    for (int r0 = 0; r0 < M; r0 += 16) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t* a_lo = reinterpret_cast<const uint32_t*>(as + (size_t)(r0 + g) * MXU_LDS + 16 * t);
        const uint32_t* a_hi = reinterpret_cast<const uint32_t*>(as + (size_t)(r0 + g + 8) * MXU_LDS + 16 * t);
#pragma unroll
        for (int c = 0; c < NQ; ++c) {
            const uint32_t wd[8] = {wv[2 * c].x, wv[2 * c].y, wv[2 * c].z, wv[2 * c].w,
                                    wv[2 * c + 1].x, wv[2 * c + 1].y, wv[2 * c + 1].z, wv[2 * c + 1].w};
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int o = 32 * c + 2 * v;             // words: the lane's 4 values at logical K 64 c + 16 t + 4 v
                mma_bf16(acc, a_lo[o], a_hi[o], a_lo[o + 1], a_hi[o + 1], wd[2 * v], wd[2 * v + 1]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = r0 + g + (i < 2 ? 0 : 8), jj = 2 * t + (i & 1);
            if (m < M) {
                const float u = bfr_(1.0f / (1.0f + ::expf(-bfr_(acc[i]))));
                pj[((size_t)s * M + m) * 8 + jj] = bfr_(u * __bfloat162float(Nn[(size_t)m * ldn + s * H + j0 + jj]));
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < JG * M * 8; e += blockDim.x) {
        const int q = e / (M * 8), m = (e / 8) % M, jj = e % 8;
        const float* pq = ps + (size_t)q * 4 * M * 8;
        const float p0 = pq[(0 * M + m) * 8 + jj], p1 = pq[(1 * M + m) * 8 + jj];
        const float p2 = pq[(2 * M + m) * 8 + jj], p3 = pq[(3 * M + m) * 8 + jj];
        Xo[(size_t)m * ldxo + (blockIdx.x * JG + q) * 8 + jj] = __float2bfloat16_rn((((p0 + p1) + p2) + p3) * inv_hc);
    }
}

void mixup_bf16(torch::Tensor d, torch::Tensor n, torch::Tensor w, torch::Tensor x, int64_t hc, int64_t jg) {
    const int M = d.size(0), H = x.size(1);
    TORCH_CHECK(hc == 4 && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(0) == 4 * H && w.size(1) == MXU_R,
                "mixup_bf16: w bf16 [4 H, 320]");
    TORCH_CHECK(d.scalar_type() == at::kFloat && d.stride(1) == 1 && d.size(1) >= MXU_R, "mixup_bf16: d fp32 [M, >=320]");
    TORCH_CHECK(n.scalar_type() == at::kBFloat16 && n.stride(1) == 1 && n.size(1) == 4 * H && n.size(0) == M,
                "mixup_bf16: n bf16 [M, 4 H]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(0) == M && H % 8 == 0, "mixup_bf16: x");
    TORCH_CHECK(M >= 1 && M <= 64, "mixup_bf16: 1..64 rows");
    const int mr = (M + 15) / 16 * 16;
    TORCH_CHECK(jg == 1 || jg == 2 || jg == 4, "mixup_bf16: jg 1, 2 or 4");
    TORCH_CHECK(H % (8 * jg) == 0, "mixup_bf16: H % (8 jg)");
    const size_t smem = (size_t)mr * MXU_LDS * 2 + (size_t)jg * 4 * M * 8 * 4;
    auto k = jg == 1 ? mixup_bf16_kernel<1> : jg == 2 ? mixup_bf16_kernel<2> : mixup_bf16_kernel<4>;
    static bool once[3] = {false, false, false};
    const int ki = jg == 1 ? 0 : jg == 2 ? 1 : 2;
    if (!once[ki]) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); once[ki] = true; }
    launch_pdl(k, dim3(H / (8 * jg)), dim3(128 * jg), smem, at::cuda::getCurrentCUDAStream(),
        (const float*)d.data_ptr(), (int)d.stride(0), (const __nv_bfloat16*)n.data_ptr(), (int)n.stride(0),
        (const __nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), M, H, 1.0f / (float)hc);
}

void bf16_cluster(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t cs) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kFloat,
                "bf16_cluster: x, w bf16, y fp32");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1) && x.size(0) >= 1 && x.size(0) <= 64, "bf16_cluster: shapes");
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1 && x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0,
                "bf16_cluster: row-major, 16-byte aligned rows");
    const int mt = (int)(x.size(0) + 15) / 16;
#define C(WK, CS)                                         \
    if (wk == WK && cs == CS) {                           \
        if (mt == 4) launch_cluster<4, WK, CS>(x, w, y);          \
        else if (mt == 3) launch_cluster<3, WK, CS>(x, w, y);     \
        else if (mt == 2) launch_cluster<2, WK, CS>(x, w, y);     \
        else launch_cluster<1, WK, CS>(x, w, y);          \
        return;                                           \
    }
    C(4, 2) C(8, 2) C(16, 2) C(4, 4) C(8, 4) C(16, 4) C(4, 8) C(8, 8)
#undef C
    TORCH_CHECK(false, "bf16_cluster: (wk, cs) not built");
}

void bf16_skinny(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t nt, int64_t pf) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kFloat,
                "bf16_skinny: x, w bf16, y fp32");
    TORCH_CHECK(w.size(1) % 64 == 0 && x.size(1) == w.size(1), "bf16_skinny: K % 64");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 64, "bf16_skinny: 1..64 rows");
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1 && x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0,
                "bf16_skinny: row-major, 16-byte aligned rows");
    const int mt = (int)(x.size(0) + 15) / 16;
#define L(WK, NT, PF)                                 \
    if (wk == WK && nt == NT && pf == PF) {           \
        if (mt == 4) launch<4, WK, NT, PF>(x, w, y);          \
        else if (mt == 3) launch<3, WK, NT, PF>(x, w, y);     \
        else if (mt == 2) launch<2, WK, NT, PF>(x, w, y);     \
        else launch<1, WK, NT, PF>(x, w, y);          \
        return;                                       \
    }
    L(1, 1, 0) L(2, 1, 0) L(4, 1, 0) L(8, 1, 0) L(16, 1, 0) L(32, 1, 0)
    L(4, 2, 0) L(8, 2, 0) L(16, 2, 0) L(32, 2, 0)
    L(1, 1, 1) L(2, 1, 1) L(4, 1, 1) L(8, 1, 1) L(16, 1, 1) L(32, 1, 1)
    L(2, 2, 1) L(4, 2, 1) L(8, 2, 1) L(16, 2, 1)
#undef L
    TORCH_CHECK(false, "bf16_skinny: (wk, nt) not built");
}
