
constexpr float RSQRT128 = 0.08838834764831845f;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

struct Bank {
    const uint32_t* const* tr;    // [G, 3] trellis pointers (gate, up, down)
    const int* kb;                 // [G, 3] bits
    const __half* const* su;       // [G, 3] suh pointers
    const __half* const* sv;       // [G, 3] svh pointers
};

// A fragment order: a matrix [R, Kin] fp16 (R a multiple of 16) as [R / 16][Kin / 16][32 lanes][4 regs]
// of half2, lane l = 4 g + t and register c holding row g + 8 (c & 1), columns 2 t + 8 (c >> 1) and the
// next: the operand registers of mma.m16n8k16 (row-major A). The half offset of element (r, k):
__device__ __forceinline__ size_t frag_at(int r, int k, int KT) {
    const int rr = r & 15, kk = k & 15;
    const int lane = (rr & 7) * 4 + ((kk & 7) >> 1), reg = (rr >> 3) + 2 * (kk >> 3);
    return (((((size_t)(r >> 4) * KT + (k >> 4)) * 32 + lane) * 4 + reg) << 1) + (kk & 1);
}

__device__ __forceinline__ void store_frag4(__half* XH, int r, int k, int KT, float v0, float v1, float v2, float v3) {
    *reinterpret_cast<__half2*>(XH + frag_at(r, k, KT)) = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    *reinterpret_cast<__half2*>(XH + frag_at(r, k + 2, KT)) = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
}

// XH[j] row PROW[i] = fp16(H(x[ROW[i]] * suh(EID[i], MAT0 + j)) / sqrt(128)), fragment order: pair i's input
// rotated with its expert's factors (rot_in_kernel's arithmetic). grid (P, Kin / 512, nmat), 4 warps, a
// 128-block a warp; XH [nmat] of R padded rows.
__global__ void __launch_bounds__(128) pair_rot_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const int* __restrict__ ROW, const int* __restrict__ EID, const int* __restrict__ PROW, Bank a, int mat0,
        __half* __restrict__ XH, int R, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.y * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int i = blockIdx.x, j = blockIdx.z;
    const int e = EID[i], m = ROW[i];
    const int k = blk * 128 + 4 * lane;
    const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)m * ldx + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(a.su[e * 3 + mat0 + j] + k);
    float v0 = bf_lo(xr.x), v1 = bf_hi(xr.x), v2 = bf_lo(xr.y), v3 = bf_hi(xr.y);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    store_frag4(XH + (size_t)j * R * Kin, PROW[i], k, Kin / 16, v0, v1, v2, v3);
}

// XH row r = fp16(H(bf16(bf16(silu(g)) * u) * suh(EID[i], down)) / sqrt(128)) from Y[r] = [g (I) | u (I)]
// bf16, r = PROW[i] (moe_down_body's arithmetic), fragment order. grid (P, I / 512), 4 warps.
__global__ void __launch_bounds__(128) pair_silu_rot_kernel(const __nv_bfloat16* __restrict__ Y,
        const int* __restrict__ EID, const int* __restrict__ PROW, Bank a, __half* __restrict__ XH, int I) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.y * 4 + warp;
    if (blk * 128 >= I) return;
    const int i = blockIdx.x, r = PROW[i];
    const int k = blk * 128 + 4 * lane;
    const __nv_bfloat16* yg = Y + (size_t)r * 2 * I;
    const uint2 gr = *reinterpret_cast<const uint2*>(yg + k);
    const uint2 ur = *reinterpret_cast<const uint2*>(yg + I + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(a.su[EID[i] * 3 + 2] + k);
    const float gv[4] = {bf_lo(gr.x), bf_hi(gr.x), bf_lo(gr.y), bf_hi(gr.y)};
    const float uv[4] = {bf_lo(ur.x), bf_hi(ur.x), bf_lo(ur.y), bf_hi(ur.y)};
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = bfr(bfr(gv[q] / (1.0f + ::expf(-gv[q]))) * uv[q]);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    float v0 = v[0] * __low2float(s01), v1 = v[1] * __high2float(s01);
    float v2 = v[2] * __low2float(s23), v3 = v[3] * __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    store_frag4(XH, r, k, I / 16, v0, v1, v2, v3);
}

// ------------------------------------------------------------------------------- the group GEMM
// One item: rows [r0, r0 + nr) of XH (fp16, fragment order, r0 a multiple of 16, nr <= 16 MT) times one
// expert matrix's 128-column block cb. 8 warps: wn = warp % 2 owns 64 columns (NTW = 4 tiles of 16), wk =
// warp / 2 a quarter of the K tiles (contiguous). The trellis tiles stream through ST cp.async stages a
// warp; the activations ride registers two K tiles ahead, one 16-byte load a lane and row tile (rows
// past nr zeroed). The four K quarters are added in order through shared memory, then one warp a row
// applies the output rotation and svh.
constexpr int GG_NTW = 4, GG_WN = 2, GG_WK = 4, GG_ST = 3, GG_RP = 132;

template <int KB, int MT, bool OF32>
__device__ __forceinline__ void group_gemm_body(const __half* __restrict__ xh, int Kin,
        const uint32_t* __restrict__ T, int ldt, const __half* __restrict__ SVH, int cb, int nr,
        void* __restrict__ Yv, int ldy, int r0, int col0, const int* __restrict__ DST, uint32_t* smem) {
    constexpr int TW = tile_u32(KB);
    constexpr int SW = GG_NTW * TW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wn = warp % GG_WN, wk = warp / GG_WN;
    const int g = lane >> 2, t = lane & 3;
    uint32_t* stage = smem + (size_t)warp * GG_ST * SW;
    float* red = reinterpret_cast<float*>(smem + (size_t)GG_WN * GG_WK * GG_ST * SW);
    const int KT = Kin / 16;
    const int per_w = (KT + GG_WK - 1) / GG_WK;
    const int k0 = min(KT, wk * per_w), k1 = min(KT, k0 + per_w);
    const int nt0 = cb * 8 + wn * GG_NTW;

    float acc[MT][GG_NTW][2][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int j = 0; j < GG_NTW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][j][h][c] = 0.f;

    auto issue = [&](int kt, int buf) {
        if (kt < k1) {
            const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
            uint4* dst = reinterpret_cast<uint4*>(stage + buf * SW);
            for (int c = lane; c < SW / 4; c += 32) cp_async16(dst + c, src + c);
        }
        cp_commit();
    };
    // xh: the item's first row tile; a row tile of one K tile is 32 lanes x 16 bytes
    const uint4* xf = reinterpret_cast<const uint4*>(xh);
    uint4 n1[MT], n2[MT];
    auto load_x = [&](int kt, uint4* dst) {
#pragma unroll
        for (int a = 0; a < MT; ++a)
            dst[a] = 16 * a < nr ? __ldg(xf + ((size_t)a * KT + kt) * 32 + lane) : make_uint4(0u, 0u, 0u, 0u);
    };
#pragma unroll
    for (int s = 0; s < GG_ST - 1; ++s) issue(k0 + s, s);
    if (k0 < k1) load_x(k0, n1);
    if (k0 + 1 < k1) load_x(k0 + 1, n2);
    for (int kt = k0; kt < k1; ++kt) {
        const int i = kt - k0;
        uint32_t xa[MT][4];
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const uint4 v = n1[a];
            const bool lo = 16 * a + g < nr, hi = 16 * a + 8 + g < nr;      // rows g and g + 8 of the tile
            xa[a][0] = lo ? v.x : 0u; xa[a][1] = hi ? v.y : 0u; xa[a][2] = lo ? v.z : 0u; xa[a][3] = hi ? v.w : 0u;
            n1[a] = n2[a];
        }
        if (kt + 2 < k1) load_x(kt + 2, n2);
        issue(kt + GG_ST - 1, (i + GG_ST - 1) % GG_ST);
        cp_wait<GG_ST - 1>();
        __syncwarp();
        const uint32_t* buf = stage + (i % GG_ST) * SW;
#pragma unroll
        for (int j = 0; j < GG_NTW; ++j) {
            FragB f0, f1;
            dq_tile<KB>(buf + j * TW, lane * 8, f0, f1);
#pragma unroll
            for (int a = 0; a < MT; ++a) {
                mma(acc[a][j][0], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f0.r[0], f0.r[1]);
                mma(acc[a][j][1], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f1.r[0], f1.r[1]);
            }
        }
        __syncwarp();
    }
    cp_wait<0>();
    for (int step = 0; step < GG_WK; ++step) {
        if (wk == step) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int j = 0; j < GG_NTW; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (wn * GG_NTW + j) * 16 + h * 8 + 2 * t;
                        float* lo = red + (16 * a + g) * GG_RP + col;
                        float* hi = red + (16 * a + 8 + g) * GG_RP + col;
                        if (step == 0) { lo[0] = acc[a][j][h][0]; lo[1] = acc[a][j][h][1]; hi[0] = acc[a][j][h][2]; hi[1] = acc[a][j][h][3]; }
                        else { lo[0] += acc[a][j][h][0]; lo[1] += acc[a][j][h][1]; hi[0] += acc[a][j][h][2]; hi[1] += acc[a][j][h][3]; }
                    }
        }
        __syncthreads();
    }
    const int n = cb * 128 + 4 * lane;
    const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
    for (int m = warp; m < nr; m += GG_WN * GG_WK) {
        const float* rr = red + m * GG_RP + 4 * lane;
        float v0 = rr[0], v1 = rr[1], v2 = rr[2], v3 = rr[3];
        wht128(v0, v1, v2, v3, lane);
        v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
        v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
        const size_t row = DST ? (size_t)DST[r0 + m] : (size_t)(r0 + m);
        if constexpr (OF32) {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(Yv) + row * ldy + col0 + n) = make_float4(v0, v1, v2, v3);
        } else {
            __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
            uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(Yv) + row * ldy + col0 + n) = o;
        }
    }
}

// grid (items upper bound x nmat N / 128), 256 threads. Groups g = 0..G-1 own CNT[g] consecutive rows
// of XH (P rows a matrix, fragment order) from PSTART[g] (a multiple of 16); IEND[g] = items of groups
// 0..g, an item = up to 16 MT rows. blockIdx.x = item (nmat N / 128) + (matrix j, column block cb): the
// CTAs of one item are adjacent, so its rows stream from memory once and from L2 for the others; matrix
// MAT0 + j of group g reads XH[j]. Output row: DST[row] (null: the row), columns j N + n.
template <int MT, bool OF32>
__global__ void __launch_bounds__(256) group_gemm_kernel(const __half* __restrict__ XH, int P, int Kin, int N,
        const int* __restrict__ CNT, const int* __restrict__ PSTART, const int* __restrict__ IEND, int G,
        Bank a, int mat0, int nmat, void* __restrict__ Y, int ldy, const int* __restrict__ DST) {
    extern __shared__ uint32_t smem[];
    const int ncb = N / 128, nyc = nmat * ncb;
    const int item = blockIdx.x / nyc, yc = blockIdx.x % nyc;
    if (item >= IEND[G - 1]) return;
    int lo = 0, hi = G - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (IEND[mid] > item) hi = mid; else lo = mid + 1;
    }
    const int gi = lo;
    const int j = item - (gi ? IEND[gi - 1] : 0);
    const int r0 = PSTART[gi] + j * 16 * MT;
    const int nr = min(16 * MT, CNT[gi] - j * 16 * MT);
    const int jm = yc / ncb, cb = yc % ncb;
    const int mat = mat0 + jm;
    const __half* xh = XH + ((size_t)jm * P + r0) * Kin;          // r0 / 16 row tiles in
    const uint32_t* T = a.tr[gi * 3 + mat];
    const __half* svh = a.sv[gi * 3 + mat];
    const int ldt = N / 16;
    switch (a.kb[gi * 3 + mat]) {
        case 2: group_gemm_body<2, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 3: group_gemm_body<3, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 4: group_gemm_body<4, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 5: group_gemm_body<5, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 6: group_gemm_body<6, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 8: group_gemm_body<8, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 7: group_gemm_body<7, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 15: group_gemm_body<15, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 25: group_gemm_body<25, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        case 35: group_gemm_body<35, MT, OF32>(xh, Kin, T, ldt, svh, cb, nr, Y, ldy, r0, jm * N, DST, smem); break;
        default: break;
    }
}

// ----------------------------------------------------------------------------------- mixer tail
// X[m, j] = bf16(sum over streams s in order of bf16(bf16(sigmoid(bf16(U[m, s H + j]))) * Nn[m, s H + j]) / hc)
// (mix_up_kernel's roundings after its up projection). grid (H / 256, M), 256 threads, hc = 4.
__global__ void __launch_bounds__(256) mix_tail_kernel(const float* __restrict__ U, const __nv_bfloat16* __restrict__ Nn,
        int ldn, __nv_bfloat16* __restrict__ Xo, int ldx, int H, float inv_hc) {
    const int j = blockIdx.x * 256 + threadIdx.x, m = blockIdx.y;
    if (j >= H) return;
    const float* u = U + (size_t)m * 4 * H + j;
    const __nv_bfloat16* nn = Nn + (size_t)m * ldn + j;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float sg = bfr(1.0f / (1.0f + ::expf(-bfr(u[(size_t)s * H]))));
        acc += bfr(sg * __bfloat162float(nn[(size_t)s * H]));
    }
    Xo[(size_t)m * ldx + j] = __float2bfloat16_rn(acc * inv_hc);
}

}  // namespace

static Bank bank_of(const torch::Tensor& tr, const torch::Tensor& kb, const torch::Tensor& su, const torch::Tensor& sv) {
    Bank a;
    a.tr = (const uint32_t* const*)tr.data_ptr();
    a.kb = (const int*)kb.data_ptr();
    a.su = (const __half* const*)su.data_ptr();
    a.sv = (const __half* const*)sv.data_ptr();
    return a;
}

void pre_pair_rot(torch::Tensor x, torch::Tensor row, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr,
                  torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M, Kin]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3 && xh.size(1) % 16 == 0,
                "xh fp16 [nmat, R, Kin], R padded rows (a multiple of 16)");
    const int nmat = xh.size(0), R = xh.size(1), Kin = xh.size(2), P = row.numel();
    TORCH_CHECK(row.dtype() == torch::kInt && eid.dtype() == torch::kInt && prow.dtype() == torch::kInt
                && eid.numel() == P && prow.numel() == P, "row, eid, prow int32 [P]");
    TORCH_CHECK(Kin % 128 == 0 && x.size(1) == Kin, "Kin");
    if (P == 0) return;
    pair_rot_kernel<<<dim3(P, (Kin / 128 + 3) / 4, nmat), 128, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const int*)row.data_ptr(), (const int*)eid.data_ptr(), (const int*)prow.data_ptr(), bank_of(tr, kb, su, sv),
        (int)mat0, (__half*)xh.data_ptr(), R, Kin);
}

void pre_pair_silu_rot(torch::Tensor y, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr, torch::Tensor kb,
                       torch::Tensor su, torch::Tensor sv, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(y.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int R = y.size(0), I = y.size(1) / 2, P = eid.numel();
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && I % 128 == 0, "y bf16 [R, 2 I]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.numel() == (int64_t)R * I, "xh fp16 [R, I]");
    TORCH_CHECK(prow.dtype() == torch::kInt && prow.numel() == P, "prow int32 [P]");
    if (P == 0) return;
    pair_silu_rot_kernel<<<dim3(P, (I / 128 + 3) / 4), 128, 0, st>>>((const __nv_bfloat16*)y.data_ptr(),
        (const int*)eid.data_ptr(), (const int*)prow.data_ptr(), bank_of(tr, kb, su, sv), (__half*)xh.data_ptr(), I);
}

template <int MT, bool OF32>
static void gg_launch(const torch::Tensor& xh, int P, int Kin, int N, const torch::Tensor& cnt, const torch::Tensor& pstart,
                      const torch::Tensor& iend, int G, Bank a, int mat0, int nmat, torch::Tensor& y,
                      const int* dst, int items, int kbmax, cudaStream_t st) {
    const size_t smem = (size_t)GG_WN * GG_WK * GG_ST * GG_NTW * tile_u32(kbmax) * 4 + (size_t)16 * MT * GG_RP * 4;
    auto kern = group_gemm_kernel<MT, OF32>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kern<<<dim3(items * nmat * (N / 128)), 256, smem, st>>>((const __half*)xh.data_ptr(), P, Kin, N,
        (const int*)cnt.data_ptr(), (const int*)pstart.data_ptr(), (const int*)iend.data_ptr(), G, a, mat0, nmat,
        y.data_ptr(), (int)y.stride(0), dst);
}

void pre_group_gemm(torch::Tensor xh, torch::Tensor cnt, torch::Tensor pstart, torch::Tensor iend, torch::Tensor tr,
                    torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, int64_t N, torch::Tensor y,
                    c10::optional<torch::Tensor> dst, int64_t items, int64_t mt, int64_t kbmax) {
    const at::cuda::OptionalCUDAGuard guard(xh.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3, "xh fp16 [nmat, P, Kin]");
    const int nmat = xh.size(0), P = xh.size(1), Kin = xh.size(2), G = cnt.numel();
    TORCH_CHECK(cnt.dtype() == torch::kInt && pstart.dtype() == torch::kInt && iend.dtype() == torch::kInt
                && pstart.numel() == G && iend.numel() == G, "cnt, pstart, iend int32 [G]");
    TORCH_CHECK(Kin % 128 == 0 && N % 128 == 0 && ((kbmax >= 2 && kbmax <= 8) || kbmax == 15 || kbmax == 25 || kbmax == 35),
                "shapes (K 2..8, or 15 / 25 / 35 for 1.5 / 2.5 / 3.5)");
    TORCH_CHECK(y.stride(1) == 1 && y.size(1) >= nmat * N, "y [rows, >= nmat N]");
    TORCH_CHECK(P % 16 == 0, "rows a matrix: a multiple of 16 (fragment order)");
    TORCH_CHECK(!dst.has_value() || (dst->dtype() == torch::kInt && dst->numel() == P), "dst int32 [P]");
    if (items == 0 || P == 0) return;
    const int* dp = dst.has_value() ? (const int*)dst->data_ptr() : nullptr;
    const Bank a = bank_of(tr, kb, su, sv);
    if (y.dtype() == torch::kFloat) {
        if (mt == 4) gg_launch<4, true>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
        else gg_launch<2, true>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
    } else {
        TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32");
        if (mt == 4) gg_launch<4, false>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
        else gg_launch<2, false>(xh, P, Kin, N, cnt, pstart, iend, G, a, mat0, nmat, y, dp, items, kbmax, st);
    }
}

void pre_mix_tail(torch::Tensor u, torch::Tensor n, torch::Tensor x, int64_t hc) {
    const at::cuda::OptionalCUDAGuard guard(u.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = x.size(0), H = x.size(1);
    TORCH_CHECK(hc == 4 && u.dtype() == torch::kFloat && u.is_contiguous() && u.size(0) == M && u.size(1) == 4 * H, "u fp32 [M, 4 H]");
    TORCH_CHECK(n.dtype() == torch::kBFloat16 && n.stride(1) == 1 && n.size(1) == 4 * H, "n bf16 [M, 4 H]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1, "x bf16 [M, H]");
    if (M == 0) return;
    mix_tail_kernel<<<dim3((H + 255) / 256, M), 256, 0, st>>>((const float*)u.data_ptr(), (const __nv_bfloat16*)n.data_ptr(),
        (int)n.stride(0), (__nv_bfloat16*)x.data_ptr(), (int)x.stride(0), H, 1.0f / (float)hc);
}
