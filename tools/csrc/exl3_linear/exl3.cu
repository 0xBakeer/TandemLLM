
// ---------------------------------------------------------------------------------------------
// EXL3 trellis kernels for the GB10 (sm_121). The bit-window extraction (dq8 / dq4 / dq2x2 and the
// aligned 2- and 4-bit forms) and the mul1 codebook decode are ported from exllamav3
// (exllamav3_ext/quant/exl3_dq.cuh, codebook.cuh), under its licence:
//
// MIT License -- Copyright (c) 2025 Turboderp
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions: The above copyright notice and this
// permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Everything else (the skinny GEMM, its K split and reduction, the cluster reduction, the fused
// output rotation, the rotation kernels) is this engine's.
// ---------------------------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;

namespace {

struct FragB { uint32_t r[2]; };

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}
#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// two mul1 codebook entries -> one f16x2 (low half = w0), exllamav3's decode_mul1_product_2
__device__ __forceinline__ uint32_t dec2(uint32_t w0, uint32_t w1) {
    uint32_t x0 = (w0 & 0xffffu) * 0x83DCD12Du;
    uint32_t x1 = (w1 & 0xffffu) * 0x83DCD12Du;
    uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    __half2 h = __halves2half2(__ushort_as_half((unsigned short)s0), __ushort_as_half((unsigned short)s1));
    __half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
    return *reinterpret_cast<uint32_t*>(&r);
}

template <int bits>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag) {
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w3 = fshift(b, a, s2);
    uint32_t w2 = fshift(b, a, s2 + bits);
    uint32_t w1 = fshift(b, a, s2 + bits * 2);
    uint32_t w0 = fshift(b, a, s2 + bits * 3);
    frag.r[0] = dec2(w0, w1);
    frag.r[1] = dec2(w2, w3);
}

template <int bits>
__device__ __forceinline__ void dq2x2(const uint32_t* ptr, int t_offset, FragB& frag) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;
        int b1 = b0 + 1 * bits;
        int b2 = b1 + 16;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[i0 % (bits * 256 / 32)];
        uint32_t b = ptr[i2 % (bits * 256 / 32)];
        uint32_t w1 = fshift(b, a, s2);
        uint32_t w0 = fshift(b, a, s2 + bits);
        frag.r[i] = dec2(w0, w1);
    }
}

template <int bits, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1) {
        w7 = fshift(b, a, s2);            w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2); w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4); w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6); w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 4) {
        w7 = fshift(b, a, s2); w6 = w7 >> bits; w5 = w6 >> bits; w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4); w2 = w3 >> bits; w1 = w2 >> bits; w0 = w1 >> bits;
    }
    frag0.r[0] = dec2(w0, w1); frag0.r[1] = dec2(w2, w3);
    frag1.r[0] = dec2(w4, w5); frag1.r[1] = dec2(w6, w7);
}

__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0.r[0] = dec2(w0, w1); frag0.r[1] = dec2(w2, w3);
    frag1.r[0] = dec2(w4, w5); frag1.r[1] = dec2(w6, w7);
}

__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    frag0.r[0] = dec2(w0, w1); frag0.r[1] = dec2(w2, w3);
    frag1.r[0] = dec2(w4, w5); frag1.r[1] = dec2(w6, w7);
}

// ENG-285: a half-integer rate (1.5, 2.5 or 3.5 bits, which exllamav3 1.6.0 gives mul1 tensors below
// 4 bpw) is ten times the rate in these kernels' bits (15, 25, 35). Its 256-weight tile is a ring of
// 256 KA + 128 bits (8 KA + 4 words, a tile 16 KA + 8 int16 wide): each weight adds KA bits and an odd
// one a bit more (exllamav3's MASK 0xAAAA), so weight i's 16-bit window ends at ring bit
// (i + 1) KA + (i + 1) / 2 (exllamav3_ext/quant/frac.cu). An integer rate's tile is 8 K words.
__host__ __device__ constexpr int tile_u32(int bits) { return bits >= 10 ? 8 * (bits / 10) + 4 : 8 * bits; }

template <int KA>
__device__ __forceinline__ void dq8_frac(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1) {
    constexpr int NW = 8 * KA + 4, R = 32 * NW;
    // t_offset is a multiple of 8: weight t_offset + j's window starts j KA + (j + 1) / 2 bits after
    // weight t_offset's, so the lane's 8 windows lie in the 3 words from the first one's
    const int s0 = ((t_offset + 1) * KA + (t_offset >> 1) - 16 + R) % R;
    const int i0 = s0 >> 5, o0 = s0 & 31;
    const uint32_t a = ptr[i0], b = ptr[(i0 + 1) % NW], c = ptr[(i0 + 2) % NW];
    uint32_t w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = o0 + j * KA + ((j + 1) >> 1);
        w[j] = r < 32 ? fshift(b, a, 48 - r) : fshift(c, b, 80 - r);
    }
    frag0.r[0] = dec2(w[0], w[1]); frag0.r[1] = dec2(w[2], w[3]);
    frag1.r[0] = dec2(w[4], w[5]); frag1.r[1] = dec2(w[6], w[7]);
}

// the 8 weights of lane (t_offset / 8) of one 16x16 tile, as two B fragments (columns 0..7, 8..15)
template <int bits>
__device__ __forceinline__ void dq_tile(const uint32_t* ptr, int idx, FragB& f0, FragB& f1) {
    if constexpr (bits >= 10) dq8_frac<bits / 10>(ptr, idx, f0, f1);
    else if constexpr (bits == 2) dq8_aligned_2bits(ptr, idx, f0, f1);
    else if constexpr (bits == 3) dq8<3, 4>(ptr, idx, f0, f1);
    else if constexpr (bits == 4) dq8_aligned_4bits(ptr, idx, f0, f1);
    else if constexpr (bits == 7) { dq2x2<7>(ptr, idx, f0); dq2x2<7>(ptr, idx + 4, f1); }
    else { dq4<bits>(ptr, idx, f0); dq4<bits>(ptr, idx + 4, f1); }   // 5, 6, 8
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* glob) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int n> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(n) : "memory"); }
#if EXL3_PDL
// SPD-245: programmatic dependent launch of the skinny GEMM (QWEN38_EXL3_PDL). The trellis is a weight, so
// its first stages are fetched before the wait; the activations and every store come after it.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }
#endif

// natural-order Walsh-Hadamard over 128 values held 4 a lane (index 4 lane + e), unnormalised
__device__ __forceinline__ void wht128(float& v0, float& v1, float& v2, float& v3, int lane) {
    float a0 = v0 + v1, a1 = v0 - v1, a2 = v2 + v3, a3 = v2 - v3;
    v0 = a0 + a2; v1 = a1 + a3; v2 = a0 - a2; v3 = a1 - a3;
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        float p0 = __shfl_xor_sync(0xffffffffu, v0, i);
        float p1 = __shfl_xor_sync(0xffffffffu, v1, i);
        float p2 = __shfl_xor_sync(0xffffffffu, v2, i);
        float p3 = __shfl_xor_sync(0xffffffffu, v3, i);
        if (lane & i) { v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3; }
        else          { v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3; }
    }
}

constexpr float RSQRT128 = 0.08838834764831845f;

// ------------------------------------------------------------------------------- input rotation
// XH[g, m, :] = fp16( H(fp32(X[m, :]) * fp32(SUH[g, :])) / sqrt(128) ). grid (Kin/128/4, M, G), 4 warps.
__global__ void __launch_bounds__(128) rot_in_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int m = blockIdx.y, g = blockIdx.z;
    const int k = blk * 128 + 4 * lane;
    const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)m * ldx + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(SUH + (size_t)g * Kin + k);
    float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
    float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + ((size_t)g * M + m) * Kin + k) = o;
}

// The MLP's activation and the down projection's input rotation in one pass, bit for bit the two
// torch kernels and rot_in: A = bf16(bf16(silu(G)) * U) with silu(x) = x / (1 + exp(-x)) in fp32 (ATen's
// silu), then XH = fp16(H(fp32(A) * SUH) / sqrt(128)). Y holds gate | up: gate at column k, up at Kin + k.
__global__ void __launch_bounds__(128) rot_in_silu_kernel(const __nv_bfloat16* __restrict__ Y, int ldy,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int m = blockIdx.y;
    const int k = blk * 128 + 4 * lane;
    const uint2 gr = *reinterpret_cast<const uint2*>(Y + (size_t)m * ldy + k);
    const uint2 ur = *reinterpret_cast<const uint2*>(Y + (size_t)m * ldy + Kin + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(SUH + k);
    float gv[4] = {__uint_as_float(gr.x << 16), __uint_as_float(gr.x & 0xffff0000u),
                   __uint_as_float(gr.y << 16), __uint_as_float(gr.y & 0xffff0000u)};
    float uv[4] = {__uint_as_float(ur.x << 16), __uint_as_float(ur.x & 0xffff0000u),
                   __uint_as_float(ur.y << 16), __uint_as_float(ur.y & 0xffff0000u)};
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float sl = __bfloat162float(__float2bfloat16_rn(gv[e] / (1.0f + ::expf(-gv[e]))));
        v[e] = __bfloat162float(__float2bfloat16_rn(sl * uv[e]));
    }
    const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
    float v0 = v[0] * __low2float(s01), v1 = v[1] * __high2float(s01);
    float v2 = v[2] * __low2float(s23), v3 = v[3] * __high2float(s23);
    wht128(v0, v1, v2, v3, lane);
    __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
    __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + (size_t)m * Kin + k) = o;
}

// The MLP of a prefill slab in one pass between the two GEMMs, bit for bit rot_out (gate and up,
// fp32 products -> bf16), ATen's silu and mul, and rot_in of the down projection: per row and
// 128-column block, YG/YU the library GEMM's fp32 products of gate and up, XH the down's rotated
// input. grid (n/128/4, M), 4 warps, one 128-block a warp.
__global__ void __launch_bounds__(128) mlp_bridge_kernel(const float* __restrict__ YG, const float* __restrict__ YU,
        int ldp, const __half* __restrict__ SVG, const __half* __restrict__ SVU,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int ldx, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= N) return;
    const int m = blockIdx.y;
    const int n = blk * 128 + 4 * lane;
    float g[4], u[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) { g[e] = YG[(size_t)m * ldp + n + e]; u[e] = YU[(size_t)m * ldp + n + e]; }
    wht128(g[0], g[1], g[2], g[3], lane);
    wht128(u[0], u[1], u[2], u[3], lane);
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float gb = __bfloat162float(__float2bfloat16_rn(g[e] * RSQRT128 * __half2float(SVG[n + e])));
        const float ub = __bfloat162float(__float2bfloat16_rn(u[e] * RSQRT128 * __half2float(SVU[n + e])));
        const float sl = __bfloat162float(__float2bfloat16_rn(gb / (1.0f + ::expf(-gb))));
        v[e] = __bfloat162float(__float2bfloat16_rn(sl * ub)) * __half2float(SUH[n + e]);
    }
    wht128(v[0], v[1], v[2], v[3], lane);
    __half2 o01 = __floats2half2_rn(v[0] * RSQRT128, v[1] * RSQRT128);
    __half2 o23 = __floats2half2_rn(v[2] * RSQRT128, v[3] * RSQRT128);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(XH + (size_t)m * ldx + n) = o;
}

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

// ------------------------------------------------------------------------------ output rotation
// Y[m, n] = H(YP[m, block]) / sqrt(128) * SVH[n], YP fp32 or fp16, Y bf16 or fp32. grid (N/128/4, M).
template <typename TI, typename TO>
__global__ void __launch_bounds__(128) rot_out_kernel(const TI* __restrict__ YP, int ldp,
        const __half* __restrict__ SVH, TO* __restrict__ Y, int ldy, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= N) return;
    const int m = blockIdx.y;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = to_f(YP[(size_t)m * ldp + n + e]);
    wht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        float r = v[e] * RSQRT128 * __half2float(SVH[n + e]);
        if constexpr (sizeof(TO) == 4) Y[(size_t)m * ldy + n + e] = r;
        else Y[(size_t)m * ldy + n + e] = __float2bfloat16_rn(r);
    }
}

// ----------------------------------------------------------------------------------- reconstruct
// WI[Kin, N] fp16 (rotated basis, row k, column n) from the trellis, columns [n_off, n_off + N).
// grid (N/128, Kin/16), 8 warps: one 16x16 tile a warp.
template <int K>
__global__ void __launch_bounds__(256) recon_kernel(const uint16_t* __restrict__ T, int ldt,
        int n_off_tiles, __half* __restrict__ WI, int ldw) {
    constexpr int TW = tile_u32(K);                // uint32 a tile
    __shared__ uint32_t s[8][TW];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kt = blockIdx.y, nt = blockIdx.x * 8 + warp;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(T) + ((size_t)kt * ldt + n_off_tiles + nt) * TW;
    for (int i = lane; i < TW; i += 32) s[warp][i] = src[i];
    __syncwarp();
    FragB f0, f1;
    dq_tile<K>(s[warp], lane * 8, f0, f1);
    const int a = lane & 3, col = lane >> 2;
    __half* out = WI + (size_t)(kt * 16) * ldw + nt * 16;
    const __half2* h00 = reinterpret_cast<const __half2*>(&f0.r[0]);
    const __half2* h01 = reinterpret_cast<const __half2*>(&f0.r[1]);
    const __half2* h10 = reinterpret_cast<const __half2*>(&f1.r[0]);
    const __half2* h11 = reinterpret_cast<const __half2*>(&f1.r[1]);
    out[(2 * a) * ldw + col]         = __low2half(*h00);
    out[(2 * a + 1) * ldw + col]     = __high2half(*h00);
    out[(2 * a + 8) * ldw + col]     = __low2half(*h01);
    out[(2 * a + 9) * ldw + col]     = __high2half(*h01);
    out[(2 * a) * ldw + col + 8]     = __low2half(*h10);
    out[(2 * a + 1) * ldw + col + 8] = __high2half(*h10);
    out[(2 * a + 8) * ldw + col + 8] = __low2half(*h11);
    out[(2 * a + 9) * ldw + col + 8] = __high2half(*h11);
}

// ---------------------------------------------------------------------------------- skinny GEMM
// Y[M, N] = rot_out( XH[member][M, Kin] @ WI ), M <= 16 MT.
// grid (S, N/128), cluster (S, 1, 1); block 32 WN WK. A CTA owns the 128 columns [128 by, +128):
// warp (wn, wk) the 16 NTW columns wn*16*NTW.. of them over its slice of the CTA's K-tile range.
// KB: bits a weight. ST: cp.async stages of a warp's (NTW tiles x one K tile) of trellis.
template <int KB, int MT, int NTW, int WN, int WK, int ST, bool OF32, bool XSM, int MINB>
__global__ void __launch_bounds__(32 * WN * WK, MINB) gemm_kernel(
        const __half* __restrict__ XH, const uint32_t* __restrict__ T, const __half* __restrict__ SVH,
        void* __restrict__ Yv, int ldy, int Mtot, int Kin, int N, int ldt, int b1, int b2, int S) {
    constexpr int TW = 8 * KB;                      // uint32 a 16x16 tile
    // XSM: the activations ride the cp.async stages with the trellis; otherwise they are loaded
    // into registers one K tile ahead (no shared memory, so more CTAs fit an SM)
    constexpr int XR = 12;                          // uint32 a staged activation row (16 fp16 + pad)
    constexpr int XW = XSM ? 16 * MT * XR : 0;      // uint32 of a stage's activation rows
    constexpr int SW = NTW * TW + XW;               // uint32 a warp's stage: trellis, then rows
    constexpr int RP = 132;                         // padded row of the reduction buffer
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wn = warp % WN, wk = warp / WN;
    const int g = lane >> 2, t = lane & 3;
    // row grid (ENG-213): blockIdx.x = chunk * S + rank. A chunk is 16 MT rows; the chunks of one
    // column block are adjacent CTAs, so the trellis the first streams is still in L2 for the next,
    // and every chunk runs the instruction sequence a call with only its rows would (row bits do not
    // depend on how many chunks there are).
    const int rank = blockIdx.x % S, chunk = blockIdx.x / S;
    const int cb = blockIdx.y;
    const int KT = Kin / 16;               // ldt: tiles a trellis row (a group member is a view)
    const int member = (cb >= b1) + (cb >= b2);
    const int M = min(16 * MT, Mtot - chunk * 16 * MT);
    const __half* xh = XH + (size_t)member * Mtot * Kin + (size_t)chunk * 16 * MT * Kin;
    Yv = OF32 ? (void*)(reinterpret_cast<float*>(Yv) + (size_t)chunk * 16 * MT * ldy)
              : (void*)(reinterpret_cast<__nv_bfloat16*>(Yv) + (size_t)chunk * 16 * MT * ldy);

    uint32_t* stage = smem + (size_t)warp * ST * SW;
    float* red = reinterpret_cast<float*>(smem + (size_t)WN * WK * ST * SW);

    // K range: the cluster's ranks, then the CTA's warps, both contiguous
    const int per_r = (KT + S - 1) / S;
    const int r0 = min(KT, rank * per_r), r1 = min(KT, r0 + per_r);
    const int per_w = (r1 - r0 + WK - 1) / WK;
    const int k0 = min(r1, r0 + wk * per_w), k1 = min(r1, k0 + per_w);
    const int nt0 = cb * 8 + wn * NTW;
    const int mrows = min(M, 16 * MT);

    // rows past M are never copied: zero them once in every stage
    if (XSM) for (int e = lane; e < ST * XW; e += 32) {
        const int st_ = e / XW, w_ = e % XW;
        if (w_ / XR >= mrows) stage[st_ * SW + NTW * TW + w_] = 0u;
    }
    __syncwarp();

    float acc[MT][NTW][2][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][j][h][c] = 0.f;

    // one stage: the NTW tiles of K tile kt (contiguous), and the K tile's 16 activations of
    // every row (two 16-byte chunks a row), all through cp.async
    auto issue = [&](int kt, int buf) {
        if (kt < k1) {
            const uint4* src = reinterpret_cast<const uint4*>(T + ((size_t)kt * ldt + nt0) * TW);
            uint4* dst = reinterpret_cast<uint4*>(stage + buf * SW);
            for (int c = lane; c < NTW * TW / 4; c += 32) cp_async16(dst + c, src + c);
            if (XSM) {
                uint32_t* xs = stage + buf * SW + NTW * TW;
                for (int c = lane; c < 2 * mrows; c += 32) {
                    const int r = c >> 1, h = c & 1;
                    cp_async16(xs + r * XR + 4 * h, xh + (size_t)r * Kin + kt * 16 + 8 * h);
                }
            }
        }
        cp_commit();
    };
    // the register path's operands, one K tile ahead
    uint32_t xn[MT][4];
    auto load_x = [&](int kt) {
        const int kk = kt * 16 + 2 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int ra = 16 * a + g, rb = ra + 8;
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(xh + (size_t)ra * Kin + kk);
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(xh + (size_t)rb * Kin + kk);
            xn[a][0] = ra < M ? __ldg(pa) : 0u;
            xn[a][1] = rb < M ? __ldg(pb) : 0u;
            xn[a][2] = ra < M ? __ldg(pa + 4) : 0u;
            xn[a][3] = rb < M ? __ldg(pb + 4) : 0u;
        }
    };

#if EXL3_PDL
    if (XSM) pdl_wait();                            // the stages carry activations
#endif
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) issue(k0 + s, s);
#if EXL3_PDL
    if (!XSM) pdl_wait();
#endif
    if (!XSM && k0 < k1) load_x(k0);
    for (int kt = k0; kt < k1; ++kt) {
        const int i = kt - k0;
        uint32_t xa[MT][4];
        if (!XSM) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int c = 0; c < 4; ++c) xa[a][c] = xn[a][c];
            if (kt + 1 < k1) load_x(kt + 1);
        }
        issue(kt + ST - 1, (i + ST - 1) % ST);
        cp_wait<ST - 1>();
        __syncwarp();
        const uint32_t* buf = stage + (i % ST) * SW;
        if (XSM) {
            // activation operands: rows g and 8 + g, K 2t.. and 2t + 8..
            const uint32_t* xs = buf + NTW * TW;
#pragma unroll
            for (int a = 0; a < MT; ++a) {
                xa[a][0] = xs[(16 * a + g) * XR + t];
                xa[a][1] = xs[(16 * a + 8 + g) * XR + t];
                xa[a][2] = xs[(16 * a + g) * XR + t + 4];
                xa[a][3] = xs[(16 * a + 8 + g) * XR + t + 4];
            }
        }
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
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
#if EXL3_PDL
    pdl_trigger();
#endif

    // the CTA's warps, in K order (wk), into the reduction buffer [16 MT][128]
    for (int step = 0; step < WK; ++step) {
        if (wk == step) {
#pragma unroll
            for (int a = 0; a < MT; ++a)
#pragma unroll
                for (int j = 0; j < NTW; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (wn * NTW + j) * 16 + h * 8 + 2 * t;
                        float* r_lo = red + (16 * a + g) * RP + col;
                        float* r_hi = red + (16 * a + 8 + g) * RP + col;
                        if (step == 0) {
                            r_lo[0] = acc[a][j][h][0]; r_lo[1] = acc[a][j][h][1];
                            r_hi[0] = acc[a][j][h][2]; r_hi[1] = acc[a][j][h][3];
                        } else {
                            r_lo[0] += acc[a][j][h][0]; r_lo[1] += acc[a][j][h][1];
                            r_hi[0] += acc[a][j][h][2]; r_hi[1] += acc[a][j][h][3];
                        }
                    }
        }
        __syncthreads();
    }

    const int nthreads = 32 * WN * WK;
    const int rows = min(M, 16 * MT);
    if (S > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        if (rank == 0) {
            for (int r = 1; r < S; ++r) {
                const float* rem = cluster.map_shared_rank(red, r);
                for (int e = threadIdx.x; e < rows * 128; e += nthreads) {
                    const int m = e >> 7, c = e & 127;
                    red[m * RP + c] += rem[m * RP + c];
                }
                __syncthreads();
            }
        }
        cluster.sync();
        if (rank != 0) return;
    }

    // output rotation: one warp a row
    for (int m = warp; m < rows; m += WN * WK) {
        const float* rr = red + m * RP + 4 * lane;
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

template <int KB, int MT, int NTW, int WN, int WK, int ST, bool OF32, bool XSM, int MINB>
void launch_gemm(const at::Tensor& xh, const at::Tensor& t, const at::Tensor& svh, at::Tensor& y,
                 int M, int Kin, int N, int ldt, int S, int b1, int b2, cudaStream_t stream) {
    constexpr int TW = 8 * KB;
    const size_t smem = (size_t)WN * WK * ST * (NTW * TW + (XSM ? 16 * MT * 12 : 0)) * 4 + (size_t)16 * MT * 132 * 4;
    auto kern = gemm_kernel<KB, MT, NTW, WN, WK, ST, OF32, XSM, MINB>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        attr = true;
    }
    cudaLaunchConfig_t cfg = {};
    const int R = (M + 16 * MT - 1) / (16 * MT);    // row chunks (ENG-213 row grid)
    cfg.gridDim = dim3(S * R, N / 128, 1);
    cfg.blockDim = dim3(32 * WN * WK, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = S > 1 ? 1 : 0;           // one CTA a column block: a plain launch
#if EXL3_PDL
    cudaLaunchAttribute at2[2] = {at[0]};
    at2[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at2[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at2;
    cfg.numAttrs += 1;
#endif
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern,
        (const __half*)xh.data_ptr(), (const uint32_t*)t.data_ptr(), (const __half*)svh.data_ptr(),
        (void*)y.data_ptr(), (int)y.stride(0), M, Kin, N, ldt, b1, b2, S);
    TORCH_CHECK(e == cudaSuccess, "exl3 gemm launch: ", cudaGetErrorString(e));
}

}  // namespace

// cfg: 0 -> MT1 NTW8 WN1 (activations staged), 1 -> MT1 NTW4 WN2 (registers, 2 CTAs an SM),
//      2 -> MT2 NTW4 WN2, 3 -> MT2 NTW2 WN4, 4 -> MT1 NTW8 WN1 (registers)
#define EXL3_WK_CASES(KB, MT, NTW, WN, OF32, XSM, MINB)                                          \
    switch (wk) {                                                                                \
        case 4: launch_gemm<KB, MT, NTW, WN, 4, EXL3_ST, OF32, XSM, MINB>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
        case 8: launch_gemm<KB, MT, NTW, WN, 8, EXL3_ST, OF32, XSM, MINB>(xh, t, svh, y, M, Kin, N, ldt, S, b1, b2, st); return; \
        default: TORCH_CHECK(false, "exl3 gemm: wk must be 4 or 8, got ", wk);                   \
    }
#define EXL3_CFG_CASES(KB, OF32)                                                                 \
    switch (cfg) {                                                                               \
        case 0: EXL3_WK_CASES(KB, 1, 8, 1, OF32, true, 1)                                        \
        case 1: EXL3_WK_CASES(KB, 1, 4, 2, OF32, false, 1)                                       \
        case 2: EXL3_WK_CASES(KB, 2, 4, 2, OF32, false, 1)                                       \
        case 3: EXL3_WK_CASES(KB, 2, 2, 4, OF32, false, 1)                                       \
        case 4: EXL3_WK_CASES(KB, 1, 8, 1, OF32, false, 1)                                       \
        default: TORCH_CHECK(false, "exl3 gemm: bad cfg ", cfg);                                 \
    }
#define EXL3_KB_CASES(OF32)                                                                      \
    switch (kb) {                                                                                \
        case 2: EXL3_CFG_CASES(2, OF32)                                                          \
        case 3: EXL3_CFG_CASES(3, OF32)                                                          \
        case 4: EXL3_CFG_CASES(4, OF32)                                                          \
        case 5: EXL3_CFG_CASES(5, OF32)                                                          \
        case 6: EXL3_CFG_CASES(6, OF32)                                                          \
        case 7: EXL3_CFG_CASES(7, OF32)                                                          \
        case 8: EXL3_CFG_CASES(8, OF32)                                                          \
        default: TORCH_CHECK(false, "exl3 gemm: K bits must be 2..8, got ", kb);                 \
    }

void exl3_gemm(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y,
               int64_t kb, int64_t cfg, int64_t wk, int64_t S, int64_t b1, int64_t b2, int64_t rmax) {
    const at::cuda::OptionalCUDAGuard guard(xh.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous() && xh.dim() == 3, "xh: fp16 [G, M, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3, "trellis: int16 [Kin/16, N/16, 16K]");
    TORCH_CHECK(t.size(2) == 16 * kb && t.stride(2) == 1 && t.stride(1) == 16 * kb, "trellis tiles must be dense, K ", kb);
    const int M = xh.size(1), Kin = xh.size(2), N = t.size(1) * 16;
    const int ldt = t.stride(0) / t.stride(1);
    TORCH_CHECK(t.size(0) * 16 == Kin, "trellis K tiles vs Kin");
    TORCH_CHECK(N % 128 == 0 && Kin % 128 == 0, "N and Kin must be multiples of 128");
    TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "y [M, N]");
    TORCH_CHECK(M >= 1 && M <= ((cfg == 2 || cfg == 3) ? 32 : 16) * rmax, "rows ", M, " for cfg ", cfg, " and ", rmax, " row chunks");
    TORCH_CHECK(S >= 1 && S <= 16, "cluster size");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh fp16 [N]");
    if (y.dtype() == torch::kFloat) { EXL3_KB_CASES(true) }
    else { TORCH_CHECK(y.dtype() == torch::kBFloat16, "y bf16 or fp32"); EXL3_KB_CASES(false) }
}

void exl3_rot_in(torch::Tensor x, torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x bf16 [M, Kin]");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.dim() == 2 && suh.is_contiguous(), "suh fp16 [G, Kin]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous(), "xh fp16 [G, M, Kin]");
    const int M = x.size(0), Kin = x.size(1), G = suh.size(0);
    TORCH_CHECK(Kin % 128 == 0 && suh.size(1) == Kin, "Kin");
    dim3 grid((Kin / 128 + 3) / 4, M, G);
    rot_in_kernel<<<grid, 128, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0),
        (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Kin);
}

void exl3_rot_in_silu(torch::Tensor y, torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(y.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.dim() == 2 && y.stride(1) == 1, "y bf16 [M, 2 Kin]");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.dim() == 2 && suh.size(0) == 1 && suh.is_contiguous(), "suh fp16 [1, Kin]");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.is_contiguous(), "xh fp16 [1, M, Kin]");
    const int M = y.size(0), Kin = suh.size(1);
    TORCH_CHECK(Kin % 128 == 0 && y.size(1) == 2 * Kin && xh.numel() == (int64_t)M * Kin, "shapes");
    TORCH_CHECK(y.stride(0) % 4 == 0 && ((uintptr_t)y.data_ptr()) % 8 == 0, "y alignment");
    dim3 grid((Kin / 128 + 3) / 4, M, 1);
    rot_in_silu_kernel<<<grid, 128, 0, st>>>((const __nv_bfloat16*)y.data_ptr(), (int)y.stride(0),
        (const __half*)suh.data_ptr(), (__half*)xh.data_ptr(), M, Kin);
}

void exl3_mlp_bridge(torch::Tensor yg, torch::Tensor yu, torch::Tensor svg, torch::Tensor svu,
                     torch::Tensor suh, torch::Tensor xh) {
    const at::cuda::OptionalCUDAGuard guard(yg.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(yg.dtype() == torch::kFloat && yu.dtype() == torch::kFloat && yg.sizes() == yu.sizes(), "yg/yu fp32, same shape");
    TORCH_CHECK(yg.stride(1) == 1 && yu.stride(1) == 1 && yg.stride(0) == yu.stride(0), "yg/yu rows");
    const int M = yg.size(0), N = yg.size(1);
    TORCH_CHECK(N % 128 == 0 && svg.numel() == N && svu.numel() == N && suh.numel() == N, "bridge columns");
    TORCH_CHECK(xh.dtype() == torch::kHalf && xh.stride(1) == 1 && xh.size(0) == M && xh.size(1) == N, "xh fp16 [M, N] view");
    TORCH_CHECK(svg.is_contiguous() && svu.is_contiguous() && suh.is_contiguous(), "factors contiguous");
    dim3 grid((N / 128 + 3) / 4, M);
    mlp_bridge_kernel<<<grid, 128, 0, st>>>((const float*)yg.data_ptr(), (const float*)yu.data_ptr(), (int)yg.stride(0),
        (const __half*)svg.data_ptr(), (const __half*)svu.data_ptr(), (const __half*)suh.data_ptr(),
        (__half*)xh.data_ptr(), (int)xh.stride(0), N);
}

void exl3_rot_out(torch::Tensor yp, torch::Tensor svh, torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard guard(yp.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int M = yp.size(0), N = yp.size(1);
    TORCH_CHECK(N % 128 == 0 && svh.numel() == N && yp.stride(1) == 1 && y.stride(1) == 1, "rot_out shapes");
    dim3 grid((N / 128 + 3) / 4, M);
    if (yp.dtype() == torch::kFloat && y.dtype() == torch::kBFloat16)
        rot_out_kernel<float, __nv_bfloat16><<<grid, 128, 0, st>>>((const float*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), N);
    else if (yp.dtype() == torch::kFloat && y.dtype() == torch::kFloat)
        rot_out_kernel<float, float><<<grid, 128, 0, st>>>((const float*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (float*)y.data_ptr(), (int)y.stride(0), N);
    else if (yp.dtype() == torch::kHalf && y.dtype() == torch::kBFloat16)
        rot_out_kernel<__half, __nv_bfloat16><<<grid, 128, 0, st>>>((const __half*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), N);
    else if (yp.dtype() == torch::kHalf && y.dtype() == torch::kFloat)
        rot_out_kernel<__half, float><<<grid, 128, 0, st>>>((const __half*)yp.data_ptr(), (int)yp.stride(0),
            (const __half*)svh.data_ptr(), (float*)y.data_ptr(), (int)y.stride(0), N);
    else TORCH_CHECK(false, "rot_out dtypes");
}

void exl3_recon(torch::Tensor t, torch::Tensor wi, int64_t kb, int64_t n_off) {
    const at::cuda::OptionalCUDAGuard guard(t.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.size(2) == 2 * tile_u32(kb) && t.stride(2) == 1 && t.stride(1) == t.size(2), "trellis");
    TORCH_CHECK(wi.dtype() == torch::kHalf && wi.stride(1) == 1, "wi fp16");
    const int Kin = t.size(0) * 16, ntn = t.size(1), N = wi.size(1);
    const int ldt = t.stride(0) / t.stride(1);
    TORCH_CHECK(wi.size(0) == Kin && N % 128 == 0 && n_off % 128 == 0 && n_off + N <= ntn * 16, "recon shapes");
    dim3 grid(N / 128, Kin / 16);
    const uint16_t* tp = (const uint16_t*)t.data_ptr();
    __half* wp = (__half*)wi.data_ptr();
    const int ld = wi.stride(0), off = n_off / 16;
    switch (kb) {
        case 2: recon_kernel<2><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 3: recon_kernel<3><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 4: recon_kernel<4><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 5: recon_kernel<5><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 6: recon_kernel<6><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 7: recon_kernel<7><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 8: recon_kernel<8><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 15: recon_kernel<15><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 25: recon_kernel<25><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        case 35: recon_kernel<35><<<grid, 256, 0, st>>>(tp, ldt, off, wp, ld); break;
        default: TORCH_CHECK(false, "recon: K bits 2..8, or 15 / 25 / 35 for 1.5 / 2.5 / 3.5");
    }
}
