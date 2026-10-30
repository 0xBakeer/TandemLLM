// rot_in_kernel's values in fragment order; grid (Kin/128/4, Mp, G): rows M..Mp-1 are zeros
__global__ void __launch_bounds__(128) rot_in_frag_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Mp, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int m = blockIdx.y, g = blockIdx.z;
    const int k = blk * 128 + 4 * lane;
    uint32_t w0 = 0u, w1 = 0u;
    if (m < M) {
        const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)m * ldx + k);
        const uint2 sr = *reinterpret_cast<const uint2*>(SUH + (size_t)g * Kin + k);
        float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
        float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
        v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
        wht128(v0, v1, v2, v3, lane);
        __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
        __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
        w0 = *reinterpret_cast<uint32_t*>(&o01); w1 = *reinterpret_cast<uint32_t*>(&o23);
    }
    __half* xh = XH + (size_t)g * Mp * Kin;
    *reinterpret_cast<uint32_t*>(xh + frag_off(m, k, Kin)) = w0;
    *reinterpret_cast<uint32_t*>(xh + frag_off(m, k + 2, Kin)) = w1;
}

// rot_in_silu_kernel's values in fragment order; grid (Kin/128/4, Mp): rows M..Mp-1 are zeros
__global__ void __launch_bounds__(128) rot_in_silu_frag_kernel(const __nv_bfloat16* __restrict__ Y, int ldy,
        const __half* __restrict__ SUH, __half* __restrict__ XH, int M, int Kin) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= Kin) return;
    const int m = blockIdx.y;
    const int k = blk * 128 + 4 * lane;
    uint32_t w0 = 0u, w1 = 0u;
    if (m < M) {
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
        w0 = *reinterpret_cast<uint32_t*>(&o01); w1 = *reinterpret_cast<uint32_t*>(&o23);
    }
    *reinterpret_cast<uint32_t*>(XH + frag_off(m, k, Kin)) = w0;
    *reinterpret_cast<uint32_t*>(XH + frag_off(m, k + 2, Kin)) = w1;
}
