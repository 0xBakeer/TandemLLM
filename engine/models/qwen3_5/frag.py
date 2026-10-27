"""SPD-201: the 27B's EXL3 projections at 33..64 rows read fragment-major activations (QWEN38_EXL3_FRAG).

The row grid's register path loads a lane's A operands as four 4-byte loads a 16-row group and K tile, and at 64
rows its warps wait on them. Here the input rotation writes its fp16 values in the mma's A-fragment order (rows in
groups of 16, zero rows to a multiple of 16, K tiles, 32 lanes, a0..a3 of m16n8k16) and the gemm takes a lane's
operands in one 16-byte load. The values and every instruction on them are the same: `check` holds rows 33..64 to
the shared kernels' bits on the 27B's shapes, plain, grouped and silu.

The source is tools/exl3_linear.py's, with the gemm's activation offset and load and its row count from the output
replaced (each replaced text checked to be there once) and the two fragment-order rotations added, compiled as
a module of its own. Only this family's weights take it: `install(eng)` tags the EXL3 blocks of the engine's
weights and wraps the shared `exl3_matmul` and `exl3_down_silu` once; every other call, every other family's
block, runs the shared functions unchanged. tools/exl3_linear.py keeps its bytes.

Per call at 48..64 rows, rotation included: -3 to -10 % on every 27B shape; at 32 rows and below flat (not taken).
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch

from engine.settings import SETTINGS as _S

ON = _S.get("EXL3_FRAG") == "1"
#: the most rows a call takes the fragment-major path at (the joint pass's row cap)
MAX_ROWS = 64
STATS = {"frag": 0, "other": 0}
#: the ids of this family's tagged blocks (an EXL3Block takes no new attribute)
TAGGED: set = set()
_ORIG = None
_MOD = None

_OFF_OLD = "    const __half* xh = XH + (size_t)member * Mtot * Kin + (size_t)chunk * 16 * MT * Kin;"
_OFF_NEW = "    const __half* xh = XH + (size_t)member * ((Mtot + 15) / 16 * 16) * Kin + (size_t)chunk * 16 * MT * Kin;"
_LOAD_OLD = """        const int kk = kt * 16 + 2 * t;
#pragma unroll
        for (int a = 0; a < MT; ++a) {
            const int ra = 16 * a + g, rb = ra + 8;
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(xh + (size_t)ra * Kin + kk);
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(xh + (size_t)rb * Kin + kk);
            xn[a][0] = ra < M ? __ldg(pa) : 0u;
            xn[a][1] = rb < M ? __ldg(pb) : 0u;
            xn[a][2] = ra < M ? __ldg(pa + 4) : 0u;
            xn[a][3] = rb < M ? __ldg(pb + 4) : 0u;
        }"""
_LOAD_NEW = """#pragma unroll
        for (int a = 0; a < MT; ++a) {
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (16 * a < M) v = __ldg(reinterpret_cast<const uint4*>(xh + (size_t)16 * a * Kin) + (size_t)kt * 32 + lane);
            xn[a][0] = v.x; xn[a][1] = v.y; xn[a][2] = v.z; xn[a][3] = v.w;
        }"""
_ROWS_OLD = "    const int M = xh.size(1), Kin = xh.size(2), N = t.size(1) * 16;"
_ROWS_NEW = ("    const int M = y.size(0), Kin = xh.size(2), N = t.size(1) * 16;\n"
             "    TORCH_CHECK(xh.size(1) == (M + 15) / 16 * 16 && cfg != 0, "
             "\"fragment-major xh [G, M to 16, Kin], a register cfg\");")
_HELPER = r"""// SPD-201: the rotated input in the mma's A-fragment order: a member's rows in groups of 16 (zero rows to a
// multiple of 16), a group's K tiles in order, a tile's 32 lanes, a lane's a0..a3 of m16n8k16, so a lane's
// operands are one 16-byte load. Only where the values sit moves.
__device__ __forceinline__ size_t frag_off(int m, int k, int Kin) {
    const int rg = m >> 4, r_hi = (m >> 3) & 1, r_lo = m & 7;
    const int kt = k >> 4, k_hi = (k >> 3) & 1, t = (k >> 1) & 3, pr = k & 1;
    return ((((((size_t)rg * (Kin >> 4) + kt) * 8 + r_lo) * 4 + t) * 2 + k_hi) * 2 + r_hi) * 2 + pr;
}
"""
_KERNELS = r"""// rot_in_kernel's values in fragment order; grid (Kin/128/4, Mp, G): rows M..Mp-1 are zeros
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
"""
_LAUNCHERS = r"""void exl3_rot_in_frag(torch::Tensor x, torch::Tensor suh, torch::Tensor xh) {
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
"""
_CPP_ADD = ("void exl3_rot_in_frag(torch::Tensor x, torch::Tensor suh, torch::Tensor xh);\n"
            "void exl3_rot_in_silu_frag(torch::Tensor y, torch::Tensor suh, torch::Tensor xh);\n")


def _source() -> str:
    from tools import exl3_linear as X
    src = X._CUDA
    for old, new in ((_OFF_OLD, _OFF_NEW), (_LOAD_OLD, _LOAD_NEW), (_ROWS_OLD, _ROWS_NEW)):
        assert src.count(old) == 1, "tools/exl3_linear.py's kernel moved: " + old.strip()[:60]
        src = src.replace(old, new)
    anchor = "constexpr float RSQRT128 = 0.08838834764831845f;\n"
    assert src.count(anchor) == 1, "tools/exl3_linear.py's kernel moved: RSQRT128"
    src = src.replace(anchor, anchor + "\n" + _HELPER + "\n" + _KERNELS)
    return src + "\n" + _LAUNCHERS


def module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        from tools import exl3_linear as X
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_PDL {int(X.PDL)}\n" + _source()    # SPD-245
        cpp = X._CPP + _CPP_ADD
        tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"qwen35_exl3_frag_{tag}", cpp_sources=[cpp], cuda_sources=[src],
                           functions=["exl3_gemm", "exl3_rot_in_frag", "exl3_rot_in_silu_frag"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _MOD


def _cfg(w, M: int):
    """The register-path cfg a fragment-major call of M rows takes, or None (the shared path)."""
    from tools import exl3_linear as X
    if not (id(w) in TAGGED and X.ROWGRID and X.SKINNY_MAX < M <= min(MAX_ROWS, X.CHUNK_UNTIL)
            and w.kb < 10):
        return None
    c = X.pick(w.tile_n, w.K, w.kb)["cfg2"]
    return c if c in (2, 3) else None


def _gemm(xf, w, M: int, c: int, dev, out=None):
    from tools import exl3_linear as X
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    module().exl3_gemm(xf, w.t, w.svh, out, w.kb, c, tile["wk"], tile["S"], w.bounds[0], w.bounds[1],
                       (M + X.SKINNY_MAX - 1) // X.SKINNY_MAX)
    return out


def matmul(x, w, *, out=None, cfg=None, S=None, wk=None, xh=None, M=None, dev=None):
    """`exl3_matmul` with this family's tagged blocks at 33..64 rows on the fragment-major path."""
    if xh is None and cfg is None and S is None and wk is None and x is not None and x.is_cuda:
        c = _cfg(w, x.shape[0])
        if c is not None:
            if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
                x = x.contiguous()
                if x.data_ptr() % 8:
                    x = x.clone()
            n = x.shape[0]
            xf = torch.empty(w.suh.shape[0], (n + 15) // 16 * 16, w.K, dtype=torch.float16, device=x.device)
            module().exl3_rot_in_frag(x, w.suh, xf)
            STATS["frag"] += 1
            return _gemm(xf, w, n, c, x.device, out)
    STATS["other"] += 1
    return _ORIG[0](x, w, out=out, cfg=cfg, S=S, wk=wk, xh=xh, M=M, dev=dev)


def down_silu(y, w):
    """`exl3_down_silu` with this family's tagged blocks at 33..64 rows on the fragment-major path."""
    if y.is_cuda:
        c = _cfg(w, y.shape[0])
        if c is not None:
            if y.stride(1) != 1 or y.stride(0) % 4 or y.data_ptr() % 8:
                y = y.contiguous()
            n = y.shape[0]
            xf = torch.empty(1, (n + 15) // 16 * 16, w.K, dtype=torch.float16, device=y.device)
            module().exl3_rot_in_silu_frag(y, w.suh, xf)
            STATS["frag"] += 1
            return _gemm(xf, w, n, c, y.device)
    return _ORIG[1](y, w)


def _tag(w) -> int:
    from tools.exl3_linear import EXL3Block
    n = 0
    for d in (getattr(w, "q", {}), getattr(w, "g", {}), getattr(w, "t", {})):
        for blk in d.values():
            if isinstance(blk, EXL3Block) and id(blk) not in TAGGED:
                TAGGED.add(id(blk))
                n += 1
    return n


def install(eng) -> int:
    """Tag `eng`'s EXL3 blocks and wrap the shared functions (once). Returns the blocks tagged."""
    if not ON:
        return 0
    from tools import exl3_linear as X
    global _ORIG
    if _ORIG is None:
        _ORIG = (X.exl3_matmul, X.exl3_down_silu)
        X.exl3_matmul, X.exl3_down_silu = matmul, down_silu
    return _tag(getattr(eng, "w", None))


def check(rows=(33, 40, 47, 48, 56, 60, 63, 64)) -> list[str]:
    """The fragment-major path against the shared one, bit for bit, on the 27B's projection shapes (plain, a group
    of two members, and the down projection's silu input). A GPU."""
    global _ORIG
    from tools import exl3_linear as X
    if _ORIG is None:
        _ORIG = (X.exl3_matmul, X.exl3_down_silu)
    out = []
    for N, K, kb in ((10240, 5120, 4), (5120, 6144, 4), (5120, 17408, 5), (5120, 5120, 3)):
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        y = (torch.randn(max(rows), 2 * K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        bad = []
        for M in rows:
            TAGGED.discard(id(w))
            ref, refd = _ORIG[0](x[:M], w), _ORIG[1](y[:M], w)
            TAGGED.add(id(w))
            got, gotd = matmul(x[:M], w), down_silu(y[:M], w)
            if not (torch.equal(ref, got) and torch.equal(refd, gotd)):
                bad.append(M)
        out.append(f"{N}x{K} K{kb}: " + ("rows 33..64 the shared kernels' bits (plain and silu)" if not bad
                                          else f"DIFFER at {bad}"))
        TAGGED.discard(id(w))
        assert not bad
    return out
