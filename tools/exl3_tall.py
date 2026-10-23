"""SPD-163 step 8: the EXL3 skinny GEMM with a 64-row tile, for the 27B's joint verify pass and packed draft.

`tools/exl3_linear.py` takes 33-64 rows as two adjacent 32-row chunks (the row grid, ENG-213): the second
chunk reads the trellis from L2, but decodes it again, and on GB10 the decode is the kernel's work. In
the 27B's served joint pass (64 rows at four or eight slots) that second decode is the 32-to-64-row step
of the joint staircase (E=4: 84 -> 100 ms, E=8: 101 -> 117 ms a pass); the packed draft (8 x 16 rows) runs
four chunks.

This module builds the same kernel source with MT = 4 (four 16-row tiles a CTA, one decode of each trellis
tile for all 64 rows) as its own extension, and leaves `tools/exl3_linear.py`'s bytes (and every other
family's fingerprint) alone; only the 27B's engine/model.py calls it. A row's arithmetic is the MT = 2
kernel's: each 16-row tile `a` keeps its own accumulator, the K range is split into the same rank and
warp slices (`pick`: a function of the weight's shape only) and reduced in the same order, so every row is
the bits the row grid gives it whatever the row count. `check()` holds that on the 27B's projection shapes
at every row count 1..128.

cfg (the column tiling, never the order): 5 -> MT4 NTW2 WN4, 6 -> MT4 NTW4 WN2 (the activations in
registers).
"""

from __future__ import annotations

import hashlib
import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tools import exl3_linear as X  # noqa: E402

#: rows a tall chunk takes
TALL = 64

_EXTRA = r"""
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
"""

_CPP = ("void exl3_gemm_tall(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y, "
        "int64_t kb, int64_t cfg, int64_t wk, int64_t S, int64_t b1, int64_t b2);\n")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        src = f"#define EXL3_ST {X.STAGES}\n" + X._CUDA + _EXTRA
        tag = hashlib.sha1((_CPP + src).encode()).hexdigest()[:10]
        _MOD = load_inline(name=f"qwen38_exl3_tall_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                           functions=["exl3_gemm_tall"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _MOD


# ---- source variants of the shared kernel (built as their own extensions; exl3_linear.py's bytes stay) ----
_MMA = """            for (int a = 0; a < MT; ++a) {
                mma(acc[a][j][0], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f0.r[0], f0.r[1]);
                mma(acc[a][j][1], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f1.r[0], f1.r[1]);
            }"""
#: "skip": a 16-row group past the chunk's rows issues no mma (its rows are never stored: the row grid's
#: last chunk of 33..48 rows paid two groups for one); every row < M runs the same instructions
_SKIP = """            for (int a = 0; a < MT; ++a) {
                if (16 * a < M) {
                    mma(acc[a][j][0], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f0.r[0], f0.r[1]);
                    mma(acc[a][j][1], xa[a][0], xa[a][1], xa[a][2], xa[a][3], f1.r[0], f1.r[1]);
                }
            }"""
#: "nomma" (timing only, wrong numbers): the mma replaced by an xor into the accumulator that keeps the decode
#: and the loads live -- the discriminator between the tensor pipe and the decode's ALU work
_NOMMA = """            for (int a = 0; a < MT; ++a) {
                acc[a][j][0][0] += __uint_as_float((f0.r[0] ^ xa[a][0]) & 0x3f800000u);
                acc[a][j][1][0] += __uint_as_float((f1.r[1] ^ xa[a][1]) & 0x3f800000u);
            }"""
_VARIANTS: dict = {}


def variant(name: str):
    """The shared kernels with the mma loop as `name` says ("skip", "nomma"), as their own extension."""
    mod = _VARIANTS.get(name)
    if mod is None:
        from torch.utils.cpp_extension import load_inline
        body = {"skip": _SKIP, "nomma": _NOMMA}[name]
        assert X._CUDA.count(_MMA) == 1, "the shared kernel's mma loop moved"
        src = f"#define EXL3_ST {X.STAGES}\n" + X._CUDA.replace(_MMA, body)
        tag = hashlib.sha1((X._CPP + src).encode()).hexdigest()[:10]
        mod = _VARIANTS[name] = load_inline(
            name=f"qwen38_exl3_{name}_{tag}", cpp_sources=[X._CPP], cuda_sources=[src], functions=["exl3_gemm"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"], verbose=False)
    return mod


def skip_matmul(x: torch.Tensor, w, *, xh: torch.Tensor | None = None, out: torch.Tensor | None = None,
                mod_name: str = "skip") -> torch.Tensor:
    """`exl3_matmul`'s row grid (33..CHUNK_UNTIL rows) on the dead-group-skip kernel: the same bits."""
    M = x.shape[0] if x is not None else xh.shape[1]
    dev = (x if x is not None else xh).device
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    variant(mod_name).exl3_gemm(X.rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, tile["cfg2"],
                                tile["wk"], tile["S"], w.bounds[0], w.bounds[1], (M + X.SKINNY_MAX - 1) // X.SKINNY_MAX)
    return out


def discriminate(shape=(10240, 5120, 4), rows=(16, 32, 48, 64)) -> list[str]:
    """The row grid's time with and without its mma (the decode and loads kept), by row count: tensor-bound
    when the full kernel grows with rows and the no-mma one does not."""
    import time
    N, K, kb = shape
    w = X.random_block(N, K, kb, seed=1)
    x = (torch.randn(max(rows), K, generator=torch.Generator().manual_seed(2)) * 0.7).to(torch.bfloat16).cuda()
    out = []
    for M in rows:
        xs = x[:M]
        res = []
        for name, fn in (("full", lambda: X.exl3_matmul(xs, w)), ("skip", lambda: skip_matmul(xs, w)),
                         ("nomma", lambda: skip_matmul(xs, w, mod_name="nomma"))):
            if M <= X.SKINNY_MAX and name != "full":
                continue
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            res.append(f"{name} {(time.perf_counter() - t0) / 50 * 1e6:.1f} us")
        out.append(f"{N}x{K} K{kb} M={M}: " + ", ".join(res))
    return out


def check_skip(rows=range(33, 129)) -> list[str]:
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        x = (torch.randn(max(rows), K, generator=torch.Generator().manual_seed(N)) * 0.7).to(torch.bfloat16).cuda()
        ref = X.exl3_matmul(x, w)
        for M in rows:
            assert torch.equal(skip_matmul(x[:M], w), X.exl3_matmul(x[:M], w)), (N, K, kb, M)
        out.append(f"{N}x{K} K{kb}: rows {min(rows)}..{max(rows)} on the dead-group skip = the row grid's bits")
    return out


#: the column tiling a tall call takes (the order is `pick`'s); per (N, Kin, kb) overrides from a sweep
CFG = int(os.environ.get("QWEN38_EXL3_TALL_CFG", "5") or 5)
_CFG_BY: dict = {}


def eligible(w, M: int) -> bool:
    """A call the tall kernel takes: an integer-rate EXL3 projection at 33..CHUNK_UNTIL rows."""
    return (isinstance(w, X.EXL3Block) and w.kb < 10 and X.SKINNY_MAX < M <= X.CHUNK_UNTIL
            and w.t.is_cuda)


def tall_matmul(x: torch.Tensor, w, *, cfg: int | None = None, xh: torch.Tensor | None = None,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """`exl3_matmul(x, w)` for 33..CHUNK_UNTIL rows, the rows in 64-row tiles (the same bits)."""
    M = x.shape[0] if x is not None else xh.shape[1]
    dev = (x if x is not None else xh).device
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    c = cfg if cfg is not None else _CFG_BY.get((w.tile_n, w.K, w.kb), CFG)
    _module().exl3_gemm_tall(X.rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, c, tile["wk"],
                             tile["S"], w.bounds[0], w.bounds[1])
    return out


#: the 27B's projection shapes (N, Kin, K bits) as the ThinkingCap EXL3 4.0 pack holds them, and the drafter
SHAPES = ((16384, 5120, 4), (10240, 5120, 4), (5120, 6144, 4), (34816, 5120, 4), (5120, 17408, 5),
          (14336, 5120, 4), (5120, 12288, 4), (1024, 5120, 4), (248320, 5120, 4), (5120, 5120, 3))


def check(rows=range(1, 129), bench: bool = True) -> list[str]:
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768                                   # the head's columns, a slice of them
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        ref = X.exl3_matmul(x, w)
        for M in rows:
            if M <= X.SKINNY_MAX:
                continue
            for c in (5, 6):
                got = tall_matmul(x[:M], w, cfg=c)
                assert torch.equal(got, ref[:M]), (N, K, kb, M, c)
            assert torch.equal(X.exl3_matmul(x[:M], w), ref[:M]), ("row grid", N, K, M)
        line = f"{N}x{K} K{kb}: rows 33..{max(rows)} through the 64-row tile (cfg 5/6) = the row grid's bits"
        if bench:
            line += "; " + _bench(x, w)
        out.append(line)
    return out


def _bench(x, w) -> str:
    import time
    res = []
    for M in (64, 128):
        xs = x[:M]
        for name, fn in (("grid", lambda: X.exl3_matmul(xs, w)),) + tuple(
                (f"tall{c}", (lambda c=c: tall_matmul(xs, w, cfg=c))) for c in (5, 6)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            res.append(f"M{M} {name} {(time.perf_counter() - t0) / 20 * 1e3:.3f} ms")
    return ", ".join(res)


if __name__ == "__main__":
    if "--discriminate" in sys.argv:
        for shape in ((10240, 5120, 4), (17408, 5120, 4), (5120, 17408, 5)):
            for line in discriminate(shape):
                print("exl3_tall:", line, flush=True)
        for line in check_skip():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    for line in check():
        print("exl3_tall:", line, flush=True)
