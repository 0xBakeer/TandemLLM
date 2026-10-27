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

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import native_build as nb  # noqa: E402
from tools import exl3_linear as X  # noqa: E402

#: rows a tall chunk takes
TALL = 64

_EXTRA = nb.source(__file__, "tall.cu")

_CPP = nb.source(__file__, "tall.cpp")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        src = f"#define EXL3_ST {X.STAGES}\n" + X._CUDA + _EXTRA
        tag = nb.tag(_CPP, src)
        _MOD = nb.load(f"qwen38_exl3_tall_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["exl3_gemm_tall"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
    return _MOD


# ---- T64 (SPD-199): one decode of each trellis tile for all 64 rows of a chunk ----------------------------
# A warp owns one 16-column tile of the CTA's 128 columns for all MT 16-row groups and walks the CTA's whole
# K range; `pick`'s WK warp slices become WK consecutive stretches of that walk. A stretch starts from zero
# and is added into the reduction buffer when it ends (the first with `=`, the rest with `+=`, empty ones
# too), so every output element is the production kernel's ((s0 + s1) + s2) ... over the same K tiles in the
# same mma chain; the cluster fold and the output rotation are the production kernel's lines.
_T64 = nb.source(__file__, "t64.cu")
_T64_CPP = nb.source(__file__, "t64.cpp")
#: column tiles a T64 warp owns (WN = 8 / NTW warps a CTA); the order is the same at any NTW
T64_NTW = int(os.environ.get("QWEN38_EXL3_T64_NTW", "2") or 2)
#: cp.async stages a T64 warp keeps in flight (one 16x16 tile a stage; a function of nothing that moves bits)
T64_ST = int(os.environ.get("QWEN38_EXL3_T64_ST", "8") or 8)
_T64_MOD = None


def _t64_module():
    global _T64_MOD
    if _T64_MOD is None:
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_T64_ST {T64_ST}\n" + X._CUDA + _T64
        tag = nb.tag(_T64_CPP, src)
        _T64_MOD = nb.load(f"qwen38_exl3_t64_{tag}", cpp_sources=[_T64_CPP], cuda_sources=[src],
                           functions=["exl3_gemm_t64"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _T64_MOD


def t64_matmul(x: torch.Tensor, w, *, mt: int = 4, ntw: int | None = None, xh: torch.Tensor | None = None,
               out: torch.Tensor | None = None) -> torch.Tensor:
    """`exl3_matmul(x, w)` with each trellis tile decoded once for 16 mt rows (the same bits)."""
    M = x.shape[0] if x is not None else xh.shape[1]
    dev = (x if x is not None else xh).device
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    _t64_module().exl3_gemm_t64(X.rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, mt,
                                T64_NTW if ntw is None else ntw, tile["wk"], tile["S"], w.bounds[0], w.bounds[1])
    return out


def check_t64(rows=(1, 2, 7, 16, 17, 31, 32, 33, 40, 48, 63, 64, 65, 96, 127, 128), bench: bool = True) -> list[str]:
    """T64 (mt 4 and 8) against the production kernels' bits at every row count named, on the 27B's shapes;
    with `bench`, the times at 32, 64 and 128 rows."""
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        for M in rows:
            ref = X.exl3_matmul(x[:M], w)
            for mt in (4, 8):
                for ntw in (1, 2, 4):
                    assert torch.equal(t64_matmul(x[:M], w, mt=mt, ntw=ntw), ref), ("t64", N, K, kb, M, mt, ntw)
        line = f"{N}x{K} K{kb} ST{T64_ST}: rows {min(rows)}..{max(rows)} through T64 (mt 4/8, ntw 1/2/4) = the production bits"
        if bench:
            line += "; " + _bench_t64(x, w)
        out.append(line)
    return out


def _bench_t64(x, w) -> str:
    import time
    res = []
    for M in (32, 64, 128):
        xs = x[:M]
        for name, fn in (("prod", lambda: X.exl3_matmul(xs, w)),) + tuple(
                (f"t{16 * mt}n{ntw}", (lambda mt=mt, ntw=ntw: t64_matmul(xs, w, mt=mt, ntw=ntw)))
                for mt in (4, 8) for ntw in (1, 2, 4)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                fn()
            torch.cuda.synchronize()
            res.append(f"M{M} {name} {(time.perf_counter() - t0) / 30 * 1e6:.0f}us")
    return ", ".join(res)


# ---- P2 (SPD-199): each trellis tile decoded once for 64 rows, the production order kept ------------------------
# A CTA owns the production kernel's 128 columns and K range, for 64 rows. pick()'s WK warp slices run in phases
# of 4: 4 slice groups x 4 row-group warps (16 rows each); at each K step a group's 4 warps decode its 8 column
# tiles once into shared memory (two buffers, one named barrier a step) and every warp of the group applies them
# to its own 16 rows, its accumulators starting from zero for the slice. A phase's slices are added into the
# reduction buffer in slice order (= then +=, empty ones too), then the next phase's: every output is the
# production ((s0 + s1) + s2) ... over the same tiles and mma chain; the cluster fold and the rotation are the
# production lines.
_P2 = nb.source(__file__, "p2.cu")
_P2_CPP = nb.source(__file__, "p2.cpp")
#: raw trellis stages a P2 slice group keeps in flight
P2_ST = int(os.environ.get("QWEN38_EXL3_P2_ST", "4") or 4)
_P2_MOD = None


def _p2_module():
    global _P2_MOD
    if _P2_MOD is None:
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_P2_ST {P2_ST}\n" + X._CUDA + _P2
        tag = nb.tag(_P2_CPP, src)
        _P2_MOD = nb.load(f"qwen38_exl3_p2_{tag}", cpp_sources=[_P2_CPP], cuda_sources=[src],
                          functions=["exl3_gemm_p2"],
                          extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                          verbose=False)
    return _P2_MOD


def p2_matmul(x: torch.Tensor, w, *, xh: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """`exl3_matmul(x, w)` with each trellis tile decoded once for 64 rows (the same bits)."""
    M = x.shape[0] if x is not None else xh.shape[1]
    dev = (x if x is not None else xh).device
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    _p2_module().exl3_gemm_p2(X.rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, tile["wk"], tile["S"],
                              w.bounds[0], w.bounds[1])
    return out


def check_p2(rows=(1, 2, 7, 16, 17, 31, 32, 33, 40, 48, 63, 64, 65, 96, 127, 128), bench: bool = True) -> list[str]:
    """P2 against the production kernels' bits at every row count named, on the 27B's shapes; with `bench`, the
    times at 32, 64 and 128 rows."""
    import time
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        bad = [M for M in rows if not torch.equal(p2_matmul(x[:M], w), X.exl3_matmul(x[:M], w))]
        line = (f"{N}x{K} K{kb} wk{X.pick(w.tile_n, w.K, w.kb)['wk']} S{X.pick(w.tile_n, w.K, w.kb)['S']} P2ST{P2_ST}: "
                + ("rows 1..128 = the production bits" if not bad else f"DIFFER at rows {bad}"))
        if bench:
            res = []
            for M in (32, 64, 128):
                xs = x[:M]
                for name, fn in (("prod", lambda: X.exl3_matmul(xs, w)), ("p2", lambda: p2_matmul(xs, w))):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(30):
                        fn()
                    torch.cuda.synchronize()
                    res.append(f"M{M} {name} {(time.perf_counter() - t0) / 30 * 1e6:.0f}us")
            line += "; " + ", ".join(res)
        out.append(line)
    return out


# ---- P2C (SPD-199): P2 with the 128 columns split over a CTA pair ---------------------------------------------
# P2 holds 8 column tiles x 16 rows a warp (64 accumulators a lane), so one 512-thread CTA fills an SM's registers.
# Here a CTA owns 64 of the block's 128 columns (4 tiles) for 64 rows: 2 slice groups x 4 row-group warps (256
# threads, 32 accumulators a lane), pick()'s WK slices in phases of 2, each tile decoded once for the 64 rows. The
# two column halves and the S K ranks of one (chunk, column block) are one cluster of 2S CTAs: each half's rank 0
# folds its ranks in rank order (the production fold), then the first CTA reads the other half's 64 columns over
# the cluster and rotates the full 128-column rows (the production lines). Every output is the production sum:
# the same tiles, mma chain, slice order, fold and rotation.
_P2C = nb.source(__file__, "p2c.cu")
_P2C_CPP = nb.source(__file__, "p2c.cpp")
_P2C_MOD = None


def _p2c_module():
    global _P2C_MOD
    if _P2C_MOD is None:
        bar = _P2[:_P2.index("template <int KB, int WK, int ST, bool OF32>")]   # bar_named
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_P2_ST {P2_ST}\n" + X._CUDA + bar + _P2C
        tag = nb.tag(_P2C_CPP, src)
        _P2C_MOD = nb.load(f"qwen38_exl3_p2c_{tag}", cpp_sources=[_P2C_CPP], cuda_sources=[src],
                           functions=["exl3_gemm_p2c"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _P2C_MOD


def p2c_matmul(x: torch.Tensor, w, *, xh: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """`exl3_matmul(x, w)` with each trellis tile decoded once for 64 rows over a column-split CTA pair (the same bits)."""
    M = x.shape[0] if x is not None else xh.shape[1]
    dev = (x if x is not None else xh).device
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    _p2c_module().exl3_gemm_p2c(X.rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, tile["wk"],
                                tile["S"], w.bounds[0], w.bounds[1])
    return out


def check_p2c(rows=(1, 2, 7, 16, 17, 31, 32, 33, 40, 48, 63, 64, 65, 96, 127, 128), bench: bool = True) -> list[str]:
    """P2C against the production kernels' bits at every row count named, on the 27B's shapes; with `bench`, the
    times at 32, 64 and 128 rows against production and P2."""
    import time
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        bad = [M for M in rows if not torch.equal(p2c_matmul(x[:M], w), X.exl3_matmul(x[:M], w))]
        tile = X.pick(w.tile_n, w.K, w.kb)
        line = (f"{N}x{K} K{kb} wk{tile['wk']} S{tile['S']}: "
                + ("rows 1..128 = the production bits" if not bad else f"DIFFER at rows {bad}"))
        if bench:
            res = []
            for M in (32, 64, 128):
                xs = x[:M]
                for name, fn in (("prod", lambda: X.exl3_matmul(xs, w)), ("p2", lambda: p2_matmul(xs, w)),
                                 ("p2c", lambda: p2c_matmul(xs, w))):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(30):
                        fn()
                    torch.cuda.synchronize()
                    res.append(f"M{M} {name} {(time.perf_counter() - t0) / 30 * 1e6:.0f}us")
            line += "; " + ", ".join(res)
        out.append(line)
    return out


# ---- H64 (SPD-199): a column-half CTA pair, every K slice live, each trellis tile decoded once for 64 rows ------
# The row grid decodes a column block's trellis once a 32-row chunk, and the discriminator (no-mma kernel) says that
# decode, not the tensor pipe, is what 64 rows pay for. A CTA here owns 64 of the block's 128 columns for 64 rows:
# pick()'s WK slices are WK live warps a column pair (the production K split, all slices at once), a warp decodes
# its 2 column tiles once and applies them to 4 row groups (64 accumulators a lane, the production budget). The two
# column halves and the S ranks of one (chunk, column block) are one cluster of 2S CTAs; each half's rank 0 folds
# its ranks in rank order, then the first CTA rotates the full 128-column rows. Every output is the production sum:
# the same tiles, mma chain, slice order, fold and rotation.
_H64 = nb.source(__file__, "h64.cu")
_H64_CPP = nb.source(__file__, "h64.cpp")
#: cp.async stages a H64 warp keeps in flight (a function of nothing that moves bits)
H64_ST = int(os.environ.get("QWEN38_EXL3_H64_ST", "4") or 4)
#: the activations one K tile ahead in registers (16 more a lane) or loaded at the step
H64_PF = os.environ.get("QWEN38_EXL3_H64_PF", "0") == "1"
_H64_MOD = None


def _h64_module():
    global _H64_MOD
    if _H64_MOD is None:
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_H64_ST {H64_ST}\n" + X._CUDA + _H64
        tag = nb.tag(_H64_CPP, src)
        _H64_MOD = nb.load(f"qwen38_exl3_h64_{tag}", cpp_sources=[_H64_CPP], cuda_sources=[src],
                           functions=["exl3_gemm_h64"],
                           extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v",
                                              "-gencode=arch=compute_121a,code=sm_121a"],
                           verbose=False)
    return _H64_MOD


def h64_matmul(x: torch.Tensor, w, *, xh: torch.Tensor | None = None, out: torch.Tensor | None = None,
               pf: bool | None = None) -> torch.Tensor:
    """`exl3_matmul(x, w)` with each trellis tile decoded once for 64 rows over a column-half CTA pair (the same bits)."""
    M = x.shape[0] if x is not None else xh.shape[1]
    dev = (x if x is not None else xh).device
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=dev)
    tile = X.pick(w.tile_n, w.K, w.kb)
    _h64_module().exl3_gemm_h64(X.rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, tile["wk"],
                                tile["S"], w.bounds[0], w.bounds[1], int(H64_PF if pf is None else pf))
    return out


def check_h64(rows=(1, 2, 7, 16, 17, 31, 32, 33, 40, 48, 63, 64, 65, 96, 127, 128), bench: bool = True) -> list[str]:
    """H64 against the production kernels' bits at every row count named, on the 27B's shapes; with `bench`, the
    times at 32, 48, 64 and 128 rows against production."""
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        bad = [(M, pf) for M in rows for pf in (False, True)
               if not torch.equal(h64_matmul(x[:M], w, pf=pf), X.exl3_matmul(x[:M], w))]
        tile = X.pick(w.tile_n, w.K, w.kb)
        line = (f"{N}x{K} K{kb} wk{tile['wk']} S{tile['S']}: "
                + (f"rows {min(rows)}..{max(rows)} = the production bits" if not bad else f"DIFFER at {bad}"))
        if bench:
            res = []
            for M in (32, 48, 64, 128):
                xs = x[:M]
                for name, fn in (("prod", lambda: X.exl3_matmul(xs, w)), ("h64", lambda: h64_matmul(xs, w, pf=False)),
                                 ("h64pf", lambda: h64_matmul(xs, w, pf=True))):
                    res.append(f"M{M} {name} {_time_us(fn):.0f}us")
            line += "; " + ", ".join(res)
        out.append(line)
    return out


def _time_us(fn, n: int = 30) -> float:
    for _ in range(3):
        fn()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1e3


# ---- fragment-major activations (SPD-199): the rotated input stored in the mma's A-fragment order ------------------
# The profile of the row grid at 64 rows: the warps wait on the activation loads (long scoreboard), four 4-byte
# loads a lane a 16-row group and K tile. Stored as [G, row groups, K tiles, 32 lanes, 4 words] (lane l's a0..a3 of
# m16n8k16), a lane's operands are one coalesced 16-byte load. The values and every instruction that uses them are
# the same: only where they sit in memory moves.
_FRAG_LOAD_OLD = nb.source(__file__, "patches/frag_load.old.cuh")
_FRAG_LOAD_NEW = nb.source(__file__, "patches/frag_load.new.cuh")
_FRAG_OFF_OLD = nb.source(__file__, "patches/frag_off.old.cuh")
_FRAG_OFF_NEW = nb.source(__file__, "patches/frag_off.new.cuh")
_FRAG_MOD = None


def _frag_module():
    global _FRAG_MOD
    if _FRAG_MOD is None:
        assert X._CUDA.count(_FRAG_LOAD_OLD) == 1 and X._CUDA.count(_FRAG_OFF_OLD) == 1, "the shared kernel moved"
        src = (f"#define EXL3_ST {X.STAGES}\n"
               + X._CUDA.replace(_FRAG_LOAD_OLD, _FRAG_LOAD_NEW).replace(_FRAG_OFF_OLD, _FRAG_OFF_NEW))
        tag = nb.tag(X._CPP, src)
        _FRAG_MOD = nb.load(f"qwen38_exl3_frag_{tag}", cpp_sources=[X._CPP], cuda_sources=[src],
                            functions=["exl3_gemm"],
                            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                            verbose=False)
    return _FRAG_MOD


def frag_layout(xh: torch.Tensor) -> torch.Tensor:
    """[G, M, Kin] fp16 (rotate_in's) -> the same values in A-fragment order, rows padded to 16 with zeros."""
    G, M, K = xh.shape
    RG = (M + 15) // 16
    p = torch.zeros(G, RG * 16, K, dtype=xh.dtype, device=xh.device)
    p[:, :M] = xh
    v = p.view(G, RG, 2, 8, K // 16, 2, 4, 2)          # g, rg, r_hi, r_lo, kt, k_hi, t, pair
    return v.permute(0, 1, 4, 3, 6, 5, 2, 7).contiguous().view(G, RG * 16, K)


def frag_matmul(x: torch.Tensor, w, *, xf: torch.Tensor | None = None, out: torch.Tensor | None = None,
                cfg: int | None = None) -> torch.Tensor:
    """`exl3_matmul(x, w)` on the fragment-major activations (a register-path cfg; the same bits)."""
    M = x.shape[0]
    Mp = (M + 15) // 16 * 16                            # the padded rows are computed and dropped
    if out is None:
        out = torch.empty(Mp, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=x.device)
    tile = X.pick(w.tile_n, w.K, w.kb)
    c = tile["cfg2"] if cfg is None else cfg
    mt = 2 if c in (2, 3) else 1
    xf = frag_layout(X.rotate_in(x, w)) if xf is None else xf
    _frag_module().exl3_gemm(xf, w.t, w.svh, out, w.kb, c, tile["wk"], tile["S"], w.bounds[0], w.bounds[1],
                             (Mp + 16 * mt - 1) // (16 * mt))
    return out[:M]


def check_frag(rows=(1, 2, 7, 16, 17, 31, 32, 33, 40, 48, 63, 64, 65, 96, 127, 128), bench: bool = True) -> list[str]:
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        bad = [(M, c) for M in rows for c in (2, 3) if not torch.equal(frag_matmul(x[:M], w, cfg=c), X.exl3_matmul(x[:M], w))]
        line = f"{N}x{K} K{kb}: " + ("rows 1..128 = the production bits" if not bad else f"DIFFER at {bad[:6]}")
        if bench:
            res = []
            for M in (1, 16, 32, 64, 128):
                xs = x[:M]
                xh = X.rotate_in(xs, w)
                xf = frag_layout(xh)
                tile = X.pick(w.tile_n, w.K, w.kb)
                res.append(f"M{M} prod {_time_us(lambda: X.exl3_matmul(None, w, xh=xh, M=M, dev=x.device)):.0f}")
                for c in (2, 3):
                    res.append(f"frag{c} {_time_us(lambda: frag_matmul(xs, w, xf=xf, cfg=c)):.0f}")
            line += "; " + ", ".join(res)
        out.append(line)
    return out


_H64F_LOAD_OLD = nb.source(__file__, "patches/h64f_load.old.cuh")
_H64F_LOAD_NEW = nb.source(__file__, "patches/h64f_load.new.cuh")
_H64F_OFF_OLD = nb.source(__file__, "patches/h64f_off.old.cuh")
_H64F_OFF_NEW = nb.source(__file__, "patches/h64f_off.new.cuh")
_H64F_MOD = None


def _h64f_module():
    global _H64F_MOD
    if _H64F_MOD is None:
        assert _H64.count(_H64F_LOAD_OLD) == 1 and _H64.count(_H64F_OFF_OLD) == 1
        body = _H64.replace(_H64F_LOAD_OLD, _H64F_LOAD_NEW).replace(_H64F_OFF_OLD, _H64F_OFF_NEW)
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_H64_ST {H64_ST}\n" + X._CUDA + body
        tag = nb.tag(_H64_CPP, src)
        _H64F_MOD = nb.load(f"qwen38_exl3_h64f_{tag}", cpp_sources=[_H64_CPP], cuda_sources=[src],
                            functions=["exl3_gemm_h64"],
                            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                            verbose=False)
    return _H64F_MOD


def h64f_matmul(x: torch.Tensor, w, *, xf: torch.Tensor | None = None, pf: bool = True) -> torch.Tensor:
    """H64 on the fragment-major activations (the same bits)."""
    M = x.shape[0]
    Mp = (M + 15) // 16 * 16
    out = torch.empty(Mp, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=x.device)
    tile = X.pick(w.tile_n, w.K, w.kb)
    xf = frag_layout(X.rotate_in(x, w)) if xf is None else xf
    _h64f_module().exl3_gemm_h64(xf, w.t, w.svh, out, w.kb, tile["wk"], tile["S"], w.bounds[0], w.bounds[1], int(pf))
    return out[:M]


def check_h64f(rows=(1, 7, 16, 17, 32, 33, 48, 63, 64, 65, 128), bench: bool = True) -> list[str]:
    out = []
    for N, K, kb in SHAPES:
        if N * K > 300_000_000:
            N = 32768
        w = X.random_block(N, K, kb, seed=N + K + kb)
        g = torch.Generator(device="cpu").manual_seed(N)
        x = (torch.randn(max(rows), K, generator=g) * 0.7).to(torch.bfloat16).cuda()
        bad = [(M, pf) for M in rows for pf in (False, True)
               if not torch.equal(h64f_matmul(x[:M], w, pf=pf), X.exl3_matmul(x[:M], w))]
        line = f"{N}x{K} K{kb}: " + ("= the production bits" if not bad else f"DIFFER at {bad[:6]}")
        if bench:
            res = []
            for M in (32, 64, 128):
                xs = x[:M]
                xh = X.rotate_in(xs, w)
                xf = frag_layout(xh)
                res.append(f"M{M} prod {_time_us(lambda: X.exl3_matmul(None, w, xh=xh, M=M, dev=x.device)):.0f}")
                res.append(f"frag2 {_time_us(lambda: frag_matmul(xs, w, xf=xf, cfg=2)):.0f}")
                for pf in (False, True):
                    res.append(f"h64f{int(pf)} {_time_us(lambda: h64f_matmul(xs, w, xf=xf, pf=pf)):.0f}")
            line += "; " + ", ".join(res)
        out.append(line)
    return out


# ---- source variants of the shared kernel (built as their own extensions; its file keeps its bytes) ----
_MMA = nb.source(__file__, "patches/mma.old.cuh")
#: "skip": a 16-row group past the chunk's rows issues no mma (its rows are never stored: the row grid's
#: last chunk of 33..48 rows paid two groups for one); every row < M runs the same instructions
_SKIP = nb.source(__file__, "patches/mma.skip.cuh")
#: "nomma" (timing only, wrong numbers): the mma replaced by an xor into the accumulator that keeps the decode
#: and the loads live -- the discriminator between the tensor pipe and the decode's ALU work
_NOMMA = nb.source(__file__, "patches/mma.nomma.cuh")
_VARIANTS: dict = {}


def variant(name: str):
    """The shared kernels with the mma loop as `name` says ("skip", "nomma"), as their own extension."""
    mod = _VARIANTS.get(name)
    if mod is None:
        body = {"skip": _SKIP, "nomma": _NOMMA}[name]
        assert X._CUDA.count(_MMA) == 1, "the shared kernel's mma loop moved"
        src = f"#define EXL3_ST {X.STAGES}\n" + X._CUDA.replace(_MMA, body)
        tag = nb.tag(X._CPP, src)
        mod = _VARIANTS[name] = nb.load(
            f"qwen38_exl3_{name}_{tag}", cpp_sources=[X._CPP], cuda_sources=[src], functions=["exl3_gemm"],
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
    if "--p2" in sys.argv:
        for line in check_p2():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    if "--h64f" in sys.argv:
        for line in check_h64f():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    if "--frag" in sys.argv:
        for line in check_frag():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    if "--h64" in sys.argv:
        for line in check_h64():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    if "--t64" in sys.argv:
        for line in check_t64():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    if "--discriminate" in sys.argv:
        for shape in ((10240, 5120, 4), (17408, 5120, 4), (5120, 17408, 5)):
            for line in discriminate(shape):
                print("exl3_tall:", line, flush=True)
        for line in check_skip():
            print("exl3_tall:", line, flush=True)
        sys.exit(0)
    for line in check():
        print("exl3_tall:", line, flush=True)
