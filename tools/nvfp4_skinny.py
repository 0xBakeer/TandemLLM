"""The W4A16 product for 1..32 rows as a CUDA kernel that feeds the tensor cores from registers.

`tools/nvfp4_linear_v2.py` reads the verify's 13.7 GB of NVFP4 projections at about 169 GB/s in
the engine and 194 GB/s cold, against a practical ceiling of about 235 GB/s on this board. Triton
decodes each weight tile into a blocked register layout and then has to move it into the tensor
core's operand layout, which it does through shared memory, for every 128-wide K step of every
tile. This kernel never does that, because of one observation about the stored format:

    `cvt.rn.f16x2.e2m1x2` turns ONE packed byte into ONE f16x2 register holding (k, k+1), the low
    half being the even element -- which is exactly how an `mma.m16n8k16` operand register holds
    two consecutive K elements.

So a thread can load 16 contiguous bytes of a weight row with one vector load and decode each byte
straight into an operand register. The only thing that has to agree is which logical K each of the
instruction's sixteen K slots stands for, and a dot product does not care about the order of its
terms, so the slots are assigned to match the bytes a thread already holds:

    lane (g, t) of a warp holds bytes 16t .. 16t+15 of a 64-byte row chunk (128 logical K).
    In the chunk's j-th k16 instruction (j = 0..7) its B registers are bytes 16t+2j and 16t+2j+1,
    i.e. logical K 32t+4j .. 32t+4j+3; its A registers are the ACTIVATION at the same four K,
    which are four consecutive bf16 of the activation row -- also a contiguous load.

The weights are used as the B operand (8 weight rows per instruction) and the activations as A
(16 rows per instruction), so a verify block of up to 16 rows is one instruction per weight tile
and 17..32 rows are two. The stored layout does not change: the prefill paths, the v1/v2 kernels
and the dequantiser read the same bytes they always have, and nothing is duplicated in memory.

Scales. Group j of a row covers logical K 16j..16j+15, eight bytes. Lane t's sixteen bytes are
groups 2t and 2t+1 of the chunk, instructions 0..3 use the first and 4..7 the second, and both
scales arrive in one 16-bit load, decoded by `cvt.rn.f16x2.e4m3x2`. The scale multiplies the
decoded weight in fp16 before the product, exactly as v2 does (exact: e2m1 x e4m3 needs six
significand bits and fp16 has eleven), and the per-tensor scale multiplies the fp32 accumulator.

Determinism. The reduction order over K is a function of the weight's shape only -- the tile table
below is keyed by (N, K) -- never of the row count, so a row's result is the same bits whether it
is verified alone or in a block of sixteen, which is what the losslessness gate compares. K is split
across the warps of a CTA and summed in warp order through shared memory; there are no atomics.

Built on first use from tools/csrc/nvfp4_skinny/ (skinny.cu, the bindings skinny.cpp) with
`torch.utils.cpp_extension.load_inline` for sm_121a (the e2m1 converter is an arch-specific instruction),
cached under ~/.cache/torch_extensions by source hash (engine/native_build.py).
"""

from __future__ import annotations

import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import native_build as nb  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QWEN38_NVFP4_SKINNY=1 routes every NVFP4 projection of 1..32 rows here instead of to v2.
SKINNY = _S.get("NVFP4_SKINNY") == "1"
SKINNY_MAX = 32
# launch with programmatic dependent launch -- the grid starts while the kernel before it
# (an RMS norm that releases its dependents at once, tools/norm_kernels.py) still runs, loads its
# first weights, and waits for the activation. The same loads in the same order: the same bits.
PDL = _S.get("SKINNY_PDL") == "1"
# the weight loads' L2 hint, compiled in (one build a value; see `ld_w`). 0 = as shipped.
LDW = int(_S.get("SKINNY_LDW"))
# /, TIMING ONLY -- the output is wrong with either on. XSTUB: the activation loads
# return a constant, so the compiler drops the activation's registers and loads (the ceiling a
# kernel that kept the activation out of the lane's registers could reach). SSTUB: the scale loads
# return a constant (the ceiling of a perfect scale stream). Never set in a served environment.
XSTUB = int(_S.get("SKINNY_XSTUB"))
SSTUB = int(_S.get("SKINNY_SSTUB"))
if XSTUB or SSTUB:
    print(f"[skinny] TIMING-ONLY BUILD: XSTUB={XSTUB} SSTUB={SSTUB} -- the output is wrong",
          flush=True)
# read the scales from a copy laid out in runs of 16 rows x one 128-wide K step (8 bytes a
# row, 128 contiguous bytes; `scale_runs`) instead of 8 bytes in each of 8 rows a scale-row apart.
# The same bytes in the same registers: the same bits. Compiled in (one build a value, like LDW);
# the copy is made on a weight's first skinny call and costs 1/9 of its bytes.
SRUN = int(_S.get("SKINNY_SRUN"))

_CUDA = nb.source(__file__, "skinny.cu")

_CPP = nb.source(__file__, "skinny.cpp")

_MOD = None
# one built module a weight-load hint, so an in-process block A/B can flip `LDW` (the
# verify graphs key on it, engine/verify_graph.py signature) without a second process
_MODS: dict = {}
_MOD_LDW = None


def _module(ldw: int | None = None):
    """The built kernel module for a weight-load hint: `LDW` by default, or a tile entry's own `ldw`
    (a hint for the 17..32-row tile alone, 2026-09-26)."""
    global _MOD, _MOD_LDW
    if ldw is not None and ldw != LDW:
        if ldw not in _MODS:
            _MODS[ldw] = _build(ldw)
        return _MODS[ldw]
    if _MOD is not None and _MOD_LDW == LDW:
        return _MOD
    if _MOD is not None and LDW in _MODS:
        _MOD, _MOD_LDW = _MODS[LDW], LDW
        return _MOD
    _MOD = _build(LDW)
    _MODS[LDW], _MOD_LDW = _MOD, LDW
    return _MOD


def _build(ldw: int):
    # the build directory is named by the source's hash, so two checkouts with different
    # kernels on one box (the service and a branch under test) never rebuild over each other
    src = f"#define LDW_HINT {ldw}\n" + _CUDA
    if XSTUB or SSTUB:
        src = f"#define XSTUB {XSTUB}\n#define SSTUB {SSTUB}\n" + src
    if SRUN:
        src = f"#define SRUN {SRUN}\n" + src
    tag = nb.tag(_CPP, src)
    return nb.load(f"qwen38_nvfp4_skinny_{tag}", cpp_sources=[_CPP],
                   cuda_sources=[src],
                   functions=["skinny", "skinny_ser"],
                   # the e2m1 converter is an arch-specific instruction: sm_121a and
                   # nothing else (an explicit arch flag also stops torch adding its own)
                   extra_cuda_cflags=["-O3", "-lineinfo",
                                      "-gencode=arch=compute_121a,code=sm_121a"],
                   verbose=False)


# (N, K) -> tile. Keyed by shape only, never by row count: the K split is the reduction order.
# Filled from the cold sweep and the in-engine A/B (docs/kernels.md).
_CONFIG: dict[tuple[int, int], dict] = {}
# A shape the table does not name -- the drafter's 4096- and 25600-wide projections -- takes the
# first sweep's winner, which is what the kb-* rows ran (ops/skinny-tiles.json names the target's).
_FALLBACK = {"nt": 4, "wk": 8, "pf": 1, "minb": 1}
# A measured table from a file, for the in-engine A/B before a table is written into this one:
# {"17408x5120": {"nt": 8, "wk": 4, "pf": 1}, ...}
if _S.get("SKINNY_TILES") and not os.path.isfile(_S.get("SKINNY_TILES")):
    # a missing table would silently serve every shape on the fallback tile
    print(f"[skinny] WARNING: QWEN38_SKINNY_TILES={_S.get('SKINNY_TILES')} does not "
          f"exist; every projection takes the fallback tile {_FALLBACK}", flush=True)
if _S.get("SKINNY_TILES") and os.path.isfile(_S.get("SKINNY_TILES")):
    import json as _json
    for _k, _v in _json.load(open(_S.get("SKINNY_TILES"))).items():
        _n, _kk = (int(v) for v in _k.split("x"))
        _CONFIG[(_n, _kk)] = {"nt": int(_v["nt"]), "wk": int(_v["wk"]), "pf": int(_v["pf"]),
                              "minb": int(_v.get("minb", 1)), "il": int(_v.get("il", 0)),
                              **({"kr": 1} if int(_v.get("kr", 0)) else {}),
                              **({"spw": int(_v["spw"])} if int(_v.get("spw", 1)) > 1 else {})}


# Two more tables for an in-process A/B: QWEN38_SKINNY_TILES_B / _C name them, and `ALT` /
# `ALT2` -- switches `tools/block_budget.py --ab tools.nvfp4_skinny:ALT` flips -- route the shapes
# each names through it (ALT2 first when both are on). Off (the default) is the first table, exactly.
def _table(env: str) -> dict:
    out: dict[tuple[int, int], dict] = {}
    path = _S.get(env[len("QWEN38_"):]) if env.startswith("QWEN38_") else os.environ.get(env)
    if path and os.path.isfile(path):
        import json as _json
        for k, v in _json.load(open(path)).items():
            n, kk = (int(x) for x in k.split("x"))
            out[(n, kk)] = {"nt": int(v["nt"]), "wk": int(v["wk"]), "pf": int(v["pf"]),
                            "minb": int(v.get("minb", 1)), "il": int(v.get("il", 0)),
                            **({"kr": 1} if int(v.get("kr", 0)) else {}),
                            **({"spw": int(v["spw"])} if int(v.get("spw", 1)) > 1 else {}),
                            **({"ldw": int(v["ldw"])} if "ldw" in v else {}),
                            **({"ser": 1, "wn": int(v["wn"]), "nb": int(v["nb"]),
                                "g": int(v.get("g", 1))} if int(v.get("ser", 0)) else {})}
    return out


_ALT, _ALT2 = _table("QWEN38_SKINNY_TILES_B"), _table("QWEN38_SKINNY_TILES_C")
# the tile for 17..32 rows. At 32 rows the 16-row winner (nt2:wk16:pf2) falls to
# 164-184 GB/s on the wide shapes, a verify of 32 rows paying ~17 ms more in this kernel than one of
# 16. A second table, read only past sixteen rows, may change the N tile and the prefetch but NEVER
# the K split: the K split is a row's summation order, so a row keeps its bits whatever the block's
# width (the losslessness gate's row independence). An entry with another `wk` is refused at load.
if _S.get("SKINNY_TILES_WIDE") and not os.path.isfile(_S.get("SKINNY_TILES_WIDE")):
    print(f"[skinny] WARNING: QWEN38_SKINNY_TILES_WIDE={_S.get('SKINNY_TILES_WIDE')} does "
          f"not exist; 17..32-row verifies take the base tile", flush=True)
_WIDE = _table("QWEN38_SKINNY_TILES_WIDE")
for (_n, _kk), _v in list(_WIDE.items()):
    _base = _CONFIG.get((_n, _kk), _FALLBACK)
    if _v["wk"] != _base["wk"] or _v.get("il", 0) != _base.get("il", 0):
        print(f"[skinny] WARNING: QWEN38_SKINNY_TILES_WIDE {_n}x{_kk} splits K as wk{_v['wk']}, "
              f"the base tile as wk{_base['wk']}: refused (a row's bits would depend on the block's "
              f"width)", flush=True)
        del _WIDE[(_n, _kk)]
# a second 17..32-row table for the in-process block A/B. QWEN38_SKINNY_TILES_WIDE_B
# names it and `WIDE_B` (off by default) routes the shapes it names through it past sixteen rows. The
# same rule as the wide table: an entry that splits K otherwise than the base tile is refused.
_WIDE_B = _table("QWEN38_SKINNY_TILES_WIDE_B")
for (_n, _kk), _v in list(_WIDE_B.items()):
    _base = _CONFIG.get((_n, _kk), _FALLBACK)
    if _v["wk"] != _base["wk"] or _v.get("il", 0) != _base.get("il", 0):
        print(f"[skinny] WARNING: QWEN38_SKINNY_TILES_WIDE_B {_n}x{_kk} splits K as wk{_v['wk']}, "
              f"the base tile as wk{_base['wk']}: refused", flush=True)
        del _WIDE_B[(_n, _kk)]
ALT = False
ALT2 = False
WIDE_B = False
# The weight-load hint for 17..32-row calls only (None: the entry's own `ldw`, else LDW). In-process A/B.
WIDE_LDW = None


def _alt(N: int, K: int) -> dict | None:
    if ALT2 and (N, K) in _ALT2:
        return _ALT2[(N, K)]
    if ALT and (N, K) in _ALT:
        return _ALT[(N, K)]
    return None


from tools.tile_warn import missing as _tile_missing  # noqa: E402


def pick(N: int, K: int, M: int = 1) -> dict:
    if M > 16 and WIDE_B and (N, K) in _WIDE_B:
        return _WIDE_B[(N, K)]
    if M > 16 and (N, K) in _WIDE:
        return _WIDE[(N, K)]
    return _alt(N, K) or _CONFIG.get((N, K), _FALLBACK)


def set_config(N: int, K: int, cfg: dict) -> None:
    _CONFIG[(N, K)] = dict(cfg)


_EMPTY: dict = {}


def nvfp4_matmul_skinny(x: torch.Tensor, w, *, out: torch.Tensor | None = None,
                        nt: int | None = None, wk: int | None = None,
                        pf: int | None = None, minb: int | None = None,
                        il: int | None = None, kr: int | None = None,
                        ser: int | None = None, wn: int | None = None,
                        nb: int | None = None, g: int | None = None,
                        spw: int | None = None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W[N, K]^T for M <= 32; W an NVFP4Block or an NVFP4Group."""
    M = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and 1 <= M <= SKINNY_MAX, x.shape
    x = x.contiguous()
    if x.data_ptr() % 16:
        # the activation is read with 16-byte loads; a view at an odd offset is copied once
        x = x.clone()
    cfg = pick(w.N, w.K, M)
    if (w.N, w.K) not in _CONFIG and _alt(w.N, w.K) is None:
        if getattr(w, "sizes", None):
            # a fused group inherits its first member's tile: same K split, so a grouped launch
            # sums each row in the order the member's own launch would
            cfg = pick(w.sizes[0], w.K, M)
            if (w.sizes[0], w.K) not in _CONFIG and _alt(w.sizes[0], w.K) is None:
                _tile_missing("nvfp4_skinny", w.sizes[0], w.K, _FALLBACK)
        else:
            _tile_missing("nvfp4_skinny", w.N, w.K, _FALLBACK)
    if nt is not None or wk is not None or pf is not None:
        # an explicit tile (a test, a bench) is the whole tile: the table entry's order and layout
        # extras -- kr, spw, ser, its own hint -- belong to the entry and are not carried over to it
        # (phase6: with kr1 as the served wide table, an explicit nt4:pf0 would have run kr1)
        cfg = {k: v for k, v in cfg.items() if k in ("nt", "wk", "pf", "minb", "il")}
    if out is None:
        out = torch.empty(M, w.N, dtype=torch.bfloat16, device=x.device)
    s2v = getattr(w, "s2v", None)
    if s2v is None:
        s2v = _EMPTY.get(x.device)
        if s2v is None:
            s2v = _EMPTY[x.device] = torch.empty(0, dtype=torch.float32, device=x.device)
    sv = w.s.view(torch.uint8)
    if SRUN:
        sv = getattr(w, "_srun", None)
        if sv is None:
            sv = w._srun = scale_runs(w.s).view(-1, 8)
    mod = _module(WIDE_LDW if M > 16 and WIDE_LDW is not None else cfg.get("ldw"))
    if (cfg.get("ser", 0) if ser is None else ser):
        # the slice-serial tile, the same K split (`wk`) summed by one warp in slice order
        mod.skinny_ser(x, w.w, sv, s2v, float(w.s2), out,
                             cfg["nt"] if nt is None else nt, cfg["wk"] if wk is None else wk,
                             cfg["wn"] if wn is None else wn, cfg["nb"] if nb is None else nb,
                             cfg.get("g", 1) if g is None else g)
        return out
    mod.skinny(x, w.w, sv, s2v, float(w.s2), out,
                     cfg["nt"] if nt is None else nt, cfg["wk"] if wk is None else wk,
                     cfg["pf"] if pf is None else pf,
                     cfg.get("minb", 1) if minb is None else minb,
                     cfg.get("il", 0) if il is None else il, int(PDL),
                     cfg.get("kr", 0) if kr is None else kr,
                     cfg.get("spw", 1) if spw is None else spw)
    return out


def scale_runs(s: torch.Tensor) -> torch.Tensor:
    """The e4m3 scales [N, K/16] as runs of 16 rows x one 128-wide K step.

    Run (G, q) holds bytes 8q .. 8q + 7 of rows 16G .. 16G + 15, in row order: 128 contiguous bytes,
    which a warp's lanes read for its 8 NT rows in one or two lines where they read 8 bytes in each
    of 8 rows before. A permutation of the same bytes; rows past N up to a multiple of 16 are zero
    and never read (the kernel clamps a row index to N - 1)."""
    N, C = s.shape
    u = s.contiguous().view(torch.uint8)
    n16 = (N + 15) // 16 * 16
    if n16 != N:
        u = torch.cat([u, u.new_zeros(n16 - N, C)])
    return u.view(n16 // 16, 16, C // 8, 8).permute(0, 2, 1, 3).contiguous().view(-1)


def use_skinny(M: int) -> bool:
    return SKINNY and 1 <= M <= SKINNY_MAX


# ------------------------------------------------------------------ gate and bench
def _exact(w) -> torch.Tensor:
    """The weight in fp32 with no rounding at all: e2m1 x e4m3 x s2, the value every kernel means."""
    from tools.nvfp4_linear import FP4_GRID, GROUP
    grid = FP4_GRID.to(w.w.device)
    lo, hi = (w.w & 0x0F).long(), (w.w >> 4).long()
    vals = torch.empty(w.N, w.K, dtype=torch.float32, device=w.w.device)
    vals[:, 0::2] = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
    vals[:, 1::2] = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
    return vals * w.s.float().repeat_interleave(GROUP, 1) * w.s2


def check(shapes=((17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
                  (12288, 5120), (1024, 5120)),
          rows=(1, 2, 7, 8, 9, 14, 16, 17, 24, 32), seed: int = 0) -> list[str]:
    """Against the exact product and against v2, on random NVFP4 weights.

    The bar is v2's own: `max|skinny - v2|` at or below `max|v2 - exact|` (two fp32 orders of the
    same fp16 products), and row independence -- a row computed alone equals the same row inside a
    block, bit for bit -- at every row count, for every tile in the sweep.
    """
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
    g = torch.Generator(device="cuda").manual_seed(seed)
    out, worst = [], 0.0
    for (N, K) in shapes:
        ref = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float32) * 0.02
        w = quantize_to_nvfp4(ref)
        wd = _exact(w)
        x = torch.randn(max(rows), K, device="cuda", generator=g).to(torch.bfloat16)
        for cfg in ({"nt": 4, "wk": 1, "pf": 0}, {"nt": 8, "wk": 4, "pf": 1},
                    {"nt": 16, "wk": 2, "pf": 0}, {"nt": 4, "wk": 8, "pf": 1},
                    {"nt": 2, "wk": 16, "pf": 2}, {"nt": 4, "wk": 8, "pf": 2, "minb": 2}):
            single = None
            for M in rows:
                xm = x[:M]
                y = nvfp4_matmul_skinny(xm, w, **cfg).float()
                y2 = nvfp4_matmul_v2(xm, w).float()
                exact = xm.float().to(torch.float16).float() @ wd.T
                d_s = (y - exact).abs().max().item()
                d_v2 = (y2 - exact).abs().max().item()
                d_sv = (y - y2).abs().max().item()
                ratio = d_sv / max(d_v2, 1e-12)
                worst = max(worst, ratio)
                if single is None:
                    single = y[:1].clone()
                indep = (y[:1] - single).abs().max().item()
                assert indep == 0.0, f"row independence {N}x{K} {cfg} M={M}: {indep}"
                assert d_s <= 2 * d_v2 + 1e-6, f"{N}x{K} {cfg} M={M}: {d_s} vs v2 {d_v2}"
            out.append(f"{N}x{K} {cfg}: last |skinny-exact| {d_s:.3e}  |v2-exact| {d_v2:.3e}  "
                       f"|skinny-v2| {d_sv:.3e}")
        del w, wd, ref
    out.append(f"worst |skinny-v2| / |v2-exact| = {worst:.3f}; row independence exact everywhere")
    return out


def bench_cold(N: int, K: int, gb: float = 2.5, rows=(1, 8, 16, 32), tiles=None,
               reps: int = 5) -> list[str]:
    """A chain of distinct weights bigger than the board's caches, so every byte is a DRAM read.

    Same method as tools/cold_bw.py: the chain is `n` independent copies, each projection timed
    over the whole chain. A ranking, never a budget -- the engine's own verify decides.
    """
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2
    one = quantize_to_nvfp4(torch.randn(N, K, device="cuda") * 0.02)
    per = one.nbytes
    n = max(4, int(gb * 1e9 / per))
    ws = []
    for _ in range(n):
        b = type(one).__new__(type(one))
        b.w, b.s, b.s2, b.N, b.K, b._bf16 = one.w.clone(), one.s.clone(), one.s2, one.N, one.K, None
        if SRUN:
            b._srun = scale_runs(b.s).view(-1, 8)       # built before the clock, as the engine does
        ws.append(b)
    del one
    tiles = tiles or [{"nt": 8, "wk": 4, "pf": 1}]
    lines = [f"### {N}x{K}  {per / 1e6:.1f} MB x {n} = {n * per / 1e9:.2f} GB"]
    for M in rows:
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        cands = [("v2", lambda b: nvfp4_matmul_v2(x, b))]
        for c in tiles:
            cands.append((tile_name(c), lambda b, c=c: nvfp4_matmul_skinny(x, b, **c)))
        for name, fn in cands:
            try:
                for b in ws[:2]:
                    fn(b)
                torch.cuda.synchronize()
            except RuntimeError as e:          # a tile the kernel refuses (shared memory, no instance)
                lines.append(f"  M={M:3d} {name:>18} refused: {str(e).splitlines()[0][:80]}")
                continue
            best = float("inf")
            for _ in range(reps):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for b in ws:
                    fn(b)
                t1.record()
                t1.synchronize()
                best = min(best, t0.elapsed_time(t1))
            lines.append(f"  M={M:3d} {name:>18} {best:8.2f} ms  {n * per / best / 1e6:7.1f} GB/s")
            RESULTS.setdefault(f"{N}x{K}", {}).setdefault(name, {})[M] = n * per / best / 1e6
    return lines


RESULTS: dict = {}


def tile_name(c: dict) -> str:
    """`nt2:wk16:pf2:mb1`, with `:il1` for the interleaved K split."""
    if c.get("ser", 0):
        return (f"nt{c['nt']}:wk{c['wk']}:sr1:wn{c['wn']}:nb{c['nb']}"
                + (f":gs{c['g']}" if c.get("g", 1) != 1 else ""))
    return (f"nt{c['nt']}:wk{c['wk']}:pf{c['pf']}:mb{c.get('minb', 1)}"
            + (":il1" if c.get("il", 0) else "") + (":kr1" if c.get("kr", 0) else "")
            + (f":sw{c['spw']}" if c.get("spw", 1) > 1 else ""))


def parse_tile(name: str) -> dict:
    keys = {"nt": "nt", "wk": "wk", "pf": "pf", "mb": "minb", "il": "il", "kr": "kr", "sr": "ser",
            "wn": "wn", "nb": "nb", "gs": "g", "sw": "spw"}
    out = {"minb": 1, "il": 0}
    if ":sr1" in name:
        out["pf"] = 2
    for part in name.split(":"):
        out[keys[part[:2]]] = int(part[2:])
    return out


def best_tiles(rows=(8, 16)) -> dict:
    """Per shape, the tile with the best mean GB/s over the verify's row counts."""
    out = {}
    for shape, by_tile in RESULTS.items():
        scored = [(sum(v.get(m, 0.0) for m in rows) / len(rows), name)
                  for name, v in by_tile.items() if name != "v2"]
        if not scored:
            continue
        gbps, name = max(scored)
        out[shape] = dict(parse_tile(name), gbps=round(gbps, 1),
                          v2_gbps=round(sum(by_tile["v2"].get(m, 0.0) for m in rows) / len(rows), 1))
    return out


def read_ceiling(gb: float = 2.5, reps: int = 5) -> str:
    """What a plain streaming read reaches here: a float sum over a buffer bigger than the caches."""
    buf = torch.empty(int(gb * 1e9 / 4), dtype=torch.float32, device="cuda").uniform_()
    buf.sum()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        buf.sum()
        t1.record()
        t1.synchronize()
        best = min(best, t0.elapsed_time(t1))
    n = buf.numel() * 4
    del buf
    return f"read ceiling: torch sum over {n / 1e9:.2f} GB in {best:.2f} ms = {n / best / 1e6:.1f} GB/s"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", default="", help="N:K[,N:K...]")
    ap.add_argument("--rows", default="1,8,16")
    ap.add_argument("--tiles", default="nt4:wk8:pf1:mb1",
                    help="nt:wk:pf:mb[:il1], comma separated (mb = CTAs an SM the registers must "
                         "allow; il1 = the interleaved K split)")
    ap.add_argument("--gb", type=float, default=2.5)
    ap.add_argument("--write-tiles", default="", help="write the best tile per shape here")
    ap.add_argument("--best-rows", default="8,16",
                    help="the row counts --write-tiles averages over (24,32 for the wide table)")
    a = ap.parse_args()
    if a.bench:
        print(read_ceiling(a.gb), flush=True)
    if a.check:
        for line in check():
            print(line, flush=True)
    tiles = [parse_tile(t) for t in a.tiles.split(",")]
    for shp in filter(None, a.bench.split(",")):
        N, K = (int(v) for v in shp.split(":"))
        for line in bench_cold(N, K, a.gb, [int(r) for r in a.rows.split(",")], tiles):
            print(line, flush=True)
    if a.write_tiles:
        import json
        best = best_tiles(tuple(int(r) for r in a.best_rows.split(",")))
        json.dump(best, open(a.write_tiles, "w"), indent=1)
        for shape, c in best.items():
            print(f"best {shape}: nt{c['nt']}:wk{c['wk']}:pf{c['pf']} {c['gbps']} GB/s "
                  f"(v2 {c['v2_gbps']})", flush=True)
