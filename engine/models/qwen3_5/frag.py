"""SPD-201: the 27B's EXL3 projections at 33..64 rows read fragment-major activations (QWEN38_EXL3_FRAG).

The row grid's register path loads a lane's A operands as four 4-byte loads a 16-row group and K tile, and at 64
rows its warps wait on them. Here the input rotation writes its fp16 values in the mma's A-fragment order (rows in
groups of 16, zero rows to a multiple of 16, K tiles, 32 lanes, a0..a3 of m16n8k16) and the gemm takes a lane's
operands in one 16-byte load. The values and every instruction on them are the same: `check` holds rows 33..64 to
the shared kernels' bits on the 27B's shapes, plain, grouped and silu.

The source is tools/exl3_linear.py's (tools/csrc/exl3_linear/exl3.cu), with the gemm's activation offset and load
and its row count from the output replaced (csrc/frag/patches/: each .old.cuh text checked to be there once, then
replaced by its .new.cuh) and the two fragment-order rotations of csrc/frag/ added, compiled as a module of its
own. Only this family's weights take it: `install(eng)` tags the EXL3 blocks of the engine's
weights and wraps the shared `exl3_matmul` and `exl3_down_silu` once; every other call, every other family's
block, runs the shared functions unchanged. tools/exl3_linear.py and its sources keep their bytes.

Per call at 48..64 rows, rotation included: -3 to -10 % on every 27B shape; at 32 rows and below flat (not taken).
"""

from __future__ import annotations

import torch

from engine import native_build as nb
from engine.settings import SETTINGS as _S

ON = _S.get("EXL3_FRAG") == "1"
#: the most rows a call takes the fragment-major path at (the joint pass's row cap)
MAX_ROWS = 64
STATS = {"frag": 0, "other": 0}
#: the ids of this family's tagged blocks (an EXL3Block takes no new attribute)
TAGGED: set = set()
_ORIG = None
_MOD = None

_OFF_OLD = nb.source(__file__, "patches/off.old.cuh")
_OFF_NEW = nb.source(__file__, "patches/off.new.cuh")
_LOAD_OLD = nb.source(__file__, "patches/load.old.cuh")
_LOAD_NEW = nb.source(__file__, "patches/load.new.cuh")
_ROWS_OLD = nb.source(__file__, "patches/rows.old.cuh")
_ROWS_NEW = nb.source(__file__, "patches/rows.new.cuh")
_HELPER = nb.source(__file__, "frag_off.cuh")
_KERNELS = nb.source(__file__, "rot_frag.cu")
_LAUNCHERS = nb.source(__file__, "rot_frag_launch.cu")
_CPP_ADD = nb.source(__file__, "rot_frag.cpp")


def _source() -> str:
    from tools import exl3_linear as X
    src = X._CUDA
    for old, new in ((_OFF_OLD, _OFF_NEW), (_LOAD_OLD, _LOAD_NEW), (_ROWS_OLD, _ROWS_NEW)):
        assert src.count(old) == 1, "tools/csrc/exl3_linear/exl3.cu moved: " + old.strip()[:60]
        src = src.replace(old, new)
    anchor = "constexpr float RSQRT128 = 0.08838834764831845f;\n"
    assert src.count(anchor) == 1, "tools/csrc/exl3_linear/exl3.cu moved: RSQRT128"
    src = src.replace(anchor, anchor + "\n" + _HELPER + "\n" + _KERNELS)
    return src + "\n" + _LAUNCHERS


def module():
    global _MOD
    if _MOD is None:
        from tools import exl3_linear as X
        src = f"#define EXL3_ST {X.STAGES}\n#define EXL3_PDL {int(X.PDL)}\n" + _source()    # SPD-245
        cpp = X._CPP + _CPP_ADD
        tag = nb.tag(cpp, src)
        _MOD = nb.load(f"qwen35_exl3_frag_{tag}", cpp_sources=[cpp], cuda_sources=[src],
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
