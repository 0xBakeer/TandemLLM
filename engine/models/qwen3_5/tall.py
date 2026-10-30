"""SPD-199: the 27B's EXL3 projections at 33..CHUNK_UNTIL rows through the decode-once tile (T64 / T128).

QWEN38_EXL3_TALL: "" (off, the row grid as before), "64" or "128": the rows a tile decodes each trellis
tile once for (tools/exl3_tall.py `t64_matmul`, every row the production kernels' bits: `check_t64`).

Only this family's weights take it: `install(eng)` tags the EXL3 blocks of the engine's weights, and the
shared `tools.exl3_linear.exl3_matmul` is wrapped once so that a tagged block at 33..CHUNK_UNTIL rows (an
integer rate the tile builds, no forced tiling) goes to the tile; every other call, every other family's
block, runs the shared function unchanged. tools/exl3_linear.py keeps its bytes.
"""

from __future__ import annotations

from engine.settings import SETTINGS as _S

MODE = (_S.get("EXL3_TALL") or "").strip()
MT = {"": 0, "64": 4, "128": 8}[MODE]
#: the K bits the tile is built for
KBITS = (3, 4, 5, 6)
_ORIG = None
STATS = {"tall": 0, "other": 0}


def _tag(w) -> int:
    from tools.exl3_linear import EXL3Block
    n = 0
    for d in (getattr(w, "q", {}), getattr(w, "g", {}), getattr(w, "t", {})):
        for blk in d.values():
            if isinstance(blk, EXL3Block) and blk.kb in KBITS and not getattr(blk, "_tall", 0):
                blk._tall = MT
                n += 1
    return n


def install(eng) -> int:
    """Tag `eng`'s EXL3 blocks for the tile and wrap the shared matmul (once). Returns the blocks tagged."""
    if not MT:
        return 0
    from tools import exl3_linear as X
    global _ORIG
    if _ORIG is None:
        _ORIG = X.exl3_matmul
        X.exl3_matmul = _dispatch
    return _tag(getattr(eng, "w", None))


def _dispatch(x, w, *, out=None, cfg=None, S=None, wk=None, xh=None, M=None, dev=None):
    mt = getattr(w, "_tall", 0)
    if mt and cfg is None and S is None and wk is None:
        from tools import exl3_linear as X
        rows = M if M is not None else (x.shape[0] if x is not None else xh.shape[1])
        if X.SKINNY_MAX < rows <= X.CHUNK_UNTIL and (x is None or x.is_cuda):
            from tools import exl3_tall as TT
            if x is not None and (x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8):
                x = x.contiguous()
                if x.data_ptr() % 8:
                    x = x.clone()
            STATS["tall"] += 1
            return TT.t64_matmul(x, w, mt=mt, xh=xh, out=out)
    STATS["other"] += 1
    return _ORIG(x, w, out=out, cfg=cfg, S=S, wk=wk, xh=xh, M=M, dev=dev)
