"""bf16 weights x bf16 activations for 1..64 rows on the tensor cores, fp32 out: NVIDIA FP4-Local's mixers
(down|inject) and router. Row-invariant (the K split depends on the weight's shape only). Sources:
tools/csrc/bf16_skinny/.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import native_build as nb  # noqa: E402

import torch  # noqa: E402

_CUDA = nb.source(__file__, "bf16.cu")
_CPP = nb.source(__file__, "bf16.cpp")
_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = nb.load(f"tandem_bf16_skinny_{nb.tag(_CPP, _CUDA)}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["bf16_skinny", "bf16_cluster", "mixup_bf16", "bf16_tiled", "bf16_tiledc"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"])
    return _MOD


def default_split(N: int, K: int) -> tuple[int, int]:
    """(wk, nt) for a weight [N, K]: a function of the shape only (row invariance)."""
    steps = K // 64
    wk = 16 if steps >= 32 else 8 if steps >= 16 else 2 if steps >= 4 else 1
    return wk, 1


#: the next K step's weights loaded during this one's products (same bits)
PF = 1

#: rows (and weight rows) from which `mm` takes the column-split kernel (`bf16_tiled`: the activations staged once a
#: CTA; the same bits), and its (warps a CTA, 8-row tiles a warp, K steps in flight). The 1.27 GB head, us skinny ->
#: tiled: 6,705 -> 5,700 at 33 rows, 7,814 -> 5,779 at 56, 8,334 -> 5,854 at 64 (1 row: 5,009 -> 5,446, kept skinny)
TILED_FROM, TILED_N, TILED = 33, 2560, (2, 2, 4)
#: `mm_cluster`'s column-split kernel above TILED_FROM rows: (warps a CTA, 8-row tiles a warp, K steps in flight)
TILED_C = (4, 1, 8)


def mm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, wk: int | None = None,
       nt: int | None = None, pf: int | None = None) -> torch.Tensor:
    """x [M, K] bf16 (M <= 64) @ w [N, K] bf16 -> fp32 [M, N]."""
    M, K = x.shape
    N = w.shape[0]
    y = out if out is not None else torch.empty(M, N, dtype=torch.float32, device=x.device)
    dwk, dnt = default_split(N, K)
    if M >= TILED_FROM and N >= TILED_N and pf is None and (dwk if wk is None else wk) in (16, 32):
        # many rows of a large weight (the head): the column-split kernel, this one's bits (the same K slices, in order)
        _module().bf16_tiled(x.contiguous(), w, y, dwk if wk is None else wk, *TILED)
        return y
    _module().bf16_skinny(x.contiguous(), w, y, dwk if wk is None else wk, dnt if nt is None else nt,
                          PF if pf is None else pf)
    return y


def mm_cluster(x: torch.Tensor, w: torch.Tensor, wk: int = 8, cs: int = 4, out: torch.Tensor | None = None) -> torch.Tensor:
    """`mm` with K split across a cluster of `cs` CTAs (a weight with few rows), reduced through distributed shared
    memory in rank order: fp32 [M, N]. Row-invariant (the order depends on the shape, wk and cs only)."""
    M = x.shape[0]
    y = out if out is not None else torch.empty(M, w.shape[0], dtype=torch.float32, device=x.device)
    if M >= TILED_FROM and wk == 16 and cs == 4:
        # many rows: each CTA 32 weight rows of its chunk, the chunk's activations staged once (`bf16_tiledc`; this
        # kernel's bits). The mixers' down|inject [324, 10,240], us: 55 -> 47 at 33 rows, 62 -> 50 at 56
        _module().bf16_tiledc(x.contiguous(), w, y, wk, cs, *TILED_C)
        return y
    _module().bf16_cluster(x.contiguous(), w, y, wk, cs)
    return y


def mix_up(d: torch.Tensor, n: torch.Tensor, w: torch.Tensor, hc: int, out: torch.Tensor | None = None,
           jg: int | None = None) -> torch.Tensor:
    """The mixer's second half for a bf16 up weight w [hc H, 320]: d [M, >=320] fp32 (down|inject), n [M, hc H] bf16 (the
    normed streams) -> x [M, H] bf16, on the tensor cores at every row count (row-invariant), 64 rows a launch. jg = 8-column
    groups a CTA (its staged activations shared by them): free to follow M, the bits do not change."""
    H = w.shape[0] // hc
    if out is None:
        out = torch.empty(d.shape[0], H, dtype=torch.bfloat16, device=d.device)
    for r0 in range(0, d.shape[0], 64):
        m = min(64, d.shape[0] - r0)
        # from MIX_PRE_FROM rows the activation silu(D / hc) is computed once, not by every CTA (the same bits)
        a = torch.empty(m, 320, dtype=torch.bfloat16, device=d.device) if m >= MIX_PRE_FROM else _NONE.get(d.device)
        if a is None:
            a = _NONE[d.device] = torch.empty(0, dtype=torch.bfloat16, device=d.device)
        _module().mixup_bf16(d[r0:r0 + 64], n[r0:r0 + 64], w, out[r0:r0 + 64], hc, MIX_JG(m) if jg is None else jg, a)
    return out


_NONE: dict = {}
#: rows from which `mix_up` computes its activations once (mixup_a_kernel) instead of in every CTA
MIX_PRE_FROM = 17


def MIX_JG(M: int) -> int:
    """`mix_up`'s column groups a CTA for M rows: 4 (80 CTAs of 512 threads) at every M measured, the 100 sites in
    3.19 / 3.24 / 3.24 / 3.33 / 3.78 ms at 1 / 4 / 8 / 16 / 32 rows (one group: 3.28 / 3.54 / 3.54 / 3.96 / 8.32)."""
    return 4
