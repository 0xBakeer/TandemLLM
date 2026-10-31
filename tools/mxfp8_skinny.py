"""The W8A16 product for MXFP8 weights (NVIDIA FP4-Local's dense layers), 1..64 rows, on the tensor cores.

MXFP8 = e4m3 values with one E8M0 scale (a power of two) per 32 along K. The weight is decoded in registers to the
bf16 bits of its exact value and multiplied by bf16 activations (W8A16: the activations are never quantised), fp32
accumulate, bf16 out. Row-invariant by construction: the K split depends on the weight's shape only, the slices are
summed in warp order, no atomics (docs/kernels.md "Row invariance"). Sources: tools/csrc/mxfp8_skinny/.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import native_build as nb  # noqa: E402

import torch  # noqa: E402

_CUDA = nb.source(__file__, "mxfp8.cu")
_CPP = nb.source(__file__, "mxfp8.cpp")
_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        tag = nb.tag(_CPP, _CUDA)
        _MOD = nb.load(f"tandem_mxfp8_skinny_{tag}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["mxfp8_skinny"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"])
    return _MOD


def default_wk(K: int) -> int:
    """K-split warps for a weight of width K: a function of K only (row invariance)."""
    steps = K // 64
    return 16 if steps >= 32 else 8 if steps >= 16 else 2 if steps >= 4 else 1


def default_nt(M: int) -> int:
    """8-row weight tiles a CTA: 1 up to 16 rows, 2 above (the 2-row-tile path runs out of registers at nt=1 and high
    wk). Free to follow M: the tile count regroups whole output columns, the bits do not change."""
    return 1 if M <= 16 else 2


#: the next K step's weights loaded during this one's products (0: off); same bits either way
PF = 1


def mm(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, wk: int | None = None, out: torch.Tensor | None = None,
       nt: int | None = None, pf: int | None = None):
    """x [M, K] bf16 (M <= 64) @ MXFP8 w [N, K] (uint8 e4m3 bytes) with scales s [N, K/32] (uint8 E8M0) -> [M, N] bf16.

    nt = 8-row weight tiles a CTA owns (1, 2, 4): the activations of a K step are loaded once and reused across them."""
    M, K = x.shape
    N = w.shape[0]
    y = out if out is not None else torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _module().mxfp8_skinny(x.contiguous(), w, s, y, default_wk(K) if wk is None else wk,
                            default_nt(M) if nt is None else nt, PF if pf is None else pf)
    return y
