"""bf16 weights x bf16 activations for 1..32 rows on the tensor cores, fp32 out: NVIDIA FP4-Local's mixers
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
                       functions=["bf16_skinny", "bf16_cluster"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"])
    return _MOD


def default_split(N: int, K: int) -> tuple[int, int]:
    """(wk, nt) for a weight [N, K]: a function of the shape only (row invariance)."""
    steps = K // 64
    wk = 16 if steps >= 32 else 8 if steps >= 16 else 2 if steps >= 4 else 1
    return wk, 1


#: the next K step's weights loaded during this one's products (same bits)
PF = 1


def mm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, wk: int | None = None,
       nt: int | None = None, pf: int | None = None) -> torch.Tensor:
    """x [M, K] bf16 (M <= 32) @ w [N, K] bf16 -> fp32 [M, N]."""
    M, K = x.shape
    N = w.shape[0]
    y = out if out is not None else torch.empty(M, N, dtype=torch.float32, device=x.device)
    dwk, dnt = default_split(N, K)
    _module().bf16_skinny(x.contiguous(), w, y, dwk if wk is None else wk, dnt if nt is None else nt,
                          PF if pf is None else pf)
    return y


def mm_cluster(x: torch.Tensor, w: torch.Tensor, wk: int = 8, cs: int = 4, out: torch.Tensor | None = None) -> torch.Tensor:
    """`mm` with K split across a cluster of `cs` CTAs (a weight with few rows), reduced through distributed shared
    memory in rank order: fp32 [M, N]. Row-invariant (the order depends on the shape, wk and cs only)."""
    M = x.shape[0]
    y = out if out is not None else torch.empty(M, w.shape[0], dtype=torch.float32, device=x.device)
    _module().bf16_cluster(x.contiguous(), w, y, wk, cs)
    return y
