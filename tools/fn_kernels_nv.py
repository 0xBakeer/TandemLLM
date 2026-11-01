"""tools/fn_kernels' kernels on NVIDIA FP4-Local's weight types (bf16 GEMV weights, fp32 norm weights), built as a
module of their own: fn_kernels' source text unchanged, with the launches for those types appended
(tools/csrc/fn_kernels_nv/). The shared module, its build and the fingerprints of the models that reach it keep
their bytes; Flash-Next's dispatch (engine/models/flashnext/ops.py) sends NVIDIA's weights here.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import native_build as nb  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402

import torch  # noqa: E402

_CUDA = nb.source(__file__, "nv.cu")
_CPP = nb.source(__file__, "nv.cpp")
_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        src = "#define EXL3_ST 3\n" + FK._HELPERS + FK._CUDA + _CUDA
        _MOD = nb.load(f"fn_kernels_nv_{nb.tag(_CPP, src)}", cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["nv_gemv", "nv_hc_norm", "nv_inject_norm", "nv_combine_inject_norm", "fn_set_pdl",
                                  "fn_set_gemv_split", "fn_set_gemv_nb", "fn_set_gemv_mc"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
        # the launch options fn_kernels sets on its own copy (the same knobs, the same kernels)
        _MOD.fn_set_pdl(int(_S.get("FN_PDL") == "1"))
        _MOD.fn_set_gemv_split(int(_S.get("FN_GEMV_SPLIT") == "1"))
        _MOD.fn_set_gemv_nb(int(_S.get("FN_GEMV_NB") == "1"))
        _MOD.fn_set_gemv_mc(int(_S.get("FN_GEMV_MC") or 0))
    return _MOD


def gemv(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, bf16: bool = False) -> torch.Tensor:
    """x [M, K] bf16 @ w [N, K] bf16 ^T -> fp32 [M, N] (bf16 = True: the sums rounded to bf16), fn_gemv's kernel."""
    if out is None:
        out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16 if bf16 else torch.float32, device=x.device)
    _module().nv_gemv(x, w, out)
    return out


def hc_norm(x: torch.Tensor, w: torch.Tensor, group: int, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """`fn_kernels.hc_norm` with an fp32 weight."""
    if out is None:
        out = torch.empty_like(x)
    _module().nv_hc_norm(x, w, out, group, eps)
    return out


def inject_norm(h0, out, d, R: int, w: torch.Tensor, hc: int, eps: float):
    """`fn_kernels.inject_norm` with an fp32 weight."""
    hn, nn = torch.empty_like(h0), torch.empty_like(h0)
    _module().nv_inject_norm(h0, out, d, R, w, hn, nn, hc, eps)
    return hn, nn


def combine_inject_norm(h0, yd, ids, wts, d, R: int, w: torch.Tensor, hc: int, eps: float):
    """`fn_kernels.combine_inject_norm` with an fp32 weight."""
    hn, nn = torch.empty_like(h0), torch.empty_like(h0)
    _module().nv_combine_inject_norm(h0, yd, ids.reshape(-1), wts.reshape(-1), d, R, w, hn, nn, hc, eps)
    return hn, nn
