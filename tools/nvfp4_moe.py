"""NVIDIA FP4-Local's routed experts (NVFP4, stacked by expert) for 1..64 rows: the plan, gate|up with the activation,
and down, on the tensor cores with bf16 activations (W4A16). Row-invariant: a pair's output is the same bits whichever
pairs share its call (tools/csrc/nvfp4_moe/moe.cu). Sources: tools/csrc/nvfp4_moe/.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import native_build as nb  # noqa: E402

import torch  # noqa: E402

_CUDA = nb.source(__file__, "moe.cu")
_CPP = nb.source(__file__, "moe.cpp")
_MOD = None
PMAX = 1024
#: groups a launch's grid holds on its y axis (a CTA walks groups y, y + GY, ...)
GY = 64


def _module():
    global _MOD
    if _MOD is None:
        _MOD = nb.load(f"tandem_nvfp4_moe_{nb.tag(_CPP, _CUDA)}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["nvfp4_moe_plan", "nvfp4_moe_gu", "nvfp4_moe_down"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"])
    return _MOD


def plan(ids: torch.Tensor, E: int) -> dict:
    """ids int32 [M, S] (the shared expert as E) -> the routed pairs by expert in groups of <= 16 (device tensors)."""
    P = ids.numel()
    dev = ids.device
    p = {"order": torch.empty(P, dtype=torch.int32, device=dev), "gstart": torch.empty(P, dtype=torch.int32, device=dev),
         "gcount": torch.empty(P, dtype=torch.int32, device=dev), "gexp": torch.empty(P, dtype=torch.int32, device=dev),
         "ng": torch.empty(1, dtype=torch.int32, device=dev)}
    _module().nvfp4_moe_plan(ids.reshape(-1).contiguous(), E, p["order"], p["gstart"], p["gcount"], p["gexp"], p["ng"])
    return p


def experts(x: torch.Tensor, ids: torch.Tensor, bank, wk_gu: int = 4, wk_d: int = 1, nt_d: int = 2) -> torch.Tensor:
    """Every routed (row, slot) pair's down output, unweighted: x [M, H] bf16, ids int32 [M, S] -> yd [M S, H] fp32
    (the shared expert's rows left unwritten: the caller's). The K splits are the weights' shapes' (row invariance)."""
    M, S = ids.shape
    P = M * S
    assert P <= PMAX, P
    p = plan(ids, bank.n)
    a = torch.empty(P, bank.I, dtype=torch.bfloat16, device=x.device)
    y = torch.empty(P, bank.H, dtype=torch.float32, device=x.device)
    gy = min(P, GY)
    m = _module()
    m.nvfp4_moe_gu(x.contiguous(), S, p["order"], p["gstart"], p["gcount"], p["gexp"], p["ng"], bank.gu_w, bank.gu_s,
                   bank.gu_s2, a, gy, wk_gu)
    m.nvfp4_moe_down(a, p["order"], p["gstart"], p["gcount"], p["gexp"], p["ng"], bank.d_w, bank.d_s, bank.d_s2, y, gy,
                     wk_d, nt_d)
    return y
