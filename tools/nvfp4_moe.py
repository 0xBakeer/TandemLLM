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
#: row tiles of a prompt chunk's groups (16 pairs a tile)
PREFILL_MT = 4
#: (wk, nt) of a prompt chunk's gate|up launch
PREFILL_GU = (1, 2)
#: groups a launch's grid holds on its y axis (a CTA walks groups y, y + GY, ...)
GY = 64


def _module():
    global _MOD
    if _MOD is None:
        _MOD = nb.load(f"tandem_nvfp4_moe_{nb.tag(_CPP, _CUDA)}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["nvfp4_moe_plan", "nvfp4_moe_gu", "nvfp4_moe_down"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"])
    return _MOD


def plan_torch(ids: torch.Tensor, E: int, group: int = 16) -> dict:
    """`plan` in torch ops for any number of pairs (a prompt chunk; not for a graph: one host read of the count),
    the same groups the plan kernel makes."""
    flat = ids.reshape(-1).long()
    P = flat.numel()
    dev = flat.device
    order = torch.argsort(flat * P + torch.arange(P, device=dev))        # by (expert, pair)
    se = flat[order]
    pos = torch.arange(P, device=dev)
    new = torch.ones(P, dtype=torch.bool, device=dev)
    new[1:] = se[1:] != se[:-1]
    start = torch.cummax(torch.where(new, pos, torch.zeros_like(pos)), 0).values      # each position's run start
    lead = (((pos - start) % group) == 0) & (se < E)
    gpos = torch.nonzero(lead).view(-1)
    end = torch.empty(P, dtype=torch.long, device=dev)                  # each position's run end
    last = torch.ones(P, dtype=torch.bool, device=dev)
    last[:-1] = new[1:]
    ends = torch.nonzero(last).view(-1) + 1
    end = ends[torch.cumsum(new.long(), 0) - 1]
    return {"order": order.to(torch.int32), "gstart": gpos.to(torch.int32),
            "gcount": torch.minimum(end[gpos] - gpos, torch.full_like(gpos, group)).to(torch.int32),
            "gexp": se[gpos].to(torch.int32),
            "ng": torch.tensor([gpos.numel()], dtype=torch.int32, device=dev), "n": gpos.numel()}


def plan(ids: torch.Tensor, E: int) -> dict:
    """ids int32 [M, S] (the shared expert as E) -> the routed pairs by expert in groups of <= 16 (device tensors).
    Up to PMAX pairs: one CTA on the device (a graph can hold it); more: `plan_torch`."""
    P = ids.numel()
    if P > PMAX:
        return plan_torch(ids, E)
    dev = ids.device
    p = {"order": torch.empty(P, dtype=torch.int32, device=dev), "gstart": torch.empty(P, dtype=torch.int32, device=dev),
         "gcount": torch.empty(P, dtype=torch.int32, device=dev), "gexp": torch.empty(P, dtype=torch.int32, device=dev),
         "ng": torch.empty(1, dtype=torch.int32, device=dev)}
    _module().nvfp4_moe_plan(ids.reshape(-1).contiguous(), E, p["order"], p["gstart"], p["gcount"], p["gexp"], p["ng"])
    return p


def experts(x: torch.Tensor, ids: torch.Tensor, bank, wk_gu: int = 4, nt_gu: int = 1, wk_d: int = 1,
            nt_d: int = 2, mt: int | None = None) -> torch.Tensor:
    """Every routed (row, slot) pair's down output, unweighted: x [M, H] bf16, ids int32 [M, S] -> yd [M S, H] fp32
    (the shared expert's rows left unwritten: the caller's). The K splits are the weights' shapes' (row invariance)."""
    M, S = ids.shape
    P = M * S
    # a group is one 16-row tile while a graph may hold the call (the plan kernel); a prompt chunk groups up to
    # 16 MT pairs of one expert, each weight decoded once for them (MT = PREFILL_MT)
    if mt is None:
        mt = 1 if P <= PMAX else PREFILL_MT
        if mt > 1:
            # a prompt chunk's rows agree with the row step on the next token, not bit for bit (engine.forward_pre):
            # the fastest split for many rows (4,000 rows x 48 layers: 556 ms against 669 at the row step's split)
            wk_gu, nt_gu = PREFILL_GU
    p = plan(ids, bank.n) if mt == 1 else plan_torch(ids, bank.n, 16 * mt)
    a = torch.empty(P, bank.I, dtype=torch.bfloat16, device=x.device)
    y = torch.empty(P, bank.H, dtype=torch.float32, device=x.device)
    gy = min(P, GY) if "n" not in p else max(1, min(p["n"], 1024))
    m = _module()
    m.nvfp4_moe_gu(x.contiguous(), S, p["order"], p["gstart"], p["gcount"], p["gexp"], p["ng"], bank.gu_w, bank.gu_s,
                   bank.gu_s2, a, gy, wk_gu, nt_gu, mt)
    m.nvfp4_moe_down(a, p["order"], p["gstart"], p["gcount"], p["gexp"], p["ng"], bank.d_w, bank.d_s, bank.d_s2, y, gy,
                     wk_d, nt_d, mt)
    return y
