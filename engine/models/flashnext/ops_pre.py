"""tools/fn_prefill_kernels as the Flash-Next engine calls it (`from engine.models.flashnext import ops_pre as PK`):
fn_prefill_kernels' own names, except a weight or bank with its own kernels and NVIDIA FP4-Local's bf16 weights
(engine/models/flashnext/ops.py says why)."""

from __future__ import annotations

import torch

from tools import fn_prefill_kernels as _PK


def __getattr__(name):
    return getattr(_PK, name)


def mm_rows(x: torch.Tensor, w) -> torch.Tensor:
    """`fn_prefill_kernels.mm_rows`; a weight with its own `mul_rows(x)` multiplies itself, a bf16 weight runs on the
    library GEMM (fp32 accumulate, the sums rounded once)."""
    if hasattr(w, "mul_rows"):
        return w.mul_rows(x).float()
    if isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and x.dtype == torch.bfloat16:
        return torch.nn.functional.linear(x, w).float()
    return _PK.mm_rows(x, w)


def moe_grouped(x, ids, bank, *a, **k):
    if hasattr(bank, "experts"):
        return bank.experts(x, ids)
    return _PK.moe_grouped(x, ids, bank, *a, **k)
