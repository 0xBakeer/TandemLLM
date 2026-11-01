"""tools/fn_kernels as the Flash-Next engine calls it (`from engine.models.flashnext import ops as K`): every name is
fn_kernels' own, except the calls whose weight carries its own kernels or is a type the EXL3 packs never have --
NVIDIA FP4-Local's (nvidia.py: `mul_rows`, `mix_up`, `experts`; bf16 GEMV weights and fp32 norm weights through
tools/fn_kernels_nv). The shared modules keep their bytes, so the other families' fingerprints do not move. Module
flags (`MOE_GROUP`, ...) are read here and written on tools/fn_kernels itself."""

from __future__ import annotations

import torch

from tools import fn_kernels as _K


def __getattr__(name):
    return getattr(_K, name)


def _nv():
    from tools import fn_kernels_nv
    return fn_kernels_nv


def gemv(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(w, "mul_rows"):
        y = w.mul_rows(x)
        return y.float() if out is None else out.copy_(y)
    if isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16:
        return _nv().gemv(x, w, out)
    return _K.gemv(x, w, out)


def gemv_bf16(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(w, "mul_rows"):
        y = w.mul_rows(x).to(torch.bfloat16)
        return y if out is None else out.copy_(y)
    if isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16:
        return _nv().gemv(x, w, out, bf16=True)
    return _K.gemv_bf16(x, w, out)


def mix_up(d: torch.Tensor, n: torch.Tensor, w_up, hc: int, out: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(w_up, "mix_up"):
        return w_up.mix_up(d, n, hc, out)
    return _K.mix_up(d, n, w_up, hc, out)


def moe(x, ids, wts, bank, order=None, bufs=None, out=None):
    if hasattr(bank, "experts"):
        return _K.moe_combine(bank.experts(x, ids), ids, wts, out=out)
    return _K.moe(x, ids, wts, bank, order=order, bufs=bufs, out=out)


def moe_experts(x, ids, bank, order=None):
    if hasattr(bank, "experts"):
        return bank.experts(x, ids)
    return _K.moe_experts(x, ids, bank, order=order)


def _f32(w) -> bool:
    return isinstance(w, torch.Tensor) and w.dtype == torch.float32


def hc_norm(x, w, group: int, eps: float, out=None):
    return _nv().hc_norm(x, w, group, eps, out) if _f32(w) else _K.hc_norm(x, w, group, eps, out)


def inject_norm(h0, out, d, R: int, w, hc: int, eps: float):
    return _nv().inject_norm(h0, out, d, R, w, hc, eps) if _f32(w) else _K.inject_norm(h0, out, d, R, w, hc, eps)


def combine_inject_norm(h0, yd, ids, wts, d, R: int, w, hc: int, eps: float):
    if _f32(w):
        return _nv().combine_inject_norm(h0, yd, ids, wts, d, R, w, hc, eps)
    return _K.combine_inject_norm(h0, yd, ids, wts, d, R, w, hc, eps)
