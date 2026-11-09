"""SPD-249 P2: a prompt chunk's hyper-connection inject and mixer inputs in fewer passes, the bits of what they replace.

The prefill path (engine/models/flashnext/engine.py `_mix_pre`) runs a site's mixer as

    di = mm_rows(n, Wd)                       # tools/fn_prefill_kernels: QWEN38_FN_PREFILL_MM picks the GEMM
    a  = silu(bf16(di[:, :R]).float() / hc).to(bf16)
    u  = mm_rows(a, Wu)                       # [M, hc H] fp32
    x  = mix_tail(u, n, hc)

Both GEMMs and `mix_tail` stay as they are, in every `mm_rows` mode; only what feeds them changes:

  inject   `inject_norm` (the inject of a layer's output and the next site's norm) in a CTA a row (4 x 128 threads,
           16-byte loads), the norm's sum in the old kernel's order (QWEN38_FN_PF_INJ_FUSED). The same launch can
           write n's fp32 form.
  mix      a in one launch with the torch chain's roundings (bf16(di), fp32 multiply by 1 / hc, silu in fp32, bf16).
           Where `mm_rows` would take `x.float() @ W32^T` (`lib32`), the GEMM gets the same fp32 values without the
           `.float()` passes: n's from the inject's launch, a's written as fp32 by the kernel. Any other mode gets the
           bf16 n and a through `mm_rows` itself (QWEN38_FN_PF_HC_FUSED).

A GPU that builds the extension only; anything else takes the replaced path.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import native_build as nb  # noqa: E402
from tools import gpu as _gpu  # noqa: E402

_CUDA = nb.source(__file__, "hc.cu")

_CPP = nb.source(__file__, "hc.cpp")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        tag = nb.tag(_CPP, _CUDA)
        _MOD = nb.load(f"fn_pf_hc_{tag}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["pf_inject_norm", "pf_mix_a"],
                       extra_cuda_cflags=["-O3", "-lineinfo", *_gpu.arch_flags("fn_pf_hc")], verbose=False)
    return _MOD


def mix_ok(n: torch.Tensor, hc: int) -> bool:
    """Whether `mix` takes a site: n bf16 [M, hc H] on a CUDA device that builds the extension."""
    return (n.is_cuda and n.dtype == torch.bfloat16 and n.dim() == 2 and n.shape[1] % hc == 0
            and _gpu.profile(n.device).inline("fn_pf_hc"))


def lib32(w) -> bool:
    """Whether `mm_rows` takes a bf16 x with this weight as `x.float() @ W32^T` (then x's fp32 form made elsewhere is
    the same GEMM input): the fp32 mode, and the tc mode's weights that are not int8."""
    from tools import fn_prefill_kernels as PK
    return PK.MM == "fp32" or (PK.MM == "tc" and not hasattr(w, "q"))


def wants_n32() -> bool:
    """Whether an inject should write n's fp32 form for the next mixer (the mode whose down GEMM reads it)."""
    from tools import fn_prefill_kernels as PK
    return PK.MM == "fp32"


def mix_a(d: torch.Tensor, R: int, hc: int, fp32: bool = True) -> torch.Tensor:
    """d fp32 [M, >= R] -> [M, R]: bf16(silu(bf16(d) / hc)) as bf16, or with `fp32` its `.float()` (the fp32 GEMM's
    input), in one launch."""
    a = torch.empty(d.shape[0], R, dtype=torch.float32 if fp32 else torch.bfloat16, device=d.device)
    _module().pf_mix_a(d, a, hc)
    return a


def mix(n: torch.Tensor, n32: torch.Tensor | None, wd, wu, R: int, hc: int):
    """A site's mixer, `_mix_pre`'s (x bf16 [M, H], di fp32 [M, R + hc]) bit for bit in the `mm_rows` mode that runs:
    the same GEMMs on the same values (see `lib32`), then `mix_tail`."""
    from tools import fn_prefill_kernels as PK
    if lib32(wd):
        di = (n32 if n32 is not None else n.float()) @ PK._wf(wd).T
    else:
        di = PK.mm_rows(n, wd)
    if lib32(wu):
        u = mix_a(di, R, hc) @ PK._wf(wu).T
    else:
        u = PK.mm_rows(mix_a(di, R, hc, fp32=False), wu)
    return PK.mix_tail(u, n, hc), di


def inject_ok(h0: torch.Tensor, out: torch.Tensor, hc: int) -> bool:
    """Whether `inject_norm` takes the launch: hc 4 streams of a width that is a multiple of 8 up to 4,096, 16-byte rows,
    a CUDA device that builds the extension."""
    G = h0.shape[1] // hc if hc else 0
    return (h0.is_cuda and hc == 4 and h0.shape[1] == 4 * G and G % 8 == 0 and G <= 4096 and h0.is_contiguous()
            and out.stride(1) == 1 and out.shape[1] == G and out.stride(0) % 8 == 0 and out.data_ptr() % 16 == 0
            and _gpu.profile(h0.device).inline("fn_pf_hc"))


def inject_norm(h0: torch.Tensor, out: torch.Tensor, d: torch.Tensor, R: int, w: torch.Tensor | None, hc: int, eps: float,
                n32: bool = False):
    """tools/fn_kernels.inject_norm's (h, n), bit for bit, a row's streams in one CTA with 16-byte loads (hc.cu
    `pf_inject_norm_kernel`: the same per-element roundings and the same order of the norm's sum), and with `n32` (and
    a norm weight) n's fp32 form from the same launch: (h, n, n32 or None)."""
    hn = torch.empty_like(h0)
    nn = torch.empty_like(h0) if w is not None else hn
    nf = torch.empty(h0.shape, dtype=torch.float32, device=h0.device) if n32 and w is not None else None
    _module().pf_inject_norm(h0, out, d, R, w, hn, nn, nf, hc, eps)
    return hn, (nn if w is not None else None), nf


def mix_reference(n: torch.Tensor, wd, wu, R: int, hc: int):
    """The path `mix` replaces (engine `_mix_pre`, the `mm_rows` mode that runs), for the tests."""
    from tools import fn_prefill_kernels as PK
    di = PK.mm_rows(n, wd)
    a = torch.nn.functional.silu(di[:, :R].to(torch.bfloat16).float() / hc).to(torch.bfloat16)
    return PK.mix_tail(PK.mm_rows(a, wu), n, hc), di
