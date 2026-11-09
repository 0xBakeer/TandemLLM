"""SPD-249 P2: the hyper-connection mixer of a prompt chunk in two launches a site, bit for bit the path it replaces.

The prefill path (engine/models/flashnext/engine.py `_mix_pre`, QWEN38_FN_PREFILL_MM=fp32) runs a site's mixer as

    di = n.float() @ Wd^T                     # tools/fn_prefill_kernels.mm_rows: the weight's fp32 form, the library GEMM
    a  = silu(bf16(di[:, :R]).float() / hc).to(bf16)
    u  = a.float() @ Wu^T                     # [M, hc H] fp32, written and read back
    x  = mix_tail(u, n, hc)

which at 4,096 rows a chunk moves n twice in fp32, u twice and a few small tensors, around two library fp32 GEMMs, in
about ten launches. Here:

  down     one launch: di and a. Each di is one fp32 fma chain over k = 0 .. K-1 starting from 0 (the library's own
           sum at every row count a chunk has; tests/gpu/test_fn_pf_hc_gpu.py), the weight float(q) * s
           (`Int8Rows.dequant`), n read as bf16; a with the torch path's roundings.
  inject   `inject_norm` (the inject of a layer's output and the next site's norm) with the same bits in a CTA a row
           (4 x 128 threads, 16-byte loads; QWEN38_FN_PF_INJ_FUSED).
  up_tail  one launch: u's chains the same way over the R columns, then mix_tail's arithmetic in the epilogue; u is
           never stored.

Every rounding fence of the replaced path is kept: di fp32, bf16(di), the fp32 / hc as a multiply by 1 / hc, silu in
fp32, a bf16, u fp32, bf16(u), bf16(sigmoid), bf16(sigmoid n), the stream sum in stream order, the final bf16. The
tile shapes (`DOWN_TILE`, `UP_TILE`) never change a sum. Only int8 mixers with group scales (QWEN38_FN_MIX int8 /
int8rows, the served forms) on a GPU that builds the extension; anything else takes the replaced path.
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

#: the CTA shapes (hc.cu `pf_hc_down` / `pf_hc_up_tail` cases); measured on a DGX Spark at 4,096 rows
DOWN_TILE = 0
UP_TILE = 0


def _module():
    global _MOD
    if _MOD is None:
        tag = nb.tag(_CPP, _CUDA)
        _MOD = nb.load(f"fn_pf_hc_{tag}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["pf_hc_up_tail", "pf_inject_norm", "pf_mix_a", "pf_chain"],
                       extra_cuda_cflags=["-O3", "-lineinfo", *_gpu.arch_flags("fn_pf_hc")], verbose=False)
    return _MOD


def mix_ok(n: torch.Tensor, hc: int) -> bool:
    """Whether `mix` takes a site: n bf16 [M, hc H] on a CUDA device that builds the extension."""
    return (n.is_cuda and n.dtype == torch.bfloat16 and n.dim() == 2 and n.shape[1] % hc == 0
            and _gpu.profile(n.device).inline("fn_pf_hc"))


def mix_a(d: torch.Tensor, R: int, hc: int) -> torch.Tensor:
    """d fp32 [M, >= R] -> fp32 [M, R]: float(bf16(silu(bf16(d) / hc))), the `mm_rows` input the replaced path makes."""
    a = torch.empty(d.shape[0], R, dtype=torch.float32, device=d.device)
    _module().pf_mix_a(d, a, hc)
    return a


def mix(n: torch.Tensor, n32: torch.Tensor | None, wd, wu, R: int, hc: int):
    """A site's mixer, `_mix_pre`'s (x bf16 [M, H], di fp32 [M, R + hc]) bit for bit: the same library fp32 GEMMs on
    the same fp32 inputs (n's fp32 form from the inject's launch when given, else `n.float()`; a's from `mix_a`), the
    weights' fp32 forms as `mm_rows` takes them (its cache included), then `mix_tail`."""
    from tools import fn_prefill_kernels as PK
    di = (n32 if n32 is not None else n.float()) @ PK._wf(wd).T
    u = mix_a(di, R, hc) @ PK._wf(wu).T
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


def chain(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x fp32 [M, K] @ w fp32 [N, K]^T with each output one fp32 fma chain over k = 0 .. K-1 from 0 (a reference)."""
    y = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    _module().pf_chain(x.contiguous(), w.contiguous(), y)
    return y


def library_id() -> str:
    """The fp32 GEMM library this process runs: torch and its CUDA, the cuBLAS build loaded (its own version), the
    device. The row counts measured by tools/fn_pf_hc_rows.py hold for this id only."""
    import ctypes
    torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")      # loads cuBLAS
    path = None
    with open("/proc/self/maps") as f:
        for line in f:
            if "libcublas.so" in line and "Lt" not in line:
                path = line.split()[-1]
                break
    ver = "unknown"
    if path:
        lib = ctypes.CDLL(path)
        v = [ctypes.c_int() for _ in range(3)]
        if all(lib.cublasGetProperty(i, ctypes.byref(v[i])) == 0 for i in range(3)):
            ver = ".".join(str(x.value) for x in v)
    cap = torch.cuda.get_device_capability()
    return f"torch {torch.__version__} cuda {torch.version.cuda} cublas {ver} {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}"


def mix_reference(n: torch.Tensor, wd, wu, R: int, hc: int):
    """The path the launches replace (engine `_mix_pre` at QWEN38_FN_PREFILL_MM=fp32), for the tests."""
    from tools import fn_prefill_kernels as PK
    di = n.float() @ wd.dequant().T
    a = torch.nn.functional.silu(di[:, :R].to(torch.bfloat16).float() / hc).to(torch.bfloat16)
    return PK.mix_tail(a.float() @ wu.dequant().T, n, hc), di
