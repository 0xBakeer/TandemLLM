"""On the board (SPD-241): the mixers' tensor-core kernels (QWEN38_FN_MIXER_MMA) on the served int8 shapes.

  * a row's bits do not depend on its call: for every M = 1..40 each row of the M-row call equals its one-row
    call, for the down|inject GEMV (int8, a scale a group of 128) and mix_up (int8, a group of 64);
  * the sums are the CUDA-core kernels' up to the order: against an fp64 reference of the dequantised weights,
    the tensor-core kernels' error is within 2x the CUDA-core kernels' and under 1e-6 of the largest output (the
    GEMV), and within two bf16 steps of the CUDA-core kernel's output (mix_up's bf16 output after a sigmoid);
  * the time of both at M = 1, 3, 24 (printed).

Run on the box: python tests/gpu/test_fn_mixer_mma_gpu.py   (under the GPU lock)
"""

from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import fn_kernels as K  # noqa: E402

H, R, HC = 2560, 320, 4
DEV = "cuda"


def _set(on: int):
    K._module().fn_set_mixer_mma(on)


def _restore():
    K._module().fn_set_mixer_mma(int(K._S.get("FN_MIXER_MMA") == "1"))


def _us(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def _gemv_case():
    g = torch.Generator(device=DEV).manual_seed(3)
    w = Int8Rows.of((torch.randn(R + HC, HC * H, generator=g, device=DEV) * 0.02).half(), 128)
    x = torch.randn(40, HC * H, generator=g, device=DEV).to(torch.bfloat16)
    return w, x


def test_gemv_rows_do_not_depend_on_the_call():
    w, x = _gemv_case()
    _set(1)
    try:
        one = torch.cat([K.gemv(x[m:m + 1], w) for m in range(40)])
        for M in range(2, 41):
            assert torch.equal(K.gemv(x[:M], w), one[:M]), M
    finally:
        _restore()
    return "M = 2..40: every row its one-row call's bits"


def test_gemv_error_within_the_cuda_core_kernels():
    w, x = _gemv_case()
    ref = x.double() @ w.dequant(torch.float64).T
    out = {}
    for on in (0, 1):
        _set(on)
        try:
            out[on] = K.gemv(x[:24], w)
        finally:
            _restore()
    e0 = ((out[0].double() - ref[:24]).abs().max() / ref[:24].abs().max()).item()
    e1 = ((out[1].double() - ref[:24]).abs().max() / ref[:24].abs().max()).item()
    # the tensor core's sum of a kstep rounds otherwise than a serial fp32 FMA chain: within 2x, at fp32's scale
    assert e1 <= 2.0 * e0 and e1 < 1e-6, (e0, e1)
    t = {(on, M): None for on in (0, 1) for M in (1, 3, 24)}
    for on, M in t:
        _set(on)
        try:
            t[on, M] = _us(lambda: K.gemv(x[:M], w))
        finally:
            _restore()
    return (f"max rel err vs fp64 {e0:.2e} (cuda cores) / {e1:.2e} (tensor cores); us M1 {t[0, 1]:.0f}->{t[1, 1]:.0f}, "
            f"M3 {t[0, 3]:.0f}->{t[1, 3]:.0f}, M24 {t[0, 24]:.0f}->{t[1, 24]:.0f}")


def _mix_case():
    g = torch.Generator(device=DEV).manual_seed(11)
    w = Int8Rows.of((torch.randn(HC * H, R, generator=g, device=DEV) * 0.05).half(), 64)
    d = torch.randn(40, R + HC, generator=g, device=DEV) * 2
    n = torch.randn(40, HC * H, generator=g, device=DEV).to(torch.bfloat16)
    return w, d, n


def test_mix_up_rows_do_not_depend_on_the_call():
    w, d, n = _mix_case()
    _set(1)
    try:
        one = torch.cat([K.mix_up(d[m:m + 1], n[m:m + 1], w, HC) for m in range(40)])
        for M in range(2, 41):
            assert torch.equal(K.mix_up(d[:M], n[:M], w, HC), one[:M]), M
    finally:
        _restore()
    return "M = 2..40: every row its one-row call's bits"


def test_mix_up_within_a_bf16_step_of_the_cuda_core_kernel():
    w, d, n = _mix_case()
    out = {}
    for on in (0, 1):
        _set(on)
        try:
            out[on] = K.mix_up(d[:24], n[:24], w, HC).float()
        finally:
            _restore()
    diff = (out[1] - out[0]).abs()
    step = out[0].abs().clamp_min(1e-3) * 2 ** -7                 # one bf16 step at the CUDA-core output
    frac = (diff > 0).float().mean().item()
    assert bool((diff <= 2 * step + 1e-6).all()), float((diff / step).max())
    t = {(on, M): None for on in (0, 1) for M in (1, 3, 24)}
    for on, M in t:
        _set(on)
        try:
            t[on, M] = _us(lambda: K.mix_up(d[:M], n[:M], w, HC))
        finally:
            _restore()
    return (f"{frac:.1%} of outputs differ, each within two bf16 steps; us M1 {t[0, 1]:.0f}->{t[1, 1]:.0f}, "
            f"M3 {t[0, 3]:.0f}->{t[1, 3]:.0f}, M24 {t[0, 24]:.0f}->{t[1, 24]:.0f}")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<52} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<52} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
