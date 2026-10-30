"""On the board (SPD-231): the register-blocked mixer kernels (QWEN38_FN_GEMV_NB) against the one-row-a-CTA
kernels they replace for M > 1.

  * gemv (gemv_split's shapes: int8 rows, int8 with group scales, fp16; fp32 and bf16 out) and mix_up (int8,
    int8 with group scales, fp16 up weights): for every M = 2..32 the output with the knob equals the output
    without it bit for bit, and every row equals its M = 1 call (which never takes the new kernels); fp16
    weights keep the old kernels (slower on the new ones, 2026-10-24);
  * the time of both at M = 4 and 32 (printed; the joint pass of 8 lanes x 4 rows is the 32-row case).

Run on the box: python tests/gpu/test_fn_gemv_nb_gpu.py   (under the GPU lock)
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


def _x(M, K_, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, K_, generator=g, device=DEV) * scale).to(torch.bfloat16)


def _both(fn):
    """fn() on the old kernels and on the new ones."""
    mod = K._module()
    mod.fn_set_gemv_nb(0)
    a = fn()
    mod.fn_set_gemv_nb(1)
    try:
        b = fn()
    finally:
        mod.fn_set_gemv_nb(int(K._S.get("FN_GEMV_NB") == "1"))
    return a, b


def _us(fn, on: int, n=50):
    mod = K._module()
    mod.fn_set_gemv_nb(on)
    try:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6
    finally:
        mod.fn_set_gemv_nb(int(K._S.get("FN_GEMV_NB") == "1"))


def test_gemv_new_equals_old_and_rows():
    out = []
    for N, Kk, kind in ((324, 10240, "int8"), (324, 10240, "int8g"), (513, 2560, "fp16"), (96, 2560, "fp16"),
                        (64, 2560, "int8g")):
        g = torch.Generator(device=DEV).manual_seed(N + Kk)
        w16 = (torch.randn(N, Kk, generator=g, device=DEV) * 0.02).half()
        w = Int8Rows.of(w16) if kind == "int8" else Int8Rows.of(w16, 128) if kind == "int8g" else w16
        x = _x(32, Kk, 3)
        one = torch.cat([K.gemv(x[m:m + 1], w) for m in range(32)])
        for M in range(2, 33):
            a, b = _both(lambda: K.gemv(x[:M], w))
            assert torch.equal(a, b), (N, Kk, kind, M, (a - b).abs().max().item())
            assert torch.equal(b, one[:M]), (N, Kk, kind, M)
            a, b = _both(lambda: K.gemv_bf16(x[:M], w))
            assert torch.equal(a, b), (N, Kk, kind, M, "bf16 out")
        t = {M: (_us(lambda: K.gemv(x[:M], w), 0), _us(lambda: K.gemv(x[:M], w), 1)) for M in (4, 32)}
        out.append(f"{N}x{Kk} {kind} M4 {t[4][0]:.0f}->{t[4][1]:.0f} us, M32 {t[32][0]:.0f}->{t[32][1]:.0f} us")
    return "; ".join(out)


def test_mix_up_new_equals_old_and_rows():
    out = []
    for kind in ("int8", "int8g", "fp16"):
        g = torch.Generator(device=DEV).manual_seed(11)
        w16 = (torch.randn(HC * H, R, generator=g, device=DEV) * 0.05).half()
        w = Int8Rows.of(w16) if kind == "int8" else Int8Rows.of(w16, 64) if kind == "int8g" else w16
        d = torch.randn(32, R + 4, generator=g, device=DEV) * 2        # the down|inject output: R + hc columns
        n = _x(32, HC * H, 6)
        one = torch.cat([K.mix_up(d[m:m + 1], n[m:m + 1], w, HC) for m in range(32)])
        for M in range(2, 33):
            a, b = _both(lambda: K.mix_up(d[:M], n[:M], w, HC))
            assert torch.equal(a, b), (kind, M, (a.float() - b.float()).abs().max().item())
            assert torch.equal(b, one[:M]), (kind, M)
        t = {M: (_us(lambda: K.mix_up(d[:M], n[:M], w, HC), 0), _us(lambda: K.mix_up(d[:M], n[:M], w, HC), 1))
             for M in (4, 32)}
        out.append(f"{kind} M4 {t[4][0]:.0f}->{t[4][1]:.0f} us, M32 {t[32][0]:.0f}->{t[32][1]:.0f} us")
    return "; ".join(out)


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<44} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<44} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
