"""NemotronH EXL3 kernels on the board (ENG-E12): the relu^2 expert bank (tools/exl3_relu2.py) against its
CPU arithmetic (each projection through tools/exl3_linear.reference_matmul), a pair's bits independent of
the call's row count (a verify block's rows are the decode step's), and the dense projections at Bolt's
shapes through tools/fn_kernels.dense against the reference. GPU only.

    python tests/gpu/test_nemotron_h_exl3_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.exl3_linear import random_block, reference_matmul  # noqa: E402
from tools.exl3_relu2 import Relu2Bank  # noqa: E402


def _bank(E: int, I: int, X: int, kbs=(2, 25, 3, 4)):
    return Relu2Bank([(random_block(I, X, kbs[e % len(kbs)], seed=e), random_block(X, I, kbs[(e + 1) % len(kbs)], seed=100 + e))
                      for e in range(E)])


def test_relu2_bank_against_its_reference():
    # Bolt's widest layer: 2,560-wide experts over the 1,024 latent, top 9
    bank = _bank(12, 2560, 1024)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(4, 1024, device="cuda", generator=g) * 2).to(torch.bfloat16)
    ids = torch.randint(0, 12, (4, 9), device="cuda", generator=g, dtype=torch.int32)
    yd = bank.moe_relu2(x, ids)
    worst = 0.0
    for p, e in enumerate(ids.reshape(-1).tolist()):
        up, down = bank.expert(e)
        a = torch.relu(reference_matmul(x[p // 9:p // 9 + 1], up)).square().to(torch.bfloat16)
        want = reference_matmul(a, down).float()
        worst = max(worst, float((yd[p] - want[0]).abs().max() / want.abs().max()))
    assert worst < 2e-2, worst
    return f"worst rel {worst:.2e}"


def test_relu2_rows_are_their_lone_calls():
    bank = _bank(8, 1536, 1024)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(32, 1024, device="cuda", generator=g)).to(torch.bfloat16)
    ids = torch.randint(0, 8, (32, 9), device="cuda", generator=g, dtype=torch.int32)
    full = bank.moe_relu2(x, ids)
    for m in (0, 7, 31):
        one = bank.moe_relu2(x[m:m + 1], ids[m:m + 1])
        assert torch.equal(one, full[m * 9:(m + 1) * 9]), m


def test_dense_projections_at_bolt_shapes():
    from engine.models.nemotron_h.engine import _exl3_mm
    worst = 0.0
    for N, K, kb in ((18560, 4096, 4), (4096, 8192, 4), (1024, 4096, 4), (5376, 4096, 2), (131072, 4096, 6)):
        w = random_block(N, K, kb, seed=N)
        x = torch.randn(8, K, device="cuda").to(torch.bfloat16)
        y = _exl3_mm(x, w).float()
        ref = reference_matmul(x, w).float()
        worst = max(worst, float((y - ref).abs().max() / ref.abs().max()))
        assert torch.equal(_exl3_mm(x[:1], w), _exl3_mm(x, w)[:1]), (N, K)
    assert worst < 2e-2, worst
    return f"worst rel {worst:.2e}"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            r = fn()
            print(f"  {name:<50} ok   {r or ''}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:<50} FAIL {type(e).__name__}: {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} failed" if fails else ""))
    sys.exit(1 if fails else 0)
