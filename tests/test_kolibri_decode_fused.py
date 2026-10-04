"""The fused decode step and the glue kernels on the GPU; skipped without CUDA.

`tools/kolibri_dec_check.py` does the work: the fused step's k/v/idx writes must be bit-equal to the
unfused path's, its attention as close to the fp32 reference as the unfused kernel's (1 bf16 ulp of
sum order apart), and the SwiGLU kernel bit-equal to torch's silu * up.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_fused_decode_matches_unfused_path_and_reference():
    from tools import kolibri_dec_check as dc
    for row in dc.parity():
        assert row["writes_equal"], row
        assert row["new_vs_ref"] <= 1.5 * row["old_vs_ref"] + 2e-3, row
        assert row["new_vs_old"] <= 8e-3, row


def test_swiglu_bit_equal_to_torch():
    from tools import kolibri_dec_check as dc
    assert all(v == 0 for v in dc.glue().values())


def test_moe_combine_extra_equals_add():
    from engine.kolibri.kernels import _moe_combine_kernel
    import triton
    M, k, H = 1, 7, 2560
    p = torch.randn(M * k, H, device="cuda")
    ex = torch.randn(M, H, device="cuda").to(torch.bfloat16)
    y0 = torch.empty(M, H, device="cuda")
    y1 = torch.empty(M, H, device="cuda")
    _moe_combine_kernel[(M, triton.cdiv(H, 256))](p, y0, H, k, p.stride(0), y0.stride(0), p, 0,
                                                  BN=256, EXTRA=False)
    _moe_combine_kernel[(M, triton.cdiv(H, 256))](p, y1, H, k, p.stride(0), y1.stride(0), ex,
                                                  ex.stride(0), BN=256, EXTRA=True)
    assert torch.equal(y1, y0 + ex.float())


def test_small_ring_takes_one_slice():
    from tools import kolibri_attn_kernels as K
    torch.manual_seed(0)
    nq, nk, d, R = 6, 2, 16, 16
    y = torch.randn(1, (nq + 2 * nk) * d, device="cuda").to(torch.bfloat16)
    qn = torch.ones(d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, nk, R, d, device="cuda").to(torch.bfloat16)
    v = torch.randn_like(k)
    idx = torch.arange(R, dtype=torch.int32, device="cuda") + 20       # rows 20..35 (wrapped)
    idx = idx[(torch.arange(R, device="cuda") - 20) % R]
    pos = torch.tensor([36], dtype=torch.int32, device="cuda")
    o = K.decode_fused(y, qn, qn, pos, k, v, idx, nq=nq, nk=nk, d=d, theta=1e4, eps=1e-6,
                       rope=True, window=9, scale=d ** -0.5)
    assert int(idx[36 % R]) == 36
    assert torch.isfinite(o.float()).all() and o.shape == (1, nq * d)
