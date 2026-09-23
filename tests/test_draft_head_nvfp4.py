"""SPD-21: the drafter's NVFP4 vocabulary head.

Quantised a row block at a time with one per-tensor scale taken over the whole head, so the result
must be exactly what quantising the dequantised head in one piece gives -- the block size is a
memory decision and may not be a numerical one. And the flag is off by default.
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_block_quantisation_equals_one_piece():
    from tools.head_gemv import FP8Head, head_to_nvfp4
    from tools.nvfp4_linear import E2M1_MAX, E4M3_MAX, quantize_to_nvfp4
    g = torch.Generator().manual_seed(0)
    w = torch.randn(300, 128, generator=g) * 0.05
    w[17, 5] = 0.9                                   # an outlier in one block only
    amax = w.abs().amax(1, keepdim=True)
    scale = (amax / 448.0).squeeze(1)
    head = FP8Head((w / amax * 448.0).clamp(-448, 448).to(torch.float8_e4m3fn), scale)
    blocked = head_to_nvfp4(head, rows=64)
    full = head.w.float() * head.s[:, None]
    s2 = float((full.abs().max() / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    one = quantize_to_nvfp4(full, scale_2=s2)
    assert blocked.s2 == one.s2
    assert torch.equal(blocked.w, one.w)
    assert torch.equal(blocked.s.view(torch.uint8), one.s.view(torch.uint8))
    assert blocked.shape == (300, 128)
    return "row blocks of 64 over 300 rows == one piece, codes, scales and s2"


def test_off_by_default():
    from engine.drafters import dflash2
    old = os.environ.pop("QWEN38_DRAFT_HEAD_NVFP4", None)
    try:
        assert importlib.reload(dflash2).DRAFT_HEAD_NVFP4 is False
    finally:
        if old is not None:
            os.environ["QWEN38_DRAFT_HEAD_NVFP4"] = old
        importlib.reload(dflash2)
    return "QWEN38_DRAFT_HEAD_NVFP4 unset -> the e4m3 head"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
