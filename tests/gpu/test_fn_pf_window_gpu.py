"""On the board (SPD-249 P1): the kernel facts Flash-Next's layer-major prefill windows (engine/models/flashnext/
layer_major.py) and the row-order pair rotation (QWEN38_FN_PF_ROT_ROWS) rest on, bit for bit.

  * `fn_prefill_kernels.moe_grouped` with the pair rotation in row order equals the grouped order's at 1 to 4,099
    rows (skewed routing: experts take several items), and over a window's rows (chunks of 2,048, 2,048 and 3
    rows) each chunk's pairs get their own call's bits;
  * the slab cache and `mm_rows`' weight cache are the joint prefill's (tests/gpu/test_fn_joint_prefill_gpu.py).

The whole-model check is tools/fn_window_check.py (logits, streams, sections, KV, every chunk line).
Run on the box: python tests/gpu/test_fn_pf_window_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import fn_prefill_kernels as PK  # noqa: E402
from test_fn_joint_prefill_gpu import H, _bank, _ids, _x  # noqa: E402


def _grouped(x, ids, bank, rot: bool):
    was = PK.ROT_ROWS
    PK.ROT_ROWS = rot
    try:
        return PK.moe_grouped(x, ids, bank)
    finally:
        PK.ROT_ROWS = was


def test_row_order_rotation_keeps_the_bits():
    bank = _bank(24, (2, 3, 4), 5, seed=11)
    cases = []
    for M in (1, 7, 64, 205, 1031, 4099):
        x, ids = _x(M, H, 300 + M, 0.5), _ids(M, 24, 10, 500 + M)
        a, b = _grouped(x, ids, bank, False), _grouped(x, ids, bank, True)
        assert torch.equal(a, b), M
        cases.append(M)
    return f"rows {cases}: equal"


def test_a_windows_rows_are_each_chunks_own():
    bank = _bank(24, (2, 3, 4), 5, seed=13)
    lens = (2048, 2048, 3)
    xs = [_x(M, H, 700 + j, 0.5) for j, M in enumerate(lens)]
    ids = [_ids(M, 24, 10, 800 + j) for j, M in enumerate(lens)]
    for rot in (False, True):
        solo = [_grouped(x, i, bank, False) for x, i in zip(xs, ids)]
        win = _grouped(torch.cat(xs), torch.cat(ids), bank, rot)
        r0 = 0
        for M, y in zip(lens, solo):
            assert torch.equal(win[r0:r0 + y.shape[0]], y), (rot, M)
            r0 += y.shape[0]
    return f"chunks {lens} in one call (both rotation orders) = each chunk's own grouped-order call"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<46} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<46} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
