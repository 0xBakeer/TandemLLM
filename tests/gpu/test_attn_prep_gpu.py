"""SPD-40 on the board: the attention layer's q/k norms and partial rotary in one launch.

Against the engine's own path (the fused RMS norm, a transpose, `apply_rope` in bf16) bit for bit:
one row, a chain of 8 and 16 rows, a tree's repeated positions, a prefill-sized block, and the query
as the strided slice of the interleaved q|gate projection the engine hands over.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools import attn_prep as AP  # noqa: E402


def test_bit_identical_to_the_engine_path():
    lines = AP.check()
    assert all(ln.endswith("bit-identical") for ln in lines), lines
    return "; ".join(lines)


def test_a_prefill_block_and_the_last_rope_row():
    g = torch.Generator(device="cuda").manual_seed(5)
    Hq, Hk, D, R, P = 24, 4, 256, 64, 2048
    emb = torch.randn(P, R, device="cuda", generator=g)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    wq = (torch.randn(D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    wk = (torch.randn(D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    T = 300
    q = torch.randn(T, Hq, 2 * D, device="cuda", generator=g).to(torch.bfloat16)[..., :D]
    k = torch.randn(T, Hk, D, device="cuda", generator=g).to(torch.bfloat16)
    pos = torch.arange(P - T, P, device="cuda")
    a = AP.attn_prep(q, k, wq, wk, cos, sin, pos, 1e-6)
    b = AP.reference(q, k, wq, wk, cos, sin, pos, 1e-6)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[0].is_contiguous() and a[0].shape == (1, Hq, T, D) and a[1].shape == (1, Hk, T, D)
    return f"T={T} ending at the table's last row: bit-identical, contiguous [1, H, T, D]"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
