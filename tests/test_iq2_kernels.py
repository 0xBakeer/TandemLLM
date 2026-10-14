"""IQ2_XXS: the dequant twin (tools/iq2_linear.py) against gguf-py's `IQ2_XXS.dequantize_blocks` bit for
bit on random blocks (every bit pattern of a block is a valid block), Q8_0 likewise; the on-the-fly GEMM
against the twin's fp32 matmul, and its rows invariant to the row count up to 33 (bit for bit). The
GEMM runs on the GPU or under TRITON_INTERPRET=1; the twin anywhere."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import iq2_linear as Q  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
KERNELS = Q.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))


def random_iq2(N: int, K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = N * (K // 256)
    b = rng.integers(0, 256, size=(nb, 66), dtype=np.uint8)
    d = (rng.standard_normal(nb) * 0.02).astype(np.float16)
    b[:, :2] = d.view(np.uint8).reshape(nb, 2)
    return b.reshape(-1)


def test_tables():
    assert Q.GRID.shape == (256, 8) and set(np.unique(Q.GRID)) == {8.0, 25.0, 43.0}
    assert Q.KSIGNS[0] == 0 and Q.KSIGNS[1] == 0x81 and Q.KSIGNS[3] == 0x03 and Q.KSIGNS[127] == 0xFF


def test_twin_equals_gguf_py():
    try:
        from gguf.constants import GGMLQuantizationType
        from gguf.quants import dequantize
    except Exception:  # noqa: BLE001
        return "skipped: gguf-py missing"
    for N, K in ((3, 256), (5, 1024)):
        raw = random_iq2(N, K, seed=N)
        want = dequantize(raw.reshape(N, -1), GGMLQuantizationType.IQ2_XXS).astype(np.float32)
        got = Q.dequant_iq2_xxs(raw, N, K).numpy()
        assert want.shape == got.shape and np.array_equal(want.view(np.uint32), got.view(np.uint32)), \
            (N, K, np.abs(want - got).max())
        rng = np.random.default_rng(N)
        nb = N * K // 32
        b8 = rng.integers(0, 256, size=(nb, 34), dtype=np.uint8)
        b8[:, :2] = (rng.standard_normal(nb) * 0.02).astype(np.float16).view(np.uint8).reshape(nb, 2)
        w8 = dequantize(b8.reshape(N, -1), GGMLQuantizationType.Q8_0).astype(np.float32)
        assert np.array_equal(w8.view(np.uint32), Q.dequant_q8_0(b8.reshape(-1), N, K).numpy().view(np.uint32))
    return "IQ2_XXS and Q8_0 bit for bit"


def test_gemm_near_the_twin_and_row_invariant():
    if not KERNELS:
        return "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
    N, K = 96, 512
    raw = random_iq2(N, K, seed=7)
    lin = Q.IQ2Linear(raw, N, K, device=DEV)
    w = Q.dequant_iq2_xxs(raw, N, K)
    assert torch.equal(lin.dequant(), w)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(33, K, generator=g).to(torch.bfloat16).to(DEV)
    y = lin.matmul(x, f32=True)
    want = Q.matmul_reference(x.cpu(), w)
    err = ((y.cpu() - want).abs().max() / want.abs().max()).item()
    assert err < 1e-5, err
    for M in (1, 2, 16, 17, 33):
        y1 = lin.matmul(x[:M], f32=True)
        assert torch.equal(y1, y[:M]), M
    return f"rel err {err:.1e}; rows 1..33 bit for bit"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
