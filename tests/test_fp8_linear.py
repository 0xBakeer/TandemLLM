"""tools/nvfp4_relu2.py FP8Linear: rows bitwise invariant to the call's row count for M = 1..33
(a verify block is 33 rows and takes the same row kernel as decode), the row kernel within bf16
tolerance of the fp32 e4m3 decode, and the M > 33 prompt path still correct."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nvfp4_relu2 import FP8Linear, HAVE_TRITON  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
KERNELS = HAVE_TRITON and torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET")

SHAPES = [(256, 512), (4096, 4096), (96, 320)]            # 320 % 128 != 0 -> the BK=64 rule


def _linear(N, K, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 40).clamp(-400, 400).to(torch.float8_e4m3fn).to(DEV)
    return FP8Linear(w, 0.0137)


def test_fp8_row_invariance():
    if not KERNELS:
        return
    for N, K in SHAPES:
        lin = _linear(N, K, N + K)
        for f32 in (False, True):
            x = torch.randn(33, K, dtype=torch.bfloat16, device=DEV)
            for M in (1, 2, 9, 16, 17, 32, 33):
                y = lin.matmul(x[:M], f32=f32)
                for r in range(M):
                    yr = lin.matmul(x[r:r + 1], f32=f32)
                    assert torch.equal(yr, y[r:r + 1]), (N, K, M, r, f32)


def test_fp8_matches_dequant():
    if not KERNELS:
        return
    for N, K in SHAPES:
        lin = _linear(N, K, N * 3 + K)
        x = torch.randn(9, K, dtype=torch.bfloat16, device=DEV)
        ref = x.float() @ lin.dequant().float().T
        y = lin.matmul(x, f32=True)
        rel = ((y - ref).norm() / (ref.norm() + 1e-9)).item()
        assert rel < 0.01, (N, K, rel)


def test_fp8_prompt_path():
    if not KERNELS:
        return
    for N, K in SHAPES:
        lin = _linear(N, K, N * 5 + K)
        x = torch.randn(70, K, dtype=torch.bfloat16, device=DEV)
        ref = x.float() @ lin.dequant().float().T
        for f32 in (False, True):
            y = lin.matmul(x, f32=f32).float()
            rel = ((y - ref).norm() / (ref.norm() + 1e-9)).item()
            assert rel < 0.01, (N, K, f32, rel)
        assert lin.matmul(x).shape == (70, N)


if __name__ == "__main__":
    for n in sorted(dir()):
        if n.startswith("test_"):
            print(n, globals()[n]() or "PASS")
