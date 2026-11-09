"""Focused Q2_K/Q3_K packed-GEMM coverage for hoisted decode fields."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import gguf_quants as Q  # noqa: E402
if Q.HAVE_TRITON:
    from triton.runtime.errors import InterpreterError  # noqa: E402
else:
    InterpreterError = RuntimeError

DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
KERNELS = Q.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))
_HALF = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
_SIZES = {2: 84, 3: 110}
_CLASSES = {2: Q.Q2KLinear, 3: Q.Q3KLinear}
_DECODERS = {2: Q.dq_q2_k, 3: Q.dq_q3_k}
_KINDS = ("ff", "alternating", "single-bit", "zero")


def _kernel_call(lin, x: torch.Tensor, f32: bool) -> torch.Tensor:
    try:
        return lin.matmul(x, f32=f32)
    except InterpreterError as exc:
        if os.environ.get("TRITON_INTERPRET") == "1" and "get_bf16" in str(exc):
            raise unittest.SkipTest("Triton interpreter lacks bf16 scalar support") from exc
        raise


def _block(tid: int, kind: str, salt: int) -> np.ndarray:
    size = _SIZES[tid]
    b = np.zeros(size, dtype=np.uint8)
    if kind == "ff":
        b.fill(0xFF)
    elif kind == "alternating":
        b[:] = np.where(np.arange(size) & 1, 0x55, 0xAA)
    elif kind == "single-bit":
        if tid == 2:
            bit = salt % 512
            b[16 + bit // 8] = np.uint8(1 << (bit % 8))
        else:
            bit = salt % 512
            b[32 + bit // 8] = np.uint8(1 << (bit % 8))
    elif kind != "zero":
        raise ValueError(kind)
    else:
        b.fill(0)
    if tid == 2:
        b[80:82] = _HALF
        b[82:84] = _HALF
    else:
        b[108:110] = _HALF
    return b


def _structured(tid: int, n: int, nb: int, seed: int) -> np.ndarray:
    blocks = np.empty((n, nb, _SIZES[tid]), dtype=np.uint8)
    for row in range(n):
        for kb in range(nb):
            kind = _KINDS[(seed + row + 3 * kb) % len(_KINDS)]
            blocks[row, kb] = _block(tid, kind, seed + row * nb + kb)
    return blocks


def _reference(raw: np.ndarray, tid: int, x: torch.Tensor) -> torch.Tensor:
    n, nb, size = raw.shape
    decoded = _DECODERS[tid](raw.reshape(n * nb, size)).reshape(n, nb * 256)
    weights = torch.from_numpy(np.ascontiguousarray(decoded))
    return x.to(torch.bfloat16).float() @ weights.T


def _assert_reference(got: torch.Tensor, ref: torch.Tensor, tag: tuple) -> None:
    got = got.cpu().float()
    err = (got - ref).abs()
    assert torch.isfinite(got).all(), tag
    assert torch.isfinite(ref).all(), tag
    assert err.max().item() <= 0.03 * max(1.0, ref.abs().max().item()) + 0.5, (
        tag,
        err.max().item(),
        ref.abs().max().item(),
    )
    assert (err.norm() / (ref.norm() + 1e-6)).item() < 0.03, tag


def test_q23k_numpy_twin_structured_shapes():
    for tid in (2, 3):
        for K in (256, 512, 4096):
            raw = _structured(tid, 35, K // 256, seed=tid + K)
            x = torch.randn(3, K, generator=torch.Generator().manual_seed(tid * 10000 + K),
                            dtype=torch.float32)
            ref = _reference(raw, tid, x)
            assert ref.shape == (3, 35)
            assert torch.isfinite(ref).all(), (tid, K)


def test_q23k_kernel_structured_reference():
    if not KERNELS:
        raise unittest.SkipTest("no triton/gpu")
    for tid in (2, 3):
        for K in (256, 512, 4096):
            raw = _structured(tid, 35, K // 256, seed=tid + K)
            x_cpu = torch.randn(3, K, generator=torch.Generator().manual_seed(tid * 10000 + K),
                                dtype=torch.float32).to(torch.bfloat16)
            lin = _CLASSES[tid](raw.reshape(-1), 35, K, device=DEV)
            ref = _reference(raw, tid, x_cpu)
            for f32 in (False, True):
                got = _kernel_call(lin, x_cpu.to(DEV), f32)
                assert got.dtype == (torch.float32 if f32 else torch.bfloat16), (tid, K, f32)
                _assert_reference(got, ref, (tid, K, f32))


def test_q23k_kernel_rows_are_bitwise_invariant():
    if not KERNELS:
        raise unittest.SkipTest("no triton/gpu")
    K = 512
    for tid in (2, 3):
        raw = _structured(tid, 35, K // 256, seed=tid + 91)
        lin = _CLASSES[tid](raw.reshape(-1), 35, K, device=DEV)
        x = torch.randn(33, K, generator=torch.Generator().manual_seed(tid + 700),
                        dtype=torch.float32).to(torch.bfloat16).to(DEV)
        for f32 in (False, True):
            solo = torch.cat([_kernel_call(lin, x[row:row + 1].clone(), f32).cpu()
                              for row in range(33)])
            for M in range(1, 34):
                got = _kernel_call(lin, x[:M], f32).cpu()
                assert torch.equal(got, solo[:M]), (tid, M, f32)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
