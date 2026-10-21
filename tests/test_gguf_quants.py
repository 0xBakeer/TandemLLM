"""tools/gguf_quants.py: every dequantizer bit-exact against gguf-py's on identical random blocks
(every bit pattern is a valid block; fp16 scale fields are pinned to a sane value so both sides
stay finite), the packed linears' matmul against the twins' fp32 matmul, and their rows invariant
to the row count bit for bit. The packed kernels run on the GPU or under TRITON_INTERPRET=1."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import gguf_quants as Q  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
KERNELS = Q.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))

# type id -> (bytes a block, block size, fp16 fields to pin sane) for the packed formats
PACKS = {2: (18, 32, [(0, 2)]), 3: (20, 32, [(0, 4)]), 6: (22, 32, [(0, 2)]), 7: (24, 32, [(0, 4)]),
         10: (84, 256, [(80, 84)]), 11: (110, 256, [(108, 110)]), 12: (144, 256, [(0, 4)]),
         13: (176, 256, [(0, 4)]), 14: (210, 256, [(208, 210)]), 17: (74, 256, [(0, 2)]),
         18: (98, 256, [(0, 2)]), 19: (50, 256, [(0, 2)]), 20: (18, 32, [(0, 2)]), 21: (110, 256, [(0, 2)]),
         22: (82, 256, [(0, 2)]), 23: (136, 256, [(0, 2)]), 29: (56, 256, [(48, 56)])}
_HALF = np.frombuffer(np.float16(0.5).tobytes(), np.uint8)


def random_blocks(tid: int, n: int, seed: int = 0) -> np.ndarray:
    ts, qk, fix = PACKS[tid]
    rng = np.random.default_rng(seed + tid)
    b = rng.integers(0, 256, size=(n, ts), dtype=np.uint8)
    for s, e in fix:
        b[:, s:e] = np.tile(_HALF, (e - s) // 2) if tid != 29 else np.uint8(0x30)
    return b


def test_dequant_matches_gguf_py():
    try:
        from gguf.constants import GGMLQuantizationType as GT
        from gguf.quants import dequantize
    except Exception:  # noqa: BLE001
        return "skipped: gguf-py missing"
    rng = np.random.default_rng(0)
    ok = 0
    for tid, (qk, ts, _fn) in sorted(Q.TYPES.items()):
        if tid in (0, 1, 24, 25, 26, 27, 28, 30):      # scalar/int types are plain views
            continue
        try:
            gq_t = GT(tid)
        except ValueError:
            continue
        b = rng.integers(0, 256, (8, ts), np.uint8)
        try:
            ref = np.asarray(dequantize(b.copy().reshape(-1), gq_t), np.float32).reshape(-1)
        except NotImplementedError:
            continue                                     # gguf-py cannot do Q8_1/Q8_K; ours can
        mine = Q.dequant(tid, b.reshape(-1).copy(), (8 * qk,)).numpy().reshape(-1)
        assert np.array_equal(np.isnan(ref), np.isnan(mine)), tid
        m = ~np.isnan(ref)
        assert np.array_equal(ref[m], mine[m]), tid
        ok += 1
    assert ok >= 20


def test_packed_linears():
    if not KERNELS:
        return "skipped: no triton/gpu"
    for tid, cls in sorted(Q.PACKED_LINEAR.items()):
        ts, qk, _ = PACKS[tid]
        shapes = [(128, 512), (256, 1024)]
        if qk == 32:
            shapes.append((64, 96))          # K/32 = 3 blocks: exercises the SB-loop tail guard
        for N, K in shapes:
            raw = random_blocks(tid, N * (K // qk)).reshape(-1)
            lin = cls(raw, N, K, device=DEV)
            w32 = lin.dequant().to(DEV)
            for M in (1, 4, 16):
                x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
                ref = torch.nn.functional.linear(x.float(), w32)
                rel = ((lin.matmul(x).float() - ref).norm() / (ref.norm() + 1e-9)).item()
                assert rel < 0.01, (tid, N, K, M, rel)


def test_row_invariance():
    """Every call width M = 1..33 (decode's 1 to the ROWS_MAX+1 verify block) gives a row bitwise
    identical to the M=1 row — the packed linears plus iq2_linear's IQ2/Q8 linears."""
    if not KERNELS:
        return "skipped: no triton/gpu"
    from tools.iq2_linear import IQ2Linear, Q8Linear
    for tid, cls in sorted(Q.PACKED_LINEAR.items()):
        ts, qk, _ = PACKS[tid]
        shapes = [(128, 512)]
        if qk == 32:
            shapes.append((64, 96))          # 3 blocks a row: exercises the K tail
        for N, K in shapes:
            lin = cls(random_blocks(tid, N * (K // qk)).reshape(-1), N, K, device=DEV)
            x = torch.randn(33, K, device=DEV, dtype=torch.bfloat16)
            one = lin.matmul(x[:1])
            for M in range(2, 34):
                assert torch.equal(one, lin.matmul(x[:M])[:1]), (tid, N, K, M)
    # iq2_linear's production linears hold the same contract (IQ2_XXS needs K % 256 == 0)
    rng = np.random.default_rng(1)
    for cls, ts, qk, shapes in ((IQ2Linear, 66, 256, [(128, 512)]), (Q8Linear, 34, 32, [(128, 512), (64, 96)])):
        for N, K in shapes:
            b = rng.integers(0, 256, (N * (K // qk), ts), dtype=np.uint8)
            b[:, 0:2] = _HALF
            lin = cls(b.reshape(-1), N, K, device=DEV)
            x = torch.randn(33, K, device=DEV, dtype=torch.bfloat16)
            one = lin.matmul(x[:1])
            for M in range(2, 34):
                assert torch.equal(one, lin.matmul(x[:M])[:1]), (cls.__name__, N, K, M)


def test_dequant_gpu_close():
    """`dequant_gpu` decodes on-device from the packed tensors (folded scales are fp16), so it is
    within ~1 bf16 ulp of the fp32 `dq_*` twin — the whole weight and a partial row slice."""
    if not (KERNELS and DEV == "cuda"):
        return "skipped: no gpu"
    for tid, cls in sorted(Q.PACKED_LINEAR.items()):
        ts, qk, _ = PACKS[tid]
        N, K = 128, 512
        lin = cls(random_blocks(tid, N * (K // qk)).reshape(-1), N, K, device=DEV)
        for got, ref in ((lin.dequant_gpu(), lin.dequant()), (lin.dequant_gpu(5, 37), lin.dequant()[5:37])):
            err = (got.cpu().float() - ref).abs()
            tol = 2**-7 * ref.abs() + 2**-7 * ref.abs().max() * 2**-8
            assert (err <= tol).all(), (tid, err.max().item())


def test_prefill_path():
    """M > ROWS_MAX + 1 takes the chunked GPU dequant + cuBLAS path; a forced tiny chunk budget
    exercises several N slices with a ragged tail."""
    if not (KERNELS and DEV == "cuda"):
        return "skipped: no gpu"
    for tid, cls in sorted(Q.PACKED_LINEAR.items()):
        ts, qk, _ = PACKS[tid]
        N, K = 128, 512
        lin = cls(random_blocks(tid, N * (K // qk)).reshape(-1), N, K, device=DEV)
        w32 = lin.dequant().to(DEV)
        x = torch.randn(70, K, device=DEV, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x.float(), w32)
        rel = ((lin.matmul(x).float() - ref).norm() / (ref.norm() + 1e-9)).item()
        assert rel < 0.01, (tid, rel)
        old = Q.PREFILL_W_BYTES
        try:
            Q.PREFILL_W_BYTES = 2 * K * 30        # 30 rows a slice: 4 full + a ragged 8
            rel = ((lin.matmul(x).float() - ref).norm() / (ref.norm() + 1e-9)).item()
        finally:
            Q.PREFILL_W_BYTES = old
        assert rel < 0.01, (tid, rel)


def test_twins_match_oracle():
    if not KERNELS:
        return "skipped: no triton/gpu"
    try:
        from gguf.constants import GGMLQuantizationType as GT
        from gguf.quants import dequantize
    except Exception:  # noqa: BLE001
        return "skipped: gguf-py missing"
    for tid, cls in sorted(Q.PACKED_LINEAR.items()):
        ts, qk, _ = PACKS[tid]
        N, K = 64, 512
        raw = random_blocks(tid, N * (K // qk), seed=9).reshape(-1)
        lin = cls(raw, N, K, device=DEV)
        ref = torch.from_numpy(np.asarray(dequantize(raw.copy(), GT(tid)), np.float32)).reshape(N, K)
        assert torch.equal(lin.dequant(), ref), tid
