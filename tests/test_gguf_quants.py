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
         13: (176, 256, [(0, 4)]), 14: (210, 256, [(208, 210)]), 20: (18, 32, [(0, 2)]), 23: (136, 256, [(0, 2)])}
_HALF = np.frombuffer(np.float16(0.5).tobytes(), np.uint8)


def random_blocks(tid: int, n: int, seed: int = 0) -> np.ndarray:
    ts, qk, fix = PACKS[tid]
    rng = np.random.default_rng(seed + tid)
    b = rng.integers(0, 256, size=(n, ts), dtype=np.uint8)
    for s, e in fix:
        b[:, s:e] = np.tile(_HALF, (e - s) // 2)
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
        for N, K in ((128, 512), (256, 1024)):
            raw = random_blocks(tid, N * (K // qk)).reshape(-1)
            lin = cls(raw, N, K, device=DEV)
            w32 = lin.dequant().to(DEV)
            for M in (1, 4, 16):
                x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
                ref = torch.nn.functional.linear(x.float(), w32)
                rel = ((lin.matmul(x).float() - ref).norm() / (ref.norm() + 1e-9)).item()
                assert rel < 0.01, (tid, N, K, M, rel)


def test_row_invariance():
    if not KERNELS:
        return "skipped: no triton/gpu"
    for tid, cls in sorted(Q.PACKED_LINEAR.items()):
        ts, qk, _ = PACKS[tid]
        N, K = 128, 512
        lin = cls(random_blocks(tid, N * (K // qk)).reshape(-1), N, K, device=DEV)
        x = torch.randn(1, K, device=DEV, dtype=torch.bfloat16)
        one = lin.matmul(x)
        many = lin.matmul(x.repeat(8, 1))
        assert torch.equal(one, many[:1]), tid


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
