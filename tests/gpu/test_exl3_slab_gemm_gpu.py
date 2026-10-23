"""On the board: `tools/exl3_slab_gemm.slab_rot` (the slab product with rot_out in its epilogue) gives
the library GEMM + rot_out_kernel's bits on every slab shape of the 27B, at row counts from a partial
chunk to a window, with every tiling it may take; and the prefill MLP under EXL3_SLAB_TRITON gives the
bridge's bits. Run on the box under the lock: python tests/gpu/test_exl3_slab_gemm_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402
from tools import exl3_slab_gemm as S  # noqa: E402

SHAPES = ((5120, 16384), (5120, 10240), (5120, 6144), (5120, 12288), (5120, 1024), (17408, 5120),
          (6144, 5120))
ROWS = (513, 700, 1024, 1350, 2048, 4096 + 904)
TILES = ((128, 128, 32, 4, 4), (128, 128, 64, 8, 3), (128, 256, 64, 8, 3))


def test_slab_rot_bits():
    mod = X._module()
    g = torch.Generator(device="cuda").manual_seed(0)
    for K, N in SHAPES:
        wi = (torch.randn(K, N, device="cuda", generator=g) / K ** 0.5).half()
        sv = (torch.rand(N, device="cuda", generator=g) + 0.5).half()
        for M in ROWS:
            xh = torch.randn(M, K, device="cuda", generator=g).half()
            ref = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            mod.exl3_rot_out(torch.mm(xh, wi, out_dtype=torch.float32), sv, ref)
            for t in TILES:
                out = torch.empty_like(ref)
                S.slab_rot(xh, wi, sv, out, tile=t)
                assert torch.equal(out, ref), (K, N, M, t)


def test_mlp_bits():
    g = X.random_block(17408, 5120, 4, seed=11)
    u = X.random_block(17408, 5120, 4, seed=21)
    gu = X.EXL3Block.group([g, u], ["gate", "up"])
    wd = X.random_block(5120, 17408, 5, seed=31)
    for M in (700, 2048):
        gen = torch.Generator(device="cpu").manual_seed(M)
        x = (torch.randn(M, 5120, generator=gen) * 0.7).to(torch.bfloat16).cuda()
        X.SLAB_TRITON = False
        ref = X.exl3_mlp_slab(x, gu, wd)
        X.SLAB_TRITON = True
        try:
            got = X.exl3_mlp_slab(x, gu, wd)
            X.slab_cache_on()
            try:
                got2 = X.exl3_mlp_slab(x, gu, wd)
            finally:
                X.slab_cache_off()
            one = X.exl3_matmul(x, g)                         # a single projection on the slab path
            X.SLAB_TRITON = False
            one_ref = X.exl3_matmul(x, g)
        finally:
            X.SLAB_TRITON = False
        assert ref is not None and torch.equal(got, ref) and torch.equal(got2, ref), M
        assert torch.equal(one, one_ref), M


if __name__ == "__main__":
    test_slab_rot_bits()
    test_mlp_bits()
    print("ok")
