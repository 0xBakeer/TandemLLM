"""The one-row kernels (NVFP4 GEMV, routed experts, head) against fp32 products of the
same weights. On a machine without CUDA they run in Triton's interpreter (run this file on its own:
the interpreter must be on before the kernels are defined); the inline-asm e2m1 decode runs only on
the GPU."""

from __future__ import annotations

import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

try:
    import pytest
except ImportError:                                                  # run as a plain script
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pytest is not None:
    triton = pytest.importorskip("triton")
from engine.kolibri import kernels as KK  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _modes(f):
    """parametrize over MODE 0/1 under pytest; the script runner below loops itself"""
    return pytest.mark.parametrize("mode", [0, 1])(f) if pytest is not None else f


def rel(a, ref):
    return ((a.float() - ref).abs().max() / ref.abs().max()).item()


def rand_nv(N, K, g):
    codes = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g)
    scale = (torch.rand(N, K // 16, generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
    s2 = torch.rand(N, generator=g) * 0.02 + 0.005
    return KK.NVFP4Linear(codes.to(DEV), scale.to(DEV), s2.to(DEV))


def asm_modes():
    return (0, 1) if DEV == "cuda" else (0,)


@_modes
def test_e2m1_decode_every_code(mode):
    """All 256 byte values through the GEMV against `fp4_unpack`: one output row per byte value."""
    for asm in asm_modes():
        N, K = 256, 32
        codes = torch.arange(256, dtype=torch.uint8).repeat_interleave(K // 2).view(N, K // 2)
        lin = KK.NVFP4Linear(codes.to(DEV), torch.ones(N, K // 16).to(torch.float8_e4m3fn).to(DEV),
                             torch.ones(N).to(DEV))
        for k in range(0, K, 2):
            x = torch.zeros(1, K)
            x[0, k] = 1.0
            x[0, k + 1] = 16.0          # every sum a multiple of 0.5 under 128: exact in bf16
            want = x @ KK.fp4_unpack(codes).T
            y = torch.empty(1, N, dtype=torch.bfloat16, device=DEV)
            KK._nv_gemv_1row[(N // 16,)](x.to(torch.bfloat16).to(DEV), lin.w, lin.s, lin.s2, y, N, K,
                                         lin.w.stride(0), lin.s.stride(0), BN=16, BK=32, ASM=asm, MODE=mode)
            assert torch.equal(y.cpu().float(), want.to(torch.bfloat16).float()), (asm, mode, k)


@_modes
def test_nv_gemv_matches_fp32(mode):
    g = torch.Generator().manual_seed(0)
    N, K = 64, 512
    lin = rand_nv(N, K, g)
    x = torch.randn(1, K, generator=g).to(torch.bfloat16).to(DEV)
    ref = x.float() @ lin.dense().T
    for asm in asm_modes():
        y = torch.empty(1, N, dtype=torch.bfloat16, device=DEV)
        KK.nv_gemv_1row(lin, x, y, (16, 256, 4, 2, mode)) if asm == KK.NV_ASM else \
            KK._nv_gemv_1row[(N // 16,)](x, lin.w, lin.s, lin.s2, y, N, K, lin.w.stride(0), lin.s.stride(0),
                                         BN=16, BK=256, ASM=asm, MODE=mode)
        assert rel(y, ref) < 8e-3, (asm, mode)        # bf16 output rounding


@_modes
def test_moe_1row2_matches_torch(mode):
    g = torch.Generator().manual_seed(1)
    E, H, I, k = 9, 256, 64, 3

    def bank(N, K):
        return KK.ExpertBank(torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(DEV),
                             (torch.rand(E, N, K // 16, generator=g) * 2 + 0.5).to(torch.float8_e4m3fn).to(DEV),
                             (torch.rand(E, generator=g) * 0.02 + 0.01).to(DEV))
    G, U, D = bank(I, H), bank(I, H), bank(H, I)
    x = torch.randn(1, H, generator=g).to(torch.bfloat16).to(DEV)
    ids = torch.tensor([[7, 2, 4]], device=DEV)
    w = torch.rand(1, k, generator=g).to(DEV)
    ref = KK.moe_experts_torch(x, ids, w, G, U, D)
    h = torch.empty(k, I, dtype=torch.bfloat16, device=DEV)
    p = torch.empty(k, H, dtype=torch.float32, device=DEV)
    KK.moe_1row2(x, ids.reshape(-1), w.reshape(-1), G, U, D, h, p,
                 ((16, 128, 4, 2, mode), (16, 64, 4, 2, mode)))
    assert rel(p.sum(0, keepdim=True), ref) < 2e-2


def test_head2_matches_fp32():
    g = torch.Generator().manual_seed(3)
    V, K = 512, 512
    w = (torch.randn(V, K, generator=g) * 0.5).to(torch.float8_e4m3fn).to(DEV)
    s = (torch.rand(V, generator=g) * 0.01 + 0.001).to(DEV)
    x = torch.randn(1, K, generator=g).to(DEV)
    ref = (x @ w.float().T) * s[None]
    y = torch.empty(1, V, dtype=torch.float32, device=DEV)
    KK._head_gemv2[(V // 16,)](x, w, s, y, V, K, w.stride(0), BN=16, BK=256)
    assert rel(y, ref) < 1e-5


def test_fused_swiglu_down_bit_equal():
    """The shared expert's down projection with the SwiGLU inside equals swiglu + matmul, bit for bit
    (GPU only: the interpreter has no libdevice exp)."""
    if DEV != "cuda":
        return
    g = torch.Generator().manual_seed(4)
    F_, N = 512, 2560
    lin = KK.FP8Linear((torch.randn(N, F_, generator=g) * 0.5).to(torch.float8_e4m3fn).to(DEV),
                       (torch.rand(N // 128, F_ // 128, generator=g) + 0.5).to(DEV))
    gu = (torch.randn(1, 2 * F_, generator=g) * 2).to(torch.bfloat16).to(DEV)
    a = torch.empty(1, F_, dtype=torch.bfloat16, device=DEV)
    KK._swiglu_kernel[(1, F_ // 512)](gu, a, F_, gu.stride(0), a.stride(0), BF=512)
    want = lin.matmul(a)
    got = lin.swiglu_matmul(gu)
    assert got is not None and torch.equal(got, want)


def test_add_rms2_with_combine_bit_equal():
    """add_rms2 over MoEParts equals the combine kernel followed by add_rms2, bit for bit."""
    g = torch.Generator().manual_seed(5)
    H, k = 2560, 6
    r = torch.randn(1, H, generator=g).to(DEV)
    p = torch.randn(k, H, generator=g).to(DEV)
    ex = torch.randn(1, H, generator=g).to(torch.bfloat16).to(DEV)
    w1 = (torch.rand(H, generator=g) + 0.5).to(torch.bfloat16).to(DEV)
    w2 = (torch.rand(H, generator=g) + 0.5).to(torch.bfloat16).to(DEV)
    for extra in (ex, None):
        parts = KK.MoEParts(p, extra)
        r0, x0 = KK.add_rms2(r, parts.combine(), w1, w2, 1e-6)
        r1, x1 = KK.add_rms2(r, parts, w1, w2, 1e-6)
        assert torch.equal(r0, r1) and torch.equal(x0, x1)


def test_rows_entry_points_bit_equal():
    """matmul_rows / moe_experts_rows / logits_rows: row i equals the one-row call on row i, bit for
    bit (the verify twins rely on it). GPU only (served shapes, slow in the interpreter)."""
    if DEV != "cuda":
        return
    saved = KK.ROWS8
    try:
        for bits in (0, 15):                      # the row loop and every multi-row kernel
            KK.ROWS8 = bits
            for M in (2, 5, 11):
                _rows_bit_equal(M)
    finally:
        KK.ROWS8 = saved


def _rows_bit_equal(M):
    g = torch.Generator().manual_seed(6 + M)
    nv = rand_nv(6144, 2560, g)
    x = torch.randn(M, 2560, generator=g).to(torch.bfloat16).to(DEV)
    yr = nv.matmul_rows(x)
    assert all(torch.equal(yr[i:i + 1], nv.matmul(x[i:i + 1])) for i in range(M))
    f8 = KK.FP8Linear((torch.randn(2560, 512, generator=g) * 0.5).to(torch.float8_e4m3fn).to(DEV),
                      (torch.rand(20, 4, generator=g) + 0.5).to(DEV))
    gu = torch.randn(M, 1024, generator=g).to(torch.bfloat16).to(DEV)
    yr = f8.matmul_rows(gu, swiglu=True)
    assert all(torch.equal(yr[i:i + 1], f8.swiglu_matmul(gu[i:i + 1])) for i in range(M))
    yr = f8.matmul_rows(gu[:, :512])
    assert all(torch.equal(yr[i:i + 1], f8.matmul(gu[i:i + 1, :512])) for i in range(M))
    E, H, I, k = 9, 2560, 512, 3

    def bank(N, K):
        return KK.ExpertBank(torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(DEV),
                             (torch.rand(E, N, K // 16, generator=g) * 2 + 0.5).to(torch.float8_e4m3fn).to(DEV),
                             (torch.rand(E, generator=g) * 0.02 + 0.01).to(DEV))
    G, U, D = bank(I, H), bank(I, H), bank(H, I)
    xm = torch.randn(M, H, generator=g).to(torch.bfloat16).to(DEV)
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(M)]).to(DEV)
    w = torch.rand(M, k, generator=g).to(DEV)
    ex = torch.randn(M, H, generator=g).to(torch.bfloat16).to(DEV)
    yr = KK.moe_experts_rows(xm, ids, w, G, U, D, ex)
    for i in range(M):
        y1 = KK.moe_experts(xm[i:i + 1], ids[i:i + 1], w[i:i + 1], G, U, D, extra=ex[i:i + 1])
        assert torch.equal(yr[i:i + 1], y1), i
    head = KK.E4M3Head((torch.randn(4096, H, generator=g) * 0.5).to(torch.float8_e4m3fn).to(DEV),
                       (torch.rand(4096, generator=g) * 0.01).to(DEV))
    hx = torch.randn(M, H, generator=g).to(DEV)
    yr = head.logits_rows(hx)
    assert all(torch.equal(yr[i:i + 1], head.logits(hx[i:i + 1])) for i in range(M))


if __name__ == "__main__":
    import inspect
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            for mode in ((0, 1) if "mode" in inspect.signature(f).parameters else (None,)):
                f(mode) if mode is not None else f()
                n += 1
                print("pass", name, "" if mode is None else f"mode={mode}", flush=True)
    print(f"{n} passed")
