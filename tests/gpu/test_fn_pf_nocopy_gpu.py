"""On the board (SPD-249 P4): QWEN38_FN_PF_NOCOPY's paths against the copies they leave out, bit for bit.

  * `fn_prefill_kernels.gated_norm_rows` (z read in place, a column slice of wider rows) equals
    `fn_kernels.gated_norm` on z's copy, both gates, row counts around the kernel's 8 rows a block;
  * `gdn_prefill_kernels.fused_chunk_prefill(view=True)` on the values as a column slice equals the call on their
    copy: the output and the state;
  * `qsa_scores_rows(lean=True)` equals the full fill on a buffer that held other values: a pool of the score
    width, a shorter one, rows of no block;
  * `mix_tail(out=...)` writes the rows of the buffer it is given and no other;
  * `ple.decode_rows(words=True)` equals the bit stream's rows on the device.

Run on the box: python tests/gpu/test_fn_pf_nocopy_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import fn_kernels as K  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

DEV = "cuda"


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def test_gated_norm_rows_is_gated_norm():
    g = torch.Generator(device=DEV).manual_seed(41)
    w = (torch.randn(128, generator=g, device=DEV) * 0.3 + 1).to(torch.bfloat16)
    out = []
    for T, HV, C in ((1, 48, 384), (5, 16, 96), (300, 48, 384), (4096, 48, 12288)):
        rows = (torch.randn(T, C + HV * 128, generator=g, device=DEV) * 2).to(torch.bfloat16)
        z = rows[:, C:]
        o = torch.randn(T * HV, 128, generator=g, device=DEV).to(torch.bfloat16)
        assert not z.is_contiguous() or T == 1
        assert PK.gated_norm_ok(o, z), (T, HV)
        for sig in (True, False):
            got = PK.gated_norm_rows(o, z, w, 1e-6, sig)
            want = K.gated_norm(o, z.reshape(-1, 128), w, 1e-6, sig)
            assert torch.equal(_bits(got), _bits(want)), (T, HV, sig)
        out.append(f"{T}x{HV}")
    assert not PK.gated_norm_ok(o, rows[:, ::2][:, :HV * 128])          # columns not adjacent
    assert not PK.gated_norm_ok(o.float(), z)
    return "bit for bit, sigmoid and silu: " + ", ".join(out)


def test_fused_chunk_prefill_reads_a_value_view():
    from tools.gdn_prefill_kernels import fused_chunk_prefill
    g = torch.Generator(device=DEV).manual_seed(42)
    out = []
    for T, Hk, H, Dk, Dv in ((300, 4, 8, 128, 128), (64, 16, 48, 128, 128), (1000, 16, 48, 128, 128)):
        kd = Hk * Dk
        y = (torch.randn(T, 2 * kd + H * Dv, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
        q = y[:, :kd].view(1, T, Hk, Dk)
        k = y[:, kd:2 * kd].view(1, T, Hk, Dk)
        v = y[:, 2 * kd:].view(1, T, H, Dv)
        assert not v.is_contiguous()
        beta = torch.rand(1, T, H, generator=g, device=DEV).to(torch.bfloat16)
        gg = -torch.rand(1, T, H, generator=g, device=DEV) * 0.05
        S0 = torch.randn(1, H, Dk, Dv, generator=g, device=DEV) * 0.01
        o1, S1 = fused_chunk_prefill(q, k, v, gg, beta, S0, bv=16)
        o2, S2 = fused_chunk_prefill(q, k, v, gg, beta, S0, bv=16, view=True)
        o3, S3 = fused_chunk_prefill(q, k, v.contiguous(), gg, beta, S0, bv=16, view=True)
        assert torch.equal(_bits(o1), _bits(o2)) and torch.equal(_bits(S1), _bits(S2)), (T, H)
        assert torch.equal(_bits(o1), _bits(o3)) and torch.equal(_bits(S1), _bits(S3)), (T, H)
        out.append(f"{T}x{H}")
    return "output and state bit for bit: " + ", ".join(out)


def test_qsa_scores_rows_lean_is_the_full_fill():
    g = torch.Generator(device=DEV).manual_seed(43)
    out = []
    for M, NB, pool_n, block in ((300, 2048, 2058, 128), (1100, 4096, 4096, 1024), (64, 9000, 8000, 1024),
                                 (17, 512, 0, 1024)):
        iq = (torch.randn(M, 4, 128, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
        pool = (torch.randn(pool_n, 128, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
        nblk = torch.randint(0, min(NB, pool_n) + 1, (M,), generator=g, device=DEV)
        want = PK.qsa_scores_rows(iq, pool, nblk, NB, block=block)
        torch.cuda.empty_cache()
        junk = torch.full((M, NB), 7.0, dtype=torch.float32, device=DEV)   # the block the lean buffer may take
        del junk
        got = PK.qsa_scores_rows(iq, pool, nblk, NB, block=block, lean=True)
        assert torch.equal(_bits(got), _bits(want)), (M, NB, pool_n)
        out.append(f"{M}x{NB} pool {pool_n}")
    return "bit for bit: " + ", ".join(out)


def test_mix_tail_writes_the_given_rows():
    g = torch.Generator(device=DEV).manual_seed(44)
    H, M = 2560, 300
    u = torch.randn(M, 4 * H, generator=g, device=DEV)
    n = torch.randn(M, 4 * H, generator=g, device=DEV).to(torch.bfloat16)
    want = PK.mix_tail(u, n, 4)
    buf = torch.full((700, H), 3.0, dtype=torch.bfloat16, device=DEV)
    got = PK.mix_tail(u, n, 4, buf[200:500])
    assert got.data_ptr() == buf[200:500].data_ptr()
    assert torch.equal(_bits(buf[200:500]), _bits(want))
    assert bool((buf[:200] == 3).all()) and bool((buf[500:] == 3).all())
    return f"{M} rows inside a buffer of 700, the others untouched"


def test_decode_rows_words_on_the_device():
    from engine.models.flashnext.ple import ROW_DIM, decode_rows, mul1_codebook
    g = torch.Generator(device=DEV).manual_seed(45)
    cb = mul1_codebook(DEV)
    out = []
    for K_ in (2, 3, 4, 5, 6, 8):
        N = 4096 * 3
        packed = torch.randint(-32768, 32768, (N, 1 + ROW_DIM * K_ // 16), generator=g, device=DEV, dtype=torch.int16)
        packed[:, 0] = (torch.rand(N, generator=g, device=DEV) * 0.1 + 0.01).half().view(torch.int16)
        heads = torch.randint(0, 3, (N,), generator=g, device=DEV)
        bias = (torch.randn(3, ROW_DIM, generator=g, device=DEV) * 0.01).half()
        a = decode_rows(packed, heads, bias, K_, cb)
        b = decode_rows(packed, heads, bias, K_, cb, words=True)
        assert torch.equal(_bits(a), _bits(b)), K_
        out.append(str(K_))
    return "bit for bit at K " + ", ".join(out)


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
