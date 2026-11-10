"""On the board (SPD-249 M1): the prefill's int8 mixer GEMM on the tensor cores (QWEN38_FN_PREFILL_MM=tc, tcn).

  * `fn_prefill_kernels.mm_i8` (the native kernel) against the fp64 sum with the exact q s weights, and a row's
    bits the same in any chunk (1 to 1,031 rows, the down and the up shape, a scale a row and a scale a group);
  * `mm_i8_tc` (Triton) the same two facts; a row count is not a compile key;
  * `mix_up_tail` under tcn (the up projection and the mixer's tail in one launch) = `mix_tail(mm_i8(a, w), n, 4)`
    bit for bit, also for n rows with a wider stride; under fp32 and tc the two calls as before;
  * a weight or shape the native kernel does not take goes the fp32 way under tcn, with fp32's bits.
  * every CTA shape (`tile`) of the native kernel and of the fused launch gives the same bits.

Run on the box: python tests/gpu/test_fn_prefill_mm_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

DEV = "cuda"
#: (K, scale group, N): the down|inject shape and the up shape of a mixer, small; a scale a row; odd tiles
SHAPES = ((1280, 128, 324), (320, 64, 1024), (320, None, 256), (64, 64, 4), (128, 64, 1028))
ROWS = (1, 15, 17, 64, 65, 200, 1031)


def _w(N, K, G, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(N, K, device=DEV, generator=g) * (0.02 + torch.rand(N, 1, device=DEV, generator=g))
    return Int8Rows.of(w.half(), G)


def _x(M, K, seed, amp=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, K, device=DEV, generator=g) * amp).to(torch.bfloat16)


def _exact(w):
    q, s = w.q, w.s
    N, K = q.shape
    ng = 1 if s.dim() == 1 else s.shape[1]
    return (q.double().view(N, ng, -1) * s.double().view(N, ng, 1)).view(N, K)


def _mode(mm):
    was, PK.MM = PK.MM, mm
    return was


def _sums(f, name):
    worst = 0.0
    for i, (K, G, N) in enumerate(SHAPES):
        w = _w(N, K, G, 10 + i)
        assert PK.mm_i8_fits(_x(1, K, 0), w), (K, G, N)
        full = None
        for M in ROWS:
            x = _x(M, K, 100 * i + M)
            y = f(x, w)
            assert y.dtype == torch.float32 and tuple(y.shape) == (M, N)
            ref = x.double() @ _exact(w).T
            e = ((y.double() - ref).abs().max() / ref.abs().max()).item()
            assert e < 2e-6, f"{name} K{K} G{G} N{N} M{M}: {e:.2e} of the largest sum"
            worst = max(worst, e)
            full = (x, y)
        x, y = full
        for sl in (slice(0, 1), slice(3, 20), slice(64, 129), slice(1030, 1031)):
            assert torch.equal(f(x[sl].contiguous(), w), y[sl]), f"{name} K{K} G{G} N{N} rows {sl}"
    return f"{len(SHAPES)} shapes x {len(ROWS)} row counts, worst {worst:.1e} of the largest fp64 sum; a row's bits in any chunk"


def test_native_sums():
    return _sums(PK.mm_i8, "mm_i8")


def test_triton_sums():
    return _sums(PK.mm_i8_tc, "mm_i8_tc")


def test_up_and_tail_in_one_launch():
    n_chk = 0
    for i, (K, G, H) in enumerate(((320, 64, 2560), (64, 64, 64), (320, None, 192), (128, 128, 128))):
        w = _w(4 * H, K, G, 40 + i)
        for M in ROWS:
            for amp in (0.3, 40.0):
                a, n = _x(M, K, 7 * M + i, amp), _x(M, 4 * H, 9 * M + i, 2.0)
                ref = PK.mix_tail(PK.mm_i8(a, w), n, 4)
                assert bool(torch.isfinite(ref.float()).all())
                was = _mode("tcn")
                try:
                    x = PK.mix_up_tail(a, w, n, 4)
                    wide = torch.zeros(M, 4 * H + 24, dtype=torch.bfloat16, device=DEV)
                    wide[:, :4 * H] = n
                    xw = PK.mix_up_tail(a, w, wide[:, :4 * H], 4)
                finally:
                    PK.MM = was
                assert x.dtype == torch.bfloat16 and tuple(x.shape) == (M, H)
                assert torch.equal(x, ref), f"K{K} G{G} H{H} M{M} amp {amp}"
                assert torch.equal(xw, ref), f"K{K} G{G} H{H} M{M} amp {amp}: n rows with a wider stride"
                for mm in ("fp32", "tc"):
                    was = _mode(mm)
                    try:
                        assert torch.equal(PK.mix_up_tail(a, w, n, 4), PK.mix_tail(PK.mm_rows(a, w), n, 4)), (mm, K, H, M)
                    finally:
                        PK.MM = was
                n_chk += 4
    return f"{n_chk} cases: tcn = mix_tail(mm_i8), fp32 and tc = their two calls"


def test_every_tile_the_same_bits():
    n_chk = 0
    for i, (K, G, N) in enumerate(SHAPES):
        w = _w(N, K, G, 10 + i)
        for M in (1, 17, 65, 200):
            x = _x(M, K, 100 * i + M)
            y = PK.mm_i8(x, w)
            for tile in (1, 2, 3, 4):
                assert torch.equal(PK.mm_i8(x, w, tile), y), f"K{K} G{G} N{N} M{M} tile {tile}"
                n_chk += 1
    was = _mode("tcn")
    try:
        for i, (K, G, H) in enumerate(((320, 64, 2560), (64, 64, 64), (320, None, 192))):
            w = _w(4 * H, K, G, 40 + i)
            for M in (1, 33, 200):
                a, n = _x(M, K, 7 * M + i), _x(M, 4 * H, 9 * M + i, 2.0)
                x = PK.mix_up_tail(a, w, n, 4)
                for tile in (1, 2, 3):
                    assert torch.equal(PK.mix_up_tail(a, w, n, 4, tile), x), f"K{K} G{G} H{H} M{M} fused tile {tile}"
                    n_chk += 1
    finally:
        PK.MM = was
    return f"{n_chk} cases"


def test_other_shapes_go_the_fp32_way():
    cases = []
    # fp16 weights; a group that is no multiple of 64; fp32 rows; a width the fused tail does not take
    w16 = (torch.randn(96, 320, device=DEV) * 0.05).half()
    odd = _w(64, 96, 32, 71)
    w40 = _w(160, 64, 64, 72)
    x = _x(33, 320, 73)
    for name, xi, wi in (("fp16 weights", x, w16), ("groups of 32", _x(33, 96, 74), odd), ("fp32 rows", x.float(), _w(64, 320, 64, 75))):
        was = _mode("fp32")
        ref = PK.mm_rows(xi, wi)
        PK.MM = "tcn"
        try:
            assert torch.equal(PK.mm_rows(xi, wi), ref), name
        finally:
            PK.MM = was
        cases.append(name)
    a, n = _x(33, 64, 76), _x(33, 160, 77)
    was = _mode("tcn")
    try:
        assert torch.equal(PK.mix_up_tail(a, w40, n, 4), PK.mix_tail(PK.mm_i8(a, w40), n, 4))
    finally:
        PK.MM = was
    return ", ".join(cases) + "; H 40 through the two calls"


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
