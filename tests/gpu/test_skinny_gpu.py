"""SPD-20 on the board: the skinny NVFP4 kernel against the exact product and against v2.

Run on the box under the lock (it needs CUDA): every projection shape the engine and the drafter
issue, every row count 1..16 plus 17, 24 and 32, both the served tile and the others the sweep
knows; row independence for EVERY row (not only row 0); odd tails in N and in the K split; zero
weights and a NaN activation that must stay in its own row; grouped weights with a scale per
column; and the flag-off path byte-identical to v2.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools import nvfp4_skinny as SK  # noqa: E402
from tools.nvfp4_linear import NVFP4Block, quantize_to_nvfp4, nvfp4_matmul  # noqa: E402
from tools.nvfp4_linear_v2 import NVFP4Group, nvfp4_matmul_v2  # noqa: E402

G = torch.Generator(device="cuda").manual_seed(0)
ROWS = list(range(1, 17)) + [17, 24, 32]
TILES = [{}, {"nt": 4, "wk": 8, "pf": 1}, {"nt": 8, "wk": 4, "pf": 0}, {"nt": 2, "wk": 16, "pf": 2},
         {"nt": 4, "wk": 8, "pf": 2, "minb": 2}, {"nt": 16, "wk": 1, "pf": 0},
         # SPD-33: the 8-row N tile, two CTAs an SM, and the interleaved K split
         {"nt": 1, "wk": 16, "pf": 2}, {"nt": 1, "wk": 16, "pf": 2, "minb": 2},
         {"nt": 2, "wk": 16, "pf": 2, "il": 1}, {"nt": 1, "wk": 8, "pf": 1, "il": 1}]


def _w(N, K, scale=0.02):
    return quantize_to_nvfp4(torch.randn(N, K, device="cuda", generator=G) * scale)


def _x(M, K):
    return torch.randn(M, K, device="cuda", generator=G).to(torch.bfloat16)


def _check(w, x, tiles=TILES):
    """max |skinny - exact| within twice v2's own distance; every row equal to itself alone."""
    wd = SK._exact(w)
    exact = x.float().to(torch.float16).float() @ wd.T
    d_v2 = (nvfp4_matmul_v2(x, w).float() - exact).abs().max().item()
    worst = 0.0
    for t in tiles:
        y = SK.nvfp4_matmul_skinny(x, w, **t)
        d = (y.float() - exact).abs().max().item()
        assert d <= 2 * d_v2 + 1e-6, (w.N, w.K, x.shape[0], t, d, d_v2)
        for r in range(x.shape[0]):
            alone = SK.nvfp4_matmul_skinny(x[r:r + 1].clone(), w, **t)
            assert torch.equal(alone[0], y[r]), (w.N, w.K, x.shape[0], t, r)
        worst = max(worst, d / max(d_v2, 1e-12))
    return worst


def test_every_model_shape_every_row_count():
    shapes = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
              (12288, 5120), (1024, 5120), (4096, 5120), (5120, 4096), (5120, 25600)]
    worst, n = 0.0, 0
    for (N, K) in shapes:
        w = _w(N, K)
        x = _x(32, K)
        for M in ROWS:
            worst = max(worst, _check(w, x[:M], tiles=[{}]))
            n += 1
        del w
    return f"{len(shapes)} shapes x {len(ROWS)} row counts, served tile: worst / v2 = {worst:.2f}, " \
           f"every row == itself alone"


def test_every_tile_on_the_widest_and_the_longest_shape():
    worst = 0.0
    for (N, K) in [(17408, 5120), (5120, 17408)]:
        w, x = _w(N, K), _x(32, K)
        for M in (1, 7, 8, 9, 16, 17, 32):
            worst = max(worst, _check(w, x[:M]))
    return f"{len(TILES)} tiles x 7 row counts x 2 shapes: worst / v2 = {worst:.2f}"


def test_odd_tails():
    out = []
    for (N, K) in [(24, 5120), (1000, 5120), (5128, 5120), (8, 128), (72, 256), (4104, 1152)]:
        w, x = _w(N, K), _x(16, K)
        for t in ({"nt": 4, "wk": 8, "pf": 1}, {"nt": 16, "wk": 4 if K >= 2048 else 1, "pf": 0},
                  {"nt": 2, "wk": 16, "pf": 2}, {"nt": 1, "wk": 16, "pf": 2},
                  {"nt": 1, "wk": 16, "pf": 2, "il": 1}):
            for M in (1, 5, 16):
                _check(w, x[:M], tiles=[t])
        out.append(f"{N}x{K}")
    return "N not a multiple of the tile, one K step, more K-split warps than steps: " + ", ".join(out)


def test_zero_weights_and_a_nan_row():
    N, K = 1024, 5120
    w = _w(N, K)
    w.w[:64] = 0                              # code 0 = +0.0
    x = _x(16, K)
    y = SK.nvfp4_matmul_skinny(x, w)
    assert torch.equal(y[:, :64], torch.zeros_like(y[:, :64])), "zero weights must give exact zeros"
    x[3, 17] = float("nan")
    y2 = SK.nvfp4_matmul_skinny(x, w)
    assert torch.isnan(y2[3]).all(), "a NaN activation must poison its own row"
    others = [r for r in range(16) if r != 3]
    assert torch.equal(y2[others], y[others]), "a NaN leaked into another row"
    return "zero rows exact zero; a NaN stays in its row, the other 15 rows bit-identical"


def test_grouped_weights_with_a_scale_per_column():
    parts = [_w(17408, 5120, 0.02), _w(17408, 5120, 0.05)]
    ref = [SK.nvfp4_matmul_skinny(_x(1, 5120), p) for p in parts]            # warm
    x = _x(16, 5120)
    each = [SK.nvfp4_matmul_skinny(x, p) for p in parts]
    g = NVFP4Group(parts, ["gate", "up"])
    both = SK.nvfp4_matmul_skinny(x, g)
    a, b = both.split(g.sizes, dim=-1)
    assert torch.equal(a, each[0]) and torch.equal(b, each[1]), "group != members"
    for t in ({"nt": 4, "wk": 8, "pf": 1}, {"nt": 2, "wk": 16, "pf": 2}):
        a, b = SK.nvfp4_matmul_skinny(x, g, **t).split(g.sizes, dim=-1)
        assert torch.equal(a, SK.nvfp4_matmul_skinny(x, parts[0], **t))
        assert torch.equal(b, SK.nvfp4_matmul_skinny(x, parts[1], **t))
    del ref
    return "gate+up as one group == each member alone, bit for bit (served tile and two others)"


def test_a_tile_too_big_for_shared_memory_is_refused():
    w, x = _w(1024, 5120), _x(8, 5120)
    try:
        SK.nvfp4_matmul_skinny(x, w, nt=16, wk=16, pf=0)
    except RuntimeError as exc:
        assert "shared memory" in str(exc), exc
        torch.cuda.synchronize()                 # and nothing was launched to fail later
        return "nt16 x wk16 (120 KB of partials) refused before launch with a sentence"
    raise AssertionError("an impossible tile was launched")


def test_the_flag_off_path_is_v2_byte_for_byte():
    w = _w(17408, 5120)
    old = SK.SKINNY
    try:
        SK.SKINNY = False
        for M in ROWS:
            x = _x(M, 5120)
            assert torch.equal(nvfp4_matmul(x, w), nvfp4_matmul_v2(x, w)), M
        SK.SKINNY = True
        x = _x(8, 5120)
        assert torch.equal(nvfp4_matmul(x, w), SK.nvfp4_matmul_skinny(x, w)), "flag on not routed"
    finally:
        SK.SKINNY = old
    return f"flag off: nvfp4_matmul == v2 bit for bit at {len(ROWS)} row counts; flag on routes"



def test_an_8_row_tile_sums_every_row_as_the_16_row_tile_does():
    """SPD-33. The N = 5120 projections (down, GDN out, attention o) run 320 CTAs of 16 weight rows
    at one CTA an SM; eight rows a CTA doubles the grid. The K split -- the summation order -- is
    the warp count's, not the row tile's, so the 8-row tile must give the served tile's bits at
    every row count, with and without the two-CTA register bound."""
    n = 0
    for (N, K) in [(5120, 17408), (5120, 6144), (6144, 5120)]:
        w, x = _w(N, K), _x(32, K)
        for M in ROWS:
            ref = SK.nvfp4_matmul_skinny(x[:M], w, nt=2, wk=16, pf=2)
            for t in ({"nt": 1, "wk": 16, "pf": 2}, {"nt": 1, "wk": 16, "pf": 2, "minb": 2},
                      {"nt": 1, "wk": 16, "pf": 1}):
                if M > 16 and t.get("minb", 1) == 2:
                    continue                       # the two-CTA bound is for up to 16 rows
                assert torch.equal(SK.nvfp4_matmul_skinny(x[:M], w, **t), ref), (N, K, M, t)
                n += 1
        del w
    return f"nt1 == nt2 at wk16, bit for bit: {n} (shape, rows, tile) cases"


def test_the_interleaved_split_is_its_own_fixed_order():
    """The interleaved K split sums in another order than the contiguous one (the lossless gate
    decides it), but its order is still the shape's: the same bits for a row alone and in a block,
    for the 8- and the 16-row tile, and the same bits twice."""
    w, x = _w(5120, 17408), _x(32, 17408)
    for M in (1, 8, 16, 17, 32):
        a = SK.nvfp4_matmul_skinny(x[:M], w, nt=2, wk=16, pf=2, il=1)
        b = SK.nvfp4_matmul_skinny(x[:M], w, nt=1, wk=16, pf=2, il=1)
        assert torch.equal(a, b), M
        assert torch.equal(a, SK.nvfp4_matmul_skinny(x[:M], w, nt=2, wk=16, pf=2, il=1)), M
    contiguous = SK.nvfp4_matmul_skinny(x[:16], w, nt=2, wk=16, pf=2)
    return (f"il1: nt1 == nt2, deterministic, row-independent; vs the contiguous split "
            f"max|d| {(a[:16].float() - contiguous.float()).abs().max().item():.2e}")


def test_programmatic_dependent_launch_changes_no_bit():
    """SPD-30. With QWEN38_SKINNY_PDL the projection is launched to start before the kernel ahead
    of it finishes, and the norms release it early; the loads and the order are the same, so every
    output is the same bits -- alone, after a norm that releases it, inside a captured graph."""
    from tools import norm_kernels as NK
    w = _w(17408, 5120)
    wn = (torch.randn(5120, device="cuda", generator=G) * 0.1).to(torch.bfloat16)
    rows = (1, 8, 16, 17, 32)
    xs = {M: (_x(M, 5120), _x(M, 5120)) for M in rows}
    xg = _x(16, 5120)
    old = (SK.PDL, SK.SKINNY)
    outs = {}
    try:
        SK.SKINNY = True
        for pdl in (False, True):
            SK.PDL = pdl
            got = []
            for M in rows:
                x, res = xs[M]
                got.append(SK.nvfp4_matmul_skinny(x, w))
                h, y = NK.add_rms_norm(res, x, wn, 1e-6)          # releases the projection early
                got.append(SK.nvfp4_matmul_skinny(y, w))
            x = xg
            SK.nvfp4_matmul_skinny(NK.rms_norm(x, wn, 1e-6), w)    # warm before the capture
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(g, stream=s):
                gy = SK.nvfp4_matmul_skinny(NK.rms_norm(x, wn, 1e-6), w)
            g.replay()
            torch.cuda.synchronize()
            got.append(gy.clone())
            outs[pdl] = got
    finally:
        SK.PDL, SK.SKINNY = old
    same = all(torch.equal(a, b) for a, b in zip(outs[False], outs[True]))
    assert same, "PDL changed an output"
    return f"{len(outs[True])} outputs (5 row counts alone and after a releasing norm, one graph): bit-identical"

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
