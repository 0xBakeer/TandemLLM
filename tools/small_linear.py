"""A small bf16 projection with a fixed reduction order: the linear-attention gate inputs `a | b`.

Every GDN layer projects its input onto two 48-wide weights, `in_proj_a` and `in_proj_b`, in bf16.
A library GEMM picks its algorithm from heuristics that depend on the row count and on the workspace
it is given -- which is not the same inside a CUDA-graph capture as outside one, and on GB10 the small-M
pick is a one-warp SM80 kernel at about 30 us a call. On 2026-09-23 (K4) a graphed verify moved chat's
text where the eager verify of the same block did not, and these two GEMMs were the suspects.

SPD-163 (QWEN38_GDN_AB): both in one call over the concatenated [96, K] weight, in two launches.

  1. `_gate_part`, grid (N / 16, KSPLIT, row slabs): one program a 16-output block, a fixed K range
     (split s of KSPLIT) and a 64-row slab. Its K loop runs in ascending BK steps, each a `tl.dot` of a
     [64, BK] bf16 tile by a [BK, 16] one into an fp32 accumulator (tensor cores, `mma.sync`). The
     slab is always 64 rows; rows past M load as zeros. Nothing in the program depends on M but the
     mask, so row r's partial is the same bits for any M and any other rows.
  2. `_gate_sum`: the KSPLIT partials of an output summed in the order s = 0, 1, ..., KSPLIT - 1 in
     fp32, rounded once to bf16 -- `F.linear`'s contract.

The same kernel serves the joint verify (every engine's rows in one call), the lone verify, its graph
and the single-token decode step, so a row's gate inputs are the same bits whichever computed them.
`python tools/small_linear.py` is the board check: rows 1..64 and two slabs against the 64-row call,
permuted and offset rows, the fp32 bound, the instruction in the PTX, and the time against F.linear.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

BM, BN, BK, KSPLIT = 64, 16, 64, 8


if HAVE_TRITON:

    @triton.jit(do_not_specialize=["M"])
    def _gate_part(X, W, P, M, N, K, KS, s_xm, s_wn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pn, s, pm = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        mm, mn = rm < M, rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k_lo = s * KS
        for k0 in range(0, KS, BK):
            rk = k_lo + k0 + tl.arange(0, BK)
            mk = (rk < K) & (rk < k_lo + KS)
            x = tl.load(X + rm[:, None] * s_xm + rk[None, :], mask=mm[:, None] & mk[None, :],
                        other=0.0)
            w = tl.load(W + rn[None, :] * s_wn + rk[:, None], mask=mn[None, :] & mk[:, None],
                        other=0.0)
            acc = tl.dot(x, w, acc)
        # P is [KSPLIT, M, N]: split s of output (m, n) at s * M * N + m * N + n
        tl.store(P + s * M * N + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & mn[None, :])

    @triton.jit(do_not_specialize=["MN"])
    def _gate_sum(P, Y, MN, S: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < MN
        acc = tl.load(P + i, mask=m, other=0.0)
        for s in tl.static_range(1, S):
            acc += tl.load(P + s * MN + i, mask=m, other=0.0)
        tl.store(Y + i, acc.to(tl.bfloat16), mask=m)


def small_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """X [M, K] bf16 @ w [N, K]^T bf16 -> [M, N] bf16, the same bits for a row whatever M and the
    other rows are."""
    M, K = x.shape
    N = w.shape[0]
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return y
    ks = triton.cdiv(triton.cdiv(K, KSPLIT), BK) * BK       # a split's K range, by K alone
    part = torch.empty(KSPLIT, M, N, dtype=torch.float32, device=x.device)
    _gate_part[(triton.cdiv(N, BN), KSPLIT, triton.cdiv(M, BM))](
        x, w, part, M, N, K, ks, x.stride(0), w.stride(0), BM=BM, BN=BN, BK=BK, num_warps=4)
    MN = M * N
    _gate_sum[(triton.cdiv(MN, 512),)](part, y, MN, S=KSPLIT, BLOCK=512, num_warps=4)
    return y


def check(K: int = 5120, N: int = 96, bench: bool = True) -> list[str]:
    out = []
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(160, K, device="cuda", generator=g).to(torch.bfloat16)
    ref = x.float() @ w.float().T
    y = small_linear(x[:64], w)
    d = (y.float() - ref[:64]).abs().max().item()
    lib = torch.nn.functional.linear(x[:64], w)
    dl = (lib.float() - ref[:64]).abs().max().item()
    # the bound: one bf16 rounding of the fp32 sum (half an ulp of |y|) plus the sum's own error
    ulp = (ref[:64].abs() * 2.0 ** -8).clamp_min(2.0 ** -133)
    assert ((y.float() - ref[:64]).abs() <= ulp + 1e-3).all(), "fp32 bound"
    out.append(f"max|y - fp32| {d:.3e} (F.linear {dl:.3e}), within half a bf16 ulp + 1e-3")
    for m in range(1, 65):
        assert torch.equal(small_linear(x[:m], w), y[:m]), f"rows 1..{m}"
    out.append("rows 1..64: every prefix gives the 64-row call's bits")
    yy = small_linear(x, w)                                 # 160 rows: three slabs
    assert torch.equal(yy[:64], y), "slab 0"
    for lo, hi in ((64, 128), (128, 160), (100, 133)):
        assert torch.equal(small_linear(x[lo:hi], w), yy[lo:hi]), f"rows {lo}..{hi}"
    out.append("160 rows (three slabs): every row is its bits in any slab, at any offset")
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(small_linear(x[:64][perm], w), y[perm]), "permutation"
    for r in (0, 7, 31, 63):
        assert torch.equal(small_linear(x[r:r + 1], w), y[r:r + 1]), f"row {r} alone"
    other = torch.randn(64, K, device="cuda", generator=g).to(torch.bfloat16)
    mix = torch.cat([x[:5], other[:20], x[5:9]])
    ym = small_linear(mix, w)
    assert torch.equal(ym[:5], y[:5]) and torch.equal(ym[25:29], y[5:9]), "neighbours"
    out.append("permuted rows, a row alone and a row among other requests' rows: the same bits")
    part = torch.empty(KSPLIT, 4, N, dtype=torch.float32, device="cuda")
    ks = triton.cdiv(triton.cdiv(K, KSPLIT), BK) * BK
    h = _gate_part[(triton.cdiv(N, BN), KSPLIT, 1)](x, w, part, 4, N, K, ks, x.stride(0), w.stride(0),
                                                    BM=BM, BN=BN, BK=BK, num_warps=4)
    ptx = getattr(h, "asm", {}).get("ptx", "")
    mma = sorted({ln.split()[0] for ln in ptx.splitlines() if "mma.sync" in ln})
    out.append(f"PTX: {', '.join(mma) or 'mma.sync not found (no PTX to read)'}")
    if bench:
        out.extend(_bench(w, K))
    return out


def _bench(w: torch.Tensor, K: int) -> list[str]:
    import time
    out = []
    a, b = w[:w.shape[0] // 2], w[w.shape[0] // 2:]
    for m in (1, 4, 8, 16, 32, 64):
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        rows = []
        for name, fn in (("gate", lambda: small_linear(x, w)),
                         ("2x F.linear", lambda: (torch.nn.functional.linear(x, a),
                                                  torch.nn.functional.linear(x, b)))):
            for _ in range(20):
                fn()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(50):
                    fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                gr.replay()
            torch.cuda.synchronize()
            rows.append(f"{name} {(time.perf_counter() - t0) / 500 * 1e6:.1f} us")
        out.append(f"M={m:2d} (graphed): " + ", ".join(rows))
    return out


if __name__ == "__main__":
    print("\n".join("small_linear: " + s for s in check()))
