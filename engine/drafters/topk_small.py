"""A small-k top-k over long rows in two passes (SPD-201, QWEN38_DF2_TOPK_FAST): the block drafter's candidates.

`torch.topk` of k = 16 over a [P, 248320] head output runs a multi-block radix select, four digit passes and a
gather, ~1.3 ms for the 56 rows of a packed draft of eight blocks. Here each program takes one row's chunk of
CHUNK columns and pulls its k largest by k max-reductions, and a second pass merges a row's chunk winners the same
way. The values are the top-k values; among exactly equal values the lowest column wins (the order of ties is
the one thing `torch.topk` leaves unspecified). Sorted descending.
"""

from __future__ import annotations

import os

import torch

ON = os.environ.get("QWEN38_DF2_TOPK_FAST", "0") == "1"
CHUNK = 4096

_K = None


def _kernels():
    global _K
    if _K is not None:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def _chunk_topk(X, sx, V, OV, OI, NCH, K: tl.constexpr, C: tl.constexpr):
        r = tl.program_id(0)
        ch = tl.program_id(1)
        cols = ch * C + tl.arange(0, C)
        x = tl.load(X + r * sx + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        big = 2147483647
        for j in range(K):
            m = tl.max(x, axis=0)
            i = tl.min(tl.where(x == m, cols, big), axis=0)
            tl.store(OV + (r * NCH + ch) * K + j, m)
            tl.store(OI + (r * NCH + ch) * K + j, i)
            x = tl.where(cols == i, float("-inf"), x)

    @triton.jit
    def _merge_topk(PV, PI, n, OV, OI, K: tl.constexpr, NP: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, NP)
        ok = o < n
        v = tl.load(PV + r * n + o, mask=ok, other=float("-inf"))
        ids = tl.load(PI + r * n + o, mask=ok, other=2147483647)
        big = 2147483647
        for j in range(K):
            m = tl.max(v, axis=0)
            i = tl.min(tl.where(v == m, ids, big), axis=0)
            tl.store(OV + r * K + j, m)
            tl.store(OI + r * K + j, i)
            v = tl.where(ids == i, float("-inf"), v)

    _K = (_chunk_topk, _merge_topk)
    return _K


def topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(values fp32 [P, k], ids int64 [P, k]) of a [P, V] CUDA tensor, sorted descending, ties to the lowest id."""
    import triton
    chunk_k, merge_k = _kernels()
    P, V = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    nch = triton.cdiv(V, CHUNK)
    pv = torch.empty(P, nch, k, dtype=torch.float32, device=x.device)
    pi = torch.empty(P, nch, k, dtype=torch.int32, device=x.device)
    chunk_k[(P, nch)](x, x.stride(0), V, pv, pi, nch, K=k, C=CHUNK, num_warps=8)
    ov = torch.empty(P, k, dtype=torch.float32, device=x.device)
    oi = torch.empty(P, k, dtype=torch.int32, device=x.device)
    n = nch * k
    merge_k[(P,)](pv, pi, n, ov, oi, K=k, NP=triton.next_power_of_2(n), num_warps=4)
    return ov, oi.long()


def check(seed: int = 0) -> list[str]:
    """Against `torch.topk` on head-shaped rows: the same values everywhere; the same ids where no two kept values
    tie (a GPU)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for P, V, k, tie in ((56, 248320, 16, False), (64, 248320, 16, True), (7, 248320, 8, False), (3, 5000, 16, True)):
        x = (torch.randn(P, V, device="cuda", generator=g) * 3).to(torch.bfloat16)
        if tie:
            x = (x.float() * 2).round().to(torch.bfloat16)
        rv, ri = torch.topk(x, k, dim=-1)
        v, i = topk(x, k)
        assert torch.equal(rv.float(), v), (P, V, k)
        clean = (rv[:, 1:] != rv[:, :-1]).all(-1) & (rv[:, -1] != x.float().topk(k + 1, dim=-1).values[:, -1].to(rv.dtype))
        assert torch.equal(ri[clean], i[clean]), (P, V, k)
        same = torch.equal(torch.sort(ri, -1).values, torch.sort(i, -1).values)
        out.append(f"[{P}, {V}] k {k}{' (ties)' if tie else ''}: values equal; ids equal on {int(clean.sum())}/{P} "
                   f"tie-free rows; id sets {'equal' if same else 'differ at tied boundaries'}")
    return out
