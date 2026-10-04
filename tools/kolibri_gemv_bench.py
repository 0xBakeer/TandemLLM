"""One-row NVFP4 / FP8 projections at Kolibri-1's decode shapes: the served kernels against split-K
variants. Weights rotate through enough copies to stay out of L2 (cold, as in a
decode step), timed in a CUDA graph.

    python tools/kolibri_gemv_bench.py [out.json]
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri.kernels import NVFP4Linear, FP8Linear, _nv_tile, _dot, DOT32  # noqa: E402

COPIES = 10


@triton.jit
def _nv_splitk(X, Wc, Ws, P, N, K, KS, swn, ssn, BN: tl.constexpr, BK: tl.constexpr,
               DOT: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    nm = rn < N
    k_lo = ps * KS
    if DOT:
        rm = tl.arange(0, 16)
        acc = tl.zeros((16, BN), dtype=tl.float32)
        for k0 in range(k_lo, k_lo + KS, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(X + rm[:, None] * 0 + kk[None, :], mask=(rm < 1)[:, None] & (kk < K)[None, :],
                        other=0.0)
            w = _nv_tile(Wc, Ws, 0, rn, nm, k0, K, 0, swn, 0, ssn, BN, BK)
            acc += _dot(x, tl.trans(w), DOT32)
        out = tl.sum(acc, 0)
    else:
        out = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(k_lo, k_lo + KS, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(X + kk, mask=kk < K, other=0.0).to(tl.float32)
            w = _nv_tile(Wc, Ws, 0, rn, nm, k0, K, 0, swn, 0, ssn, BN, BK).to(tl.float32)
            out += tl.sum(w * x[None, :], 1)
    tl.store(P + ps * N + rn, out, mask=nm)


@triton.jit
def _planes_out(P, S2, Y, N, S: tl.constexpr, BN: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    nm = rn < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for s in range(S):
        acc += tl.load(P + s * N + rn, mask=nm, other=0.0)
    tl.store(Y + rn, (acc * tl.load(S2 + rn, mask=nm, other=0.0)).to(tl.bfloat16), mask=nm)


def nv_splitk(lin: NVFP4Linear, x, S, BN, BK, warps, stages, dot=True):
    N, K = lin.N, lin.K
    KS = triton.cdiv(triton.cdiv(K, S), BK) * BK
    P = torch.empty(S, N, dtype=torch.float32, device=x.device)
    _nv_splitk[(triton.cdiv(N, BN), S)](x, lin.w, lin.s, P, N, K, KS, lin.w.stride(0),
                                        lin.s.stride(0), BN=BN, BK=BK, DOT=dot,
                                        num_warps=warps, num_stages=stages)
    y = torch.empty(1, N, dtype=torch.bfloat16, device=x.device)
    _planes_out[(triton.cdiv(N, 512),)](P, lin.s2, y, N, S=S, BN=512)
    return y


def graph_us(fns, reps=4) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in fns:
                f()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / (reps * len(fns)))
    return sorted(ts)[2]


def rand_nv(N, K, dev):
    codes = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(N, K // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn)
    return NVFP4Linear(codes, scale, torch.full((N,), 0.01, device=dev))


def main():
    dev = "cuda"
    res = []
    x = torch.randn(1, 6144, device=dev).to(torch.bfloat16)
    for name, N, K in (("q", 6144, 2560), ("o", 2560, 6144)):
        lins = [rand_nv(N, K, dev) for _ in range(COPIES)]
        xx = x[:, :K].contiguous()
        ref = lins[0].matmul(xx).float()
        base = graph_us([lambda l=l: l.matmul(xx) for l in lins])
        ideal = (lins[0].nbytes) / 273e9 * 1e6
        print(json.dumps({"shape": name, "served_us": base, "ideal_us": ideal}), flush=True)
        best = None
        grid = json.loads(os.environ.get("GEMV_GRID", "null")) or [
            (1, 2, 3, 4), (32, 64, 128), (128, 256), (4, 8), (3,), (True, False)]
        for S, BN, BK, warps, stages, dot in itertools.product(*grid):
            if K % S:
                continue
            try:
                got = nv_splitk(lins[0], xx, S, BN, BK, warps, stages, dot).float()
                err = ((got - ref).abs().max() / ref.abs().max()).item()
                t = graph_us([lambda l=l: nv_splitk(l, xx, S, BN, BK, warps, stages, dot) for l in lins])
            except Exception as e:                                   # noqa: BLE001
                continue
            row = {"shape": name, "S": S, "BN": BN, "BK": BK, "warps": warps, "stages": stages,
                   "dot": dot, "us": t, "rel_err": err}
            res.append(row)
            if best is None or t < best["us"]:
                best = row
        print(json.dumps({"best": best}), flush=True)
        top = sorted([r for r in res if r["shape"] == name], key=lambda r: r["us"])[:6]
        for r in top:
            print(json.dumps(r), flush=True)
    if len(sys.argv) > 1:
        json.dump(res, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
