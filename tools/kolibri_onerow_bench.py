"""One-row (decode) MoE experts and FP8 projections: the served kernels against the one-row kernels,
cold weights (rotating experts / copies), CUDA-graph timing, and the max relative
difference of the outputs.

    python tools/kolibri_onerow_bench.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri import kernels as KK  # noqa: E402
from tools.kolibri_gemv_bench import graph_us  # noqa: E402

E, H, I, TOPK = 385, 2560, 512, 6


def bank(N, K, dev):
    codes = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(E, N, K // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn)
    return KK.ExpertBank(codes, scale, torch.full((E,), 0.02, device=dev))


def moe1(x, ids, w, G, U, D, bn, bk, bnd, bkd, warps, stages):
    k = ids.shape[1]
    h = torch.empty(k, I, dtype=torch.bfloat16, device=x.device)
    KK._moe_gate_up_1row[(k, triton.cdiv(I, bn))](
        x, ids, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
        G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
        BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    p = torch.empty(k, H, dtype=torch.float32, device=x.device)
    KK._moe_down_1row[(k, triton.cdiv(H, bnd))](
        h, ids, D.codes, D.scale, D.scale_2, w, p, I, H, h.stride(0), D.codes.stride(0),
        D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
        BN=bnd, BK=bkd, num_warps=warps, num_stages=stages)
    y = torch.empty(1, H, dtype=torch.float32, device=x.device)
    KK._moe_combine_kernel[(1, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), p, 0,
                                                    BN=256, EXTRA=False, num_warps=4)
    return y


def fp8_1row(lin, x, bn, bk, warps, stages):
    y = torch.empty(1, lin.N, dtype=torch.bfloat16, device=x.device)
    KK._fp8_gemv_1row[(lin.N // bn,)](x, lin.w, lin.s, y, lin.N, lin.K, lin.w.stride(0),
                                     lin.s.stride(0), BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return y


def main():
    dev = "cuda"
    torch.manual_seed(0)
    G, U, D = bank(I, H, dev), bank(I, H, dev), bank(H, I, dev)
    x = (torch.randn(1, H, device=dev)).to(torch.bfloat16)
    idss = [torch.randperm(384, device=dev)[:TOPK][None] for _ in range(12)]
    w = torch.rand(1, TOPK, device=dev)
    ref = KK.moe_experts(x, idss[0], w, G, U, D)
    served = graph_us([lambda i=i: KK.moe_experts(x, i, w, G, U, D) for i in idss])
    print(json.dumps({"moe_served_us": served}), flush=True)
    rows = []
    grid = json.loads(os.environ.get("MOE_GRID", "null")) or [
        (16, 32, 64), (128, 256, 512), (16, 32, 64), (128, 256, 512), (4,), (2, 3)]
    for bn, bk, bnd, bkd, warps, stages in itertools.product(*grid):
        try:
            got = moe1(x, idss[0], w.reshape(-1), G, U, D, bn, bk, bnd, bkd, warps, stages)
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            t = graph_us([lambda i=i: moe1(x, i, w.reshape(-1), G, U, D, bn, bk, bnd, bkd, warps, stages)
                          for i in idss])
        except Exception as e:                                       # noqa: BLE001
            print("skip", bn, bk, bnd, bkd, repr(e)[:100], flush=True)
            continue
        rows.append({"bn": bn, "bk": bk, "bnd": bnd, "bkd": bkd, "warps": warps, "stages": stages,
                     "us": t, "rel_err": err})
    rows.sort(key=lambda r: r["us"])
    for r in rows[:6]:
        print(json.dumps(r), flush=True)
    if os.environ.get("NO_FP8"):
        return
    # FP8 one-row projections at the served shapes
    for N, K in ((1024, 2560), (2560, 512), (7168, 2560), (2560, 6144)):
        lins = []
        for _ in range(max(4, int(60e6 // (N * K)))):
            wq = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
            sc = torch.rand(-(-N // 128), -(-K // 128), device=dev) + 0.5
            lins.append(KK.FP8Linear(wq, sc))
        xx = torch.randn(1, K, device=dev).to(torch.bfloat16)
        r0 = lins[0].matmul(xx).float()
        sv = graph_us([lambda l=l: l.matmul(xx) for l in lins])
        best = None
        for bn, bk, warps, stages in itertools.product((4, 8, 16, 32), (128, 256, 512), (2, 4, 8), (2, 3)):
            if K % bk or N % bn:
                continue
            try:
                got = fp8_1row(lins[0], xx, bn, bk, warps, stages).float()
                err = ((got - r0).abs().max() / r0.abs().max()).item()
                t = graph_us([lambda l=l: fp8_1row(l, xx, bn, bk, warps, stages) for l in lins])
            except Exception as e:                                   # noqa: BLE001
                continue
            if best is None or t < best["us"]:
                best = {"bn": bn, "bk": bk, "warps": warps, "stages": stages, "us": t, "rel_err": err}
        print(json.dumps({"fp8": [N, K], "served_us": sv, "ideal_us": N * K / 273e3, "best": best}),
              flush=True)


if __name__ == "__main__":
    main()
