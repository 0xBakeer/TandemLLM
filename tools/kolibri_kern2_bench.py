"""The lane-accumulator one-row kernels against the earlier ones at Kolibri-1's decode shapes, cold weights (copies
rotate so they come from DRAM), CUDA-graph timing, and each output's error against an fp32 product
of the same weights (the served kernel's error printed beside it).

    python tools/kolibri_kern2_bench.py [nv] [moe] [head]      # all three without arguments
    NV_GRID='[[16,32],[256,512],[4,8],[3],[0,1],[0,1]]'                # BN, BK, warps, stages, mode, asm
    MOE_GRID='[[8,16],[256,512],[4],[3],[0,1]]'                       # gate/up and down each
    HEAD_GRID='[[8,16,32],[256,512],[4,8],[2,3]]'
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri import kernels as KK  # noqa: E402
from tools.kolibri_gemv_bench import graph_us, rand_nv  # noqa: E402

E, H, I, TOPK, V = 385, 2560, 512, 6, 128000


def rel(a, ref):
    return ((a.float() - ref).abs().max() / ref.abs().max()).item()


def grid(name, default):
    return list(itertools.product(*(json.loads(os.environ.get(name, "null")) or default)))


def emit(row):
    print(json.dumps(row), flush=True)


def bench_nv(dev):
    for name, N, K in (("q", 6144, 2560), ("o", 2560, 6144)):
        lins = [rand_nv(N, K, dev) for _ in range(10)]
        for l in lins:
            l.s2 = (torch.rand(N, device=dev) * 0.02 + 0.005)
        x = torch.randn(1, K, device=dev).to(torch.bfloat16)
        ref = x.float() @ lins[0].dense().T
        KK.NV1ROW = False
        served = graph_us([lambda l=l: l.matmul(x) for l in lins])
        emit({"nv": name, "served_us": served, "served_err": rel(lins[0].matmul(x), ref),
              "ideal_us": lins[0].nbytes / 273e3})
        rows = []
        for bn, bk, w, st, mode, asm in grid("NV_GRID", [(8, 16, 32), (256, 512), (2, 4, 8), (2, 3), (0, 1), (0, 1)]):
            if N % bn or K % bk:
                continue
            def run(l, bn=bn, bk=bk, w=w, st=st, mode=mode, asm=asm):
                y = torch.empty(1, N, dtype=torch.bfloat16, device=dev)
                KK._nv_gemv_1row[(N // bn,)](x, l.w, l.s, l.s2, y, N, K, l.w.stride(0), l.s.stride(0),
                                             BN=bn, BK=bk, ASM=asm, MODE=mode, num_warps=w, num_stages=st)
                return y
            try:
                err = rel(run(lins[0]), ref)
                t = graph_us([lambda l=l: run(l) for l in lins])
            except Exception as e:                                   # noqa: BLE001
                emit({"nv": name, "skip": [bn, bk, w, st, mode, asm], "err": repr(e)[:160]})
                continue
            rows.append({"nv": name, "cfg": [bn, bk, w, st, mode, asm], "us": t, "err": err})
        rows.sort(key=lambda r: r["us"])
        for r in rows[:8]:
            emit(r)
        KK.NV1ROW = True


def bank(N, K, dev):
    codes = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(E, N, K // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn)
    return KK.ExpertBank(codes, scale, torch.rand(E, device=dev) * 0.02 + 0.01)


def bench_moe(dev):
    torch.manual_seed(0)
    G, U, D = bank(I, H, dev), bank(I, H, dev), bank(H, I, dev)
    x = torch.randn(1, H, device=dev).to(torch.bfloat16)
    idss = [torch.randperm(384, device=dev)[:TOPK].contiguous() for _ in range(12)]
    rw = torch.rand(TOPK, device=dev)
    ref = KK.moe_experts_torch(x, idss[0][None], rw[None], G, U, D)

    def full(ids, cfg):
        h = torch.empty(TOPK, I, dtype=torch.bfloat16, device=dev)
        p = torch.empty(TOPK, H, dtype=torch.float32, device=dev)
        if cfg is None:
            KK._moe_1row_ke3(x, ids, rw, G, U, D, h, p)
        else:
            KK.moe_1row2(x, ids, rw, G, U, D, h, p, cfg)
        return h, p
    h0, p0 = full(idss[0], None)
    served = graph_us([lambda i=i: full(i, None) for i in idss])
    emit({"moe": "served", "us": served, "err": rel(p0.sum(0, keepdim=True), ref)})
    # gate/up alone, then down alone (down reads the served h)
    gu, dn = [], []
    for bn, bk, w, st, mode in grid("MOE_GRID", [(8, 16, 32), (256, 512), (2, 4, 8), (2, 3), (0, 1)]):
        cfg = ((bn, bk, w, st, mode), (16, 128, 4, 3, 0))
        if I % bn or H % bk:
            continue
        def g1(ids, cfg=cfg):
            h = torch.empty(TOPK, I, dtype=torch.bfloat16, device=dev)
            (bn_, bk_, w_, st_, md_), _ = cfg
            KK._moe_gate_up_1row2[(TOPK, I // bn_)](
                x, ids, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
                G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
                BN=bn_, BK=bk_, ASM=KK.NV_ASM, MODE=md_, num_warps=w_, num_stages=st_)
            return h
        try:
            e = rel(g1(idss[0]), h0.float())
            t = graph_us([lambda i=i: g1(i) for i in idss])
            gu.append({"moe": "gate_up", "cfg": cfg[0], "us": t, "diff_vs_served": e})
        except Exception as ex:                                      # noqa: BLE001
            emit({"moe": "gate_up", "skip": cfg[0], "err": repr(ex)[:160]})
        if H % bn or I % bk:
            continue
        def d1(ids, bn=bn, bk=bk, w=w, st=st, mode=mode):
            p = torch.empty(TOPK, H, dtype=torch.float32, device=dev)
            KK._moe_down_1row2[(TOPK, H // bn)](
                h0, ids, D.codes, D.scale, D.scale_2, rw, p, I, H, h0.stride(0), D.codes.stride(0),
                D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
                BN=bn, BK=bk, ASM=KK.NV_ASM, MODE=mode, num_warps=w, num_stages=st)
            return p
        try:
            e = rel(d1(idss[0]), p0)
            t = graph_us([lambda i=i: d1(i) for i in idss])
            dn.append({"moe": "down", "cfg": [bn, bk, w, st, mode], "us": t, "diff_vs_served": e})
        except Exception as ex:                                      # noqa: BLE001
            emit({"moe": "down", "skip": [bn, bk, w, st, mode], "err": repr(ex)[:160]})
    for rows in (gu, dn):
        rows.sort(key=lambda r: r["us"])
        for r in rows[:6]:
            emit(r)
    if gu and dn:
        cfg = (tuple(gu[0]["cfg"]), tuple(dn[0]["cfg"]))
        h1, p1 = full(idss[0], cfg)
        t = graph_us([lambda i=i: full(i, cfg) for i in idss])
        emit({"moe": "best_pair", "cfg": cfg, "us": t, "err": rel(p1.sum(0, keepdim=True), ref),
              "served_us": served})


def bench_head(dev):
    w = (torch.randn(V, H, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.rand(V, device=dev) * 0.01 + 0.001
    head = KK.E4M3Head(w, s)
    x = torch.randn(1, H, device=dev)
    ref = (x @ w.float().T) * s[None]
    KK.HEAD2 = False
    sv = graph_us([lambda: head.logits(x)], reps=3)
    sv_err = rel(head.logits(x), ref)
    KK.HEAD2 = True
    rows = []
    for bn, bk, wp, st in grid("HEAD_GRID", [(8, 16, 32), (256, 512), (4, 8), (2, 3)]):
        def run(bn=bn, bk=bk, wp=wp, st=st):
            y = torch.empty(1, V, dtype=torch.float32, device=dev)
            KK._head_gemv2[(V // bn,)](x, head.w, head.s, y, V, H, head.w.stride(0), BN=bn, BK=bk,
                                       num_warps=wp, num_stages=st)
            return y
        try:
            err = rel(run(), ref)
            t = graph_us([run], reps=3)
        except Exception as e:                                       # noqa: BLE001
            emit({"head": "skip", "cfg": [bn, bk, wp, st], "err": repr(e)[:160]})
            continue
        rows.append({"cfg": [bn, bk, wp, st], "us": t, "err": err})
    rows.sort(key=lambda r: r["us"])
    emit({"head": [V, H], "served_us": sv, "served_err": sv_err, "ideal_us": V * H / 273e3, "best": rows[:4]})


def bench_rows(dev):
    """Verify rows (the verify twins): the multi-row kernels (ROWS8) against the row loop, us a call."""
    torch.manual_seed(1)
    G, U, D = bank(I, H, dev), bank(I, H, dev), bank(H, I, dev)
    head = KK.E4M3Head((torch.randn(V, H, device=dev) * 0.5).to(torch.float8_e4m3fn),
                       torch.rand(V, device=dev) * 0.01)
    nvs = [rand_nv(6144, 2560, dev) for _ in range(4)] + [rand_nv(2560, 6144, dev) for _ in range(4)]
    f8 = {}
    for N, K in ((1024, 2560), (7168, 2560), (2560, 6144), (2560, 512)):
        f8[(N, K)] = [KK.FP8Linear((torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn),
                                   torch.rand(-(-N // 128), -(-K // 128), device=dev) + 0.5) for _ in range(3)]
    for M in (1, 2, 4, 8, 16):
        row = {"M": M}
        for on in (False, True):
            KK.ROWS8 = 15 if on else 0
            tag = "rows8" if on else "loop"
            xs = torch.randn(M, H, device=dev)
            row[f"head_{tag}"] = graph_us([lambda: head.logits_rows(xs)], reps=2)
            x2 = xs.to(torch.bfloat16)
            x6 = torch.randn(M, 6144, device=dev).to(torch.bfloat16)
            row[f"nv_{tag}"] = graph_us([lambda l=l: l.matmul_rows(x2 if l.K == 2560 else x6) for l in nvs])
            for (N, K), ls in f8.items():
                xk = torch.randn(M, 2 * K if K == 512 else K, device=dev).to(torch.bfloat16)
                row[f"fp8_{N}x{K}_{tag}"] = graph_us([lambda l=l: l.matmul_rows(xk, swiglu=K == 512) for l in ls])
            ids = torch.stack([torch.randperm(384, device=dev)[:TOPK] for _ in range(M)])
            w = torch.rand(M, TOPK, device=dev)
            row[f"moe_{tag}"] = graph_us([lambda: KK.moe_experts_rows(x2, ids, w, G, U, D)])
        KK.ROWS8 = 2
        emit({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})


def main():
    dev = "cuda"
    torch.manual_seed(0)
    which = sys.argv[1:] or ["nv", "moe", "head"]
    with torch.inference_mode():
        for w in which:
            {"nv": bench_nv, "moe": bench_moe, "head": bench_head, "rows": bench_rows}[w](dev)


if __name__ == "__main__":
    main()
