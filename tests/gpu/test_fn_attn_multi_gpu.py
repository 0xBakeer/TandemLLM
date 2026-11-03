"""On the board (SPD-249): a joint pass's attention for several lanes in one launch each.

`fn_kernels.attn_prep_multi` and `fn_seq_kernels.attn_rows_multi` over the rows of 3 lanes (each its own caches,
positions, slots, control, a dense and a sparse lane, different row counts) against `attn_prep` / `attn_rows` called a
lane at a time: the queries, the cache rows written, the indexer queries and the attention outputs bit for bit.

Run on the box: python tests/gpu/test_fn_attn_multi_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import fn_kernels as K  # noqa: E402
from tools import fn_seq_kernels as SK  # noqa: E402

DEV = "cuda"
HQ, HK, D, ROT, CAP = 16, 2, 256, 64, 4096
IHD, NHI = 128, 4
L0 = [700, 1500, 300]                   # each lane's committed length (its block starts there)
MS = [3, 5, 2]                          # each lane's rows (a chain)


def _lane(g, j, L, M, sparse):
    kc = torch.randn(HK, CAP, D, generator=g, device=DEV).bfloat16()
    vc = torch.randn(HK, CAP, D, generator=g, device=DEV).bfloat16()
    ikc = torch.randn(CAP, IHD, generator=g, device=DEV).bfloat16()
    pos = torch.arange(L, L + M, device=DEV, dtype=torch.int64)
    slot = pos.clone()
    ctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    anc = torch.zeros(M, SK.ANC, dtype=torch.int32, device=DEV)
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    for r in range(M):
        anc[r, :r + 1] = torch.arange(L, L + r + 1, dtype=torch.int32)
        if sparse:
            keep = torch.sort(torch.randperm(L, generator=torch.Generator().manual_seed(17 * j + r))[:L // 3]).values
            vis = torch.cat([keep.int(), torch.arange(L, L + r + 1, dtype=torch.int32)])
            sel[r, :vis.numel()] = vis.to(DEV)
            ctl[r] = torch.tensor([vis.numel(), L, 0, 0], dtype=torch.int32)
        else:
            ctl[r] = torch.tensor([L + r + 1, L, 1, 0], dtype=torch.int32)
    iq = torch.zeros(M, NHI, IHD, dtype=torch.bfloat16, device=DEV) if sparse else None
    return {"kc": kc, "vc": vc, "ikc": ikc, "pos": pos, "slot": slot, "ctl": ctl, "anc": anc, "sel": sel, "iq": iq}


def test_attn_multi_equals_a_lane_at_a_time():
    g = torch.Generator(device=DEV).manual_seed(5)
    lanes = [_lane(g, j, L, M, sparse=(j == 1)) for j, (L, M) in enumerate(zip(L0, MS))]
    Mt = sum(MS)
    qkv = torch.randn(Mt, HQ * 2 * D + 2 * HK * D, generator=g, device=DEV).bfloat16()
    iqk = torch.randn(Mt, NHI * IHD + IHD, generator=g, device=DEV).bfloat16()
    qw = torch.randn(D, generator=g, device=DEV).bfloat16() * 0.1
    kw = torch.randn(D, generator=g, device=DEV).bfloat16() * 0.1
    qln = torch.randn(IHD, generator=g, device=DEV).bfloat16() * 0.1
    ang = torch.arange(CAP, device=DEV).float()[:, None] * torch.linspace(0.001, 0.5, ROT, device=DEV)[None, :]
    cos, sin = ang.cos().bfloat16(), ang.sin().bfloat16()
    lone = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()} for x in lanes]
    # a lane at a time (the path today)
    qs, outs, r0 = [], [], 0
    for x, M in zip(lone, MS):
        q = K.attn_prep(qkv[r0:r0 + M], iqk[r0:r0 + M], x["pos"], qw, kw, cos, sin, x["kc"], x["vc"], x["ikc"], HQ, HK,
                        NHI * IHD, 1e-6, slot=x["slot"], qln=qln if x["iq"] is not None else None, iq=x["iq"])
        qs.append(q)
        outs.append(SK.attn_rows(q, x["kc"], x["vc"], x["ctl"], x["anc"], x["sel"]))
        r0 += M
    # every lane in one launch each
    ap, ar, lane_of, r0 = [], [], [], 0
    for j, (x, M) in enumerate(zip(lanes, MS)):
        ap.append(K.attn_prep_entry(r0, M, x["kc"], x["vc"], x["ikc"], x["pos"], x["slot"], x["iq"]))
        ar.append(SK.attn_rows_entry(r0, x["kc"], x["vc"], x["ctl"], x["anc"], x["sel"]))
        lane_of += [j] * M
        r0 += M
    ap = torch.tensor(ap, dtype=torch.int64, device=DEV)
    ar = torch.tensor(ar, dtype=torch.int64, device=DEV)
    lane = torch.tensor(lane_of, dtype=torch.int32, device=DEV)
    q = K.attn_prep_multi(qkv, iqk, ap, qw, kw, cos, sin, HQ, HK, NHI * IHD, IHD, NHI, 1e-6, qln=qln)
    kc0 = lanes[0]["kc"]
    o = SK.attn_rows_multi(q, ar, lane, (kc0.stride(0), kc0.stride(1), kc0.stride(0), kc0.stride(1)), HK)
    torch.cuda.synchronize()
    assert torch.equal(q, torch.cat(qs)), "queries"
    for x, y in zip(lanes, lone):
        for k in ("kc", "vc", "ikc"):
            assert torch.equal(x[k], y[k]), k
        if x["iq"] is not None:
            assert torch.equal(x["iq"], y["iq"]), "indexer queries"
    assert torch.equal(o, torch.cat(outs)), "attention outputs"
    y1 = K.gate_mul(o, qkv, D)
    y0 = torch.cat([K.gate_mul(oo, qkv[a:a + M], D) for oo, a, M in zip(outs, [0, MS[0], MS[0] + MS[1]], MS)])
    assert torch.equal(y1, y0), "gate"
    return f"{len(MS)} lanes, {Mt} rows (dense and sparse): queries, caches, indexer queries, outputs and gate bit-equal"


if __name__ == "__main__":
    print(" ", test_attn_multi_equals_a_lane_at_a_time())
