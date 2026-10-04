"""The fused decode step (`decode_fused`) against the unfused decode path and the fp32 reference.

    python tools/kolibri_dec_check.py [--sweep] [--json out.json]

Per sliding (ring) and full layer, on random data shaped like Kolibri-1 (48 q heads, 4 KV heads of
128, a ring of 640 slots, window 513): the k/v/idx the step writes must be bit-equal to the older
path's writes; the attention output is compared with the fp32 reference (max |d|), next to the older
kernel's own error. Then the time of one layer's attention in a CUDA graph (100 layers captured),
older path against fused; `--sweep` tries the launch shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import kolibri_attn_kernels as K  # noqa: E402
from tools.attn_kernels import decode_attention_dev, reference  # noqa: E402

NQ, NK, D, R, WIN, THETA, EPS = 48, 4, 128, 640, 513, 10000.0, 1e-6
SCALE = 1.0 / math.sqrt(D)


def ring_state(L, dev):
    idx = torch.full((R,), -1, dtype=torch.int32, device=dev)
    n = min(L, R)
    ar = torch.arange(L - n, L, device=dev)
    idx[ar % R] = ar.to(torch.int32)
    k = torch.randn(1, NK, R, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, NK, R, D, device=dev, dtype=torch.bfloat16)
    return k, v, idx


def old_ring(y, qn, kn, p32, k, v, idx):
    """attn_decode's unfused sliding step (engine/kolibri/attn.py with KOLIBRI_DEC_FUSED=0)."""
    posl = p32.long()
    q, kk = K.qk_prep(y, qn, kn, posl, NQ, NK, D, THETA, EPS, True)
    vv = y[:, (NQ + NK) * D:].reshape(1, NK, D)
    sl = posl % R
    idx.index_copy_(0, sl, p32)
    k[0].index_copy_(1, sl, kk.transpose(0, 1))
    v[0].index_copy_(1, sl, vv.transpose(0, 1))
    qlo = (p32 - (WIN - 1)).to(torch.int32)
    one = torch.ones(1, 1, dtype=torch.int8, device=y.device)
    o = K.ring_attention(q.transpose(0, 1)[None], k, v, idx, 0, qlo, one, scale=SCALE, startp=p32)
    return o[0].transpose(0, 1).reshape(1, -1), q


def old_full(y, qn, kn, p32, k, v, max_len):
    posl = p32.long()
    q, kk = K.qk_prep(y, qn, kn, posl, NQ, NK, D, THETA, EPS, False)
    vv = y[:, (NQ + NK) * D:].reshape(1, NK, D)
    k[0].index_copy_(1, posl, kk.transpose(0, 1))
    v[0].index_copy_(1, posl, vv.transpose(0, 1))
    one = torch.ones(1, 1, dtype=torch.int8, device=y.device)
    lenp = torch.cat([p32, p32 + 1])
    o = decode_attention_dev(q.transpose(0, 1)[None], k, v, lenp, one,
                             max(1024, 1 << (max_len - 1).bit_length()), scale=SCALE)
    return o[0].transpose(0, 1).reshape(1, -1), q


def new(y, qn, kn, p32, k, v, idx, **kw):
    return K.decode_fused(y, qn, kn, p32, k, v, idx, nq=NQ, nk=NK, d=D, theta=THETA, eps=EPS,
                          rope=idx is not None, window=WIN, scale=SCALE, **kw)


def ring_ref(q, k, v, idx, pos):
    qlo = torch.tensor([pos - (WIN - 1)], dtype=torch.int32, device=q.device)
    one = torch.ones(1, 1, dtype=torch.bool, device=q.device)
    return K.ring_reference(q.transpose(0, 1)[None], k, v, idx, pos, qlo, one).float()[0, :, 0].reshape(1, -1)


def inputs(dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    y = torch.randn(1, (NQ + 2 * NK) * D, device=dev, dtype=torch.bfloat16, generator=g) * 3
    qn = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    kn = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    return y, qn, kn


def parity(dev="cuda", kw=None) -> list:
    kw = kw or {}
    out = []
    for L in (1, 5, 300, 512, 513, 700, 5000, 100003):
        y, qn, kn = inputs(dev, L)
        k, v, idx = ring_state(L, dev)
        p32 = torch.tensor([L], dtype=torch.int32, device=dev)
        k1, v1, i1 = k.clone(), v.clone(), idx.clone()
        o_old, q = old_ring(y, qn, kn, p32, k1, v1, i1)
        k2, v2, i2 = k.clone(), v.clone(), idx.clone()
        o_new = new(y, qn, kn, p32, k2, v2, i2, **kw)
        ref = ring_ref(q, k1, v1, i1, L)
        out.append({"layer": "ring", "pos": L,
                    "writes_equal": bool(torch.equal(k1, k2) and torch.equal(v1, v2) and torch.equal(i1, i2)),
                    "old_vs_ref": (o_old.float() - ref).abs().max().item(),
                    "new_vs_ref": (o_new.float() - ref).abs().max().item(),
                    "new_vs_old": (o_new.float() - o_old.float()).abs().max().item()})
    max_len = 40000
    for L in (0, 1, 31, 32, 33, 500, 1023, 1024, 8191, 32768, 39990):
        y, qn, kn = inputs(dev, 7 + L)
        k = torch.randn(1, NK, max_len, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, NK, max_len, D, device=dev, dtype=torch.bfloat16)
        p32 = torch.tensor([L], dtype=torch.int32, device=dev)
        k1, v1 = k.clone(), v.clone()
        o_old, q = old_full(y, qn, kn, p32, k1, v1, max_len)
        k2, v2 = k.clone(), v.clone()
        o_new = new(y, qn, kn, p32, k2, v2, None, **kw)
        one = torch.ones(1, 1, dtype=torch.bool, device=dev)
        ref = reference(q.transpose(0, 1)[None], k1[:, :, :L + 1], v1[:, :, :L + 1], L, one,
                        scale=SCALE).float()[0, :, 0].reshape(1, -1)
        out.append({"layer": "full", "pos": L,
                    "writes_equal": bool(torch.equal(k1, k2) and torch.equal(v1, v2)),
                    "old_vs_ref": (o_old.float() - ref).abs().max().item(),
                    "new_vs_ref": (o_new.float() - ref).abs().max().item(),
                    "new_vs_old": (o_new.float() - o_old.float()).abs().max().item()})
        del k, v, k1, v1, k2, v2
    return out


def glue(dev="cuda") -> dict:
    """The glue kernels against the torch ops they replace, bit for bit."""
    from engine.kolibri.kernels import swiglu
    out = {}
    g = torch.Generator(device=dev).manual_seed(3)
    for M, F_ in ((1, 2048), (1, 1536), (7, 2048)):
        gu = (torch.randn(M, 2 * F_, device=dev, generator=g) * 4).to(torch.bfloat16)
        ref = (torch.nn.functional.silu(gu.float()[:, :F_]) * gu.float()[:, F_:]).to(torch.bfloat16)
        got = swiglu(gu, F_)
        out[f"swiglu_{M}x{F_}_ulps_differ"] = int((got != ref).sum())
    return out


def graph_us(fn, n=100) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / n)
    return sorted(ts)[2]


def timing(dev="cuda", kw=None) -> list:
    kw = kw or {}
    y, qn, kn = inputs(dev, 1)
    out = []
    k, v, idx = ring_state(5000, dev)
    p32 = torch.tensor([5000], dtype=torch.int32, device=dev)
    out.append({"layer": "ring", "old_us": graph_us(lambda: old_ring(y, qn, kn, p32, k, v, idx)),
                "new_us": graph_us(lambda: new(y, qn, kn, p32, k, v, idx, **kw))})
    max_len = 262144
    k = torch.zeros(1, NK, max_len, D, device=dev, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    for L in (1024, 8192, 32768, 131072):
        p32 = torch.tensor([L], dtype=torch.int32, device=dev)
        out.append({"layer": "full", "pos": L,
                    "old_us": graph_us(lambda: old_full(y, qn, kn, p32, k, v, max_len), 20),
                    "new_us": graph_us(lambda: new(y, qn, kn, p32, k, v, None, **kw), 20)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {"parity": parity(), "glue": glue(), "timing": timing()}
    print(json.dumps(res["glue"]), flush=True)
    for r in res["parity"] + res["timing"]:
        print(json.dumps(r), flush=True)
    if a.sweep:
        res["sweep"] = []
        for ch in (32, 64, 128):
            for ns in (8, 16, 32, 48):
                for bn in (32, 64):
                    for w in (4, 8):
                        kw = dict(ring_ch=ch, full_ns=ns, bn=bn, num_warps=w)
                        t = timing(kw=kw)
                        row = {"kw": kw, "ring": t[0]["new_us"], "full": [x["new_us"] for x in t[1:]]}
                        res["sweep"].append(row)
                        print(json.dumps(row), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
