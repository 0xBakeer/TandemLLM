"""On the board (SPD-150): the e4m3 KV cache's kernels for Flash-Next (QWEN38_FN_KV_FP8) and Qwen3.5 MoE
(QWEN38_Q35M_KV_FP8).

  * tools/kv_fp8.kv_quant2 codes a call's staged rows as `quantize_kv` does (the scales equal, the codes on at least
    99.9 % of the values: the Triton cast and torch's may round a tie differently), at the rows' slots only;
  * fn_seq_kernels.attn_rows on the codes and scales gives the bits of attn_rows on the decoded bf16 cache, for dense
    rows, QSA-selected rows and tree rows (its loads decode a code times its scale to bf16, the dot is the bf16 one);
  * qwen3_5_moe.kernels.attn_rows the same;
  * a row's result on the coded cache does not depend on the other rows of its call (a verify block's row is the
    decode step's): M rows at once equal each row alone;
  * kv_quant2 and attn_rows captured in a CUDA graph replay to the eager bits;
  * against the bf16 cache the result stays close (cosine printed and held above 0.99).

Not a speed or quality measurement: the decode rate and the KL against the exact cache are measured on a pack.

Run on the box: python tests/gpu/test_kv_fp8_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import fn_seq_kernels as SK  # noqa: E402
from tools import kv_fp8 as KQ  # noqa: E402

DEV = "cuda"
HQ, HK, AD = 24, 2, 256


def _cache(cap, seed, scale=3.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kc = (torch.randn(HK, cap, AD, generator=g, device=DEV) * scale).to(torch.bfloat16)
    vc = torch.randn(HK, cap, AD, generator=g, device=DEV).to(torch.bfloat16)
    return kc, vc, g


def _coded(kc, vc):
    k8, ks = KQ.quantize_kv(kc)
    v8, vs = KQ.quantize_kv(vc)
    return k8, ks, v8, vs, KQ.kv_dequant(k8, ks), KQ.kv_dequant(v8, vs)


def _dense_rows(M, n0, L, cap):
    """M chain rows at positions n0 .. n0 + M - 1 of a block that starts at L: dense, ancestors at L.."""
    ctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    t = torch.arange(M, device=DEV)
    ctl[:, 0], ctl[:, 1], ctl[:, 2] = (n0 + t + 1).int(), L, 1
    anc = torch.zeros(M, SK.ANC, dtype=torch.int32, device=DEV)
    for i in range(M):
        anc[i, :n0 + i + 1 - L] = torch.arange(L, n0 + i + 1, dtype=torch.int32, device=DEV)
    return ctl, anc, torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)


def _selected_rows(M, ctx, g):
    """The last M rows of a ctx-token context: 512 blocks of 4 (ascending) and the open block."""
    p = torch.arange(ctx - M, ctx, device=DEV)
    nblk = (p + 1) // 4
    sc = torch.rand(M, int(nblk.max()), generator=g, device=DEV)
    sc.masked_fill_(torch.arange(sc.shape[1], device=DEV)[None, :] >= nblk[:, None], -1.0)
    blk = torch.sort(torch.topk(sc, 512, dim=1).indices, dim=1).values
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    ctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    for i in range(M):
        tok = (blk[i, :, None] * 4 + torch.arange(4, device=DEV)).flatten()
        tail = torch.arange(int(nblk[i]) * 4, int(p[i]) + 1, device=DEV)
        v = torch.cat([tok, tail])
        sel[i, :v.numel()] = v.int()
        ctl[i, 0], ctl[i, 1], ctl[i, 2] = v.numel(), ctx, 0
    return ctl, torch.zeros(M, SK.ANC, dtype=torch.int32, device=DEV), sel


def test_kv_quant2_codes_as_quantize_kv():
    kt, vt, _ = _cache(64, 1)
    kd = torch.zeros(HK, 300, AD, dtype=torch.uint8, device=DEV)
    vd = torch.zeros_like(kd)
    ks = torch.zeros(HK, 300, device=DEV)
    vs = torch.zeros_like(ks)
    idx = torch.arange(32, device=DEV)
    slot = torch.arange(200, 232, device=DEV)
    KQ.kv_quant2(kt, vt, idx, kd, ks, vd, vs, slot)
    ck, sk = KQ.quantize_kv(kt[:, :32])
    cv, sv = KQ.quantize_kv(vt[:, :32])
    assert torch.allclose(ks[:, 200:232], sk) and torch.allclose(vs[:, 200:232], sv)
    same = min((kd[:, 200:232] == ck).float().mean().item(), (vd[:, 200:232] == cv).float().mean().item())
    assert same >= 0.999, same
    assert not kd[:, :200].any() and not kd[:, 232:].any() and not ks[:, :200].any()
    return f"codes equal on {same:.5f} of the values"


def test_flash_next_rows_on_codes():
    out = []
    for name, cap, rows in (("dense", 2051, lambda: _dense_rows(16, 2000, 2000, 2051)),
                            ("selected", 32768, None), ("tree", 2100, lambda: _dense_rows(8, 2040, 2040, 2100))):
        kc, vc, g = _cache(cap, 7)
        ctl, anc, sel = rows() if rows is not None else _selected_rows(16, cap, g)
        M = ctl.shape[0]
        q = torch.randn(M, HQ, AD, generator=g, device=DEV).to(torch.bfloat16)
        k8, ks, v8, vs, kd, vd = _coded(kc, vc)
        o8 = SK.attn_rows(q, k8, v8, ctl, anc, sel, ks=ks, vs=vs)
        od = SK.attn_rows(q, kd, vd, ctl, anc, sel)
        assert torch.equal(o8, od), f"{name}: the coded cache's rows differ from the decoded cache's"
        one = torch.cat([SK.attn_rows(q[i:i + 1], k8, v8, ctl[i:i + 1], anc[i:i + 1], sel[i:i + 1], ks=ks, vs=vs)
                         for i in range(M)])
        assert torch.equal(o8, one), f"{name}: a row depends on the other rows of its call"
        ob = SK.attn_rows(q, kc, vc, ctl, anc, sel)
        c = torch.nn.functional.cosine_similarity(o8.float().flatten(), ob.float().flatten(), 0).item()
        assert c > 0.99, (name, c)
        out.append(f"{name} {c:.5f}")
    return "cosine to the bf16 cache: " + ", ".join(out)


def test_qwen35_moe_rows_on_codes():
    from engine.models.qwen3_5_moe import kernels as AK
    kc, vc, g = _cache(4096, 11)
    M, L = 8, 4000
    ctl = torch.zeros(M, 2, dtype=torch.int32, device=DEV)
    ctl[:, 0] = torch.arange(L + 1, L + M + 1, dtype=torch.int32, device=DEV)
    ctl[:, 1] = L
    anc = torch.zeros(M, 33, dtype=torch.int32, device=DEV)
    for i in range(M):
        anc[i, :i + 1] = torch.arange(L, L + i + 1, dtype=torch.int32, device=DEV)
    q = torch.randn(M, HQ, AD, generator=g, device=DEV).to(torch.bfloat16)
    k8, ks, v8, vs, kd, vd = _coded(kc, vc)
    ns = AK.class_of(L + M, 8192)
    o8 = AK.attn_rows(q, k8, v8, ctl, anc, ns, ks=ks, vs=vs).clone()
    od = AK.attn_rows(q, kd, vd, ctl, anc, ns).clone()
    assert torch.equal(o8, od)
    one = torch.cat([AK.attn_rows(q[i:i + 1], k8, v8, ctl[i:i + 1], anc[i:i + 1], ns, ks=ks, vs=vs).clone()
                     for i in range(M)])
    assert torch.equal(o8, one)
    return "the decoded cache's bits; each row its own"


def test_in_a_graph():
    kt, vt, g = _cache(64, 3)
    kc, vc, _ = _cache(2100, 4)
    k8, ks, v8, vs, _, _ = _coded(kc, vc)
    ctl, anc, sel = _dense_rows(16, 2040, 2040, 2100)
    q = torch.randn(16, HQ, AD, generator=g, device=DEV).to(torch.bfloat16)
    idx = torch.arange(16, device=DEV)
    slot = torch.arange(2040, 2056, device=DEV)
    bufs: dict = {}
    KQ.kv_quant2(kt, vt, idx, k8, ks, v8, vs, slot)
    ref = SK.attn_rows(q, k8, v8, ctl, anc, sel, bufs=bufs, ks=ks, vs=vs).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        KQ.kv_quant2(kt, vt, idx, k8, ks, v8, vs, slot)
        SK.attn_rows(q, k8, v8, ctl, anc, sel, bufs=bufs, ks=ks, vs=vs)
    torch.cuda.current_stream().wait_stream(s)
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        KQ.kv_quant2(kt, vt, idx, k8, ks, v8, vs, slot)
        o = SK.attn_rows(q, k8, v8, ctl, anc, sel, bufs=bufs, ks=ks, vs=vs)
    k8[:, 2040:2056].zero_()
    ks[:, 2040:2056].zero_()
    G.replay()
    torch.cuda.synchronize()
    assert torch.equal(o, ref)
    return "captured coding and attention replay to the eager bits"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
