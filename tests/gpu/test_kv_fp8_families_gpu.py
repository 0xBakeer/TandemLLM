"""On the board (SPD-150): the e4m3 KV cache of LFM2 (LFM2_KV_FP8) and NemotronH (NH_KV_FP8) on the GPU.

  * tools/kv_fp8.kv_quant2 (Triton) against its torch twin: the codes of a row equal quantize_kv's (all but
    a rare rounding tie), the scales equal; rows outside `slot` untouched;
  * kernels.attn_rows with the scales (the FP8 branch) against the torch twin on the decoded cache, every
    row of an M-row call the one-row call's bit for bit, and the bf16 launch without them unchanged;
  * each family's tiny engine with the e4m3 cache (tests/test_kv_fp8_families.py 8 and 9, on the GPU): the
    first prompt chunk the bf16 engine's bit for bit, decode close to it, a verify block the decode steps'
    bit for bit, a moved path's codes and scales committed, the self-check;
  * graphs on: a DecodeGraph's logits, captured VerifyGraph blocks and CommitGraph commits give the eager
    bits and the same codes, scales and states.

Run on the box: python tests/gpu/test_kv_fp8_families_gpu.py   (under flock ~/.qwen38-box.flock)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.lfm2_moe import kernels as K  # noqa: E402
from tests import test_kv_fp8_families as F  # noqa: E402
from tools import kv_fp8 as KV  # noqa: E402


def test_kv_quant2_against_its_twin():
    g = torch.Generator(device="cuda").manual_seed(1)
    out = []
    for Hk, D in ((8, 64), (2, 128)):
        R, cap = 33, 4096
        src = (torch.randn(Hk, R, D, device="cuda", generator=g) * 4).to(torch.bfloat16)
        vsrc = (torch.randn(Hk, R, D, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
        idx = torch.arange(R, device="cuda")
        slot = torch.randperm(cap, device="cuda", generator=g)[:R]
        kd = torch.full((Hk, cap, D), 7, dtype=torch.uint8, device="cuda")
        vd, ks, vs = kd.clone(), torch.full((Hk, cap), -1.0, device="cuda"), torch.full((Hk, cap), -1.0, device="cuda")
        KV.kv_quant2(src, vsrc, idx, kd, ks, vd, vs, slot)
        torch.cuda.synchronize()
        mask = torch.ones(cap, dtype=torch.bool, device="cuda")
        mask[slot] = False
        assert bool((kd[:, mask] == 7).all()) and bool((ks[:, mask] == -1).all())
        for x, d, s in ((src, kd, ks), (vsrc, vd, vs)):
            codes, sc = KV.quantize_kv(x)
            same = float((d[:, slot] == codes).float().mean())
            assert same >= 0.999 and torch.allclose(s[:, slot], sc), (Hk, D, same)
        out.append(f"Hk {Hk} D {D}: {same:.4f} of the codes equal")
    return ", ".join(out)


def test_attn_rows_on_codes():
    g = torch.Generator(device="cuda").manual_seed(2)
    out = []
    for Hq, Hk, D in ((32, 8, 64), (32, 2, 128)):
        cap, L = 3000, 2600
        M = 9
        kc = torch.randn(Hk, cap, D, device="cuda", generator=g).to(torch.bfloat16)
        vc = torch.randn(Hk, cap, D, device="cuda", generator=g).to(torch.bfloat16)
        k8, ks = KV.quantize_kv(kc)
        v8, vs = KV.quantize_kv(vc)
        q = torch.randn(M, Hq, D, device="cuda", generator=g).to(torch.bfloat16)
        ctl = torch.tensor([[L + i + 1, L] for i in range(M)], dtype=torch.int32, device="cuda")
        anc = torch.zeros(M, K.ANC, dtype=torch.int32)
        for i in range(M):
            anc[i, :i + 1] = torch.arange(L, L + i + 1)
        anc = anc.cuda()
        ns = K.class_of(L + M, cap)
        got = K.attn_rows(q, k8, v8, ctl, anc, ns, ks=ks, vs=vs)
        ref = K.attn_rows_reference(q, k8, v8, ctl, anc, ks=ks, vs=vs)
        err = F._rel(got, ref)
        assert err < 2 ** -6, (Hq, Hk, D, err)
        for m in range(M):
            one = K.attn_rows(q[m:m + 1], k8, v8, ctl[m:m + 1], anc[m:m + 1], ns, ks=ks, vs=vs)
            assert torch.equal(got[m], one[0]), (Hk, D, m)
        exact = K.attn_rows(q, kc, vc, ctl, anc, ns)
        assert F._rel(exact, K.attn_rows_reference(q, kc, vc, ctl, anc)) < 2 ** -6
        out.append(f"Hk {Hk} D {D}: {err:.1e} of the twin, e4m3 vs bf16 {F._rel(got, exact):.3f}")
    return "; ".join(out)


def test_tiny_engines_on_the_gpu():
    assert F.DEV == "cuda"
    return "; ".join(f"{fn.__name__}: {fn()}" for fn in (F.test_8_lfm2_engine, F.test_9_nemotron_engine))


def _graphs(build, mod, prompt, cap):
    cfg, eng, _ = build(True)
    a, b = eng.new_state(cap), eng.new_state(cap)
    eng.prefill(prompt, a)
    eng.prefill(prompt, b)
    dg = mod.DecodeGraph(eng, a)
    for t in (7, 19, 44):
        ga = dg(t).clone()
        assert torch.equal(ga[0], F._decode(eng, b, [t])[0]), "a graph step differs from the eager step"
    vg, cg = {}, {}
    for _ in range(2):                                   # the second round replays the captured graphs
        ta = eng.verify([5, 6, 7, 8], [-1, 0, 1, 1], a, graphs=vg)
        tb = eng.verify([5, 6, 7, 8], [-1, 0, 1, 1], b)
        assert ta == tb
        eng.commit(a, [0, 1, 3], graphs=cg)
        eng.commit(b, [0, 1, 3])
        assert a.length == b.length
        n = a.length
        for li in cfg.attention_layers:
            for d in ("k", "v", "ks", "vs"):
                assert torch.equal(getattr(a, d)[li][:, :n], getattr(b, d)[li][:, :n]), (d, li)
        for x, y in zip(list(a.conv.values()) + list(getattr(a, "ssm", {}).values()),
                        list(b.conv.values()) + list(getattr(b, "ssm", {}).values())):
            assert torch.equal(x, y)
    return f"{len(dg.graphs)} decode, {len(vg)} verify, {len(cg)} commit graphs"


def test_graphs_with_the_e4m3_cache():
    from engine.models.lfm2_moe import engine as LE
    from engine.models.nemotron_h import engine as NE
    from tests.test_lfm2_moe_engine import PROMPT as LP
    from tests.test_nemotron_h_engine import PROMPT as NP
    return (f"LFM2 {_graphs(F._lfm, LE, LP, 128)}; NemotronH {_graphs(F._nh, NE, NP, 256)}")


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<36} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<36} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
