"""On the board (SPD-245): several prompts in one layer-major forward (engine/model.py `prefill_joint`)
leave every engine as its own forward does, bit for bit; and the skinny-order slab product
(tools/exl3_slab_gemm.py `skinny_matmul`) gives the skinny kernel's bits at any row count.

One process, one copy of the weights, four engines. Each prompt (random tokens; lengths 512, 300,
77 and 512 from a primed state at 512) is first forwarded alone (`forward(tokens, start, last_only)`,
the server's prefill of a prompt of at most EXL3_CHUNK_UNTIL rows), then all four together. Checked
with `torch.equal`: the last-row logits, the post-norm hidden rows, every KV row written, S and the
convolution tails. Then the product alone, against `matmul` 512 rows at a time, on every projection
shape of two layers, at 40 / 300 / 512 / 1000 / 4096 rows.

Run on the box with the served environment (QWEN38_EXL3_CHUNK_UNTIL=512, the 27B EXL3 profile):
    python tests/gpu/test_prefill_joint_gpu.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import engine.model as M  # noqa: E402
import engine.models.qwen3_5.adapter  # noqa: E402,F401
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import exl3_slab_gemm as SG  # noqa: E402


def snap(eng, start, T):
    L = start + T
    kv = [(eng.kv.k[i][..., :L, :].clone(), eng.kv.v[i][..., :L, :].clone()) for i in range(eng.kv.k.shape[0])]
    return {"S": eng.state.S.clone(), "conv": eng.state.conv.clone(), "kv": kv,
            "hid": eng.hidden_post_norm.clone(), "pre": eng.hidden_pre_norm.clone(), "len": eng.kv.length}


def same(a, b, tag, bad):
    for k in ("S", "conv", "hid", "pre"):
        if not torch.equal(a[k], b[k]):
            bad.append(f"{tag} {k}: max|d| {(a[k].float() - b[k].float()).abs().max().item():.3e}")
    for i, ((k1, v1), (k2, v2)) in enumerate(zip(a["kv"], b["kv"])):
        if not (torch.equal(k1, k2) and torch.equal(v1, v2)):
            bad.append(f"{tag} kv{i}")
            break
    if a["len"] != b["len"]:
        bad.append(f"{tag} len {a['len']} {b['len']}")


def test_joint_equals_alone(w, cfg):
    engs = [M.Qwen38Engine(cfg, w, max_len=4096, device="cuda") for _ in range(4)]
    g = torch.Generator().manual_seed(5)
    spec = [(512, 0), (300, 0), (77, 0), (512, 512)]
    toks = [torch.randint(1000, 100000, (T,), generator=g).cuda() for T, _ in spec]
    prime = torch.randint(1000, 100000, (512,), generator=g).cuda()

    def ready(k):
        e = engs[k]
        e.reset()
        if spec[k][1]:
            e.forward(prime, start=0, last_only=True)

    alone = []
    with torch.no_grad():
        for k, (T, st) in enumerate(spec):
            ready(k)
            assert M.prefill_joint_ok(engs[k], T, st), (T, st)
            lg = engs[k].forward(toks[k], start=st, last_only=True).float().clone()
            torch.cuda.synchronize()
            alone.append((lg, snap(engs[k], st, T)))
        for k in range(4):
            ready(k)
        t0 = time.perf_counter()
        out = M.prefill_joint([(engs[k], toks[k], spec[k][1]) for k in range(4)])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    bad = []
    for k, (T, st) in enumerate(spec):
        if not torch.equal(out[k].float(), alone[k][0]):
            bad.append(f"prompt {k} logits")
        same(snap(engs[k], st, T), alone[k][1], f"prompt {k}", bad)
    assert not bad, bad
    del engs
    torch.cuda.empty_cache()
    return f"4 prompts (512, 300, 77, 512 after 512) joint == alone: logits, hidden, KV, S, conv; {dt * 1e3:.0f} ms"


def test_skinny_order_product(w, cfg):
    U = X.CHUNK_UNTIL
    g = torch.Generator(device="cuda").manual_seed(0)
    n = 0
    for layer in (0, 3):
        p = f"layers.{layer}"
        lin = cfg.is_linear(layer)
        for b in (w.group(f"{p}.linear_attn.qkvz" if lin else f"{p}.self_attn.qkv"),
                  w.proj(f"{p}.linear_attn.out_proj" if lin else f"{p}.self_attn.o_proj"),
                  w.group(f"{p}.mlp.gate_up"), w.proj(f"{p}.mlp.down_proj")):
            for rows in (40, 300, 512, 1000, 4096):
                x = (torch.randn(rows, b.K, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
                ref = torch.cat([b.matmul(x[r:r + U]) for r in range(0, rows, U)], 0)
                got = SG.skinny_matmul(x, b)
                assert torch.equal(ref, got), (p, b.N, b.K, b.kb, rows, int((ref != got).sum()))
                n += 1
    return f"{n} shape x row cases bit-equal to the skinny kernel"


if __name__ == "__main__":
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    fails = 0
    for name, fn in (("skinny_order_product", test_skinny_order_product), ("joint_equals_alone", test_joint_equals_alone)):
        try:
            print(f"ok   {name}: {fn(w, cfg)}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}", flush=True)
    print(f"{2 - fails} passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
