"""The grouped expert kernels against the per-pair ones, in one process (ENG-241, the joint pass's
largest part): the same rows, ids and bank through both, every pair's output compared bit for bit and
both timed; then the load-time self-check with the grouped kernels (verify rows of every width against
the decode step's single row, on real routing).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_moe_group_check.py --pack ~/flashnext-packs/3.05 \
        --rows 2,5,8,15,24,32 --layers 0,24,47 --json out.json

Rows: each layer's router on random inputs (bf16, the residual stream's scale), and a second set where
the rows repeat (every pair of an expert in one group of up to 8). Timing: CUDA events around each
call, the median of `--reps`.

SPD-249 `--variants 'gy=48;stage=1;gr=4,stage=1'`: the grouped kernels with grid rows (FN_MOE_GY), the
gate|up rows staged once (FN_MOE_STAGE) or a group cap (FN_MOE_GR) against today's grouped call, bit
for bit, with the time and the expert bytes a second (the trellis tiles of the distinct experts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.serving import build  # noqa: E402
from tools import fn_kernels as K  # noqa: E402


def timed(fn, reps: int, inner: int = 1) -> float:
    """The median over `reps` of a call's time; `inner` calls back to back between the events (the queue
    stays full: the launch gaps of one call do not count)."""
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    return sorted(ts)[len(ts) // 2]


def parse_variants(spec: str) -> list:
    """`gy=48;stage=1;gr=4,stage=1` -> [(label, {gy, stage, gr})], today's grouped call (gy 0, no stage,
    gr 8) first; each is timed and compared with `y1`."""
    base = {"gy": 0, "stage": 0, "gr": 8}
    out = [("today", dict(base))] if spec.strip() else []
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        v = dict(base)
        for kv in part.split(","):
            k, _, val = kv.partition("=")
            assert k.strip() in base, f"unknown variant knob {k!r}"
            v[k.strip()] = int(val)
        out.append((part, v))
    return out


def set_variant(v: dict) -> None:
    mod = K._module()
    mod.fn_set_moe_gy(v["gy"])
    mod.fn_set_moe_gr(v["gr"])
    K.MOE_STAGE = bool(v["stage"])


def expert_bytes(bank, ids) -> int:
    """The routed experts' trellis bytes a call reads: gate, up and down of every distinct expert in
    `ids` (`bank.keep` holds (trellis, suh, svh) a projection, in the pointer order)."""
    n = 0
    for e in torch.unique(ids).cpu().tolist():
        for m in range(3):
            t = bank.keep[(e * 3 + m) * 3]
            n += t.numel() * t.element_size()
    return n


def replay(a, fn, K) -> dict:
    """SPD-249: every layer's call of a real joint pass (x, ids from fn_joint_check --dump-moe) through today's
    grouped kernels and each variant: bit-equal to today, time and GB/s a layer, and the pass's sum (the experts
    of one verify pass)."""
    d = torch.load(a.replay)
    variants = parse_variants(a.variants) or [("today", {"gy": 0, "stage": 0, "gr": 8})]
    tot = {lab: {"ms": 0.0, "bytes": 0, "equal": True} for lab, _ in variants}
    us = []
    for rec in d["records"]:
        bank = fn.banks[rec["layer"]]
        x, ids = rec["x"].cuda(), rec["ids"].cuda()
        us.append(int(torch.unique(ids[:, :-1]).numel()))
        nb = expert_bytes(bank, ids)
        K.MOE_GROUP = True
        set_variant({"gy": 0, "stage": 0, "gr": 8})
        y1 = K.moe_experts(x, ids, bank).clone()
        for lab, v in variants:
            set_variant(v)
            yg = K.moe_experts(x, ids, bank).clone()
            tg = timed(lambda: K.moe_experts(x, ids, bank), a.reps, inner=10)
            t = tot[lab]
            t["ms"] += tg
            t["bytes"] += nb
            t["equal"] = t["equal"] and bool(torch.equal(yg, y1))
    set_variant({"gy": 0, "stage": 0, "gr": 8})
    out = {"rows": d["rows"], "layers": len(d["records"]), "distinct_routed_mean": round(sum(us) / max(1, len(us)), 1),
           "variants": {lab: {"equal": t["equal"], "pass_ms": round(t["ms"], 3), "pass_mb": round(t["bytes"] / 1e6, 1),
                              "gbs": round(t["bytes"] / t["ms"] / 1e6, 1)} for lab, t in tot.items()}}
    print(f"[group] replay {out}", flush=True)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--rows", default="2,5,8,15,24,32")
    ap.add_argument("--layers", default="0,24,47")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--variants", default="", help="SPD-249: e.g. 'gy=48;stage=1;gr=4,stage=1' (FN_MOE_GY/STAGE/GR)")
    ap.add_argument("--replay", help="SPD-249: real routing from tools/fn_joint_check.py --dump-moe instead of random rows")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    eng, _, _ = build(a.pack, 4096, tok, spec=False, log=lambda s: None)
    fn = eng.fn
    c = fn.cfg
    g = torch.Generator(device="cuda").manual_seed(7)
    res = {"rows": []}
    if a.replay:
        res["replay"] = replay(a, fn, K)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        return
    for li in [int(x) for x in a.layers.split(",")]:
        bank = fn.banks[li]
        for M in [int(x) for x in a.rows.split(",")]:
            for kind in ("random", "repeated"):
                x = (torch.randn(M, c.hidden_size, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
                if kind == "repeated":
                    x = x[:1].repeat(M, 1).contiguous()
                logits = K.gemv(x, fn.W[fn.pre(li) + "mlp.router"])
                ids, _ = K.router(logits, c.num_experts, c.num_experts_per_tok)
                K.MOE_GROUP = False
                y0 = K.moe_experts(x, ids, bank).clone()
                t0 = timed(lambda: K.moe_experts(x, ids, bank), a.reps)
                K.MOE_GROUP = True
                y1 = K.moe_experts(x, ids, bank).clone()
                t1 = timed(lambda: K.moe_experts(x, ids, bank), a.reps)
                variants = parse_variants(a.variants)
                if variants:
                    nb = expert_bytes(bank, ids)
                    r_gy = []
                    for lab, v in variants:
                        set_variant(v)
                        yg = K.moe_experts(x, ids, bank).clone()
                        tg = timed(lambda: K.moe_experts(x, ids, bank), a.reps, inner=10)
                        r_gy.append({"variant": lab, **v, "equal": bool(torch.equal(yg, y1)),
                                     "ms": round(tg, 4), "gbs": round(nb / tg / 1e6, 1)})
                    set_variant({"gy": 0, "stage": 0, "gr": 8})
                K.MOE_GROUP = False
                experts = int(torch.unique(ids).numel())
                r = {"layer": li, "rows": M, "kind": kind, "pairs": int(ids.numel()), "experts": experts,
                     "equal": bool(torch.equal(y0, y1)), "max_diff": float((y0 - y1).abs().max()),
                     "per_pair_ms": round(t0, 3), "grouped_ms": round(t1, 3)}
                if variants:
                    r["variants"] = r_gy
                    r["expert_mb"] = round(nb / 1e6, 2)
                print(f"[group] {r}", flush=True)
                res["rows"].append(r)
    K.MOE_GROUP = True
    text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    check = fn.self_check(tok(text, add_special_tokens=False)["input_ids"], log=lambda s: print(s, flush=True))
    res["self_check_grouped"] = check
    print(f"[group] self-check with the grouped kernels: {check}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
