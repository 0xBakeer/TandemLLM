"""Bytes and memory of a Kolibri-1 checkpoint on the DGX Spark, from its config.json alone.

Kolibri-1 (Aleph Alpha, `model_type: kolibri1`) is a 50-layer mixture of experts: 384 routed experts,
top 6, one ungated shared expert, every layer MoE, GQA attention with 48 query heads and 4 KV heads of
128, and a 4:1 pattern of sliding-window (513 tokens) and full-attention layers. This tool counts its
parameters by group, the bytes a decode step reads under each weight format, the KV cache at a
context length, the bytes a verify of R rows reads (the routed experts it touches grow with R), and
the bandwidth floor of each on the Spark.

    python tools/kolibri_budget.py CONFIG.json [--ctx 262144] [--bw 273] [--json]

The formats: `bf16` (2 bytes a weight), `fp8` (1 byte plus one fp32 scale per 128 by 128 block, as
Aleph Alpha ships it), `nvfp4` (0.5625 bytes: an e2m1 code and an e4m3 scale per 16 weights). The
router (`mlp.gate`), the norms and the embedding stay BF16 in every profile; the head is BF16 in
`bf16` and `fp8` (as shipped) and e4m3 with a row scale in the NVFP4 profiles (`tools/quant_head.py`).
"""

from __future__ import annotations

import argparse
import json

BYTES = {"bf16": 2.0, "fp8": 1.0 + 4.0 / (128 * 128), "nvfp4": 0.5625, "e4m3row": 1.0}


def counts(c: dict) -> dict:
    H, L = c["hidden_size"], c["num_hidden_layers"]
    nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, k = c["num_experts"], c["num_experts_per_tok"]
    fi, fs, V = c["moe_intermediate_size"], c["shared_expert_intermediate_size"], c["vocab_size"]
    attn = H * nq * hd + 2 * H * nkv * hd + nq * hd * H
    expert = 3 * H * fi
    return {
        "layers": L, "experts": E, "top_k": k,
        "vocab": V, "embed": V * H, "head": V * H,
        "attn_per_layer": attn, "attn": attn * L,
        "expert_each": expert, "routed_all": expert * E * L, "routed_active": expert * k * L,
        "shared": 3 * H * fs * L, "router": H * E * L, "bias": E * L,
        "norms": L * (4 * H + 2 * hd) + H,
        "kv_bytes_token_layer_bf16": 2 * nkv * hd * 2,
        "full_layers": sum(t == "full_attention" for t in c["layer_types"]),
        "sliding_layers": sum(t == "sliding_attention" for t in c["layer_types"]),
        "window": c["sliding_window"],
    }


def profiles() -> dict[str, dict[str, str]]:
    """Which format each group takes. `nvfp4-attn8` keeps attention in the shipped FP8."""
    return {
        "bf16": {"attn": "bf16", "routed": "bf16", "shared": "bf16", "head": "bf16"},
        "fp8": {"attn": "fp8", "routed": "fp8", "shared": "fp8", "head": "bf16"},
        "nvfp4-experts": {"attn": "fp8", "routed": "nvfp4", "shared": "fp8", "head": "e4m3row"},
        "nvfp4": {"attn": "nvfp4", "routed": "nvfp4", "shared": "nvfp4", "head": "e4m3row"},
    }


def resident(n: dict, p: dict[str, str]) -> float:
    """Bytes held in memory for the weights."""
    head = n["head"] * BYTES[p["head"]] + (n["vocab"] * 4 if p["head"] == "e4m3row" else 0)
    return (n["embed"] * 2 + head + n["attn"] * BYTES[p["attn"]] + n["routed_all"] * BYTES[p["routed"]]
            + n["shared"] * BYTES[p["shared"]] + n["router"] * 2 + n["bias"] * 4 + n["norms"] * 2)


def union_uniform(E: int, k: int, rows: int) -> float:
    """Expected distinct experts a layer touches for `rows` tokens if routing were uniform and
    independent. Real routing of consecutive tokens is correlated, so this is an upper estimate;
    `tools/kolibri_ref.py --union` measures it on real text."""
    return E * (1.0 - (1.0 - k / E) ** rows)


def step_bytes(n: dict, p: dict[str, str], rows: int = 1, union: float | None = None) -> float:
    """Weight bytes one forward of `rows` rows reads: every non-expert weight once, and each
    distinct routed expert of each layer once."""
    u = union if union is not None else union_uniform(n["experts"], n["top_k"], rows)
    head = n["head"] * BYTES[p["head"]]
    return (n["attn"] * BYTES[p["attn"]] + n["shared"] * BYTES[p["shared"]] + n["router"] * 2 + head
            + n["layers"] * u * n["expert_each"] * BYTES[p["routed"]])


def kv_bytes(n: dict, ctx: int, kv_bytes_el: float = 2.0, extra_rows: int = 0) -> float:
    """KV held for one sequence: full layers keep every row, sliding layers a ring of the window
    plus `extra_rows` (the rows a verify may write past the window before the commit)."""
    per = n["kv_bytes_token_layer_bf16"] / 2 * kv_bytes_el
    return per * (n["full_layers"] * ctx + n["sliding_layers"] * min(ctx, n["window"] + extra_rows))


def kv_read(n: dict, ctx: int, kv_bytes_el: float = 2.0) -> float:
    """KV a one-token decode step reads at context `ctx`."""
    per = n["kv_bytes_token_layer_bf16"] / 2 * kv_bytes_el
    return per * (n["full_layers"] * ctx + n["sliding_layers"] * min(ctx, n["window"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--bw", type=float, default=273.0, help="GB/s for the floor")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    c = json.load(open(a.config))
    n = counts(c)
    total = (n["embed"] + n["head"] + n["attn"] + n["routed_all"] + n["shared"] + n["router"]
             + n["bias"] + n["norms"])
    active = n["attn"] + n["routed_active"] + n["shared"] + n["router"] + n["head"] + n["norms"] + n["bias"]
    out = {"params_total": total, "params_active_per_token": active + c["hidden_size"], "groups": n,
           "profiles": {}}
    for name, p in profiles().items():
        row = {"resident_gb": resident(n, p) / 1e9, "decode_weight_gb": step_bytes(n, p, 1) / 1e9}
        for ctx in (1024, 32768, 131072, a.ctx):
            for kvn, el in (("bf16", 2.0), ("fp8", 1.0)):
                b = step_bytes(n, p, 1) + kv_read(n, ctx, el)
                row[f"decode_ctx{ctx}_kv{kvn}"] = {"gb": b / 1e9, "floor_tok_s": a.bw / (b / 1e9)}
        row["verify_rows"] = {r: {"union_uniform": union_uniform(n["experts"], n["top_k"], r),
                                  "gb": step_bytes(n, p, r) / 1e9}
                              for r in (1, 2, 4, 8, 16, 24, 32)}
        out["profiles"][name] = row
    out["kv_gb"] = {ctx: {"bf16": kv_bytes(n, ctx, 2.0, 32) / 1e9, "fp8": kv_bytes(n, ctx, 1.0, 32) / 1e9}
                    for ctx in (1024, 32768, 131072, 262144, 1048576)}
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"parameters {total:,}  active a token {out['params_active_per_token']:,}")
    print(f"KV a token: {n['kv_bytes_token_layer_bf16'] * n['full_layers']:,} B in the {n['full_layers']} full layers (bf16); "
          f"sliding layers keep {n['window']} rows")
    for ctx, v in out["kv_gb"].items():
        print(f"  KV at {ctx:>9,}: {v['bf16']:6.2f} GB bf16  {v['fp8']:6.2f} GB fp8")
    for name, row in out["profiles"].items():
        print(f"{name:14s} resident {row['resident_gb']:6.2f} GB   decode weights {row['decode_weight_gb']:5.2f} GB")
        for ctx in (1024, 32768, 131072, a.ctx):
            r = row[f"decode_ctx{ctx}_kvbf16"]
            print(f"    ctx {ctx:>7,}: {r['gb']:5.2f} GB a token, floor {r['floor_tok_s']:6.1f} tok/s at {a.bw:g} GB/s")
        print("    verify rows -> GB (uniform routing): " + ", ".join(
            f"{r}:{v['gb']:.2f}" for r, v in row["verify_rows"].items()))


if __name__ == "__main__":
    main()
