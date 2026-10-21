"""ENG-401: an NVFP4 pack of Flint from its BF16 checkpoint, on one GPU (the Spark), no calibration data.

    PYTHONPATH=<repo> python tools/flint_nvfp4.py --model <bf16 dir> --out <pack dir> [--head nvfp4|bf16]

Every projection the model multiplies by -- attention q/k/v/o, the dense layers' and the shared experts' SwiGLU,
the 12,032 routed experts' gate/up/down and, with `--head nvfp4` (the default), the head -- becomes NVFP4: e2m1
codes two a byte (`X.weight` uint8 [N, K / 2], the even K in the low nibble), an e4m3 scale per group of 16
(`X.weight_scale` [N, K / 16]) and an fp32 tensor scale (`X.weight_scale_2`). Each group's scale is the one of
tools/quant_nvfp4.py's clip search (`quantize_clipped`) that minimises the group's squared error, with no
activation weighting (no calibration corpus). Projections the engine runs as one launch share one tensor scale:
q, k and v; a SwiGLU's gate and up (an expert's too); they are quantised as one matrix and stored as their own
tensors. Everything else (norms, sinks, the router, the embedding) is copied as stored. config.json gains
`quantization_config` {"quant_method": "nvfp4", "group_size": 16, ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flint.config import config_from_dict  # noqa: E402
from tools.quant_nvfp4 import CLIP_RATIOS, quantize_clipped  # noqa: E402

SHARD_BYTES = 4 << 30
COPY = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "LICENSE")


class Reader:
    def __init__(self, d: str):
        self.d = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h: dict = {}

    def get(self, key: str) -> torch.Tensor:
        f = self.wm[key]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(key)


def groups(cfg) -> list[tuple[str, ...]]:
    """The projections quantised together (one tensor scale), by stored key without `.weight`."""
    out: list[tuple[str, ...]] = []
    for li in range(cfg.num_hidden_layers):
        a = f"model.layers.{li}.self_attn."
        out += [(a + "q_proj", a + "k_proj", a + "v_proj"), (a + "o_proj",)]
        m = f"model.layers.{li}.mlp."
        if cfg.is_moe(li):
            for e in range(cfg.num_experts):
                x = m + f"experts.{e}."
                out += [(x + "gate_proj", x + "up_proj"), (x + "down_proj",)]
            if cfg.shared_expert_intermediate_size:
                x = m + "shared_expert."
                out += [(x + "gate_proj", x + "up_proj"), (x + "down_proj",)]
        else:
            out += [(m + "gate_proj", m + "up_proj"), (m + "down_proj",)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="nvfp4", choices=("nvfp4", "bf16"))
    ap.add_argument("--keep", default="", help="comma-separated parts left in BF16: attn, dense, shared, experts")
    ap.add_argument("--bf16-expert-layers", default="",
                    help="comma-separated layers whose routed experts stay BF16 (the top layers' massive activations)")
    a = ap.parse_args()
    t0 = time.time()
    raw = json.load(open(os.path.join(a.model, "config.json")))
    cfg = config_from_dict(raw, a.model)
    src = Reader(a.model)
    os.makedirs(a.out, exist_ok=True)
    keep = set(filter(None, a.keep.split(",")))
    part = lambda k: ("attn" if ".self_attn." in k else "shared" if ".shared_expert." in k
                      else "experts" if ".experts." in k else "dense")
    xl = {int(x) for x in a.bf16_expert_layers.split(",") if x.strip()}
    layer_of = lambda k: int(k.split(".")[2]) if k.startswith("model.layers.") else -1
    gs = [g for g in groups(cfg) if part(g[0]) not in keep
          and not (part(g[0]) == "experts" and layer_of(g[0]) in xl)]
    if a.head == "nvfp4":
        gs.append(("lm_head",))
    quantized = {k for g in gs for k in g}
    shard, shard_bytes, n_shard, wm = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if not shard:
            return
        n_shard += 1
        name = f"model-{n_shard:05d}.safetensors"
        save_file(shard, os.path.join(a.out, name), metadata={"format": "pt"})
        for k in shard:
            wm[k] = name
        shard, shard_bytes = {}, 0

    def put(k: str, t: torch.Tensor):
        nonlocal shard_bytes
        shard[k] = t.contiguous().cpu()
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= SHARD_BYTES:
            flush()

    err_sum, err_n = 0.0, 0
    for i, g in enumerate(gs):
        ws = [src.get(k + ".weight") for k in g]
        W = torch.cat(ws).cuda()
        blk = quantize_clipped(W, None, ratios=CLIP_RATIOS)
        if i % 997 == 0:
            deq = blk.dequant().float()
            e = float(((deq - W.float()).pow(2).mean() / W.float().pow(2).mean()).sqrt())
            err_sum, err_n = err_sum + e, err_n + 1
        r = 0
        for k, w in zip(g, ws):
            n = w.shape[0]
            put(k + ".weight", blk.w[r:r + n])
            put(k + ".weight_scale", blk.s[r:r + n])
            put(k + ".weight_scale_2", torch.tensor(blk.s2, dtype=torch.float32))
            r += n
        if i % 2000 == 0:
            print(f"[nvfp4] {i}/{len(gs)} groups, {time.time() - t0:.0f} s", flush=True)
    for k in src.wm:
        base = k[:-len(".weight")] if k.endswith(".weight") else None
        if base in quantized:
            continue
        put(k, src.get(k))
    flush()
    json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(a.out, f)) for f in set(wm.values()))},
               "weight_map": dict(sorted(wm.items()))}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"),
              indent=1)
    for f in COPY:
        if os.path.isfile(os.path.join(a.model, f)):
            shutil.copy(os.path.join(a.model, f), os.path.join(a.out, f))
    raw["quantization_config"] = {"quant_method": "nvfp4", "group_size": 16, "scale_dtype": "float8_e4m3fn",
                                  "tensor_scale": "fp32", "head": a.head, "bf16_parts": sorted(keep),
                                  "bf16_expert_layers": sorted(xl), "scales": "clip search, unweighted",
                                  "producer": "TandemLLM tools/flint_nvfp4.py"}
    json.dump(raw, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(json.dumps({"groups": len(gs), "shards": n_shard, "seconds": round(time.time() - t0),
                      "sampled_rel_rms_error": round(err_sum / max(err_n, 1), 4),
                      "gb": round(sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 1e9, 2)}))


if __name__ == "__main__":
    main()
