"""A tiny random Kolibri-1 on disk, in the served layout, for CPU tests of the engine.

    python tools/kolibri_tiny.py OUT_DIR

writes `OUT_DIR/release/` (Aleph Alpha's FP8 layout: e4m3 projections with fp32 128x128 block
scales, BF16 norms/router/embedding/head, a config and a word-level tokenizer) and `OUT_DIR/set/`
(the published set's layout: `layers/{L}.safetensors` with NVFP4 experts, shared expert and attention,
BF16 norms and router, fp32 router bias; `outside.safetensors` with the embedding, the final norm
and the e4m3 head with fp32 row scales). Every dimension the FP8 blocks touch is a multiple of 128.

`reference_logits(dir, ids, attn)` is `tools/kolibri_ref.layer_forward` over the same weights
dequantised to fp32: what the engine must reproduce.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import kolibri_nvfp4 as kn  # noqa: E402

CFG = {
    "hidden_size": 128, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 64, "rms_norm_eps": 1e-6, "vocab_size": 96, "rope_theta": 10000.0,
    "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "norm_topk_prob": False, "sliding_window": 5,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"], "eos_token_id": 95,
}


def tensors(c=CFG, seed=0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    H, A, KV = c["hidden_size"], c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
    Fi, Fs, E, V = c["moe_intermediate_size"], c["shared_expert_intermediate_size"], c["num_experts"], c["vocab_size"]

    def r(*s, scale=0.08):
        return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)

    def nrm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)
    t = {"model.embed_tokens.weight": r(V, H, scale=1.0), "lm_head.weight": r(V, H, scale=0.2),
         "model.norm.weight": nrm(H)}
    for L in range(c["num_hidden_layers"]):
        p = f"model.layers.{L}."
        for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
            t[p + n + ".weight"] = nrm(H)
        t[p + "self_attn.q_proj.weight"] = r(A, H)
        t[p + "self_attn.k_proj.weight"] = r(KV, H)
        t[p + "self_attn.v_proj.weight"] = r(KV, H)
        t[p + "self_attn.o_proj.weight"] = r(H, A)
        t[p + "self_attn.q_norm.weight"] = nrm(c["head_dim"])
        t[p + "self_attn.k_norm.weight"] = nrm(c["head_dim"])
        t[p + "mlp.gate.weight"] = r(E, H, scale=0.3)
        t[p + "moe.router.expert_bias"] = r(E, scale=0.1).float()
        for e in range(E):
            t[p + f"mlp.experts.{e}.gate_proj.weight"] = r(Fi, H)
            t[p + f"mlp.experts.{e}.up_proj.weight"] = r(Fi, H)
            t[p + f"mlp.experts.{e}.down_proj.weight"] = r(H, Fi)
        t[p + "mlp.shared_experts.gate_proj.weight"] = r(Fs, H)
        t[p + "mlp.shared_experts.up_proj.weight"] = r(Fs, H)
        t[p + "mlp.shared_experts.down_proj.weight"] = r(H, Fs)
    return t


def fp8_block(v: torch.Tensor):
    N, K = v.shape
    nb, kb = -(-N // 128), -(-K // 128)
    s = torch.empty(nb, kb)
    codes = torch.empty(N, K, dtype=torch.float8_e4m3fn)
    for i in range(nb):
        for j in range(kb):
            blk = v[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128].float()
            s[i, j] = blk.abs().max().clamp_min(1e-12) / 448.0
            codes[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (blk / s[i, j]).to(torch.float8_e4m3fn)
    return codes, s


def nvfp4(v: torch.Tensor):
    c, s, s2 = kn.clip_batched(v.float()[None], None)
    return c[0], s[0], s2[0, 0, 0].reshape(())


def fp8_here(layout: str, c: dict, L: int, k: str) -> bool:
    """Whether projection `k` of layer L is FP8 in the set's layout. "nvfp4": none. "kvfullsh8"
    (the published set): k and v everywhere, all attention on full layers, the shared expert."""
    if layout != "kvfullsh8":
        return False
    full = c["layer_types"][L] == "full_attention"
    return (".k_proj." in k or ".v_proj." in k or ("self_attn." in k and full) or "shared_experts." in k)


def write(out: str, c=CFG, seed=0, layout: str = "nvfp4") -> tuple[str, str]:
    t = tensors(c, seed)
    rel, st = os.path.join(out, "release"), os.path.join(out, "set")
    os.makedirs(rel, exist_ok=True)
    os.makedirs(os.path.join(st, "layers"), exist_ok=True)
    r = {}
    for k, v in t.items():
        if k.endswith("_proj.weight"):
            r[k], r[k[: -len("weight")] + "weight_scale_inv"] = fp8_block(v)
        else:
            r[k] = v
    save_file({k: v.contiguous() for k, v in r.items()}, os.path.join(rel, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in r}},
              open(os.path.join(rel, "model.safetensors.index.json"), "w"))
    json.dump(c, open(os.path.join(rel, "config.json"), "w"))
    for L in range(c["num_hidden_layers"]):
        p = f"model.layers.{L}."
        lay = {}
        for k, v in t.items():
            if not k.startswith(p):
                continue
            n = "layers." + k[len("model.layers."):]
            if k.endswith("_proj.weight") and fp8_here(layout, c, L, k):
                b = n[: -len(".weight")]
                lay[b + ".weight"], lay[b + ".weight_scale"] = fp8_block(v)    # the fp32 table as weight_scale
            elif k.endswith("_proj.weight"):
                b = n[: -len(".weight")]
                lay[b + ".weight"], lay[b + ".weight_scale"], lay[b + ".weight_scale_2"] = nvfp4(v)
            else:
                lay[n] = v
        save_file({k: v.contiguous() for k, v in lay.items()}, os.path.join(st, "layers", f"{L}.safetensors"))
    hc, hs = kn.quant_head_e4m3(t["lm_head.weight"])
    save_file({"embed_tokens.weight": t["model.embed_tokens.weight"], "norm.weight": t["model.norm.weight"],
               "lm_head.weight": hc, "lm_head.weight_scale": hs}, os.path.join(st, "outside.safetensors"))
    json.dump(c, open(os.path.join(st, "config.json"), "w"))
    if layout != "nvfp4":
        json.dump({"layout": layout}, open(os.path.join(st, "manifest.json"), "w"))
    return st, rel


def reference_logits(st: str, rel: str, ids: list[int], attn: str = "fp8") -> torch.Tensor:
    """fp32 logits [T, V] of `tools/kolibri_ref.layer_forward` over the served weights, dequantised:
    experts and shared expert from the set's NVFP4, attention from the release's FP8 (attn="fp8")
    or the set's NVFP4 (attn="set"), the e4m3 head."""
    from safetensors import safe_open
    from tools import kolibri_ref as kr
    c = json.load(open(os.path.join(rel, "config.json")))
    ck = kr.Ckpt(rel)
    E = c["num_experts"]
    with safe_open(os.path.join(st, "outside.safetensors"), framework="pt") as f:
        emb = f.get_tensor("embed_tokens.weight").float()
        fn = f.get_tensor("norm.weight").float()
        head = f.get_tensor("lm_head.weight").float() * f.get_tensor("lm_head.weight_scale")[:, None]
    r = emb[torch.tensor(ids)]
    for L in range(c["num_hidden_layers"]):
        with safe_open(os.path.join(st, "layers", f"{L}.safetensors"), framework="pt") as f:
            def nv(b):
                w_ = f.get_tensor(b + ".weight")
                if w_.dtype == torch.float8_e4m3fn:          # FP8 block, fp32 table
                    sc = f.get_tensor(b + ".weight_scale")
                    N, K = w_.shape
                    return w_.float() * sc.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
                return kn.dequant(w_, f.get_tensor(b + ".weight_scale"),
                                  f.get_tensor(b + ".weight_scale_2"), torch.float32)
            p = f"layers.{L}."
            w = {k: f.get_tensor(p + n + ".weight").float() for k, n in kr.LayerW.NAMES.items()
                 if k not in ("q", "k", "v", "o", "sg", "su", "sd")}
            for k, n in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
                w[k] = (ck.get(f"model.layers.{L}.self_attn.{n}.weight", torch.float32) if attn == "fp8"
                        else nv(p + "self_attn." + n))
            for k, n in (("sg", "gate_proj"), ("su", "up_proj"), ("sd", "down_proj")):
                w[k] = nv(p + "mlp.shared_experts." + n)
            w["bias"] = f.get_tensor(p + "moe.router.expert_bias").float()
            for k, n in (("eg", "gate_proj"), ("eu", "up_proj"), ("ed", "down_proj")):
                w[k] = torch.stack([nv(p + f"mlp.experts.{e}.{n}") for e in range(E)])
        r, _ = kr.layer_forward(r, kr.LayerW(**w), c, L)
    return kr.rms(r, fn, c["rms_norm_eps"]).float() @ head.T


if __name__ == "__main__":
    print(write(sys.argv[1]))
