"""A NemotronH GGUF (llama.cpp's `nemotron_h_moe`, NVIDIA's Bolt IQ2 release) into the engine's forms.

The config comes from the GGUF's own metadata (`config_from_gguf`): a layer is attention where its
`attention.head_count_kv` entry is non-zero, MoE where its `feed_forward_length` entry is, Mamba2
otherwise; the expert count of a MoE layer is its router's row count (Bolt's per-layer counts), its
top k `expert_used_count` (a scalar or one entry a layer). The last `nextn_predict_layers` blocks are
the MTP head. A `config.json` beside the file, when there is one, is read instead and checked against
the metadata.

Tensors, by type:

  IQ2_XXS 2-D        an `IQ2Linear` (tools/iq2_linear.py): decoded on the fly by its own kernel
  IQ2_XXS experts    an `IQ2Bank` when both the up and the down stack are IQ2_XXS
  F32 / F16 / BF16   as bf16 (fp32 for the router's bias, A, dt_bias, D)
  Q8_0               dequantised to bf16 at load (the twin, bit for bit with gguf-py)
  any other type     refused by name (GGUFFile.open): one kernel per type found in a release

llama.cpp stores some Mamba2 tensors in its own forms: `ssm_a` is A itself (-exp(A_log)), `ssm_d` and
`ssm_a` are [heads, 1], `ssm_norm` is [groups, group size], `ssm_conv1d` is [conv_dim, K]; they are put
back in the checkpoint's shapes, A kept as A (`mixer.A`, which the engine reads before A_log).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from engine.models.nemotron_h.config import NemotronHConfig, validate
from engine.models.nemotron_h.gguf import GGUFFile
from engine.models.nemotron_h.load import ExpertsRelu2

ARCHS = ("nemotron_h_moe", "nemotron_h")
F32_NAMES = ("gate.e_score_correction_bias", "mixer.A", "mixer.dt_bias", "mixer.D")


def _arr(v, n: int) -> list:
    return list(v) if isinstance(v, list) else [v] * n


def config_from_gguf(g: GGUFFile, path: str = "") -> NemotronHConfig:
    arch = g.arch()
    if arch not in ARCHS:
        raise ValueError(f"{g.path}: GGUF architecture {arch!r}; this reader takes {', '.join(ARCHS)}")
    m = {k[len(arch) + 1:]: v for k, v in g.meta.items() if k.startswith(arch + ".")}
    n_all = int(m["block_count"])
    n_mtp = int(m.get("nextn_predict_layers", 0) or 0)
    n = n_all - n_mtp
    kv = _arr(m.get("attention.head_count_kv", 0), n_all)
    ffn = _arr(m.get("feed_forward_length", 0), n_all)
    used = _arr(m.get("expert_used_count", 0), n_all)
    types, k, width, experts = [], {}, {}, {}
    for li in range(n):
        if int(kv[li]):
            types.append("attention")
        elif int(ffn[li]):
            types.append("moe")
            gate = g.tensors.get(f"blk.{li}.ffn_gate_inp.weight")
            if gate is None:
                raise ValueError(f"blk.{li}: a MoE layer without ffn_gate_inp.weight")
            k[li], width[li], experts[li] = int(used[li]), int(ffn[li]), int(gate.shape[0])
        else:
            types.append("mamba")
    mtp_moe = {}
    if n_mtp:
        if n_mtp != 1:
            raise ValueError(f"{n_mtp} MTP blocks: the head runs one")
        gate = g.tensors.get(f"blk.{n}.ffn_gate_inp.weight")
        mtp_moe = {1: (int(used[n]), int(ffn[n]) or int(m.get("expert_feed_forward_length", 0)),
                       int(gate.shape[0]))}
    inner = int(m["ssm.inner_size"])
    heads = int(m["ssm.time_step_rank"])
    latent = g.tensors.get(f"blk.{next(iter(k), 0)}.ffn_latent_down.weight")
    eos = g.meta.get("tokenizer.ggml.eos_token_id")
    cfg = NemotronHConfig(
        path=path or os.path.dirname(g.path), hidden_size=int(m["embedding_length"]),
        num_attention_heads=int(_arr(m["attention.head_count"], n_all)[next(i for i, t in enumerate(types) if t == "attention")]),
        num_key_value_heads=int(next(int(x) for x in kv if int(x))), head_dim=int(m.get("attention.key_length", 128)),
        vocab_size=int(m.get("vocab_size") or g.tensors["token_embd.weight"].shape[0]),
        rms_norm_eps=float(m.get("attention.layer_norm_rms_epsilon", 1e-5)),
        max_position_embeddings=int(m.get("context_length", 262144)), layer_types=types, mamba_num_heads=heads,
        mamba_head_dim=inner // heads, ssm_state_size=int(m["ssm.state_size"]), n_groups=int(m["ssm.group_count"]),
        conv_kernel=int(m.get("ssm.conv_kernel", 4)), chunk_size=128, moe_k=k, moe_ffn=width, moe_experts=experts,
        moe_latent_size=int(latent.shape[0]) if latent is not None else None,
        shared_ffn=int(m.get("expert_shared_feed_forward_length", 0)),
        routed_scaling_factor=float(m.get("expert_weights_scale", 1.0)), tie_word_embeddings="output.weight" not in g.tensors,
        eos_token_id=int(eos) if eos is not None else 2, mtp_layer_types=["attention", "moe"] if n_mtp else [],
        mtp_moe=mtp_moe, eos_token_ids=[int(eos)] if eos is not None else [2], architecture="gguf:" + arch)
    if not bool(m.get("expert_weights_norm", True)):
        raise ValueError("expert_weights_norm false: this engine implements the renormalised top k only")
    if int(m.get("expert_group_count", 1) or 1) != 1:
        raise ValueError("expert_group_count != 1: this engine implements one group")
    validate(cfg)
    return cfg


def _names(cfg: NemotronHConfig) -> dict:
    """{stored GGUF name: canonical name} for the trunk and the MTP head."""
    out = {"token_embd.weight": "embeddings.weight", "output_norm.weight": "norm_f.weight", "output.weight": "lm_head.weight"}
    mamba = {"ssm_in.weight": "in_proj.weight", "ssm_conv1d.weight": "conv1d.weight", "ssm_conv1d.bias": "conv1d.bias",
             "ssm_dt.bias": "dt_bias", "ssm_a": "A", "ssm_d": "D", "ssm_norm.weight": "norm.weight",
             "ssm_out.weight": "out_proj.weight"}
    attn = {"attn_q.weight": "q_proj.weight", "attn_k.weight": "k_proj.weight", "attn_v.weight": "v_proj.weight",
            "attn_output.weight": "o_proj.weight"}
    moe = {"ffn_gate_inp.weight": "gate.weight", "exp_probs_b.bias": "gate.e_score_correction_bias",
           "ffn_latent_down.weight": "fc1_latent_proj.weight", "ffn_latent_up.weight": "fc2_latent_proj.weight",
           "ffn_up_exps.weight": "experts.up_proj", "ffn_down_exps.weight": "experts.down_proj",
           "ffn_up_shexp.weight": "shared_experts.up_proj.weight", "ffn_down_shexp.weight": "shared_experts.down_proj.weight"}
    for li, t in enumerate(cfg.layer_types):
        out[f"blk.{li}.attn_norm.weight"] = f"layers.{li}.norm.weight"
        for a, b in {"mamba": mamba, "attention": attn, "moe": moe}[t].items():
            out[f"blk.{li}.{a}"] = f"layers.{li}.mixer.{b}"
    if cfg.mtp_layer_types:
        j = cfg.num_hidden_layers
        out.update({f"blk.{j}.nextn.eh_proj.weight": "mtp.layers.0.eh_proj.weight",
                    f"blk.{j}.nextn.enorm.weight": "mtp.layers.0.enorm.weight",
                    f"blk.{j}.nextn.hnorm.weight": "mtp.layers.0.hnorm.weight",
                    f"blk.{j}.attn_norm.weight": "mtp.layers.0.norm.weight",
                    f"blk.{j}.post_attention_norm.weight": "mtp.layers.1.norm.weight",
                    f"blk.{j}.nextn.shared_head_norm.weight": "mtp.layers.1.final_layernorm.weight"})
        for a, b in attn.items():
            out[f"blk.{j}.{a}"] = f"mtp.layers.0.mixer.{b}"
        for a, b in moe.items():
            out[f"blk.{j}.{a}"] = f"mtp.layers.1.mixer.{b}"
    return out


def _dense(g: GGUFFile, name: str, device, want) -> torch.Tensor:
    from tools import iq2_linear as Q
    t = g.tensors[name]
    raw = g.raw(name)
    N = t.shape[0] if len(t.shape) > 1 else 1
    K = int(np.prod(t.shape)) // N
    if t.type_name in ("F32", "F16", "BF16"):
        dt = {"F32": np.float32, "F16": np.float16}.get(t.type_name)
        x = torch.from_numpy(np.array(raw).view(dt)) if dt else torch.from_numpy(np.array(raw).view(np.int16)).view(torch.bfloat16)
        return x.reshape(t.shape).to(device=device, dtype=want)
    if t.type_name == "Q8_0":
        return Q.dequant_q8_0(np.array(raw), N, K).reshape(t.shape).to(device=device, dtype=want)
    raise ValueError(f"{name}: {t.type_name} has no reader here")


def load_gguf(path: str, device="cuda", mtp: bool = True, log=print):
    """(cfg, W, experts, timing, g) of a NemotronH GGUF, in load.py's forms."""
    from tools import iq2_linear as Q
    t0 = time.time()
    g = GGUFFile.open(path)
    cfg = config_from_gguf(g)
    cj = os.path.join(os.path.dirname(path), "config.json")
    if os.path.isfile(cj):
        from engine.models.nemotron_h.config import config_from_dict
        c2 = config_from_dict(json.load(open(cj)), os.path.dirname(path))
        for f in ("layer_types", "moe_k", "moe_ffn", "moe_experts", "hidden_size", "ssm_state_size", "mamba_num_heads"):
            if getattr(c2, f) != getattr(cfg, f):
                raise ValueError(f"config.json and the GGUF metadata disagree on {f}")
    names = _names(cfg)
    missing = sorted(set(names) - set(g.tensors) - ({"output.weight"} if cfg.tie_word_embeddings else set()))
    missing = [m for m in missing if not (m.endswith(("ffn_latent_down.weight", "ffn_latent_up.weight")) and not cfg.moe_latent_size)]
    missing = [m for m in missing if mtp or not names[m].startswith("mtp.")]
    if missing:
        raise ValueError(f"{path}: {len(missing)} tensor(s) missing, first {missing[0]}")
    W, experts, packed = {}, {}, 0
    for stored, canon in names.items():
        if stored not in g.tensors or (not mtp and canon.startswith("mtp.")):
            continue
        t = g.tensors[stored]
        if canon.endswith(("experts.up_proj", "experts.down_proj")):
            continue
        if t.type_name == "IQ2_XXS" and len(t.shape) == 2:
            W[canon] = Q.IQ2Linear(np.array(g.raw(stored)), t.shape[0], t.shape[1], device=device)
            packed += 1
            continue
        want = torch.float32 if canon.endswith(F32_NAMES) else torch.bfloat16
        x = _dense(g, stored, device, want)
        if canon.endswith(("mixer.A", "mixer.D")):
            x = x.reshape(-1)
        elif canon.endswith("mixer.norm.weight"):
            x = x.reshape(-1)
        elif canon.endswith("conv1d.weight"):
            x = x.reshape(cfg.conv_dim, 1, cfg.conv_kernel)
        W[canon] = x.contiguous()
    if cfg.tie_word_embeddings:
        W["lm_head.weight"] = W["embeddings.weight"]
    for pre in [f"layers.{li}.mixer." for li in cfg.attention_layers] + (["mtp.layers.0.mixer."] if mtp and cfg.mtp_layer_types else []):
        ws = [W[pre + f"{x}_proj.weight"] for x in "qkv"]
        if all(torch.is_tensor(w) for w in ws):
            W[pre + "qkv"] = torch.cat(ws).contiguous()
    keys = [(li, f"blk.{li}.") for li in cfg.moe_layers] + ([(("mtp", 1), f"blk.{cfg.num_hidden_layers}.")] if mtp and cfg.mtp_layer_types else [])
    for key, b in keys:
        up, down = g.tensors[b + "ffn_up_exps.weight"], g.tensors[b + "ffn_down_exps.weight"]
        E, I, X = up.shape
        if up.type_name == down.type_name == "IQ2_XXS":
            experts[key] = Q.IQ2Bank(np.array(g.raw(up.name)), np.array(g.raw(down.name)), E, I, X, device=device)
            packed += 2
        else:
            experts[key] = ExpertsRelu2(_dense(g, up.name, device, torch.bfloat16).reshape(E, I, X).contiguous(),
                                        _dense(g, down.name, device, torch.bfloat16).reshape(E, X, I).contiguous())
    timing = {"total_s": round(time.time() - t0, 2), "packed": packed, "types": g.types(), "mtp": mtp}
    if log:
        log(f"[nh-gguf] {path}: {timing}")
    return cfg, W, experts, timing, g
