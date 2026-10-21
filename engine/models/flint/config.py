"""The Flint configuration (`flint`), read from the checkpoint.

Flint is a pre-norm decoder of grouped-query attention layers, each followed by an MLP. Every attention
layer has per-head RMS norms on q and k and one learned sink logit per query head
(`self_attn.softmax_offset`: an extra softmax column with no value). `layer_types` makes a layer a
sliding-window layer (`sliding_window` keys counting the row itself, NEOX rotary at `rope_theta`) or a
full-attention layer; the full layers are `no_rope_layer_indices` and take no positional encoding. The
MLP is a dense SwiGLU on `mlp_only_layers` and elsewhere a mixture of `num_experts` SwiGLU experts:
sigmoid scores, the top `num_experts_per_tok` (an `expert_bias`, when a checkpoint has one, for the
choice only), the chosen scores renormalised to sum to one, plus an ungated shared expert of
`shared_expert_intermediate_size`.

As for the other families, a field this module does not know stops the load with its name, and fields
whose value would change the arithmetic are checked against the one value the engine implements
(`FIXED`). An EXL3 pack's `config.json` is the checkpoint's with the converter's `quantization_config`
added; the pack reader takes that block (engine/models/flint/pack.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from engine.config import resolve_snapshot

FAMILY_TYPES = ("flint",)
ARCHITECTURES = ("FlintForCausalLM",)

KNOWN = {
    "hidden_size", "intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps",
    "rope_theta", "rope_parameters", "max_position_embeddings", "layer_types", "sliding_window", "use_sliding_window",
    "no_rope_layer_indices", "mlp_only_layers", "num_experts", "num_experts_per_tok", "eos_token_id",
    "bos_token_id", "pad_token_id", "model_type", "architectures",
}
IGNORED = {
    "dtype", "torch_dtype", "initializer_range", "output_router_logits", "use_cache", "transformers_version",
    "quantization_config", "router_aux_loss_coef", "attention_dropout",
}
FIXED = {"hidden_act": "silu", "norm_topk_prob": True, "attention_bias": False, "tie_word_embeddings": False,
         "softmax_type": "learnable", "router_score_function": "sigmoid", "decoder_sparse_step": 1}


def is_flint(raw: dict) -> bool:
    t = raw.get("text_config") or raw
    return str(t.get("model_type", raw.get("model_type", ""))) in FAMILY_TYPES or any(
        a in ARCHITECTURES for a in raw.get("architectures") or [])


@dataclass
class FlintConfig:
    path: str
    hidden_size: int
    intermediate_size: int                 # the dense layers' MLP
    moe_intermediate_size: int             # a routed expert's
    shared_expert_intermediate_size: int   # the shared expert's (0: none)
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    layer_types: list[str]                 # "sliding_attention" | "full_attention"
    sliding_window: int                    # keys a sliding row sees, itself included (0: no window)
    no_rope_layers: list[int]
    mlp_only_layers: list[int]
    num_experts: int
    num_experts_per_tok: int
    eos_token_id: int
    bos_token_id: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)   # generation stop ids
    # the routed experts' renormalisation: sum + ROUTE_EPS (the published routing's 1e-20)
    route_eps: float = 1e-20

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def gqa(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_sliding(self, layer: int) -> bool:
        return self.sliding_window > 0 and self.layer_types[layer] == "sliding_attention"

    def has_rope(self, layer: int) -> bool:
        return layer not in self.no_rope_layers

    def is_moe(self, layer: int) -> bool:
        return layer not in self.mlp_only_layers

    @property
    def moe_layers(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_moe(i)]

    @property
    def sliding_layers(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_sliding(i)]

    @property
    def full_layers(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if not self.is_sliding(i)]

    def kv_bytes_per_token(self, elem: int = 2) -> int:
        """K and V of every layer for one token (bf16: 100,352 bytes on Flint; the sliding layers keep every
        row today, the window is applied when they are read)."""
        return self.num_hidden_layers * 2 * self.kv_dim * elem


def config_from_dict(raw: dict, path: str = "", gen: dict | None = None) -> FlintConfig:
    t = raw.get("text_config") or raw
    n = t["num_hidden_layers"]
    rope = t.get("rope_parameters") or {}
    eos = t.get("eos_token_id", raw.get("eos_token_id"))
    eos_list = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    if not eos_list:
        raise ValueError("Flint needs eos_token_id in its config")
    gen_eos: list[int] = []
    if gen:
        e = gen.get("eos_token_id")
        gen_eos = e if isinstance(e, list) else ([e] if e is not None else [])
    hidden = t["hidden_size"]
    use_sw = bool(t.get("use_sliding_window", True))
    cfg = FlintConfig(
        path=path,
        hidden_size=hidden,
        intermediate_size=t["intermediate_size"],
        moe_intermediate_size=t["moe_intermediate_size"],
        shared_expert_intermediate_size=int(t.get("shared_expert_intermediate_size") or 0),
        num_hidden_layers=n,
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t.get("num_key_value_heads", t["num_attention_heads"]),
        head_dim=t.get("head_dim") or hidden // t["num_attention_heads"],
        vocab_size=t["vocab_size"],
        rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 10000.0))),
        max_position_embeddings=t.get("max_position_embeddings", 262144),
        layer_types=list(t.get("layer_types") or ["full_attention"] * n),
        sliding_window=int(t.get("sliding_window") or 0) if use_sw else 0,
        no_rope_layers=[int(i) for i in t.get("no_rope_layer_indices") or []],
        mlp_only_layers=[int(i) for i in t.get("mlp_only_layers") or []],
        num_experts=t["num_experts"],
        num_experts_per_tok=t["num_experts_per_tok"],
        eos_token_id=int(eos_list[0]),
        bos_token_id=t.get("bos_token_id"),
        eos_token_ids=[int(x) for x in (gen_eos or eos_list)],
    )
    validate(cfg, t)
    return cfg


def validate(cfg: FlintConfig, t: dict) -> None:
    unknown = sorted(set(t) - KNOWN - IGNORED - set(FIXED))
    if unknown:
        raise ValueError(f"Flint config field(s) this engine does not know: {', '.join(unknown)}")
    for k, want in FIXED.items():
        if k in t and t[k] != want:
            raise ValueError(f"Flint config {k}={t[k]!r}: this engine implements {want!r} only")
    rope = t.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or t.get("rope_scaling"):
        raise ValueError("Flint rope scaling: this engine implements the plain rotary only")
    if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {"sliding_attention", "full_attention"}:
        raise ValueError(f"layer_types must list sliding_attention or full_attention for each of "
                         f"{cfg.num_hidden_layers} layers")
    if any(not 0 <= i < cfg.num_hidden_layers for i in cfg.no_rope_layers + cfg.mlp_only_layers):
        raise ValueError("no_rope_layer_indices / mlp_only_layers out of range")
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
    if cfg.head_dim % 2:
        raise ValueError("the rotary needs an even head_dim")
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError("num_experts_per_tok out of range")
    if any(cfg.layer_types[i] == "sliding_attention" for i in range(cfg.num_hidden_layers)) and \
            t.get("use_sliding_window", True) and cfg.sliding_window < 2:
        raise ValueError("a sliding layer needs sliding_window >= 2")


def load_flint_config(path: str | None = None) -> FlintConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    if not is_flint(raw):
        raise ValueError(f"{snap}: not a Flint config")
    gen = None
    gp = os.path.join(snap, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    return config_from_dict(raw, snap, gen)
