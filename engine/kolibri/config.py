"""Kolibri-1's shape, read from the checkpoint's `config.json`.

The same keys the plugin and `tools/kolibri_ref.py` read; nothing here is guessed. `sliding_window`
is 513 in the release: a sliding layer sees the current token and the 512 before it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class KolibriConfig:
    hidden: int
    layers: int
    nq: int
    nkv: int
    hd: int
    window: int
    rope_theta: float
    eps: float
    layer_types: list
    experts: int
    topk: int
    inter: int
    shared_inter: int
    vocab: int
    norm_topk_prob: bool
    eos: int
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, c: dict) -> "KolibriConfig":
        return cls(hidden=c["hidden_size"], layers=c["num_hidden_layers"], nq=c["num_attention_heads"],
                   nkv=c["num_key_value_heads"], hd=c["head_dim"], window=c["sliding_window"],
                   rope_theta=float(c["rope_theta"]), eps=float(c["rms_norm_eps"]),
                   layer_types=list(c["layer_types"]), experts=c["num_experts"],
                   topk=c["num_experts_per_tok"], inter=c["moe_intermediate_size"],
                   shared_inter=c.get("shared_expert_intermediate_size", c["moe_intermediate_size"]),
                   vocab=c["vocab_size"], norm_topk_prob=bool(c.get("norm_topk_prob", False)),
                   eos=int(c.get("eos_token_id") or 127906), raw=c)

    @classmethod
    def load(cls, path: str) -> "KolibriConfig":
        """A checkpoint directory or a `config.json`."""
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def sliding(self, L: int) -> bool:
        return self.layer_types[L] == "sliding_attention"

    @property
    def q_size(self) -> int:
        return self.nq * self.hd

    @property
    def kv_size(self) -> int:
        return self.nkv * self.hd
