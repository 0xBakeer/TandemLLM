"""The model adapter's data types (ENG-221 contract, ENG-E8 layering): what a model tells the shared
methods about itself, as data the methods read instead of branching on the model.

An adapter is the engine object of one architecture with the verbs the server's loop and the caches
call (forward, forward_tree, forward_block, commit_tree, rollback_to, picks, hidden_post_norm, kv, state,
settle, discard_pending, trim_pending, reset) and a `cost()` returning a `CostModel`. Hot verbs are the
engine's own bound methods, never wrapped; cold ones may be installed on the engine class by the model's
adapter module (`engine/models/qwen3_5/adapter.py` for the 27B, whose `engine/model.py` keeps its bytes).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostModel:
    """Bytes the caches, the memory governor and the dashboard price a model by, from its state layout."""

    section_bytes: int              # the fixed state a snapshot copies whole
    view_bytes_per_token: int       # the position-indexed buffers a snapshot slices, a token
    parts: dict = field(default_factory=dict)   # section bytes by kind ("recurrent", "conv", ...)

    def snapshot_cost(self) -> dict:
        """The dashboard's `cache.snapshot_cost` row (contract 1.3 keys)."""
        return {"recurrent_bytes": int(self.parts.get("recurrent", 0)), "conv_bytes": int(self.parts.get("conv", 0)),
                "kv_bytes_per_token": int(self.view_bytes_per_token)}
