"""The 27B's cold adapter verbs (ENG-E8 step 0): installed on `Qwen38Engine` when this module is
imported, so the shared server and caches call `eng.cost()`, `eng.discard_pending()` and `eng.manifest()` on either
model without a branch, and `engine/model.py` (in the 27B's arithmetic manifest) keeps its bytes. The
server's loader takes the class from here; the hot verbs stay the engine's own methods."""

from __future__ import annotations

from engine.methods.manifest import Manifest
from engine.model import Qwen38Engine
from engine.models.adapter import CostModel

#: what the 27B's stored rows depend on (fingerprint v3, ENG-243): the forward, verify and commit
#: bodies, the KV pages and the tree they index, the vision tower (image rows are prompt rows), the
#: drafters whose state is cached, and the knob registry (the defaults of these files' knobs); the
#: `tools/` kernels they import and the tile tables their knobs name follow (engine/methods/manifest.py)
MODULES = ("engine/config.py", "engine/drafters/__init__.py", "engine/drafters/dflash2.py",
           "engine/drafters/draft_graph.py", "engine/drafters/dspark.py", "engine/drafters/mtp.py",
           "engine/gdn.py", "engine/kvpool.py", "engine/linear.py", "engine/loader.py", "engine/model.py",
           "engine/settings.py", "engine/tree.py", "engine/verify_graph.py", "engine/vision.py")


def cost(eng) -> CostModel:
    """The 27B's snapshot bytes: the GDN state and conv tails whole, K and V of every attention layer
    a token (bf16) -- the formula the dashboard used (server/app.py `cache_stats`, before 2026-10-11)."""
    cfg = eng.cfg
    rec = eng.state.S.numel() * 4
    conv = eng.state.conv.numel() * eng.state.conv.element_size()
    return CostModel(section_bytes=rec + conv,
                     view_bytes_per_token=len(cfg.attention_layers) * cfg.num_key_value_heads * cfg.head_dim * 2 * 2,
                     parts={"recurrent": rec, "conv": conv})


def discard_pending(eng) -> None:
    """Forget an uncommitted verify (after the state was overwritten by a restore): the fields the
    caches reset by hand until ENG-157 moves them to this call."""
    eng._pending_walk = False
    eng._pend = None


def manifest(eng=None) -> Manifest:
    return Manifest(arch="qwen3_5", modules=MODULES)


Qwen38Engine.cost = cost
Qwen38Engine.discard_pending = discard_pending
Qwen38Engine.manifest = manifest

__all__ = ["Qwen38Engine", "cost", "discard_pending", "manifest"]
