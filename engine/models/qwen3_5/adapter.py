"""The 27B's cold adapter verbs (ENG-E8): installed on `Qwen38Engine` when this module is imported, so
the shared server and caches read `eng.caps` and call `eng.cost()`, `eng.cache_state()`,
`eng.discard_pending()`, `eng.resume_at(n)`, `eng.settle()`, `eng.manifest()` and `eng.precapture()` on
either model without a branch, and `engine/model.py` (in
the 27B's arithmetic manifest) keeps its bytes. The server's loader takes the class from here; the
hot verbs stay the engine's own methods."""

from __future__ import annotations

from engine.methods.manifest import Manifest
from engine.model import VERIFY_ROWS, Qwen38Engine
from engine.models.adapter import Capabilities, CostModel
from engine.models.qwen3_5 import attn_group
from engine.models.state import Qwen38State
from engine.settings import SETTINGS as _S

#: what the 27B's stored rows depend on (fingerprint v3, ENG-243): the forward, verify and commit
#: bodies, the KV pages and the tree they index, the vision tower (image rows are prompt rows), the
#: drafters whose state is cached, and the knob registry (the defaults of these files' knobs); the
#: `tools/` kernels they import and the tile tables their knobs name follow (engine/methods/manifest.py)
MODULES = ("engine/config.py", "engine/drafters/__init__.py", "engine/drafters/dflash2.py",
           "engine/drafters/draft_graph.py", "engine/drafters/dspark.py", "engine/drafters/mtp.py",
           "engine/gdn.py", "engine/kvpool.py", "engine/linear.py", "engine/loader.py", "engine/model.py",
           "engine/models/qwen3_5/attn_group.py", "engine/settings.py", "engine/tree.py",
           "engine/verify_graph.py", "engine/vision.py")


def cost(eng) -> CostModel:
    """The 27B's snapshot bytes: the GDN state and conv tails whole (the state's own count,
    `GDNState.nbytes`, as the caches budgeted before ENG-157), K and V of every attention layer a
    token (bf16); the parts as the dashboard showed them (server/app.py `cache_stats`, before
    2026-10-11)."""
    cfg = eng.cfg
    rec = eng.state.S.numel() * 4
    conv = eng.state.conv.numel() * eng.state.conv.element_size()
    return CostModel(section_bytes=eng.state.nbytes,
                     view_bytes_per_token=len(cfg.attention_layers) * cfg.num_key_value_heads * cfg.head_dim * 2 * 2,
                     parts={"recurrent": rec, "conv": conv})


def discard_pending(eng) -> None:
    """Forget an uncommitted verify and the traces of the last block (after the state was
    overwritten by a restore): a chain's walked state, a folded commit, the block's trace and tree."""
    eng._pending_walk = False
    eng._pend = None                  # a commit pending on the state just overwritten
    eng._trace = None
    eng.tree = None
    eng.trace = None


def cache_state(eng):
    """What the caches keep of a sequence (engine/models/state.py): S and conv, the KV views."""
    return Qwen38State(eng.state, eng.kv)


def resume_at(eng, n: int) -> None:
    """The engine continues at length `n` from the state the caller just wrote into it."""
    discard_pending(eng)
    eng.kv.length = n
    eng.state.primed = n > 0


def settle(eng) -> None:
    """Land a chain accepted in full whose walked state is not in yet (before a snapshot)."""
    eng._settle()


def precapture(eng, drafter) -> int | None:
    """The verify graphs for every width at the first context class, at load (None: graphs off). A
    router that sizes every block on the staircase (QWEN38_LEN_MODE calc/wide) submits any row count
    up to 32; the others stay at 2..16 and capture the rest lazily. SPD-199: the decode-once EXL3 tile
    (QWEN38_EXL3_TALL) is installed first, so every graph captures the path the calls take."""
    from engine.models.qwen3_5 import frag, tall
    n = tall.install(eng)
    if n:
        print(f"[exl3] decode-once tile for 33+ rows: {tall.MODE} rows, {n} blocks", flush=True)
    n = frag.install(eng)
    if n:
        print(f"[exl3] fragment-major activations for 33..{frag.MAX_ROWS} rows: {n} blocks", flush=True)
    if eng._graphs_for(2, 0) is None:
        return None
    import engine.model as _M
    return eng._graphs.precapture(widths=range(2, min(_M.VERIFY_ROWS, getattr(drafter, "max_nodes", 31) + 1) + 1)
                                  if getattr(drafter, "calc", False) else range(2, 17))


def memory_parts(eng) -> dict:
    """The dashboard's memory split of the 27B (server/memsplit.py): its weights, the KV cache's
    tensors held in the allocator (pages of the shared pool are counted by the pool), the GDN state."""
    from engine import kvpool
    w, kv, st = getattr(eng, "w", None), getattr(eng, "kv", None), getattr(eng, "state", None)

    def nb(*ts):
        return sum(int(t.numel() * t.element_size()) for t in ts if t is not None)
    return {"weights": getattr(w, "bytes_fp8", 0) + getattr(w, "bytes_other", 0),
            "kv_pool": nb(*(t for t in (getattr(kv, k, None) for k in ("k", "v", "ks", "vs"))
                            if t is not None and not kvpool.paged(t))),
            "gdn_state": nb(*(getattr(st, k, None) for k in ("S", "conv", "_spare")))}


def resident_files(eng) -> list:
    """Files the engine keeps in the page cache on purpose: none (its weights are in device memory)."""
    return []


def manifest(eng=None) -> Manifest:
    return Manifest(arch="qwen3_5", modules=MODULES)


#: SPD-153: a chain-shaped tree block shares the joint verify pass of a parallel round. Alone, a chain
#: delegates to its chain kernel (`forward_tree`); in a joint round that made it a pass of its own while
#: the other slots waited: 856 lone 77 ms passes against 434 joint ones in an 8-slot run on 2026-10-23.
#: In the joint pass it is verified on the tree path, as every tree block there (`_tree_begin`), so the
#: exactness rule is the joint pass's own (docs/exactness.md, Parallel requests). 0: the old rule.
JOINT_CHAINS = _S.get("JOINT_CHAINS") != "0"
#: SPD-199: the joint pass's row cap below the proven one (`joint_policy`); 0: the proven one
JOINT_MAX_ROWS = max(0, int(_S.get("JOINT_MAX_ROWS") or 0))
_tree_batchable_alone = Qwen38Engine.tree_batchable


def tree_batchable(eng, parents) -> bool:
    """Whether this block can share a joint pass (the server asks only while two or more slots decode):
    any tree block of 2 to VERIFY_ROWS rows, chains included (`JOINT_CHAINS`)."""
    if not JOINT_CHAINS:
        return _tree_batchable_alone(eng, parents)
    return 2 <= len(parents) <= VERIFY_ROWS


def joint_policy(eng) -> dict:
    """SPD-159/155: the joint verify pass this family's kernels keep row-invariant. On EXL3 weights the
    skinny kernel takes 33-64 rows as adjacent 32-row chunks of the same arithmetic, and an 8 x 8-row joint
    pass gave each engine's rows bit for bit its own eager verify (2026-10-23, the ThinkingCap EXL3 4.0
    pack): 64 rows. NVFP4 sends 33-64 rows to its prefill tile (tools/nvfp4_linear.py) and FP8 is not
    proven: 32 rows there. The cut is the auto one either way (it stays within the cap).

    SPD-199: QWEN38_JOINT_MAX_ROWS lowers the cap (never past the proven one). On the EXL3 4.0 pack a
    projection's 33-64 rows cost far more than its 32 (10240 x 5120: 86 us at 32 rows, 141 at 48, 143 at 64;
    the cost stays without the tensor-core step, tools/exl3_tall.py), so 48 buys nothing over 64: the arms
    are 32 and 64."""
    exl3 = getattr(getattr(eng, "w", None), "exl3_source", None) is not None
    proven = 64 if exl3 else 32
    pol = {"max_rows": min(proven, JOINT_MAX_ROWS) if JOINT_MAX_ROWS else proven, "cut": "auto"}
    # SPD-223: the grouped GDN verify carries trees of at most MAXD levels (tools/gdn_verify_kernels.py);
    # a deeper one (the lookup's deep chain: 32) is cut to it in a joint pass, never handed to the kernel
    from tools.gdn_verify_kernels import MAXD
    pol["max_depth"] = int(MAXD)
    from engine.models.qwen3_5 import joint_graphs as JG
    if JG.ON:
        # SPD-163: a joint cut keeps each tree within the graphs' row bucket, so the pass is graphed
        pol["tree_cap"] = JG.t_pad
    from engine.models.qwen3_5 import roundplan as RP
    rp = RP.planner_for(eng, pol["max_rows"])
    if rp is not None:
        # SPD-187: the round planner (shadow: logged only; on: the round's bucket)
        pol["planner"], pol["ctx_class"] = rp, RP.ctx_class
    return pol


def _joint_multi(items: list) -> list:
    """SPD-163: the 27B's joint verify pass (`server/app.py _run_round`): from a CUDA graph where it
    qualifies (engine/models/qwen3_5/joint_graphs.py, QWEN38_JOINT_GRAPHS), else the eager pass."""
    from engine.models.qwen3_5 import joint_graphs as JG
    return JG.forward_tree_multi(items)


def joint_graphs_report(eng) -> dict | None:
    """SPD-163: `/health`'s `parallel.joint_graphs` (None while QWEN38_JOINT_GRAPHS is off)."""
    from engine.models.qwen3_5 import joint_graphs as JG
    return JG.graphs_for(eng).report() if JG.ON else None


Qwen38Engine.forward_tree_multi = staticmethod(_joint_multi)
Qwen38Engine.joint_graphs_report = joint_graphs_report
Qwen38Engine.caps = Capabilities(caches=True, side_slots=True)
Qwen38Engine.tree_batchable = tree_batchable
Qwen38Engine.joint_policy = joint_policy
Qwen38Engine.joint_attn = staticmethod(attn_group.run_attn)      # SPD-245
Qwen38Engine.cost = cost
Qwen38Engine.discard_pending = discard_pending
Qwen38Engine.cache_state = cache_state
Qwen38Engine.resume_at = resume_at
Qwen38Engine.settle = settle
Qwen38Engine.manifest = manifest
Qwen38Engine.resident_files = resident_files
Qwen38Engine.memory_parts = memory_parts
Qwen38Engine.precapture = precapture

__all__ = ["Qwen38Engine", "cache_state", "cost", "discard_pending", "manifest", "precapture", "resume_at",
           "settle"]
