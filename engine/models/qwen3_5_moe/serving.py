"""The Qwen3.5 MoE trunk behind the server's own generation loop (ENG-391): Flash-Next's serving adapter
(engine/models/flashnext/serving.py) over `Qwen35MoeEngine`, so the request path is the 27B's code --
penalties, the think budget, pattern stops, logprobs, tool calls, the caches on the prefill's 4,096-row
grid -- and the drafter is the shared MTP tree drafter over the pack's own MTP layer, with the lookup
in the same cut (StairCut, priced online or from ops/prices/<model>/<profile>.<device>.json).

  forward, one token      the captured decode step (a graph per attention class)
  forward, more tokens    the prefill kernels, every row's post-norm hidden kept for the MTP layer
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks
  commit_tree             `commit` from captured graphs (the folded GDN commit included)
"""

from __future__ import annotations

import os

import numpy as np

from engine.methods.device import device_id
from engine.methods.manifest import Manifest
from engine.methods.mtp_drafter import MTPTreeDrafter
from engine.methods.staircut import PriceModel, StairCut
from engine.models.adapter import Capabilities
from engine.models.flashnext.serving import FlashNextServing
from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
from engine.settings import SETTINGS as _S

#: fingerprint v3: the files that compute this family's rows (Flash-Next's kernels and engine follow it:
#: the class is built on them)
MODULES = ("engine/models/qwen3_5_moe/__init__.py", "engine/models/qwen3_5_moe/config.py",
           "engine/models/qwen3_5_moe/engine.py", "engine/models/qwen3_5_moe/kernels.py",
           "engine/models/qwen3_5_moe/load.py", "engine/models/qwen3_5_moe/pack.py",
           "engine/models/qwen3_5_moe/ref.py", "engine/models/flashnext/engine.py",
           "engine/models/flashnext/load.py", "engine/models/flashnext/ref.py", "engine/models/flashnext/vision.py",
           "engine/settings.py")
TABLES = ("ops/fn-exl3-tiles.json", "ops/fn-dense-splits.json")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: the model's name in ops/prices/ and models/
MODEL_ID = "ornith-1.5-122b"


def manifest() -> Manifest:
    return Manifest(arch="qwen3_5_moe", modules=MODULES, data=TABLES)


class Qwen35MoeServing(FlashNextServing):
    """`Qwen35MoeEngine` with one state and its graphs, in the server's engine interface. One sequence
    slot for now: the joint verify of several slots is Flash-Next's mixer-shaped pass (not built here)."""

    caps = Capabilities(caches=True, side_slots=False, anchor_grid=4096, shadow=False, side_lane=False)

    def price_class(self, n: int) -> int:
        """The draft prices' context class at length `n`: 0 up to 4,096, then 4,096 x 2^k (a verify's
        attention grows with the context)."""
        c = 4096
        if n <= c:
            return 0
        while c < n and c < 262_144:
            c *= 2
        return c

    def manifest(self) -> Manifest:
        return manifest()


def profile_of(pack: str) -> str:
    return os.environ.get("QSE_WEIGHTS") or "ornith-1.5-122b-" + os.path.basename(os.path.normpath(pack))


def prices(pack: str, log=print) -> PriceModel:
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(pack)}.{device_id()}.json"),
                           log=log)


def build(pack: str, max_len: int, tok, spec: bool = True, depth: int = 3, mark=lambda stage: None, log=print,
          corpus: str | None = None, kv_pool=None):
    """The pack's weights, the engine in the loop's interface, and with `spec` the MTP tree drafter over
    the pack's MTP layer (the hot draft head of FN_DRAFT_HOT ids when the vocabulary is the 27B's), the
    lookup with Q35M_LOOKUP, gated by the load-time self-check of the expert kernels (a failed check
    serves plain decode). Returns (serving, drafter or None, info)."""
    from engine.models.qwen3_5_moe.load import load as q_load
    idx, W, banks, timing = q_load(pack, "cuda", mtp=spec, log=log)
    mark("weights")
    eng_ = Qwen35MoeEngine(idx.cfg, W, banks, device="cuda", max_len=max_len)
    if kv_pool is not None:
        kv_pool()
    eng = Qwen35MoeServing(eng_, max_len)
    mark("engine")
    spec = spec and timing.get("mtp", False)
    drafter, check, hot = None, None, int(_S.get("FN_DRAFT_HOT"))
    if spec:
        if hot and idx.cfg.vocab_size == 248_320:
            eng_.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
        else:
            hot = 0
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check = eng_.self_check(tok(text, add_special_tokens=False)["input_ids"], log=log)
        if check["ok"]:
            ngram = None
            if _S.get("Q35M_LOOKUP") == "1":
                from engine.drafters.ngram import NgramDrafter
                from engine.tokfp import fingerprint
                ngram = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(pack), min_order=3,
                                     max_depth=16, min_corpus_order=8)
                log(f"[server] Qwen3.5 MoE lookup in the cut: corpus "
                    f"{(corpus or 'none') if ngram.corpus is not None else 'none'}")
            drafter = MTPTreeDrafter(eng, eng.add_mtp_head(depth), StairCut(prices(pack, log), siblings=False), ngram)
            drafter.capture_all()
        else:
            log("[server] Qwen3.5 MoE self-check FAILED: speculation off, plain decode only")
    mark("drafters")
    info = {"pack": pack, "load": timing, "self_check": check,
            "draft_head": f"hot {hot}" if spec and hot else ("full" if spec else None),
            "depth": depth if drafter is not None else 0}
    return eng, drafter, info


def serve(a, ctx) -> dict:
    """A Qwen3.5 MoE EXL3 pack for the server (`--model <pack>`, the registry's `serve`): the engine and
    its MTP drafter in the server loop's interface; the pack's EXL3 vision tower when it has one."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    spec = a.drafter != "none"
    if a.drafter not in ("mtp", "none"):
        ctx.log(f"[server] the Qwen3.5 MoE trunk drafts with its own MTP layer: --drafter {a.drafter} taken as mtp")
    tok = AutoTokenizer.from_pretrained(a.model)
    depth = a.depth if a.depth >= 2 else 3
    pool = ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
    eng, drafter, info = build(a.model, a.max_len, tok, spec=spec, depth=depth, mark=ctx.mark, log=ctx.log,
                               corpus=a.corpus, kv_pool=pool)

    def tower_of(vcfg):
        from engine.models.flashnext.vision import load_tower
        from engine.models.qwen3_5_moe.pack import Qwen35MoePackIndex
        return load_tower(Qwen35MoePackIndex.open(a.model), a.model)
    ctx.vision(a.model, tower_of)
    ctx.mark("vision")
    if drafter is not None:
        a.depth = depth
        if drafter.ngram is not None:
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
    describe = (f"Qwen3.5 MoE {a.model}: load {info['load'].get('total_s')} s, drafter "
                f"{'MTP chain + StairCut, depth ' + str(a.depth) if drafter is not None else 'none'}"
                f"{' + lookup' if getattr(drafter, 'ngram', None) is not None else ''}, "
                f"max_len {a.max_len:,},")
    return {"engine": eng, "tok": tok, "drafter": drafter, "path": a.model, "device": "cuda",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "weights_info": {"format": "exl3", "weights": a.model,
                                         "drafters": "mtp (the pack's own layer)" if drafter is not None else None,
                                         "draft_head": info["draft_head"], "load_s": info["load"].get("total_s"),
                                         "self_check": info["self_check"]}}}
