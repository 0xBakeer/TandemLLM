"""The 27B family served (`qwen3_5`, the default family): its builder, its drafter stack and its side slots, which the
server calls through its registry entry (engine/models/qwen3_5/arch.py `serve`) and the builder's `side_slots` hooks.
Moved from server/app.py (ENG-406): the same steps over the same arguments; what the server provides comes in
`ctx` (the KV pool, the vision tower, its state, its price table, the side slots' constant)."""

from __future__ import annotations

from engine.settings import SETTINGS as _S
from engine.spec import Relax


def serve(a, ctx) -> dict:
    """The 27B (and every `qwen3_5` checkpoint): its weights (NVFP4, FP8 head), the KV pool, the
    engine, the tokenizer, the vision tower and the drafter `--drafter` names. `ctx` as for a
    registry builder (engine/models `Arch.serve`): the KV pool, the vision tower, the server's state and price
    table through it. Moved from server/app.py (ENG-406), the same steps."""
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5.adapter import Qwen38Engine   # with the adapter's cold verbs
    from transformers import AutoTokenizer
    cfg = load_config(a.model)
    ctx.keys(cfg.path, [cfg.path, a.nvfp4, a.fp8_head])
    w = Weights(cfg.path, skip_mtp=a.drafter in ("none", "dflash2", "merged", "lenrouter"),
                nvfp4=a.nvfp4,
                fp8_head=a.fp8_head)
    ctx.mark("weights")
    ctx.kv_pool()
    eng = Qwen38Engine(cfg, w, max_len=a.max_len)
    ctx.mark("engine")
    tok = AutoTokenizer.from_pretrained(cfg.path)
    ctx.vision(cfg.path)
    ctx.mark("vision")
    drafter = None
    if a.drafter == "mtp":
        from engine.drafters.mtp import MTPDrafter
        drafter = MTPDrafter(eng, max_len=a.max_len, depth=a.depth, draft_head=a.draft_head)
    elif a.drafter == "router":
        from engine.router import RouterDrafter
        drafter = RouterDrafter(eng, max_len=a.max_len, depth=a.depth)
    elif a.drafter == "merged":
        # The configuration the 11:18 gate passed on, plus the tree: the block drafter priced
        # against the lookup drafter every step, both putting candidates in one verify call.
        from engine.drafters.dflash2 import DFlash2Drafter
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, VERIFY_MS, VERIFY_MS_NVFP4
        table = VERIFY_MS_NVFP4 if a.nvfp4 or _S.get("NVFP4") else VERIFY_MS
        head = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, max_len=a.max_len,
                              path=a.dflash2_path, draft_head=a.draft_head)
        head.tree_temp = a.df2_temp
        head._build()
        # `--budget` counts nodes INCLUDING the anchor, because that is what the measured curve is
        # keyed by and where its cliff is: 16 nodes cost 164.4 ms and 17 cost 172. So the drafters
        # get one fewer.
        ng = NgramDrafter(corpus_path=a.corpus, tokenizer_sha=ctx.state.get("tokenizer_sha"), min_order=3, max_depth=16,
                          node_budget=a.budget - 1, branch_top_k=3, min_expected=0.2,
                          alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                          verify_base_ms=table[min(table)],
                          verify_per_node_ms=(table[max(table)] - table[min(table)])
                          / (max(table) - min(table)))
        drafter = MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                               node_budget=a.budget - 1, mtp_ms_per_token=0.0,
                               head_fixed_ms=35.0, adaptive_depth=False,
                               rollback_ms=6.2, verify_ms_table=table)
        a.depth = a.budget - 1
    elif a.drafter == "dflash2":
        from engine.drafters.dflash2 import DFlash2Drafter
        drafter = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=a.dflash2_blocks,
                                 path=a.dflash2_path, draft_head=a.draft_head,
                                 max_len=a.max_len)
        drafter._build()
        # The block width, not `--depth`, is what this drafter proposes per verify pass.
        a.depth = (drafter.cfg.block_size - 1) * a.dflash2_blocks
    elif a.drafter == "lenrouter":
        drafter = make_lenrouter(a, eng, a.max_len, ctx)
    if a.narrow_arm == "off" and a.drafter != "lenrouter":
        raise SystemExit("[server] --narrow-arm off is the length router's (--drafter lenrouter)")
    ctx.mark("drafters")
    relax = Relax(a.relax_tau, a.relax_rank)
    print(f"[server] {w.report()}")
    if relax.on:
        print(f"[server] LOSSY ACCEPT RULE ON: tau={relax.tau} rank={relax.rank}. Output is not "
              f"greedy and not reproducible against a run with tau 1 and rank 1. See "
              f"docs/exactness.md")
    describe = (f"drafter={a.drafter} depth={a.depth} k={a.k or a.depth} "
                f"nvfp4={w.nvfp4_source or 'off'} fp8_head={(w.fp8_head_source if str(w.fp8_head_source).startswith('build') else 'on') if w.fp8_head_source else 'off'} ")
    return {"engine": eng, "tok": tok, "drafter": drafter, "path": cfg.path, "device": "cuda",
            "cfg": cfg, "w": w, "describe": describe,
            "serving": {"tree": bool(a.tree), "sampled_tree": a.sampled_tree, "relax": relax},
            "side_slots": side_slots(a, ctx, cfg, w)}


def make_lenrouter(a, eng, max_len: int, ctx, corpus_from=None):
    """The served drafter stack (`--drafter lenrouter`) over `eng`, its draft caches `max_len`
    long: slot 0's at the engine's window, a side slot's (`--parallel N`) at its own. Sets
    `a.depth`. `corpus_from`: the lookup drafter whose corpus (and suffix store) a side slot's
    lookup drafter reads too, instead of loading its own copy."""
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.lenrouter import LengthRouter, NoArm
    if not a.dflash2_ckpt16:
        raise SystemExit("--drafter lenrouter needs --dflash2-ckpt16")
    if a.narrow_arm == "off":
        # SPD-75: the wide arm alone; the router keeps the narrow arm's place released
        small = NoArm(eng, block=8)
    else:
        small = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, path=a.dflash2_path,
                               draft_head=a.draft_head, max_len=max_len, block=8)
        small._build()
    large = DFlash2Drafter(eng, a.dflash2_ckpt16, blocks=1, path=a.dflash2_path,
                           draft_head=a.draft_head, max_len=max_len, block=16)
    large._build()
    if a.tree:
        # The combined configuration: each arm is the lookup drafter's tree merged with that
        # arm's own lattice, and the length router chooses the node budget. One NgramDrafter
        # for both arms -- its suffix index is updated in `observe`, and two arms each
        # observing every block would index every token twice.
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, served_tree_table, tree_nodes
        tree_table = served_tree_table(ctx.prices.get("tree_ms"))
        ng = NgramDrafter(corpus_path=a.corpus if corpus_from is None else "",
                          tokenizer_sha=ctx.state.get("tokenizer_sha"), min_order=3, max_depth=16,
                          node_budget=large.cfg.block_size - 1, branch_top_k=3,
                          min_expected=0.2, alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                          verify_base_ms=tree_table[8],
                          verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
        if corpus_from is not None:
            ng.corpus = corpus_from.corpus           # the same corpus and suffix store, read-only
        arms = [MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                             node_budget=tree_nodes(head.cfg.block_size) - 1, mtp_ms_per_token=0.0,
                             head_fixed_ms=ctx.prices.get("head_fixed_ms", 27.0), adaptive_depth=False,
                             rollback_ms=ctx.prices.get("rollback_ms", 6.4),
                             verify_ms_table=dict(tree_table), tree_ms_table=dict(tree_table))
                if not isinstance(head, NoArm) else head for head in (small, large)]
        try:
            drafter = LengthRouter(arms[0], arms[1], fixed=a.len_fixed,
                                   explore_period=a.len_explore, tree=True, ngram=ng,
                                   verify_table=ctx.prices.get("lenrouter_tree_ms"),
                                   draft_table=ctx.prices.get("lenrouter_draft_ms"),
                                   latch=a.len_latch, drop_idle=a.drop_idle, deep=a.deep,
                                   deep_after=a.deep_after, latch_table=dict(tree_table),
                                   learn_cost=a.learn_cost)
        except ValueError as exc:
            raise SystemExit(f"[server] {exc}")
    else:
        try:
            drafter = LengthRouter(small, large, fixed=a.len_fixed,
                                   explore_period=a.len_explore, latch=a.len_latch,
                                   width_trim=not a.len_latch, drop_idle=a.drop_idle,
                                   learn_cost=a.learn_cost)
        except ValueError as exc:
            raise SystemExit(f"[server] {exc}")
    # The router may propose the wide block on any step, so the loop's cap has to be the wide
    # one; asking it for fewer would silently pin it to the narrow length. With the deep chain
    # it is the deep width, for the same reason.
    a.depth = max(large.cfg.block_size - 1, a.deep - 1 if a.tree else 0)
    return drafter


def side_slots(a, ctx, cfg, w) -> dict:
    """The server's side-slot hooks for the 27B (`--parallel N`, server/app.py `_build_side_slots`): an engine
    over the same weights at the side window with its own drafter stack over the same checkpoints and the
    lookup's corpus, its bytes estimate, its graph precapture, and the drafters it is built for."""

    def check() -> None:
        if a.drafter not in ("lenrouter", "none"):
            raise SystemExit(f"[parallel] --parallel {a.parallel} is built for --drafter lenrouter (or none), "
                             f"not {a.drafter}")

    def make(L: int, drafter0):
        from engine.models.qwen3_5.adapter import Qwen38Engine   # with the adapter's cold verbs
        e2 = Qwen38Engine(cfg, w, max_len=L)
        e2.graphs_off = a.side_graphs == "off"
        ng0 = getattr(drafter0, "ngram", None)
        d2 = make_lenrouter(a, e2, L, ctx, corpus_from=ng0) if a.drafter == "lenrouter" else None
        return e2, d2

    def nbytes(L: int, paged: bool) -> tuple[int, int]:
        # SRV-307: with the pool a slot's window is addresses; only its rotary table (256 B a token) and its
        # constant are memory at load
        return (256 if paged else 106_752), int(ctx.side_const_gb() * (1 << 30))

    def precapture(e2, d2) -> None:
        if a.side_precapture == "on" and e2._graphs_for(2, 0) is not None:
            e2._graphs.precapture(widths=range(2, 33) if getattr(d2, "calc", False) else range(2, 17))

    return {"check": check, "make": make, "bytes": nbytes, "precapture": precapture}
