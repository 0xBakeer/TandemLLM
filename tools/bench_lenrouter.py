"""Fixed eight, fixed sixteen, and the router, on the five workloads, in one process.

The comparison this tool exists for is the one phase 5 could not make: the 17:14 table was three
separate runs of `bench_decode.py`, and the router has to be measured against both fixed lengths on
the same weights, the same prompts and the same allocation history or the comparison is with a
different afternoon.

Both fixed baselines run through `LengthRouter` with `fixed=` set rather than through
`DFlash2Drafter` directly, so the three rows differ in the routing policy and in nothing else --
not in which object holds the tap, not in how many drafters are resident, not in the cost of
keeping two draft caches current. A baseline that does not pay a cost the routed run pays would
flatter the router by exactly that cost.

**Thinking is off.** The template's default is on, the atlas row sends `enable_thinking: false`, and
the difference between the two regimes reached 72 % on the same prompt with the same drafter
(SPEED-LEDGER 15:50). Every number here is the regime the row is measured in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.lenrouter import LengthRouter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_spec, generate_spec_tree  # noqa: E402
from tools.bench_decode import PROMPTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--ckpt8", required=True)
    ap.add_argument("--ckpt16", required=True)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--configs", default="fixed8,fixed16,router")
    ap.add_argument("--latch", action="store_true",
                    help="one width decision a request instead of one a block; see "
                         "engine/lenrouter.py::_choose_latched")
    ap.add_argument("--explore", type=int, default=32)
    ap.add_argument("--no-trim", action="store_true",
                    help="turn off the per-step width choice made from the wide drafter's own "
                         "lattice, leaving only the choice of which drafter runs")
    ap.add_argument("--tree", action="store_true",
                    help="verify a TREE per step instead of a chain. Each arm becomes a "
                         "MergedRouter -- the lookup drafter's tree and the block drafter's "
                         "lattice in one node set -- and the length router chooses the node budget")
    ap.add_argument("--corpus", default=os.environ.get("QWEN38_CORPUS", ""))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids
    print(f"[bench] {w.report()}")

    small = DFlash2Drafter(eng, os.path.expanduser(a.ckpt8), blocks=1, max_len=a.max_len, block=8)
    small._build()
    large = DFlash2Drafter(eng, os.path.expanduser(a.ckpt16), blocks=1, max_len=a.max_len, block=16)
    large._build()

    def build_router():
        if not a.tree:
            return LengthRouter(small, large, explore_period=a.explore,
                                width_trim=(not a.no_trim) and not a.latch, latch=a.latch)
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter
        # Track B's measured NVFP4 tiles, not the 11:47 curve: 16 nodes verify in 129.2 ms where
        # they used to take 164.4, and 24 is a bucket the kernel pays a whole second tile for.
        tree_table = {8: 121.7, 16: 129.2, 32: 163.2}
        ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16, node_budget=15,
                          branch_top_k=3, min_expected=0.2, alpha=0.6, corpus_weight=0.5,
                          min_corpus_order=8, verify_base_ms=tree_table[8],
                          verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
        arms = []
        for head, budget in ((small, small.cfg.block_size - 1),
                             (large, large.cfg.block_size - 1)):
            arms.append(MergedRouter(ng, head, mtp_depth=budget, node_budget=budget,
                                     mtp_ms_per_token=0.0, head_fixed_ms=27.0,
                                     adaptive_depth=False, rollback_ms=6.4,
                                     verify_ms_table=dict(tree_table),
                                     tree_ms_table=dict(tree_table)))
        return LengthRouter(arms[0], arms[1], explore_period=a.explore, tree=True, ngram=ng,
                            latch=a.latch)

    run_one = generate_spec_tree if a.tree else generate_spec

    # Warm up BOTH widths before anything is timed, through the verify path this run will use, and
    # then throw the router away.
    #
    # The first run of the first process autotunes Triton, and the first table this tool printed
    # says how much that is worth: the `prose` row, which ran first, read a verify of 260.2 ms and a
    # draft of 68.5 ms against the 115.5 and 26.8 every later row settled at, and its fixed-8
    # figure came out at 11.95 tok/s against phase 5's 18.53 on the same prompt with the same
    # checkpoint. A warm-up is not a nicety here: the first configuration measured was 35 % slower
    # than itself, and the router looked 35 % better than a baseline that was paying for the
    # compiler. The ledger has this trap from phase 4, on prefill, and it is the same one.
    warm = build_router()
    warm.learn_cost = False
    warm_ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPTS["prose"]}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False),
                   return_tensors="pt").input_ids[0].to(a.device)
    for width in (8, 16):
        warm.fixed = width
        run_one(eng, warm_ids, 48, warm, large.cfg.block_size - 1, eos)
    print("[bench] warmed both widths, 2 x 48 tokens; the cost estimates start from here")

    router = build_router()

    wanted = [c.strip() for c in a.configs.split(",") if c.strip()]
    rows = []
    for name, text in PROMPTS.items():
        if a.only and name != a.only:
            continue
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False),
                  return_tensors="pt").input_ids[0].to(a.device)
        print(f"\n### {name}  ({ids.numel()} prompt tokens, thinking off)")
        for label, fixed in (("fixed8", 8), ("fixed16", 16), ("router", 0),
                             ("mix3", 0), ("mix4", 0)):
            if label not in wanted:
                continue
            router.fixed = fixed
            # `mixN` pins the width on a fixed schedule -- one narrow block in every N, chosen by
            # the block counter and nothing else. It isolates the COST OF SWITCHING from the cost
            # of choosing badly: it switches as often as the router does and it knows nothing.
            router.mix_period = int(label[3:]) if label.startswith("mix") else 0
            router.attach()
            _, st = run_one(eng, ids, a.new, router, large.cfg.block_size - 1, eos)
            print("   ", st.line(label))
            print("     ", router.report())
            rows.append({"workload": name, "config": label, "tok_s": st.tok_s,
                         "accept_len": st.accept_len, "accept_rate": st.accept_rate,
                         "blocks": st.blocks, "tokens": st.tokens,
                         "small": router.stats["small"], "large": router.stats["large"],
                         "trims": router.stats["trims"], "forced": router.stats["forced"],
                         "width_hist": dict(router.stats["width_hist"])})
            # `generate_spec` calls `reset()` at the top of every run, which clears the arms, the
            # ceiling rate and the calibration and keeps the learned costs -- so each row here
            # starts from the same cold policy a request gets, and the cost constants improve
            # across the sweep exactly as they would in a long-lived server.

    print("\n" + "=" * 96)
    hdr = [c for c in ("fixed8", "fixed16", "router", "mix3", "mix4") if c in wanted]
    print(f"{'workload':10s} " + " ".join(f"{h:>18s}" for h in hdr) + "   best fixed   router vs")
    means = {h: [] for h in hdr}
    for name in dict.fromkeys(r["workload"] for r in rows):
        cells, by = [], {}
        for h in hdr:
            r = next((x for x in rows if x["workload"] == name and x["config"] == h), None)
            by[h] = r
            cells.append(f"{r['tok_s']:8.2f}/{r['accept_len']:5.2f}" if r else " " * 14)
            if r:
                means[h].append(r["tok_s"])
        fixed = [by[h]["tok_s"] for h in ("fixed8", "fixed16") if by.get(h)]
        best = max(fixed) if fixed else 0.0
        rel = (by["router"]["tok_s"] / best - 1) * 100 if by.get("router") and best else 0.0
        print(f"{name:10s} " + " ".join(f"{c:>18s}" for c in cells)
              + f"   {best:8.2f}   {rel:+6.2f} %")
    print("-" * 96)
    print(f"{'mean':10s} " + " ".join(
        f"{sum(means[h]) / len(means[h]):>18.2f}" if means[h] else " " * 18 for h in hdr))
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        json.dump(rows, open(a.json_out, "w"), indent=1)
        print(f"[bench] -> {a.json_out}")


if __name__ == "__main__":
    main()
