"""Single-stream decode throughput across a workload mix.

Five prompts, chosen so that the two regimes a lookup drafter lives between are both represented:
fresh generation, where a suffix memory knows nothing, and reproduction, where most of the output
is already somewhere in the context. Reporting a single tok/s for this model without saying which
regime it came from is how a recipe ends up quoting a number nobody can reproduce.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.mtp import MTPDrafter  # noqa: E402
from engine.router import RouterDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_greedy, generate_spec  # noqa: E402

SNIPPET = '''
def load_shard(path, device):
    """Read one safetensors shard and return its tensors, quantised weights kept as codes."""
    out = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            if ".visual." in key:
                continue
            out[key] = f.get_tensor(key)
    return out


def step_bytes(weights, n_layers):
    """Bytes one autoregressive step has to read, by group."""
    per_layer = {}
    for name, tensor in weights.items():
        if not name.startswith("layers."):
            continue
        idx = int(name.split(".")[1])
        per_layer[idx] = per_layer.get(idx, 0) + tensor.numel() * tensor.element_size()
    layers = sum(v for k, v in per_layer.items() if k < n_layers)
    head = weights["lm_head.weight"]
    return layers + head.numel() * head.element_size()
'''

PROMPTS = {
    "prose": ("Write three paragraphs about why the memory system, and not the arithmetic units, "
              "sets the speed of a language model that generates one token at a time."),
    "chat": ("I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
             "Explain in plain language what that means for running a 27-billion-parameter model."),
    "code": ("Write a Python function that reads a safetensors file header and prints every tensor "
             "name, dtype and shape, sorted by the number of bytes it occupies."),
    "edit": ("Here is a Python module:\n\n```python" + SNIPPET + "```\n\n"
             "Rewrite it with type hints on every function and parameter. Change nothing else: "
             "keep the same function names, the same docstrings, the same logic and the same "
             "order. Output the complete module."),
    "quote": ("Read this passage and then repeat it back to me word for word, exactly as written, "
              "with no commentary:\n\n" + open(os.path.join(
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "bench", "refprompt.txt")).read().split("\n\n")[0]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--ks", default="8,16")
    ap.add_argument("--baseline", action="store_true", help="also time the non-speculative path")
    ap.add_argument("--only", default=None)
    ap.add_argument("--mtp", default=None,
                    help="comma separated draft depths for the checkpoint's mtp head")
    ap.add_argument("--mtp-hidden", default="post", choices=["post", "pre"])
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--router", default=None,
                    help="comma separated mtp depths for the engram+mtp router")
    ap.add_argument("--dflash2", default=None,
                    help="comma separated chain lengths in blocks for the block drafter; "
                         "1 = one 8-wide block (7 proposals), 2 = two chained (14)")
    ap.add_argument("--dflash2-path", default="greedy", choices=["greedy", "viterbi", "both"],
                    help="how the block drafter walks its own lattice")
    ap.add_argument("--dflash2-head", default=None,
                    help="reduced-vocabulary head for the block drafter's own head read")
    ap.add_argument("--dflash2-head-both", action="store_true",
                    help="run every block-drafter configuration twice, with and without that head")
    ap.add_argument("--dflash2-ckpt", default=None)
    ap.add_argument("--dflash2-tap", default="entry", choices=["entry", "output", "both"],
                    help="which hidden state target_layer_ids names")
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids

    walks = ["greedy", "viterbi"] if a.dflash2_path == "both" else [a.dflash2_path]
    heads = [None, a.dflash2_head] if a.dflash2_head_both else [a.dflash2_head]
    taps = ["entry", "output"] if a.dflash2_tap == "both" else [a.dflash2_tap]
    dflash2_configs = [(int(b), wk, hd, tp) for b in (a.dflash2.split(",") if a.dflash2 else [])
                       for wk in walks for hd in heads for tp in taps]
    dflash2_cache: dict[tuple, DFlash2Drafter] = {}

    rows = []
    for name, text in PROMPTS.items():
        if a.only and name != a.only:
            continue
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True),
                  return_tensors="pt").input_ids[0].to(a.device)
        print(f"\n### {name}  ({ids.numel()} prompt tokens)")
        if a.baseline:
            _, st = generate_greedy(eng, ids, a.new, eos)
            print("   ", st.line("no drafter"))
            rows.append((name, "none", 0, st))
        for d in ([int(x) for x in a.mtp.split(",")] if a.mtp else []):
            md = MTPDrafter(eng, max_len=a.max_len, hidden=a.mtp_hidden, depth=d)
            _, st = generate_spec(eng, ids, a.new, md, d, eos)
            print("   ", st.line(f"mtp-{a.mtp_hidden} d={d}"))
            rows.append((name, f"mtp-{a.mtp_hidden}", d, st))
        for d in ([int(x) for x in a.router.split(",")] if a.router else []):
            rd = RouterDrafter(EngramDrafter(), MTPDrafter(eng, max_len=a.max_len,
                                                          hidden=a.mtp_hidden), mtp_depth=d)
            _, st = generate_spec(eng, ids, a.new, rd, max(d, 16), eos)
            print("   ", st.line(f"router d={d}"))
            print(f"     router chose engram {rd.stats['engram']}x "
                  f"({rd.stats['engram_tokens']} tok), mtp {rd.stats['mtp']}x "
                  f"({rd.stats['mtp_tokens']} tok)")
            rows.append((name, "router", d, st))
        for nb, walk, hd, tp in dflash2_configs:
            dd = dflash2_cache.get((nb, walk, hd, tp))
            if dd is None:
                dd = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=nb, path=walk, tap=tp,
                                    draft_head=hd or "", max_len=a.max_len)
                dd._build()
                dflash2_cache[(nb, walk, hd, tp)] = dd
            dd.attach()
            width = (dd.cfg.block_size - 1) * nb
            _, st = generate_spec(eng, ids, a.new, dd, width, eos)
            dd.detach()
            tag = ("+head" if hd else "") + ("" if tp == "entry" else "/out")
            print("   ", st.line(f"dflash2-{walk[:3]}{tag} b={nb}"))
            rows.append((name, f"df2-{walk[:3]}{tag}", width, st))

        for k in ([] if a.no_engram else [int(x) for x in a.ks.split(",")]):
            eg = EngramDrafter()
            _, st = generate_spec(eng, ids, a.new, eg, k, eos)
            print("   ", st.line(f"engram k={k}"))
            print(f"     engram hit {eg.stats['hits']}/{eg.stats['calls']} calls, "
                  f"orders {dict(sorted(eg.stats['order_hist'].items(), reverse=True))}")
            rows.append((name, "engram", k, st))

    print("\n" + "=" * 92)
    print(f"{'workload':10s} {'drafter':12s} {'k':>3} {'tok/s':>8} {'acc/block':>10} "
          f"{'draft acc':>10} {'rollbacks':>10} {'tok':>5}")
    for name, d, k, st in rows:
        print(f"{name:10s} {d:12s} {k:3d} {st.tok_s:8.2f} {st.accept_len:10.2f} "
              f"{st.accept_rate * 100:9.1f}% {st.rollbacks:10d} {st.tokens:5d}")


if __name__ == "__main__":
    main()
