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
from engine.drafters.engram import EngramDrafter  # noqa: E402
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
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids

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
        for k in [int(x) for x in a.ks.split(",")]:
            eg = EngramDrafter()
            _, st = generate_spec(eng, ids, a.new, eg, k, eos)
            print("   ", st.line(f"engram k={k}"))
            print(f"     engram hit {eg.stats['hits']}/{eg.stats['calls']} calls, "
                  f"orders {dict(sorted(eg.stats['order_hist'].items(), reverse=True))}")
            rows.append((name, "engram", k, st))

    print("\n" + "=" * 92)
    print(f"{'workload':10s} {'drafter':10s} {'k':>3} {'tok/s':>8} {'acc/block':>10} "
          f"{'draft acc':>10} {'rollbacks':>10} {'tok':>5}")
    for name, d, k, st in rows:
        print(f"{name:10s} {d:10s} {k:3d} {st.tok_s:8.2f} {st.accept_len:10.2f} "
              f"{st.accept_rate * 100:9.1f}% {st.rollbacks:10d} {st.tokens:5d}")


if __name__ == "__main__":
    main()
