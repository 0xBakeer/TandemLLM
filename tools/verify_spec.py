"""The M2 correctness gate: speculation must not change greedy output.

Runs the same prompts three ways -- no drafter, a drafter that is wrong on purpose, and the engram
drafter -- and requires the three token sequences to be identical. The adversarial drafter is the
one that matters: it drives the rollback path on nearly every block, at every possible accept
length, which a drafter that is usually right would almost never do.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.fixed import AdversarialDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_greedy, generate_spec  # noqa: E402

PROMPTS = {
    "prose": "Write three paragraphs about why the memory system, and not the arithmetic units, "
             "sets the speed of a language model that generates one token at a time.",
    "chat": "I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
            "Explain in plain language what that means for running a 27-billion-parameter model, "
            "and what I should expect.",
    "code": "Write a Python function that reads a safetensors file header and prints every tensor "
            "name, dtype and shape, sorted by the number of bytes it occupies. Include a short "
            "docstring and a main guard.",
}


def build(args):
    cfg = load_config(args.model)
    w = Weights(cfg.path, device=args.device, skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=args.max_len, device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    return cfg, w, eng, tok


def encode(tok, text: str, chat: bool, device: str) -> torch.Tensor:
    if chat:
        text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids[0].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    cfg, w, eng, tok = build(a)
    ok = True
    for name, text in PROMPTS.items():
        if a.only and name != a.only:
            continue
        ids = encode(tok, text, a.chat, a.device)
        base, sb = generate_greedy(eng, ids, a.new)
        print(sb.line(f"{name}/no drafter"))

        adv = AdversarialDrafter(cfg.vocab_size, seed=1)
        got, sa = generate_spec(eng, ids, a.new, adv, a.k)
        print(sa.line(f"{name}/adversarial"))
        same = got[:len(base)] == base
        ok &= same
        print(f"    adversarial identical to greedy: {'yes' if same else 'NO'}")
        if not same:
            for i, (x, y) in enumerate(zip(base, got)):
                if x != y:
                    print(f"    first difference at {i}: greedy {x!r} ({tok.decode([x])!r}) "
                          f"vs spec {y!r} ({tok.decode([y])!r})")
                    break

        # a drafter that is right some of the time, to exercise partial accepts of every length
        half = AdversarialDrafter(cfg.vocab_size, seed=2, truth=base, accept_prefix=a.k // 2)
        got2, sh = generate_spec(eng, ids, a.new, half, a.k)
        print(sh.line(f"{name}/half-right"))
        same2 = got2[:len(base)] == base
        ok &= same2
        print(f"    half-right identical to greedy:  {'yes' if same2 else 'NO'}")

        eg = EngramDrafter()
        got3, se = generate_spec(eng, ids, a.new, eg, a.k)
        print(se.line(f"{name}/engram"))
        same3 = got3[:len(base)] == base
        ok &= same3
        print(f"    engram identical to greedy:      {'yes' if same3 else 'NO'}")
        print(f"    engram hit rate {eg.stats['hits']}/{eg.stats['calls']}, "
              f"orders {eg.stats['order_hist']}")
        print()

    print("=" * 78)
    print(f"GATE  speculation does not change greedy output   {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
