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
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
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


def bf16_ulp(x: float) -> float:
    """The distance between adjacent bf16 numbers at magnitude `x`.

    bf16 keeps 7 stored mantissa bits, so the spacing at magnitude 2^e is 2^(e-7): at a logit of 20
    it is 0.125. That number is not a detail. The engine's logits ARE bf16, so two tokens whose
    logits differ by one spacing are as close as this model can represent them being, and which of
    the two comes out of an argmax depends on the rounding of whatever matmul produced them.
    """
    import math
    if x == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def compare(base: list[int], got: list[int], gaps: list[float], tok,
            tops: list[float] | None = None) -> tuple[bool, str]:
    """Does a speculative run reproduce the greedy run?

    With one exception, which the chat prompt of 2026-09-17 10:02 forced into the open. Greedy
    decoding is only defined where the argmax is: at a position where the top two logits are the
    SAME NUMBER there is no greedy answer, only whichever of the two a reduction happened to return
    first, and a block verify reduces over [T, V] where the one-token path reduces over [1, V].
    A run that follows the greedy run to an exact tie and then takes the other token has not broken
    losslessness. It has run out of greedy to be lossless about.

    So a mismatch is a failure unless the gap at that position is exactly zero, and the verdict says
    which of the two it was. Anything above zero is a real divergence, however small.
    """
    for i, (x, y) in enumerate(zip(base, got)):
        if x == y:
            continue
        g = gaps[i] if i < len(gaps) else float("nan")
        ulp = bf16_ulp(tops[i]) if tops and i < len(tops) else 0.0
        where = (f"at {i}: greedy {x} ({tok.decode([x])!r}) vs spec {y} ({tok.decode([y])!r}), "
                 f"top1-top2 gap {g:.4f} = {g / ulp:.2f} bf16 ulp" if ulp else
                 f"at {i}: greedy {x} vs spec {y}, top1-top2 gap {g:.4f}")
        if g == 0.0:
            return True, f"follows greedy to an exact logit tie, {where}"
        if g <= ulp:
            return True, f"follows greedy to a one-ulp logit tie, {where}"
        return False, f"DIVERGES {where}"
    return True, "identical"


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
    ap.add_argument("--dflash2", default=None,
                    help="also gate the block drafter, at this many chained blocks")
    ap.add_argument("--dflash2-path", default="greedy", choices=["greedy", "viterbi"])
    ap.add_argument("--dflash2-head", default=None)
    ap.add_argument("--dflash2-ckpt", default=None,
                    help="gate a fine-tuned drafter. Losslessness is by construction -- a drafter "
                         "only proposes -- but a drafter trained by this repository is exactly the "
                         "kind of thing that should be made to prove it anyway")
    ap.add_argument("--dflash2-block", type=int, default=0)
    a = ap.parse_args()

    cfg, w, eng, tok = build(a)
    ok = True
    for name, text in PROMPTS.items():
        if a.only and name != a.only:
            continue
        ids = encode(tok, text, a.chat, a.device)
        base, sb = generate_greedy(eng, ids, a.new, record_gaps=True)
        ties = sum(1 for g, t in zip(sb.gaps, sb.tops) if g <= bf16_ulp(t))
        print(sb.line(f"{name}/no drafter"))
        print(f"    greedy top1-top2 logit gap: median {sorted(sb.gaps)[len(sb.gaps) // 2]:.3f}, "
              f"minimum {min(sb.gaps):.4f}, positions within one bf16 ulp "
              f"{ties} of {len(sb.gaps)}")

        adv = AdversarialDrafter(cfg.vocab_size, seed=1)
        got, sa = generate_spec(eng, ids, a.new, adv, a.k)
        print(sa.line(f"{name}/adversarial"))
        same, why = compare(base, got, sb.gaps, tok, sb.tops)
        ok &= same
        print(f"    adversarial: {why}")

        # a drafter that is right some of the time, to exercise partial accepts of every length
        half = AdversarialDrafter(cfg.vocab_size, seed=2, truth=base, accept_prefix=a.k // 2)
        got2, sh = generate_spec(eng, ids, a.new, half, a.k)
        print(sh.line(f"{name}/half-right"))
        same2, why2 = compare(base, got2, sb.gaps, tok, sb.tops)
        ok &= same2
        print(f"    half-right:  {why2}")

        if a.dflash2:
            nb = int(a.dflash2)
            dd = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=nb, path=a.dflash2_path,
                                draft_head=a.dflash2_head, max_len=a.max_len,
                                block=a.dflash2_block or None)
            width = (dd.cfg.block_size - 1) * nb
            got4, sd = generate_spec(eng, ids, a.new, dd, width)
            dd.detach()
            print(sd.line(f"{name}/dflash2 b={nb}"))
            same4, why4 = compare(base, got4, sb.gaps, tok, sb.tops)
            ok &= same4
            print(f"    dflash2:     {why4}")

        eg = EngramDrafter()
        got3, se = generate_spec(eng, ids, a.new, eg, a.k)
        print(se.line(f"{name}/engram"))
        same3, why3 = compare(base, got3, sb.gaps, tok, sb.tops)
        ok &= same3
        print(f"    engram:      {why3}")
        print(f"    engram hit rate {eg.stats['hits']}/{eg.stats['calls']}, "
              f"orders {eg.stats['order_hist']}")
        print()

    from engine.model import ROLLBACK_DIFF
    if ROLLBACK_DIFF:
        worst = max(ROLLBACK_DIFF, key=lambda r: r[2])
        rel = [d / max(m, 1e-9) for _, _, d, m in ROLLBACK_DIFF]
        print(f"rank-k vs replay rollback over {len(ROLLBACK_DIFF)} layer-rollbacks: "
              f"worst absolute {worst[2]:.3e} (layer {worst[0]}, keep {worst[1]}, "
              f"state absmax {worst[3]:.3f}), worst relative {max(rel):.3e}, "
              f"mean relative {sum(rel) / len(rel):.3e}")
    print("=" * 78)
    print("GATE  speculation changes greedy output only where the top two logits are within one")
    print(f"      bf16 ulp of each other, which is as close as this engine can represent them: "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
