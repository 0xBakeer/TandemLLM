"""Measure the relaxed accept rule on the board, with its quality cost beside every speed number.

The rule is lossy, so a throughput figure from it means nothing on its own. This tool produces the
two columns together, in one process, on the same weights and the same prompts:

    speed      tokens a second, accepted tokens a block, and how many of those accepts were
               relaxed ones rather than argmax agreements
    quality    the negative log-likelihood the EXACT TARGET assigns to the text that came out,
               per token, against the same target's own greedy text -- plus a degeneration check,
               because an n-gram repetition is the failure that an average NLL hides

The likelihood is taken under the same weight set that generated it, teacher-forced in one pass. That
isolates the accept rule: the quantisation gate in `tools/quality_gate.py` is the one that compares
weight sets, and mixing the two questions would make neither answerable.

Everything is off by default. `--taus 1.0` alone reproduces the lossless engine and is the control
every other row is read against.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import Relax, generate_spec  # noqa: E402

PROMPTS = {
    "prose": ("Write three paragraphs about why the memory system, and not the arithmetic units, "
              "sets the speed of a language model that generates one token at a time."),
    "chat": ("I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
             "Explain in plain language what that means for running a 27-billion-parameter model."),
    "science": ("Explain how a sodium-potassium pump maintains the resting potential of a neuron, "
                "and why the cell spends so much of its energy budget on it."),
    "code": ("Write a Python function that reads a safetensors file header and prints every tensor "
             "name, dtype and shape, sorted by the number of bytes it occupies."),
}


def teacher_forced_nll(eng: Qwen38Engine, ids: torch.Tensor, first: int, chunk: int = 512) -> float:
    """Mean NLL, in nats, that the target assigns to `ids[first:]` given everything before it."""
    eng.reset()
    total = 0.0
    counted = 0
    pos = 0
    with torch.no_grad():
        while pos < ids.numel():
            end = min(pos + chunk, ids.numel())
            logits = eng.forward(ids[pos:end], start=pos)
            logprob = torch.log_softmax(logits[0].float(), dim=-1)
            # logits[i] predicts ids[pos + i + 1]
            lo = max(first, pos + 1)
            hi = end
            if hi > lo:
                rows = torch.arange(lo - pos - 1, hi - pos - 1, device=ids.device)
                target = ids[lo:hi]
                total += float(-logprob[rows].gather(1, target[:, None].long()).sum())
                counted += int(hi - lo)
            pos = end
    return total / max(counted, 1)


def distinct_ngrams(tokens: list[int], n: int = 4) -> float:
    if len(tokens) <= n:
        return 1.0
    grams = {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    return len(grams) / (len(tokens) - n + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--nvfp4", default=None)
    parser.add_argument("--fp8-head", default=None)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--taus", default="1.0,0.3,0.1,0.02")
    parser.add_argument("--ranks", default="")
    parser.add_argument("--dflash2-blocks", type=int, default=1)
    parser.add_argument("--only", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cfg = load_config(args.model)
    weights = Weights(cfg.path, device="cuda", skip_mtp=False,
                      nvfp4=args.nvfp4, fp8_head=args.fp8_head)
    eng = Qwen38Engine(cfg, weights, max_len=args.max_len, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids
    drafter = DFlash2Drafter(eng, None, blocks=args.dflash2_blocks, path="greedy",
                             max_len=args.max_len)
    drafter._build()
    width = (drafter.cfg.block_size - 1) * args.dflash2_blocks

    settings: list[Relax] = [Relax(tau=float(t)) for t in args.taus.split(",") if t]
    settings += [Relax(rank=int(r)) for r in args.ranks.split(",") if r]

    rows = []
    for name, text in PROMPTS.items():
        if args.only and name != args.only:
            continue
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True),
                  return_tensors="pt").input_ids[0].cuda()
        print(f"\n### {name}  ({ids.numel()} prompt tokens, {args.new} new)")
        print(f"{'rule':<12}{'tok/s':>8}{'acc/blk':>9}{'relaxed':>9}{'NLL/tok':>9}{'dNLL':>9}"
              f"{'uniq-4':>8}{'same as greedy':>16}")
        base_nll = None
        base_out = None
        for rule in settings:
            drafter.attach()
            out, st = generate_spec(eng, ids, args.new, drafter, width, eos, relax=rule)
            drafter.detach()
            full = torch.cat([ids, torch.tensor(out, device=ids.device, dtype=ids.dtype)])
            nll = teacher_forced_nll(eng, full, ids.numel())
            if base_nll is None:
                base_nll, base_out = nll, list(out)
            shared = 0
            for a, b in zip(base_out, out):
                if a != b:
                    break
                shared += 1
            label = f"tau {rule.tau:g}" if rule.tau < 1.0 else (
                f"rank {rule.rank}" if rule.rank > 1 else "lossless")
            print(f"{label:<12}{st.tok_s:>8.2f}{st.accept_len:>9.2f}{st.relaxed:>9d}"
                  f"{nll:>9.4f}{nll - base_nll:>9.4f}{distinct_ngrams(out):>8.3f}"
                  f"{shared / max(len(out), 1) * 100:>15.1f}%")
            rows.append({"workload": name, "rule": label, "tau": rule.tau, "rank": rule.rank,
                         "tok_s": st.tok_s, "accept_len": st.accept_len, "relaxed": st.relaxed,
                         "nll": nll, "dnll": nll - base_nll,
                         "uniq4": distinct_ngrams(out),
                         "prefix_shared": shared / max(len(out), 1),
                         "text": tok.decode(out)})
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(rows, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
