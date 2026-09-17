"""The gate an architecture edit has to pass: held-out loss, argmax agreement, free generation.

Teacher-forced loss on its own is not a quality gate, and the reason is written down in
notes/ARCHITECTURE.md 1.9: it never lets an error compound, so it cannot see a model that fails to
stay on its own trajectory. It rated a degenerate configuration better than a healthy one on the
earlier engine on this board and cost a day. So this tool reports three things and a change is only
accepted if all three hold:

  1. **held-out NLL**, on prose and on code separately, teacher-forced against the FP8 engine's own
     numbers on exactly the same tokens. The threshold is a delta, not an absolute: the absolute
     value belongs to the text.
  2. **argmax agreement** on the same positions, reported both overall and restricted to positions
     where the FP8 engine is confident, because on a flat distribution over 248,320 tokens the
     argmax is decided by rounding.
  3. **a free generation** of several hundred tokens on prompts from domains the calibration corpus
     never contained, with the repetition statistics printed, because that is the only one of the
     three that can see the failure mode that matters.

Two passes over the same tokens, one per configuration, in one process, so the comparison is not
across two allocation histories.

    python tools/quality_gate.py --nvfp4 ~/nvfp4/mlp-rtn.safetensors --tokens 2048
    python tools/quality_gate.py --nvfp4 ~/nvfp4/mlp-rtn.safetensors --gen 900
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = {
    "prose": os.path.join(HERE, "bench", "heldout_prose.txt"),
    "code": os.path.join(HERE, "bench", "heldout_code.txt"),
}

# Prompts for the generation gate. None of these domains is in bench/calib.txt, which is Python
# source and package documentation: the point is to generate outside the calibration set.
GEN_PROMPTS = [
    ("recipe", "Explain, in one continuous passage of prose, how a lock and a weir differ and why "
               "a canal needs both."),
    ("letter", "Write a short formal letter to a landlord reporting a broken heating system and "
               "asking for a repair date."),
    ("sql", "Write a SQL schema for a lending library with members, titles, copies and loans, then "
            "one query that lists the overdue loans."),
    ("german", "Erklaere in einfachem Deutsch, warum ein Zug beim Bremsen laenger braucht als ein "
               "Auto."),
    ("maths", "Prove that the square root of two is irrational, writing the argument out in full "
              "sentences."),
]


def teacher_forced(engine: Qwen38Engine, ids: torch.Tensor, chunk: int) -> tuple:
    """Returns (sum NLL, count, argmax [T-1], logit gap [T-1]) over the whole sequence."""
    nll = 0.0
    n = 0
    argmax = []
    gap = []
    engine.reset()
    start = 0
    for c0 in range(0, ids.numel() - 1, chunk):
        piece = ids[c0:c0 + chunk]
        logits = engine.forward(piece.to(engine.device), start=start)[0].float()
        start += piece.numel()
        tgt = ids[c0 + 1:c0 + 1 + piece.numel()].to(engine.device)
        m = tgt.numel()
        lp = F.log_softmax(logits[:m], dim=-1)
        nll += float(-lp.gather(1, tgt[:, None]).sum())
        n += m
        top2 = logits[:m].topk(2, dim=-1)
        argmax.append(top2.indices[:, 0].cpu())
        gap.append((top2.values[:, 0] - top2.values[:, 1]).cpu())
        del logits, lp, top2
    return nll, n, torch.cat(argmax), torch.cat(gap)


def generate(engine: Qwen38Engine, tok, prompt: str, new: int) -> str:
    msg = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=False)[0]
    engine.reset()
    out = []
    logits = engine.forward(ids.to(engine.device), start=0, last_only=True)[0, -1]
    pos = ids.numel()
    for _ in range(new):
        nxt = int(logits.argmax())
        out.append(nxt)
        if nxt == tok.eos_token_id:
            break
        logits = engine.forward(torch.tensor([nxt], device=engine.device), start=pos,
                                last_only=True)[0, -1]
        pos += 1
    return tok.decode(out, skip_special_tokens=True)


def repetition(text: str) -> dict:
    words = text.split()
    if len(words) < 20:
        return {"words": len(words), "uniq4": 1.0, "longest_repeat": 0}
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    uniq = len(set(grams)) / len(grams)
    longest = 0
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        longest = max(longest, run)
    return {"words": len(words), "uniq4": uniq, "longest_repeat": longest}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--nvfp4", required=True, help="NVFP4 MLP file or snapshot directory")
    ap.add_argument("--tokens", type=int, default=2048, help="held-out tokens per corpus")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=0, help="free-generation length, 0 to skip")
    ap.add_argument("--baseline", default="fp8", choices=("fp8", "none"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = load_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)

    corpus = {}
    for name, path in CORPORA.items():
        ids = tok(open(path).read(), return_tensors="pt").input_ids[0][: args.tokens]
        corpus[name] = ids
        print(f"[corpus] {name:5s} {ids.numel()} tokens from {os.path.basename(path)}")

    results: dict[str, dict] = {}
    ref_argmax: dict[str, torch.Tensor] = {}
    ref_gap: dict[str, torch.Tensor] = {}

    for tag in (["fp8", "nvfp4"] if args.baseline == "fp8" else ["nvfp4"]):
        w = Weights(cfg.path, skip_mtp=True, nvfp4=args.nvfp4 if tag == "nvfp4" else None)
        eng = Qwen38Engine(cfg, w, max_len=max(args.chunk, 4096) + 64)
        print(f"\n=== {tag} === {w.report()}")
        res = {}
        for name, ids in corpus.items():
            t0 = time.time()
            nll, n, am, gp = teacher_forced(eng, ids, args.chunk)
            res[name] = {"nll": nll / n, "n": n}
            if tag == "fp8":
                ref_argmax[name], ref_gap[name] = am, gp
            else:
                if name in ref_argmax:
                    agree = (am == ref_argmax[name]).float()
                    conf = ref_gap[name] >= 1.0
                    res[name]["agree"] = float(agree.mean())
                    res[name]["agree_conf"] = float(agree[conf].mean()) if conf.any() else float("nan")
                    res[name]["n_conf"] = int(conf.sum())
            print(f"  {name:5s} NLL {res[name]['nll']:.4f} over {n} tokens "
                  f"({time.time() - t0:.1f} s)")
        if args.gen:
            res["gen"] = {}
            for gname, prompt in GEN_PROMPTS:
                t0 = time.time()
                text = generate(eng, tok, prompt, args.gen)
                st = repetition(text)
                res["gen"][gname] = st
                print(f"  gen/{gname:7s} {st['words']:4d} words  uniq-4gram {st['uniq4']:.3f}  "
                      f"longest word repeat {st['longest_repeat']}  ({time.time() - t0:.1f} s)")
                print(f"      {text[:160]!r}")
        results[tag] = res
        del eng, w
        torch.cuda.empty_cache()

    if "fp8" in results:
        print("\n--- delta, NVFP4 MLPs against FP8, same tokens")
        for name in corpus:
            a = results["fp8"][name]["nll"]
            b = results["nvfp4"][name]["nll"]
            r = results["nvfp4"][name]
            print(f"  {name:5s} NLL {a:.4f} -> {b:.4f}   delta {b - a:+.4f} nats   "
                  f"argmax {r.get('agree', float('nan')):.4f} "
                  f"(confident positions {r.get('agree_conf', float('nan')):.4f}, "
                  f"n={r.get('n_conf', 0)})")
        worst = max(results["nvfp4"][n]["nll"] - results["fp8"][n]["nll"] for n in corpus)
        print(f"  worst delta {worst:+.4f} nats   gate <= +0.05   "
              f"{'PASS' if worst <= 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
