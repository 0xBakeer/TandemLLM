"""SPD-241: agreement of Flash-Next's decode and verify rows with the mixers on tensor cores (QWEN38_FN_MIXER_MMA)
against the CUDA-core mixers, one run of each (separate processes: each captures its own graphs).

    PYTHONPATH=<repo>:~/fn-ref/site QWEN38_FN_MIXER_MMA=0 python tools/fn_mma_agree.py --pack P --out base.pt
    PYTHONPATH=<repo>:~/fn-ref/site QWEN38_FN_MIXER_MMA=1 python tools/fn_mma_agree.py --pack P --against base.pt

Per prompt (`--classes` at `--ctx` tokens, the prefill unchanged by the knob): `--new` greedy tokens from the
decode step (one row a call), then the same tokens again as chain blocks of `--block` rows (the verify, every row
committed). The base run writes every row's logits (fp16) and its tokens; the run `--against` it follows the base
run's tokens and reports, per path, top-1 agreement with the base, mean and p99 of KL(base || this) and the max
logit difference; and in each run whether the verify rows equal the decode rows bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.serving import FlashNextServing, build  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


def rows(s, ids, toks, block: int):
    """(decode rows, verify rows) [len(toks), vocab] fp32 of `toks` after the prompt `ids`, on fresh states."""
    n = ids.numel()
    dec = [s.forward(ids, 0).view(1, -1).float().clone()]
    for i, t in enumerate(toks[:-1]):
        dec.append(s.forward(torch.tensor([t], device="cuda"), n + i).view(1, -1).float().clone())
    s.reset()
    first = s.forward(ids, 0).view(1, -1).float().clone()
    ver, pos = [first], n
    i = 0
    while i < len(toks) - 1:
        blk = toks[i:i + block]
        lg = s.forward_block(torch.tensor(blk, device="cuda"), pos).float().clone()
        s.rollback_to(len(blk))
        ver.append(lg)
        pos += len(blk)
        i += len(blk)
    ver = torch.cat(ver)[:len(toks)]
    return torch.cat(dec), ver


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--out", help="the base run: write tokens and rows here")
    ap.add_argument("--against", help="follow this base run's tokens and compare with its rows")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    L = a.ctx + a.new + 1024
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda m: print(m, flush=True))
    base = torch.load(a.against) if a.against else None
    saved, report = {}, {"mma": os.environ.get("QWEN38_FN_MIXER_MMA", "0"), "classes": {}}
    for cls in a.classes.split(","):
        ids = torch.tensor(class_prompt(tok, cls, a.ctx), device="cuda")
        if base is None:
            s = FlashNextServing(eng.fn, L)
            x = s.forward(ids, 0)
            toks = [int(x.view(-1).argmax())]
            for i in range(a.new - 1):
                x = s.forward(torch.tensor([toks[-1]], device="cuda"), ids.numel() + i)
                toks.append(int(x.view(-1).argmax()))
        else:
            toks = base[cls]["toks"]
        s = FlashNextServing(eng.fn, L)
        dec, ver = rows(s, ids, toks, a.block)
        r = {"verify_equals_decode": bool(torch.equal(dec, ver))}
        if base is None:
            saved[cls] = {"toks": toks, "dec": dec.half().cpu(), "ver": ver.half().cpu()}
        else:
            for path, mine in (("decode", dec), ("verify", ver)):
                ref = base[cls][path[:3]].float().cuda()
                lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(mine.half().float(), -1)
                kl = (lp.exp() * (lp - lq)).sum(-1)
                r[path] = {"top1": round(float((ref.argmax(-1) == mine.argmax(-1)).float().mean()), 4),
                           "kl_mean": float(kl.mean()), "kl_p99": float(torch.quantile(kl, 0.99)),
                           "max_logit_diff": round(float((ref - mine.half().float()).abs().max()), 4)}
        report["classes"][cls] = r
        print(f"[agree] {cls}: {r}", flush=True)
    if base is None and a.out:
        torch.save(saved, a.out)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
