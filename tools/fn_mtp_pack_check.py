"""SPD-235 on the board: the packed MTP draft (`FlashNextEngine.mtp_chain_multi`, the round runner's
QSE_RUNNER_MTP_PACK) against each head's own graph draft.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_mtp_pack_check.py --pack ~/flashnext-packs/3.05 \\
        --tokens 900,1000,1100,1200,1300,1400,1500,1600 --rounds 8 [--arch qwen3_5_moe]

One engine, one serving slot a prompt for its prefill (the trunk's streams of the prompt, `keep_stack`), and
two MTP heads a slot primed with the same rows: each round every lane feeds both heads the same 1..depth + 1
rows of its prompt's streams with their next tokens; the lone twins draft from their graphs one after the
other, the packed twins together. Per round: whether every lane's drafted tokens (top-2 a depth) are its lone
twin's, and the largest probability difference; then the time of both (a draft only proposes, so tokens that
agree at near ties are a speed matter, not a correctness one).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import draft_meet  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", default="900,1000,1100,1200")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--arch", default="flashnext", choices=("flashnext", "qwen3_5_moe"))
    a = ap.parse_args()
    if a.arch == "qwen3_5_moe":
        from engine.models.qwen3_5_moe.serving import Qwen35MoeServing as Serving, build
    else:
        from engine.models.flashnext.serving import FlashNextServing as Serving, build
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    lens = [int(x) for x in a.tokens.split(",")]
    L = max(lens) + 4096
    eng, _, _ = build(a.pack, L, tok, spec=True, depth=a.depth, log=lambda s: print(s, flush=True))
    fn = eng.fn
    prompts = [torch.tensor(class_prompt(tok, ("prose", "code", "chat")[i % 3], n), device="cuda")
               for i, n in enumerate(lens)]
    slots = [eng] + [Serving(fn, L) for _ in prompts[1:]]
    lone, packed, streams = [], [], []
    for s, ids in zip(slots, prompts):
        fn.prefill(ids, s.st, keep_stack=True)
        rows = fn.prompt_stack.clone() if getattr(fn, "prompt_stack", None) is not None else s.last_streams()
        T = ids.numel()
        h1, h2 = s.add_mtp_head(a.depth), Serving(fn, L).add_mtp_head(a.depth)
        k = T // 2                                       # prime both on the first half, feed from there
        for h in (h1, h2):
            h.prime(rows[:k], ids[1:k + 1].tolist())
        lone.append(h1)
        packed.append(h2)
        streams.append((rows, ids, k))
    same_all, worst, t_lone, t_pack = True, 0.0, 0.0, 0.0
    for r in range(a.rounds):
        feeds = []
        for i, (rows, ids, k) in enumerate(streams):
            n = 1 + (r + i) % (a.depth + 1)
            feeds.append((rows[k:k + n], ids[k + 1:k + n + 1].tolist()))
            streams[i] = (rows, ids, k + n)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs_l = []
        for h, (rw, nx) in zip(lone, feeds):
            h.launch(rw, nx)
            outs_l.append(h.read())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        got = []
        draft_meet.MEET = lambda head: (got.append(head) or True)
        try:
            for h, (rw, nx) in zip(packed, feeds):
                h.launch(rw, nx)
        finally:
            draft_meet.MEET = None
        type(got[0]).run_packed(got)
        outs_p = [h.read() for h in packed]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if r:
            t_lone, t_pack = t_lone + t1 - t0, t_pack + t2 - t1
        same = [ol[0] == op[0] for ol, op in zip(outs_l, outs_p)]
        top1 = [[x[0] for x in ol[0]] == [x[0] for x in op[0]] for ol, op in zip(outs_l, outs_p)]
        diff = max(abs(pa - pb) for ol, op in zip(outs_l, outs_p) for ra, rb in zip(ol[1], op[1])
                   for pa, pb in zip(ra, rb))
        worst = max(worst, diff)
        same_all &= all(top1)
        print(f"[mtp-pack] round {r}: top-2 equal {same}, top-1 equal {top1}, max prob diff {diff:.4f}", flush=True)
    n = max(1, a.rounds - 1)
    print(f"[mtp-pack] {len(prompts)} lanes, depth {a.depth}: top-1 all equal {same_all}, worst prob diff "
          f"{worst:.4f}; lone {t_lone / n * 1e3:.2f} ms a round, packed {t_pack / n * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
