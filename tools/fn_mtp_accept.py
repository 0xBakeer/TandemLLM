"""ENG-153 groundwork: how far the MTP layer's greedy draft chain agrees with the target, per depth.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_mtp_accept.py --pack ~/fn2/packs/3.05 \
        --classes prose,code,chat --ctx 1024 --new 192 --depth 6 --json out.json

The target decodes greedy through its decode graph (plain, no speculation). The MTP layer is primed
over the prompt (its own KV cache, as the trunk's streams and the next tokens); then at every
committed token t it is fed the real pair (the trunk's streams at t, the token t + 1) and drafts a
greedy chain of `--depth` tokens for t + 2 .. t + 1 + depth from its own streams (the deeper feeds
rolled back). Drafts are scored against the text the target actually produced: per depth k, the
share of rounds whose first k drafts are all right, and the expected tokens a round of a chain of
depth k (1 + the accepted drafts). Nothing here measures speed; the draft runs eager. The full head
is used for drafting (no hot-vocabulary slice).

`--topk 2,4,16` (SPD-249): the head's top-k at every depth of the chain as well. Per k and depth d, the
share of rounds on the path (drafts 1..d-1 right) whose true token d is in the top-k (`coverage`), and
the share whose chain first misses at d with the true token in the top-k (`miss_in_topk`: what a sibling
or a corrector over the top-k could recover there). `hot_share`: the true tokens inside the served hot
draft vocabulary (the first FN_DRAFT_HOT ids of ops/fn-hot-vocab.npy). `--dump F`: per class the prompt
ids, the greedy text's ids and each round's chain, for tools/fn_lookup_hits.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


def chain_topk(eng: FlashNextEngine, stack_row, next_token: int, st_m: FNState, depth: int, k: int):
    """`FlashNextEngine.mtp_chain` with each depth's top-k ids as well: (greedy chain, [top-k ids a depth])."""
    logits, h = eng.mtp_feed(stack_row, torch.tensor([next_token]), st_m)
    keep = st_m.length
    out, tops = [], []
    for d in range(depth):
        row = logits[-1]
        tops.append([int(t) for t in torch.topk(row.float(), k).indices.tolist()] if k > 1 else [int(row.argmax())])
        out.append(int(row.argmax()))
        if d + 1 < depth:
            logits, h = eng.mtp_feed(h[-1:], torch.tensor([out[-1]]), st_m)
    st_m.length = keep
    st_m.clamp_pool()
    return out, tops


def score(drafts: list, tops: list | None, toks: list, depth: int, ks: tuple = (), hot=None) -> dict:
    """Scores the chains against the greedy text: drafts[i] (and tops[i], the top-k ids a depth) predict
    toks[i + 1], toks[i + 2], ... CPU only."""
    hits = [0] * depth
    rounds = 0
    tokens_round = [0.0] * depth
    on_path = [0] * depth                              # rounds whose drafts 1..d-1 are right
    cov = {k: [0] * depth for k in ks}
    miss = {k: [0] * depth for k in ks}
    for i, d in enumerate(drafts):
        truth = toks[i + 1: i + 1 + depth]
        if len(truth) < depth:
            break
        rounds += 1
        acc = 0
        for j in range(depth):
            if d[j] == truth[j]:
                acc += 1
            else:
                break
        for j in range(depth):
            if acc > j:
                hits[j] += 1
            tokens_round[j] += 1 + min(acc, j + 1)
        if tops is not None:
            for j in range(min(acc + 1, depth)):       # depths on the path: 0..acc (acc = the first miss)
                on_path[j] += 1
                for k in ks:
                    if truth[j] in tops[i][j][:k]:
                        cov[k][j] += 1
                        if j == acc:
                            miss[k][j] += 1
    out = {"rounds": rounds, "accept_at_depth": [round(h / max(rounds, 1), 4) for h in hits],
           "tokens_a_round_chain": [round(t / max(rounds, 1), 3) for t in tokens_round]}
    if tops is not None:
        out["on_path"] = on_path
        out["coverage"] = {k: [round(c / max(n, 1), 4) for c, n in zip(cov[k], on_path)] for k in ks}
        out["miss_in_topk"] = {k: [round(m / max(rounds, 1), 4) for m in miss[k]] for k in ks}
    if hot is not None:
        hs = set(int(t) for t in hot)
        out["hot_share"] = round(sum(1 for t in toks[1:] if t in hs) / max(1, len(toks) - 1), 4)
    return out


@torch.no_grad()
def run_class(eng: FlashNextEngine, ids: list[int], new: int, depth: int, eos: set, ks: tuple = (), hot=None) -> tuple:
    st = eng.new_state(eng.max_len)
    logits = eng.prefill(torch.tensor(ids), st, keep_stack=True)
    S = eng.prompt_stack                               # the prompt's streams [L, hc H]
    L = len(ids)
    st_m = FNState(cap=eng.max_len)
    x = int(logits[0].argmax())                        # x_L
    if L > 1:
        eng.mtp_feed(S[: L - 1], torch.tensor(ids[1:L]), st_m)          # prime over the prompt
    g = DecodeGraph(eng, st)
    g.capture()
    stack_t = S[L - 1: L]
    toks = [x]
    drafts, tops = [], ([] if ks else None)
    kmax = max(ks) if ks else 1
    while len(toks) < new and toks[-1] not in eos:
        nxt = toks[-1]
        if ks:
            d, t = chain_topk(eng, stack_t, nxt, st_m, depth, kmax)   # for tokens after `nxt`
            drafts.append(d)
            tops.append(t)
        else:
            drafts.append(eng.mtp_chain(stack_t, nxt, st_m, depth))
        toks.append(g.greedy(nxt))
        stack_t = eng.stack.clone()
    r = {"prompt_tokens": L, "new_tokens": len(toks)}
    r.update(score(drafts, tops, toks, depth, ks, hot))
    return r, {"prompt": list(ids), "toks": toks, "drafts": drafts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--topk", default="", help="SPD-249: e.g. 2,4,16: the head's top-k coverage a depth")
    ap.add_argument("--hot", type=int, default=32768, help="the hot draft vocabulary's size for hot_share (0: off)")
    ap.add_argument("--dump", help="SPD-249: prompts, greedy texts and chains (JSON) for tools/fn_lookup_hits.py")
    ap.add_argument("--json")
    a = ap.parse_args()
    ks = tuple(int(k) for k in a.topk.split(",") if k)
    hot = None
    if a.hot:
        import numpy as np
        hot = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops",
                                   "fn-hot-vocab.npy"))[: a.hot].tolist()
    from transformers import AutoTokenizer
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=True)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=4096)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "knobs": {"ctx": a.ctx, "new": a.new, "depth": a.depth, "greedy": True,
                                                     "thinking": False, "draft_head": "full (248,320)", "topk": list(ks),
                                                     "hot": a.hot},
           "classes": {}}
    dump = {}
    for c in a.classes.split(","):
        r, d = run_class(eng, class_prompt(tok, c, a.ctx), a.new, a.depth, eos, ks, hot)
        res["classes"][c] = r
        dump[c] = d
        print(f"[mtp] {c}: {json.dumps(r)}", flush=True)
    if a.dump:
        json.dump({"depth": a.depth, "classes": dump}, open(a.dump, "w"))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
