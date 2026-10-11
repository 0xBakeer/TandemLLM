"""ENG-227's proof on the board: greedy speculative output == plain greedy output, token for token.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_spec_check.py --pack ~/fn2/packs/3.05 \
        --classes prose,code,chat --ctx 1024 --new 256 --modes oracle,oracle-eager,mtp --json out.json

Per class (tools/fn_generate.py's prompts at `--ctx`): the plain greedy text through the captured
decode step (`DecodeGraph`), then the same prompt through `SpecDecoder` with each draft source:

  oracle        trees built from the plain text with planted errors: a true chain of 0..`--depth`
                tokens, a wrong sibling subtree before a true child at random (so accepted rows sit
                past their depth and the commit moves them), a wrong token after the chain. Any
                drafter must give the plain text; this one makes long accepted paths through moved
                tree rows. Verify blocks from captured graphs.
  oracle-eager  the same trees, verify eager (no graphs)
  mtp           the pack's MTP layer as a greedy chain drafter of `--depth` (eager drafting)
  mtp-graph     the same chain from captured graphs (`MTPGraphDrafter`); its drafts should match
                `mtp`'s closely (the fused trunk's arithmetic in place of the reference's)

Recorded per class and mode: equal (the speculative text is the plain one), the first differing
index, rounds, rows verified, accepted drafts, rounds whose path was moved, and for `mtp` the
tokens a round. Speeds here are not measurements (the drafters are eager).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.models.flashnext.spec import Drafter, MTPChainDrafter, MTPGraphDrafter, SpecDecoder  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


class OracleDrafter(Drafter):
    """Trees from a known continuation `truth` (truth[0] = the first anchor) with planted errors."""

    def __init__(self, truth: list[int], depth: int, vocab: int, seed: int):
        self.truth, self.depth, self.vocab = truth, depth, vocab
        self.rng = random.Random(seed)
        self.i = 0                                    # the anchor's index in truth

    def _wrong(self, avoid: int) -> int:
        while True:
            t = self.rng.randrange(self.vocab)
            if t != avoid:
                return t

    def propose(self, anchor: int):
        P, i, rng = self.truth, self.i, self.rng
        toks, par = [anchor], [-1]

        def add(p, t):
            toks.append(t)
            par.append(p)
            return len(toks) - 1
        a = rng.randint(0, min(self.depth, len(P) - 1 - i))
        cur = 0
        for j in range(1, a + 1):
            if rng.random() < 0.4:
                w = add(cur, self._wrong(P[i + j]))
                if rng.random() < 0.5:
                    add(w, self._wrong(-1))
            cur = add(cur, P[i + j])
        nxt = P[i + a + 1] if i + a + 1 < len(P) else -1
        add(cur, self._wrong(nxt))
        return toks, par

    def accepted(self, stack, nexts):
        self.i += len(nexts)


def plain(eng: FlashNextEngine, ids: list[int], new: int, eos: set) -> tuple[list[int], float]:
    st = eng.new_state(eng.max_len)
    logits = eng.prefill(torch.tensor(ids), st)
    g = DecodeGraph(eng, st)
    g.capture()
    out = [int(logits[0].argmax())]
    torch.cuda.synchronize()
    t0 = time.time()
    while len(out) < new and out[-1] not in eos:
        out.append(g.greedy(out[-1]))
    return out, (len(out) - 1) / (time.time() - t0)


def spec(eng: FlashNextEngine, ids: list[int], new: int, eos: set, mode: str, truth: list[int], depth: int,
         seed: int) -> tuple[list[int], dict]:
    st = eng.new_state(eng.max_len)
    logits = eng.prefill(torch.tensor(ids), st)
    first = int(logits[0].argmax())
    if mode.startswith("oracle"):
        dr = OracleDrafter(truth, depth, eng.cfg.vocab_size, seed)
    else:
        dr = (MTPGraphDrafter if mode == "mtp-graph" else MTPChainDrafter)(eng, depth, eng.max_len)
        dr.prime(eng.stack.clone(), ids, first)
    sd = SpecDecoder(eng, st, dr, graphs=mode != "oracle-eager")
    out = sd.generate(first, new, eos)
    s = dict(sd.stats)
    s["tokens_a_round"] = round((len(out) - 1) / max(s["rounds"], 1), 3)
    for k in ("draft_s", "verify_s", "commit_s"):
        s[k] = round(s[k], 2)
    return out, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--modes", default="oracle,oracle-eager,mtp,mtp-graph")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    modes = a.modes.split(",")
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=any(m.startswith("mtp") for m in modes))
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=4096)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": a.ctx, "new": a.new, "depth": a.depth, "modes": modes, "seed": a.seed, "greedy": True,
                     "thinking": False, "kv_dtype": "bf16", "max_len": 4096}, "classes": {}}
    sc = eng.self_check(class_prompt(tok, "chat", 0))
    res["self_check"] = sc
    for c in a.classes.split(","):
        ids = class_prompt(tok, c, a.ctx)
        P, tps = plain(eng, ids, a.new, eos)
        row = {"prompt_tokens": len(ids), "plain_tokens": len(P), "plain_tok_s": round(tps, 2), "modes": {}}
        for m in modes:
            S, s = spec(eng, ids, a.new, eos, m, P, a.depth, a.seed)
            diff = next((i for i in range(min(len(S), len(P))) if S[i] != P[i]), None)
            if diff is None and len(S) != len(P):
                diff = min(len(S), len(P))
            s.update(equal=S == P, first_diff=diff, tokens=len(S))
            row["modes"][m] = s
            print(f"[spec] {c} {m}: {json.dumps(s)}", flush=True)
        row["text_head"] = tok.decode(P[:40], skip_special_tokens=True)
        res["classes"][c] = row
    res["mem_end_gib"] = mem_available_gib()
    res["all_equal"] = all(m["equal"] for r in res["classes"].values() for m in r["modes"].values())
    print(f"[spec] all_equal={res['all_equal']}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
