"""SPD-249 on the CPU: how often the lookup drafter (engine/drafters/ngram.py, as the server builds it for
Flash-Next) proposes a chain whose first tokens are the true continuation, over logged texts.

    PYTHONPATH=<repo> python tools/fn_lookup_hits.py --pack ~/flashnext-packs/3.05 --corpus <corpus dir> \
        --mtp-dump mtp.json --texts c8.json,c8code.json --json out.json

Inputs: `--mtp-dump` (tools/fn_mtp_accept.py --dump: per class the prompt ids, the greedy text's ids and the
MTP chain drafted at every position) and `--texts` (bench JSONs with "prompts" and "texts" per row: the
prompt and the sampled completion, tokenized here with the pack's tokenizer). For every position i of a
completion (the committed context: the prompt and the completion up to and including token i) the drafter,
primed with the prompt and fed the completion as it commits, is asked for its candidates (depth 16):

  fired       the share of positions with any candidate;
  hit2_top    the share whose best candidate's first 2 tokens are the next 2 true tokens (hitN_top for N);
  hit2_any    the same for any of the best `branch_top_k` (3) candidates;
  len_top     the mean matched length of the best candidate where it fired;
  source      local (the request's own text) / corpus / both, where it fired.

With MTP chains: `beyond_mtp` = the share of positions where the best candidate matches 2+ tokens while the
MTP chain (depth `--mtp-depth`) matches fewer, and `gain_tokens` = the mean of max(0, lookup's matched
length - MTP's) a position (both capped at `--cap`). `rounds`: the same counted only at the positions a
chain-only round would verify from (each round advances 1 + the MTP chain's accepted tokens, depth
`--mtp-depth`). Nothing here measures speed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.ngram import NgramDrafter  # noqa: E402


def matched(chain: list[int], truth: list[int]) -> int:
    n = 0
    for a, b in zip(chain, truth):
        if a != b:
            break
        n += 1
    return n


def scan(drafter: NgramDrafter, prompt: list[int], comp: list[int], mtp: list | None = None,
         mtp_depth: int = 2, cap: int = 16, depth: int = 16, ns: tuple = (2, 3, 4)) -> dict:
    """One completion: per position i (the context ends at comp[i]) the drafter's candidates against
    comp[i + 1:]. `mtp[i]`: the MTP chain drafted after comp[i] (the same positions), or None."""
    drafter.reset()
    drafter.prime(list(prompt) + list(comp[:1]))
    ctx = list(prompt) + list(comp[:1])
    st = {"positions": 0, "fired": 0, "hit_top": {n: 0 for n in ns}, "hit2_any": 0, "len_top": 0,
          "source": {"local": 0, "corpus": 0, "both": 0}, "beyond_mtp": 0, "gain_tokens": 0,
          "rounds": 0, "rounds_fired": 0, "rounds_hit2": 0, "rounds_beyond_mtp": 0, "mtp_len": 0}
    next_round = 0
    for i in range(len(comp) - 1):
        truth = comp[i + 1: i + 1 + cap]
        if len(truth) < 2:
            break
        _, cands = drafter.candidates(ctx, depth)
        st["positions"] += 1
        top = cands[0][0] if cands else []
        lt = matched(top, truth) if cands else 0
        if cands:
            st["fired"] += 1
            st["source"][drafter.last_source] = st["source"].get(drafter.last_source, 0) + 1
            st["len_top"] += lt
            for n in ns:
                st["hit_top"][n] += lt >= n
            st["hit2_any"] += any(matched(c, truth) >= 2 for c, _ in cands[:drafter.branch_top_k])
        lm = None
        if mtp is not None and i < len(mtp):
            lm = matched(mtp[i][:mtp_depth], truth)
            st["mtp_len"] += lm
            if lt >= 2 and lm < 2:
                st["beyond_mtp"] += 1
            st["gain_tokens"] += max(0, min(lt, cap) - lm)
        if i == next_round:
            st["rounds"] += 1
            st["rounds_fired"] += bool(cands)
            st["rounds_hit2"] += lt >= 2
            if lm is not None:
                st["rounds_beyond_mtp"] += lt >= 2 and lm < 2
            next_round = i + 1 + (lm if lm is not None else 0)
        ctx.append(comp[i + 1])
        drafter.observe([comp[i + 1]])
    return st


def merge(a: dict, b: dict) -> dict:
    out = {}
    for k in set(a) | set(b):
        x, y = a.get(k, 0), b.get(k, 0)
        out[k] = merge(x or {}, y or {}) if isinstance(x, dict) or isinstance(y, dict) else x + y
    return out


def summary(st: dict, with_mtp: bool) -> dict:
    p, f = max(1, st["positions"]), max(1, st["fired"])
    out = {"positions": st["positions"], "fired": round(st["fired"] / p, 4),
           **{f"hit{n}_top": round(v / p, 4) for n, v in sorted(st["hit_top"].items())},
           "hit2_any": round(st["hit2_any"] / p, 4), "len_top": round(st["len_top"] / f, 3),
           "source": {k: round(v / f, 3) for k, v in st["source"].items()},
           "rounds": st["rounds"], "rounds_fired": round(st["rounds_fired"] / max(1, st["rounds"]), 4),
           "rounds_hit2": round(st["rounds_hit2"] / max(1, st["rounds"]), 4)}
    if with_mtp:
        out.update({"mtp_len": round(st["mtp_len"] / p, 3), "beyond_mtp": round(st["beyond_mtp"] / p, 4),
                    "gain_tokens": round(st["gain_tokens"] / p, 4),
                    "rounds_beyond_mtp": round(st["rounds_beyond_mtp"] / max(1, st["rounds"]), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--corpus", default="", help="the corpus directory the server loads (empty: none)")
    ap.add_argument("--mtp-dump", default="")
    ap.add_argument("--texts", default="", help="comma-separated bench JSONs (rows with prompts and texts)")
    ap.add_argument("--mtp-depth", type=int, default=2, help="the MTP chain's depth a round (8 lanes: 2, one lane: 4)")
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from engine.tokfp import fingerprint
    tok = AutoTokenizer.from_pretrained(a.pack)
    # the server's (engine/models/flashnext/serving.py build)
    drafter = NgramDrafter(corpus_path=a.corpus, tokenizer_sha=fingerprint(a.pack), min_order=3, max_depth=16,
                           min_corpus_order=8)
    res = {"knobs": {"corpus": a.corpus if drafter.corpus is not None else None, "min_order": 3,
                     "min_corpus_order": 8, "mtp_depth": a.mtp_depth, "cap": a.cap}, "sets": {}}
    if a.mtp_dump:
        d = json.load(open(a.mtp_dump))
        for c, x in d["classes"].items():
            st = scan(drafter, x["prompt"], x["toks"], x["drafts"], a.mtp_depth, a.cap)
            res["sets"][f"mtp:{c}"] = summary(st, True)
            print(f"[lookup] mtp:{c}: {json.dumps(res['sets'][f'mtp:{c}'])}", flush=True)
    for f in [x for x in a.texts.split(",") if x]:
        d = json.load(open(f))
        tot = None
        for r in d["rows"]:
            if r.get("warm"):
                continue
            for p, t in zip(r.get("prompts") or [], r.get("texts") or []):
                pi = tok(p, add_special_tokens=False)["input_ids"]
                ci = tok(t, add_special_tokens=False)["input_ids"]
                st = scan(drafter, pi, ci, None, a.mtp_depth, a.cap)
                tot = st if tot is None else merge(tot, st)
        if tot is not None:
            name = os.path.basename(f)
            res["sets"][name] = summary(tot, False)
            print(f"[lookup] {name}: {json.dumps(res['sets'][name])}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
