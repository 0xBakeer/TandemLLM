"""SPD-143: how much a drafter could buy on a model, offline: the lookup's acceptance on recorded agentic
conversations of the Chat tab's store, under the model's tokenizer, priced with the model's measured verify
staircase, against ideal drafters of a fixed per-token acceptance. Read-only; prints aggregates only (no text
leaves the store).

    PYTHONPATH=<repo> python tools/lookup_headroom.py --db ~/.tandemllm/chat/chat.sqlite3 --tokenizer <model dir> \
        --stair stair.json --bench bench.json [--conversations 40] [--json out.json]

A turn is one assistant message of a conversation whose assistant wrote tool calls: its target is the text the
assistant produced (reasoning, answer and each call's input, in order), its context the conversation before it
(user text, earlier assistant text and calls, tool outputs) -- the texts as the store keeps them, joined by
blank lines, not the chat template's rendering, and written by the model that served those runs, not by this one
(a proxy for this model's own text). The lookup (engine/drafters/ngram.py: the request's own text, no corpus, no
suffix store) proposes its best branch at every position; a round takes the first k rows of it, k by StairCut's
rule -- the expected tokens (1 + the calibrated chance of each row being right) over the verify price of k + 1
rows, against the plain step -- and commits the rows that match and one more. The calibration is the store's own:
the chance a branch is right down to a row, by the row's score tenth. An ideal drafter of acceptance a proposes k
rows each right with chance a given the ones above it; its best k is the same rule's.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.ngram import NgramDrafter  # noqa: E402
from tools.fn_lookup_sim import lookup_path, match  # noqa: E402


def prices(stair: dict, bench: dict) -> dict:
    ctx = min(stair["rows"], key=int)
    V = {int(r): v["ms"] for r, v in stair["rows"][ctx].items()}
    V[1] = bench["graph"]["ms_token"]
    return V


def vp(V: dict, rows: int) -> float:
    ks = sorted(V)
    if rows in V:
        return V[rows]
    step = (V[ks[-1]] - V[ks[-2]])
    return V[ks[-1]] + step * (rows - ks[-1])


def turns(db: str, n_conv: int, tok, ctx_cap: int, tgt_cap: int):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    convs = [r[0] for r in con.execute(
        "select conversation_id, max(created) m from messages where role='assistant' and parts like '%tool_call%' "
        "group by conversation_id order by m desc limit ?", (n_conv,))]
    for cid in convs:
        rows = con.execute("select role, parts from messages where conversation_id=? order by created", (cid,)).fetchall()
        ctx: list[str] = []
        for role, parts in rows:
            try:
                ps = json.loads(parts)
            except ValueError:
                continue
            texts = []
            for p in ps if isinstance(ps, list) else []:
                t = p.get("type")
                if t in ("text", "thinking"):
                    texts.append(p.get("text") or "")
                elif t == "tool_call":
                    texts.append(f"{p.get('name')} {p.get('input') or ''}")
                    out = (p.get("output") or {}).get("content")
                    if role == "assistant" and out:
                        texts.append("\0" + str(out))          # a tool output: context of the next step, not target
            if role == "assistant":
                tgt = [t for t in texts if not t.startswith("\0")]
                if tgt:
                    c_ids = tok("\n\n".join(ctx), add_special_tokens=False)["input_ids"][-ctx_cap:]
                    t_ids = tok("\n\n".join(tgt), add_special_tokens=False)["input_ids"][:tgt_cap]
                    yield c_ids, t_ids
            ctx += [t.lstrip("\0") for t in texts]


def main() -> None:
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/.tandemllm/chat/chat.sqlite3"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--stair", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--conversations", type=int, default=40)
    ap.add_argument("--ctx-cap", type=int, default=32768)
    ap.add_argument("--tgt-cap", type=int, default=3000)
    ap.add_argument("--max-rows", type=int, default=16)
    ap.add_argument("--json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    V = prices(json.load(open(a.stair)), json.load(open(a.bench)))
    plain = V[1]
    data = list(turns(a.db, a.conversations, tok, a.ctx_cap, a.tgt_cap))
    look = NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)
    # pass 1: calibration by score tenth (the chance a branch is right down to a row)
    hit, n = [0.0] * 10, [0.0] * 10
    paths = []
    for c, t in data:
        look.reset()
        look.prime(c + t[:1])
        per = []
        for i in range(len(t) - 1):
            toks, sc = lookup_path(look, c + t[:i + 1], t[i])
            m = match(toks, t, i)
            per.append((toks, sc, m))
            for j, v in enumerate(sc):
                b = min(int(v * 10), 9)
                hit[b] += j < m
                n[b] += 1
            look.observe(t[i + 1:i + 2])
        paths.append(per)
    cal = [(h + 1) / (k + 2) for h, k in zip(hit, n)]
    # pass 2: the rounds
    tokens = rounds = drafted_rounds = 0
    ms = 0.0
    acc_hist: dict = {}
    for (c, t), per in zip(data, paths):
        i = 0
        while i < len(t) - 1:
            toks, sc, m = per[i]
            q, best_k, best = 1.0, 0, 1.0 / plain
            for k in range(1, min(len(toks), a.max_rows - 1) + 1):
                q_k = 1.0 + sum(_cum(cal, sc, j) for j in range(k))
                v = q_k / vp(V, k + 1)
                if v > best:
                    best, best_k = v, k
            acc = min(best_k, m)
            ms += vp(V, best_k + 1) if best_k else plain
            got = acc + 1
            tokens += got
            rounds += 1
            drafted_rounds += best_k > 0
            acc_hist[acc] = acc_hist.get(acc, 0) + 1
            i += got
    res = {"turns": len(data), "target_tokens": sum(len(t) for _, t in data),
           "context_tokens_mean": round(sum(len(c) for c, _ in data) / max(len(data), 1)),
           "plain_ms": plain, "verify_ms": {k: V[k] for k in sorted(V)},
           "lookup": {"tok_s": round(1e3 * tokens / ms, 1), "plain_tok_s": round(1e3 / plain, 1),
                      "speedup": round((tokens / ms) * plain, 3), "tokens_a_round": round(tokens / rounds, 3),
                      "drafted_round_pct": round(100 * drafted_rounds / rounds, 1),
                      "accepted_hist": {str(k): acc_hist[k] for k in sorted(acc_hist)[:17]}},
           "calibration_by_score_tenth": [round(x, 3) for x in cal], "ideal": {}}
    for acc_p in (0.5, 0.7, 0.9):
        best = (1.0 / plain, 0)
        for k in range(1, a.max_rows):
            e = sum(acc_p ** j for j in range(k + 1))
            best = max(best, (e / vp(V, k + 1), k))
        res["ideal"][str(acc_p)] = {"best_rows": best[1] + 1, "tok_s": round(1e3 * best[0], 1),
                                    "speedup": round(best[0] * plain, 3)}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def _cum(cal: list, sc: list, j: int) -> float:
    """The calibrated chance that rows 0..j of a branch are all right (row j's own score tenth: the scores
    are already the path's)."""
    return cal[min(int(sc[j] * 10), 9)]


if __name__ == "__main__":
    main()
