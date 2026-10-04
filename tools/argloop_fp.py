"""The tool-argument loop guard's calibration: legit writes that must never fire, loops that must.

    python tools/argloop_fp.py --db ~/.local/share/opencode/opencode.db \
        --tokenizer /path/to/tokenizer.json [--json out.json]

Legit: every `write` content and `edit` newString in the opencode database (any model, any
status but the guard's own kind of failure). Loops: the `invalid` tool parts whose text is a
call cut by the token limit, and the text parts of assistant messages that ended on `length` with
an open `<tool_call>`. Each text is tokenized with the served tokenizer and streamed through
`server.argloop.ArgLoopGuard` token by token, exactly as the server feeds it.

Prints counts and numbers only, never a file's text: per set, how many fire under the defaults,
and for the legit set the largest periodic stretch (tokens, repeats) and window ratio any check
saw, so a threshold can be set above them with a margin. For the loops, the token at which the
guard fires.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import argloop  # noqa: E402

OPEN = "<tool_call>"


def legit(db) -> list[tuple[str, str]]:
    out = []
    q = ("select json_extract(data,'$.tool'), data from part "
         "where json_extract(data,'$.type')='tool' and json_extract(data,'$.tool') in "
         "('write','edit','multiedit','patch','apply_patch')")
    for tool, data in db.execute(q):
        try:
            inp = json.loads(data)["state"].get("input") or {}
        except (ValueError, KeyError, AttributeError):
            continue
        for key in ("content", "newString", "new_string", "patch", "patchText"):
            v = inp.get(key)
            if isinstance(v, str) and v:
                out.append((tool, v))
    return out


def loops(db) -> list[tuple[str, str]]:
    out = []
    q = ("select p.data from part p join message m on p.message_id = m.id "
         "where json_extract(m.data,'$.finish')='length' and json_extract(p.data,'$.type')='text'")
    for (data,) in db.execute(q):
        t = json.loads(data).get("text") or ""
        i = t.find(OPEN)
        if i >= 0 and t.find("</tool_call>", i) < 0:
            out.append(("length_text", t[i + len(OPEN):]))
    return out


def stream(tok, text: str, cfg: dict, context_chars: int = 0, every_all: bool = False):
    """Feed `text` token by token; returns (verdict, stats). `every_all`: also record the
    largest periodic stretch and ratio any check saw (the legit set's margins)."""
    enc = tok.encode(text, add_special_tokens=False)
    ends = [e for _, e in enc.offsets]
    g = argloop.ArgLoopGuard(cfg, context_chars)
    stats = {"tokens": len(ends), "chars": len(text), "max_span_tokens": 0,
             "max_span_repeats": 0.0, "max_ratio": 0.0}
    for i, e in enumerate(ends):
        v = g.feed(i + 1, e, lambda e=e: text[:e])
        if every_all and g._offs and len(g._offs) % cfg["every"] == 0 and len(g._offs) >= 256:
            blk = text[:e]
            p, span = argloop.periodic_tail(blk, cfg["max_period"], cfg["probe"])
            if p:
                st = g._tokens_from(len(blk) - span)
                if span / p >= 2 and st > stats["max_span_tokens"]:
                    stats["max_span_tokens"], stats["max_span_repeats"] = st, span / p
            if len(g._offs) >= 4096:
                stats["max_ratio"] = max(stats["max_ratio"],
                                         argloop.ratio(blk[-cfg["window_chars"]:]))
        if v is not None:
            return v, stats
    return None, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/opencode/opencode.db"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE threshold override")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    cfg = dict(argloop.DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(float(v)) if isinstance(cfg[k], int) else float(v)
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rep = {"cfg": cfg, "legit": {}, "loops": []}
    L = legit(db)
    fired, top_span, top_ratio, n_tok = [], [], [], []
    for tool, text in L:
        v, st = stream(tok, text, cfg, every_all=True)
        n_tok.append(st["tokens"])
        top_span.append((st["max_span_tokens"], round(st["max_span_repeats"], 1), st["tokens"]))
        top_ratio.append((round(st["max_ratio"], 2), st["tokens"]))
        if v is not None:
            fired.append(dict(v.as_dict(), tool=tool, total_tokens=st["tokens"]))
    top_span.sort(reverse=True)
    top_ratio.sort(reverse=True)
    n_tok.sort()
    rep["legit"] = {"texts": len(L), "fired": fired, "tokens_max": n_tok[-1] if n_tok else 0,
                    "tokens_p99": n_tok[int(len(n_tok) * 0.99)] if n_tok else 0,
                    "over_1024_tokens": sum(1 for t in n_tok if t >= 1024),
                    "top_periodic_stretch": top_span[:8], "top_window_ratio": top_ratio[:8]}
    for kind, text in loops(db):
        v, st = stream(tok, text, cfg)
        rep["loops"].append({"kind": kind, "tokens": st["tokens"], "chars": st["chars"],
                             "fired": v.as_dict() if v is not None else None})
    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
