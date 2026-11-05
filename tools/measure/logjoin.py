"""The engine log's `[req]` lines, by request id: the slot a request ran on (`slot=`), its passes
(`blocks=`), committed tokens and decode time, and whether it drafted (`accept=` other than `-`).
Read only from a local file (`--engine-log`, or the launched server's own log)."""

from __future__ import annotations

import re

REQ = re.compile(r"\[req\] (\S+) (?:stream|json) prompt=(\d+) completion=(\d+)")
FIELD = re.compile(r" (slot|blocks|committed|decode_ms|accept)=(\S+)")


def parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = REQ.search(line)
        if not m:
            continue
        f = dict(FIELD.findall(line[m.end():]))
        rec = {"prompt": int(m.group(2)), "completion": int(m.group(3))}
        if "slot" in f:
            rec["slot"] = int(f["slot"])
        if "blocks" in f and "committed" in f:
            b, c = int(f["blocks"]), int(f["committed"])
            rec["blocks"], rec["committed"] = b, c
            rec["tokens_per_pass"] = round(c / b, 2) if b else None
        if "decode_ms" in f and "committed" in f:
            ms = float(f["decode_ms"])
            rec["engine_decode_tok_s"] = round(int(f["committed"]) / (ms / 1e3), 2) if ms > 0 else None
        if "accept" in f:
            rec["drafting"] = f["accept"] != "-"
        out[m.group(1)] = rec
    return out


def enrich(rows: list[dict], log: dict) -> int:
    """Each row's `log`: the [req] records of its request ids. Returns the rows joined."""
    n = 0
    for r in rows:
        got = [log[i] for i in r.get("ids") or [] if i in log]
        if got:
            r["log"] = got
            n += 1
    return n
