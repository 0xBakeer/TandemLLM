"""The tool-argument loop guard on the live engine: one opencode-shaped turn whose `write` is
asked to repeat one map row hundreds of times (the 2026-10-04 shape, on purpose), then the next
turn of the same session.

    python tools/argloop_live.py --base http://127.0.0.1:8000 --key d7-test [--admin TOKEN]

Prints JSON: the first turn's finish, the stop sentence, tokens and seconds; whether a call
reached the client; the second turn's monitor rows (A1's GIVE_UP on the first request, A1's
PROMPT_TAIL_NOTE on the second). Streams, as opencode does. The ask is a legit request whose
honest answer IS a periodic write, so the guard firing here is the mechanism, not a false
positive measurement (that is tools/argloop_fp.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

UA = "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
ROW = '"' + " " * 94 + '",'
WRITE = {"type": "function", "function": {
    "name": "write", "description": "Write a file to the local filesystem.",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string", "description": "absolute path"},
        "content": {"type": "string", "description": "the file's content"}},
        "required": ["filePath", "content"]}}}
READ = {"type": "function", "function": {
    "name": "read", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}},
                   "required": ["filePath"]}}}


def system_text(chars: int = 24000) -> str:
    """A system text of `chars` characters (the session is keyed by `x-session-id`)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = "You are opencode, an interactive CLI coding agent.\n<env>\n  Working directory: " \
           "/tmp/d7-guard\n</env>\n\n"
    with open(os.path.join(here, "docs", "server.md")) as f:
        text += f.read()[:chars]
    return text


def stream(base, key, body, sid):
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": UA, "x-session-id": sid, "X-Tandem-Profile": "coding"})
    out = {"content": "", "reasoning": "", "calls": 0, "finish": None, "id": None}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: {"):
                continue
            ch = json.loads(ln[6:])
            out["id"] = out["id"] or ch.get("id")
            if ch.get("usage"):
                out["usage"] = ch["usage"]
            for c in ch.get("choices") or ():
                d = c.get("delta") or {}
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
                out["calls"] += sum(1 for tc in d.get("tool_calls") or () if tc.get("id"))
                out["finish"] = c.get("finish_reason") or out["finish"]
    out["s"] = round(time.time() - t0, 1)
    return out


def events(base, admin, since_ms):
    h = {"Authorization": f"Bearer {admin}"} if admin else {}
    req = urllib.request.Request(f"{base}/v1/conductor/events?since={since_ms}&limit=500",
                                 headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        got = json.loads(r.read())
    rows = got.get("events") or got.get("rows") or []
    return [{k: e.get(k) for k in ("request_id", "task", "decision", "reason", "detail")}
            for e in rows if e.get("task") == "A1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="d7-test")
    ap.add_argument("--admin", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--system-chars", type=int, default=24000,
                    help="keep it small on a box someone is using")
    a = ap.parse_args()
    sid = f"d7-argloop-{int(time.time())}"
    since = int(time.time() * 1000)
    user = (f"Use the write tool to create /tmp/d7-guard/level.js. It must contain `const LEVEL "
            f"= [` on the first line, then exactly {a.rows} lines that are each exactly {ROW} "
            f"(a double quote, 94 spaces, a double quote, a comma), then `];`. Write every one "
            f"of the {a.rows} lines out in full in the content argument; do not generate them "
            f"with code. Call the tool now.")
    msgs = [{"role": "system", "content": system_text(a.system_chars)}, {"role": "user", "content": user}]
    body = {"model": "TandemLLM", "messages": msgs, "tools": [WRITE, READ], "stream": True,
            "max_tokens": 65536, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    first = stream(a.base, a.key, body, sid)
    rep = {"session": sid, "first": {k: first.get(k) for k in ("id", "finish", "calls", "s",
                                                                 "usage")},
           "stop_sentence": first["content"].strip()[-400:]}
    msgs2 = msgs + [{"role": "assistant", "content": first["content"]},
                    {"role": "user", "content": "Go on."}]
    second = stream(a.base, a.key, dict(body, messages=msgs2, max_tokens=4096), sid)
    rep["second"] = {k: second.get(k) for k in ("id", "finish", "calls", "s")}
    try:
        rep["a1_rows"] = events(a.base, a.admin, since)
    except Exception as exc:                                       # noqa: BLE001
        rep["a1_rows"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(rep, indent=1))
    ok = ("[TandemLLM stopped the" in first["content"] and first["calls"] == 0
          and first["finish"] == "stop")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
