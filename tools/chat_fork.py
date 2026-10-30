"""A recorded Chat-tab conversation cut at a step and run on from there by the live model, for a
few steps only (SRV-667 C).

A bug a long run met at step N (a stub copied as content, a shader error the model could not
read, a stall on the wrong files) is checked at that step without the hours before it: the
conversation is replayed through a chatd of this tree up to the step before N
(tools/chat_trace.py, the fake engine, seconds), and from step N on each model call goes to the
live engine with the request chatd builds, at the chat's own settings. The model's calls run as
in the replay: file tools on the replay's scratch workspace (the files as the record left them),
the recorded tools (render_page, the web, the library) and any bash command the record does not
hold answer that they did not run. At most `--steps` model calls (default 3, at most 12), then
the run ends; `--samples` runs it again that many times (the sampling's spread).

    python tools/chat_fork.py <trace.json> --at TURN:STEP [--steps 3] [--samples 1]
        [--url http://127.0.0.1:8000] [--patch result.txt] [--store DIR] [--out fork.jsonl]
        [--max-tokens 8192]

`--patch`: a file whose text replaces the result of the last recorded call before the step (the
result as the fix returns it, e.g. render_page's new shader-error lines). `--store`: the chat data
folder whose blobs hold the recorded screenshots (chat_trace.py). Each forked step is printed:
its reasoning (cut), text and calls with their results; `--out` writes them, one JSON a line.
Every request goes under its own `x-session-id` (`chat-fork-...`): the conversation's own cache is
not touched, and the first forked call reads the conversation cold (on a long chat that is the
fork's cost, minutes, not the hours of the run). Standard library and the test suite's helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import chat_trace  # noqa: E402

MAX_STEPS = 12


def as_step(message: dict) -> dict:
    """An OpenAI chat message as the fake engine's script step: text, reasoning, calls."""
    out = {"content": message.get("content") or ""}
    r = message.get("reasoning_content") or message.get("reasoning")
    if r:
        out["reasoning"] = r
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_raw": fn.get("arguments")}
        calls.append({"name": fn.get("name") or "", "arguments": args})
    if calls:
        out["calls"] = calls
    return out


def engine(url: str, max_tokens: int | None = None, timeout: float = 1800.0):
    """`live(body, headers)` against an OpenAI-compatible engine: the body as chatd built it,
    not streamed, under a session of its own."""
    session = "chat-fork-" + uuid.uuid4().hex[:8]

    def call(body: dict, headers: dict) -> dict:
        b = {k: v for k, v in body.items() if k != "stream_options"}
        b["stream"] = False
        if max_tokens:
            b["max_tokens"] = min(int(b.get("max_tokens") or max_tokens), int(max_tokens))
        # chatd's own headers (the client kind, the cost line), never the hop's (a length)
        h = {k: v for k, v in headers.items() if k.lower().startswith(("x-qse-", "x-tandem-"))
             and k.lower() != "x-tandem-parent-session"}
        h.update({"Content-Type": "application/json", "x-session-id": session})
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(b).encode(), headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as f:
            out = json.load(f)
        msg = ((out.get("choices") or [{}])[0].get("message")) or {}
        return {**as_step(msg), "_usage": out.get("usage") or {}}
    return call


def fork(conv: dict, at: tuple[int, int], live, steps: int = 3, store: str | None = None,
         patch: str | None = None) -> dict:
    """The conversation replayed to the step before `at` (turn, step) and run on by `live` for at
    most `steps` model calls. Returns {"steps": [...], "requests": [...], "snapshot", "fork_at"}:
    each forked step's text, reasoning and calls with the results chatd gave them."""
    steps = max(1, min(int(steps), MAX_STEPS))
    got: list[dict] = []
    bodies: list[dict] = []

    def bounded(body: dict, headers: dict):
        if len(got) >= steps:
            return None
        step = live(body, headers)
        bodies.append(body)
        got.append({k: v for k, v in step.items() if not k.startswith("_")} |
                   {"usage": step.get("_usage") or {}})
        return {k: v for k, v in step.items() if not k.startswith("_")}
    res = chat_trace.replay(conv, store=store, fork=at, live=bounded, patch=patch)
    # the forked steps' calls as chatd ran them: the last message's parts from the fork step on
    last = next(m for m in reversed(res["snapshot"]["messages"]) if m["role"] == "assistant")
    first = at[1]
    ran = [p for p in last["parts"] if p.get("type") == "tool_call" and
           int(p.get("step") or 0) >= first]
    k = 0
    for s in got:
        for c in s.get("calls") or []:
            if k < len(ran):
                c["status"] = ran[k].get("status")
                c["result"] = ((ran[k].get("output") or {}).get("content") or "")
            k += 1
    return {"steps": got, "requests": bodies, "snapshot": res["snapshot"], "fork_at": at,
            "replayed": len(res["requests"]) - len(bodies)}


def _cut(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f" ... ({len(text):,} characters)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--at", required=True, help="TURN:STEP, both from 1 (the first forked step)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--patch", default=None, help="a file: the new result of the call before")
    ap.add_argument("--store", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        turn, step = (int(x) for x in a.at.split(":"))
    except ValueError:
        ap.error("--at is TURN:STEP, e.g. 1:97")
    if a.steps > MAX_STEPS:
        ap.error(f"--steps is at most {MAX_STEPS}: a fork checks a step, it is not a run")
    patch = open(a.patch).read() if a.patch else None
    conv = chat_trace.load(a.trace)
    out = open(a.out, "w") if a.out else None
    for n in range(max(1, a.samples)):
        res = fork(conv, (turn, step), engine(a.url, a.max_tokens), a.steps, a.store, patch)
        print(f"sample {n + 1}: turn {turn} from step {step}, {res['replayed']} requests "
              f"replayed, {len(res['steps'])} live step(s)")
        for i, s in enumerate(res["steps"]):
            u = s.get("usage") or {}
            print(f"  step {step + i}: prompt {u.get('prompt_tokens', '?')}, "
                  f"completion {u.get('completion_tokens', '?')}")
            if s.get("reasoning"):
                print("    thinking:", _cut(s["reasoning"], 400).replace("\n", " "))
            if s.get("content"):
                print("    text:", _cut(s["content"], 400).replace("\n", " "))
            for c in s.get("calls") or []:
                print(f"    call {c['name']} {_cut(json.dumps(c['arguments']), 200)} -> "
                      f"{c.get('status')}: {_cut(c.get('result') or '', 200)}".replace("\n", " "))
            if out:
                out.write(json.dumps({"sample": n + 1, "step": step + i, **s},
                                     ensure_ascii=False) + "\n")
    if out:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
