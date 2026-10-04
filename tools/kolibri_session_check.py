"""Client sessions against the Kolibri server: conversation switches, an Open WebUI-shaped chat
and a restart in the middle of a conversation. Each check PASS or FAIL, numbers in the line.

  switch     a long conversation (`--long` tokens), then a new short chat over three turns (turn 2
             and 3 reuse its own turns), then the long conversation again (still cached), then the
             chat again (parked while the long one ran, still cached)
  owui       Open WebUI's shape: a streamed chat turn, then its title and tags side requests
             (non-streamed, the history pasted into a task prompt), then turn 2 streamed; turn 2
             must reuse turn 1, the side answers must be clean JSON
  restart-write / restart-check
             turn 1 of a conversation, saved to `--state`; after a stop and a start, turn 2 from
             that file must reuse turn 1's prompt from the session the drain wrote

    python tools/kolibri_session_check.py switch --long 16000
    python tools/kolibri_session_check.py restart-write --state /tmp/conv.json
    bash ops/kolibri-stop.sh && KOLIBRI_SET=... bash ops/kolibri-start.sh
    python tools/kolibri_session_check.py restart-check --state /tmp/conv.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kolibri_serve_check import fields, post  # noqa: E402

FAILS: list[str] = []


def check(name, ok, **kw):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(kw, ensure_ascii=False)}", flush=True)
    if not ok:
        FAILS.append(name)


def cached(usage) -> int:
    return int(((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def chat(B, M, msgs, max_tokens=120, stream=False, **kw):
    body = {"model": M, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none", **kw}
    t0 = time.perf_counter()
    if stream:
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})
        events, first = post(B, "/chat/completions", body, stream=True, timeout=1800)
        f = fields(events)
        return f["content"], f["usage"], first, f
    r = post(B, "/chat/completions", body, timeout=1800)
    t = (r.get("timings") or {}).get("ttft_ms")
    return (r["choices"][0]["message"].get("content") or "", r.get("usage"),
            (t / 1e3) if t else time.perf_counter() - t0, r)


def long_doc(n_tokens: int) -> str:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench")
    parts = [open(os.path.join(here, f)).read() for f in
             ("heldout_prose.txt", "heldout_de.txt", "heldout_code.txt", "calib.txt")
             if os.path.isfile(os.path.join(here, f))]
    body, k = "", 0
    while len(body) < n_tokens * 3.6:
        body += parts[k % len(parts)] + "\n\n"
        k += 1
    return body


def stats(B):
    import urllib.request
    try:
        return json.loads(urllib.request.urlopen(B + "/cache/stats", timeout=10).read())
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def run_switch(B, M, n_long):
    run = time.time_ns()
    longm = [{"role": "user", "content": f"Dokument {run}.\n\n" + long_doc(n_long)
              + "\n\nFasse den ersten Absatz in einem Satz zusammen."}]
    a, u, t, _ = chat(B, M, longm, max_tokens=60)
    check("switch: long turn 1", u["prompt_tokens"] > n_long * 0.5, prompt=u["prompt_tokens"],
          ttft_s=round(t, 2))
    longm += [{"role": "assistant", "content": a}]
    msgs = [{"role": "system", "content": f"Du bist ein knapper Assistent. Lauf {run}."}]
    prev = 0
    for turn, q in enumerate(["Nenne drei Vögel, die in Deutschland brüten.",
                              "Welcher davon ist der kleinste?",
                              "Und wie schwer ist er ungefähr?"], 1):
        msgs.append({"role": "user", "content": q})
        a, u, t, _ = chat(B, M, msgs, max_tokens=80, stream=True)
        c = cached(u)
        # the previous prompt is the reusable part: the template renders the answer in the
        # history differently from the generation prompt's open assistant turn
        ok = c == 0 if turn == 1 else c >= prev - 8
        prev = u["prompt_tokens"]
        check(f"switch: new chat turn {turn}", ok, prompt=u["prompt_tokens"], cached=c,
              ttft_s=round(t or 0, 3))
        msgs.append({"role": "assistant", "content": a})
    longm.append({"role": "user", "content": "Welche Sprachen kommen im Dokument vor?"})
    a, u, t, _ = chat(B, M, longm, max_tokens=40)
    c = cached(u)
    check("switch: back to the long conversation, still cached", c >= u["prompt_tokens"] - 200,
          prompt=u["prompt_tokens"], cached=c, ttft_s=round(t, 3))
    msgs.append({"role": "user", "content": "Danke. Wo überwintert er?"})
    a, u, t, _ = chat(B, M, msgs, max_tokens=60, stream=True)
    c = cached(u)
    check("switch: back to the chat, still cached", c >= prev - 8,
          prompt=u["prompt_tokens"], cached=c, ttft_s=round(t or 0, 3))
    kp = stats(B).get("kolibri_prefix", {})
    print("stats", json.dumps({k: kp.get(k) for k in ("requests", "cold", "parked", "unparked",
                                                         "park_evicted", "park_skipped",
                                                         "park_ms", "unpark_ms",
                                                         "parked_conversations")}))


OWUI_TITLE = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.
### Guidelines:
- The title should clearly represent the main theme or subject of the conversation.
- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.
- Write the title in the chat's primary language; default to English if multilingual.
- Prioritize accuracy over excessive creativity; keep it clear and simple.
### Output:
JSON format: { "title": "your concise title here" }
### Chat History:
<chat_history>
%s
</chat_history>"""

OWUI_TAGS = """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3 more specific subtopic tags.
### Output:
JSON format: { "tags": ["tag1", "tag2", "tag3"] }
### Chat History:
<chat_history>
%s
</chat_history>"""


def run_owui(B, M):
    run = time.time_ns()
    sysmsg = {"role": "system", "content": "You are a helpful assistant. Today is Sunday. " * 40
              + f"(session {run})"}
    msgs = [sysmsg, {"role": "user", "content": "Explain in a paragraph how hummingbirds hover."}]
    body = {"model": M, "messages": msgs, "stream": True, "max_tokens": 300}   # OWUI: no usage ask
    events, first = post(B, "/chat/completions", body, stream=True, timeout=600)
    f = fields(events)
    a1 = f["content"]
    check("owui: turn 1 streamed (release sampling, thinking as the server defaults)",
          bool(a1.strip()) and f["finish"] in ("stop", "length"), finish=f["finish"],
          chars=len(a1), ttft_s=round(first or 0, 3), think_in_content="<think>" in a1)
    hist = f"USER: {msgs[1]['content']}\nASSISTANT: {a1}"
    for name, tmpl, key in (("title", OWUI_TITLE, "title"), ("tags", OWUI_TAGS, "tags")):
        r = post(B, "/chat/completions", {"model": M, "stream": False, "max_tokens": 1000,
                                          "messages": [{"role": "user", "content": tmpl % hist}]})
        txt = r["choices"][0]["message"].get("content") or ""
        body_txt = txt.split("</think>")[-1].strip()
        s, e = body_txt.find("{"), body_txt.rfind("}")
        try:
            val = json.loads(body_txt[s:e + 1])[key]
            ok = bool(val)
        except Exception:  # noqa: BLE001
            val, ok = None, False
        check(f"owui: {name} side request answers JSON", ok, value=val,
              raw_head=txt[:120], usage=r.get("usage"))
    msgs += [{"role": "assistant", "content": a1},
             {"role": "user", "content": "And how fast do their wings beat?"}]
    events, first = post(B, "/chat/completions", {**body, "messages": msgs,
                                                  "stream_options": {"include_usage": True}},
                         stream=True, timeout=600)
    f2 = fields(events)
    u = f2["usage"] or {}
    c = cached(u)
    check("owui: turn 2 reuses turn 1 after the side requests",
          c >= (u.get("prompt_tokens") or 0) - 200 and c > 0, prompt=u.get("prompt_tokens"),
          cached=c, ttft_s=round(first or 0, 3), usage_keys=sorted(u))


def run_restart_write(B, M, path, n_long):
    run = time.time_ns()
    msgs = [{"role": "user", "content": f"Text {run}.\n\n" + long_doc(n_long)
             + "\n\nWorum geht es im ersten Abschnitt? Ein Satz."}]
    a, u, t, _ = chat(B, M, msgs, max_tokens=60)
    msgs.append({"role": "assistant", "content": a})
    json.dump({"messages": msgs, "prompt": u["prompt_tokens"], "ttft_s": t}, open(path, "w"))
    check("restart: turn 1 written", bool(a), prompt=u["prompt_tokens"], ttft_s=round(t, 2))


def run_restart_check(B, M, path):
    st = json.load(open(path))
    msgs = st["messages"] + [{"role": "user", "content": "Und im zweiten? Ein Satz."}]
    a, u, t, _ = chat(B, M, msgs, max_tokens=60)
    c = cached(u)
    s = stats(B)
    check("restart: turn 2 after the restart reuses turn 1", c >= st["prompt"] - 8,
          turn1_prompt=st["prompt"], prompt=u["prompt_tokens"], cached=c, ttft_s=round(t, 3),
          turn1_ttft_s=round(st["ttft_s"], 2), restored=(s.get("kolibri_session") or {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("switch", "owui", "restart-write", "restart-check", "all"))
    ap.add_argument("--base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Kolibri-1")
    ap.add_argument("--long", type=int, default=16000)
    ap.add_argument("--state", default="/tmp/kolibri-restart-conv.json")
    a = ap.parse_args()
    if a.mode in ("switch", "all"):
        run_switch(a.base, a.model, a.long)
    if a.mode in ("owui", "all"):
        run_owui(a.base, a.model)
    if a.mode == "restart-write":
        run_restart_write(a.base, a.model, a.state, a.long)
    if a.mode == "restart-check":
        run_restart_check(a.base, a.model, a.state)
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all passed", flush=True)
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(main())
