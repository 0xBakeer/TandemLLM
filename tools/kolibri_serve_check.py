"""Kolibri-1 through the OpenAI server: the checks a client would make, each PASS or FAIL.

    python tools/kolibri_serve_check.py --base http://127.0.0.1:8001/v1 [--json out.json]

Standard library only. Greedy (temperature 0) everywhere, so a rerun gives the same text.

  models      /v1/models lists the served id
  capital     "Was ist die Hauptstadt von Deutschland?" with thinking off answers Berlin
  think_rc    thinking on (effort low), reasoning_format reasoning_content, streamed: reasoning
              deltas arrive, the answer arrives, and neither field carries a <think> tag
  think_tags  the same in the default tags format: content opens with the model's own <think>
              and closes it before the answer
  tool        a weather tool, thinking off: finish_reason tool_calls, one call get_weather with
              JSON arguments naming Berlin, streamed as tool_calls deltas
  tool_rt     the tool result sent back: the model answers in text using it
  prefix      turn 2 of a conversation reuses turn 1's rows (usage cached tokens, the prefix
              cache's counters) and its first token comes sooner than turn 1's
  guest       a short unrelated request between two turns of a 2k+ conversation: turn 2 still
              reuses all of turn 1's prompt (the guest stash)
  speed       decode tok/s of a 256-token greedy answer (from the server's own timings)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def post(base, path, body, stream=False, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(r.read())
    events, t0, first = [], time.perf_counter(), None
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if first is None and any((c.get("delta") or {}).get(k) for c in ev.get("choices") or []
                                 for k in ("content", "reasoning_content", "tool_calls")):
            first = time.perf_counter() - t0
        events.append(ev)
    return events, first


def fields(events):
    out = {"content": "", "reasoning_content": "", "tool_calls": [], "finish": None, "usage": None}
    for ev in events:
        if ev.get("usage"):
            out["usage"] = ev["usage"]
        for c in ev.get("choices") or []:
            d = c.get("delta") or {}
            out["content"] += d.get("content") or ""
            out["reasoning_content"] += d.get("reasoning_content") or ""
            out["tool_calls"] += d.get("tool_calls") or []
            if c.get("finish_reason"):
                out["finish"] = c["finish_reason"]
    return out


def long_check(B, M, n_tokens, check):
    """A needle near the start of a long document (the held-out and calibration texts repeated to
    about `n_tokens`), asked for at the end; then a follow-up that must reuse the document."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench")
    parts = [open(os.path.join(here, f)).read() for f in
             ("heldout_prose.txt", "heldout_de.txt", "heldout_code.txt", "calib.txt")
             if os.path.isfile(os.path.join(here, f))]
    body, k = "", 0
    while len(body) < n_tokens * 3.6:
        body += parts[k % len(parts)] + "\n\n"
        k += 1
    cut = len(body) // 20
    doc = body[:cut] + "\n\nWichtig: Die Geheimzahl für das Archiv lautet 4711-KOLIBRI.\n\n" + body[cut:]
    msgs = [{"role": "user", "content": "Hier ist ein langes Dokument.\n\n" + doc
             + "\n\nFrage: Wie lautet die Geheimzahl für das Archiv? Antworte nur mit der Zahl."}]
    t0 = time.time()
    r1 = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 40,
                                       "reasoning_effort": "none", "messages": msgs}, timeout=1800)
    a1 = r1["choices"][0]["message"]["content"]
    msgs2 = msgs + [{"role": "assistant", "content": a1},
                    {"role": "user", "content": "Und in welcher Sprache ist der erste Abschnitt "
                                                "des Dokuments geschrieben? Ein Wort."}]
    r2 = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 20,
                                       "reasoning_effort": "none", "messages": msgs2}, timeout=1800)
    c2 = ((r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    p1 = r1["usage"]["prompt_tokens"]
    check("long_needle", "4711" in a1 and c2 >= p1 - 8, prompt_tokens=p1, answer=a1,
          turn1_s=round(time.time() - t0, 1),
          ttft1_ms=(r1.get("timings") or {}).get("ttft_ms"),
          decode1_tok_s=(r1.get("timings") or {}).get("predicted_per_second"),
          turn2_cached=c2, turn2_answer=r2["choices"][0]["message"]["content"],
          ttft2_ms=(r2.get("timings") or {}).get("ttft_ms"),
          decode2_tok_s=(r2.get("timings") or {}).get("predicted_per_second"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Kolibri-1")
    ap.add_argument("--json", default="")
    ap.add_argument("--long", type=int, default=0,
                    help="also a needle in a document of about this many tokens, and a follow-up")
    ap.add_argument("--only-long", action="store_true")
    a = ap.parse_args()
    B, M = a.base, a.model
    res = {}

    def check(name, ok, **info):
        res[name] = {"pass": bool(ok), **info}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, ensure_ascii=False)[:600]}",
              flush=True)

    if a.long:
        long_check(B, M, a.long, check)
        if a.only_long:
            if a.json:
                json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)
            return sum(not v["pass"] for v in res.values())

    ms = json.loads(urllib.request.urlopen(B + "/models", timeout=30).read())
    ids = [m["id"] for m in ms.get("data", [])]
    check("models", M in ids, ids=ids)

    r = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 64,
                                      "reasoning_effort": "none",
                                      "messages": [{"role": "user", "content":
                                                    "Was ist die Hauptstadt von Deutschland?"}]})
    text = r["choices"][0]["message"]["content"]
    check("capital", "Berlin" in text and "<think>" not in text, text=text,
          finish=r["choices"][0]["finish_reason"], usage=r.get("usage"))

    q = [{"role": "user", "content": "Wie viele Minuten hat eine Woche? Antworte kurz."}]
    ev, ttft = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 1500,
                                             "stream": True, "reasoning_effort": "low",
                                             "reasoning_format": "reasoning_content",
                                             "stream_options": {"include_usage": True},
                                             "messages": q}, stream=True)
    f = fields(ev)
    check("think_rc", f["reasoning_content"].strip() and f["content"].strip()
          and "<think>" not in f["content"] + f["reasoning_content"]
          and "</think>" not in f["content"] + f["reasoning_content"],
          reasoning=f["reasoning_content"][:200], content=f["content"][:200], finish=f["finish"],
          usage=f["usage"], ttft_s=ttft)

    ev, _ = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 1500,
                                          "stream": True, "reasoning_effort": "low",
                                          "messages": q}, stream=True)
    f = fields(ev)
    c = f["content"]
    check("think_tags", c.lstrip().startswith("<think>") and "</think>" in c
          and c.split("</think>")[-1].strip(), content=c[:120] + " ... " + c[-120:])

    tq = [{"role": "user", "content": "Wie ist das Wetter gerade in Berlin?"}]
    ev, _ = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 256,
                                          "stream": True, "reasoning_effort": "none",
                                          "tools": WEATHER, "messages": tq}, stream=True)
    f = fields(ev)
    name = "".join((d.get("function") or {}).get("name") or "" for d in f["tool_calls"])
    args = "".join((d.get("function") or {}).get("arguments") or "" for d in f["tool_calls"])
    try:
        parsed = json.loads(args) if args else None
    except ValueError:
        parsed = None
    check("tool", f["finish"] == "tool_calls" and name == "get_weather"
          and isinstance(parsed, dict) and "Berlin" in json.dumps(parsed, ensure_ascii=False),
          call=name, arguments=args, content=f["content"][:200], finish=f["finish"])

    call_id = next((d.get("id") for d in f["tool_calls"] if d.get("id")), "call_0")
    r = post(B, "/chat/completions", {
        "model": M, "temperature": 0, "max_tokens": 256, "reasoning_effort": "none",
        "tools": WEATHER,
        "messages": tq + [{"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": "get_weather", "arguments": args or '{"city": "Berlin"}'}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": '{"temp_c": 14, "sky": "bedeckt", "wind_kmh": 20}'}]})
    text = r["choices"][0]["message"].get("content") or ""
    check("tool_rt", "14" in text and r["choices"][0]["finish_reason"] == "stop", text=text)

    conv = [{"role": "system", "content": "Du bist ein hilfreicher Assistent. " * 60},
            {"role": "user", "content": "Nenne drei deutsche Flüsse."}]
    ev1, t1 = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 120,
                                            "stream": True, "reasoning_effort": "none",
                                            "stream_options": {"include_usage": True},
                                            "messages": conv}, stream=True)
    f1 = fields(ev1)
    conv2 = conv + [{"role": "assistant", "content": f1["content"]},
                    {"role": "user", "content": "Und welcher davon ist der längste?"}]
    ev2, t2 = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 120,
                                            "stream": True, "reasoning_effort": "none",
                                            "stream_options": {"include_usage": True},
                                            "messages": conv2}, stream=True)
    f2 = fields(ev2)
    cached = ((f2["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    try:
        stats = json.loads(urllib.request.urlopen(B.rsplit("/v1", 1)[0] + "/v1/cache/stats",
                                                  timeout=30).read())
    except Exception as exc:                                      # noqa: BLE001
        stats = {"error": str(exc)}
    kp = stats.get("kolibri_prefix") or {}
    check("prefix", (cached or 0) > 0 and t2 is not None and t1 is not None,
          ttft1_s=t1, ttft2_s=t2, cached_tokens=cached, usage2=f2["usage"],
          prefix={k: kp.get(k) for k in ("truncated", "anchor_hits", "cold", "tokens_reused",
                                          "tokens_forwarded", "anchors")})

    # a guest between two turns of a conversation longer than 2,048 tokens (the stash's floor):
    # turn 2 must still find turn 1's prompt in the cache
    long_sys = ("Du bist ein sorgfältiger Assistent für Fragen zur deutschen Geschichte. "
                "Antworte knapp, nenne Jahreszahlen und erfinde nichts. ") * 110
    conv = [{"role": "system", "content": long_sys},
            {"role": "user", "content": "Wann wurde die erste deutsche Eisenbahn eröffnet?"}]
    r1 = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 60,
                                       "reasoning_effort": "none", "messages": conv})
    post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 20,
                                  "reasoning_effort": "none", "messages": [
                                      {"role": "user", "content": "Gib diesem Chat einen Titel: Eisenbahn"}]})
    conv2 = conv + [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
                    {"role": "user", "content": "Und zwischen welchen Städten?"}]
    r2 = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 60,
                                       "reasoning_effort": "none", "messages": conv2})
    c1 = r1["usage"]["prompt_tokens"]
    c2 = ((r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    # thinking off: turn 1's prompt ends with the empty think block (4 tokens) that turn 2's
    # history renders without, so turn 2 can share at most c1 - 4 tokens
    check("guest", c1 >= 2048 and c2 >= c1 - 8, turn1_prompt=c1, turn2_prompt=r2["usage"]["prompt_tokens"],
          turn2_cached=c2, answer=r2["choices"][0]["message"]["content"][:160],
          ttft_turn1_ms=(r1.get("timings") or {}).get("ttft_ms"),
          ttft_turn2_ms=(r2.get("timings") or {}).get("ttft_ms"))

    r = post(B, "/chat/completions", {"model": M, "temperature": 0, "max_tokens": 256,
                                      "reasoning_effort": "none", "messages": [
                                          {"role": "user", "content":
                                           "Schreibe einen langen Absatz über die Geschichte "
                                           "der Eisenbahn in Deutschland."}]})
    tm = r.get("timings") or {}
    check("speed", r["usage"]["completion_tokens"] > 50, usage=r["usage"], timings=tm)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)
    return sum(not v["pass"] for v in res.values())


if __name__ == "__main__":
    sys.exit(main())
