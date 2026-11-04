"""api: the request shapes a client sends, on each side: a stream and a plain answer of the chat
endpoint, a client that cancels mid-stream, a client that leaves while it waits for a lane, a
forced tool call, and the /health contract. After each, the server must be idle again and answer
the next request. A check the base fails too is SKIPPED (not this change's)."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request

from tools import parallel_card as pc
from tools.gate.suites import Ctx, Plan, first
from tools.gate.verdict import case_row, skipped

NAME = "api"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Name three rivers in Europe, one line each."}]


def _post(base: str, path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _conn(base: str, path: str, body: dict) -> http.client.HTTPConnection:
    host, port = base.split("//")[1].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=600)
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    return c


def idle_then_ok(ctx: Ctx, limit_s: float = 120) -> tuple[bool, str]:
    """The server is idle again (nothing running or waiting) and answers a short request."""
    t = time.monotonic()
    while time.monotonic() - t < limit_s:
        p = pc.health(ctx.base).get("parallel") or {}
        inf = pc.health(ctx.base).get("inflight") or {}
        if not p.get("running") and not inf.get("waiting") and not p.get("decoding"):
            break
        time.sleep(1.0)
    else:
        return False, f"not idle after {limit_s:.0f} s"
    try:
        r = _post(ctx.base, "/v1/chat/completions", {"model": ctx.cl.model, "messages": ASK, "max_tokens": 8})
        return bool(r.get("choices")), f"idle after {time.monotonic() - t:.1f} s, next request ok"
    except (urllib.error.URLError, OSError) as exc:
        return False, f"the next request failed: {exc}"


def stream(ctx: Ctx) -> tuple[bool, str]:
    c = _conn(ctx.base, "/v1/chat/completions", {"model": ctx.cl.model, "messages": ASK, "max_tokens": 48,
                                                "stream": True, "stream_options": {"include_usage": True}})
    r = c.getresponse()
    if r.status != 200:
        return False, f"HTTP {r.status}"
    text, done, usage = "", False, None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        ev = json.loads(data)
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            text += (d.get("content") or "") + (d.get("reasoning_content") or "")
    c.close()
    ok = done and bool(text) and bool(usage and usage.get("completion_tokens"))
    return ok, f"{len(text)} characters, [DONE] {done}, usage {bool(usage)}"


def plain(ctx: Ctx) -> tuple[bool, str]:
    r = _post(ctx.base, "/v1/chat/completions", {"model": ctx.cl.model, "messages": ASK, "max_tokens": 48})
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    ok = bool(text) and ch.get("finish_reason") in ("stop", "length")
    return ok, f"{len(text)} characters, finish {ch.get('finish_reason')}"


def cancel(ctx: Ctx) -> tuple[bool, str]:
    c = _conn(ctx.base, "/v1/completions", {"model": ctx.cl.model, "prompt": ctx.cl.render(ASK),
                                          "max_tokens": 1024, "stream": True, "temperature": 0})
    r = c.getresponse()
    events = 0
    for raw in r:
        if raw.startswith(b"data:"):
            events += 1
        if events >= 5:
            break
    c.close()
    ok, why = idle_then_ok(ctx)
    return ok, f"closed after {events} events; {why}"


def end_ids(tok) -> list[int]:
    """The model's end tokens (its tokenizer's end of text and the chat template's end of turn)."""
    if tok is None:
        return []
    ids = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
    for t in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != getattr(tok, "unk_token_id", None):
            ids.append(i)
    return sorted({int(i) for i in ids if i is not None})


def waited_without_text(c: http.client.HTTPConnection, limit_s: float = 3.0) -> bool:
    """True when no streamed text came back within `limit_s` (the request waited for a lane). The
    engine may send its headers early while a request waits (`X-Tandem-Wait`), so headers alone do
    not count; a `data:` line does."""
    t0 = time.monotonic()
    try:
        r = c.getresponse()
        while time.monotonic() - t0 < limit_s:
            line = r.fp.readline()
            if not line:
                return False                     # closed: answered or refused, not held
            if line.startswith(b"data:"):
                return False
    except (socket.timeout, TimeoutError):
        return True
    except (OSError, http.client.HTTPException):
        return False
    return True


def leave_waiting(ctx: Ctx) -> tuple[bool, str]:
    """Every lane busy with a long stream, one more client that waits and leaves, then the long
    streams leave too."""
    lanes = (ctx.health.get("parallel") or {}).get("max_running") or 1
    h0 = pc.health(ctx.base).get("inflight") or {}
    ask = "Write the numbers from one to three thousand in words, one a line, and do not stop early."
    ask += ctx.pack.get("hold_word", "")
    body = {"model": ctx.cl.model, "prompt": ctx.cl.render([{"role": "user", "content": ask}]),
            "max_tokens": 4096, "stream": True, "temperature": 0}
    ends = end_ids(ctx.cl.tok)
    if ends:
        body["logit_bias"] = {str(i): -100 for i in ends}      # the lanes stay busy until the client leaves
    conns, stop = [], threading.Event()

    def hold(c):
        try:
            r = c.getresponse()
            for _ in r:
                if stop.is_set():
                    break
        except Exception:                                             # noqa: BLE001
            pass

    ths = []
    for _ in range(lanes):
        c = _conn(ctx.base, "/v1/completions", body)
        conns.append(c)
        t = threading.Thread(target=hold, args=(c,), daemon=True)
        t.start()
        ths.append(t)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60:
        if ((pc.health(ctx.base).get("parallel") or {}).get("running") or 0) >= lanes:
            break
        time.sleep(0.5)
    running = (pc.health(ctx.base).get("parallel") or {}).get("running") or 0
    waiter = _conn(ctx.base, "/v1/completions", dict(body, max_tokens=16))
    waiter.sock.settimeout(3.0)
    held = waited_without_text(waiter)
    waiter.close()
    time.sleep(1.0)
    stop.set()
    for c in conns:
        c.close()
    for t in ths:
        t.join(10)
    ok, why = idle_then_ok(ctx)
    h1 = pc.health(ctx.base).get("inflight") or {}
    errs = (h1.get("errors") or 0) - (h0.get("errors") or 0)
    detail = (f"{running} of {lanes} lanes busy, the extra request "
              f"{'held with no text for 3 s' if held else 'NOT held (answered or closed)'}, all left; "
              f"errors +{errs}, abandoned +{(h1.get('abandoned') or 0) - (h0.get('abandoned') or 0)}; {why}")
    return ok and errs == 0 and held and running >= lanes, detail


def tool_call(ctx: Ctx) -> tuple[bool, str]:
    r = _post(ctx.base, "/v1/chat/completions", {
        "model": ctx.cl.model, "max_tokens": 256, "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "What is the weather in Hamburg? Use the tool."}]})
    ch = (r.get("choices") or [{}])[0]
    calls = (ch.get("message") or {}).get("tool_calls") or []
    if not calls:
        return False, f"no tool call, finish {ch.get('finish_reason')}"
    fn = calls[0].get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return False, f"the arguments are not JSON: {fn.get('arguments')!r:.80}"
    ok = fn.get("name") == "get_weather" and isinstance(args, dict)
    return ok, f"{fn.get('name')}({json.dumps(args)[:60]}), finish {ch.get('finish_reason')}"


#: the /health fields a client and the dashboard read, and their types
HEALTH = {"status": str, "version": str, "max_len": int, "parallel.max_running": int,
          "parallel.running": int, "inflight.waiting": int, "inflight.errors": int, "weights": dict}


def health_contract(ctx: Ctx) -> tuple[bool, str]:
    h = pc.health(ctx.base)
    bad = []
    for key, typ in HEALTH.items():
        v = h
        for part in key.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if not isinstance(v, typ) or isinstance(v, bool) and typ is int:
            bad.append(key)
    models = pc._get(ctx.base + "/v1/models")
    ids = [m.get("id") for m in models.get("data") or []]
    if ctx.cl.model not in ids:
        bad.append(f"/v1/models lacks {ctx.cl.model}")
    if h.get("status") != "ok":
        bad.append(f"status {h.get('status')!r}")
    return not bad, ("all fields present" if not bad else "missing or wrong: " + ", ".join(bad))


CASES = (("health", "/health contract", health_contract), ("stream", "chat stream", stream),
         ("plain", "chat answer", plain), ("tool", "forced tool call", tool_call),
         ("cancel", "cancel mid-stream", cancel), ("leave", "client leaves while waiting", leave_waiting))


def measure(ctx: Ctx) -> dict:
    out = {}
    for cid, _, fn in CASES:
        try:
            ok, why = fn(ctx)
        except Exception as exc:                                     # noqa: BLE001
            ok, why = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        out[cid] = {"ok": ok, "why": why}
        ctx.p(f"api {cid}: {'ok' if ok else 'FAILED'} ({why})")
    return out


def rows(raw: dict, plan: Plan) -> list[dict]:
    b, h = first(raw, "base"), first(raw, "head")
    if not h:
        return [skipped("api.all", NAME, "all checks", "the head did not run")]
    return [case_row(f"api.{cid}", NAME, label, (b or {}).get(cid, {}).get("ok") if b else None,
                     h[cid]["ok"], h[cid]["why"]) for cid, label, _ in CASES]
