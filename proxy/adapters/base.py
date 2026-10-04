"""The adapter interface, and the OpenAI-compatible behaviour every kind starts from.

One class per backend kind. The relay copies bytes; the adapter only reads them, for the ledger:

    routes()          the client paths this kind serves through the relay
    capabilities()    what the harness (P4) may ask of it: continue_final_message, the choice
                      field, where reasoning arrives, which cache counts it reports, how thinking
                      is switched
    headers(key)      the upstream headers: the key as a bearer, a JSON body
    forward(headers)  the client headers passed on (local engines read X-Tandem and the session
                      headers; a cloud API gets none of them)
    prepare(...)      the upstream body: the model id the backend knows, and a stream asked for
                      its usage frame when the client did not ask (the frame is then dropped)
    read_frame(...)   one event of a stream: does it carry a token (content, reasoning or a
                      tool-call delta), its usage, its finish reason, is it the frame to drop
    read_body(...)    the same for a whole JSON answer
    read_usage(obj)   the provider's usage object in the ledger's normalised fields; a field the
                      provider did not send is None, never 0
    price(usage, p)   the cost in micro-dollars from the model's price row, or None
    metrics()         P2 and P3: the scrape table; None here

Usage, normalised:

    prompt_tokens       all prompt tokens, the cached ones included
    completion_tokens   all output tokens, the reasoning ones included
    cached_tokens       prompt tokens served from the provider's cache
    cache_write_tokens  prompt tokens written to a cache at a price of their own (Anthropic later)
    reasoning_tokens    output tokens spent reasoning
"""

from __future__ import annotations

import json

CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
EMBEDDINGS = "/v1/embeddings"
RESPONSES = "/v1/responses"

ENDPOINTS = {CHAT: "chat", COMPLETIONS: "completions", EMBEDDINGS: "embeddings",
             RESPONSES: "responses"}

# the client headers a local engine reads (our engine's conductor, session identity)
LOCAL_HEADERS = ("x-tandem", "x-session-id", "x-conversation-id", "x-request-id")


def _int(x):
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _get(obj, *path):
    for p in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(p)
    return obj


class Stamps:
    """What one response's frames said, filled by `read_frame`; the relay adds the times."""

    __slots__ = ("usage", "finish_reason", "reasoning", "tools", "tokens_seen", "error",
                 "model", "think_open")

    def __init__(self):
        self.usage = None              # the provider's usage object, as sent
        self.finish_reason = None
        self.reasoning = False         # reasoning text seen (a field, or <think> in content)
        self.tools = set()             # tool-call indices seen
        self.tokens_seen = 0           # frames that carried a token (content, reasoning, tool)
        self.error = None              # an error object inside the stream
        self.model = None
        self.think_open = False


def sse_data(event: bytes) -> bytes | None:
    """The `data:` payload of one SSE event (several data lines joined by newlines), or None."""
    parts = []
    for line in event.splitlines():
        if line.startswith(b"data:"):
            parts.append(line[5:].lstrip(b" "))
    return b"\n".join(parts) if parts else None


class Adapter:
    kind = "generic"
    cloud = False

    def __init__(self, backend: dict):
        self.backend = backend

    # ----------------------------------------------------------------- what it serves
    def routes(self) -> tuple[str, ...]:
        return (CHAT, COMPLETIONS, EMBEDDINGS)

    def capabilities(self) -> dict:
        return {"continue_final_message": False, "choice": "text",
                "reasoning_field": "reasoning_content",
                "cache_counts": ["prompt_tokens_details.cached_tokens"],
                "thinking_switch": "chat_template_kwargs"}

    def metrics(self):
        return None

    # ----------------------------------------------------------------- the request
    def headers(self, key: str | None) -> list[tuple[str, str]]:
        out = [("Content-Type", "application/json"), ("Accept", "application/json, "
                                                                "text/event-stream")]
        if key:
            out.append(("Authorization", f"Bearer {key}"))
        return out

    def forward(self, client_headers) -> list[tuple[str, str]]:
        if self.cloud:
            return []
        out = []
        for name in LOCAL_HEADERS:
            try:
                v = client_headers.get(name)
            except Exception:                                      # noqa: BLE001
                v = None
            if v:
                out.append((name, v))
        return out

    def inject_usage(self) -> bool:
        v = self.backend.get("inject_usage")
        if v is None:
            from proxy.config import KINDS
            v = KINDS.get(self.kind, KINDS["generic"])["inject_usage"]
        return bool(v)

    def prepare(self, path: str, raw: bytes, body: dict, upstream_model: str
                ) -> tuple[bytes, bool]:
        """(the upstream body, whether the stream was asked for a usage frame the client did not
        ask for). The client's bytes go out unchanged when nothing needs to change."""
        changed = False
        injected = False
        if body.get("model") != upstream_model:
            body = dict(body)
            body["model"] = upstream_model
            changed = True
        if path in (CHAT, COMPLETIONS) and body.get("stream") and self.inject_usage():
            opts = body.get("stream_options")
            if not (isinstance(opts, dict) and opts.get("include_usage") is True):
                body = dict(body)
                body["stream_options"] = dict(opts) if isinstance(opts, dict) else {}
                body["stream_options"]["include_usage"] = True
                changed = injected = True
        if not changed:
            return raw, False
        return json.dumps(body, ensure_ascii=False).encode(), injected

    # ----------------------------------------------------------------- the answer
    def read_frame(self, path: str, event: bytes, st: Stamps, injected: bool) -> tuple[bool, bool]:
        """(the frame carries a token, drop it before the client)."""
        data = sse_data(event)
        if data is None or data == b"[DONE]":
            return False, False
        try:
            obj = json.loads(data)
        except ValueError:
            return False, False
        if not isinstance(obj, dict):
            return False, False
        if path == RESPONSES:
            return self._read_response_event(obj, st), False
        if "error" in obj and not obj.get("choices"):
            st.error = obj.get("error")
            st.finish_reason = "error"
            return False, False
        if obj.get("model") and st.model is None:
            st.model = obj["model"]
        token = False
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            token |= self._read_choice(ch, st)
            if ch.get("finish_reason"):
                st.finish_reason = ch["finish_reason"]
        u = obj.get("usage")
        drop = False
        if isinstance(u, dict):
            st.usage = u
            # the usage-only frame the proxy asked for: choices empty, nothing else in it
            drop = injected and not obj.get("choices")
        if token:
            st.tokens_seen += 1
        return token, drop

    def _read_choice(self, ch: dict, st: Stamps) -> bool:
        token = False
        d = ch.get("delta")
        if not isinstance(d, dict):
            d = {}
        text = ch.get("text")                                  # /v1/completions
        if isinstance(text, str) and text:
            token = True
        for f in ("reasoning_content", "reasoning"):
            v = d.get(f)
            if isinstance(v, str) and v:
                st.reasoning = True
                token = True
        c = d.get("content")
        if isinstance(c, str) and c:
            token = True
            if "<think>" in c:
                st.reasoning = True
        tc = d.get("tool_calls")
        if isinstance(tc, list):
            for i, call in enumerate(tc):
                idx = call.get("index", i) if isinstance(call, dict) else i
                st.tools.add(idx)
                token = True
        if isinstance(d.get("function_call"), dict):
            st.tools.add(0)
            token = True
        return token

    def _read_response_event(self, obj: dict, st: Stamps) -> bool:
        t = obj.get("type") or ""
        if t in ("response.output_text.delta", "response.refusal.delta"):
            st.tokens_seen += 1
            return True
        if t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            st.reasoning = True
            st.tokens_seen += 1
            return True
        if t == "response.function_call_arguments.delta":
            st.tokens_seen += 1
            return True
        if t == "response.output_item.added":
            item = obj.get("item") or {}
            if item.get("type") in ("function_call", "custom_tool_call"):
                st.tools.add(obj.get("output_index", len(st.tools)))
            if item.get("type") == "reasoning":
                st.reasoning = True
            return False
        if t in ("response.completed", "response.incomplete", "response.failed"):
            resp = obj.get("response") or {}
            if isinstance(resp.get("usage"), dict):
                st.usage = resp["usage"]
            if resp.get("model"):
                st.model = resp["model"]
            st.finish_reason = {"response.completed": "stop", "response.incomplete": "length",
                                "response.failed": "error"}[t]
            if t == "response.incomplete":
                reason = _get(resp, "incomplete_details", "reason")
                if reason and reason != "max_output_tokens":
                    st.finish_reason = str(reason)
            return False
        if t == "error":
            st.error = obj.get("error") or obj
            st.finish_reason = "error"
        return False

    def read_body(self, path: str, obj, st: Stamps) -> None:
        """A whole JSON answer (no stream)."""
        if not isinstance(obj, dict):
            return
        if path == RESPONSES:
            if isinstance(obj.get("usage"), dict):
                st.usage = obj["usage"]
            st.finish_reason = {"completed": "stop", "incomplete": "length",
                                "failed": "error"}.get(obj.get("status"), obj.get("status"))
            for item in obj.get("output") or []:
                if isinstance(item, dict) and item.get("type") in ("function_call",
                                                                   "custom_tool_call"):
                    st.tools.add(len(st.tools))
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    st.reasoning = True
            return
        if "error" in obj and not obj.get("choices"):
            st.error = obj["error"]
            st.finish_reason = "error"
        if isinstance(obj.get("usage"), dict):
            st.usage = obj["usage"]
        st.model = obj.get("model")
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            if ch.get("finish_reason"):
                st.finish_reason = ch["finish_reason"]
            msg = ch.get("message") or {}
            if isinstance(msg, dict):
                if msg.get("reasoning_content") or msg.get("reasoning"):
                    st.reasoning = True
                if isinstance(msg.get("content"), str) and "<think>" in msg["content"]:
                    st.reasoning = True
                for i, _ in enumerate(msg.get("tool_calls") or []):
                    st.tools.add(i)

    def read_usage(self, obj, path: str = CHAT) -> dict:
        """The normalised usage; every field None when `obj` is not a usage object."""
        out = {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None,
               "cache_write_tokens": None, "reasoning_tokens": None}
        if not isinstance(obj, dict):
            return out
        if "input_tokens" in obj and "prompt_tokens" not in obj:       # the Responses API
            out["prompt_tokens"] = _int(obj.get("input_tokens"))
            out["completion_tokens"] = _int(obj.get("output_tokens"))
            out["cached_tokens"] = _int(_get(obj, "input_tokens_details", "cached_tokens"))
            out["reasoning_tokens"] = _int(_get(obj, "output_tokens_details",
                                                "reasoning_tokens"))
            return out
        out["prompt_tokens"] = _int(obj.get("prompt_tokens"))
        out["completion_tokens"] = _int(obj.get("completion_tokens"))
        if path == EMBEDDINGS and out["completion_tokens"] is None and \
                out["prompt_tokens"] is not None:
            out["completion_tokens"] = 0
        out["cached_tokens"] = _int(_get(obj, "prompt_tokens_details", "cached_tokens"))
        if out["cached_tokens"] is None:
            # DeepSeek's shape, also sent by hosts that copy its API
            out["cached_tokens"] = _int(obj.get("prompt_cache_hit_tokens"))
        r = _int(_get(obj, "completion_tokens_details", "reasoning_tokens"))
        if r is None:
            r = _int(obj.get("reasoning_tokens"))            # SGLang's UsageInfo
        out["reasoning_tokens"] = r
        return out

    def price(self, usage: dict, price: dict | None, at: float | None = None) -> int | None:
        return self.cost(usage, price)

    @staticmethod
    def cost(usage: dict, price: dict | None) -> int | None:
        """Micro-dollars: (prompt - cached) x input + cached x cached input + cache write x its
        price + completion x output, prices in USD per million tokens (so USD per million x tokens
        is micro-dollars). None without a price, without the counts, or without a price a non-zero
        count needs."""
        if not price or usage.get("prompt_tokens") is None or \
                usage.get("completion_tokens") is None:
            return None
        pin, pout = price.get("input"), price.get("output")
        if pin is None or pout is None:
            return None
        cached = usage.get("cached_tokens") or 0
        pc = price.get("cached_input")
        pc = pin if pc is None else pc
        cw = usage.get("cache_write_tokens") or 0
        pcw = price.get("cache_write")
        if cw and pcw is None:
            pcw = pin
        cost = ((usage["prompt_tokens"] - cached) * pin + cached * pc + cw * (pcw or 0)
                + usage["completion_tokens"] * pout)
        return int(round(cost))

    # ----------------------------------------------------------------- discovery
    models_path = "/models"

    def parse_models(self, obj) -> list[dict]:
        """`GET /models` into [{id, context_length}]."""
        out = []
        data = obj.get("data") if isinstance(obj, dict) else None
        for m in data or []:
            if isinstance(m, dict) and isinstance(m.get("id"), str):
                ctx = None
                for k in ("context_length", "max_model_len", "context_window"):
                    if isinstance(m.get(k), int):
                        ctx = m[k]
                        break
                out.append({"id": m["id"], "context_length": ctx})
        return out
