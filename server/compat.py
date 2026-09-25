"""The OpenAI request surface, one disposition per field (SRV-17), and the checks that go with it.

Silently ignoring a field a caller depends on is the worst answer a server can give: the client
cannot tell an ignored `n` or `logit_bias` from a model that happened to answer that way. So every
OpenAI field has exactly one of three dispositions here, and `tests/test_compat.py` pins the table,
so a field cannot be added -- or quietly dropped -- without the test saying so:

  * HONOURED -- the field works;
  * NEUTRAL  -- accepted, and by its own definition it does not change the output (`store`, a
    latency hint such as `prediction`, a label such as `metadata`); the RUNBOOK says so;
  * REFUSED  -- a 400 naming the field (for the values the engine cannot serve).

Fields that are not OpenAI's (the engine's own -- `top_k`, `min_p`, `repetition_penalty`,
`chat_template_kwargs`, ... -- and whatever else a client adds) are not refused: clients send
extensions for other servers, and a 400 for those would break them without helping anyone.

`tool_choice` (SRV-14) lives here too: auto / none / required / a named function, checked against
the request's `tools`.
"""

from __future__ import annotations

HONOURED, NEUTRAL, REFUSED = "honoured", "neutral", "refused"

CHAT_FIELDS = {
    "messages": HONOURED, "model": HONOURED, "stream": HONOURED, "stream_options": HONOURED,
    "max_tokens": HONOURED, "max_completion_tokens": HONOURED, "stop": HONOURED,
    "temperature": HONOURED, "top_p": HONOURED, "seed": HONOURED,
    "presence_penalty": HONOURED, "frequency_penalty": HONOURED, "logit_bias": HONOURED,
    "logprobs": HONOURED, "top_logprobs": HONOURED, "n": HONOURED,
    "tools": HONOURED, "tool_choice": HONOURED, "parallel_tool_calls": HONOURED,
    "reasoning_effort": HONOURED, "response_format": HONOURED,
    # a conversation label for the state cache (server/app.py::conversation_id); no identity
    "user": NEUTRAL, "metadata": NEUTRAL,
    "store": NEUTRAL, "service_tier": NEUTRAL, "prompt_cache_key": NEUTRAL,
    "safety_identifier": NEUTRAL,
    # Predicted outputs change latency, never the text: honouring it is a speed question
    "prediction": NEUTRAL,
    # text is the only modality; `["text"]` is accepted, anything else refused
    "modalities": HONOURED,
    "audio": REFUSED, "functions": REFUSED, "function_call": REFUSED,
    "web_search_options": REFUSED, "verbosity": REFUSED,
}

COMPLETION_FIELDS = {
    "prompt": HONOURED, "model": HONOURED, "stream": HONOURED, "stream_options": HONOURED,
    "max_tokens": HONOURED, "stop": HONOURED, "temperature": HONOURED, "top_p": HONOURED,
    "seed": HONOURED, "presence_penalty": HONOURED, "frequency_penalty": HONOURED,
    "logit_bias": HONOURED, "logprobs": HONOURED, "n": HONOURED,
    "user": NEUTRAL,
    # `best_of: 1` and `echo: false` are the defaults and accepted; anything else is refused
    "best_of": HONOURED, "echo": HONOURED, "suffix": REFUSED,
}

MAX_N = 16
MAX_TOP_LOGPROBS = 20
MAX_COMPLETION_LOGPROBS = 5


class Refusal(ValueError):
    """A request the engine will not serve as asked: a 400 that names the field."""

    def __init__(self, param: str, message: str):
        super().__init__(message)
        self.param = param

    def body(self) -> dict:
        return {"error": {"message": str(self), "type": "invalid_request_error",
                          "param": self.param}}


def _int(body: dict, name: str, lo: int, hi: int, default: int) -> int:
    v = body.get(name)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise Refusal(name, f"{name} must be an integer in [{lo}, {hi}], got {v!r}")
    return v


def check(body: dict, chat: bool) -> None:
    """Refuse what the engine cannot serve, naming the field. Everything else passes."""
    table = CHAT_FIELDS if chat else COMPLETION_FIELDS
    for name, disp in table.items():
        if disp == REFUSED and body.get(name) is not None:
            hint = {"functions": "; use tools", "function_call": "; use tool_choice"}.get(name, "")
            raise Refusal(name, f"{name} is not supported by this engine{hint}")
    n = _int(body, "n", 1, MAX_N, 1)
    if n > 1 and body.get("stream"):
        raise Refusal("n", "n > 1 is served for non-streamed requests only")
    rf = body.get("response_format")
    if rf is not None:
        kind = rf.get("type") if isinstance(rf, dict) else None
        if kind != "text":
            raise Refusal("response_format",
                          f"response_format type {kind!r} is not supported (text only)")
    if chat:
        mods = body.get("modalities")
        if mods is not None and list(mods) != ["text"]:
            raise Refusal("modalities", "text is the only modality this engine serves")
        lp = body.get("logprobs")
        if lp is not None and not isinstance(lp, bool):
            raise Refusal("logprobs", "logprobs must be true or false")
        _int(body, "top_logprobs", 0, MAX_TOP_LOGPROBS, 0)
        if body.get("top_logprobs") and not lp:
            raise Refusal("top_logprobs", "top_logprobs needs logprobs: true")
        ptc = body.get("parallel_tool_calls")
        if ptc is not None and not isinstance(ptc, bool):
            raise Refusal("parallel_tool_calls", "parallel_tool_calls must be true or false")
        tool_choice(body)
    else:
        _int(body, "logprobs", 0, MAX_COMPLETION_LOGPROBS, 0)
        if body.get("echo"):
            raise Refusal("echo", "echo is not supported by this engine")
        if _int(body, "best_of", 1, MAX_N, 1) not in (1, n):
            raise Refusal("best_of", "best_of must equal n: the engine returns every choice")


def top_logprobs(body: dict, chat: bool) -> int | None:
    """None when the request did not ask for log-probabilities, else how many alternatives."""
    if chat:
        return int(body.get("top_logprobs") or 0) if body.get("logprobs") else None
    lp = body.get("logprobs")
    return None if lp is None else int(lp)


def tool_names(body: dict) -> list[str]:
    out = []
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out.append(fn["name"])
    return out


def tool_choice(body: dict) -> tuple[str, str | None]:
    """`(mode, name)`: auto / none / required / named, validated against the request's tools."""
    tc = body.get("tool_choice")
    if tc is None or tc == "auto":
        return "auto", None
    if tc == "none":
        return "none", None
    names = tool_names(body)
    if tc == "required":
        if not names:
            raise Refusal("tool_choice", "tool_choice required needs tools")
        return "required", None
    if isinstance(tc, dict) and tc.get("type") == "function" and isinstance(tc.get("function"),
                                                                              dict):
        name = tc["function"].get("name")
        if not isinstance(name, str) or name not in names:
            raise Refusal("tool_choice", f"tool_choice names {name!r}, which is not in tools")
        return "named", name
    raise Refusal("tool_choice", "tool_choice must be auto, none, required or "
                                 '{"type": "function", "function": {"name": ...}}')


def directive(mode: str, name: str | None) -> str | None:
    """SRV-14's prompt-level enforcement: the sentence the system turn ends with, or None."""
    if mode == "required":
        return "You must call at least one of the functions above in this reply."
    if mode == "named":
        return f"You must call the function {name} in this reply."
    return None


def logit_bias(body: dict, vocab: int) -> dict[int, float] | None:
    """`{token id: bias}` from OpenAI's `{"id": value}`, ids in the vocabulary, values in
    [-100, 100]; None when absent or empty."""
    raw = body.get("logit_bias")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise Refusal("logit_bias", "logit_bias must map token ids to numbers")
    out: dict[int, float] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            raise Refusal("logit_bias", f"logit_bias key {k!r} is not a token id") from None
        if not 0 <= tid < vocab:
            raise Refusal("logit_bias", f"logit_bias token id {tid} is outside the vocabulary")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not -100 <= v <= 100:
            raise Refusal("logit_bias", f"logit_bias value for {tid} must be in [-100, 100]")
        if v:
            out[tid] = float(v)
    return out or None
