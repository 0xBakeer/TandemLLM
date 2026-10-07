"""Plain API clients get a clean answer (SRV-245, SRV-247; UX2's normal-user run, 2026-10-07).

On 0.3.48-test an OpenAI SDK or curl client that named no `reasoning_format` got the thinking in
`content` as `<think>...</think>`; `stop: ["\\n\\n"]` ended a response at the first blank line of
the reasoning (55 tokens, no answer); a 60k-token prompt's progress lines went into `content`; and
a memory question from a client that offered no tools ended with the model's own `tool` call
(a GET on the memory search) handed over with finish `tool_calls`.

  1  the default under a conductor profile: every client kind gets `reasoning_content` (Hermes
     keeps `tags`); `reasoning_format: tags` keeps the tags; the raw model keeps the server's
  2  a stop string applies to the answer: one the reasoning holds does not cut it, and a blank
     line does not match the template's blank lines after `</think>`; streamed and not
  3  the answer does not start with the blank lines after the reasoning
  4  state lines never in a plain client's `content`: in `reasoning_content`, or none with tags
  5  a request without tools never gets `tool_calls`: the call is read out of the answer and
     dropped, finish `stop`, the journal row names it; streamed and not; with tools it is a call

Run: python tests/test_api_clients.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_ui_states as tu  # noqa: E402
from server.conductor import states  # noqa: E402

CURL = {"User-Agent": "curl/8.5.0"}
SDK = {"User-Agent": "OpenAI/Python 2.24.0"}


def ask(text, headers=CURL, stream=True, **extra):
    body = {"model": "TandemLLM", "stream": stream, "max_tokens": 2000,
            "messages": [{"role": "user", "content": text}], **extra}
    return tu._go(body, headers)


def finish_of(out) -> str | None:
    if not out["raw"].lstrip().startswith("data:"):
        return json.loads(out["raw"])["choices"][0]["finish_reason"]
    fin = None
    for ln in out["raw"].splitlines():
        if ln.startswith("data: {"):
            for ch in json.loads(ln[6:]).get("choices") or ():
                fin = ch.get("finish_reason") or fin
    return fin


# ------------------------------------------------------------------ 1 the default

def test_1_sdk_and_curl_get_reasoning_content():
    tu.serve("on")
    for h in (CURL, SDK):
        for stream in (True, False):
            out = ask("Say hello in one short sentence.", h, stream)
            assert "<think>" not in out["content"] and "</think>" not in out["content"], out
            assert "The question has" in out["reasoning"], out["reasoning"][:80]
    tags = ask("Say hello in one short sentence.", CURL, reasoning_format="tags")
    assert tags["content"].startswith("<think>\n") and "</think>" in tags["content"]
    # the raw model: the server's format, as the release
    raw = tu._go({"model": "TandemLLM", "stream": False, "max_tokens": 200,
                  "messages": [{"role": "user", "content": "Say hello."}]},
                 dict(CURL, **{"X-Tandem": "raw"}))
    assert raw["reasoning"] == "" and "reasoning_content" not in raw["raw"], raw["raw"][:200]


# ------------------------------------------------------------------ 2-3 stops and the answer

def test_2_a_stop_string_applies_to_the_answer():
    tu.serve("on")
    for stream in (True, False):
        # "question" is in the fake's reasoning only; before SRV-245 it cut the response there
        out = ask("Write two short paragraphs about cats.", stream=stream, stop=["question"])
        assert out["content"] and "<think>" not in out["content"], out
        assert finish_of(out) == "stop"
        # a blank line: the template's blank lines after </think> are not the answer's
        nl = ask("Write two short paragraphs about cats.", stream=stream, stop=["\n\n"])
        assert len(nl["content"]) > 20, nl["content"]
    # in the answer, a stop still cuts: the fake's answer is words; cut at its third space
    full = ask("Write two short paragraphs about cats.", stream=False)["content"]
    word = full.split(" ")[2]
    cut = ask("Write two short paragraphs about cats.", stream=False, stop=[" " + word])
    assert full.startswith(cut["content"]) and len(cut["content"]) < len(full), cut["content"]
    # tags asked for: the block whole, the answer cut
    t = ask("Write two short paragraphs about cats.", stream=True, stop=["question"],
            reasoning_format="tags")
    assert "</think>" in t["content"] and t["content"].split("</think>", 1)[1].strip()


def test_3_the_answer_does_not_start_with_blank_lines():
    tu.serve("on")
    for h in (CURL, SDK):
        for stream in (True, False):
            out = ask("Say hello in one short sentence.", h, stream)
            assert out["content"] and not out["content"].startswith("\n"), repr(out["content"][:20])


# ------------------------------------------------------------------ 4 state lines

def test_4_state_lines_never_in_a_plain_clients_content():
    tu.serve("on")
    out = ask("Say hello in one short sentence.")
    assert states.MARKER not in out["content"], out["content"][:200]
    assert states.MARKER in out["reasoning"], "the lines go to reasoning_content"
    tags = ask("Say hello in one short sentence.", reasoning_format="tags")
    assert states.MARKER not in tags["content"], tags["content"][:200]
    flat = ask("Say hello in one short sentence.", stream=False, reasoning_format="tags")
    assert states.MARKER not in flat["content"], flat["content"][:200]


# ------------------------------------------------------------------ 5 no tools, no tool_calls

def test_5_no_tools_no_tool_calls():
    cond, _ = tu.serve("on")
    for stream in (True, False):
        out = ask("What is my UX1 pet? FAKE_CALL_NOTOOLS", stream=stream)
        assert out["calls"] == [] and '"tool_calls"' not in out["raw"], out["raw"][-400:]
        assert "<tool_call>" not in out["content"] and "/v1/memory" not in out["content"]
        assert out["content"].startswith("Let me check my memory"), out["content"]
        assert finish_of(out) == "stop", finish_of(out)
    # with tools offered, the same answer is a call
    TOOL = [{"type": "function", "function": {"name": "tool", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}}}}]
    out = ask("What is my UX1 pet? FAKE_CALL_NOTOOLS", tools=TOOL)
    assert out["calls"] and finish_of(out) == "tool_calls", out["raw"][-300:]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
