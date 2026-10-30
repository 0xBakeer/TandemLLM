"""#93: with thinking off, reasoning the model writes anyway goes to its field (Ornith on long contest
problems: its working-out, a lone `</think>`, then the answer; or a block it opens at its start).

  1  the rule: only the FIRST `</think>` alone on its line outside a code block, after no `<think>` of the
     model's own, splits (the working-out's own code blocks before it included, as Ornith writes them); a
     tag inside a code block or in a line of text, and every later one, stays as written
                                                            test_the_rule
  2  the formats: `reasoning_content` moves the working-out to its field, `tags` closes it as a block at
     the content's start, `both` does both                  test_the_formats
  3  streaming: the text before the tag has gone out, so only the tag line is dropped, in any piece
     size; a tag in a code block or in a line of text, and every later one, passes    test_the_stream_filter
  4  the server, raw and conductor, streamed and not: the tester's shapes (a lone `</think>`, an empty
     pair, a block opened at the start), a call after the split still parses, usage counts the moved
     text as reasoning tokens, and thinking on is untouched     test_the_server
  5  a system-first family's thinking-off line ends the system prompt instead of a developer message
     its template refuses (the 500 on Ornith)              test_a_system_first_line

CPU, the fake engine in this process. Run: python tests/test_think_off.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server.stream import OffClose, off_full, off_split  # noqa: E402

WORK = "Let me analyze this.\nWe need X.\nSo X = 5.\n"
LONE = WORK + "</think>\n\nThe answer is 5."
FENCED = "Here is the code:\n```python\nprint('</think>')\n```\nDone."
INCODE = "Here is the code:\n```\n</think>\n```\nDone."
WORKED = "Try it:\n```python\nx = 1\n```\nThat checks out.\n</think>\n\nThe answer is 1."
LATER = "Use this:\n```\nx\n```\nand then </think> stays."


def test_the_rule():
    assert off_split(LONE) == (WORK.rstrip("\n"), "The answer is 5.")
    assert off_split(FENCED) is None and off_split(LATER) is None and off_split(INCODE) is None
    assert off_split(WORKED) == ("Try it:\n```python\nx = 1\n```\nThat checks out.", "The answer is 1.")
    assert off_split("<think>\nhmm\n</think>\n\nok") is None             # the self-open block's case
    assert off_split("a <think> b </think> c") is None                  # not alone on a line
    assert off_split("plan\n<think>\nmore\n</think>\nok") is None       # its own pair: as written
    two = LONE + "\n```\nprint('</think>')\n```"
    assert off_split(two) == (WORK.rstrip("\n"), "The answer is 5.\n```\nprint('</think>')\n```")
    assert off_split("no tags at all") is None


def test_the_formats():
    c, r = off_full(LONE, "reasoning_content")
    assert (c, r) == ("The answer is 5.", WORK.rstrip("\n"))
    c, r = off_full(LONE, "tags")
    assert r is None and c == "<think>\n" + WORK + "</think>\n\nThe answer is 5."
    c, r = off_full(LONE, "both")
    assert r == WORK.rstrip("\n") and c.startswith("<think>\n") and c.endswith("The answer is 5.")
    assert off_full(FENCED, "tags") is None


def _stream(text, size):
    f = OffClose()
    out = "".join(f.push(text[i:i + size]) for i in range(0, len(text), size)) + f.finish()
    return out, f.dropped


def test_the_stream_filter():
    for size in (1, 2, 3, 7, 1000):
        assert _stream(LONE, size) == (WORK + "\nThe answer is 5.", 1), size
        assert _stream(WORKED, size) == (WORKED.replace("</think>\n", ""), 1), size
        for keep in (FENCED, LATER, INCODE, "a <think> b </think> c"):
            assert _stream(keep, size) == (keep, 0), (size, keep)
        twice = LONE + "\n</think>\nagain"
        assert _stream(twice, size) == (WORK + "\nThe answer is 5.\n</think>\nagain", 1), size


def _ask(text, stream, fmt="reasoning_content", thinking=False, tools=None, mode="raw"):
    from test_toolcheck import Req, serve
    serve(mode)
    body = {"messages": [{"role": "user", "content": text}], "stream": stream, "max_tokens": 2000,
            "chat_template_kwargs": {"enable_thinking": thinking}, "reasoning_format": fmt}
    if tools:
        body["tools"] = tools
    head, raw = Req(body, {"X-Tandem": "raw"} if mode == "raw" else None).response()
    out = {"content": "", "reasoning": "", "calls": [], "usage": None}
    if not stream:
        d = json.loads(raw)
        m = d["choices"][0]["message"]
        out.update(content=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
                   calls=[c["function"] for c in m.get("tool_calls") or []], usage=d.get("usage"))
        return out
    calls = {}
    for ln in raw.splitlines():
        if not ln.startswith("data: {"):
            continue
        d = json.loads(ln[6:])
        out["usage"] = d.get("usage") or out["usage"]
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            out["content"] += dl.get("content") or ""
            out["reasoning"] += dl.get("reasoning_content") or ""
            for tc in dl.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                c["name"] += (tc.get("function") or {}).get("name") or ""
                c["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    out["calls"] = [calls[k] for k in sorted(calls)]
    return out


ESC = LONE.replace("\n", "\\n")
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                          "required": ["path"]}}}]


def test_the_server():
    for mode in ("raw", "on"):
        # a lone `</think>`: moved (not streamed), dropped (streamed)
        got = _ask("FAKE_ANSWER " + ESC, stream=False, mode=mode)
        assert got["content"] == "The answer is 5." and got["reasoning"] == WORK.rstrip("\n"), (mode, got)
        assert got["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, got["usage"]
        got = _ask("FAKE_ANSWER " + ESC, stream=True, mode=mode)
        assert "</think>" not in got["content"] and got["content"].endswith("The answer is 5."), (mode, got)
        # tags: a closed block at the content's start
        got = _ask("FAKE_ANSWER " + ESC, stream=False, fmt="tags", mode=mode)
        assert got["content"].startswith("<think>\n") and got["content"].count("</think>") == 1, got
        # a block the model opened at its start, never closed (cut at the limit): all reasoning
        got = _ask("FAKE_ANSWER <think>\\nworking it out", stream=False, mode=mode)
        assert got["content"] == "" and "working it out" in got["reasoning"], (mode, got)
        # an empty pair first: dropped, the answer as written
        for stream in (False, True):
            got = _ask("FAKE_ANSWER <think>\\n\\n</think>\\n\\nLooking at it: 7.", stream=stream, mode=mode)
            assert got["content"] == "Looking at it: 7." and "</think>" not in got["content"], (mode, stream, got)
        # a call after the split still parses, both ways
        call = json.dumps({"name": "read", "arguments": {"path": "a.py"}})
        for stream in (False, True):
            got = _ask("FAKE_ANSWER " + ESC + "\nFAKE_CALL_JSON " + call, stream=stream, tools=TOOLS, mode=mode)
            assert [c["name"] for c in got["calls"]] == ["read"], (mode, stream, got)
            assert json.loads(got["calls"][0]["arguments"]) == {"path": "a.py"}
        # code that prints the tag stays as written
        for stream in (False, True):
            got = _ask("FAKE_ANSWER " + FENCED.replace("\n", "\\n"), stream=stream, mode=mode)
            assert got["content"] == FENCED and got["reasoning"] == "", (mode, stream, got)
    # thinking on: the release's split, untouched
    got = _ask("FAKE_REASONING plan\nFAKE_ANSWER done", stream=False, thinking=True)
    assert got["content"] == "done" and got["reasoning"].strip() == "plan", got
    return "raw and conductor, streamed and not"


def test_a_system_first_line():
    from engine.models.adapter import PromptSpec, ReasoningSpec, ServeSpec
    from server import app
    from test_modelspec import _prompt, _serve  # noqa: F401
    note = "Do not show your reasoning."
    spec = ServeSpec(reasoning=ReasoningSpec(off_note=note), prompt=PromptSpec(system_first=True))
    _serve(spec)
    body = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    client = app.thinking_mod.read(body, app._effort_caps())
    ids = app.build_prompt(body, client, count="ids")
    text = app.STATE["tok"].decode(ids, skip_special_tokens=False)
    assert "Be brief.\n\n" + note in text and "developer" not in text, text[-300:]
    body["messages"] = body["messages"][1:]
    text = app.STATE["tok"].decode(app.build_prompt(body, client, count="ids"), skip_special_tokens=False)
    assert note in text and text.index(note) < text.index("Say hi."), text[-300:]


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
