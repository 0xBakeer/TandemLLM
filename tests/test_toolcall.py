"""The Qwen XML tool-call parser, unit-tested on a CPU (ENG-27)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.toolcall import ToolCallBuffer, parse_tool_calls  # noqa: E402

BLOCK = ("<tool_call>\n<function=write_file>\n<parameter=path>\n/app/a.html\n</parameter>\n"
         "<parameter=content>\n<p>hi</p>\n</parameter>\n</function>\n</tool_call>")


def test_parse_single_block():
    content, calls = parse_tool_calls("I'll write it now.\n\n" + BLOCK)
    assert content == "I'll write it now.\n\n"
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "write_file"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "/app/a.html", "content": "<p>hi</p>"}
    assert calls[0]["id"].startswith("call_")


def test_consecutive_duplicate_calls_are_dropped():
    content, calls = parse_tool_calls(BLOCK + "\n" + BLOCK)
    assert content == "\n"
    assert len(calls) == 1, "the echoed duplicate must not be executed twice"


def test_parse_two_blocks_and_text_between():
    text = "<tool_call><function=one><parameter=a>1</parameter></function></tool_call> and " + BLOCK
    content, calls = parse_tool_calls(text)
    assert content == " and "
    assert [c["function"]["name"] for c in calls] == ["one", "write_file"]


def test_unparseable_block_stays_in_content():
    bad = "<tool_call>no function here</tool_call>"
    content, calls = parse_tool_calls("x " + bad)
    assert calls == [] and bad in content


def test_buffer_releases_content_and_collects_calls():
    # The buffer holds up to len("<tool_call>")-1 characters back so a split opener is never
    # shown; content is what feed() releases plus flush() at the end.
    b = ToolCallBuffer()
    content = "".join(b.feed("hello ") + b.feed("world"))
    assert b.calls == [] and not content.endswith("world"), "the tail is held, not lost"
    content += "".join(b.feed(" <tool_c"))
    content += "".join(b.feed("all>"))
    assert b.feed("<function=write_file><parameter=path>/x</parameter></function>") == []
    assert b.feed("</tool_call>") == []
    assert len(b.calls) == 1
    assert json.loads(b.calls[0]["function"]["arguments"]) == {"path": "/x"}
    content += b.flush()
    assert content.strip() == "hello world", "the held tail comes out at flush"


def test_buffer_flushes_an_unclosed_block_as_content():
    b = ToolCallBuffer()
    assert b.feed("text <tool_call><function=foo>") == ["text "]
    left = b.flush()
    assert left.startswith("<tool_call>") and "foo" in left
    assert b.calls == []


def test_buffer_releases_an_unparseable_block():
    b = ToolCallBuffer()
    out = "".join(b.feed("a <tool_call>junk</tool_call> b"))
    out += b.flush()
    assert out == "a <tool_call>junk</tool_call> b"
    assert b.calls == []


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:52s} ok")
            passed += 1
    print(f"{passed} passed")
