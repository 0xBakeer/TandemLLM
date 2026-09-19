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
    assert content == "hello world", "plain content flows with no delay"
    content += "".join(b.feed(" <tool_c"))
    content += "".join(b.feed("all>"))
    assert b.feed("<function=write_file><parameter=path>/x</parameter></function>") == []
    assert b.feed("</tool_call>") == []
    assert len(b.calls) == 1
    assert json.loads(b.calls[0]["function"]["arguments"]) == {"path": "/x"}
    content += b.flush()
    assert content.strip() == "hello world", "the held tail comes out at flush"


def test_buffer_does_not_delay_plain_content():
    # Certification regression: holding a fixed 9-char tail delayed the first delta and moved the
    # row's TTFT +14 %. With no partial opener, everything flows immediately.
    b = ToolCallBuffer()
    assert "".join(b.feed("the quick brown fox")) == "the quick brown fox"
    assert b.feed("x") == ["x"]
    assert b.feed("<") == []                      # a lone '<' could begin an opener
    assert b.feed("t") == []                      # still could
    assert "".join(b.feed("he answer")) == "<the answer"  # dis-proven: released whole


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


def _stream_charwise(b: ToolCallBuffer, text: str) -> tuple[str, list[dict]]:
    """Feed one character at a time -- the harshest split -- and collect content + deltas."""
    content, deltas = "", []
    for ch in text:
        content += "".join(b.feed(ch))
        deltas += b.drain_deltas()
    return content, deltas


def _arguments_from(deltas: list[dict]) -> str:
    return "".join(d["function"]["arguments"] for d in deltas if "arguments" in d["function"])


def test_streams_the_call_while_the_block_is_written():
    # The rolex_svg fix: a whole-file call used to be withheld and then dumped whole at the very
    # end. The call's name and argument fragments must flow as the model writes them.
    b = ToolCallBuffer()
    content, deltas = _stream_charwise(b, "I'll write it now.\n\n" + BLOCK + " done")
    assert content.startswith("I'll write it now.\n\n")
    assert len(deltas) > 5, "arguments must arrive as many fragments, not one lump"
    start = deltas[0]
    assert start["index"] == 0 and start["id"].startswith("call_")
    assert start["function"] == {"name": "write_file", "arguments": ""}
    assert json.loads(_arguments_from(deltas)) == {"path": "/app/a.html", "content": "<p>hi</p>"}
    assert b.calls[0]["id"] == start["id"], "the collected call keeps the streamed id"
    assert start["id"] in b.streamed_ids, "and the end-of-stream sweep must skip it"
    assert content == "I'll write it now.\n\n done"


def test_streamed_values_are_trimmed_and_escaped():
    value = 'he said "hi" then \\ and\nnew line  '
    block = ("<tool_call><function=note><parameter=text>\n" + value + "\n</parameter>"
             "</function></tool_call>")
    b = ToolCallBuffer()
    _, deltas = _stream_charwise(b, block)
    assert json.loads(_arguments_from(deltas)) == {"text": 'he said "hi" then \\ and\nnew line'}


def test_streamed_value_with_angle_bracket():
    # A content parameter is HTML; its `<` must not end the value.
    block = ("<tool_call><function=write_file><parameter=content>\n<!DOCTYPE html>\n</parameter>"
             "</function></tool_call>")
    b = ToolCallBuffer()
    _, deltas = _stream_charwise(b, block)
    assert json.loads(_arguments_from(deltas)) == {"content": "<!DOCTYPE html>"}


def test_unknown_shape_never_streams_partially():
    b = ToolCallBuffer()
    content, deltas = _stream_charwise(b, "a <tool_call>junk</tool_call> b")
    assert deltas == [], "an unrecognised block must not vend half a call"
    assert content + b.flush() == "a <tool_call>junk</tool_call> b"
    assert b.calls == []


def test_two_blocks_stream_with_increasing_index():
    b = ToolCallBuffer()
    text = ("<tool_call><function=one><parameter=a>1</parameter></function></tool_call>"
            "<tool_call><function=two><parameter=b>2</parameter></function></tool_call>")
    _, deltas = _stream_charwise(b, text)
    starts = [d for d in deltas if "id" in d]
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["function"]["name"] for s in starts] == ["one", "two"]
    assert len(b.calls) == 2 and b.streamed_ids == {s["id"] for s in starts}


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:52s} ok")
            passed += 1
    print(f"{passed} passed")
