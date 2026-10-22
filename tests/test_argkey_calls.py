"""SRV-683, SRV-687, SRV-688: the arg-key tool-call form (tool_format "argkey") and a model that opens its own reasoning block
(ReasoningSpec.opens_itself): a block rewritten into the JSON tier with the values as written and typed by the
schema, streamed in any piece sizes, a bad block left as text; the reasoning split in every format."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.stream import Reasoning, split_full  # noqa: E402
from server.toolcall import ToolCallBuffer, call_format, diagnose, parse_tool_calls  # noqa: E402
from server.toolformats.argkey import ArgKeyCalls, argkey_calls, rewrite_argkey  # noqa: E402

TEXT = ("Sure.<tool_call>write_file<arg_key>path</arg_key><arg_value>/a b.txt</arg_value><arg_key>content</arg_key>"
        "<arg_value>x\n  y\n</arg_value><arg_key>n</arg_key><arg_value>3</arg_value></tool_call><tool_call>get_time"
        "</tool_call>")


def test_rewrite_and_types():
    content, calls = parse_tool_calls(rewrite_argkey(TEXT, eos=True), eos=True,
                                      types={"write_file": {"n": frozenset({"integer"})}})
    assert content == "Sure."
    assert [c["function"]["name"] for c in calls] == ["write_file", "get_time"]
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "/a b.txt", "content": "x\n  y\n", "n": 3}      # the string exactly as written


def test_stream_any_pieces():
    want = rewrite_argkey(TEXT, eos=True)
    for size in (1, 2, 5, 13, len(TEXT)):
        r = ArgKeyCalls()
        got = "".join(r.feed(TEXT[i:i + size]) for i in range(0, len(TEXT), size)) + r.finish(eos=True)
        assert got == want, size


def test_bad_block_is_text():
    bad = "<tool_call>f<arg_key>x</arg_key>oops</tool_call>"
    assert argkey_calls("f<arg_key>x</arg_key>oops") is None
    assert rewrite_argkey(bad) == bad
    assert rewrite_argkey("<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value>", eos=False).startswith("<tool_call>f")


TYPES = {"write_file": {"n": frozenset({"integer"})}}


def _buffer(text: str, size: int, **kw) -> tuple[str, dict, list]:
    """The stream's buffer in the arg-key format fed `text` in pieces of `size`: (content, the
    calls as a client assembles them by index, the argument deltas per call)."""
    b = ToolCallBuffer(names=["write_file", "get_time"], types=TYPES, fmt="argkey", **kw)
    content, deltas = "", []
    for i in range(0, len(text), size):
        content += "".join(b.feed(text[i:i + size]))
        deltas += b.drain_deltas()
    left, tail = b.finish(eos=True)
    content += left
    deltas += b.drain_deltas() + tail
    calls: dict = {}
    for d in deltas:
        c = calls.setdefault(d["index"], {"name": "", "args": "", "n": 0})
        c["name"] += d["function"].get("name") or ""
        c["args"] += d["function"].get("arguments") or ""
        c["n"] += 1
    return content, calls, b


def test_buffer_streams_the_arguments_as_written():
    """SRV-687: the buffer reads the arg-key form itself; a string argument goes out in many
    deltas as it is written, exactly (no trimming), and the calls equal the whole-answer parse."""
    _, want = parse_tool_calls(rewrite_argkey(TEXT, eos=True), eos=True, types=TYPES)
    for size in (1, 2, 5, 13, len(TEXT)):
        content, calls, b = _buffer(TEXT, size)
        assert content == "Sure.", (size, content)
        assert [(c["name"], json.loads(c["args"])) for c in calls.values()] == \
            [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in want], size
        if size <= 5:
            assert calls[0]["n"] > 5, (size, calls[0])                 # streamed, not one delta
    _, calls, _ = _buffer("<tool_call>write_file<arg_key>content</arg_key><arg_value>  two  \n"
                          "</arg_value></tool_call>", 3)
    assert json.loads(calls[0]["args"]) == {"content": "  two  \n"}


def test_buffer_holds_and_falls_back_as_written():
    """Held (the conductor's checks): one whole delta a call; a block that is not the form is held
    as the text the model wrote, and without holding it goes out as that text."""
    content, calls, b = _buffer(TEXT, 4, hold=True)
    held = b.take_held()
    assert content == "Sure." and not calls and [c["function"]["name"] for c in held] == \
        ["write_file", "get_time"]
    bad = "<tool_call>write_file<arg_key>x</arg_key>oops</tool_call>"
    content, calls, b = _buffer("Hi." + bad, 3, hold=True)
    assert b.take_held() == [{"bad": "write_file<arg_key>x</arg_key>oops", "closed": True}]
    content, calls, _ = _buffer("Hi." + bad, 3)
    assert content == "Hi." + bad and not calls


def test_diagnose_and_format_line_name_no_placeholder():
    """SRV-688: C4's reason and its format line in the arg-key form, with a real tool and key."""
    assert diagnose("write_file<arg_key>x</arg_key>oops", "argkey") == \
        "argument 'x' has no <arg_value> after its </arg_key>"
    assert diagnose("<arg_key>x</arg_key>", "argkey") == "the block does not begin with the tool's name"
    assert diagnose("f<arg_key>x</arg_key><arg_value>1", "argkey") == "argument 'x' has no </arg_value>"
    line = call_format("argkey", "write_file", "path")
    assert line.startswith("<tool_call>write_file, ") and "<arg_key>path</arg_key>" in line
    for fmt in (None, "argkey", "pythonic"):
        for text in (call_format(fmt, "read", "filePath"), diagnose("<function>\nx"),
                     diagnose("hello")):
            assert "NAME" not in text and "KEY" not in text, (fmt, text)


def test_self_opened_reasoning_every_format():
    pieces = ["<th", "ink>", "\nplan a", "nd b</th", "ink>\n\nAnswer", "!"]
    for fmt in ("tags", "reasoning_content", "both"):
        r = Reasoning(fmt, in_think=False, self_open=True)
        out = [x for p in pieces for x in r.push(p)] + r.finish()
        content = "".join(t for f, t in out if f == "content")
        reasoning = "".join(t for f, t in out if f == "reasoning")
        assert content.strip() == "Answer!", (fmt, out)
        if fmt != "tags":
            assert reasoning.strip() == "plan and b", (fmt, out)
        r = Reasoning(fmt, in_think=False, self_open=True)
        out = [x for p in ("  He", "llo") for x in r.push(p)] + r.finish()
        assert out == [("content", "  He"), ("content", "llo")], (fmt, out)
    assert split_full("<think>\nx y</think>\n\nAns", "reasoning_content", in_think=False, self_open=True) == ("Ans", "x y")
    assert split_full("<think>x</think>A", "tags", in_think=False, self_open=True) == ("<think>x</think>A", None)
    assert split_full("<think>x</think>A", "reasoning_content", in_think=False) == ("<think>x</think>A", None)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
