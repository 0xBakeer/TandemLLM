"""SRV-683: the arg-key tool-call form (tool_format "argkey") and a model that opens its own reasoning block
(STATE["self_think"]): a block rewritten into the JSON tier with the values as written and typed by the
schema, streamed in any piece sizes, a bad block left as text; the reasoning split in every format."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.stream import Reasoning, split_full  # noqa: E402
from server.toolcall import ArgKeyCalls, argkey_calls, parse_tool_calls, rewrite_argkey  # noqa: E402

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
