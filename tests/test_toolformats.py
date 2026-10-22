"""ENG-407 (ENG-E17): every tool-call form behind one interface, the contract each must keep.

For every registered `ToolFormat` (server/toolformats) and every family's `ServeSpec.tool_format`:

  1  the family's form is registered; the default is the XML form
  2  a call rendered in the form parses back to the same name and arguments: whole (`rewrite` +
     `parse_tool_calls`) and streamed through the ToolCallBuffer in pieces of 1 to N characters, with
     the form's stream rewriter in front when it has one; a typed argument comes back typed
  3  the forced-call constraint accepts the form's forced rendering, and a call of another form is not
     accepted where the two differ
  4  C4's line shows the offered tool and argument and no placeholder; a broken block's reason names none
  5  the argument loop guard reads the open call's name

CPU. Run: python tests/test_toolformats.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine import grammar as G  # noqa: E402
from engine.models import REGISTRY  # noqa: E402
from server import toolformats  # noqa: E402
from server.toolcall import ToolCallBuffer, diagnose, parse_tool_calls  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                     "n": {"type": "integer"}}, "required": ["path", "content"]}}}]
TYPES = {"write_file": {"n": frozenset({"integer"})}}
ARGS = {"path": "/tmp/a b.txt", "content": "line one\nline two", "n": 3}


def _formats():
    toolformats.get(None)
    return dict(toolformats.FORMATS)


def test_every_family_form_is_registered():
    fmts = _formats()
    assert toolformats.get(None).name == toolformats.DEFAULT and toolformats.DEFAULT in fmts
    for name, a in REGISTRY.items():
        assert a.spec.tool_format in fmts, (name, a.spec.tool_format)
    try:
        toolformats.get("no-such-form")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown form must fail")
    return f"{len(fmts)} forms: {sorted(fmts)}"


def _stream(fmt, text: str, size: int) -> list:
    b = ToolCallBuffer(names=["write_file"], types=TYPES, fmt=fmt)
    rw = fmt.stream_rewriter()
    content = ""
    for i in range(0, len(text), size):
        piece = text[i:i + size]
        content += "".join(b.feed(rw.feed(piece) if rw is not None else piece))
        b.drain_deltas()
    if rw is not None:
        content += "".join(b.feed(rw.finish(eos=True)))
    left, _ = b.finish(eos=True)
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in b.calls], content + left


def test_a_rendered_call_reads_back_in_every_form():
    for name, fmt in _formats().items():
        text = "Writing it." + fmt.render("write_file", ARGS)
        content, calls = parse_tool_calls(fmt.rewrite(text, eos=True), eos=True, types=TYPES)
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
        assert got == [("write_file", ARGS)] and content == "Writing it.", (name, got, content)
        for size in (1, 2, 3, 7, len(text)):
            calls, content = _stream(fmt, text, size)
            assert calls == [("write_file", ARGS)] and content == "Writing it.", (name, size, calls, content)
    return f"{len(_formats())} forms, whole and streamed"


def test_the_forced_constraint_takes_the_forms_own_rendering():
    fmts = _formats()
    for name, fmt in fmts.items():
        pat, typed = fmt.forced_regex(TOOLS, ["write_file"], many=False)
        assert typed == {"write_file": {"n"}}, (name, typed)
        d = G.compile_regex(pat)
        own = fmt.forced_render("write_file", ARGS)
        assert d.accept[d.walk(0, own.encode())], (name, own)
        for other in fmts.values():
            alt = other.forced_render("write_file", ARGS)
            if alt != own:
                assert not d.accept[d.walk(0, alt.encode())], (name, other.name)
        assert set(fmt.markers) <= set(G.CALL_MARKERS) or name != toolformats.DEFAULT


def test_the_notes_show_a_real_tool_and_no_placeholder():
    for name, fmt in _formats().items():
        line = fmt.example("write_file", "path")
        assert "write_file" in line and "path" in line, (name, line)
        for t in (line, fmt.example("write_file"), diagnose("<function>\nx", name), diagnose("hello", name),
                  diagnose("write_file<arg_key>x</arg_key>oops", name)):
            assert "NAME" not in t and "KEY" not in t, (name, t)


def test_the_block_reader_base_takes_nothing():
    r = toolformats.BlockReader()
    r.reset()
    r.stream(None, 0)
    assert r.takes("anything") is False and r.parse("x", None) is None


def test_the_loop_guard_reads_the_open_calls_name():
    for name, fmt in _formats().items():
        text = fmt.forced_render("write_file", ARGS)
        opened = text.split("<tool_call>", 1)[1][:40] if "<tool_call>" in text else ""
        assert fmt.open_call_name(opened) == "write_file", (name, opened)
        assert fmt.open_call_name("") is None


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
