"""SRV-691 (QA SRV-692): a call's argument names matched to the offered tool's schema before it leaves the server.

  1  the rules: exact; case, `_` and `-`; the synonym table (a path for the one path-like parameter, query /
     pattern / regex, cmd / command, content / text / contents); nothing when the target is ambiguous or
     already set; a value is never changed but for the target's JSON type
  2  the gate: a call not streamed yet is renamed only when it then has its required parameters
  3  the stream: the buffer renames a key as it is decided, in every tool-call form, and the final call is
     the one that streamed (no second call from the sweep); a held call is renamed at its closer
  4  the server, both paths: opencode's `read` with `path` reaches the client as `filePath`, streamed and
     not, raw and under the conductor, which journals the rename

The real failure (2026-10-22, Flint 4.0 at its card's sampling, SRV-690): `read` with
{"path": "/tmp/proj/src/config.py"} where opencode's schema requires `filePath`.
CPU. Run: python tests/test_callrepair.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import callrepair as CR  # noqa: E402
from server import toolformats  # noqa: E402
from server.toolcall import ToolCallBuffer, schema_types  # noqa: E402

OC = json.load(open(os.path.join(HERE, "fixtures", "opencode_tools.json")))["tools"]
SCHEMAS = CR.schemas_of(OC)
READ = SCHEMAS["read"]


def _ev(tool, frm, to, kind="argument"):
    return {"kind": kind, "tool": tool, "from": frm, "to": to}


def test_the_rules():
    assert CR.rename_key("filePath", READ, set()) is None                       # exact
    assert CR.rename_key("file_path", READ, set()) == "filePath"                # case and underscore
    assert CR.rename_key("FILEPATH", READ, set()) == "filePath"
    assert CR.rename_key("path", READ, set()) == "filePath"                     # the one path-like parameter
    assert CR.rename_key("file", READ, set()) == "filePath"
    assert CR.rename_key("path", READ, {"filePath"}) is None                    # already set
    assert CR.rename_key("cmd", SCHEMAS["bash"], set()) == "command"
    assert CR.rename_key("query", SCHEMAS["grep"], set()) == "pattern"
    assert CR.rename_key("text", SCHEMAS["write"], set()) == "content"
    assert CR.rename_key("color", READ, set()) is None                          # no rule
    two = {"type": "object", "properties": {"src_path": {"type": "string"}, "dst_path": {"type": "string"}}}
    assert CR.rename_key("path", two, set()) is None                            # ambiguous
    same = {"type": "object", "properties": {"filePath": {}, "file_path": {}}}
    assert CR.rename_key("FilePath", same, set()) is None                       # two by case alone
    assert CR.rename_key("x", {"type": "object"}, set()) is None and CR.rename_key("x", None, set()) is None


def test_the_real_failure_and_the_gate():
    args, r = CR.repair_args("read", {"path": "/tmp/proj/src/config.py"}, READ)
    assert args == {"filePath": "/tmp/proj/src/config.py"} and r == [_ev("read", "path", "filePath")]
    # gated: the rename does not give the call its required parameters -> as written
    edit = SCHEMAS["edit"]
    args, r = CR.repair_args("edit", {"path": "/x", "old": "a"}, edit)
    assert r == [] and args == {"path": "/x", "old": "a"}
    args, r = CR.repair_args("edit", {"path": "/x", "old": "a"}, edit, gate=False)
    assert args["filePath"] == "/x" and r == [_ev("edit", "path", "filePath")]
    # typed: a renamed value takes its target's JSON type, as the parser converts it
    s = {"type": "object", "properties": {"timeout": {"type": "integer"}, "command": {"type": "string"}},
         "required": ["command"]}
    args, r = CR.repair_args("bash", {"cmd": "ls", "Timeout": "120"}, s, kinds={"timeout": frozenset({"integer"})})
    assert args == {"command": "ls", "timeout": 120}, args
    assert CR.repair_args("x", "not a dict", READ) == ("not a dict", [])
    call = {"id": "c", "type": "function", "function": {"name": "read", "arguments": '{"path": "/a"}'}}
    c2, r = CR.repair_call(call, SCHEMAS)
    assert json.loads(c2["function"]["arguments"]) == {"filePath": "/a"} and c2["id"] == "c"
    bad = {"function": {"name": "read", "arguments": "{not json"}}
    assert CR.repair_call(bad, SCHEMAS) == (bad, [])
    assert CR.repair_call({"function": {"name": "nope", "arguments": "{}"}}, SCHEMAS)[1] == []
    calls, r = CR.repair_calls([call, bad], SCHEMAS)
    assert len(calls) == 2 and r == [_ev("read", "path", "filePath")]
    assert CR.schemas_of([{"name": "flat", "parameters": {"type": "object"}}, "junk", {"type": "function"}]) == \
        {"flat": {"type": "object"}}


def _stream(text: str, fmt=None, hold=False, size=3):
    b = ToolCallBuffer(names=list(SCHEMAS), types=schema_types(OC), fmt=fmt, schemas=SCHEMAS, hold=hold)
    rw = b.fmt.stream_rewriter() if b.fmt is not None else None
    deltas = []
    for i in range(0, len(text), size):
        p = text[i:i + size]
        b.feed(rw.feed(p) if rw is not None else p)
        deltas += b.drain_deltas()
    if rw is not None:
        b.feed(rw.finish(eos=True))
    _, tail = b.finish(eos=True)
    deltas += b.drain_deltas() + tail
    sent = {}
    for d in deltas:
        c = sent.setdefault(d["index"], "")
        sent[d["index"]] = c + ((d.get("function") or {}).get("arguments") or "")
    return b, {k: json.loads(v) for k, v in sent.items() if v}


def test_the_stream_renames_as_it_goes_in_every_form():
    for name in ("xml", "argkey", "pythonic"):
        fmt = toolformats.get(name)
        text = "Reading." + fmt.render("read", {"path": "/tmp/proj/src/config.py", "offset": 40})
        b, sent = _stream(text, fmt=None if name == "xml" else fmt)
        assert list(sent.values()) == [{"filePath": "/tmp/proj/src/config.py", "offset": 40}], (name, sent)
        assert len(b.calls) == 1 and json.loads(b.calls[0]["function"]["arguments"])["filePath"], name
        assert b.repairs == [_ev("read", "path", "filePath")], (name, b.repairs)
    # held (the conductor's checks): renamed at the closer, gated
    b, _ = _stream(toolformats.get("xml").render("read", {"path": "/a"}), hold=True)
    held = b.take_held()
    assert json.loads(held[0]["function"]["arguments"]) == {"filePath": "/a"} and b.repairs


def test_a_tool_the_request_did_not_offer():
    """SRV-693: an offered equivalent when exactly one and its required parameters are named; else a
    search or a listing as quoted bash; else as written."""
    c = {"function": {"name": "read_file", "arguments": json.dumps({"path": "/a.txt"})}}
    c2, ev = CR.repair_call(c, SCHEMAS)
    assert c2["function"]["name"] == "read" and json.loads(c2["function"]["arguments"]) == {"filePath": "/a.txt"}
    assert ev[0] == _ev("read_file", "read_file", "read", kind="tool") and ev[1] == _ev("read", "path", "filePath")
    # no list tool in opencode's set: a listing as bash, quoted
    c2, ev = CR.repair_call({"function": {"name": "ls", "arguments": json.dumps({"path": "/tmp/my dir"})}}, SCHEMAS)
    assert c2["function"]["name"] == "bash" and json.loads(c2["function"]["arguments"]) == \
        {"command": "ls -la -- '/tmp/my dir'"}, c2
    only_bash = {"bash": SCHEMAS["bash"]}
    c2, ev = CR.repair_call({"function": {"name": "grep", "arguments": json.dumps(
        {"pattern": "it's", "path": "src", "include": "*.py"})}}, only_bash)
    assert json.loads(c2["function"]["arguments"])["command"] == "grep -rn --include='*.py' -- 'it'\"'\"'s' src", c2
    # nothing to map to: as written
    for name, args, schemas in (("grep", {"pattern": "x"}, {"read": READ}),              # no bash
                                ("grep", {"path": "src"}, only_bash),                    # no pattern
                                ("frobnicate", {"x": 1}, SCHEMAS),                        # not a known kind
                                ("write_file", {"path": "/a"}, SCHEMAS)):                 # required content missing
        call = {"function": {"name": name, "arguments": json.dumps(args)}}
        assert CR.repair_call(call, schemas) == (call, []), name
    # a stream holds such a call and maps it whole at the closer
    b, sent = _stream(toolformats.get("xml").render("read_file", {"path": "/a.txt"}))
    assert [c["function"]["name"] for c in b.calls] == ["read"] and list(sent.values()) == [{"filePath": "/a.txt"}]


def test_c4_writes_an_unmapped_tool_again():
    """SRV-693 under the conductor: a call to a tool the request did not offer and the server could not map
    is written again with the offered list (C4)."""
    from test_toolcheck import ask, rows, serve
    mon = serve("on")
    call = "FAKE_CALL_JSON " + json.dumps({"name": "frobnicate", "arguments": {"x": 1}})
    ask(call, stream=False, tools=OC)
    act = rows(mon, task="C4", decision="REGENERATE_CALL")
    assert act and "frobnicate" in act[0]["text"] and "read" in act[0]["text"], mon.journal.events()


def test_the_server_on_both_paths():
    from test_toolcheck import ask, rows, serve
    call = "FAKE_CALL_JSON " + json.dumps({"name": "read", "arguments": {"path": "/tmp/proj/src/config.py"}})
    for mode in ("raw", "on"):
        for stream in (True, False):
            mon = serve(mode)
            got = ask(call, stream=stream, tools=OC)
            assert got["calls"] and json.loads(got["calls"][0]["args"]) == {"filePath": "/tmp/proj/src/config.py"}, \
                (mode, stream, got)
            if mon is not None:
                assert rows(mon, task="ARGS", decision="RENAME"), (stream, mon.journal.events())
    return "raw and conductor, streamed and not"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
