"""A memory note written as a `tandem_memory` tool call (2026-11-03): the call is the block.

With the tools of a chat in view the model wrote its note as Qwen's tool call; no client offers
that tool, so chatd answered that it is not available, the model wrote the blocks again with no
answer around them, and the client got an empty turn. Here, string handling only (standard
library, no torch):

  1  the shared parser (server/memblock.py `call_block`, `tool_blocks`): the XML form and the JSON
     form as the block they mean; a parameter that is not a field (a `why`) kept in the body;
     the background writer reads its calls with the same function
                                                        test_call_block_*, test_writer_shares_it
  2  the stream filter with `calls`: the call is held and hidden at every split of the stream,
     written as its block; another tool's call passes as it came, held only until its function
     is named; a block and a call in one answer; the call cut before `</tool_call>` at the end
     token; the caps (too many, too long) as for a block; `calls` off: the call passes
                                                        test_filter_*
  3  `without_notes`: the answer as the client gets it, the notes counted (SP2 reads it so)
                                                        test_without_notes
  4  E1 case M (server/conductor/monitor/tier0.py): an answer of notes alone continues after a
     hidden result, a tool response after a call; an answer with text, or a note in the
     reasoning, does not                                test_empty_turn_case_m

Run: python tests/test_memory_calls.py
"""

from __future__ import annotations

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import memblock  # noqa: E402

CALL = ("<tool_call>\n<function=tandem_memory>\n<parameter=op>\nadd\n</parameter>\n"
        "<parameter=scope>\nproject\n</parameter>\n<parameter=kind>\ngotcha\n</parameter>\n"
        "<parameter=title>\nthe pagoda checker wants the summer scene within 1.0\n</parameter>\n"
        "<parameter=why>\nfound twice in one session\n</parameter>\n<parameter=body>\nGate (a) "
        "compares the summer scene with night 1; offerings.js recolours it by season.\n"
        "</parameter>\n</function>\n</tool_call>")
BASH = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
JSON_CALL = ('<tool_call>\n{"name": "tandem_memory", "arguments": {"op": "add", "scope": "user", '
             '"kind": "rule", "title": "short answers", "tags": ["style", "answers"], '
             '"body": "The user wants answers of at most five sentences."}}\n</tool_call>')
BLOCK = ('<tandem_memory op="add" scope="user" kind="rule">\ntitle: metric units\n\nThe user '
         'measures in metres and kilograms.\n</tandem_memory>')


def feed_all(f, text, step):
    out = [f.feed(text[i:i + step]) for i in range(0, len(text), step)]
    return "".join(out) + f.finish()


def test_call_block_xml():
    b = memblock.call_block(CALL)
    op = memblock.parse(b)
    assert (op["op"], op["scope"], op["kind"]) == ("add", "project", "gotcha"), op
    assert op["title"] == "the pagoda checker wants the summer scene within 1.0"
    assert op["body"].startswith("Gate (a) compares the summer scene")
    assert op["body"].endswith("why: found twice in one session"), op["body"]
    # without the `<tool_call>` opener (a tokenizer that drops it) the function still reads
    assert memblock.call_block(CALL[len("<tool_call>\n"):]) == b
    assert memblock.call_block(BASH) is None and memblock.call_block("text") is None


def test_call_block_json():
    op = memblock.parse(memblock.call_block(JSON_CALL))
    assert (op["op"], op["scope"], op["kind"], op["title"]) == ("add", "user", "rule",
                                                                  "short answers"), op
    assert op["tags"] == "style, answers" and op["body"].startswith("The user wants")
    s = json.dumps({"name": "tandem_memory", "arguments": json.dumps({"op": "archive", "id": "n1",
                                                                      "reason": "done"})})
    op = memblock.parse(memblock.call_block(f"<tool_call>{s}</tool_call>"))
    assert (op["op"], op["id"], op["reason"]) == ("archive", "n1", "done"), op
    other = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    assert memblock.call_block(other) is None


def test_writer_shares_it():
    from server.conductor import memwriter
    text = "x " + CALL + "\n" + BASH + "\n" + JSON_CALL
    assert memwriter.tool_blocks(text) == memblock.tool_blocks(text)
    got = memblock.tool_blocks(text)
    assert [memblock.parse(b)["title"] for b in got] == [
        "the pagoda checker wants the summer scene within 1.0", "short answers"], got
    # a call left open before another: both functions read, as the writer read them before
    two = CALL[:-len("</tool_call>")] + CALL
    assert len(memblock.tool_blocks(two)) == 2


def test_filter_hides_the_call_at_every_split():
    text = "The scene passes now.\n\n" + CALL + "\n"
    for step in (1, 2, 3, 5, 7, 11, 64, 1000):
        got = []
        f = memblock.MemFilter(lambda b: got.append(b), calls=True)
        assert feed_all(f, text, step) == "The scene passes now.\n\n\n", step
        assert got == [memblock.call_block(CALL)], step
        assert f.hidden == [CALL] and f.taken == [CALL] and f.called == 1 and not f.failed, step


def test_filter_passes_other_calls():
    text = "Let me look.\n" + BASH + "\nand " + JSON_CALL.replace("tandem_memory", "web_search")
    for step in (1, 4, 9, 1000):
        f = memblock.MemFilter(lambda b: None, calls=True)
        assert feed_all(f, text, step) == text, step
        assert f.hidden == [] and f.called == 0
    # held only until the function is named: `<tool_call>\n<function=bash>` goes on at once
    f = memblock.MemFilter(lambda b: None, calls=True)
    assert f.feed("ok <tool_call>\n<function=ba") == "ok "
    assert f.feed("sh>\n<parameter") == "<tool_call>\n<function=bash>\n<parameter"
    # a `<tool_call>` that names no function within its first characters is not ours
    f = memblock.MemFilter(lambda b: None, calls=True)
    s = "<tool_call>" + "x" * 80
    assert feed_all(f, s, 3) == s and f.hidden == []


def test_filter_block_and_call_in_one_answer():
    text = "Done.\n" + BLOCK + "\n" + CALL + "\n" + BASH
    got = []
    f = memblock.MemFilter(lambda b: got.append(b), calls=True)
    assert feed_all(f, text, 6) == "Done.\n\n\n" + BASH
    assert got == [BLOCK, memblock.call_block(CALL)] and f.called == 1 and f.blocks == 2


def test_filter_call_cut_at_the_end_token():
    cut = "Noted. " + CALL[:-len("\n</tool_call>")]
    f = memblock.MemFilter(lambda b: None, calls=True)
    assert feed_all(f, cut, 8) == "Noted. " and f.called == 1 and f.taken, f.failed
    # cut inside its parameters: it never closed, it goes out as the text it was
    half = CALL[:120]
    f = memblock.MemFilter(lambda b: None, calls=True)
    assert feed_all(f, "x " + half, 5) == "x " + half and f.failed == ["unclosed"]


def test_filter_caps_and_refusals():
    f = memblock.MemFilter(lambda b: "cap", calls=True)
    assert feed_all(f, "a " + CALL, 13) == "a " and f.failed == ["cap"] and f.hidden == [CALL]
    f = memblock.MemFilter(lambda b: None, calls=True)
    assert feed_all(f, (CALL + "\n") * 5, 17) == "\n" * 5 and f.failed == ["too_many"]
    long = CALL.replace("by season.", "by season. " + "y" * 2100)
    f = memblock.MemFilter(lambda b: None, calls=True)
    assert feed_all(f, long, 100) == "" and f.failed == ["too_long"] and f.hidden == [long]
    # a closed call whose arguments are no note is ours all the same: hidden, refused (schema)
    f = memblock.MemFilter(lambda b: None, calls=True)
    assert feed_all(f, "<tool_call>{\"name\": \"tandem_memory\", \"arguments\": 3}</tool_call>",
                    5) == "" and f.failed == ["schema"]


def test_filter_calls_off():
    text = "Done.\n" + CALL + "\n" + BLOCK
    f = memblock.MemFilter(lambda b: None)
    assert feed_all(f, text, 4) == "Done.\n" + CALL + "\n" and f.called == 0


def test_without_notes():
    assert memblock.without_notes("Done. " + CALL) == ("Done. ", 0, 1)
    assert memblock.without_notes(BLOCK + "\n" + CALL) == ("\n", 1, 1)
    assert memblock.without_notes(CALL, calls=False) == (CALL, 0, 0)
    assert memblock.without_notes("plain") == ("plain", 0, 0)


def test_empty_turn_case_m():
    from server.conductor.monitor import tier0
    from server.conductor.monitor.observe import FinishEvent
    task = {"close": "x</think>", "nudges": ["go on"], "tries": 2}
    sup = types.SimpleNamespace(req=types.SimpleNamespace(tools_offered=True, n=1),
                                task_acts={}, tool_names=["bash"], tool_types={})

    class Task(dict):
        id = "E1"
    act = tier0.empty_turn(FinishEvent("eos", False, "\n", 0, 40, notes=1, note_calls=1),
                           Task(task), sup)
    assert act.kind == "CONTINUE_HIDDEN" and act.detail == {"case": "M", "act": 1, "notes": 1,
                                                            "call": True}, act.detail
    assert act.text == tier0.NOTED
    act = tier0.empty_turn(FinishEvent("eos", False, "", 0, 40, notes=1),
                           Task(task, noted="Noted."), sup)
    assert act.detail["call"] is False and act.text == "Noted."
    # text around the note: not empty, nothing to do
    assert tier0.empty_turn(FinishEvent("eos", False, "Done.", 0, 40, notes=1), Task(task),
                            sup) is None
    # no note: case B as before
    act = tier0.empty_turn(FinishEvent("eos", False, "", 0, 40), Task(task), sup)
    assert act.detail["case"] == "B"


def test_case_m_resolves_to_a_tool_response():
    from server.conductor.monitor import executor, policy as policy_mod
    pol = policy_mod.load_default()
    seen = []
    sup = types.SimpleNamespace(policy=pol, lift_ids=None,
                                monitor=types.SimpleNamespace(ids=lambda s: seen.append(s) or [1]))
    act = policy_mod.Action(policy_mod.CONTINUE_HIDDEN, text="Saved.", detail={"case": "M",
                                                                               "call": True})
    assert executor.Supervisor._resolve(sup, act) and act.visible == []
    assert seen[-1].startswith("<|im_end|>\n<|im_start|>user\n<tool_response>\nSaved.\n"), seen
    act = policy_mod.Action(policy_mod.CONTINUE_HIDDEN, text="Saved.", detail={"case": "M",
                                                                               "call": False})
    executor.Supervisor._resolve(sup, act)
    assert seen[-1].startswith("<|im_end|>\n<|im_start|>user\nSaved.<|im_end|>"), seen


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"ok {name}")
    print(f"{passed} passed")
