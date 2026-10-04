"""Memory notes through the real handler (2026-10-05, P-M1): the fake engine in process, a
conductor, opencode-shaped requests. Needs CPU torch (the box's CPU suite).

The test matrix:

  1  memory off (the seeds; every profile stored before it): a block the model writes goes out
     as text, nothing is written, the prompt carries no memory note
                                                        test_off_leaves_the_block_alone
  2  on, streamed: the block is not in the client's content, the note is on disk (provisional,
     qwen-inline), the state line says "remembered: short answers", the journal row's extra
     names the write without its text                  test_on_streamed_write
  3  on, a block before a tool call: the block is held back, the call goes out whole
                                                        test_on_block_before_a_call
  4  a refused block and an unclosed block go out as text; nothing is written
                                                        test_refused_and_unclosed_go_out
  5  non-streamed: the content without the block, the note on disk
                                                        test_non_streamed_write
  6  the next turns: the card and the index go in once at the tail and replay at their place on
     the tool turn after it (the prompt starts with the one before it up to the notes' end);
     the bodies go in on a new user message, never on a tool turn
                                                        test_the_reads_and_their_replay
  7  the hidden span of a taken block: its tokens found in the committed context
                                                        test_hidden_span
  8  shadow: the card and the write, no index and no bodies
                                                        test_shadow

Run: python tests/test_memory_router.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_ui_states import OPENCODE_UA, TOOLS, _go, lines, serve  # noqa: E402

from server import app  # noqa: E402

SYSTEM = "You are opencode.\nWorking directory: /home/user/pagoda\nPlatform: linux"
BLOCK_OPEN = "<tandem_memory"


def setup(memory="on", states="on"):
    cond, seen = serve(states)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        if memory is not None:
            doc["memory"] = memory
        cond.store.put("profiles", name, doc, note="memory", author="test")
    return cond, seen


def ask(text, history=(), stream=True, sid="ses_mem1", **extra):
    body = {"model": "TandemLLM", "stream": stream, "max_tokens": 2000, "tools": TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": SYSTEM}, *history,
                         {"role": "user", "content": text}], **extra}
    return _go(body, {"User-Agent": OPENCODE_UA, "x-session-id": sid})


def test_off_leaves_the_block_alone():
    cond, seen = setup(memory=None)
    out = ask("Please keep it short. FAKE_MEMORY")
    assert BLOCK_OPEN in out["content"], out["content"]
    assert cond.notes.notes("owner") == []
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "Read one in full" not in text and "<tandem_memory op=" not in text


def test_on_streamed_write():
    cond, seen = setup()
    out = ask("Please keep it short. FAKE_MEMORY")
    assert BLOCK_OPEN not in out["content"] and out["content"].strip(), out["content"]
    ns = cond.notes.notes("owner")
    assert len(ns) == 1 and ns[0].title == "short answers" and ns[0].status == "provisional"
    assert ns[0].writer == "qwen-inline" and ns[0].scope == "user"
    assert "remembered: short answers" in lines(out["reasoning"]) + lines(out["content"]), out
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "<tandem_memory op=" in text                      # the card went in


def test_on_block_before_a_call():
    cond, _ = setup()
    call = json.dumps({"name": "bash", "arguments": {"command": "ls -la"}})
    out = ask("Keep it short. FAKE_MEMORY_PROJECT FAKE_CALL_JSON " + call)
    assert BLOCK_OPEN not in out["content"], out["content"]
    args = "".join((c.get("function") or {}).get("arguments") or "" for c in out["calls"])
    assert json.loads(args) == {"command": "ls -la"}, out["calls"]
    ns = cond.notes.notes("owner")
    assert len(ns) == 1 and ns[0].scope == "project" and ns[0].project == "/home/user/pagoda"


def test_refused_and_unclosed_go_out():
    cond, _ = setup()
    out = ask("Keep it short. FAKE_MEMORY_BAD")
    assert '<tandem_memory op="add" scope="user" kind="poem">' in out["content"]
    out = ask("Keep it short again. FAKE_MEMORY_OPEN")
    assert BLOCK_OPEN in out["content"]
    assert cond.notes.notes("owner") == []


def test_non_streamed_write():
    cond, _ = setup()
    out = ask("Please keep it short. FAKE_MEMORY", stream=False)
    assert BLOCK_OPEN not in out["content"] and out["content"].strip()
    assert [n.title for n in cond.notes.notes("owner")] == ["short answers"]


def _tail_notes(text):
    return text.count("<runtime-note>")


def test_the_reads_and_their_replay():
    cond, seen = setup()
    cond.notes.write("owner", {"op": "add", "scope": "project", "kind": "gotcha",
                               "title": "the summer scene checker gate",
                               "body": "Gate (a) wants the summer scene within 1.0 of night 1."},
                     writer="human", project="/home/user/pagoda")
    q1 = "How is the summer scene checked by the gate?"
    a1 = ask(q1, sid="ses_mem2")
    p1 = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "Read one in full" in p1 and "within 1.0 of night 1" in p1, p1[-1500:]
    hist = [{"role": "user", "content": q1},
            {"role": "assistant", "content": a1["content"], "reasoning_content": a1["reasoning"],
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.js b.js"}]
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000, "tools": TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": SYSTEM}, *hist]}
    _go(body, {"User-Agent": OPENCODE_UA, "x-session-id": "ses_mem2"})
    p2 = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    # replayed at their place: the second prompt starts with the first up to the notes' end
    end1 = p1.rindex("</runtime-note>") + len("</runtime-note>")
    assert p2.startswith(p1[:end1]), (p1[end1 - 300:end1], p2[end1 - 300:end1 + 100])
    assert p2.count("Read one in full") == 1 and p2.count("within 1.0 of night 1") == 1


def test_hidden_span():
    cond, seen = setup()
    ask("Please keep it short. FAKE_MEMORY", sid="ses_mem3")
    tok = app.STATE["tok"]
    ctx = app.STATE.get("last_ctx") or []
    from server import memblock
    n_prompt = len(seen[-1])
    gen = list(ctx[n_prompt:])
    _, found = memblock.strip_blocks(tok.decode(gen, skip_special_tokens=True))
    assert ctx and found, "the fake engine kept the committed context and the block"
    spans = app._memory_spans(found, gen, n_prompt)
    assert spans and tok.decode(spans[0][1], skip_special_tokens=True).strip() == \
        found[0].strip(), spans


def test_shadow():
    cond, seen = setup(memory="shadow")
    cond.notes.write("owner", {"op": "add", "scope": "user", "kind": "reference",
                               "title": "the summer scene checker gate",
                               "body": "Gate (a) wants the summer scene within 1.0 of night 1."},
                     writer="human")
    out = ask("How is the summer scene checked? FAKE_MEMORY", sid="ses_mem4")
    assert BLOCK_OPEN not in out["content"]
    p = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "<tandem_memory op=" in p and "Read one in full" not in p and "within 1.0" not in p
    assert len(cond.notes.notes("owner")) == 2


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
