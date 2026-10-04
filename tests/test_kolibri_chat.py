"""Kolibri-1 serving on the CPU: template fixes, the reasoning block the model opens itself, Hermes
JSON tool calls, and the decode loop with its ring prefix cache over a tiny random Kolibri.

The template tests need the release's tokenizer files (`tokenizer.json`, `tokenizer_config.json`):
`KOLIBRI_TOKENIZER=dir` (the downloaded set holds them); they are skipped without it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri import chat  # noqa: E402
from server.stream import MODEL_OPENS, Reasoning, split_full  # noqa: E402
from server.toolcall import ToolCallBuffer, parse_tool_calls  # noqa: E402

TOKDIR = os.environ.get("KOLIBRI_TOKENIZER") or ""
HAVE_TOK = os.path.isfile(os.path.join(TOKDIR, "tokenizer_config.json"))
needs_tok = pytest.mark.skipif(not HAVE_TOK, reason="Kolibri tokenizer files not here")


@pytest.fixture(scope="module")
def tok():
    return chat.load_tokenizer(TOKDIR)


@needs_tok
def test_thinking_default_is_model_opened(tok):
    text = tok.render([{"role": "user", "content": "Hallo"}])
    assert text.endswith("<|im_start|>assistant\n")
    assert "Reasoning effort is set to high" in text
    assert chat.thinking_mode(text) == MODEL_OPENS


@needs_tok
def test_enable_thinking_false_beats_a_default_effort(tok):
    text = tok.render([{"role": "user", "content": "Hallo"}], enable_thinking=False,
                      reasoning_effort="medium")
    assert text.endswith("<think>\n\n</think>\n\n")
    assert "Reasoning is disabled" in text
    assert chat.thinking_mode(text) is False


@needs_tok
def test_effort_words(tok):
    for word, sentence in (("low", "set to low"), ("minimal", "set to low"),
                           ("medium", "set to medium"), ("xhigh", "set to high"),
                           ("none", "Reasoning is disabled")):
        assert sentence in tok.render([{"role": "user", "content": "x"}], reasoning_effort=word)


@needs_tok
def test_content_parts_and_developer_role(tok):
    text = tok.render([{"role": "developer", "content": [{"type": "text", "text": "Be brief."}]},
                       {"role": "user", "content": [{"type": "text", "text": "Hi "},
                                                    {"type": "image_url", "image_url": {"url": "x"}},
                                                    {"type": "text", "text": "there"}]}])
    assert text.startswith("<|im_start|>system\nBe brief.")
    assert "<|im_start|>user\nHi there<|im_end|>" in text
    tok.apply_chat_template([{"role": "user", "content": [{"type": "image_url"}]}],
                            add_generation_prompt=True, tokenize=False)
    assert tok.dropped_parts == 1


@needs_tok
def test_tools_and_tool_results(tok):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "d",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}}}}]
    msgs = [{"role": "user", "content": "Wetter in Berlin?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "12 Grad"}]
    text = tok.render(msgs, tools=tools)
    assert "<tools>" in text and '"name": "get_weather"' in text
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>' in text
    assert "<|im_start|>user\n<tool_response>\n12 Grad\n</tool_response><|im_end|>" in text
    ids = tok(text, add_special_tokens=False).input_ids
    assert chat.TOOL_CALL_ID in ids and chat.IM_END in ids


@needs_tok
def test_special_ids(tok):
    for s, i in (("<think>", chat.THINK_OPEN_ID), ("</think>", chat.THINK_CLOSE_ID),
                 ("<|im_end|>", chat.IM_END), ("<tool_call>", chat.TOOL_CALL_ID)):
        assert tok.convert_tokens_to_ids(s) == i
    # <think> is not a special token: a decode that skips specials keeps it
    assert tok.decode([chat.THINK_OPEN_ID, chat.IM_END], skip_special_tokens=True) == "<think>"


def _run(fmt, pieces):
    r = Reasoning(fmt, in_think=MODEL_OPENS)
    out = []
    for p in pieces:
        out += r.push(p)
    out += r.finish()
    return out


def _field(out, f):
    return "".join(t for k, t in out if k == f)


PIECES = ["<thi", "nk>", "\nLet me ", "think.</th", "ink>\n\nThe answer is 4."]


def test_reasoning_model_opens_tags():
    out = _run("tags", PIECES)
    assert "".join(t for _, t in out) == "<think>\nLet me think.</think>\n\nThe answer is 4."
    assert _field(out, "content") == "\n\nThe answer is 4."


def test_reasoning_model_opens_reasoning_content():
    out = _run("reasoning_content", PIECES)
    assert _field(out, "reasoning") == "Let me think."
    assert _field(out, "content") == "The answer is 4."


def test_reasoning_model_opens_both():
    out = _run("both", PIECES)
    assert _field(out, "reasoning") == "Let me think."
    assert _field(out, "tagged") + _field(out, "content") == \
        "<think>\nLet me think.</think>\n\nThe answer is 4."


def test_reasoning_model_writes_no_block():
    for fmt in ("tags", "reasoning_content", "both"):
        out = _run(fmt, ["\n", "<", "b>Hello</b>"])
        assert _field(out, "content") == "\n<b>Hello</b>"
        assert _field(out, "reasoning") == ""


def test_split_full_model_opens():
    text = "<think>\nplan\n</think>\n\nanswer"
    assert split_full(text, "reasoning_content", in_think=MODEL_OPENS) == ("answer", "plan\n")
    assert split_full(text, "tags", in_think=MODEL_OPENS) == (text, None)
    assert split_full("plain", "reasoning_content", in_think=MODEL_OPENS) == ("plain", None)


def test_reasoning_head_model_opens():
    from server.app import _reasoning_head
    assert _reasoning_head("<think>\na</think>\n\nb", "model") == ("<think>\na</think>", "\n\nb")
    assert _reasoning_head("no block", "model") == ("", "no block")


CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>'


def test_hermes_tool_call_parse():
    text, calls = parse_tool_calls("Let me check.\n" + CALL, names={"get_weather"})
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"city": "Berlin"}'
    assert "tool_call" not in text


def test_hermes_tool_call_stream():
    buf = ToolCallBuffer(names={"get_weather"})
    shown = []
    for i in range(0, len(CALL), 7):
        shown += buf.feed(CALL[i:i + 7])
    rest, calls = buf.finish(eos=True)
    deltas = buf.drain_deltas()
    got = calls or [d for d in deltas]
    assert "tool_call" not in "".join(shown) + rest
    assert got, (shown, rest, calls, deltas)


# --------------------------------------------------------------------- the serving loop, tiny model
class _App:
    """The parts of server/app.py the loop uses."""

    def __init__(self):
        from server import app as real
        self.STATE = {}
        self.BlockStats = real.BlockStats


def _engine(d, max_len=1024, ring=16):
    """The engine on the tiny model, with a 16-row ring (window 5: 12 rows back)."""
    from engine.kolibri.model import KolibriEngine
    from tools import kolibri_tiny as tiny
    st, rel = tiny.write(d)
    eng = KolibriEngine.load(st, rel, device="cpu", max_len=max_len, attention="kernel",
                             log=lambda s: None)
    eng.attn.ring_rows = ring
    eng.kv = eng.attn.make_kv(max_len, "cpu")
    return eng


def _gen(app, served, ids, n=6):
    from server.kolibri_serve import make_generate_stream
    app.STATE["engine"] = served
    g = make_generate_stream(app)
    return list(g(torch.tensor(ids), n, set()))


def test_serving_loop_and_prefix_cache():
    from server.kolibri_serve import RingPrefix, Served
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        pc = RingPrefix(chunk=8, anchor_bytes=1 << 30)
        served = Served(eng, pc)
        app = _App()
        p1 = [int(x) for x in torch.randint(0, 90, (30,))]
        out1 = _gen(app, served, p1)
        assert app.STATE["last_prefill"]["reused"] == 0
        assert sorted(pc.anchors) == [8, 16, 24, 30]
        # turn 2 keeps all of turn 1 and its answer: truncate path (within the ring's reach)
        p2 = p1 + out1[:-1] + [3, 4, 5]
        out2 = _gen(app, served, p2)
        assert app.STATE["last_prefill"]["kind"] == "truncate"
        assert app.STATE["last_prefill"]["reused"] == len(p1) + len(out1) - 1
        # the same prompt again: its end anchor and stored logits, nothing forwarded
        again = _gen(app, served, p2)
        assert again == out2
        assert app.STATE["last_prefill"]["forwarded"] == 0
        # the same prompt cold gives the same tokens
        eng.reset()
        pc2 = RingPrefix(chunk=8)
        cold = _gen(app, Served(eng, pc2), p2)
        assert cold == out2
        # a prompt that leaves turn 1 early: anchor path (the ring would not reach)
        eng.reset()
        pc3 = RingPrefix(chunk=8)
        s3 = Served(eng, pc3)
        long = [int(x) for x in torch.randint(0, 90, (700,))]
        _gen(app, s3, long, n=2)
        p4 = long[:500] + [1, 2, 3]
        out4 = _gen(app, s3, p4)
        assert app.STATE["last_prefill"]["kind"] == "anchor"
        assert app.STATE["last_prefill"]["reused"] == 496
        eng.reset()
        cold4 = _gen(app, Served(eng, RingPrefix(chunk=8)), p4)
        assert cold4 == out4


def test_guest_between_turns_keeps_the_conversation():
    from server.kolibri_serve import RingPrefix, Served
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        pc = RingPrefix(chunk=8, park_min=64)
        served = Served(eng, pc)
        app = _App()
        conv = [int(x) for x in torch.randint(0, 90, (300,))]
        out1 = _gen(app, served, conv, n=40)              # 339 rows: 300 prompt + 39 decoded
        guest = [int(x) for x in torch.randint(0, 90, (40,))]
        _gen(app, served, guest, n=20)                     # shares nothing: the conversation parks
        assert pc.stats["parked"] == 1 and len(pc.parked) == 1
        turn2 = conv + [7, 8, 9]                           # the answer dropped, as a template does
        out2 = _gen(app, served, turn2)
        assert pc.stats["unparked"] == 1
        assert app.STATE["last_prefill"]["kind"] == "anchor"
        assert app.STATE["last_prefill"]["reused"] == 300  # the prompt's end anchor
        assert pc.parked == []                             # the 59-row guest is below park_min
        eng.reset()
        cold = _gen(app, Served(eng, RingPrefix(chunk=8)), turn2)
        assert cold == out2


def test_new_chat_after_a_long_conversation_reuses_its_own_turns():
    """After a long conversation a later short request must not stay its guest forever: a new
    short chat reuses its own turns. Long conversation, then a new chat over three
    turns (reuse from turn 2), then the long conversation again (still cached), then the chat."""
    from server.kolibri_serve import RingPrefix, Served
    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        pc = RingPrefix(chunk=8, park_min=64)
        served = Served(eng, pc)
        app = _App()
        long = [int(x) for x in torch.randint(0, 90, (400,))]
        _gen(app, served, long, n=20)
        chat = [int(x) for x in torch.randint(0, 90, (50,))]
        out = _gen(app, served, chat, n=20)
        assert app.STATE["last_prefill"]["reused"] == 0
        for turn in (2, 3):
            chat = chat + out[:-1] + [10 + turn, 11 + turn]
            out = _gen(app, served, chat, n=20)
            lp = app.STATE["last_prefill"]
            assert lp["reused"] >= len(chat) - 25, (turn, lp)
            assert lp["kind"] in ("truncate", "anchor")
        assert [e.length for e in pc.parked] == [419]      # the long conversation, untouched
        long2 = long + [5, 6, 7]
        out_long = _gen(app, served, long2)
        assert app.STATE["last_prefill"]["reused"] == 400
        assert [e.length for e in pc.parked] == [len(chat) + 19]   # the chat parked in turn
        chat4 = chat + out[:-1] + [20, 21]
        out4 = _gen(app, served, chat4)
        assert app.STATE["last_prefill"]["reused"] >= len(chat4) - 25
        # bit for bit what a fresh engine answers
        for ids, got in ((long2, out_long), (chat4, out4)):
            eng.reset()
            assert _gen(app, Served(eng, RingPrefix(chunk=8)), ids) == got


def test_a_conversation_larger_than_the_budget_is_not_parked():
    from server.kolibri_serve import RingPrefix, Served
    torch.manual_seed(2)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        per = eng.kv.bytes_per_token
        pc = RingPrefix(chunk=8, park_min=64, stash_bytes=100 * per)   # 100 rows
        served = Served(eng, pc)
        app = _App()
        conv = [int(x) for x in torch.randint(0, 90, (300,))]
        _gen(app, served, conv, n=10)
        guest = [int(x) for x in torch.randint(0, 90, (60,))]
        _gen(app, served, guest, n=5)
        assert pc.stats["park_skipped"] == 1 and pc.parked == []
        turn2 = conv + [7, 8, 9]
        out2 = _gen(app, served, turn2)
        assert app.STATE["last_prefill"]["reused"] == 0
        eng.reset()
        cold = _gen(app, Served(eng, RingPrefix(chunk=8)), turn2)
        assert cold == out2


def test_parked_conversations_share_the_budget_oldest_out():
    from server.kolibri_serve import RingPrefix, Served
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        per = eng.kv.bytes_per_token
        ring = eng.kv.ring.nbytes
        pc = RingPrefix(chunk=8, park_min=64, park_anchors=2,
                        stash_bytes=500 * per + 6 * ring)
        served = Served(eng, pc)
        app = _App()
        convs = [[int(x) for x in torch.randint(0, 90, (200,))] for _ in range(4)]
        for c in convs:
            _gen(app, served, c, n=5)
        # three parked at 204 rows each do not fit 500 rows: the oldest went
        assert pc.stats["park_evicted"] >= 1
        assert sum(e.nbytes for e in pc.parked) <= pc.stash_bytes
        assert convs[0] not in [e.tokens[:200] for e in pc.parked]
        _gen(app, served, convs[2] + [1, 2])
        assert app.STATE["last_prefill"]["reused"] == 200


def test_session_survives_a_restart_bit_for_bit():
    """A planned stop writes the live conversation; a new engine reads it back with every tensor
    equal, and the next turn reuses it and answers what the engine that never stopped answers."""
    from server.kolibri_serve import (RingPrefix, Served, load_session, save_session,
                                      session_fingerprint)
    torch.manual_seed(5)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        pc = RingPrefix(chunk=8)
        app = _App()
        conv = [int(x) for x in torch.randint(0, 90, (300,))]
        out1 = _gen(app, Served(eng, pc), conv, n=40)
        fp = session_fingerprint(eng, d, d)
        path = os.path.join(d, "sess", "live.bin")
        w = save_session(pc, eng, path, fp)
        assert w["live"]["tokens"] == 339 and os.path.isfile(path)
        L = eng.kv.length
        before = [t.narrow(3, 0, L).clone() for t in (eng.kv.k, eng.kv.v)]
        ring = [t.clone() for t in eng.kv.ring.tensors()]
        turn2 = conv + out1[:-1] + [7, 8, 9]
        ref = _gen(app, Served(eng, pc), turn2)
        ref_prefill = dict(app.STATE["last_prefill"])
        # "restart": a second engine, empty
        eng2 = _engine(d)
        pc2 = RingPrefix(chunk=8)
        r = load_session(pc2, eng2, path, fp)
        assert r["exact"] and r["live"]["tokens"] == 339 and not os.path.exists(path)
        assert all(torch.equal(a, b.narrow(3, 0, L)) for a, b in zip(before, (eng2.kv.k, eng2.kv.v)))
        assert all(torch.equal(a, b) for a, b in zip(ring, eng2.kv.ring.tensors()))
        got = _gen(app, Served(eng2, pc2), turn2)
        assert got == ref
        assert app.STATE["last_prefill"]["reused"] == ref_prefill["reused"] > 300
        # the same prompt as the session's last: the end anchor and its logits came back too
        eng3 = _engine(d)
        pc3 = RingPrefix(chunk=8)
        save_session(pc2, eng2, path, fp)
        assert load_session(pc3, eng3, path, {**fp, "ring": 1})["skipped"] == ["fingerprint"]
        assert eng3.kv.length == 0


def test_parked_conversations_survive_a_restart_too():
    """Open WebUI's order: a chat turn, then its title request. The chat is parked when the server
    stops; after the restart turn 2 of the chat still reuses turn 1."""
    from server.kolibri_serve import (RingPrefix, Served, load_session, save_session,
                                      session_fingerprint)
    torch.manual_seed(6)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        pc = RingPrefix(chunk=8, park_min=64)
        app = _App()
        conv = [int(x) for x in torch.randint(0, 90, (300,))]
        _gen(app, Served(eng, pc), conv, n=40)
        title = [int(x) for x in torch.randint(0, 90, (30,))]
        _gen(app, Served(eng, pc), title, n=5)
        assert len(pc.parked) == 1
        fp = session_fingerprint(eng, d, d)
        path = os.path.join(d, "sess", "live.bin")
        w = save_session(pc, eng, path, fp)
        assert w["live"]["tokens"] == 34 and [r["tokens"] for r in w["parked"]] == [339]
        turn2 = conv + [7, 8, 9]
        ref = _gen(app, Served(eng, pc), turn2)
        eng2 = _engine(d)
        pc2 = RingPrefix(chunk=8, park_min=64)
        r = load_session(pc2, eng2, path, fp)
        assert r["exact"] and [p["tokens"] for p in r["parked"]] == [339]
        assert os.listdir(os.path.dirname(path)) == []
        got = _gen(app, Served(eng2, pc2), turn2)
        assert got == ref and app.STATE["last_prefill"]["reused"] == 300


def test_a_torn_session_file_is_dropped_not_served():
    from server.kolibri_serve import (RingPrefix, Served, load_session, save_session,
                                      session_fingerprint)
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        pc = RingPrefix(chunk=8)
        app = _App()
        conv = [int(x) for x in torch.randint(0, 90, (200,))]
        _gen(app, Served(eng, pc), conv, n=10)
        fp = session_fingerprint(eng, d, d)
        path = os.path.join(d, "sess", "live.bin")
        save_session(pc, eng, path, fp)
        with open(path, "r+b") as f:                   # one byte of the K rows, flipped
            n = int.from_bytes(f.read(8), "little")
            f.seek(8 + n + 100)
            x = f.read(1)
            f.seek(8 + n + 100)
            f.write(bytes([x[0] ^ 0xFF]))
        eng2 = _engine(d)
        pc2 = RingPrefix(chunk=8)
        r = load_session(pc2, eng2, path, fp)
        assert r["exact"] is False and eng2.kv.length == 0 and pc2.tokens == []
        _gen(app, Served(eng2, pc2), conv + [1, 2])
        assert app.STATE["last_prefill"]["reused"] == 0


def _stream(buf_pieces, **kw):
    buf = ToolCallBuffer(**kw)
    shown, deltas = [], []
    for p in buf_pieces:
        shown += buf.feed(p)
        deltas += buf.drain_deltas()
    rest, late = buf.finish(eos=True)
    return "".join(shown) + rest, deltas, late, buf


def test_hermes_arguments_stream_live():
    text = ('Ich schaue nach.\n<tool_call>\n{"name": "write_file", "arguments": {"path": "a.txt", '
            '"content": "Zeile 1\\nZeile 2 mit \\"Zitat\\" und {Klammern}"}}\n</tool_call>')
    for step in (1, 3, 7, 50):
        shown, deltas, late, buf = _stream([text[i:i + step] for i in range(0, len(text), step)],
                                           names={"write_file"})
        assert shown == "Ich schaue nach.\n"
        assert late == []                                  # nothing left for the sweep
        assert deltas[0]["function"]["name"] == "write_file"
        args = "".join(d["function"]["arguments"] for d in deltas)
        assert json.loads(args) == {"path": "a.txt",
                                    "content": 'Zeile 1\nZeile 2 mit "Zitat" und {Klammern}'}
        assert len(deltas) > 2 or step == 50               # the arguments came in pieces
        assert len(buf.calls) == 1 and buf.calls[0]["id"] == deltas[0]["id"]


def test_hermes_two_calls_and_other_shapes():
    two = ('<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>\n'
           '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>')
    shown, deltas, late, buf = _stream([two[i:i + 5] for i in range(0, len(two), 5)])
    names = [d["function"].get("name") for d in deltas if d.get("id")]
    assert names == ["a", "b"] and late == [] and shown.strip() == ""
    # arguments as a string, or keys the other way round: not streamed, sent whole at closure
    for odd in ('<tool_call>\n{"name": "a", "arguments": "{\\"x\\": 1}"}\n</tool_call>',
                '<tool_call>\n{"arguments": {"x": 1}, "name": "a"}\n</tool_call>'):
        shown, deltas, late, buf = _stream([odd[i:i + 4] for i in range(0, len(odd), 4)])
        assert deltas == [] and len(late) == 1 and json.loads(late[0]["function"]["arguments"]) == {"x": 1}


def test_hermes_typed_parameters_wait_for_closure():
    from server.toolcall import schema_types
    tools = [{"type": "function", "function": {"name": "n", "parameters": {
        "type": "object", "properties": {"k": {"type": "integer"}}}}}]
    t = '<tool_call>\n{"name": "n", "arguments": {"k": "5"}}\n</tool_call>'
    shown, deltas, late, buf = _stream([t[i:i + 3] for i in range(0, len(t), 3)],
                                       types=schema_types(tools))
    assert deltas == [] and len(late) == 1
