"""0.3.26-test: the tool-argument loop guard (server/argloop.py).

The 2026-10-04 16:18 shape: a `write` whose content falls into one 97-character tile-map row
repeated until the token limit (65,536 tokens, 327 s, nothing written). Here:

  1  the guard's rules on text: a periodic tail fires within `span_tokens` of the loop's start;
     legit shapes do not (a tile map with a few blank rows, a data table, minified code, a long
     varied file); the size rule needs size, the context multiple and the compression together
  2  through the handler with the fake engine (one token a character): the looping call is
     stopped within a few thousand tokens, the client gets no call and no block text, one
     sentence, finish stop; the journal has A1's GIVE_UP
  3  the session's next request carries the hidden note at the prompt's tail, once
  4  a long legit write streams through whole
  5  without held calls (no acting monitor) the guard only logs: nothing of the stream changes
  6  `--argloop off` and `log`

Run: python tests/test_argloop.py
"""

from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app, argloop  # noqa: E402
from test_toolcheck import SYSTEM, ask, ctx_text, policy, rows, serve  # noqa: E402

# opencode's system prompt is longer than the monitor's lineage window: one session
LONG = SYSTEM + "\n" + "Follow the project's rules. " * 200

ROW = '"' + " " * 94 + '",\n'


def _stream(text: str, cfg=None, context_chars: int = 0, per_token: int = 1):
    """Feed `text` as tokens of `per_token` characters; the verdict and the tokens fed."""
    g = argloop.ArgLoopGuard(cfg, context_chars)
    n = 0
    for i in range(per_token, len(text) + per_token, per_token):
        n += 1
        v = g.feed(n, min(i, len(text)), lambda i=i: text[:i])
        if v is not None:
            return v, n
    return None, n


def _code(n_lines: int, seed: int = 1) -> str:
    r = random.Random(seed)
    names = ["player", "enemy", "coin", "tile", "camera", "score", "level", "vx", "vy", "dt"]
    out = []
    for i in range(n_lines):
        a, b = r.choice(names), r.choice(names)
        out.append(f"  if ({a}.x > {b}.y + {r.randint(0, 999)}) {{ {a}.v = {r.random():.3f}; }}"
                   f" // {i}")
    return "\n".join(out)


# ------------------------------------------------------------------ 1: the rules on text
def test_1_the_incident_shape_fires_soon_after_the_loop_starts():
    head = _code(60)
    text = head + "\nconst LEVEL = [\n" + ROW * 3000
    # the served tokenizer reads the row as ~5 tokens: 20 characters a token
    v, n = _stream(text, per_token=20)
    assert v is not None and v.rule == "periodic" and v.period == len(ROW), v
    onset = len(head) // 20
    assert n - onset <= argloop.DEFAULTS["span_tokens"] + 2 * argloop.DEFAULTS["every"] + 64, \
        (n, onset)


def test_1_a_short_unit_loop_fires_too():
    v, _ = _stream(_code(30) + "\n" + "0, " * 20000, per_token=2)
    assert v is not None and v.rule == "periodic" and v.period in (3, 6, 9), v


def test_1_legit_shapes_do_not_fire():
    blank = "  " + '"' + " " * 60 + '",\n'
    tilemap = "const MAP = [\n" + blank * 12 + ('"' + "#" * 60 + '",\n') * 2 + "];\n"
    table = "".join(f"  {{id: {i}, name: 'item{i}', price: {i * 3 % 17}.99}},\n"
                    for i in range(3000))
    minified = "".join(f"function f{i}(a,b){{return a*{i}+b-{i % 7}}};" for i in range(4000))
    levels = "".join(f"// level {k}\n" + tilemap.replace("#" * 60, "#" * k + "?" * (60 - k))
                     for k in range(1, 30))
    for text in (_code(5000), levels + _code(200), table, minified,
                 _code(300) + tilemap + _code(300)):
        for per in (1, 4):
            v, _ = _stream(text, per_token=per)
            assert v is None, (v, text[:80])


def test_1_the_size_rule_needs_size_context_and_repetition():
    big = _code(30000)                  # ~2 MB, varied: compresses ~4x
    cfg = {"size_tokens": 1000, "min_tokens": 64, "every": 256}
    assert _stream(big, cfg, per_token=4)[0] is None
    # near-repetition no period catches: a counter in every row
    rows_ = "".join(f'"{" " * 90}{i:06d}",\n' for i in range(60000))
    v, _ = _stream(rows_, cfg, per_token=8)
    assert v is not None and v.rule == "size" and v.ratio >= 16, v
    # the same text, with a message in context bigger than it can grow to: no fire
    v, _ = _stream(rows_, cfg, context_chars=10 ** 8, per_token=8)
    assert v is None


def test_1_periodic_tail_and_the_messages():
    p, span = argloop.periodic_tail("x" * 50 + "abc" * 400, 4096)
    assert p == 3 and span == 1200
    assert argloop.periodic_tail(_code(50), 4096) == (0, 0)
    msgs = [{"role": "system", "content": "s" * 10},
            {"role": "tool", "content": [{"type": "text", "text": "t" * 500}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "write", "arguments": "a" * 900}}]}]
    assert argloop.longest_message_chars(msgs) == 900
    note = argloop.note_text({"rule": "periodic", "period": 98, "repeats": 211, "tokens": 2176})
    assert "98-character" in note and "211 times" in note and "{why}" not in note


# ------------------------------------------------------------------ 2, 3: through the handler
def test_2_the_looping_write_is_stopped_with_one_sentence():
    mon = serve()
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=60000)
    assert got["calls"] == {} and "<tool_call>" not in got["content"], got["content"][:200]
    assert "LEVEL" not in got["content"]
    assert got["content"].strip().startswith("[TandemLLM stopped the `write` call"), got["content"]
    assert got["finish"] == "stop", got["finish"]
    n = len(app.STATE["last_ctx"]) - len(app.STATE["tok"].encode_text(""))
    g = rows(mon, task="A1", decision="GIVE_UP")
    assert g and g[0]["detail"]["rule"] == "periodic" and g[0]["detail"]["name"] == "write", g
    # the fake writes one token a character: the loop starts ~750 tokens into the block
    assert g[0]["detail"]["tokens"] < 3000, g[0]["detail"]
    assert n < 10000, n


def test_3_the_next_request_carries_the_hidden_note_once():
    mon = serve()
    ask("FAKE_ARG_LOOP write the game", max_tokens=60000, system=LONG)
    ask("FAKE_ARG_LOOP write the game", max_tokens=200, system=LONG,
        history=[{"role": "assistant", "content": "[TandemLLM stopped the `write` call]"},
                 {"role": "user", "content": "go on"}])
    notes = rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")
    assert len(notes) == 1, mon.journal.events()
    assert "Runtime note: your last tool call was stopped" in ctx_text()
    ask("go on, read the tool docs", max_tokens=200, system=LONG,
        history=[{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}])
    assert len(rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")) == 1
    assert "Runtime note: your last tool call was stopped" not in ctx_text()


def test_4_a_long_legit_write_goes_through_whole():
    mon = serve()
    content = _code(400)
    import json as _j
    got = ask("FAKE_CALL_JSON " + _j.dumps({"name": "write", "arguments": {
        "filePath": "/tmp/a.js", "content": content}}), max_tokens=60000)
    assert list(got["calls"]) == [0], got["content"][:200]
    assert _j.loads(got["calls"][0]["args"])["content"].strip() == content.strip()
    assert not rows(mon, task="A1")


def test_4_a_call_streamed_live_is_stopped_too():
    """The live coding profile's tool gate is off (no SP3 task acts): calls stream as they are
    written. The client keeps the call it was sent part of (never closed, as the token limit
    left it on 2026-10-04), gets no block text as content, one sentence, finish stop; the
    session's next request has the note."""
    mon = serve(pol=policy(C4="off", C3="off", C1="off", C2="off"))
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=60000, system=LONG)
    assert list(got["calls"]) == [0] and got["calls"][0]["name"] == "write", got["calls"].keys()
    args = got["calls"][0]["args"]
    assert args.startswith("{") and 1000 < len(args) < 6000, len(args)
    assert "<tool_call>" not in got["content"] and "LEVEL" not in got["content"]
    assert "[TandemLLM stopped the `write` call" in got["content"], got["content"][-300:]
    assert got["finish"] == "stop", got["finish"]
    g = rows(mon, task="A1", decision="GIVE_UP")
    assert g and g[0]["detail"]["streamed"] is True, g
    ask("FAKE_ARG_LOOP write the game", max_tokens=200, system=LONG,
        history=[{"role": "assistant", "content": "stopped"}, {"role": "user", "content": "go"}])
    assert len(rows(mon, task="A1", decision="PROMPT_TAIL_NOTE")) == 1


def test_5_without_a_monitor_the_guard_only_logs():
    serve(mode="raw")
    got = ask("FAKE_ARG_LOOP write the game", max_tokens=6000)
    assert got["finish"] == "length", got["finish"]
    assert "stopped the" not in got["content"]


def test_6_off_and_log():
    for flag in ("off", "log"):
        mon = serve()
        app.STATE["argloop"] = flag
        got = ask("FAKE_ARG_LOOP write the game", max_tokens=6000)
        assert "stopped the" not in got["content"] and not rows(mon, task="A1"), flag


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
