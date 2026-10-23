"""`cache_prompt: false` (#98, server/app.py `cache_prompt_off`, `_cold_release`), on the random 4-layer
model through the handler: the prompt is read whole, no prompt cache is read or written, and the
conversation in place is there for the next request that leaves the field out.

  1  only `false` turns the prompt cache off; `true`, a missing field or another value keep it
                                                        test_only_false_turns_it_off
  2  a long cold request after a conversation: `timings.cache_n` 0, every prompt token forwarded,
     no session snapshot taken; the conversation was parked, and its next turn without the flag
     resumes at its end anchor, as before the cold request
                                                        test_a_cold_request_reads_whole_and_the_conversation_comes_back
  3  a short cold request is a guest: the rows it overwrote come back, the end anchor stays
                                                        test_a_short_cold_request_is_a_guest
  4  the same cold request twice reads its prompt whole twice (no cache of its own)
                                                        test_a_cold_request_finds_nothing_of_its_own

Run: python tests/test_cache_prompt.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_end_anchor as E  # noqa: E402  (the CPU environment, the random model, the handler helpers)

from engine import cache  # noqa: E402
from server import app  # noqa: E402
from test_app_loop import Req  # noqa: E402
from test_cache import tokens  # noqa: E402

BIG = 1 << 30


def _chat(messages, cold=None):
    body = {"messages": messages, "max_tokens": 40, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if cold is not None:
        body["cache_prompt"] = cold
    head, raw = Req("/v1/chat/completions", body).response()
    assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
    return json.loads(raw)


def test_only_false_turns_it_off():
    assert app.cache_prompt_off({"cache_prompt": False})
    for body in ({}, {"cache_prompt": True}, {"cache_prompt": None}, {"cache_prompt": "false"},
                 {"cache_prompt": 0}):
        assert not app.cache_prompt_off(body), body


def _conversation(user_len=90, **res_kw):
    res = E._state(**res_kw)
    tok = app.STATE["tok"] = E.ChatTok()
    user = tok.decode(tokens(user_len, seed=80))
    _chat([{"role": "user", "content": user}])
    n1 = user_len + 3
    out = list(app.STATE["last_ctx"][n1:])
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": tok.decode(out)},
            {"role": "user", "content": tok.decode(E.TOOL)}]
    return res, tok, msgs


def test_a_cold_request_reads_whole_and_the_conversation_comes_back():
    store = cache.StateStore(BIG, E.CH)
    res, tok, msgs = _conversation(park_bytes=BIG, store=store)
    e = res.end["at"]
    sent = len(tok.apply_chat_template(msgs)["input_ids"][0])
    puts = store.stats["puts"]
    r = _chat(msgs, cold=False)
    last = app.STATE["last_prefill"]
    assert last["reused"] == 0 and last["forwarded"] == sent, (last, sent)
    assert r["timings"]["cache_n"] == 0 and r["timings"]["prompt_n"] == sent, r["timings"]
    assert store.stats["puts"] == puts, store.stats
    # its rows are nobody's: nothing parks or spills them later
    assert res.owner is None and res.valid == 0 and res.end is None, (res.owner, res.valid)
    assert res.stats["parks"] >= 1, res.stats
    # the same turn without the flag: the conversation is back, at its end anchor
    r = _chat(msgs)
    last = app.STATE["last_prefill"]
    assert last["kind"] == "end" and last["reused"] == e, (last, e)
    assert r["timings"]["cache_n"] == e, r["timings"]
    return f"cold read {sent} tokens, then resumed at {e}"


def test_a_short_cold_request_is_a_guest():
    res, tok, msgs = _conversation(user_len=300)
    e, owner = res.end["at"], res.owner
    short = [{"role": "user", "content": tok.decode(tokens(20, seed=81))}]
    r = _chat(short, cold=False)
    assert app.STATE["last_prefill"]["reused"] == 0 and r["timings"]["cache_n"] == 0
    assert res.owner == owner and res._stash is not None, (res.owner, res._stash)
    _chat(msgs)
    last = app.STATE["last_prefill"]
    assert last["kind"] == "end" and last["reused"] == e, (last, e)


def test_a_cold_request_finds_nothing_of_its_own():
    res, tok, msgs = _conversation(park_bytes=BIG)
    sent = len(tok.apply_chat_template(msgs)["input_ids"][0])
    for _ in range(2):
        r = _chat(msgs, cold=False)
        assert r["timings"]["cache_n"] == 0 and r["timings"]["prompt_n"] == sent, r["timings"]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
