"""SPD-215: the resident prefix keeps the state at the end of the last prompt (`keep_prompt_end`).

An agent client sends the conversation back with its own words for the model's answer (a tool loop
that summarises, a bench that writes a fixed reply): the copy parts from the rows right after the
prompt. Without the prompt's end the next turn resumes at the grid line below it and reads the
prompt's tail again; with it the turn resumes at the prompt's end and reads only what is new.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_end_anchor as E  # noqa: E402  (the served loop on the random model)

from engine import cache  # noqa: E402


def _turns(on: bool, n1: int = 30):
    keep, cache.PROMPT_END = cache.PROMPT_END, on
    try:
        res = E._state(end=True)
        prompt = E.tokens(100, seed=60)
        out1, _, _ = E.turn(prompt, n1)
        answer = E.tokens(n1, seed=61)               # the client's own words for the answer
        out2, pre2, _ = E.turn(prompt + answer + E.TOOL, 24)
        return res, out2, pre2
    finally:
        cache.PROMPT_END = keep


def test_the_next_turn_resumes_at_the_prompts_end():
    """The copy parts right after the prompt (100 tokens, a 16-token grid): with the prompt's end it
    resumes at 100 and forwards only the new tokens; without it at the grid line 96."""
    res, out_on, pre_on = _turns(True)
    assert pre_on["kind"] == "pend" and pre_on["reused"] == 100, pre_on
    assert res.stats["pend_hits"] == 1 and res.stats["pend_kept"] >= 2, res.stats
    _, out_off, pre_off = _turns(False)
    assert pre_off["kind"] == "resident" and pre_off["reused"] == 96, pre_off
    assert out_on == out_off, "the greedy tokens are the grid resume's"


def test_a_prompt_that_changed_before_the_end_does_not_take_it():
    """The prompt's end is offered only when the new prompt holds every token up to it."""
    keep, cache.PROMPT_END = cache.PROMPT_END, True
    try:
        E._state(end=True)
        prompt = E.tokens(100, seed=60)
        E.turn(prompt, 10)
        other = prompt[:50] + E.tokens(50, seed=62) + E.TOOL
        _, pre, _ = E.turn(other, 8)
        assert pre.get("kind") != "pend" and int(pre.get("reused") or 0) <= 48, pre
    finally:
        cache.PROMPT_END = keep


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
