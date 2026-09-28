"""The overthinking signal, unit-tested on a CPU.

`ThinkBudget` is the engine's way of saying something to a model that is circling. Two detectors
guard the reasoning block: a short pattern repeated (the pattern-stop machinery, gentler
thresholds), and novelty collapse of 8-grams over two windows. Plus the budget that already
existed. The claims worth testing without a model: each detector fires on its own kind of stall,
neither fires on ordinary varied text, the budget still works, and the whole thing is inactive
outside the block.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.spec import ThinkBudget  # noqa: E402

OPEN, CLOSE, TEXT = 100, 101, 102


class FakeTok:
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return {"<think>": OPEN, "</think>": CLOSE}.get(text, -1)

    def __call__(self, text, add_special_tokens=False):
        seq = {"<think>": [7, 8, 9], "</think>": [10, 11, 12]}.get(text, [TEXT, TEXT])
        return type("R", (), {"input_ids": seq})()


def budget(**kw) -> ThinkBudget:
    return ThinkBudget(FakeTok(), **kw)


def various(n: int, seed: int = 5):
    x = seed
    out = []
    for _ in range(n):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(x % 100000)
    return out


def test_opens_only_when_the_prompt_ends_inside_the_block():
    t = budget(stall=False)
    t.start([1, 2, OPEN])
    assert t.inside
    t.start([OPEN, CLOSE])
    assert not t.inside, "the prompt closed the block itself"
    t.start([1, 2, 3])
    assert not t.inside


def test_budget_still_fires():
    t = budget(budget=10, stall=False)
    t.start([OPEN])
    t.observe(various(9))
    assert not t.hit
    t.observe(various(1))
    assert t.hit


def test_loop_detector_fires_on_a_period_two_cycle():
    t = budget(stall=True)                 # count=4, so eight cycle tokens are enough
    t.start([OPEN])
    t.observe([7, 9] * 5)
    assert t.hit and t.reason.startswith("loop("), f"reason={t.reason}"


def test_novelty_detector_fires_on_a_recycled_window():
    t = budget(stall=True)
    t.start([OPEN])
    block = various(ThinkBudget.STALL_WINDOW, seed=7)
    t.observe(block * 3)                   # windows 2 and 3 are identical
    assert t.hit and t.reason == "novelty", f"reason={t.reason}"


def test_varied_text_never_fires():
    t = budget(stall=True)
    t.start([OPEN])
    t.observe(various(600, seed=11))
    assert not t.hit, f"varied reasoning must not be closed, reason={t.reason}"


def test_nothing_fires_outside_the_block_or_after_it_closes():
    t = budget(stall=True)
    t.start([1, 2, 3])                     # never inside
    t.observe([7, 9] * 40)
    assert not t.hit
    t.start([OPEN])
    t.observe([7, 9] * 3)
    t.observe([CLOSE])
    assert t.done and not t.hit, "after the block closes there is nothing to signal"


def test_literal_tags_arm_the_budget_too():
    # The model sometimes writes the literal characters instead of the special token; a budget
    # watching only the special id never arms (e05-e10, 2026-09-19). Feed the literal sequence.
    t = budget(budget=5, stall=False)
    t.start([1, 2, 3])                       # no tags in the prompt
    t.observe([7, 8, 9])                     # the literal "<think>"
    assert t.inside, "a literal open tag must arm the block"
    t.observe(various(4))
    assert not t.hit
    t.observe(various(1))
    assert t.hit


def test_literal_close_ends_the_block():
    t = budget(stall=False)
    t.start([7, 8, 9])                       # the prompt ends with the literal open
    assert t.inside
    t.observe(various(3))
    t.observe([10, 11, 12])                  # the literal "</think>"
    assert t.done and not t.hit


def test_stall_can_be_turned_off():
    t = budget(stall=False)
    t.start([OPEN])
    t.observe([7, 9] * 40)
    assert not t.hit, "with the stall detector off only the budget applies"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:52s} ok")
            passed += 1
    print(f"{passed} passed")
