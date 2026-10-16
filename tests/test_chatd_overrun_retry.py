# the engine's own 400 is better data than the guard's estimate: a request one token past the
# window used to fail the run outright — a relay backend with no max_output on its card asked a
# 262,144-token window for 260,976 output tokens against 1,169 of input. One retry, capped at the
# room the engine reports. Run: python tests/test_chatd_overrun_retry.py — no sandbox, no engine.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.runs import _overrun_room

# the two shapes seen live, verbatim
VLLM = ("This model's maximum context length is 262144 tokens. However, you requested 260976 "
        "output tokens and your prompt contains at least 1169 input tokens, for a total of at "
        "least 262145 tokens. Please reduce the length of the input prompt or the number of "
        "requested output tokens. (parameter=input_tokens)")
OPENAI = ("This model's maximum context length is 8192 tokens. However, you requested 9500 "
          "prompt tokens and 400 completion tokens, please reduce the prompt length")


def test_the_engine_numbers_become_the_room():
    assert _overrun_room(VLLM) == 262144 - 1169 - 256


def test_openai_wording_parses_too():
    assert _overrun_room(OPENAI) == 0          # 8192 - 9500 - 256 < 256: no room to retry into


def test_openai_with_room():
    assert _overrun_room(
        "This model's maximum context length is 8192 tokens. However, you requested 7000 "
        "prompt tokens and 400 completion tokens") == 8192 - 7000 - 256


def test_other_400s_stay_fatal():
    for m in ("HTTP 400", "Invalid model 'qwen'",
              "maximum context length is 262144 tokens",          # counts missing
              "you requested 400 output tokens"):
        assert _overrun_room(m) == 0, m


def test_commas_tolerated():
    assert _overrun_room("maximum context length is 262,144 tokens. you requested 1,169 "
                         "input tokens") == 262144 - 1169 - 256


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
