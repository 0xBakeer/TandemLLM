# a 400 saying the request is over the window is retried once at half the output — sizing by the
# engine's refusal numbers is self-referential: two refusals of requests 4,097 tokens apart both
# said "a total of at least 262,145", because the input number is back-computed from window + 1.
# Run: python tests/test_chatd_overrun_retry.py — no sandbox, no engine.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd.runs import is_overrun

# the shapes seen live, verbatim — the refusal arrives with the split computed two different
# ways depending on how far the engine got before refusing
VLLM = ("This model's maximum context length is 262144 tokens. However, you requested 260976 "
        "output tokens and your prompt contains at least 1169 input tokens, for a total of at "
        "least 262145 tokens. Please reduce the length of the input prompt or the number of "
        "requested output tokens. (parameter=input_tokens)")
VLLM_TEXT = ("This model's maximum context length is 262144 tokens. However, you requested "
             "262143 output tokens and your prompt contains 300 characters (more than 128 "
             "characters, which is the upper bound for 1 input tokens). "
             "(parameter=input_text, value=300)")
OPENAI = ("This model's maximum context length is 8192 tokens. However, you requested 9500 "
          "prompt tokens and 400 completion tokens, please reduce the prompt length")


def test_the_live_shapes_recognised():
    assert is_overrun(VLLM) and is_overrun(VLLM_TEXT) and is_overrun(OPENAI)


def test_other_400s_stay_fatal():
    for m in ("HTTP 400", "Invalid model 'qwen'", "maximum context is configured to 262144",
              "you requested 400 output tokens", ""):
        assert not is_overrun(m), m


def test_commas_tolerated():
    assert is_overrun("This model's maximum context length is 262,144 tokens. However, ...")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
