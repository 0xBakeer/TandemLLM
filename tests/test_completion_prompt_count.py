"""SRV-443 (QA SRV-444): a /v1/completions text prompt through `build_prompt` answers every `count`
form a chat prompt does: the host ids `_par_side_need` reads before it picks a slot, the length,
and (the length, not inside <think>). Before, the text branch returned the tensor for each, and
`_shared_head` raised on it: every /v1/completions request with parallel slots was a 500.
SPD-249: the counts never touch the device (they run before any lock, beside an idle graph capture); only
the tensor the caller asks for goes to it.

Run: python tests/test_completion_prompt_count.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok, setup  # noqa: E402

from server import app  # noqa: E402


def test_a_text_prompt_answers_every_count():
    setup()
    app.STATE["tok"] = QTok()
    body = {"prompt": "Once upon a time"}
    ids, kind, in_think = app.build_prompt(body)
    want = QTok().encode("Once upon a time")
    assert kind == "text" and not in_think and ids.tolist() == want
    got = app.build_prompt(body, count="ids")
    assert isinstance(got, list) and got == want and all(isinstance(t, int) for t in got)
    assert app.build_prompt(body, count=True) == len(want)
    assert app.build_prompt(body, count="think") == (len(want), False)
    assert app._shared_head(got, want + [7]) == len(want)
    assert app.build_prompt({"prompt": ["first", "second"]}, count="ids") == QTok().encode("first")


def test_a_text_prompt_counts_on_the_host():
    """SPD-249: `count` of a text prompt makes no copy to the device (a device whose tensors cannot be read back,
    `meta`, stands in for the GPU: a count that went there could not be read); the prompt itself goes there."""
    setup()
    app.STATE["tok"] = QTok()
    old = app.STATE.get("device")
    app.STATE["device"] = "meta"
    try:
        body = {"prompt": "Once upon a time"}
        want = QTok().encode("Once upon a time")
        assert app.build_prompt(body, count="ids") == want
        assert app.build_prompt(body, count=True) == len(want)
        assert app.build_prompt(body, count="think") == (len(want), False)
        ids, kind, _ = app.build_prompt(body)
        assert kind == "text" and ids.device.type == "meta" and ids.numel() == len(want)
    finally:
        app.STATE["device"] = old


if __name__ == "__main__":
    test_a_text_prompt_answers_every_count()
    test_a_text_prompt_counts_on_the_host()
    print("  test_a_text_prompt_answers_every_count ok\n  test_a_text_prompt_counts_on_the_host ok\n2 passed")
