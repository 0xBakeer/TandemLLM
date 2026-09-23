"""The serving loop and its HTTP handler, on a CPU, with the random 4-layer model.

`server/app.py`'s `generate_stream` is its own copy of the decode loop -- a generator, so a token
reaches the socket the moment it is accepted -- and the gate tools never run it: they drive
`engine/spec.py`. So the claims about the SERVED loop are tested here, against the loop itself:

  * `X-Engine-Stop` reaches a non-streamed client, and a streamed one gets the marker in band
    instead, since its headers left before the guard fired (ENG-104);

Run: python tests/test_app_loop.py
"""

from __future__ import annotations

import io
import json
import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.penalty import PenaltySpec  # noqa: E402
from engine.spec import Relax, ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from test_window_edge import FixedDrafter, _engine  # noqa: E402

OPEN, CLOSE = 90, 91
CLOSING = [80, 81, CLOSE]          # the forced phrase; like the real one it ends with the closer


class FakeTok:
    """Just enough tokenizer for the loop, the think budget and the handler."""

    unk_token_id = -1
    eos_token_id = None

    def convert_tokens_to_ids(self, text):
        return {"<think>": OPEN, "</think>": CLOSE}.get(text, -1)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        if text == "<think>":
            ids = [60, 61]
        elif text == "</think>":
            ids = [62, 63]
        elif text == ThinkBudget.PHRASE:
            ids = list(CLOSING)
        else:
            ids = [1 + (ord(c) % 50) for c in text]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return type("R", (), {"input_ids": ids})()

    def decode(self, seq, skip_special_tokens=True):
        return "".join(chr(ord("a") + int(t) % 26) for t in seq)


def serve(drafter=None, tree=False, max_len=256, k=7):
    eng = _engine(max_len)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=drafter, k=k, tree=tree, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=False,
                     prefix_chunk=0, session_cache=False, tok=FakeTok(), device="cpu",
                     max_len=max_len, default_max_tokens=64, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None)
    return eng


class StopOn:
    """A pattern guard that fires on one exact block -- the forced close, here."""

    def __init__(self, block):
        self.block, self.hit, self.label = list(block), False, "stub"

    def observe(self, new):
        if list(new) == self.block:
            self.hit = True
        return self.hit


# ------------------------------------------------------------------ ENG-104 nit 5

class Req(app.Handler):
    """The handler without a socket: the body in, the raw response bytes out."""

    def __init__(self, path, body):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}
        self.path, self.command, self.request_version = path, "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST " + path, ("127.0.0.1", 0)
        self.close_connection = True

    def response(self):
        self.do_POST()
        head, _, body = self.wfile.getvalue().partition(b"\r\n\r\n")
        return head.decode(), body.decode()


class HitAfter:
    """Stands in for `PatternStop`: fires on its third block."""

    def __init__(self, *_):
        self.n, self.hit, self.label = 0, False, "size=1 count=2 tok=[1]"

    def observe(self, new):
        self.n += 1
        self.hit = self.hit or self.n >= 3
        return self.hit


def test_the_stop_header_is_non_streaming_only_and_the_stream_says_it_in_band():
    real = app.PatternStop
    app.PatternStop = HitAfter
    try:
        serve()
        app.STATE["pattern_stop"] = (1, 1, 2)
        head, body = Req("/v1/completions", {"prompt": "hello", "max_tokens": 20}).response()
        assert "X-Engine-Stop: pattern-stop(" in head, head
        assert app.GUARD_MARKER in json.loads(body)["choices"][0]["text"]
        head, body = Req("/v1/completions", {"prompt": "hello", "max_tokens": 20,
                                             "stream": True}).response()
        assert "X-Engine-Stop" not in head, "a streamed response's headers precede the guard"
        text = "".join(json.loads(line[6:])["choices"][0]["text"]
                       for line in body.splitlines()
                       if line.startswith("data: {") and json.loads(line[6:])["choices"])
        assert text.endswith(app.GUARD_MARKER), "the streamed client gets the stop in band"
    finally:
        app.PatternStop = real


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
