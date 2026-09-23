"""The serving loop and its HTTP handler, on a CPU, with the random 4-layer model.

`server/app.py`'s `generate_stream` is its own copy of the decode loop -- a generator, so a token
reaches the socket the moment it is accepted -- and the gate tools never run it: they drive
`engine/spec.py`. So the claims about the SERVED loop are tested here, against the loop itself:

  * a pattern-stop hit inside the forced reasoning close ends the stream there, on every verify
    path, with the KV holding exactly what the stream says was written (SRV-18);
  * `X-Engine-Stop` reaches a non-streamed client, and a streamed one gets the marker in band
    instead, since its headers left before the guard fired (ENG-104);
  * the single-token step closes the reasoning block like the verified ones (SRV-19), and brings
    the drafter current like engine/spec.py's does (SRV-20);
  * a seed reproduces a sampled request whatever the drafter and the width choices do (ENG-103);
  * a graceful stop lets the generation in flight finish (SRV-21);
  * a failed non-streamed request is logged and counted like a streamed one (SRV-22);
  * a tool call written inside the reasoning block is not a call (SRV-23);

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


# ------------------------------------------------------------------ SRV-18

def _forced_close_with_guard(drafter, tree):
    eng = serve(drafter, tree=tree)
    prompt = torch.tensor([5, 6, 7, 8, OPEN])
    think = ThinkBudget(FakeTok(), budget=3, stall=False)
    guard = StopOn(CLOSING)
    out = list(app.generate_stream(prompt, 60, set(), think, pstop=guard))
    ctx = app.STATE["last_ctx"]
    return eng, out, ctx, guard


def test_a_guard_hit_in_the_forced_close_ends_the_stream():
    """SRV-18. `_force_close` returned early when the closing phrase tripped the pattern guard,
    and the caller carried on decoding as if nothing had happened: the guard's stop was lost, and
    the next step forwarded `ctx[-1]` -- the phrase's last token, already in the KV -- a second
    time, one row further on. The stream must end on the phrase, with every token in `ctx` in the
    KV exactly once."""
    for label, drafter, tree in (("chain", FixedDrafter(97, 5, tree_mode=False), False),
                                 ("tree", FixedDrafter(97, 5, tree_mode=True), True)):
        eng, out, ctx, guard = _forced_close_with_guard(drafter, tree)
        assert guard.hit, label
        assert out[-len(CLOSING):] == CLOSING, f"{label}: the stream must end on the phrase: {out}"
        assert out.count(CLOSING[0]) >= 1 and len(out) < 30, f"{label}: decoding went on: {out}"
        assert eng.kv.length == len(ctx), (
            f"{label}: kv.length {eng.kv.length} vs len(ctx) {len(ctx)} -- the forced forward "
            f"wrote every token of ctx, and nothing may be written after it")


# ------------------------------------------------------------------ the single-token step

def test_the_reasoning_close_fires_on_the_single_token_path():
    """SRV-19. The budget and the stall signal closed the block only after a VERIFIED block; the
    single-token step -- every step of a drafter-less server, and a declined step of any other --
    observed the token and never looked at `think.hit`. A drafter-less server never closed a
    reasoning block at all. With the guard on the phrase, the stream also ends there (SRV-18)."""
    serve(None)
    think = ThinkBudget(FakeTok(), budget=3, stall=False)
    out = list(app.generate_stream(torch.tensor([5, 6, 7, 8, OPEN]), 30, set(), think))
    assert think.done, f"the block was never closed: {out}"
    i = next(i for i in range(len(out)) if out[i:i + len(CLOSING)] == CLOSING)
    assert i == 4, f"closed after the 4th token (3 in the budget + the one that crossed it): {out}"
    assert len(out) > i + len(CLOSING), "the answer follows the phrase"
    eng, out, ctx, guard = _forced_close_with_guard(None, False)
    assert guard.hit and out[-len(CLOSING):] == CLOSING and eng.kv.length == len(ctx), out


class Cursor:
    """A block drafter's position discipline, as DFlash2 has it: its cache covers the positions it
    was synced for, and it declines whenever the anchor is not the first position it lacks."""

    def __init__(self, decline_on):
        self.ctx_len, self.calls, self.decline_on = 0, 0, decline_on
        self.covered, self.after = set(), 0

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.covered.update(range(first_pos, first_pos + len(tokens)))
        self.ctx_len = first_pos + len(tokens)

    def propose(self, ctx, k):
        self.calls += 1
        if self.calls == self.decline_on or len(ctx) - 1 != self.ctx_len:
            return []
        if self.calls > self.decline_on:
            self.after += 1
        return [(len(ctx) * 7 + i) % 97 for i in range(3)]


def test_a_declined_step_keeps_the_drafter_current():
    """SRV-20. A declined step forwards one token and did not sync the drafter, as engine/spec.py
    does. A drafter whose cache is indexed by position -- the block drafter -- was then one
    position behind for good, declined every later step, and the request finished one token a
    forward. After one decline the drafter must be current again and proposing."""
    dr = Cursor(decline_on=3)
    eng = serve(dr)
    out = list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 40, set()))
    assert len(out) == 40
    assert dr.after > 0, "the drafter never proposed again after its one decline"
    committed = len(app.STATE["last_ctx"]) - 1         # the last token is decided, not forwarded
    missing = set(range(committed)) - dr.covered
    assert not missing, f"positions the drafter was never synced for: {sorted(missing)}"


# ------------------------------------------------------------------ ENG-103

class QDrafter:
    """A drafter that SAMPLES its proposals from its own distribution, as DFlash2 does under a
    sampled request, and carries q for the accept. Its q is deliberately unlike the target's."""

    def __init__(self, width, seed=4):
        self.width, self.sampler, self.last_q = width, None, None
        g = torch.Generator().manual_seed(seed)
        self.q = torch.softmax(torch.randn(97, generator=g) * 3.0, dim=-1)

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def set_sampling(self, sampler):
        self.sampler = sampler if sampler is not None and sampler.on else None

    def propose(self, ctx, k):
        s, n = self.sampler, min(k, self.width)
        if s is None:
            return [int(self.q.argmax())] * n
        self.last_q = [self.q] * n
        if getattr(s, "coupled", False):
            return [s.pick_at(self.q, len(ctx) + i) for i in range(n)]
        return [s.pick(self.q) for _ in range(n)]


class Wobble(FixedDrafter):
    """Changes its block width from step to step, the way the length router's latch lands on a
    different width when the box runs at a different speed."""

    def __init__(self, seed):
        super().__init__(97, 3, tree_mode=False)
        self.g = torch.Generator().manual_seed(seed)

    def propose(self, ctx, count):
        self.width = int(torch.randint(1, 9, (1,), generator=self.g))
        return super().propose(ctx, count)


def _sampled(drafter, tree=False, seed=42):
    serve(drafter, tree=tree)
    app.STATE["sampled_tree"] = tree
    s = app.Sampler(temperature=0.9, top_p=0.95, seed=seed)
    return list(app.generate_stream(torch.tensor([5, 6, 7, 8, 9]), 48, set(), sampler=s))


def test_a_seed_reproduces_the_request_whatever_the_drafter_does():
    """ENG-103. One generator stream fed both the drafter's proposals and the accept's draws, in an
    order the drafter decided -- and the served length router decides its width from wall-clock
    timing. Two runs of one seeded request under different load drew differently and diverged.
    With position-keyed draws the output is a function of (prompt, params, seed) alone: no
    drafter, two fixed widths, two different width schedules, a sampling drafter at two widths,
    and the tree walk all emit the same tokens."""
    ref = _sampled(None)
    runs = {
        "chain width 3": _sampled(FixedDrafter(97, 3, tree_mode=False)),
        "chain width 7": _sampled(FixedDrafter(97, 7, tree_mode=False)),
        "width schedule A": _sampled(Wobble(1)),
        "width schedule B": _sampled(Wobble(2)),
        "sampling drafter 4": _sampled(QDrafter(4)),
        "sampling drafter 8": _sampled(QDrafter(8)),
        "tree walk": _sampled(FixedDrafter(97, 5, tree_mode=True), tree=True),
    }
    bad = {label: out for label, out in runs.items() if out != ref}
    assert not bad, f"diverged from the drafter-less run {ref}: {bad}"
    assert _sampled(None, seed=43) != ref, "and the seed is what the output depends on"


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



# ------------------------------------------------------------------ SRV-22

def test_a_failed_non_streamed_request_is_logged_and_counted():
    """SRV-22. `_log_request` is "one line per generation, always, whatever happened to it", and
    it is what counts `errors` for /health and /metrics. The streamed path calls it on an
    exception; the non-streamed one let the exception go straight to `do_POST`'s 500, so a failed
    JSON request left no [req] line and no count -- /health said errors: 0 over a failing server."""
    import contextlib

    def broken(*a, **k):
        yield 5
        raise RuntimeError("the engine fell over")

    serve()
    real = app.generate_stream
    app.generate_stream = broken
    before = app.INFLIGHT["errors"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            head, body = Req("/v1/completions", {"prompt": "hello", "max_tokens": 8}).response()
    finally:
        app.generate_stream = real
    assert head.startswith("HTTP/1.1 500"), head
    assert app.INFLIGHT["errors"] == before + 1, "the failure must be counted"
    assert "finish=error" in out.getvalue() and "RuntimeError" in out.getvalue(), out.getvalue()


# ------------------------------------------------------------------ SRV-23

class CharTok(FakeTok):
    """One token per character, so a scripted generation can spell any text -- tags included."""

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        text = "".join(m["content"] for m in messages) + "\n<think>\n"
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def decode(self, seq, skip_special_tokens=True):
        return "".join(chr(int(t)) for t in seq)


THOUGHT = ("I could call <tool_call>\n<function=delete_file>\n<parameter=path>\n/home/user/a.txt"
           "\n</parameter>\n</function>\n</tool_call> but I should ask first.\n</think>\n\n")


def _chat(answer, stream):
    serve()
    app.STATE["tok"] = CharTok()
    real = app.generate_stream
    app.generate_stream = lambda *a, **k: iter([ord(c) for c in THOUGHT + answer])
    try:
        return Req("/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "tidy up"}], "stream": stream,
                    "tools": [{"type": "function", "function": {"name": "delete_file"}}]}
                   ).response()
    finally:
        app.generate_stream = real


def test_a_tool_call_inside_the_reasoning_is_not_a_call():
    """SRV-23. In `tags` format the reasoning travels in `content`, and both the streamed tool-call
    buffer and the non-streamed parser read all of it: a call the model only DELIBERATES about --
    "I could call delete_file ... but I should ask first" -- went out as a real `tool_calls` entry
    with finish_reason tool_calls, for a client to execute. Only the answer can call a tool; the
    thought stays visible as text."""
    head, body = _chat("Should I delete a.txt?", stream=False)
    msg = json.loads(body)["choices"][0]
    assert "tool_calls" not in msg["message"], msg
    assert msg["finish_reason"] == "length" and "<tool_call>" in msg["message"]["content"]
    head, body = _chat("Should I delete a.txt?", stream=True)
    assert '"tool_calls"' not in body and "finish_reason\": \"tool_calls" not in body, body[-400:]
    # and a call in the ANSWER still is one, on both paths
    call = ("<tool_call>\n<function=delete_file>\n<parameter=path>\n/home/user/a.txt\n"
            "</parameter>\n</function>\n</tool_call>")
    head, body = _chat(call, stream=False)
    msg = json.loads(body)["choices"][0]["message"]
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["delete_file"], msg
    head, body = _chat(call, stream=True)
    assert body.count('"name": "delete_file"') == 1, body[-600:]


# ------------------------------------------------------------------ SRV-21

_DRAIN_CHILD = r"""
import os, sys, time
sys.path.insert(0, os.path.join({root!r}, "tests"))
import test_app_loop as t
from http.server import ThreadingHTTPServer
from server import app

t.serve()


def slow(*a, **k):
    for tok in (5, 6, 7, 8, 9, 10, 11, 12):
        time.sleep(0.3)
        yield tok


app.generate_stream = slow
httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
print(httpd.server_address[1], flush=True)
app.serve_until_drained(httpd)
"""


def test_a_graceful_stop_lets_the_generation_in_flight_finish():
    """SRV-21. SIGTERM set `draining` and shut the listener down -- and then `main()` returned.
    ThreadingHTTPServer's handler threads are DAEMONS, so the interpreter exited under the stream
    it had promised to finish: RUNBOOK's "it drains first", stop.sh's grace period and every
    hold's stop cut the generation in flight instead. A streamed request that is running when the
    signal lands must end with its finish chunk and [DONE], and the process must exit 0."""
    import signal
    import socket
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    child = subprocess.Popen([sys.executable, "-c", _DRAIN_CHILD.format(root=root)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             env=env)
    try:
        port = int(child.stdout.readline())
        body = json.dumps({"prompt": "hello", "max_tokens": 8, "stream": True}).encode()
        s = socket.create_connection(("127.0.0.1", port), timeout=30)
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        got = s.recv(65536)
        while b"data: {" not in got:
            got += s.recv(65536)
        child.send_signal(signal.SIGTERM)             # mid-stream: the first token is out
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            got += chunk
        assert b"data: [DONE]" in got, f"the stream was cut: {got[-300:]!r}"
        assert got.count(b'"text"') >= 8, "every token reached the client"
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            child.kill()

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
