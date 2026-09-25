"""The OpenAI-compat corners (SRV-17), on a CPU, through the served loop and the real handler.

Three kinds of claim:

  * **exactness under speculation.** `logit_bias` is a deterministic transform of every target row,
    so a greedy request decodes to the same tokens with or without a drafter, chain or tree -- the
    test drafts the UNBIASED continuation, so the bias is exactly where drafts get rejected. A
    seeded sampled request with `min_p` and `logit_bias` emits the same tokens with no drafter, a
    chain and a tree walk (the keyed draws of ENG-103), and `logprobs` changes no token;
  * **the contract.** One request per OpenAI field through the handler: it works, or it is a 400
    that names the field -- never a silent ignore; and the field table itself is pinned, so a
    field cannot be added or dropped without this file changing;
  * **the shapes.** `n` choices, `logprobs` in both endpoints' shapes, streamed and not, and a
    request without the new fields answered byte for byte as before.

Run: python tests/test_compat.py
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

from engine.penalty import PenaltySpec, PenaltyState  # noqa: E402
from engine.spec import Relax  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from server import app, compat  # noqa: E402
from test_window_edge import _engine  # noqa: E402

PROMPT = [5, 6, 7, 8, 9]
EOS = 96


class Tok:
    """One character per token: ids 0..95 are the printable ASCII from space, 96 is the end."""

    unk_token_id = -1
    eos_token_id = EOS

    def convert_tokens_to_ids(self, text):
        return -1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(ord(c) - 32) % 96 for c in text]
        return type("R", (), {"input_ids": torch.tensor([ids]) if return_tensors else ids})()

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        text = "".join(str(m.get("content") or "") for m in messages)
        return {"input_ids": torch.tensor([[(ord(c) - 32) % 96 for c in text]])}

    def decode(self, seq, skip_special_tokens=True):
        return "".join("" if int(t) == EOS and skip_special_tokens else chr(32 + int(t) % 96)
                       for t in seq)


def serve(drafter=None, tree=False, k=6):
    eng = _engine(256)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=drafter, k=k, tree=tree, sampled_tree="det" if tree
                     else False, relax=Relax(1.0, 1), verbose=False, state_store=None,
                     prefix_cache=False, prefix_chunk=0, session_cache=False, tok=Tok(),
                     device="cpu", max_len=256, default_max_tokens=24, pen_spec=PenaltySpec(),
                     model="t", cfg_eos=EOS, think_budget=0, think_stall=False,
                     reasoning_format="tags", request_timeout=0.0, max_queue=8, queue_timeout=5.0,
                     pattern_stop=None, response_cache=None, suffix_store=None)
    return eng


class Oracle:
    """Drafts a reference continuation with every third token wrong: long accepts, rejections at
    known places, and blocks of every length."""

    def __init__(self, ref: list[int], width: int = 5):
        self.ref, self.width = ref, width

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def _seq(self, ctx, count, corrupt=True):
        i = len(ctx) - len(PROMPT)
        out = []
        for j in range(min(self.width, max(1, count))):
            t = self.ref[i + j] if i + j < len(self.ref) else 0
            out.append((t + 1) % 95 if corrupt and (i + j) % 3 == 2 else t)
        return out

    def propose(self, ctx, count):
        return self._seq(ctx, count)


class TreeOracle(Oracle):
    """The same, as a tree: the uncorrupted path rides beside the corrupted one, so the walk has a
    real choice to make."""

    def propose_tree(self, ctx, count):
        a, b = self._seq(ctx, count), self._seq(ctx, count, corrupt=False)
        seqs = [(a, 0.9)] + ([(b[:3], 0.5)] if b[:3] != a[:3] else [])
        return DraftTree.from_sequences(ctx[-1], seqs)


def _run(drafter=None, tree=False, bias=None, sampler=None, lpr=None, n=24):
    serve(drafter, tree=tree)
    spec = PenaltySpec(bias=bias)
    pen = PenaltyState(spec, 97, "cpu") if spec.on else None
    kw = {"lpr": lpr} if lpr is not None else {}
    return list(app.generate_stream(torch.tensor(PROMPT), n, set(), pen=pen, sampler=sampler,
                                    **kw))


# ------------------------------------------------------------------ exactness under speculation

def _accepted() -> int:
    """Draft tokens the last run's blocks accepted (the loop's own first-miss histogram)."""
    bs = app.STATE.get("blocks")
    return sum(a * n for h in bs.accept.values() for a, n in h.items()) if bs else 0


def test_logit_bias_is_exact_under_speculation():
    plain = _run()
    # push one of the plain answer's later tokens down: the biased answer shares the plain one's
    # first tokens and departs mid-answer, so the unbiased drafts are accepted up to the bias and
    # rejected exactly where it acts
    at = next(i for i in range(6, len(plain)) if plain[i] not in plain[:i])
    bias = {plain[at]: -8.0}
    ref = _run(bias=bias)
    split = next(i for i, (a, b) in enumerate(zip(plain, ref)) if a != b)
    assert 3 <= split < len(ref) - 3, f"the bias must act mid-answer, it acted at {split}"
    runs, took = {}, {}
    for label, dr, tree in (("chain, unbiased drafts", Oracle(plain), False),
                            ("chain, biased drafts", Oracle(ref), False),
                            ("tree, unbiased drafts", TreeOracle(plain), True),
                            ("tree, biased drafts", TreeOracle(ref), True)):
        runs[label] = _run(dr, tree=tree, bias=bias)
        took[label] = _accepted()
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, f"differs from the drafter-less biased run {ref}: {bad}"
    assert all(took.values()), f"every run must accept drafted tokens: {took}"
    return f"4 speculative runs == the drafter-less run; the bias acts at token {split}; drafts accepted {took}"


def test_a_seeded_sample_with_min_p_and_logit_bias_ignores_the_drafter():
    def s():
        return app.Sampler(temperature=0.9, top_p=0.95, min_p=0.05, seed=17)
    bias = {3: 2.0, 11: -4.0}
    ref = _run(bias=bias, sampler=s())
    plain = _run(sampler=s())
    assert ref != plain
    runs = {"chain": _run(Oracle(plain), bias=bias, sampler=s()),
            "tree walk": _run(TreeOracle(plain), tree=True, bias=bias, sampler=s())}
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, f"differs from the drafter-less keyed sample {ref}: {bad}"
    return "min_p + logit_bias, seeded: chain and tree walk == no drafter"


def test_logprobs_change_no_token_and_agree_across_paths():
    plain = _run()
    got = {}
    for label, dr, tree in (("none", None, False), ("chain", Oracle(plain), False),
                            ("tree", TreeOracle(plain), True)):
        rec = app.lp_mod.Recorder(3)
        out = _run(dr, tree=tree, lpr=rec)
        assert out == plain, f"{label}: logprobs changed the output"
        assert [e[0] for e in rec.entries[:len(out)]] == out, label
        got[label] = rec.entries[:len(out)]
    for label in ("chain", "tree"):
        for (t, lp, alts), (t0, lp0, alts0) in zip(got[label], got["none"]):
            assert abs(lp - lp0) < 1e-4, (label, lp, lp0)
            assert [a for a, _ in alts] == [a for a, _ in alts0]
            assert alts[0][0] == t, "greedy: the chosen token is the top alternative"
    return "3 paths, 24 tokens: same ids, logprobs within 1e-4 of the single-step rows"


# ------------------------------------------------------------------ the handler

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
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            self.do_POST()
        head, _, body = self.wfile.getvalue().partition(b"\r\n\r\n")
        return head.decode(), body.decode()


def chat(**extra):
    serve()
    body = dict({"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 12},
                **extra)
    return Req("/v1/chat/completions", body).response()


def completion(**extra):
    serve()
    return Req("/v1/completions", dict({"prompt": "hello there", "max_tokens": 12},
                                       **extra)).response()


def _events(raw: str) -> list[dict]:
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]


def test_the_field_table_is_pinned():
    chat_fields = {
        "messages", "model", "stream", "stream_options", "max_tokens", "max_completion_tokens",
        "stop", "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty",
        "logit_bias", "logprobs", "top_logprobs", "n", "tools", "tool_choice",
        "parallel_tool_calls", "reasoning_effort", "response_format", "user", "metadata", "store",
        "service_tier", "prompt_cache_key", "safety_identifier", "prediction", "modalities",
        "audio", "functions", "function_call", "web_search_options", "verbosity"}
    assert set(compat.CHAT_FIELDS) == chat_fields, set(compat.CHAT_FIELDS) ^ chat_fields
    comp_fields = {"prompt", "model", "stream", "stream_options", "max_tokens", "stop",
                   "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty",
                   "logit_bias", "logprobs", "n", "user", "best_of", "echo", "suffix"}
    assert set(compat.COMPLETION_FIELDS) == comp_fields, set(compat.COMPLETION_FIELDS) ^ comp_fields
    return f"{len(chat_fields)} chat + {len(comp_fields)} completion fields, one disposition each"


# (field, value, "ok" | the 400's param) -- one request per field and value
CHAT_MATRIX = [
    ("audio", {"voice": "alloy", "format": "wav"}, "audio"),
    ("functions", [{"name": "f", "parameters": {}}], "functions"),
    ("function_call", "auto", "function_call"),
    ("web_search_options", {}, "web_search_options"),
    ("verbosity", "low", "verbosity"),
    ("modalities", ["text"], "ok"),
    ("modalities", ["text", "audio"], "modalities"),
    ("response_format", {"type": "text"}, "ok"),
    ("response_format", {"type": "json_object"}, "response_format"),
    ("response_format", {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
     "response_format"),
    ("n", 2, "ok"),
    ("n", 0, "n"),
    ("n", 17, "n"),
    ("logprobs", True, "ok"),
    ("logprobs", "yes", "logprobs"),
    ("top_logprobs", 3, "top_logprobs"),              # without logprobs: true
    ("parallel_tool_calls", False, "ok"),
    ("parallel_tool_calls", "no", "parallel_tool_calls"),
    ("tool_choice", "sometimes", "tool_choice"),
    ("tool_choice", "required", "tool_choice"),       # no tools
    ("logit_bias", {"5": 10}, "ok"),
    ("logit_bias", {"999999": 1}, "logit_bias"),
    ("logit_bias", {"5": 101}, "logit_bias"),
    ("logit_bias", {"x": 1}, "logit_bias"),
    ("logit_bias", [1, 2], "logit_bias"),
    ("min_p", 0.1, "ok"),
    ("min_p", 1.5, "temperature"),                   # the sampler's own 400 names its group
    ("user", "someone", "ok"),
    ("metadata", {"conversation_id": "c1"}, "ok"),
    ("store", True, "ok"),
    ("service_tier", "auto", "ok"),
    ("prompt_cache_key", "k", "ok"),
    ("safety_identifier", "s", "ok"),
    ("prediction", {"type": "content", "content": "hi"}, "ok"),
    ("seed", 3, "ok"),
    ("stop", ["zz"], "ok"),
]


def test_every_chat_field_works_or_is_refused_by_name():
    for field, value, want in CHAT_MATRIX:
        head, body = chat(**{field: value})
        if want == "ok":
            assert head.startswith("HTTP/1.1 200"), (field, value, body[:300])
        else:
            assert head.startswith("HTTP/1.1 400"), (field, value, head, body[:200])
            assert json.loads(body)["error"]["param"] == want, (field, value, body)
    head, _ = chat(n=2, stream=True)
    assert head.startswith("HTTP/1.1 400"), "n > 1 streamed is refused, not served as n = 1"
    return f"{len(CHAT_MATRIX) + 1} requests: every one works or names its field"


def test_every_completion_field_works_or_is_refused_by_name():
    matrix = [("echo", True, "echo"), ("echo", False, "ok"), ("best_of", 2, "best_of"),
              ("best_of", 1, "ok"), ("suffix", "tail", "suffix"), ("logprobs", 2, "ok"),
              ("logprobs", 6, "logprobs"), ("n", 2, "ok"), ("logit_bias", {"5": -100}, "ok")]
    for field, value, want in matrix:
        head, body = completion(**{field: value})
        if want == "ok":
            assert head.startswith("HTTP/1.1 200"), (field, value, body[:300])
        else:
            assert head.startswith("HTTP/1.1 400") and json.loads(body)["error"]["param"] == want, \
                (field, value, body[:200])
    return f"{len(matrix)} requests"


def _strip_times(payload: dict) -> dict:
    p = dict(payload)
    for k in ("id", "created", "timings", "metrics"):
        p.pop(k, None)
    return p


def test_neutral_fields_do_not_change_the_answer():
    _, base = chat()
    for field, value in (("store", True), ("metadata", {"conversation_id": "c1"}),
                         ("service_tier", "auto"), ("prompt_cache_key", "k"),
                         ("prediction", {"type": "content", "content": "hello"}),
                         ("safety_identifier", "s"), ("user", "u"),
                         ("response_format", {"type": "text"}), ("modalities", ["text"]),
                         ("parallel_tool_calls", True), ("logit_bias", {}), ("min_p", 0.0)):
        _, body = chat(**{field: value})
        assert _strip_times(json.loads(body)) == _strip_times(json.loads(base)), field
    return "12 fields: the same answer, usage included"


def test_a_request_without_the_new_fields_is_answered_as_before():
    """The chunks and the JSON body of a plain request carry `logprobs: null`, one choice at index
    0, and nothing new -- the bytes a client saw from rc5."""
    _, body = chat()
    p = json.loads(body)
    assert list(p) == ["id", "object", "created", "model", "usage", "choices", "timings",
                       "metrics"]
    assert list(p["choices"][0]) == ["index", "finish_reason", "logprobs", "message"]
    assert p["choices"][0]["logprobs"] is None and len(p["choices"]) == 1
    _, raw = chat(stream=True)
    for ev in _events(raw):
        for c in ev["choices"]:
            assert c["logprobs"] is None and c["index"] == 0
    return "keys in rc5's order, logprobs null on every chunk"


def test_logit_bias_steers_the_answer():
    _, body = chat(logit_bias={"33": 100}, max_tokens=6)       # id 33 is "A"
    assert json.loads(body)["choices"][0]["message"]["content"] == "AAAAAA"
    return "+100 on one token: every token is that token"


def test_n_choices():
    _, body = chat(n=3)
    p = json.loads(body)
    assert [c["index"] for c in p["choices"]] == [0, 1, 2]
    texts = {c["message"]["content"] for c in p["choices"]}
    assert len(texts) == 1, "greedy: every choice is the greedy answer"
    one = json.loads(chat()[1])
    assert p["usage"]["completion_tokens"] == 3 * one["usage"]["completion_tokens"]
    s = dict(n=3, temperature=1.0, seed=5, max_tokens=16)
    a, b = json.loads(chat(**s)[1]), json.loads(chat(**s)[1])
    ta = [c["message"]["content"] for c in a["choices"]]
    assert ta == [c["message"]["content"] for c in b["choices"]], "a seed reproduces every choice"
    assert len(set(ta)) == 3, f"sampled choices differ: {ta}"
    first = json.loads(chat(temperature=1.0, seed=5, max_tokens=16)[1])
    assert first["choices"][0]["message"]["content"] == ta[0], "choice 0 is the n = 1 answer"
    return f"greedy x3 identical; seeded x3 distinct and reproduced: {ta}"


def test_chat_logprobs_both_paths():
    _, body = chat(logprobs=True, top_logprobs=2)
    p = json.loads(body)
    content = p["choices"][0]["logprobs"]["content"]
    assert len(content) == p["usage"]["completion_tokens"] - \
        (1 if p["choices"][0]["finish_reason"] == "stop" else 0)
    assert "".join(e["token"] for e in content) == p["choices"][0]["message"]["content"]
    for e in content:
        assert e["logprob"] <= 0.0 and len(e["top_logprobs"]) == 2
        assert e["top_logprobs"][0]["token"] == e["token"], "greedy picks the top alternative"
        assert bytes(e["bytes"]).decode() == e["token"]
    _, raw = chat(logprobs=True, top_logprobs=2, stream=True)
    streamed = [e for ev in _events(raw) for c in ev["choices"]
                for e in ((c.get("logprobs") or {}).get("content") or [])]
    assert [e["token"] for e in streamed] == [e["token"] for e in content]
    assert all(abs(a["logprob"] - b["logprob"]) < 1e-6 for a, b in zip(streamed, content))
    return f"{len(content)} tokens, the same entries streamed and not"


def test_completion_logprobs_legacy_shape():
    _, body = completion(logprobs=2)
    c = json.loads(body)["choices"][0]
    lp = c["logprobs"]
    assert set(lp) == {"tokens", "token_logprobs", "top_logprobs", "text_offset"}
    assert "".join(lp["tokens"]) == c["text"]
    assert lp["text_offset"] == [sum(len(t) for t in lp["tokens"][:i]) for i in range(len(lp["tokens"]))]
    assert all(len(t) <= 2 for t in lp["top_logprobs"])
    _, raw = completion(logprobs=2, stream=True)
    toks, offs = [], []
    for ev in _events(raw):
        got = ev["choices"][0].get("logprobs")
        if got:
            toks += got["tokens"]
            offs += got["text_offset"]
    assert toks == lp["tokens"] and offs == lp["text_offset"]
    return "tokens / token_logprobs / top_logprobs / text_offset, the same streamed"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:62s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
