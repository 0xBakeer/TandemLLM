"""Structured outputs, on a CPU: the automaton, the masks, and the loop under a constraint.

  * the regex engine accepts exactly what Python's `re.fullmatch` accepts, over random strings;
  * a code point range becomes UTF-8 byte sequences that cover it exactly (brute force);
  * a JSON schema compiles to a language that holds its valid instances -- compact or indented --
    and none of the invalid ones;
  * a state's token mask is exactly the brute-force walk of every token's bytes;
  * in the served loop a constrained greedy request decodes to the same tokens with no drafter, a
    chain and a tree (drafting the unconstrained continuation, so the mask is where drafts die),
    and every answer that ends is in the language;
  * reasoning is not constrained; the closing tag starts the constraint;
  * through the handler: json_schema, json_object, a choice list and a regex give answers in their
    languages on both transports; a constraint with tools, a bad schema and a constraint the
    vocabulary cannot write are 400s that say so; `--no-structured-outputs` refuses as before.

Run: python tests/test_grammar.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import grammar as G  # noqa: E402
from server import app  # noqa: E402
import test_compat as C  # noqa: E402

PATTERNS = [
    r"a|bc", r"(ab)*c?", r"[0-9]{2,4}", r"x[a-c]+y|z{0,2}", r"(?:ab|a)(?:bc|c)", r"[^ab]\d",
    r"\w+\s\w*", r"(a|b){3}", r"[a-e]{1,}q{2,}", r"\.[\]\\]", r"(?:[a-z]\d)?[A-Z]",
]


def _match(p, s):
    d = G.compile_regex(p)
    return bool(d.accept[d.walk(d.start, s.encode())])


def test_the_regex_engine_agrees_with_python():
    rng = random.Random(3)
    alphabet = "abcxyzqAZ019 .]\\\t"
    n = 0
    for p in PATTERNS:
        d = G.compile_regex(p)
        py = re.compile(p)
        for _ in range(400):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
            want = py.fullmatch(s) is not None
            got = bool(d.accept[d.walk(d.start, s.encode())])
            assert got == want, (p, s, got, want)
            n += 1
    for bad in ("(a", "a{3,2}", "a*?", "(?=a)", "\\q", "[a", "^a"):
        try:
            G.compile_regex(bad)
            raise AssertionError(f"{bad!r} must be refused")
        except G.GrammarError:
            pass
    return f"{len(PATTERNS)} patterns x 400 random strings == re.fullmatch; 7 bad patterns refused"


def test_utf8_ranges_cover_exactly():
    rng = random.Random(5)
    for _ in range(60):
        a = rng.choice([0, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, rng.randint(0, 0x10FFFF)])
        b = min(0x10FFFF, a + rng.choice([0, 1, 63, 64, 2047, 70000]))
        seqs = G.utf8_sequences(a, b)
        for cp in list(range(max(0, a - 3), a + 4)) + list(range(b - 3, b + 4)) + \
                [rng.randint(a, b) for _ in range(50)]:
            if cp < 0 or cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
                continue
            enc = chr(cp).encode()
            hit = any(len(s) == len(enc) and all(x <= y <= z for (x, z), y in zip(s, enc))
                      for s in seqs)
            assert hit == (a <= cp <= b), (hex(a), hex(b), hex(cp))
    d = G.compile_regex("[^a]")
    for ch in ("é", "🤦", "中", "\x7f"):
        assert d.accept[d.walk(0, ch.encode())], ch
    half = "🤦".encode()[:2]
    assert d.walk(0, half) != d.dead and not d.accept[d.walk(0, half)], "half a character is a live prefix"
    return "60 random ranges, boundaries and samples; half a 4-byte character is live, not accepting"


SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "city": {"enum": ["Berlin", "Paris"]}, "ok": {"type": "boolean"},
    "score": {"type": "number"}, "extra": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/p"}]}},
    "required": ["name", "age"], "$defs": {"p": {"type": "object", "properties": {
        "x": {"type": "integer"}}, "required": ["x"]}}}


def test_a_schema_holds_its_instances_and_nothing_else():
    d = G.compile_regex(G.WS + G.schema_regex(SCHEMA) + G.WS)
    good = [{"name": "Ann", "age": 31}, {"name": "Ann", "age": -2, "city": "Paris"},
            {"name": "a\"b\\u00e9", "age": 0, "tags": ["x", "y"], "ok": True, "score": 1.5e-3,
             "extra": {"x": 4}},
            {"name": "Ünïcødé 🙂", "age": 7, "extra": None}]
    bad = [{"age": 31}, {"name": "Ann"}, {"name": "Ann", "age": 1.5},
           {"name": "Ann", "age": 1, "city": "Rome"}, {"name": "Ann", "age": 1, "tags": ["a"] * 4},
           {"name": "Ann", "age": 1, "unknown": 1}, {"age": 1, "name": "Ann"},
           {"name": "Ann", "age": 1, "extra": {"y": 1}}]
    for inst in good:
        for text in (json.dumps(inst), json.dumps(inst, indent=2), json.dumps(inst, ensure_ascii=False)):
            assert d.accept[d.walk(0, text.encode())], text
    for inst in bad:
        assert not d.accept[d.walk(0, json.dumps(inst).encode())], inst
    for bad_schema in ({"type": "object", "properties": {"a": {"$ref": "#/$defs/a"}},
                        "$defs": {"a": {"$ref": "#/$defs/a"}}}, {"type": "tuple"}, [1]):
        try:
            G.schema_regex(bad_schema)
            raise AssertionError(f"{bad_schema} must be refused")
        except G.GrammarError:
            pass
    obj = G.compile_regex(G.json_object_regex(3))
    for text in ('{}', '{"a": [1, {"b": null}], "c": "d"}', ' {"a":1} '):
        assert obj.accept[obj.walk(0, text.encode())], text
    for text in ('[]', '"x"', '{"a": 1', '{"a": [[[[1]]]]}'):
        assert not obj.accept[obj.walk(0, text.encode())], text
    return (f"{len(good)} instances x 3 spellings in, {len(bad)} out; a $ref cycle and an unknown "
            f"type refused; json_object: {obj.states} states")


def _toy_vocab():
    rng = random.Random(9)
    toks = [bytes([b]) for b in range(32, 127)] + [None]          # the last one: special
    for _ in range(300):
        toks.append("".join(rng.choice('{}":, abc123') for _ in range(rng.randint(2, 5))).encode())
    toks += ["é".encode(), "é".encode()[:1], "é".encode()[1:]]
    return G.Vocab(toks, len(toks) + 3), toks


def test_a_mask_is_the_brute_force_walk():
    v, toks = _toy_vocab()
    d = G.compile_regex(r'\{"a": ?[0-9]+(, ?"b": ?"[a-cé]*")?\}')
    for state in range(d.states):
        m = v.mask(d, state)
        for i, b in enumerate(toks):
            want = b is not None and d.walk(state, b) != d.dead
            assert m[i] == want, (state, i, b)
        assert not m[len(toks):].any(), "ids past the tokenizer are never allowed"
    return f"{d.states} states x {len(toks)} tokens, split UTF-8 tokens included"


# ------------------------------------------------------------------ the loop under a constraint

class TreeOracle(C.TreeOracle):
    pass


def _grammar(pattern):
    return G.Grammar(pattern, G.Vocab.from_tokenizer(C.Tok(), 97))


def _run(pattern, drafter=None, tree=False, n=40, in_think=False, prompt=None, bias=None):
    C.serve(drafter, tree=tree)
    cons = G.Constraint(_grammar(pattern), {C.EOS}, "cpu", think_end=90, in_think=in_think)
    pen = cons
    if bias:
        from engine.penalty import PenaltySpec, PenaltyState
        pen = G.LogitChain([PenaltyState(PenaltySpec(bias=bias), 97, "cpu"), cons])
    return list(app.generate_stream(torch.tensor(prompt or C.PROMPT), n, {C.EOS}, pen=pen))


LANG = r'\{"k": "[a-z]{2,6}", "n": [0-9]{1,3}\}'


def test_a_constrained_request_is_exact_under_speculation():
    free = C._run(n=40)
    ref = _run(LANG)
    text = C.Tok().decode(ref)
    assert re.fullmatch(LANG, text) and ref[-1] == C.EOS, (text, ref[-3:])
    runs, took = {}, {}
    for label, dr, tree in (("chain, free drafts", C.Oracle(free), False),
                            ("chain, constrained drafts", C.Oracle(ref), False),
                            ("tree, free drafts", C.TreeOracle(free), True),
                            ("tree, constrained drafts", C.TreeOracle(ref), True)):
        runs[label] = _run(LANG, dr, tree)
        took[label] = C._accepted()
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, f"differs from the drafter-less constrained run {ref}: {bad}"
    assert took["chain, constrained drafts"] and took["tree, constrained drafts"], took
    return f"{text!r}: 4 speculative runs == no drafter; drafts accepted {took}"


def test_reasoning_is_not_constrained():
    # a prompt that ends inside the reasoning block: nothing is masked until the closing token
    # (id 90 here) -- a forced close is exactly that token -- and then the language starts
    cons = G.Constraint(_grammar(LANG), {C.EOS}, "cpu", think_end=90, in_think=True)
    cons.seed([1, 2])
    row = torch.zeros(97)
    cons.apply_single(row)
    assert torch.isfinite(row).all(), "inside the reasoning every token is allowed"
    lg = torch.zeros(3, 97)
    cons.apply_chain(lg, [5, 90])
    assert torch.isfinite(lg[:2]).all() and not torch.isfinite(lg[2]).all()
    assert torch.isfinite(lg[2][(ord("{") - 32) % 96]), "after the close, the answer opens with {"
    cons.commit([5, 90])
    assert not cons.thinking
    return "rows before the close are free, the row after it is the language's first state"


def test_the_impossible_state_fails_loudly():
    g = _grammar("é")                         # no token of the toy vocabulary writes 0xC3
    try:
        g.mask(g.dfa.start, frozenset({C.EOS}), "cpu")
        raise AssertionError("an empty mask must raise")
    except G.GrammarError as exc:
        assert "no legal token" in str(exc)
    return "a state no token can leave raises GrammarError"


def _ask(stream=False, **extra):
    C.serve()
    body = dict({"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 60,
                 "stream": stream}, **extra)
    head, raw = C.Req("/v1/chat/completions", body).response()
    if head.startswith("HTTP/1.1 200") and stream:
        ev = C._events(raw)
        text = "".join(c["delta"].get("content") or "" for e in ev for c in e["choices"])
        finish = [c["finish_reason"] for e in ev for c in e["choices"] if c["finish_reason"]]
        err = [e.get("error") for e in ev if e.get("error")]
        return head, text, finish[-1] if finish else None, err
    if head.startswith("HTTP/1.1 200"):
        c = json.loads(raw)["choices"][0]
        return head, c["message"]["content"], c["finish_reason"], []
    return head, raw, None, []


def test_the_handler_serves_every_constraint_form():
    schema = {"type": "object", "properties": {"k": {"type": "string", "maxLength": 4},
                                               "n": {"type": "integer"}}, "required": ["k", "n"]}
    forms = {
        "json_schema": ({"response_format": {"type": "json_schema",
                                             "json_schema": {"name": "s", "schema": schema}}},
                        lambda t: set(json.loads(t)) >= {"k", "n"}),
        "json_object": ({"response_format": {"type": "json_object"}},
                        lambda t: isinstance(json.loads(t), dict)),
        "choice": ({"structured_outputs": {"choice": ["red", "green", "blue"]}},
                   lambda t: t in ("red", "green", "blue")),
        "regex": ({"structured_outputs": {"regex": r"[0-9]{3}-[a-z]{2}"}},
                  lambda t: re.fullmatch(r"[0-9]{3}-[a-z]{2}", t)),
    }
    seen = {}
    for label, (extra, ok) in forms.items():
        for stream in (False, True):
            head, text, finish, err = _ask(stream, **extra)
            assert head.startswith("HTTP/1.1 200"), (label, head, text[:200])
            if finish == "stop":
                assert ok(text), (label, stream, text)
            else:
                assert finish == "length", (label, finish)
            seen[f"{label}/{'stream' if stream else 'json'}"] = finish
    return f"8 answers in their languages ({sorted(set(seen.values()))})"


def test_the_handler_refuses_what_it_cannot_serve():
    tools = [{"type": "function", "function": {"name": "f"}}]
    cases = [({"response_format": {"type": "json_object"}, "tools": tools}, "tools"),
             ({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "x"}}}},
              "response_format"),
             ({"structured_outputs": {"regex": "(a"}}, "structured_outputs"),
             ({"structured_outputs": {"grammar": "root ::= x"}}, "structured_outputs"),
             ({"structured_outputs": {"choice": ["a"]}, "response_format": {"type": "json_object"}},
              "structured_outputs")]
    for extra, param in cases:
        head, raw, _, _ = _ask(**extra)
        assert head.startswith("HTTP/1.1 400") and json.loads(raw)["error"]["param"] == param, \
            (extra, head, raw[:200])
    head, raw, _, _ = _ask(response_format={"type": "json_object"}, tools=tools, tool_choice="none")
    assert head.startswith("HTTP/1.1 200"), "tool_choice none takes the tools out of the request"
    # a constraint the vocabulary cannot write: non-streamed, a 400 that names it; streamed, the
    # stream closes with finish_reason error and the reason
    head, raw, _, _ = _ask(structured_outputs={"regex": "é+"})
    assert head.startswith("HTTP/1.1 400") and "no legal token" in raw, (head, raw[:300])
    head, text, finish, err = _ask(True, structured_outputs={"regex": "é+"})
    assert finish == "error" and err and "no legal token" in err[0]["message"], (finish, err)
    C.serve()
    app.STATE["structured_outputs"] = False
    head, raw = C.Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                              "response_format": {"type": "json_object"}}).response()
    assert head.startswith("HTTP/1.1 400"), "--no-structured-outputs refuses as rc5 did"
    return f"{len(cases)} refusals by name; the impossible constraint on both transports; the flag off"


def test_an_unconstrained_request_never_builds_a_vocabulary():
    C.serve()
    C.chat()
    assert "grammar_vocab" not in app.STATE, "a plain request must not pay for the constraint"
    return "no vocabulary, no grammar, the same processors"


def test_the_mask_cache_is_bounded_and_forgets_dropped_grammars():
    v, _ = _toy_vocab()
    old = G.MASK_CACHE
    G.MASK_CACHE = 5
    try:
        G._MASKS.clear()
        g = G.Grammar(r"[a-c]{8}", v)
        first = g.mask(0, frozenset(), "cpu")
        for s in range(1, 8):
            g.mask(s, frozenset(), "cpu")
        assert len(G._MASKS) == 5 and not g.cached(0, "cpu"), "the least recently used goes first"
        assert torch.equal(g.mask(0, frozenset(), "cpu"), first), "and comes back the same"
        G._GRAMMARS.clear()
        for i in range(3):
            G.grammar_for(f"x{i}", v, keep=2).mask(0, frozenset(), "cpu")
        live = {id(x) for x in G._GRAMMARS.values()}
        assert {k[0] for k in G._MASKS} - {id(g)} <= live, "a dropped grammar leaves no masks"
    finally:
        G.MASK_CACHE = old
        G._MASKS.clear()
        G._GRAMMARS.clear()
    return "5 masks kept of 8, LRU order; a grammar dropped from the cache takes its masks along"


# ------------------------------------------------------------------ tool_choice as a mask

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
             "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "set", "parameters": {
             "type": "object", "properties": {"n": {"type": "integer"}, "mode": {"enum": ["a", "b"]},
                                              "tags": {"type": "array", "items": {"type": "string"}}},
             "required": ["n"]}}},
         {"type": "function", "function": {"name": "now"}}]


def test_the_closer_free_text_is_exactly_the_strings_without_it():
    d = G.compile_regex(G.avoiding("</parameter>"))
    rng = random.Random(1)
    for _ in range(20000):
        s = "".join(rng.choice("</parameter>xyz\n") for _ in range(rng.randint(0, 30)))
        if rng.random() < 0.2:
            cut = rng.randint(0, len(s))
            s = s[:cut] + rng.choice(["</parameter>", "</paramete", "<</parameter",
                                      "</param</parameter>"]) + s[cut:]
        assert bool(d.accept[d.walk(0, s.encode())]) == ("</parameter>" not in s), s
    return "20,000 random strings: accepted exactly when `</parameter>` is absent"


def test_the_tool_call_language():
    pat, typed = G.tool_call_regex(TOOLS, ["read_file", "set", "now"])
    assert typed == {"set": {"n", "tags"}}
    d = G.compile_regex(pat)
    ok = ["<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n"
          "</function>\n</tool_call>",
          "\n<tool_call>\n<function=set>\n<parameter=n>\n3\n</parameter>\n<parameter=mode>\nb\n"
          "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n"
          "</tool_call>",
          '<tool_call><function=set><parameter=n>-2</parameter><parameter=tags>["x", "y"]'
          "</parameter></function></tool_call>",
          "<tool_call>\n<function=read_file>\n<parameter=path>\na </par b\n</parameter>\n"
          "</function>\n</tool_call>"]
    bad = ["I will read it.<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n"
           "</parameter>\n</function>\n</tool_call>",
           "<tool_call>\n<function=write_file>\n</function>\n</tool_call>",
           "<tool_call>\n<function=set>\n<parameter=mode>\na\n</parameter>\n</function>\n"
           "</tool_call>",
           "<tool_call>\n<function=set>\n<parameter=n>\nthree\n</parameter>\n</function>\n"
           "</tool_call>",
           "<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n</parameter>\n"
           "<parameter=mode>\na\n</parameter>\n</function>\n</tool_call>",
           "<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n</parameter>\n</function>\n"
           "</tool_call> and then some text", ""]
    for t in ok:
        assert d.accept[d.walk(0, t.encode())], t
    for t in bad:
        assert not d.accept[d.walk(0, t.encode())], t
    one = G.compile_regex(G.tool_call_regex(TOOLS, ["now"], many=False)[0])
    two = "<tool_call><function=now></function></tool_call>"
    assert one.accept[one.walk(0, two.encode())] and not one.accept[one.walk(0, (two * 2).encode())]
    assert one.accept[one.walk(0, (two + "\n" * 8).encode())]
    assert one.walk(0, (two + "\n" * 9).encode()) == one.dead, "trailing whitespace is bounded"
    return (f"{len(ok)} calls in; prose first, an unknown function, a missing required parameter, "
            f"a mistyped integer, an undeclared parameter, trailing text and nothing at all out; "
            f"many=False: one call")


# an enum only: the random model, free to add digits to an integer, never stops writing one
PICK = [{"type": "function", "function": {"name": "pick", "parameters": {
            "type": "object", "properties": {"color": {"enum": ["red", "green"]}},
            "required": ["color"]}}},
        {"type": "function", "function": {"name": "stop", "parameters": {
            "type": "object", "properties": {}}}}]


def test_the_tool_constraint_is_exact_under_speculation():
    # the random model, left a choice of whitespace between the tags, takes a space forever; a bias
    # against the space (id 0) walks it through the call -- and chains a bias before the mask
    pat, _ = G.tool_call_regex(PICK, ["pick"], many=False)
    free = C._run(n=120)
    ref = _run(pat, n=120, bias={0: -50.0})
    text = C.Tok().decode(ref)
    assert ref[-1] == C.EOS and re.fullmatch(r"[ \t\n\r]*<tool_call>.*</tool_call>[ \t\n\r]*",
                                             text), text
    runs = {label: _run(pat, dr, tree, n=120, bias={0: -50.0})
            for label, dr, tree in (("chain, free drafts", C.Oracle(free), False),
                                    ("chain, constrained drafts", C.Oracle(ref), False),
                                    ("tree, free drafts", C.TreeOracle(free), True),
                                    ("tree, constrained drafts", C.TreeOracle(ref), True))}
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, f"differs from the drafter-less run: {bad}"
    return f"{text!r}: no drafter, chain and tree agree"


def test_the_handler_forces_the_call():
    from server.toolcall import parse_tool_calls  # noqa: F401 -- the path under test
    seen = {}
    for label, extra, allowed in (("required", {"tool_choice": "required"}, {"pick", "stop"}),
                                  ("named", {"tool_choice": {"type": "function",
                                                             "function": {"name": "pick"}}},
                                   {"pick"}),
                                  ("many", {"tool_choice": "required", "parallel_tool_calls": True},
                                   {"pick", "stop"})):
        for stream in (False, True):
            C.serve()
            body = dict({"messages": [{"role": "user", "content": "hello there"}],
                         "max_tokens": 120, "stream": stream, "tools": PICK,
                         "logit_bias": {"0": -50}}, **extra)
            head, raw = C.Req("/v1/chat/completions", body).response()
            assert head.startswith("HTTP/1.1 200"), (label, head, raw[:200])
            if stream:
                ev = C._events(raw)
                names = [d["function"]["name"] for e in ev for c in e["choices"]
                         for d in (c["delta"].get("tool_calls") or []) if d.get("function", {}).get("name")]
                finish = [c["finish_reason"] for e in ev for c in e["choices"] if c["finish_reason"]][-1]
                args = None
            else:
                c = json.loads(raw)["choices"][0]
                names = [t["function"]["name"] for t in c["message"].get("tool_calls") or []]
                finish = c["finish_reason"]
                args = [json.loads(t["function"]["arguments"]) for t in c["message"].get("tool_calls") or []]
            assert finish == "tool_calls", (label, stream, finish, raw[-300:])
            if finish == "tool_calls":
                assert names and set(names) <= allowed, (label, names)
                if label != "many":
                    assert len(names) == 1, ("one call unless parallel_tool_calls is true", names)
                for a in args or []:
                    if "color" in a:
                        assert a["color"] in ("red", "green"), a
            seen[f"{label}/{'stream' if stream else 'json'}"] = (finish, names)
    return f"{seen}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:58s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
