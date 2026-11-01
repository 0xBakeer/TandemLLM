"""The conductor's task router (SRV-129, QA SRV-130) and profile resolution (SRV-117, QA SRV-118),
on the checkpoint's own chat template (tests/fixtures/qwen38_chat_template.jinja, rendered by
transformers) through the server's own `build_prompt`, with the classifiers replaced by fakes.

SRV-129's Gherkin, row by row:
  1. a code request gets the xhigh sentence in the tail note and no budget below the backstop;
  2. a summary thinks off through `enable_thinking: false`, and the render equals thinking on up
     to its last line (effort medium); at any other effort it is budget 1 instead;
  3. a design request gets the ui-design text before the last <|im_end|>, and the clean prompt
     (what the client sent, what the suffix store keeps) has none of it;
  4. a tool-result turn asks no model;
  5. a vague follow-up below the switch threshold keeps the class;
  6. the decider down: Qwen decides; both down: the session's class, else the default;
  7. shadow changes nothing and journals what it would have done;
  8. the note is replayed at the same place, so turn 2's prompt starts with turn 1's prompt;
  9. a compaction drops the note and the class block comes back once;
 10. one model (2026-10-01): /v1/models lists TandemLLM alone, the raw engine's old name is a
     hidden alias that plans exactly as TandemLLM, and `X-Tandem: raw` (or the locked `raw` /
     `bench` profile by header) gets an inactive plan (nothing changes);
 11. a client budget wins;
 12. a class added to the router is used by the next request;
 13. the test box explains a decision with the exact note.
Plus: no word decides a class (with every classifier down, opposite requests get the same class),
a skill the client loaded is not injected again, a class with several packages asks a second
question, the checker result of turn N+1 lands on turn N's row, and SRV-117's resolution order.

Run: python tests/test_conductor_router.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from server import app  # noqa: E402
from server.conductor import inject, schema, signals  # noqa: E402
from server.conductor.classify import Answer  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from server.conductor.router import COMPOSITE, resolve  # noqa: E402
from server.conductor.store import ConfigError  # noqa: E402

TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                             "qwen38_chat_template.jinja")).read()
SPECIAL = {"<|im_start|>": 1_100_001, "<|im_end|>": 1_100_002, "<think>": 1_100_003,
           "</think>": 1_100_004}
INV = {v: k for k, v in SPECIAL.items()}
TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"{n} a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}} for n in ("read", "bash", "skill")]
SYSTEM = ("You are opencode, an interactive CLI tool that helps users with software engineering "
          "tasks. Skills available: pencil-sketch, spiral-portrait.")


class QTok:
    """The real template, one id per character, the four special tokens as single ids."""

    unk_token_id = -1
    eos_token_id = SPECIAL["<|im_end|>"]

    def convert_tokens_to_ids(self, text):
        return SPECIAL.get(text, -1)

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for s, t in SPECIAL.items():
                if text.startswith(s, i):
                    out.append(t)
                    i += len(s)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text or "")
        return type("R", (), {"input_ids": torch.tensor([ids]) if return_tensors else ids})()

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None,
                            return_dict=False, tools=None, tokenize=True, **kw):
        text = render_jinja_template([messages], tools=tools, chat_template=TEMPLATE,
                                     add_generation_prompt=add_generation_prompt, **kw)[0][0]
        if not tokenize:
            return text
        return {"input_ids": torch.tensor([self.encode(text)])}

    def decode(self, seq, skip_special_tokens=True):
        return "".join(("" if skip_special_tokens else INV[int(t)]) if int(t) in INV
                       else chr(int(t)) for t in seq)


class Fake:
    """A classifier stage: `fn(text, options)` -> (label, confidence) or "error:<why>"."""

    def __init__(self, kind, fn):
        self.kind, self.fn, self.calls = kind, fn, []

    def choice(self, state, question, options, max_chars=None):
        # the router's context line (SRV-153, tests/test_classifier_digest.py) follows the user's
        # text on its own last line; these fakes answer from the text, as the rows were written
        text = state.split("\nprev=", 1)[0]
        self.calls.append((text, dict(options)))
        r = self.fn(text, options)
        if isinstance(r, str):
            return Answer(self.kind, error=r)
        label, conf = r
        return Answer(self.kind, label, conf, {label: conf}, 1.5, model="fake")


def setup(decider=None, qwen=None, effort="medium"):
    """A conductor on a temp dir, the server's state for `build_prompt`, and the two fakes."""
    app.STATE.clear()
    app.STATE.update(tok=QTok(), device="cpu", reasoning_effort=effort, segments=None,
                     model="qwen38-spark-engine")
    fakes = {"decider": Fake("decider", decider or (lambda t, o: "decider_unreachable")),
             "qwen": Fake("qwen", qwen or (lambda t, o: "qwen_unreachable"))}
    cond = Conductor(tempfile.mkdtemp(prefix="cond-"), state=app.STATE, journal=False,
                     factory=lambda kind, role, timeout: fakes[kind])
    return cond, fakes


def body(messages, tools=True, **kw):
    b = {"model": COMPOSITE, "messages": messages, **kw}
    if tools:
        b["tools"] = TOOLS
    return b


def turn(cond, b, headers=None, kind="opencode"):
    """What the server does with one request: plan, render, apply. Returns (plan, ids, clean)."""
    plan = cond.plan(b, headers or {"x-session-id": "ses-1"}, client_kind=kind,
                     request_id="req", chat=True)
    prompt, _, in_think = app.build_prompt(b)
    out, plan.in_think_after = plan.apply_prompt(prompt, app.STATE["tok"], in_think)
    ids = out.tolist()
    return plan, ids, plan.clean(prompt.tolist())


def text_of(ids) -> str:
    return QTok().decode(ids, skip_special_tokens=False)


def by_label(label, conf):
    return lambda t, o: (label, conf)


CODE = "Refactor the lexer in src/lexer.ts so every token carries its line number."


# ------------------------------------------------------------------ SRV-117: resolution

def test_only_tandemllm_is_listed():
    cond, _ = setup()
    names = [m["id"] for m in cond.models("qwen38-spark-engine", 1)]
    assert names == ["TandemLLM"], "one model: the raw name is not listed"
    # the card answers for the one model and for the hidden alias, nothing else
    assert cond.model_card("TandemLLM", "qwen38-spark-engine", 1)["id"] == "TandemLLM"
    assert cond.model_card("qwen38-spark-engine", "qwen38-spark-engine", 1)["id"] == "TandemLLM"
    assert cond.model_card("gpt-4o", "qwen38-spark-engine", 1) is None


def test_the_old_name_is_a_hidden_alias():
    """The raw engine's old name plans exactly as TandemLLM: same profile, same class, same note."""
    cond, fakes = setup(decider=by_label("code_edit", 0.9))
    st = cond.store
    for name in ("qwen38-spark-engine", "QWEN38-SPARK-ENGINE", "qwen3.8-27b", ""):
        r = resolve(st, name, "qwen38-spark-engine", {}, "opencode")
        assert (r.profile, r.mode, r.source) == ("coding", "on", "client"), (name, r.profile)
        r = resolve(st, name, "qwen38-spark-engine", {"X-Tandem-Profile": "overnight"}, "opencode")
        assert (r.profile, r.mode) == ("overnight", "on")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
    a = body(msgs)
    plan_a, ids_a, _ = turn(cond, a, headers={"x-session-id": "alias-a"})
    b = dict(body(msgs), model="qwen38-spark-engine")
    plan_b, ids_b, _ = turn(cond, b, headers={"x-session-id": "alias-b"})
    assert plan_a.active and plan_b.active and ids_a == ids_b
    assert plan_b.row["class_id"] == "code_edit" and plan_b.row["profile"] == "coding"


def test_raw_is_a_header_not_a_name():
    """`X-Tandem: raw`, or the locked raw / bench profile by header, is the release's path."""
    cond, fakes = setup(decider=by_label("code_edit", 0.9))
    st = cond.store
    for hdr in ({"X-Tandem": "raw"}, {"X-Tandem": "off"}, {"X-Tandem-Profile": "bench"},
                {"X-Tandem-Profile": "raw"}):
        for name in (COMPOSITE, "qwen38-spark-engine"):
            r = resolve(st, name, "qwen38-spark-engine", hdr, "opencode")
            assert r.mode == "raw", (hdr, name)
            b = dict(body([{"role": "user", "content": CODE}]), model=name)
            plan, ids, clean = turn(cond, b, headers=dict(hdr, **{"x-session-id": "raw"}))
            assert not plan.active and ids == clean and not fakes["decider"].calls
            assert plan.think_budget(0) == 0 and not plan.changed and plan.in_think_after
    r = resolve(st, COMPOSITE, "q", {"X-Tandem-Profile": "bench"}, "opencode")
    assert (r.profile, r.mode, dict(r.headers())) == ("bench", "raw", {"X-Tandem-Profile": "bench"})


def test_the_header_the_client_and_the_default_pick_a_profile():
    cond, _ = setup()
    st = cond.store
    r = resolve(st, COMPOSITE, "q", {"X-Tandem-Profile": "overnight"}, "open-webui")
    assert (r.profile, r.source) == ("overnight", "header")
    assert resolve(st, COMPOSITE, "q", {}, "opencode").profile == "coding"
    assert resolve(st, COMPOSITE, "q", {}, "open-webui").profile == "chat"
    r = resolve(st, COMPOSITE, "q", {}, "curl")
    assert (r.profile, r.source) == ("chat", "default")
    try:
        resolve(st, COMPOSITE, "q", {"X-Tandem-Profile": "nope"}, "curl")
        raise AssertionError("an unknown profile is the client's 400")
    except ConfigError as exc:
        assert exc.status == 400 and "nope" in str(exc)


def test_x_tandem_lowers_and_never_raises():
    cond, _ = setup()
    st = cond.store
    assert resolve(st, COMPOSITE, "q", {"X-Tandem": "raw"}, "opencode").mode == "raw"
    assert resolve(st, COMPOSITE, "q", {"X-Tandem": "off"}, "opencode").mode == "raw"
    assert resolve(st, COMPOSITE, "q", {"X-Tandem": "shadow"}, "opencode").mode == "shadow"
    _, doc = st.get("profiles", "coding")
    doc["mode"] = "shadow"
    st.put("profiles", "coding", doc, note="shadow", author="t")
    assert resolve(st, COMPOSITE, "q", {"X-Tandem": "on"}, "opencode").mode == "shadow"
    # the frame's global switch (--conductor) lowers too
    assert resolve(st, COMPOSITE, "q", {}, "open-webui", global_mode="off").mode == "raw"
    assert resolve(st, COMPOSITE, "q", {"X-Tandem-Profile": "bench"}, "curl").mode == "raw"


# ------------------------------------------------------------------ SRV-129

def test_a_code_request_thinks_at_xhigh():
    cond, fakes = setup(decider=by_label("code_edit", 0.91))
    plan, ids, clean = turn(cond, body([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": CODE}]))
    row = plan.row
    assert (row["class_id"], row["source"], row["confidence"]) == ("code_edit", "decider", 0.91)
    assert row["effort"] == "xhigh" and plan.think_budget(0) == 0, "no cap below the backstop"
    note = plan.injected[0]["text"]
    assert schema.XHIGH_NOTE in note, note
    assert "Read a file before you change it" in note, "the class's rules come with it"
    # the note sits in the last turn, right before its <|im_end|> and the generation prompt
    t = text_of(ids)
    assert t.endswith(note + "<|im_end|>\n<|im_start|>assistant\n<think>\n"), t[-400:]
    assert text_of(clean) == t.replace(note, ""), "everything else is the client's prompt"
    assert len(fakes["decider"].calls) == 1 and not fakes["qwen"].calls


def test_a_summary_does_not_think_and_keeps_the_prefix():
    """Thinking off closes the generation prompt's `<think>` at the tail: the prompt equals the
    template's own `enable_thinking: false` render where that render keeps the head (medium), and
    at every effort the head is the thinking-on render's, token for token."""
    for effort in ("medium", "xhigh", "low"):
        cond, _ = setup(decider=by_label("summarize", 0.88), effort=effort)
        msgs = [{"role": "user", "content": "Summarize the text below in five bullet points.\n\n"
                                            + "The engine reads one sequence. " * 20}]
        plan, ids, clean = turn(cond, body(msgs, tools=False))
        assert plan.row["thinking"] == "off" and plan.in_think_after is False, effort
        on = text_of(clean)
        got = text_of(ids)
        assert on.endswith("assistant\n<think>\n")
        assert got == on[:-len("<think>\n")] + "<think>\n\n</think>\n\n", effort
        off = QTok().apply_chat_template(msgs, tokenize=False, reasoning_effort=effort,
                                         enable_thinking=False)
        assert (got == off) == (effort == "medium"), "the template path would change the head"
        # the next turn renders the answer's empty think block the same way: the prefix holds
        nxt = QTok().apply_chat_template(msgs + [{"role": "assistant", "content": "- one"},
                                                 {"role": "user", "content": "thanks"}],
                                         tokenize=False, reasoning_effort=effort)
        assert nxt.startswith(got), effort
        assert plan.think_budget(0) == 0


def test_the_real_template_puts_effort_at_the_head():
    """Why the router never re-renders for effort or thinking: xhigh and low render at the start
    of the system turn, and `enable_thinking: false` drops that sentence, so either change is a
    new prompt from the first tokens."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    x = QTok().apply_chat_template(msgs, tokenize=False, reasoning_effort="xhigh")
    lo = QTok().apply_chat_template(msgs, tokenize=False, reasoning_effort="low")
    xo = QTok().apply_chat_template(msgs, tokenize=False, reasoning_effort="xhigh",
                                    enable_thinking=False)
    for other in (lo, xo):
        first = next(i for i, (a, b) in enumerate(zip(x, other)) if a != b)
        assert first < 60, first


def test_a_design_request_loads_the_design_skill_hidden():
    cond, _ = setup(decider=by_label("design_ui", 0.84))
    plan, ids, clean = turn(cond, body([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": "Make the settings page look "
                                                                    "calmer and more modern."}]))
    note = plan.injected[0]["text"]
    assert "# UI design" in note and plan.injected[0]["parts"][1]["name"] == "ui-design"
    assert "# UI design" in text_of(ids) and "# UI design" not in text_of(clean)
    inj = plan.injected[0]
    assert inj["pos"] == len(clean) - len(QTok().encode("<|im_end|>\n<|im_start|>assistant\n"
                                                        "<think>\n"))
    assert inj["tokens"] == len(QTok().encode(note)) and "design_ui" in inj["reason"]
    assert plan.row["acted"] == 1 and plan.changed


def _session_turns(cond, first_user, answers):
    """An opencode session: the first user message, then tool turns."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first_user}]
    out = [turn(cond, body(list(msgs)))]
    for i, a in enumerate(answers):
        msgs += [{"role": "assistant", "content": "", "reasoning_content": f"step {i}",
                  "tool_calls": [{"id": f"c{i}", "type": "function",
                                  "function": {"name": "bash",
                                               "arguments": json.dumps({"command": a})}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": f"ran {a}"}]
        out.append(turn(cond, body(list(msgs))))
    return msgs, out


def test_a_tool_result_turn_asks_no_model_and_replays_the_note():
    cond, fakes = setup(decider=by_label("code_edit", 0.91))
    _, turns = _session_turns(cond, CODE, ["ls", "cat src/lexer.ts"])
    assert len(fakes["decider"].calls) == 1, "only the user's message is classified"
    (p1, ids1, _), (p2, ids2, _), (p3, ids3, _) = turns
    assert p2.row["source"] == "sticky" and p2.row["class_id"] == "code_edit"
    assert not p2.injected and p2.row["replayed"]["count"] == 1
    # turn 2 starts with turn 1's whole prompt, note included: the resident prefix holds
    assert ids2[:len(ids1)] == ids1, "turn 2's prompt begins with turn 1's"
    assert ids3[:len(ids2)] == ids2
    assert text_of(ids3).count("<runtime-note>") == 1


def test_a_chat_without_a_system_message_keeps_its_session_and_replays_the_note():
    """0.3.14's live check (2026-10-02 22:07): a raw chat with no system message and no session
    id. Its key was the hash of the first two messages -- the user's alone on turn 1, the user's
    and the first answer on turn 2 -- so turn 2 was another session, turn 1's note was not
    replayed, and its prompt parted from the resident rows at the note: the follow-up resumed
    nowhere (`cache_n` 0). The key is the head up to the first user message now."""
    cond, fakes = setup(decider=by_label("code_edit", 0.91))
    hdr = {"User-Agent": "curl/8.5"}                  # no session id (`turn`'s default has one)
    m1 = [{"role": "user", "content": CODE}]
    p1, ids1, _ = turn(cond, body(m1, tools=False), headers=hdr, kind="other")
    assert p1.row["session_key"].startswith("h:"), p1.row["session_key"]
    assert p1.injected, "turn 1 has its note"
    m2 = m1 + [{"role": "assistant", "content": "Done.", "reasoning_content": "Read the lexer."},
               {"role": "user", "content": "Now say it in one sentence."}]
    p2, ids2, clean2 = turn(cond, body(m2, tools=False), headers=hdr, kind="other")
    assert p2.row["session_key"] == p1.row["session_key"], (p1.row["session_key"],
                                                            p2.row["session_key"])
    assert p2.row["replayed"]["count"] == 1, p2.row["replayed"]
    assert ids2[:len(ids1)] == ids1, "turn 2's prompt begins with turn 1's, note included"
    assert ids2 != clean2
    # with a system message the head is the system text and the first user message, as before
    sh = signals.shape(body([{"role": "system", "content": "s"}] + m2, tools=False))
    sh1 = signals.shape(body([{"role": "system", "content": "s"}] + m1, tools=False))
    assert sh.first_hash == sh1.first_hash


def test_a_vague_follow_up_keeps_the_class():
    """The decider is sure of the first message; the follow-up only reaches Qwen's threshold
    (0.55) at 0.65, which is below the session's switch threshold (0.8)."""
    answers = {CODE: ("code_edit", 0.91), "ok, now make it nicer": ("design_ui", 0.65)}
    cond, _ = setup(decider=lambda t, o: answers[t], qwen=lambda t, o: answers[t])
    msgs = [{"role": "user", "content": CODE}]
    turn(cond, body(list(msgs)))
    msgs += [{"role": "assistant", "content": "Done."},
             {"role": "user", "content": "ok, now make it nicer"}]
    plan, _, _ = turn(cond, body(list(msgs)))
    assert (plan.row["class_id"], plan.row["kept"]) == ("code_edit", "below_switch")
    assert [s["label"] for s in plan.row["stages"]] == ["design_ui", "design_ui"]
    assert plan.row["stages"][0].get("below") and plan.row["stages"][1]["stage"] == "qwen"
    # a clear switch at 0.85 changes it
    answers["ok, now make it nicer"] = ("design_ui", 0.85)
    msgs[-1] = {"role": "user", "content": "ok, now make it nicer "}
    answers["ok, now make it nicer "] = ("design_ui", 0.85)
    plan, _, _ = turn(cond, body(list(msgs)))
    assert plan.row["class_id"] == "design_ui" and plan.row["source"] == "decider"


def test_the_decider_down_and_everything_down():
    cond, fakes = setup(decider=lambda t, o: "decider_unreachable",
                        qwen=by_label("debug", 0.8))
    plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]))
    assert (plan.row["class_id"], plan.row["source"]) == ("debug", "qwen")
    assert plan.row["stages"][0]["error"] == "decider_unreachable"
    # both down, a new session: the profile's default class (coding: code_edit)
    cond, fakes = setup()
    plan, _, _ = turn(cond, body([{"role": "user", "content": "Summarize this file."}]),
                      headers={"x-session-id": "s2"})
    assert (plan.row["class_id"], plan.row["kept"]) == ("code_edit", "fail_open")
    # both down in a session that has a class: it keeps it
    cond, fakes = setup(decider=by_label("design_ui", 0.9))
    msgs = [{"role": "user", "content": "Restyle the header."}]
    turn(cond, body(list(msgs)))
    fakes["decider"].fn = lambda t, o: "decider_timeout"
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "And the footer."}]
    plan, _, _ = turn(cond, body(list(msgs)))
    assert (plan.row["class_id"], plan.row["kept"]) == ("design_ui", "fail_open")


def test_no_word_decides_a_class():
    """With every classifier down, a request to summarize and a request to fix a crash get the
    same class: nothing in the router reads what the user wrote."""
    cond, _ = setup()
    got = set()
    for i, text in enumerate(("Summarize this article for me please.",
                              "Fix the crash in main.py, it throws a KeyError.",
                              "Draw a spiral portrait of this photo.", "hi!")):
        plan, _, _ = turn(cond, body([{"role": "user", "content": text}]),
                          headers={"x-session-id": f"s{i}"})
        got.add(plan.row["class_id"])
    assert got == {"code_edit"}, got


def test_shadow_changes_nothing():
    cond, _ = setup(decider=by_label("design_ui", 0.9))
    _, doc = cond.store.get("profiles", "coding")
    doc["decisions"].update(effort="shadow", packages="shadow", rules="shadow")
    cond.store.put("profiles", "coding", doc, note="shadow", author="t")
    plan, ids, clean = turn(cond, body([{"role": "user", "content": "Restyle the page."}]))
    assert ids == clean and not plan.changed and plan.think_budget(0) == 0
    sh = plan.row["shadow"]
    assert sh["effort"]["effort"] == "high"
    assert {n["name"] for n in sh["notes"]} == {"high", "ui-design"}
    # classify in shadow: the class is journaled, the policy is the default class's
    doc["decisions"].update(classify="shadow", effort="on")
    cond.store.put("profiles", "coding", doc, note="classify shadow", author="t")
    plan, ids, clean = turn(cond, body([{"role": "user", "content": "Restyle the page too."}]),
                            headers={"x-session-id": "s9"})
    assert plan.row["class_id"] == "code_edit" and plan.row["shadow"]["class_id"] == "design_ui"


def test_the_profile_mode_shadow_lowers_every_decision():
    cond, _ = setup(decider=by_label("code_edit", 0.95))
    plan, ids, clean = turn(cond, body([{"role": "user", "content": CODE}]),
                            headers={"x-session-id": "s1", "X-Tandem": "shadow"})
    assert plan.res.mode == "shadow" and set(plan.decisions.values()) <= {"off", "shadow"}
    assert ids == clean and dict(plan.headers())["X-Tandem-Profile"] == "coding"


def test_a_pasted_document_from_another_client_is_not_a_compaction():
    """SRV-175 (UT1, 2026-10-06, chatcmpl-6f4facab1bf446f28932fc6a): a fresh chat with a long
    pasted document and "summarize", no system message, no tools, from a client that is not
    opencode, matched opencode's bare compaction shape and was journaled `compaction: served
    client, why shape`, thinking off, with the line "the conversation was compacted". It is a
    plain turn now, its class decided by the decider; opencode's shape keeps its path."""
    doc = [{"role": "user", "content": "Summarize the key idea of each in three bullets.\n\n"
                                       + "Doc one. " * 1200}]
    for kind in ("other", "open-webui", "hermes"):
        cond, fakes = setup(decider=by_label("summarize", 0.9))
        plan, _, _ = turn(cond, body(doc, tools=False), headers={"x-session-id": f"p-{kind}"},
                          kind=kind)
        assert not plan.compaction, kind
        assert plan.row["source"] != "structural", (kind, plan.row["source"])
        assert "compaction" not in plan.row.get("extra", {}), (kind, plan.row["extra"])
        assert len(fakes["decider"].calls) == 1, "the decider was asked, as for any new message"
    cond, _ = setup(decider=by_label("design_ui", 0.9))
    plan, _, _ = turn(cond, body(doc, tools=False), headers={"x-session-id": "p-oc"})
    assert plan.compaction and (plan.row["class_id"], plan.row["source"]) == ("summarize",
                                                                              "structural")
    return "3 other clients: a plain turn the decider classes; opencode: the compaction path"


def test_a_compaction_drops_the_note_and_the_block_returns():
    cond, fakes = setup(decider=by_label("design_ui", 0.9))
    msgs, turns = _session_turns(cond, "Restyle the dashboard cards.", ["ls"])
    assert turns[1][0].row["replayed"]["count"] == 1
    compact = [{"role": "user", "content": "Summarize our conversation so far. " + "x " * 1500}]
    plan, ids, clean = turn(cond, body(compact, tools=False))
    assert (plan.row["class_id"], plan.row["source"]) == ("summarize", "structural")
    assert plan.row["thinking"] == "off" and not plan.row.get("replayed")
    # after it, opencode goes on with a shorter history: the old note's position is gone (the
    # prompt parts before it), the session is still design_ui, and its block comes back once
    after = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Summary so far: cards restyled halfway. Continue."},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "c9", "type": "function", "function": {"name": "bash",
                                                               "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c9", "content": "ok"}]
    n_calls = len(fakes["decider"].calls)
    plan, ids, clean = turn(cond, body(after))
    assert plan.row["class_id"] == "design_ui" and plan.row["source"] == "sticky"
    assert plan.row["replayed"]["count"] == 0, "the old note did not survive the compaction"
    assert len(plan.injected) == 1 and "# UI design" in plan.injected[0]["text"]
    assert text_of(ids).count("# UI design") == 1
    assert len(fakes["decider"].calls) == n_calls, "a tool turn asks no model"
    # and the turn after that replays the new note instead of adding another
    after += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c10", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c10", "content": "ok again"}]
    plan2, ids2, _ = turn(cond, body(after))
    assert not plan2.injected and plan2.row["replayed"]["count"] == 1
    assert ids2[:len(ids)] == ids


def test_a_client_budget_wins():
    cond, _ = setup(decider=by_label("chat", 0.9))
    plan, _, _ = turn(cond, body([{"role": "user", "content": "hello"}], tools=False,
                                 max_reasoning_tokens=4096))
    assert plan.row["budget_source"] == "client" and plan.think_budget(4096) == 4096
    plan, _, _ = turn(cond, body([{"role": "user", "content": "hello again"}], tools=False),
                      headers={"x-session-id": "s3"})
    assert plan.row["effort"] == "low" and plan.think_budget(0) == 1024


def test_a_failing_checker_raises_the_effort_one_level():
    cond, _ = setup(decider=by_label("design_ui", 0.9))
    msgs, _ = _session_turns(cond, "Restyle the cards.", [])
    msgs += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content":
            "pagodacheck  /demo\nRESULT: FAIL  <- (a) summer island colour differs by 3.5 > 1.0"}]
    plan, _, _ = turn(cond, body(list(msgs)))
    assert plan.row["effort"] == "xhigh", "high + a FAIL = xhigh for this turn"


def test_the_checker_result_labels_the_previous_turn():
    cond, _ = setup(decider=by_label("code_edit", 0.9))
    msgs, turns = _session_turns(cond, CODE, [])
    first = turns[0][0].row["id"]
    msgs += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content":
            "sketchcheck  a.png   (style: spiral)\nRESULT: PASS"}]
    turn(cond, body(list(msgs)))
    row = cond.journal.get(first)
    assert row["next_checker"][0]["verdict"] == "PASS"


def test_a_loaded_skill_is_not_injected_again_and_packages_get_a_second_question():
    cond, fakes = setup(decider=lambda t, o: ("spiral-portrait", 0.8) if "none" in o
                        else ("drawing", 0.9))
    plan, ids, _ = turn(cond, body([{"role": "user", "content": "Draw her as one spiral line."}]))
    ch = plan.row["extra"]["package_choice"]
    assert ch["candidates"] == ["pencil-sketch", "spiral-portrait", "one-line-portrait"]
    assert ch["chosen"] == "spiral-portrait" and ch["how"] == "decider"
    assert set(fakes["decider"].calls[1][1]) == set(ch["candidates"]) | {"none"}
    # the skill file is not on this machine's test path: journaled, not injected
    notes = plan.row["extra"].get("notes") or []
    assert any(n.startswith(("package_file_missing", "package_")) for n in notes) or \
        any(p["name"] == "spiral-portrait" for p in plan.injected[0]["parts"])
    # the client loaded it itself: no second copy, no question
    cond, fakes = setup(decider=by_label("drawing", 0.9))
    msgs = [{"role": "user", "content": "Draw her as one spiral line."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "k", "type": "function",
                 "function": {"name": "skill", "arguments": {"name": "spiral-portrait"}}}]},
            {"role": "tool", "tool_call_id": "k", "content": "loaded"},
            {"role": "user", "content": "Now draw the second one."}]
    plan, _, _ = turn(cond, body(msgs), headers={"x-session-id": "s7"})
    assert "already_loaded:spiral-portrait" in plan.row["extra"]["notes"]
    assert "spiral-portrait" not in plan.row["extra"]["package_choice"]["candidates"]


def test_a_client_pins_and_blocks_packages():
    """SRV-361: `tandem.packages` in the body. A pinned package goes in whatever the class picks,
    a blocked one never does (not even as the class's pick), and a body without the field plans
    as before."""
    def rig():
        cond, fakes = setup(decider=lambda t, o: ("tiny-skill", 0.9) if "none" in o
                            else ("three_d", 0.9))
        _, p = cond.store.get("packages")
        for n in ("tiny-skill", "other-skill", "pinned-skill"):
            p["packages"][n] = {"title": n, "kind": "skill", "source": "inline",
                                "description": f"The {n} recipe for a scene.",
                                "body": f"# {n} body", "max_tokens": 3000, "enabled": True}
        cond.store.put("packages", None, p, note="pins", author="t")
        _, r = cond.store.get("router")
        for c in r["classes"]:
            if c["id"] == "three_d":
                c["policy"]["packages"] = ["tiny-skill", "other-skill"]
        cond.store.put("router", None, r, note="pins", author="t")
        return cond, fakes
    msg = [{"role": "user", "content": "Build a voxel island in 3D."}]
    cond, _ = rig()
    plan, _, _ = turn(cond, body(msg))
    names = [x["name"] for x in plan.injected[0]["parts"] if x["kind"] == "package"]
    assert names == ["tiny-skill"] and "package_pinned" not in plan.row["extra"]
    cond, _ = rig()
    plan, _, _ = turn(cond, dict(body(msg), tandem={"packages": {"pin": ["pinned-skill", "gone"]}}))
    names = [x["name"] for x in plan.injected[0]["parts"] if x["kind"] == "package"]
    assert names == ["tiny-skill", "pinned-skill"], names
    assert plan.row["extra"]["package_pinned"] == ["pinned-skill"]
    assert "pin_unknown:gone" in plan.row["extra"]["notes"]
    cond, _ = rig()
    plan, _, _ = turn(cond, dict(body(msg), tandem={"packages": {"block": ["tiny-skill"]}}))
    names = [x["name"] for x in (plan.injected[0]["parts"] if plan.injected else [])
             if x["kind"] == "package"]
    assert "tiny-skill" not in names and plan.row["extra"]["package_blocked"] == ["tiny-skill"]


def test_a_general_package_goes_in_first_and_the_skill_is_picked_among_the_rest():
    """2026-10-02: a package marked `general` (the coding package) goes in on every turn of a
    class that lists it, before the task skill; the second question picks among the skills
    alone. A class whose only package is general asks no second question. A stored packages
    document without the flag (every one before it) is read as before."""
    cond, fakes = setup(decider=lambda t, o: ("tiny-skill", 0.9) if "none" in o
                        else ("three_d", 0.9))
    _, p = cond.store.get("packages")
    p["packages"]["coding"] = {"title": "Coding", "kind": "package", "source": "inline",
                               "description": "How programming work must happen.",
                               "body": "# Coding core\n\nRun after every step.",
                               "max_tokens": 3000, "enabled": True, "general": True}
    for n in ("tiny-skill", "other-skill"):
        p["packages"][n] = {"title": n, "kind": "skill", "source": "inline",
                            "description": f"The {n} recipe for a scene.",
                            "body": f"# {n} body", "max_tokens": 3000, "enabled": True}
    cond.store.put("packages", None, p, note="coding general", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "three_d":
            c["policy"]["packages"] = ["coding", "tiny-skill", "other-skill"]
        if c["id"] == "code_edit":
            c["policy"]["packages"] = ["coding"]
    cond.store.put("router", None, r, note="coding on three_d", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": "Build a voxel island in 3D."}]))
    text = plan.injected[0]["text"]
    assert [x["name"] for x in plan.injected[0]["parts"] if x["kind"] == "package"] == \
        ["coding", "tiny-skill"], plan.injected[0]["parts"]
    assert text.index("# Coding core") < text.index("# tiny-skill body")
    ch = plan.row["extra"]["package_choice"]
    assert ch["candidates"] == ["tiny-skill", "other-skill"] and ch["chosen"] == "tiny-skill"
    assert "coding" not in fakes["decider"].calls[1][1], "the general one is not in the choice"
    assert plan.row["extra"]["package_general"] == ["coding"]
    cond, fakes = setup(decider=by_label("code_edit", 0.9))
    cond.store.put("packages", None, p, note="coding general", author="t")
    cond.store.put("router", None, r, note="coding on code_edit", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]))
    assert "# Coding core" in plan.injected[0]["text"] and len(fakes["decider"].calls) == 1
    assert "package_choice" not in plan.row["extra"]


def test_a_new_class_from_the_config_is_used_next():
    cond, fakes = setup(decider=lambda t, o: ("data_analysis", 0.9) if "data_analysis" in o
                        else ("chat", 0.9))
    _, r = cond.store.get("router")
    r["classes"].append({"id": "data_analysis", "label": "Data analysis",
                         "description": "Analyse a table or a CSV file and report numbers.",
                         "policy": {"effort": "high", "packages": [], "rules": ["Show the "
                                                                               "query."]}})
    v = cond.store.put("router", None, r, note="add data_analysis", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": "What is the mean of col b?"}]))
    assert plan.row["class_id"] == "data_analysis" and plan.row["router_version"] == v
    assert "- Show the query." in plan.injected[0]["text"]


def test_the_test_box_explains():
    cond, _ = setup(decider=lambda t, o: "decider_timeout", qwen=by_label("design_ui", 0.7))
    out = cond.router.explain("Make the login page prettier.", "coding")
    assert out["class_id"] == "design_ui" and out["source"] == "qwen"
    assert [s["stage"] for s in out["stages"]] == ["decider", "qwen"]
    assert out["note"].startswith("\n\n<runtime-note>") and "# UI design" in out["note"]
    assert out["effort"] == "high" and out["profile"] == "coding"
    assert not cond.router.sessions_live(), "the test box keeps no session"


def test_the_monitor_hand_over():
    """The frame's monitor runs under the profile: never above its mode (the raw name runs it
    raw), with its task modes, caps and the thresholds that map onto the policy file."""
    from server.conductor.monitor import policy as mpol
    base = mpol.load()
    cond, _ = setup(decider=by_label("code_edit", 0.9))
    plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]),
                      headers={"x-session-id": "s1", "X-Tandem": "shadow"})
    assert cond.monitor_mode(plan, "on") == "shadow"
    raw, _, _ = turn(cond, dict(body([{"role": "user", "content": CODE}]),
                                model="qwen38-spark-engine"),
                     headers={"x-session-id": "s1", "X-Tandem": "raw"})
    assert cond.monitor_mode(raw, "on") == "raw" and cond.monitor_policy(raw, base) is None
    alias, _, _ = turn(cond, dict(body([{"role": "user", "content": CODE}]),
                                  model="qwen38-spark-engine"), headers={"x-session-id": "s3"})
    assert cond.monitor_mode(alias, "on") == "on", "the alias runs the monitor as TandemLLM"
    assert cond.monitor_mode(None, "on") == "on", "without the config the frame is unchanged"
    _, doc = cond.store.get("profiles", "coding")
    doc["tasks"]["D1"]["mode"] = "shadow"
    doc["tasks"]["D1"]["thresholds"]["repeat_share"] = 0.75
    doc["tasks"]["D2"]["thresholds"]["backstop"] = 24000
    doc["caps"]["acts_per_request"] = 1
    cond.store.put("profiles", "coding", doc, note="tune", author="t")
    plan, _, _ = turn(cond, body([{"role": "user", "content": CODE}]),
                      headers={"x-session-id": "s2"})
    pol = cond.monitor_policy(plan, base)
    assert pol.tasks["D1"].mode == "shadow" and pol.data["tasks"]["D1"]["arms"]["repeat"]["act"] == 0.75
    assert pol.data["tasks"]["D2"]["cap"] == 24000 and pol.caps["acts_per_request"] == 1
    assert base.tasks["D1"].mode == "act", "the server's policy is untouched"
    assert cond.monitor_policy(plan, base) is pol, "cached per profile version"


def test_an_in_engine_decider_needs_no_conductor_change():
    """The decider is an interface: a backend the engine registers (`engine:<name>`) answers the
    same Jev questions as the :8010 sidecar, and switching is one role endpoint in the config."""
    from server.conductor import classify

    class InEngine(classify.Decider):
        def __init__(self, model, timeout_ms):
            self.model = model

        def limits(self):
            return {"context_tokens": 256, "question_tokens": 96, "max_options": 20}

        def ask(self, state, questions):
            q = questions["q"]
            assert q["type"] == "choice" and len(state) <= int((256 - 96) * 3.5)
            return {"model": f"in-engine:{self.model}", "answers": {"q": {
                "type": "choice", "choice": "debug", "probabilities": {"debug": 0.93},
                "confidence": 0.93}}}

    classify.register_backend("test", InEngine)
    try:
        app.STATE.clear()
        app.STATE.update(tok=QTok(), device="cpu", reasoning_effort="medium", segments=None,
                         model="qwen38-spark-engine")
        cond = Conductor(tempfile.mkdtemp(prefix="cond-"), state=app.STATE, journal=False)
        _, roles = cond.store.get("roles")
        roles["decider"].update(endpoint="engine:test", model="laya-next")
        roles["classifier"]["mode"] = "off"
        cond.store.put("roles", None, roles, note="the decider in the engine", author="t")
        plan, _, _ = turn(cond, body([{"role": "user", "content": "It crashes. " * 200}]))
        assert (plan.row["class_id"], plan.row["source"]) == ("debug", "decider")
        assert plan.row["stages"][0]["model"] == "in-engine:laya-next"
        assert cond.role_test("decider")["ok"]
    finally:
        classify.BACKENDS.pop("test", None)


def test_signals_shape():
    s = signals.shape(body([{"role": "system", "content": "s"},
                            {"role": "user", "content": "do it"},
                            {"role": "assistant", "content": "RESULT: FAIL <- quoted, not run"},
                            {"role": "tool", "content": "sketchcheck x.png\nRESULT: PASS"}]))
    assert s.tool_turn and s.user_text == "do it" and s.tools and not s.compaction
    assert [r["verdict"] for r in s.results] == ["PASS"], "assistant text is never a result"


def test_stats_count_only_the_stages_that_were_asked():
    """A stage answered from the text cache and a stage whose role is off are not calls: with the
    decider off, the classifier figures are Qwen's (2026-10-01, the tile read 0 / 0 ms)."""
    from server.conductor import decisions
    now = 1_790_000_000_000
    row = lambda stages: {"ts_ms": now, "class_id": "code_edit", "source": "qwen", "stages": stages}
    rows = [row([{"stage": "decider", "error": "decider_role_off", "ms": 0.0},
                 {"stage": "qwen", "label": "code_edit", "ms": 550.0}]),
            row([{"stage": "decider", "error": "decider_role_off", "ms": 0.0, "cached": True},
                 {"stage": "qwen", "label": "code_edit", "ms": 550.0, "cached": True}]),
            row([{"stage": "decider", "error": "decider_unreachable", "ms": 1.0},
                 {"stage": "qwen", "label": "debug", "ms": 600.0}])]
    st = decisions.stats(rows, now_ms=now)
    assert st["classifier"]["qwen"]["calls"] == 2 and st["classifier"]["decider"]["calls"] == 1
    assert st["failures"] == {"decider_unreachable": 1}, st["failures"]


def test_two_chats_that_open_the_same_way_are_two_sessions():
    """SRV-225 (UX2 run): Open WebUI sends no chat id, so the session key is the hash
    of the head up to the first user message, and every chat that opened with the same sentence
    was one session (A01's new chat: turns 27, 21,216 session tokens, four notes replayed at its
    first message). A conversation's very first turn now starts its own session, and its later
    turns find it again by their first turn; the rows say how (`extra.session`)."""
    cond, fakes = setup(decider=by_label("code_edit", 0.91))
    hdr = {"User-Agent": "Open WebUI"}
    first = [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": CODE}]

    def go(msgs):
        return turn(cond, body(msgs, tools=False), headers=hdr, kind="open-webui")[0]

    a1 = go(first)
    ka = a1.row["session_key"]
    assert ka.startswith("h:") and a1.session.turns == 1 and a1.injected
    assert a1.row["extra"]["session"] == "new"
    # the first turn's tool rounds and its next turns stay in that session
    a_tool = first + [{"role": "assistant", "content": "",
                       "tool_calls": [{"id": "1", "type": "function", "function": {
                           "name": "read", "arguments": "{\"path\": \"/a\"}"}}]},
                      {"role": "tool", "tool_call_id": "1", "content": "x = 1"}]
    a1b = go(a_tool)
    assert a1b.row["session_key"] == ka and a1b.row["extra"]["session"] == "continued"
    a2_msgs = a_tool + [{"role": "assistant", "content": "Done A."},
                        {"role": "user", "content": "Now add tests."}]
    a2 = go(a2_msgs)
    assert a2.row["session_key"] == ka and a2.row["replayed"]["count"] >= 1
    turns_a = a2.session.turns
    # a new chat with the same opening: its own session, turn 1, nothing replayed
    b1 = go(first)
    kb = b1.row["session_key"]
    assert kb != ka and kb.startswith(ka + "#"), (ka, kb)
    assert b1.session.turns == 1 and b1.injected and not (b1.row["replayed"] or {}).get("count")
    assert b1.row["extra"]["session"] == "new"
    # both go on in their own sessions, also interleaved
    b2 = go(first + [{"role": "assistant", "content": "Done B, another answer."},
                     {"role": "user", "content": "Now add tests."}])
    assert b2.row["session_key"] == kb
    a3 = go(a2_msgs + [{"role": "assistant", "content": "Tests added."},
                       {"role": "user", "content": "Run them."}])
    assert a3.row["session_key"] == ka and a3.session.turns == turns_a + 1
    # a client that sends its chat id gets a `c:` key, whatever the opening
    c1 = turn(cond, body(first, tools=False), headers={"X-OpenWebUI-Chat-Id": "chat-1"},
              kind="open-webui")[0]
    c2 = turn(cond, body(first, tools=False), headers={"X-OpenWebUI-Chat-Id": "chat-2"},
              kind="open-webui")[0]
    assert (c1.row["session_key"], c2.row["session_key"]) == ("c:chat-1", "c:chat-2")
    assert c1.row["extra"]["session"] == "id"


def test_without_generations_the_second_chat_joins_the_first():
    """The A side of the A/B (`QWEN38_SESSION_GENERATIONS=off`): the bug."""
    from server.conductor import router as R
    was = R.SESSION_GENERATIONS
    R.SESSION_GENERATIONS = "off"
    try:
        cond, _ = setup(decider=by_label("code_edit", 0.91))
        hdr = {"User-Agent": "Open WebUI"}
        first = [{"role": "system", "content": "You are helpful."},
                 {"role": "user", "content": CODE}]
        a = turn(cond, body(first, tools=False), headers=hdr, kind="open-webui")[0]
        b = turn(cond, body(first, tools=False), headers=hdr, kind="open-webui")[0]
        assert a.row["session_key"] == b.row["session_key"] and b.session.turns == 2
    finally:
        R.SESSION_GENERATIONS = was


def test_session_shape_counts_people_and_hashes_the_first_turn():
    m1 = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    sh = signals.shape(body(m1, tools=False))
    assert (sh.n_user, sh.fresh, sh.turn1_hash) == (1, True, None)
    m2 = m1 + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    sh2 = signals.shape(body(m2, tools=False))
    assert sh2.n_user == 2 and not sh2.fresh and sh2.turn1_hash
    m3 = m2 + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
    assert signals.shape(body(m3, tools=False)).turn1_hash == sh2.turn1_hash
    other = m1 + [{"role": "assistant", "content": "hello there"},
                  {"role": "user", "content": "more"}]
    assert signals.shape(body(other, tools=False)).turn1_hash != sh2.turn1_hash
    # a tool response in a user message is not a person's message
    tr = m1 + [{"role": "assistant", "content": "", "tool_calls": []},
               {"role": "user", "content": "<tool_response>ok</tool_response>"}]
    assert signals.shape(body(tr, tools=False)).n_user == 1


# ------------------------------------------------------------------ SPD-249: qwen_busy

def _busy_setup(eta):
    cond, fakes = setup(decider=lambda t, o: "decider_unreachable", qwen=by_label("debug", 0.8))
    fakes["qwen"].local = True                       # the server's own port: its queue is known
    asked = []

    def hook(texts, max_new):
        asked.append((tuple(texts), max_new))
        return eta
    cond.router.own_eta = hook
    return cond, fakes, asked


def test_qwen_busy_skips_the_stage_at_once_and_the_cascade_goes_on():
    """SPD-249: a burst of chats queued the router's questions behind slot 0 and each waited out
    its 8 s timeout; when the server says the answer cannot come in time, the stage is skipped
    at once (`qwen_busy`, journaled with the estimate) and the cascade goes on: here to the
    profile's default class, as a timeout ends."""
    cond, fakes, asked = _busy_setup(20_000.0)
    plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]))
    st = [s for s in plan.row["stages"] if s["stage"] == "qwen"]
    assert st and st[0]["error"] == "qwen_busy" and st[0]["skipped"], plan.row["stages"]
    assert st[0]["eta_ms"] == 20_000.0 and st[0]["timeout_ms"] > 0, st[0]
    assert not fakes["qwen"].calls, "a busy stage is never asked"
    assert (plan.row["class_id"], plan.row["kept"]) == ("code_edit", "fail_open")
    assert asked and len(asked[0][0]) == 2 and asked[0][1] == 24, asked
    assert "KeyError" in asked[0][0][1], "the hook counts the message the stage would send"
    # not before the slot's turn ends (a client's turn holds slot 0): the same, no estimate
    cond, fakes, _ = _busy_setup(float("inf"))
    plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail."}]))
    st = [s for s in plan.row["stages"] if s["stage"] == "qwen"]
    assert st[0]["error"] == "qwen_busy" and st[0]["eta_ms"] is None and not fakes["qwen"].calls
    json.dumps(plan.row["stages"])                   # the journal row stays JSON


def test_qwen_free_or_unknown_or_remote_is_asked():
    """SPD-249: an answer that comes in time, no estimate (no prefill rate yet), or a stage on
    another endpoint (whose queue the server cannot see): Qwen is asked as before."""
    for eta in (150.0, None):
        cond, fakes, asked = _busy_setup(eta)
        plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]))
        assert (plan.row["class_id"], plan.row["source"]) == ("debug", "qwen"), (eta, plan.row)
        assert len(fakes["qwen"].calls) == 1 and asked
    cond, fakes, asked = _busy_setup(20_000.0)
    fakes["qwen"].local = False
    plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]))
    assert plan.row["source"] == "qwen" and not asked, "a remote stage never reads this queue"


def test_qwen_busy_is_a_failure_not_a_call():
    from server.conductor import decisions
    now = 1_790_000_000_000
    rows = [{"ts_ms": now, "class_id": "code_edit", "source": "default",
             "stages": [{"stage": "qwen", "error": "qwen_busy", "ms": 0.0, "skipped": True}]},
            {"ts_ms": now, "class_id": "debug", "source": "qwen",
             "stages": [{"stage": "qwen", "label": "debug", "ms": 600.0}]}]
    st = decisions.stats(rows, now_ms=now)
    assert st["classifier"]["qwen"]["calls"] == 1, st["classifier"]
    assert st["failures"] == {"qwen_busy": 1}, st["failures"]


def test_the_own_eta_estimator():
    """SPD-249 (`queue_wait.own_eta_ms`): a free slot is the prefill alone; behind slot 0 the
    holder's time and the rows ahead come first; a client's turn on slot 0 is never in time;
    no rate, no estimate."""
    from server import queue_wait as qw
    assert qw.own_eta_ms(1000, None, free=True) is None
    assert qw.own_eta_ms(1000, 500.0, free=True) == 2000.0
    assert qw.own_eta_ms(1000, 500.0, free=False, ahead_rows=3000, holder_ms=0.0) == 8000.0
    assert qw.own_eta_ms(1000, 500.0, free=False, ahead_rows=0, holder_ms=1500.0) == 3500.0
    assert qw.own_eta_ms(1000, 500.0, free=False, holder_ms=-40.0) == 2000.0
    assert qw.own_eta_ms(1000, 500.0, free=False, holder_ms=None) == float("inf")
    # the shared GPU turn: before each of its turns every other holder runs one unit
    assert qw.turn_wait_ms(3, [200.0, 6000.0]) == 18_600.0 and qw.turn_wait_ms(3, []) == 0.0
    assert qw.own_eta_ms(1000, 500.0, free=True, turn_ms=600.0) == 2600.0
    assert qw.own_eta_ms(1000, 500.0, free=False, holder_ms=100.0, turn_ms=600.0) == 2700.0


def test_the_servers_own_eta_reads_its_queue():
    """SPD-249 (`app._own_eta_ms`, the router's hook): slot 0 free; slot 0 held with the side
    lane's jobs pending ahead; held by another own question (its prefill left, the own waiters);
    held by a client's turn for a question too long for the lane; held by no request."""
    import time as _t
    from server import preempt
    setup()
    app.cache_peek.SOLO_RATES._q.clear()          # SRV-763: read before the peek's rate
    real_rate = app._peek_rate
    app._peek_rate = lambda: 1000.0                  # one row a ms
    lane = app.STATE["lane"] = preempt.Lane(lambda job: None, prompt_max=2048, new_max=64)
    rows = lambda texts: app.OWN_TEMPLATE_ROWS + sum(len(t) for t in texts)   # QTok: a char a token
    short, long_ = ("pick one", "x" * 100), ("pick one", "y" * 3000)
    try:
        assert app._own_eta_ms(short) == float(rows(short))
        assert app.LOCK.acquire(timeout=5)
        try:
            app.HOLDER["who"] = {"own": False, "rank": app.RANK_TURN, "t_hold": 0.0}
            lane.submit(preempt.Job([1] * 500, 24, set()))
            lane.submit(preempt.Job([1] * 700, 24, set()))
            assert app._own_eta_ms(short) == float(1200 + rows(short))
            assert app._own_eta_ms(long_) == float("inf"), "too long for the lane: the turn's end"
            app.HOLDER["who"] = {"own": True, "rows": 4000, "t_hold": _t.monotonic() - 1.0}
            me = {"own": True, "rows": 2500, "rank": app.RANK_TURN, "t": 0.0}
            with app.QUEUE:
                app.WAITERS.append(me)
            try:
                got = app._own_eta_ms(long_)
                want = 3000.0 + 2500 + rows(long_)            # 4 s of prefill, 1 s done
                assert abs(got - want) < 50.0, (got, want)
            finally:
                with app.QUEUE:
                    app.WAITERS.remove(me)
            app.HOLDER["who"] = None
            assert app._own_eta_ms(long_) is None, "a lock no request took: no basis"
        finally:
            app.HOLDER["who"] = None
            app.LOCK.release()
        app._peek_rate = lambda: None
        assert app._own_eta_ms(short) is None
    finally:
        app._peek_rate = real_rate
        app.STATE["lane"] = None


def test_a_free_side_slot_beside_a_background_prefill_is_no_immediate_start():
    """SPD-249, live 2026-10-29: slot 0 ran a background prepare/memory prefill, a side slot was free,
    and a 449-token question took 17.9 s against the 8 s timeout: the turn is shared, so each of the
    question's turns (its prefill chunk, its answer blocks) waits a unit of slot 0's prefill. Now its
    estimate charges that, and the router skips it as qwen_busy; beside the round runner's lanes alone
    it charges a round a turn and is asked."""
    from types import SimpleNamespace as NS
    from server import preempt
    from server import round_runner as RR
    from server import slots as S
    cond, fakes = setup(decider=lambda t, o: "decider_unreachable", qwen=by_label("debug", 0.8))
    fakes["qwen"].local = True
    cond.router.own_eta = app._own_eta_ms
    app.cache_peek.SOLO_RATES._q.clear()          # SRV-763: read before the peek's rate
    real_rate, real_rr = app._peek_rate, RR.RUNNER
    app._peek_rate = lambda: 600.0
    app.STATE["prefix_chunk"] = 4096
    app.STATE["lane"] = preempt.Lane(lambda job: None, prompt_max=2048, new_max=64)
    pool = S.Pool([S.Slot(0, 65_536), S.Slot(1, 65_536), S.Slot(2, 65_536)])
    app.STATE["pool"] = pool
    RR.RUNNER = None
    assert app.LOCK.acquire(timeout=5)
    try:
        app.HOLDER["who"] = {"own": False, "rank": app.RANK_PREP, "t_hold": 0.0}    # slot 0's own job
        texts = ("Pick one class id.", "x" * 415)
        eta = app._own_eta_ms(texts)
        rows = app.OWN_TEMPLATE_ROWS + sum(len(t) for t in texts)
        chunk_ms = 1e3 * 4096 / 600.0
        want = 1e3 * rows / 600.0 + (1 + app.OWN_ANSWER_TURNS) * chunk_ms
        assert abs(eta - want) < 1.0 and eta > 8000, (eta, want)
        plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]))
        st = [x for x in plan.row["stages"] if x["stage"] == "qwen"]
        assert st and st[0]["error"] == "qwen_busy" and not fakes["qwen"].calls, plan.row["stages"]
        # the runner's lanes on slots 1 and 2 instead, slot 0 free: a round a turn, asked in time
        app.HOLDER["who"] = None
        app.LOCK.release()
        RR.RUNNER = NS(stats={"rounds": 10}, ms={"round": 1750.0})                 # 175 ms a round
        for i in (1, 2):
            pool.slots[i].lock.acquire()
            pool.slots[i].phase = "runner"
        eta = app._own_eta_ms(texts)
        assert abs(eta - (1e3 * rows / 600.0 + (1 + app.OWN_ANSWER_TURNS) * 175.0)) < 1.0, eta
        plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]),
                          headers={"x-session-id": "ses-2"})
        assert plan.row["source"] == "qwen" and len(fakes["qwen"].calls) == 1, plan.row["stages"]
        for i in (1, 2):
            pool.slots[i].phase = ""
            pool.slots[i].lock.release()
        assert app._own_eta_ms(texts) == 1e3 * rows / 600.0, "nothing else runs: the prefill alone"
    finally:
        if app.LOCK.locked():
            app.HOLDER["who"] = None
            app.LOCK.release()
        app._peek_rate, RR.RUNNER = real_rate, real_rr
        app.STATE["lane"] = None
        app.STATE.pop("pool", None)


def test_a_burst_in_the_router_skips_the_questions_that_cannot_fit():
    """SPD-249, live 2026-10-29: 8 chats at once, the box idle when each asked, and 16 of 32 router
    questions ran out their 8 s, because the other chats' questions and prefills came right after.
    The other requests inside the router and the clients waiting for a slot now count: each shares
    the turn with the question for at least its rows. A short class question still fits a burst of 8;
    the long package question does not; alone, both are asked."""
    from server import preempt
    from server import slots as S
    cond, fakes = setup(decider=lambda t, o: "decider_unreachable", qwen=by_label("debug", 0.8))
    fakes["qwen"].local = True
    cond.router.own_eta = app._own_eta_ms
    app.cache_peek.SOLO_RATES._q.clear()          # SRV-763: read before the peek's rate
    real_rate = app._peek_rate
    app._peek_rate = lambda: 600.0
    app.STATE["prefix_chunk"] = 4096
    app.STATE["lane"] = preempt.Lane(lambda job: None, prompt_max=2048, new_max=64)
    app.STATE["pool"] = S.Pool([S.Slot(0, 65_536)] + [S.Slot(i, 65_536) for i in range(1, 8)])
    short, long_ = ("Pick one class id.", "x" * 415), ("Pick the packages.", "y" * 2330)
    rows = lambda texts: app.OWN_TEMPLATE_ROWS + sum(len(t) for t in texts)     # QTok: a char a token
    me = [{"own": False, "rank": app.RANK_TURN, "t": 0.0} for _ in range(2)]
    try:
        assert app._burst_others() == 0
        alone = app._own_eta_ms(long_)
        assert abs(alone - 1e3 * rows(long_) / 600.0) < 1.0 and alone < 8000, alone
        with app._planning():
            with app._planning():
                assert app.PLANNING["n"] == 2 and app._burst_others() == 1
        assert app.PLANNING["n"] == 0
        app.PLANNING["n"] = 8                                      # a burst of 8 chats in the router
        e_short, e_long = app._own_eta_ms(short), app._own_eta_ms(long_)
        assert abs(e_short - 8e3 * rows(short) / 600.0) < 1.0 and e_short < 8000, e_short
        assert abs(e_long - 8e3 * rows(long_) / 600.0) < 1.0 and e_long > 8000, e_long
        with app.QUEUE:
            app.WAITERS.extend(me)                                 # two clients waiting for a slot
        assert app._burst_others() == 9
        assert abs(app._own_eta_ms(short) - 10e3 * rows(short) / 600.0) < 1.0
        with app.QUEUE:
            for w in me:
                app.WAITERS.remove(w)
        # through the router, in the same burst: the question as this router asks it (one char a token
        # here, so its rows are its characters) fits at a fast prefill and is skipped at a slow one
        app._peek_rate = lambda: 60_000.0
        plan, _, _ = turn(cond, body([{"role": "user", "content": "The tests fail with a KeyError."}]))
        st = [x for x in plan.row["stages"] if x["stage"] == "qwen"]
        assert st[0].get("error") is None and len(fakes["qwen"].calls) == 1, st
        app._peek_rate = lambda: 600.0
        plan, _, _ = turn(cond, body([{"role": "user", "content": "The build breaks on a missing import."}]),
                          headers={"x-session-id": "ses-2"})        # another text: no cached answer
        st = [x for x in plan.row["stages"] if x["stage"] == "qwen"]
        assert st[0]["error"] == "qwen_busy" and len(fakes["qwen"].calls) == 1, st
    finally:
        app.PLANNING["n"] = 0
        app._peek_rate = real_rate
        app.STATE["lane"] = None
        app.STATE.pop("pool", None)


# ------------------------------------------------------------------ SPD-249: notes at the head

class _NotesHead:
    """`QWEN38_NOTES_HEAD` on for one test."""

    def __enter__(self):
        from server.conductor import router as R
        self.R, self.was = R, R.NOTES_HEAD
        R.NOTES_HEAD = "on"

    def __exit__(self, *exc):
        self.R.NOTES_HEAD = self.was


def test_notes_head_off_leaves_every_note_at_the_tail():
    from server.conductor import router as R
    assert R.NOTES_HEAD == "off", "default off until the quality gate passes"
    cond, _ = setup(decider=by_label("code_edit", 0.91))
    plan, ids, _ = turn(cond, body([{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]))
    assert plan.head_end is None and text_of(ids).endswith(plan.injected[0]["text"] + "<|im_end|>\n"
                                                           "<|im_start|>assistant\n<think>\n")


def test_notes_head_puts_the_class_note_at_the_end_of_the_system_message():
    """With the client's system message: the class's note goes before that message's `<|im_end|>`,
    its text as it is; the user's message and the generation prompt follow untouched; the grid mark
    (`head_end`) is where the note ends; the clean prompt is the client's."""
    with _NotesHead():
        cond, _ = setup(decider=by_label("code_edit", 0.91))
        plan, ids, clean = turn(cond, body([{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": CODE}]))
    note = plan.injected[0]["text"]
    t, c = text_of(ids), text_of(clean)
    i = c.index(SYSTEM) + len(SYSTEM)
    assert t == c[:i] + note + c[i:], (t[:400], c[:400])
    assert text_of(ids[:plan.head_end]).endswith(note) and plan.head_ids == ids[:plan.head_end]
    assert t.endswith(CODE + "<|im_end|>\n<|im_start|>assistant\n<think>\n")


def test_notes_head_without_a_system_message_adds_one_and_chats_share_it():
    """No system message: the notes go in a system message of their own at the head, and two chats with
    different first messages share the served prompt through the head's end (the prefix the cache
    serves); the varying message follows."""
    with _NotesHead():
        cond, _ = setup(decider=by_label("code_edit", 0.91))
        a, ids_a, clean_a = turn(cond, body([{"role": "user", "content": CODE}], tools=False),
                                 headers={"x-session-id": "head-a"})
        b, ids_b, _ = turn(cond, body([{"role": "user", "content": "Rename the parser's entry point "
                                                                     "to parseModule everywhere."}],
                                      tools=False), headers={"x-session-id": "head-b"})
    note = a.injected[0]["text"]
    t = text_of(ids_a)
    head = "<|im_start|>system\n" + note + "<|im_end|>\n"
    first = text_of(clean_a).index("<|im_start|>")
    assert t[first:].startswith(head), t[:300]
    assert a.head_end == b.head_end and ids_a[:a.head_end] == ids_b[:b.head_end]
    assert text_of(ids_a[:a.head_end]).endswith(head)
    assert a.injected[0]["reason"].endswith("(head)")


def test_notes_head_keeps_the_head_on_later_turns_and_puts_new_notes_at_the_tail():
    """Turn 2 replays the head note where it sat; a note this turn adds goes to the last message's
    end, as before, so the conversation's head never changes under it."""
    with _NotesHead():
        cond, _ = setup(decider=by_label("code_edit", 0.91))
        msgs, turns = _session_turns(cond, CODE, ["ls"])
        (p1, ids1, _), (p2, ids2, _) = turns
        assert p1.head_end and ids2[:p1.head_end] == ids1[:p1.head_end]
        assert p2.head_end == p1.head_end and not p2.fresh
        msgs += [{"role": "assistant", "content": "done"},
                 {"role": "user", "content": "Now make the sky blue in the header."}]
        fakes_label = by_label("design_ui", 0.9)
        cond.router.factory = lambda kind, role, timeout: Fake(kind, fakes_label if kind == "decider"
                                                               else (lambda t, o: "qwen_unreachable"))
        cond.router._adapters.clear()
        p3, ids3, _ = turn(cond, body(list(msgs)))
    assert ids3[:p1.head_end] == ids1[:p1.head_end], "the head holds"
    assert p3.injected and all(x["at"] >= p1.head_end for x in p3.injected), p3.injected
    assert text_of(ids3).endswith(p3.injected[-1]["text"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n")

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
