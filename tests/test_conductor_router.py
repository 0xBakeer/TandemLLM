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


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
