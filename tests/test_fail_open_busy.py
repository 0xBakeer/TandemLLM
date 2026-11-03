"""SRV-763 (QA SRV-764): a turn whose class nobody decided, mostly because the server could not
answer the class question in time.

Live 2026-10-31 (fd627ac3, the Chat tab, 8 chats at once): the router's class question was skipped
(`qwen_busy`) or ran out its 8 s (`qwen_timeout`) on 92 of 113 conductor turns, and every new chat
fell back to the chat-app profile's default class, coding's code_edit, at xhigh with 5,956 tokens
of coding packages; the package question then waited out its own 8 s or was skipped too. The same
prompts one at a time routed to research, plan and writing at medium or high.

The test matrix (SRV-763's Gherkin):

  1  a new Chat-tab chat whose class question is skipped busy takes the profile's undecided class
     (chat), thinks at most at the profile's default level, gets its general packages alone and
     asks no package question; the row says so          test_1_a_new_chat_under_load_is_a_plain_chat
  2  a class question that timed out is the same         test_2_a_timeout_is_load_too
  3  a missing classifier (refused, unreachable) is no load: the default class, as before (SRV-534)
                                                         test_3_a_missing_classifier_keeps_the_default
  4  a session with a decided class keeps it and its level; no package question on that turn
                                                         test_4_a_decided_session_keeps_its_class
  5  the undecided class holds on the chat's tool turns (the same note) and gives way to the next
     answer even below the switch; a decided class still holds below it
                                                         test_5_the_busy_guess_gives_way_to_an_answer
  6  QWEN38_FAIL_OPEN=class: the default class's whole policy and the package question, as before
                                                         test_6_the_knob_restores_the_old_fallback
  7  a profile without an undecided class (opencode's coding, overnight, hermes) keeps its default
     class's whole policy (code_edit at xhigh, its general packages and core) and its stickiness; only
     the package question is skipped               test_7_opencode_keeps_code_edit_and_its_level
  8  an asked stage's row carries the estimate beside its time
                                                         test_8_an_asked_stage_has_its_estimate
  9  the library question is not asked on such a turn    test_9_the_library_question_waits_too
 10  `undecided_class` names a router class; chat-app ships "chat"; the migration gives a stored chat-app
     one when it has none, never twice, never over a choice, never without the class
                                                         test_10_*
 11  a first message below the bar takes the undecided class with its whole policy; opencode keeps
     code_edit                                   test_11_answers_below_the_bar
 12  a busy guess is a new session's class to a later message: one the box answers below the bar
     drops the lean (the class's whole policy and the package question); a build class over the
     bar's margin takes it (SRV-255); a classifier missing later drops the lean too
                                                 test_12_a_busy_guess_is_new_to_the_next_message
 13  below the bar with a build class on top that leads by less than the margin: the default class
     (code_edit and its files-care), as before; on a busy guess too
                                                 test_13_a_build_class_on_top_keeps_the_default

Run: python tests/test_fail_open_busy.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_conductor_router import body, by_label, setup, turn  # noqa: E402

from server.conductor import libsearch, migrations, schema  # noqa: E402
from server.conductor import router as router_mod  # noqa: E402
from server.conductor.store import Store  # noqa: E402

ESSAY = "Write a short essay about the history of tea in East Asia."
FREE, BUSY = 150.0, 20_000.0


def _box(qwen=None, decider=None, eta=None, chat_effort="high"):
    """A conductor with the shipped chat-app profile, packages on code_edit (a general one, a core,
    two task skills) and on chat and plan (a general one on chat, two the package question picks
    among), and the server's estimate for the Qwen stage (`eta["ms"]`, changeable per turn)."""
    cond, fakes = setup(decider=decider or (lambda t, o: "decider_unreachable"),
                        qwen=qwen or by_label("research", 0.9))
    fakes["qwen"].local = True                      # the server's own port: its queue is known
    eta = eta if eta is not None else {"ms": BUSY}
    cond.router.own_eta = lambda texts, max_new: eta["ms"]
    _, p = cond.store.get("packages")
    for n, general in (("coding", True), ("core-recipe", False), ("task-skill", False),
                       ("task-two", False), ("chat-core", True), ("chat-help", False),
                       ("chat-two", False)):
        p["packages"][n] = {"title": n, "kind": "package", "source": "inline",
                            "description": f"The {n} text.", "body": f"# {n} body",
                            "max_tokens": 3000, "enabled": True, "general": general}
    cond.store.put("packages", None, p, note="busy test", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"].update(packages=["coding", "core-recipe", "task-skill", "task-two"],
                               core=["core-recipe"])
        if c["id"] == "chat":
            c["policy"].update(packages=["chat-core", "chat-help", "chat-two"], effort=chat_effort)
        if c["id"] == "plan":
            c["policy"].update(packages=["chat-help", "chat-two"])
    cond.store.put("router", None, r, note="busy test", author="t")
    cond.store.put("profiles", "chat-app", copy.deepcopy(schema.SHIPPED_PROFILES["chat-app"]),
                   note="shipped", author="t")
    return cond, fakes, eta


def _parts(plan) -> list[str]:
    return [p["name"] for n in plan.injected for p in n["parts"] if p.get("kind") == "package"]


def _package_asked(fakes) -> bool:
    """A package question went to a stage: its options carry `none` (router.NONE_OPTION)."""
    return any(router_mod.NONE_OPTION in o for f in fakes.values() for _, o in f.calls)


def _chat(cond, text=ESSAY, sid="chat-1", msgs=None):
    m = msgs or [{"role": "user", "content": text}]
    return turn(cond, body(m), headers={"x-session-id": sid}, kind="chat-app")


def _probs_qwen(fakes, probs: dict):
    """The Qwen fake answers the class question with `probs` (its top as the label) and the
    package question with its first option at 0.9."""
    from server.conductor.classify import Answer
    fake = fakes["qwen"]

    def choice(state, question, options, max_chars=None):
        fake.calls.append((state.split("\nprev=", 1)[0], dict(options)))
        if router_mod.NONE_OPTION in options:
            pick = next(o for o in options if o != router_mod.NONE_OPTION)
            return Answer("qwen", pick, 0.9, {pick: 0.9}, 1.5, model="fake")
        top = max(probs.items(), key=lambda kv: kv[1])
        return Answer("qwen", top[0], top[1], dict(probs), 1.5, model="fake")
    fake.choice = choice


def _build_classes(cond):
    """design_ui and three_d build on this box: their policies name `files-care`
    (router.BUILD_MARK), as the shipped router's do, and SRV-255's rule is the shipped one."""
    _, p = cond.store.get("packages")
    p["packages"][router_mod.BUILD_MARK] = {
        "title": "files care", "kind": "package", "source": "inline",
        "description": "Never overwrite the user's files.", "body": "# files-care body",
        "max_tokens": 3000, "enabled": True, "general": True}
    cond.store.put("packages", None, p, note="busy test", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] in ("design_ui", "three_d"):
            c["policy"]["packages"] = [router_mod.BUILD_MARK]
    r["classifier"]["below_accept"] = dict(schema.SHIPPED_ROUTER["classifier"]["below_accept"])
    cond.store.put("router", None, r, note="busy test", author="t")


def _block_packages(plan) -> list[str]:
    """The packages of the turn's block (a later turn replays it in place: `injected` is short)."""
    return sorted(p["name"] for p in (plan.block or {}).get("parts", ()) if p["kind"] == "package")


def _session(cond):
    """The router's one session in these tests."""
    ss = list(cond.router._sessions.values())
    assert len(ss) == 1, [s.key for s in ss]
    return ss[0]


def test_1_a_new_chat_under_load_is_a_plain_chat():
    cond, fakes, _ = _box()
    plan, _, _ = _chat(cond)
    row = plan.row
    assert row["profile"] == "chat-app", row["profile"]
    assert (row["class_id"], row["source"], row["kept"]) == ("chat", "fallback", "fail_open"), row
    assert row["stages"][-1]["error"] == "qwen_busy" and not fakes["qwen"].calls
    assert row["extra"]["undecided_class"] == {"from": "code_edit", "to": "chat", "why": "busy"}, \
        row["extra"]
    assert row["extra"]["fail_open"] == {"lean": True, "effort": "medium"}, row["extra"]
    assert row["effort"] == "medium", "chat's high capped at the profile's default level"
    assert _parts(plan) == ["chat-core"], _parts(plan)
    assert row["extra"]["package_choice"]["how"] == "fail_open", row["extra"]["package_choice"]
    assert not _package_asked(fakes), "no package question on a busy turn"
    text = "".join(n["text"] for n in plan.injected)
    assert "coding body" not in text and schema.XHIGH_NOTE not in text
    assert plan.fail_open and plan.lean


def test_2_a_timeout_is_load_too():
    cond, fakes, _ = _box(qwen=lambda t, o: "qwen_timeout", eta={"ms": FREE})
    plan, _, _ = _chat(cond)
    row = plan.row
    assert (row["class_id"], row["kept"], row["effort"]) == ("chat", "fail_open", "medium"), row
    assert row["stages"][-1]["error"] == "qwen_timeout"
    assert len(fakes["qwen"].calls) == 1, "the class question alone, no package question after it"
    assert _parts(plan) == ["chat-core"] and not _package_asked(fakes)


def test_3_a_missing_classifier_keeps_the_default():
    cond, fakes, _ = _box(qwen=lambda t, o: "qwen_unreachable", eta={"ms": FREE},
                          decider=lambda t, o: "decider_refused")
    plan, _, _ = _chat(cond)
    row = plan.row
    assert (row["class_id"], row["kept"], row["effort"]) == ("code_edit", "fail_open", "xhigh"), row
    assert "undecided_class" not in row["extra"] and "fail_open" not in row["extra"], row["extra"]
    assert "coding" in _parts(plan) and "core-recipe" in _parts(plan)
    assert not plan.fail_open and not plan.lean


def test_4_a_decided_session_keeps_its_class():
    eta = {"ms": FREE}
    cond, fakes, _ = _box(qwen=lambda t, o: ("chat-help", 0.9) if "none" in o else ("plan", 0.9),
                          eta=eta)
    m1 = [{"role": "user", "content": "Plan a three-day trip to Kyoto for two."}]
    p1, _, _ = _chat(cond, msgs=m1)
    assert (p1.row["class_id"], p1.row["effort"]) == ("plan", "high"), p1.row
    assert _package_asked(fakes) and "chat-help" in _parts(p1)
    eta["ms"] = BUSY
    for f in fakes.values():
        f.calls.clear()
    m2 = m1 + [{"role": "assistant", "content": "Day 1: temples."},
               {"role": "user", "content": "Add a day for Nara and a budget for each day."}]
    p2, _, _ = _chat(cond, msgs=m2)
    row = p2.row
    assert (row["class_id"], row["kept"], row["effort"]) == ("plan", "fail_open", "high"), row
    assert row["extra"]["fail_open"] == {"lean": False} and "undecided_class" not in row["extra"]
    assert not _package_asked(fakes), "the package question waits for a free box"
    assert row["extra"]["package_choice"]["how"] in ("fail_open", "sticky"), row["extra"]


def test_5_the_busy_guess_gives_way_to_an_answer():
    eta = {"ms": BUSY}
    cond, fakes, _ = _box(qwen=by_label("research", 0.6), eta=eta)
    m1 = [{"role": "user", "content": ESSAY}]
    p1, ids1, _ = _chat(cond, msgs=m1)
    assert p1.row["class_id"] == "chat" and p1.lean
    # its tool turn: the same class, still lean, the same note, nothing asked
    m2 = m1 + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read",
                                                      "arguments": "{\"path\": \"a\"}"}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    eta["ms"] = FREE
    p2, ids2, _ = _chat(cond, msgs=m2)
    assert (p2.row["class_id"], p2.row["source"]) == ("chat", "sticky"), p2.row
    assert p2.lean and p2.row["effort"] == "medium" and not p2.injected
    assert p2.block["key"] == p1.block["key"] and ids2[:len(ids1)] == ids1
    assert not fakes["qwen"].calls and not _package_asked(fakes)
    # the next user message, the box free: research at 0.6 is below the switch (0.8), and the
    # busy class was no answer, so research it is
    m3 = m2 + [{"role": "assistant", "content": "Tea came from Yunnan."},
               {"role": "user", "content": "Compare green and black tea processing."}]
    p3, _, _ = _chat(cond, msgs=m3)
    row = p3.row
    assert (row["class_id"], row["source"]) == ("research", "qwen"), row
    assert row["extra"]["guess_replaced"] == {"was": "chat", "guess": "busy"}, row["extra"]
    assert not p3.lean
    # a decided class still holds below the switch
    m4 = m3 + [{"role": "assistant", "content": "Oxidation."},
               {"role": "user", "content": "Write me a short note about it."}]
    fakes["qwen"].fn = by_label("chat", 0.6)
    p4, _, _ = _chat(cond, msgs=m4)
    assert (p4.row["class_id"], p4.row["kept"]) == ("research", "below_switch"), p4.row


def test_6_the_knob_restores_the_old_fallback():
    old = router_mod.FAIL_OPEN
    router_mod.FAIL_OPEN = "class"
    try:
        cond, fakes, _ = _box(qwen=lambda t, o: "qwen_timeout", eta={"ms": FREE})
        plan, _, _ = _chat(cond)
        row = plan.row
        assert (row["class_id"], row["kept"], row["effort"]) == ("code_edit", "fail_open",
                                                                 "xhigh"), row
        assert "undecided_class" not in row["extra"] and "coding" in _parts(plan)
        assert _package_asked(fakes), "the package question is asked after the timeout, as before"
    finally:
        router_mod.FAIL_OPEN = old


def test_7_opencode_keeps_code_edit_and_its_level():
    for kind, extra, prof in (("opencode", {}, "coding"),
                              ("opencode", {"X-Tandem-Profile": "overnight"}, "overnight"),
                              ("opencode", {"X-Tandem-Profile": "hermes"}, "hermes")):
        eta = {"ms": BUSY}
        cond, fakes, _ = _box(qwen=by_label("research", 0.6), eta=eta)
        if prof not in cond.store.names():
            cond.store.put("profiles", prof, copy.deepcopy(schema.SHIPPED_PROFILES[prof]),
                           note="shipped", author="t")
        h = {"x-session-id": f"oc-{prof}", **extra}
        m1 = [{"role": "user", "content": "ok, run the tests"}]
        plan, _, _ = turn(cond, body(m1), headers=h, kind=kind)
        row = plan.row
        assert row["profile"] == prof and "undecided_class" not in row["extra"], row
        assert (row["class_id"], row["kept"], row["effort"]) == ("code_edit", "fail_open",
                                                                 "xhigh"), (prof, row)
        assert row["extra"]["fail_open"] == {"lean": False}, row["extra"]
        assert sorted(_parts(plan)) == ["coding", "core-recipe"], _parts(plan)
        assert row["extra"]["package_choice"]["how"] == "fail_open", "the package question skipped"
        assert not _package_asked(fakes) and not fakes["qwen"].calls
        assert plan.fail_open and not plan.lean
        assert _session(cond).guess is None, "no guess: the session is decided as on the base"
        # the next message, the box free: research at 0.6 is below the switch (0.8), and
        # code_edit holds, as on the base
        eta["ms"] = FREE
        m2 = m1 + [{"role": "assistant", "content": "Done."},
                   {"role": "user", "content": "Explain the failing test to me."}]
        p2, _, _ = turn(cond, body(m2), headers=h, kind=kind)
        assert (p2.row["class_id"], p2.row["source"], p2.row["kept"]) == \
            ("code_edit", "sticky", "below_switch"), (prof, p2.row)
        assert "guess_replaced" not in p2.row["extra"] and not p2.lean


def test_8_an_asked_stage_has_its_estimate():
    cond, fakes, _ = _box(eta={"ms": FREE})
    plan, _, _ = _chat(cond)
    st = [s for s in plan.row["stages"] if s["stage"] == "qwen"][0]
    assert st.get("error") is None and st["eta_ms"] == FREE and "ms" in st, st
    assert plan.row["class_id"] == "research"
    # both rows (asked with its estimate, skipped busy) are what the decisions contract allows
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                           "contract", "conductor-v1", "decisions.schema.json")) as f:
        c = json.load(f)
    busy, _, _ = _chat(_box()[0])
    for row in (st, busy.row["stages"][-1]):
        for item in (c["$defs"]["decision"]["properties"]["stages"]["items"], c["$defs"]["stage"]):
            assert schema.validate(json.loads(json.dumps(row)), item) == [], (row, item)


def test_9_the_library_question_waits_too():
    class R:
        def cascade(self, *a, **k):
            raise AssertionError("no question on a busy turn")

    class P:
        locked, fail_open = False, True
    assert libsearch._ask(R(), P(), [], "text", None, {"max_picks": 1}, {}, {}) == \
        ([], "fail_open", [])


def test_10_undecided_class_names_a_router_class_and_chat_app_ships_chat():
    prof = copy.deepcopy(schema.SHIPPED_PROFILES["chat-app"])
    assert prof["undecided_class"] == "chat"
    kw = dict(name="chat-app", router=schema.SHIPPED_ROUTER, packages=schema.SHIPPED_PACKAGES,
              profiles=schema.SHIPPED_PROFILES)
    assert not schema.check("profiles", prof, **kw)
    prof["undecided_class"] = "nope"
    assert any("undecided_class" in e for e in schema.check("profiles", prof, **kw))
    assert "undecided_class" not in schema.SHIPPED_PROFILES["coding"]
    assert "undecided_class" in schema.DERIVED_PROFILES["chat-app"][1]


MID = "2026-10-31-chat-app-undecided-class"
ONLY = [m for m in migrations.MIGRATIONS if m[0] == MID]


def _store_with(doc: dict) -> Store:
    st = Store(tempfile.mkdtemp(prefix="busy-mig-"))
    st.put("profiles", "chat-app", doc, note="stored", author="t")
    return st


def test_10_the_migration_gives_a_stored_chat_app_the_undecided_class():
    assert len(ONLY) == 1
    old = copy.deepcopy(schema.SHIPPED_PROFILES["chat-app"])
    del old["undecided_class"]
    st = _store_with(old)
    got = migrations.run(st, migrations=ONLY)
    assert got[0]["result"] == "changed" and got[0]["changes"] == ["undecided_class=chat"], got
    assert st.get("profiles", "chat-app")[1]["undecided_class"] == "chat"
    assert migrations.run(st, migrations=ONLY) == [], "never twice"
    # a stored choice stays
    mine = dict(old, undecided_class="plan")
    st = _store_with(mine)
    assert migrations.run(st, migrations=ONLY)[0]["result"] == "unchanged"
    assert st.get("profiles", "chat-app")[1]["undecided_class"] == "plan"
    # someone took it out after a version had it
    st = _store_with(mine)
    st.put("profiles", "chat-app", old, note="removed", author="owner")
    assert migrations.run(st, migrations=ONLY)[0]["result"] == "unchanged"
    assert "undecided_class" not in st.get("profiles", "chat-app")[1]
    # a stored router without the class: nothing to name
    class NoChat:
        def get(self, kind, name=None):
            r = copy.deepcopy(schema.ROUTER)
            r["classes"] = [c for c in r["classes"] if c["id"] != "chat"]
            return 1, r
    change = ONLY[0][3]
    assert change(old, [], store=NoChat()) is None
    assert change(old, [])[0]["undecided_class"] == "chat"


def test_11_answers_below_the_bar():
    cond, fakes, _ = _box(qwen=lambda t, o: ("chat-help", 0.9) if "none" in o else ("research", 0.3),
                          eta={"ms": FREE})
    plan, _, _ = _chat(cond)
    row = plan.row
    assert (row["class_id"], row["source"], row["kept"]) == ("chat", "fallback", "no_answer"), row
    assert row["extra"]["undecided_class"] == {"from": "code_edit", "to": "chat", "why": "below"}
    assert row["effort"] == "high" and "fail_open" not in row["extra"], "the class's whole policy"
    assert _package_asked(fakes) and sorted(_parts(plan)) == ["chat-core", "chat-help"]
    assert not plan.lean and not plan.fail_open
    # the next message's answer replaces it below the switch
    fakes["qwen"].fn = by_label("research", 0.6)
    m2 = [{"role": "user", "content": ESSAY}, {"role": "assistant", "content": "Tea."},
          {"role": "user", "content": "Compare green and black tea processing."}]
    p2, _, _ = _chat(cond, msgs=m2)
    assert p2.row["class_id"] == "research" and p2.row["extra"]["guess_replaced"] == \
        {"was": "chat", "guess": "below"}, p2.row
    # opencode: coding's default class, held below the switch, as before
    cond, fakes, _ = _box(qwen=by_label("research", 0.3), eta={"ms": FREE})
    p1, _, _ = turn(cond, body([{"role": "user", "content": ESSAY}]),
                    headers={"x-session-id": "oc-2"}, kind="opencode")
    assert (p1.row["class_id"], p1.row["kept"], p1.row["effort"]) == ("code_edit", "no_answer",
                                                                      "xhigh"), p1.row
    fakes["qwen"].fn = by_label("research", 0.6)
    p2, _, _ = turn(cond, body(m2), headers={"x-session-id": "oc-2"}, kind="opencode")
    assert (p2.row["class_id"], p2.row["kept"]) == ("code_edit", "below_switch"), p2.row


def test_12_a_busy_guess_is_new_to_the_next_message():
    eta = {"ms": BUSY}
    cond, fakes, _ = _box(qwen=lambda t, o: ("chat-help", 0.9) if "none" in o else ("research", 0.3),
                          eta=eta)
    m1 = [{"role": "user", "content": ESSAY}]
    p1, _, _ = _chat(cond, msgs=m1)
    assert p1.row["class_id"] == "chat" and p1.lean
    # the next message, the box free, its answer below the bar (research 0.3): no longer lean,
    # chat's whole policy (high on this box) and the package question
    eta["ms"] = FREE
    m2 = m1 + [{"role": "assistant", "content": "Tea came from Yunnan."},
               {"role": "user", "content": "Tell me more about the tea ceremony."}]
    p2, _, _ = _chat(cond, msgs=m2)
    row = p2.row
    assert (row["class_id"], row["source"], row["kept"]) == ("chat", "fallback", "no_answer"), row
    assert not p2.lean and not p2.fail_open and "fail_open" not in row["extra"], row["extra"]
    assert row["effort"] == "high", "the class's whole policy"
    assert _package_asked(fakes) and _block_packages(p2) == ["chat-core", "chat-help"], p2.block
    assert _session(cond).guess == "below", _session(cond).guess
    # a later busy message makes it lean again; the guess still gives way to an answer
    eta["ms"] = BUSY
    for f in fakes.values():
        f.calls.clear()
    m3 = m2 + [{"role": "assistant", "content": "Matcha."},
               {"role": "user", "content": "And in Korea?"}]
    p3, _, _ = _chat(cond, msgs=m3)
    assert p3.lean and p3.row["effort"] == "medium" and not _package_asked(fakes), p3.row

    # SRV-255 on a busy guess: a build class over the margin is the first classified message's
    cond, fakes, _ = _box(qwen=by_label("research", 0.3), eta=eta)
    _build_classes(cond)
    eta["ms"] = BUSY
    p1, _, _ = _chat(cond, msgs=m1)
    assert p1.row["class_id"] == "chat" and p1.lean
    eta["ms"] = FREE
    _probs_qwen(fakes, {"three_d": 0.38, "chat": 0.2, "research": 0.1})
    m2 = m1 + [{"role": "assistant", "content": "Tea."},
               {"role": "user", "content": "Create a new folder and create there a pagoda garden"}]
    p2, _, _ = _chat(cond, msgs=m2)
    row = p2.row
    assert (row["class_id"], row["kept"]) == ("three_d", "below_build"), row
    assert row["extra"]["guess_replaced"] == {"was": "chat", "guess": "busy"}, row["extra"]
    assert not p2.lean

    # a classifier missing on the later message: not the box's load, so the lean goes
    cond, fakes, _ = _box(qwen=by_label("research", 0.3), eta=eta)
    eta["ms"] = BUSY
    p1, _, _ = _chat(cond, msgs=m1)
    assert p1.lean
    eta["ms"] = FREE
    fakes["qwen"].fn = lambda t, o: "qwen_unreachable"
    m2 = m1 + [{"role": "assistant", "content": "Tea."},
               {"role": "user", "content": "Tell me more."}]
    p2, _, _ = _chat(cond, msgs=m2)
    assert (p2.row["class_id"], p2.row["kept"]) == ("chat", "fail_open"), p2.row
    assert not p2.lean and p2.row["effort"] == "high", p2.row


def test_13_a_build_class_on_top_keeps_the_default():
    # design_ui 0.40 leads three_d 0.35 by less than the margin (0.1): `_below_build` declines,
    # and the top class builds, so the new chat keeps the default class with its files-care
    cond, fakes, _ = _box(eta={"ms": FREE})
    _build_classes(cond)
    _probs_qwen(fakes, {"design_ui": 0.40, "three_d": 0.35, "chat": 0.1})
    plan, _, _ = _chat(cond, text="Make me a landing page with a 3D hero in a new folder")
    row = plan.row
    assert (row["class_id"], row["source"], row["kept"]) == ("code_edit", "fallback",
                                                             "no_answer"), row
    assert row["extra"]["undecided_builds"] == {"top": "design_ui", "kept": "code_edit"}, \
        row["extra"]
    assert "undecided_class" not in row["extra"] and row["effort"] == "xhigh"
    assert not plan.lean and not plan.fail_open
    assert _session(cond).guess is None, "decided as on the base: code_edit holds below the switch"
    # a prose class on top: the undecided class
    cond, fakes, _ = _box(eta={"ms": FREE})
    _build_classes(cond)
    _probs_qwen(fakes, {"research": 0.40, "design_ui": 0.35})
    plan, _, _ = _chat(cond)
    assert (plan.row["class_id"], plan.row["extra"]["undecided_class"]["why"]) == \
        ("chat", "below"), plan.row
    # on a busy guess: the same, and the guess ends
    eta = {"ms": BUSY}
    cond, fakes, _ = _box(qwen=by_label("research", 0.3), eta=eta)
    _build_classes(cond)
    p1, _, _ = _chat(cond)
    assert p1.row["class_id"] == "chat" and p1.lean
    eta["ms"] = FREE
    _probs_qwen(fakes, {"design_ui": 0.40, "three_d": 0.35})
    m2 = [{"role": "user", "content": ESSAY}, {"role": "assistant", "content": "Tea."},
          {"role": "user", "content": "Now build a page about it with a 3D teapot."}]
    p2, _, _ = _chat(cond, msgs=m2)
    assert (p2.row["class_id"], p2.row["kept"], p2.row["effort"]) == ("code_edit", "no_answer",
                                                                      "xhigh"), p2.row
    assert p2.row["extra"]["guess_replaced"] == {"was": "chat", "guess": "busy"}
    assert not p2.lean


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
