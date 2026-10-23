"""SRV-703 (QA SRV-704): memory states where it is on, and a model whose prompt has no place for the
conductor's notes reads none and says so. On the router test kit of tests/test_conductor_router.py
(the real Qwen template, fake classifier stages). Needs CPU torch (the box's CPU suite).

The cause (2026-10-22, the box's journal): Flint's template ends no message with a token, so
`inject.insertion_point` (the last `<|im_end|>`) found none, and every turn of a profile with memory
on read `no_insertion_point` while its row said `memory: on`. No card, index or body went in, so no
injection was journaled, the model wrote no block, and the writer had no session to write.

SRV-703's Gherkin, row by row:
  1  an owner chat on a profile with memory on: the index goes in and the journal records the
     injection; the background writer sees the session (and is due once it is idle)
                                                   test_1_memory_on_injects_and_the_writer_sees_it
  1b the model's fit says its prompt holds no note (Flint's adapter): nothing goes in, the prompt
     is the rendered one token for token, the row says memory off for the model and names no
     error; the writer sees no session              test_1b_no_note_place_reads_none_and_says_so
  1c Flint's adapter declares it, and the fit asks for no package, effort sentence, rule or memory
                                                   test_1c_flint_declares_no_note_place
  1d a prompt with no `<|im_end|>` has no insertion point (the cause on the box)
                                                   test_1d_a_prompt_without_im_end
  2  the Notes tab's "where": each profile's switch by name with its clients (absent: not set),
     the served model's state; the notes route carries it
                                                   test_2_where_names_each_profile_and_the_model
  2b `none` adds nothing (no card, no off note, no row part); `off` adds the off note once
                                                   test_2b_none_adds_nothing_off_tells_the_model
  3  every shipped profile states its memory (tests/test_conductor_derived.py checks the values)
                                                   test_3_every_shipped_profile_states_its_memory

Run: python tests/test_memory_where.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import body, by_label, setup  # noqa: E402

from server import app  # noqa: E402
from server.conductor import inject, memwriter, modelfit, schema  # noqa: E402
from server.conductor import memory as memory_mod  # noqa: E402
from server.conductor import routes  # noqa: E402


class _Journal:
    def __init__(self):
        self.rows = []

    def note(self, kind, task, *, decision=None, detail=None, **_):
        self.rows.append((kind, task, decision, detail))


def _switch(cond, value, name="chat"):
    """The kit's store holds the built-in profiles; the switch as a box would have it."""
    _, doc = cond.store.get("profiles", name)
    if value is None:
        doc.pop("memory", None)
    else:
        doc["memory"] = value
    cond.store.put("profiles", name, doc, note=f"memory {value}", author="test")


def _owner_notes(cond):
    for t in ("garden: tomatoes in the south bed", "units: metric"):
        cond.notes.write("owner", {"op": "add", "scope": "user", "kind": "reference", "title": t,
                                   "body": f"The user noted {t}."}, writer="human")


def turn(cond, b, headers=None, kind="opencode"):
    """The kit's turn (plan, render, apply) as the owner: the engine names `owner` for a request on
    loopback, and memory reads and writes nothing without a user."""
    plan = cond.plan(b, headers or {"x-session-id": "ses-1"}, client_kind=kind,
                     request_id="req", chat=True, user="owner")
    prompt, _, in_think = app.build_prompt(b)
    out, plan.in_think_after = plan.apply_prompt(prompt, app.STATE["tok"], in_think)
    return plan, out.tolist(), plan.clean(prompt.tolist())


def _kinds(plan) -> set:
    return {k for n in plan.injected for k in n["kinds"]}


def _chat(cond, text="What do you know about my garden?", sid="ses-w1"):
    return turn(cond, body([{"role": "user", "content": text}], tools=False),
                headers={"x-session-id": sid}, kind="open-webui")


def test_1_memory_on_injects_and_the_writer_sees_it():
    modelfit.install(None, "model")
    cond, _ = setup(decider=by_label("chat", 0.95))
    _switch(cond, "on")
    _owner_notes(cond)
    plan, ids, clean = _chat(cond)
    assert plan.row["extra"]["memory"]["mode"] == "on", plan.row["extra"].get("memory")
    assert {"memory"} <= _kinds(plan) and len(ids) > len(clean), "the index went in"
    assert "no_insertion_point" not in plan.row["extra"].get("errors", [])
    j = _Journal()
    cond.memory.journal = j
    memory_mod.mark_injected(cond.memory, plan)
    inj = [r for r in j.rows if r[:2] == ("memory_notes", "injected")]
    assert inj and inj[0][2] == "index" and len(inj[0][3]["ids"]) == 2, j.rows
    # the background writer: the session is its (app.py hands it Plan.memory), due once idle
    w = cond.memory.writer
    w.seen("conv-w1", plan.memory, "What do you know about my garden?")
    cfg = memwriter.settings_of(cond.memory.cfg())
    assert w.state("conv-w1")["mode"] == "on"
    assert w.due("conv-w1", int(cfg["min_new_tokens"]) + 10, 0.0, cfg) is None
    assert w.due("conv-w1", int(cfg["min_new_tokens"]) + 10, float(cfg["idle_s"]) + 1, cfg) == "idle"


def test_1b_no_note_place_reads_none_and_says_so():
    modelfit.install({"notes": False}, "model")
    try:
        cond, _ = setup(decider=by_label("chat", 0.95))
        _switch(cond, "on")
        _owner_notes(cond)
        plan, ids, clean = _chat(cond, sid="ses-w2")
        assert plan.row["extra"]["memory"] == {"mode": "off", "why": "model_notes"}
        assert plan.memory is None and not plan.injected and ids == clean, "nothing went in"
        assert "no_insertion_point" not in plan.row["extra"].get("errors", []), plan.row["extra"]
        assert plan.row["extra"]["model_fit"]["notes"] is False
        # the writer is never handed a session (app.py calls seen() with Plan.memory)
        cond.memory.writer.seen("conv-w2", plan.memory, "x")
        assert cond.memory.writer.state("conv-w2") is None
        # a coding turn with packages planned: none goes in either, and the row says why
        plan, ids, clean = turn(cond, body([{"role": "user", "content":
                                            "Refactor the lexer in src/lexer.ts."}]),
                                headers={"x-session-id": "ses-w3"})
        assert not plan.injected and ids == clean
        assert "no_insertion_point" not in plan.row["extra"].get("errors", [])
    finally:
        modelfit.install(None, "model")


def test_1c_flint_declares_no_note_place():
    from engine.models.flint import serving
    fit, src = modelfit.resolve(serving.FlintServing.caps.conductor, "model")
    assert src == "model" and fit["notes"] is False
    assert (fit["memory"], fit["packages"], fit["effort_notes"], fit["rules"]) == (False, "none",
                                                                                 False, False)
    # the setting can still lift it for an A/B, key by key
    fit, _ = modelfit.resolve(serving.FlintServing.caps.conductor, "notes=1")
    assert fit["notes"] is True and fit["memory"] is True
    # the other families keep theirs
    fit, _ = modelfit.resolve(None, "model")
    assert fit["notes"] is True and fit == modelfit.DEFAULTS


def test_1d_a_prompt_without_im_end():
    im_end = 151645
    assert inject.insertion_point([1, 2, im_end, 3, 4], im_end) == 2
    assert inject.insertion_point([1, 2, 3, 4], im_end) is None     # a Flint prompt's shape
    assert inject.insertion_point([1, 2, 3], None) is None


def test_2_where_names_each_profile_and_the_model():
    modelfit.install(None, "model")
    cond, _ = setup()
    _switch(cond, "on")
    _switch(cond, "off", "coding")
    w = memory_mod.where(cond.store)
    by = {p["name"]: p for p in w["profiles"]}
    assert w["model"] == {"memory": True, "why": None, "words": None}
    assert "raw" not in by and "bench" not in by, "the locked raw profiles are left out"
    assert by["chat"]["memory"] == "on" and by["chat"]["clients"] == ["open-webui", "dashboard"]
    assert (by["coding"]["memory"], by["coding"]["set"]) == ("off", True)
    # a profile stored without the key reads as `none`, not set
    _switch(cond, None)
    by = {p["name"]: p for p in memory_mod.where(cond.store)["profiles"]}
    assert (by["chat"]["memory"], by["chat"]["set"]) == ("none", False)
    assert sum(1 for p in by.values() if p["default"]) <= 1
    modelfit.install({"notes": False}, "model")
    try:
        w = memory_mod.where(cond.store)
        assert w["model"]["memory"] is False and w["model"]["why"] == "model_notes"
        assert "no place" in w["model"]["words"]
        status, out, _ = routes._notes(cond, "GET", [], {}, {})
        assert status == 200 and out["where"] == w
    finally:
        modelfit.install(None, "model")


def test_2b_none_adds_nothing_off_tells_the_model():
    modelfit.install(None, "model")
    for switch, note in (("none", False), ("off", True)):
        cond, _ = setup(decider=by_label("chat", 0.95))
        _switch(cond, switch)
        plan, ids, clean = _chat(cond, sid=f"ses-{switch}")
        text = app.STATE["tok"].decode(ids, skip_special_tokens=False)
        assert (memory_mod.OFF_NOTE[:40] in text) is note, switch
        assert "memory_index" not in {p["kind"] for n in plan.injected for p in n["parts"]}
        if switch == "none":
            assert plan.row["extra"].get("memory") is None, plan.row["extra"].get("memory")
        else:
            assert plan.row["extra"]["memory"] == {"mode": "off", "why": "profile"}


def test_3_every_shipped_profile_states_its_memory():
    for name, p in schema.SHIPPED_PROFILES.items():
        assert p.get("memory") in schema.MEMORY_MODES, name


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
