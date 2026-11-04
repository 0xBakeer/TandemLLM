"""A rule about the user from a tool turn is kept at session scope (2026-11-04).

The store's guard (`rule_from_tool`, server/conductor/notes.py): a `rule` or `ruling` about the
user needs the user's own message in the turn. In a chat with no working directory every project
note is a user note, so the guard refused most rules the model wrote during a tool run. Now such a
rule is kept for its conversation only: session scope, provisional, the same writer and kind, and
the result, the journal and the log say `demoted: rule_from_tool`. It never becomes a user or
project rule; without a session it is refused as before. String handling only (no torch):

  1  the store: the Chat tab (no project), a project, a user turn, a ruling, no session; the
     demoted note is a session note (file, visibility, changelog), the same rule again revises it
                                                        test_store_*
  2  unchanged: an update of a user rule from a tool turn is refused; caps and an empty body are
     refused; a project rule from a tool turn with a project is a project note as before
                                                        test_unchanged_*
  3  promotion only by a user turn: the same rule from the user's message is a new user note
                                                        test_a_user_turn_writes_the_user_rule
  4  every writer: the inline block and the call form (Memory.take through the stream filter),
     the background writer, the journal rows; `X-Tandem-Memory: read` stores nothing, memory off
     has no filter                                      test_take_*, test_writer_*, test_switches

All notes are synthetic. Run: python tests/test_memory_rule_scope.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import memory as memory_mod  # noqa: E402
from server.conductor.notes import CAPS, NoteStore, Refused  # noqa: E402

PROJ = "/home/user/pagoda"
SESS = "c:chat-test-1"
BLOCK = ('<tandem_memory op="add" scope="{scope}" kind="{kind}">\ntitle: {title}\n\n'
         'The user wants metric units in every answer.\n</tandem_memory>')
CALL = ("<tool_call>\n<function=tandem_memory>\n<parameter=op>\nadd\n</parameter>\n"
        "<parameter=scope>\nproject\n</parameter>\n<parameter=kind>\nrule\n</parameter>\n"
        "<parameter=title>\nmetric units\n</parameter>\n<parameter=body>\nThe user wants metric "
        "units in every answer.\n</parameter>\n</function>\n</tool_call>")


class Journal:
    def __init__(self):
        self.rows = []

    def note(self, kind, task, **kw):
        self.rows.append((kind, task, kw))
        return {}


class UI:
    def __init__(self):
        self.lines = []

    def add(self, kind, **facts):
        self.lines.append((kind, facts))


def store() -> NoteStore:
    return NoteStore(tempfile.mkdtemp(prefix="mem-scope-"),
                     settings=lambda: {"dedupe_threshold": 0.6})


def rule(st, title="metric units", scope="user", kind="rule", project=None, session=SESS,
         from_user=False, writer="qwen-inline"):
    return st.write("owner", {"op": "add", "scope": scope, "kind": kind, "title": title,
                              "body": f"The user wants {title} in every answer."},
                    writer=writer, project=project, session=session, request_id="req-1",
                    from_user=from_user)


def refused(fn, why):
    try:
        fn()
    except Refused as exc:
        assert exc.why == why, (exc.why, why)
        return
    raise AssertionError(f"not refused ({why})")


# ------------------------------------------------------------------------------ the store
def test_store_chat_tab_without_a_project():
    st = store()
    r = rule(st, scope="project")                 # rescoped to user, then kept for the session
    assert (r["scope"], r["kind"], r["rescoped"], r["demoted"]) == \
        ("session", "rule", "project", "rule_from_tool"), r
    n = st.get("owner", r["id"])
    assert n.scope == "session" and n.session == SESS and n.project is None
    assert n.status == "provisional" and n.writer == "qwen-inline" and n.kind == "rule"
    assert "/sessions/" in n.path and "/owner/user/" not in n.path, n.path
    assert "at session scope (rule_from_tool)" in n.changelog[0], n.changelog
    assert st.visible(n, None, SESS) and not st.visible(n, None, "c:chat-test-2")
    assert [x.scope for x in st.notes("owner")] == ["session"]
    assert st.stats["demoted"] == 1 and st.stats["writes"] == 1
    # the same rule again in a tool turn: a revision of the session note, still demoted
    again = rule(st, scope="project")
    assert again["id"] == r["id"] and again["revised"] and again["demoted"] == "rule_from_tool"
    assert len(st.notes("owner")) == 1


def test_store_with_a_project():
    st = store()
    r = rule(st, project=PROJ, writer="qwen-writer")
    assert r["scope"] == "session" and r["demoted"] == "rule_from_tool" and "rescoped" not in r
    n = st.get("owner", r["id"])
    assert n.session == SESS and n.project is None and n.writer == "qwen-writer"


def test_store_ruling_and_no_session():
    st = store()
    r = rule(st, title="the release waits for the self-test", kind="ruling")
    assert r["scope"] == "session" and r["kind"] == "ruling" and r["demoted"] == "rule_from_tool"
    refused(lambda: rule(st, title="tabs", session=None), "rule_from_tool")
    refused(lambda: rule(st, title="tabs", scope="project", session=None), "rule_from_tool")
    assert len(st.notes("owner")) == 1


def test_store_user_turn_and_other_kinds_unchanged():
    st = store()
    r = rule(st, from_user=True)
    assert r["scope"] == "user" and "demoted" not in r, r
    g = rule(st, title="the build needs node 20", kind="gotcha")
    assert g["scope"] == "user" and "demoted" not in g, g
    assert st.stats["demoted"] == 0


# ------------------------------------------------------------------------------ unchanged
def test_unchanged_refusals():
    st = store()
    u = rule(st, from_user=True)
    refused(lambda: st.write("owner", {"op": "update", "id": u["id"], "title": "metric units",
                                       "body": "The user wants SI units in every answer."},
                             writer="qwen-inline", session=SESS, from_user=False),
            "rule_from_tool")
    assert st.get("owner", u["id"]).body.startswith("The user wants metric units")
    refused(lambda: st.write("owner", {"op": "add", "scope": "user", "kind": "rule",
                                       "title": "no body", "body": ""},
                             writer="qwen-inline", session=SESS, from_user=False), "schema")
    refused(lambda: rule(st, title="x" * (CAPS["title_chars"] + 1)), "cap")
    words = ("alder birch cedar dogwood elm fir ginkgo hazel ilex juniper kauri larch maple "
             "nutmeg oak pine quince rowan spruce teak umbrella vine walnut yew").split()
    for i in range(CAPS["session_notes"]):
        w = words[i]
        st.write("owner", {"op": "add", "scope": "session", "kind": "reference",
                           "title": f"{w} {w}", "body": f"{w} {w} {w} {w}."},
                 writer="qwen-inline", session=SESS)
    assert len([n for n in st.notes("owner") if n.scope == "session"]) == CAPS["session_notes"]
    refused(lambda: rule(st, title="one rule too many"), "cap")


def test_unchanged_project_rule_with_a_project():
    st = store()
    r = rule(st, scope="project", project=PROJ)
    assert r["scope"] == "project" and "demoted" not in r, r


def test_a_user_turn_writes_the_user_rule():
    st = store()
    s = rule(st)
    u = rule(st, from_user=True)
    assert s["scope"] == "session" and u["scope"] == "user" and u["id"] != s["id"]
    assert st.get("owner", s["id"]).scope == "session"


# ------------------------------------------------------------------------------ every writer
def _ctx(**kw):
    return {"mode": "on", "user": "owner", "project": None, "session": SESS, "from_user": False,
            "request_id": "r-scope", "writer": "qwen-inline", "written": [], **kw}


def test_take_block_and_call_in_a_tool_turn():
    for raw in (BLOCK.format(scope="project", kind="rule", title="metric units"), CALL):
        st, j, ui = store(), Journal(), UI()
        mem = memory_mod.Memory(st, j)
        ctx = _ctx()
        f = mem.filter(ctx, ui)
        out = f.feed("The tests pass.\n\n" + raw) + f.finish()
        assert out.strip() == "The tests pass." and f.failed == [] and f.hidden == [raw], out
        ns = st.notes("owner")
        assert [(n.scope, n.kind, n.session) for n in ns] == [("session", "rule", SESS)], ns
        w = ctx["written"][0]
        assert (w["scope"], w["demoted"], w["rescoped"], w["why"]) == \
            ("session", "rule_from_tool", "project", None), w
        row = [kw for k, t, kw in j.rows if (k, t) == ("memory_notes", "write")][-1]
        assert row["decision"] == "add" and row["detail"]["demoted"] == "rule_from_tool", row
        assert "metric" not in str(row), "the journal row has no text"
        assert ui.lines == [("remembered", {"title": "metric units"})], ui.lines


def test_take_user_turn_and_no_session():
    st = store()
    mem = memory_mod.Memory(st, Journal())
    ctx = _ctx(from_user=True)
    f = mem.filter(ctx)
    f.feed(BLOCK.format(scope="user", kind="rule", title="metric units"))
    f.finish()
    assert [n.scope for n in st.notes("owner")] == ["user"] and "demoted" not in ctx["written"][0]
    ctx = _ctx(session=None)
    f = mem.filter(ctx)
    f.feed(BLOCK.format(scope="user", kind="rule", title="tabs not spaces"))
    f.finish()
    assert f.failed == ["rule_from_tool"] and len(st.notes("owner")) == 1, f.failed


def test_switches():
    st = store()
    mem = memory_mod.Memory(st, Journal())
    ctx = _ctx(read_only=True)
    f = mem.filter(ctx)
    raw = BLOCK.format(scope="user", kind="rule", title="metric units")
    assert (f.feed("Done.\n" + raw) + f.finish()).strip() == "Done."
    assert ctx["written"][0]["why"] == "read_only" and st.notes("owner") == [], ctx["written"]
    assert mem.filter(_ctx(mode="off")) is None and mem.filter(None) is None


def test_writer_keeps_the_rule_for_the_session():
    st, j = store(), Journal()
    mem = memory_mod.Memory(st, j)
    w = mem.writer
    w.seen("conv-1", _ctx(), None)
    text = BLOCK.format(scope="user", kind="rule", title="metric units") + "\n\n" + \
        CALL.replace("metric units", "short answers")
    run = w.apply("conv-1", text, {"mode": "on", "max_blocks": 4}, "idle", 10, {})
    assert run["refused"] == [] and len(run["ops"]) == 2, run
    assert {(o["scope"], o["demoted"]) for o in run["ops"]} == {("session", "rule_from_tool")}
    ns = st.notes("owner")
    assert sorted((n.scope, n.writer) for n in ns) == [("session", "qwen-writer")] * 2, ns


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"ok {name}")
    print(f"{passed} passed")
