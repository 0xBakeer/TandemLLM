"""Plan mode, the todo list and rewind in chatd (SRV-363): each Gherkin scenario of the ticket
against the fake engine and a workspace on disk. CPU only.

Run: python -m pytest -q tests/test_chatd_plans.py
"""

from __future__ import annotations

import os

from chatd_util import Chat, run_all
from test_chatd_agent import calls, requests, small_window

from chatd.plans import APPROVED, PLAN_REFUSAL, PLAN_RULE, plan_items

PLAN = "FAKE_SAY Here is the plan:\\n1. Read the config\\n2. Add the **flag**\\n3. Write a test" \
       "\\n4. Run the suite"


def parts(snap, kind):
    return [p for p in snap["messages"][-1]["parts"] if p["type"] == kind]


def test_plan_mode_only_reads():
    c = Chat()
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"plan_mode": True, "effort": "low",
                                                         "tools": ["read_file", "write_file",
                                                                   "edit_file", "glob"]})
        c.svc.workspace(cid).write("a.txt", b"one")
        c.send(cid, calls(("edit_file", {"path": "a.txt", "old": "one", "new": "two"}),
                          ("read_file", {"path": "a.txt"})))
        s = c.wait_idle(cid, 20)
        edit, read = parts(s, "tool_call")
        assert edit["status"] == "failed" and edit["output"]["content"] == PLAN_REFUSAL
        assert read["status"] == "done" and "one" in read["output"]["content"]
        assert s["messages"][-1]["status"] == "done"                # the run went on
        assert c.svc.workspace(cid).read("a.txt") == b"one"
        first = requests(c)[0]
        assert sorted(t["function"]["name"] for t in first["tools"]) == ["glob", "read_file",
                                                                        "todo_write"]
        assert first["reasoning_effort"] == "high"
        assert PLAN_RULE in first["messages"][0]["content"]
    finally:
        c.close()


def test_approving_a_plan_makes_the_todo_list_and_runs_it():
    c = Chat()
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"plan_mode": True})
        sse = c.events(cid)
        c.send(cid, PLAN)
        c.wait_idle(cid, 20)
        r = c.ok("POST", f"/conversations/{cid}/plan/approve", {})
        assert [t["text"] for t in r["todos"]] == ["Read the config", "Add the flag",
                                                  "Write a test", "Run the suite"]
        assert all(t["state"] == "pending" for t in r["todos"])
        s = c.wait_idle(cid, 20)
        assert s["settings"]["plan_mode"] is False and len(s["todos"]) == 4
        user = s["messages"][-2]
        assert user["role"] == "user" and user["parts"][0]["text"] == APPROVED
        last = requests(c)[-1]
        assert "todo_write" in [t["function"]["name"] for t in last["tools"]]
        assert "[ ] Read the config" in last["messages"][-1]["content"]
        assert PLAN_RULE not in (last["messages"][0].get("content") or "")
        sse.until(lambda f: f[1] == "todos.changed")
        sse.close()
        # an edited list goes in as it is
        c.ok("PATCH", f"/conversations/{cid}/settings", {"plan_mode": True})
        c.send(cid, PLAN)
        c.wait_idle(cid, 20)
        r = c.ok("POST", f"/conversations/{cid}/plan/approve", {"items": ["only this"]})
        assert [t["text"] for t in r["todos"]] == ["only this"]
        c.wait_idle(cid, 20)
    finally:
        c.close()


def test_todo_write_keeps_the_list_and_it_survives_compaction():
    c = Chat()
    try:
        small_window(c, window=5000)
        c.svc.learn_ratio = lambda *a: None
        cid = c.new()
        c.svc.conv(cid).tok_ratio = 0.5
        c.ok("PATCH", f"/conversations/{cid}/settings", {"tools": ["read_file"]})
        items = [{"text": "a", "state": "done"}, {"text": "b", "state": "done"},
                 {"text": "c", "state": "pending"}, {"text": "d", "state": "in_progress"}]
        c.ok("PUT", f"/conversations/{cid}/todos", {"items": [{"text": "x"}]})
        c.send(cid, calls(("todo_write", {"items": items})))
        s = c.wait_idle(cid, 20)
        assert parts(s, "tool_call")[0]["output"]["content"] == "todo list saved: 4 items, 2 done"
        assert [t["state"] for t in s["todos"]] == ["done", "done", "pending", "in_progress"]
        long = "lorem ipsum dolor " * 120
        for i in range(3):
            c.send(cid, f"turn {i}: {long}")
            c.wait_idle(cid, 20)
        s = c.snap(cid)
        assert s["compactions"], "the long turns compacted"
        last = requests(c)[-1]
        assert "[x] a\n[x] b\n[ ] c\n[~] d" in last["messages"][-1]["content"]
        # the list sits on the user message: every later request renders it the same way
        users = [m for m in s["messages"] if m["role"] == "user"]
        assert sum(1 for p in users[-1]["parts"] if p["type"] == "todos") == 1
        st, err, _ = c.call("PUT", f"/conversations/{cid}/todos", {"items": [{"text": ""}]})
        assert st == 400
        # rewinding to the last message brings back the list its run started with
        c.ok("PUT", f"/conversations/{cid}/todos", {"items": [{"text": "a", "state": "done"}]})
        r = c.ok("POST", f"/conversations/{cid}/rewind", {"message_id": users[-1]["id"]})
        assert [t["state"] for t in r["conversation"]["todos"]] == ["done", "done", "pending",
                                                                    "in_progress"]
    finally:
        c.close()


def test_rewind_restores_the_files_and_moves_the_head():
    c = Chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("keep.txt", b"v0")
        c.send(cid, "first")
        c.wait_idle(cid, 20)
        c.send(cid, "second")
        s = c.wait_idle(cid, 20)
        second = s["messages"][-2]
        for i, step in enumerate([("write_file", {"path": "keep.txt", "content": "v1"}),
                                  ("write_file", {"path": "new.txt", "content": "made"}),
                                  ("edit_file", {"path": "keep.txt", "old": "v1", "new": "v2"})]):
            c.ok("PATCH", "/tools/" + step[0], {"policy": "always"})
            c.send(cid, calls(step))
            c.wait_idle(cid, 20)
        assert ws.read("keep.txt") == b"v2" and ws.read("new.txt") == b"made"
        n_before = len(c.svc._messages(cid))
        r = c.ok("POST", f"/conversations/{cid}/rewind", {"message_id": second["id"]})
        assert r["text"] == "second" and r["files"] == 2
        assert ws.read("keep.txt") == b"v0"
        assert not os.path.exists(os.path.join(ws.root, "new.txt"))
        conv = r["conversation"]
        assert conv["head_id"] == second["parent_id"]
        assert [m["parts"][0]["text"] for m in conv["messages"] if m["role"] == "user"] == \
            ["first"]
        assert len(c.svc._messages(cid)) == n_before             # nothing deleted
        assert r["conversation"]["todos"] == []                 # no open list back then
        # a new version of the second message: the old branch is one pager step away
        c.send(cid, "second, again")
        s = c.wait_idle(cid, 20)
        mine = s["messages"][-2]
        assert mine["parts"][0]["text"] == "second, again"
        assert mine["siblings"] == {"index": 1, "count": 2}
        b = c.ok("POST", f"/conversations/{cid}/branch", {"message_id": mine["id"], "index": 0})
        assert any(m["parts"][0]["text"] == "second" for m in b["messages"]
                   if m["role"] == "user")
        st, _, _ = c.call("POST", f"/conversations/{cid}/rewind", {"message_id": "nope"})
        assert st == 404
    finally:
        c.close()


def test_plan_items_from_an_answer():
    assert plan_items("Plan:\n1. a\n2) b\n\nNotes:\n- not a step") == ["a", "b"]
    assert plan_items("- one\n* two\n- [ ] three") == ["one", "two", "three"]
    assert plan_items("just do it") == ["just do it"]
    assert plan_items("") == []


if __name__ == "__main__":
    run_all(globals())
