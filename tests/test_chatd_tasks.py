"""Workers (VIS-E15, SRV-579; QA SRV-580): the task tool and the worker runs. CPU only: the model is
tests/fake_backend.py with a script (`fake.script`) that plays the main agent and each worker by
the request's `x-session-id`; the sandbox, where one is needed, the fake runtime of
tests/test_chatd_agent.py.

  1  only the task result reaches the main context: the worker writes a 40 KB file, the main
     agent's next request carries none of it and grew by at most 2,000 tokens
                                                  test_only_the_task_result_reaches_the_main_context
  2  the worker's first request: the worker block, the worker tools, one user message with the
     task's number, title, brief, expect and the todo list, byte for byte; nothing of a chat of 30
     messages                                     test_the_worker_starts_with_a_fresh_context
  3  one sandbox, one /workspace, the chat's rules: the worker's bash runs in the chat's container,
     its file is the main agent's, a deny rule of the chat refuses its call
                                                  test_the_worker_shares_the_chats_sandbox_and_rules
  4  the worker takes no run slot: two holders with a second chat, a third chat waits
                                                  test_the_worker_takes_no_run_slot_of_its_own
  5  session ids chat-<conv>-task-1 and -task-2, the worker's headers; the chat's unchanged
                                                  test_workers_have_their_own_sessions_and_headers
  6  task conversations stay out of the list, search, the rail and titles; delete and purge take
     them along                                   test_task_conversations_stay_hidden_and_go_with_the_chat
  7  one task at a time: a step's second task call answers ONE_TASK
                                                  test_one_task_at_a_time
  8  at most 8 tasks in a run: the ninth answers the limit, the run goes on
                                                  test_at_most_8_tasks_in_a_run
  9  depth 1: a worker's tools have no task and no todo_write; plan mode lists no task
                                                  test_a_worker_cannot_delegate
 10  a worker compacts its own conversation: the rows are its, the brief stays byte for byte, the
     main context is unchanged                    test_a_worker_compacts_its_own_conversation
 11  the events on the chat's log (task.started, task.phase, task.done, the worker's parts with
     task_id, the parent's `task` phase) and both endpoints
                                                  test_the_events_and_the_endpoints
 12  the schema step: v7 to v8, a v8 file refused by a chatd of v7 (the rollback restores the backup)
                                                  test_the_schema_step_and_its_rollback
 13  Agents off: the stored chats' requests are origin/main's, byte for byte (tests/chatd_replay.py)
                                                  test_agents_off_requests_are_byte_identical_to_main

Run: PYTHONPATH=.:tests python3 tests/test_chatd_tasks.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import threading
import time

import chatd_replay
from chatd_util import Chat, run_all
from test_chatd_agent import chat_with_sandbox, small_window

from chatd import compaction as comp
from chatd import render
from chatd import tools as T
from chatd.store import MIGRATIONS, Store, StoreError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIEF = ("Goal: /workspace/hello.txt holds the word hi\n"
         "Files: hello.txt (new), nothing else is read\n"
         "Constraints: change no other file in /workspace\n"
         "Check: read_file hello.txt shows the word hi")
REPORT = ("Status: done\nSummary: wrote hello.txt\nChanged files:\n- hello.txt: new, one line\n"
          "Checks:\n- read_file hello.txt: hi\nExpected:\n- nothing asked\nOpen problems:\n- none")


def hdr(h: dict, k: str) -> str | None:
    return {a.lower(): b for a, b in h.items()}.get(k.lower())


def since_user(body: dict) -> list[dict]:
    """The messages after the request's last user message: the run's steps so far."""
    msgs = body.get("messages") or []
    i = max((k for k, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    return msgs[i + 1:]


def task_call(title="Hello file", brief=BRIEF, expect="nothing"):
    return {"name": "task", "arguments": {"title": title, "brief": brief, "expect": expect}}


class Script:
    """The fake's script: `main(body, k)` for the chat's requests and `worker(body, k, n)` for a
    worker's, `k` the tool results so far in the run (from 0), `n` the task's number."""

    def __init__(self, main=None, worker=None):
        self.main = main or (lambda b, k: {"calls": [task_call()]} if k == 0 else
                             {"content": "All done."})
        self.worker = worker or (lambda b, k, n: {"calls": [{"name": "write_file", "arguments": {
            "path": "hello.txt", "content": "hi"}}]} if k == 0 else {"content": REPORT})

    def __call__(self, body: dict, h: dict):
        if body.get("stream") is not True:
            return None                       # a compaction's or a title's call: the fake's words
        s = hdr(h, "x-session-id") or ""
        if "-task-" in s:                 # a worker has one user message (a tail is one more)
            k = sum(1 for m in body.get("messages") or [] if m.get("role") == "tool")
            return self.worker(body, k, int(s.rsplit("-", 1)[1]))
        return self.main(body, sum(1 for m in since_user(body) if m.get("role") == "tool"))


def agents_chat(c, **settings) -> str:
    cid = c.new()
    c.ok("PATCH", f"/conversations/{cid}/settings", {"agents": True, **settings})
    return cid


def worker_calls(c, n=None) -> list[tuple[dict, dict]]:
    return [(h, b) for h, b in c.fake.calls if "-task-" in (hdr(h, "x-session-id") or "") and
            (n is None or (hdr(h, "x-session-id") or "").endswith(f"-task-{n}")) and
            b.get("stream")]


def main_calls(c, cid) -> list[tuple[dict, dict]]:
    return [(h, b) for h, b in c.fake.calls if hdr(h, "x-session-id") == f"chat-{cid}" and
            b.get("stream")]


def task_part(snap) -> dict:
    return next(p for p in snap["messages"][-1]["parts"]
                if p["type"] == "tool_call" and p["name"] == "task")


# ---------------------------------------------------------------------- the main context
def test_only_the_task_result_reaches_the_main_context():
    c = Chat()
    try:
        body40 = "".join(f"fn shade{i}(p: vec3f) -> f32 {{ return sin(p.x * {i}.0); }}\n"
                         for i in range(700))            # ~40 KB of WGSL
        assert len(body40) > 40_000

        def worker(b, k, n):
            if k == 0:
                return {"calls": [{"name": "write_file", "arguments": {"path": "src/ocean.wgsl",
                                                                       "content": body40}}]}
            return {"content": REPORT.replace("hello.txt", "src/ocean.wgsl")}
        c.fake.script = Script(lambda b, k: {"content": "ok"}, worker)
        cid = agents_chat(c)
        c.send(cid, "context " + "ocean sunset words " * 3000)   # the main context: ~18,000 tokens
        c.wait_idle(cid, 30)
        c.fake.script.main = lambda b, k: {"calls": [task_call("Ocean shader")]} if k == 0 else \
            {"content": "The shader is written."}
        c.send(cid, "write the shader")
        s = c.wait_idle(cid, 40)
        assert open(os.path.join(c.data, "workspaces", cid, "src", "ocean.wgsl")).read() == body40
        first, second = [b for _, b in main_calls(c, cid)][-2:]
        assert comp.estimate(first, None) > 18_000
        assert "shade3(" not in json.dumps(second)
        assert comp.estimate(second, None) - comp.estimate(first, None) <= 2000
        res = second["messages"][-1]
        assert res["role"] == "tool" and "src/ocean.wgsl" in res["content"]
        assert task_part(s)["status"] == "done"
    finally:
        c.close()


def test_the_worker_starts_with_a_fresh_context():
    c = Chat()
    try:
        c.fake.script = Script(lambda b, k: {"content": "ok"})
        cid = agents_chat(c)
        for i in range(15):                                    # 30 messages
            c.send(cid, f"earlier message number {i}")
            c.wait_idle(cid, 20)
        c.ok("PUT", f"/conversations/{cid}/todos", {"items": [
            {"text": "Scene skeleton with WebGPU and a WebGL2 fallback", "state": "done"},
            {"text": "Ocean shader: fix the horizon banding", "state": "in_progress"},
            {"text": "Sun and sky colours at sunset", "state": "pending"}]})
        brief = ("Goal: the horizon shows no banding at 1280x800\n"
                 "Files: src/ocean.wgsl (the ocean's fragment shader)\n"
                 "Constraints: keep the WebGL2 fallback as it is\n"
                 "Check: render_page index.html, no console errors")
        c.fake.script.main = lambda b, k: {"calls": [task_call(
            "Ocean shader: fix the horizon banding", brief,
            "the FPS line from the console after render_page")]} if k == 0 else {"content": "ok"}
        c.send(cid, "fix the banding")
        c.wait_idle(cid, 30)
        h, b = worker_calls(c)[0]
        assert [m["role"] for m in b["messages"]] == ["system", "user"]
        today = _dt.date.today().isoformat()
        assert b["messages"][0]["content"] == (
            "You are a worker in TandemLLM. Another assistant, the main agent, talks with the user "
            "and gave you one task. You do not see that chat. Everything you know about the task "
            "is in the brief below. The files are in /workspace, which you share with the main "
            "agent.\n\nDo the task with your tools. Change only what the brief asks for. Check "
            "your work the way the brief's Check line says, and run that check before you report. "
            "If the brief is unclear or the task is impossible, do what you can, stop, and say "
            "what is missing in the report. Do not guess what the user wants beyond the brief.\n\n"
            "You have at most 40 model calls. A note from the chat can appear at the end of a "
            "tool result. Read it. If it changes or cancels your goal, finish the current step "
            "safely and report.\n\nEnd with a report in exactly this shape, and nothing after "
            "it:\n\nStatus: done | partial | failed\nSummary: one to three lines on what you did\n"
            "Changed files:\n- path: what changed, in one line (or \"none\")\nChecks:\n- what you "
            "ran: the result (screenshot ids such as shot-3 when you used render_page)\nExpected:"
            "\n- each item the brief's expect asks for: the answer (or \"nothing asked\")\nOpen "
            "problems:\n- what is not done, broken or uncertain, and any question for the user (or "
            "\"none\")\n\nNever paste file contents into the report. The main agent reads files "
            "itself when it needs them.\n\n"
            f"You are the assistant in TandemLLM's chat. Today is {today}. You can use tools. "
            "Files live in /workspace, this chat's own Linux sandbox; the internet is reachable "
            "through a proxy, the local network is not. Read a file before you edit it, and "
            "prefer edit_file for small changes. Use present to show the user a page, a file or "
            "this run's changes. When the task is done, answer in plain words; do not call tools "
            "you do not need.")
        assert b["messages"][1]["content"] == (
            "Task 1: Ocean shader: fix the horizon banding\nBrief:\n" + brief +
            "\nThe report must also contain: the FPS line from the console after render_page\n"
            "The chat's todo list (the main agent keeps it; you cannot change it):\n"
            "- [done] Scene skeleton with WebGPU and a WebGL2 fallback\n"
            "- [in_progress] Ocean shader: fix the horizon banding\n"
            "- [pending] Sun and sky colours at sunset")
        assert "earlier message number" not in json.dumps(b)
        assert [t["function"]["name"] for t in b["tools"]] == [
            "read_file", "write_file", "edit_file", "glob", "grep", "web_search", "web_fetch",
            "present"]                   # WORKER_TOOLS of this box (no sandbox, no library)
    finally:
        c.close()


def test_the_worker_shares_the_chats_sandbox_and_rules():
    c = chat_with_sandbox()
    try:
        def worker(b, k, n):
            if k == 0:
                return {"calls": [{"name": "write_file", "arguments": {"path": "w.txt",
                                                                       "content": "from worker"}}]}
            if k == 1:
                return {"calls": [{"name": "bash", "arguments": {"command": "ls /workspace"}}]}
            return {"content": REPORT}

        def main(b, k):
            if k == 0:
                return {"calls": [task_call()]}
            if k == 1:
                return {"calls": [{"name": "glob", "arguments": {"pattern": "*.txt"}}]}
            return {"content": "ok"}
        c.fake.script = Script(main, worker)
        cid = agents_chat(c, permission_mode="auto")
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        box = c.svc.sandboxes
        assert set(box.boxes) == {cid}                      # one container, the chat's
        assert ["bash", "-lc", "ls /workspace"] in box.ran
        assert open(os.path.join(c.data, "workspaces", cid, "w.txt")).read() == "from worker"
        glob = [p for p in s["messages"][-1]["parts"] if p.get("name") == "glob"][0]
        assert "/workspace/w.txt" in glob["output"]["content"]
        tcid = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]["task_conversation_id"]
        assert not os.path.exists(os.path.join(c.data, "workspaces", tcid))
        assert c.svc.workspace(tcid).root == c.svc.workspace(cid).root
        # a deny rule of the chat refuses the worker's call
        c.svc.add_rule("bash", None, "deny", cid)
        n_ran = len(box.ran)
        c.send(cid, "again")
        c.wait_idle(cid, 30)
        t2 = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][1]
        d = c.ok("GET", f"/conversations/{cid}/tasks/{t2['id']}")
        bash = [p for m in d["messages"] for p in m["parts"] if p.get("name") == "bash"][0]
        assert bash["status"] == "denied" and len(box.ran) == n_ran
    finally:
        c.close()


def test_the_worker_takes_no_run_slot_of_its_own():
    c = Chat(max_runs=2)
    try:
        slow = {"content": "x" * 400, "tps": 40}              # 80 pieces: two seconds
        cids = {}

        def script(body, h):
            if body.get("stream") is not True:
                return None
            s = hdr(h, "x-session-id") or ""
            k = sum(1 for m in since_user(body) if m.get("role") == "tool")
            if s == f"chat-{cids.get('b')}":
                return {"calls": [task_call()]} if k == 0 else {"content": "ok"}
            if "-task-" in s:
                return {"content": "y" * 600, "tps": 40} if k == 0 else {"content": REPORT}
            return slow
        c.fake.script = script
        cids["a"] = c.new()
        cids["b"] = agents_chat(c)
        cids["c"] = c.new()
        c.send(cids["a"], "long answer")
        c.wait(lambda: c.svc.conv(cids["a"]).run is not None and
               c.svc.conv(cids["a"]).run.state == "running", 10)
        c.send(cids["b"], "give a task")
        conv_b = c.svc.conv(cids["b"])
        c.wait(lambda: conv_b.run is not None and conv_b.run.child is not None and
               conv_b.run.child.state == "running", 10)
        worker = conv_b.run.child
        holders = set(c.svc.slots.holders)
        assert len(holders) == 2 and conv_b.run in holders and worker not in holders
        assert worker.slot_owner is conv_b.run
        c.send(cids["c"], "third chat")
        c.wait(lambda: (c.snap(cids["c"])["run"] or {}).get("now") ==
               "Waiting for a free run slot", 10)
        for k in ("a", "b", "c"):
            c.wait_idle(cids[k], 30)
        assert c.snap(cids["c"])["messages"][-1]["status"] == "done"
        assert not c.svc.slots.holders
    finally:
        c.close()


def test_workers_have_their_own_sessions_and_headers():
    c = Chat()
    try:
        c.fake.script = Script(lambda b, k: {"calls": [task_call(f"Task {k}")]} if k < 2 else
                               {"content": "two done"})
        cid = agents_chat(c)
        c.send(cid, "two tasks")
        c.wait_idle(cid, 40)
        sessions = [hdr(h, "x-session-id") for h, _ in worker_calls(c)]
        assert sessions == [f"chat-{cid}-task-1"] * 2 + [f"chat-{cid}-task-2"] * 2, sessions
        for h, _ in worker_calls(c):
            assert hdr(h, "X-Tandem-Worker") == "1" and hdr(h, "X-Tandem-Memory") == "off"
            assert hdr(h, "X-Tandem-Parent-Session") == f"chat-{cid}"
            assert hdr(h, "X-QSE-Client") == "chat-app"
        for h, _ in main_calls(c, cid):
            assert hdr(h, "X-Tandem-Worker") is None and hdr(h, "X-Tandem-Memory") is None
        tasks = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"]
        assert [t["session"] for t in tasks] == [f"chat-{cid}-task-1", f"chat-{cid}-task-2"]
    finally:
        c.close()


def test_task_conversations_stay_hidden_and_go_with_the_chat():
    c = Chat(titles=True)
    try:
        c.fake.script = Script(lambda b, k: {"calls": [task_call(f"Hidden {k}")]}
                               if k < 2 else {"content": "done"},
                               lambda b, k, n: {"calls": [{"name": "write_file", "arguments": {
                                   "path": "z.txt", "content": "zebra"}}]} if k == 0 else
                               {"content": REPORT})
        cid = agents_chat(c)
        c.send(cid, "two tasks please")
        c.wait_idle(cid, 40)
        tcids = [t["task_conversation_id"] for t in
                 c.ok("GET", f"/conversations/{cid}/tasks")["tasks"]]
        assert len(tcids) == 2
        assert [x["id"] for x in c.ok("GET", "/conversations")["conversations"]] == [cid]
        assert c.ok("GET", "/conversations?q=zebra")["conversations"] == []   # worker text only
        rail = " ".join(r["data"] for r in c.svc.store.all("SELECT data FROM global_events"))
        assert not any(t in rail for t in tcids)
        time.sleep(0.5)                                     # the title thread of the chat
        titles = [hdr(h, "x-session-id") for h, b in c.fake.calls
                  if (hdr(h, "x-session-id") or "").startswith("chat-title-")]
        assert all(t == f"chat-title-{cid}" for t in titles)
        for t in tcids:
            assert c.svc.store.one("SELECT title FROM conversations WHERE id = ?",
                                   (t,))["title"].startswith("Task ")
        c.ok("DELETE", f"/conversations/{cid}")
        assert all(c.svc.store.one("SELECT deleted FROM conversations WHERE id = ?",
                                   (t,))["deleted"] for t in tcids)
        c.ok("POST", f"/conversations/{cid}/restore")
        assert not any(c.svc.store.one("SELECT deleted FROM conversations WHERE id = ?",
                                       (t,))["deleted"] for t in tcids)
        c.ok("DELETE", f"/conversations/{cid}?forever=1")
        for t in tcids:
            assert c.svc.store.one("SELECT 1 FROM conversations WHERE id = ?", (t,)) is None
            assert c.svc.store.one("SELECT 1 FROM messages WHERE conversation_id = ?", (t,)) is None
        assert c.svc.store.one("SELECT 1 FROM tasks WHERE conversation_id = ?", (cid,)) is None
    finally:
        c.close()


def test_one_task_at_a_time():
    c = Chat()
    try:
        c.fake.script = Script(lambda b, k: {"calls": [task_call("First"), task_call("Second")]}
                               if k == 0 else {"content": "ok"})
        cid = agents_chat(c)
        c.send(cid, "two at once")
        s = c.wait_idle(cid, 30)
        calls = [p for p in s["messages"][-1]["parts"] if p.get("name") == "task"]
        assert calls[0]["status"] == "done"
        assert calls[1]["status"] == "failed" and calls[1]["output"]["content"] == \
            "one task at a time: call it again after you read this report"
        assert [t["title"] for t in c.ok("GET", f"/conversations/{cid}/tasks")["tasks"]] == \
            ["First"]
    finally:
        c.close()


def test_at_most_8_tasks_in_a_run():
    c = Chat()
    try:
        c.fake.script = Script(lambda b, k: {"calls": [task_call(f"Job {k}")]} if k < 9 else
                               {"content": "nine asked"})
        cid = agents_chat(c)
        c.send(cid, "nine tasks")
        s = c.wait_idle(cid, 90)
        calls = [p for p in s["messages"][-1]["parts"] if p.get("name") == "task"]
        assert len(calls) == 9 and all(p["status"] == "done" for p in calls[:8])
        assert calls[8]["status"] == "failed" and \
            "this run has given 8 tasks, its limit" in calls[8]["output"]["content"]
        assert len(c.ok("GET", f"/conversations/{cid}/tasks")["tasks"]) == 8
        assert s["messages"][-1]["status"] == "done"            # the run went on
    finally:
        c.close()


def test_a_worker_cannot_delegate():
    c = chat_with_sandbox()
    try:
        c.fake.script = Script()
        cid = agents_chat(c)
        c.send(cid, "go")
        c.wait_idle(cid, 30)
        names = [t["function"]["name"] for t in worker_calls(c)[0][1]["tools"]]
        assert "task" not in names and "todo_write" not in names and "bash" in names
        assert names == [t for t in T.WORKER_TOOLS if t in names]           # BUILTINS order
        main = [t["function"]["name"] for t in main_calls(c, cid)[0][1]["tools"]]
        assert main[-1] == "task" or "task" in main
        # plan mode keeps the read tools and the todo list: no task, Agents mode or not
        settings = {**c.svc.settings_of(cid), "plan_mode": True}
        info = c.svc.model_info(settings.get("model"))
        assert "task" not in [t.name for t in c.svc.tools_for(settings, info, cid, agents=True)]
        # no picker lists it
        assert "task" not in [t["id"] for t in c.ok("GET", "/tools")]
        assert "task" not in c.ok("GET", "/capabilities")["defaults"]["tools"]
    finally:
        c.close()


def test_a_worker_compacts_its_own_conversation():
    c = Chat()
    try:
        small_window(c, window=12_000)
        c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
        cid = agents_chat(c)
        ws = c.svc.workspace(cid)
        ws.write("big.txt", "".join(f"line {i} of the big file with some words\n"
                                    for i in range(2000)).encode())

        def worker(b, k, n):
            if k < 8:
                return {"calls": [{"name": "read_file", "arguments": {
                    "path": "big.txt", "offset": 1 + 100 * k, "limit": 100}}]}
            return {"content": REPORT}
        c.fake.script = Script(worker=worker)
        c.svc.conv(cid).tok_ratio = 0.5
        c.send(cid, "read it all")
        s = c.wait_idle(cid, 60)
        tcid = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]["task_conversation_id"]
        rows = c.svc.store.all("SELECT conversation_id, boundary_id FROM compactions")
        assert rows and all(r["conversation_id"] == tcid for r in rows), [dict(r) for r in rows]
        assert "@" in rows[0]["boundary_id"]
        first = worker_calls(c)[0][1]["messages"][1]["content"]
        after = [b for _, b in worker_calls(c) if "Summary of the earlier part" in
                 b["messages"][0]["content"]]
        assert after and all(b["messages"][1] == {"role": "user", "content": first}
                             for b in after)
        assert s["compactions"] == []                        # the chat's own: none
        assert task_part(s)["status"] == "done"
    finally:
        c.close()


def test_the_events_and_the_endpoints():
    c = Chat()
    try:
        c.fake.script = Script()
        cid = agents_chat(c)
        ev = c.events(cid)
        c.send(cid, "go")
        frames = ev.until(lambda f: f[1] == "run.done", 30)
        ev.close()
        kinds = [f[1] for f in frames]
        evs = [{**json.loads(f[2] or "{}"), "type": f[1]} for f in frames]
        started = next(e for e in evs if e["type"] == "task.started")
        tid = started["task"]["id"]
        assert started["task"]["n"] == 1 and started["task"]["status"] == "running"
        assert kinds.index("task.started") < kinds.index("task.done")
        phases = [e for e in evs if e["type"] == "task.phase"]
        assert phases and all(e["task_id"] == tid and e["max_steps"] == 40 for e in phases)
        nows = [e["now"] for e in evs if e["type"] == "run.phase" and e["phase"] == "task"]
        assert any(n.startswith("Task 1: Hello file, call 1 of 40: ") for n in nows), nows
        worker_parts = [e for e in evs if e["type"].startswith(("part.", "tool.", "message.")) and
                        e.get("task_id")]
        assert worker_parts and all(e["task_id"] == tid for e in worker_parts)
        assert {"message.created", "part.added", "tool.started", "tool.done"} <= \
            {e["type"] for e in worker_parts}
        done = next(e for e in evs if e["type"] == "task.done")
        assert done["task_id"] == tid and done["status"] == "done"
        assert done["stats"]["steps"] == 2 and done["stats"]["completion_tokens"] > 0
        # the worker's own run events stay inside chatd
        run_ids = {e["run"]["id"] for e in evs if e["type"] == "run.started"}
        assert len(run_ids) == 1
        lst = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"]
        assert [t["id"] for t in lst] == [tid] and lst[0]["status"] == "done"
        d = c.ok("GET", f"/conversations/{cid}/tasks/{tid}")
        assert d["task"]["id"] == tid and int(d["seq"]) > 0
        roles = [m["role"] for m in d["messages"]]
        assert roles == ["user", "assistant"]
        call = next(p for p in d["messages"][1]["parts"] if p["type"] == "tool_call")
        assert json.loads(call["input"])["path"] == "hello.txt" and \
            call["output"]["content"].startswith("wrote /workspace/hello.txt")
        st, _, _ = c.call("GET", f"/conversations/{cid}/tasks/nope")
        assert st == 404
        # the parent's turn counts its worker
        a = c.snap(cid)["messages"][-1]
        assert a["stats"]["workers"]["count"] == 1 and a["stats"]["workers"]["prompt_tokens"] > 0
        assert c.snap(cid)["context"]["worker"] is None
    finally:
        c.close()


def test_the_schema_step_and_its_rollback():
    d = tempfile.mkdtemp(prefix="chatd-schema-")
    path = os.path.join(d, "chat.sqlite3")
    old = {k: v for k, v in MIGRATIONS.items() if k < 8}
    s7 = Store(path, migrations=old)
    assert s7.schema_version == 7
    s7.run("INSERT INTO conversations (id, title, created, updated, settings, ui) VALUES "
           "('c1', 't', 1, 1, '{}', '{}')")
    s7.db.close()
    s8 = Store(path)
    assert s8.migrated == [8] and s8.schema_version == 8
    assert s8.one("SELECT kind, parent_id FROM conversations WHERE id = 'c1'")[:] == ("chat", None)
    s8.run("INSERT INTO tasks (id, conversation_id, n, title, brief, expect, task_conversation_id, "
           "status) VALUES ('t1', 'c1', 1, 't', 'b', 'e', 'tc1', 'running')")
    try:
        s8.run("INSERT INTO tasks (id, conversation_id, n, title, brief, expect, "
               "task_conversation_id, status) VALUES ('t2', 'c1', 1, 't', 'b', 'e', 'tc2', 'x')")
        raise AssertionError("a second task 1 in the chat")
    except Exception as exc:                                    # noqa: BLE001
        assert "UNIQUE" in str(exc)
    s8.db.close()
    try:
        Store(path, migrations=old)
        raise AssertionError("a v7 chatd read a v8 file")
    except StoreError as exc:
        assert "newer than this chatd" in str(exc)


def test_agents_off_requests_are_byte_identical_to_main():
    with open(os.path.join(ROOT, "tests", "fixtures", "chatd_replay_main.json")) as fh:
        want = json.load(fh)
    c = chatd_replay.chat()
    try:
        got = chatd_replay.build(c.svc)
    finally:
        c.close()
    assert sorted(got) == sorted(want)
    for name in want:
        assert got[name]["headers"] == want[name]["headers"], name
        assert got[name]["body"] == want[name]["body"], name
    assert render.headers("x") == {"x-session-id": "chat-x", "X-QSE-Client": "chat-app",
                                   "X-Tandem-Cost": "1"}


if __name__ == "__main__":
    run_all(globals())
