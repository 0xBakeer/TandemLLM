"""chatd phase 2 (SRV-351, SRV-353, SRV-355, SRV-357): the agent loop, tools, permissions, the
context guard and compaction, the workspace, snapshots, the egress filter, the memory guard and
the terminals. CPU only: the model is tests/fake_backend.py (`FAKE_CALLS` makes it call our tools),
the sandbox a fake with the runtime's interface (the real one is the Spark's box test).

Run: python -m pytest -q tests/test_chatd_agent.py
"""

from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
import threading
import time

from chatd_util import KEY, Chat, is_done, run_all, text_of

from chatd import egress, sandbox as sb_mod, tools as T
from chatd.snapshots import Snapshots
from chatd.workspace import ChangedError, PathError, Workspace, etag


def calls(*c) -> str:
    return "FAKE_CALLS " + json.dumps([{"name": n, "arguments": a} for n, a in c])


def parts_of(snap, role="assistant"):
    return [p for m in snap["messages"] if m["role"] == role for p in m["parts"]]


def requests(c):
    return [b for b in c.fake.bodies if "model" in b and b.get("stream")]


# ---------------------------------------------------------------------- a fake sandbox runtime
class FakeBoxes(sb_mod.Sandboxes):
    """The runtime's interface with the container engine replaced: `run` answers from `script`,
    `open_exec` gives a socketpair whose far end echoes (a fake PTY)."""

    def __init__(self, data):
        super().__init__(data, engine=None, log=lambda line: None)
        self.script = {}
        self.ran = []

    def available(self):
        return True

    def _start(self, cid):
        for d in self.paths(cid):
            os.makedirs(d, exist_ok=True)
        return "fake-" + cid

    def _proxy(self, b):
        pass

    def stop(self, cid, reason="user"):
        b = self.box(cid)
        b.state, b.stopped_reason = "stopped", reason
        self._changed(b)

    def run(self, cid, cmd, *, timeout=600, on_output=None, cancel=None):
        self.ran.append(cmd)
        line = cmd[-1]
        for key, (out, code, delay) in self.script.items():
            if key in line:
                for chunk in out:
                    if cancel is not None and cancel.wait(delay):
                        return -15, False
                    on_output and on_output(chunk)
                if delay * len(out) > timeout:
                    return -9, True
                return code, False
        return 0, False

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        a, b = socket.socketpair()

        def echo():
            b.sendall(b"sandbox:/workspace$ ")
            while True:
                try:
                    d = b.recv(4096)
                except OSError:
                    return
                if not d:
                    return
                b.sendall(d)
        threading.Thread(target=echo, daemon=True).start()
        return "exec-1", a, b""


def chat_with_sandbox():
    c = Chat()
    c.svc.sandboxes = FakeBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    c.svc._caps = None
    return c


# ---------------------------------------------------------------------- the loop
def test_a_tool_round_writes_a_file_and_the_model_reads_the_result():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, calls(("write_file", {"path": "notes/a.txt", "content": "hello"})))
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        call = next(p for p in a["parts"] if p["type"] == "tool_call")
        assert call["name"] == "write_file" and call["status"] == "done", call
        assert "wrote /workspace/notes/a.txt (5 bytes)" in call["output"]["content"]
        assert open(os.path.join(c.data, "workspaces", cid, "notes", "a.txt")).read() == "hello"
        assert a["status"] == "done" and "done0" in text_of(a)
        assert a["stats"]["steps"] == 2
        second = requests(c)[-1]["messages"]
        assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"][
            "name"] == "write_file"
        assert second[-1]["role"] == "tool" and "wrote /workspace/notes/a.txt" in \
            second[-1]["content"]
        assert [t["function"]["name"] for t in requests(c)[0]["tools"]][:3] == \
            ["read_file", "write_file", "edit_file"]
        assert requests(c)[0]["parallel_tool_calls"] is True
        assert "diff" not in json.dumps(c.ok("GET", f"/conversations/{cid}/sandbox/diff"))[:0]
        d = c.ok("GET", f"/conversations/{cid}/sandbox/diff?run_id={a['run_id']}")
        assert d["files"][0]["path"] == "/workspace/notes/a.txt" and d["files"][0]["status"] == \
            "added"
    finally:
        c.close()


def test_an_unknown_tool_and_bad_arguments_are_results_not_crashes():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, calls(("rm_everything", {}), ("read_file", {"path": "../../etc/passwd"})))
        s = c.wait_idle(cid, 20)
        p = [x for x in parts_of(s) if x["type"] == "tool_call"]
        assert p[0]["status"] == "failed" and "not available" in p[0]["output"]["content"]
        assert p[1]["status"] == "failed" and "not allowed" in p[1]["output"]["content"]
        assert s["messages"][-1]["status"] == "done"
    finally:
        c.close()


def test_read_calls_run_in_parallel_and_results_keep_order():
    c = Chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        for i in range(3):
            ws.write(f"f{i}.txt", f"file {i}\n".encode())
        c.send(cid, calls(*[("read_file", {"path": f"f{i}.txt"}) for i in range(3)]))
        c.wait_idle(cid, 20)
        tool_msgs = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"]
        assert [("file %d" % i) in m["content"] for i, m in enumerate(tool_msgs)] == [True] * 3
    finally:
        c.close()


def test_the_step_cap_ends_interrupted_and_continue_goes_on():
    c = Chat()
    try:
        cid = c.new()
        c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": 1})
        c.send(cid, calls(("glob", {"pattern": "**/*"})))
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "interrupted" and "limit of 1 model call" in a["parts"][-1]["message"]
        c.ok("POST", f"/conversations/{cid}/continue")
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "done" and "done0" in text_of(a)
        assert len(s["messages"]) == 2
    finally:
        c.close()


def test_the_same_call_three_times_stops_the_run():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, calls(("glob", {"pattern": "*.py"})) + " FAKE_CALLS_ALWAYS")
        s = c.wait_idle(cid, 30)
        a = s["messages"][-1]
        assert [p["type"] for p in a["parts"]].count("tool_call") == 3
        assert "three times" in a["parts"][-1]["message"] and a["status"] == "done"
    finally:
        c.close()


def test_an_empty_turn_is_retried_once_with_a_nudge_then_an_error():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_EMPTY_ANSWER hello")              # thinking, then nothing
        s = c.wait_idle(cid, 20)
        reqs = requests(c)
        assert len(reqs) == 2 and reqs[-1]["messages"][-1]["content"].startswith("(Your last turn")
        assert s["messages"][-1]["status"] == "done"       # the fake answers the nudge
        assert "Your last turn" not in json.dumps(c.snap(cid))       # the nudge is not stored
        c.svc.step_empty = lambda run: True                 # a model that stays empty
        c.send(cid, "again")
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "error"
        assert "nothing, twice" in s["messages"][-1]["parts"][-1]["message"]
        assert len(requests(c)) == 4
    finally:
        c.close()


def test_a_steer_queued_during_the_tools_goes_in_after_the_results():
    c = chat_with_sandbox()
    try:
        c.svc.sandboxes.script["sleep"] = (["working\n"] * 10, 0, 0.1)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "sleep 1"})))
        c.wait(lambda: any(p.get("status") == "running" and p["type"] == "tool_call"
                           for p in parts_of(c.snap(cid))), 10)
        c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "s1", "text": "also this"})
        c.wait(lambda: len(c.snap(cid)["messages"]) == 4 and c.snap(cid)["run"] is None, 30)
        sent = requests(c)[-1]["messages"]
        roles = [m["role"] for m in sent if m["role"] != "system"]
        assert roles[-3:] == ["assistant", "tool", "user"] and sent[-1]["content"] == "also this"
    finally:
        c.close()


# ---------------------------------------------------------------------- permissions
def test_a_prompt_waits_two_devices_see_it_and_deny_with_a_note_reaches_the_model():
    c = Chat()
    try:
        cid = c.new()
        a = c.events(cid)
        b = c.events(cid)
        for x in (a, b):
            x.until(lambda f: f[1] == "current")
        c.send(cid, calls(("web_fetch", {"url": "http://example.org/x"})))
        asked = [x.until(lambda f: f[1] == "tool.asking")[-1] for x in (a, b)]
        ev = json.loads(asked[0][2])
        assert ev["summary"] == "Fetch http://example.org/x" and ev["risk"] == "network"
        assert ev["suggested_rule"] == {"tool": "web_fetch", "match": "example.org"}
        s = c.snap(cid)
        assert s["status"] == "waiting" and s["run"]["phase"] == "permission"
        time.sleep(1.5)                                 # a prompt does not time out
        assert c.snap(cid)["run"]["phase"] == "permission"
        call_id = ev["call_id"]
        c.ok("POST", f"/conversations/{cid}/permissions/{call_id}",
             {"decision": "deny", "note": "use the cached copy"})
        st, obj, _ = c.call("POST", f"/conversations/{cid}/permissions/{call_id}",
                            {"decision": "once"})
        assert st == 409                                # the second device was too late
        for x in (a, b):
            done = x.until(lambda f: f[1] == "tool.done")[-1]
            assert json.loads(done[2])["status"] == "denied"
            x.close()
        c.wait_idle(cid, 20)
        tool_msg = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"][0]
        assert tool_msg["content"] == "The user denied this call. Their note: use the cached copy"
    finally:
        c.close()


def test_a_run_waiting_for_permission_gives_its_run_slot_back():
    # live (rc-0.8.0, 2026-10-13): a chat waiting for a bash prompt held one of the two run slots,
    # and two other chats ran one after the other
    c = Chat(max_runs=2)
    try:
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("web_fetch", {"url": "http://example.org/x"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        ids = [c.new() for _ in range(2)]
        for x in ids:
            c.send(x, "FAKE_SLOW FAKE_LONG 20 x")
        c.wait(lambda: c.fake.max_streams == 2, 10)      # both at once beside the waiting chat
        for x in ids:
            c.wait_idle(x, 20)
            assert "free run slot" not in ((c.snap(x)["run"] or {}).get("now") or "")
        assert c.snap(cid)["run"]["phase"] == "permission"
        # the answer: the run takes a slot again for its next call, and ends
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}", {"decision": "deny"})
        c.wait_idle(cid, 20)
        assert is_done(sse.until(is_done)[-1])
        sse.close()
    finally:
        c.close()


def test_always_allow_writes_a_rule_and_the_next_call_runs_without_asking():
    c = Chat()
    try:
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("web_fetch", {"url": "http://192.168.1.1/admin"})))
        ev = json.loads(sse.until(lambda f: f[1] == "tool.asking")[-1][2])
        c.ok("POST", f"/conversations/{cid}/permissions/{ev['call_id']}",
             {"decision": "always", "scope": "chat"})
        c.wait_idle(cid, 20)
        call = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"][0]
        assert call["status"] == "failed" and "local network address" in call["output"]["content"]
        rules = c.ok("GET", "/permissions/rules")
        assert rules[0]["tool"] == "web_fetch" and rules[0]["match"] == "192.168.1.1"
        assert rules[0]["scope"] == "chat"
        c.send(cid, calls(("web_fetch", {"url": "http://192.168.1.1/other"})))
        c.wait_idle(cid, 20)
        frames = sse.until(is_done)
        assert not any(f[1] == "tool.asking" for f in frames)
        sse.close()
        c.call("DELETE", f"/permissions/rules/{rules[0]['id']}")
        assert c.ok("GET", "/permissions/rules") == []
    finally:
        c.close()


def test_rule_matching():
    bash = T.BY_NAME["bash"]
    allow_npm = [{"tool": "bash", "match": "npm install", "decision": "allow"}]
    assert T.decide(bash, {"command": "npm install three"}, allow_npm, [], {}) == "allow"
    assert T.decide(bash, {"command": "npm installx three"}, allow_npm, [], {}) == "ask"
    assert T.decide(bash, {"command": "npm install a && rm -rf /"}, allow_npm, [], {}) == "ask"
    assert T.decide(bash, {"command": "npm install $(curl x)"}, allow_npm, [], {}) == "ask"
    both = allow_npm + [{"tool": "bash", "match": "git status", "decision": "allow"}]
    assert T.decide(bash, {"command": "npm install && git status"}, both, [], {}) == "allow"
    deny = [{"tool": "bash", "match": "npm", "decision": "deny"}]
    assert T.decide(bash, {"command": "npm install x"}, allow_npm + deny, [], {}) == "deny"
    # chat rules before global ones
    assert T.decide(bash, {"command": "npm install x"}, allow_npm,
                    [{"tool": "bash", "match": "npm", "decision": "deny"}], {}) == "allow"
    assert T.decide(bash, {"command": "ls"}, [], [], {"policies": {"bash": "always"}}) == "allow"
    w = T.BY_NAME["write_file"]
    assert T.decide(w, {}, [], [], {"policies": {"write_file": "ask"}}) == "ask"
    assert T.decide(w, {}, [], [], {}) == "allow"
    mcp = T.Tool("mcp__x1__search", "", {}, default="ask")
    assert T.decide(mcp, {}, [], [{"tool": "mcp__x1__*", "decision": "allow"}], {}) == "allow"
    assert T.bash_prefix("npm install three") == "npm install"
    assert T.bash_prefix("python3 app.py") == "python3"
    assert T.bash_prefix("ls -la && rm x") == "ls"
    wf = T.BY_NAME["web_fetch"]
    rule = [{"tool": "web_fetch", "match": "example.org", "decision": "allow"}]
    assert T.decide(wf, {"url": "https://docs.example.org/a"}, rule, [], {}) == "allow"
    assert T.decide(wf, {"url": "https://example.org.evil.com/"}, rule, [], {}) == "ask"


# ---------------------------------------------------------------------- bash in the sandbox
def test_bash_streams_its_output_and_a_stop_cancels_it():
    c = chat_with_sandbox()
    try:
        c.svc.sandboxes.script["pytest"] = (["collected 2\n", "..\n", "2 passed\n"], 0, 0.05)
        c.svc.sandboxes.script["sleep"] = (["tick\n"] * 100, 0, 0.1)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("bash", {"command": "pytest -q"})))
        frames = sse.until(is_done, 20)
        outs = [json.loads(f[2])["delta"] for f in frames if f[1] == "tool.output"]
        assert "".join(outs) == "collected 2\n..\n2 passed\n"
        call = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"][0]
        assert call["status"] == "done" and call["output"]["exit_code"] == 0
        assert c.snap(cid)["sandbox"]["state"] == "running"
        # a long command, then Stop
        c.send(cid, calls(("bash", {"command": "sleep 100"})))
        c.wait(lambda: any(f[1] == "tool.started" for f in [sse.frames.get(timeout=10)]), 10)
        time.sleep(0.4)
        c.ok("POST", f"/conversations/{cid}/stop")
        s = c.wait_idle(cid, 20)
        call = [p for p in parts_of(s) if p["type"] == "tool_call"][-1]
        assert call["status"] == "cancelled" and s["messages"][-1]["status"] == "stopped"
        sse.close()
    finally:
        c.close()


def test_a_long_output_is_cut_for_the_model_and_saved_whole():
    c = chat_with_sandbox()
    try:
        big = "x" * 100_000 + "\n"
        c.svc.sandboxes.script["cat big"] = ([big], 0, 0)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "cat big"})))
        c.wait_idle(cid, 20)
        tool_msg = [m for m in requests(c)[-1]["messages"] if m["role"] == "tool"][0]["content"]
        assert len(tool_msg) < 40_000 and "the whole output is in /workspace/.tandem/outputs/" \
            in tool_msg
        call = [p for p in parts_of(c.snap(cid)) if p["type"] == "tool_call"][0]
        saved = os.path.join(c.data, "workspaces", cid, ".tandem", "outputs", call["id"] + ".txt")
        assert os.path.getsize(saved) == len(big)
        assert call["output"]["truncated"] is True and len(call["output"]["content"]) <= 30_010
    finally:
        c.close()


# ---------------------------------------------------------------------- the context guard
def small_window(c, window=3000, backend="engine"):
    real = c.svc.model_info

    def info(mid):
        out = dict(real(mid))
        out.update(max_len=window, backend=backend, max_output=None)
        return out
    c.svc.model_info = info


def test_compaction_summarizes_older_turns_and_the_model_gets_the_summary():
    c = Chat()
    try:
        small_window(c, window=5000)
        c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
        cid = c.new()
        c.svc.conv(cid).tok_ratio = 0.5
        long = "lorem ipsum dolor " * 120                          # ~2,200 characters a turn
        for i in range(3):
            c.send(cid, f"turn {i}: {long}")
            c.wait_idle(cid, 20)
        c.send(cid, "and now?")
        s = c.wait_idle(cid, 30)
        assert len(s["compactions"]) >= 1
        cp = s["compactions"][-1]
        assert cp["summary"].startswith("w0") and cp["before_tokens"] > 0
        summ = [b for b in c.fake.bodies if b.get("stream") is False and
                b.get("tandem", {}).get("compaction")]
        assert summ and summ[-1]["messages"][-1]["content"].startswith("Write a summary")
        last = requests(c)[-1]
        assert last["tandem"] == {"compaction": False}
        system = last["messages"][0]
        assert system["role"] == "system" and "Summary of the earlier part" in system["content"]
        users = [m["content"] for m in last["messages"] if m["role"] == "user"]
        assert len(users) == 2 and users[-1] == "and now?" and users[0].startswith("turn 2")
        assert last["max_tokens"] < 5000
        assert len(s["messages"]) == 8                          # the store keeps every message
        # the thread's divider sits above the first message the model still reads in full
        kept = next(m for m in s["messages"] if m["id"] == cp["boundary_id"])
        assert kept["role"] == "user" and kept["parts"][0]["text"].startswith("turn 2")
    finally:
        c.close()


def test_a_request_that_cannot_fit_fails_before_the_engine_is_asked():
    c = Chat()
    try:
        small_window(c, window=1200)
        cid = c.new()
        n = len(requests(c))
        c.send(cid, "word " * 2000)
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "error" and "window is 1,200" in a["parts"][-1]["message"]
        assert len(requests(c)) == n
    finally:
        c.close()


def test_earlier_reasoning_is_left_out_when_nothing_else_helps():
    c = Chat()
    try:
        small_window(c, window=2600)
        cid = c.new()
        c.send(cid, "FAKE_REASONING FAKE_LONG 6 " + "w " * 300)
        c.wait_idle(cid, 20)
        c.svc.conv(cid).tok_ratio = 0.3
        # make the first answer's reasoning big in the store
        conv = c.svc.conv(cid)
        mid = c.snap(cid)["messages"][1]["id"]
        row = c.svc.store.one("SELECT parts FROM messages WHERE id = ?", (mid,))
        parts = json.loads(row[0])
        parts[0]["text"] = "thinking " * 900
        c.svc.store.run("UPDATE messages SET parts = ? WHERE id = ?", (json.dumps(parts), mid))
        c.send(cid, "next")
        s = c.wait_idle(cid, 20)
        assert s["messages"][-1]["status"] == "done", s["messages"][-1]
        last = requests(c)[-1]
        assert not any("reasoning_content" in m for m in last["messages"])
        assert conv is c.svc.conv(cid)
    finally:
        c.close()


# ---------------------------------------------------------------------- the workspace
def test_workspace_paths_refuse_escapes():
    d = tempfile.mkdtemp()
    ws = Workspace(d)
    ws.ensure()
    ws.write("a/b.txt", b"hi")
    assert ws.read("/workspace/a/b.txt") == b"hi" and ws.read("a/b.txt") == b"hi"
    for bad in ("../x", "/etc/passwd", "a/../../x", "/workspace/../etc"):
        try:
            ws.resolve(bad)
            raise AssertionError(bad)
        except PathError:
            pass
    outside = tempfile.mkdtemp()
    with open(os.path.join(outside, "secret"), "w") as fh:
        fh.write("no")
    os.symlink(outside, os.path.join(d, "link"))
    os.symlink(os.path.join(outside, "secret"), os.path.join(d, "s.txt"))
    for bad in ("link/secret", "s.txt"):
        try:
            ws.read(bad)
            raise AssertionError(bad)
        except (PathError, OSError):
            pass
    assert "/workspace/link" not in ws.glob("**/*")
    e = etag(b"hi")
    ws.write("a/b.txt", b"new", if_match=e)
    try:
        ws.write("a/b.txt", b"x", if_match=e)
        raise AssertionError("a stale etag must fail")
    except ChangedError:
        pass
    assert ws.grep("ne.") == ["/workspace/a/b.txt:1: new"]


def test_the_files_api():
    c = Chat()
    try:
        cid = c.new()
        base = f"/conversations/{cid}/sandbox"
        st, _, _ = c.call("PUT", f"{base}/file?path=/workspace/src/app.py",
                          raw=b"print('hi')\n", headers={"Content-Type": "text/plain"})
        assert st == 204
        f = c.ok("GET", f"{base}/file?path=/workspace/src/app.py")
        assert f["content"] == "print('hi')\n" and f["etag"] == etag(b"print('hi')\n")
        st, _, _ = c.call("PUT", f"{base}/file?path=src/app.py", raw=b"x",
                          headers={"If-Match": "stale", "Content-Type": "text/plain"})
        assert st == 412
        assert [x["path"] for x in c.ok("GET", f"{base}/files?path=/workspace")] == \
            ["/workspace/src"]
        st, obj, _ = c.call("GET", f"{base}/file?path=../../x")
        assert st == 400
        boundary = "B"
        raw = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"up.txt"
               f"\"\r\nContent-Type: text/plain\r\n\r\nuploaded\r\n--{boundary}--\r\n").encode()
        up = c.ok("POST", f"{base}/upload?dir=/workspace/in", raw=raw,
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert up[0]["path"] == "/workspace/in/up.txt"
        st, body, h = c.call("GET", f"{base}/view/src/app.py")
        assert st == 200 and "frame-ancestors 'self'" in h["Content-Security-Policy"]
        a = c.ok("GET", f"/conversations/{cid}/artifacts/" + "file%3A%2Fworkspace%2Fsrc%2Fapp.py")
        assert a["kind"] == "file" and a["content"] == "print('hi')\n"
        st, _, _ = c.call("DELETE", f"{base}/file?path=in/up.txt")
        assert st == 204
    finally:
        c.close()


def test_snapshots_diff_and_undo_with_renames_and_deletes():
    d = tempfile.mkdtemp()
    ws = Workspace(os.path.join(d, "ws"))
    ws.ensure()
    ws.write("keep.txt", b"one\ntwo\n")
    ws.write("old_name.py", b"x = 1\n")
    ws.write("gone.txt", b"bye\n")
    snaps = Snapshots(os.path.join(d, "snaps"))
    snaps.take("c1", ws, "r1", "before")
    ws.write("keep.txt", b"one\nTWO\n")
    ws.write("new_name.py", ws.read("old_name.py"))
    ws.delete("old_name.py")
    ws.delete("gone.txt")
    snaps.take("c1", ws, "r1", "after")
    files = {f["path"]: f for f in snaps.diff("c1", "r1")["files"]}
    assert files["/workspace/keep.txt"]["status"] == "modified"
    assert files["/workspace/keep.txt"]["additions"] == 1 and \
        files["/workspace/keep.txt"]["deletions"] == 1
    assert files["/workspace/new_name.py"]["status"] == "added"
    assert files["/workspace/old_name.py"]["status"] == "deleted"
    assert files["/workspace/gone.txt"]["status"] == "deleted"
    n = snaps.undo("c1", ws, "r1", "u1")
    assert n == 4
    assert ws.read("keep.txt") == b"one\ntwo\n" and ws.read("old_name.py") == b"x = 1\n"
    assert ws.read("gone.txt") == b"bye\n"
    try:
        ws.read("new_name.py")
        raise AssertionError("the added file must be gone")
    except (PathError, OSError):
        pass
    snaps.undo("c1", ws, "u1", "u2")                      # the undo can be undone
    assert ws.read("keep.txt") == b"one\nTWO\n" and ws.read("new_name.py") == b"x = 1\n"


# ---------------------------------------------------------------------- the egress filter
def test_the_egress_filter_refuses_the_lan_and_this_box():
    for bad in ("10.0.0.1", "192.168.1.131", "172.20.0.1", "127.0.0.1", "169.254.1.1",
                "100.64.0.1", "::1", "fd00::1", "fe80::1", "::ffff:192.168.1.1", "0.0.0.0",
                "224.0.0.1"):
        assert not egress.allowed(bad), bad
    for good in ("1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"):
        assert egress.allowed(good), good
    for host in ("localhost", "127.0.0.1"):
        try:
            egress.resolve(host, 80)
            raise AssertionError(host)
        except egress.Refused:
            pass
    d = tempfile.mkdtemp()
    mode = {"m": "on"}
    p = egress.Proxy(os.path.join(d, "proxy.sock"), lambda: mode["m"])
    p.start()
    try:
        def ask(req: bytes) -> bytes:
            s = socket.socket(socket.AF_UNIX)
            s.connect(p.path)
            s.sendall(req)
            out = b""
            while True:
                x = s.recv(4096)
                if not x:
                    break
                out += x
            s.close()
            return out
        assert ask(b"CONNECT 192.168.1.1:443 HTTP/1.1\r\nHost: x\r\n\r\n").startswith(
            b"HTTP/1.1 403")
        assert b"local network" in ask(b"GET http://127.0.0.1:8000/v1/models HTTP/1.1\r\n"
                                       b"Host: 127.0.0.1\r\n\r\n")
        mode["m"] = "off"
        assert b"network is off" in ask(b"CONNECT example.org:443 HTTP/1.1\r\n\r\n")
        assert p.stats["refused"] == 3
    finally:
        p.close()


# ---------------------------------------------------------------------- the memory guard
class FakeEngineAPI:
    def __init__(self):
        self.calls = []

    def call(self, method, path, body=None, timeout=60):
        self.calls.append((method, path))
        if path.startswith("/images/"):
            return 200, {}
        if path.endswith("/json") and "containers" in path:
            return 404, None
        if path.startswith("/containers/create"):
            return 201, {"Id": "ctr-" + str(len(self.calls))}
        if path.endswith("/start"):
            return 204, None
        if path.endswith("/exec"):
            return 201, {"Id": "e"}
        return 204, None

    def hijack(self, path, body):
        a, b = socket.socketpair()
        b.close()
        return a, b""


def test_the_memory_guard_admits_stops_and_kills_by_the_numbers():
    d = tempfile.mkdtemp()
    mem = {"avail": 60}
    clock = {"t": 1000.0}
    api = FakeEngineAPI()
    sb = sb_mod.Sandboxes(d, engine=api, meminfo_fn=lambda: {"MemAvailable": int(
        mem["avail"] * sb_mod.GiB)}, clock=lambda: clock["t"])
    sb._proxy = lambda b: None
    sb.run = lambda *a, **k: (0, False)
    sb.min_start_gb = 30
    sb.max_running = 1
    assert sb.admit() is None
    mem["avail"] = 25
    assert "30 GiB start line" in sb.admit()
    mem["avail"] = 22                                     # 22 - 4 = 18 < 19
    sb.min_start_gb = 0
    assert "leaves the engine under 19" in sb.admit()
    mem["avail"] = 60
    sb.ensure("c1")
    assert sb.box("c1").state == "running"
    sb.ensure("c2")                                       # the cap: the idle c1 makes room
    assert sb.box("c1").state == "stopped" and sb.box("c2").state == "running"
    assert sb.box("c1").stopped_reason == "idle"
    mem["avail"] = 14
    assert sb.guard_once() == "stop c2" and sb.box("c2").stopped_reason == "memory_guard"
    mem["avail"] = 60
    sb.ensure("c3")
    mem["avail"] = 11
    assert sb.guard_once() == "kill" and sb.box("c3").state == "stopped"
    mem["avail"] = 60
    sb.ensure("c4")
    clock["t"] += 31 * 60
    assert sb.guard_once() == "idle c4"
    cancel = threading.Event()
    mem["avail"] = 20
    threading.Timer(0.3, cancel.set).start()
    try:
        sb.ensure("c5", cancel=cancel)
        raise AssertionError("must wait, then give up when stopped")
    except sb_mod.Unavailable:
        assert sb.box("c5").state == "stopped"


# ---------------------------------------------------------------------- terminals
def ws_open(port, path):
    s = socket.create_connection(("127.0.0.1", port))
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
              f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nX-Tandem-Forward: {KEY}"
              f"\r\n\r\n".encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(1)
    assert head.startswith(b"HTTP/1.1 101"), head
    return s


def ws_send(s, data: bytes, op=2):
    mask = os.urandom(4)
    n = len(data)
    head = bytes([0x80 | op]) + (bytes([0x80 | n]) if n < 126 else
                                 bytes([0x80 | 126]) + n.to_bytes(2, "big"))
    s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))


def ws_recv(s, timeout=5.0) -> bytes:
    s.settimeout(timeout)
    out = b""
    try:
        while True:
            h = s.recv(2)
            if len(h) < 2:
                break
            n = h[1] & 0x7F
            if n == 126:
                n = int.from_bytes(s.recv(2), "big")
            data = b""
            while len(data) < n:
                data += s.recv(n - len(data))
            out += data
            s.settimeout(0.3)
    except socket.timeout:
        pass
    return out


def test_a_terminal_keeps_its_scrollback_and_a_second_attach_sees_it():
    c = chat_with_sandbox()
    try:
        cid = c.new()
        s1 = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/terminals/shell/ws")
        assert b"sandbox:/workspace$" in ws_recv(s1)
        ws_send(s1, b"echo first\n")
        assert b"echo first" in ws_recv(s1)
        s1.close()
        s2 = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/terminals/shell/ws")
        back = ws_recv(s2)
        assert b"sandbox:/workspace$" in back and b"echo first" in back
        ws_send(s2, json.dumps({"type": "resize", "cols": 100, "rows": 30}).encode(), op=1)
        s2.close()
        assert c.ok("GET", f"/conversations/{cid}/terminals")[0] == {
            "session": "shell", "title": "Terminal", "owner": "user", "alive": True}
    finally:
        c.close()


def test_an_agent_call_is_a_read_only_terminal():
    c = chat_with_sandbox()
    try:
        c.svc.sandboxes.script["make"] = (["building\n", "done\n"], 0, 0)
        c.ok("PATCH", "/tools/bash", {"policy": "always"})
        cid = c.new()
        c.send(cid, calls(("bash", {"command": "make all"})))
        s = c.wait_idle(cid, 20)
        call = [p for p in parts_of(s) if p["type"] == "tool_call"][0]
        sock = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/terminals/{call['id']}/ws")
        out = ws_recv(sock)
        assert b"$ make all" in out and b"building\r\ndone\r\n" in out
        sock.close()
        agent = [t for t in c.ok("GET", f"/conversations/{cid}/terminals")
                 if t["owner"] == "agent"]
        assert agent and agent[0]["session"] == call["id"]
    finally:
        c.close()


def test_present_opens_a_pane_on_every_device():
    c = Chat()
    try:
        cid = c.new()
        c.svc.workspace(cid).write("site/index.html", b"<h1>hi</h1>")
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, calls(("present", {"kind": "preview", "path": "site/index.html"})))
        ev = json.loads(sse.until(lambda f: f[1] == "pane.open")[-1][2])
        assert ev["pane"] == {"kind": "preview", "ref": "file:/workspace/site/index.html",
                              "title": None}
        sse.close()
        a = c.ok("GET", f"/conversations/{cid}/artifacts/file%3A%2Fworkspace%2Fsite%2Findex.html")
        assert a["kind"] == "preview" and a["url"].endswith("/sandbox/view/site/index.html")
    finally:
        c.close()


def test_an_empty_message_never_reaches_the_engine():
    """2026-10-12: a client sent `content` instead of `text`; chatd took it as an empty message
    and the engine got a user turn with nothing in it (and answered something unrelated)."""
    c = Chat()
    try:
        cid = c.new()
        st, err, _ = c.call("POST", f"/conversations/{cid}/messages",
                            {"client_msg_id": "e1", "content": "Say hello in three words."})
        assert st == 400 and err["error"]["type"] == "empty_message"
        assert "`text`, not `content`" in err["error"]["message"]
        for body in ({"client_msg_id": "e2", "text": "   "}, {"client_msg_id": "e3"}):
            assert c.call("POST", f"/conversations/{cid}/messages", body)[0] == 400
            assert c.call("POST", f"/conversations/{cid}/queue", body)[0] == 400
        assert c.call("POST", f"/conversations/{cid}/messages",
                      {"client_msg_id": "e4", "text": 5})[0] == 400
        assert not [b for b in c.fake.bodies if "model" in b and "session" not in b]
        assert c.snap(cid)["messages"] == []
        # an empty message stored before this check (an old database): refused before the engine
        um = c.svc._insert_message(cid, None, "user", [{"type": "text", "text": ""}], "done")
        with c.svc.conv(cid).lock:
            c.svc._start_run(c.svc.conv(cid), um["id"], None)
        s = c.wait_idle(cid, 20)
        a = s["messages"][-1]
        assert a["status"] == "error" and not [b for b in c.fake.bodies
                                               if "model" in b and "session" not in b]
        # with a file it goes
        att = c.svc.store_blob(cid, "notes.txt", "text/plain", b"one line")
        c.ok("POST", f"/conversations/{cid}/messages", {"client_msg_id": "e5", "text": "",
                                                        "attachments": [att["id"]]})
        c.wait_idle(cid, 20)
        assert "one line" in requests(c)[-1]["messages"][-1]["content"]
    finally:
        c.close()


# ---------------------------------------------------------------------- the port preview
class RelayBoxes(FakeBoxes):
    """`relay.py connect PORT` played on the host: the exec's stdin goes raw to 127.0.0.1:PORT,
    its answer comes back in the attach stream's 8-byte frames, as the real relay's does."""

    def __init__(self, data):
        super().__init__(data)
        self.relays = 0

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        if "/opt/tandem/relay.py" not in cmd:
            return super().open_exec(cid, cmd, tty=tty, stdin=stdin, env=env, workdir=workdir)
        a, b = socket.socketpair()
        srv = socket.create_connection(("127.0.0.1", int(cmd[-1])))
        self.relays += 1

        def up():
            try:
                while True:
                    d = b.recv(65536)
                    if not d:
                        break
                    srv.sendall(d)
                srv.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        def down():
            while True:
                d = srv.recv(65536)
                if not d:
                    break
                b.sendall(bytes([1, 0, 0, 0]) + len(d).to_bytes(4, "big") + d)
            # shut down before the close: on Linux a close() while `up` blocks in b.recv leaves
            # the socket open (the blocked call holds it), the reader never sees the end and the
            # preview hung on the Spark only (macOS wakes the reader)
            try:
                b.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            b.close()
            self.relays -= 1
        threading.Thread(target=up, daemon=True).start()
        threading.Thread(target=down, daemon=True).start()
        return "exec-relay", a, b""

    def exit_code(self, exec_id):
        return False, 0, None


def app_server():
    """A dev server in the sandbox: a page with root-relative links, and a WebSocket echo."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from chatd.terminals import WsConn, ws_accept

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if ws_accept(self):
                ws = WsConn(self)
                while True:
                    m = ws.recv()
                    if m is None:
                        return
                    ws.send(m[0], b"echo:" + m[1])
            body = b'<script src="/app.js"></script><a href="/about">about</a>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "x=1")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_the_port_preview_proxies_pages_and_websockets():
    c = Chat()
    c.svc.sandboxes = RelayBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    app = app_server()
    port = app.server_address[1]
    try:
        cid = c.new()
        base = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/")
        assert st == 503                                   # the sandbox is not running
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/")
        body = body.decode()
        assert st == 200 and f'src="{base}/app.js"' in body and f'href="{base}/about"' in body
        assert "frame-ancestors 'self'" in h["Content-Security-Policy"] and "Set-Cookie" not in h
        sock = ws_open(c.port, base + "/socket")
        ws_send(sock, b"hello", op=1)
        assert ws_recv(sock) == b"echo:hello"
        ws_send(sock, b"\x00\x01binary", op=2)
        assert ws_recv(sock) == b"echo:\x00\x01binary"
        sock.close()
        # the browser gone, the relay ends
        c.wait(lambda: c.svc.sandboxes.relays == 0, 10)
    finally:
        app.shutdown()
        c.close()


if __name__ == "__main__":
    run_all(globals())
