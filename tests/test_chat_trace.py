"""SRV-667: a recorded Chat-tab conversation replayed through chatd in seconds (tools/chat_trace.py).

A synthetic trace of the owner's reef run's shape (2026-10-20): a long write, the model sending
an earlier write's stub as content, a bash whose exit code is not 0 after a warning on stderr, a
plan-mode turn and its Approve. The fork (tools/chat_fork.py) cuts it at a step and runs a
stand-in live model from there. The replay must hold every check of `chat_trace.check`; with the
call cap at 2 the chat ends idle. With CHAT_TRACES set (a folder of chat snapshots, the box's
~/ornith/replay-r*.json and ~/chat-analysis/chat.json) every one of them is replayed too.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import chat_trace as T  # noqa: E402

from chatd import render  # noqa: E402

PAGE = "".join(f"export const reef{i} = Math.sin({i} * 0.1);\n" for i in range(120))


def _call(k, step, name, args, out, code=None, status="done"):
    return {"type": "tool_call", "id": f"c{k}", "name": name, "step": step, "status": status,
            "input": json.dumps(args), "output": {"content": out, "exit_code": code,
                                                  "is_error": status == "failed"}}


def _trace() -> dict:
    stub = render.wrote_stub("/workspace/reef/js/effects.js", PAGE)
    return {"settings": {"effort": "auto", "thinking": True, "permission_mode": "yolo",
                         "tools": ["bash", "read_file", "write_file", "edit_file", "glob",
                                   "todo_write"], "max_steps": 100},
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "Build the reef"}]},
                {"role": "assistant", "parts": [
                    {"type": "thinking", "text": "write the effects", "step": 1},
                    _call(1, 1, "write_file", {"path": "/workspace/reef/js/effects.js",
                                               "content": PAGE}, "wrote"),
                    {"type": "text", "text": "Again:", "step": 2},
                    _call(2, 2, "write_file", {"path": "/workspace/reef/js/effects.js",
                                               "content": stub}, "refused", status="failed"),
                    _call(3, 3, "bash", {"command": "cat > app.js <<'EOF'\nx\nEOF\nnode --check app.js"},
                          "(node:25) Warning: To load an ES module ...\nSyntaxError\n[exit code 1]", 1,
                          status="failed"),
                    _call(4, 4, "bash", {"command": "ls"}, "app.js\n", 0),
                    {"type": "text", "text": "Done.", "step": 5}]},
                {"role": "user", "parts": [{"type": "text", "text": "Split it. Rewrite the plan."},
                                           {"type": "plan", "text": "Plan mode is on"}]},
                {"role": "assistant", "parts": [
                    _call(5, 1, "read_file", {"path": "/workspace/reef/js/effects.js"}, ""),
                    {"type": "text", "text": "1. core.js\n2. main.js", "step": 2}]},
                {"role": "user", "parts": [{"type": "text", "text": T.APPROVED}]},
                {"role": "assistant", "parts": [
                    _call(6, 1, "edit_file", {"path": "/workspace/reef/js/effects.js",
                                              "old": "reef0 =", "new": "reef0 ="}, "edited"),
                    {"type": "text", "text": "Built.", "step": 2}]}]}


def test_a_trace_replays_and_every_check_holds():
    res = T.replay(_trace())
    bad = T.check(res)
    assert not bad, bad
    calls = {p["id"]: p for p in res["calls"]}
    assert len(res["requests"]) >= 7, len(res["requests"])
    stubbed = next(p for p in res["calls"] if p["name"] == "write_file" and p["status"] == "failed")
    assert "is the text that write sent, correct as it is" in stubbed["output"]["content"]
    bash = [p for p in res["calls"] if p["name"] == "bash"]
    assert [p["status"] for p in bash] == ["failed", "done"], [p["status"] for p in bash]
    assert res["statuses"][-1] == "idle", res["statuses"]
    assert calls                                                   # (ids are chatd's own)


def test_the_call_cap_leaves_the_chat_idle():
    res = T.replay(_trace(), cap=2)
    assert res["snapshot"]["messages"][-1]["status"] == "interrupted"
    assert not T.check(res, cap=2), T.check(res, cap=2)
    assert res["snapshot"]["status"] == "idle"


def test_a_fork_runs_the_live_model_from_the_step_for_a_few_steps():
    """SRV-667 C (tools/chat_fork.py): the record up to the step before the fork, then the live
    model, at most `steps` calls; its file calls run on the replay's workspace as the record left
    it; `patch` replaces the result of the last recorded call before the fork."""
    import chat_fork as F
    seen = []
    plan = [{"content": "", "reasoning": "check the file",
             "calls": [{"name": "read_file", "arguments": {"path": "/workspace/reef/js/effects.js"}},
                       {"name": "bash", "arguments": {"command": "npm test"}}]},
            {"content": "", "calls": [{"name": "glob", "arguments": {"pattern": "**/*.js"}}]},
            {"content": "never asked"}]

    def live(body, headers):
        seen.append((body, headers))
        return dict(plan[len(seen) - 1], _usage={"prompt_tokens": 7})
    res = F.fork(_trace(), (1, 3), live, steps=2, patch="refused: the new message")
    assert len(seen) == 2 and len(res["steps"]) == 2, len(seen)
    first = seen[0][0]["messages"]
    # the record up to step 2 (its patched result), nothing of step 3 or later
    tools = [m for m in first if m["role"] == "tool"]
    assert len(tools) == 2 and tools[-1]["content"] == "refused: the new message", tools
    assert not any("node --check" in json.dumps(m) for m in first)
    read, bash = res["steps"][0]["calls"]
    assert read["status"] == "done" and "reef0 = Math.sin(0 * 0.1)" in read["result"], read
    assert bash["status"] == "failed" and "not run in the fork" in bash["result"], bash
    assert res["steps"][1]["calls"][0]["name"] == "glob"
    assert res["steps"][0]["usage"] == {"prompt_tokens": 7}
    # the step must exist; the CLI caps the steps
    try:
        F.fork(_trace(), (1, 9), live)
        raise AssertionError("a step past the record forked")
    except ValueError as exc:
        assert "no step 9 in turn 1" in str(exc)


def test_the_fork_cli_against_an_engine():
    """tools/chat_fork.py end to end against a stand-in engine on localhost: the body chatd built,
    not streamed, under the fork's own session; --out holds the steps."""
    import http.server
    import tempfile
    import threading
    import chat_fork as F
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append((body, {"x-session-id": self.headers.get("x-session-id")}))
            msg = {"role": "assistant", "content": "The plan.", "reasoning_content": "think"}
            data = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 11, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d = tempfile.mkdtemp()
    try:
        trace, out = os.path.join(d, "t.json"), os.path.join(d, "f.jsonl")
        with open(trace, "w") as fh:
            json.dump(_trace(), fh)
        rc = F.main([trace, "--at", "1:4", "--steps", "2", "--samples", "2",
                     "--url", f"http://127.0.0.1:{srv.server_port}", "--out", out])
        assert rc == 0
        assert len(got) == 2                     # one live call a sample: the text ends the run
        body, headers = got[0]
        assert body["stream"] is False and body["messages"][-1]["role"] == "tool"
        assert headers["x-session-id"].startswith("chat-fork-")
        assert got[0][1]["x-session-id"] != got[1][1]["x-session-id"]
        rows = [json.loads(x) for x in open(out)]
        assert [r["sample"] for r in rows] == [1, 2] and rows[0]["content"] == "The plan."
        assert rows[0]["step"] == 4 and rows[0]["usage"]["prompt_tokens"] == 11
    finally:
        srv.shutdown()


def test_the_recorded_traces():
    folder = os.environ.get("CHAT_TRACES")
    if not folder:
        return "skipped: CHAT_TRACES not set"
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        res = T.replay(T.load(path))
        bad = T.check(res)
        assert not bad, (path, bad[:5])
        out.append(f"{os.path.basename(path)} {len(res['requests'])} requests")
    return ", ".join(out)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
