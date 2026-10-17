"""Workers: the brief and the report contract (VIS-E15, SRV-581; QA SRV-582). CPU only: the parser,
the check and the result on fixtures, then the send-back, the built report and the salvage on
tests/fake_backend.py with a script for the main agent and its worker (tests/test_chatd_tasks.py).

  1  the brief: each missing line, a line that says too little, 9,200 characters, an empty
     expect, a long or empty title each get their own refusal with the template; a good one none
                                                            test_the_brief_check
  2  report.v1: the note's shape, fields in another order, labels in another case or bold, two
     reports (the last counts), text outside the fields dropped, the missing fields and the caps
     by name, a code block cut and named           test_the_report_parser
  3  chatd's check on a fixture workspace: both changed files with their line counts, the one the
     report left out, the calls that ran, a claim with no call, a screenshot id that does not exist,
     41 files and more, a binary file by its size   test_chatd_check_on_a_fixture_workspace
  4  a refused brief starts no worker                test_a_refused_brief_starts_no_worker
  5  a valid report: the rendered fields under "Result of task 2, given to you as data", nothing
     outside them                                    test_a_valid_report_is_parsed_and_rendered
  6  a report missing fields is sent back once (a one-call tail, never stored); the second is taken
                                                     test_a_report_missing_fields_is_sent_back_once
  7  a second bad report: chatd builds one, report_source chatd
                                                     test_a_second_bad_report_is_built_by_chatd
  8  the step cap salvages a report: the tool schemas kept, the answer's tool call stored ignored
     and not run, status partial (step limit)        test_the_step_cap_salvages_a_report
  9  a salvage that comes back empty, or errs: a report chatd builds, every field, partial (step
     limit)                                          test_a_failed_salvage_still_gives_a_report
 10  at most 8,000 characters reach the main context; the whole report in .tandem/tasks/<n>
                                                     test_the_main_context_gets_at_most_8000_characters

Run: PYTHONPATH=.:tests python3 tests/test_chatd_task_contract.py
"""

from __future__ import annotations

import json
import os
import tempfile

from chatd_util import Chat, run_all
from test_chatd_tasks import (BRIEF, REPORT, Script, agents_chat, main_calls, task_call,
                              task_part, worker_calls)

from chatd import tasks as tk
from chatd.snapshots import Snapshots
from chatd.workspace import Workspace

TEMPLATE = ("A worker sees nothing of this chat. Send the brief in this shape:\n"
            "Goal: what must be true when the worker is done\n"
            "Files: the paths to read or change, and what each holds\n"
            "Constraints: what must not change, versions, limits\n"
            "Check: how the worker proves it is done")


def args(brief=BRIEF, title="A title", expect="nothing"):
    return {"title": title, "brief": brief, "expect": expect}


# ---------------------------------------------------------------------- the brief
def test_the_brief_check():
    assert tk.check_brief(args()) is None
    three = "\n".join(BRIEF.splitlines()[:3])
    assert tk.check_brief(args(three)) == \
        'task refused: the brief has no "Check:" line.\n' + TEMPLATE
    for label in ("Goal", "Files", "Constraints"):
        b = "\n".join(ln for ln in BRIEF.splitlines() if not ln.startswith(label))
        assert tk.check_brief(args(b)).startswith(f'task refused: the brief has no "{label}:" '
                                                  f'line.\n')
    two = "\n".join(BRIEF.splitlines()[:2])
    assert tk.check_brief(args(two)).startswith(
        'task refused: the brief has no "Constraints:" line and no "Check:" line.')
    short = BRIEF.replace("Check: read_file hello.txt shows the word hi", "Check: it works")
    assert tk.check_brief(args(short)).startswith(
        'task refused: the "Check:" line says too little (at least 10 characters after the '
        'label).')
    big = BRIEF + "\n" + "x" * (9200 - len(BRIEF) - 1)
    assert len(big) == 9200
    assert tk.check_brief(args(big)).startswith(
        "task refused: the brief is 9,200 characters; at most 8,000.")
    assert tk.check_brief(args(expect=" ")).startswith("task refused: expect is empty")
    assert tk.check_brief(args(title="t" * 95)).startswith(
        "task refused: the title is 95 characters; at most 80.")
    assert tk.check_brief(args(title="")).startswith("task refused: the title is empty.")
    # labels in another case and with spaces are lines all the same
    assert tk.check_brief(args(BRIEF.replace("Goal:", "goal :"))) is None
    assert all(tk.check_brief(a).endswith(TEMPLATE) for a in (
        args(three), args(big), args(title=""), args(expect="")))


# ---------------------------------------------------------------------- the report
def test_the_report_parser():
    rep, problems = tk.parse_report(REPORT)
    assert problems == [] and rep["schema"] == "report.v1" and rep["status"] == "done"
    assert rep["summary"] == "wrote hello.txt"
    assert rep["changed_files"] == [{"path": "hello.txt", "change": "new, one line"}]
    assert rep["checks"] == [{"what": "read_file hello.txt", "result": "hi", "call_id": None,
                              "shot": None}]
    assert rep["expected"] == ["nothing asked"] and rep["open_problems"] == []
    # another order, other cases, bold labels, a preamble and a closing line outside the fields
    other = ("Here is my report.\n\n**Summary:** fixed the banding\n  CHECKS :\n- render_page "
             "index.html: 0 console errors, shot-2 (call call_7)\nchanged FILES:\n"
             "- /workspace/src/ocean.wgsl: normals in f32\n  and smoothstep\nStatus: partial\n"
             "Expected:\n- the FPS line: fps 58.9\nOpen problems:\n- the sky is flat\n\n"
             "Thanks, that is all!")
    rep, problems = tk.parse_report(other)
    assert problems == [], problems
    assert rep["status"] == "partial" and rep["summary"] == "fixed the banding"
    assert rep["changed_files"] == [{"path": "src/ocean.wgsl",
                                     "change": "normals in f32 and smoothstep"}]
    assert rep["expected"] == [{"item": "the FPS line", "answer": "fps 58.9"}]
    assert rep["checks"][0] == {"what": "render_page index.html",
                                "result": "0 console errors, shot-2 (call call_7)",
                                "call_id": "call_7", "shot": "shot-2"}
    assert "Thanks" not in json.dumps(rep) and "Here is my report" not in json.dumps(rep)
    # two reports: the last one counts
    rep, _ = tk.parse_report(REPORT.replace("wrote hello.txt", "first try") + "\n\n" + REPORT)
    assert rep["summary"] == "wrote hello.txt"
    # missing fields and caps, by name
    no_checks = REPORT.replace("Checks:\n- read_file hello.txt: hi\n", "").replace(
        "\nOpen problems:\n- none", "")
    assert tk.parse_report(no_checks)[1] == ["Checks", "Open problems"]
    assert tk.parse_report("I wrote the file, all good.") == (None, list(tk.FIELDS))
    assert tk.parse_report(REPORT.replace("Status: done", "Status: finished"))[1] == \
        ["a Status of done, partial or failed"]
    rep, problems = tk.parse_report(REPORT.replace("wrote hello.txt", "s" * 700))
    assert problems == ["a Summary of at most 600 characters"] and len(rep["summary"]) <= 610
    many = "\n".join(f"- f{i}.txt: new" for i in range(45))
    rep, problems = tk.parse_report(REPORT.replace("- hello.txt: new, one line", many))
    assert problems == ["at most 40 Changed files"] and len(rep["changed_files"]) == 40
    rep, problems = tk.parse_report(REPORT.replace("new, one line", "w" * 250))
    assert problems == ["Changed files lines of at most 200 characters"]
    # a code block inside a field is cut and named: no file body rides in a report
    code = "x = 1\n" * 500                                  # 3,000 characters
    rep, problems = tk.parse_report(REPORT.replace("Summary: wrote hello.txt",
                                                   "Summary: wrote it:\n```js\n" + code + "```"))
    assert problems == [] and "x = 1" not in json.dumps(rep)
    assert rep["summary"] == "wrote it: [code block of 2,999 characters removed]", rep["summary"]
    assert rep["cut"] == ["[code block of 2,999 characters removed]"]


def _part(cid, name, args, out, step, **kw):
    return {"type": "tool_call", "id": cid, "name": name, "input": json.dumps(args),
            "status": kw.pop("status", "done"), "step": step,
            "output": {"content": out, "is_error": kw.pop("is_error", False),
                       "exit_code": kw.pop("exit_code", None), "images": []},
            "model_content": out, **kw}


def test_chatd_check_on_a_fixture_workspace():
    d = tempfile.mkdtemp(prefix="chatd-check-")
    ws = Workspace(os.path.join(d, "ws"))
    ws.ensure()
    snaps = Snapshots(os.path.join(d, "snaps"))
    ws.write("src/ocean.wgsl", "".join(f"old {i}\n" for i in range(20)).encode())
    ws.write("src/main.js", "a\nb\nc\n".encode())
    snaps.take("c", ws, "t1", "before")
    ws.write("src/ocean.wgsl", ("".join(f"old {i}\n" for i in range(8)) +
                                "".join(f"new {i}\n" for i in range(41))).encode())
    ws.write("src/main.js", "a\nB\nc\nd\nE\n".encode())
    ws.write("sky.png", b"\x89PNG\r\n\x1a\n\0\0" + b"\1" * 1000)
    snaps.take("c", ws, "t1", "after")
    diff = snaps.diff("c", "t1")
    parts = [_part("call_31", "render_page", {"path": "index.html"},
                   "rendered file:///workspace/index.html\nno console errors", 3, shot="shot-1",
                   pane={"kind": "image", "ref": "file:/workspace/.tandem/shots/call_31.png"}),
             _part("call_29", "bash", {"command": "npm test"}, "ok", 2, exit_code=0),
             _part("call_30", "read_file", {"path": "src/main.js"}, "1\ta", 2)]
    rep, _ = tk.parse_report(REPORT.replace("- hello.txt: new, one line",
                                            "- src/ocean.wgsl: normals in f32").replace(
        "- read_file hello.txt: hi", "- render_page index.html: no errors, shot-1\n"
                                     "- bash \"npm test\": exit 0\n- read_file src/main.js: read\n"
                                     "- curl the dev server: 200\n- looked at shot-9: fine"))
    lines = tk.chatd_check(diff, parts, rep)
    assert lines[0] == ("- Changed in /workspace (snapshot diff): sky.png (binary, 1,010 bytes), "
                        "src/main.js (+3 -1), src/ocean.wgsl (+41 -12)"), lines[0]
    assert lines[1] == "- Changed but not in the report: sky.png, src/main.js"
    assert lines[2] == ('- Checks that ran: render_page index.html (call call_31: 0 console '
                        'errors, shot-1); bash "npm test" (call call_29: exit 0)'), lines[2]
    assert lines[3] == ('- Claimed in Checks with no matching call: "curl the dev server: 200"; '
                        '"looked at shot-9: fine"'), lines[3]
    assert lines[4] == ("- Screenshots (open one with read_file): shot-1 "
                        "/workspace/.tandem/shots/call_31.png")
    # no render_page call: that claim is named
    lines = tk.chatd_check(diff, parts[1:], rep)
    assert '"render_page index.html: no errors, shot-1"' in lines[3]
    # 52 files: the first 40 and how many more
    for i in range(52):
        ws.write(f"many/f{i:02d}.txt", b"x\n")
    snaps.take("c", ws, "t2", "before")
    for i in range(52):
        ws.write(f"many/f{i:02d}.txt", b"y\n")
    snaps.take("c", ws, "t2", "after")
    line = tk.chatd_check(snaps.diff("c", "t2"), [], None)[0]
    assert line.count("(+1 -1)") == 40 and line.endswith(", and 12 more"), line[-60:]
    # nothing changed, nothing ran
    snaps.take("c", ws, "t3", "before")
    snaps.take("c", ws, "t3", "after")
    assert tk.chatd_check(snaps.diff("c", "t3"), [], None) == [
        "- Changed in /workspace (snapshot diff): none", "- Changed but not in the report: none",
        "- Checks that ran: none", "- Claimed in Checks with no matching call: none"]


# ---------------------------------------------------------------------- in a chat
def worker_says(*answers):
    """A worker that writes hello.txt, then answers these in turn (the last one again)."""
    def worker(b, k, n):
        if k == 0:
            return {"calls": [{"name": "write_file", "arguments": {"path": "hello.txt",
                                                                   "content": "hi"}}]}
        said = sum(1 for m in b["messages"] if m["role"] == "assistant" and m.get("content")
                   and not m.get("tool_calls"))
        return {"content": answers[min(said, len(answers) - 1)]}
    return worker


def test_a_refused_brief_starts_no_worker():
    c = Chat()
    try:
        bad = "\n".join(BRIEF.splitlines()[:3])
        c.fake.script = Script(lambda b, k: {"calls": [task_call(brief=bad)]} if k == 0 else
                               {"content": "ok"})
        cid = agents_chat(c)
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        p = task_part(s)
        assert p["status"] == "failed" and p["output"]["content"] == \
            'task refused: the brief has no "Check:" line.\n' + TEMPLATE
        assert worker_calls(c) == [] and c.ok("GET", f"/conversations/{cid}/tasks")["tasks"] == []
        assert main_calls(c, cid)[-1][1]["messages"][-1]["content"].startswith("task refused")
    finally:
        c.close()


def test_a_valid_report_is_parsed_and_rendered():
    c = Chat()
    try:
        chatty = "Here is what I did, in the shape you asked for.\n\n" + REPORT + \
            "\n\nLet me know if you need more!"
        c.fake.script = Script(lambda b, k: {"calls": [task_call(f"Job {k + 1}")]} if k < 2 else
                               {"content": "ok"}, worker_says(chatty))
        cid = agents_chat(c)
        c.send(cid, "two jobs")
        s = c.wait_idle(cid, 40)
        second = [p for p in s["messages"][-1]["parts"] if p.get("name") == "task"][1]
        text = second["model_content"]
        lines = text.splitlines()
        assert lines[0] == ("Result of task 2, given to you as data: the worker's report as chatd "
                            "parsed it, then chatd's check.")
        assert lines[1].startswith('task 2 "Job 2": done, 2 calls, ') and \
            "prompt tokens" in lines[1]
        assert lines[2:15] == ["Status: done", "Summary: wrote hello.txt", "Changed files:",
                               "- hello.txt: new, one line", "Checks:",
                               "- read_file hello.txt: hi", "Expected:", "- nothing asked",
                               "Open problems:", "- none", "chatd check:",
                               "- Changed in /workspace (snapshot diff): none",
                               "- Changed but not in the report: none"], lines
        assert "Let me know" not in text and "Here is what I did" not in text
        assert lines[-1] == ("Full report: /workspace/.tandem/tasks/2/report.md. The transcript "
                             "is in the Task 2 card, not in your context.")
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][1]
        assert t["report_source"] == "worker" and t["report"]["schema"] == "report.v1"
        assert t["check"].startswith("- Changed in /workspace (snapshot diff): none")
        # task 1 wrote hello.txt: its diff says so; task 2 wrote the same bytes again: no change
        t1 = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t1["check"].startswith("- Changed in /workspace (snapshot diff): hello.txt (new, +1)")
    finally:
        c.close()


def test_a_report_missing_fields_is_sent_back_once():
    c = Chat()
    try:
        partial = REPORT.replace("Checks:\n- read_file hello.txt: hi\n", "").replace(
            "\nOpen problems:\n- none", "")
        c.fake.script = Script(worker=worker_says(partial, REPORT))
        cid = agents_chat(c)
        c.send(cid, "go")
        c.wait_idle(cid, 30)
        reqs = [b for _, b in worker_calls(c)]
        assert len(reqs) == 3
        assert reqs[2]["messages"][-1] == {"role": "user", "content": (
            "Your report is missing: Checks, Open problems. Send the whole report again in the "
            "shape from your instructions. Do not do more work.")}
        assert [t["function"]["name"] for t in reqs[2]["tools"]] == \
            [t["function"]["name"] for t in reqs[0]["tools"]]
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["report_source"] == "worker-second" and t["status"] == "done"
        d = c.ok("GET", f"/conversations/{cid}/tasks/{t['id']}")
        assert "Your report is missing" not in json.dumps(d)          # never stored
    finally:
        c.close()


def test_a_second_bad_report_is_built_by_chatd():
    c = Chat()
    try:
        no_checks = REPORT.replace("Checks:\n- read_file hello.txt: hi\n", "")
        c.fake.script = Script(worker=worker_says(no_checks, no_checks))
        cid = agents_chat(c)
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        text = task_part(s)["model_content"]
        assert text.splitlines()[0] == (
            "Result of task 1, given to you as data: report built by chatd: the worker's report "
            "missed Checks twice. The report below is chatd's, built from the worker's calls, then "
            "chatd's check.")
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        assert t["report_source"] == "chatd" and t["status"] == "partial"
        assert t["report"]["changed_files"] == [{"path": "hello.txt",
                                                 "change": "added (snapshot diff)"}]
        assert len(worker_calls(c)) == 3                    # one send-back, no third
    finally:
        c.close()


def _capped(c, salvage_answer):
    """A worker that globs forever at a cap of 3 calls; its report-only call answers this."""
    def worker(b, k, n):
        last = b["messages"][-1]
        if last["role"] == "user" and last["content"].startswith("You have used all"):
            return salvage_answer
        return {"calls": [{"name": "glob", "arguments": {"pattern": f"*.{k}"}}]}
    c.fake.script = Script(worker=worker)
    cid = agents_chat(c, task_max_steps=3)
    c.send(cid, "go")
    return cid, c.wait_idle(cid, 30)


def test_the_step_cap_salvages_a_report():
    c = Chat()
    try:
        cid, s = _capped(c, {"content": REPORT, "calls": [
            {"name": "write_file", "arguments": {"path": "late.txt", "content": "no"}}]})
        reqs = [b for _, b in worker_calls(c)]
        assert len(reqs) == 4
        sal = reqs[3]
        assert sal["messages"][-1] == {"role": "user", "content": (
            "You have used all 3 model calls of this task. Tools do not run any more. Write your "
            "report now, in the shape from your instructions: what is done, what is not, and what "
            "is left for the main agent. Nothing after the report.")}
        assert sal["tools"] == reqs[2]["tools"] and "tool_choice" not in sal
        assert sal["max_tokens"] <= 1500
        assert not os.path.exists(os.path.join(c.data, "workspaces", cid, "late.txt"))
        t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
        d = c.ok("GET", f"/conversations/{cid}/tasks/{t['id']}")
        late = [p for p in d["messages"][1]["parts"] if p.get("name") == "write_file"]
        assert late and late[0]["status"] == "ignored"
        assert t["status"] == "partial (step limit)" and t["report"]["status_note"] == \
            "step limit"
        assert t["report_source"] == "worker" and t["steps"] == 4
        text = task_part(s)["model_content"]
        assert text.splitlines()[1].startswith('task 1 "Hello file": partial (step limit), 4 calls')
        assert "Status: partial (step limit)" in text
    finally:
        c.close()


def test_a_failed_salvage_still_gives_a_report():
    for answer in ({"content": ""}, {"status": 400, "message": "bad request"}):
        c = Chat()
        try:
            cid, s = _capped(c, answer)
            t = c.ok("GET", f"/conversations/{cid}/tasks")["tasks"][0]
            assert t["report_source"] == "chatd" and t["status"] == "partial (step limit)", t
            rep = t["report"]
            assert rep["status_note"] == "step limit" and rep["summary"].startswith(
                "report built by chatd: the worker's model calls ran out before its report")
            text = task_part(s)["model_content"]
            for field in ("Status: partial (step limit)", "Summary: ", "Changed files:",
                          "Checks:", "Expected:", "Open problems:", "chatd check:"):
                assert field in text, (field, text)
            assert s["messages"][-1]["status"] == "done"
        finally:
            c.close()


def test_the_main_context_gets_at_most_8000_characters():
    c = Chat()
    try:
        long = REPORT.replace("Open problems:\n- none", "Open problems:\n" + "\n".join(
            f"- problem {i}: " + "q" * 270 for i in range(40)))
        c.fake.script = Script(worker=worker_says(long))
        cid = agents_chat(c)
        c.send(cid, "go")
        s = c.wait_idle(cid, 30)
        text = task_part(s)["model_content"]
        whole = open(os.path.join(c.data, "workspaces", cid, ".tandem", "tasks", "1",
                                  "report.md")).read()
        assert len(whole) > 11_000 and len(text) <= 8000
        assert "[... the report is cut here; the whole of it is in " \
               "/workspace/.tandem/tasks/1/report.md]" in text
        assert text.splitlines()[0].startswith("Result of task 1, given to you as data")
        assert "chatd check:" in text and text.endswith("not in your context.")
        assert main_calls(c, cid)[-1][1]["messages"][-1]["content"] == text
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
