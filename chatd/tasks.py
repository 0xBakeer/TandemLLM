"""Workers: the task tool's runtime (VIS-E15, SRV-579; the plan is the Memo note "Chat: main agent
and workers (context folding), 2026-10-17"). Mixed into `chatd.service.Service`.

THE TASK. In Agents mode the main agent gives hands-on work to a worker: `task {title, brief,
expect}` (chatd/tools.py). A worker is a full run (chatd/runs.py) on a hidden conversation of kind
`task` whose parent is the chat: the same loop, retries, compaction guard, verify phase and
permission handling as any run, so a loop fix lands once. Its context starts fresh: the worker
block (packages/system/workers/PACKAGE.md, then the chat-app tool rules), the worker tools
(tools.py WORKER_TOOLS) and one user message with the task's number, title, brief, expect and the
chat's todo list. Nothing else of the chat goes in, and only the task result comes back.

SHARED. Every sandbox, workspace, setting and rule lookup of a task conversation resolves to its
chat (`root_cid`, `settings_of`): one container, one /workspace, the chat's permission mode and
rules. The worker runs in its parent's run slot (`Run.slot_owner`) while the parent waits in its
task call; it gives the slot back while a permission prompt waits, as any run does.

LIMITS. A step with two task calls runs the first; each other answers tools.ONE_TASK. A run gives
at most `task_max_per_run` (8) tasks, a worker makes at most `task_max_steps` (40) model calls and
has no task tool (depth 1).

THE CHAT SEES IT. The worker's session is `chat-<chat>-task-<n>` (render.task_session_id) and its
headers turn the memory notes off. Its transcript's events go to the chat's log with `task_id`
(service.py TASK_FORWARD), beside `task.started`, `task.phase`, `task.metrics` (ephemeral, once a
second) and `task.done`; meanwhile the parent's phase is `task`, its Now line "Task 2: <title>, call
12 of 40: <the worker's Now>". `GET .../tasks` lists a chat's tasks, `GET .../tasks/{id}` gives one
with its transcript. A snapshot of the workspace at the worker's start (key: the task's id) is the
base of the task's diff and a rewind point.
"""

from __future__ import annotations

import json
import os
import re

from chatd import render
from chatd import context as ctx_mod
from chatd.ids import iso, now_ms, ulid
from chatd.store import dumps
from chatd.tools import Result

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages",
                   "system", "workers")
RESULT_MAX_CHARS = 8000
_TEXTS: dict = {}


def texts() -> dict:
    """The package's texts: `worker` (PACKAGE.md), `main` (references/main.md) and the sections
    of references/brief.md and report.md by their `## ` headings, read once."""
    if not _TEXTS:
        def read(rel: str) -> str:
            with open(os.path.join(PKG, rel), encoding="utf-8") as fh:
                return fh.read()
        _TEXTS["worker"] = read("PACKAGE.md").strip()
        _TEXTS["main"] = read("references/main.md").strip()
        for rel in ("references/brief.md", "references/report.md"):
            for head, body in re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", read(rel), re.M | re.S):
                _TEXTS[head.strip().lower()] = body.strip()
    return _TEXTS


def worker_block(max_steps: int) -> str:
    return texts()["worker"].replace("{max_steps}", str(max_steps))


def first_message(n: int, title: str, brief: str, expect: str, todos: list[dict]) -> str:
    """The worker's one user message (the plan note's layout)."""
    lines = [f"Task {n}: {title}", "Brief:", brief.strip(),
             f"The report must also contain: {expect.strip()}"]
    if todos:
        lines.append("The chat's todo list (the main agent keeps it; you cannot change it):")
        lines += [f"- [{t['state']}] {t['text']}" for t in todos]
    return "\n".join(lines)


def final_text(parts: list[dict]) -> str:
    """The worker's last words: the text of the steps after its last tool call."""
    last = max((int(p.get("step") or 1) for p in parts if p.get("type") == "tool_call"),
               default=0)
    return "".join(p.get("text") or "" for p in parts
                   if p.get("type") == "text" and int(p.get("step") or 1) > last).strip()


def wire(row) -> dict:
    """The contract's `Task` (dashboard/src/chat/CONTRACT.md "Tasks (VIS-E15)")."""
    r = dict(row)
    return {"id": r["id"], "n": r["n"], "title": r["title"], "brief": r["brief"],
            "expect": r["expect"], "status": r["status"], "attempt": r["attempt"],
            "conversation_id": r["conversation_id"],
            "task_conversation_id": r["task_conversation_id"],
            "run_id": r["run_id"], "call_id": r["call_id"],
            "session": render.task_session_id(r["conversation_id"], r["n"]),
            "steps": r["steps"], "max_steps": r["max_steps"],
            "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
            "cached_tokens": r["cached_tokens"],
            "started_at": iso(r["started"]) if r["started"] else None,
            "ended_at": iso(r["ended"]) if r["ended"] else None,
            "report": json.loads(r["report"]) if r["report"] else None,
            "report_source": r["report_source"], "check": r["check_text"], "error": r["error"]}


class TasksMixin:
    # ------------------------------------------------------------------ who asks
    def task_row(self, task_id: str | None) -> dict | None:
        r = self.store.one("SELECT * FROM tasks WHERE id = ?", (task_id,)) if task_id else None
        return dict(r) if r is not None else None

    def session_of(self, conv) -> str:
        """The engine's session of a conversation's requests: a worker's own, else the chat's."""
        if conv.parent_id is None:
            return render.session_id(conv.id)
        return render.task_session_id(conv.parent_id, self.task_row(conv.task_id)["n"])

    def headers_of(self, conv) -> dict:
        if conv.parent_id is None:
            return render.headers(conv.id)
        return render.headers(conv.id, session=self.session_of(conv),
                              parent_session=render.session_id(conv.parent_id), worker=True)

    def _head(self, conv, run=None) -> dict:
        """The system block's own part by who asks: a worker's block."""
        if conv.parent_id is None:
            return {}
        task = run.task if run is not None and run.task else self.task_row(conv.task_id)
        return {"worker": worker_block(int(task["max_steps"]))}

    # ------------------------------------------------------------------ the tool
    def start_task(self, parent, part: dict, args: dict) -> Result:
        """The task call's executor (on the parent's tool thread): a worker on a new task
        conversation, in the parent's slot; its result when it ends."""
        settings = self.settings_of(parent.conv.id)
        if parent.parent is not None:
            return Result("task: a worker cannot give tasks", is_error=True)
        limit = int(settings.get("task_max_per_run") or 8)
        if parent.tasks_given >= limit:
            return Result(f"task refused: this run has given {limit} tasks, its limit. Answer the "
                          f"user with what the reports say, or ask whether to go on.",
                          is_error=True)
        title, brief, expect = (str(args.get(k) or "").strip() for k in ("title", "brief",
                                                                         "expect"))
        parent.tasks_given += 1
        row = self._new_task(parent, part, title, brief, expect, settings)
        return self._run_task(parent, row)

    def _new_task(self, parent, part: dict, title: str, brief: str, expect: str,
                  settings: dict) -> dict:
        from chatd.service import EMPTY_UI
        conv = parent.conv
        root = conv.id
        with conv.lock:
            n = 1 + int(self.store.one("SELECT MAX(n) FROM tasks WHERE conversation_id = ?",
                                       (root,))[0] or 0)
            tid, tcid, t = ulid(), ulid(), now_ms()
            steps = int(settings.get("task_max_steps") or 40)
            self.store.run("INSERT INTO conversations (id, title, title_auto, created, updated, "
                           "model, settings, ui, kind, parent_id) VALUES (?, ?, 0, ?, ?, ?, '{}', "
                           "?, 'task', ?)", (tcid, f"Task {n}: {title}"[:200], t, t,
                                             settings.get("model"), dumps(EMPTY_UI), root))
            self.store.run("INSERT INTO tasks (id, conversation_id, run_id, call_id, n, title, "
                           "brief, expect, task_conversation_id, status, max_steps, started) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                           (tid, root, parent.id, part["id"], n, title, brief, expect, tcid, steps,
                            t))
            text = first_message(n, title, brief, expect, self.todos(root))
            self._insert_message(tcid, None, "user", [{"type": "text", "text": text}], "done")
        try:                                     # the task's diff starts here (SRV-581's check)
            self.snaps.take(root, self.workspace(root), tid, "before")
        except OSError:
            pass
        row = self.task_row(tid)
        self.emit(conv, {"type": "task.started", "run_id": parent.id, "task": wire(row)},
                  run_id=parent.id)
        self.log(f"[task] {tid} {n} started conv={root} run={parent.id}")
        return row

    def _run_task(self, parent, row: dict, continue_mid: str | None = None) -> Result:
        tconv = self.conv(row["task_conversation_id"])
        user = self.store.one("SELECT id FROM messages WHERE conversation_id = ? AND role = 'user' "
                              "ORDER BY id LIMIT 1", (tconv.id,))["id"]
        with tconv.lock:
            worker = self._start_run(tconv, user, parent.started_by, parent=parent, task=row,
                                     continue_mid=continue_mid)
        parent.child = worker
        try:
            while worker.thread.is_alive() and not parent.cancel.is_set():
                worker.thread.join(1.0)
        finally:
            parent.child = None
        row = self.task_row(row["id"])
        parent.workers.append({"task_id": row["id"], "prompt_tokens": row["prompt_tokens"],
                               "completion_tokens": row["completion_tokens"],
                               "cached_tokens": row["cached_tokens"],
                               "ms": int((row["ended"] or now_ms()) - row["started"])})
        return Result(self.task_result(row), is_error=row["status"] == "failed")

    def task_result(self, row: dict) -> str:
        """What the main agent reads: the worker's last words, at most RESULT_MAX_CHARS."""
        rep = json.loads(row["report"]) if row["report"] else {}
        text = rep.get("text") or f"task {row['n']} ended {row['status']} with no report"
        return text[:RESULT_MAX_CHARS]

    # ------------------------------------------------------------------ the worker's end
    def task_ended(self, worker, outcome: str, stats: dict) -> None:
        """From finish_run, under the task conversation's lock: the tasks row, the task's
        snapshot after, `task.done` on the chat."""
        conv, task = worker.conv, worker.task
        root = conv.parent_id
        status = {"completed": "done", "cancelled": "cancelled", "interrupted": "interrupted",
                  "failed": "failed"}[outcome]
        try:
            self.snaps.take(root, self.workspace(root), task["id"], "after")
        except OSError:
            pass
        m = self._messages(conv.id).get(worker.message_id)
        report = {"text": final_text(m["parts"] if m else [])}
        fig = {"prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in worker.step_usage),
               "cached_tokens": sum(int(u.get("cached_tokens") or 0) for u in worker.step_usage),
               "completion_tokens": int(stats.get("completion_tokens") or 0)}
        self._end_task(worker, status, report, "worker", None, fig)

    def _end_task(self, worker, status: str, report: dict | None, source: str | None,
                  check: str | None, fig: dict) -> None:
        task = worker.task
        t = now_ms()
        # a worker resumed by Continue adds its calls to the ones before
        self.store.run("UPDATE tasks SET status = ?, steps = ?, prompt_tokens = prompt_tokens + ?, "
                       "completion_tokens = completion_tokens + ?, cached_tokens = cached_tokens "
                       "+ ?, ended = ?, report = ?, report_source = ?, check_text = ?, error = ? "
                       "WHERE id = ?",
                       (status, worker.step, fig["prompt_tokens"], fig["completion_tokens"],
                        fig["cached_tokens"], t, dumps(report) if report is not None else None,
                        source, check, worker.error, task["id"]))
        row = self.task_row(task["id"])
        parent = self.conv(worker.conv.parent_id)
        self.emit(parent, {"type": "task.done", "task_id": task["id"], "status": status,
                           "report": report, "check": check,
                           "stats": {"steps": row["steps"], "max_steps": row["max_steps"],
                                     "prompt_tokens": row["prompt_tokens"],
                                     "completion_tokens": row["completion_tokens"],
                                     "cached_tokens": row["cached_tokens"],
                                     "ms": int(t - row["started"]),
                                     "report_source": source}},
                  run_id=worker.parent.id if worker.parent else None)

    # ------------------------------------------------------------------ live figures
    def task_phase(self, worker) -> None:
        """A worker's phase: `task.phase` on its chat, and the parent's Now line."""
        p, t = worker.parent, worker.task
        ev = {"type": "task.phase", "task_id": t["id"], "phase": worker.phase, "now": worker.now,
              "step": worker.step, "max_steps": worker.max_steps}
        if worker.tool:
            ev["tool"] = worker.tool
        self.emit(p.conv, ev, run_id=p.id)
        if p.active:
            p.set_phase("task", f"Task {t['n']}: {t['title']}, call {worker.step} of "
                                f"{worker.max_steps}: {worker.now}")

    def task_metrics(self, worker) -> None:
        """Once a second from the ticker (service.py `_metrics`): the worker's tokens so far."""
        self.emit(worker.parent.conv, {
            "type": "task.metrics", "task_id": worker.task["id"],
            "prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in worker.step_usage),
            "completion_tokens": worker.tokens, "decode_tps": worker.metrics["decode_tps"]},
            persist=False)

    def worker_split(self, worker, body: dict, path: list[dict]) -> None:
        """At each of a worker's requests: the parts of its context, estimated from the
        characters at its learned ratio (the engine counts only the whole prompt's)."""
        from chatd.agent import estimated_split
        est = estimated_split(body, worker.conv.tok_ratio, None)
        user = path[0] if path and path[0].get("role") == "user" else {}
        worker.ctx_est = {"system": est["system"], "tools": est["tools"],
                          "brief": ctx_mod.estimate_tokens(render.text_of(user.get("parts") or [])),
                          "files": ctx_mod.file_tokens(path)}

    def worker_context(self, run) -> dict | None:
        """The Context panel's `worker`: the running worker's own window (null when none runs)."""
        w = getattr(run, "child", None) if run is not None else None
        if w is None or not w.active:
            return None
        t = w.task
        last = w.step_usage[-1] if w.step_usage else {}
        used = int(last.get("prompt_tokens") or 0) + int(last.get("completion_tokens") or 0) \
            or int(w.request_est or 0)
        est = getattr(w, "ctx_est", None) or {"system": 0, "tools": 0, "brief": 0, "files": 0}
        prompt = int(last.get("prompt_tokens") or 0)
        cached = int(last.get("cached_tokens") or 0)
        settings = self.settings_of(w.conv.id)
        return {"task_id": t["id"], "n": t["n"], "title": t["title"], "used": used,
                "window": int(self.model_info(settings.get("model")).get("max_len") or 0),
                "split": {**est, "steps": max(0, used - sum(est.values()))},
                "step": w.step, "max_steps": w.max_steps,
                "cache": {"warm": cached, "cold": max(0, prompt - cached),
                          "source": last.get("cache_source") or "none"}}

    # ------------------------------------------------------------------ routes
    def list_tasks(self, cid: str) -> dict:
        """GET /conversations/{id}/tasks: the chat's tasks, by number."""
        self._row(cid)
        return {"tasks": [wire(r) for r in self.store.all(
            "SELECT * FROM tasks WHERE conversation_id = ? ORDER BY n", (cid,))]}

    def get_task(self, cid: str, task_id: str) -> dict:
        """GET /conversations/{id}/tasks/{task_id}: the task with its transcript (calls with
        their arguments and results), the running one's from memory, and the chat's `seq` it
        includes: the chat's later events with this `task_id` apply on top."""
        from chatd.service import ApiError
        conv = self.conv(cid)
        r = self.store.one("SELECT * FROM tasks WHERE id = ? AND conversation_id = ?",
                           (task_id, cid))
        if r is None:
            raise ApiError(404, "not_found", "no such task in this chat")
        tconv = self.conv(r["task_conversation_id"])
        # under the task conversation's lock only: its changes and their events on the chat's log
        # happen under it (the lock order is task, then chat), so the transcript read here holds
        # every event up to `seq` and none after
        with tconv.lock:
            seq = conv.last_seq
            row = self._row(tconv.id)
            branch, msgs = self._branch(tconv, row["head_id"])
            kids = self._children(msgs)
            return {"task": wire(self.task_row(task_id)),
                    "messages": [self._wire_message(m, kids) for m in branch], "seq": str(seq)}
