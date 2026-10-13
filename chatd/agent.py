"""The loop's side of the service: parts by step, tool calls, permissions, the request with the
context guard, compaction (SRV-351, SRV-353). Mixed into `chatd.service.Service`; everything here
runs under the conversation's lock when it touches the conversation, like the rest of the service.
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from chatd import compaction as comp
from chatd import egress, render
from chatd import tools as tools_mod
from chatd.ids import iso, now_ms, ulid
from chatd.plans import PLAN_REFUSAL, PLAN_RULE, PLAN_TOOLS
from chatd.runs import NUDGE, GuardError
from chatd.sandbox import Unavailable
from chatd.store import dumps
from chatd.tools import PathError, Result, ToolError

EVENT_OUTPUT_CAP = 64 * 1024        # tool.output per call in events (CONTRACT section 3)
READ_PARALLEL = 4


class ApiConflict(Exception):
    pass


class ToolCtx:
    """What an executor may use for one call."""

    def __init__(self, svc, run, part: dict):
        self.svc, self.run, self.part = svc, run, part
        self.call_id = part["id"]
        self.run_id = run.id
        self.cid = run.conv.id
        self.ws = svc.workspace(self.cid)
        self.engine = svc.engine
        self._out: list[str] = []
        self.quiet = False

    def exec(self, cmd: list[str], timeout: float = 600, quiet: bool = False):
        sb = self.svc.sandboxes

        def waiting(why):
            self.run.set_phase("tool", f"Waiting for memory to start the sandbox ({why})")
        sb.ensure(self.cid, cancel=self.run.cancel, on_wait=waiting)
        self.run.set_phase("tool", self._running_words())
        self.quiet = quiet
        return sb.run(self.cid, cmd, timeout=timeout, on_output=self.output,
                      cancel=self.run.cancel)

    def _running_words(self) -> str:
        name = self.part.get("name")
        if name == "bash":
            try:
                cmd = json.loads(self.part.get("input") or "{}").get("command") or ""
            except ValueError:
                cmd = ""
            return f"Running bash: {cmd[:120]}"
        return f"Running {name}"

    def output(self, text: str) -> None:
        self._out.append(text)
        if not self.quiet:
            self.svc.tool_output(self.run, self.part, text)

    def output_text(self) -> str:
        return "".join(self._out)

    def fetch(self, url: str):
        return egress.fetch(url)

    def present(self, pane: dict) -> None:
        self.svc.emit(self.run.conv, {"type": "pane.open", "pane": pane}, run_id=self.run_id)

    def changed(self, path: str) -> None:
        pass


class AgentMixin:
    # ================================================================== parts by step
    def new_step(self, run) -> None:
        with run.conv.lock:
            self.flush(run)
            if run.open_kind == "thinking":
                self.close_thinking(run)
            run.open_kind, run.open_index = None, None

    def _pending_add(self, run, key: tuple, text: str) -> None:
        run.pend[key] = run.pend.get(key, "") + text

    def append(self, run, kind: str, text: str) -> None:
        conv = run.conv
        with conv.lock:
            if run.open_kind != kind:
                self.flush(run)
                if run.open_kind == "thinking":
                    self.close_thinking(run)
                part = ({"type": "thinking", "text": "", "tokens": None, "done": False}
                        if kind == "thinking" else {"type": "text", "text": ""})
                part["step"] = run.step
                conv.live["parts"].append(part)
                run.open_kind, run.open_index = kind, len(conv.live["parts"]) - 1
                if kind == "thinking":
                    run.think_started = time.monotonic()
                self.emit(conv, {"type": "part.added", "message_id": run.message_id,
                                 "part": dict(part)}, run_id=run.id)
            self._pending_add(run, (run.open_index, "text"), text)
            if time.monotonic() - run.last_flush >= 0.05:
                self.flush(run)

    def add_call(self, run, call_id: str, name: str) -> int:
        conv = run.conv
        with conv.lock:
            self.flush(run)
            if run.open_kind == "thinking":
                self.close_thinking(run)
            run.open_kind, run.open_index = None, None
            tool = self.tool(name)
            part = {"type": "tool_call", "id": call_id, "name": name,
                    "source": tool.source if tool else {"kind": "builtin"},
                    "risk": tool.risk if tool else None, "input": "", "output": None,
                    "status": "running", "started_at": None, "ended_at": None,
                    "permission": None, "pane": None, "step": run.step}
            conv.live["parts"].append(part)
            self.emit(conv, {"type": "part.added", "message_id": run.message_id,
                             "part": dict(part)}, run_id=run.id)
            return len(conv.live["parts"]) - 1

    def call_part(self, run, k: int) -> dict:
        return run.conv.live["parts"][k]

    def append_input(self, run, k: int, text: str) -> None:
        with run.conv.lock:
            self._pending_add(run, (k, "input"), text)

    def tool_output(self, run, part: dict, text: str) -> None:
        with run.conv.lock:
            self._pending_add(run, (part["id"], "output"), text)

    def flush(self, run) -> None:
        conv = run.conv
        with conv.lock:
            run.last_flush = time.monotonic()
            if not run.pend or conv.live is None:
                return
            pend, run.pend = run.pend, {}
            parts = conv.live["parts"]
            for (key, field), delta in pend.items():
                if field == "output":
                    part = next((p for p in parts if p.get("id") == key), None)
                    if part is None:
                        continue
                    sent = run.out_sent.get(key, 0)
                    if sent >= EVENT_OUTPUT_CAP:
                        continue
                    delta = delta[:EVENT_OUTPUT_CAP - sent]
                    run.out_sent[key] = sent + len(delta)
                    out = part.get("output") or {"content": "", "is_error": False, "bytes": 0}
                    out["content"] += delta
                    out["bytes"] += len(delta.encode())
                    part["output"] = out
                    self.emit(conv, {"type": "tool.output", "message_id": run.message_id,
                                     "call_id": key, "delta": delta}, run_id=run.id)
                    continue
                part = parts[key]
                if field == "input":
                    part["input"] = (part.get("input") or "") + delta
                else:
                    part["text"] = (part.get("text") or "") + delta
                ev = {"type": "part.delta", "message_id": run.message_id, "index": key,
                      "delta": delta}
                ev["field"] = field
                self.emit(conv, ev, run_id=run.id)

    def drop_step(self, run) -> None:
        """A stream that broke while a call was being written: those calls end `cancelled`, and
        the retry is a step of its own."""
        with run.conv.lock:
            self.flush(run)
            for k in run.step_parts:
                p = run.conv.live["parts"][k]
                p["status"], p["ended_at"] = "cancelled", iso(now_ms())
                self.emit(run.conv, {"type": "tool.done", "message_id": run.message_id,
                                     "call_id": p["id"], "status": "cancelled",
                                     "is_error": False}, run_id=run.id)
            run.step_parts, run.tc_parts = [], {}
        run.step += 1

    def step_empty(self, run) -> bool:
        live = run.conv.live
        if live is None:
            return True
        return not any((p.get("text") or "").strip() or p.get("type") == "tool_call"
                       for p in live["parts"] if p.get("step") == run.step
                       and p.get("type") != "thinking")

    def queued(self, conv) -> bool:
        return self.store.one("SELECT 1 FROM queue WHERE conversation_id = ? LIMIT 1",
                              (conv.id,)) is not None

    # ================================================================== the registry
    def tool(self, name: str):
        t = tools_mod.BY_NAME.get(name)
        if t is not None:
            return t
        mcp = getattr(self, "mcp", None)
        t = mcp.tool(name) if mcp is not None else None
        owui = getattr(self, "owui", None)
        return t if t is not None or owui is None else owui.tool(name)

    def _prefs(self) -> dict:
        return {r["tool"]: dict(r) for r in self.store.all("SELECT * FROM tool_prefs")}

    def available_builtins(self) -> list:
        ok = []
        sandbox = self.sandboxes.available()
        library = self.library_available()
        for t in tools_mod.BUILTINS:
            if t.sandbox and not sandbox:
                continue
            if t.needs == "library" and not library:
                continue
            ok.append(t)
        return ok

    def tools_for(self, settings: dict, info: dict, cid: str | None = None,
                  listing: bool = True) -> list:
        """The chat's tools, in the fixed order: built-ins, MCP servers by name, then imported
        Open WebUI toolkits by name (they need the sandbox). `listing` False: an MCP server whose
        tools were never listed is not started for it (the cache peek's body)."""
        if not info.get("tools", True):
            return []
        prefs = self._prefs()
        enabled = set(settings.get("tools") or [])
        if cid and (settings.get("plan_mode") or self.todos(cid)):
            enabled.add("todo_write")
        out = [t for t in self.available_builtins() if t.name in enabled and
               (prefs.get(t.name) is None or prefs[t.name]["enabled"] in (None, 1))]
        if settings.get("plan_mode"):
            # SRV-363: the read tools and the todo list, nothing that changes the world
            return [t for t in out if t.name in PLAN_TOOLS]
        mcp = getattr(self, "mcp", None)
        if mcp is not None:
            out += mcp.tools_for(settings, cid if listing else None)
        owui = getattr(self, "owui", None)
        if owui is not None and self.sandboxes.available():
            out += [t for t in owui.tools_for()
                    if prefs.get(t.name) is None or prefs[t.name]["enabled"] in (None, 1)]
        return out

    def tool_infos(self) -> list[dict]:
        prefs = self._prefs()
        return [self._info(t, prefs) for t in self.available_builtins() + (
            self.mcp.all_tools() if getattr(self, "mcp", None) else []) + (
            self.owui.all_tools() if getattr(self, "owui", None) else [])]

    def _info(self, t, prefs: dict) -> dict:
        p = prefs.get(t.name) or {}
        default = {"allow": "always", "ask": "ask", "deny": "never"}[t.default]
        return {"id": t.name, "name": t.name, "description": t.description,
                "source": t.source, "enabled": p.get("enabled") in (None, 1),
                "policy": p.get("policy") or default, "risk": t.risk,
                "schema_tokens": len(json.dumps(t.openai())) // 4}

    def set_tool(self, name: str, patch: dict) -> dict:
        if self.tool(name) is None:
            from chatd.service import ApiError
            raise ApiError(404, "not_found", "no such tool")
        cur = self._prefs().get(name) or {}
        enabled = patch.get("enabled", None if cur.get("enabled") is None else bool(cur["enabled"]))
        policy = patch.get("policy", cur.get("policy"))
        if policy not in (None, "ask", "always", "never"):
            from chatd.service import ApiError
            raise ApiError(400, "bad_request", "policy is ask, always or never")
        self.store.run("INSERT OR REPLACE INTO tool_prefs VALUES (?, ?, ?)",
                       (name, None if enabled is None else int(bool(enabled)), policy))
        return self._info(self.tool(name), self._prefs())

    # ================================================================== permissions
    def rules(self, cid: str | None = None) -> list[dict]:
        rows = self.store.all("SELECT * FROM permission_rules ORDER BY created")
        return [{"id": r["id"], "tool": r["tool"], "match": r["match"], "decision": r["decision"],
                 "scope": "chat" if r["conversation_id"] else "all",
                 "conversation_id": r["conversation_id"], "created_at": iso(r["created"])}
                for r in rows if cid is None or r["conversation_id"] in (None, cid)]

    def add_rule(self, tool: str, match: str | None, decision: str, cid: str | None) -> dict:
        rid = ulid()
        self.store.run("INSERT INTO permission_rules VALUES (?, ?, ?, ?, ?, ?)",
                       (rid, tool, match or None, decision, cid, now_ms()))
        return next(r for r in self.rules() if r["id"] == rid)

    def delete_rule(self, rid: str) -> None:
        self.store.run("DELETE FROM permission_rules WHERE id = ?", (rid,))

    def _decide(self, cid: str, tool, args: dict, settings: dict) -> tuple[str, str | None]:
        """(allow | ask | deny, the permission mode that allowed it or None): the rules, then the
        chat's mode (SRV-387/SRV-391, tools.decide_mode)."""
        rules = self.rules(cid)
        chat = [r for r in rules if r["conversation_id"] == cid]
        glob = [r for r in rules if r["conversation_id"] is None]
        pref = (self._prefs().get(tool.name) or {}).get("policy")
        mode = self.permission_mode(settings)
        fetched = self._fetched_domains(cid) if tool.name == "web_fetch" and mode == "auto" else ()
        return tools_mod.decide_mode(tool, args, chat, glob, settings, pref, mode, frozenset(fetched))

    def _fetched_domains(self, cid: str) -> set[str]:
        """The domains this chat's web_fetch calls have reached (a call that ran: done or failed
        after it was allowed), which auto mode fetches again without asking."""
        out = set()
        conv = self.conv(cid)
        live = (conv.live or {}).get("parts") or []
        rows = self.store.all("SELECT parts FROM messages WHERE conversation_id = ? AND role = "
                              "'assistant'", (cid,))
        for parts in [json.loads(r["parts"]) for r in rows] + [live]:
            for p in parts:
                if p.get("type") == "tool_call" and p.get("name") == "web_fetch" and \
                        p.get("status") in ("done", "failed"):
                    try:
                        url = json.loads(p.get("input") or "{}").get("url")
                    except ValueError:
                        continue
                    if url:
                        out.add(tools_mod.domain_of(str(url)))
        return out

    def answer_permission(self, cid: str, call_id: str, body: dict) -> None:
        """POST /permissions/{call}: the first answer wins; a second device gets 409."""
        from chatd.service import ApiError
        conv = self.conv(cid)
        run = conv.run
        w = run.waiting.get(call_id) if run is not None else None
        if w is None or w["answer"] is not None:
            raise ApiError(409, "already_answered", "this prompt is not open (answered on another "
                                                    "device, or the run ended)")
        decision = body.get("decision")
        if decision not in ("once", "always", "deny"):
            raise ApiError(400, "bad_request", "decision is once, always or deny")
        w["answer"] = {"decision": decision, "scope": body.get("scope") or "chat",
                       "note": str(body.get("note") or "")[:2000]}
        w["event"].set()

    def _permit(self, run, part: dict, tool, args: dict, settings: dict) -> tuple[str, str]:
        """allow, deny (with the model's note) or cancel. A call the chat's mode allowed runs
        without a prompt and is still recorded: the part's `approval` and a `tool.auto_approved`
        event in the chat's log."""
        d, by = self._decide(run.conv.id, tool, args, settings)
        if d == "allow" and by:
            conv = run.conv
            with conv.lock:
                part["approval"] = {"by": "mode", "mode": by}
                self.emit(conv, {"type": "tool.auto_approved", "message_id": run.message_id,
                                 "call_id": part["id"], "name": tool.name, "mode": by,
                                 "summary": tools_mod.summary(tool.name, args)}, run_id=run.id)
            return "allow", ""
        if d == "allow":
            return "allow", ""
        if d == "deny":
            return "deny", "A permission rule denies this call."
        conv = run.conv
        rule = tools_mod.suggested_rule(tool.name, args)
        words = tools_mod.summary(tool.name, args)
        w = {"event": threading.Event(), "answer": None}
        with conv.lock:
            self.flush(run)
            part["status"] = "asking"
            part["permission"] = {"summary": words, "suggested_rule": rule}
            run.waiting[part["id"]] = w
            self.emit(conv, {"type": "tool.asking", "message_id": run.message_id,
                             "call_id": part["id"], "summary": words, "risk": tool.risk,
                             "suggested_rule": rule}, run_id=run.id)
        run.set_phase("permission", f"Waiting for permission: {tool.name}",
                      tool={"call_id": part["id"], "name": tool.name})
        self.emit_global({"type": "tool.asking", "conversation_id": conv.id,
                          "call_id": part["id"], "summary": words})
        self.summary_changed(conv, status="waiting")
        # the answer can take hours: the run's engine slot goes back meanwhile (live: one chat
        # waiting for a bash prompt held one of the two, and two other chats ran one at a time)
        self.slots.release_if(run)
        w["event"].wait()
        run.waiting.pop(part["id"], None)
        part["permission"] = None
        if w["answer"] is None:
            return "cancel", ""
        a = w["answer"]
        if a["decision"] == "deny":
            note = a["note"]
            return "deny", "The user denied this call." + (f" Their note: {note}" if note else "")
        if a["decision"] == "always":
            self.add_rule(rule["tool"], rule.get("match"), "allow",
                          None if a["scope"] == "all" else conv.id)
        return "allow", ""

    # ================================================================== running the calls
    def run_tools(self, run, idxs: list[int]) -> str:
        conv = run.conv
        self.new_step(run)
        with conv.lock:
            row = self._row(conv.id)
            from chatd.service import DEFAULT_SETTINGS
            settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
            parts = [conv.live["parts"][k] for k in idxs]
        info = self.model_info(settings.get("model"))
        enabled = {t.name: t for t in self.tools_for(settings, info, conv.id)}
        jobs = []
        for part in parts:
            tool = enabled.get(part.get("name") or "")
            try:
                args = tools_mod.parse_args(part.get("input") or "")
            except ToolError as exc:
                self._finish_call(run, part, Result(str(exc), is_error=True))
                continue
            if tool is None and settings.get("plan_mode") and self.tool(part.get("name") or ""):
                self._finish_call(run, part, Result(PLAN_REFUSAL, is_error=True))
                continue
            if tool is None:
                self._finish_call(run, part, Result(
                    f"The tool {part.get('name')!r} is not available in this chat. Available: "
                    f"{', '.join(enabled) or 'none'}.", is_error=True))
                continue
            sig = (tool.name, json.dumps(args, sort_keys=True))
            run.recent_calls.append(sig)
            if len(run.recent_calls) == 3 and len(set(run.recent_calls)) == 1:
                run.notice = (f"The model called {tool.name} with the same arguments three times "
                              f"in a row; the run stopped here.")
                self._finish_call(run, part, Result("Stopped: the same call three times in a row.",
                                                    is_error=True))
                continue
            jobs.append((part, tool, args))
        i = 0
        while i < len(jobs):
            if run.cancel.is_set():
                for part, _, _ in jobs[i:]:
                    self._finish_call(run, part, Result("stopped"), status="cancelled")
                return "cancelled"
            part, tool, args = jobs[i]
            group = [jobs[i]]
            if tool.risk == "read" and not tool.sandbox:
                while i + len(group) < len(jobs) and len(group) < READ_PARALLEL:
                    nxt = jobs[i + len(group)]
                    if nxt[1].risk != "read" or nxt[1].sandbox:
                        break
                    group.append(nxt)
            allowed = []
            for part, tool, args in group:
                verdict, note = self._permit(run, part, tool, args, settings)
                if verdict == "cancel":
                    self._finish_call(run, part, Result("stopped"), status="cancelled")
                    return "cancelled"
                if verdict == "deny":
                    self._finish_call(run, part, Result(note), status="denied")
                    continue
                allowed.append((part, tool, args))
            if len(allowed) > 1:
                with ThreadPoolExecutor(READ_PARALLEL) as pool:
                    list(pool.map(lambda j: self._exec_call(run, *j), allowed))
            for j in allowed if len(allowed) == 1 else []:
                self._exec_call(run, *j)
            i += len(group)
        return "cancelled" if run.cancel.is_set() and run.stop_mode else "ok"

    def _exec_call(self, run, part: dict, tool, args: dict) -> None:
        conv = run.conv
        with conv.lock:
            part["status"], part["started_at"] = "running", iso(now_ms())
            self.emit(conv, {"type": "tool.started", "message_id": run.message_id,
                             "call_id": part["id"], "pane": None}, run_id=run.id)
        ctx = ToolCtx(self, run, part)
        if not tool.sandbox:
            run.set_phase("tool", ctx._running_words(), tool={"call_id": part["id"],
                                                               "name": tool.name})
        if tool.risk in ("write", "exec") and not run.snapshotted:
            run.snapshotted = True
            try:
                self.snaps.take(conv.id, ctx.ws, run.id, "before")
            except OSError:
                pass
        try:
            res = tool.executor(ctx, args)
        except (ToolError, PathError, Unavailable, egress.Refused) as exc:
            res = Result(str(exc), is_error=True)
        except tools_mod.ChangedError:
            res = Result("the file changed meanwhile; read it again", is_error=True)
        except FileNotFoundError:
            res = Result("no such file", is_error=True)
        except Exception as exc:                                    # noqa: BLE001
            self.log(f"[tool] {part['id']} {tool.name} failed: {type(exc).__name__}")
            res = Result(f"the tool failed ({type(exc).__name__})", is_error=True)
        status = "cancelled" if run.cancel.is_set() and run.stop_mode else None
        self._finish_call(run, part, res, status=status, ctx=ctx)

    def _finish_call(self, run, part: dict, res: Result, status: str | None = None,
                     ctx: ToolCtx | None = None) -> None:
        conv = run.conv
        images = []
        for mime, data in res.images:
            images.append(self.store_blob(conv.id, f"{part['id']}.png", mime, data))
        with conv.lock:
            self.flush(run)
            ui, cut = tools_mod.for_ui(res.content)
            st = status or ("failed" if res.is_error else "done")
            out = {"content": ui, "is_error": bool(res.is_error) or st == "failed",
                   "bytes": len(res.content.encode()), "truncated": cut,
                   "exit_code": res.exit_code, "images": images}
            if st in ("denied", "cancelled") and not res.content:
                out = None
            part["output"] = out
            part["model_content"] = tools_mod.for_model(ctx or ToolCtx(self, run, part), part["id"],
                                                        res.content)
            part["status"], part["ended_at"] = st, iso(now_ms())
            if res.pane:
                part["pane"] = res.pane
            self.emit(conv, {"type": "tool.done", "message_id": run.message_id,
                             "call_id": part["id"], "status": st,
                             "is_error": bool(out and out["is_error"]), "output": out},
                      run_id=run.id)

    # ================================================================== the request
    def _compaction_for(self, cid: str, path: list[dict]) -> tuple | None:
        """The newest compaction on `path`: (its summary, the first message kept, None) for one
        before a user turn; for one inside a long run (boundary `<message id>@<step>`), (its
        summary, that run's user message, (the run's message index, its first step kept))."""
        ids = {m["id"]: i for i, m in enumerate(path) if m.get("id")}
        best = None
        for r in self.store.all("SELECT * FROM compactions WHERE conversation_id = ? ORDER BY "
                                "created", (cid,)):
            mid, _, step = str(r["boundary_id"]).partition("@")
            if mid not in ids:
                continue
            i = ids[mid]
            if step:
                u = next((k for k in range(i - 1, -1, -1) if path[k]["role"] == "user"), 0)
                best = (r["summary"], u, (i, int(step)))
            else:
                best = (r["summary"], i, None)
        return best

    @staticmethod
    def _step_from(c) -> tuple[int, int] | None:
        return (c[2][0] - c[1], c[2][1]) if c and c[2] else None

    def compactions(self, cid: str, path: list[dict]) -> list[dict]:
        ids = {m["id"] for m in path}
        return [{"id": r["id"], "at": iso(r["created"]), "before_tokens": r["before_tokens"],
                 "after_tokens": r["after_tokens"], "summary": r["summary"],
                 "duration_ms": r["duration_ms"], "boundary_id": r["boundary_id"]}
                for r in self.store.all("SELECT * FROM compactions WHERE conversation_id = ? "
                                        "ORDER BY created", (cid,))
                if str(r["boundary_id"]).partition("@")[0] in ids]

    def _system(self, settings: dict, tools: list, summary: str | None) -> str | None:
        prompt = None
        if settings.get("prompt_id"):
            r = self.store.one("SELECT template FROM prompts WHERE id = ?",
                               (settings["prompt_id"],))
            prompt = r["template"] if r else None
        skills = self.skill_index(settings) if hasattr(self, "skill_index") else None
        today = _dt.date.today().isoformat() if tools else None
        return render.system_block(today=today, tools=bool(tools), summary=summary,
                                   skills=skills, prompt=prompt,
                                   plan=PLAN_RULE if settings.get("plan_mode") else None)

    def peek_body(self, conv, settings: dict, branch: list[dict]) -> dict:
        """The body of the chat's next call as `request_for` builds it (the system block, the
        tools, the effort, the skills field), over the branch as it is: what the engine's cache
        peek renders (SRV-349). The head must be the next request's, or the peek finds nothing."""
        if settings.get("plan_mode") and settings.get("effort") not in ("xhigh",):
            settings = {**settings, "effort": "high"}
        info = self.model_info(settings.get("model"))
        tools = self.tools_for(settings, info, conv.id, listing=False)
        c = self._compaction_for(conv.id, branch)
        b = c[1] if c else 0
        ours = info.get("backend") == "engine"
        return render.body(settings.get("model") or info.get("id") or "default", settings,
                           branch[b:], system=self._system(settings, tools, c[0] if c else None),
                           vision=bool(info.get("vision")), blob=self._blob_of,
                           tools=[t.openai() for t in tools],
                           tandem=self.tandem_field(settings) if ours else None,
                           inline_images=ours, step_from=self._step_from(c))

    def request_for(self, run) -> tuple[bytes, dict]:
        """The body of the run's next call, after the context guard (chatd/compaction.py)."""
        from chatd.service import DEFAULT_SETTINGS
        conv = run.conv
        if run.step and run.conv.live is not None:
            self.new_step(run)
        with conv.lock:
            row = self._row(conv.id)
            settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
            msgs = self._messages(conv.id)
            path = [msgs[m] for m in self._path(msgs, run.user_message_id)]
            live = conv.live
            if live is not None and live["parts"]:
                # with its id: a compaction inside the run names its steps by it
                path.append({"id": run.message_id, "role": "assistant",
                             "parts": [dict(p) for p in live["parts"]
                                                             if p.get("step", 1) < run.step or
                                                             run.continue_final]})
        if settings.get("plan_mode") and settings.get("effort") not in ("xhigh",):
            settings = {**settings, "effort": "high"}           # a plan turn thinks hard
        user = next((m for m in reversed(path) if m.get("role") == "user"), None)
        if user is not None and not render.text_of(user.get("parts") or []).strip() and \
                not user.get("attachments"):
            # never a turn with an empty user message (the 2026-10-12 empty request)
            raise GuardError("The message is empty: nothing was sent to the model. Write the "
                             "message again, or attach a file.")
        info = self.model_info(settings.get("model"))
        tools = self.tools_for(settings, info, conv.id)
        schemas = [t.openai() for t in tools]
        model = settings.get("model") or info.get("id") or "default"
        window = int(info.get("max_len") or 0)
        ours = info.get("backend") == "engine"
        user_i = next((i for i in range(len(path) - 1, -1, -1) if path[i]["role"] == "user"), 0)

        def build(keep_from: int = 0) -> dict:
            c = self._compaction_for(conv.id, path)
            b = c[1] if c else 0
            return render.body(model, settings, path[b:],
                               system=self._system(settings, tools, c[0] if c else None),
                               vision=bool(info.get("vision")), blob=self._blob_of,
                               continue_final=run.continue_final, tools=schemas,
                               keep_reasoning_from=max(0, keep_from - b),
                               nudge=NUDGE if run.nudge else None,
                               tandem=self.tandem_field(settings) if ours else None,
                               inline_images=ours, step_from=self._step_from(c))
        if ours and run.step <= 1 and not run.remark:
            gone = self.missing_pins(settings)
            if gone:
                run.remark = (f"The pinned skill{'s' if len(gone) > 1 else ''} "
                              f"{', '.join(gone)} {'are' if len(gone) > 1 else 'is'} no longer "
                              f"in the library; the request went without.")
        body = build()
        est = comp.estimate(body, conv.tok_ratio)
        if window and est > int(window * comp.COMPACT_SHARE) and run.compacted != run.step:
            run.compacted = run.step            # once a step: a long run compacts again later
            if self.compact(conv, run, [m for m in path if m.get("id")]):
                body = build()
                est = comp.estimate(body, conv.tok_ratio)
        if window and est + comp.MIN_ANSWER > window:
            body = build(keep_from=user_i)
            est = comp.estimate(body, conv.tok_ratio)
        if window and est + comp.MIN_ANSWER > window:
            raise GuardError(
                f"This conversation is about {est:,} tokens and the model's window is "
                f"{window:,}: there is no room left for an answer. Start a new chat, or edit an "
                f"earlier message to branch off before the long part.")
        if window:
            room = window - est - 256
            cap = int(info.get("max_output") or 0)
            body["max_tokens"] = max(256, min(room, cap) if cap else room)
        # the text's characters and the images' tokens apart: the ratio is learned on text only
        run.request_chars, run.request_images = comp.measure(body)
        run.request_est = est
        self.store.run("UPDATE runs SET request = ?, state = 'running', step = ? WHERE id = ?",
                       (dumps({"messages": len(body["messages"]), "tools": len(schemas),
                               "continue": run.continue_final, "estimate": est}),
                        run.step, run.id))
        return render.encode(body), render.headers(conv.id, settings)

    # ================================================================== compaction
    def compact(self, conv, run=None, path: list[dict] | None = None, force: bool = False) -> bool:
        """Summarize everything before the last two user turns (chatd/compaction.py)."""
        from chatd.service import DEFAULT_SETTINGS
        with conv.lock:
            row = self._row(conv.id)
            settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
            if path is None:
                msgs = self._messages(conv.id)
                path = [msgs[m] for m in self._path(msgs, row["head_id"])]
        b = comp.boundary(path)
        prior = self._compaction_for(conv.id, path)
        sb = None
        done = prior[2] if prior else None             # a long run's compaction: (index, step)
        if b is None or (prior is not None and (done[0] if done else prior[1]) >= b):
            # one long run: its earlier steps instead (its user message stays)
            prior_step = done[1] if done and done[0] == len(path) - 1 else 0
            sb = comp.step_boundary(path, prior_step)
            if sb is None:
                return False
        info = self.model_info(settings.get("model"))
        window = int(info.get("max_len") or 0)
        start = prior[1] if prior else 0
        system = render.system_block(today=None, tools=False, summary=prior[0] if prior else None)
        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        if sb is None:
            for k in range(start, b):
                msgs.extend(render.message(path[k], False, self._blob_of,
                                           from_step=done[1] if done and done[0] == k else 0))
        else:
            i, keep_step = sb
            first = done[1] if done and done[0] == i else 0
            run_msg = dict(path[i], parts=[p for p in path[i].get("parts") or []
                                           if int(p.get("step") or 1) < keep_step])
            for m in path[start:i]:
                msgs.extend(render.message(m, False, self._blob_of))
            msgs.extend(render.message(run_msg, False, self._blob_of, from_step=first))
            b = i
        before = comp.estimate({"messages": msgs}, conv.tok_ratio)
        if window and before + comp.SUMMARY_TOKENS + 512 > window:
            ratio = conv.tok_ratio or comp.DEFAULT_RATIO
            msgs = comp.shrink(msgs, int((window - comp.SUMMARY_TOKENS - 1024) / ratio))
        msgs.append({"role": "user", "content": comp.INSTRUCTIONS})
        body = {"model": settings.get("model") or info.get("id") or "default", "messages": msgs,
                "stream": False, "max_tokens": comp.SUMMARY_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False}}
        if info.get("backend") == "engine":
            body["tandem"] = {"compaction": True}
        if run is not None:
            run.set_phase("prefill", "Compacting the conversation")
        t0 = time.monotonic()
        st, obj = self.engine.post_json("/v1/chat/completions", body,
                                        render.headers(conv.id, settings), timeout=900.0)
        if st != 200 or not isinstance(obj, dict):
            self.log(f"[compact] {conv.id} failed ({st})")
            return False
        try:
            text = obj["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            return False
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()
        if len(text) < 40:
            return False
        usage = obj.get("usage") or {}
        before_tokens = int(usage.get("prompt_tokens") or before)
        if sb is None:
            rest, bid = [x for m in path[b:] for x in render.message(m, False)], path[b]["id"]
        else:
            u = next((k for k in range(b - 1, -1, -1) if path[k]["role"] == "user"), 0)
            rest = [x for m in path[u:b] for x in render.message(m, False)]
            rest += render.message(path[b], False, from_step=sb[1])
            bid = f"{path[b]['id']}@{sb[1]}"              # the run's message and its first step kept
        kept = comp.estimate({"messages": [{"content": text}] + rest}, conv.tok_ratio)
        cid = ulid()
        ms = int((time.monotonic() - t0) * 1000)
        with conv.lock:
            self.store.run("INSERT INTO compactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (cid, conv.id, bid, text, before_tokens, kept, now_ms(), ms,
                            run.id if run else None))
            self.emit(conv, {"type": "compaction", "compaction": {
                "id": cid, "at": iso(now_ms()), "before_tokens": before_tokens,
                "after_tokens": kept, "summary": text, "duration_ms": ms,
                "boundary_id": bid}})
        self.log(f"[compact] {conv.id} {before_tokens} -> {kept} tokens in {ms} ms")
        return True

    def compact_now(self, cid: str) -> None:
        from chatd.service import ApiError
        conv = self.conv(cid)
        if conv.run is not None and conv.run.active:
            raise ApiError(409, "run_active", "a run is on; compaction happens before its next "
                                              "model call when it is needed")
        threading.Thread(target=self._compact_bg, args=(conv,), daemon=True).start()

    def _compact_bg(self, conv) -> None:
        if not self.compact(conv, force=True):
            self.emit(conv, {"type": "context.changed", "context": conv.context}, persist=False) \
                if conv.context else None
