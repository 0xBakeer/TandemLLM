"""Tier 0 checks on held tool calls (SRV-113): C4 malformed, C3 repeated, C1 risk floors, C2 project
rules. Registered with `tier0.register` at import, so a task in the policy file names them like any
other rule, and their parameters (every pattern, list, limit and text) are the policy's.

The checks run when the answer ends (the end token, `Supervisor.check_answer`): one
`ToolCallEvent` for every call and for every block that does not parse, before any of them is
released. A rule returns

    REPAIR_CALL      a deterministic fix: `detail["arguments"]` for a call, `detail["calls"]` for a
                     block that does not parse (what it parses into once fixed)
    REGENERATE_CALL  its `text` is the reminder the model gets in a hidden turn; every call of the
                     answer is then discarded and the model writes the set again
    GIVE_UP          (C4, a call the token limit cut) nothing of it goes out
    PASS + reason    an observation, journaled; nothing changes
    None             nothing found

`detail["mode"]` lowers what the task may do with one finding (C4's schema check runs in shadow
while C4 itself acts). Everything a rule reads is the event, the policy task and the request: its
messages are the session's tool history (earlier calls, their results, the system prompt with
opencode's working directory and AGENTS.md). Never the reasoning; never the model's logits.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex

from server import toolcall
from server.conductor.monitor.policy import GIVE_UP, PASS, REGENERATE_CALL, REPAIR_CALL, Action
from server.conductor.monitor.session import args_hash
from server.conductor.monitor.tier0 import register

_TEXT = {"type": "string", "minLength": 1}
_TEXTS = {"type": "array", "items": _TEXT, "minItems": 1,
          "description": "the n-th reminder for the n-th regeneration in a request (the last one "
                         "repeats)"}
_STRS = {"type": "array", "items": _TEXT}
_COUNT = {"type": "integer", "minimum": 0}
_MODE = {"enum": ["off", "shadow", "act"]}


# ------------------------------------------------------------------ what the request says

def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _hash(content) -> str:
    raw = content if isinstance(content, str) else json.dumps(content, sort_keys=True,
                                                               ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _args(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        got = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def tool_history(messages) -> list[dict]:
    """Every call the session made, oldest first: `{name, args, args_hash, result_hash,
    result_line}`, the result matched by `tool_call_id` (by position when a client sends none);
    `result_hash` is None while a call has no result."""
    calls: list[dict] = []
    by_id: dict[str, dict] = {}
    pending: list[dict] = []
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            pending = []
            for c in m.get("tool_calls") or ():
                fn = (c or {}).get("function") or {}
                rec = {"name": str(fn.get("name") or ""), "args": _args(fn.get("arguments")),
                       "args_hash": args_hash(fn.get("arguments") or "{}"),
                       "result_hash": None, "result_line": ""}
                calls.append(rec)
                pending.append(rec)
                if c.get("id"):
                    by_id[str(c["id"])] = rec
        elif m.get("role") == "tool":
            rec = by_id.get(str(m.get("tool_call_id") or "")) or (pending.pop(0) if pending
                                                                 else None)
            if rec is None:
                continue
            if rec in pending:
                pending.remove(rec)
            content = m.get("content")
            rec["result_hash"] = _hash(content)
            rec["result_line"] = next((ln.strip() for ln in _text(content).splitlines()
                                       if ln.strip()), "")
    return calls


def system_text(messages) -> str:
    for m in messages or ():
        if isinstance(m, dict) and m.get("role") == "system":
            return _text(m.get("content"))
        break
    return ""


def _memo(sup, key, make):
    got = sup.memo.get(key)
    if got is None:
        got = sup.memo[key] = make()
    return got


def _search(pattern: str, text: str, group: int = 1) -> str | None:
    try:
        m = re.search(pattern, text, re.M)
    except re.error:
        return None
    return m.group(group) if m and m.groups() else (m.group(0) if m else None)


def project_root(sup, task) -> str | None:
    """The working directory the client states in its system prompt (opencode's `<env>`)."""
    return _memo(sup, "tool:root", lambda: (_search(task["root"], system_text(sup.req.messages))
                                            or "").rstrip("/") or "") or None


def _home(root: str | None, task) -> str | None:
    return _search(task["home"], root or "") if root else None


def _norm(path: str, root: str | None, home: str | None) -> str | None:
    """`path` as an absolute, normalised path; None when it cannot be placed (a `~` with no
    known home, a relative path with no root)."""
    p = path.strip().strip("'\"")
    if p.startswith("~"):
        if not home:
            return None
        p = home + p[1:]
    elif not p.startswith("/"):
        if not root:
            return None
        p = posixpath.join(root, p)
    return posixpath.normpath(p)


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _strings(args: dict, keys) -> list[tuple[str, str]]:
    """(key, value) of the string arguments named in `keys` (all of them when `keys` is empty)."""
    return [(k, v) for k, v in args.items()
            if isinstance(v, str) and (not keys or k in keys)]


# ------------------------------------------------------------------ the argument schema

_JSON_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list}


def _is(value, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool) or \
            (isinstance(value, float) and value.is_integer())
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "null":
        return value is None
    t = _JSON_TYPES.get(kind)
    return True if t is None else isinstance(value, t)


def schema_problems(arguments: str, schema) -> list[str]:
    """What the arguments break of the tool's `parameters`: required keys that are missing and
    values whose JSON type the schema does not allow. The two checks a client's validator makes
    first; nothing it would not refuse."""
    if not isinstance(schema, dict):
        return []
    try:
        args = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return ["the arguments are not a JSON object"]
    if not isinstance(args, dict):
        return ["the arguments are not a JSON object"]
    out = [f"missing required '{k}'" for k in schema.get("required") or ()
           if isinstance(k, str) and k not in args]
    props = schema.get("properties") or {}
    for k, v in args.items():
        sub = props.get(k) if isinstance(props, dict) else None
        t = sub.get("type") if isinstance(sub, dict) else None
        kinds = t if isinstance(t, list) else [t] if isinstance(t, str) else []
        if kinds and not any(_is(v, x) for x in kinds):
            out.append(f"'{k}' is {_json_name(v)}, the schema says {'/'.join(kinds)}")
    return out


def _json_name(v) -> str:
    """A value's JSON type, in the words a schema uses (the reminder quotes it to the model)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "a boolean"
    if isinstance(v, (int, float)):
        return "a number"
    return {str: "a string", dict: "an object", list: "an array"}.get(type(v), type(v).__name__)


def _nth(texts: list, n: int) -> str:
    return texts[min(n, len(texts) - 1)]


# ------------------------------------------------------------------ C4: malformed

@register("malformed_call", "SP3", params={
    "texts": dict(_TEXTS, description="the reminder after a block that does not parse; {error} "
                                      "is the parser's reason"),
    "regenerations": dict(_COUNT, description="regenerations a request may spend on unreadable "
                                              "calls; then the block goes out as content, as "
                                              "without the checks"),
    "repair": {"type": "boolean", "description": "try the deterministic fixes first (JSON with "
                                                 "trailing commas or Python literals, an unquoted "
                                                 "name, a missing </parameter>)"},
    "schema": dict(_MODE, description="the arguments against the tool's JSON schema (required "
                                      "keys, types): at most this mode"),
    "schema_text": dict(_TEXT, description="the reminder for a schema problem: {name}, {problem}"),
    "cut": {"enum": ["content", "give_up"],
            "description": "a call the token limit cut: content (as without the checks) or "
                           "GIVE_UP (nothing of half a call goes out)"},
}, required=("texts", "regenerations", "repair", "schema", "schema_text", "cut"))
def malformed_call(ev, task, sup):
    """C4: a call that does not parse is repaired or written again; arguments that break the
    tool's schema are journaled (shadow in wave 1)."""
    if ev.cut:
        if task["cut"] == "give_up":
            return Action(GIVE_UP, reason="cut_by_length", detail={"chars": len(ev.block)})
        return Action(PASS, reason="cut_by_length(content)", detail={"chars": len(ev.block)})
    done = sup.task_acts.get(task.id, 0)
    if ev.error is not None:
        if task["repair"]:
            got = toolcall.repair(ev.block, sup.tool_types)
            if got is not None:
                calls, fixed = got
                return Action(REPAIR_CALL, reason=f"repair({', '.join(fixed)})",
                              detail={"calls": calls, "fixed": fixed, "error": ev.error})
        if done >= task["regenerations"]:
            return Action(PASS, reason=f"fallback(regenerations={done})",
                          detail={"error": ev.error})
        return Action(REGENERATE_CALL, reason=f"unreadable({ev.error})",
                      text=_nth(task["texts"], done).replace("{error}", ev.error),
                      detail={"error": ev.error, "act": done + 1})
    if ev.block and task["repair"]:
        # it parsed, but without a parameter whose closer never came: close it and keep it
        got = toolcall.repair(ev.block, sup.tool_types)
        same = [c for c in (got[0] if got else ()) if c["function"]["name"] == ev.name]
        if same and same[0]["function"]["arguments"] != ev.arguments:
            return Action(REPAIR_CALL, reason=f"repair({', '.join(got[1])})",
                          detail={"arguments": same[0]["function"]["arguments"],
                                  "fixed": got[1]})
    if task["schema"] == "off" or ev.schema is None:
        return None
    problems = schema_problems(ev.arguments, ev.schema)
    if not problems:
        return None
    if done >= task["regenerations"]:
        return Action(PASS, reason=f"schema({problems[0]})", detail={"problems": problems})
    return Action(REGENERATE_CALL, reason=f"schema({problems[0]})",
                  text=task["schema_text"].replace("{name}", ev.name)
                  .replace("{problem}", "; ".join(problems[:3])),
                  detail={"problems": problems, "mode": task["schema"]})


# ------------------------------------------------------------------ C3: the same call again

@register("repeated_call", "SP3", params={
    "repeats": {"type": "integer", "minimum": 2,
                "description": "the call that would be the n-th in a row with the same name and "
                               "arguments, after n-1 identical results"},
    "texts": dict(_TEXTS, description="the reminder: {name}, {count} (the earlier runs), "
                                      "{first_line} (of their result)"),
    "first_line_chars": {"type": "integer", "minimum": 8},
}, required=("repeats", "texts", "first_line_chars"))
def repeated_call(ev, task, sup):
    """C3: the same call a third time after two identical results -> write something else.
    A call whose earlier results differ (a screenshot, a file being written) passes."""
    if ev.error is not None or not ev.name:
        return None
    need = task["repeats"] - 1
    hist = _memo(sup, "tool:history", lambda: tool_history(sup.req.messages))
    if len(hist) < need:
        return None
    last = hist[-need:]
    h = args_hash(ev.arguments)
    if any(c["name"] != ev.name or c["args_hash"] != h for c in last):
        return None
    results = {c["result_hash"] for c in last}
    if None in results or len(results) != 1:
        return None
    line = last[-1]["result_line"][:task["first_line_chars"]]
    text = (_nth(task["texts"], sup.task_acts.get(task.id, 0)).replace("{name}", ev.name)
            .replace("{count}", str(need)).replace("{first_line}", line))
    return Action(REGENERATE_CALL, reason=f"repeated({ev.name} x{task['repeats']})", text=text,
                  detail={"name": ev.name, "count": need, "first_line": line})


# ------------------------------------------------------------------ C1: risk floors

_FLOOR = {
    "type": "object",
    "properties": {
        "id": _TEXT, "label": _TEXT,
        "kind": {"enum": ["pattern", "rm_outside_root", "write_outside_root"],
                 "description": "pattern: the regex matches an argument; rm_outside_root: an "
                                "`rm` the regex finds removes a path outside the project root; "
                                "write_outside_root: a path argument lies outside the root"},
        "tools": dict(_STRS, description="tool names it applies to; empty: every tool"),
        "args": dict(_STRS, description="argument keys it reads; empty: every string argument"),
        "pattern": {"type": "string",
                    "description": "a regular expression; for rm_outside_root its group "
                                   "`targets` holds the paths"},
    },
    "required": ["id", "label", "kind", "tools", "args", "pattern"],
    "additionalProperties": False,
}


def _floor_hit(floor: dict, ev, args: dict, root: str | None, home: str | None) -> str | None:
    """What a floor matched in a call (a short excerpt), or None."""
    if floor["tools"] and ev.name not in floor["tools"]:
        return None
    for _, val in _strings(args, floor["args"]):
        if floor["kind"] == "pattern":
            got = _search(floor["pattern"], val, 0)
            if got:
                return got[:120]
        elif floor["kind"] == "rm_outside_root":
            if not root:
                continue
            try:
                found = list(re.finditer(floor["pattern"], val, re.M))
            except re.error:
                return None
            cwd = _norm(args["workdir"], root, home) if isinstance(args.get("workdir"), str) \
                else root
            for m in found:
                try:
                    targets = shlex.split(m.group("targets"))
                except (ValueError, IndexError):
                    continue
                for t in targets:
                    p = _norm(t, cwd, home)
                    if p is None or not _inside(p, root):
                        return f"rm {t}"
        elif floor["kind"] == "write_outside_root":
            if not root:
                continue
            p = _norm(val, root, home)
            if p is None or not _inside(p, root):
                return val[:120]
    return None


@register("risk_floor", "SP3", params={
    "floors": {"type": "array", "items": _FLOOR},
    "root": dict(_TEXT, description="a regex whose first group is the project root, read from the "
                                    "system prompt (opencode states its working directory)"),
    "home": dict(_TEXT, description="a regex whose first group is the home directory, read from "
                                    "the root, for paths that start with ~"),
    "text": dict(_TEXT, description="the reminder when a floor acts: {label}, {match}"),
}, required=("floors", "root", "home", "text"))
def risk_floor(ev, task, sup):
    """C1: the risk floors (arbiter's guard_policy as a list): rm -rf outside the project root,
    git push, writes outside the root, a download piped into a shell."""
    if ev.error is not None or not ev.name:
        return None
    args = _args(ev.arguments)
    root = project_root(sup, task)
    home = _home(root, task)
    for floor in task["floors"]:
        hit = _floor_hit(floor, ev, args, root, home)
        if hit:
            return Action(REGENERATE_CALL, reason=f"floor({floor['id']})",
                          text=task["text"].replace("{label}", floor["label"])
                          .replace("{match}", hit),
                          detail={"floor": floor["id"], "match": hit, "root": root})
    return None


# ------------------------------------------------------------------ C2: project rules

@register("project_rules", "SP3", params={
    "requires": {"type": "string", "description": "a regex the system prompt must match for the "
                                                  "rules to apply (the AGENTS.md text opencode "
                                                  "sends); empty: always"},
    "path_keys": dict(_STRS, description="argument keys that hold a file path"),
    "write_tools": dict(_STRS, description="tools that create or overwrite a file"),
    "content_keys": dict(_STRS, description="argument keys that hold a written file's text"),
    "numbered_copy": {"type": "object", "properties": {
        "enabled": {"type": "boolean"},
        "suffix": dict(_TEXT, description="a regex for what a numbered copy adds to a known "
                                          "file's stem")},
        "required": ["enabled", "suffix"], "additionalProperties": False,
        "description": "a new file named like a file the session already used, plus a number"},
    "path_prefixes": dict(_STRS, description="paths under these break the rule (/tmp/)"),
    "max_lines": {"type": "integer", "minimum": 1,
                  "description": "a written file longer than this breaks the rule"},
    "root": dict(_TEXT, description="as C1's: the project root from the system prompt"),
    "home": dict(_TEXT, description="as C1's"),
    "text": dict(_TEXT, description="the reminder when a rule acts: {rule}, {path}, {detail}"),
}, required=("requires", "path_keys", "write_tools", "content_keys", "numbered_copy",
             "path_prefixes", "max_lines", "root", "home", "text"))
def project_rules(ev, task, sup):
    """C2: AGENTS.md rules that text alone can decide: a numbered copy of a file the session
    knows, a path under /tmp, a written file over 400 lines."""
    if ev.error is not None or not ev.name:
        return None
    if task["requires"] and not _search(task["requires"], system_text(sup.req.messages), 0):
        return None
    args = _args(ev.arguments)
    root = project_root(sup, task)
    home = _home(root, task)
    paths = [(k, v) for k, v in _strings(args, task["path_keys"])]

    def act(rule: str, path: str, detail: str):
        return Action(REGENERATE_CALL, reason=f"rule({rule})",
                      text=task["text"].replace("{rule}", rule).replace("{path}", path)
                      .replace("{detail}", detail),
                      detail={"rule": rule, "path": path, "detail": detail})

    for _, p in paths:
        full = _norm(p, root, home) or p
        for pre in task["path_prefixes"]:
            if full.startswith(pre) or p.startswith(pre):
                return act("path_prefix", p, f"under {pre}")
    if ev.name in task["write_tools"]:
        nc = task["numbered_copy"]
        if nc["enabled"]:
            known = _memo(sup, "tool:known", lambda: {
                (_norm(v, root, home) or v) for c in tool_history(sup.req.messages)
                for _, v in _strings(c["args"], task["path_keys"])})
            for _, p in paths:
                full = _norm(p, root, home) or p
                if full in known:
                    continue
                d, base = posixpath.split(full)
                stem, ext = posixpath.splitext(base)
                for k in known:
                    kd, kb = posixpath.split(k)
                    ks, ke = posixpath.splitext(kb)
                    if kd != d or ke != ext or not ks or ks == stem:
                        continue
                    try:
                        if re.fullmatch(re.escape(ks) + f"(?:{nc['suffix']})", stem):
                            return act("numbered_copy", p, f"a copy of {kb}")
                    except re.error:
                        break
        for _, text in _strings(args, task["content_keys"]):
            n = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
            if n > task["max_lines"]:
                return act("max_lines", paths[0][1] if paths else "", f"{n} lines, over "
                                                                      f"{task['max_lines']}")
    return None
