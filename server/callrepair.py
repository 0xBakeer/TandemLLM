"""Argument names matched to the offered tool's schema before a call leaves the server (SRV-691), and a call to a
tool the request did not offer mapped to an offered equivalent (SRV-693).

A model that names an argument the way another harness does -- opencode's `read` takes `filePath`, and
Flint 4.0 at its card's sampling wrote `path` in about 2 of 8 asks (SRV-690) -- has the call refused by
the client, which validates the arguments against the tool's schema. Here a call's argument names are
matched to the schema's, in order:

  1  exact: kept
  2  the same name but for case, `_` and `-` (file_path = filePath = filepath)
  3  a synonym of a small table: a path (path, file, filename, file_path) for the schema's one path-like
     parameter; query, pattern and regex; cmd and command; content, text and contents

and renamed only when exactly one parameter of the schema is the target and the call does not set it
already; a call not streamed yet is renamed only when it then has every required parameter. A value is
never changed or invented. Format-independent: it works on the JSON tier every tool-call form is read
into (server/toolformats), on both paths (raw and conductor). Every change is returned as an event
({kind: tool|argument, tool, from, to}) for the journal.

A call to a tool the request did not offer (`map_tool`) goes to the one offered tool of the same kind
(grep, glob, list, read, write, edit, bash by another harness's word) when its arguments then name that
tool's required parameters, else, for a search or a listing, to bash with a quoted command when bash is
offered; otherwise it goes out as written and the conductor's C4 has it written again.

The stream's buffer (server/toolcall.py `ToolCallBuffer`) renames a key when the key is decided, with the
same rule (`rename_key`), so what went out live and the call the closer parses agree.
"""

from __future__ import annotations

import json
import re

#: synonym groups, by normalized name (`_norm`); the path group also takes any parameter ending in "path"
_PATH = frozenset({"path", "file", "filename", "filepath", "fname"})
_GROUPS = (_PATH, frozenset({"query", "pattern", "regex"}), frozenset({"cmd", "command"}),
           frozenset({"content", "text", "contents"}))


def _norm(k: str) -> str:
    return re.sub(r"[\s_\-]", "", str(k)).lower()


def _props(schema) -> dict:
    p = schema.get("properties") if isinstance(schema, dict) else None
    return p if isinstance(p, dict) else {}


def _required(schema) -> list:
    r = schema.get("required") if isinstance(schema, dict) else None
    return [k for k in r if isinstance(k, str)] if isinstance(r, list) else []


def _group(n: str):
    return next((g for g in _GROUPS if n in g), None)


def rename_key(key: str, schema, used) -> str | None:
    """The schema parameter `key` stands for, or None to keep `key`: none when `key` is the schema's or
    the target is not exactly one parameter the call has not set (`used`)."""
    props = _props(schema)
    if not props or key in props:
        return None
    n = _norm(key)
    same = [p for p in props if _norm(p) == n]
    if same:
        free = [p for p in same if p not in used]
        return free[0] if len(same) == 1 and free else None
    g = _group(n)
    if g is None:
        return None
    cands = [p for p in props if _norm(p) in g or (g is _PATH and _norm(p).endswith("path"))]
    if len(cands) != 1 or cands[0] in used:
        return None
    return cands[0]


def repair_args(name: str, args: dict, schema, gate: bool = True, kinds: dict | None = None) -> tuple[dict, list]:
    """`(arguments, renames)` of one call: its keys matched to `schema` (`rename_key`) in the call's own
    order, a key counting as set from where it is written on (as the stream decides it). `gate`: rename
    only when the call then has every required parameter (a call that streamed live keeps the renames it
    streamed: `gate=False`). `kinds` ({parameter: JSON types}): a renamed string value is converted as the
    parser converts the target's (server/toolcall.py `convert`)."""
    if not isinstance(args, dict) or not _props(schema):
        return args, []
    out, renames, used = {}, [], set()
    for k, v in args.items():
        to = rename_key(k, schema, used)
        if to is None:
            out[k] = v
            used.add(k)
            continue
        if kinds and to in kinds and isinstance(v, str):
            from server.toolcall import convert
            v = convert(v, kinds[to])
        out[to] = v
        used.add(to)
        renames.append({"kind": "argument", "tool": name, "from": k, "to": to})
    if not renames:
        return args, []
    if gate and not all(r in out for r in _required(schema)):
        return args, []
    return out, renames


def repair_call(call: dict, schemas: dict | None, gate: bool = True, types: dict | None = None) -> tuple[dict, list]:
    """`(call, events)`: an OpenAI call (`{"function": {"name", "arguments"}}`, arguments a JSON object's
    text) whose tool the request did not offer mapped to an offered one where that is unambiguous (SRV-693,
    `map_tool`), then its argument names matched to its tool's schema in `schemas` ({name: parameters});
    `types` as the parser's (server/toolcall.py `schema_types`). An event is {kind: tool|argument, tool,
    from, to}."""
    fn = call.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return call, []
    events = []
    name = fn.get("name")
    if schemas and name not in schemas:
        got = map_tool(name, args, schemas)
        if got is None:
            return call, []
        name, args, events = got
    schema = (schemas or {}).get(name)
    if not schema:
        return call, []
    new, renames = repair_args(name, args, schema, gate=gate, kinds=(types or {}).get(name))
    events += renames
    if not events:
        return call, []
    return dict(call, function=dict(fn, name=name, arguments=json.dumps(new))), events


def repair_calls(calls: list, schemas: dict | None, types: dict | None = None) -> tuple[list, list]:
    """Every call of a list (the non-streamed answer), and all their renames."""
    out, renames = [], []
    for c in calls:
        c2, r = repair_call(c, schemas, types=types)
        out.append(c2)
        renames += r
    return out, renames


#: tools a model names by another harness's word, by the word the client may offer (normalized, `_norm`);
#: SRV-693
_TOOLS = (frozenset({"grep", "search", "rg", "ripgrep", "grepsearch", "searchfiles"}),
          frozenset({"glob", "findfiles", "filesearch", "globfiles"}),
          frozenset({"list", "ls", "listdir", "listdirectory", "listfiles"}),
          frozenset({"read", "readfile", "cat", "view", "openfile"}),
          frozenset({"write", "writefile", "createfile"}),
          frozenset({"edit", "editfile", "strreplace", "replaceinfile"}),
          frozenset({"bash", "shell", "run", "runcommand", "execute", "exec", "terminal", "runshellcommand"}))


def _shell_of(group, args: dict) -> str | None:
    """The bash command for a search or a listing the client has no tool for (SRV-693): read-only only (grep,
    ls; never a write, a delete or a redirection), every value quoted, none invented; None when the call has
    no pattern (a search) or is any other kind of tool, which C4 has written again."""
    import shlex
    pick = {_norm(k): v for k, v in args.items() if isinstance(v, (str, int, float))}
    if "grep" in group:
        pat = next((pick[k] for k in ("pattern", "query", "regex") if k in pick), None)
        if pat is None:
            return None
        inc = pick.get("include")
        path = next((pick[k] for k in ("path", "directory", "dir") if k in pick), None)
        return ("grep -rn " + (f"--include={shlex.quote(str(inc))} " if inc else "") + "-- "
                + shlex.quote(str(pat)) + (" " + shlex.quote(str(path)) if path else ""))
    if "ls" in group:
        path = next((pick[k] for k in ("path", "directory", "dir") if k in pick), None)
        return "ls -la" + (" -- " + shlex.quote(str(path)) if path else "")
    return None


def map_tool(name: str, args: dict, schemas: dict) -> tuple[str, dict, list] | None:
    """SRV-693: a call to a tool the request did not offer, as `(tool, arguments, events)` of an offered one
    when exactly one offered tool is its equivalent and the call's arguments then name its required
    parameters, else a search or a listing as a quoted bash command when bash is offered; None otherwise
    (nothing invented: the conductor's C4 has it written again)."""
    n = _norm(name)
    group = next((g for g in _TOOLS if n in g), None)
    if group is None:
        return None
    same = [o for o in schemas if _norm(o) in group]
    if len(same) == 1:
        new, renames = repair_args(same[0], args, schemas[same[0]])
        if all(r in new for r in _required(schemas[same[0]])) and set(new) <= set(_props(schemas[same[0]])):
            return same[0], new, [{"kind": "tool", "tool": name, "from": name, "to": same[0]}] + renames
    shell = [o for o in schemas if _norm(o) in _TOOLS[-1]]
    cmd = _shell_of(group, args) if len(shell) == 1 else None
    if cmd is None:
        return None
    key = next((p for p in _props(schemas[shell[0]]) if _norm(p) in ("command", "cmd")), None)
    if key is None:
        return None
    return shell[0], {key: cmd}, [{"kind": "tool", "tool": name, "from": name, "to": shell[0], "command": cmd}]


def schemas_of(tools) -> dict:
    """{name: parameters} of a request's tools (the flat form read as a function tool)."""
    out = {}
    for t in tools or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn.get("name"), str) and isinstance(fn.get("parameters"), dict):
            out[fn["name"]] = fn["parameters"]
    return out
