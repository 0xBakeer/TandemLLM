"""The tools a chat can offer the model, their executors, and the permission rules (SRV-353).

THE REGISTRY. Each tool: name, description, JSON schema, source (builtin, mcp:<server>, user,
skill), risk (read, write, exec, network), the default permission (allow, ask, deny), whether it
needs the chat's sandbox, and its executor. Tools go to the model in one fixed order (the
built-ins as listed, then MCP servers by name, then user tools), so the prompt head stays
byte-stable and the prefix cache holds. MCP tools reach the model as `mcp__<server>__<tool>`.

THE EXECUTORS run on the host where nothing executes (read_file, write_file, edit_file, glob,
grep: chatd/workspace.py, symlink-safe), in the sandbox where something does (bash, render_page),
and through the egress filter where the network is touched (web_fetch: the same private-range
refusal as the sandbox's proxy, chatd/egress.py). A result for the model is at most 16,000 tokens:
a longer one is written to /workspace/.tandem/outputs/<call>.txt and the model gets its first and
last 4,000 tokens and the path. The part the UI keeps is cut at 30,000 characters (CONTRACT 5).

PERMISSIONS (Claude Code's model). A call is decided by: the chat's rules (deny, then allow), the
global rules (deny, then allow), the chat's policy for the tool (always, never, ask), "auto-accept
edits" for write tools, and the tool's default. A rule names a tool and, for some, what it may do:
`bash` a command prefix by whole words ("npm install" matches `npm install three`, never `npm
installx`; a command with `$(...)` or backticks matches no rule, and a compound command matches
only when every part does), `web_fetch` a domain and its subdomains, an MCP tool a glob on its
name (`mcp__x1__*`).
"""

from __future__ import annotations

import fnmatch
import html.parser
import json
import os
import re
import shlex
import urllib.parse

from chatd.workspace import ChangedError, PathError

MODEL_LIMIT_CHARS = 64_000          # ~16,000 tokens
MODEL_EDGE_CHARS = 16_000           # ~4,000 tokens kept from each end
UI_LIMIT_CHARS = 30_000
BASH_TIMEOUT_S = 600
OUTPUTS_DIR = ".tandem/outputs"
SHOTS_DIR = ".tandem/shots"


class Tool:
    def __init__(self, name, description, schema, *, risk="read", default="allow", sandbox=False,
                 source=None, executor=None, needs=None):
        self.name = name
        self.description = description
        self.schema = schema
        self.risk = risk
        self.default = default
        self.sandbox = sandbox
        self.source = source or {"kind": "builtin"}
        self.executor = executor
        self.needs = needs              # a capability the box must have ("library")

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name,
                                                 "description": self.description,
                                                 "parameters": self.schema}}


class Result:
    def __init__(self, content: str = "", *, is_error=False, exit_code=None, images=None,
                 pane=None):
        self.content = content
        self.is_error = is_error
        self.exit_code = exit_code
        self.images = images or []      # [(mime, bytes)]
        self.pane = pane


class ToolError(Exception):
    """A call that cannot run as asked: its message is the tool result the model reads."""


def _obj(props: dict, required: list) -> dict:
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}


# ---------------------------------------------------------------------- executors
def _bash(ctx, a):
    cmd = str(a.get("command") or "").strip()
    if not cmd:
        raise ToolError("command is empty")
    if a.get("background"):
        log = f"/workspace/{OUTPUTS_DIR}/{ctx.call_id}.log"
        wrapped = (f"mkdir -p /workspace/{OUTPUTS_DIR} && nohup bash -lc {shlex.quote(cmd)} "
                   f"> {log} 2>&1 & echo started pid $!")
        code, _ = ctx.exec(["bash", "-lc", wrapped], timeout=30)
        return Result(f"{ctx.output_text().strip()}\nThe command runs in the background; its "
                      f"output goes to {log}.", exit_code=code)
    timeout = min(BASH_TIMEOUT_S, max(1, int(a.get("timeout_s") or BASH_TIMEOUT_S)))
    code, timed_out = ctx.exec(["bash", "-lc", cmd], timeout=timeout)
    out = ctx.output_text()
    if timed_out:
        return Result(out + f"\n[timed out after {timeout} s; the command was stopped]",
                      is_error=True, exit_code=code)
    tail = f"\n[exit code {code}]" if code else ""
    return Result(out + tail, is_error=bool(code), exit_code=code)


def _read(ctx, a):
    path = str(a.get("path") or "")
    st = ctx.ws.stat(path)
    if os.path.isdir(ctx.ws.resolve(path)):
        raise ToolError(f"{path} is a directory; use glob to list it")
    data = ctx.ws.read(path, limit=4 * 1024 * 1024)
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        mime = "image/" + path.lower().rsplit(".", 1)[1].replace("jpg", "jpeg")
        return Result(f"{path}: an image, {st.st_size:,} bytes", images=[(mime, data)])
    if b"\0" in data[:4096]:
        return Result(f"{path}: a binary file, {st.st_size:,} bytes; not shown")
    lines = data.decode("utf-8", "replace").splitlines()
    off = max(1, int(a.get("offset") or 1))
    lim = int(a.get("limit") or 2000)
    chunk = lines[off - 1:off - 1 + lim]
    body = "\n".join(f"{i:6d}\t{ln}" for i, ln in enumerate(chunk, off))
    more = f"\n[{len(lines) - (off - 1 + len(chunk))} more lines]" if off - 1 + len(chunk) < \
        len(lines) else ""
    return Result(body + more)


def _write(ctx, a):
    path, content = str(a.get("path") or ""), a.get("content")
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    ctx.ws.write(path, content.encode())
    ctx.changed(path)
    return Result(f"wrote {ctx.ws.show(ctx.ws.resolve(path))} ({len(content.encode()):,} bytes)")


def _edit(ctx, a):
    path, old, new = str(a.get("path") or ""), a.get("old"), a.get("new")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        raise ToolError("old and new must be strings, old not empty")
    text = ctx.ws.read(path).decode("utf-8")
    n = text.count(old)
    if n == 0:
        raise ToolError("old was not found in the file; read it again and copy it exactly")
    if n > 1 and not a.get("replace_all"):
        raise ToolError(f"old occurs {n} times; give more context to make it unique, or set "
                        f"replace_all")
    out = text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1)
    ctx.ws.write(path, out.encode())
    ctx.changed(path)
    return Result(f"edited {ctx.ws.show(ctx.ws.resolve(path))}: {n if a.get('replace_all') else 1}"
                  f" replacement(s), {len(old.splitlines()) or 1} line(s) out, "
                  f"{len(new.splitlines()) or 1} in")


def _glob(ctx, a):
    hits = ctx.ws.glob(str(a.get("pattern") or "**/*"), str(a.get("path") or "/workspace"))
    return Result("\n".join(hits) if hits else "no files match")


def _grep(ctx, a):
    hits = ctx.ws.grep(str(a.get("pattern") or ""), str(a.get("path") or "/workspace"),
                       a.get("include") or None, bool(a.get("ignore_case")))
    return Result("\n".join(hits) if hits else "no matches")


def _fetch(ctx, a):
    url = str(a.get("url") or "")
    status, ctype, body = ctx.fetch(url)
    text = body.decode("utf-8", "replace")
    if "html" in ctype:
        text = html_to_text(text)
    return Result(f"HTTP {status} {ctype}\n\n{text}", is_error=status >= 400)


def _render(ctx, a):
    w = min(2400, max(320, int(a.get("width") or 1280)))
    h = min(2400, max(240, int(a.get("height") or 800)))
    if a.get("port"):
        url = f"http://127.0.0.1:{int(a['port'])}/{str(a.get('path') or '').lstrip('/')}"
    else:
        full = ctx.ws.resolve(str(a.get("path") or "index.html"), must_exist=True)
        url = "file://" + ctx.ws.show(full)
    shot = f"/workspace/{SHOTS_DIR}/{ctx.call_id}.png"
    cmd = (f"mkdir -p /workspace/{SHOTS_DIR} && timeout 60 chromium --headless=new --no-sandbox "
           f"--disable-gpu --hide-scrollbars --enable-logging=stderr --v=0 "
           f"--window-size={w},{h} --virtual-time-budget=3000 --screenshot={shot} "
           f"{shlex.quote(url)} 2>&1 | grep -E 'CONSOLE|Uncaught|ERROR' | grep -v -E "
           f"'dbus|Fontconfig|gpu|GPU|sandbox' | head -40")
    ctx.exec(["bash", "-lc", cmd], timeout=90, quiet=True)
    log = ctx.output_text()
    try:
        png = ctx.ws.read(shot)
    except (PathError, OSError):
        raise ToolError("the page did not render: " + (log.strip() or "no screenshot")) from None
    errors = [ln for ln in log.splitlines() if ln.strip()]
    msg = f"rendered {url} at {w}x{h}; screenshot {shot}\n"
    msg += ("console:\n" + "\n".join(errors[:40])) if errors else "no console errors"
    return Result(msg, images=[("image/png", png)],
                  pane={"kind": "image", "ref": f"file:{shot}", "title": "screenshot"})


PANE_KINDS = ("preview", "files", "diff", "markdown", "image", "terminal", "editor")


def _present(ctx, a):
    kind = str(a.get("kind") or "preview")
    if kind not in PANE_KINDS:
        raise ToolError(f"kind must be one of {', '.join(PANE_KINDS)}")
    if a.get("port"):
        ref = f"port:{int(a['port'])}"
    elif kind == "diff":
        ref = f"diff:{ctx.run_id}"
    elif kind == "terminal":
        ref = "shell"
    elif kind == "files":
        ref = ctx.ws.show(ctx.ws.resolve(str(a.get("path") or "/workspace")))
    else:
        full = ctx.ws.resolve(str(a.get("path") or ""), must_exist=True)
        ref = "file:" + ctx.ws.show(full) if kind != "editor" else ctx.ws.show(full)
    pane = {"kind": kind, "ref": ref, "title": str(a.get("title") or "")[:80] or None}
    ctx.present(pane)
    return Result(f"opened a {kind} pane for the user", pane=pane)


def _library_search(ctx, a):
    q = str(a.get("query") or "").strip()
    if not q:
        raise ToolError("query is empty")
    st, obj = ctx.engine.get_json("/v1/library/search?" + urllib.parse.urlencode({"q": q, "k": 5}))
    if st != 200 or not isinstance(obj, list):
        raise ToolError(f"the library did not answer ({st})")
    if not obj:
        return Result("nothing in the library matches")
    return Result("\n".join(f"- {x.get('name')}: {x.get('description')}" for x in obj))


def _library_read(ctx, a):
    name = str(a.get("name") or "").strip()
    st, obj = ctx.engine.get_json("/v1/library/" + urllib.parse.quote(name, safe=""))
    if st != 200 or not isinstance(obj, dict):
        raise ToolError(f"no skill {name!r} in the library")
    ref = obj.get("reference") or ""
    return Result(f"# {obj.get('name')}\n\n{obj.get('body') or ''}" +
                  (f"\n\n## Reference\n\n{ref}" if ref else ""))


def _todo_write(ctx, a):
    items = a.get("items")
    try:
        got = ctx.svc.set_todos(ctx.cid, items)
    except Exception as exc:                                       # noqa: BLE001
        raise ToolError(str(exc)) from None
    done = sum(1 for i in got if i["state"] == "done")
    return Result(f"todo list saved: {len(got)} items, {done} done")


BUILTINS = [
    Tool("bash", "Run a shell command in this chat's own Linux sandbox. The working directory is "
         "/workspace (the chat's files, kept across sessions). Python 3, Node, git, ripgrep, jq, "
         "curl and Chromium are installed; pip and npm reach the internet through a proxy; the "
         "local network is not reachable. Output streams to the user. Timeout 600 s; set "
         "background true for servers and watchers (their output goes to a log file). Start dev "
         "servers on 127.0.0.1 and show them with present(kind='preview', port=N).",
         _obj({"command": _S, "timeout_s": _I, "background": _B}, ["command"]),
         risk="exec", default="ask", sandbox=True, executor=_bash),
    Tool("read_file", "Read a file in /workspace, with line numbers. offset and limit are line "
         "numbers (default: the first 2,000 lines). Images come back as images.",
         _obj({"path": _S, "offset": _I, "limit": _I}, ["path"]), executor=_read),
    Tool("write_file", "Create or replace a file in /workspace with the given content.",
         _obj({"path": _S, "content": _S}, ["path", "content"]), risk="write",
         executor=_write),
    Tool("edit_file", "Replace an exact string in a file in /workspace. old must occur exactly "
         "once unless replace_all is true; copy it from read_file without the line numbers.",
         _obj({"path": _S, "old": _S, "new": _S, "replace_all": _B}, ["path", "old", "new"]),
         risk="write", executor=_edit),
    Tool("glob", "Find files in /workspace by a glob pattern (** crosses directories).",
         _obj({"pattern": _S, "path": _S}, ["pattern"]), executor=_glob),
    Tool("grep", "Search file contents in /workspace with a regular expression; answers "
         "path:line: text.", _obj({"pattern": _S, "path": _S, "include": _S, "ignore_case": _B},
                                  ["pattern"]), executor=_grep),
    Tool("web_fetch", "Fetch a public web page or file (http or https) and return its text. Local "
         "network addresses are refused.", _obj({"url": _S}, ["url"]), risk="network",
         default="ask", executor=_fetch),
    Tool("render_page", "Open an HTML file in /workspace (path) or a port in the sandbox (port) in "
         "headless Chromium; returns a screenshot and the console errors.",
         _obj({"path": _S, "port": _I, "width": _I, "height": _I}, []), sandbox=True,
         executor=_render),
    Tool("present", "Show the user something in a pane beside the chat: a preview of an HTML file "
         "or a port, a file, the diff of this run, a Markdown file, an image, or the terminal.",
         _obj({"kind": {"type": "string", "enum": list(PANE_KINDS)}, "path": _S, "port": _I,
               "title": _S}, ["kind"]), executor=_present),
    Tool("library_search", "Search TandemLLM's skill library for a how-to before unfamiliar "
         "work; answers names and one line each.", _obj({"query": _S}, ["query"]),
         executor=_library_search, needs="library"),
    Tool("library_read", "Read one skill from the library by its name.",
         _obj({"name": _S}, ["name"]), executor=_library_read, needs="library"),
    Tool("todo_write", "Keep the task's todo list, which the user sees beside the chat: send the "
         "whole list each time, every item with a state (pending, in_progress, done). Mark an "
         "item in_progress when you start it and done when it is done.",
         _obj({"items": {"type": "array", "items": _obj({
             "id": _S, "text": _S, "state": {"type": "string",
                                            "enum": ["pending", "in_progress", "done"]}},
             ["text", "state"])}}, ["items"]), executor=_todo_write),
]
BY_NAME = {t.name: t for t in BUILTINS}


# ---------------------------------------------------------------------- permissions
_SUB = re.compile(r"^[a-z][a-z0-9_-]*$")
_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")


def bash_prefix(command: str) -> str:
    """The rule a "always allow" on a bash call suggests: the first word, plus the second when
    it is a subcommand (`npm install`, `git status`), never the whole line."""
    try:
        words = shlex.split(_SPLIT.split(command.strip())[0])
    except ValueError:
        words = command.split()
    if not words:
        return ""
    if len(words) > 1 and _SUB.match(words[1]):
        return f"{words[0]} {words[1]}"
    return words[0]


def _bash_matches(match: str, command: str) -> bool:
    if "$(" in command or "`" in command:
        return False
    want = match.split()
    if not want:
        return False
    for part in _SPLIT.split(command.strip()):
        if not part:
            continue
        try:
            words = shlex.split(part)
        except ValueError:
            return False
        if words[:len(want)] != want:
            return False
    return True


def domain_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def rule_matches(rule: dict, tool: str, args: dict) -> bool:
    if not fnmatch.fnmatchcase(tool, rule.get("tool") or ""):
        return False
    m = rule.get("match")
    if not m:
        return True
    if tool == "bash":
        return _bash_matches(m, str(args.get("command") or ""))
    if tool == "web_fetch":
        d = domain_of(str(args.get("url") or ""))
        return d == m or d.endswith("." + m)
    return False


def suggested_rule(tool: str, args: dict) -> dict:
    if tool == "bash":
        return {"tool": "bash", "match": bash_prefix(str(args.get("command") or ""))}
    if tool == "web_fetch":
        return {"tool": "web_fetch", "match": domain_of(str(args.get("url") or ""))}
    return {"tool": tool}


def _bash_parts(command: str) -> list[dict] | None:
    """A compound command as its parts (each matched on its own); None when it hides a command
    (`$(...)`, backticks), which no rule may allow."""
    if "$(" in command or "`" in command:
        return None
    return [{"command": p} for p in _SPLIT.split(command.strip()) if p.strip()]


def decide(tool: Tool, args: dict, chat_rules: list[dict], global_rules: list[dict],
           settings: dict, global_policy: str | None = None) -> str:
    """allow | ask | deny for one call. The chat's rules come before the global ones; within a
    scope a deny comes before an allow. A compound bash command is allowed only when every part is
    (by this scope's rules or an earlier scope's), and denied when any part is."""
    if tool.name == "bash":
        parts = _bash_parts(str(args.get("command") or ""))
        allows: list[dict] = []
        for rules in (chat_rules, global_rules):
            denies = [r for r in rules if r.get("decision") == "deny"]
            allows += [r for r in rules if r.get("decision") == "allow"]
            for p in parts or [args]:
                if any(rule_matches(r, "bash", p) for r in denies):
                    return "deny"
            if parts and all(any(rule_matches(r, "bash", p) for r in allows) for p in parts):
                return "allow"
    else:
        for rules in (chat_rules, global_rules):
            for want in ("deny", "allow"):
                for r in rules:
                    if r.get("decision") == want and rule_matches(r, tool.name, args):
                        return want
    pol = (settings.get("policies") or {}).get(tool.name) or global_policy
    if pol == "always":
        return "allow"
    if pol == "never":
        return "deny"
    if pol == "ask":
        return "ask"
    if tool.risk == "write" and settings.get("auto_accept_edits"):
        return "allow"
    return tool.default


def summary(tool: str, args: dict) -> str:
    """The prompt's one line: what the call wants to do."""
    if tool == "bash":
        return f"Run: {str(args.get('command') or '')[:300]}"
    if tool == "web_fetch":
        return f"Fetch {str(args.get('url') or '')[:300]}"
    if tool == "write_file":
        n = len(str(args.get("content") or "").encode())
        return f"Write {args.get('path')} ({n:,} bytes)"
    if tool == "edit_file":
        return f"Edit {args.get('path')}"
    if tool.startswith("mcp__"):
        _, server, name = tool.split("__", 2)
        return f"{server}: {name} with {len(args)} argument{'s' if len(args) != 1 else ''}"
    return f"{tool} with {len(args)} argument{'s' if len(args) != 1 else ''}"


# ---------------------------------------------------------------------- results
def for_model(ctx, call_id: str, text: str) -> str:
    """The result the model reads: whole up to ~16,000 tokens; else the ends and the file."""
    if len(text) <= MODEL_LIMIT_CHARS:
        return text
    path = f"{OUTPUTS_DIR}/{call_id}.txt"
    try:
        ctx.ws.write(path, text.encode())
        where = f"/workspace/{path}"
    except (PathError, OSError):
        where = "(could not be saved)"
    cut = len(text) - 2 * MODEL_EDGE_CHARS
    return (text[:MODEL_EDGE_CHARS] + f"\n\n[... {cut:,} characters left out; the whole output is "
            f"in {where} ...]\n\n" + text[-MODEL_EDGE_CHARS:])


def for_ui(text: str) -> tuple[str, bool]:
    if len(text) <= UI_LIMIT_CHARS:
        return text, False
    return text[:UI_LIMIT_CHARS // 2] + "\n[...]\n" + text[-UI_LIMIT_CHARS // 2:], True


def parse_args(raw: str) -> dict:
    if not raw or not raw.strip():
        return {}
    try:
        v = json.loads(raw)
    except ValueError as exc:
        raise ToolError(f"the arguments are not valid JSON ({exc.msg}); send a JSON object") \
            from None
    if not isinstance(v, dict):
        raise ToolError("the arguments must be a JSON object")
    return v


class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
             "article", "pre", "table", "ul", "ol", "header", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.href = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.out.append("#" * int(tag[1]) + " ")
        if tag == "li":
            self.out.append("- ")
        if tag == "a":
            self.href = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag == "a" and self.href and self.href.startswith("http"):
            self.out.append(f" ({self.href})")
            self.href = None

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(doc: str) -> str:
    p = _Text()
    try:
        p.feed(doc)
    except Exception:                                               # noqa: BLE001
        pass
    text = "".join(p.out)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


__all__ = ["Tool", "Result", "ToolError", "BUILTINS", "BY_NAME", "decide", "rule_matches",
           "suggested_rule", "summary", "bash_prefix", "for_model", "for_ui", "parse_args",
           "html_to_text", "PathError", "ChangedError"]
