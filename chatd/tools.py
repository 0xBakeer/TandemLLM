"""The tools a chat can offer the model, their executors, and the permission rules (SRV-353).

THE REGISTRY. Each tool: name, description, JSON schema, source (builtin, mcp:<server>, user,
skill), risk (read, write, exec, network), the default permission (allow, ask, deny), whether it
needs the chat's sandbox, and its executor. Tools go to the model in one fixed order (the
built-ins as listed, then MCP servers by name, then user tools), so the prompt head stays
byte-stable and the prefix cache holds. MCP tools reach the model as `mcp__<server>__<tool>`.

THE EXECUTORS run on the host where nothing executes (read_file, write_file, edit_file, glob,
grep: chatd/workspace.py, symlink-safe), in the sandbox where something does (bash, render_page),
and through the egress filter where the network is touched (web_fetch: the same private-range
refusal as the sandbox's proxy, chatd/egress.py; web_search: the configured provider's host only,
chatd/websearch.py). A result for the model is at most 16,000 tokens:
a longer one is written to /workspace/.tandem/outputs/<call>.txt and the model gets its first and
last 4,000 tokens and the path. The part the UI keeps is cut at 30,000 characters (CONTRACT 5).

PERMISSIONS (Claude Code's model; the chat's mode on top: `decide_mode`). A call is decided by: the chat's rules (deny, then allow), the
global rules (deny, then allow), the chat's policy for the tool (always, never, ask), "auto-accept
edits" for write tools, and the tool's default. A rule names a tool and, for some, what it may do:
`bash` a command prefix by whole words ("npm install" matches `npm install three`, never `npm
installx`; a command with `$(...)` or backticks matches no rule, and a compound command matches
only when every part does), `web_fetch` a domain and its subdomains, an MCP tool a glob on its
name (`mcp__x1__*`).
"""

from __future__ import annotations

import fnmatch
import hashlib
import html.parser
import json
import os
import re
import shlex
import stat
import time
import urllib.parse

from chatd import render, sandbox as sandbox_mod
from chatd.workspace import ChangedError, PathError

MODEL_LIMIT_CHARS = 64_000          # ~16,000 tokens
MODEL_EDGE_CHARS = 16_000           # ~4,000 tokens kept from each end
UI_LIMIT_CHARS = 30_000
BASH_TIMEOUT_S = 600
OUTPUTS_DIR = ".tandem/outputs"
SHOTS_DIR = ".tandem/shots"


class Tool:
    def __init__(self, name, description, schema, *, risk="read", default="allow", sandbox=False,
                 source=None, executor=None, needs=None, side_effects=None):
        self.name = name
        self.description = description
        self.schema = schema
        self.risk = risk
        self.default = default
        self.sandbox = sandbox
        self.source = source or {"kind": "builtin"}
        self.executor = executor
        self.needs = needs              # a capability the box must have ("library")
        # whether a call reaches past the chat's sandbox (an MCP tool not marked read-only, an
        # Open WebUI toolkit that imports a network or process module); None: a built-in
        self.side_effects = side_effects

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name,
                                                 "description": self.description,
                                                 "parameters": self.schema}}


class Result:
    def __init__(self, content: str = "", *, is_error=False, exit_code=None, images=None,
                 pane=None, data=None):
        self.content = content
        self.is_error = is_error
        self.exit_code = exit_code
        self.images = images or []      # [(mime, bytes)]
        self.pane = pane
        self.data = data                # structured output for the UI (`output.data`)


class ToolError(Exception):
    """A call that cannot run as asked: its message is the tool result the model reads; `data`,
    when set, is the call's `output.data` for the chat."""

    def __init__(self, message: str, data: dict | None = None):
        super().__init__(message)
        self.data = data


def oom_words() -> str:
    return f"out of memory ({sandbox_mod.limit_words()} sandbox limit)"


def oom_data() -> dict:
    return {"sandbox": {"reason": "oom", "limit_gib": sandbox_mod.LIMIT_BYTES / sandbox_mod.GiB}}


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
    if getattr(ctx, "oom", False):
        return Result(out + f"\n[{oom_words()}: the kernel killed a process of this command "
                            f"(exit code {code}). Use less memory: fewer parallel jobs or threads "
                            f"(the sandbox has 4 CPUs while nproc counts the whole box, so pass "
                            f"-j4, -threads 4 and the like), smaller batches, files on disk "
                            f"instead of data in memory]",
                      is_error=True, exit_code=code, data=oom_data())
    tail = f"\n[exit code {code}]" if code else ""
    return Result(out + tail, is_error=bool(code), exit_code=code)


def _read(ctx, a):
    path = str(a.get("path") or "")
    st = ctx.ws.stat(path)
    if stat.S_ISDIR(st.st_mode):
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
    if render.WROTE_STUB.match(content.strip()):
        # an earlier write's content reads as this stub in the request (chatd/render.py)
        raise ToolError("content is the stub of an earlier write, not the file's text; send the "
                        "whole text, or use edit_file for a change")
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
        # the request shows an earlier write's or read's text as a stub (chatd/render.py)
        raise ToolError("old was not found in the file; read it again with read_file and copy it "
                        "exactly (an earlier write or read of it may show only a stub)")
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


def _search(ctx, a):
    from chatd.websearch import SearchError, as_text
    try:
        res = ctx.search(a.get("query"), a.get("max_results"))
    except SearchError as exc:
        raise ToolError(str(exc)) from None
    return Result(as_text(res), data={"provider": res["provider"], "query": res["query"],
                                      "results": res["results"]})


def _ports(ctx) -> list[dict] | None:
    """What listens in the sandbox now (relay.py ports); None when that could not be read."""
    mark = len(ctx.output_text())
    ctx.exec(sandbox_mod.relay_cmd("ports"), timeout=20, quiet=True)
    lines = ctx.output_text()[mark:].strip().splitlines()
    try:
        got = json.loads(lines[-1]) if lines else None
    except ValueError:
        return None
    return got if isinstance(got, list) else None


def _ports_words(ports: list[dict] | None) -> str:
    if ports is None:
        return "what listens in the sandbox could not be read"
    if not ports:
        return "nothing listens on any port in the sandbox"
    return "listening now: " + ", ".join(
        f"{p['port']} on {' and '.join(p.get('addresses') or [])}" +
        (f" ({p['command']})" if p.get("command") else "") for p in ports)


# what Chromium writes beside the page's console: its own pages, the WebGL fallback, its GL driver
STATUS_MARK = "HTTP_STATUS "
CONSOLE_NOISE = "source: chrome://|Automatic fallback to software WebGL|GL Driver Message"
_CONSOLE_HEAD = re.compile(r"^\[[\d:./]+:[A-Z]+:CONSOLE(?:\((\d+)\)|:(\d+))?\]\s*")


def console_line(line: str) -> str:
    """`[86:86:1016/215118.672506:INFO:CONSOLE:422] "msg", source: url (422)` -> `"msg",
    source: url (422)`: the page's message, without Chromium's process, thread and clock."""
    return _CONSOLE_HEAD.sub("", line.strip())


def _render(ctx, a):
    w = min(2400, max(320, int(a.get("width") or 1280)))
    h = min(2400, max(240, int(a.get("height") or 800)))
    if a.get("port"):
        # localhost, as curl in the sandbox dials it (::1, then 127.0.0.1): live (2026-10-16)
        # Vite 5 listened on [::1]:5173 only and Chromium sent to 127.0.0.1 was refused
        port = int(a["port"])
        ports = _ports(ctx)
        if ports is not None and port not in {p.get("port") for p in ports}:
            raise ToolError(f"nothing listens on port {port} in the sandbox, on any address; "
                            f"{_ports_words(ports)}")
        path = str(a.get("path") or "")
        row = next((p for p in ports or [] if p.get("port") == port), None)
        cwd = ((row or {}).get("cwd") or "").rstrip("/")
        if cwd and path.startswith(cwd + "/"):
            # a file's path where the URL's goes: the server serves its working directory (live
            # 2026-10-17, /workspace/_test.html on a server from /workspace was its 404 page)
            path = path[len(cwd) + 1:]
        url = f"http://localhost:{port}/{path.lstrip('/')}"
    else:
        full = ctx.ws.resolve(str(a.get("path") or "index.html"), must_exist=True)
        url = "file://" + ctx.ws.show(full)
        cwd = ""
    kit = _render_kit(ctx, url, w, a.get("port") and int(a["port"]), cwd)
    if kit is not None:
        return kit
    shot = f"/workspace/{SHOTS_DIR}/{ctx.call_id}.png"
    # WebGL through SwiftShader by name (the automatic fallback is deprecated and says so on
    # every page); the page's console only: Chromium's own pages (chrome://) and its services
    # (GCM registration, `ERROR:...cc` lines) are not the page's
    # a server's answer is said when it is no page (404, 500): the screenshot is its error page
    status = (f"echo \"{STATUS_MARK}$(curl -s -o /dev/null -m 10 -w '%{{http_code}}' "
              f"{shlex.quote(url)})\"; " if a.get("port") else "")
    # memory: /dev/shm is Docker's 64 MB (the shared memory goes to /tmp instead), one renderer
    cmd = (f"mkdir -p /workspace/{SHOTS_DIR} && {status}timeout 60 chromium --headless=new "
           f"--no-sandbox --disable-gpu --use-angle=swiftshader --enable-unsafe-swiftshader "
           f"--disable-dev-shm-usage --renderer-process-limit=1 --no-first-run "
           f"--disable-background-networking --hide-scrollbars --enable-logging=stderr --v=0 "
           f"--window-size={w},{h} --virtual-time-budget=3000 --screenshot={shot} "
           f"{shlex.quote(url)} 2>&1 | grep -E 'CONSOLE|Uncaught' | grep -v -E "
           f"'{CONSOLE_NOISE}' | head -40")
    mark = len(ctx.output_text())
    ctx.exec(["bash", "-lc", cmd], timeout=90, quiet=True)
    out = ctx.output_text()[mark:].splitlines()
    code = next((ln[len(STATUS_MARK):].strip() for ln in out if ln.startswith(STATUS_MARK)), "")
    log = "\n".join(console_line(ln) for ln in out if not ln.startswith(STATUS_MARK))
    try:
        png = ctx.ws.read(shot)
    except (PathError, OSError):
        if getattr(ctx, "oom", False):
            raise ToolError(f"the page did not render: {oom_words()}, the kernel killed Chromium. "
                            f"A lighter page or a smaller width and height may fit",
                            oom_data()) from None
        raise ToolError("the page did not render: " + (log.strip() or "no screenshot") +
                        (f" ({url}; {_ports_words(_ports(ctx))})" if a.get("port") else "")
                        ) from None
    errors = quote_lines(ctx, [ln for ln in log.splitlines() if ln.strip()],
                         a.get("port") and int(a["port"]), cwd)
    # a new render of the page as it is now, each call (its own file): live (2026-10-13) the
    # model read a correct clock as "still tilted" four times and took the screenshots for stale
    msg = (f"rendered {url} at {w}x{h} just now (a fresh render of the current file, "
           f"{time.strftime('%H:%M:%S')}, image sha256 {hashlib.sha256(png).hexdigest()[:12]}); "
           f"screenshot {shot}\n")
    if code.isdigit() and not 200 <= int(code) < 400:
        msg += (f"the server answered HTTP {code} for {url}: the screenshot is its error page, not "
                f"the app\n")
    msg += ("console:\n" + "\n".join(errors[:40])) if errors else "no console errors"
    return Result(msg, images=[("image/png", png)],
                  pane={"kind": "image", "ref": f"file:{shot}", "title": "screenshot"})


# a script and line in an error: `http://localhost:5174/js/scene/pond.js:62` (the kit) or
# `source: http://localhost:5174/js/scene/pond.js (62)` (Chromium's log)
_AT_LINE = re.compile(r"(?:https?://(?:localhost|127\.0\.0\.1):(\d+)|file://)(/[^\s():\"']+?)"
                      r"(?::(\d+)(?::\d+)?|\s\((\d+)\))")


def quote_lines(ctx, errors: list[str], port: int | None, cwd: str) -> list[str]:
    """Each error that names a script of this page and a line, followed by that line of the file
    in the chat's files: live (2026-10-17, pagoda replay 3) "Cannot read properties of undefined
    (reading 'set')" at pond.js (62) went to the model 13 times, and it edited main.js for 20
    minutes; line 62 was `petal.position.set(...)` on a geometry."""
    out = []
    for e in errors:
        out.append(e)
        m = _AT_LINE.search(e)
        if not m:
            continue
        p, path, line = m.group(1), m.group(2), m.group(3) or m.group(4)
        if p is not None:
            if port is None or int(p) != port or not cwd:
                continue
            path = cwd.rstrip("/") + path
        try:
            src = ctx.ws.read(path, limit=512 * 1024).decode("utf-8", "replace").splitlines()
            text = src[int(line) - 1].strip()
        except (PathError, OSError, IndexError, ValueError):
            continue
        out.append(f"    {ctx.ws.show(ctx.ws.resolve(path))}:{line}: {text[:200]}")
    return out


def kit_ok(ctx) -> bool:
    """The laya kit in the chat's files (chatd/verify.py `vendor_kit`); False without the source
    tree (an installed package): render_page then shoots with Chromium's own flags."""
    from chatd import verify
    return verify.vendor_kit(ctx.ws)


def _render_kit(ctx, url: str, w: int, port: int | None = None, cwd: str = "") -> Result | None:
    """render_page through the laya kit's page check (Chrome under DevTools, real time): the
    page's console errors, uncaught exceptions and failed loads, after the page has loaded. Live
    (2026-10-17, pagoda replay 3) a module the page imports answered 404, the app stayed on its
    loading screen, and Chromium's own screenshot mode logged nothing: render_page said "no
    console errors" six times over the same picture. None when the kit is not there or its run
    gives no report (the caller shoots with Chromium's own flags)."""
    if not kit_ok(ctx):
        return None
    from chatd import verify
    shots = f"/workspace/{SHOTS_DIR}/{ctx.call_id}"
    cmd = (f"cd /workspace && python3 {verify.KIT_IN_BOX} page {shlex.quote(url)} --no-laya "
           f"--widths {w} --shots {shots} --timeout 60")
    mark = len(ctx.output_text())
    ctx.exec(["bash", "-lc", cmd], timeout=90, quiet=True)
    out = ctx.output_text()[mark:]
    try:
        rep = json.loads(out[out.index("{"):out.rindex("}") + 1])
        png = ctx.ws.read(f"{shots}/page-{w}.png")
    except (ValueError, PathError, OSError):
        if getattr(ctx, "oom", False):
            raise ToolError(f"the page did not render: {oom_words()}, the kernel killed Chromium. "
                            f"A lighter page or a smaller width and height may fit",
                            oom_data()) from None
        return None
    h = 844 if w < 600 else 900
    shot = f"{shots}/page-{w}.png"
    msg = (f"rendered {url} at {w}x{h} just now (a fresh render of the current file, "
           f"{time.strftime('%H:%M:%S')}, image sha256 {hashlib.sha256(png).hexdigest()[:12]}); "
           f"screenshot {shot}\n")
    errors = quote_lines(ctx, rep.get("errors") or [], port, cwd)
    text = (((rep.get("widths") or [{}])[0]).get("text") or "").strip()
    if text:
        msg += f'visible text: "{text[:200]}"\n'
    msg += ("console errors, exceptions and failed loads:\n" + "\n".join(errors[:40])) \
        if errors else "no console errors and no failed loads"
    return Result(msg, images=[("image/png", png)],
                  pane={"kind": "image", "ref": f"file:{shot}", "title": "screenshot"})


PANE_KINDS = ("preview", "files", "diff", "markdown", "image", "terminal", "editor", "video",
              "audio", "pdf")
# a file's own kind wins over these: a video presented as a preview plays in the player (VIS-159)
_BY_FILE = ("preview", "markdown", "image", "video", "audio", "pdf")


def _present(ctx, a):
    from chatd.panes import media_kind, mime_of
    kind = str(a.get("kind") or "preview")
    if kind not in PANE_KINDS:
        raise ToolError(f"kind must be one of {', '.join(PANE_KINDS)}")
    if kind in _BY_FILE and not a.get("port") and a.get("path"):
        kind = media_kind(mime_of(str(a["path"]))) or kind
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
    if kind == "preview":
        return _present_preview(ctx, pane)
    ctx.present(pane)
    return Result(f"opened a {kind} pane for the user", pane=pane)


def _present_preview(ctx, pane):
    """A preview's real outcome for the model: what the pane targets (the port that listens, a
    page's dev server, or nothing yet), what the server answered, and what the user's pane loaded
    (its own report: errors, failed files), so the model cannot claim a blank pane works. Live
    (2026-10-16) `present` said done for port 4173 while the user's pane was white."""
    from chatd.preview import REPORT_WAIT_S, probe_words, report_words, server_words
    from chatd.service import ApiError
    t0 = time.time()
    try:
        art = ctx.svc.artifact(ctx.cid, pane["ref"])
    except ApiError as exc:
        raise ToolError(str(exc)) from None
    ctx.present(pane)
    w = art.get("waiting")
    if w:
        return Result(f"opened a preview pane, but it shows nothing yet: {w['message']} "
                      f"{_ports_words(ctx.svc.listening(ctx.cid))}. Start the server (bash with "
                      f"background true), then present it again. The preview does not work yet: "
                      f"do not tell the user it does.", is_error=True, pane=pane)
    lines, bad = [], False
    if art.get("port"):
        lines.append("opened the preview pane on " + server_words(
            {"port": art["port"], "command": art.get("server")}) + ".")
        if art.get("note"):
            lines.append(art["note"])
        words, b = probe_words(ctx.svc.probe(ctx.cid, art["port"]))
        lines.append(words)
        bad = bad or b
    else:
        lines.append(f"opened the preview pane on {pane['ref'][5:]}.")
    viewers = len(ctx.svc.conv(ctx.cid).fan)
    rep = ctx.svc.wait_report(ctx.cid, pane["ref"], t0, REPORT_WAIT_S if viewers else 0)
    words, b = report_words(rep, viewers)
    lines.append(words)
    bad = bad or b
    if bad:
        lines.append("The preview does not work yet: fix this before telling the user it does.")
    return Result("\n".join(lines), is_error=bad, pane=pane)


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
    from chatd.verify import kit_note
    text = f"# {obj.get('name')}\n\n{obj.get('body') or ''}" + (f"\n\n## Reference\n\n{ref}"
                                                                if ref else "")
    return Result(text + kit_note(text, ctx.ws))


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
    Tool("web_search", "Search the web; answers a numbered list of results (title, URL, snippet, "
         "date when known). Cite the URLs you use, and read a page in full with web_fetch. "
         "max_results: 1 to 20, default from the chat settings.",
         _obj({"query": _S, "max_results": _I}, ["query"]), risk="network", executor=_search,
         needs="web_search"),
    Tool("web_fetch", "Fetch a public web page or file (http or https) and return its text. Local "
         "network addresses are refused.", _obj({"url": _S}, ["url"]), risk="network",
         default="ask", executor=_fetch),
    Tool("render_page", "Open an HTML file in /workspace (path) or a port in the sandbox (port) in "
         "headless Chromium; returns a screenshot and the console errors.",
         _obj({"path": _S, "port": _I, "width": _I, "height": _I}, []), sandbox=True,
         executor=_render),
    Tool("present", "Show the user something in a pane beside the chat: a preview of an HTML file "
         "or a port, a file, the diff of this run, a Markdown file, an image, a video, an audio "
         "file, a PDF, or the terminal.",
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
# the programs whose second word is a subcommand worth its own rule (`git push` is not `git status`);
# for any other program the second word is an argument (`grep tilt` would allow only that search)
_SUBCOMMANDS = {"git", "npm", "pnpm", "yarn", "npx", "pip", "pip3", "uv", "poetry", "cargo", "go",
                "docker", "podman", "kubectl", "apt", "apt-get", "brew", "make", "dotnet", "gh"}
_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")


def bash_prefix(command: str) -> str:
    """The rule a "always allow" suggests for one command: the first word, plus the second when
    it is a subcommand of a program that has them (`npm install`, `git status`; _SUBCOMMANDS),
    never the whole line. Of a chained command,
    its first part (bash_prefixes has them all)."""
    try:
        words = shlex.split(_SPLIT.split(command.strip())[0])
    except ValueError:
        words = command.split()
    if not words:
        return ""
    if len(words) > 1 and words[0] in _SUBCOMMANDS and _SUB.match(words[1]):
        return f"{words[0]} {words[1]}"
    return words[0]


def bash_prefixes(command: str, allows: list[dict] = ()) -> list[str]:
    """The prefixes "always allow" saves for a chained command: one per part that no allow rule in
    `allows` covers yet, each once, in order (`cd /w && node a.js > run.log 2>&1; echo $?; cat
    run.log` with cd and node allowed: echo, cat). The live bug of 2026-10-13: only the first
    part's prefix was offered, so the rule never covered the command and the prompt came back."""
    out: list[str] = []
    for p in _bash_parts(command) or []:
        if any(rule_matches(r, "bash", p) for r in allows):
            continue
        pre = bash_prefix(p["command"])
        if pre and pre not in out:
            out.append(pre)
    return out


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


def suggested_rule(tool: str, args: dict, allows: list[dict] = ()) -> dict:
    """What "always allow" saves. For bash, `matches`: a prefix per part not yet covered by
    `allows` (the chat's and the global allow rules), and `match` the same joined for the prompt
    ("cd, node, echo, cat"); a command that hides a command (`$(...)`, backticks) gets none, since
    no rule may allow it."""
    if tool == "bash":
        cmd = str(args.get("command") or "")
        pres = bash_prefixes(cmd, allows) if _bash_parts(cmd) is not None else []
        return {"tool": "bash", "match": ", ".join(pres), "matches": pres}
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


# ---------------------------------------------------------------------- permission modes
# SRV-387/SRV-391: a mode per chat on top of the rules. ask: as above. auto (the default): what runs
# in the chat's own sandbox runs without a prompt (bash, the workspace's files, render_page); what
# leaves it still asks (web_fetch to a domain this chat has not fetched, an MCP tool not marked
# read-only, an Open WebUI toolkit with side effects), and so does rm aimed at /, /workspace or the
# home directory. yolo: nothing asks. In every mode a deny rule (or the policy never) refuses, an
# explicit ask policy asks, and plan mode keeps its read-only tool list (chatd/agent.py tools_for).
MODES = ("ask", "auto", "yolo")
DEFAULT_MODE = "auto"
_RM_TARGET = re.compile(r"^(?:/|/workspace|~|\$HOME|\$\{HOME\}|/root|/home(?:/[^/]+)?)(?:/\*?)?$")


def risky_rm(command: str) -> bool:
    """rm or rmdir aimed at /, /workspace or a home directory (or a command that hides what it
    runs and mentions rm)."""
    if "$(" in command or "`" in command:
        return bool(re.search(r"\brm(?:dir)?\b", command))
    for part in _SPLIT.split(command.strip()):
        try:
            words = shlex.split(part)
        except ValueError:
            return bool(re.search(r"\brm(?:dir)?\b", part))
        while words and words[0] in ("sudo", "command", "exec", "nice", "nohup", "time"):
            words = words[1:]
        if words and os.path.basename(words[0]) in ("rm", "rmdir") and \
                any(_RM_TARGET.match(w) for w in words[1:] if not w.startswith("-")):
            return True
    return False


def leaves_sandbox(tool: Tool, args: dict, fetched: set | frozenset = frozenset()) -> bool:
    """A call auto mode still asks for: it reaches past the chat's sandbox."""
    if tool.name == "web_fetch":
        return domain_of(str(args.get("url") or "")) not in fetched
    if tool.side_effects is not None:
        return bool(tool.side_effects)
    return tool.risk == "network"


def decide_mode(tool: Tool, args: dict, chat_rules: list[dict], global_rules: list[dict],
                settings: dict, global_policy: str | None = None, mode: str = "ask",
                fetched: set | frozenset = frozenset()) -> tuple[str, str | None]:
    """(allow | ask | deny, the mode that allowed it or None). `decide` first: its deny, allow and
    ask stand except an ask that came from the tool's default, which the mode may turn into an
    allow. `fetched`: the domains this chat's web_fetch has already reached."""
    d = decide(tool, args, chat_rules, global_rules, settings, global_policy)
    if d != "ask" or mode not in ("auto", "yolo"):
        return d, None
    if (settings.get("policies") or {}).get(tool.name) == "ask" or global_policy == "ask":
        return "ask", None
    if mode == "yolo":
        return "allow", "yolo"
    if leaves_sandbox(tool, args, fetched):
        return "ask", None
    if tool.name == "bash" and risky_rm(str(args.get("command") or "")):
        return "ask", None
    return "allow", "auto"


def summary(tool: str, args: dict) -> str:
    """The prompt's one line: what the call wants to do."""
    if tool == "bash":
        return f"Run: {str(args.get('command') or '')[:300]}"
    if tool == "web_fetch":
        return f"Fetch {str(args.get('url') or '')[:300]}"
    if tool == "web_search":
        return f"Search the web for \"{str(args.get('query') or '')[:300]}\""
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


__all__ = ["Tool", "Result", "ToolError", "BUILTINS", "BY_NAME", "decide", "decide_mode", "MODES",
           "DEFAULT_MODE", "risky_rm", "leaves_sandbox", "rule_matches",
           "suggested_rule", "summary", "bash_prefix", "bash_prefixes", "for_model", "for_ui", "parse_args",
           "html_to_text", "PathError", "ChangedError"]
