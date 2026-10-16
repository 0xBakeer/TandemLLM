"""The preview's target and what loads in it (SRV-511, chatd/preview.py): the port proxy's CORS and
shim, a page's own directory on the view route, the target a `port:N` or a bundler project's
index.html resolves to, the dev server the pane starts, and what `present` tells the model."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time

from chatd_util import Chat, run_all
from test_chatd_agent import RelayBoxes

from chatd import preview as PV, sandbox as sb_mod, tools as T
from chatd.workspace import Workspace

_spec = importlib.util.spec_from_file_location("relay", sb_mod.RELAY)
relay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay)


class PortBoxes(RelayBoxes):
    """RelayBoxes, and `relay.py ports` answers `self.listen`, `relay.py probe` runs on the host."""

    def __init__(self, data):
        super().__init__(data)
        self.listen: list[dict] = []

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        if cmd[:2] == ["python3", "-c"] and len(cmd) > 3 and cmd[3] in ("ports", "probe"):
            if cmd[3] == "ports":
                rows = [dict(r) for r in self.listen]
                for r in rows:
                    r.setdefault("http", True)
                out = json.dumps(rows).encode()
            else:
                out = json.dumps(relay.probe(int(cmd[4]), cmd[5] if len(cmd) > 5 else "/")).encode()
            a, b = socket.socketpair()
            b.sendall(bytes([1, 0, 0, 0]) + len(out).to_bytes(4, "big") + out)
            b.close()
            return "exec-ports", a, b""
        return super().open_exec(cid, cmd, tty=tty, stdin=stdin, env=env, workdir=workdir)


def chat():
    c = Chat()
    c.svc.sandboxes = PortBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    return c


def server(pages: dict):
    """A dev server: path -> (status, type, body, extra headers); records each request's headers."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            st, ctype, body, extra = pages.get(self.path.split("?")[0],
                                               (404, "text/plain", b"no", {}))
            self.send_response(st)
            self.send_header("Content-Type", ctype)
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


VITE_INDEX = (b'<!DOCTYPE html><html><head>\n<script type="module" src="/@vite/client"></script>'
              b'<link rel="stylesheet" href="style.css"></head><body><canvas></canvas>'
              b'<script type="module" src="/src/main.js"></script></body></html>')
VITE_CLIENT = (b'const base$1 = "/" || "/";\nconst socketHost = `${null || importMetaUrl.hostname}'
               b':${hmrPort || importMetaUrl.port}${"/"}`;\nconst base = "/" || "/";\n'
               b'import "/node_modules/vite/dist/client/env.mjs";')


# ---------------------------------------------------------------------- the port proxy
def test_the_port_proxy_lets_the_opaque_preview_read_its_modules_and_adds_the_shim():
    """Live (2026-10-16, WebKit via tandem.khaled.de): "Origin null is not allowed by
    Access-Control-Allow-Origin" for /src/main.js and /@vite/client (a dark pane) and for
    dist/assets/*.css and *.js under `vite preview` (a white one). Each answer now allows the
    opaque origin; the server sees a local client; a page gets the shim before Vite's client; the
    client's base is the port's prefix (hot updates import `base + path`)."""
    c = chat()
    app, seen = server({
        "/": (200, "text/html", VITE_INDEX, {"Access-Control-Allow-Origin": "http://localhost",
                                              "X-Frame-Options": "DENY"}),
        "/@vite/client": (200, "text/javascript", VITE_CLIENT, {}),
        "/style.css": (200, "text/css", b"body{background:url(/bg.png)}", {}),
    })
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        base = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        h_null = {"Origin": "null", "Referer": "https://tandem.khaled.de/x",
                  "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "script"}
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/", headers=h_null)
        assert st == 200 and h["Access-Control-Allow-Origin"] == "null" and "X-Frame-Options" not in h
        page = body if isinstance(body, bytes) else body.encode()
        head = page.index(b"<head>") + len(b"<head>")
        assert page[head:head + 8] == b"<script>", "the shim comes first in the head"
        assert page.index(b"tandemPreview") < page.index(f'src="{base}/@vite/client"'.encode())
        assert f'"{base}"'.encode() in page[:page.index(b"</script>")] and b",L=%d," % port in page
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/@vite/client",
                             headers=h_null)
        js = body if isinstance(body, bytes) else body.encode()
        assert h["Access-Control-Allow-Origin"] == "null"
        assert f'const base$1 = "{base}/" || "/";'.encode() in js
        assert f'const base = "{base}/" || "/";'.encode() in js
        assert f'import "{base}/node_modules/vite/dist/client/env.mjs"'.encode() in js
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/style.css")
        assert "Access-Control-Allow-Origin" not in h, "only the opaque origin is allowed"
        for path, hdrs in seen:
            low = {k.lower() for k in hdrs}
            assert not low & {"origin", "referer", "sec-fetch-site", "sec-fetch-mode"}, (path, low)
        assert any("sec-fetch-dest" in {k.lower() for k in hd} for _, hd in seen)
    finally:
        app.shutdown()
        c.close()


def test_the_view_route_serves_a_built_page_from_its_own_directory():
    """dist/index.html of `vite build` asks for /assets/index-x.js: on the view route that is the
    page's own directory, and its module may be read by the opaque preview."""
    c = chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("app/dist/index.html", b'<html><head><script type="module" crossorigin '
                                        b'src="/assets/index-x.js"></script><link rel="stylesheet" '
                                        b'crossorigin href="/assets/a.css"></head></html>')
        ws.write("app/dist/assets/index-x.js", b"document.body.dataset.ok = 1")
        view = f"/v1/chat-app/conversations/{cid}/sandbox/view/app/dist"
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/view/app/dist/index.html",
                             headers={"Origin": "null"})
        page = body if isinstance(body, bytes) else body.encode()
        assert st == 200 and h["Access-Control-Allow-Origin"] == "null"
        assert f'src="{view}/assets/index-x.js"'.encode() in page
        assert f'href="{view}/assets/a.css"'.encode() in page
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/view/app/dist/assets/index-x.js",
                             headers={"Origin": "null"})
        assert st == 200 and h["Access-Control-Allow-Origin"] == "null"
        # the artifact of a built page is the page itself: it runs as a file
        a = c.ok("GET", f"/conversations/{cid}/artifacts/file%3A%2Fworkspace%2Fapp%2Fdist%2Findex.html")
        assert a["url"].endswith("/sandbox/view/app/dist/index.html") and "waiting" not in a
    finally:
        c.close()


# ---------------------------------------------------------------------- the target
def ws_with_vite_project() -> Workspace:
    root = tempfile.mkdtemp()
    ws = Workspace(root)
    ws.ensure()
    ws.write("pagoda-garden/package.json", json.dumps({
        "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
        "devDependencies": {"vite": "^5.4.0", "three": "^0.170.0"}}).encode())
    ws.write("pagoda-garden/index.html", b'<html><head><link rel="stylesheet" href="style.css">'
                                         b'</head><body><canvas class="webgl"></canvas>'
                                         b'<script type="module" src="/src/main.js"></script>'
                                         b'</body></html>')
    ws.write("pagoda-garden/src/main.js", b"import * as THREE from 'three';\n"
                                          b"import { CONFIG } from './config.js';\n")
    ws.write("pagoda-garden/dist/index.html", b'<script type="module" crossorigin '
                                              b'src="/assets/index-B.js"></script>')
    ws.write("pagoda-garden/dist/assets/index-B.js", b'const a=1;export{a};')
    ws.write("plain/index.html", b'<script type="module" src="app.js"></script>')
    ws.write("plain/app.js", b'import { x } from "./x.js";')
    ws.write("mapped/index.html", b'<script type="importmap">{"imports":{"three":"https://c/t.js"}}'
                                  b'</script><script type="module">import * as T from "three"'
                                  b'</script>')
    ws.write("ts/index.html", b'<script type="module" src="/src/main.ts"></script>')
    return ws


def test_a_bundler_projects_source_page_needs_its_dev_server_and_a_built_or_plain_page_does_not():
    ws = ws_with_vite_project()
    try:
        need = PV.needs_server(ws, "/workspace/pagoda-garden/index.html")
        assert need == {"reason": 'its script /src/main.js imports the package "three" by name',
                        "dir": "/workspace/pagoda-garden", "command": "npm run dev",
                        "bundler": "vite"}
        assert PV.needs_server(ws, "/workspace/pagoda-garden/dist/index.html") is None
        assert PV.needs_server(ws, "/workspace/plain/index.html") is None
        assert PV.needs_server(ws, "/workspace/mapped/index.html") is None, "an import map resolves it"
        ts = PV.needs_server(ws, "/workspace/ts/index.html")
        assert ts["reason"] == "its script /src/main.ts needs compiling" and ts["command"] is None
        assert PV.project_of(ws, "/workspace/pagoda-garden/src/main.js")["dir"] == \
            "/workspace/pagoda-garden"
        assert PV.project_of(ws, "/workspace/plain/index.html") is None
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_the_preview_picks_the_port_that_listens_or_the_projects_newest_server():
    vite = {"port": 5173, "cwd": "/workspace/pagoda-garden", "started": 200, "command": "vite"}
    other = {"port": 8000, "cwd": "/workspace/api", "started": 300, "command": "python3 -m http.server"}
    db = {"port": 5432, "cwd": "/", "started": 400, "http": False}
    assert PV.pick([vite, other], want=5173) is vite
    assert PV.pick([vite, other, db], want=4173) is other, "the newest server that answers HTTP"
    assert PV.pick([vite, other], want=4173, project="/workspace/pagoda-garden") is vite
    assert PV.pick([other], project="/workspace/pagoda-garden") is None
    assert PV.pick([], want=4173) is None and PV.pick(None) is None


def test_a_port_or_page_artifact_resolves_to_what_listens_or_waits():
    c = chat()
    try:
        cid = c.new()
        boxes = c.svc.sandboxes
        ws = c.svc.workspace(cid)
        src = ws_with_vite_project()
        for rel in ("pagoda-garden/package.json", "pagoda-garden/index.html",
                    "pagoda-garden/src/main.js"):
            ws.write(rel, src.read(rel))
        shutil.rmtree(src.root, ignore_errors=True)
        page = "file%3A%2Fworkspace%2Fpagoda-garden%2Findex.html"
        # the sandbox stopped: the pane waits and names the command
        a = c.ok("GET", f"/conversations/{cid}/artifacts/port%3A4173")
        assert "url" not in a and "the sandbox is stopped" in a["waiting"]["message"]
        a = c.ok("GET", f"/conversations/{cid}/artifacts/{page}")
        assert a["waiting"]["command"] == "npm run dev" and \
            a["waiting"]["dir"] == "/workspace/pagoda-garden" and \
            "the source of a Vite project" in a["waiting"]["message"]
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        boxes.listen = [{"port": 5173, "addresses": ["::1"], "cwd": "/workspace/pagoda-garden",
                         "started": 9, "command": "node /workspace/pagoda-garden/node_modules/.bin/vite"}]
        a = c.ok("GET", f"/conversations/{cid}/artifacts/port%3A4173")
        assert a["url"].endswith("/sandbox/ports/5173/") and a["port"] == 5173
        assert a["note"].startswith("Nothing listens on port 4173; this is port 5173 (node ")
        a = c.ok("GET", f"/conversations/{cid}/artifacts/port%3A5173")
        assert a["url"].endswith("/sandbox/ports/5173/") and "note" not in a
        a = c.ok("GET", f"/conversations/{cid}/artifacts/{page}")
        assert a["url"].endswith("/sandbox/ports/5173/") and "it runs from its dev server" in a["note"]
        got = c.ok("GET", f"/conversations/{cid}/sandbox/ports")
        assert [p["port"] for p in got["ports"]] == [5173] and got["ports"][0]["http"] is True
    finally:
        c.close()


def test_the_pane_starts_the_projects_dev_server_from_its_package_json():
    c = chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("game/package.json", json.dumps({"scripts": {"dev": "vite --host"},
                                                  "devDependencies": {"vite": "5"}}).encode())
        ws.write("loose/a.txt", b"x")
        got = c.ok("POST", f"/conversations/{cid}/sandbox/serve", {"dir": "/workspace/game"})
        assert got == {"dir": "/workspace/game", "command": "npm run dev",
                       "log": "/workspace/.tandem/outputs/dev-server.log"}
        line = c.svc.sandboxes.ran[-1][-1]
        assert "cd /workspace/game && { [ -d node_modules ] || npm install; } && exec npm run dev" \
            in line and "nohup" in line and line.endswith("& echo started")
        st, body, _ = c.call("POST", f"/conversations/{cid}/sandbox/serve", {"dir": "/workspace/loose"})
        assert st == 400 and body["error"]["type"] == "no_dev_server"
        st, body, _ = c.call("POST", f"/conversations/{cid}/sandbox/serve", {"dir": "/etc"})
        assert st == 400
    finally:
        c.close()


# ---------------------------------------------------------------------- what present says
class Ctx:
    def __init__(self, svc, cid):
        self.svc, self.cid = svc, cid
        self.ws = svc.workspace(cid)
        self.panes = []
        self.run_id = "run-1"

    def present(self, pane):
        self.panes.append(pane)


def test_present_reports_the_server_and_what_the_users_pane_loaded():
    c = chat()
    app, _ = server({"/": (200, "text/html", VITE_INDEX, {}),
                     "/@vite/client": (200, "text/javascript", b"1", {})})
    port = app.server_address[1]
    try:
        cid = c.new()
        ctx = Ctx(c.svc, cid)
        # nothing listens: the pane opens and waits; the model hears it shows nothing
        res = T._present(ctx, {"kind": "preview", "port": 4173})
        assert res.is_error and ctx.panes[-1]["ref"] == "port:4173"
        assert "shows nothing yet" in res.content and "do not tell the user it does" in res.content
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        c.svc.sandboxes.listen = [{"port": port, "cwd": "/workspace/app", "started": 5,
                                   "command": "node vite"}]
        # no browser has the chat open: the server's answer, and that no pane has loaded it
        res = T._present(ctx, {"kind": "preview", "port": 4173})
        assert res.content.startswith(f"opened the preview pane on port {port} (node vite).")
        assert f"Nothing listens on port 4173; this is port {port}" in res.content
        assert "answered GET / 200 text/html; 1 of 3 scripts and style sheets answered, failing: " \
               "style.css 404" in res.content and res.is_error
        assert "No browser has this chat open" in res.content

        # a browser has it open: the pane's own report comes back
        conv = c.svc.conv(cid)

        class Sub:
            def send(self, *a):
                pass
        conv.fan.add(Sub())

        def report(errors, failed):
            time.sleep(0.3)
            c.ok("POST", f"/conversations/{cid}/preview/report", {
                "ref": "port:4173", "url": f"https://x/v1/chat-app/conversations/{cid}/sandbox/"
                                           f"ports/{port}/", "title": "Pagoda Garden",
                "errors": errors, "failed": failed, "elements": 3, "text": 0, "canvas": 1})
        threading.Thread(target=report, args=([], ["script https://x/src/main.js"]),
                         daemon=True).start()
        res = T._present(ctx, {"kind": "preview", "port": 4173})
        assert f'The user\'s pane loaded ports/{port}/ (title "Pagoda Garden"); 1 resource(s) ' \
               "failed: script https://x/src/main.js." in res.content and res.is_error
        assert "fix this before telling the user it does" in res.content
        threading.Thread(target=report, args=([], []), daemon=True).start()
        rep = c.svc.wait_report(cid, "port:4173", time.time(), 5)
        assert PV.report_words(rep, 1) == (
            f'The user\'s pane loaded ports/{port}/ (title "Pagoda Garden"); no errors, 3 elements, '
            "1 canvas.", False)
        assert PV.report_words(None, 0)[0].startswith("No browser has this chat open")
        assert PV.report_words(dict(rep, elements=0, canvas=0), 1)[1], "an empty page is a problem"
    finally:
        app.shutdown()
        c.close()


def test_present_of_a_static_page_needs_no_server():
    c = chat()
    try:
        cid = c.new()
        ctx = Ctx(c.svc, cid)
        ctx.ws.write("site/index.html", b"<h1>hi</h1>")
        res = T._present(ctx, {"kind": "preview", "path": "site/index.html"})
        assert not res.is_error and ctx.panes[-1] == {"kind": "preview",
                                                       "ref": "file:/workspace/site/index.html",
                                                       "title": None}
        assert res.content.startswith("opened the preview pane on /workspace/site/index.html.")
    finally:
        c.close()


# ---------------------------------------------------------------------- the shim
def test_the_shim_sends_runtime_urls_and_the_hot_reload_socket_to_the_ports_prefix():
    """The shim as a browser runs it (node, with the few globals it touches): fetch, XHR, a
    WebSocket and an image's src that start with `/` or name localhost:PORT go to the prefix;
    another host and an already prefixed URL stay."""
    node = shutil.which("node")
    if not node:
        print("no node here: the shim's run is skipped")
        return
    prefix = "/v1/chat-app/conversations/C1/sandbox/ports/5173"
    shim = PV.page_shim(prefix, 5173).decode()[len("<script>"):-len("</script>")]
    harness = r"""
const got = [];
globalThis.location = new URL('https://tandem.khaled.de/v1/chat-app/conversations/C1/sandbox/ports/5173/');
globalThis.window = globalThis;
globalThis.fetch = (u) => { got.push(['fetch', String(u)]); return Promise.resolve(); };
globalThis.Request = class { constructor(u) { this.url = u; } };
globalThis.XMLHttpRequest = class {}; XMLHttpRequest.prototype.open = function (m, u) { got.push(['xhr', u]); };
globalThis.WebSocket = class { constructor(u, p) { got.push(['ws', u, p]); } };
class El { setAttribute(n, v) { got.push(['attr', n, v]); } }
globalThis.Element = El;
globalThis.HTMLAnchorElement = class extends El {};
globalThis.HTMLImageElement = class extends El {};
Object.defineProperty(HTMLImageElement.prototype, 'src', { configurable: true, get() { return this._s; }, set(v) { this._s = v; got.push(['img', v]); } });
globalThis.addEventListener = () => {};
globalThis.document = { title: '', body: null, getElementsByTagName: () => [] };
globalThis.parent = { postMessage() {} };
""" + shim + r"""
fetch('/textures/a.jpg'); fetch('https://cdn.example/x.js'); fetch(location.href + 'data.json');
new XMLHttpRequest().open('GET', '/api/x?y=1');
new WebSocket('wss://tandem.khaled.de:/?token=T', 'vite-hmr');
new WebSocket('ws://localhost:5173/?token=T', 'vite-hmr');
new WebSocket('ws://localhost:9999/');
const img = new HTMLImageElement(); img.src = '/textures/b.png';
img.setAttribute('src', '/c.png'); new HTMLAnchorElement().setAttribute('href', '/about');
console.log(JSON.stringify(got));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
    try:
        out = subprocess.run([node, fh.name], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(fh.name)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    p = "https://tandem.khaled.de" + prefix
    assert got == [
        ["fetch", p + "/textures/a.jpg"], ["fetch", "https://cdn.example/x.js"],
        ["fetch", p + "/data.json"], ["xhr", p + "/api/x?y=1"],
        ["ws", "wss://tandem.khaled.de" + prefix + "/?token=T", "vite-hmr"],
        ["ws", "wss://tandem.khaled.de" + prefix + "/?token=T", "vite-hmr"],
        ["ws", "ws://localhost:9999/", None],
        ["img", p + "/textures/b.png"], ["attr", "src", p + "/c.png"], ["attr", "href", "/about"]]


if __name__ == "__main__":
    run_all(globals())
