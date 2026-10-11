"""chatd's routes (CONTRACT section 2), SSE (section 3) and the forward key.

Every route answers under `/v1/chat-app/...` (the path the engine server forwards unchanged) and
needs `X-Tandem-Forward: <key>`; anything without it gets 403. `GET /health` is the one exception
(the watchdog's): it says `ok` and the version, nothing else.

The forward adds what only the engine server knows: `X-Tandem-Password-Set` and
`X-Tandem-Signed-In` (the capabilities' `session`), and passes the browser's User-Agent, from which
the snapshot names the device a run was started on.

Phase 2 routes (tools, packages, MCP, prompts, presets, permissions, compaction, the sandbox, its
files and terminals, artifacts) answer as the contract says for a box without them: the lists are
empty, a write is 501 `not_available`, and every sandbox route is 503 `sandbox_unavailable`.
"""

from __future__ import annotations

import hmac
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from chatd import __version__
from chatd.events import HEARTBEAT, HEARTBEAT_S, control, frame
from chatd.service import ApiError, Service, device_label

PREFIX = "/v1/chat-app"
MAX_JSON = 1 << 20
MAX_UPLOAD = 51 * 1024 * 1024
_ID = r"([A-Za-z0-9_.:-]{1,80})"

# (method, pattern, handler name)
ROUTES = [
    ("GET", r"/capabilities", "capabilities"),
    ("GET", r"/session", "session"),
    ("GET", r"/conversations", "list"),
    ("POST", r"/conversations", "create"),
    ("GET", rf"/conversations/{_ID}", "get"),
    ("PATCH", rf"/conversations/{_ID}", "patch"),
    ("DELETE", rf"/conversations/{_ID}", "delete"),
    ("POST", r"/folders", "folder"),
    ("GET", rf"/conversations/{_ID}/events", "events"),
    ("POST", rf"/conversations/{_ID}/messages", "send"),
    ("POST", rf"/conversations/{_ID}/queue", "queue"),
    ("DELETE", rf"/conversations/{_ID}/queue/{_ID}", "unqueue"),
    ("POST", rf"/conversations/{_ID}/queue/{_ID}/now", "send_now"),
    ("POST", rf"/conversations/{_ID}/stop", "stop"),
    ("POST", rf"/conversations/{_ID}/continue", "continue_"),
    ("POST", rf"/conversations/{_ID}/messages/{_ID}/regenerate", "regenerate"),
    ("POST", rf"/conversations/{_ID}/messages/{_ID}/edit", "edit"),
    ("POST", rf"/conversations/{_ID}/branch", "branch"),
    ("PATCH", rf"/conversations/{_ID}/settings", "settings"),
    ("GET", r"/events", "global_events"),
    ("POST", r"/attachments", "upload"),
    ("GET", rf"/attachments/{_ID}", "attachment"),
    # phase 2: empty lists, writes not available
    ("GET", r"/(tools|packages|mcp/servers|prompts|presets|permissions/rules)", "empty_list"),
    ("GET", rf"/conversations/{_ID}/terminals", "empty_list_c"),
    ("*", rf"/conversations/{_ID}/(sandbox|terminals)(/.*)?", "sandbox"),
    ("*", r"/(tools|packages|mcp|prompts|presets|permissions)(/.*)?", "not_available"),
    ("POST", rf"/conversations/{_ID}/(compact|permissions/{_ID})", "not_available"),
    ("GET", rf"/conversations/{_ID}/artifacts/{_ID}", "no_artifact"),
]
_COMPILED = [(m, re.compile(p + "$"), n) for m, p, n in ROUTES]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "chatd"
    disable_nagle_algorithm = True            # an SSE frame goes out at once, not after 40 ms
    svc: Service = None                       # set by make_server
    key: str = ""

    def log_message(self, fmt, *args):        # ids and statuses only; never bodies
        pass

    # ------------------------------------------------------------------ plumbing
    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              headers: tuple = ()) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def _empty(self, code: int = 204) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_JSON:
            raise ApiError(413, "too_large", "the body is too large")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except ValueError:
            raise ApiError(400, "bad_request", "the body is not JSON") from None
        if not isinstance(obj, dict):
            raise ApiError(400, "bad_request", "the body must be a JSON object")
        return obj

    @property
    def viewer(self) -> str | None:
        return device_label(self.headers.get("User-Agent"))

    def _authorized(self) -> bool:
        got = self.headers.get("X-Tandem-Forward") or ""
        return bool(self.key) and hmac.compare_digest(got.encode(), self.key.encode())

    def _dispatch(self, method: str) -> None:
        split = urlsplit(self.path)
        path = split.path
        if path in ("/health", "/healthz") and method == "GET":
            return self._json(200, {"status": "ok", "service": "chatd", "version": __version__})
        if not self._authorized():
            self.close_connection = True
            return self._json(403, {"error": {"type": "forbidden",
                                              "message": "chatd answers the engine server's "
                                                         "forward only"}})
        if path.startswith(PREFIX):
            path = path[len(PREFIX):]
        path = path.rstrip("/") or "/"
        self.query = {k: v[-1] for k, v in parse_qs(split.query).items()}
        for m, rx, name in _COMPILED:
            if m not in (method, "*"):
                continue
            hit = rx.match(path)
            if hit:
                try:
                    return getattr(self, "r_" + name)(*[unquote(g) if g else g
                                                         for g in hit.groups()])
                except ApiError as exc:
                    return self._json(exc.status, exc.body())
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                    return
                except Exception as exc:                           # noqa: BLE001
                    self.svc.log(f"[http] {method} {name} failed: {type(exc).__name__}")
                    return self._json(500, {"error": {"type": "internal",
                                                      "message": type(exc).__name__}})
        return self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PUT(self):
        self._dispatch("PUT")

    # ------------------------------------------------------------------ routes
    def r_capabilities(self):
        pw = (self.headers.get("X-Tandem-Password-Set") or "0") == "1"
        signed = (self.headers.get("X-Tandem-Signed-In") or "1") == "1"
        self._json(200, self.svc.capabilities(password_set=pw, signed_in=signed))

    def r_session(self):
        pw = (self.headers.get("X-Tandem-Password-Set") or "0") == "1"
        signed = (self.headers.get("X-Tandem-Signed-In") or "1") == "1"
        self._json(200, {"password_set": pw, "signed_in": signed, "sandbox": "not_installed"})

    def r_list(self):
        q = self.query
        self._json(200, self.svc.list(q.get("q", ""), q.get("archived", "0") in ("1", "true"),
                                      q.get("folder") or None))

    def r_create(self):
        self._json(200, self.svc.create(self._body(), self.viewer))

    def r_get(self, cid):
        self._json(200, self.svc.snapshot(cid, self.viewer))

    def r_patch(self, cid):
        self._json(200, self.svc.patch(cid, self._body()))

    def r_delete(self, cid):
        self.svc.delete(cid)
        self._empty(204)

    def r_folder(self):
        self._json(200, self.svc.create_folder(str(self._body().get("name") or "")))

    def r_send(self, cid):
        self._json(202, self.svc.send(cid, self._body(), self.viewer))

    def r_queue(self, cid):
        self._json(202, self.svc.queue(cid, self._body(), self.viewer))

    def r_unqueue(self, cid, qid):
        self.svc.unqueue(cid, qid)
        self._empty(204)

    def r_send_now(self, cid, qid):
        self.svc.send_now(cid, qid, self.viewer)
        self._json(202, {})

    def r_stop(self, cid):
        self.svc.stop(cid)
        self._json(202, {})

    def r_continue_(self, cid):
        self._json(202, self.svc.continue_run(cid, self.viewer))

    def r_regenerate(self, cid, mid):
        self._body()
        self._json(202, self.svc.regenerate(cid, mid, self.viewer))

    def r_edit(self, cid, mid):
        self._json(202, self.svc.edit(cid, mid, self._body(), self.viewer))

    def r_branch(self, cid):
        self._json(200, self.svc.switch_branch(cid, self._body(), self.viewer))

    def r_settings(self, cid):
        self._json(200, self.svc.update_settings(cid, self._body()))

    def r_upload(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ApiError(413, "too_large", "an attachment may be at most 50 MB")
        raw = self.rfile.read(n) if n else b""
        self._json(200, self.svc.upload(self.headers.get("Content-Type") or "", raw))

    def r_attachment(self, att_id):
        mime, data = self.svc.attachment(att_id)
        self._send(200, data, mime, (("Content-Disposition", "inline"),
                                     ("X-Content-Type-Options", "nosniff"),
                                     ("Content-Security-Policy", "default-src 'none'; sandbox")))

    def r_empty_list(self, *_):
        self._json(200, [])

    def r_empty_list_c(self, cid):
        self.svc.conv(cid)
        self._json(200, [])

    def r_sandbox(self, *_):
        self._json(503, {"error": {"type": "sandbox_unavailable",
                                   "message": "the sandbox is not installed on this box yet "
                                              "(phase 2)"}})

    def r_not_available(self, *_):
        self._json(501, {"error": {"type": "not_available",
                                   "message": "not available yet: tools, permissions, MCP, "
                                              "prompts, presets and compaction come in phase 2"}})

    def r_no_artifact(self, *_):
        self._json(404, {"error": {"type": "not_found", "message": "no such artifact"}})

    # ------------------------------------------------------------------ SSE
    def _after(self) -> int | None:
        raw = self.headers.get("Last-Event-ID") or self.query.get("after")
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            return -1          # an id from somewhere else: reset

    def _stream(self, attach, detach) -> None:
        sub, replay = attach(self._after())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        w = self.wfile
        try:
            if sub is None:
                w.write(control("reset"))
                w.flush()
                return
            for seq, kind, data in replay:
                w.write(frame(seq, kind, data))
            w.write(control("current"))
            w.flush()
            while True:
                items = sub.take(HEARTBEAT_S)
                if items is None:
                    return
                if sub.overflow:
                    w.write(control("reset"))
                    w.flush()
                    return
                if not items:
                    w.write(HEARTBEAT)
                else:
                    w.write(b"".join(frame(*it) for it in items))
                w.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            if sub is not None:
                detach(sub)
                sub.close()

    def r_events(self, cid):
        self.svc.conv(cid)
        self._stream(lambda after: self.svc.attach(cid, after),
                     lambda sub: self.svc.detach(cid, sub))

    def r_global_events(self):
        self._stream(self.svc.attach_global, lambda sub: self.svc.gfan.remove(sub))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(svc: Service, host: str, port: int, key: str) -> Server:
    handler = type("ChatdHandler", (Handler,), {"svc": svc, "key": key})
    return Server((host, port), handler)


def serve_in_thread(srv: Server) -> threading.Thread:
    t = threading.Thread(target=srv.serve_forever, name="chatd-http", daemon=True)
    t.start()
    return t
