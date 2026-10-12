"""Terminals and the port preview (SRV-357, Chat 5). Mixed into `chatd.service.Service`.

TERMINALS. `shell` is the user's own PTY in the chat's sandbox (`bash -l` through an exec with a
TTY). chatd holds it, not the browser: the last 1 MiB of output stays in a ring buffer, a device
that attaches gets that first and then the live bytes, several devices may attach and type at once,
and a terminal nobody has attached for 30 minutes is closed. Any other session name is an agent
call id: a read-only view of that call's output (its scrollback, then its live `tool.output`), so
the user's shell and the agent never type into the same PTY.

    WS /conversations/{id}/terminals/{session}/ws   binary frames = PTY bytes both ways;
                                                    text frames = {"type":"resize","cols","rows"}

THE PORT PREVIEW. The container has no network, so a dev server inside it is reached through an
exec of `relay.py connect <port>`: chatd writes the HTTP request to its stdin and reads the answer
from its stdout. Each answer carries the preview CSP (CONTRACT section 5); an HTML answer's
absolute `/` URLs are rewritten to the port's prefix (a dev server started without `--base`).
WebSocket upgrades (hot reload) are not passed through yet.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import socket
import struct
import threading
import time

from chatd.events import Subscriber
from chatd.sandbox import Unavailable, demux

RING = 1024 * 1024
IDLE_CLOSE_S = 30 * 60
GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ---------------------------------------------------------------------- RFC 6455, the server side
def ws_accept(handler) -> bool:
    key = handler.headers.get("Sec-WebSocket-Key")
    if (handler.headers.get("Upgrade") or "").lower() != "websocket" or not key:
        return False
    accept = base64.b64encode(hashlib.sha1(key.encode() + GUID).digest()).decode()
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()
    handler.wfile.flush()
    handler.close_connection = True
    return True


class WsConn:
    def __init__(self, handler):
        self.r = handler.rfile
        self.w = handler.wfile
        self.lock = threading.Lock()
        self.closed = False

    def send(self, opcode: int, data: bytes) -> None:
        n = len(data)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([n])
        elif n < 65536:
            head += bytes([126]) + struct.pack(">H", n)
        else:
            head += bytes([127]) + struct.pack(">Q", n)
        with self.lock:
            if self.closed:
                return
            try:
                self.w.write(head + data)
                self.w.flush()
            except (OSError, ValueError):
                self.closed = True

    def binary(self, data: bytes) -> None:
        self.send(0x2, data)

    def text(self, s: str) -> None:
        self.send(0x1, s.encode())

    def recv(self) -> tuple[int, bytes] | None:
        """The next whole message (fragments joined); pings answered here. None: closed."""
        buf, first_op = b"", None
        while True:
            h = self.r.read(2)
            if len(h) < 2:
                return None
            fin, op = h[0] & 0x80, h[0] & 0x0F
            masked, n = h[1] & 0x80, h[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", self.r.read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self.r.read(8))[0]
            mask = self.r.read(4) if masked else b""
            data = self.r.read(n) if n else b""
            if masked:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 0x8:
                self.send(0x8, data[:2])
                self.closed = True
                return None
            if op == 0x9:
                self.send(0xA, data)
                continue
            if op == 0xA:
                continue
            if first_op is None:
                first_op = op
            buf += data
            if fin:
                return first_op, buf


class Terminal:
    def __init__(self, svc, cid: str, session: str, exec_id: str, sock: socket.socket, first: bytes):
        self.svc, self.cid, self.session = svc, cid, session
        self.exec_id, self.sock = exec_id, sock
        self.ring = bytearray(first[-RING:])
        self.clients: set[WsConn] = set()
        self.lock = threading.Lock()
        self.alive = True
        self.created = time.time()
        self.detached_at = time.time()
        threading.Thread(target=self._pump, name="chatd-term", daemon=True).start()

    def _pump(self) -> None:
        while True:
            try:
                d = self.sock.recv(65536)
            except OSError:
                d = b""
            if not d:
                break
            with self.lock:
                self.ring += d
                if len(self.ring) > RING:
                    del self.ring[:len(self.ring) - RING]
                clients = list(self.clients)
            for c in clients:
                c.binary(d)
        self.alive = False
        for c in list(self.clients):
            c.binary(b"\r\n\x1b[2m[the shell ended]\x1b[0m\r\n")

    def write(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except OSError:
            self.alive = False

    def resize(self, cols: int, rows: int) -> None:
        try:
            self.svc.sandboxes.engine.call("POST", f"/exec/{self.exec_id}/resize?h={int(rows)}"
                                                   f"&w={int(cols)}", timeout=5)
        except OSError:
            pass

    def close(self) -> None:
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class TerminalsMixin:
    def _terms(self) -> dict:
        if not hasattr(self, "_terminals"):
            self._terminals = {}
        return self._terminals

    def _calls(self, cid: str) -> list[dict]:
        conv = self.conv(cid)
        with conv.lock:
            row = self._row(cid)
            branch, _ = self._branch(conv, row["head_id"])
        return [p for m in branch if m["role"] == "assistant" for p in m["parts"]
                if p.get("type") == "tool_call"]

    def terminals_list(self, cid: str) -> list[dict]:
        out = []
        t = self._terms().get((cid, "shell"))
        if t is not None:
            out.append({"session": "shell", "title": "Terminal", "owner": "user",
                        "alive": t.alive})
        for p in self._calls(cid)[-20:]:
            if p.get("name") in ("bash", "render_page"):
                try:
                    cmd = json.loads(p.get("input") or "{}").get("command") or p["name"]
                except ValueError:
                    cmd = p["name"]
                out.append({"session": p["id"], "title": str(cmd)[:60], "owner": "agent",
                            "alive": p.get("status") == "running"})
        return out

    def terminal_open(self, cid: str, session: str = "") -> dict:
        t = self._shell(cid)
        return {"session": t.session, "title": "Terminal", "owner": "user", "alive": t.alive}

    def _shell(self, cid: str) -> Terminal:
        terms = self._terms()
        t = terms.get((cid, "shell"))
        if t is not None and t.alive:
            return t
        self.sandboxes.ensure(cid)
        exec_id, sock, first = self.sandboxes.open_exec(
            cid, ["bash", "-l"], tty=True, stdin=True,
            env=["TERM=xterm-256color", "PS1=\\[\\e[32m\\]sandbox\\[\\e[0m\\]:\\[\\e[34m\\]\\w"
                 "\\[\\e[0m\\]$ "])
        t = terms[(cid, "shell")] = Terminal(self, cid, "shell", exec_id, sock, first)
        return t

    def terminal_ws(self, handler, cid: str, session: str) -> None:
        self.conv(cid)
        if not ws_accept(handler):
            raise self._api_error(400, "bad_request", "a WebSocket upgrade is expected")
        ws = WsConn(handler)
        if session != "shell":
            return self._call_view(ws, cid, session)
        try:
            t = self._shell(cid)
        except Unavailable as exc:
            ws.binary(f"\x1b[31mThe sandbox is not available: {exc}\x1b[0m\r\n".encode())
            ws.send(0x8, b"")
            return
        box = self.sandboxes.box(cid)
        box.busy += 1
        with t.lock:
            ws.binary(bytes(t.ring))
            t.clients.add(ws)
        try:
            while t.alive:
                m = ws.recv()
                if m is None:
                    break
                op, data = m
                if op == 0x2:
                    t.write(data)
                elif op == 0x1:
                    try:
                        c = json.loads(data)
                    except ValueError:
                        continue
                    if c.get("type") == "resize":
                        t.resize(c.get("cols") or 80, c.get("rows") or 24)
                    elif c.get("type") == "input":
                        t.write(str(c.get("data") or "").encode())
                self.sandboxes.touch(cid)
        finally:
            with t.lock:
                t.clients.discard(ws)
                if not t.clients:
                    t.detached_at = time.time()
            box.busy -= 1

    def _call_view(self, ws: WsConn, cid: str, call_id: str) -> None:
        """An agent call's output, read-only: what it wrote so far, then live."""
        conv = self.conv(cid)
        sub = Subscriber()
        with conv.lock:
            part = next((p for p in self._calls(cid) if p.get("id") == call_id), None)
            live = conv.live
            if part is None and live is not None:
                part = next((p for p in live["parts"] if p.get("id") == call_id), None)
            if part is None:
                ws.binary(f"no output for {call_id}\r\n".encode())
                ws.send(0x8, b"")
                return
            conv.fan.add(sub)
            try:
                cmd = json.loads(part.get("input") or "{}").get("command") or part.get("name")
            except ValueError:
                cmd = part.get("name")
            head = f"\x1b[2m$ {cmd}\x1b[0m\r\n" + ((part.get("output") or {}).get("content")
                                                    or "")
            ws.binary(head.replace("\r\n", "\n").replace("\n", "\r\n").encode())
            running = part.get("status") == "running"
        try:
            while running and not ws.closed:
                items = sub.take(15.0)
                if items is None:
                    break
                for _, kind, data in items:
                    if kind not in ("tool.output", "tool.done"):
                        continue
                    ev = json.loads(data)
                    if ev.get("call_id") != call_id:
                        continue
                    if kind == "tool.done":
                        running = False
                        break
                    ws.binary(ev["delta"].replace("\r\n", "\n").replace("\n", "\r\n").encode())
        finally:
            with conv.lock:
                conv.fan.remove(sub)
        ws.binary(b"\r\n\x1b[2m[the command ended]\x1b[0m\r\n")

    def close_idle_terminals(self) -> None:
        now = time.time()
        for key, t in list(self._terms().items()):
            if not t.alive or (not t.clients and now - t.detached_at > IDLE_CLOSE_S):
                t.close()
                self._terms().pop(key, None)

    # ------------------------------------------------------------------ the port preview
    def proxy_port(self, handler, cid: str, port: int, rest: str) -> None:
        from chatd.panes import PREVIEW_CSP
        self.conv(cid)
        box = self.sandboxes.box(cid)
        if box.state != "running":
            raise self._api_error(503, "sandbox_stopped", "the sandbox is not running; start it "
                                                          "or ask the agent to start the server")
        q = handler.path.split("?", 1)
        path = (rest or "/") + (f"?{q[1]}" if len(q) > 1 else "")
        if (handler.headers.get("Upgrade") or "").lower() == "websocket":
            return self._proxy_ws(handler, cid, port, path)
        n = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(n) if n else b""
        lines = [f"{handler.command} {path} HTTP/1.1", f"Host: 127.0.0.1:{port}",
                 "Connection: close"]
        for k, v in handler.headers.items():
            if k.lower() in ("host", "connection", "cookie", "authorization", "upgrade",
                             "keep-alive", "x-tandem-forward") or k.lower().startswith("x-tandem"):
                continue
            lines.append(f"{k}: {v}")
        exec_id, sock, first = self.sandboxes.open_exec(
            cid, ["python3", "/opt/tandem/relay.py", "connect", str(port)], stdin=True)
        box.busy += 1
        try:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
            chunks: list[bytes] = []
            demux(sock, first, lambda s, d: chunks.append(d) if s == 1 else None)
        finally:
            box.busy -= 1
            try:
                sock.close()
            except OSError:
                pass
        raw = b"".join(chunks)
        head, _, payload = raw.partition(b"\r\n\r\n")
        status_line, _, hdrs = head.partition(b"\r\n")
        parts = status_line.split(b" ", 2)
        status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 502
        keep = []
        ctype = "application/octet-stream"
        for h in hdrs.split(b"\r\n"):
            k, _, v = h.decode("latin-1").partition(":")
            kl = k.strip().lower()
            if not kl or kl in ("content-length", "connection", "transfer-encoding",
                                "content-security-policy", "set-cookie", "keep-alive"):
                continue
            if kl == "content-type":
                ctype = v.strip()
                continue
            keep.append((k.strip(), v.strip()))
        if b"transfer-encoding: chunked" in hdrs.lower():
            payload = _dechunk(payload)
        if ctype.startswith("text/html"):
            prefix = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}/"
            payload = re.sub(rb'(src|href|action)=(["\'])/(?!/)',
                             lambda m: m.group(1) + b"=" + m.group(2) + prefix.encode(), payload)
        handler._send(status, payload, ctype, tuple(keep) + (
            ("Content-Security-Policy", PREVIEW_CSP), ("X-Content-Type-Options", "nosniff")))

    def _proxy_ws(self, handler, cid: str, port: int, path: str) -> None:
        """A WebSocket from the preview to a server in the sandbox (a dev server's live reload, an
        app's socket): the upgrade goes to it through the relay, its 101 comes back as it is, then
        bytes both ways until either side closes. When the browser closes its side the relay's
        stdin closes (it then closes its side to the server); a relay still there 5 s later is
        ended."""
        lines = [f"GET {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Upgrade: websocket",
                 "Connection: Upgrade"]
        for k, v in handler.headers.items():
            kl = k.lower()
            if kl.startswith("sec-websocket-") or kl in ("origin", "user-agent", "accept-language"):
                lines.append(f"{k}: {v}")
        box = self.sandboxes.box(cid)
        exec_id, sock, first = self.sandboxes.open_exec(
            cid, ["python3", "/opt/tandem/relay.py", "connect", str(port)], stdin=True)
        box.busy += 1
        handler.close_connection = True
        out = handler.wfile

        def down(stream: int, data: bytes) -> None:
            if stream == 1:
                out.write(data)
                out.flush()

        def up() -> None:
            try:
                while True:
                    data = handler.rfile.read1(65536)
                    if not data:
                        break
                    sock.sendall(data)
                    self.sandboxes.touch(cid)
            except OSError:
                pass
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            time.sleep(5)
            self.sandboxes.end_exec(exec_id)
        try:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
            threading.Thread(target=up, daemon=True).start()
            demux(sock, first, down)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            box.busy -= 1
            try:
                sock.close()
            except OSError:
                pass


def _dechunk(data: bytes) -> bytes:
    out, i = b"", 0
    while i < len(data):
        j = data.find(b"\r\n", i)
        if j < 0:
            break
        try:
            n = int(data[i:j].split(b";")[0], 16)
        except ValueError:
            break
        if n == 0:
            break
        out += data[j + 2:j + 2 + n]
        i = j + 2 + n + 2
    return out
