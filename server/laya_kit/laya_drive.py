#!/usr/bin/env python3
"""Laya plays the software under test: TandemLLM's playtest driver (.tandem/laya_drive.py).

Each step the driver reads the program's state (a small JSON object), asks Laya through TandemLLM
which action from a fixed set comes next, applies it as real input and logs both. A death, a win,
a stuck spot or a step budget ends what it says; `report` turns the log into a short text for you.

  proc     python3 .tandem/laya_drive.py proc "python3 game.py --headless" --actions ACTIONS
           the program prints one JSON state per line and reads one action per line
           ("reset" starts the next episode; its exit ends one, and it is started again)
  cmd      python3 .tandem/laya_drive.py cmd --state "CMD" --act "CMD {action}" [--reset "CMD"]
           a command prints the state, another applies the action (a CLI, an API with curl)
  browser  python3 .tandem/laya_drive.py browser URL --keys run=ArrowRight,jump=Space,wait=
           a Chrome it starts (or --cdp http://127.0.0.1:9222) over the DevTools protocol, real
           key and mouse input; the page defines window.tandem_state() (and tandem_reset())
  report   python3 .tandem/laya_drive.py report [.tandem/drive-log.jsonl]
  serve-log  python3 .tandem/laya_drive.py serve-log [--port 8765]: the in-page kit's log sink

ACTIONS: `run=move right,jump=jump straight up,wait=do nothing` (label=what it does) or
`run,jump,wait`. A state may carry its own choices, for a UI: "actions": ["save", "cancel"] or
"controls": [{"id": "save", "label": "Save", "x": 120, "y": 40} | {..., "selector": "#save"}].
The state's fields the driver reads: dead / over / lost (an episode lost), won / win, lives (a
drop is a death), x and y (or player.x/y, or pos) for stuck spots and the report, cause (why it
died). Everything else goes to Laya as it is: keep it under ~1,000 characters, what matters for the
next move (nearby obstacles, gaps, enemies relative to the player, velocity, lives, score).

Python: `from laya_drive import Driver; Driver(actions).run(get_state, apply, reset)`.
Stdlib only. TANDEM_URL (default https://ai-api.khaled.de) is where Laya is.
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import hashlib
import http.client
import json
import os
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

LOG = ".tandem/drive-log.jsonl"
QUESTION = "Which action should the player take now to stay alive and make progress?"
STATE_CHARS = 1000


# ---------------------------------------------------------------------------- Laya
class Laya:
    """One persistent connection to /v1/laya/ask; a driver session of its own."""

    def __init__(self, url: str | None = None, session: str | None = None, timeout: float = 30.0):
        self.url = urllib.parse.urlsplit((url or os.environ.get("TANDEM_URL")
                                          or "https://ai-api.khaled.de").rstrip("/"))
        self.session = session or f"drive-{os.getpid()}-{int(time.time())}"
        self.timeout = timeout
        self.conn = None

    def _post(self, path: str, body: dict) -> tuple[int, dict, dict]:
        data = json.dumps(body).encode()
        for attempt in (0, 1):
            if self.conn is None:
                cls = (http.client.HTTPSConnection if self.url.scheme == "https"
                       else http.client.HTTPConnection)
                self.conn = cls(self.url.netloc, timeout=self.timeout)
            try:
                self.conn.request("POST", self.url.path + path, data,
                                  {"Content-Type": "application/json"})
                r = self.conn.getresponse()
                raw = r.read()
                try:
                    out = json.loads(raw or b"{}")
                except ValueError:
                    out = {"error": {"type": "bad_json", "message": raw[:200].decode("utf-8", "replace")}}
                return r.status, out, dict(r.getheaders())
            except (OSError, http.client.HTTPException):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
        raise OSError("unreachable")

    def pick(self, question: str, choices: dict, text: str, deadline_ms: int = 15000) -> dict:
        """{label, p, sure, ms} or {error, retry}."""
        t0 = time.perf_counter()
        try:
            st, out, hdr = self._post("/v1/laya/ask", {
                "question": question, "choices": choices, "text": text,
                "session": self.session, "deadline_ms": deadline_ms})
        except OSError as exc:
            return {"error": f"unreachable: {exc}", "ms": round((time.perf_counter() - t0) * 1e3, 1)}
        ms = round((time.perf_counter() - t0) * 1e3, 1)
        if st != 200:
            err = (out.get("error") or {}) if isinstance(out, dict) else {}
            return {"error": err.get("type") or str(st), "message": err.get("message"),
                    "retry": hdr.get("Retry-After"), "ms": ms}
        return {"label": out["label"], "p": out["p"], "sure": out["sure"], "ms": ms,
                "laya_ms": out.get("ms")}

    def end(self, summary: dict) -> dict | None:
        try:
            st, out, _ = self._post("/v1/laya/drive/end", {"session": self.session,
                                                           "summary": summary})
            return out if st == 200 else None
        except OSError:
            return None


# ---------------------------------------------------------------------------- the loop
def parse_actions(spec) -> dict:
    """`run=move right,jump` or a list or a dict -> {label: meaning}."""
    if isinstance(spec, dict):
        return {str(k): str(v or k) for k, v in spec.items()}
    if isinstance(spec, str):
        spec = [s for s in spec.split(",") if s.strip()]
    out = {}
    for a in spec:
        k, _, v = str(a).partition("=")
        out[k.strip()] = v.strip() or k.strip()
    return out


def _num(v):
    return round(v, 1) if isinstance(v, float) else v


def compact(state, limit: int = STATE_CHARS) -> str:
    """The state as short JSON: floats to one decimal, lists cut until it fits."""
    def walk(v, cut):
        if isinstance(v, dict):
            return {k: walk(x, cut) for k, x in v.items() if x is not None}
        if isinstance(v, list):
            return [walk(x, cut) for x in v[:cut]]
        return _num(v)
    for cut in (64, 16, 8, 4, 2, 1):
        s = json.dumps(walk(state, cut), separators=(",", ":"))
        if len(s) <= limit:
            return s
    return s[:limit]


def position(state):
    """(x, y) from x/y, player.x/y or pos, else None."""
    if not isinstance(state, dict):
        return None
    for src in (state, state.get("player") if isinstance(state.get("player"), dict) else None):
        if src and isinstance(src.get("x"), (int, float)):
            return (round(src["x"], 1), round(src["y"], 1) if isinstance(src.get("y"), (int, float)) else None)
    pos = state.get("pos")
    if isinstance(pos, (list, tuple)) and pos and isinstance(pos[0], (int, float)):
        return tuple(round(p, 1) for p in pos[:2])
    return None


def lost(state) -> bool:
    return isinstance(state, dict) and any(state.get(k) is True for k in ("dead", "over", "lost", "game_over"))


def won(state) -> bool:
    return isinstance(state, dict) and any(state.get(k) is True for k in ("won", "win"))


def choices_of(state, fixed: dict) -> dict:
    """The step's choices: the state's own (`actions` / `controls`), else the fixed set."""
    if isinstance(state, dict):
        if isinstance(state.get("controls"), list) and state["controls"]:
            return {str(c.get("id")): str(c.get("label") or c.get("id"))[:200]
                    for c in state["controls"] if isinstance(c, dict) and c.get("id")}
        if isinstance(state.get("actions"), list) and state["actions"]:
            return {str(a): fixed.get(str(a), str(a)) for a in state["actions"]}
    return fixed


class Driver:
    """The decision loop. `get_state()` -> dict; `apply(action, state)`; `reset()` -> None.
    `every_ms` 0: step as fast as Laya answers (a turn-based program); > 0: one step per tick
    (a real-time one). `fallback` is applied when Laya cannot answer (busy, rate, unreachable)."""

    def __init__(self, actions, question: str = QUESTION, url: str | None = None,
                 session: str | None = None, log: str = LOG, every_ms: int = 0,
                 steps: int = 600, episodes: int = 3, fallback: str | None = None,
                 stuck_steps: int = 20, laya: Laya | None = None, label: str = ""):
        self.actions = parse_actions(actions)
        self.question, self.every = question, every_ms / 1000.0
        self.steps, self.episodes, self.stuck_steps = steps, episodes, stuck_steps
        self.fallback = fallback if fallback is not None else (
            "wait" if "wait" in self.actions else next(iter(self.actions)))
        self.laya = laya or Laya(url, session)
        # a real-time program does not wait for a busy engine: two ticks, then the fallback
        self.deadline_ms = max(250, 2 * every_ms) if every_ms else 15000
        self.log_path, self.label = log, label
        self.counts = collections.Counter()

    def _write(self, row: dict) -> None:
        self._f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._f.flush()

    def run(self, get_state, apply, reset=None) -> dict:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._f = open(self.log_path, "a")
        t0 = time.monotonic()
        stamp = lambda: int((time.monotonic() - t0) * 1000)          # noqa: E731
        self._write({"run": self.laya.session, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "label": self.label, "actions": self.actions, "every_ms": int(self.every * 1000),
                     "steps": self.steps, "episodes": self.episodes,
                     "question": hashlib.sha256(self.question.encode()).hexdigest()[:12]})
        c = self.counts
        try:
            for ep in range(1, self.episodes + 1):
                if ep > 1 and reset is not None:
                    reset()
                c["episodes"] += 1
                state = get_state()
                lives = state.get("lives") if isinstance(state, dict) else None
                spot, same, flagged, last_pos, step = position(state), 0, set(), None, 0
                end = "death" if lost(state) else "win" if won(state) else None
                while end is None and step < self.steps:
                    step += 1
                    tick = time.monotonic()
                    ch = choices_of(state, self.actions)
                    v = self.laya.pick(self.question, ch, compact(state), self.deadline_ms)
                    if "error" in v:
                        a, why = (self.fallback if self.fallback in ch else next(iter(ch))), v["error"]
                        c["fallbacks"] += 1
                        c["busy"] += v["error"] == "busy"
                    else:
                        a, why = v["label"], None
                    c["steps"] += 1
                    c[f"a:{a}"] += 1
                    row = {"ep": ep, "step": step, "t": stamp(), "s": compact(state, 600), "a": a,
                           "p": v.get("p"), "sure": v.get("sure"), "ms": v.get("ms")}
                    if why:
                        row["why"] = why
                    self._write(row)
                    apply(a, state)
                    if self.every:
                        time.sleep(max(0.0, self.every - (time.monotonic() - tick)))
                    # where a death happens: the place of the decision before it (a game puts
                    # the player back somewhere else at once)
                    last_pos = position(state)
                    state = get_state()
                    if lost(state):
                        end = "death"
                        break
                    if won(state):
                        end = "win"
                        break
                    nl = state.get("lives") if isinstance(state, dict) else None
                    if isinstance(lives, (int, float)) and isinstance(nl, (int, float)) and nl < lives:
                        c["deaths"] += 1
                        self._write({"ep": ep, "step": step, "t": stamp(), "event": "death",
                                     "s": compact(state, 600), "pos": last_pos,
                                     "cause": state.get("cause"), "lives": nl})
                    lives = nl
                    pos = position(state)
                    if pos is not None and pos == spot:
                        same += 1
                        if same == self.stuck_steps and pos not in flagged:
                            flagged.add(pos)
                            c["stuck"] += 1
                            self._write({"ep": ep, "step": step, "t": stamp(), "event": "stuck",
                                         "pos": pos, "steps": same, "s": compact(state, 600)})
                    else:
                        spot, same = pos, 0
                end = end or "timeout"
                c[{"death": "deaths", "win": "wins", "timeout": "timeouts"}[end]] += 1
                self._write({"ep": ep, "step": step, "t": stamp(), "event": end,
                             "s": compact(state, 600),
                             "pos": last_pos if end == "death" and last_pos else position(state),
                             "cause": state.get("cause") if isinstance(state, dict) else None})
        finally:
            summary = {k: v for k, v in c.items() if ":" not in k}
            srv = self.laya.end(summary)
            self._write({"end": self.laya.session, "t": stamp(), "summary": summary,
                         "actions": {k[2:]: v for k, v in c.items() if k.startswith("a:")},
                         "laya": {k: srv.get(k) for k in ("ms_p50", "ms_p95", "unsure", "errors")}
                         if srv else None})
            self._f.close()
        return dict(self.counts)


# ---------------------------------------------------------------------------- proc and cmd
class Proc:
    """A program speaking JSON lines: a state line out, an action line in."""

    def __init__(self, cmd: str):
        self.cmd = cmd
        self.p = None

    def _start(self):
        self.p = subprocess.Popen(self.cmd, shell=True, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def _read(self) -> dict:
        while True:
            line = self.p.stdout.readline()
            if not line:
                return {"over": True, "cause": f"the program exited ({self.p.wait()})"}
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue

    def state(self) -> dict:
        if self.p is None:
            self._start()
        return self._read()

    def apply(self, action: str, _state=None) -> None:
        try:
            self.p.stdin.write(action + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def reset(self) -> None:
        if self.p is not None and self.p.poll() is None:
            self.apply("reset")
        else:
            self._start()

    def close(self) -> None:
        if self.p is not None and self.p.poll() is None:
            self.p.terminate()


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout


def cmd_state(cmd: str):
    def get():
        out = sh(cmd).strip()
        try:
            return json.loads(out[out.find("{"):]) if "{" in out else {"output": out[-STATE_CHARS:]}
        except ValueError:
            return {"output": out[-STATE_CHARS:]}
    return get


# ---------------------------------------------------------------------------- browser (CDP)
KEYS = {"ArrowLeft": 37, "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40, "Space": 32,
        "Enter": 13, "Escape": 27, "Shift": 16, "Control": 17, "Tab": 9, "Backspace": 8}


def key_event(name: str) -> dict:
    """CDP Input.dispatchKeyEvent fields for `ArrowRight`, `Space`, `a`, `KeyA`, `1`."""
    if name == "Space" or name == " ":
        return {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32, "text": " "}
    if name in KEYS:
        return {"key": name, "code": name if name not in ("Shift", "Control") else name + "Left",
                "windowsVirtualKeyCode": KEYS[name]}
    ch = name[-1] if name.startswith(("Key", "Digit")) else name
    ch = ch.lower() if len(ch) == 1 else ch
    code = ("Digit" + ch) if ch.isdigit() else ("Key" + ch.upper())
    return {"key": ch, "code": code, "windowsVirtualKeyCode": ord(ch.upper()), "text": ch}


def find_chrome() -> str | None:
    cands = [os.environ.get("CHROME"),
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium",
             *(shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome")),
             *sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"))),
             *sorted(glob.glob(os.path.expanduser(
                 "~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*")))]
    return next((c for c in cands if c and os.access(c, os.X_OK)), None)


class CDP:
    """The DevTools protocol over one websocket, stdlib only: evaluate, keys, mouse."""

    def __init__(self, ws_url: str):
        u = urllib.parse.urlsplit(ws_url)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise OSError("the DevTools websocket closed during the handshake")
            head += chunk
        first = head.split(b"\r\n")[0]
        if b" 101 " not in first:
            raise OSError(f"DevTools handshake: {first!r}")
        self.buf = head.split(b"\r\n\r\n", 1)[1]
        self.n = 0

    def _send(self, text: str) -> None:
        data = text.encode()
        mask = os.urandom(4)
        n = len(data)
        hdr = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else
                               bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else
                               bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.sock.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _need(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise OSError("the DevTools websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _recv(self) -> str:
        parts = []
        while True:
            b0, b1 = self._need(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._need(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._need(8))[0]
            data = self._need(n)
            if b0 & 0x0F == 0x8:
                raise OSError("the DevTools websocket closed")
            if b0 & 0x0F in (0x1, 0x0):
                parts.append(data)
                if b0 & 0x80:
                    return b"".join(parts).decode("utf-8", "replace")

    def call(self, method: str, **params):
        self.n += 1
        me = self.n
        self._send(json.dumps({"id": me, "method": method, "params": params}))
        while True:
            msg = json.loads(self._recv())
            if msg.get("id") == me:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result") or {}

    def evaluate(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(f"page: {r['exceptionDetails'].get('text')} "
                               f"{(r['exceptionDetails'].get('exception') or {}).get('description', '')[:300]}")
        return (r.get("result") or {}).get("value")

    def key(self, name: str, down: bool) -> None:
        ev = key_event(name)
        if not down:
            ev.pop("text", None)
        self.call("Input.dispatchKeyEvent", type=("keyDown" if down else "keyUp"), **ev)

    def click(self, x: float, y: float) -> None:
        for t in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)


class Browser:
    """A page under CDP: a Chrome this starts (headless) or one at `cdp`."""

    def __init__(self, url: str, cdp: str | None = None, headless: bool = True,
                 keys: dict | None = None, width: int = 1280, height: int = 800):
        self.proc, self.tmp = None, None
        if cdp is None:
            chrome = find_chrome()
            if chrome is None:
                raise SystemExit("no Chrome or Chromium found (set CHROME=...); or start one with "
                                 "--remote-debugging-port=9222 and pass --cdp http://127.0.0.1:9222")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            self.tmp = tempfile.mkdtemp(prefix="laya-drive-chrome-")
            self.proc = subprocess.Popen(
                [chrome, f"--remote-debugging-port={port}", f"--user-data-dir={self.tmp}",
                 "--no-first-run", "--no-default-browser-check", f"--window-size={width},{height}",
                 # Linux without unprivileged user namespaces (Ubuntu 23.10+) has no sandbox for
                 # it; Playwright starts Chromium the same way there
                 *(["--no-sandbox"] if sys.platform.startswith("linux") else []),
                 *(["--headless=new"] if headless else []), "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cdp = f"http://127.0.0.1:{port}"
        target = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(cdp + "/json", timeout=2) as r:
                    target = next((t for t in json.load(r) if t.get("type") == "page"), None)
                if target:
                    break
            except OSError:
                pass
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(f"Chrome exited at its start ({self.proc.returncode}): {chrome}")
            time.sleep(0.1)
        if target is None:
            raise SystemExit(f"no page at {cdp}")
        self.cdp = CDP(target["webSocketDebuggerUrl"])
        self.cdp.call("Page.enable")
        self.cdp.call("Page.navigate", url=url)
        self.keys = keys or {}
        self.held: set = set()
        self.wait_ready()

    def wait_ready(self, timeout: float = 15.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                if self.cdp.evaluate("document.readyState === 'complete' && "
                                     "typeof (window.tandem_state || window.__state) === 'function'"):
                    return
            except RuntimeError:
                pass
            time.sleep(0.1)
        raise SystemExit("the page has no window.tandem_state() (or __state()) after 15 s")

    def state(self) -> dict:
        return self.cdp.evaluate("(window.tandem_state || window.__state)()") or {}

    def apply(self, action: str, state=None) -> None:
        ctl = next((c for c in (state or {}).get("controls") or ()
                    if isinstance(c, dict) and str(c.get("id")) == action), None)
        if ctl is not None:
            self.release()
            if "x" in ctl and "y" in ctl:
                return self.cdp.click(ctl["x"], ctl["y"])
            r = self.cdp.evaluate(f"(() => {{ const e = document.querySelector({json.dumps(ctl.get('selector') or '#' + action)}); "
                                  f"if (!e) return null; const b = e.getBoundingClientRect(); "
                                  f"return [b.x + b.width / 2, b.y + b.height / 2]; }})()")
            if r:
                self.cdp.click(*r)
            return
        # every step presses its keys anew: a jump on two steps is two jumps
        self.release()
        for k in [k for k in self.keys.get(action, "").split("+") if k]:
            self.cdp.key(k, True)
            self.held.add(k)

    def release(self) -> None:
        for k in self.held:
            self.cdp.key(k, False)
        self.held = set()

    def reset(self) -> None:
        self.release()
        if self.cdp.evaluate("typeof window.tandem_reset === 'function'"):
            self.cdp.evaluate("window.tandem_reset()")
        else:
            self.cdp.call("Page.reload")
            self.wait_ready()

    def close(self) -> None:
        try:
            self.release()
        except (OSError, RuntimeError):
            pass
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------- the report
def report(path: str = LOG, last: int = 6, run: str | None = None) -> str:
    """The log's last run (or `run`) as a short text: outcomes, deaths with where and why and the
    decisions before them, stuck spots, the actions, Laya's numbers."""
    rows = [json.loads(ln) for ln in open(path) if ln.strip().startswith("{")]
    starts = [i for i, r in enumerate(rows) if "run" in r and (run is None or r["run"] == run)]
    if not starts:
        return f"{path}: no run"
    i0 = starts[-1]
    head = rows[i0]
    body = []
    for r in rows[i0 + 1:]:
        if "run" in r:
            break
        body.append(r)
    steps = [r for r in body if "a" in r]
    events = [r for r in body if "event" in r]
    endrow = next((r for r in body if "end" in r), None)
    outc = collections.Counter(e["event"] for e in events if e["event"] in ("death", "win", "timeout"))
    ms = sorted(r["ms"] for r in steps if isinstance(r.get("ms"), (int, float)))
    pct = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))] if ms else None    # noqa: E731
    eps = len({r["ep"] for r in body if "ep" in r})
    out = [f"playtest {head['run']} ({head.get('at')}{', ' + head['label'] if head.get('label') else ''}): "
           f"{eps} episodes, {len(steps)} steps; {outc['win']} won, {outc['death']} deaths, "
           f"{outc['timeout']} hit the step budget"
           + ("" if endrow else " (the log has no end: the run was cut off)")]
    if ms:
        unsure = sum(1 for r in steps if r.get("sure") is False)
        fb = sum(1 for r in steps if r.get("why"))
        out.append(f"Laya: {pct(0.5):.0f} ms p50, {pct(0.95):.0f} ms p95 a step; not sure on "
                   f"{unsure} of {len(steps)}" + (f"; {fb} fallbacks ({', '.join(sorted({r['why'] for r in steps if r.get('why')}))})" if fb else ""))
    hist = collections.Counter(r["a"] for r in steps)
    if hist:
        out.append("actions: " + ", ".join(f"{a} {n * 100 // len(steps)}%" for a, n in hist.most_common()))
    by_ep = collections.defaultdict(list)
    for r in steps:
        by_ep[r["ep"]].append(r)
    deaths = [e for e in events if e["event"] == "death"]
    for n, e in enumerate(deaths[:8], 1):
        where = f" at {fmt_pos(e.get('pos'))}" if e.get("pos") else ""
        why = f": {e['cause']}" if e.get("cause") else ""
        out.append(f"death {n} (episode {e['ep']}, step {e['step']}){where}{why}")
        before = [r for r in by_ep[e["ep"]] if r["step"] <= e["step"]][-last:]
        for r in before:
            p = f" p={r['p']:.2f}" if isinstance(r.get("p"), (int, float)) else ""
            flag = "" if r.get("sure") is not False else " (unsure)"
            out.append(f"   step {r['step']}: {r['a']}{p}{flag}{' [' + r['why'] + ']' if r.get('why') else ''}  {r['s'][:160]}")
    if len(deaths) > 8:
        out.append(f"... {len(deaths) - 8} deaths more")
    spots = collections.Counter(fmt_pos(e.get("pos")) for e in deaths if e.get("pos"))
    if spots and max(spots.values()) > 1:
        out.append("deaths by place: " + ", ".join(f"{p} x{n}" for p, n in spots.most_common(5)))
    for e in [e for e in events if e["event"] == "stuck"][:5]:
        out.append(f"stuck (episode {e['ep']}, step {e['step']}) at {fmt_pos(e.get('pos'))} for "
                   f"{e['steps']} steps: {e['s'][:160]}")
    for e in [e for e in events if e["event"] == "timeout"][:3]:
        out.append(f"step budget hit (episode {e['ep']}) at {fmt_pos(e.get('pos')) or e['s'][:120]}")
    return "\n".join(out)


def fmt_pos(p) -> str:
    if not p:
        return ""
    return "x=%s y=%s" % (p[0], p[1]) if len(p) > 1 and p[1] is not None else f"x={p[0]}"


# ---------------------------------------------------------------------------- the log sink
def serve_log(port: int, path: str = LOG) -> None:
    """POST /log (JSON lines) appends to the log; the in-page kit's sink. CORS open."""
    import http.server
    lock = threading.Lock()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    class H(http.server.BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
            lines = [ln for ln in data.splitlines() if ln.strip().startswith("{")]
            with lock, open(path, "a") as f:
                for ln in lines:
                    f.write(ln.strip() + "\n")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"lines": len(lines)}).encode())

        def log_message(self, *a):
            pass

    print(f"drive log sink: POST http://127.0.0.1:{port}/log -> {path}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


# ---------------------------------------------------------------------------- the command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--actions", default="", help="label=meaning,... (or labels)")
        p.add_argument("--question", default=QUESTION)
        p.add_argument("--episodes", type=int, default=3)
        p.add_argument("--steps", type=int, default=600, help="step budget per episode")
        p.add_argument("--every-ms", type=int, default=None,
                       help="one step per tick (browser: 250; else 0, as fast as Laya answers)")
        p.add_argument("--fallback", default=None, help="the action when Laya cannot answer (default wait or the first)")
        p.add_argument("--stuck", type=int, default=20, help="steps on one spot that count as stuck")
        p.add_argument("--log", default=LOG)
        p.add_argument("--url", default=None, help="TandemLLM (default $TANDEM_URL or https://ai-api.khaled.de)")
        p.add_argument("--label", default="", help="a word for the report's head (the build, the fix)")
        p.add_argument("--no-report", action="store_true")

    p = sub.add_parser("proc")
    p.add_argument("command")
    common(p)
    p = sub.add_parser("cmd")
    p.add_argument("--state", required=True)
    p.add_argument("--act", required=True, help="the command, {action} is replaced")
    p.add_argument("--reset", default=None)
    common(p)
    p = sub.add_parser("browser")
    p.add_argument("page", help="the page's URL")
    p.add_argument("--keys", default="", help="action=Key+Key,...: run=ArrowRight,jump=Space,wait=")
    p.add_argument("--cdp", default=None, help="a running Chrome's DevTools (http://127.0.0.1:9222)")
    p.add_argument("--headed", action="store_true")
    common(p)
    p = sub.add_parser("report")
    p.add_argument("log", nargs="?", default=LOG)
    p.add_argument("--last", type=int, default=6, help="decisions shown before each death")
    p = sub.add_parser("serve-log")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--log", default=LOG)
    a = ap.parse_args(argv)

    if a.mode == "report":
        print(report(a.log, a.last))
        return 0
    if a.mode == "serve-log":
        serve_log(a.port, a.log)
        return 0
    keys = {}
    if a.mode == "browser":
        for part in [x for x in a.keys.split(",") if x.strip()]:
            k, _, v = part.partition("=")
            keys[k.strip()] = v.strip()
    # a UI's state may list its own controls each step; then the fixed set is only the fallback
    actions = parse_actions(a.actions) if a.actions else ({k: k for k in keys} or {"wait": "do nothing"})
    every = a.every_ms if a.every_ms is not None else (250 if a.mode == "browser" else 0)
    drv = Driver(actions, a.question, a.url, log=a.log, every_ms=every, steps=a.steps,
                 episodes=a.episodes, fallback=a.fallback, stuck_steps=a.stuck, label=a.label)
    if a.mode == "proc":
        prog = Proc(a.command)
        try:
            drv.run(prog.state, prog.apply, prog.reset)
        finally:
            prog.close()
    elif a.mode == "cmd":
        drv.run(cmd_state(a.state), lambda act, s: sh(a.act.replace("{action}", shlex.quote(act))),
                (lambda: sh(a.reset)) if a.reset else None)
    else:
        b = Browser(a.page, a.cdp, headless=not a.headed, keys=keys)
        try:
            drv.run(b.state, b.apply, b.reset)
        finally:
            b.close()
    if not a.no_report:
        print(report(a.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
