#!/usr/bin/env python3
"""What the engine is doing when it cannot say so itself (OPS-65, the UT1 self-test, 2026-10-06).

During a deploy hold (07:29 to 07:51) every URL of the service, the dashboard included, answered the
reverse proxy's fallback: a JSON 503 saying "vLLM cold start takes ~5 min", on a white page, for
22 minutes. Two parts fix that, both stdlib, neither touching the engine:

  THE STATUS FILE. ops/start.sh, ops/stop.sh and ops/hold.sh write what they do to
  `logs/engine-status.json` (`QSE_STATUS_FILE`): `{state, since, reason, expected_s, version,
  starts: [{at, s}]}`. `state` is `stopping`, `held` (a hold runs a command with the service
  stopped; its reason is the hold file's first line), `starting`, `up`, `failed` or `down`. A start
  records how long it took; the last five are the estimate for the next one.

  THE RESPONDER (`serve`, opt-in: `QSE_STATUS_PORT` in serve.env, 127.0.0.1 only). The reverse
  proxy sends a request here when the engine does not answer (nginx `error_page 502 503 504`, see
  docs/operations.md). A browser's page request (`/dashboard/...` or `Accept: text/html`) gets a
  small page that says what is going on, since when and about how long, checks `/health` every
  5 s and opens the dashboard when the engine answers. Everything else gets the OpenAI error shape,
  status 503, `Retry-After`, and the same facts under `engine`. The facts come from the file and
  from the engine itself: a file that says `up` while the engine does not answer reads
  "not answering" from the moment the responder first saw it so.

  python3 ops/engine_status.py set starting|up|held|stopping|failed|down [--reason R]
                                   [--expected-s N] [--took-s N]
  python3 ops/engine_status.py show
  python3 ops/engine_status.py serve --port 8009
  python3 ops/engine_status.py ensure --port 8009      # serve in the background unless it answers
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATES = ("starting", "up", "held", "stopping", "failed", "down")
KEEP_STARTS = 5
#: a start takes about this long when none was measured yet (the 27B engine: 3 to 5 minutes)
DEFAULT_START_S = 300
WORDS = {"starting": "starting", "held": "stopped for maintenance", "stopping": "stopping",
         "failed": "failed to start", "down": "stopped", "up": "not answering",
         "not_answering": "not answering"}


def path() -> str:
    return os.environ.get("QSE_STATUS_FILE") or os.path.join(REPO, "logs", "engine-status.json")


def read(p: str | None = None) -> dict:
    try:
        with open(p or path()) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def write(state: str, *, reason: str | None = None, expected_s: float | None = None,
          took_s: float | None = None, p: str | None = None, now: float | None = None) -> dict:
    """The new state, written atomically; a start's duration joins `starts`."""
    if state not in STATES:
        raise SystemExit(f"state must be one of {', '.join(STATES)}")
    p = p or path()
    now = time.time() if now is None else now
    d = read(p)
    starts = [s for s in d.get("starts") or [] if isinstance(s, dict)]
    if took_s is not None:
        starts = (starts + [{"at": int(now), "s": round(float(took_s), 1)}])[-KEEP_STARTS:]
    # the version of the tree the status file belongs to (logs/ sits in it), not this script's
    tree = os.path.dirname(os.path.dirname(os.path.abspath(p)))
    version = None
    for d in (tree, REPO):
        try:
            with open(os.path.join(d, "VERSION")) as f:
                version = f.read().strip()
            break
        except OSError:
            continue
    out = {"state": state, "since": int(now), "reason": (reason or "")[:200] or None,
           "expected_s": expected_s, "version": version, "starts": starts}
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, p)
    return out


def typical_start(d: dict) -> float:
    s = [float(x["s"]) for x in d.get("starts") or [] if isinstance(x, dict) and x.get("s")]
    return float(statistics.median(s)) if s else float(DEFAULT_START_S)


def facts(d: dict, now: float, answering: bool | None = None, seen_down: float | None = None) -> dict:
    """What the page and the JSON say: the state in words, since when, about how long is left
    (a hold: its expected length; a start: the median of the recent starts), Retry-After."""
    state = d.get("state") or "down"
    since = d.get("since")
    if state == "up" and answering is False:
        state, since = "not_answering", int(seen_down or now)
    left = None
    if state == "held" and d.get("expected_s") and since:
        left = max(0.0, since + float(d["expected_s"]) + typical_start(d) - now)
    elif state in ("starting", "stopping") and since:
        left = max(0.0, since + typical_start(d) - now)
    elif state == "not_answering":
        left = typical_start(d)
    retry = int(min(300, max(5, (left or 60) / 4)))
    return {"state": state, "words": WORDS.get(state, state), "since": since,
            "since_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since)) if since else None,
            "reason": d.get("reason"), "left_s": round(left) if left is not None else None,
            "typical_start_s": round(typical_start(d)), "version": d.get("version"),
            "retry_after_s": retry}


def _mins(s: float) -> str:
    return f"{max(1, round(s / 60))} min" if s >= 90 else f"{max(1, round(s))} s"


def message(f: dict) -> str:
    """One sentence for the API error: what, since when, about how long."""
    out = f"TandemLLM is {f['words']}"
    if f.get("since_iso"):
        out += f" (since {f['since_iso']})"
    if f.get("reason"):
        out += f": {f['reason']}"
    if f.get("left_s") is not None:
        out += f"; back in about {_mins(f['left_s'])}" if f["left_s"] > 0 else \
            "; back any moment"
    return out + ". Retry after the Retry-After seconds."


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TandemLLM is {words}</title>
<style>
:root{{--bg:#0e1116;--fg:#e6e9ef;--muted:#9aa3b2;--accent:#ff8a3d;--card:#161b22;--line:#2a313c}}
@media (prefers-color-scheme:light){{:root{{--bg:#f6f7f9;--fg:#14181f;--muted:#5b6472;--card:#fff;--line:#dfe3ea}}}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;padding:16px}}
main{{max-width:520px;width:100%;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}}p{{margin:8px 0}}.muted{{color:var(--muted);font-size:14px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent);margin-right:8px;
animation:p 1.4s ease-in-out infinite}}@keyframes p{{50%{{opacity:.3}}}}
dl{{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:16px 0 0;font-size:14px}}dt{{color:var(--muted)}}dd{{margin:0}}
</style></head><body><main data-engine-status="{state}">
<h1><span class="dot"></span><span id="what">TandemLLM is {words}</span></h1>
<p id="why">{why}</p>
<dl><dt>since</dt><dd id="since" data-ts="{since}">{since_text}</dd>
<dt>back in</dt><dd id="left">{left}</dd><dt>version</dt><dd>{version}</dd></dl>
<p class="muted" id="poll">This page checks every 5 s and opens the dashboard when the engine answers.</p>
</main><script>
const loc=(s)=>s?new Date(s*1000).toLocaleTimeString([], {{hour:'2-digit',minute:'2-digit'}}):'—';
const sinceEl=document.getElementById('since');sinceEl.textContent=loc(+sinceEl.dataset.ts||0);
async function poll(){{try{{const r=await fetch('/health',{{cache:'no-store'}});
if(r.ok){{location.reload();return}}const j=await r.json();const e=j.engine||{{}};
if(e.words)document.getElementById('what').textContent='TandemLLM is '+e.words;
if(e.reason!==undefined)document.getElementById('why').textContent=e.reason||'';
if(e.since)sinceEl.textContent=loc(e.since);
document.getElementById('left').textContent=e.left_s==null?'unknown':(e.left_s>=90?Math.round(e.left_s/60)+' min':Math.max(1,Math.round(e.left_s))+' s');
}}catch(_){{}}document.getElementById('poll').textContent='Last checked '+new Date().toLocaleTimeString()+'; checks every 5 s.'}}
setInterval(poll,5000);
</script></body></html>"""


def page(f: dict) -> str:
    left = "unknown" if f.get("left_s") is None else _mins(f["left_s"]) if f["left_s"] > 0 \
        else "any moment"
    since_text = f.get("since_iso") or "—"
    return PAGE.format(state=html.escape(f["state"]), words=html.escape(f["words"]),
                       why=html.escape(f.get("reason") or ""), since=int(f.get("since") or 0),
                       since_text=html.escape(since_text), left=html.escape(left),
                       version=html.escape(str(f.get("version") or "—")))


def engine_answers(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:                                              # noqa: BLE001
        return False


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "tandem-status"
    status_file: str | None = None
    engine_health = "http://127.0.0.1:8000/health"
    seen_down: list = [None]

    def _facts(self) -> dict:
        now = time.time()
        d = read(self.status_file)
        answering = engine_answers(self.engine_health) if d.get("state") == "up" else None
        if answering is False and self.seen_down[0] is None:
            self.seen_down[0] = now
        if answering:
            self.seen_down[0] = None
        return facts(d, now, answering, self.seen_down[0])

    def _send(self, code: int, body: bytes, ctype: str, retry: int) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", str(retry))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _answer(self) -> None:
        f = self._facts()
        path = self.path.split("?")[0]
        wants_page = (self.command in ("GET", "HEAD")
                      and (path == "/" or path.startswith("/dashboard")
                           or "text/html" in (self.headers.get("Accept") or "")))
        if wants_page:
            return self._send(503, page(f).encode(), "text/html; charset=utf-8",
                              f["retry_after_s"])
        body = {"error": {"message": message(f), "type": "provider_unavailable",
                          "code": "engine_" + f["state"]}, "engine": f}
        if path == "/health":
            body["status"] = f["state"]
        self._send(503, json.dumps(body).encode(), "application/json", f["retry_after_s"])

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _answer

    def do_OPTIONS(self):
        self._answer()

    def log_message(self, *a):
        pass


def serve(port: int, host: str = "127.0.0.1", engine_port: int = 8000) -> None:
    Handler.status_file = path()
    Handler.engine_health = f"http://127.0.0.1:{engine_port}/health"
    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(f"[status] serving {Handler.status_file} on {host}:{port}", flush=True)
    srv.serve_forever()


def ensure(port: int, engine_port: int = 8000) -> str:
    """The responder in the background (its own session, stdio to a log), unless something
    answers on the port already."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return "running"
    except OSError:
        pass
    log = os.path.join(os.path.dirname(path()), "engine-status-server.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", "--port", str(port),
                          "--engine-port", str(engine_port)], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=out, start_new_session=True, close_fds=True,
                         env=dict(os.environ, QSE_STATUS_FILE=path()))
    return "started"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the engine's status file and responder (OPS-65)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("state", choices=STATES)
    s.add_argument("--reason")
    s.add_argument("--expected-s", type=float)
    s.add_argument("--took-s", type=float)
    sub.add_parser("show")
    for name in ("serve", "ensure"):
        p = sub.add_parser(name)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--engine-port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "set":
        write(a.state, reason=a.reason, expected_s=a.expected_s, took_s=a.took_s)
    elif a.cmd == "show":
        print(json.dumps(facts(read(), time.time()), indent=1))
    elif a.cmd == "serve":
        serve(a.port, engine_port=a.engine_port)
    else:
        print(f"[status] responder {ensure(a.port, a.engine_port)} on :{a.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
