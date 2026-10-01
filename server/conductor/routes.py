"""`/v1/conductor/*`: the dashboard's API over the conductor (SRV-119, SRV-129, VIS-42).

Reads follow the dashboard's rule (open with the login off, the admin token or session with it
on). Writes (PUT, restore, mode, label) need the admin bearer token or a dashboard session, or a
tool on the box itself: auth rule `conductor_write` (server/auth.py). Every write is a new
version with a note; nothing edits a version in place.

    GET  /v1/conductor/config                    loaded versions, load errors, journal, roles
    GET  /v1/conductor/meta                      enums, the task catalogue, the schemas
    GET  /v1/conductor/profiles                  tiles
    GET  /v1/conductor/{kind}[/{name}]           settings, router, roles, packages; profiles/{name}
    PUT  /v1/conductor/{kind}[/{name}]           {doc, note, base_version?}: a new version
    GET  .../versions   .../versions/{v}   .../diff?from=&to=
    POST .../restore {version, note?}            a new version equal to an old one
    POST /v1/conductor/profiles/{name}/mode      {mode, note?}: the three-way switch
    POST /v1/conductor/roles/{role}/test         one question, its latency and answer
    GET  /v1/conductor/packages/scan             skills found in skill_dirs on the box
    POST /v1/conductor/classify                  {text, profile?, tools?}: the test box
    GET  /v1/conductor/decisions[/{id}]          the live feed and one decision
    POST /v1/conductor/decisions/{id}/label      {label: right|wrong|clear, note?}
    GET  /v1/conductor/stats?hours=24            the Overview's tiles and ribbon
    GET  /v1/conductor/sessions[/{key}[/export?format=json|md]]
    GET  /v1/conductor/evals                     classifier and suite reports on the box
"""

from __future__ import annotations

import glob
import json
import os
import time
from urllib.parse import parse_qs, unquote, urlsplit

from server.conductor import decisions as dec_mod
from server.conductor import schema, store as store_mod
from server.conductor.store import ConfigError

CONTRACT = "1.0"
DOCS = ("settings", "router", "roles", "packages")


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind = status, kind

    def body(self) -> dict:
        return {"error": {"type": self.kind, "message": str(self)}}


def _author(handler, a) -> str:
    """Who wrote a version: how they were let in, and from where (never a token)."""
    h = handler.headers
    if a.admin_bearer(h):
        how = "admin token"
    elif a.session(h) is not None:
        how = "dashboard session"
    else:
        how = "box"
    who = (h.get("X-Real-IP") or handler._peer() or "")[:64]
    return f"{how} {who}".strip()


def _profile_tile(cond, name: str) -> dict:
    v, doc = cond.store.get("profiles", name)
    meta = cond.store.meta("profiles", name)
    tasks = [t["mode"] for t in doc["tasks"].values()]
    decs = list(doc["decisions"].values())
    return {"name": name, "version": v, "mode": doc["mode"], "description": doc["description"],
            "clients": doc["clients"], "default_class": doc["default_class"],
            "locked": schema.LOCKED.get(name),
            "decisions": doc["decisions"],
            "counts": {"tasks": {m: tasks.count(m) for m in schema.TASK_MODES},
                       "decisions": {m: decs.count(m) for m in schema.MODES3}},
            "last_change": {k: meta.get(k) for k in ("version", "note", "author", "created_at")},
            "error": meta.get("error")}


def handle(handler, cond, method: str, body: dict | None, auth) -> None:
    """Answer one `/v1/conductor/*` request on `handler` (server/app.py's Handler). Reads, the
    test box, a dry validation and a role test follow the dashboard's rule; writes need
    `conductor_write`."""
    split = urlsplit(handler.path)
    parts = [unquote(p) for p in split.path.split("/")[3:] if p]
    q = {k: v[-1] for k, v in parse_qs(split.query).items()}
    reads = parts[:1] == ["classify"] or parts[-1:] in (["check"], ["test"])
    write = method in ("PUT", "POST") and not reads
    if not handler._allowed("conductor_write" if write else "dashboard"):
        return
    body = body if isinstance(body, dict) else {}
    try:
        status, payload, raw = _route(handler, cond, method, parts, q, body, auth)
    except ConfigError as exc:
        return handler._json(exc.status, exc.body())
    except ApiError as exc:
        return handler._json(exc.status, exc.body())
    if raw is not None:
        ctype, data = raw
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", ctype)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True
        return
    payload.setdefault("contract_version", CONTRACT)
    return handler._json(status, payload, extra_headers=(("Cache-Control", "no-store"),))


def _doc_ref(parts: list[str]) -> tuple[str, str | None, list[str]]:
    """(kind, name, the rest of the path) for a document route."""
    kind = parts[0]
    if kind == "profiles":
        if len(parts) < 2:
            raise ApiError(404, "not_found", "which profile?")
        return kind, parts[1], parts[2:]
    return kind, None, parts[1:]


def _base(body: dict) -> int | None:
    b = body.get("base_version")
    if b is None:
        return None
    try:
        return int(b)
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", "base_version is a version number")


def _route(handler, cond, method, parts, q, body, auth):
    st = cond.store
    if not parts:
        raise ApiError(404, "not_found", "no route /v1/conductor")
    head = parts[0]

    if head == "config" and method == "GET":
        return 200, {"server": {"composite": "TandemLLM", "raw_model": cond.state.get("model"),
                                "reasoning_effort": cond.state.get("reasoning_effort"),
                                "global_mode": cond.state.get("conductor_mode"),
                                "self_url": cond.self_url()},
                     "store": st.status(), "journal": cond.journal.info(),
                     "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, None
    if head == "meta" and method == "GET":
        return 200, schema.meta(), None
    if head == "profiles" and len(parts) == 1 and method == "GET":
        _, settings = st.get("settings")
        return 200, {"default_profile": settings["default_profile"],
                     "profiles": [_profile_tile(cond, n) for n in st.names()]}, None
    if head == "classify" and method == "POST":
        text = body.get("text")
        msgs = body.get("messages")
        if not (isinstance(text, str) and text.strip()) and not isinstance(msgs, list):
            raise ApiError(400, "bad_request", "send text (the prompt to test) or messages")
        prof = body.get("profile")
        if prof is not None and not st.exists("profiles", str(prof)):
            raise ApiError(400, "bad_request", f"no profile {prof!r}")
        return 200, cond.router.explain(text or "", prof, tools=bool(body.get("tools", True)),
                                        messages=msgs if isinstance(msgs, list) else None), None
    if head == "decisions":
        if len(parts) == 1 and method == "GET":
            rows = cond.journal.rows(since_ms=int(q.get("since") or 0),
                                     limit=min(500, int(q.get("limit") or 100)),
                                     session=q.get("session") or None,
                                     before_ms=int(q["before"]) if q.get("before") else None)
            return 200, {"decisions": rows}, None
        if len(parts) == 2 and method == "GET":
            r = cond.journal.get(parts[1])
            if r is None:
                raise ApiError(404, "not_found", f"no decision {parts[1]}")
            return 200, {"decision": r}, None
        if len(parts) == 3 and parts[2] == "label" and method == "POST":
            if cond.journal.get(parts[1]) is None:
                raise ApiError(404, "not_found", f"no decision {parts[1]}")
            label = body.get("label")
            if label not in ("right", "wrong", "clear"):
                raise ApiError(400, "bad_request", "label is right, wrong or clear")
            note = str(body.get("note") or "")[:400]
            cond.journal.update(parts[1], label=None if label == "clear" else label,
                                label_note=note or None,
                                labelled_at=int(time.time() * 1000) if label != "clear" else None)
            return 200, {"id": parts[1], "label": None if label == "clear" else label}, None
    if head == "stats" and method == "GET":
        hours = max(1, min(168, int(q.get("hours") or 24)))
        rows = cond.journal.rows(since_ms=int((time.time() - hours * 3600) * 1000), limit=100000)
        out = dec_mod.stats(rows, hours=hours)
        out["roles"] = cond.roles_health()
        return 200, out, None
    if head == "sessions" and method == "GET":
        if len(parts) == 1:
            rows = cond.journal.rows(since_ms=int((time.time() - 7 * 86400) * 1000),
                                     limit=100000)
            return 200, {"sessions": dec_mod.sessions(rows)[:200],
                         "live": cond.router.sessions_live()}, None
        key = parts[1]
        rows = cond.journal.rows(session=key, limit=2000)
        if len(parts) == 3 and parts[2] == "export":
            fmt = q.get("format") or "json"
            if fmt == "md":
                return 200, {}, ("text/markdown; charset=utf-8",
                                 dec_mod.export_md(key, rows).encode())
            notes = [dict(n, decision=r["id"], ts_ms=r["ts_ms"]) for r in
                     sorted(rows, key=lambda x: x["ts_ms"]) for n in (r.get("injected") or [])]
            return 200, {"session": key, "turns": len(rows), "interventions": len(notes),
                         "injections": notes}, None
        if len(parts) == 2:
            live = next((s for s in cond.router.sessions_live() if s["key"] == key), None)
            if not rows and live is None:
                raise ApiError(404, "not_found", f"no session {key}")
            return 200, {"session": key, "live": live,
                         "turns": sorted(rows, key=lambda r: r["ts_ms"])}, None
    if head == "evals" and method == "GET":
        reports = []
        for f in sorted(glob.glob(os.path.join(cond.root, "evals", "*.json")), reverse=True)[:50]:
            try:
                with open(f) as fh:
                    rep = json.load(fh)
                rep["file"] = os.path.basename(f)
                reports.append(rep)
            except (OSError, ValueError):
                continue
        return 200, {"reports": reports,
                     "engine": {"version": cond.state.get("version"),
                                "code_sha": cond.state.get("code_sha")}}, None
    if head == "roles" and len(parts) == 3 and parts[2] == "test" and method == "POST":
        return 200, cond.role_test(parts[1]), None
    if head == "roles" and len(parts) == 1 and method == "GET":
        v, doc = st.get("roles")
        return 200, {"version": v, "doc": doc, "meta": st.meta("roles"),
                     "health": cond.roles_health(force=q.get("fresh") == "1")}, None
    if head == "packages" and len(parts) == 2 and parts[1] == "scan" and method == "GET":
        _, doc = st.get("packages")
        return 200, {"skills": cond.library.scan(doc)}, None
    if head == "profiles" and len(parts) == 3 and parts[2] == "mode" and method == "POST":
        name = parts[1]
        mode = body.get("mode")
        if mode not in schema.PROFILE_MODES:
            raise ApiError(400, "bad_request", f"mode is one of {', '.join(schema.PROFILE_MODES)}")
        if not st.exists("profiles", name):
            raise ApiError(404, "not_found", f"no profile {name}")
        v, doc = st.get("profiles", name)
        doc = dict(doc, mode=mode)
        nv = st.put("profiles", name, doc, note=body.get("note") or f"mode to {mode}",
                    author=_author(handler, auth), base=_base(body))
        return 200, {"name": name, "version": nv, "mode": mode,
                     "loaded_at": st.meta("profiles", name).get("loaded_at")}, None

    if head in DOCS or head == "profiles":
        kind, name, rest = _doc_ref(parts)
        if not st.exists(kind, name) and not (method == "PUT" and kind == "profiles"):
            raise ApiError(404, "not_found", f"no {kind} {name or ''}".strip())
        if method == "GET" and not rest:
            v, doc = st.get(kind, name)
            out = {"kind": kind, "name": name, "version": v, "doc": doc,
                   "meta": st.meta(kind, name)}
            if kind == "profiles":
                out["locked"] = schema.LOCKED.get(name)
            if kind == "packages":
                out["status"] = cond.library.status(doc)
            return 200, out, None
        if method == "PUT" and not rest:
            doc = body.get("doc")
            if not isinstance(doc, dict):
                raise ApiError(400, "bad_request", "send {doc, note}")
            nv = st.put(kind, name, doc, note=str(body.get("note") or ""),
                        author=_author(handler, auth), base=_base(body))
            return 200, {"kind": kind, "name": name, "version": nv,
                         "loaded_at": st.meta(kind, name).get("loaded_at")}, None
        if method == "GET" and rest == ["versions"]:
            return 200, {"kind": kind, "name": name, "versions": st.versions(kind, name)}, None
        if method == "GET" and len(rest) == 2 and rest[0] == "versions":
            try:
                v = int(rest[1])
            except ValueError:
                raise ApiError(400, "bad_request", "a version is a number")
            return 200, {"kind": kind, "name": name, "version": v,
                         "doc": st.version(kind, name, v)}, None
        if method == "GET" and rest == ["diff"]:
            cur = st.get(kind, name)[0]
            try:
                a = int(q.get("from") if q.get("from") is not None else cur)
                b = int(q.get("to") if q.get("to") is not None else cur)
            except ValueError:
                raise ApiError(400, "bad_request", "from and to are version numbers")
            return 200, {"kind": kind, "name": name, "from": a, "to": b,
                         "changes": store_mod.diff(st.version(kind, name, a),
                                                   st.version(kind, name, b))}, None
        if method == "POST" and rest == ["check"]:
            doc = body.get("doc")
            errs = st.check(kind, name, doc) if isinstance(doc, dict) else ["$: send {doc}"]
            cur_v, cur = st.get(kind, name)
            return 200, {"valid": not errs, "errors": errs[:50],
                         "changes": store_mod.diff(cur, doc) if isinstance(doc, dict) else [],
                         "base_version": cur_v}, None
        if method == "POST" and rest == ["restore"]:
            try:
                v = int(body.get("version"))
            except (TypeError, ValueError):
                raise ApiError(400, "bad_request", "send {version}")
            nv = st.restore(kind, name, v, author=_author(handler, auth), note=body.get("note"))
            return 200, {"kind": kind, "name": name, "version": nv, "restored_from": v,
                         "loaded_at": st.meta(kind, name).get("loaded_at")}, None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/{'/'.join(parts)}")
