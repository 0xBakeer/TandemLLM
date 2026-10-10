"""`/v1/setup/*`: the setup guide's routes (docs/setup.md, contract 1.0; the shapes are
dashboard/src/api/types.ts `Setup*`).

    GET  /v1/setup/models                 a tile per models/<id>/model.yaml, and the files that did not load
    GET  /v1/setup/models/{id}            the model.yaml as JSON, with `wizard_fields` (the step schema)
    GET  /v1/setup/models/{id}/steps      the step schema and the tool definitions Wren is given
    GET  /v1/setup/device                 the box, read only (memory, GPU, disk, profiles on disk)
    POST /v1/setup/estimate {selection}   the memory estimate (server/setup/estimate.py)
    GET  /v1/setup/assistant/status       Wren (server/setup/wren.py); a poll keeps it loaded
    POST /v1/setup/assistant/load         idempotent; a memory refusal is state `refused`
    POST /v1/setup/assistant/unload       the memory back
    POST /v1/setup/assistant/chat         {messages, page} -> SSE of SetupChatEvent (`data: {...}`);
                                          /v1/setup/chat is the same route

Every route is an admin route (server/auth.py, kind `setup`): open while no admin password is set,
the admin token, a dashboard session or a tool on the box itself once one is. Nothing here runs on a
chat request's path: server/app.py imports this module the first time a /v1/setup route is called.
"""

from __future__ import annotations

from urllib.parse import unquote

from server.setup import device as device_mod
from server.setup import estimate as estimate_mod
from server.setup import registry, tools

CONTRACT = "1.0"
NO_STORE = (("Cache-Control", "no-store"),)


def _err(h, status: int, code: str, message: str, **extra) -> None:
    h._json(status, {"error": dict({"type": code, "code": code, "message": message}, **extra)},
            extra_headers=NO_STORE)


def busy(app) -> bool:
    """Requests are in flight or waiting (what apply and reset refuse to run beside)."""
    inf = app.INFLIGHT
    return bool(inf.get("running") or inf.get("waiting") or inf.get("lane"))


def wren_of(app):
    from server.setup import wren
    return wren.get(app, device_facts=lambda: device_facts(app),
                    setup_mode=bool(app.STATE.get("setup_mode")))


def assistant_status(app) -> dict:
    return wren_of(app).status()


def device_facts(app) -> dict:
    """What Wren reads of the box each turn: the device answer without Wren's own block."""
    d = device_mod.read(app.STATE, inflight=app.INFLIGHT, busy=busy(app), assistant={},
                        gpu=(app.STATE.get("gpu_sampler").sample()
                             if app.STATE.get("gpu_sampler") is not None else None))
    return {k: d.get(k) for k in ("mode", "preset", "mem_total_gb", "mem_available_gb", "gpu",
                                  "disk", "active_profile", "profiles_on_disk", "hf_token_present",
                                  "password_set")}


def device(app) -> dict:
    sampler = app.STATE.get("gpu_sampler")
    if sampler is None:
        from server import dashboard_api
        sampler = app.STATE.setdefault("gpu_sampler", dashboard_api.GpuSampler())
    return device_mod.read(app.STATE, inflight=app.INFLIGHT, busy=busy(app),
                           assistant=assistant_status(app), gpu=sampler.sample())


def _model(h, mid: str) -> dict | None:
    m = registry.get(mid)
    if m is None:
        _err(h, 404, "unknown_model", f"no model {mid!r} in models/")
    return m


def handle(h, app, method: str, path: str, body) -> None:
    if not h._allowed("setup"):
        return
    sub = path[len("/v1/setup"):].strip("/")
    parts = [unquote(x) for x in sub.split("/")] if sub else []
    try:
        if method == "GET":
            return _get(h, app, parts)
        return _post(h, app, parts, body)
    except estimate_mod.SelectionError as exc:
        return h._json(exc.status, exc.body(), extra_headers=NO_STORE)


def _get(h, app, parts: list[str]) -> None:
    if parts == ["models"]:
        models, errors = registry.load_all()
        return h._json(200, {"contract_version": CONTRACT,
                             "models": [registry.tile(m) for m in models.values()],
                             "errors": errors}, extra_headers=NO_STORE)
    if len(parts) == 2 and parts[0] == "models":
        m = _model(h, parts[1])
        if m is None:
            return None
        return h._json(200, dict(m, contract_version=CONTRACT,
                                 wizard_fields=registry.wizard_fields(m)), extra_headers=NO_STORE)
    if len(parts) == 3 and parts[0] == "models" and parts[2] == "steps":
        m = _model(h, parts[1])
        if m is None:
            return None
        return h._json(200, {"contract_version": CONTRACT, "model": m["id"],
                             "steps": registry.wizard_fields(m), "tools": tools.definitions(m)},
                       extra_headers=NO_STORE)
    if parts == ["device"]:
        return h._json(200, device(app), extra_headers=NO_STORE)
    if parts == ["assistant", "status"]:
        w = wren_of(app)
        w.touch()
        return h._json(200, w.status(), extra_headers=NO_STORE)
    return _err(h, 404, "not_found", f"no route /v1/setup/{'/'.join(parts)}")


def selection_model(body) -> dict:
    """The model a selection names, or SelectionError (400 unknown_model)."""
    mid = body.get("model") if isinstance(body, dict) else None
    m = registry.get(mid) if isinstance(mid, str) else None
    if m is None:
        raise estimate_mod.SelectionError(400, "unknown_model", f"no model {mid!r} in models/")
    return m


def _post(h, app, parts: list[str], body) -> None:
    if parts == ["estimate"]:
        sel = body.get("selection", body) if isinstance(body, dict) else body
        m = selection_model(sel)
        mem = device_mod.meminfo()
        out = estimate_mod.estimate(m, estimate_mod.check(m, sel),
                                    available_gb=round(mem["MemAvailable"], 2)
                                    if "MemAvailable" in mem else None)
        return h._json(200, out, extra_headers=NO_STORE)
    if parts == ["assistant", "load"]:
        return h._json(200, wren_of(app).load(), extra_headers=NO_STORE)
    if parts == ["assistant", "unload"]:
        return h._json(200, wren_of(app).unload("asked"), extra_headers=NO_STORE)
    if parts in (["assistant", "chat"], ["chat"]):
        return sse(h, wren_of(app).chat(body))
    return _err(h, 404, "not_found", f"no route POST /v1/setup/{'/'.join(parts)}")


def sse(h, events) -> None:
    """An event stream of `data: {json}` frames; a reader that goes away closes the generator,
    which stops what it was producing."""
    import json
    try:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("X-Accel-Buffering", "no")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        for ev in events:
            h.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
            h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        h.close_connection = True
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()
