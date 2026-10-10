"""`/v1/setup/*`: the setup guide's routes (docs/setup.md, contract 1.0; the shapes are
dashboard/src/api/types.ts `Setup*`).

    GET  /v1/setup/models                 a tile per models/<id>/model.yaml, and the files that did not load
    GET  /v1/setup/models/{id}            the model.yaml as JSON, with `wizard_fields` (the step schema)
    GET  /v1/setup/models/{id}/steps      the step schema and the tool definitions Wren is given
    GET  /v1/setup/device                 the box, read only (memory, GPU, disk, profiles on disk)
    POST /v1/setup/estimate {selection}   the memory estimate (server/setup/estimate.py)

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


def assistant_status(app) -> dict:
    w = app.STATE.get("wren")
    if w is not None:
        return w.status()
    return {"name": "Wren", "model": None, "fallback": False, "state": "off",
            "reason": "this server has no assistant", "load_s": None, "device_gib": None,
            "claim_gib": None, "can_load": False}


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
    return _err(h, 404, "not_found", f"no route POST /v1/setup/{'/'.join(parts)}")
