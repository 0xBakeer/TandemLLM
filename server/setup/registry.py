"""The model registry: one folder per base model under models/, its model.yaml (schema 1).

    models/<id>/model.yaml           the tile, the weight profiles, the measured table, the tools that
                                     run beside the model, parallelism, methods, the device presets
    models/<id>/README.md            optional
    models/<id>/measurements/*       optional raw data (the measured table's own files)
    models/schema/model.schema.json  SCHEMA below as a JSON Schema file, for editors (`--schema`)

`load_all()` reads every model.yaml with the safe loader, checks it against SCHEMA and the cross rules
below, and returns the models as the JSON the dashboard gets (GET /v1/setup/models/{id}): the file as
written, with each measured cell's list `[tok/s, min, ms a round, tokens a round, n(, ttft s)]` as an
object. A file that does not load is reported with its path and the rest still list; a folder with no
model.yaml is skipped. The result is kept until a file's mtime moves.

The cross rules: a schema version this code does not read is refused whole (a schema 2 must not be
half-read); ids unique per list; exactly one recommended profile; every measured row, asset,
dependency and default names something that exists; the tools that are on by default satisfy their
own dependencies and conflicts, and no dependency goes round in a circle; a field's default and
recommended value pass the field's own check; every `docs` link is a file in the repository. What the
repository adds (tests/test_setup_registry.py): every serve.env key a model names is one the engine
or the scripts read (`unregistered_keys`), and every ops/profiles/<name>.env has an entry
(`unlisted_profiles`).

    python -m server.setup.registry --check [model.yaml ...]   exit 1 and the problems when one fails
    python -m server.setup.registry --schema                   print models/schema/model.schema.json
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import threading

from server.conductor.schema import validate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
SCHEMA_VERSION = 1

STEPS = (("models", "Models"), ("overview", "Model"), ("weights", "Weights"),
         ("components", "Tools"), ("parallelism", "Parallelism and memory"),
         ("methods", "Methods and advanced"), ("review", "Review"), ("install", "Install"))
MODES = ("greedy_512", "greedy_128", "sampled_512", "sampled_128", "long_greedy", "long_sampled")
TASKS = ("prose", "chat", "code", "long32k", "long64k")
GROUPS = ("speed", "roles", "caches", "disk", "features")
FIELD_TYPES = ("enum", "int", "number", "bool", "string")
ENV_KEY = r"^[A-Z][A-Z0-9_]*$"
MODEL_ID = r"^[a-z0-9][a-z0-9.-]{0,63}$"
# a string field without a pattern of its own: what a value written into serve.env may hold
SAFE_VALUE = r"^[A-Za-z0-9._:-]{1,64}$"

_NUM = {"type": "number"}
_NUM0 = {"type": "number", "minimum": 0}
_STR = {"type": "string", "maxLength": 2000}
_ID = {"type": "string", "pattern": r"^[a-z0-9][a-z0-9._-]{0,63}$"}
_IDS = {"type": "array", "items": _ID}
_NOTE = {"type": ["string", "null"], "maxLength": 2000}


def _obj(props: dict, required=None) -> dict:
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else list(required),
            "additionalProperties": False}


_CELL = {"type": "array", "items": _NUM0}
_QUALITY = _obj({"kl_vs_bf16": {"type": ["number", "null"]}, "top1_pct": {"type": ["number", "null"]},
                 "ppl": {"type": ["number", "null"]}, "date": _STR, "source": _STR})
_ASSET = _obj({"id": _ID, "label": _STR, "path": _STR, "also": {"type": "array", "items": _STR},
               "fetchable": {"type": "boolean"}, "protected": {"type": "boolean"},
               "hf_repo": _NOTE, "source": _STR})
_PROFILE_REQUIRED = ("id", "label", "format", "bpw", "disk_gb", "weights_gib", "drafters_gib", "hf_repo",
                     "hf_private", "recommended", "badges", "quality", "assets_required", "note")
# SRV-337, all optional (a 27B profile has none of them): the repository branch, the largest window
# that fits the device preset (null: none fits), a lookup table beside the weights (warm: in the page
# cache; cold: read past it through a row cache), a tool's memory under this profile
_PROFILE = _obj({"id": _ID, "label": _STR, "format": {"enum": ["exl3", "nvfp4"]},
                 "bpw": {"type": ["number", "null"]}, "disk_gb": _NUM0, "weights_gib": _NUM0,
                 "drafters_gib": _NUM0, "hf_repo": _STR, "hf_private": {"type": "boolean"},
                 "recommended": {"type": "boolean"}, "badges": {"type": "array", "items": _STR},
                 "quality": _QUALITY, "assets_required": _IDS, "note": _NOTE,
                 "hf_revision": _NOTE,
                 "max_window": {"type": ["integer", "null"], "minimum": 0},
                 "table": {"type": ["object", "null"],
                           "properties": {"gib": _NUM0, "policy": {"enum": ["warm", "cold"]}, "cache_gib": _NUM0},
                           "required": ["gib", "policy", "cache_gib"], "additionalProperties": False},
                 "components_gib": {"type": "object", "additionalProperties": _NUM0}},
                required=_PROFILE_REQUIRED)
_LOSSLESS = _obj({"identical": {"type": "integer", "minimum": 0},
                  "near_tie": {"type": "integer", "minimum": 0},
                  "other": {"type": "integer", "minimum": 0},
                  "max_near_tie_ulp": {"type": ["number", "null"]}})
_ROW = _obj({"profile": _ID, "build": _STR, "date": _STR, "comparable": {"type": "boolean"},
             "note": _NOTE,
             "cells": {"type": "object", "additionalProperties": False,
                       "properties": {m: {"type": "object", "additionalProperties": False,
                                          "properties": {t: _CELL for t in TASKS}}
                                      for m in MODES}},
             "ttft_s": {"type": "object"},
             "lossless": {"type": ["object", "null"]}},
            required=("profile", "build", "date", "comparable", "cells", "ttft_s", "lossless"))
_COMPONENT = _obj({"id": _ID, "label": _STR, "group": {"enum": list(GROUPS)}, "what": _STR,
                   "default": {"type": "boolean"}, "required": {"type": "boolean"},
                   "memory_gib": _NUM0, "memory_max_gib": {"type": ["number", "null"]},
                   "elastic": {"type": "boolean"}, "disk_gb": _NUM0,
                   "depends_on": _IDS, "conflicts": _IDS,
                   # serve.env key -> {"on": value, "off": value}; checked in _cross (the keys are
                   # upper case, which the conductor's validator does not take as map keys)
                   "knobs": {"type": "object"},
                   "assets": _IDS, "docs": _STR, "note": _NOTE})
_FIELD = _obj({"id": _ID, "label": _STR, "what": _STR, "type": {"enum": list(FIELD_TYPES)},
               "options": {"type": "array", "items": _obj({"value": _STR, "label": _STR})},
               "min": _NUM, "max": _NUM, "pattern": _STR,
               "default": {"type": ["string", "number", "boolean"]},
               "recommended": {"type": ["string", "number", "boolean"]},
               "knob": {"type": ["string", "null"]}, "docs": _NOTE},
              required=("id", "label", "what", "type", "default", "recommended", "knob", "docs"))
_RANGE = _obj({"min": _NUM0, "max": _NUM0, "default": _NUM0})
_SRANGE = _obj({"min": _NUM0, "max": _NUM0, "default": _NUM0, "step": _NUM0})
SCHEMA = _obj({
    "schema": {"const": SCHEMA_VERSION},
    "id": {"type": "string", "pattern": MODEL_ID},
    "name": _STR, "family": _STR, "params": _STR, "pitch": _STR, "architecture": _STR,
    "license": _STR, "icon": _NOTE, "recommended": {"type": "boolean"},
    "links": {"type": "array", "items": _obj({"label": _STR, "url": _STR})},
    "devices": {"type": "array", "items": _obj({"id": _ID, "label": _STR, "total_gib": _NUM0,
                                                "reserved_gib": _NUM0, "note": _NOTE})},
    "assets": {"type": "array", "items": _ASSET},
    "quality_source": _STR,
    "profiles": {"type": "array", "items": _PROFILE},
    "measurements": _obj({"build": _STR, "date": _STR, "method": _STR, "source": _STR,
                          "rows": {"type": "array", "items": _ROW}}),
    "memory": _obj({"parts": {"type": "array",
                              "items": _obj({"id": _ID, "label": _STR, "gib": _NUM0,
                                             "kind": {"enum": ["fixed", "reserve"]},
                                             "source": _STR})},
                    "sources": {"type": "array", "items": _STR},
                    # SRV-337: parts that grow with the window (slot 0's), e.g. a prefill's score buffer
                    "per_window_parts": {"type": "array",
                                         "items": _obj({"id": _ID, "label": _STR,
                                                        "bytes_per_token": {"type": "integer", "minimum": 0},
                                                        "max_gib": _NUM0,
                                                        "kind": {"enum": ["fixed", "reserve"]},
                                                        "source": _STR},
                                                       required=("id", "label", "bytes_per_token", "kind",
                                                                 "source"))}},
                   required=("parts", "sources")),
    "components": {"type": "array", "items": _COMPONENT},
    "parallelism": _obj({"slots": _RANGE, "window": _SRANGE, "side_window": _SRANGE,
                         "kv_pool": _obj({"default": {"type": "boolean"}}),
                         "floor_gb": _RANGE, "kv_hard_floor_gb": _NUM0,
                         "kv_bytes_per_token": {"type": "integer", "minimum": 1},
                         "typical_fill_tokens": {"type": "integer", "minimum": 1},
                         "side_slot_const_gib": _NUM0,
                         "knobs": _obj({k: {"type": "string", "pattern": ENV_KEY}
                                        for k in ("slots", "window", "side_window", "kv_pool",
                                                  "floor_gb")}),
                         "fixed": {"type": "object"}}),
    "methods": {"type": "array", "items": _FIELD},
    "advanced": {"type": "array", "items": _FIELD},
}, required=("schema", "id", "name", "family", "params", "pitch", "architecture", "license",
             "links", "devices", "assets", "profiles", "measurements", "memory", "components",
             "parallelism", "methods", "advanced"))


class RegistryError(ValueError):
    def __init__(self, path: str, problems: list[str]):
        super().__init__(f"{path}: " + "; ".join(problems[:8]))
        self.path, self.problems = path, problems


def field_error(f: dict, value) -> str | None:
    """Why `value` is not a valid value of field `f` (a method, an advanced setting or a step
    field), or None. Strings that came from a client are written into serve.env, so a string is
    only ever an option's value or one that matches the field's pattern."""
    t = f["type"]
    if t == "bool":
        return None if isinstance(value, bool) else "expected true or false"
    if t == "enum":
        opts = [o["value"] if isinstance(o, dict) else o for o in f.get("options") or []]
        return None if isinstance(value, str) and value in opts else f"expected one of {', '.join(opts)}"
    if t in ("int", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                (t == "int" and not float(value).is_integer()):
            return "expected an integer" if t == "int" else "expected a number"
        if value != value or value in (float("inf"), float("-inf")):
            return "expected a number"
        if f.get("min") is not None and value < f["min"]:
            return f"at least {f['min']:g}"
        if f.get("max") is not None and value > f["max"]:
            return f"at most {f['max']:g}"
        return None
    pattern = f.get("pattern") or SAFE_VALUE
    if not isinstance(value, str) or "\n" in value or not re.fullmatch(pattern, value):
        return f"letters, digits and . _ : - only ({pattern})"
    return None


def _docs_ok(link: str, root: str) -> bool:
    path = link.split("#", 1)[0]
    return bool(path) and not os.path.isabs(path) and ".." not in path.split("/") and \
        os.path.isfile(os.path.join(root, path))


def _cycle(comps: list[dict]) -> str | None:
    """A component that depends on itself through others, or None."""
    deps = {c["id"]: c["depends_on"] for c in comps}
    for start in deps:
        seen, todo = set(), list(deps[start])
        while todo:
            d = todo.pop()
            if d == start:
                return start
            if d not in seen and d in deps:
                seen.add(d)
                todo += deps[d]
    return None


def _cross(m: dict, root: str) -> list[str]:
    errs = []
    for key in ("profiles", "components", "methods", "advanced", "devices", "assets"):
        ids = [x["id"] for x in m[key]]
        if len(set(ids)) != len(ids):
            errs.append(f"$.{key}: two entries share an id")
    fields = [f["id"] for f in m["methods"] + m["advanced"]]
    if len(set(fields)) != len(fields):
        errs.append("$.methods/$.advanced: a field id is used twice")
    if not m["devices"]:
        errs.append("$.devices: at least one device preset")
    rec = [p["id"] for p in m["profiles"] if p["recommended"]]
    if len(rec) != 1:
        errs.append(f"$.profiles: exactly one recommended profile, not {len(rec)}")
    assets = {a["id"] for a in m["assets"]}
    for i, p in enumerate(m["profiles"]):
        for a in p["assets_required"]:
            if a not in assets:
                errs.append(f"$.profiles[{i}].assets_required: no asset {a!r}")
        if not p["quality"]["date"] or not p["quality"]["source"]:
            errs.append(f"$.profiles[{i}].quality: a measured value needs its date and source")
    profiles = {p["id"] for p in m["profiles"]}
    rows = m["measurements"]["rows"]
    for i, r in enumerate(rows):
        if r["profile"] not in profiles:
            errs.append(f"$.measurements.rows[{i}]: no profile {r['profile']!r}")
        if not r["comparable"] and not r.get("note"):
            errs.append(f"$.measurements.rows[{i}]: a row that is not comparable says why (note)")
        if r["comparable"] and r["build"] != m["measurements"]["build"]:
            errs.append(f"$.measurements.rows[{i}]: build {r['build']} is not the table's "
                        f"{m['measurements']['build']}; mark it comparable: false")
        for mode, cells in r["cells"].items():
            for task, c in cells.items():
                if len(c) not in (5, 6):
                    errs.append(f"$.measurements.rows[{i}].cells.{mode}.{task}: 5 or 6 numbers")
    comps = {c["id"]: c for c in m["components"]}
    for i, c in enumerate(m["components"]):
        for d in c["depends_on"] + c["conflicts"]:
            if d not in comps or d == c["id"]:
                errs.append(f"$.components[{i}]: no other component {d!r}")
        for a in c["assets"]:
            if a not in assets:
                errs.append(f"$.components[{i}].assets: no asset {a!r}")
        if c["required"] and not c["default"]:
            errs.append(f"$.components[{i}]: a required component is on by default")
        if c["default"]:
            for d in c["depends_on"]:
                if d in comps and not comps[d]["default"]:
                    errs.append(f"$.components[{i}]: on by default, but {d!r} it depends on is not")
            for d in c["conflicts"]:
                if d in comps and comps[d]["default"]:
                    errs.append(f"$.components[{i}]: on by default together with {d!r}, a conflict")
        if c["memory_max_gib"] is not None and c["memory_max_gib"] < c["memory_gib"]:
            errs.append(f"$.components[{i}].memory_max_gib: under memory_gib")
        for k, v in c["knobs"].items():
            if not re.match(ENV_KEY, k):
                errs.append(f"$.components[{i}].knobs: {k!r} is not a serve.env key")
            if not isinstance(v, dict) or set(v) != {"on", "off"} or \
                    not all(x is None or isinstance(x, str) for x in v.values()):
                errs.append(f"$.components[{i}].knobs.{k}: {{\"on\": string|null, \"off\": "
                            f"string|null}} (quote on and off: YAML reads them as booleans)")
            elif any(isinstance(x, str) and "\n" in x for x in v.values()):
                errs.append(f"$.components[{i}].knobs.{k}: one line only")
        if not _docs_ok(c["docs"], root):
            errs.append(f"$.components[{i}].docs: no file {c['docs']!r} in the repository")
    loop = _cycle(m["components"])
    if loop:
        errs.append(f"$.components: {loop!r} depends on itself")
    p = m["parallelism"]
    for k in ("slots", "window", "side_window", "floor_gb"):
        r = p[k]
        if not r["min"] <= r["default"] <= r["max"]:
            errs.append(f"$.parallelism.{k}: the default is outside min..max")
    for k, v in (p.get("fixed") or {}).items():
        if not re.match(ENV_KEY, k) or not isinstance(v, str) or not re.fullmatch(SAFE_VALUE, v):
            errs.append(f"$.parallelism.fixed.{k}: a serve.env key and a plain string value")
    for group in ("methods", "advanced"):
        for i, f in enumerate(m[group]):
            if f["type"] == "enum" and not f.get("options"):
                errs.append(f"$.{group}[{i}]: an enum needs options")
            if f["knob"] is not None and not re.match(ENV_KEY, f["knob"]):
                errs.append(f"$.{group}[{i}].knob: {f['knob']!r} is not a serve.env key")
            for which in ("default", "recommended"):
                why = field_error(f, f[which])
                if why:
                    errs.append(f"$.{group}[{i}].{which}: {why}")
            if f.get("docs") and not _docs_ok(f["docs"], root):
                errs.append(f"$.{group}[{i}].docs: no file {f['docs']!r} in the repository")
    return errs


def _cell(c: list) -> dict:
    out = {"tok_s": c[0], "min": c[1], "ms_round": c[2], "tok_round": c[3], "n": int(c[4])}
    if len(c) > 5:
        out["ttft_s"] = c[5]
    return out


def parse(data, path: str = "model.yaml", root: str | None = None) -> dict:
    """A model.yaml's data, checked, as the API's JSON; RegistryError with every problem."""
    if isinstance(data, dict) and data.get("schema") != SCHEMA_VERSION:
        raise RegistryError(path, [f"$.schema: {data.get('schema')!r}; this server reads schema "
                                   f"{SCHEMA_VERSION} only (a newer file needs a newer TandemLLM)"])
    errs = validate(data, SCHEMA)
    if not errs:
        errs = _cross(data, root or ROOT)
    if errs:
        raise RegistryError(path, errs)
    m = dict(data)
    m["measurements"] = dict(data["measurements"], rows=[
        dict(r, note=r.get("note"),
             cells={mode: {t: _cell(c) for t, c in cells.items()} for mode, cells in r["cells"].items()})
        for r in data["measurements"]["rows"]])
    m.setdefault("recommended", False)
    m["parallelism"] = dict(data["parallelism"], fixed=dict(data["parallelism"].get("fixed") or {}))
    m.pop("quality_source", None)
    return m


def load(path: str, root: str | None = None) -> dict:
    import yaml
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(path, [f"cannot read: {exc}"]) from None
    m = parse(data, path, root)
    if os.path.basename(os.path.dirname(path)) != m["id"]:
        raise RegistryError(path, [f"the folder is not named after the id {m['id']!r}"])
    return m


_CACHE: dict = {"key": None, "models": {}, "errors": []}
_LOCK = threading.Lock()


def load_all(models_dir: str | None = None) -> tuple[dict[str, dict], list[dict]]:
    """id -> model, and [{path, message}] for the files that did not load."""
    d = models_dir or MODELS_DIR
    paths = sorted(glob.glob(os.path.join(d, "*", "model.yaml")))
    key = (d, tuple((p, os.path.getmtime(p)) for p in paths))
    with _LOCK:
        if _CACHE["key"] == key:
            return _CACHE["models"], _CACHE["errors"]
        models, errors = {}, []
        root = os.path.dirname(os.path.abspath(d))
        for p in paths:
            try:
                m = load(p, root)
            except RegistryError as exc:
                errors.append({"path": os.path.relpath(p, root), "message": str(exc)})
                continue
            models[m["id"]] = m
        _CACHE.update(key=key, models=models, errors=errors)
        return models, errors


def get(model_id: str, models_dir: str | None = None) -> dict | None:
    return load_all(models_dir)[0].get(model_id)


def recommended_profile(m: dict) -> dict:
    return next(p for p in m["profiles"] if p["recommended"])


def profile(m: dict, pid: str) -> dict | None:
    return next((p for p in m["profiles"] if p["id"] == pid), None)


def window_refusal(profile_id: str, max_len: int, models: dict[str, dict] | None = None) -> str | None:
    """SRV-335: why a start of `profile_id` with `max_len` must not load, or None. A profile whose entry
    names its largest window on the device (`max_window`) refuses a longer one, and null means it fits
    no window there; a profile without the key (the 27B's) or in no entry is never refused here."""
    for m in (models if models is not None else load_all()[0]).values():
        p = profile(m, profile_id)
        if p is None or "max_window" not in p:
            continue
        dev, mw = m["devices"][0]["label"], p["max_window"]
        if mw is None:
            return f"{profile_id} fits no window on the {dev} (models/{m['id']}/model.yaml: max_window null)"
        if max_len > mw:
            return (f"--max-len {max_len:,} is above {profile_id}'s largest window, {mw:,} on the {dev} "
                    f"(models/{m['id']}/model.yaml)")
        return None
    return None


def component(m: dict, cid: str) -> dict | None:
    return next((c for c in m["components"] if c["id"] == cid), None)


def asset(m: dict, aid: str) -> dict | None:
    return next((a for a in m["assets"] if a["id"] == aid), None)


def fields(m: dict) -> dict[str, tuple[str, dict]]:
    """field id -> (methods | advanced, the field)."""
    return {f["id"]: (g, f) for g in ("methods", "advanced") for f in m[g]}


def tile(m: dict) -> dict:
    """The models step's tile: the fastest greedy-512 cell of the recommended profile."""
    rec = recommended_profile(m)
    best = None
    for r in m["measurements"]["rows"]:
        if r["profile"] != rec["id"] or not r["comparable"]:
            continue
        for task, c in (r["cells"].get("greedy_512") or {}).items():
            if best is None or c["tok_s"] > best["tok_s"]:
                best = {"tok_s": c["tok_s"], "task": task, "profile": rec["id"]}
    return {"id": m["id"], "name": m["name"], "family": m["family"], "params": m["params"],
            "pitch": m["pitch"], "recommended": bool(m.get("recommended")), "icon": m.get("icon"),
            "best_speed": best}


def _field(f: dict) -> dict:
    out = {"field": f["id"], "label": f["label"], "type": f["type"]}
    if f["type"] == "enum":
        out["options"] = [o["value"] for o in f["options"]]
    for k in ("min", "max", "pattern"):
        if f.get(k) is not None:
            out[k] = f[k]
    return out


def wizard_fields(m: dict) -> list[dict]:
    """The steps and their fields: what the step forms show and what Wren's tool calls are checked
    against (`set_field(step, field, value)`, `toggle(component, on)`, `select_profile(id)`)."""
    p = m["parallelism"]
    per_step = {
        "weights": [{"field": "profile", "label": "Weight profile", "type": "enum",
                     "options": [x["id"] for x in m["profiles"]]}],
        "components": [{"field": c["id"], "label": c["label"], "type": "bool"}
                       for c in m["components"]],
        "parallelism": [
            {"field": "slots", "label": "Parallel requests", "type": "int",
             "min": p["slots"]["min"], "max": p["slots"]["max"]},
            {"field": "window", "label": "Context window, first slot", "type": "int",
             "min": p["window"]["min"], "max": p["window"]["max"]},
            {"field": "side_window", "label": "Context window, other slots", "type": "int",
             "min": p["side_window"]["min"], "max": p["side_window"]["max"]},
            {"field": "kv_pool", "label": "Shared KV pool", "type": "bool"},
            {"field": "floor_gb", "label": "Memory kept free (GiB)", "type": "number",
             "min": p["floor_gb"]["min"], "max": p["floor_gb"]["max"]}],
        "methods": [_field(f) for f in m["methods"] + m["advanced"]],
    }
    return [{"index": i, "id": sid, "label": label, "fields": per_step.get(sid, [])}
            for i, (sid, label) in enumerate(STEPS)]


def step_field(m: dict, step, name: str) -> dict | None:
    """The field `name` of step `step` (its id or index), or None."""
    for s in wizard_fields(m):
        if step in (s["id"], s["index"]):
            return next((f for f in s["fields"] if f["field"] == name), None)
    return None


def env_keys(m: dict) -> dict[str, str]:
    """Every serve.env key the model names -> where (for the knob test and the plan)."""
    out = {}
    for c in m["components"]:
        for k in c["knobs"]:
            out[k] = f"components.{c['id']}"
    for f in m["methods"] + m["advanced"]:
        if f["knob"]:
            out[f["knob"]] = f"{'methods' if f in m['methods'] else 'advanced'}.{f['id']}"
    for k, v in m["parallelism"]["knobs"].items():
        out[v] = f"parallelism.{k}"
    for k in m["parallelism"].get("fixed") or {}:
        out[k] = "parallelism.fixed"
    return out


_ASSIGN = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", re.M)


def serve_env_keys(root: str | None = None) -> set[str]:
    """The keys ops/serve.env and the profiles assign, commented-out template lines included
    (`#QSE_KV_POOL=on` documents a key the scripts read), plus the keys ops/start.sh reads."""
    r = root or ROOT
    keys = set()
    for p in [os.path.join(r, "ops", "serve.env")] + glob.glob(os.path.join(r, "ops", "profiles", "*.env")):
        with open(p) as fh:
            keys |= set(_ASSIGN.findall(fh.read()))
    with open(os.path.join(r, "ops", "start.sh")) as fh:
        keys |= set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", fh.read()))
    return keys


def server_env_reads(root: str | None = None) -> set[str]:
    """The QSE_* variables server/ reads from its environment (`os.environ.get("QSE_...")`)."""
    r = root or ROOT
    out = set()
    for p in glob.glob(os.path.join(r, "server", "**", "*.py"), recursive=True):
        with open(p) as fh:
            out |= set(re.findall(r"""environ(?:\.get\(|\[)\s*["'](QSE_[A-Z0-9_]+)["']""", fh.read()))
    return out


def unregistered_keys(m: dict, root: str | None = None) -> list[str]:
    """Keys the model names that nothing reads: a QWEN38_* key outside engine/settings.py's
    registry; any other key that ops/serve.env, a profile or ops/start.sh never names and, for a
    QSE_* key, that server/ never reads from its environment either."""
    from engine.settings import KNOBS
    known = serve_env_keys(root)
    reads = server_env_reads(root)
    bad = []
    for k, where in sorted(env_keys(m).items()):
        if k.startswith("QWEN38_"):
            if k[len("QWEN38_"):] not in KNOBS:
                bad.append(f"{k} ({where}): not in engine/settings.py")
        elif k not in known and k not in reads:
            bad.append(f"{k} ({where}): not in ops/serve.env, a profile or ops/start.sh")
    return bad


def profile_names(root: str | None = None) -> list[str]:
    r = root or ROOT
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(r, "ops", "profiles", "*.env"))
                  if not os.path.basename(p).startswith("_"))


def unlisted_profiles(models: dict[str, dict], root: str | None = None) -> list[str]:
    listed = {p["id"] for m in models.values() for p in m["profiles"]}
    return [n for n in profile_names(root) if n not in listed]


def hf_hub() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")


def expand(path: str, repo: str | None = None) -> str:
    """`${HOME}`, `${REPO}` and `${HF_HUB}` in a registry path (the only three it uses)."""
    return (path.replace("${HOME}", os.path.expanduser("~")).replace("${REPO}", repo or ROOT)
            .replace("${HF_HUB}", hf_hub()))


def asset_present(a: dict, repo: str | None = None) -> bool:
    """Every path of the asset exists (a glob matches)."""
    return all(glob.glob(expand(p, repo)) for p in [a["path"], *a.get("also", [])])


def json_schema() -> dict:
    """SCHEMA as models/schema/model.schema.json."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/0xBakeer/TandemLLM/models/schema/model.schema.json",
            "title": "TandemLLM model registry entry, schema 1 (models/<id>/model.yaml)", **SCHEMA}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m server.setup.registry")
    ap.add_argument("--check", action="store_true", help="check model.yaml files (default: all)")
    ap.add_argument("--schema", action="store_true", help="print the JSON Schema")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    if a.schema:
        print(json.dumps(json_schema(), indent=2, sort_keys=True))
        return 0
    paths = a.paths or sorted(glob.glob(os.path.join(MODELS_DIR, "*", "model.yaml")))
    bad = 0
    for p in paths:
        try:
            m = load(p)
        except RegistryError as exc:
            bad += 1
            print(f"FAIL {p}")
            for line in exc.problems:
                print(f"  {line}")
            continue
        extra = unregistered_keys(m)
        if extra:
            bad += 1
            print(f"FAIL {p}")
            for line in extra:
                print(f"  {line}")
            continue
        print(f"ok   {p}: {m['id']}, {len(m['profiles'])} profiles, {len(m['components'])} tools")
    if not a.paths:
        models, _ = load_all()
        for n in unlisted_profiles(models):
            bad += 1
            print(f"FAIL ops/profiles/{n}.env: no model lists this profile")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
