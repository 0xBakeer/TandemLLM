"""Wren's tools (SRV-325): what the assistant may ask the setup page to do, and the check of a call.

The browser executes every call; the server never applies one. The server checks a call against the
model's step schema (`registry.wizard_fields`) before it streams it, and the browser checks it again
before it acts. There is no tool to finish, apply, reset, delete, load or unload anything: those are
the user's buttons.

    set_field(step, field, value)   a field of a step (the step's id, or its index 0..7)
    toggle(component, on)           a tool beside the model on or off
    select_profile(id)              the weight profile
    go_to_step(n)                   step n, 0..7
    next(), previous()              one step on or back
    explain(field)                  point at a field, a tool or a profile while Wren explains it
"""

from __future__ import annotations

import json
import re

from server.setup import registry

NAMES = ("set_field", "toggle", "select_profile", "go_to_step", "next", "previous", "explain")


def definitions(m: dict) -> list[dict]:
    """The OpenAI tool definitions Wren is given, with the model's own ids in the enums."""
    steps = registry.wizard_fields(m)
    step_ids = [s["id"] for s in steps if s["fields"]]
    comps = [c["id"] for c in m["components"] if not c["required"]]
    names = sorted({f["field"] for s in steps for f in s["fields"]} | set(comps)
                   | {p["id"] for p in m["profiles"]} | {s["id"] for s in steps})

    def fn(name, description, props=None, required=()):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props or {},
                           "required": list(required), "additionalProperties": False}}}

    return [
        fn("set_field", "Set a field of a step to a value, like the number of slots or the port.",
           {"step": {"type": "string", "enum": step_ids},
            "field": {"type": "string"},
            "value": {"type": ["string", "number", "boolean"]}}, ("step", "field", "value")),
        fn("toggle", "Switch a tool beside the model on or off.",
           {"component": {"type": "string", "enum": comps}, "on": {"type": "boolean"}},
           ("component", "on")),
        fn("select_profile", "Pick the weight profile.",
           {"id": {"type": "string", "enum": [p["id"] for p in m["profiles"]]}}, ("id",)),
        fn("go_to_step", "Open a step by its number, 0 to 7.",
           {"n": {"type": "integer", "minimum": 0, "maximum": len(steps) - 1}}, ("n",)),
        fn("next", "Go to the next step."),
        fn("previous", "Go back one step."),
        fn("explain", "Point at a field, a tool or a weight profile on the page while you explain it.",
           {"field": {"type": "string", "enum": names}}, ("field",)),
    ]


def _value_error(f: dict, value) -> str | None:
    spec = dict(f, options=[{"value": o} for o in f.get("options") or []])
    return registry.field_error(spec, value)


def check(m: dict, name: str, args) -> tuple[dict | None, str | None]:
    """(the call's arguments, None) when `name(args)` is a valid call on model `m`; else
    (None, why) in a sentence the model can act on. `args` may be the JSON text of a tool call."""
    if name not in NAMES:
        return None, f"there is no tool {name!r}; the tools are {', '.join(NAMES)}"
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return None, "the arguments are not valid JSON"
    if not isinstance(args, dict):
        return None, "the arguments must be a JSON object"
    want = {"set_field": {"step", "field", "value"}, "toggle": {"component", "on"},
            "select_profile": {"id"}, "go_to_step": {"n"}, "next": set(), "previous": set(),
            "explain": {"field"}}[name]
    if set(args) != want:
        return None, (f"{name} takes {', '.join(sorted(want)) or 'no arguments'}, "
                      f"not {', '.join(sorted(args)) or 'none'}")
    steps = registry.wizard_fields(m)
    if name == "set_field":
        step = args["step"]
        if isinstance(step, str) and re.fullmatch(r"\d", step):
            step = int(step)
        f = registry.step_field(m, step, args["field"]) if isinstance(args["field"], str) else None
        if f is None:
            return None, f"step {args['step']!r} has no field {args['field']!r}"
        why = _value_error(f, args["value"])
        if why:
            return None, f"{args['field']}: {why}"
        return args, None
    if name == "toggle":
        c = registry.component(m, args["component"]) if isinstance(args["component"], str) else None
        if c is None:
            return None, f"there is no tool {args['component']!r}"
        if not isinstance(args["on"], bool):
            return None, "on must be true or false"
        if c["required"] and not args["on"]:
            return None, f"{c['label']} cannot be switched off"
        return args, None
    if name == "select_profile":
        if not isinstance(args["id"], str) or registry.profile(m, args["id"]) is None:
            return None, (f"there is no weight profile {args['id']!r}; they are "
                          f"{', '.join(p['id'] for p in m['profiles'])}")
        return args, None
    if name == "go_to_step":
        n = args["n"]
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < len(steps):
            return None, f"n is a step number from 0 to {len(steps) - 1}"
        return args, None
    if name == "explain":
        names = ({f["field"] for s in steps for f in s["fields"]} | {s["id"] for s in steps}
                 | {c["id"] for c in m["components"]} | {p["id"] for p in m["profiles"]})
        if args["field"] not in names:
            return None, f"there is nothing called {args['field']!r} on the page"
        return args, None
    return args, None                                   # next, previous
