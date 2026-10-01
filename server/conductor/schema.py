"""The conductor's configuration: what each document holds, its defaults, and its validation.

Seven kinds of document, every one versioned by `server/conductor/store.py`:

    settings   the default profile and the journal's scope and retention
    router     the task classes (id, description, default policy), the classifier stages and
               their thresholds, stickiness, the effort levels and the injection wrapper
    roles      role -> endpoint, model, timeout, memory claim, mode (a role is a kind of
               question; only this file names models)
    packages   skills and packages the router may inject: inline text or a SKILL.md on the box
    monitor    the session monitor's policy: every task with its rule and parameters (the tool
               checks' floors, project rules and reminders among them), caps, texts, templates.
               Version 0 is the server's policy file (ops/supervisor.yaml); its schema is the
               monitor's own, built from the rule registry (server/conductor/monitor/policy.py)
    profiles   one document per profile: modes, client mapping, per-class policy overrides and
               the session monitor's task settings (mode, thresholds, the tool-gate packs)
    residency  the residency planner's limits (OPS-41): the MemAvailable floor and the margin a
               role load must leave above it, the settle time after a stop, the load order, and
               each role's memory claim with its basis (measured, a declared peak, declared)
    deciders   the decision models the engine serves itself (ENG-179, SRV-131): each model's load
               policy (on_start, on_demand, off) and which model answers each question id

The schemas are the JSON Schema 2020-12 subset `validate` below understands (and `ajv` in the
dashboard's contract test): `type`, `properties`, `required`, `additionalProperties` (false or a
schema), `items`, `enum`, `const`, `minimum`, `maximum`, `minLength`, `maxLength`, `maxItems`,
`pattern`. `tools/conductor_contract.py --write` puts them in `docs/contract/conductor-v1/`, and a
test holds the files to these dicts. Standard library only, like the rest of the server.

`check(kind, doc, ...)` is the schema plus the rules a schema cannot say: a policy names classes
and packages that exist, a task set to act whose role is offline is refused, an injection wrapper
carries `{text}`.
"""

from __future__ import annotations

import copy
import re

MODES3 = ("off", "shadow", "on")            # a routing decision type
TASK_MODES = ("off", "shadow", "act")       # a session-monitor task (SRV-109 wording)
PROFILE_MODES = ("raw", "shadow", "on")     # the profile's master switch
EFFORTS = ("off", "low", "medium", "high", "xhigh")
DECISIONS = ("classify", "effort", "packages", "rules")
CLIENT_KINDS = ("open-webui", "openai-sdk", "opencode", "curl", "dashboard", "other")
LOCKED = {"raw": "raw passthrough, byte-identical to the release",
          "bench": "raw passthrough, byte-identical; benchmarks and the paper"}
ID_RE = r"^[a-z][a-z0-9_-]{0,39}$"

# --------------------------------------------------------------------------- the task catalogue
# The session monitor's tasks (TandemLLM 1, section 3). The frame (SRV-109) executes them; the
# profile only says their mode, thresholds and cap. `unit` and `why` are for the dashboard's
# editor; `role` is the role a Tier 1 task needs before it may act.
TASKS: dict[str, dict] = {
    "D1": {"label": "reasoning loop", "family": "thinking", "tier": "0+1", "role": "decider",
           "thresholds": {"repeat_share": (0.7, "share"), "grey_low": (0.4, "share"),
                          "min_tokens": (2048, "tokens"), "every_tokens": (64, "tokens")},
           "why": "3 of 3 failures caught at 2.1k to 3.0k tokens; 4 of 80 long healthy turns "
                  "closed at 0.6 to 0.7"},
    "D1b": {"label": "no close on literal runs", "family": "thinking", "tier": "0", "role": None,
            "thresholds": {"min_span": (128, "tokens")},
            "why": "51 of 52 stall closes in the database were hex or digit runs"},
    "D2": {"label": "reasoning backstop", "family": "thinking", "tier": "0", "role": None,
           "thresholds": {"backstop": (32768, "tokens"), "answer_room": (2048, "tokens")},
           "why": "one turn reasoned 65,536 tokens and wrote nothing (2026-09-30 19:05Z)"},
    "D3": {"label": "already decided", "family": "thinking", "tier": "1", "role": "decider",
           "thresholds": {"act_at": (0.9, "probability"), "every_tokens": (1024, "tokens")},
           "why": "hypothesis: long reasoning after the next call is chosen; shadow first"},
    "D4": {"label": "thinking per step", "family": "thinking", "tier": "0+1", "role": None,
           "thresholds": {}, "why": "the thinking controller (SRV-127)"},
    "E1": {"label": "empty stop with tools", "family": "end of turn", "tier": "0", "role": None,
           "thresholds": {"tries": (2, "count")},
           "why": "two empty stops ended the 2026-09-30 night (19:33Z, 19:58Z)"},
    "E2": {"label": "done without evidence", "family": "end of turn", "tier": "1",
           "role": "decider", "thresholds": {"claim_at": (0.85, "probability"),
                                             "admit_at": (0.6, "probability")},
           "why": "hypothesis: PASS claimed without a checker run; shadow first"},
    "C1": {"label": "risk guard", "family": "tool calls", "tier": "0+1", "role": "decider",
           "thresholds": {"act_at": (0.85, "probability")},
           "why": "guard floors plus the decider in the grey band"},
    "C2": {"label": "project rules", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"max_file_lines": (400, "lines")},
           "why": "AGENTS.md rules the model broke on 2026-09-30: numbered copies, /tmp paths"},
    "C3": {"label": "same call thrice", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"repeats": (3, "count")},
           "why": "one todowrite loop ran 49 times in 10 minutes (2026-09-26, SRV-45)"},
    "C4": {"label": "malformed call", "family": "tool calls", "tier": "0", "role": None,
           "thresholds": {"repairs": (2, "count")},
           "why": "opencode refused 19 calls on 2026-09-26; a cut write reached it as broken JSON"},
    "S1": {"label": "gate progress", "family": "session", "tier": "0", "role": None,
           "thresholds": {"fails": (3, "count")},
           "why": "gate (a) failed 15.6, 12.2, 3.5, 12.3, 3.5 without progress (2026-09-30)"},
    "S2": {"label": "context health", "family": "session", "tier": "0", "role": None,
           "thresholds": {}, "why": "night 1: PROGRESS.yaml last read at 02:28, five "
                                    "compactions after it"},
    "S3": {"label": "skill routing", "family": "session", "tier": "0+1", "role": None,
           "thresholds": {"act_at": (0.9, "probability")}, "why": "SRV-123"},
    "S4": {"label": "stall watchdog", "family": "session", "tier": "0", "role": None,
           "thresholds": {"minutes": (20, "minutes")},
           "why": "the run sat idle 21.2 minutes after an empty stop (2026-09-30); the "
                  "overnight driver's job"},
    "HINT": {"label": "speed hints", "family": "speed", "tier": "0", "role": None,
             "thresholds": {}, "why": "not gated: no SPD measurement yet"},
}


def _tasks(act=(), shadow=()) -> dict:
    out = {}
    for tid, meta in TASKS.items():
        mode = "act" if tid in act else "shadow" if tid in shadow else "off"
        out[tid] = {"mode": mode, "cap": 1 if tid.startswith(("E", "C")) else 2,
                    "thresholds": {k: v[0] for k, v in meta["thresholds"].items()}}
    return out


# --------------------------------------------------------------------------- seed documents
# What a box starts with before anyone writes a version (version 0, built in). The texts are
# the defaults the dashboard shows grey.

XHIGH_NOTE = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate "
              "key assumptions, consider plausible alternatives, and prioritize correctness, "
              "consistency, and clarity in the final answer.")
LOW_NOTE = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
            "to the conclusion without unnecessary elaboration.")
HIGH_NOTE = ("Reasoning effort for this step is high. Think the task through and check your plan "
             "against the code before you act.")

ROUTER: dict = {
    "classes": [
        {"id": "code_edit", "label": "Code edit",
         "description": "Write, change, refactor or extend code or files in a software project.",
         "policy": {"effort": "xhigh", "packages": [], "rules": [
             "Read a file before you change it, and run the project's check or tests after each "
             "change."]}},
        {"id": "debug", "label": "Debug",
         "description": "Find and fix a bug, a failing test, an error message or a crash.",
         "policy": {"effort": "xhigh", "packages": [], "rules": [
             "Reproduce the failure first, change one thing at a time, and quote the result line "
             "of the check."]}},
        {"id": "design_ui", "label": "UI design",
         "description": "The look of a web page or an app: layout, visual style, colours, "
                        "typography, UI and UX.",
         "policy": {"effort": "high", "packages": ["ui-design"], "rules": []}},
        {"id": "drawing", "label": "Drawing",
         "description": "Draw, paint or animate a picture or a portrait with code: a sketch, "
                        "line art, SVG or canvas art.",
         "policy": {"effort": "high",
                    "packages": ["pencil-sketch", "spiral-portrait", "one-line-portrait"],
                    "rules": []}},
        {"id": "three_d", "label": "3D scene",
         "description": "A 3D scene or model: Three.js, WebGL, voxels, a photo rebuilt in 3D.",
         "policy": {"effort": "high", "packages": ["photo-to-3d", "voxel-seasons"], "rules": []}},
        {"id": "plan", "label": "Plan",
         "description": "Plan an approach, a design or an architecture before building it, or "
                        "break the work into steps.",
         "policy": {"effort": "high", "packages": [], "rules": []}},
        {"id": "research", "label": "Research",
         "description": "Look something up, compare options, or explain how something works.",
         "policy": {"effort": "medium", "packages": [], "rules": []}},
        {"id": "summarize", "label": "Summarize",
         "description": "Summarize, shorten or pull the key points out of text that is given, or "
                        "compress a conversation.",
         "policy": {"effort": "off", "packages": [], "rules": []}},
        {"id": "chat", "label": "Chat",
         "description": "Small talk, a greeting, or a short question with a short answer.",
         "policy": {"effort": "low", "packages": [], "rules": []}},
    ],
    "classifier": {
        "stages": [
            # 0.8, not 0.6: on the labelled set (216 prompts, the Spark, 2026-10-01) the cascade
            # was right on 95.4 % at 0.8 against 93.5 % at 0.6; half the decider's confident
            # mistakes sat between 0.6 and 0.8 and Qwen got them right (tools/conductor_eval.py)
            {"kind": "decider", "enabled": True, "accept": 0.8, "timeout_ms": 400},
            {"kind": "qwen", "enabled": True, "accept": 0.55, "timeout_ms": 8000},
        ],
        "question": "What kind of work does this request ask for?",
        "digest_chars": 2000,
        "package_question": "Which of these instructions fits this request best?",
        "package_accept": 0.6,
        "qwen_prompt": ("You sort requests to a coding assistant into task classes. Read the "
                        "request and answer with the id of the one class that fits it best.\n\n"
                        "Classes:\n{classes}"),
    },
    "structural": {"compaction": "summarize", "tool_turn": "keep"},
    "stickiness": {"switch": 0.8},
    "effort": {
        "levels": {
            "off": {"thinking": False, "budget": 0, "note": ""},
            "low": {"thinking": True, "budget": 1024, "note": LOW_NOTE},
            "medium": {"thinking": True, "budget": 0, "note": ""},
            "high": {"thinking": True, "budget": 0, "note": HIGH_NOTE},
            "xhigh": {"thinking": True, "budget": 0, "note": XHIGH_NOTE},
        },
        "raise_on_fail": True,
    },
    "injection": {
        "wrapper": ("\n\n<runtime-note>\nThe runtime that serves you adds this note. The user did "
                    "not write it and does not see it. Follow it and do not mention it.\n\n{text}\n"
                    "</runtime-note>"),
        "max_note_tokens": 8000,
        "max_replay_tokens": 16000,
        "replay": True,
    },
}

UI_DESIGN = """# UI design

Build interfaces that look designed, not assembled.

1. Start from the content. Decide what the user must see first on the page, then second. One primary action per screen; make it the strongest element.
2. Use a spacing scale (4, 8, 12, 16, 24, 32, 48 px) and nothing between. Align everything to it. Group related items with less space, separate groups with more.
3. Type: one family for UI text, at most one for display. A scale such as 12, 14, 16, 20, 24, 32 px. Body text 14 to 16 px, line height 1.4 to 1.6, lines at most 75 characters.
4. Colour: define tokens first (background, surface, border, text, muted text, one accent, success, warning, danger). Text contrast at least 4.5:1. Use the accent for the primary action and the current state only.
5. Layout: CSS grid or flexbox, no absolute positioning for layout. Mobile first: one column at 390 px with no horizontal scroll, then widen at breakpoints. Tap targets at least 44 px.
6. States: every component needs hover, focus-visible, active, disabled, loading, empty and error states. Never leave a blank area while data loads.
7. Motion: 150 to 250 ms, ease-out, only to explain a change. Respect prefers-reduced-motion.
8. Accessibility: semantic HTML first, labels for every input, visible focus, keyboard reachable controls, alt text.
9. Check your work: open the page at 390 px and at 1440 px, compare the result with the brief, and fix alignment and spacing before adding features.
"""

PACKAGES: dict = {
    "skill_dirs": ["~/.config/opencode/skills", "~/.qwen38-spark-engine/conductor/skills"],
    "packages": {
        "ui-design": {"title": "UI design", "kind": "skill", "source": "inline",
                      "description": "Rules for layout, spacing, type, colour, states and "
                                     "responsive checks when building a web or app interface.",
                      "body": UI_DESIGN, "max_tokens": 3000, "enabled": True},
        "pencil-sketch": {"title": "Pencil sketch", "kind": "skill", "source": "skill:pencil-sketch",
                          "description": "Turn a reference image into an animated graphite pencil "
                                         "sketch on a canvas: tone, contours, hatching.",
                          "body": "", "max_tokens": 4000, "enabled": True},
        "spiral-portrait": {"title": "Spiral portrait", "kind": "skill",
                            "source": "skill:spiral-portrait",
                            "description": "A portrait drawn as one spiral line whose width follows "
                                           "the tone of a reference image.",
                            "body": "", "max_tokens": 4000, "enabled": True},
        "one-line-portrait": {"title": "One-line portrait", "kind": "skill",
                              "source": "skill:one-line-portrait",
                              "description": "A portrait drawn as one continuous line through a "
                                             "stipple of the image (TSP art).",
                              "body": "", "max_tokens": 4000, "enabled": True},
        "photo-to-3d": {"title": "Photo to 3D", "kind": "skill", "source": "skill:photo-to-3d",
                        "description": "Rebuild one photo of a real place as a Three.js scene "
                                       "that matches the photo from a hero camera.",
                        "body": "", "max_tokens": 4000, "enabled": True},
        "voxel-seasons": {"title": "Voxel seasons", "kind": "skill", "source": "skill:voxel-seasons",
                          "description": "Give a voxel island four seasons: palette, snow, "
                                         "particles, weather, sound, timelapse camera.",
                          "body": "", "max_tokens": 4000, "enabled": True},
    },
}

# `self` for a `jev` role: the decision models this engine serves (engine/decision, asked in
# process, SRV-131); the arbiter sidecar on :8010 is gone (OPS-45 re-scoped, 2026-10-01). Its
# memory claim is the residency document's.
ROLES: dict = {
    "decider": {"endpoint": "self", "model": "auto", "contract": "jev",
                "timeout_ms": 400, "memory_gb": 0.0, "mode": "on"},
    "classifier": {"endpoint": "self", "model": "qwen38-spark-engine", "contract": "openai-chat",
                   "timeout_ms": 8000, "memory_gb": 0.0, "mode": "on"},
    "librarian": {"endpoint": "", "model": "", "contract": "rerank", "timeout_ms": 800,
                  "memory_gb": 0.0, "mode": "off"},
    "reader": {"endpoint": "", "model": "", "contract": "openai-chat", "timeout_ms": 30000,
               "memory_gb": 0.0, "mode": "off"},
    "judge": {"endpoint": "self", "model": "qwen38-spark-engine", "contract": "openai-chat",
              "timeout_ms": 30000, "memory_gb": 0.0, "mode": "off"},
}

# OPS-41: what may load beside the engine. A role loads when MemAvailable minus its claim stays at
# or above floor + margin (15 + 6 GiB: the 2026-09-11 wedge and the decider design note), one at a
# time in `order`, never during a hold. The decider's claim is the MemAvailable it took beside the
# engine (31.0 -> 22.8 GiB, 2026-10-01), not the 5.1 GB nvidia-smi shows; a role that grows
# (arbiter's batcher at 64 rows) claims its peak.
RESIDENCY: dict = {
    "floor_gib": 15.0, "margin_gib": 6.0, "settle_s": 5,
    "order": ["decider", "librarian", "reader"],
    "claims": {
        "decider": {"gib": 8.2, "basis": "measured",
                    "note": "arbiter with Laya's three checkpoints beside the engine: MemAvailable "
                            "31.0 -> 22.8 GiB (2026-10-01); nvidia-smi shows 5.1 GB"},
        "reader": {"gib": 12.0, "basis": "peak",
                   "note": "a Nemotron Nano 9B in FP8, declared peak 10 to 12 GB; not resident "
                           "until RND-21 earns it"},
    },
}
BASES = ("measured", "peak", "declared")

# The decision models the engine serves (stage A of the decision-model design): which load when
# the engine starts (`on_start`, after /health and the page-cache drop, through the residency
# planner), which load when first asked (`on_demand`), which never (`off`); and which model
# answers each question id the conductor asks (`auto`: the decider role's model, by default the
# family's router). English is the only
# one on start: every conductor question is English. A model `QWEN38_DECIDER_MODELS` does not
# serve is ignored here.
DECIDER_LOADS = ("on_start", "on_demand", "off")
DECIDERS: dict = {
    "models": [{"name": "laya-english", "load": "on_start"},
               {"name": "laya-typed-decisions", "load": "on_demand"},
               {"name": "laya-multilingual", "load": "on_demand"}],
    "questions": {"classify": "auto", "packages": "auto"},
}

SETTINGS: dict = {"default_profile": "chat",
                  "journal": {"retention_days": 30, "digest": "tools"}}


def _profile(desc, mode, clients, default_class, decisions, act=(), shadow=(), caps=(2, 6),
             notify=False, lock=None, packs=False) -> dict:
    return {"description": desc, "mode": mode, "clients": list(clients),
            "default_class": default_class,
            "decisions": dict(zip(DECISIONS, decisions)),
            "routing": {},
            "thinking": {"default": "medium", "hints": True, "client_override": True},
            "tasks": _tasks(act, shadow),
            "caps": {"acts_per_request": caps[0], "acts_per_session_hour": caps[1]},
            # the tool-gate packs: off, a pack's task (C1 risk, C2 project, C3 same_call,
            # C4 malformed) does not run whatever its mode says
            "toolgate": {"risk": packs, "project": packs, "same_call": packs,
                         "malformed": packs, "tier1_timeout_ms": 500},
            # S1's fails, S4's minutes and NOTIFY (hook.Conductor.CONTEXT); NOTIFY is on only
            # in the overnight profile (Khaled, 2026-09-30: allowed, off by default)
            "context": {"s1_runs": 3, "s4_minutes": 20, "notify": notify},
            "hints": {"stair_width": False, "prefix_retention": False},
            **({"locked": lock} if lock else {})}


PROFILES: dict[str, dict] = {
    "raw": _profile("The raw engine: no conductor, byte-identical to the release.", "raw", [],
                    "chat", ("off",) * 4, lock=LOCKED["raw"]),
    "bench": _profile("Benchmarks and the paper: the raw engine, locked.", "raw", [], "chat",
                      ("off",) * 4, lock=LOCKED["bench"]),
    "chat": _profile("Open WebUI and the Playground: class-aware effort, no tool gate.", "on",
                     ("open-webui", "dashboard"), "chat", ("on", "on", "on", "off"),
                     act=("D1", "D1b", "D2")),
    "coding": _profile("opencode, interactive: every routing decision on, Tier 0 acts, Tier 1 "
                       "in shadow.", "on", ("opencode",), "code_edit", ("on",) * 4,
                       act=("D1", "D1b", "D2", "E1", "C3", "C4"),
                       shadow=("D3", "E2", "C1", "C2", "S3"), packs=True),
    "overnight": _profile("Unattended opencode runs: coding plus the session tasks, raised caps.",
                          "on", (), "code_edit", ("on",) * 4,
                          act=("D1", "D1b", "D2", "E1", "C3", "C4", "S1", "S2", "S4"),
                          shadow=("D3", "E2", "C1", "C2", "S3"), caps=(3, 12), packs=True,
                          notify=True),
}


def monitor_seed(path: str | None = None) -> dict:
    """The monitor's policy file (default ops/supervisor.yaml) as a document."""
    from server.conductor.monitor import policy as mpolicy
    return copy.deepcopy(mpolicy.load(path).data)


def doc_errors(kind: str, doc) -> list[str]:
    """The schema violations of a document: the monitor's policy by its own validator (its
    schema has a branch per rule), every other kind by `validate`."""
    if kind == "monitor":
        from server.conductor.monitor import policy as mpolicy
        try:
            mpolicy.Policy(doc)
        except (mpolicy.PolicyError, KeyError, TypeError, AttributeError) as exc:
            return [str(exc)]
        return []
    return validate(doc, SCHEMAS[kind])


def seed(kind: str, name: str | None = None) -> dict | None:
    """The built-in version 0 of a document, a fresh copy."""
    if kind == "monitor":
        return monitor_seed()
    src = {"settings": SETTINGS, "router": ROUTER, "roles": ROLES, "packages": PACKAGES,
           "residency": RESIDENCY, "deciders": DECIDERS}.get(kind)
    if kind == "profiles":
        src = PROFILES.get(name or "")
    return copy.deepcopy(src) if src is not None else None


# --------------------------------------------------------------------------- JSON Schemas
def _obj(props: dict, required=None, extra=False) -> dict:
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else list(required),
            "additionalProperties": extra}


_STR = {"type": "string", "maxLength": 20000}
_TEXT = {"type": "string", "maxLength": 200000}
_ID = {"type": "string", "pattern": ID_RE}
_PROB = {"type": "number", "minimum": 0, "maximum": 1}
_MS = {"type": "integer", "minimum": 1, "maximum": 120000}

POLICY = _obj({"effort": {"enum": list(EFFORTS)},
               "packages": {"type": "array", "items": _ID, "maxItems": 8},
               "rules": {"type": "array", "items": {"type": "string", "maxLength": 2000},
                         "maxItems": 16},
               "note": {"type": "string", "maxLength": 4000}},
              required=())

SCHEMAS: dict[str, dict] = {
    "settings": _obj({"default_profile": _ID,
                      "journal": _obj({"retention_days": {"type": "integer", "minimum": 1,
                                                          "maximum": 400},
                                       "digest": {"enum": ["tools", "all", "none"]}})}),
    "router": _obj({
        "classes": {"type": "array", "maxItems": 20, "items": _obj({
            "id": _ID, "label": {"type": "string", "minLength": 1, "maxLength": 60},
            "description": {"type": "string", "minLength": 8, "maxLength": 600},
            "policy": POLICY})},
        "classifier": _obj({
            "stages": {"type": "array", "maxItems": 4, "items": _obj({
                "kind": {"enum": ["decider", "qwen"]}, "enabled": {"type": "boolean"},
                "accept": _PROB, "timeout_ms": _MS})},
            "question": {"type": "string", "minLength": 4, "maxLength": 400},
            "digest_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
            "package_question": {"type": "string", "minLength": 4, "maxLength": 400},
            "package_accept": _PROB,
            "qwen_prompt": {"type": "string", "minLength": 10, "maxLength": 4000}}),
        "structural": _obj({"compaction": {"type": "string", "maxLength": 40},
                            "tool_turn": {"enum": ["keep", "classify"]}}),
        "stickiness": _obj({"switch": _PROB}),
        "effort": _obj({
            "levels": _obj({lv: _obj({"thinking": {"type": "boolean"},
                                      "budget": {"type": "integer", "minimum": 0,
                                                 "maximum": 262144},
                                      "note": {"type": "string", "maxLength": 4000}})
                            for lv in EFFORTS}),
            "raise_on_fail": {"type": "boolean"}}),
        "injection": _obj({"wrapper": {"type": "string", "minLength": 6, "maxLength": 4000},
                           "max_note_tokens": {"type": "integer", "minimum": 16,
                                               "maximum": 32768},
                           "max_replay_tokens": {"type": "integer", "minimum": 0,
                                                 "maximum": 65536},
                           "replay": {"type": "boolean"}}),
    }),
    "roles": {"type": "object", "additionalProperties": _obj({
        "endpoint": {"type": "string", "maxLength": 400}, "model": {"type": "string",
                                                                   "maxLength": 200},
        "contract": {"enum": ["jev", "openai-chat", "rerank"]}, "timeout_ms": _MS,
        "memory_gb": {"type": "number", "minimum": 0, "maximum": 128},
        "mode": {"enum": list(MODES3)}})},
    "packages": _obj({
        "skill_dirs": {"type": "array", "items": {"type": "string", "maxLength": 400},
                       "maxItems": 8},
        "packages": {"type": "object", "additionalProperties": _obj({
            "title": {"type": "string", "minLength": 1, "maxLength": 80},
            "kind": {"enum": ["skill", "package"]},
            "source": {"type": "string", "pattern": r"^(inline|skill:[a-z0-9][a-z0-9_.-]*|file:.+)$"},
            "description": {"type": "string", "minLength": 8, "maxLength": 600},
            "body": _TEXT, "max_tokens": {"type": "integer", "minimum": 16, "maximum": 32768},
            "enabled": {"type": "boolean"}})}}),
    "profiles": _obj({
        "description": {"type": "string", "maxLength": 400},
        "mode": {"enum": list(PROFILE_MODES)},
        "clients": {"type": "array", "items": {"enum": list(CLIENT_KINDS)}, "maxItems": 8},
        "default_class": _ID,
        "decisions": _obj({d: {"enum": list(MODES3)} for d in DECISIONS}),
        "routing": {"type": "object", "additionalProperties": POLICY},
        "thinking": _obj({"default": {"enum": list(EFFORTS)}, "hints": {"type": "boolean"},
                          "client_override": {"const": True}}),
        "tasks": _obj({tid: _obj({"mode": {"enum": list(TASK_MODES)},
                                  "cap": {"type": "integer", "minimum": 0, "maximum": 10},
                                  "thresholds": _obj({k: {"type": "number", "minimum": 0}
                                                      for k in meta["thresholds"]})})
                       for tid, meta in TASKS.items()}),
        "caps": _obj({"acts_per_request": {"type": "integer", "minimum": 0, "maximum": 10},
                      "acts_per_session_hour": {"type": "integer", "minimum": 0,
                                                "maximum": 100}}),
        "toolgate": _obj({"risk": {"type": "boolean"}, "project": {"type": "boolean"},
                          "same_call": {"type": "boolean"}, "malformed": {"type": "boolean"},
                          "tier1_timeout_ms": _MS}),
        "context": _obj({"s1_runs": {"type": "integer", "minimum": 1, "maximum": 20},
                         "s4_minutes": {"type": "integer", "minimum": 1, "maximum": 600},
                         "notify": {"type": "boolean"}}),
        "hints": _obj({"stair_width": {"type": "boolean"},
                       "prefix_retention": {"type": "boolean"}}),
        "locked": {"type": "string", "maxLength": 200},
    }, required=("description", "mode", "clients", "default_class", "decisions", "routing",
                 "thinking", "tasks", "caps", "toolgate", "context", "hints")),
}


def _monitor_schema() -> dict:
    from server.conductor.monitor import policy as mpolicy
    out = mpolicy.schema()
    out.pop("$schema", None)
    out.pop("title", None)
    return out


SCHEMAS["monitor"] = _monitor_schema()
# OPS-47: the engine's memory limits the dashboard overrides (server/memlimits.py), by their
# serve.env names; a key not set here is the flag's. MAX_LEN is not among them (Khaled's hard rule).
_GIB_LIMIT = {"type": "number", "minimum": 0, "maximum": 128}
ENGINE_LIMITS = _obj({
    "MEM_CEILING_GIB": _GIB_LIMIT, "MEM_PEAK_GIB": _GIB_LIMIT,
    "RESERVE_SLACK_GIB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_STASH_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_GUEST_MAX": {"type": "integer", "minimum": 0, "maximum": 262144},
    "RESIDENT_PIN_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "RESIDENT_PARK_GB": {"type": "number", "minimum": 0, "maximum": 32},
    "CACHE_GB": {"type": "number", "minimum": 0, "maximum": 64},
    "STORE_ENTRY_GB": {"type": "number", "minimum": 0, "maximum": 16},
    "VISION_EMBED_MB": {"type": "number", "minimum": 0, "maximum": 8192},
    "SUFFIX_STORE_MB": {"type": "number", "minimum": 0, "maximum": 4096},
    "PREFIX_CHUNK": {"type": "integer", "minimum": 64, "maximum": 8192},
    "NARROW_ARM": {"enum": [0, 1]}}, required=())
SCHEMAS["residency"] = _obj({
    "floor_gib": {"type": "number", "minimum": 0, "maximum": 128},
    "margin_gib": {"type": "number", "minimum": 0, "maximum": 64},
    "settle_s": {"type": "integer", "minimum": 0, "maximum": 120},
    "order": {"type": "array", "items": _ID, "maxItems": 16},
    "claims": {"type": "object", "additionalProperties": _obj({
        "gib": {"type": "number", "minimum": 0, "maximum": 128},
        "basis": {"enum": list(BASES)},
        "note": {"type": "string", "maxLength": 400}})},
    "engine": ENGINE_LIMITS},
    required=("floor_gib", "margin_gib", "settle_s", "order", "claims"))
SCHEMAS["deciders"] = _obj({
    "models": {"type": "array", "maxItems": 16,
               "items": _obj({"name": _ID, "load": {"enum": list(DECIDER_LOADS)}})},
    "questions": {"type": "object", "additionalProperties": _ID}})

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Violations of `schema` by `value`, one line each with its JSON path; [] when valid."""
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](value) for x in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    errs: list[str] = []
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r:.60} is not one of {', '.join(map(str, schema['enum']))}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} is above {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: {value!r:.60} does not match {schema['pattern']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}: unknown key {k!r}")
            elif isinstance(extra, dict):
                if not re.search(ID_RE, str(k)):
                    errs.append(f"{path}: key {k!r} is not an id ({ID_RE})")
                errs += validate(v, extra, f"{path}.{k}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


def check(kind: str, doc, *, name: str | None = None, router: dict | None = None,
          packages: dict | None = None, profiles: dict | None = None,
          roles_up: dict | None = None) -> list[str]:
    """The schema of `kind` and the cross-document rules. `router`, `packages`, `profiles`:
    the current documents a reference is checked against; `roles_up`: role -> bool, whether
    the role answers now (None skips the offline rule)."""
    schema = SCHEMAS.get(kind)
    if schema is None:
        return [f"$: unknown document kind {kind!r}"]
    errs = doc_errors(kind, doc)
    if errs or kind == "monitor":
        return errs
    pkg_names = set((packages or PACKAGES)["packages"])
    if kind == "router":
        ids = [c["id"] for c in doc["classes"]]
        if len(set(ids)) != len(ids):
            errs.append("$.classes: two classes share an id")
        if not ids:
            errs.append("$.classes: at least one class")
        for i, c in enumerate(doc["classes"]):
            for p in c["policy"].get("packages", []):
                if p not in pkg_names:
                    errs.append(f"$.classes[{i}].policy.packages: no package {p!r}")
        comp = doc["structural"]["compaction"]
        if comp not in ("keep", "classify") and comp not in ids:
            errs.append(f"$.structural.compaction: no class {comp!r} (or keep, classify)")
        if "{text}" not in doc["injection"]["wrapper"]:
            errs.append("$.injection.wrapper: must contain {text}")
        if "{classes}" not in doc["classifier"]["qwen_prompt"]:
            errs.append("$.classifier.qwen_prompt: must contain {classes}")
    if kind == "profiles":
        cls = {c["id"] for c in (router or ROUTER)["classes"]}
        if doc["default_class"] not in cls:
            errs.append(f"$.default_class: no class {doc['default_class']!r}")
        for cid, pol in doc["routing"].items():
            if cid not in cls:
                errs.append(f"$.routing: no class {cid!r}")
            for p in pol.get("packages", []):
                if p not in pkg_names:
                    errs.append(f"$.routing.{cid}.packages: no package {p!r}")
        if name in LOCKED:
            errs.append(f"$: profile {name!r} is locked ({LOCKED[name]})")
        if roles_up is not None:
            for tid, t in doc["tasks"].items():
                role = TASKS[tid]["role"]
                if t["mode"] == "act" and TASKS[tid]["tier"] == "1" and role and \
                        not roles_up.get(role):
                    errs.append(f"$.tasks.{tid}.mode: act needs the {role} role, which is "
                                f"offline")
        others = {n: p for n, p in (profiles or {}).items() if n != name}
        for c in doc["clients"]:
            for n, p in others.items():
                if c in p.get("clients", []):
                    errs.append(f"$.clients: {c!r} already maps to profile {n!r}")
    if kind == "settings":
        if doc["default_profile"] not in set(profiles or PROFILES) | set(PROFILES):
            errs.append(f"$.default_profile: no profile {doc['default_profile']!r}")
    if kind == "residency" and len(set(doc["order"])) != len(doc["order"]):
        errs.append("$.order: a role is listed twice")
    if kind == "residency":
        eng = doc.get("engine") or {}
        if (eng.get("MEM_CEILING_GIB") is not None and eng.get("MEM_PEAK_GIB") is not None
                and eng["MEM_PEAK_GIB"] < eng["MEM_CEILING_GIB"]):
            errs.append("$.engine: MEM_PEAK_GIB is under MEM_CEILING_GIB (the peak is the ceiling "
                        "plus a prefill's workspace)")
    if kind == "deciders":
        names = [m["name"] for m in doc["models"]]
        if len(set(names)) != len(names):
            errs.append("$.models: a model is listed twice")
    if kind == "packages":
        used = set()
        for c in (router or ROUTER)["classes"]:
            used |= set(c["policy"].get("packages", []))
        for p in sorted(used - set(doc["packages"])):
            errs.append(f"$.packages: {p!r} is used by a class and cannot be removed")
        for n, p in doc["packages"].items():
            if p["source"] == "inline" and not p["body"].strip():
                errs.append(f"$.packages.{n}.body: an inline package needs its text")
    return errs


def meta() -> dict:
    """What the dashboard's editors render from: the enums, the task catalogue with units and
    defaults, the locked profiles and the schemas themselves."""
    return {"modes": list(MODES3), "task_modes": list(TASK_MODES),
            "profile_modes": list(PROFILE_MODES), "efforts": list(EFFORTS),
            "decisions": list(DECISIONS), "client_kinds": list(CLIENT_KINDS),
            "locked": dict(LOCKED),
            "tasks": {tid: {"label": m["label"], "family": m["family"], "tier": m["tier"],
                            "role": m["role"], "why": m["why"],
                            "thresholds": {k: {"default": v[0], "unit": v[1]}
                                           for k, v in m["thresholds"].items()}}
                      for tid, m in TASKS.items()},
            "schemas": copy.deepcopy(SCHEMAS)}
