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

from server.conductor.states import KINDS as STATE_KINDS

MODES3 = ("off", "shadow", "on")            # a routing decision type
TASK_MODES = ("off", "shadow", "act")       # a session-monitor task (SRV-109 wording)
PROFILE_MODES = ("raw", "shadow", "on")     # the profile's master switch
EFFORTS = ("off", "low", "medium", "high", "xhigh")
SIZES = ("small", "medium", "large")          # SRV-157: a task's size, from structural signals
SIZE_SIGNALS = ("session_tokens", "files", "checks", "turns")
# a profile without `thinking.by_size` (every version stored before SRV-157) journals the size and
# the effort it would pick, and changes nothing
BY_SIZE_DEFAULT = "shadow"
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

# The in-context stage's question: a hidden user turn after the conversation, answered with one
# letter (a single token each in Qwen's vocabulary; the router has at most 20 classes), so one
# prefill's last row is the answer's whole distribution. `{classes}` is one `A: id - description`
# line per class
CONTEXT_QUESTION = ("The runtime that serves you asks one question before you answer, and the user "
                    "does not see it. What kind of work does the user's latest message in this "
                    "conversation ask for? Answer with the letter of the one class that fits it "
                    "best, and nothing else.\n\nClasses:\n{classes}")
CONTEXT_STAGE: dict = {"mode": "off", "min_tokens": 8192, "followups": True, "accept": 0.55,
                       "question": CONTEXT_QUESTION}
# ENG-181: a linear probe on Qwen's own rows (server/probe.py, trained by tools/probe_train.py).
# `file`: a probe under ~/.qwen38-spark-engine/probe/probes/ (or a full path). shadow: the answer
# of the turn's own snapshot is journaled beside the router's (extra.probe); on: where the plan
# decides under the engine lock (the in-context stage), the probe answers first and the question
# is asked only below `accept`. Off by default
PROBE_STAGE: dict = {"mode": "off", "file": "", "accept": 0.9}

ROUTER: dict = {
    "classes": [
        {"id": "code_edit", "label": "Code edit",
         "description": "Write, change, refactor or extend code or files in a software project.",
         # SRV-157: eval round 1 (2026-10-01), minicalc took 7.4 against 6.5 minutes at xhigh
         "policy": {"effort": "xhigh", "packages": [],
                    "effort_by_size": {"small": "high", "medium": "xhigh", "large": "xhigh"},
                    "rules": [
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
        # SRV-153: one line of what the router knows (previous class, last tools, file types,
        # checker status, compaction) at the digest's end, out of the tail's share. Off: with
        # Laya alone it failed the ship rule on 2026-10-01 (box CPU, 216 short prompts 0.500 ->
        # 0.324 at 0.8; follow-ups that change the task right 10 -> 1 of 12, the line pulls it to
        # the previous class); the Qwen stage is not measured yet
        "context_line": False,
        "package_question": "Which of these instructions fits this request best?",
        "package_accept": 0.6,
        "qwen_prompt": ("You sort requests to a coding assistant into task classes. Read the "
                        "request and answer with the id of the one class that fits it best.\n\n"
                        "Classes:\n{classes}"),
        # Qwen decides inside the conversation it already holds (2026-10-02): a long prompt or a
        # follow-up is classified by the engine itself on the prompt as the client sent it (the
        # resident prefix and its anchors hit, only the tail and this question are prefilled),
        # thinking off, one letter. Laya keeps the short first messages. Off by default
        "context_stage": dict(CONTEXT_STAGE),
        "probe_stage": dict(PROBE_STAGE),
    },
    "structural": {"compaction": "summarize", "tool_turn": "keep"},
    "stickiness": {"switch": 0.8},
    "effort": {
        "levels": {
            "off": {"thinking": False, "budget": 0, "note": ""},
            "low": {"thinking": True, "budget": 1024, "note": LOW_NOTE},
            # real caps (2026-10-02): before them medium, high and xhigh differed by their note
            # alone. The cap closes the reasoning with the monitor's hidden close (SP1); 0 is no
            # cap below the backstop (D2). A profile's `thinking.budgets` overrides a level
            "medium": {"thinking": True, "budget": 8192, "note": ""},
            "high": {"thinking": True, "budget": 16384, "note": HIGH_NOTE},
            "xhigh": {"thinking": True, "budget": 0, "note": XHIGH_NOTE},
        },
        "raise_on_fail": True,
        # SRV-157: the size signals and their thresholds (a value below `small` is small, up to
        # `medium` medium, above it large). First guesses from eval round 1 (2026-10-01):
        # minicalc and csvq added 25k to 49k prompt tokens over 7 to 16 turns, spiral 54 to 90
        # turns, photo3d 144 to 192
        "size": {
            "signals": {
                "session_tokens": {"enabled": True, "small": 20000, "medium": 60000},
                "files": {"enabled": True, "small": 3, "medium": 8},
                "checks": {"enabled": True, "small": 4, "medium": 8},
                "turns": {"enabled": True, "small": 30, "medium": 100},
            },
            "reset_on_compaction": True,
        },
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

# SRV-133, the role contract (Harmony 1): what every role declares beside its endpoint. `kind` is
# what a task asks for, never a model; `window` when it may use the Spark's GPU (idle windows, in
# decode beside Qwen, or any: not on the GPU); `preemptible` whether a Qwen request cuts it;
# `latency_budget_ms` how long its caller waits before `fallback` applies; `writes` what it may
# write (only the reasoner writes tokens the client sees, the honesty rule); `tokenizer_sha256`
# the tokenizer it reads (the engine records its own, ENG-129), which decides the forms of sharing
# it may declare (Harmony 2): `text` always, `tokens` (ids, digest chunks, the stores) only with
# the engine's own hash, `states` (Qwen's hidden rows) only for a probe inside the engine.
# P1 (2026-10-02): a `generative` role is a small model inside the engine that writes a few tokens
# of text (the narrator); its window `between_rounds` is the request's own thread between two of
# Qwen's rounds, Qwen waiting, for at most its latency budget; it writes `state_lines` (the
# framed lines of states.py in the reasoning, never content, never Qwen's prompt); its contract
# `generate` is the in-process call (engine/generative/slot.py).
ROLE_KINDS = ("reasoner", "decider", "probe", "clerk", "librarian", "tool", "generative")
WINDOWS = ("idle", "in_decode", "any", "between_rounds")
FALLBACKS = ("PASS", "stale", "skip", "next-role")
WRITES = ("client_visible", "hidden_notes", "memory_files", "nothing", "state_lines")
SHARING = ("text", "tokens", "states")
CONTRACTS = ("jev", "openai-chat", "rerank", "generate")
# the longest a role may hold Qwen between two rounds, per call
BETWEEN_ROUNDS_MAX_MS = 1000
TOKENIZER_NONE = "none"
# what a roles document stored before SRV-133 gets for the fields it lacks (journaled at load)
ROLE_DEFAULTS = {"window": "idle", "preemptible": True, "fallback": "PASS", "writes": "nothing",
                 "tokenizer_sha256": TOKENIZER_NONE, "sharing": ["text"]}
# the kind of a role the documents before SRV-133 named, else the kind its contract implies
ROLE_KIND_OF = {"decider": "decider", "classifier": "reasoner", "judge": "reasoner",
                "librarian": "librarian", "reader": "clerk", "narrator": "generative"}
CONTRACT_KIND = {"jev": "decider", "rerank": "librarian", "openai-chat": "clerk",
                 "generate": "generative"}


def _role(endpoint, model, contract, timeout_ms, mode, kind, window, preemptible=True,
          writes="nothing") -> dict:
    return {"endpoint": endpoint, "model": model, "contract": contract, "timeout_ms": timeout_ms,
            "memory_claim_gib": 0.0, "mode": mode, "kind": kind, "window": window,
            "preemptible": preemptible, "latency_budget_ms": timeout_ms, "fallback": "PASS",
            "tokenizer_sha256": TOKENIZER_NONE, "writes": writes, "sharing": ["text"]}


# `self` for a `jev` role: the decision models this engine serves (engine/decision, asked in
# process, SRV-131); the arbiter sidecar on :8010 is gone (OPS-45 re-scoped, 2026-10-01). Its
# memory claim is the residency document's. The classifier and the judge are Qwen itself through
# this server's port: their requests take the engine lock like any other, so their window is any.
ROLES: dict = {
    "decider": _role("self", "auto", "jev", 400, "on", "decider", "idle"),
    "classifier": _role("self", "qwen38-spark-engine", "openai-chat", 8000, "on", "reasoner",
                        "any", preemptible=False),
    "librarian": _role("", "", "rerank", 800, "off", "librarian", "any"),
    "reader": _role("", "", "openai-chat", 30000, "off", "clerk", "idle"),
    "judge": _role("self", "qwen38-spark-engine", "openai-chat", 30000, "off", "reasoner",
                   "any", preemptible=False),
    # P1: MiniCPM5-2B inside the engine writes the state lines (server/conductor/narrator.py);
    # a profile's `ui.narrator` switches it on, this role's mode and budget say how
    "narrator": _role("self", "minicpm5-2b", "generate", 150, "on", "generative",
                      "between_rounds", writes="state_lines"),
}

# OPS-41: what may load beside the engine. A role loads when MemAvailable minus its claim stays at
# or above floor + margin (15 + 6 GiB: the 2026-09-11 wedge and the decider design note), one at a
# time in `order`, never during a hold. The decider's claim is the MemAvailable it took beside the
# engine (31.0 -> 22.8 GiB, 2026-10-01), not the 5.1 GB nvidia-smi shows; a role that grows
# (arbiter's batcher at 64 rows) claims its peak.
RESIDENCY: dict = {
    "floor_gib": 15.0, "margin_gib": 6.0, "settle_s": 5,
    "order": ["decider", "narrator", "librarian", "reader"],
    "claims": {
        "decider": {"gib": 8.2, "basis": "measured",
                    "note": "arbiter with Laya's three checkpoints beside the engine: MemAvailable "
                            "31.0 -> 22.8 GiB (2026-10-01); nvidia-smi shows 5.1 GB"},
        "narrator": {"gib": 6.0, "basis": "declared",
                     "note": "MiniCPM5-2B in BF16 (5.03 GB of weights), its 1,024-row cache, a "
                             "step's workspace and the pool's slack; loads only when a profile "
                             "with ui.narrator on asks for a line"},
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

# SRV-137: the session transcripts on the NVMe (server/conductor/transcripts.py): `tools` writes
# a block for a request with tools (and a compaction), counters for the rest (Khaled's decision 2);
# `all` writes every request, `none` nothing. Tool results and call arguments are clipped at
# `clip_chars`, a loop the D1 guard cut keeps `loop_chars` of its start and end. The secret-pattern
# refusal has no switch: it always runs
TRANSCRIPT_SCOPES = ("tools", "all", "none")
SETTINGS: dict = {"default_profile": "chat",
                  "journal": {"retention_days": 30, "digest": "tools"},
                  "memory": {"transcripts": {"scope": "tools", "retention_days": 30,
                                             "clip_chars": 4000, "loop_chars": 200}}}


def _profile(desc, mode, clients, default_class, decisions, act=(), shadow=(), caps=(2, 6),
             notify=False, lock=None, packs=False) -> dict:
    return {"description": desc, "mode": mode, "clients": list(clients),
            "default_class": default_class,
            "decisions": dict(zip(DECISIONS, decisions)),
            "routing": {},
            "thinking": {"default": "medium", "hints": True, "client_override": True,
                         "by_size": BY_SIZE_DEFAULT},
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
            # TandemLLM's steps as lines in the reasoning the client shows (states.py); off
            "ui": {"states": "off"},
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
    schema has a branch per rule), every other kind by `validate`; a roles document's roles by
    the contract's rules too."""
    if kind == "monitor":
        from server.conductor.monitor import policy as mpolicy
        try:
            mpolicy.Policy(doc)
        except (mpolicy.PolicyError, KeyError, TypeError, AttributeError) as exc:
            return [str(exc)]
        return []
    errs = validate(doc, SCHEMAS[kind])
    if kind == "roles" and not errs:
        for name, r in doc.items():
            errs += role_rules(r, f"$.{name}")
    return errs


def role_rules(r: dict, path: str = "$", engine_tokenizer: str | None | bool = False) -> list[str]:
    """The contract's rules a schema cannot say, for one role whose fields are valid.
    `engine_tokenizer`: the hash the engine recorded (None: it recorded none), checked against a
    role that shares token ids; False skips that check (a stored document loads without it)."""
    errs = []
    kind, sharing = r["kind"], r["sharing"]
    if r["writes"] == "client_visible" and kind != "reasoner":
        errs.append(f"{path}.writes: only the reasoner writes tokens the client sees (the honesty "
                    f"rule); a {kind} may write hidden_notes, memory_files or nothing")
    if r["window"] == "in_decode" and kind not in ("decider", "probe"):
        errs.append(f"{path}.window: in_decode is for a decider or a probe only; during decode the "
                    f"GPU belongs to Qwen (a {kind} runs idle or any)")
    if r["window"] == "between_rounds":
        if kind != "generative":
            errs.append(f"{path}.window: between_rounds is for a generative role only (a small "
                        f"model inside the engine that Qwen waits for)")
        elif r["latency_budget_ms"] > BETWEEN_ROUNDS_MAX_MS:
            errs.append(f"{path}.latency_budget_ms: a role between Qwen's rounds holds Qwen at most "
                        f"{BETWEEN_ROUNDS_MAX_MS} ms a call")
    if r["writes"] == "state_lines" and kind != "generative":
        errs.append(f"{path}.writes: state_lines are written by a generative role only")
    if len(set(sharing)) != len(sharing):
        errs.append(f"{path}.sharing: a form is listed twice")
    if "states" in sharing and kind != "probe":
        errs.append(f"{path}.sharing: states (Qwen's own hidden rows) is for a probe inside the "
                    f"engine only")
    if "tokens" in sharing:
        sha = r["tokenizer_sha256"]
        if sha == TOKENIZER_NONE:
            errs.append(f"{path}.tokenizer_sha256: sharing tokens needs the role's tokenizer hash "
                        f"(equal to the engine's); none shares text only")
        elif engine_tokenizer is None:
            errs.append(f"{path}.sharing: the engine recorded no tokenizer hash (ENG-129), so "
                        f"token sharing cannot be checked; share text")
        elif engine_tokenizer is not False and sha != engine_tokenizer:
            errs.append(f"{path}.tokenizer_sha256: {sha[:12]}… is not the engine's tokenizer "
                        f"({engine_tokenizer[:12]}…); a different tokenizer shares text only")
    return errs


def role_errors(fields: dict, path: str = "$", engine_tokenizer: str | None | bool = False
                ) -> list[str]:
    """One role's contract (the ROLE_CONTRACT fields, every one required) and its rules: what
    the decision runtime's registry checks each model it serves against (SRV-133)."""
    errs = validate(fields, _obj(ROLE_CONTRACT, extra=True), path)
    return errs or role_rules(fields, path, engine_tokenizer)


def sharing_forms(r: dict, engine_tokenizer: str | None) -> list[str]:
    """The forms of sharing a role may use, as its tokenizer hash decides them (Harmony 2):
    text always; token ids when its hash is the engine's; Qwen's states for a probe."""
    out = ["text"]
    if engine_tokenizer and r.get("tokenizer_sha256") == engine_tokenizer:
        out.append("tokens")
    if r.get("kind") == "probe":
        out.append("states")
    return out


def upgrade(kind: str, doc) -> tuple[dict, list[str]]:
    """A document stored before a field existed, with the field's default, and what was filled
    (one `role.field=value` each). Only roles have such fields (SRV-133): `memory_gb` becomes
    `memory_claim_gib`, the latency budget is the role's timeout, the kind follows the role's name
    or its contract, the rest are ROLE_DEFAULTS. Anything else comes back as it is."""
    if kind != "roles" or not isinstance(doc, dict):
        return doc, []
    out, applied = {}, []
    for name, r in doc.items():
        if not isinstance(r, dict):
            out[name] = r
            continue
        r = dict(r)
        if "memory_claim_gib" not in r and "memory_gb" in r:
            r["memory_claim_gib"] = r.pop("memory_gb")
            applied.append(f"{name}.memory_claim_gib={r['memory_claim_gib']} (memory_gb)")
        if "kind" not in r:
            r["kind"] = ROLE_KIND_OF.get(name) or CONTRACT_KIND.get(r.get("contract"), "clerk")
            applied.append(f"{name}.kind={r['kind']}")
        if "latency_budget_ms" not in r and isinstance(r.get("timeout_ms"), int):
            r["latency_budget_ms"] = r["timeout_ms"]
            applied.append(f"{name}.latency_budget_ms={r['timeout_ms']} (timeout_ms)")
        for k, v in ROLE_DEFAULTS.items():
            if k not in r:
                r[k] = copy.deepcopy(v)
                applied.append(f"{name}.{k}={v if not isinstance(v, list) else ','.join(v)}")
        out[name] = r
    if "narrator" not in out:
        # P1: a roles document stored before the narrator existed gets the built-in one
        out["narrator"] = copy.deepcopy(ROLES["narrator"])
        applied.append("narrator=built-in")
    return out, applied


def context_stage(router: dict) -> dict:
    """The router's in-context stage with its defaults: a router stored before it existed has
    none, and its stage is off."""
    return {**CONTEXT_STAGE, **((router.get("classifier") or {}).get("context_stage") or {})}


def probe_stage(router: dict) -> dict:
    """The router's probe stage with its defaults (off for a router stored before it)."""
    return {**PROBE_STAGE, **((router.get("classifier") or {}).get("probe_stage") or {})}


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
# reasoning tokens, 0 = no cap below the backstop; at most the context (256k, a hard rule)
_BUDGET = {"type": "integer", "minimum": 0, "maximum": 262144}

POLICY = _obj({"effort": {"enum": list(EFFORTS)},
               "effort_by_size": _obj({z: {"enum": list(EFFORTS)} for z in SIZES}, required=()),
               "packages": {"type": "array", "items": _ID, "maxItems": 8},
               "rules": {"type": "array", "items": {"type": "string", "maxLength": 2000},
                         "maxItems": 16},
               "note": {"type": "string", "maxLength": 4000}},
              required=())

# SRV-133: the contract fields, in the roles document and in every decision model the engine
# serves (engine/decision/base.py, checked by the registry with `role_errors`)
ROLE_CONTRACT = {
    "kind": {"enum": list(ROLE_KINDS)},
    "memory_claim_gib": {"type": "number", "minimum": 0, "maximum": 128},
    "window": {"enum": list(WINDOWS)},
    "preemptible": {"type": "boolean"},
    "latency_budget_ms": {"type": "integer", "minimum": 1, "maximum": 600000},
    "fallback": {"enum": list(FALLBACKS)},
    "tokenizer_sha256": {"type": "string", "pattern": r"^(none|[0-9a-f]{64})$"},
    "writes": {"enum": list(WRITES)},
    "sharing": {"type": "array", "items": {"enum": list(SHARING)}, "maxItems": len(SHARING)},
}

SCHEMAS: dict[str, dict] = {
    "settings": _obj({"default_profile": _ID,
                      "journal": _obj({"retention_days": {"type": "integer", "minimum": 1,
                                                          "maximum": 400},
                                       "digest": {"enum": ["tools", "all", "none"]}}),
                      "memory": _obj({"transcripts": _obj({
                          "scope": {"enum": list(TRANSCRIPT_SCOPES)},
                          "retention_days": {"type": "integer", "minimum": 1, "maximum": 400},
                          "clip_chars": {"type": "integer", "minimum": 200, "maximum": 100000},
                          "loop_chars": {"type": "integer", "minimum": 0, "maximum": 4000}})},
                          required=())},
                     required=("default_profile", "journal")),
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
            "context_line": {"type": "boolean"},
            "package_question": {"type": "string", "minLength": 4, "maxLength": 400},
            "package_accept": _PROB,
            "qwen_prompt": {"type": "string", "minLength": 10, "maxLength": 4000},
            "context_stage": _obj({
                "mode": {"enum": ["off", "on"]},
                "min_tokens": {"type": "integer", "minimum": 0, "maximum": 262144},
                "followups": {"type": "boolean"},
                "accept": _PROB,
                "question": {"type": "string", "minLength": 10, "maxLength": 4000}},
                required=("mode",)),
            "probe_stage": _obj({
                "mode": {"enum": ["off", "shadow", "on"]},
                "file": {"type": "string", "maxLength": 400},
                "accept": _PROB},
                required=("mode",))},
            required=("stages", "question", "digest_chars", "package_question", "package_accept",
                      "qwen_prompt")),
        "structural": _obj({"compaction": {"type": "string", "maxLength": 40},
                            "tool_turn": {"enum": ["keep", "classify"]}}),
        "stickiness": _obj({"switch": _PROB}),
        "effort": _obj({
            "levels": _obj({lv: _obj({"thinking": {"type": "boolean"},
                                      "budget": _BUDGET,
                                      "note": {"type": "string", "maxLength": 4000}})
                            for lv in EFFORTS}),
            "raise_on_fail": {"type": "boolean"},
            "size": _obj({
                "signals": _obj({n: _obj({"enabled": {"type": "boolean"},
                                          "small": {"type": "number", "minimum": 0},
                                          "medium": {"type": "number", "minimum": 0}})
                                 for n in SIZE_SIGNALS}, required=()),
                "reset_on_compaction": {"type": "boolean"}})},
            required=("levels", "raise_on_fail")),
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
        "contract": {"enum": list(CONTRACTS)}, "timeout_ms": _MS,
        "mode": {"enum": list(MODES3)}, **ROLE_CONTRACT})},
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
                          "client_override": {"const": True},
                          "by_size": {"enum": list(MODES3)},
                          "budgets": _obj({lv: _BUDGET for lv in EFFORTS if lv != "off"},
                                          required=())},
                         required=("default", "hints", "client_override")),
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
        "ui": _obj({"states": {"enum": ["off", "on"]},
                    "kinds": _obj({k: {"type": "boolean"} for k in STATE_KINDS}, required=()),
                    # P1: the narrator writes the lines (absent: off)
                    "narrator": {"enum": ["off", "on"]}},
                   required=("states",)),
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
          roles_up: dict | None = None, engine_tokenizer: str | None = None) -> list[str]:
    """The schema of `kind` and the cross-document rules. `router`, `packages`, `profiles`:
    the current documents a reference is checked against; `roles_up`: role -> bool, whether
    the role answers now (None skips the offline rule); `engine_tokenizer`: the tokenizer hash
    the engine recorded, which a role that shares token ids must equal (None: it recorded none)."""
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
        cs = doc["classifier"].get("context_stage") or {}
        if "question" in cs and "{classes}" not in cs["question"]:
            errs.append("$.classifier.context_stage.question: must contain {classes}")
        ps = doc["classifier"].get("probe_stage") or {}
        if ps.get("mode", "off") != "off" and not ps.get("file"):
            errs.append("$.classifier.probe_stage.file: a probe file is needed when the mode is "
                        "not off")
        for n, c in ((doc["effort"].get("size") or {}).get("signals") or {}).items():
            if c["medium"] < c["small"]:
                errs.append(f"$.effort.size.signals.{n}: medium ({c['medium']:g}) is below small "
                            f"({c['small']:g})")
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
    if kind == "roles":
        for rname, r in doc.items():
            if "tokens" in r["sharing"]:
                errs += role_rules(r, f"$.{rname}", engine_tokenizer)
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
            "sizes": list(SIZES), "size_signals": list(SIZE_SIGNALS),
            "decisions": list(DECISIONS), "client_kinds": list(CLIENT_KINDS),
            "state_kinds": list(STATE_KINDS),
            "locked": dict(LOCKED),
            # SRV-133: the role contract's values, for the roles editor's selects
            "roles": {"kinds": list(ROLE_KINDS), "windows": list(WINDOWS),
                      "fallbacks": list(FALLBACKS), "writes": list(WRITES),
                      "sharing": list(SHARING), "contracts": list(CONTRACTS),
                      "defaults": copy.deepcopy(ROLE_DEFAULTS)},
            "tasks": {tid: {"label": m["label"], "family": m["family"], "tier": m["tier"],
                            "role": m["role"], "why": m["why"],
                            "thresholds": {k: {"default": v[0], "unit": v[1]}
                                           for k, v in m["thresholds"].items()}}
                      for tid, m in TASKS.items()},
            "schemas": copy.deepcopy(SCHEMAS)}
