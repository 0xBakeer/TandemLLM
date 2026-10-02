"""TandemLLM's own steps, shown to the user (`ui.states`, 2026-10-02).

A client shows Qwen's reasoning and its answer. What the conductor does around them -- the class
it picked, a skill it loaded, a cap that closed the reasoning, a tool call it repaired -- was
visible only on the dashboard. With a profile's `ui.states: on` each step is one short line in the
stream, at the moment it happens. Where (`ui.states_channel`, `channel()`): `reasoning`, as every
profile stored before S1 keeps it, puts it in the request's `reasoning_format` -- a
`reasoning_content` delta, or inside the `<think>` block of `tags` (a step after the block closed
has no place in `tags` and is left out there); `content` (S1, the seed profiles' value) sends it as
a content delta of its own, outside the thinking (server/app.py; in `tags` a line said inside the
open block goes out right after `</think>`). The texts are templates (`TEXTS`), no model and no
latency; every line keeps a `summary` field for a clerk-written one later (not built).

PROGRESS (S1). Three kinds carry the numbers of a wait (`PROGRESS`, `ui.progress`, `progress()`):
the queue (`waiting: opencode's title request runs first, position 1`), a long prefill (`reading
35,189 tokens at ~750 tok/s, about 47 s left`) and a long thinking (`thinking 4 min, 9,800
tokens`). They are always the template, never the narrator's.

THE MARKER. Each line goes out framed as

    "\\n› <text>\\n\\n"

`›` (U+203A) and a space open the line, a blank line closes it: one paragraph of its own in a
client that renders markdown (opencode), one plain line in one that does not (Hermes), and the
same in both formats. The frame is self-delimiting, so `strip_text` takes it out exactly wherever
it sits: at the head or the end of the reasoning, between two parts a client joined, or between
two characters of Qwen's own text. Only a framed line whose text is one of the templates is
taken, so a line of Qwen's own that starts with `›` stays. A narrated line is free text: it closes
with a WORD JOINER (U+2060) before the blank line (narrator.frame), and the strip takes a framed
line of any text that ends in it.

QWEN NEVER SEES THEM. opencode's chat-completions provider sends every earlier assistant message's
reasoning back as `reasoning_content` (its parts joined with nothing between them), and the Hermes
agent keeps the reasoning on the message it sends back; the Qwen3.8 template renders an assistant
message's `reasoning_content` (trimmed) into the prompt. So the server strips every framed line from
the `reasoning_content` and the string or list content of the assistant messages (where the content
channel's lines come back) before anything reads them (`strip_messages`, in the handler before the
router and before the prompt is rendered): the prompt, and with it the resident prefix and the
notes' positions, are the ones the request would have without the feature. A client that trims the
reasoning loses the frame's outer newlines, so a line at the very start or end is also taken
without them; whatever whitespace is left at an end the template trims. The lines are not tokens:
they are never in `usage`, the suffix store or the response cache key; the transcript keeps them as
a block of their own and the journal row as `extra.states`.
"""

from __future__ import annotations

import re
import time

MARKER = "› "
# one line's text: no newline, at most this long
MAX_CHARS = 200
# what each kind of step says; `{...}` are the step's facts
TEXTS = {
    "route": "deciding: {cls}, {effort}",
    "compaction": "compaction seen",
    "package": "retrieving: {pkind} {name}",
    "memory": "reading memory: {n} notes",
    "steer": "steering: {label}",
    "cap": "thinking closed at the {cap} cap",
    "backstop": "thinking closed at the {cap} backstop",
    "loop": "loop detected, steering",
    "close": "closing reasoning",
    "continue": "empty answer, continuing",
    "check": "checking the tool call",
    "checks": "checking {n} tool calls",
    "repair": "repairing the tool call",
    "regenerate": "asking for the tool call again",
    "give_up": "giving up",
    # S1: progress, the numbers of a wait the client would otherwise see nothing of; templated,
    # never the narrator's (PROGRESS)
    "queue": "waiting: {reason}, position {n}",
    "prefill": "reading {n} tokens at ~{rate} tok/s, about {left} left",
    "thinking": "thinking {elapsed}, {n} tokens",
}
# the profile's per-kind switches (`ui.kinds`) name these; `check` covers `checks`
KINDS = tuple(k for k in TEXTS if k != "checks")
PROGRESS = ("queue", "prefill", "thinking")
# where the lines go (`ui.states_channel`): the reasoning, as they always did and as every profile
# stored before S1 keeps them, or the content, outside the thinking (a new profile's default)
CHANNELS = ("reasoning", "content")
# `ui.progress` when the profile says nothing: on with the lines
PROGRESS_DEFAULT = {"mode": "on", "prefill_every_s": 10, "thinking_every_s": 120,
                    "thinking_every_tokens": 4096}
# who set a cap, in words, for the narrator
CAP_BY = {"route": "the router, for the task's effort", "client": "the client's budget",
          "reserve": "the room the answer needs under max_tokens"}


def schema_label(task: str) -> str | None:
    """A monitor task's label (`D1` -> "reasoning loop"), for the narrator."""
    from server.conductor import schema
    return (schema.TASKS.get(task) or {}).get("label")

# a line the server writes: the marker and one of the templates, nothing else, so a line of
# Qwen's own that happens to start with `›` is never taken
_TEXT = "(?:" + "|".join((r"[^\n]{1,%d}" % MAX_CHARS).join(
    re.escape(part) for part in re.split(r"\{\w+\}", t)) for t in TEXTS.values()) + ")"
# a narrated line: any text on one line, closed by the narrator's sentinel (U+2060)
_TEXT = "(?:" + _TEXT + "|[^\n\u2060]{1,%d}\u2060)" % MAX_CHARS
_MID = re.compile(r"\n" + re.escape(MARKER) + _TEXT + r"\n\n")
_HEAD = re.compile(r"\A" + re.escape(MARKER) + _TEXT + r"(?:\n\n|\n?\Z)")
_TAIL = re.compile(r"\n" + re.escape(MARKER) + _TEXT + r"\n?\Z")


def channel(ui_doc: dict | None) -> str:
    """Where a profile's lines go: its `ui.states_channel`, else the reasoning (a profile stored
    before S1)."""
    return (ui_doc or {}).get("states_channel") or "reasoning"


def progress(ui_doc: dict | None) -> dict:
    """A profile's progress lines with their defaults (`ui.progress`, absent: on)."""
    return {**PROGRESS_DEFAULT, **((ui_doc or {}).get("progress") or {})}


def duration(s: float) -> str:
    """`47 s`, `4 min`: seconds below a minute and a half, whole minutes above."""
    s = max(0.0, float(s))
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


def rate(tps: float) -> str:
    """`750`, `1,230`: a rate to two or three significant figures."""
    return f"{round(float(tps), -1 if tps >= 100 else 0):,.0f}"


def frame(text: str) -> str:
    """One step as it goes out: the marker, the text on one line, a blank line after it."""
    text = " ".join(str(text).split())[:MAX_CHARS]
    return "\n" + MARKER + text + "\n\n"


def strip_text(text: str) -> str:
    """`text` without any framed line: exact wherever the frame is whole, and at the ends also
    when a client trimmed the frame's outer newlines."""
    if MARKER not in text:
        return text
    prev = None
    while prev != text:
        prev = text
        text = _MID.sub("", text)
        text = _HEAD.sub("", text, count=1)
        text = _TAIL.sub("", text, count=1)
    return text


def strip_messages(messages) -> tuple[list, int]:
    """The messages with every framed line taken out of the assistant messages' reasoning and
    content, and how many characters went. The same list when nothing changes (the common case:
    no assistant message carries the marker)."""
    if not isinstance(messages, list):
        return messages, 0
    out, removed = None, 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        new = None
        for key in ("reasoning_content", "reasoning", "content"):
            v = m.get(key)
            if isinstance(v, str) and MARKER in v:
                s = strip_text(v)
                if s != v:
                    new = dict(new or m)
                    new[key] = s
                    removed += len(v) - len(s)
            elif key == "content" and isinstance(v, list):
                parts = None
                for j, p in enumerate(v):
                    t = p.get("text") if isinstance(p, dict) else None
                    if isinstance(t, str) and MARKER in t:
                        s = strip_text(t)
                        if s != t:
                            parts = parts or list(v)
                            parts[j] = dict(p, text=s)
                            removed += len(t) - len(s)
                if parts is not None:
                    new = dict(new or m)
                    new["content"] = parts
        if new is not None:
            out = out or list(messages)
            out[i] = new
    return (out if out is not None else messages), removed


def into_full(content: str, reasoning: str | None, fmt: str, in_think: bool, ui: "States",
              where: str = "reasoning") -> tuple[str, str | None]:
    """A non-streamed answer's `(content, reasoning_content)` with the request's lines. In the
    reasoning: those said before the generation at the head of the reasoning, the rest at its end,
    in the `reasoning_content` field (formats `reasoning_content` and `both`; created when the turn
    did not think) and inside the `<think>` block of the content (`tags` and `both`, when it has
    one). In the content (`where`, S1): all of them at the head of the content, in their order."""
    if where == "content":
        return "".join(ui.shown(e) for e in ui.log) + content, reasoning
    cut = len(ui.log) if ui.head is None else ui.head
    head = "".join(ui.shown(e) for e in ui.log[:cut])
    tail = "".join(ui.shown(e) for e in ui.log[cut:])
    if fmt in ("reasoning_content", "both"):
        reasoning = head + (reasoning or "") + tail
    opener = "<think>\n"
    if fmt in ("tags", "both") and in_think and content.startswith(opener):
        end = content.find("</think>")
        end = len(content) if end < 0 else end
        content = opener + head + content[len(opener):end] + tail + content[end:]
    return content, reasoning


class States:
    """One request's steps: what is waiting to go out, and everything it said. `narrator`: the
    request's narrator (narrator.Lines), called once per line as it goes out; None: templates."""

    def __init__(self, kinds: dict | None = None, narrator=None):
        self.kinds = kinds or {}
        self.narrator = narrator
        self.pending: list[dict] = []
        self.log: list[dict] = []
        self.head: int | None = None       # how many were said before the generation started
        self.t0 = time.time()

    def on(self, kind: str) -> bool:
        base = "check" if kind == "checks" else kind
        return bool(self.kinds.get(base, True))

    def add(self, kind: str, extra: dict | None = None, **facts) -> None:
        """One step: its template filled with `facts`; `extra` are facts for the narrator only."""
        if not self.on(kind):
            return
        try:
            text = TEXTS[kind].format(**facts)
        except (KeyError, IndexError, ValueError):
            return
        e = {"kind": kind, "text": text, "summary": None,
             "at_ms": int((time.time() - self.t0) * 1000)}
        if self.narrator is not None and kind not in PROGRESS:
            e["facts"] = {**facts, **(extra or {})}
        self.log.append(e)
        self.pending.append(e)

    def shown(self, e: dict) -> str:
        """The line as it goes out: the narrator's, written now (once), or the template's (always
        for the progress lines, whose numbers no model rewrites)."""
        if self.narrator is None or e["kind"] in PROGRESS:
            return frame(e["text"])
        if "narrator" not in e:
            try:
                e["summary"], e["narrator"] = self.narrator(e)
            except Exception as exc:                               # noqa: BLE001
                e["summary"], e["narrator"] = None, {"why": "error",
                                                     "error": f"{type(exc).__name__}: {exc}"[:200]}
        if e["summary"]:
            from server.conductor.narrator import frame as narrated
            return narrated(e["summary"])
        return frame(e["text"])

    def take(self) -> list[str]:
        out, self.pending = self.pending, []
        return [self.shown(e) for e in out]

    def take_kinds(self) -> list[tuple[str, str]]:
        """`take` with each line's kind: `(kind, line)`."""
        out, self.pending = self.pending, []
        return [(e["kind"], self.shown(e)) for e in out]

    # ------------------------------------------------------------------ from the conductor
    def route(self, row: dict | None, injected, packages: dict, compaction: bool) -> None:
        """The router's steps, once its plan is applied: a compaction it saw, the class a model
        decided this turn (not a class kept on a tool turn) with its effort, every package a
        note of this turn carries."""
        if row is None:
            return
        if compaction:
            self.add("compaction")
        if row.get("stages") and row.get("class_id"):
            eff = row.get("effort")
            effort = ("thinking off" if eff == "off" else f"{eff} effort") if eff else "default"
            self.add("route", cls=row["class_id"], effort=effort)
        for n in injected or []:
            for p in n.get("parts") or []:
                if p.get("kind") == "package":
                    meta = (packages or {}).get(p.get("name")) or {}
                    self.add("package", pkind=meta.get("kind") or "skill", name=p.get("name"))

    def monitor(self, ev: dict) -> None:
        """A session-monitor decision that acts (the `Supervisor.on_event` hook)."""
        d, task = ev.get("decision"), str(ev.get("task") or "")
        detail = ev.get("detail") or {}
        call = detail.get("call") if isinstance(detail.get("call"), dict) else {}
        # what the narrator may say beyond the template: the tools and files of a check, the
        # call a repair fixed, the reason the monitor gave
        extra = {k: v for k, v in (("tools", detail.get("tools")), ("files", detail.get("files")),
                                   ("tool", call.get("name")),
                                   ("check", schema_label(task) if task else None)) if v}
        why = dict(extra, reason=ev.get("reason")) if ev.get("reason") else extra
        if d == "CHECK":
            n = int(detail.get("calls") or 1)
            self.add("check", extra) if n == 1 else self.add("checks", extra, n=n)
        elif d == "CLOSE_REASONING":
            m = re.match(r"budget\((\d+)\)", str(ev.get("reason") or ""))
            if task in ("route", "client", "reserve") and m:
                self.add("cap", {"set by": CAP_BY[task]}, cap=f"{int(m.group(1)):,}")
            elif task == "D2":
                m = re.search(r"(\d+)", str(ev.get("reason") or ""))
                self.add("backstop", cap=f"{int(m.group(1)):,}") if m else self.add("close")
            elif task in ("D1", "D1b"):
                self.add("loop", extra)
            else:
                self.add("close", extra)
        elif d == "CONTINUE_HIDDEN":
            self.add("continue")
        elif d == "REPAIR_CALL":
            self.add("repair", why)
        elif d == "REGENERATE_CALL":
            self.add("regenerate", why)
        elif d == "GIVE_UP":
            self.add("give_up", why)
        elif d == "PROMPT_TAIL_NOTE":
            from server.conductor import schema
            self.add("steer", label=(schema.TASKS.get(task) or {}).get("label") or task)
