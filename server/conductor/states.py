"""TandemLLM's own steps, shown to the user in the reasoning channel (`ui.states`, 2026-10-02).

A client shows Qwen's reasoning and its answer. What the conductor does around them -- the class
it picked, a skill it loaded, a cap that closed the reasoning, a tool call it repaired -- was
visible only on the dashboard. With a profile's `ui.states: on` each step is one short line in the
stream's reasoning, at the moment it happens, in the request's `reasoning_format`: a
`reasoning_content` delta, or inside the `<think>` block of `tags` (a step after the block closed
has no place in `tags` and is left out there). The texts are templates (`TEXTS`), no model and no
latency; every line keeps a `summary` field for a clerk-written one later (not built).

THE MARKER. Each line goes out framed as

    "\\n› <text>\\n\\n"

`›` (U+203A) and a space open the line, a blank line closes it: one paragraph of its own in a
client that renders markdown (opencode), one plain line in one that does not (Hermes), and the
same in both formats. The frame is self-delimiting, so `strip_text` takes it out exactly wherever
it sits: at the head or the end of the reasoning, between two parts a client joined, or between
two characters of Qwen's own text. Only a framed line whose text is one of the templates is
taken, so a line of Qwen's own that starts with `›` stays.

QWEN NEVER SEES THEM. opencode's chat-completions provider sends every earlier assistant message's
reasoning back as `reasoning_content` (its parts joined with nothing between them), and the Hermes
agent keeps the reasoning on the message it sends back; the Qwen3.8 template renders an assistant
message's `reasoning_content` (trimmed) into the prompt. So the server strips every framed line from
the `reasoning_content` and the string content of the assistant messages before anything reads them
(`strip_messages`, in the handler before the router and before the prompt is rendered): the prompt,
and with it the resident prefix and the notes' positions, are the ones the request would have
without the feature. A client that trims the reasoning loses the frame's outer newlines, so a line
at the very start or end is also taken without them; whatever whitespace is left at an end the
template trims. The lines are not tokens: they are never in `usage`, the suffix store or the
response cache key; the transcript keeps them as a block of their own and the journal row as
`extra.states`.
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
}
# the profile's per-kind switches (`ui.kinds`) name these; `check` covers `checks`
KINDS = tuple(k for k in TEXTS if k != "checks")

# a line the server writes: the marker and one of the templates, nothing else, so a line of
# Qwen's own that happens to start with `›` is never taken
_TEXT = "(?:" + "|".join((r"[^\n]{1,%d}" % MAX_CHARS).join(
    re.escape(part) for part in re.split(r"\{\w+\}", t)) for t in TEXTS.values()) + ")"
_MID = re.compile(r"\n" + re.escape(MARKER) + _TEXT + r"\n\n")
_HEAD = re.compile(r"\A" + re.escape(MARKER) + _TEXT + r"(?:\n\n|\n?\Z)")
_TAIL = re.compile(r"\n" + re.escape(MARKER) + _TEXT + r"\n?\Z")


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


def into_full(content: str, reasoning: str | None, fmt: str, in_think: bool, ui: "States"
              ) -> tuple[str, str | None]:
    """A non-streamed answer's `(content, reasoning_content)` with the request's lines: those
    said before the generation at the head of the reasoning, the rest at its end, in the
    `reasoning_content` field (formats `reasoning_content` and `both`; created when the turn did
    not think) and inside the `<think>` block of the content (`tags` and `both`, when it has one)."""
    cut = len(ui.log) if ui.head is None else ui.head
    head = "".join(frame(e["text"]) for e in ui.log[:cut])
    tail = "".join(frame(e["text"]) for e in ui.log[cut:])
    if fmt in ("reasoning_content", "both"):
        reasoning = head + (reasoning or "") + tail
    opener = "<think>\n"
    if fmt in ("tags", "both") and in_think and content.startswith(opener):
        end = content.find("</think>")
        end = len(content) if end < 0 else end
        content = opener + head + content[len(opener):end] + tail + content[end:]
    return content, reasoning


class States:
    """One request's steps: what is waiting to go out, and everything it said."""

    def __init__(self, kinds: dict | None = None):
        self.kinds = kinds or {}
        self.pending: list[str] = []
        self.log: list[dict] = []
        self.head: int | None = None       # how many were said before the generation started
        self.t0 = time.time()

    def on(self, kind: str) -> bool:
        base = "check" if kind == "checks" else kind
        return bool(self.kinds.get(base, True))

    def add(self, kind: str, **facts) -> None:
        if not self.on(kind):
            return
        try:
            text = TEXTS[kind].format(**facts)
        except (KeyError, IndexError, ValueError):
            return
        self.log.append({"kind": kind, "text": text, "summary": None,
                         "at_ms": int((time.time() - self.t0) * 1000)})
        self.pending.append(frame(text))

    def take(self) -> list[str]:
        out, self.pending = self.pending, []
        return out

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
        if d == "CHECK":
            n = int((ev.get("detail") or {}).get("calls") or 1)
            self.add("check") if n == 1 else self.add("checks", n=n)
        elif d == "CLOSE_REASONING":
            m = re.match(r"budget\((\d+)\)", str(ev.get("reason") or ""))
            if task in ("route", "client", "reserve") and m:
                self.add("cap", cap=f"{int(m.group(1)):,}")
            elif task == "D2":
                m = re.search(r"(\d+)", str(ev.get("reason") or ""))
                self.add("backstop", cap=f"{int(m.group(1)):,}") if m else self.add("close")
            elif task in ("D1", "D1b"):
                self.add("loop")
            else:
                self.add("close")
        elif d == "CONTINUE_HIDDEN":
            self.add("continue")
        elif d == "REPAIR_CALL":
            self.add("repair")
        elif d == "REGENERATE_CALL":
            self.add("regenerate")
        elif d == "GIVE_UP":
            self.add("give_up")
        elif d == "PROMPT_TAIL_NOTE":
            from server.conductor import schema
            self.add("steer", label=(schema.TASKS.get(task) or {}).get("label") or task)
