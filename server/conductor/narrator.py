"""The narrator: a small model inside the engine writes the state lines' one-line summaries (P1).

G1's state lines (states.py, `ui.states`) are templates: "deciding: code_edit, high effort",
"checking the tool call". Each line keeps a `summary` field for a model-written one. With a
profile's `ui.narrator: on` the narrator writes it: the event's kind, its template and its facts
(the class and its label, the effort, a cap, the tool names and files of a check, the counts) go
to the small model of the `narrator` role (MiniCPM5-2B, engine/generative), which answers with one
plain line, and that line goes out instead of the template.

THE LIMITS. One line is at most `MAX_WORDS` words and `max_tokens` tokens (24), written inside the
role's latency budget (`latency_budget_ms` in the roles document, 150 ms by default), from the call
to the last token. A request spends at most `REQUEST_LINES` budgets on its lines; after that its
lines are the templates. Whatever fails -- the model not loaded yet (its first line starts the
load), refused by the residency planner, busy, the budget gone before the end token, the token
cap reached, too many words, an error -- the line is the template, and the journal says why.

QWEN IS NEVER BESIDE IT. The narrator runs on the request's own thread when the line goes out:
before the prefill for the router's line, between two of Qwen's rounds for the monitor's, after
the last token for the tool checks. Qwen waits meanwhile and nothing of Qwen runs at the same time
(Harmony 3: during decode the GPU is Qwen's); the engine's lock is held by the request anyway, and
the narrator holds it at most its budget and one step a line. The model's own stream is
synchronised before Qwen goes on (engine/decision/device.py).

THE FRAME. A narrated line goes out as `"\\n› " + text + "\\u2060\\n\\n"`: the template's frame
with a WORD JOINER (U+2060, invisible, not whitespace, so no client's trim and no template's
`|trim` removes it) before the closing newlines. The strip (states.py) takes a framed line of any
text that ends in it, so a narrated line, which no template matches, never reaches Qwen's prompt
either; Qwen does not write U+2060 at the end of a `› ` line of its own.

`roles.narrator.mode`: `on` writes the lines, `shadow` writes them into the journal only (the
client sees the templates: line quality can be read without anyone seeing them), `off` never asks
the model.
"""

from __future__ import annotations

import re

from engine.settings import SETTINGS as _S

MAX_WORDS = 14
MAX_TOKENS = 24
# a request's lines share this many line budgets; the rest are templates
REQUEST_LINES = 4
# the try route's ceilings (the measurement lifts the budget to read what the model writes)
TRY_MAX_BUDGET_MS = 10_000
TRY_MAX_TOKENS = 64
TRY_MAX_EVENTS = 64
SENTINEL = "\u2060"
OFF_WHY = ("this engine serves no narrator: QWEN38_NARRATOR_MODEL is empty (set it in "
           "ops/serve.env, e.g. minicpm5-2b, and restart)")

# The prompt (2026-10-02, read on the box CPU against 12 sample events): without examples and with
# the template beside the event, MiniCPM5-2B copied the template word for word; with the template
# left out, the facts in plain words (a class's label, "very high effort") and three examples it
# writes a line of its own. The system turn is the same for every line, so its rows stay in the
# model's cache and only the event is prefilled.
SYSTEM = ("You turn an event inside an AI coding engine into the status line its user reads while "
          "waiting. Write like a friendly colleague: plain words, present tense, 4 to 12 words, "
          "no quotes, no emoji, no markdown, no full stop. Do not copy the engine's label; say "
          "identifiers in plain words. Use only the facts given; never invent file names, numbers "
          "or reasons.\n\n"
          "Examples:\n"
          "Event: the engine chose what kind of task this is and how hard to think about it\n"
          "Facts: task: UI design; effort: medium effort\n"
          "Line: Treating this as UI design work, with moderate thought\n\n"
          "Event: the engine loaded a skill into the prompt\n"
          "Facts: kind: skill; name: voxel-seasons\n"
          "Line: Pulling in the voxel-seasons skill\n\n"
          "Event: the engine checks the model's tool call before the client runs it\n"
          "Facts: tools: write; files: web/index.html\n"
          "Line: Double-checking the write to web/index.html before it runs")
# the template's fact names, and words, as the model reads them
FACT_NAMES = {"cls": "task", "pkind": "kind", "n": "count"}
EFFORT_WORDS = {"xhigh effort": "very high effort", "thinking off": "no thinking"}

# what each kind of step means, for the model
EVENTS = {
    "route": "the engine chose what kind of task this is and how hard to think about it",
    "compaction": "the long conversation so far was replaced by a short summary to save room",
    "package": "the engine loaded a skill into the prompt",
    "memory": "the engine read notes from its memory",
    "steer": "the engine added a hidden steering note for the model",
    "cap": "the thinking budget ran out, so the reasoning was closed",
    "backstop": "the reasoning reached the hard backstop and was closed",
    "loop": "the reasoning started repeating itself, so the engine steered it",
    "close": "the engine closed the reasoning",
    "continue": "the model stopped without writing an answer, so the engine makes it go on",
    "check": "the engine checks the model's tool call before the client runs it",
    "checks": "the engine checks the model's tool calls before the client runs them",
    "repair": "a tool call was malformed and the engine fixed it",
    "regenerate": "a tool call was wrong and the model writes it again",
    "give_up": "the engine gave up after the tool call failed again",
}
FACT_CHARS = 120
_BAD = re.compile(r"[\u203a\u2060<>\n]|\|im_|</?think")


def messages(entry: dict, labels: dict | None = None) -> list[dict]:
    """The model's prompt for one line: the fixed system turn and the event with its facts, in
    words (`labels`: class id -> the router's label for it). Facts pass the secret filter."""
    from server.conductor.redact import redact
    facts = []
    for k, v in (entry.get("facts") or {}).items():
        if v is None or v == "" or v == []:
            continue
        v = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        if k == "cls":
            v = (labels or {}).get(v) or v.replace("_", " ")
        elif k == "effort":
            v = EFFORT_WORDS.get(v, v)
        v = redact(" ".join(v.split()))[0][:FACT_CHARS]
        facts.append(f"{FACT_NAMES.get(k, k)}: {v}")
    kind = entry["kind"]
    user = (f"Event: {EVENTS.get(kind, kind)}\nFacts: {'; '.join(facts) if facts else 'none'}"
            f"\nLine:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def clean(text: str) -> tuple[str | None, str | None]:
    """(the line, None), or (None, why it cannot be one)."""
    s = (text or "").strip().split("\n", 1)[0].strip()
    if s.lower().startswith("line:"):
        s = s[5:]
    s = s.strip("\"'`*_ ").rstrip(".").strip()
    s = " ".join(s.split())
    if not s:
        return None, "empty"
    if _BAD.search(s):
        return None, "markup"
    if len(s.split()) > MAX_WORDS:
        return None, "words"
    from server.conductor.states import MAX_CHARS
    if len(s) > MAX_CHARS:
        return None, "chars"
    return s, None


def frame(text: str) -> str:
    """A narrated line as it goes out (states.frame plus the sentinel)."""
    from server.conductor.states import MARKER
    return "\n" + MARKER + text + SENTINEL + "\n\n"


class Narrator:
    """The `narrator` role over its slot: one line per state, inside the role's budget."""

    def __init__(self, slot, role=lambda: None, labels=lambda: {}):
        """`role()`: the roles document's `narrator` entry (its mode and latency budget);
        `labels()`: class id -> the router's label."""
        self.slot, self.role, self.labels = slot, role, labels

    def settings(self) -> dict:
        r = self.role() or {}
        return {"mode": r.get("mode", "on"), "budget_ms": int(r.get("latency_budget_ms") or 150)}

    def for_request(self) -> "Lines | None":
        """What one request's States calls per line; None when the role is off."""
        s = self.settings()
        if s["mode"] == "off":
            return None
        return Lines(self, s["budget_ms"], shadow=s["mode"] == "shadow")

    def line(self, entry: dict, *, budget_ms: float, max_tokens: int = MAX_TOKENS,
             held: bool = True) -> tuple[str | None, dict]:
        """(the narrated line or None, what happened: `why` None or the reason for the template,
        `ms`, `tokens`, `finish`, the raw text when it was refused)."""
        r = self.slot.run(messages(entry, self.labels()), max_new=max_tokens, budget_ms=budget_ms,
                          held=held)
        info = {k: r[k] for k in ("ms", "tokens", "finish", "prompt_tokens", "reused")
                if k in r}
        if r.get("why"):
            return None, dict(info, why=r["why"], **({"error": r["error"]} if r.get("error")
                                                     else {}))
        if r["finish"] != "stop":
            return None, dict(info, why=r["finish"], raw=r["text"][:200])
        text, why = clean(r["text"])
        if text is None:
            return None, dict(info, why=why, raw=r["text"][:200])
        return text, dict(info, why=None)

    def try_events(self, events: list, *, budget_ms: float | None = None,
                   max_tokens: int | None = None) -> list[dict]:
        """`POST /v1/conductor/narrator/try`: each event's template and line, in the idle window
        (the engine's lock taken per line)."""
        from server.conductor.states import TEXTS
        budget = float(budget_ms or self.settings()["budget_ms"])
        cap = int(max_tokens or MAX_TOKENS)
        out = []
        for ev in events:
            kind, facts = ev.get("kind"), dict(ev.get("facts") or {})
            try:
                text = TEXTS[kind].format(**facts)
            except (KeyError, IndexError, ValueError):
                out.append({"kind": kind, "error": f"no template for {kind!r} with these facts"})
                continue
            entry = {"kind": kind, "text": text, "facts": dict(facts, **(ev.get("extra") or {}))}
            line, info = self.line(entry, budget_ms=budget, max_tokens=cap, held=False)
            out.append({"kind": kind, "template": text, "line": line,
                        "shown": line if line is not None else text, **info})
        return out


class Lines:
    """One request's narrator: the role's budget per line, `REQUEST_LINES` budgets in all."""

    def __init__(self, narrator: Narrator, budget_ms: float, *, shadow: bool = False):
        self.narrator, self.budget_ms, self.shadow = narrator, float(budget_ms), shadow
        self.left_ms = REQUEST_LINES * self.budget_ms

    def __call__(self, entry: dict) -> tuple[str | None, dict]:
        if self.left_ms <= 0:
            return None, {"why": "request_budget"}
        budget = min(self.budget_ms, self.left_ms)
        line, info = self.narrator.line(entry, budget_ms=budget)
        self.left_ms -= float(info.get("ms") or 0.0)
        if self.shadow:
            return None, dict(info, shadow=line)
        return line, info


# ---------------------------------------------------------------------- the server's wiring
def configured() -> str:
    return (_S.get("NARRATOR_MODEL") or "").strip()


def start(state: dict, gate, *, fake: bool = False, log=print):
    """The narrator for this server, or None when `QWEN38_NARRATOR_MODEL` is empty. Nothing
    loads here: the model loads when a profile's first narrated line asks for it."""
    name = configured()
    if not name:
        return None
    from engine import generative
    from engine.generative.slot import Slot
    from server import decider_api
    from server.conductor import residency, schema
    model = generative.model(name, _S.get("NARRATOR_DIR"), allow_fake=fake)
    device = None
    if not fake:
        from engine.decision.device import Device
        device = Device(_S.get("NARRATOR_DEVICE"))

    def journal(rec):
        cond = state.get("conductor")
        if cond is not None:
            residency.Journal(cond.root).write(rec)

    slot = Slot(model, role="narrator", gate=gate,
                planner=decider_api.planner(state, fake=fake, role="narrator"), device=device,
                log=lambda s: log(s, flush=True),
                contract=lambda f, n: schema.role_errors(f, n, state.get("tokenizer_sha")),
                journal=journal, mem_available=lambda: decider_api._mem_available_gib(fake))
    log(f"[narrator] {name} on {device or 'no device (fake)'}: loads when a profile with "
        f"ui.narrator on first asks for a line, through the residency planner", flush=True)
    return Narrator(slot, role=lambda: role_doc(state), labels=lambda: class_labels(state))


def role_doc(state: dict) -> dict:
    """The roles document's `narrator` (the built-in one in a document stored before it)."""
    from server.conductor import schema
    cond = state.get("conductor")
    roles = cond.store.peek("roles")[1] if cond is not None else schema.seed("roles")
    return roles.get("narrator") or schema.seed("roles")["narrator"]


def class_labels(state: dict) -> dict:
    """class id -> label, from the router document."""
    cond = state.get("conductor")
    if cond is None:
        return {}
    return {c["id"]: c["label"] for c in cond.store.peek("router")[1].get("classes") or []}


def status(state: dict) -> dict:
    """`GET /v1/conductor/narrator/status`."""
    n = state.get("narrator")
    if n is None:
        return {"enabled": False, "why": OFF_WHY}
    return {"enabled": True, "why": None, "settings": n.settings(), "slot": n.slot.status(),
            "limits": {"max_words": MAX_WORDS, "max_tokens": MAX_TOKENS,
                       "request_lines": REQUEST_LINES}}
