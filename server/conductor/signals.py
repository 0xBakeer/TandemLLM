"""What a request's structure says, and what the checkers said (SRV-123).

Two kinds of signal, both read without judging what anybody meant:

  * `shape(body)`: roles, tools, message counts, the last user message's text and whether the turn
    only carries tool results, whether it is opencode's compaction request, which skills the client
    already loaded. These are structural facts; no word list decides anything here (Khaled's
    standing rule: intent comes from a model, never from keyword routing).
  * `checker_results(text)`: the `RESULT: PASS|FAIL <- reasons` line the demo checkers print
    (`sketchcheck`, `pagodacheck`, `scenecheck`, and the eval suites' `calccheck` and
    `taskcheck`) and the
    numbers in its reasons. Only tool results are read, never assistant text, so a RESULT line
    the model quotes is not a result. A new checker needs one entry in `CHECKERS` (its header
    line), nothing else; the session monitor passes its own list and patterns from its policy.
"""

from __future__ import annotations

import hashlib
import json
import re

# A checker's first line names it and what it checked; the gate id carries that target so a
# checker run on the wrong file is a different gate.
CHECKERS = {
    "sketchcheck": re.compile(r"^sketchcheck\s+(\S+)", re.M),
    "pagodacheck": re.compile(r"^pagodacheck\s+(\S+)", re.M),
    "scenecheck": re.compile(r"^scenecheck\s+(\S+)", re.M),
    "calccheck": re.compile(r"^calccheck\s+(\S+)", re.M),
    "taskcheck": re.compile(r"^taskcheck\s+(\S+)", re.M),
}
_RESULT = re.compile(r"^RESULT:\s*(PASS|FAIL)\b[ \t]*(?:<-[ \t]*(.*))?$", re.M)
_GATE = re.compile(r"^\(([a-z0-9]{1,3})\)\s*")
_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)")
_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_ .-]*?)\s+(-?\d+(?:\.\d+)?)\s*(?:<=|>=|<|>)")

# opencode's compaction request: no tools, no system text, the history as one user message
# (docs/caches.md). The length floor keeps a short one-line request without a system prompt out.
COMPACTION_MIN_CHARS = 2000


def _text(content) -> str:
    """The text of a message's content: a string, or the text parts of a list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _images(content) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and
               (p.get("type") in ("image_url", "image") or "image_url" in p))


def checker_results(text: str, checkers: dict | None = None, gate=None,
                    reading=None) -> list[dict]:
    """Every RESULT line in one tool result: `{checker, target, verdict, reasons, readings}`,
    each reading `{gate, name, value, op, limit}` with `gate` the checker's own label ("a") or
    the metric name. A tool result may hold two checker runs; each RESULT line pairs with the
    nearest header above it. `checkers` (name -> header pattern with the target as its group),
    `gate` (a reason's gate label) and `reading` (value, comparison, limit as three groups)
    replace the built-in patterns; patterns may be compiled or strings."""
    def rx(p):
        return re.compile(p, re.M) if isinstance(p, str) else p
    checkers = CHECKERS if checkers is None else {k: rx(v) for k, v in checkers.items()}
    gate_rx = _GATE if gate is None else rx(gate)
    num_rx = _NUM if reading is None else rx(reading)
    out = []
    heads = sorted(((m.start(), name, m.group(1)) for name, h in checkers.items()
                    for m in h.finditer(text)))
    for m in _RESULT.finditer(text):
        above = [h for h in heads if h[0] < m.start()]
        checker, target = (above[-1][1], above[-1][2]) if above else ("unknown", "")
        reasons = [r.strip() for r in (m.group(2) or "").split(";") if r.strip()]
        readings = []
        for r in reasons:
            g = gate_rx.match(r)
            n = num_rx.search(r)
            if n is None:
                continue
            name_m = _NAME.search(gate_rx.sub("", r, count=1))
            name = (name_m.group(1).strip() if name_m else "").strip(" :")
            op = n.group(2)
            readings.append({"gate": g.group(1) if g else (name or "value"), "name": name,
                             "value": float(n.group(1)), "op": op,
                             "limit": float(n.group(3))})
        out.append({"checker": checker, "target": target, "verdict": m.group(1),
                    "reasons": reasons, "readings": readings})
    return out


class Shape:
    """The structural facts of one chat request."""

    __slots__ = ("tools", "n_messages", "has_system", "last_role", "tool_turn", "user_text",
                 "user_index", "user_hash", "images", "compaction", "results", "loaded_skills",
                 "system_text", "tool_names_called", "first_hash")

    def __repr__(self):
        return (f"Shape(tools={self.tools}, last={self.last_role}, tool_turn={self.tool_turn}, "
                f"compaction={self.compaction}, user_index={self.user_index})")


def _is_tool_response(text: str) -> bool:
    t = text.strip()
    return t.startswith("<tool_response>") and t.endswith("</tool_response>")


def shape(body: dict, skill_names: tuple = ()) -> Shape:
    """The facts `router.py` decides from. `skill_names`: the installed skills, for the exact-name
    check of which ones the client already loaded."""
    s = Shape()
    msgs = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    tools = body.get("tools") or []
    tc = body.get("tool_choice")
    s.tools = bool(tools) and tc != "none"
    s.n_messages = len(msgs)
    s.has_system = bool(msgs) and msgs[0].get("role") == "system"
    s.system_text = _text(msgs[0].get("content")) if s.has_system else ""
    s.last_role = msgs[-1].get("role") if msgs else None
    # the last message a person wrote: a user message that is not only tool responses
    s.user_index, s.user_text, s.images = None, "", 0
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user" and not _is_tool_response(_text(m.get("content"))):
            s.user_index, s.user_text = i, _text(m.get("content"))
            s.images = _images(m.get("content"))
            break
    s.tool_turn = s.last_role == "tool" or (
        s.last_role == "user" and s.user_index is not None and s.user_index != len(msgs) - 1)
    s.user_hash = (hashlib.sha256(f"{s.user_index}\x00{s.user_text}".encode()).hexdigest()[:24]
                   if s.user_index is not None else None)
    s.compaction = (not s.tools and not s.has_system and len(msgs) == 1 and
                    s.last_role == "user" and len(s.user_text) >= COMPACTION_MIN_CHARS)
    # the checker results that arrived since the model last wrote
    s.results = []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            break
        if m.get("role") in ("tool", "user"):
            s.results[:0] = checker_results(_text(m.get("content")))
    # skills the client already loaded: a call to its skill tool naming one, exact names only
    loaded, called = set(), []
    for m in msgs:
        for c in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            fn = (c or {}).get("function") or {}
            name = str(fn.get("name") or "")
            called.append(name)
            if "skill" in name.lower():
                args = fn.get("arguments")
                blob = args if isinstance(args, str) else json.dumps(args or {})
                for sk in skill_names:
                    if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(sk)}(?![A-Za-z0-9_-])", blob):
                        loaded.add(sk)
    s.loaded_skills = loaded
    s.tool_names_called = called
    first = [m for m in msgs[:2]]
    s.first_hash = hashlib.sha256(json.dumps(
        [(m.get("role"), _text(m.get("content"))[:4096]) for m in first],
        ensure_ascii=False).encode()).hexdigest()[:24] if first else None
    return s


def skills_named(system_text: str, skill_names) -> list[str]:
    """Installed skills the system prompt names by their exact id (AGENTS.md lists them)."""
    return [n for n in skill_names
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(n)}(?![A-Za-z0-9_-])", system_text)]


def digest(text: str, max_chars: int) -> str:
    """The text a classifier sees: the whole message when it fits, else its head and tail
    (the task is usually stated first, the latest detail last), marked where it was cut."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head - 7
    return text[:head].rstrip() + "\n[...]\n" + text[-tail:].lstrip()
