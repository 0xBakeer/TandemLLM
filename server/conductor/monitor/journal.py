"""The journal: every decision the monitor made, with the text it hid and what came of it.

The transcript a client keeps differs from what the model saw whenever the monitor acts, so each
hidden injection is recorded with its exact text, where it went, why, and the outcome. This module
keeps the newest `keep` events in memory and prints one `[conductor]` line per event -- the line
names the task, the decision and the counts, never a text. SRV-115 adds the SQLite table next to
the usage ledger, the Prometheus counters and the export through `sinks`: a sink is called with
every event after it is recorded, and a sink that raises is dropped from nothing and fails nothing.

An event is a dict. The keys the monitor writes:

    ts, request_id, session, sp, task, rule, tier, mode ("shadow" | "act"),
    decision (an action, or PASS), reason, score, detail, text, hidden_tokens, at_tokens,
    cap (why an act became GIVE_UP), outcome (tool_call | text | empty | gave_up | ...)
"""

from __future__ import annotations

import threading
import time
from collections import deque


class Journal:
    def __init__(self, keep: int = 4096, echo: bool = True):
        self._events: deque = deque(maxlen=int(keep))
        self._lock = threading.Lock()
        self.sinks: list = []
        self.echo = echo

    def write(self, event: dict) -> dict:
        """Record one event (a `ts` is added when it has none) and hand it to the sinks."""
        event.setdefault("ts", time.time())
        with self._lock:
            self._events.append(event)
        if self.echo:
            print(_line(event), flush=True)
        for sink in list(self.sinks):
            try:
                sink(event)
            except Exception:                                      # noqa: BLE001
                pass                       # a sink must never fail a request
        return event

    def events(self, request_id: str | None = None, session: str | None = None) -> list[dict]:
        with self._lock:
            evs = list(self._events)
        return [e for e in evs
                if (request_id is None or e.get("request_id") == request_id)
                and (session is None or e.get("session") == session)]

    def actions(self, request_id: str | None = None) -> list[dict]:
        """The events that executed something: mode act and a decision other than PASS."""
        return [e for e in self.events(request_id)
                if e.get("mode") == "act" and e.get("decision") not in (None, "PASS")]


def _line(e: dict) -> str:
    parts = [f"[conductor] {e.get('request_id', '-')} {e.get('sp', '-')}",
             f"task={e.get('task') or '-'}", f"mode={e.get('mode', '-')}",
             f"decision={e.get('decision', '-')}"]
    for k in ("reason", "cap", "hidden_tokens", "at_tokens", "outcome"):
        v = e.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={v}")
    return " ".join(parts)
