"""The conductor's task router: which profile, which kind of task, and what that turn gets.

Two halves per request, split by the engine lock.

`Router.plan()` runs BEFORE the lock (it may ask the decider, or Qwen through this server's own port,
and must never hold the lock while it does):

  1. Resolve the profile (SRV-117): the server serves one model, `TandemLLM`. Every model name
     resolves to it -- the raw engine's old name is a hidden alias, so no client breaks -- and goes
     to `X-Tandem-Profile`, else the profile mapped to the client kind, else the default.
     `X-Tandem: raw|off|shadow` lowers the mode and never raises it; `X-Tandem: raw` (or the
     locked `raw` / `bench` profile by header) is how benchmarks reach the release's path.
  2. Decide the class of the turn (SRV-129). Structure first: a turn that only carries tool
     results keeps the session's class; opencode's compaction request is `summarize`. A new user
     message goes through the classifier stages (the decider, then Qwen) until one answers at or above
     its threshold. A session changes class only at the stickiness threshold (0.8). Nobody sure:
     the session's class, else the profile's default. No word list decides anything.
  3. Turn the class into a policy: an effort level (budget, thinking on or off, a note), the
     packages (a second choice question when a class has several candidates), and rules.
     Each decision type is `off`, `shadow` or `on` per profile; shadow is journaled only.
     A class with an `effort_by_size` map (SRV-157) picks its effort from the session's size
     (the tokens it added, the files its calls named, the gates a checker named, the turns) under
     the class's effort as the ceiling; the size only rises within a task, and a compaction starts
     it over. The profile's `thinking.by_size` says off, shadow or on.
     The level's budget is a real cap (medium 8,192, high 16,384, xhigh none below the backstop;
     a profile's `thinking.budgets` overrides a level): the server closes the reasoning there
     with the session monitor's hidden close (SP1), so the client sees only the closer.
     A turn whose newest message is a tool result may think at its own level (F2: the router's
     `structural.tool_turn_effort`, a profile's `thinking.tool_turn_effort` over it; `inherit`,
     the default, is the class's): its switch and cap, journaled as `effort_source: tool_turn`.

The classifier's digest (SRV-153) is sized per stage, the decider's to its own `limits()` and
Qwen's to `digest_chars`, head and tail, and carries one context line the router writes from what
it knows: the previous class, the last tools, the file types, the checker's status, a compaction.

`Plan.apply_prompt()` runs UNDER the lock, on the rendered prompt ids: it replays the session's
earlier notes at their positions (server/conductor/inject.py) and adds this turn's note when the
active block is not already in the prompt. `Plan.finish()` writes the outcome.

Everything a plan did is in its journal row (server/conductor/decisions.py).
"""

from __future__ import annotations

import collections
import hashlib
import threading
import time

from server import thinking
from server.conductor import incontext, inject, outcomes, schema, signals, transcripts
from server.conductor.classify import ASKING, Answer, Qwen, decider_for
from server.conductor.store import ConfigError

COMPOSITE = "TandemLLM"
RANK = {"raw": 0, "off": 0, "shadow": 1, "on": 2}
SESSIONS_MAX = 64
TEXT_CACHE_MAX = 512
EFFORT_ORDER = ("off", "low", "medium", "high", "xhigh")
NONE_OPTION = "none"


def _lower(a: str, b: str | None) -> str:
    """The lower of two modes; `off` and `raw` rank the same."""
    if b is None:
        return a
    return a if RANK.get(a, 0) <= RANK.get(b, 0) else ("raw" if b in ("off", "raw") else b)


def _header(headers, name: str) -> str:
    try:
        return (headers.get(name) or "").strip()
    except Exception:                                              # noqa: BLE001
        return ""


class Resolution:
    __slots__ = ("profile", "version", "doc", "mode", "source", "requested")

    def __init__(self, profile, version, doc, mode, source, requested):
        self.profile, self.version, self.doc = profile, version, doc
        self.mode, self.source, self.requested = mode, source, requested

    def headers(self) -> tuple:
        """`X-Tandem-Profile`; `X-Tandem` itself goes out with every completion response
        (server/app.py's `Handler.send_response`), set from this resolution's mode."""
        return (("X-Tandem-Profile", self.profile),)


def is_alias(model: str, raw_model: str) -> bool:
    """The raw engine's old name: served as `TandemLLM`, never listed (SRV-117, 2026-10-01)."""
    return bool(model) and model.lower() == str(raw_model or "").lower() and \
        model.lower() != COMPOSITE.lower()


def resolve(store, model: str, raw_model: str, headers, client_kind: str,
            global_mode: str | None = None) -> Resolution:
    """The profile and mode of one request; `ConfigError(400)` for an unknown profile header.
    `model` and `raw_model` are kept in the signature for the callers; neither picks the profile."""
    requested = _header(headers, "X-Tandem-Profile")
    forced = _header(headers, "X-Tandem").lower()
    # One model (Khaled, 2026-10-01): `model` is not consulted. `TandemLLM`, the raw engine's old
    # name (a hidden alias) and any other name resolve the same way; raw is a header, not a name.
    if requested:
        if not store.exists("profiles", requested):
            raise ConfigError(400, f"no conductor profile {requested!r}")
        name, source = requested, "header"
    else:
        name, source = None, "default"
        for n in store.names():
            if n in schema.LOCKED:
                continue
            if client_kind in store.peek("profiles", n)[1].get("clients", []):
                name, source = n, "client"
                break
        if name is None:
            name = store.peek("settings")[1]["default_profile"]
    v, doc = store.peek("profiles", name)
    mode = doc.get("mode", "raw")
    if forced in ("raw", "off"):
        mode = "raw"
    elif forced == "shadow":
        mode = _lower(mode, "shadow")
    mode = _lower(mode, global_mode)
    return Resolution(name, v, doc, mode, source, requested or None)


class Injection:
    """One note in a session: where it sits in the client's prompt and what it is."""

    __slots__ = ("pos", "hash", "ids", "text", "parts", "kinds", "key", "turn", "at", "row")

    def __init__(self, pos, hsh, ids, text, parts, kinds, key, turn, row):
        self.pos, self.hash, self.ids, self.text = pos, hsh, ids, text
        self.parts, self.kinds, self.key, self.turn, self.row = parts, kinds, key, turn, row
        self.at = time.time()


class Session:
    __slots__ = ("key", "cls", "conf", "source", "since", "turns", "user_hash", "last_row",
                 "notes", "history", "package_pick", "lock", "compacted", "size_tier",
                 "prompt_first", "prompt_last", "evidence", "acts")

    def __init__(self, key):
        self.key = key
        self.cls = self.conf = self.source = self.user_hash = self.last_row = None
        self.compacted = False             # a compaction request came; the next class says so
        self.size_tier = None              # the largest size tier this task reached (SRV-157)
        self.prompt_first = self.prompt_last = None   # prompt tokens: first and latest turn
        self.since = time.time()
        self.turns = 0
        self.notes: list[Injection] = []
        self.history: collections.deque = collections.deque(maxlen=64)
        self.package_pick: dict = {}
        self.lock = threading.Lock()
        self.evidence: outcomes.Evidence | None = None   # J1: the snapshots' state
        self.acts: collections.deque = collections.deque(maxlen=64)  # J1: the notes' times


def session_key(body: dict, headers, sh: signals.Shape) -> str:
    """A conversation id if the client sends one (opencode sends `x-session-id`), else a hash of
    the conversation's first two messages, which stay the same on every turn."""
    for k in ("conversation_id", "session_id"):
        v = body.get(k)
        if isinstance(v, str) and v:
            return "c:" + v[:120]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("conversation_id") or meta.get("session_id") or meta.get("chat_id")
        if isinstance(v, str) and v:
            return "c:" + v[:120]
    for h in ("x-session-id", "X-Conversation-Id"):
        v = _header(headers, h)
        if v:
            return "c:" + v[:120]
    return "h:" + (sh.first_hash or "empty")


class Plan:
    """What the router decided for one request; `active` is False for raw requests."""

    def __init__(self, res: Resolution):
        self.res = res
        self.active = False
        self.row: dict | None = None
        self.thinking_off = False                    # close the generation prompt's <think>
        self.budget: int | None = None               # None = leave the request's budget
        self.budget_source = "server"
        self.block: dict | None = None               # the note for this turn, if one is due
        self.session: Session | None = None
        self.router_cfg: dict | None = None
        self.decisions: dict = {}
        self.injected: list[dict] = []
        self.clean_ids: list[int] | None = None
        self.changed = False
        self.compaction = False                      # opencode's compaction request
        # SRV-137: what the transcript writer keeps of this request -- full, counters or None --
        # and the body it reads its new messages from (a reference, read after the request ended)
        self.transcript: str | None = None
        self.body: dict | None = None
        # the in-context stage, when it is due: Qwen decides the class under the engine lock
        # (`Router.decide_in_context`) and the rest of the plan waits for it
        self.context: dict | None = None
        # ENG-181: the client's insertion point and the notes spliced before it, for the row the
        # probe snapshots read (`probe_row`)
        self._probe_at: tuple | None = None
        # C1: every note in the served prompt as (start, end, kinds, package names), for the disk
        # tier's entries (engine/disk_cache.py names a prefix that holds a package by it)
        self.spans: list[tuple] = []
        # J1: the turn's snapshot (the transcript writer stores it), its outcome, the notes this
        # plan adds besides the class's block (an outcome's note, the cards; each injected once,
        # then replayed), and the acts it took (the monitor's caps count them)
        self.snapshot: dict | None = None
        self.outcome: dict | None = None
        self.judge: list[dict] | None = None          # Laya's questions, after the request
        self.ask: dict | None = None                  # Qwen's, in place before the prefill
        self.extras: list[dict] = []
        self.acts: list[float] = []
        self.t0 = time.perf_counter()

    # ---------------------------------------------------------------- what app.py calls
    def headers(self) -> tuple:
        return self.res.headers()

    def think_budget(self, budget: int) -> int:
        """The request's reasoning budget after the plan: the client's own number always wins."""
        if self.budget is None:
            return budget
        return self.budget

    @property
    def decides_effort(self) -> bool:
        """The router sets this turn's effort level, so a client's effort word yields to it."""
        return self.active and self.decisions.get("effort") == "on"

    def apply_prompt(self, prompt, tok, in_think: bool):
        """The rendered prompt with the session's notes replayed, this turn's note added and, when
        the plan turned thinking off, the generation prompt's open `<think>` closed. Returns
        (the prompt to prefill, whether it ends inside `<think>`); the same objects when nothing
        changes. A failure here serves the prompt as rendered and says so in the journal row."""
        if not self.active or self.session is None:
            return prompt, in_think
        if not self.session.notes and self.block is None and not self.thinking_off and \
                not self.extras:
            return prompt, in_think
        try:
            return self._apply(prompt, tok, in_think)
        except Exception as exc:                                   # noqa: BLE001
            self.row["extra"].setdefault("errors", []).append(f"apply:{type(exc).__name__}")
            print(f"[conductor] the prompt was served as rendered: {type(exc).__name__}: {exc}",
                  flush=True)
            return prompt, in_think

    def _apply(self, prompt, tok, in_think: bool):
        import torch
        sess, cfg = self.session, self.router_cfg or {}
        inj_cfg = cfg.get("injection") or {}
        ids = prompt.tolist()
        self.clean_ids = ids
        tail_ids, closed_ids = None, None
        if self.thinking_off and in_think:
            # the template's own shape for `enable_thinking: false`, spliced at the tail instead of
            # rendered: the head (and its reasoning_effort sentence) stays token for token, and
            # the next turn renders the answer's empty think block the same way
            tail_ids = list(tok(inject.THINK_OPEN, add_special_tokens=False).input_ids)
            closed_ids = list(tok(inject.THINK_OFF, add_special_tokens=False).input_ids)
            if ids[-len(tail_ids):] != tail_ids:
                tail_ids = None
                self.row["extra"].setdefault("notes", []).append("thinking_off:no_open_tail")
        im_end = inject.special_id(tok, inject.IM_END)
        at = inject.insertion_point(ids, im_end)
        notes, adds = [], []
        if at is None:
            if sess.notes or self.block is not None or self.extras:
                self.row["extra"].setdefault("errors", []).append("no_insertion_point")
        else:
            with sess.lock:
                live, hashes = self._live(ids, at)
                cap = int(inj_cfg.get("max_note_tokens", 8000))
                keys = {n.key for n in live}
                if self.block is not None and self.block["key"] not in keys:
                    text = cfg["injection"]["wrapper"].replace("{text}", self.block["text"])
                    nids = list(tok(text, add_special_tokens=False).input_ids)
                    if len(nids) > cap:
                        self.row["extra"].setdefault("errors", []).append(
                            f"note_too_long:{len(nids)}")
                    else:
                        adds.append((Injection(at, hashes[at], nids, text, self.block["parts"],
                                               self.block["kinds"], self.block["key"], sess.turns,
                                               self.row["id"]), self.block["reason"]))
                # J1: the outcome's note and the cards, after the class's block; a card goes in
                # only while no live note carries the same card (the first turn, after a
                # compaction), an outcome's note once
                parts = {(p["kind"], p["name"]) for n in live for p in n.parts}
                for x in self.extras:
                    if x["key"] in keys or (x.get("unless") and tuple(x["unless"]) in parts):
                        continue
                    text = cfg["injection"]["wrapper"].replace("{text}", x["text"])
                    nids = list(tok(text, add_special_tokens=False).input_ids)
                    if len(nids) > cap:
                        self.row["extra"].setdefault("errors", []).append(
                            f"note_too_long:{len(nids)}")
                        continue
                    adds.append((Injection(at, hashes[at], nids, text, x["parts"], x["kinds"],
                                           x["key"], sess.turns, self.row["id"]), x["reason"]))
                    parts |= {(p["kind"], p["name"]) for p in x["parts"]}
                notes = live + [n for n, _ in adds]
                sess.notes = notes
            self._probe_at = (at, notes)
        if not notes and tail_ids is None:
            return prompt, in_think
        out, starts = inject.splice(ids, [(n.pos, n.ids) for n in notes])
        self.spans = [(starts[i], starts[i] + len(n.ids), sorted(n.kinds),
                       [p["name"] for p in (n.parts or []) if p.get("kind") == "package"])
                      for i, n in enumerate(notes)]
        if notes:
            replayed = notes[:len(notes) - len(adds)]
            self.row["replayed"] = {"count": len(replayed),
                                    "tokens": sum(len(n.ids) for n in replayed),
                                    "positions": [n.pos for n in replayed]}
        for i, (new, reason) in enumerate(adds):
            self.injected.append({"pos": new.pos, "at": starts[len(notes) - len(adds) + i],
                                  "tokens": len(new.ids), "text": new.text, "parts": new.parts,
                                  "kinds": sorted(new.kinds), "reason": reason})
            self.row["injected"] = self.injected
            self.row["acted"] = 1
        if tail_ids is not None:
            out = out[:len(out) - len(tail_ids)] + closed_ids
            in_think = False
            self.row["acted"] = 1
        self.changed = True
        self.row["changed"] = 1
        return torch.tensor(out, dtype=prompt.dtype, device=prompt.device), in_think

    def _live(self, ids: list[int], at: int) -> tuple[list, dict]:
        """The session's earlier notes this prompt still carries the text before (their
        decision types on, the newest within `max_replay_tokens`), and the prefix hashes.
        Called with the session's lock held."""
        inj_cfg = (self.router_cfg or {}).get("injection") or {}
        sess = self.session
        positions = [n.pos for n in sess.notes if n.pos <= at] + [at]
        hashes = inject.hashes_at(ids, positions)
        live = []
        for n in sess.notes:
            if n.pos > at or hashes.get(n.pos) != n.hash:
                continue                          # the prompt parted before the note
            if not inj_cfg.get("replay", True):
                continue
            if any(self.decisions.get(k) != "on" for k in n.kinds):
                continue                          # its decision type was turned off
            live.append(n)
        budget = int(inj_cfg.get("max_replay_tokens", 16000))
        while live and sum(len(n.ids) for n in live) > budget:
            live.pop(0)                           # the oldest goes first
        return live, hashes

    def replayed_ids(self, ids: list[int], tok) -> list[int]:
        """The client's prompt with the session's earlier notes replayed where they sit, and no
        note of this turn: the prompt the resident prefix holds up to the last message, which
        the in-context stage asks its question after. Nothing in the session changes."""
        if self.session is None or not self.session.notes:
            return list(ids)
        at = inject.insertion_point(ids, inject.special_id(tok, inject.IM_END))
        if at is None:
            return list(ids)
        with self.session.lock:
            live, _ = self._live(ids, at)
        return inject.splice(ids, [(n.pos, n.ids) for n in live])[0] if live else list(ids)

    def probe_row(self, prompt, tok) -> int | None:
        """The served prompt's position of the last message's last token, after
        `apply_prompt` (server/probe.py): from the insertion point and the notes spliced before
        it, or, for a prompt the plan did not change, from the prompt's own tail."""
        if self._probe_at is not None:
            return inject.content_row(*self._probe_at)
        tail = prompt[-64:].tolist()
        at = inject.insertion_point(tail, inject.special_id(tok, inject.IM_END))
        return None if at is None else inject.content_row(int(prompt.numel()) - len(tail) + at)

    def clean(self, prompt_ids: list[int]) -> list[int]:
        """The prompt as the client sent it, without notes: what the suffix store may keep."""
        return self.clean_ids if self.clean_ids is not None else prompt_ids

    def finish(self, rec, journal) -> None:
        """The turn ended: its outcome goes into the row, the row into the journal, and its prompt
        tokens into the session's size (a compaction request's prompt is not the session's)."""
        if self.row is None:
            return
        pt = getattr(rec, "prompt_tokens", None)
        if self.session is not None and pt and not self.compaction:
            with self.session.lock:
                if self.session.prompt_first is None:
                    self.session.prompt_first = int(pt)
                self.session.prompt_last = int(pt)
        self.row["outcome"] = {
            "status": getattr(rec, "status", None), "finish_reason": getattr(rec, "finish_reason", None),
            "tool_calls": int(getattr(rec, "tool_calls", 0) or 0),
            "tool_names": list(getattr(rec, "tool_names", None) or [])[:8],
            "completion_tokens": getattr(rec, "completion_tokens", None),
            "reasoning_tokens": getattr(rec, "reasoning_tokens", None),
            "prompt_tokens": getattr(rec, "prompt_tokens", None),
            "ttft_ms": round(rec.ttft_ms, 1) if getattr(rec, "ttft_ms", None) else None,
            "total_ms": round(rec.total_ms, 1) if getattr(rec, "total_ms", None) else None,
        }
        ui = getattr(rec, "states", None)
        if ui is not None and ui.log:
            self.row["extra"]["states"] = [
                {"kind": e["kind"], "text": e["text"], "at_ms": e["at_ms"],
                 # P1: the narrator's line and what happened (absent with the narrator off)
                 **({"summary": e["summary"], "narrator": e["narrator"]}
                    if "narrator" in e else {})} for e in ui.log]
        self.row.setdefault("acted", 0)
        self.row.setdefault("changed", 0)
        journal.close_row(self.row)


class Router:
    def __init__(self, store, library, journal, *, raw_model=lambda: "qwen38-spark-engine",
                 self_url=lambda: None, server_effort=lambda: None, global_mode=lambda: None,
                 factory=None, probe=lambda: None, monitor_session=lambda key: None):
        self.store, self.library, self.journal = store, library, journal
        # J1: the session monitor's state for a session key (its acts count against the caps)
        self.monitor_session = monitor_session
        self.probe = probe                  # server/probe.py ProbeStage, or None (ENG-181)
        self.raw_model, self.self_url = raw_model, self_url
        self.server_effort, self.global_mode = server_effort, global_mode
        self.factory = factory or self._make
        self._sessions: collections.OrderedDict[str, Session] = collections.OrderedDict()
        self._texts: collections.OrderedDict = collections.OrderedDict()
        self._adapters: dict = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- classifiers
    def _make(self, kind: str, role: dict, timeout_ms: int):
        key = (kind, role.get("endpoint"), role.get("model"), timeout_ms)
        a = self._adapters.get(key)
        if a is not None:
            return a
        endpoint = role.get("endpoint") or ""
        if endpoint == "self" and kind != "decider":
            # Qwen through this server's own port; a decider named `self` is asked in process
            endpoint = self.self_url() or ""
        if not endpoint:
            return None
        a = (decider_for(endpoint, role.get("model") or "auto", timeout_ms) if kind == "decider"
             else Qwen(endpoint, role.get("model") or self.raw_model(), timeout_ms))
        if a is None:
            return None
        self._adapters[key] = a
        return a

    def _stage_adapter(self, stage: dict, roles: dict, qid: str | None = None):
        role = roles.get("decider" if stage["kind"] == "decider" else "classifier") or {}
        if role.get("mode", "off") == "off":
            return None, "role_off"
        if stage["kind"] == "decider" and qid:
            # a model the `deciders` document names for this question; `auto` leaves the role's
            _, dec = self.store.peek("deciders")
            named = dec["questions"].get(qid)
            if named and named != "auto":
                role = dict(role, model=named)
        a = self.factory(stage["kind"], role, int(stage["timeout_ms"]))
        return (a, None) if a is not None else (None, "no_endpoint")

    def cascade(self, text: str, options: dict[str, str], router: dict, roles: dict, *,
                question: str | None = None, accept: float | None = None,
                qid: str = "classify", line: str | None = None) -> tuple:
        """Ask the enabled stages in order until one answers an option at or above its threshold.
        Returns (the accepted Answer or None, every stage's answer as dicts). `text` is the user's
        message; each stage gets its own digest of it (the decider's sized to its `limits()`,
        Qwen's to `digest_chars`), with the context `line` at its end when given. `qid`: the
        question id (`classify`, `packages`) the `deciders` document maps to a decision model."""
        token = ASKING.set(qid)
        try:
            return self._cascade(text, options, router, roles, question, accept, qid, line)
        finally:
            ASKING.reset(token)

    def _cascade(self, text, options, router, roles, question, accept, qid, line) -> tuple:
        clf = router["classifier"]
        chars = int(clf.get("digest_chars", 2000))
        stages, accepted = [], None
        for st in clf["stages"]:
            if not st.get("enabled", True):
                continue
            a, why = self._stage_adapter(st, roles, qid)
            if a is None:
                stages.append(Answer(st["kind"], error=f"{st['kind']}_{why}").as_dict())
                continue
            if st["kind"] == "decider":
                room = a.state_chars() if hasattr(a, "state_chars") else chars
                ans = a.choice(signals.digest(text, min(chars, room), line),
                               question or clf["question"], options)
            else:
                listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
                ans = a.choice(signals.digest(text, chars, line),
                               clf["qwen_prompt"].replace("{classes}", listing), options,
                               max_chars=chars)
            d = ans.as_dict()
            need = accept if accept is not None else float(st["accept"])
            d["accept"] = need
            if ans.error is None and ans.label not in options:
                d["error"] = f"{st['kind']}_unknown_label"
            elif ans.error is None and ans.confidence < need:
                d["below"] = True
            elif ans.error is None:
                accepted = ans
            stages.append(d)
            if accepted is not None:
                break
        return accepted, stages

    # ---------------------------------------------------------------- sessions
    def session(self, key: str) -> Session:
        with self._lock:
            s = self._sessions.get(key)
            if s is None:
                s = self._sessions[key] = Session(key)
            self._sessions.move_to_end(key)
            while len(self._sessions) > SESSIONS_MAX:
                self._sessions.popitem(last=False)
            return s

    def sessions_live(self) -> list[dict]:
        with self._lock:
            ss = list(self._sessions.values())
        return [{"key": s.key, "class_id": s.cls, "confidence": s.conf, "source": s.source,
                 "turns": s.turns, "notes": len(s.notes),
                 "note_tokens": sum(len(n.ids) for n in s.notes)} for s in ss]

    def _cached(self, key):
        with self._lock:
            v = self._texts.get(key)
            if v is not None:
                self._texts.move_to_end(key)
            return v

    def _remember(self, key, value) -> None:
        with self._lock:
            self._texts[key] = value
            while len(self._texts) > TEXT_CACHE_MAX:
                self._texts.popitem(last=False)

    # ---------------------------------------------------------------- the plan
    def plan(self, body: dict, headers, *, client_kind: str, request_id: str,
             chat: bool = True, model: str | None = None, dry: bool = False,
             count_tokens=None) -> Plan:
        """`dry`: the test box's run, on a throwaway session and without a journal row.
        `count_tokens()`: the prompt's length in tokens, asked only when the in-context stage
        needs it."""
        res = resolve(self.store, str(model or body.get("model") or self.raw_model()),
                      self.raw_model(), headers, client_kind, self.global_mode())
        plan = Plan(res)
        if res.mode == "raw" or not chat:
            return plan
        prof = res.doc
        router_v, router = self.store.peek("router")
        _, packages = self.store.peek("packages")
        _, roles = self.store.peek("roles")
        _, settings = self.store.peek("settings")
        plan.active, plan.router_cfg = True, router
        dec = {d: _lower(prof["decisions"].get(d, "off"), res.mode) for d in schema.DECISIONS}
        dec = {d: ("off" if m == "raw" else m) for d, m in dec.items()}
        plan.decisions = dec
        classes = {c["id"]: c for c in router["classes"]}
        sh = signals.shape(body, tuple(packages["packages"]))
        sess = Session("dry") if dry else self.session(session_key(body, headers, sh))
        plan.session = sess
        plan.compaction = sh.compaction
        if not dry:
            plan.body = body
            plan.transcript = transcripts.Transcripts.scope_of(
                transcripts.settings_of(settings), sh.tools, sh.compaction)
        digest_scope = settings["journal"]["digest"]
        keep_text = digest_scope == "all" or (digest_scope == "tools" and sh.tools)
        # SRV-153: the context line, from the session as it stood before this turn
        line = None
        if router["classifier"].get("context_line") and sh.user_text.strip() and \
                not sh.compaction:
            facts = signals.line_facts(sh)
            with sess.lock:
                line = signals.context_line(sess.cls, sess.conf, facts["tools"], facts["exts"],
                                            facts["checker"], sess.compacted)
        row = {"request_id": request_id, "session_key": sess.key, "client_kind": client_kind,
               "model": body.get("model"), "profile": res.profile,
               "profile_version": res.version, "router_version": router_v, "mode": res.mode,
               "decisions": dict(dec), "stages": [], "shadow": None, "injected": [],
               "replayed": None, "acted": 0, "changed": 0, "extra": {},
               "digest": signals.digest(sh.user_text, int(router["classifier"]["digest_chars"]),
                                        line) if keep_text and sh.user_text else None}
        if line:
            row["extra"]["context_line"] = line
        plan.row = row
        # the previous turn's outcome label: the checker result this request carries
        if sh.results and sess.last_row and not dry:
            self.journal.update(sess.last_row, next_checker=sh.results)

        # ---- the class
        t0 = time.perf_counter()
        default = prof["default_class"] if prof["default_class"] in classes else \
            router["classes"][0]["id"]
        cls, conf, source, kept, stages = None, None, None, None, []
        structural = router["structural"]
        if dec["classify"] == "off":
            cls, source = default, "profile"
        elif sh.compaction and structural["compaction"] in classes:
            cls, conf, source = structural["compaction"], 1.0, "structural"
        elif sh.tool_turn and sess.cls in classes and structural["tool_turn"] == "keep":
            # SRV-161: a tool's media in a user message is a tool turn too, journaled as such
            cls, conf, source = sess.cls, sess.conf, "sticky"
            kept = "tool_media" if sh.tool_media else None
        elif sh.user_hash is not None and sh.user_hash == sess.user_hash and sess.cls in classes:
            cls, conf, source = sess.cls, sess.conf, "sticky"
        elif not sh.user_text.strip():
            cls, source, kept = (sess.cls if sess.cls in classes else default), "fallback", \
                "no_text"
        else:
            options = {c: classes[c]["description"] for c in classes}
            with sess.lock:
                sess.compacted = False             # the line said so once
            due = None if dry else self._context_due(router, body, sh, count_tokens)
            if due is not None and self._probe_first(router):
                due = dict(due, probe=True)
            if due is not None:
                # Qwen decides inside the conversation, under the engine lock right before the
                # turn's own prefill (`decide_in_context`); the rest of the plan waits for it
                row["extra"]["context_stage"] = due
                plan.context = {"t0": t0, "options": options, "due": due,
                                "args": (body, router, prof, packages, roles, sh, sess, classes,
                                         default, dec, dry)}
                return plan
            # the deciders document says which model answers; a change there re-asks, and so
            # does another context line
            ck = (router_v, self.store.peek("deciders")[0],
                  hashlib.sha256(sh.user_text.encode()).hexdigest(), line)
            hit = self._cached(ck)
            if hit is not None:
                ans, stages = hit
                stages = [dict(s, cached=True) for s in stages]
            else:
                ans, stages = self.cascade(sh.user_text, options, router, roles, line=line)
                if ans is not None or all(s.get("error") is None for s in stages):
                    self._remember(ck, (ans, stages))
            cls, conf, source, kept = self._pick(ans, stages, router, sess, classes, default)
        return self._conclude(plan, t0, cls, conf, source, kept, stages, body, router, prof,
                              packages, roles, sh, sess, classes, default, dec, dry)

    # ---------------------------------------------------------------- Qwen in the conversation
    def _context_due(self, router, body, sh, count_tokens) -> dict | None:
        """Whether this new user message goes to the in-context stage instead of the digest
        cascade: the stage is on, and the message follows an earlier answer in the conversation
        (`followups`) or the prompt has at least `min_tokens`. A short first message stays the
        decider's. Images go through the stage as the turn's prompt carries them."""
        cs = schema.context_stage(router)
        if cs["mode"] != "on" or count_tokens is None:
            return None
        msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        if cs["followups"] and sh.user_index is not None and any(
                m.get("role") == "assistant" for m in msgs[:sh.user_index]):
            return {"due": "followup"}
        try:
            n = int(count_tokens())
        except Exception:                                          # noqa: BLE001
            return None                     # the render failed: the cascade, as before
        return {"due": "long", "tokens": n} if n >= int(cs["min_tokens"]) else None

    def _probe_first(self, router) -> bool:
        """The probe stage decides first on this plan's lock path: mode on, its file loaded."""
        ps = schema.probe_stage(router)
        st = self.probe()
        return ps["mode"] == "on" and st is not None and st.ready(ps)

    def _probe_stage(self, router, prompt_ids, tok, rows, plan, due) -> dict:
        """The probe on a prefill of the turn's prompt up to its insertion point (`rows(ids)`
        hands back that prefill's last rows, their names and its counts)."""
        ps, st = schema.probe_stage(router), self.probe()
        t0 = time.perf_counter()
        try:
            ids = plan.replayed_ids(prompt_ids, tok)
            at = inject.insertion_point(ids, inject.special_id(tok, inject.IM_END))
            if at is None or at < 1:
                raise incontext.ContextError("no_insertion_point")
            vec, names, info = rows(ids[:at])
            d = st.stage(ps, vec, names, reason=due)
            d.update(prompt_tokens=at, prefilled=int(info.get("forwarded", 0)),
                     reused=int(info.get("reused", 0)))
        except ConnectionError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            d = {"stage": "probe", "label": None, "confidence": 0.0, "probs": {}, "ms": 0.0,
                 "error": "probe_" + getattr(exc, "why", type(exc).__name__), "model": ps["file"],
                 "accept": float(ps["accept"]), "reason": due}
        d["stage_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        return d

    def decide_in_context(self, plan: "Plan", prompt_ids: list[int], tok, runner,
                          rows=None) -> None:
        """The in-context stage, under the engine lock before the turn's prefill: build its
        prompt from the turn's (the session's notes replayed), let `runner(ids, letter_ids)`
        prefill it and hand back the letters' logits with its counts, and finish the plan from
        the answer. Every failure is the stage's error in the journal, and the class falls back
        as the cascade's does when nobody answered. With the probe stage on (ENG-181) the probe
        answers first, on `rows(ids)`'s prefill up to the insertion point; at or above its
        threshold the question is not asked."""
        ctx, plan.context = plan.context, None
        if ctx is None:
            return
        body, router, prof, packages, roles, sh, sess, classes, default, dec, dry = ctx["args"]
        cs = schema.context_stage(router)
        t0 = time.perf_counter()
        stages = []
        if ctx["due"].get("probe") and rows is not None:
            try:
                pd = self._probe_stage(router, prompt_ids, tok, rows, plan, ctx["due"]["due"])
            except ConnectionError as exc:
                pd = {"stage": "probe", "label": None, "confidence": 0.0, "probs": {}, "ms": 0.0,
                      "error": "probe_client_gone", "model": None, "reason": ctx["due"]["due"]}
                cls, conf, source, kept = self._pick(None, [pd], router, sess, classes, default)
                self._conclude(plan, t0, cls, conf, source, kept, [pd], body, router, prof,
                               packages, roles, sh, sess, classes, default, dec, dry)
                raise exc
            stages.append(pd)
            if pd["error"] is None and not pd.get("below") and pd["label"] in classes:
                ans = Answer("probe", pd["label"], pd["confidence"], pd["probs"])
                cls, conf, source, kept = self._pick(ans, stages, router, sess, classes, default)
                self._conclude(plan, t0, cls, conf, source, kept, stages, body, router, prof,
                               packages, roles, sh, sess, classes, default, dec, dry)
                return
        tq = time.perf_counter()
        stage = {"stage": "qwen_context", "label": None, "confidence": 0.0, "probs": {},
                 "ms": 0.0, "error": None, "model": None, "accept": float(cs["accept"]),
                 "reason": ctx["due"]["due"]}
        ans, gone = None, None
        try:
            text, letters = incontext.listing(ctx["options"])
            ids, at = incontext.prompt(plan.replayed_ids(prompt_ids, tok), tok,
                                       cs["question"].replace("{classes}", text))
            lids = incontext.letter_ids(tok, letters)
            logits, info = runner(ids, lids, at)
            label, conf, probs = incontext.answer(logits, list(ctx["options"]))
            stage.update(label=label, confidence=round(conf, 4),
                         probs={k: round(v, 4) for k, v in probs.items()},
                         prompt_tokens=len(ids), prefilled=int(info.get("forwarded", 0)),
                         reused=int(info.get("reused", 0)), model=info.get("model"))
            if conf < float(cs["accept"]):
                stage["below"] = True
            else:
                ans = Answer("qwen_context", label, conf, probs)
        except ConnectionError as exc:
            # the client left during the stage's prefill: the row is finished, then the request
            # ends as any request whose client left
            stage["error"], gone = "qwen_context_client_gone", exc
        except Exception as exc:                                   # noqa: BLE001
            stage["error"] = "qwen_context_" + getattr(exc, "why", type(exc).__name__)
            if getattr(exc, "why", None) is None:
                print(f"[conductor] the in-context stage failed: {type(exc).__name__}: {exc}",
                      flush=True)
        stage["ms"] = round((time.perf_counter() - tq) * 1e3, 2)
        stages.append(stage)
        cls, conf, source, kept = self._pick(ans, stages, router, sess, classes, default)
        self._conclude(plan, t0, cls, conf, source, kept, stages, body, router, prof, packages,
                       roles, sh, sess, classes, default, dec, dry)
        if gone is not None:
            raise gone

    def settle_context(self, plan: "Plan", why: str = "not_run") -> None:
        """A plan whose in-context stage never ran (the request ended before the engine lock):
        its row gets the session's class, as when nobody answered."""
        ctx, plan.context = plan.context, None
        if ctx is None:
            return
        body, router, prof, packages, roles, sh, sess, classes, default, dec, dry = ctx["args"]
        stages = [{"stage": "qwen_context", "label": None, "confidence": 0.0, "probs": {},
                   "ms": 0.0, "error": f"qwen_context_{why}", "model": None,
                   "reason": ctx["due"]["due"]}]
        cls, conf, source, kept = self._pick(None, stages, router, sess, classes, default)
        self._conclude(plan, time.perf_counter(), cls, conf, source, kept, stages, body, router,
                       prof, packages, roles, sh, sess, classes, default, dec, dry)

    def _pick(self, ans, stages, router, sess, classes, default) -> tuple:
        """The class from the accepted answer (or none): the session's when nobody was sure or
        the answer is below the switch, else the answer's."""
        if ans is None:
            return (sess.cls if sess.cls in classes else default), None, "fallback", (
                "fail_open" if stages and all(s.get("error") for s in stages) else "no_answer")
        if sess.cls in classes and sess.cls != ans.label and \
                ans.confidence < float(router["stickiness"]["switch"]):
            return sess.cls, sess.conf, "sticky", "below_switch"
        return ans.label, ans.confidence, ans.stage, None

    def _conclude(self, plan, t0, cls, conf, source, kept, stages, body, router, prof, packages,
                  roles, sh, sess, classes, default, dec, dry) -> Plan:
        """The class is known: the session follows it, and the policy, the effort, the note and
        the journal row come from it."""
        row = plan.row
        row["classifier_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        row["stages"] = stages
        model_cls, model_conf, model_source = cls, conf, source
        if dec["classify"] == "shadow":
            # journal the class; the policy runs as the profile's default class
            row["shadow"] = {"class_id": cls, "confidence": conf, "source": source}
            cls, source = default, "profile"
        row.update(class_id=cls, confidence=conf, source=source, kept=kept)
        with sess.lock:
            # the session follows the model's class in shadow too, so shadow's stickiness is
            # the one `on` would have
            # a compaction request is its own kind of turn; the work after it goes on in the
            # session's class, so the structural class never becomes the session's
            sess.turns += 1
            if sh.compaction:
                sess.compacted = True
            follow = dec["classify"] != "off" and model_source != "structural"
            if follow and model_cls != sess.cls:
                sess.history.append({"at": time.time(), "from": sess.cls, "to": model_cls,
                                     "confidence": model_conf, "source": model_source})
            if follow:
                sess.cls, sess.conf, sess.source = model_cls, model_conf, model_source
            if sh.user_hash is not None:
                sess.user_hash = sh.user_hash

        # ---- the policy
        base = classes[cls]["policy"]
        ov = prof["routing"].get(cls) or {}
        policy = {"effort": ov.get("effort") or base.get("effort") or prof["thinking"]["default"],
                  "packages": list(ov["packages"] if "packages" in ov else base.get("packages", [])),
                  "rules": list(ov["rules"] if "rules" in ov else base.get("rules", [])),
                  "note": ov.get("note") or base.get("note") or "",
                  "effort_by_size": ov["effort_by_size"] if "effort_by_size" in ov
                  else base.get("effort_by_size")}
        self._size(plan, router, prof, sh, sess, policy, row)
        if not dry:
            self._observe(plan, body, router, prof, sh, sess, row)
        level = self._turn_level(plan, router, prof, sh, policy, row)
        self._effort(plan, body, router, prof, sh, self._outcome_level(plan, level, row), row)
        self._block(plan, body, router, roles, packages, sh, sess, cls, policy, row)
        if not dry:
            self._cards(plan, body, router, prof, packages, sh, cls, row)
            self._outcome_note(plan, body, router, prof, packages, sh, sess, row)
        if dry:
            return plan
        self.journal.open_row(row)
        with sess.lock:
            sess.last_row = row["id"]
        return plan

    def explain(self, text: str, profile: str | None = None, *, tools: bool = True,
                messages: list | None = None) -> dict:
        """The dashboard's test box: the whole decision for a pasted prompt, with the exact note,
        touching no session and writing no journal row."""
        body = {"model": COMPOSITE,
                "messages": messages or [{"role": "user", "content": text}]}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": "bash", "description": "run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
        headers = {"X-Tandem-Profile": profile} if profile else {}
        plan = self.plan(body, headers, client_kind="dashboard", request_id="explain", dry=True)
        out = {"profile": plan.res.profile, "profile_version": plan.res.version,
               "mode": plan.res.mode, "active": plan.active}
        if not plan.active:
            return out
        row = plan.row
        cfg = plan.router_cfg
        note = (cfg["injection"]["wrapper"].replace("{text}", plan.block["text"])
                if plan.block else None)
        out.update({k: row.get(k) for k in ("class_id", "confidence", "source", "kept", "stages",
                                             "classifier_ms", "effort", "budget", "budget_source",
                                             "thinking", "decisions", "shadow", "extra")})
        out["note"] = note
        out["note_parts"] = plan.block["parts"] if plan.block else []
        out["thinking_off"] = plan.thinking_off
        return out

    @staticmethod
    def size_signals(sh: signals.Shape, sess: Session) -> dict:
        """SRV-157's signals, all structural: the prompt tokens the session added after its first
        turn (opencode's own head, about 24k tokens, is not the task's size), the distinct files
        its tool calls named, the most gates one known checker's RESULT line named (None: no
        known checker spoke yet, so it does not vote), and the turn index."""
        added = 0
        if sess.prompt_first is not None and sess.prompt_last is not None:
            added = max(0, sess.prompt_last - sess.prompt_first)
        return {"session_tokens": added, "files": len(sh.files), "checks": sh.max_checks,
                "turns": sess.turns}

    def _size(self, plan, router, prof, sh, sess, policy, row) -> None:
        """SRV-157: the size of the task picks the effort within the class's ceiling. Each signal
        is small below its `small` threshold, medium up to its `medium` one, large above; the
        task's size is the largest any signal reached, and it never falls within the task (a
        compaction starts it over). With the profile's `thinking.by_size` on, the class's
        `effort_by_size` level for that size (never above the class's effort) becomes the turn's
        effort; in shadow it is journaled beside the class's effort, which the turn keeps."""
        mode = (prof.get("thinking") or {}).get("by_size", schema.BY_SIZE_DEFAULT)
        cfg = router["effort"].get("size") or schema.ROUTER["effort"]["size"]
        if sh.compaction:
            if cfg.get("reset_on_compaction", True):
                with sess.lock:
                    sess.size_tier = None
            return
        if mode == "off" or plan.decisions["effort"] == "off":
            return
        sig = self.size_signals(sh, sess)
        tiers = {}
        for name, v in sig.items():
            c = cfg["signals"].get(name)
            if not c or not c.get("enabled", True) or v is None:
                continue
            tiers[name] = 0 if v < c["small"] else 1 if v <= c["medium"] else 2
        now = max(tiers.values(), default=0)
        with sess.lock:
            tier = max(now, sess.size_tier or 0)
            sess.size_tier = tier
        info = {"mode": mode, "size": schema.SIZES[tier], "score": tier,
                "now": schema.SIZES[now],
                "signals": {n: {"value": sig[n], "size": schema.SIZES[t]}
                            for n, t in tiers.items()}}
        by = policy.get("effort_by_size")
        if by:
            ceiling = policy["effort"]
            want = by.get(schema.SIZES[tier]) or ceiling
            picked = want if EFFORT_ORDER.index(want) <= EFFORT_ORDER.index(ceiling) else ceiling
            info.update(ceiling=ceiling, effort=picked, applied=mode == "on")
            if mode == "on":
                policy["effort"] = picked
            else:
                row["shadow"] = dict(row.get("shadow") or {}, size={
                    "size": info["size"], "score": tier, "effort": picked, "ceiling": ceiling})
        row["extra"]["size"] = info

    @staticmethod
    def _turn_level(plan, router, prof, sh, policy, row) -> str:
        """The level this turn thinks at: the class's (after its size), or on a turn whose newest
        message is a tool result (SRV-161's tool media included, never a compaction) the
        `tool_turn_effort` level when one is set (F2). The first turn and every new user message
        keep the class's. Only the thinking switch and the cap follow it; the note is still the
        class's, so the prompt the session's notes sit in does not change."""
        want = schema.tool_turn_effort(router, prof)
        if want == "inherit" or not sh.tool_turn or sh.compaction or \
                plan.decisions["effort"] == "off":
            return policy["effort"]
        row["extra"].update(effort_source="tool_turn", class_effort=policy["effort"])
        return want

    def _effort(self, plan, body, router, prof, sh, level, row) -> None:
        dec = plan.decisions["effort"]
        raised = False
        if router["effort"].get("raise_on_fail") and prof["thinking"].get("hints") and \
                level in ("low", "medium", "high") and any(r["verdict"] == "FAIL"
                                                            for r in sh.results):
            level = EFFORT_ORDER[EFFORT_ORDER.index(level) + 1]
            raised = True
        lv = router["effort"]["levels"][level]
        # the profile's own cap for the level, else the router's (0 = none below the backstop)
        own = (prof["thinking"].get("budgets") or {}).get(level)
        cap = int(lv["budget"] if own is None else own)
        # the client's own number, in any shape (server/thinking.py), always wins; its effort
        # word or switch yields to the level, and an explicit on or off is the client's
        client = thinking.read(body, strict=False)
        client_budget = client.budget if client.kind == "number" else None
        client_thinking = client.thinking
        budget, source, thinking_on = None, "server", "on"
        if client_budget is not None:
            source = "client"
        if not lv["thinking"]:
            thinking_on = "client" if client_thinking is not None else "off"
        elif client_budget is None and cap > 0:
            budget, source = cap, "class" if own is None else "profile"
        would = {"effort": level, "budget": budget, "budget_source": source,
                 "thinking": thinking_on, "raised_on_fail": raised}
        if dec == "on":
            plan.budget = budget
            plan.budget_source = source
            plan.thinking_off = thinking_on == "off"
            if budget is not None:
                row["acted"] = 1
                plan.changed = True
            row.update(effort=level, budget=budget, budget_source=source, thinking=thinking_on)
        else:
            row.update(effort=None, budget=None, budget_source="client" if client_budget is not None
                       else "server", thinking="client" if client_thinking is not None else "on")
            if dec == "shadow":
                row["shadow"] = dict(row.get("shadow") or {}, effort=would)

    def _block(self, plan, body, router, roles, packages, sh, sess, cls, policy, row) -> None:
        """The note this class wants in the prompt: its effort sentence, its package, its rules.
        Parts whose decision type is on go into `plan.block`; shadow parts are journaled."""
        dec = plan.decisions
        parts: list[tuple[str, dict]] = []                # (decision type, part)
        base_level = policy["effort"]
        note = router["effort"]["levels"][base_level]["note"]
        if dec["effort"] != "off" and note:
            parts.append(("effort", {"kind": "effort", "name": base_level, "text": note}))
        cands = [p for p in policy["packages"] if p not in sh.loaded_skills]
        skipped = [p for p in policy["packages"] if p in sh.loaded_skills]
        if skipped:
            row["extra"].setdefault("notes", []).append(f"already_loaded:{','.join(skipped)}")
        # a general package (the coding package, 2026-10-02) goes in whole and first; the router
        # picks among the rest, the task skills, as before
        general = [p for p in cands if (packages["packages"].get(p) or {}).get("general")]
        cands = [p for p in cands if p not in general]
        if dec["packages"] != "off":
            for g in general:
                text, err = self.library.text(packages, g)
                if text is None:
                    row["extra"].setdefault("notes", []).append(f"{err}:{g}")
                else:
                    parts.append(("packages", {"kind": "package", "name": g, "text": text}))
            if general:
                row["extra"]["package_general"] = general
        if dec["packages"] != "off" and cands:
            chosen, how = None, None
            if len(cands) == 1:
                chosen, how = cands[0], "only_candidate"
            else:
                pk = (cls, sh.user_hash)
                hit = sess.package_pick.get(pk)
                if hit is not None:
                    chosen, how = hit
                else:
                    opts = {p: packages["packages"][p]["description"] for p in cands
                            if p in packages["packages"]}
                    opts[NONE_OPTION] = "None of these fits the request."
                    ans, st = self.cascade(
                        sh.user_text or "", opts, router, roles,
                        question=router["classifier"]["package_question"],
                        accept=float(router["classifier"]["package_accept"]), qid="packages")
                    row["extra"]["package_stages"] = st
                    if ans is not None and ans.label != NONE_OPTION:
                        chosen, how = ans.label, ans.stage
                    else:
                        how = "none" if ans is not None else "no_answer"
                    if len(sess.package_pick) > 32:
                        sess.package_pick.clear()
                    sess.package_pick[pk] = (chosen, how)
            row["extra"]["package_choice"] = {"candidates": cands, "chosen": chosen, "how": how}
            if chosen:
                text, err = self.library.text(packages, chosen)
                if text is None:
                    row["extra"].setdefault("notes", []).append(f"{err}:{chosen}")
                else:
                    parts.append(("packages", {"kind": "package", "name": chosen, "text": text}))
        if dec["rules"] != "off" and (policy["rules"] or policy["note"]):
            text = "\n".join(f"- {r}" for r in policy["rules"])
            if policy["note"]:
                text = (text + "\n\n" if text else "") + policy["note"]
            parts.append(("rules", {"kind": "rules", "name": cls, "text": text}))
        on = [(k, p) for k, p in parts if dec[k] == "on"]
        shadow = [(k, p) for k, p in parts if dec[k] == "shadow"]
        if shadow:
            row["shadow"] = dict(row.get("shadow") or {}, notes=[
                {"decision": k, "kind": p["kind"], "name": p["name"], "chars": len(p["text"])}
                for k, p in shadow])
        if not on:
            return
        key = hashlib.sha256(repr([(cls, p["kind"], p["name"],
                                    hashlib.sha256(p["text"].encode()).hexdigest()[:12])
                                   for _, p in on]).encode()).hexdigest()[:20]
        reason = (f"class {cls} ({row.get('source')}"
                  + (f", {row['confidence']:.2f}" if row.get("confidence") is not None else "")
                  + ")")
        plan.block = {"key": key, "text": "\n\n".join(p["text"] for _, p in on),
                      "parts": [{"kind": p["kind"], "name": p["name"]} for _, p in on],
                      "kinds": {k for k, _ in on}, "reason": reason}

    # ---------------------------------------------------------------- J1: snapshots, outcomes
    def _observe(self, plan, body, router, prof, sh, sess, row) -> None:
        """The snapshot and the Tier 0 outcome of a turn whose newest message is a tool result
        (server/conductor/outcomes.py), under the profile's `outcomes` switches; all off (every
        profile stored before J1) leaves the plan untouched. A compaction starts the re-read
        watch. A failure is the row's error and the turn goes on as without it."""
        so = schema.profile_outcomes(prof)
        if all(v == "off" for v in so.values()):
            return
        if so["mode"] == "act" or so["cards"] == "on" or so["evaluate"] == "act" or \
                so["retrieve"] == "on":
            plan.decisions["outcomes"] = "on"       # their notes replay
        if not (sh.tool_turn or sh.compaction) or (so["snapshots"] == "off" and
                                                    so["mode"] == "off" and so["judge"] == "off"):
            return
        cfg = schema.outcomes(router)
        now = time.time()
        with sess.lock:
            if sess.evidence is None:
                sess.evidence = outcomes.Evidence(now)
            ev = sess.evidence
            if sh.compaction:
                outcomes.compaction(ev)
                return
            try:
                snap = outcomes.observe(body, sh, ev, cfg, now=now,
                                        keep_media=so["snapshots"] == "on")
                label, why = outcomes.classify(snap, ev, cfg)
                sig = outcomes.settle(ev, label, why)
            except Exception as exc:                               # noqa: BLE001
                row["extra"].setdefault("errors", []).append(f"snapshot:{type(exc).__name__}")
                print(f"[conductor] the snapshot failed: {type(exc).__name__}: {exc}", flush=True)
                return
        snap["outcome"] = {"label": label, "why": why, "mode": so["mode"]}
        if so["judge"] == "shadow":
            from server.conductor import judge
            plan.judge = judge.questions(snap, cfg)        # asked after the request (step 7)
        snap["_narrow_kept"] = snap.pop("_narrow", None) or []
        if so["snapshots"] == "on":
            plan.snapshot = snap
        plan.outcome = {"label": label, "why": why, "sig": sig, "snap": snap, "cfg": cfg,
                        "so": so, "now": now, "effort": None, "note": None,
                        "modes": self._label_modes(so, cfg, label)}
        if so["mode"] != "off":
            row["extra"]["outcome"] = {"label": label, "mode": so["mode"], "why": why[:2],
                                       "ms": snap["ms"]}
        ac = cfg["evaluate"].get("ask") or {}
        if so["evaluate"] != "off" and ac.get("mode", "off") == "shadow" and \
                any(n["q"] == "failure" for n in snap.get("_narrow_kept") or ()):
            # step 6: Qwen asked in place on the narrowed case, under the engine lock before the
            # turn's own prefill (`ask_in_place`), journaled beside the twin
            n = [x for x in snap["_narrow_kept"] if x["q"] == "failure"][-1]
            plan.ask = {"twin": n.get("twin"), "call": n.get("call")}
        snap.pop("_narrow_kept", None)

    # the outcomes the evaluation points (E-c, step 6) and retrieval (step 8) answer
    EVALUATED = ("stuck", "low_quality")

    @staticmethod
    def _label_modes(so, cfg, label) -> dict:
        """What an outcome may do under the profile's switches: `note` and `effort`, each off,
        shadow or act. The plan's step 3 labels (`acting`) follow `outcomes.mode`; stuck and
        low quality follow `evaluate` (the E-c note, xhigh) and `retrieve` (the package)."""
        out = {"note": "off", "effort": "off", "evaluate": "off", "retrieve": "off"}
        mode = so["mode"] if so["mode"] != "off" else "off"
        if label in cfg["acting"]:
            out["note"] = mode
        if cfg["effort"].get(label):
            out["effort"] = mode
        if label in Router.EVALUATED:
            ev = so["evaluate"]
            rt = {"on": "act", "shadow": "shadow"}.get(so["retrieve"], "off")
            out["evaluate"], out["retrieve"] = ev, rt
            best = max((ev, rt, out["note"]), key=("off", "shadow", "act").index)
            out["note"] = best if best != "off" else out["note"]
            if ev != "off":
                out["effort"] = ev
        return out

    def _outcome_level(self, plan, level: str, row) -> str:
        """The effort after the outcome: the outcome's level (never below the turn's), held for
        `raise_turns` tool turns or until a progress. Shadow journals the level it would set."""
        o = plan.outcome
        if o is None or o["so"]["mode"] == "off":
            return level
        cfg, ev = o["cfg"], plan.session.evidence
        lv = cfg["effort"].get(o["label"]) if o["modes"]["effort"] != "off" else None
        if o["modes"]["evaluate"] != "off":
            lv = cfg["evaluate"]["effort"]
        with plan.session.lock:
            if lv:
                ev.raised = {"level": lv, "left": max(0, int(cfg["raise_turns"]) - 1),
                             "act": o["modes"]["effort"] == "act"}
            elif ev.raised and ev.raised["left"] > 0:
                lv = ev.raised["level"]
                ev.raised["left"] -= 1
            else:
                ev.raised = None
        if not lv or EFFORT_ORDER.index(lv) <= EFFORT_ORDER.index(level):
            return level
        applied = (o["modes"]["effort"] == "act" or (ev.raised or {}).get("act", False)) and \
            plan.decisions["effort"] == "on"
        o["effort"] = {"from": level, "to": lv, "applied": applied}
        row["extra"]["outcome"]["effort"] = o["effort"]
        if not applied:
            row["shadow"] = dict(row.get("shadow") or {}, outcome={"effort": lv, "from": level})
            return level
        return lv

    def _acts_free(self, plan, prof, now: float) -> str | None:
        """None when a note may go in under the profile's caps (2 a request, 6 a session hour,
        the monitor's acts of the session counted), else the cap's name."""
        caps = prof["caps"]
        if int(caps["acts_per_request"]) <= len(plan.acts):
            return "per_request"
        sess = plan.session
        mine = sum(1 for t in sess.acts if now - t < 3600.0)
        try:
            ms = self.monitor_session(sess.key)
            theirs = ms.acts_last_hour(now) if ms is not None else 0
        except Exception:                                          # noqa: BLE001
            theirs = 0
        if max(mine, theirs) >= int(caps["acts_per_session_hour"]):
            return "per_session_hour"
        return None

    def _outcome_note(self, plan, body, router, prof, packages, sh, sess, row) -> None:
        """The outcome's note (act: injected once per signature, under the caps; shadow: what
        would have gone in), and the outcome's journal row (`kind: outcome`)."""
        o = plan.outcome
        if o is None or o["so"]["mode"] == "off":
            return
        cfg, label, why, ev = o["cfg"], o["label"], o["why"], sess.evidence
        nmode = o["modes"]["note"]
        note = {"due": nmode != "off" and label in cfg["notes"], "applied": False,
                "mode": nmode}
        if note["due"]:
            with sess.lock:
                seen = o["sig"] in ev.noted
            text = self._note_text(plan, body, router, packages, sh, sess, label, why, o,
                                   cfg, row, note) if not seen else None
            if seen:
                note["skipped"] = "noted"
            elif text is None:
                note["skipped"] = "no_text"
            else:
                note["chars"] = len(text)
                cap = self._acts_free(plan, prof, o["now"]) if nmode == "act" else None
                if cap is not None:
                    note["cap"] = cap
                else:
                    with sess.lock:
                        ev.noted.add(o["sig"])
                    if nmode == "act":
                        plan.extras.append({"key": f"outcome:{row.get('request_id')}:"
                                                   f"{sess.turns}", "text": text,
                                            "parts": [{"kind": "outcome", "name": label}],
                                            "kinds": {"outcomes"},
                                            "reason": f"outcome {label}"})
                        plan.acts.append(o["now"])
                        with sess.lock:
                            sess.acts.append(o["now"])
                        note["applied"] = True
                    else:
                        row["shadow"] = dict(row.get("shadow") or {}, outcome=dict(
                            (row.get("shadow") or {}).get("outcome") or {}, note=label,
                            chars=len(text)))
        o["note"] = note
        row["extra"]["outcome"]["note"] = {k: v for k, v in note.items() if k != "due"} \
            if note["due"] else None
        self.journal.note("outcome", "outcome", decision=label, session_key=sess.key,
                          request_id=row.get("request_id"),
                          detail={"mode": o["so"]["mode"], "why": why, "effort": o["effort"],
                                  "note": note, "class": row.get("class_id"),
                                  "snapshot": outcomes.facts_row(o["snap"])})

    def _note_text(self, plan, body, router, packages, sh, sess, label, why, o, cfg, row,
                   note) -> str | None:
        """The outcome's note: its template filled; for stuck and low quality the E-c re-plan
        note (evaluate) and the retrieved package (retrieve), as their switches say."""
        snap, modes = o["snap"], o["modes"]
        tmpl = cfg["notes"].get(label)
        if label in self.EVALUATED and modes["evaluate"] != "off":
            tmpl = cfg["notes"]["stuck"]                         # E-c: re-plan from the evidence
            if modes["retrieve"] != "off":
                tmpl += "\n\n{package}"
        elif label in self.EVALUATED and modes["retrieve"] == "off":
            tmpl = tmpl if "{package}" not in (tmpl or "") else None
        if not tmpl:
            return None
        fill = {"why": "; ".join(why[:2]) or label.replace("_", " "),
                "since": f"{snap['snapshot'] - (plan.session.evidence.progress_snap or 0)} tool "
                         f"turns"}
        if "{digest}" in tmpl:
            fill["digest"] = outcomes.digest(snap, label, why, cfg)
        if "{card}" in tmpl:
            fill["card"] = self._tool_card(body, cfg, packages, row)
        if "{package}" in tmpl:
            name, text = self._retrieve(router, packages, sh, sess, label, o, cfg, row)
            note["package"] = name
            if text is None:
                return None
            fill["package"] = text
        for k, v in fill.items():
            tmpl = tmpl.replace("{" + k + "}", v)
        return tmpl.strip()

    def _retrieve(self, router, packages, sh, sess, label, o, cfg, row) -> tuple:
        """Step 8: the package that fits the session's work, among `retrieve.packages` (the
        kind skills), by the router's package question (packages.fit: the decider or Qwen picks a
        description; no word decides), once per outcome signature. (name, text) or (None,
        None)."""
        rc = cfg["retrieve"]
        cands = [p for p in rc["packages"] if (packages["packages"].get(p) or {}).get(
            "enabled", False)]
        if not cands:
            return None, None
        pk = ("retrieve", o["sig"])
        hit = sess.package_pick.get(pk)
        if hit is None:
            _, roles = self.store.peek("roles")
            opts = {p: packages["packages"][p]["description"] for p in cands}
            opts[NONE_OPTION] = "None of these fits the work."
            text = sh.user_text or ""
            line = "files=" + ",".join(dict.fromkeys(
                e for e in (signals.ext_of(f) for f in sh.files[-8:]) if e)) or None
            ans, st = self.cascade(text, opts, router, roles, question=rc["question"],
                                   accept=float(router["classifier"]["package_accept"]),
                                   qid="packages", line=line)
            row["extra"]["retrieve_stages"] = st
            hit = ans.label if ans is not None and ans.label != NONE_OPTION else None
            if len(sess.package_pick) > 32:
                sess.package_pick.clear()
            sess.package_pick[pk] = hit
        if hit is None:
            return None, None
        return hit, self.library.text(packages, hit)[0]

    def _tool_card(self, body, cfg, packages, row) -> str:
        client = row.get("client_kind") or "other"
        name = cfg["cards"]["pitfalls"].get(client)
        pit = self.library.text(packages, name)[0] if name else None
        return outcomes.tool_card(body, cfg, pit, client)

    def _cards(self, plan, body, router, prof, packages, sh, cls, row) -> None:
        """The client tool card (B1) and the harness card (B2) for a coding class with tools:
        kept live in the session's notes (in again after a compaction dropped them), each once.
        Shadow journals what would go in."""
        so = schema.profile_outcomes(prof)
        if so["cards"] == "off" or not sh.tools or sh.compaction or \
                plan.decisions["packages"] == "off":
            return
        cfg = schema.outcomes(router)
        cc = cfg["cards"]
        if cls not in cc["classes"]:
            return
        items = [("tools", self._tool_card(body, cfg, packages, row))]
        harness, err = self.library.text(packages, cc["harness"])
        if harness is not None:
            items.append(("harness", harness))
        elif err:
            row["extra"].setdefault("notes", []).append(f"{err}:{cc['harness']}")
        if so["cards"] == "shadow":
            row["shadow"] = dict(row.get("shadow") or {}, cards=[
                {"name": n, "chars": len(t)} for n, t in items])
            return
        for name, text in items:
            plan.extras.insert(0 if name == "harness" else len(plan.extras), {
                "key": f"card:{name}:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
                "text": text, "parts": [{"kind": "card", "name": name}],
                "kinds": {"outcomes"}, "reason": f"card {name}", "unless": ("card", name)})

    def ask_in_place(self, plan: "Plan", prompt_ids: list[int], tok, runner) -> dict | None:
        """Step 6, shadow: Qwen asked in place on the turn's narrowed case (a run's output with
        no exit code and no runner's counts), under the engine lock before the turn's own
        prefill, through the in-context stage's forward (`runner(ids, letter ids, at)`: the
        prompt up to its last message, the session's notes replayed, plus one hidden question;
        the turn then prefills over these rows). The answer is journaled beside the Tier 0 twin
        (`kind: outcome`, task `in_place`); it changes nothing. A failure is the row's error."""
        ask, plan.ask = plan.ask, None
        o = plan.outcome
        if ask is None or o is None:
            return None
        ac = o["cfg"]["evaluate"]["ask"]
        t0 = time.perf_counter()
        d = {"question": "failure", "twin": ask.get("twin"), "call": ask.get("call"),
             "error": None}
        label = "unknown"
        try:
            text, letters = incontext.listing(ac["options"])
            ids, at = incontext.prompt(plan.replayed_ids(prompt_ids, tok), tok,
                                       ac["question"] + "\n\n" + text)
            logits, info = runner(ids, incontext.letter_ids(tok, letters), at)
            got, conf, probs = incontext.answer(logits, list(ac["options"]))
            d.update(confidence=round(conf, 4), probs={k: round(v, 4) for k, v in probs.items()},
                     prefilled=int(info.get("forwarded", 0)), reused=int(info.get("reused", 0)))
            if conf >= float(ac["accept"]):
                label = got
                d["agree"] = (got == ask.get("twin")) if ask.get("twin") else None
        except ConnectionError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            d["error"] = getattr(exc, "why", None) or type(exc).__name__
        d["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        if plan.row is not None:
            plan.row["extra"].setdefault("outcome", {})["in_place"] = {
                "label": label, "ms": d["ms"], "error": d["error"]}
        return self.journal.note("outcome", "in_place", decision=label, detail=d,
                                 session_key=plan.session.key if plan.session else None,
                                 request_id=(plan.row or {}).get("request_id"))
