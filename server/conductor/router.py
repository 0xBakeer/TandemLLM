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
from server.conductor import incontext, inject, schema, signals, transcripts
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
                 "prompt_first", "prompt_last")

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
        if not self.session.notes and self.block is None and not self.thinking_off:
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
        notes, new = [], None
        if at is None:
            if sess.notes or self.block is not None:
                self.row["extra"].setdefault("errors", []).append("no_insertion_point")
        else:
            with sess.lock:
                live, hashes = self._live(ids, at)
                if self.block is not None and self.block["key"] not in {n.key for n in live}:
                    text = cfg["injection"]["wrapper"].replace("{text}", self.block["text"])
                    nids = list(tok(text, add_special_tokens=False).input_ids)
                    cap = int(inj_cfg.get("max_note_tokens", 8000))
                    if len(nids) > cap:
                        self.row["extra"].setdefault("errors", []).append(
                            f"note_too_long:{len(nids)}")
                    else:
                        new = Injection(at, hashes[at], nids, text, self.block["parts"],
                                        self.block["kinds"], self.block["key"], sess.turns,
                                        self.row["id"])
                notes = live + ([new] if new is not None else [])
                sess.notes = notes
        if not notes and tail_ids is None:
            return prompt, in_think
        out, starts = inject.splice(ids, [(n.pos, n.ids) for n in notes])
        if notes:
            replayed = [n for n in notes if n is not new]
            self.row["replayed"] = {"count": len(replayed),
                                    "tokens": sum(len(n.ids) for n in replayed),
                                    "positions": [n.pos for n in replayed]}
        if new is not None:
            self.injected.append({"pos": new.pos, "at": starts[-1], "tokens": len(new.ids),
                                  "text": new.text, "parts": new.parts, "kinds": sorted(new.kinds),
                                  "reason": self.block["reason"]})
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
                 factory=None):
        self.store, self.library, self.journal = store, library, journal
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

    def decide_in_context(self, plan: "Plan", prompt_ids: list[int], tok, runner) -> None:
        """The in-context stage, under the engine lock before the turn's prefill: build its
        prompt from the turn's (the session's notes replayed), let `runner(ids, letter_ids)`
        prefill it and hand back the letters' logits with its counts, and finish the plan from
        the answer. Every failure is the stage's error in the journal, and the class falls back
        as the cascade's does when nobody answered."""
        ctx, plan.context = plan.context, None
        if ctx is None:
            return
        body, router, prof, packages, roles, sh, sess, classes, default, dec, dry = ctx["args"]
        cs = schema.context_stage(router)
        t0 = time.perf_counter()
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
        stage["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        stages = [stage]
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
        self._effort(plan, body, router, prof, sh, policy, row)
        self._block(plan, body, router, roles, packages, sh, sess, cls, policy, row)
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

    def _effort(self, plan, body, router, prof, sh, policy, row) -> None:
        dec = plan.decisions["effort"]
        level = policy["effort"]
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

