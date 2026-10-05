"""The memory notes in a request: what the router puts in, and what the model's blocks write
(2026-10-05; the design: Memo "Memory for TandemLLM: design advice", sections 2 and 3).

A profile's `memory` switch (server/conductor/schema.py, absent: off):

  off      nothing: no card, no filter on the stream, no read. The request is the release's.
  shadow   the card goes in (the model may write) and its blocks are stored; nothing is read back,
           so Khaled can read what a model would remember before it is read back to it.
  on       the card, the writes, and the three reads below.

THE CARD. `tandem-memory` (packages/inline/tandem-memory.md, about 150 tokens): the block's shape
and one sentence on when to write one. A tail note kept live in the session like the harness
card: in once, replayed, in again after a compaction.

THE READS (on), each a tail note that the router's replay keeps at its position, so each costs its
prefill once:
  - the index, once a session (the first turn, and again after a compaction): the pinned ratified
    rules (at most `pinned_tokens`), then one line per note this request may see (user, project,
    session), at most `index_tokens`. Notes written later in the session do not join it: that
    would move the tail; the model knows them because it wrote them.
  - the bodies, on a new user message (never a tool turn, never a compaction): BM25 over the
    message's digest, the top `bodies` notes above `min_score` that share at least `min_words`
    of the message's words (BM25's scale moves with the number of notes; a shared-word floor
    does not), within `body_tokens`, each headed
    by its id, kind, status and date. A note already in the session's notes is not offered again.
    Their note has its own frame (BODIES_WRAPPER): the notes are the user's, so the model may
    say what one says, and the head asks it to answer from a note that answers the message.
  - the `memory` skill (packages/skills/memory), when Laya says the message asks to remember,
    forget or recall something (`intent`; shadow journals the answer only). No word list.

THE WRITES. `Memory.take()` is the stream's `MemFilter` callback (server/memblock.py): a closed
block is parsed and written (server/conductor/notes.py) with the request's user, project and
session, and its `from_user` (a tool turn cannot write a rule about the user); a refusal sends the
block out as text. A write is one journal row (`kind: memory_notes`, task `write`): op, id, scope,
kind, why, the redaction counts. Never a title or a body: they stay in the files.

THE USER. `owner` on the engine; a request the proxy relayed names the proxy user (the proxy's key
label in process, `X-Tandem-User` from a trusted proxy address). None: nothing is read or written.
The project is the working directory opencode writes into its system prompt (`Working directory:`),
never a tool argument or the model's text.
"""

from __future__ import annotations

import collections
import hashlib
import os
import threading
import time

from server.conductor import notes as notes_mod
from server.conductor import packages as packages_mod
from server.conductor import schema, signals
from server.conductor.transcripts import WORKDIR
from server import memblock
from server.conductor.libsearch import words as signals_words

CARD = "tandem-memory"
SKILL = "memory"
_SHIPPED = {CARD: os.path.join(packages_mod.SHIPPED_ROOT, "packages", "inline", "tandem-memory.md"),
            SKILL: os.path.join(packages_mod.SHIPPED_ROOT, "packages", "skills", "memory",
                                "SKILL.md")}
CHARS_PER_TOKEN = 3.5
INTENT_QUESTION = ("Does this message ask the assistant to remember something, forget something, "
                   "or recall what it remembered earlier?")
INTENT_OPTIONS = {"yes": "Yes: it asks to remember, forget or recall something.",
                  "no": "No: it asks for something else."}
SESSIONS_MAX = 4096
INDEX_HEAD = ("Your memory: notes the runtime keeps for this user{project}. The user can read and "
              "edit them. Provisional notes are yours, not confirmed by the user. Read one in "
              "full: GET {tandem_url}/v1/memory/<id>; search: "
              "GET {tandem_url}/v1/memory/search?q=<words>.")
BODIES_HEAD = ("Notes from your memory that match this message. When one answers it, answer from "
               "the note and say it comes from your notes; a ratified note is confirmed by the "
               "user. Read files only for what the notes do not say, and when they disagree with "
               "a note, tell the user both.")
# The bodies' own frame (0.3.39-test). The router's wrapper says "do not mention it", which kept
# the model from naming a note it was given: CLIP1's fresh session (2026-10-05 21:35) had the note
# attached and answered from the files. The notes are the user's, so this frame lets the model
# say what a note says. Same shape as the wrapper: two newlines first, one runtime-note.
BODIES_WRAPPER = ("\n\n<runtime-note>\nThe runtime that serves you adds this note from your "
                  "memory. The user does not see it, but the notes are the user's: tell the user "
                  "what a note says when it answers the message.\n\n{text}\n</runtime-note>")


def project_of(sh) -> str | None:
    m = WORKDIR.search(getattr(sh, "system_text", None) or "")
    return m.group(1).rstrip("/") if m else None


def _tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


class Memory:
    """One per conductor: the note store, the sessions' users and projects (for the model's
    routes, which a session token addresses), and the request side."""

    def __init__(self, store: notes_mod.NoteStore, journal=None, settings=lambda: {}):
        self.store, self.journal, self.settings = store, journal, settings
        self.sessions: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.stats = {"blocks": 0, "taken": 0, "failed": 0, "index_notes": 0, "body_notes": 0,
                      "intent_yes": 0}
        from server.conductor.memwriter import Writer
        self.writer = Writer(self)

    def cfg(self) -> dict:
        try:
            return schema.notes(self.settings())
        except Exception:                                          # noqa: BLE001
            return dict(schema.NOTES)

    def seen(self, key: str, user: str, project: str | None) -> None:
        with self._lock:
            self.sessions[key] = {"user": user, "project": project, "at": time.time()}
            self.sessions.move_to_end(key)
            while len(self.sessions) > SESSIONS_MAX:
                self.sessions.popitem(last=False)

    def session(self, key: str | None) -> dict | None:
        with self._lock:
            return dict(self.sessions[key]) if key in self.sessions else None

    # ---------------------------------------------------------------- the model's writes
    def take(self, ctx: dict, raw: str, ui=None) -> str | None:
        """One closed block of the request whose memory context is `ctx` (Plan.memory): None when
        it was written (hidden from the client), else why not (it goes out as text)."""
        self.stats["blocks"] += 1
        op, why, res = None, None, None
        try:
            op = memblock.parse(raw)
            if op["op"] == "propose":
                props = getattr(self, "proposals", None)
                if props is None:
                    raise notes_mod.Refused("no_proposals", "package proposals are off here")
                res = props.propose(op, user=ctx["user"], session=ctx["session"],
                                    request_id=ctx.get("request_id"),
                                    evidence=self.evidence(ctx.get("session")))
            else:
                res = self.store.write(ctx["user"], op, writer=ctx.get("writer", "qwen-inline"),
                                       project=ctx.get("project"), session=ctx.get("session"),
                                       request_id=ctx.get("request_id"),
                                       from_user=bool(ctx.get("from_user")))
        except memblock.BlockError as exc:
            why = exc.why
        except notes_mod.Refused as exc:
            why = exc.why
        except Exception as exc:                                   # noqa: BLE001
            why = "error"
            print(f"[memory] a block failed: {type(exc).__name__}: {exc}", flush=True)
        detail = {"op": (op or {}).get("op"), "why": why, "writer": ctx.get("writer",
                                                                           "qwen-inline"),
                  "chars": len(raw)}
        if res is not None:
            detail.update({k: res.get(k) for k in ("id", "scope", "kind", "revised", "redacted",
                                                    "name", "n", "kept") if res.get(k)})
        ctx.setdefault("written", []).append(detail)
        if why is None:
            self.stats["taken"] += 1
            if ui is not None and not res.get("kept"):     # a ratified note it doubted: no line
                title = (res.get("name") if detail["op"] == "propose" else res.get("title")) or ""
                kind = ("forgot" if detail["op"] == "archive" else
                        "proposed" if detail["op"] == "propose" else "remembered")
                ui.add(kind, title=_line(title))
        else:
            self.stats["failed"] += 1
        if self.journal is not None:
            self.journal.note("memory_notes", "write", decision=why or detail["op"],
                              detail=dict(detail, user=ctx.get("user")),
                              session_key=ctx.get("session"), request_id=ctx.get("request_id"))
        return why

    def evidence(self, session: str | None) -> list[str]:
        """The J1 outcome rows of the session that showed a gap (low quality, stuck): what a
        package proposal cites; [] when there are none."""
        if not session or self.journal is None:
            return []
        try:
            rows = self.journal.rows(kind="outcome", session=session, limit=200)
        except Exception:                                          # noqa: BLE001
            return []
        return [r["id"] for r in rows if r.get("decision") in ("low_quality", "stuck")][:8]

    def filter(self, ctx: dict | None, ui=None) -> memblock.MemFilter | None:
        """The stream's filter for a request whose memory is not off."""
        if not ctx or ctx.get("mode") not in ("on", "shadow"):
            return None
        return memblock.MemFilter(lambda raw: self.take(ctx, raw, ui))


def _line(title: str) -> str:
    t = " ".join(str(title).split())
    return t if len(t) <= 80 else t[:77].rstrip() + "..."


# ------------------------------------------------------------------------ the router's part
def _text(router, packages: dict, name: str) -> str | None:
    """A memory package's text: the packages document's, else the shipped file (a box whose
    document has no such package yet)."""
    if name in (packages.get("packages") or {}):
        text, _ = router.library.text(packages, name)
        if text:
            return text
    path = _SHIPPED.get(name)
    try:
        with open(path) as f:
            text = f.read()
    except (OSError, TypeError):
        return None
    _, body = packages_mod.split_frontmatter(text)
    return packages_mod.with_url(body.strip(), getattr(router.library._req, "url", None))


def attach(router, plan, body, router_doc, prof, packages, roles, sh, sess, cls, row,
           dry: bool, user: str | None, cap: str | None = None) -> None:
    """The memory's part of a plan (the module's header). Nothing at all with the switch off."""
    mem = getattr(router, "memory", None)
    if mem is None:
        return
    mode = schema.profile_memory(prof)
    if plan.res.mode == "shadow" and mode == "on":
        mode = "shadow"
    if cap is not None and cap != "on":
        mode = "off" if cap in ("off", "raw") or mode == "off" else "shadow"
    user = notes_mod.user_id(user)
    if mode == "off" or plan.res.mode == "raw" or sh.compaction or not user or \
            not notes_mod.valid_user(user) or mem.store.repo:
        return
    project = project_of(sh)
    plan.memory = {"mode": mode, "user": user, "project": project, "session": sess.key,
                   "from_user": not sh.tool_turn, "request_id": row.get("request_id"),
                   "writer": "qwen-inline", "written": [],
                   # the background writer's digest of the session (memory only, never journaled)
                   "text": None if sh.tool_turn else (sh.user_text or "")[:1000]}
    plan.decisions["memory"] = "on"                  # its notes replay (card, index, bodies)
    info = {"mode": mode}
    row["extra"]["memory"] = info
    if dry:
        return
    mem.seen(sess.key, user, project)
    card = _text(router, packages, CARD)
    if card:
        plan.extras.append({"key": f"memory_card:{_h(card)}", "text": card,
                            "parts": [{"kind": "memory_card", "name": CARD}],
                            "kinds": {"memory"}, "reason": "memory card",
                            "unless": ("memory_card", CARD)})
    if mode != "on":
        return
    cfg = mem.cfg()
    t0 = time.perf_counter()
    statuses = notes_mod.STATUSES if cfg["provisional"] else ("ratified",)
    # the index, once a session (the `unless` keeps it out while a live note carries it)
    idx = index_text(mem.store, user, project, sess.key, cfg, statuses,
                     getattr(router.library._req, "url", None))
    if idx is not None:
        text, ids = idx
        plan.extras.append({"key": f"memory_index:{_h(text)}", "text": text,
                            "parts": [{"kind": "memory_index", "name": "index"}],
                            "kinds": {"memory"}, "reason": f"memory index ({len(ids)} notes)",
                            "unless": ("memory_index", "index")})
        info["index"] = {"notes": len(ids), "tokens": _tokens(text)}
        plan.memory["index_ids"] = ids
    # the bodies, on a new user message
    if not sh.tool_turn and (sh.user_text or "").strip() and int(cfg["bodies"]) > 0:
        live = _live_ids(sess)
        query = signals.digest(sh.user_text or "", int(cfg["digest_chars"]), None)
        hits = mem.store.search(user, query, project=project, session=sess.key,
                                k=int(cfg["bodies"]) + len(live), statuses=statuses)
        qw = set(signals_words(query))
        need = min(int(cfg["min_words"]), len(qw))
        hits = [(n, s) for n, s in hits if s >= float(cfg["min_score"]) and n.id not in live
                and len(qw & set(signals_words(f"{n.title} {' '.join(n.tags)} {n.body}")))
                >= need]
        hits = hits[:int(cfg["bodies"])]
        text, used = bodies_text(hits, int(cfg["body_tokens"]))
        info["bodies"] = [{"id": n.id, "score": round(s, 3)} for n, s in hits if n.id in used]
        if used:
            plan.extras.append({"key": f"memory_bodies:{_h(text)}", "text": text,
                                "parts": [{"kind": "memory", "name": i} for i in used],
                                "kinds": {"memory"}, "reason": f"memory ({len(used)} notes)",
                                "wrapper": BODIES_WRAPPER})
            plan.memory["body_ids"] = used
    info["search_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    _intent(router, plan, packages, roles, router_doc, sh, row, cfg, info)


def _intent(router, plan, packages, roles, router_doc, sh, row, cfg, info) -> None:
    """Laya's question on a new user message: does it ask to remember, forget or recall?
    Asked of the decider stages only (about 20 ms), never Qwen; on: a yes attaches the skill."""
    if cfg["intent"] == "off" or sh.tool_turn or not (sh.user_text or "").strip():
        return
    clf = router_doc["classifier"]
    stages = [dict(s, enabled=True) for s in clf["stages"] if s["kind"] == "decider"]
    if not stages:
        info["intent"] = {"answer": None, "why": "no_decider"}
        return
    doc = dict(router_doc, classifier=dict(clf, stages=stages))
    t0 = time.perf_counter()
    ans, st = router.cascade(sh.user_text or "", INTENT_OPTIONS, doc, roles,
                             question=INTENT_QUESTION, accept=0.6, qid="memory_intent")
    got = ans.label if ans is not None else None
    info["intent"] = {"answer": got, "mode": cfg["intent"],
                      "ms": round((time.perf_counter() - t0) * 1e3, 2),
                      **({"why": st[-1].get("error")} if ans is None and st else {})}
    if got != "yes":
        return
    router.memory.stats["intent_yes"] += 1
    if cfg["intent"] != "on":
        row["shadow"] = dict(row.get("shadow") or {}, memory_skill=SKILL)
        return
    text = _text(router, packages, SKILL)
    if text:
        plan.extras.append({"key": f"memory_skill:{_h(text)}", "text": text,
                            "parts": [{"kind": "memory_skill", "name": SKILL}],
                            "kinds": {"memory"}, "reason": "memory skill (intent)",
                            "unless": ("memory_skill", SKILL)})
        info["skill"] = True


def index_text(store, user, project, session, cfg, statuses, url) -> tuple[str, list] | None:
    """The index note: the pinned ratified rules in full, then the index lines, within their
    token caps. None when the user has no note this request may see."""
    groups = store.index_lines(user, project, session)
    if not groups:
        return None
    pinned = [n for n in store.notes(user) if n.pinned and n.status == "ratified" and
              store.visible(n, project, session)]
    parts = [INDEX_HEAD.replace("{project}", f" and the project {project}" if project else "")
             .replace("{tandem_url}", url or packages_mod.URL_DEFAULT)]
    ids = []
    room = int(cfg["pinned_tokens"])
    if pinned:
        lines = ["Pinned rules (confirmed by the user; follow them):"]
        for n in pinned:
            t = f"- [{n.id}] {n.title}: {' '.join((n.body or '').split())}"
            if _tokens(t) > room:
                break
            room -= _tokens(t)
            lines.append(t)
            ids.append(n.id)
        if len(lines) > 1:
            parts.append("\n".join(lines))
    room = int(cfg["index_tokens"])
    words = {"user": "About the user", "project": "This project", "session": "This conversation"}
    for scope, lines in groups:
        keep = []
        for ln in lines:
            if _tokens(ln) > room:
                keep.append("... more notes; search them")
                room = 0
                break
            room -= _tokens(ln)
            keep.append(ln)
            if ln.startswith("- ["):
                ids.append(ln[3:15])
        if keep:
            parts.append(f"{words[scope]}:\n" + "\n".join(keep))
        if room <= 0:
            break
    return "\n\n".join(parts), list(dict.fromkeys(ids))


def bodies_text(hits, budget: int) -> tuple[str, list]:
    out, used = [BODIES_HEAD], []
    for n, _ in hits:
        t = (f"[{n.id}] {n.kind} · {n.status} · {(n.updated or '')[:10]}\n{n.title}\n"
             f"{(n.body or '').strip()}")
        if _tokens(t) > budget:
            continue
        budget -= _tokens(t)
        out.append(t)
        used.append(n.id)
    return "\n\n".join(out), used


def _live_ids(sess) -> set:
    with sess.lock:
        return {p["name"] for n in sess.notes for p in (n.parts or ())
                if p.get("kind") == "memory"}


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def mark_injected(mem: "Memory | None", plan) -> None:
    """After the prompt was applied: the notes that went in count as used (the dashboard's
    `uses in 7 days`), and one journal row names them (ids only)."""
    ctx = getattr(plan, "memory", None)
    if mem is None or not ctx or ctx.get("mode") != "on":
        return
    went = {p.get("kind"): True for x in (plan.injected or []) for p in x.get("parts") or []}
    ids = []
    if went.get("memory_index"):
        ids += ctx.get("index_ids") or []
    if went.get("memory"):
        ids += ctx.get("body_ids") or []
    if not ids:
        return
    mem.store.mark_used(ctx["user"], list(dict.fromkeys(ids)))
    if mem.journal is not None:
        mem.journal.note("memory_notes", "injected", decision="index" if went.get(
            "memory_index") else "bodies", detail={"ids": list(dict.fromkeys(ids)),
                                                   "index": bool(went.get("memory_index")),
                                                   "bodies": ctx.get("body_ids") or [],
                                                   "user": ctx["user"]},
                         session_key=ctx.get("session"), request_id=ctx.get("request_id"))
    # each read counted as itself: a first turn carries the index and the bodies, and counting
    # both as the index read "body_notes 0" for a session whose note body went in (0.3.39-test)
    if went.get("memory_index"):
        mem.stats["index_notes"] += len(ctx.get("index_ids") or [])
    if went.get("memory"):
        mem.stats["body_notes"] += len(ctx.get("body_ids") or [])
