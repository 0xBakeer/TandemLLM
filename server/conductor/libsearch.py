"""The package library: a search index over its skills, and the router's retrieval from it (R1).

The owner, 2026-10-03: "multiple packages about many things and subjects so the model can search and
use them later". 44 skills were written (opencode's SKILL.md: frontmatter `name` and
`description`, the text that goes in, then an optional `## Reference` with snippets) and an
INDEX.md whose table names the router classes each skill fits. On the box they sit under
~/.qwen38-spark-engine/conductor/skills/library/<name>/SKILL.md, and each is a package of the
`packages` document with `library: true` and source `skill:<name>` (the library directory is one
of `skill_dirs`).

THE INDEX (`SkillIndex`): BM25 over every enabled library package's name, description and text
(the text before `## Reference`), the name's words counting three times and the description's
twice. Words are lower-case runs of letters and digits, a plural `s` dropped, a few function
words left out; no embeddings. It is rebuilt when the `packages` version moves, a SKILL.md's or
INDEX.md's mtime moves, or a file comes or goes, looked at at most once a second. A search over
the 44 skills takes tens of microseconds; a rebuild about two milliseconds.

THE RETRIEVAL (`attach`, called by the router at the end of a plan, under the profile's `library`
switch, `off` when absent): on a new user message (not a tool turn, not a compaction), and on a
tool turn whose outcome is one of the section's `outcomes` (J1's low quality and stuck), the
digest of the user's message (and of the outcome's reasons) is searched, the class's fit from
INDEX.md filtering the candidates. The top `candidates` go to the router's package question
(packages.fit: the decider or Qwen picks a description, or none; no word decides), up to
`max_picks` skills. `on`: each goes in as a note at the prompt's tail, the router's own mechanism,
replayed at its position by every later turn (exact cache), and a skill already live in the
session is not offered again. `shadow`: the row says what would have gone in. The `card` package
(how the model searches the library itself) is kept live like the harness card. A pick is asked
once per message (or outcome signature) and session; with `LIBRARY_HOLD=class`, on a class whose
package answer holds for the class, once per class (VIS-119). A plan decided under the engine lock
(the in-context stage) cannot ask Qwen through the server's own port, so it asks the decider stages
only, and journals `locked` when there is none. A shadow question bound for Qwen on the server's
own port is asked after the turn, when the server is idle (server/conductor/deferred.py); the
proxy's stage override (`--backends`) leaves this server's own requests on that port.

The row's `extra.library`: the trigger, the query's size, the candidates with their scores, the
picks and how they were made, the search's and the question's time. No text of the message.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
import types

from server.conductor import deferred, schema, signals
from server.conductor.classify import PARENT
from server.conductor.packages import SHIPPED_ROOT, split_frontmatter, split_reference
from server.knobs import SETTINGS as _S

#: VIS-119: `message` (the default): a library question on every new user message, as before;
#: `class`: on a class whose package answer holds for the session's class (`package_sticky:
#: class`), a new user message the class question keeps in that class asks none, the class's first
#: answer holds, picks and none alike. An outcome's question is asked as before
LIBRARY_HOLD = (_S.get("LIBRARY_HOLD") or "message").strip().lower()

K1, B = 1.2, 0.75
WEIGHTS = {"name": 3, "description": 2, "body": 1}
RECHECK_S = 1.0
# the repository's copy of the library's INDEX.md (packages/library), read when the configured one
# is not on the box: a new box serves the shipped library without an install step
SHIPPED_INDEX = os.path.join(SHIPPED_ROOT, "packages", "library", "INDEX.md")
NONE_OPTION = "none"
_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset((
    "a an and are as at be but by can do does for from has have how i if in into is it its me my "
    "of on or our so than that the their then there these this to up use using was we what when "
    "where which while who will with you your").split())


def words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in STOP:
            out.append(w)
    return out


def index_path(cfg: dict) -> str:
    """The INDEX.md the library reads: the configured file, else the shipped one when the
    configured directory does not exist at all (a box that never installed the library; a
    directory without its INDEX.md is read as one: every skill in every class)."""
    idx = os.path.expanduser(cfg.get("index") or "")
    if (idx and not os.path.exists(idx) and not os.path.isdir(os.path.dirname(idx))
            and os.path.exists(SHIPPED_INDEX)):
        return SHIPPED_INDEX
    return idx


def index_classes(text: str) -> dict[str, list[str]]:
    """name -> classes from INDEX.md's table rows `| name | description | body | ref | classes |`
    (the group rows have a bold name and no classes)."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "name" or \
                not re.match(r"^[a-z0-9][a-z0-9_.-]*$", cells[0]):
            continue
        cls = [c.strip() for c in cells[-1].split(",") if re.match(r"^[a-z][a-z0-9_-]*$",
                                                                    c.strip())]
        if cls:
            out[cells[0]] = cls
    return out


class Entry:
    __slots__ = ("name", "description", "body", "reference", "classes", "path", "enabled")

    def __init__(self, name, description, body, reference, classes, path, enabled):
        self.name, self.description, self.body = name, description, body
        self.reference, self.classes, self.path, self.enabled = reference, classes, path, enabled

    def card(self) -> dict:
        return {"name": self.name, "description": self.description, "classes": self.classes}


class TextIndex:
    """BM25 over a list of documents with weighted fields: the one index class the library's
    skills (`SkillIndex`) and the memory notes (server/conductor/notes.py) share. A document is
    a dict of field -> text; `weights` says how often a field's words count. Words are `words()`.
    Built whole (the corpora are hundreds of documents at most); `scores()` is one pass over the
    query words' postings."""

    def __init__(self):
        self._post: dict[str, list[tuple[int, float]]] = {}
        self._len: list[float] = []
        self._avg = 1.0
        self.n = 0

    def build(self, docs: list[dict], weights: dict) -> None:
        post: dict[str, dict[int, float]] = {}
        lens = []
        for i, d in enumerate(docs):
            tf: dict[str, float] = {}
            for field, w in weights.items():
                for t in words(d.get(field) or ""):
                    tf[t] = tf.get(t, 0.0) + w
            lens.append(sum(tf.values()))
            for t, f in tf.items():
                post.setdefault(t, {})[i] = f
        self._post = {t: list(d.items()) for t, d in post.items()}
        self._len = lens
        self._avg = (sum(lens) / len(lens)) if lens else 1.0
        self.n = len(docs)

    def scores(self, query: str) -> list[float]:
        """Every document's BM25 score for `query` (each query word once)."""
        n = self.n
        score = [0.0] * n
        if not n:
            return score
        for t in set(words(query)):
            post = self._post.get(t)
            if not post:
                continue
            idf = math.log(1.0 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for i, f in post:
                score[i] += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * self._len[i] / self._avg))
        return score


class SkillIndex:
    """BM25 over the library packages; `library` reads their files (packages.Library)."""

    def __init__(self, library, clock=time.monotonic):
        self.library, self.clock = library, clock
        self.entries: list[Entry] = []
        self.by_name: dict[str, Entry] = {}
        self._sig = None
        self._checked = -1e9
        self.text = TextIndex()
        self.builds = 0
        self.build_ms = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def names(doc: dict) -> list[str]:
        return [n for n, p in (doc.get("packages") or {}).items() if p.get("library")]

    def _signature(self, version, doc: dict, cfg: dict) -> tuple:
        sig = [version]
        for n in self.names(doc):
            path = self.library.path(doc, n)
            try:
                sig.append((n, path, os.stat(path).st_mtime_ns) if path else (n, None, None))
            except OSError:
                sig.append((n, path, None))
        idx = index_path(cfg)
        try:
            sig.append(os.stat(idx).st_mtime_ns if idx else None)
        except OSError:
            sig.append(None)
        return tuple(sig) + (idx,)

    def ensure(self, version, doc: dict, cfg: dict | None = None) -> None:
        """Rebuild when the documents or the files moved (looked at once a second at most)."""
        cfg = cfg if cfg is not None else schema.LIBRARY
        now = self.clock()
        if self._sig is not None and self._sig[0] == version and now - self._checked < RECHECK_S:
            return
        with self._lock:
            self._checked = now
            sig = self._signature(version, doc, cfg)
            if sig == self._sig:
                return
            self._build(doc, cfg)
            self._sig = sig

    def _build(self, doc: dict, cfg: dict) -> None:
        t0 = time.perf_counter()
        classes = {}
        idx = index_path(cfg)
        if idx:
            classes = index_classes(self.library.file_text(idx) or "")
        entries = []
        for n in self.names(doc):
            p = doc["packages"][n]
            if p.get("source", "inline") == "inline":
                meta, raw, path = {}, p.get("body") or "", None
            else:
                path = self.library.path(doc, n)
                text = self.library.file_text(path) if path else None
                if text is None:
                    continue
                meta, raw = split_frontmatter(text)
            body, ref = split_reference(raw)
            entries.append(Entry(n, p.get("description") or meta.get("description", ""),
                                 body.strip(), ref.strip(), classes.get(n), path,
                                 bool(p.get("enabled", True))))
        self.text.build([{"name": e.name.replace("-", " "), "description": e.description,
                          "body": e.body} for e in entries], WEIGHTS)
        self.entries = entries
        self.by_name = {e.name: e for e in entries}
        self.builds += 1
        self.build_ms = round((time.perf_counter() - t0) * 1e3, 3)

    def search(self, query: str, k: int = 5, cls: str | None = None, exclude=(),
               min_score: float = 0.0, enabled_only: bool = True) -> list[tuple[Entry, float]]:
        """The `k` best entries for `query` (each query word once), above `min_score` and above
        0; with `cls`, only entries whose INDEX.md classes name it (an entry INDEX.md does not
        list stays in)."""
        if not self.entries:
            return []
        score = self.text.scores(query)
        out = []
        for i, s in enumerate(score):
            e = self.entries[i]
            if s <= 0 or s < min_score or e.name in exclude or (enabled_only and not e.enabled):
                continue
            if cls is not None and e.classes is not None and cls not in e.classes:
                continue
            out.append((e, s))
        out.sort(key=lambda x: (-x[1], x[0].name))
        return out[:max(0, int(k))]


# ------------------------------------------------------------------------ the router's retrieval
def _live_names(sess) -> set[str]:
    with sess.lock:
        return {p["name"] for n in sess.notes for p in (n.parts or ())
                if p.get("kind") == "library"}


def _ask(router, plan, cands, text, line, cfg, router_doc, roles,
         about: str | None = None) -> tuple[list, str, list]:
    """The package question over the candidates, `max_picks` times (each time without the picks
    before it), until it answers none. (picks, how, the stages' answers)."""
    if plan.locked:
        clf = router_doc["classifier"]
        stages = [s for s in clf["stages"] if s["kind"] != "qwen" and s.get("enabled", True)]
        if not stages:
            return [], "locked", []
        router_doc = dict(router_doc, classifier=dict(clf, stages=stages))
    if getattr(plan, "fail_open", False):
        # SRV-763: the class question could not be answered in time; this one would wait as long
        return [], "fail_open", []
    picks, how, asked = [], "none", []
    left = [e for e, _ in cands]
    for _ in range(int(cfg["max_picks"])):
        if not left:
            break
        opts = {e.name: e.description for e in left}
        opts[NONE_OPTION] = "None of these fits the request."
        ans, st = router.cascade(text, opts, router_doc, roles, question=cfg["question"],
                                 accept=float(cfg["accept"]), qid="packages", line=line,
                                 about=about)
        asked.append(st)
        if ans is None:
            how = how if picks else "no_answer"
            break
        if ans.label == NONE_OPTION:
            break
        picks.append(ans.label)
        how = ans.stage
        left = [e for e in left if e.name != ans.label]
    return picks, how, asked


def _hold_key(plan, trigger: str, cls: str):
    """VIS-119 (`LIBRARY_HOLD`): the session key of the class's library answer, for a new user
    message on a class whose package answer holds for the class; None otherwise."""
    if LIBRARY_HOLD != "class" or trigger != "message" or \
            getattr(plan, "sticky", "message") != "class":
        return None
    return ("library", "class", cls)


def _keep(sess, key, hold, picks, how) -> None:
    """An answer into the session's picks: the message's key, and the class's when it holds."""
    with sess.lock:
        if len(sess.package_pick) > 32:
            sess.package_pick.clear()
        sess.package_pick[key] = (list(picks), how)
        if hold is not None:
            sess.package_pick[hold] = (list(picks), how)


#: what `_ask` reads of a plan, for a question asked after the turn: never under the engine lock
_AFTER_TURN = types.SimpleNamespace(locked=False)


def _deferred_job(router, hits, text, line, cfg, router_doc, roles, about, sess, key, row, info,
                  trigger) -> dict:
    """VIS-119: the shadow library question as a job for server/conductor/deferred.py: asked with
    the candidates the plan found; its answer goes into the session's picks for the message (not
    the class's: a shadow question asked off the path holds nothing for the class) and, as a new
    `extra.library` (and `shadow.library` for a pick), into the request's journal row. The job
    keeps no reference to the plan (its body, tools and ids), only what the question reads."""
    def land(new: dict, picks=()) -> None:
        extra = dict(row.get("extra") or {}, library=new)
        fields = {"extra": extra}
        row["extra"] = extra
        if picks:
            fields["shadow"] = row["shadow"] = dict(row.get("shadow") or {}, library={
                "chosen": list(picks), "trigger": trigger})
        if row.get("id"):
            router.journal.update(row["id"], **fields)

    def run(waited_ms: float) -> None:
        hit = sess.package_pick.get(key)
        if hit is not None:
            # the same message's earlier job (a regenerate inside the idle time) answered it
            land(dict(info, chosen=list(hit[0]), how=hit[1], cached=True,
                      deferred={"waited_ms": round(float(waited_ms), 1)}), hit[0])
            return
        PARENT.set(row.get("request_id"))          # VIS-60: the question names its turn
        t0 = time.perf_counter()
        picks, how, asked = _ask(router, _AFTER_TURN, hits, text, line, cfg, router_doc, roles,
                                 about)
        if how not in ("no_answer", "locked"):
            _keep(sess, key, None, picks, how)
        land(dict(info, chosen=list(picks), how=how, stages=asked,
                  ask_ms=round((time.perf_counter() - t0) * 1e3, 2),
                  deferred={"waited_ms": round(float(waited_ms), 1)}), picks)

    def drop(why: str) -> None:
        land(dict(info, deferred={"dropped": why}))

    return {"what": "library", "run": run, "drop": drop}


def attach(router, plan, body, router_doc, prof, packages, roles, sh, sess, cls, row,
           dry: bool) -> None:
    """The library's part of a plan (the module's header). Nothing at all with the switch off."""
    mode = schema.profile_library(prof)
    if plan.res.mode == "shadow" and mode == "on":
        mode = "shadow"
    if mode == "off" or plan.res.mode == "raw" or sh.compaction:
        return
    if mode == "on":
        plan.decisions["library"] = "on"           # its notes replay
    cfg = schema.library(router_doc)
    if not dry:
        _card(router, plan, packages, sh, cls, row, cfg, mode)
    o = plan.outcome
    if o is not None and o["label"] in cfg["outcomes"]:
        trigger, key = f"outcome:{o['label']}", ("library", "outcome", o["sig"])
        exts = ",".join(dict.fromkeys(e for e in (signals.ext_of(f) for f in sh.files[-8:]) if e))
        line = "; ".join(o["why"][:3]) + (f" files={exts}" if exts else "")
    elif not sh.tool_turn and sh.user_text.strip():
        trigger, key, line = "message", ("library", "message", cls, sh.user_hash), None
    else:
        return
    v, _ = router.store.peek("packages")
    t0 = time.perf_counter()
    router.index.ensure(v, packages, cfg)
    query = signals.digest(sh.user_text or "", int(cfg["digest_chars"]), line)
    # FX6: a package this turn's note already carries (a class's core skill) is not picked again
    live = _live_names(sess) | set(sh.loaded_skills) | {
        p["name"] for p in ((plan.block or {}).get("parts") or ()) if p.get("kind") == "package"}
    hits = router.index.search(query, int(cfg["candidates"]), cls if cfg["class_filter"] else None,
                               exclude=live, min_score=float(cfg["min_score"]))
    info = {"mode": mode, "trigger": trigger, "query_chars": len(query),
            "candidates": [{"name": e.name, "score": round(s, 3)} for e, s in hits],
            "search_ms": round((time.perf_counter() - t0) * 1e3, 3), "chosen": [], "how": None}
    if live:
        info["live"] = sorted(live)
    row["extra"]["library"] = info
    if not hits:
        info["how"] = "no_match"
        return
    t1 = time.perf_counter()
    hit = sess.package_pick.get(key)
    want = deferred.mode() if mode == "shadow" and not dry and not plan.locked else "inline"
    # VIS-119: a shadow answer the request never reads, from Qwen through this server's own port
    # first, is asked after the turn (`idle`) or not at all (`off`); a turn that failed open
    # (SRV-763) asks nothing, after it neither
    later = hit is None and want != "inline" and not getattr(plan, "fail_open", False) and \
        router.own_port_first(router_doc, roles, "packages")
    # ... and has no class hold: asked off the request's path it costs the turn nothing, and the
    # shadow journal keeps an answer per message
    hold = None if later else _hold_key(plan, trigger, cls)
    held = False
    if hit is None and hold is not None and not getattr(plan, "class_new", True):
        hit = sess.package_pick.get(hold)
        held = hit is not None
        if held:
            info["held"] = "class"          # VIS-119: the class's answer, no question
    if hit is not None:
        picks, how = hit
        info["cached"] = True
    elif later:
        info.update(chosen=[], how="deferred" if want == "idle" else "shadow_off")
        if want == "idle":
            from server.conductor.router import _about
            plan.deferred.append(_deferred_job(
                router, hits, sh.user_text or "", line, cfg, router_doc, roles,
                _about(cls, {c["id"]: c for c in router_doc["classes"]}), sess, key, row, info,
                trigger))
        return
    else:
        from server.conductor.router import _about
        picks, how, asked = _ask(router, plan, hits, sh.user_text or "", line, cfg, router_doc,
                                 roles, _about(cls, {c["id"]: c for c in router_doc["classes"]}))
        info["stages"] = asked
        if how not in ("no_answer", "locked", "fail_open") and not dry:
            _keep(sess, key, hold, picks, how)
    info.update(chosen=list(picks), how=how, ask_ms=round((time.perf_counter() - t1) * 1e3, 2))
    if not picks:
        return
    if mode == "shadow":
        if held:
            # VIS-119: an answer held from the class's first message is no answer for this one;
            # the shadow journal counts the messages that were asked (`extra.library.held`)
            return
        row["shadow"] = dict(row.get("shadow") or {}, library={"chosen": list(picks),
                                                                "trigger": trigger})
        return
    if trigger != "message" and not dry:
        cap = router._acts_free(plan, prof, o["now"])
        if cap is not None:
            info["cap"] = cap
            return
    added = []
    for name in picks:
        text, err = router.library.text(packages, name)
        if text is None:
            row["extra"].setdefault("notes", []).append(f"{err}:{name}")
            continue
        text = cfg["header"].replace("{name}", name) + "\n\n" + text
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        plan.extras.append({"key": f"library:{name}:{sha}", "text": text,
                            "parts": [{"kind": "library", "name": name}], "kinds": {"library"}, "reason": f"library {name} ({trigger})",
                            "unless": ("library", name)})
        added.append(name)
    if added and trigger != "message" and not dry:
        plan.acts.append(o["now"])
        with sess.lock:
            sess.acts.append(o["now"])
    info["applied"] = added


def _card(router, plan, packages, sh, cls, row, cfg, mode) -> None:
    """The library card (how the model searches the library itself), live in the session's
    notes on the card classes with tools, as the harness card; shadow journals it."""
    name = cfg.get("card")
    if not name or not sh.tools or cls not in cfg["card_classes"]:
        return
    fits = getattr(router, "fits", None)
    if fits is not None and not fits(plan, packages, name):
        # SRV-243: the library is read with curl from a shell; a chat without one gets no card
        row["extra"].setdefault("notes", []).append(f"tools_missing:{name}")
        return
    text, err = router.library.text(packages, name)
    if text is None:
        row["extra"].setdefault("notes", []).append(f"{err}:{name}")
        return
    if mode == "shadow":
        row["shadow"] = dict(row.get("shadow") or {}, library_card={"name": name,
                                                                     "chars": len(text)})
        return
    plan.extras.append({"key": f"library_card:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
                        "text": text, "parts": [{"kind": "library_card", "name": name}],
                        "kinds": {"library"}, "reason": "library card",
                        "unless": ("library_card", name)})
