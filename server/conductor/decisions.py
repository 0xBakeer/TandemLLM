"""The router's journal: every decision, what it injected, and what came of it.

One row per `TandemLLM` request the router saw, shadow and on alike: the class, its confidence,
which stage decided (and every stage's answer and time), the effort set and where the budget came
from, every note with its exact text, token position and reason, the notes replayed, what shadow
would have done, and the turn's outcome (finish, tool calls, reasoning tokens, time). The next
request of the session fills `next_checker` with the checker's RESULT line and numbers, and the
dashboard's "right / wrong" buttons fill `label`. That is the honesty record (the session export)
and the flywheel's data (TandemLLM 5).

Storage mirrors the usage ledger: stdlib sqlite3, WAL, a queue and one writer thread, so nothing
here can slow or fail a request; a full queue drops the row and counts it. The live feed reads a
ring of the last 500 rows, which also holds the requests still running. Rows older than
`retention_days` (30) go once a day. The user's text (`digest`) is kept only for requests with
tools unless the settings widen it; chats without tools keep counters (Khaled's decision 2).
"""

from __future__ import annotations

import collections
import json
import os
import queue
import sqlite3
import threading
import time
import uuid

DDL = """
CREATE TABLE IF NOT EXISTS route_decisions (
  id TEXT PRIMARY KEY, ts_ms INTEGER NOT NULL, request_id TEXT, session_key TEXT,
  client_kind TEXT, model TEXT, profile TEXT, profile_version INTEGER, router_version INTEGER,
  mode TEXT, class_id TEXT, confidence REAL, source TEXT, kept TEXT, stages TEXT,
  classifier_ms REAL, effort TEXT, budget INTEGER, budget_source TEXT, thinking TEXT,
  decisions TEXT, injected TEXT, replayed TEXT, shadow TEXT, digest TEXT, outcome TEXT,
  next_checker TEXT, label TEXT, label_note TEXT, labelled_at INTEGER, acted INTEGER,
  changed INTEGER, extra TEXT);
CREATE INDEX IF NOT EXISTS route_ts ON route_decisions(ts_ms);
CREATE INDEX IF NOT EXISTS route_session ON route_decisions(session_key, ts_ms);
"""
JSON_COLS = ("stages", "decisions", "injected", "replayed", "shadow", "outcome", "next_checker",
             "extra")
COLS = ("id", "ts_ms", "request_id", "session_key", "client_kind", "model", "profile",
        "profile_version", "router_version", "mode", "class_id", "confidence", "source", "kept",
        "stages", "classifier_ms", "effort", "budget", "budget_source", "thinking", "decisions",
        "injected", "replayed", "shadow", "digest", "outcome", "next_checker", "label",
        "label_note", "labelled_at", "acted", "changed", "extra")
RING = 500
QUEUE_MAX = 5000


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Journal:
    def __init__(self, path: str | None, retention_days: int = 30):
        self.path = os.path.realpath(os.path.expanduser(path)) if path else None
        self.retention_days = int(retention_days)
        self.ring: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(QUEUE_MAX)
        self.dropped = 0
        self.errors = 0
        self._thread = None
        self._stop = threading.Event()
        self._next_prune = 0.0
        if self.path:
            os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
            conn = self._connect()
            conn.executescript(DDL)
            conn.commit()
            conn.close()
            self._thread = threading.Thread(target=self._run, name="conductor-journal",
                                            daemon=True)
            self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------ writes
    def open_row(self, row: dict) -> dict:
        """A decision made at request start: visible in the live feed at once."""
        row.setdefault("id", new_id())
        row.setdefault("ts_ms", int(time.time() * 1000))
        with self._lock:
            self.ring[row["id"]] = row
            while len(self.ring) > RING:
                self.ring.popitem(last=False)
        return row

    def close_row(self, row: dict) -> None:
        """The turn ended: the outcome is in the row; write it."""
        self._submit(("upsert", dict(row)))

    def update(self, rid: str, **fields) -> bool:
        """`next_checker` or a human label on an earlier row."""
        with self._lock:
            r = self.ring.get(rid)
            if r is not None:
                r.update(fields)
        self._submit(("update", rid, fields))
        return True

    def _submit(self, item) -> None:
        if self.path is None:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        conn = self._connect()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                item = None
            batch = [item] if item is not None else []
            while len(batch) < 200:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if batch:
                try:
                    for it in batch:
                        if it[0] == "upsert":
                            r = it[1]
                            conn.execute(
                                f"INSERT OR REPLACE INTO route_decisions ({', '.join(COLS)}) "
                                f"VALUES ({', '.join('?' * len(COLS))})",
                                tuple(json.dumps(r.get(c), ensure_ascii=False)
                                      if c in JSON_COLS and r.get(c) is not None else r.get(c)
                                      for c in COLS))
                        else:
                            _, rid, fields = it
                            sets = [f"{k} = ?" for k in fields if k in COLS]
                            if sets:
                                conn.execute(
                                    f"UPDATE route_decisions SET {', '.join(sets)} WHERE id = ?",
                                    tuple(json.dumps(v, ensure_ascii=False) if k in JSON_COLS
                                          and v is not None else v
                                          for k, v in fields.items() if k in COLS) + (rid,))
                    conn.commit()
                except sqlite3.Error as exc:
                    self.errors += 1
                    print(f"[conductor] journal write failed: {exc}", flush=True)
            if time.time() >= self._next_prune:
                self._next_prune = time.time() + 86400
                try:
                    cut = int((time.time() - self.retention_days * 86400) * 1000)
                    conn.execute("DELETE FROM route_decisions WHERE ts_ms < ?", (cut,))
                    conn.commit()
                except sqlite3.Error:
                    self.errors += 1
        conn.close()

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while not self._q.empty() and time.time() < end:
            time.sleep(0.02)
        time.sleep(0.05 if self.path else 0)

    def close(self) -> None:
        self.flush()
        self._stop.set()

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _decode(r: sqlite3.Row | tuple) -> dict:
        d = dict(zip(COLS, r))
        for c in JSON_COLS:
            if isinstance(d.get(c), str):
                try:
                    d[c] = json.loads(d[c])
                except ValueError:
                    pass
        return d

    def rows(self, *, since_ms: int = 0, limit: int = 100, session: str | None = None,
             before_ms: int | None = None) -> list[dict]:
        """Newest first: the ring (running and recent) merged with the file."""
        with self._lock:
            ring = [dict(r) for r in self.ring.values()]
        ring = [r for r in ring if r["ts_ms"] >= since_ms and
                (session is None or r.get("session_key") == session) and
                (before_ms is None or r["ts_ms"] < before_ms)]
        seen = {r["id"] for r in ring}
        disk: list[dict] = []
        if self.path:
            q = "SELECT " + ", ".join(COLS) + " FROM route_decisions WHERE ts_ms >= ?"
            args: list = [since_ms]
            if session is not None:
                q += " AND session_key = ?"
                args.append(session)
            if before_ms is not None:
                q += " AND ts_ms < ?"
                args.append(before_ms)
            q += " ORDER BY ts_ms DESC LIMIT ?"
            args.append(int(limit))
            try:
                conn = self._connect()
                disk = [self._decode(r) for r in conn.execute(q, args)]
                conn.close()
            except sqlite3.Error:
                disk = []
        out = ring + [r for r in disk if r["id"] not in seen]
        out.sort(key=lambda r: r["ts_ms"], reverse=True)
        return out[:limit]

    def get(self, rid: str) -> dict | None:
        with self._lock:
            if rid in self.ring:
                return dict(self.ring[rid])
        if not self.path:
            return None
        try:
            conn = self._connect()
            r = conn.execute("SELECT " + ", ".join(COLS) + " FROM route_decisions WHERE id = ?",
                             (rid,)).fetchone()
            conn.close()
        except sqlite3.Error:
            return None
        return self._decode(r) if r else None

    def info(self) -> dict:
        return {"path": self.path, "ring": len(self.ring), "queue": self._q.qsize(),
                "dropped": self.dropped, "errors": self.errors,
                "retention_days": self.retention_days}


# ------------------------------------------------------------------ aggregates
def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return round(v[i], 2)


def stats(rows: list[dict], *, hours: int = 24, now_ms: int | None = None) -> dict:
    """The Overview's tiles and ribbon, from journal rows of the last `hours`."""
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - hours * 3600_000
    rows = [r for r in rows if r["ts_ms"] >= since]
    by_class: dict[str, int] = collections.Counter(r.get("class_id") or "none" for r in rows)
    by_source = collections.Counter(r.get("source") or "none" for r in rows)
    by_effort = collections.Counter(r.get("effort") or "none" for r in rows)
    stage_ms: dict[str, list[float]] = collections.defaultdict(list)
    fails: dict[str, int] = collections.Counter()
    calls: dict[str, int] = collections.Counter()
    for r in rows:
        for s in r.get("stages") or []:
            # a stage answered from the text cache asked nothing, and a stage whose role is off was
            # never asked: neither is a call, a latency or a failure (the classifier tile read
            # "0 / 0 ms" with the decider off on 2026-10-01)
            if s.get("cached") or str(s.get("error") or "").endswith("role_off"):
                continue
            calls[s["stage"]] += 1
            stage_ms[s["stage"]].append(float(s.get("ms") or 0))
            if s.get("error"):
                fails[s["error"]] += 1
    acted = [r for r in rows if r.get("acted")]
    shadow = [r for r in rows if r.get("shadow")]
    injected_tokens = sum(sum(n.get("tokens") or 0 for n in (r.get("injected") or []))
                          for r in rows)
    hourly = []
    for h in range(hours):
        lo = since + h * 3600_000
        hi = lo + 3600_000
        inh = [r for r in rows if lo <= r["ts_ms"] < hi]
        hourly.append({"start_ms": lo, "total": len(inh),
                       "by_class": dict(collections.Counter(r.get("class_id") or "none"
                                                            for r in inh)),
                       "acted": sum(1 for r in inh if r.get("acted")),
                       "shadow": sum(1 for r in inh if r.get("shadow"))})
    labelled = [r for r in rows if r.get("label") in ("right", "wrong")]
    decided = [r for r in rows if r.get("source") in ("decider", "qwen")]
    # the promotion rule's input: per profile and decision type that ran in shadow, how many of
    # its decisions a person labelled and how many of those were right
    prec: dict[tuple, dict] = {}
    for r in rows:
        for d, m in (r.get("decisions") or {}).items():
            if m != "shadow":
                continue
            p = prec.setdefault((r.get("profile"), d), {"profile": r.get("profile"),
                                                        "decision": d, "decisions": 0,
                                                        "labelled": 0, "agree": 0})
            p["decisions"] += 1
            if r.get("label") in ("right", "wrong"):
                p["labelled"] += 1
                p["agree"] += r["label"] == "right"
    for p in prec.values():
        p["precision"] = round(p["agree"] / p["labelled"], 4) if p["labelled"] else None
    classes: dict[str, dict] = {}
    for r in rows:
        c = classes.setdefault(r.get("class_id") or "none", {"class_id": r.get("class_id") or "none",
                                                             "decisions": 0, "labelled": 0,
                                                             "agree": 0})
        c["decisions"] += 1
        if r.get("label") in ("right", "wrong"):
            c["labelled"] += 1
            c["agree"] += r["label"] == "right"
    return {
        "hours": hours, "decisions": len(rows), "acted": len(acted), "shadow": len(shadow),
        "changed": sum(1 for r in rows if r.get("changed")),
        "injections": sum(1 for r in rows if r.get("injected")),
        "injected_tokens": injected_tokens,
        "by_class": dict(by_class), "by_source": dict(by_source), "by_effort": dict(by_effort),
        "classifier": {k: {"calls": calls[k], "p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95)}
                       for k, v in stage_ms.items()},
        "failures": dict(fails),
        "fail_open_rate": round(sum(1 for r in rows if r.get("kept") in
                                    ("fail_open", "no_answer")) / len(rows), 4) if rows else None,
        "model_decided": len(decided),
        "labels": {"right": sum(1 for r in labelled if r["label"] == "right"),
                   "wrong": sum(1 for r in labelled if r["label"] == "wrong")},
        "precision": sorted(prec.values(), key=lambda p: (str(p["profile"]), p["decision"])),
        "classes": sorted(classes.values(), key=lambda c: -c["decisions"]),
        "hourly": hourly,
        # the session monitor's counts (SRV-115) once its journal is on this server
        "monitor": None,
    }


def sessions(rows: list[dict]) -> list[dict]:
    """One line per session: client, profile, the class now, turns, notes, last checker."""
    out: dict[str, dict] = {}
    for r in sorted(rows, key=lambda x: x["ts_ms"]):
        k = r.get("session_key") or "-"
        s = out.setdefault(k, {"key": k, "client_kind": r.get("client_kind"),
                               "profile": r.get("profile"), "started_ms": r["ts_ms"],
                               "turns": 0, "notes": 0, "note_tokens": 0, "classes": {},
                               "class_id": None, "last_ms": r["ts_ms"], "last_checker": None,
                               "labels": 0})
        s["turns"] += 1
        s["last_ms"] = r["ts_ms"]
        s["class_id"] = r.get("class_id")
        s["profile"] = r.get("profile") or s["profile"]
        s["classes"][r.get("class_id") or "none"] = s["classes"].get(r.get("class_id") or "none",
                                                                     0) + 1
        for n in r.get("injected") or []:
            s["notes"] += 1
            s["note_tokens"] += n.get("tokens") or 0
        if r.get("next_checker"):
            s["last_checker"] = (r["next_checker"] or [{}])[-1].get("verdict")
        if r.get("label"):
            s["labels"] += 1
    return sorted(out.values(), key=lambda s: s["last_ms"], reverse=True)


def export_md(key: str, rows: list[dict]) -> str:
    """The honest record of a session: every hidden text the model saw, in order."""
    rows = sorted(rows, key=lambda x: x["ts_ms"])
    notes = [(r, n) for r in rows for n in (r.get("injected") or [])]
    lines = [f"# Session {key}", "",
             f"{len(notes)} intervention{'s' if len(notes) != 1 else ''} in {len(rows)} "
             f"turn{'s' if len(rows) != 1 else ''}. The client's transcript does not contain the "
             f"texts below; the model saw each at the position given (tokens, counted in the "
             f"client's prompt) and on every later turn of the session until a compaction.", ""]
    for r in rows:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["ts_ms"] / 1000)) + "Z"
        lines.append(f"## {t} · {r.get('class_id')} ({r.get('source')}, "
                     f"{(r.get('confidence') or 0):.2f}) · effort {r.get('effort')} · "
                     f"{r.get('mode')}")
        out = r.get("outcome") or {}
        lines.append(f"Outcome: {out.get('finish_reason')}, {out.get('tool_calls', 0)} tool "
                     f"calls, {out.get('reasoning_tokens')} reasoning tokens.")
        for n in r.get("injected") or []:
            lines += ["", f"Injected at token {n.get('pos')} ({n.get('tokens')} tokens), "
                          f"because {n.get('reason')}:", "", "```text", n.get("text", ""), "```"]
        if r.get("replayed"):
            rp = r["replayed"]
            lines.append(f"Replayed {rp.get('count', 0)} earlier note(s), "
                         f"{rp.get('tokens', 0)} tokens.")
        if r.get("shadow"):
            lines.append(f"Shadow (not applied): {json.dumps(r['shadow'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)
