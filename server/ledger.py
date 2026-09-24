"""The usage ledger: one SQLite row per request, kept for at least 400 days, with no text (SRV-28).

Prometheus on this cluster keeps 15 days and stores series, not requests; a year of "what did I
spend" needs its own store. This is it: stdlib `sqlite3`, one file, WAL, one writer thread.

WHAT A ROW IS. `server/usage.py::RequestRecord.row()` -- counts, timings, the finish reason, the
cache source, a client id. Never the prompt, the answer, a tool name or argument, a conversation
id, the `user` field, an IP address or a raw token. `client_id` is `k:` + the first 12 hex digits of
sha256(the bearer token), or `anon`; `client_kind` is one of a fixed set, read from the
User-Agent. Refused requests (503/429) and rejected ones (400) are rows too, with null token
fields, so the request count is honest.

IT NEVER SLOWS OR FAILS A REQUEST. `submit()` is a `put_nowait` on a bounded queue (10,000); a full
queue drops the row and counts it. The writer commits in batches at most every second. A disk or
SQLite error drops that batch, is counted, and is logged at most once a minute. The drain on
SIGTERM calls `close()`, which flushes what is queued.

OFF BY DEFAULT, AND NEVER FROM THE ENVIRONMENT. The server takes the path from `--usage-ledger`
only. `ops/serve.env` sets `QSE_USAGE_LEDGER` and `ops/start.sh` turns it into the flag for the
:8000 service; the server itself does not read the variable, because `ops/gate.sh` exports all of
serve.env before it launches row3, verify_spec and the rest, each with a server of its own, and
none of those may write the operator's ledger. A test or e2e run whose path resolves to the production
file is refused outright.

RETENTION. Rows older than `--usage-retention-days` (400; less is refused outside tests) are
deleted once a day at 04:00 local, then `PRAGMA incremental_vacuum`. Once an ISO week, an online
backup (the sqlite3 backup API) goes to `backup/ledger-<year>-W<week>.sqlite3` beside the file,
the newest 8 kept.

    sqlite3 ~/.qwen38-spark-engine/usage/ledger.sqlite3 \
        "select datetime(ts_ms/1000,'unixepoch','localtime'), finish_reason, prompt_tokens,
                completion_tokens, round(decode_tps,1) from requests order by id desc limit 10"
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import queue
import sqlite3
import threading
import time

PRODUCTION = "~/.qwen38-spark-engine/usage/ledger.sqlite3"
SCHEMA_VERSION = 1
MIN_RETENTION_DAYS = 400
QUEUE_MAX = 10_000
BACKUP_KEEP = 8

COLUMNS = ("ts_ms", "request_id", "model", "client_id", "client_kind", "endpoint", "stream",
           "status", "finish_reason", "prompt_tokens", "cached_tokens", "completion_tokens",
           "reasoning_tokens", "queue_ms", "prompt_ms", "ttft_ms", "decode_ms", "total_ms",
           "decode_tps", "prefill_tps", "blocks", "draft_tokens", "draft_accepted", "tool_calls",
           "thinking", "cache_source", "max_tokens", "error_type", "engine_version", "code_sha")

# Schema v1, exactly as the design note §2.3 has it. Forward-only migrations key on
# meta.schema_version; there is only the first.
DDL = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, request_id TEXT NOT NULL,
  model TEXT NOT NULL, client_id TEXT NOT NULL, client_kind TEXT NOT NULL,
  endpoint TEXT NOT NULL, stream INTEGER NOT NULL, status INTEGER NOT NULL, finish_reason TEXT,
  prompt_tokens INTEGER, cached_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
  queue_ms REAL, prompt_ms REAL, ttft_ms REAL, decode_ms REAL, total_ms REAL,
  decode_tps REAL, prefill_tps REAL, blocks INTEGER, draft_tokens INTEGER, draft_accepted INTEGER,
  tool_calls INTEGER NOT NULL DEFAULT 0, thinking INTEGER NOT NULL DEFAULT 0,
  cache_source TEXT, max_tokens INTEGER, error_type TEXT, engine_version TEXT, code_sha TEXT);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts_ms);
CREATE INDEX IF NOT EXISTS requests_dim_ts ON requests(model, client_id, ts_ms);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

KINDS = ("open-webui", "openai-sdk", "curl", "dashboard", "other")


def resolve(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def is_production(path: str) -> bool:
    return resolve(path) == resolve(PRODUCTION)


def check_config(path: str | None, retention_days: int, *, test: bool) -> None:
    """Refuse the two configurations that must never start: a test writing the real ledger, and
    a retention shorter than the year-plus the Usage view promises."""
    if path and test and is_production(path):
        raise SystemExit(f"[ledger] REFUSING: a test run may not write the production ledger "
                         f"({resolve(PRODUCTION)}); give it a path of its own")
    if path and retention_days < MIN_RETENTION_DAYS and not test:
        raise SystemExit(f"[ledger] --usage-retention-days {retention_days} is below the "
                         f"{MIN_RETENTION_DAYS}-day minimum (a year of history plus a margin); "
                         f"shorter retention is for tests only (QSE_TEST=1)")


def client_of(headers) -> tuple[str, str]:
    """`(client_id, client_kind)` of a request, with nothing personal in either.

    The id is a hash prefix of the bearer token -- Open WebUI's connection key, the dashboard's
    own -- so two clients are told apart without the token ever being stored; VIS-1's token table
    replaces it with a name later. The kind is a fixed set read from the User-Agent: Open WebUI's
    outbound client is aiohttp's default agent (`Python/3.x aiohttp/3.x`), the only aiohttp caller
    here; a browser cannot set a User-Agent from `fetch`, so the dashboard is recognised by the
    `Referer` a same-origin fetch from `/dashboard/` carries (or an explicit `X-QSE-Client:
    dashboard`).
    """
    def get(name):
        try:
            return headers.get(name) or ""
        except Exception:                                          # noqa: BLE001
            return ""
    auth = get("Authorization").strip()
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    cid = "k:" + hashlib.sha256(token.encode()).hexdigest()[:12] if token else "anon"
    ua = get("User-Agent").lower()
    referer = get("Referer")
    from_dashboard = "/dashboard/" in referer.split("?")[0] if referer else False
    if (get("X-QSE-Client").strip().lower() == "dashboard" or "qse-dashboard" in ua
            or from_dashboard):
        kind = "dashboard"
    elif "open-webui" in ua or "openwebui" in ua or "aiohttp" in ua:
        kind = "open-webui"
    elif ua.startswith("openai/") or "openai/python" in ua or "openai/js" in ua:
        kind = "openai-sdk"
    elif ua.startswith("curl/"):
        kind = "curl"
    else:
        kind = "other"
    return cid, kind


def next_local(hour: int, now: float) -> float:
    """The next `hour`:00 local time strictly after `now` (epoch seconds)."""
    t = _dt.datetime.fromtimestamp(now).astimezone()
    cand = t.replace(hour=hour, minute=0, second=0, microsecond=0)
    if cand.timestamp() <= now:
        cand = (t + _dt.timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return cand.timestamp()


class Ledger:
    """The writer. One per server; `open()` starts its thread, `close()` flushes and stops it."""

    def __init__(self, path: str, *, retention_days: int = MIN_RETENTION_DAYS,
                 queue_max: int = QUEUE_MAX, flush_s: float = 1.0, backup_keep: int = BACKUP_KEEP,
                 clock=time.time, log=None):
        self.path = resolve(path)
        self.retention_days = int(retention_days)
        self.flush_s = float(flush_s)
        self.backup_keep = int(backup_keep)
        self.clock = clock
        self.log = log or (lambda msg: print(msg, flush=True))
        self._q: queue.Queue = queue.Queue(maxsize=int(queue_max))
        self.stats = {"rows": 0, "dropped": 0, "failed": 0, "pruned": 0, "backups": 0}
        self._stop = threading.Event()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None
        self._last_error_log = 0.0
        self._next_prune = 0.0
        self._local = threading.local()
        self.since_ms: int | None = None

    # ----------------------------------------------------------------- lifecycle
    def open(self) -> "Ledger":
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        self._conn = self._connect()
        row = self._conn.execute("SELECT value FROM meta WHERE key='created_at'").fetchone()
        if row is None:
            now_ms = int(self.clock() * 1000)
            self._conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                                   [("schema_version", str(SCHEMA_VERSION)),
                                    ("created_at", str(now_ms))])
            self._conn.commit()
        oldest = self._conn.execute("SELECT MIN(ts_ms) FROM requests").fetchone()[0]
        created = int(self._conn.execute(
            "SELECT value FROM meta WHERE key='created_at'").fetchone()[0])
        self.since_ms = min(created, oldest) if oldest is not None else created
        self._next_prune = next_local(4, self.clock())
        self._thread = threading.Thread(target=self._run, name="usage-ledger", daemon=True)
        self._thread.start()
        return self

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        # auto_vacuum has to be set before the first table exists to take effect
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(DDL)
        conn.commit()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return conn

    def close(self, timeout: float = 10.0) -> None:
        """Flush what is queued, then stop. Called by the SIGTERM drain."""
        self._closing.set()
        self.flush(timeout)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every submitted row is committed or given up on."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    # ----------------------------------------------------------------- writing
    def submit(self, row: dict) -> bool:
        """Hand a row to the writer. Never blocks: a full queue drops it and counts it."""
        try:
            self._q.put_nowait(row)
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    @property
    def queued(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        batch: list[dict] = []
        last = time.monotonic()
        while True:
            try:
                batch.append(self._q.get(timeout=0.25))
            except queue.Empty:
                pass
            now = time.monotonic()
            if batch and (now - last >= self.flush_s or len(batch) >= 1000 or self._stop.is_set()
                          or self._q.empty()):
                self._write(batch)
                for _ in batch:
                    self._q.task_done()
                batch = []
                last = now
            if self.clock() >= self._next_prune:
                self.maintain()
            if self._stop.is_set() and self._q.empty() and not batch:
                return

    def _write(self, batch: list[dict]) -> None:
        try:
            if self._conn is None:
                self._conn = self._connect()
            self._conn.executemany(
                f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES "
                f"({', '.join('?' * len(COLUMNS))})",
                [tuple(r.get(c) for c in COLUMNS) for r in batch])
            self._conn.commit()
            self.stats["rows"] += len(batch)
        except (sqlite3.Error, OSError) as exc:
            self.stats["failed"] += len(batch)
            try:
                self._conn.rollback()
            except Exception:                                      # noqa: BLE001
                self._conn = None
            self._error(f"write failed, {len(batch)} rows lost: {type(exc).__name__}: {exc}")

    def _error(self, msg: str) -> None:
        now = self.clock()
        if now - self._last_error_log >= 60.0 or not self._last_error_log:
            self._last_error_log = now
            self.log(f"[ledger] {msg} ({self.stats['failed']} lost, {self.stats['dropped']} "
                     f"dropped so far; logged at most once a minute)")

    # ----------------------------------------------------------------- retention and backup
    def maintain(self) -> None:
        """The daily 04:00 pass: prune, vacuum, the weekly backup. Runs on the writer thread."""
        self._next_prune = next_local(4, self.clock())
        try:
            n = self.prune()
            if n:
                self.log(f"[ledger] pruned {n} rows older than {self.retention_days} days")
            self.backup()
        except (sqlite3.Error, OSError) as exc:
            self._error(f"maintenance failed: {type(exc).__name__}: {exc}")

    def prune(self, now: float | None = None) -> int:
        if self._conn is None:
            self._conn = self._connect()
        cutoff = int(((self.clock() if now is None else now) - self.retention_days * 86400) * 1000)
        cur = self._conn.execute("DELETE FROM requests WHERE ts_ms < ?", (cutoff,))
        self._conn.commit()
        self._conn.execute("PRAGMA incremental_vacuum")
        self._conn.commit()
        self.stats["pruned"] += cur.rowcount
        return cur.rowcount

    def backup(self, now: float | None = None) -> str | None:
        """This ISO week's online backup, unless it exists; the newest `backup_keep` kept."""
        if self._conn is None:
            self._conn = self._connect()
        y, w, _ = _dt.date.fromtimestamp(self.clock() if now is None else now).isocalendar()
        d = os.path.join(os.path.dirname(self.path), "backup")
        os.makedirs(d, mode=0o700, exist_ok=True)
        dest = os.path.join(d, f"ledger-{y}-W{w:02d}.sqlite3")
        if os.path.exists(dest):
            return None
        tmp = dest + ".part"
        out = sqlite3.connect(tmp)
        try:
            self._conn.backup(out)
        finally:
            out.close()
        os.replace(tmp, dest)
        os.chmod(dest, 0o600)
        self.stats["backups"] += 1
        old = sorted(f for f in os.listdir(d) if f.startswith("ledger-") and f.endswith(".sqlite3"))
        for f in old[:-self.backup_keep] if self.backup_keep else []:
            os.remove(os.path.join(d, f))
        return dest

    # ----------------------------------------------------------------- reading
    def reader(self) -> sqlite3.Connection:
        """A read-only connection for the calling thread (the dashboard API's handlers)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
            conn.execute("PRAGMA query_only=1")
            self._local.conn = conn
        return conn

    def info(self) -> dict:
        """Size, age and health, for /v1/dashboard/summary and /system and for /metrics."""
        size = 0
        for suffix in ("", "-wal"):
            try:
                size += os.path.getsize(self.path + suffix)
            except OSError:
                pass
        try:
            rows, oldest = self.reader().execute(
                "SELECT COUNT(*), MIN(ts_ms) FROM requests").fetchone()
        except sqlite3.Error:
            rows, oldest = None, None
        return {"enabled": True, "path": self.path, "rows": rows, "bytes": size,
                "oldest_ms": oldest, "since_ms": self.since_ms,
                "retention_days": self.retention_days, "queue": self.queued,
                "dropped": self.stats["dropped"] + self.stats["failed"],
                "written": self.stats["rows"]}
