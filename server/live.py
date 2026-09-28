"""The live view of the requests in flight: `GET /v1/dashboard/live`.

the operator, 2026-09-26: "The tok/s prefill, etc requests counts decode for one and for all together.
Live data." Everything the dashboard had until now moves when a request ENDS -- `/metrics`
counters in `_log_request`, the ledger row in `_account` -- so a 40 s answer showed nothing for
40 s and then a jump. This module reads the `RequestRecord` every request already carries WHILE it
runs, once a second, and answers with one row per request plus the totals.

THE HOT PATH, because that is the constraint this was built under. Per token the server pays one
integer add: `RequestRecord.track` (the handler thread's loop, which already stamps `t_first` and
`t_last` with `perf_counter`) does `self.n_live += 1`. No lock, no allocation, no callback. Per
request there are two lock acquisitions here, `register` before the queue and `finish` in the
handler's `finally`. The sampler thread wakes once a second ONLY while a request is in flight or a
live stream is open (`_wake`); an idle box with the dashboard closed pays nothing. A tick reads a
handful of integers off the records and the running `BlockStats`, appends one small dict to a
bounded deque, and sleeps. Nothing here takes the engine lock and nothing in `engine/` changed.

THE NUMBERS ARE THE RESPONSE'S NUMBERS. A finished row is computed from the same record that
writes `timings` on the finish chunk: `decode_tps` divides committed tokens (the first
token is the prefill's) by `t_last - t_first`, `prefill_tps` divides forwarded tokens by
`prompt_ms`, `ttft_ms` is queue plus prompt. While the request runs the same formulas take `now`
for the end. The prefill's reused / forwarded split comes from `STATE["last_prefill"]`, which is
this request's the instant its first token exists, since the engine serves one request at a time
and the record absorbs it only at the end.

Definitions (also in the Memo note "Live speed panel -- design (2026-09-26)"):

    phase             queued until the engine lock, prefill until the first token, decode until the
                      end, then done (a refusal or a 400 is done too, with its status)
    tokens            tokens the handler has received so far; equals completion_tokens at the end
    decode_tps        (tokens - 1) / seconds since the first token (running average)
    decode_tps_now    tokens over the last two samples (about 2 s); null before two samples exist
    prefill_tps       forwarded / prompt_ms, from the first token on; null while prefilling and
                      for a response-cache replay
    now.decode_tps    every in-flight request's tokens over the last two samples; null when
                      nothing decoded in that window
    sample            one per second: tokens over the second (all requests) and the rate of a
                      prefill that finished in that second, else null

CONTRACT 1.1, additive: every 1.0 field keeps its meaning. Each event
gains `seq` (also its SSE `id:`), an `engine` block (idle, busy, waiting_for_client, draining), a
per-request `activity` (server/activity.py: the state, its label, prefill progress, decode figures,
the tool being called, the client's socket, the stop) and `timeline` (the last 16 transitions), and
`recent` (the last 20 finished requests for 15 minutes, each with its stop sentence).

Cadence: while a request is in flight AND a stream is open the sampler ticks `QSE_LIVE_HZ` times a
second (default 4; 1, 2 or 4; 0 = the activity off); otherwise once a second, and not at all when
nothing is in flight and no stream is open. `history` keeps one sample a second either way. A tick
builds the snapshot and its bytes ONCE and every open stream writes those bytes. Per tick the
sampler also checks each in-flight request's socket (`select` plus `recv(MSG_PEEK)`, as the handler
does) so a client that left is on the page within one tick, whatever the handler is doing.
`--live-activity off` keeps contract 1.1 with the new fields null at one event a second.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import threading
import time

from server import activity as act

CONTRACT = "1.1"
INTERVAL_S = 1.0
KEEP_S = 30.0            # a finished request stays this long with its final numbers
HISTORY = 300            # samples kept: five minutes at one a second
MAX_STREAMS = 4
MINUTE_S = 60.0
DEFAULT_HZ = 4
HZ_ALLOWED = (0, 1, 2, 4)
RECENT_N = 20            # finished requests kept with their stop
RECENT_KEEP_S = 900.0    # ... for fifteen minutes
WAIT_KEEP_S = 600.0      # waiting_for_client ends after ten minutes
PING_S = 15.0            # a stream gets `: ping` this often


def hz_from_env(value) -> int:
    """`QSE_LIVE_HZ`: 1, 2 or 4 events a second while busy; 0 turns the activity off."""
    if value is None or str(value).strip() == "":
        return DEFAULT_HZ
    try:
        hz = int(str(value).strip())
    except ValueError:
        raise SystemExit(f"QSE_LIVE_HZ must be one of {HZ_ALLOWED}, got {value!r}") from None
    if hz not in HZ_ALLOWED:
        raise SystemExit(f"QSE_LIVE_HZ must be one of {HZ_ALLOWED}, got {hz}")
    return hz


def _r(x, nd: int = 2):
    return round(float(x), nd) if x is not None else None


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_ms(ts: float) -> str:
    d = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


_SEP = (",", ":")
_RAW_REQ, _RAW_RECENT = "\u0001requests\u0001", "\u0001recent\u0001"


def _dumps(x) -> str:
    return json.dumps(x, separators=_SEP)


def encode(snap: dict, raw: dict | None = None) -> bytes:
    """One `event: live` block: the id line is the snapshot's `seq`. `raw` carries parts already
    encoded (the finished rows and `recent`, which change only when a request ends), spliced
    in for the placeholders `_snapshot_locked(raw=True)` leaves."""
    body = _dumps(snap)
    if raw is not None:
        body = body.replace(_dumps(_RAW_REQ), raw["requests"], 1)
        body = body.replace(_dumps(_RAW_RECENT), raw["recent"], 1)
    return f"event: live\nid: {snap.get('seq', 0)}\ndata: {body}\n\n".encode()


class LiveRegistry:
    """The requests in flight, a 30 s tail of finished ones, a 5-minute ring of samples, and
    (1.1) the last 20 stops.

    `blocks` returns the decode loop's running `BlockStats` (or None); `last_prefill` returns
    `STATE["last_prefill"]`; `engine` returns the engine block's plain facts (model, version,
    draining, kv, memory, store). `clock` is wall time for the payload, `perf` the counter the
    records are stamped with; tests inject both.
    """

    def __init__(self, *, blocks=None, last_prefill=None, engine=None, clock=time.time,
                 perf=time.perf_counter, keep_s: float = KEEP_S, history: int = HISTORY,
                 interval: float = INTERVAL_S, hz: int = DEFAULT_HZ, activity: bool = True,
                 queue_timeout=None, sock_closed=None):
        self.blocks_fn = blocks or (lambda: None)
        self.last_prefill_fn = last_prefill or (lambda: None)
        self.engine_fn = engine or (lambda: {})
        self.queue_timeout_fn = queue_timeout or (lambda: None)
        self.sock_closed = sock_closed or act.socket_closed
        self.clock, self.perf = clock, perf
        self.keep_s, self.interval = float(keep_s), float(interval)
        self.hz = int(hz)
        self.activity = bool(activity) and self.hz > 0
        self.fast_interval = (1.0 / self.hz if self.activity and self.hz > 1 else self.interval)
        self._lock = threading.Lock()
        self._live: dict[str, object] = {}                  # request_id -> RequestRecord, arrival order
        self._done: collections.deque = collections.deque()  # (rec, ended_perf)
        self._ends: collections.deque = collections.deque()  # perf times of every finish (1-minute count)
        self._samples: collections.deque = collections.deque(maxlen=int(history))
        self._marks: dict[str, collections.deque] = {}      # request_id -> (perf, tokens) x 3
        self._fast: dict[str, collections.deque] = {}       # request_id -> (perf, tokens, blocks)
        self._tooltok: dict[str, list] = {}                 # request_id -> [tokens, open at n]
        self._recent: collections.deque = collections.deque(maxlen=RECENT_N)  # (ended, row)
        self._recent_raw: str | None = None                 # `recent` encoded, until it changes
        self._done_raw: dict[str, tuple] = {}               # a finished row, encoded once
        self._waiting: dict | None = None                   # the last finish was tool_calls
        self._changed_p = self.perf()                       # the engine's last arrival/finish
        self._finished_tokens = 0
        self._served = self._errors = self._refused = 0
        self._credited: set[str] = set()                    # prefills already put in a sample
        self._pending_prefill: list[float] = []             # rates of prefills finished between ticks
        self._last_prefill: tuple[float, float] | None = None   # (perf at first token, tok/s)
        self._prev: tuple[float, int, bool] | None = None   # (perf, tokens, something decoding)
        self._last_sample_p: float | None = None
        self._was_fast = False
        self.samples_taken = 0
        # the event every open stream writes: (seq, bytes), made once a tick
        self.seq = 0
        self._event: tuple[int, bytes] | None = None
        self.encodes = 0
        self.tick_us = {"n": 0, "last": 0.0, "max": 0.0, "sum": 0.0, "cpu_sum": 0.0,
                        "cpu_max": 0.0}
        # the sampler thread
        self._streams = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._tick_cv = threading.Condition()
        self._ticks = 0
        self._thread: threading.Thread | None = None
        self._active_until = 0.0

    # ----------------------------------------------------------------- per request (twice each)
    def register(self, rec) -> None:
        now = self.perf()
        with self._lock:
            self._live[rec.request_id] = rec
            self._changed_p = now
            w, self._waiting = self._waiting, None
            if w is not None and self.activity and now - w["ended"] <= WAIT_KEEP_S:
                conv = getattr(rec, "conv", None)
                rec.continues = {"request_id": w["request_id"],
                                 "gap_ms": _r((now - w["ended"]) * 1e3, 1),
                                 "tool_names": list(w["tool_names"]),
                                 "conv_match": bool(conv) and conv == w["conv"]}
        self._wake.set()

    def finish(self, rec) -> None:
        now = self.perf()
        recent = None
        if self.activity:
            end = rec.t_end if rec.t_end is not None else now
            try:
                rec.stop = act.stop_of(rec, end)
                recent = self._recent_row(rec, end)
            except act.READ_ERRORS:                          # never break the handler's finally
                rec.stop = recent = None
        with self._lock:
            self._live.pop(rec.request_id, None)
            self._marks.pop(rec.request_id, None)
            self._fast.pop(rec.request_id, None)
            tt = self._tooltok.pop(rec.request_id, None)
            if tt is not None:
                rec.tool_tokens_live = tt[0] + (int(rec.n_live) - tt[1] if tt[1] is not None else 0)
            self._done.append((rec, now))
            self._ends.append(now)
            self._changed_p = now
            self._finished_tokens += int(getattr(rec, "n_live", 0))
            if rec.finish_reason == "refused":
                self._refused += 1
            else:
                self._served += 1
                if rec.finish_reason == "error":
                    self._errors += 1
            tps = self._prefill_tps(rec, done=True)
            if tps is not None and rec.request_id not in self._credited:
                self._credited.add(rec.request_id)
                self._pending_prefill.append(tps)
                self._last_prefill = (rec.t_first, tps)
            if recent is not None:
                self._recent.append((now, recent))
                self._recent_raw = None
                stop = rec.stop or {}
                self._waiting = ({"request_id": rec.request_id, "ended": now,
                                  "tool_names": list(getattr(rec, "tool_names", None) or []),
                                  "conv": getattr(rec, "conv", None),
                                  "client_kind": rec.client_kind}
                                 if stop.get("reason") == "tool_calls" else None)
            self._active_until = now + 2 * self.interval     # one more tick closes the series
            self._prune(now)
        self._wake.set()

    def _recent_row(self, rec, end: float) -> dict:
        stop = rec.stop or {}
        path = act.path_of(rec)
        return {"request_id": rec.request_id, "ended_at": _iso(self.clock()),
                "client_kind": rec.client_kind, "path": path,
                "tokens": int(rec.completion_tokens or rec.n_live or 0),
                "elapsed_ms": _r((end - rec.t_arrival) * 1e3, 1),
                "ttft_ms": _r(rec.ttft_ms), "decode_tps": _r(rec.decode_tps),
                "tool_names": list(getattr(rec, "tool_names", None) or []),
                "stop": stop}

    # ----------------------------------------------------------------- the sampler
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="qse-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._tick_cv:
            self._tick_cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def subscribe(self) -> bool:
        """One more live stream; False when the cap is reached."""
        with self._lock:
            if self._streams >= MAX_STREAMS:
                return False
            self._streams += 1
        self._wake.set()
        return True

    def unsubscribe(self) -> None:
        with self._lock:
            self._streams = max(0, self._streams - 1)

    def wait_tick(self, timeout: float) -> bool:
        """Block until the sampler has taken its next sample (True) or `timeout` passed."""
        with self._tick_cv:
            n = self._ticks
            self._tick_cv.wait_for(lambda: self._ticks != n or self._stop.is_set(), timeout)
            return self._ticks != n

    def wait_event(self, after: int, timeout: float) -> bool:
        """Block until an event newer than `after` (a `seq`) is there (True) or `timeout` passed.
        A stream writer waits on this, not on `wait_tick`: a tick that lands between the writer's
        read of `event()` and its next wait is then written, not skipped."""
        with self._tick_cv:
            return self._tick_cv.wait_for(
                lambda: (self._event is not None and self._event[0] > after) or self._stop.is_set(),
                timeout) and self._event is not None and self._event[0] > after

    def _active(self) -> bool:
        with self._lock:
            return bool(self._live) or self._streams > 0 or self.perf() < self._active_until

    def _fast_now(self) -> bool:
        """The fast cadence: a request in flight and somebody watching (caller holds the lock)."""
        return self.activity and self.hz > 1 and bool(self._live) and self._streams > 0

    def cadence(self) -> float:
        with self._lock:
            return self.fast_interval if self._fast_now() else self.interval

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._active():
                self._wake.wait()
                self._wake.clear()
                continue
            self.tick()
            self._stop.wait(self.cadence())

    def event(self) -> tuple[int, bytes] | None:
        """The newest tick's `(seq, bytes)`: what every open stream writes."""
        return self._event

    def first_event(self) -> tuple[int, bytes]:
        """A new stream's first event: the full snapshot with `history` (its own encode)."""
        snap = self.snapshot(history=True)
        return snap["seq"], encode(snap)

    def tick(self) -> None:
        """One sample, and one encoded event when a stream is open. Public so tests can drive it
        with their own clock."""
        t_real, t_cpu = time.perf_counter(), time.thread_time()
        now_p, now_c = self.perf(), self.clock()
        if self.activity:
            # the client check, outside the lock: a syscall per in-flight request
            with self._lock:
                watched = [r for r in self._live.values()
                           if getattr(r, "sock", None) is not None
                           and getattr(r, "client_gone_at", None) is None]
            for r in watched:
                if self.sock_closed(r.sock):
                    r.client_gone_at = now_p
        snap = None
        with self._lock:
            live = list(self._live.values())
            fast = self._fast_now()
            # one history sample a second: every tick at 1 Hz, every fourth at 4 Hz, and no
            # extra one on the tick where the cadence drops back
            due = (self._last_sample_p is None
                   or now_p - self._last_sample_p >= 0.9 * self.interval
                   or not (fast or self._was_fast))
            self._was_fast = fast
            if self.activity:
                bs = self.blocks_fn()
                for r in live:
                    ring = self._fast.get(r.request_id)
                    if ring is None:
                        ring = self._fast[r.request_id] = collections.deque(
                            maxlen=2 * max(1, self.hz) + 2)
                    n = int(r.n_live)
                    ring.append((now_p, n, int(getattr(bs, "blocks", 0) or 0)
                                 if bs is not None and r.t_first is not None else 0))
                    self._tool_mark(r, n)
            if due:
                self._sample(live, now_p, now_c)
            self._prune(now_p)
            if self._streams > 0:
                snap, raw = self._snapshot_locked(now_p, now_c, history=False, raw=True)
        if snap is not None:
            self.seq += 1
            snap["seq"] = self.seq
            self._event = (self.seq, encode(snap, raw))
            self.encodes += 1
        # wall time includes waiting for the GIL behind the decode loop; CPU time is the cost
        us, cpu = (time.perf_counter() - t_real) * 1e6, (time.thread_time() - t_cpu) * 1e6
        t = self.tick_us
        t["n"] += 1
        t["last"] = us
        t["sum"] += us
        t["max"] = max(t["max"], us)
        t["cpu_sum"] += cpu
        t["cpu_max"] = max(t["cpu_max"], cpu)
        with self._tick_cv:
            self._ticks += 1
            self._tick_cv.notify_all()

    def _tool_mark(self, r, n: int) -> None:
        """Tokens written inside tool-call blocks, counted between the sampler's own looks
        (approximate: a block that opens and closes between two ticks is not counted)."""
        refs = getattr(r, "live_refs", None)
        tbuf = refs[1] if refs else None
        if tbuf is None:
            return
        is_open = act._safe(lambda: act._tool_open(tbuf), False)
        tt = self._tooltok.get(r.request_id)
        if tt is None:
            tt = self._tooltok[r.request_id] = [0, None]
        if is_open and tt[1] is None:
            tt[1] = n
        elif not is_open and tt[1] is not None:
            tt[0] += n - tt[1]
            tt[1] = None

    def _sample(self, live, now_p: float, now_c: float) -> None:
        """The one-a-second sample of, unchanged."""
        total = self._finished_tokens + sum(int(getattr(r, "n_live", 0)) for r in live)
        decoding = False
        for r in live:
            if r.t_first is None:
                continue
            decoding = True
            m = self._marks.get(r.request_id)
            if m is None:
                m = self._marks[r.request_id] = collections.deque(maxlen=3)
            m.append((now_p, int(r.n_live)))
            if r.request_id not in self._credited:
                tps = self._prefill_tps(r, done=False)
                if tps is not None:
                    self._credited.add(r.request_id)
                    self._pending_prefill.append(tps)
                    self._last_prefill = (r.t_first, tps)
        decode_tps = None
        if self._prev is not None:
            t0, n0, was = self._prev
            dt = now_p - t0
            if dt > 0 and (decoding or was):
                decode_tps = _r((total - n0) / dt)
        prefill_tps = _r(self._pending_prefill[-1]) if self._pending_prefill else None
        self._pending_prefill.clear()
        self._samples.append({"t": _r(now_c, 3), "decode_tps": decode_tps,
                              "prefill_tps": prefill_tps,
                              "running": sum(1 for r in live if r.t_first is not None),
                              "waiting": sum(1 for r in live if r.t_lock is None),
                              "tokens": total})
        self._prev = (now_p, total, decoding)
        self._last_sample_p = now_p
        self.samples_taken += 1

    # ----------------------------------------------------------------- the payload
    def snapshot(self, history: bool = True) -> dict:
        now_p, now_c = self.perf(), self.clock()
        with self._lock:
            self._prune(now_p)
            out = self._snapshot_locked(now_p, now_c, history=history)
        out["seq"] = self.seq
        return out

    def _snapshot_locked(self, now_p: float, now_c: float, *, history: bool,
                         raw: bool = False):
        """The payload. `raw=True` (the tick's event): the finished rows and `recent` come back
        already encoded, beside placeholders in the dict, as `(dict, raw parts)`."""
        live = list(self._live.values())
        bs = self.blocks_fn()
        lp = self.last_prefill_fn()
        rows = [self._row(r, now_p, None, bs) for r in live]
        rows.sort(key=lambda x: {"decode": 0, "prefill": 1, "queued": 2}[x["phase"]])
        acts: dict[str, dict] = {}
        if self.activity:
            by_id = {r.request_id: r for r in live}
            waiting = [r for r in live if r.t_lock is None]
            place = {r.request_id: i + 1 for i, r in enumerate(waiting)}
            for row in rows:
                rid = row["request_id"]
                a, tl = self._activity(by_id[rid], now_p, bs, lp, place.get(rid))
                row["activity"], row["timeline"] = a, tl
                acts[rid] = a
        else:
            for row in rows:
                row["activity"] = row["timeline"] = None
        raw_parts = None
        if raw:
            done = [self._done_json(r, e, now_p) for r, e in reversed(self._done)]
            raw_parts = {"requests": "[" + ",".join([_dumps(x) for x in rows] + done) + "]",
                         "recent": self._recent_json()}
            rows = _RAW_REQ
        else:
            for r, ended in reversed(self._done):
                row = self._row(r, now_p, ended, None)
                if self.activity:
                    row["activity"], row["timeline"] = self._done_activity(r, now_p, ended)
                else:
                    row["activity"] = row["timeline"] = None
                rows.append(row)
        samples = list(self._samples)
        counts = {"in_flight": len(live),
                  "queued": sum(1 for r in live if r.t_lock is None),
                  "prefilling": sum(1 for r in live if r.t_lock is not None
                                    and r.t_first is None),
                  "decoding": sum(1 for r in live if r.t_first is not None),
                  "completed_1m": sum(1 for t in self._ends if now_p - t <= MINUTE_S),
                  "served": self._served, "errors": self._errors,
                  "refused": self._refused}
        last = self._last_prefill
        out = {"contract_version": CONTRACT, "seq": self.seq,
               "generated_at": _iso_ms(now_c),
               "interval_s": self.fast_interval if self._fast_now() else self.interval,
               "engine": self._engine(now_p, live, acts),
               "counts": counts,
               "now": {"decode_tps": self._now_decode(samples),
                       "prefill_tps": _r(last[1]) if last else None,
                       "prefilling": counts["prefilling"] > 0,
                       "tokens_per_block": self._now_tpb(live, bs),
                       "last_prefill_ms_ago": _r((now_p - last[0]) * 1e3, 1) if last
                       else None},
               "requests": rows,
               "recent": (_RAW_RECENT if raw else
                          [row for _, row in reversed(self._recent)] if self.activity
                          else None),
               "sample": samples[-1] if samples else None,
               "sampler": {"ticks": self.tick_us["n"], "encodes": self.encodes,
                           "tick_us_last": _r(self.tick_us["last"], 1),
                           "tick_us_mean": _r(self.tick_us["sum"] / self.tick_us["n"], 1)
                           if self.tick_us["n"] else None,
                           "tick_us_max": _r(self.tick_us["max"], 1),
                           "tick_cpu_us_mean": _r(self.tick_us["cpu_sum"] / self.tick_us["n"], 1)
                           if self.tick_us["n"] else None,
                           "tick_cpu_us_max": _r(self.tick_us["cpu_max"], 1)}}
        if history:
            out["history"] = samples
        return (out, raw_parts) if raw else out

    def _recent_json(self) -> str:
        if not self.activity:
            return "null"
        if self._recent_raw is None:
            self._recent_raw = _dumps([row for _, row in reversed(self._recent)])
        return self._recent_raw

    def _done_json(self, rec, ended: float, now_p: float) -> str:
        """A finished row: everything but its age is fixed, so it is encoded once and the two
        age fields (`ended_ms_ago`, `activity.since_ms`) are spliced in per tick."""
        c = self._done_raw.get(rec.request_id)
        if c is None:
            row = self._row(rec, now_p, ended, None)
            row.pop("ended_ms_ago")
            act_json = None
            if self.activity:
                a, row["timeline"] = self._done_activity(rec, now_p, ended)
                a.pop("since_ms")
                act_json = _dumps(a)
            else:
                row["activity"] = row["timeline"] = None
            c = self._done_raw[rec.request_id] = (_dumps(row), act_json)
        ago = repr(_r((now_p - ended) * 1e3, 1))
        head = '{"ended_ms_ago":' + ago
        if c[1] is not None:
            head += ',"activity":{"since_ms":' + ago + "," + c[1][1:]
        return head + "," + c[0][1:]

    # ----------------------------------------------------------------- 1.1: the activity
    def _fast_mark(self, rid: str, now_p: float):
        ring = self._fast.get(rid)
        if not ring:
            return None
        best = None
        for t, n, b in ring:
            if now_p - t >= 0.9:
                best = (t, n, b)
        return best

    def _continues(self, rec) -> dict | None:
        c = getattr(rec, "continues", None)
        if not c:
            return None
        out = {"request_id": c["request_id"], "gap_ms": c["gap_ms"],
               "tool_names": c["tool_names"], "inferred": False}
        if c.get("conv_match"):
            return out
        info = getattr(rec, "prefill_info", None) or {}
        kind, start = info.get("kind"), info.get("start")
        if kind in ("session", "resident") and start:
            out["inferred"] = True
            return out
        return None

    def _activity(self, rec, now_p: float, bs, lp, place) -> tuple[dict, list]:
        state = act.state_of(rec)
        events = act.events_of(rec)
        since = (now_p - events[-1][0]) * 1e3 if events else None
        refs = getattr(rec, "live_refs", None) or (None, None)
        think, tbuf = refs
        running_bs = bs if (rec.t_first is not None and rec.cache_source != "response") else None
        tt = self._tooltok.get(rec.request_id)
        tool_tokens = (tt[0] + (int(rec.n_live) - tt[1] if tt[1] is not None else 0)) if tt else 0
        gone = getattr(rec, "client_gone_at", None)
        sock = getattr(rec, "sock", None)
        last = rec.t_last if rec.t_last is not None else (rec.t_lock or rec.t_arrival)
        tool = act.tool_of(rec) if tbuf is not None else None
        if tool is not None and state != "tool_call" and not tool.get("calls_done"):
            tool = None
        a = {"state": state, "label": "", "since_ms": _r(since, 1),
             "constrained": getattr(rec, "constrained", None),
             "queue": ({"place": place, "wait_ms": _r((now_p - rec.t_arrival) * 1e3, 1),
                        "timeout_s": self.queue_timeout_fn(), "place_is_estimate": True}
                       if state == "queued" else None),
             "prefill": act.prefill_of(rec, now_p, lp if rec.prompt_n is None else None),
             "decode": act.decode_of(rec, now_p, running_bs,
                                     self._fast_mark(rec.request_id, now_p), tool_tokens),
             "tool": tool,
             "reasoning": ({"closed_by": act.closed_by(think),
                            "tokens": act._safe(lambda: int(think.n))}
                           if think is not None and rec.thinking else None),
             "client": {"connected": (gone is None) if sock is not None else None,
                        "silent_ms": _r((now_p - last) * 1e3, 1),
                        "gone_ms": _r((now_p - gone) * 1e3, 1) if gone is not None else None},
             "continues": self._continues(rec),
             "step": getattr(rec, "step", None) if state == "finishing" else None,
             "stop": None}
        a["label"] = act.label(a)
        if state == "prefilling":
            # the vision tower runs inside the prefill; the state stays `prefilling`
            # (the contract's enum) and the label says which image is being encoded
            enc = act.encoding_label(rec)
            if enc:
                a["label"] = enc
        return a, act.timeline_of(rec)

    def _done_activity(self, rec, now_p: float, ended: float) -> tuple[dict, list]:
        stop = getattr(rec, "stop", None) or {}
        end = rec.t_end if rec.t_end is not None else ended
        refs = getattr(rec, "live_refs", None) or (None, None)
        think = refs[0]
        gone = getattr(rec, "client_gone_at", None)
        a = {"state": "done", "label": "", "since_ms": _r((now_p - ended) * 1e3, 1),
             "constrained": getattr(rec, "constrained", None), "queue": None,
             "prefill": act.prefill_of(rec, now_p),
             "decode": act.decode_of(rec, now_p, None, None,
                                     int(getattr(rec, "tool_tokens_live", 0) or 0), live=False),
             "tool": None,
             "reasoning": ({"closed_by": act.closed_by(think),
                            "tokens": act._safe(lambda: int(think.n))}
                           if think is not None and rec.thinking else None),
             "client": {"connected": (gone is None) if getattr(rec, "sock", None) is not None
                        else None, "silent_ms": stop.get("silent_ms"),
                        "gone_ms": None},
             "continues": self._continues(rec), "step": None, "stop": stop or None}
        a["label"] = act.label(a)
        return a, act.timeline_of(rec, end, stop.get("reason"))

    def _engine(self, now_p: float, live, acts: dict) -> dict:
        e = {}
        try:
            e = dict(self.engine_fn() or {})
        except Exception:                                  # noqa: BLE001 -- a display, never a crash
            e = {}
        waiting = None
        w = self._waiting
        if self.activity and w is not None and not live and now_p - w["ended"] <= WAIT_KEEP_S:
            waiting = {"request_id": w["request_id"], "tool_names": list(w["tool_names"]),
                       "since_ms": _r((now_p - w["ended"]) * 1e3, 1),
                       "client_kind": w["client_kind"]}
        if e.get("draining"):
            state, label, since = "draining", "Shutting down", None
        elif live:
            running = next((r for r in live if r.t_lock is not None), live[0])
            a = acts.get(running.request_id)
            state = "busy"
            label = a["label"] if a else ("Queued" if running.t_lock is None else "Busy")
            since = a["since_ms"] if a else None
        elif waiting is not None:
            state, label = "waiting_for_client", act.waiting_label(waiting)
            since = waiting["since_ms"]
        else:
            state, label = "idle", "Idle"
            since = _r((now_p - self._changed_p) * 1e3, 1)
        return {"state": state, "label": label, "since_ms": since,
                "model": e.get("model"), "version": e.get("version"),
                "draining": bool(e.get("draining")), "kv": e.get("kv"),
                "memory": e.get("memory"), "store": e.get("store"),
                "waiting_for_client": waiting}

    # ----------------------------------------------------------------- 1.0 helpers
    @staticmethod
    def _now_decode(samples: list) -> float | None:
        if len(samples) < 3:
            return None
        a, b = samples[-3], samples[-1]
        if not any(s["running"] for s in samples[-3:]):
            return None
        dt = b["t"] - a["t"]
        return _r((b["tokens"] - a["tokens"]) / dt) if dt > 0 else None

    @staticmethod
    def _now_tpb(live, bs) -> float | None:
        if bs is None or not getattr(bs, "blocks", 0):
            return None
        rec = next((r for r in live if r.t_first is not None), None)
        if rec is None or rec.n_live < 2:
            return None
        return _r((rec.n_live - 1) / bs.blocks)

    def _prefill_tps(self, rec, *, done: bool) -> float | None:
        if rec.t_first is None or rec.cache_source == "response":
            return None
        if done or rec.prompt_n is not None:
            return rec.prefill_tps
        lp = self.last_prefill_fn()
        if not lp:
            return None
        ms = rec.prompt_ms
        fwd = int(lp.get("forwarded") or 0)
        return fwd / (ms / 1e3) if fwd and ms else None

    def _row(self, rec, now_p: float, ended: float | None, bs) -> dict:
        live = ended is None
        if not live:
            phase = "done"
        elif rec.t_lock is None:
            phase = "queued"
        elif rec.t_first is None:
            phase = "prefill"
        else:
            phase = "decode"
        n = int(rec.n_live) if live else int(rec.completion_tokens)
        cached, forwarded, source = rec.cached_tokens, rec.prompt_n, rec.cache_source
        prefill_tps = self._prefill_tps(rec, done=not live)
        if live and phase == "decode" and rec.prompt_n is None and source != "response":
            lp = self.last_prefill_fn() or {}
            cached = int(lp.get("reused") or 0)
            forwarded = int(lp.get("forwarded") or 0)
            if cached:
                source = str(lp.get("kind") or "prefix")
        elif forwarded is None and rec.prompt_tokens is not None:
            forwarded = int(rec.prompt_tokens) - int(cached or 0)
        if phase == "decode":
            blocks = int(getattr(bs, "blocks", 0)) if bs is not None else 0
            decode_ms = (now_p - rec.t_first) * 1e3
            decode_tps = (n - 1) / (decode_ms / 1e3) if n > 1 and decode_ms > 0 else None
            elapsed = (now_p - rec.t_arrival) * 1e3
        elif phase == "done":
            blocks = int(rec.blocks or 0)
            decode_ms, decode_tps, elapsed = rec.predicted_ms, rec.decode_tps, rec.total_ms
        else:
            blocks, decode_ms, decode_tps = 0, None, None
            elapsed = (now_p - rec.t_arrival) * 1e3
        marks = self._marks.get(rec.request_id) if live else None
        now_tps = None
        if marks and len(marks) >= 2:
            (t0, n0), (t1, n1) = marks[0], marks[-1]
            now_tps = (n1 - n0) / (t1 - t0) if t1 > t0 else None
        return {
            "request_id": rec.request_id, "phase": phase,
            "finish_reason": rec.finish_reason if phase == "done" else None,
            "status": int(rec.status), "model": rec.model,
            "client": {"id": rec.client_id, "kind": rec.client_kind},
            "endpoint": rec.endpoint, "stream": bool(rec.stream), "thinking": bool(rec.thinking),
            "temperature": _r(getattr(rec, "temperature", None)),
            "prompt_tokens": rec.prompt_tokens,
            "cached_tokens": int(cached or 0) if rec.prompt_tokens is not None else None,
            "forwarded_tokens": forwarded if rec.prompt_tokens is not None else None,
            "tokens": n, "blocks": blocks or None,
            "tokens_per_block": _r((n - 1) / blocks) if blocks and n > 1 else None,
            "elapsed_ms": _r(elapsed), "queue_ms": _r(rec.queue_ms),
            "prompt_ms": _r(rec.prompt_ms), "ttft_ms": _r(rec.ttft_ms),
            "decode_ms": _r(decode_ms), "prefill_tps": _r(prefill_tps),
            "decode_tps": _r(decode_tps), "decode_tps_now": _r(now_tps),
            "cache_source": source if rec.prompt_tokens is not None else None,
            "max_tokens": rec.max_tokens,
            "ended_ms_ago": _r((now_p - ended) * 1e3, 1) if ended is not None else None,
        }

    def _prune(self, now_p: float) -> None:
        while self._done and now_p - self._done[0][1] > self.keep_s:
            rec, _ = self._done.popleft()
            self._credited.discard(rec.request_id)
            self._done_raw.pop(rec.request_id, None)
        while self._ends and now_p - self._ends[0] > MINUTE_S:
            self._ends.popleft()
        while self._recent and now_p - self._recent[0][0] > RECENT_KEEP_S:
            self._recent.popleft()
            self._recent_raw = None
