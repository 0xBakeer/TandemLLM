"""The live view of the requests in flight (SRV-34): `GET /v1/dashboard/live`.

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
writes `timings` on the finish chunk (SRV-27): `decode_tps` divides committed tokens (the first
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
"""

from __future__ import annotations

import collections
import datetime as _dt
import threading
import time

CONTRACT = "1.0"
INTERVAL_S = 1.0
KEEP_S = 30.0            # a finished request stays this long with its final numbers
HISTORY = 300            # samples kept: five minutes at one a second
MAX_STREAMS = 4
MINUTE_S = 60.0


def _r(x, nd: int = 2):
    return round(float(x), nd) if x is not None else None


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LiveRegistry:
    """The requests in flight, a 30 s tail of finished ones, and a 5-minute ring of samples.

    `blocks` returns the decode loop's running `BlockStats` (or None); `last_prefill` returns
    `STATE["last_prefill"]`. `clock` is wall time for the payload, `perf` the counter the records
    are stamped with; tests inject both.
    """

    def __init__(self, *, blocks=None, last_prefill=None, clock=time.time,
                 perf=time.perf_counter, keep_s: float = KEEP_S, history: int = HISTORY,
                 interval: float = INTERVAL_S):
        self.blocks_fn = blocks or (lambda: None)
        self.last_prefill_fn = last_prefill or (lambda: None)
        self.clock, self.perf = clock, perf
        self.keep_s, self.interval = float(keep_s), float(interval)
        self._lock = threading.Lock()
        self._live: dict[str, object] = {}                  # request_id -> RequestRecord, arrival order
        self._done: collections.deque = collections.deque()  # (rec, ended_perf)
        self._ends: collections.deque = collections.deque()  # perf times of every finish (1-minute count)
        self._samples: collections.deque = collections.deque(maxlen=int(history))
        self._marks: dict[str, collections.deque] = {}      # request_id -> (perf, tokens) x 3
        self._finished_tokens = 0
        self._served = self._errors = self._refused = 0
        self._credited: set[str] = set()                    # prefills already put in a sample
        self._pending_prefill: list[float] = []             # rates of prefills finished between ticks
        self._last_prefill: tuple[float, float] | None = None   # (perf at first token, tok/s)
        self._prev: tuple[float, int, bool] | None = None   # (perf, tokens, something decoding)
        self.samples_taken = 0
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
        with self._lock:
            self._live[rec.request_id] = rec
        self._wake.set()

    def finish(self, rec) -> None:
        now = self.perf()
        with self._lock:
            self._live.pop(rec.request_id, None)
            self._marks.pop(rec.request_id, None)
            self._done.append((rec, now))
            self._ends.append(now)
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
            self._active_until = now + 2 * self.interval     # one more tick closes the series
            self._prune(now)
        self._wake.set()

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

    def _active(self) -> bool:
        with self._lock:
            return bool(self._live) or self._streams > 0 or self.perf() < self._active_until

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._active():
                self._wake.wait()
                self._wake.clear()
                continue
            self.tick()
            self._stop.wait(self.interval)

    def tick(self) -> None:
        """One sample. Public so tests can drive it with their own clock."""
        now_p, now_c = self.perf(), self.clock()
        with self._lock:
            live = list(self._live.values())
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
            self.samples_taken += 1
            self._prune(now_p)
        with self._tick_cv:
            self._ticks += 1
            self._tick_cv.notify_all()

    # ----------------------------------------------------------------- the payload
    def snapshot(self, history: bool = True) -> dict:
        now_p, now_c = self.perf(), self.clock()
        with self._lock:
            self._prune(now_p)
            live = list(self._live.values())
            bs = self.blocks_fn()
            rows = [self._row(r, now_p, None, bs) for r in live]
            rows.sort(key=lambda x: {"decode": 0, "prefill": 1, "queued": 2}[x["phase"]])
            rows += [self._row(r, now_p, ended, None) for r, ended in reversed(self._done)]
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
            out = {"contract_version": CONTRACT, "generated_at": _iso(now_c),
                   "interval_s": self.interval, "counts": counts,
                   "now": {"decode_tps": self._now_decode(samples),
                           "prefill_tps": _r(last[1]) if last else None,
                           "prefilling": counts["prefilling"] > 0,
                           "tokens_per_block": self._now_tpb(live, bs),
                           "last_prefill_ms_ago": _r((now_p - last[0]) * 1e3, 1) if last
                           else None},
                   "requests": rows,
                   "sample": samples[-1] if samples else None}
            if history:
                out["history"] = samples
            return out

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
        while self._ends and now_p - self._ends[0] > MINUTE_S:
            self._ends.popleft()
