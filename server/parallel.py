"""Every request in flight at once, on every backend: `GET /v1/dashboard/parallel`.

The Live panel (server/live.py) follows the engine, which serves one request at a time. With the
proxy (proxy/) one server relays to our engine, vLLM, SGLang and cloud APIs side by side, so tens
of streams can run together. This module keeps one lane per request in flight and the sums over
them, per backend and in total, and answers them as one JSON document.

WHERE THE NUMBERS COME FROM. Nothing is estimated; a figure a backend does not give is null and the
page says "not reported".

    the engine's own requests   the same record the Live panel reads: tokens (`n_live`), the
                                prefill's progress and rate (server/activity.py), the state
    relayed requests            the proxy's record (proxy/measure.py): TTFT at the proxy, the
                                token frames with their arrival times, what the newest frame
                                carried (reasoning, content, a tool call); at the end the
                                backend's `usage` (tokens, cached tokens) and the cost
    a frame is a token          only where it was checked: after each streamed answer the frames
                                are compared with the `usage.completion_tokens` the backend sent
                                (within 2 tokens + 1 %: a role frame, a think-end token), apart
                                for answers with text only and answers with tool calls (DeepSeek
                                streams a call's arguments several tokens a frame). A backend and
                                model with 2 passing answers of a kind and no failing one among
                                its last 8 counts those frames as tokens live; until then the
                                lane's live tokens and tok/s are null and it shows its frames
                                (and a call's argument characters)
    cost so far                 completion tokens so far x the output price, marked partial: the
                                prompt's part is known when the usage arrives

THE HOT PATH. Per token the relay already did `frames += 1`; it now also writes the frame's kind
(one attribute). The board does its work in the live registry's once-a-second sample (under that
lock: a few integer reads per request) and when a viewer asks, at most every `BUILD_S` whatever
the number of viewers; a poll with `since` gets the series' new samples only.

Definitions (also docs/contract/dashboard-v1/parallel.schema.json):

    lane.state          queued, prefilling (engine), waiting (relayed: sent, no token yet),
                        thinking, tool_call, writing, finishing, done
    lane.decode_now     tokens over the last NOW_S (2 s) of samples (a lane younger than that:
                        its decode_avg); null without a live count
    lane.decode_avg     (tokens - 1) / (last token - first token)
    lane.ttft_ms        arrival to the first token (relayed: at the proxy, network included)
    totals.decode_tps   the sum of the lanes' decode_now; `decode_unreported` lanes decode
                        without a live count
    series              one sample a second per backend: the tokens its lanes decoded in that
                        second (a lane without a live count adds nothing)
    window              TTFT percentiles, cache hit and cost are over requests finished in the
                        last STATS_S (300 s); cost_per_min over the last 60 s
"""

from __future__ import annotations

import collections
import json
import threading
import time

from server import activity as act

CONTRACT = "1.0"
NOW_S = 2.0              # the lanes' now-rate window
DONE_KEEP_S = 8.0        # a finished lane stays this long, with its final numbers
STATS_S = 300.0          # TTFT, cache hit, cost: requests finished in the last five minutes
COST_MIN_S = 60.0        # cost per minute: the last minute
HISTORY = 300            # series samples kept, one a second
BUILD_S = 0.4            # a snapshot is built at most this often, whatever the viewers
FRAME_TOL_TOKENS = 2     # frames vs usage: a role frame, a think-end token the stream never shows
FRAME_TOL_PCT = 1.0
FRAME_PASS_N = 2         # passing answers before frames count as tokens
FRAME_KEEP = 8           # ... with no failing answer among the last this many
ENGINE_ID = "engine"     # the backend id of the engine's own requests


def _r(x, nd: int = 2):
    return round(float(x), nd) if x is not None else None


def _pct(sorted_xs: list, q: float):
    if not sorted_xs:
        return None
    k = min(len(sorted_xs) - 1, max(0, int(round(q * (len(sorted_xs) - 1)))))
    return sorted_xs[k]


def is_relayed(rec) -> bool:
    return getattr(rec, "backend", None) is not None and hasattr(rec, "frames")


class FrameCheck:
    """Per backend and model: do its stream frames carry one token each? Fed by finished
    streamed answers, in two kinds: answers with text only (reasoning, content) and answers
    with tool calls (some APIs send a call's arguments several tokens a frame). A kind is
    `verified` once FRAME_PASS_N of its answers passed and none of its last FRAME_KEEP failed."""

    def __init__(self):
        self.text: collections.deque = collections.deque(maxlen=FRAME_KEEP)
        self.tools: collections.deque = collections.deque(maxlen=FRAME_KEEP)
        self.last = None              # (frames, tokens) of the newest answer

    @property
    def results(self) -> list:
        return list(self.text) + list(self.tools)

    def add(self, frames: int, tokens: int, tools: bool = False) -> bool:
        ok = abs(tokens - frames) <= FRAME_TOL_TOKENS + tokens * FRAME_TOL_PCT / 100.0
        (self.tools if tools else self.text).append(ok)
        self.last = (frames, tokens)
        return ok

    @staticmethod
    def _ok(d) -> bool:
        return sum(d) >= FRAME_PASS_N and all(d)

    @property
    def verified(self) -> bool:
        """Text frames are tokens."""
        return self._ok(self.text)

    @property
    def tools_verified(self) -> bool:
        """Tool-call frames are tokens too."""
        return self._ok(self.text) and self._ok(self.tools)

    def view(self) -> dict:
        return {"verified": self.verified, "tools_verified": self.tools_verified,
                "checked": len(self.results), "passed": int(sum(self.results)),
                "last": ({"frames": self.last[0], "tokens": self.last[1]}
                         if self.last else None)}


class ParallelBoard:
    """The lanes and the sums. Attach it to a `LiveRegistry` (it adds itself to `observers`).

    `engine_name` returns the engine's served model name (None on a proxy without an engine);
    `cost` returns a relayed request's cost of `tokens` completion tokens `(rec, tokens) ->
    micro-dollars | None` at its model's price (proxy/service.py `live_cost`); `kinds` returns `{backend id: kind}` for backends with
    nothing in flight (the per-backend rows list every backend that ran in the window)."""

    def __init__(self, registry, *, engine_name=None, cost=None, clock=time.time,
                 perf=time.perf_counter):
        self.reg = registry
        self.engine_name = engine_name or (lambda: None)
        self.cost = cost or (lambda rec, tokens: None)
        self.clock, self.perf = clock, perf
        self._lock = threading.Lock()
        self._marks: dict[str, collections.deque] = {}     # rid -> (perf, tokens) per sample
        self._last: dict[str, tuple] = {}                  # rid -> (backend, tokens) at the last sample
        self._ended: dict[str, int] = {}                   # rid -> tokens decoded after its last sample
        self._done: collections.deque = collections.deque()  # (end perf, lane, stats row)
        self._series: collections.deque = collections.deque(maxlen=HISTORY)
        self._frames: dict[tuple, FrameCheck] = {}
        self._kinds: dict[str, str] = {}
        self._built: tuple[float, dict] | None = None
        self.builds = 0
        self.build_us = {"last": 0.0, "max": 0.0}
        registry.observers.append(self)

    # ----------------------------------------------------------------- tokens of a record
    def _frame_check(self, rec) -> FrameCheck:
        key = (rec.backend, getattr(rec, "backend_model", None) or rec.model)
        fc = self._frames.get(key)
        if fc is None:
            fc = self._frames[key] = FrameCheck()
        return fc

    def _check(self, rec) -> FrameCheck | None:
        return self._frames.get((rec.backend, getattr(rec, "backend_model", None) or rec.model))

    def _verified(self, rec) -> bool:
        """The record's frames so far are its tokens: text frames checked, and tool-call frames
        checked too when it has any."""
        fc = self._check(rec)
        if fc is None or not fc.verified:
            return False
        return int(getattr(rec, "frames_tool", 0) or 0) == 0 or fc.tools_verified

    def tokens_of(self, rec) -> tuple[int | None, str | None]:
        """(tokens so far, where the count comes from) of a record in flight."""
        if not is_relayed(rec):
            return int(getattr(rec, "n_live", 0) or 0), "engine"
        frames = int(getattr(rec, "frames", 0) or 0)
        if frames == 0:
            return 0, "frames"
        if self._verified(rec):
            return frames, "frames"
        return None, None

    # ----------------------------------------------------------------- the registry's calls
    def on_sample(self, live, now_p: float, now_c: float) -> None:
        """Once a second, under the registry's lock: each lane's (t, tokens), and the series'
        sample (per backend, the tokens decoded since the last sample)."""
        by: dict[str, float] = {}
        with self._lock:
            seen = set()
            for rec in live:
                rid = rec.request_id
                seen.add(rid)
                bid = self._bid(rec)
                if rec.t_first is None:
                    continue
                n, _src = self.tokens_of(rec)
                if n is None:
                    continue
                ring = self._marks.get(rid)
                if ring is None:
                    ring = self._marks[rid] = collections.deque(maxlen=int(NOW_S) + 3)
                ring.append((now_p, n))
                prev = self._last.get(rid)
                by[bid] = by.get(bid, 0.0) + max(0, n - (prev[1] if prev else 0))
                self._last[rid] = (bid, n)
            for rid, extra in self._ended.items():
                bid = self._last.pop(rid, (None, 0))[0]
                if bid is not None:
                    by[bid] = by.get(bid, 0.0) + max(0, extra)
            self._ended.clear()
            for rid in [r for r in self._last if r not in seen]:
                self._last.pop(rid, None)
            prev_t = self._series[-1]["_p"] if self._series else None
            dt = (now_p - prev_t) if prev_t is not None else 1.0
            dt = dt if 0.2 < dt < 5.0 else 1.0
            self._series.append({"t": _r(now_c, 3), "_p": now_p,
                                 "by": {k: _r(v / dt, 1) for k, v in by.items()},
                                 "in_flight": len(live)})

    def on_finish(self, rec, now_p: float) -> None:
        """A request ended: its final lane (kept DONE_KEEP_S), its stats row (STATS_S), the
        frame check of a streamed relayed answer, and the tokens it decoded after the last
        sample (for the series)."""
        tokens = self._final_tokens(rec)
        with self._lock:
            hx = getattr(rec, "hx", None)
            single = hx is None or len(getattr(hx, "usages", None) or []) <= 1
            if (is_relayed(rec) and rec.stream and single and int(rec.frames or 0) > 0
                    and getattr(rec, "usage_source", None) == "backend" and rec.completion_tokens
                    and rec.finish_reason in ("stop", "length", "tool_calls")):
                self._frame_check(rec).add(int(rec.frames), int(rec.completion_tokens),
                                           tools=int(getattr(rec, "frames_tool", 0) or 0) > 0)
            last = self._last.get(rec.request_id)
            if last is not None and tokens is not None:
                self._ended[rec.request_id] = max(0, tokens - last[1])
            self._marks.pop(rec.request_id, None)
            lane = self._lane(rec, now_p, ended=now_p, final_tokens=tokens)
            self._done.append((now_p, lane, self._stats_row(rec, lane, now_p)))
            self._prune(now_p)
            self._built = None

    # ----------------------------------------------------------------- one lane
    def _bid(self, rec) -> str:
        return rec.backend if is_relayed(rec) else ENGINE_ID

    def _final_tokens(self, rec) -> int | None:
        if is_relayed(rec):
            if getattr(rec, "usage_source", None) == "backend" and rec.completion_tokens is not None:
                return int(rec.completion_tokens)
            return None
        return int(rec.completion_tokens or rec.n_live or 0)

    def _state(self, rec, ended: float | None) -> str:
        if ended is not None:
            return "done"
        if not is_relayed(rec):
            s = act.state_of(rec)
            return "replaying" if s == "replaying" else s
        if getattr(rec, "t_sent", None) is None:
            return "queued"
        if rec.t_first is None:
            return "waiting"
        k = getattr(rec, "live_kind", None)
        return {"reasoning": "thinking", "tool": "tool_call"}.get(k, "writing")

    def _lane(self, rec, now_p: float, ended: float | None = None,
              final_tokens: int | None = None) -> dict:
        relayed = is_relayed(rec)
        bid = self._bid(rec)
        kind = (getattr(rec, "backend_kind", None) or "generic") if relayed else "tandemllm"
        model = ((getattr(rec, "backend_model", None) or rec.model) if relayed
                 else (getattr(rec, "model_served", None) or self.engine_name() or rec.model))
        state = self._state(rec, ended)
        end = ended if ended is not None else now_p
        if ended is not None:
            tokens, src = final_tokens, ("usage" if relayed else "engine")
            if tokens is None and relayed and getattr(rec, "frames", 0) and self._verified(rec):
                tokens, src = int(rec.frames), "frames"
        else:
            tokens, src = self.tokens_of(rec)
        # decode rates
        now_tps = avg_tps = None
        t_last = rec.t_last
        if rec.t_first is not None and tokens is not None and tokens > 1:
            if relayed:
                span = (t_last - rec.t_first) if t_last is not None else None
                n_span = tokens               # frames checked as tokens, or the usage at the end
                if span and span > 0 and n_span and n_span > 1:
                    avg_tps = (n_span - 1) / span
            else:
                span = ((t_last if ended is not None and t_last is not None else end)
                        - rec.t_first)
                if span > 0:
                    avg_tps = (tokens - 1) / span
        if ended is None:
            ring = self._marks.get(rec.request_id)
            if ring and len(ring) >= 2:
                t1, n1 = ring[-1]
                i = len(ring) - 2
                while i > 0 and t1 - ring[i - 1][0] <= NOW_S * 1.05:
                    i -= 1
                t0, n0 = ring[i]
                if t1 > t0:
                    now_tps = (n1 - n0) / (t1 - t0)
            elif tokens is not None and avg_tps is not None:
                # younger than the window: its whole decode so far is the last NOW_S
                now_tps = avg_tps
            # a stream that went quiet (a tool call the backend holds, a stall) reads 0
            if now_tps is not None and relayed and t_last is not None and now_p - t_last > NOW_S:
                now_tps = 0.0
        # thinking tokens
        thinking = None
        if relayed:
            if src == "frames":
                thinking = int(getattr(rec, "frames_reasoning", 0) or 0)
            elif ended is not None and getattr(rec, "reasoning_tokens", None) is not None:
                thinking = int(rec.reasoning_tokens)
        else:
            think = (getattr(rec, "live_refs", None) or (None, None))[0]
            if think is not None and rec.thinking:
                thinking = act._safe(lambda: min(int(think.n), int(tokens or 0)))
            elif ended is not None:
                thinking = int(rec.reasoning_tokens or 0) if rec.thinking else 0
        # prefill
        prefill = None
        if not relayed and rec.t_lock is not None:
            p = act.prefill_of(rec, now_p, self.reg.last_prefill_fn() if ended is None else None)
            if p:
                prefill = {"done": p.get("done"), "total": p.get("total"), "pct": p.get("pct"),
                           "cached": p.get("cached"),
                           "tps": p.get("tps_live") or p.get("tps_avg"),
                           "tps_avg": p.get("tps_avg"), "eta_ms": p.get("eta_ms")}
        # the prompt and its cache
        prompt = rec.prompt_tokens
        cached = rec.cached_tokens if (not relayed or rec.usage_source == "backend") else None
        if relayed and ended is None:
            prompt = cached = None                       # the usage arrives at the end
        if not relayed and ended is None:
            cached = (prefill or {}).get("cached")       # the record absorbs it at the end
        hit = (cached / prompt) if (cached is not None and prompt) else None
        # the cost
        cost, partial = None, False
        if relayed:
            if ended is not None:
                cost = getattr(rec, "cost_micros", None)
            else:
                if tokens:
                    try:
                        cost = self.cost(rec, tokens)
                    except Exception:                              # noqa: BLE001
                        cost = None
                    partial = cost is not None
        tool = None
        if state == "tool_call":
            if relayed:
                tool = getattr(rec, "live_tool", None)
            else:
                tool = (act.tool_of(rec) or {}).get("name")
        ttft = None
        if rec.t_first is not None:
            ttft = (rec.t_first - rec.t_arrival) * 1e3
        label = None
        if not relayed and ended is None:
            label = act.label({"state": state, "prefill": prefill and {
                "done": prefill["done"], "total": prefill["total"], "pct": prefill["pct"]},
                "tool": {"name": tool}})
        lane = {
            "id": rec.request_id, "backend": bid, "kind": kind, "model": model,
            "client": {"kind": rec.client_kind, "user": getattr(rec, "user", None),
                       "session": getattr(rec, "conv", None)},
            "state": state, "label": label, "tool": tool,
            "elapsed_ms": _r((end - rec.t_arrival) * 1e3, 0),
            "ttft_ms": _r(ttft, 1),
            "waiting_ms": _r((end - rec.t_arrival) * 1e3, 0) if rec.t_first is None else None,
            "tokens": tokens, "tokens_source": src,
            "frames": int(rec.frames) if relayed else None,
            "tool_frames": int(getattr(rec, "frames_tool", 0) or 0) if relayed else None,
            "tool_chars": int(getattr(rec, "tool_chars", 0) or 0) if relayed else None,
            "thinking_tokens": thinking,
            "decode_now": _r(now_tps, 1), "decode_avg": _r(avg_tps, 1),
            "prefill": prefill,
            "prompt_tokens": prompt, "cached_tokens": cached, "cache_hit": _r(hit, 4),
            "cost_micros": cost, "cost_partial": partial,
            "max_tokens": getattr(rec, "max_tokens", None),
        }
        if ended is not None:
            lane["finish"] = rec.finish_reason
            lane["status"] = int(rec.status or 0)
            lane["decode_now"] = None
        return lane

    def _stats_row(self, rec, lane: dict, now_p: float) -> dict:
        return {"backend": lane["backend"], "kind": lane["kind"], "ttft_ms": lane["ttft_ms"],
                "prompt": lane["prompt_tokens"], "cached": lane["cached_tokens"],
                "cost": lane["cost_micros"], "tokens": lane["tokens"],
                "error": rec.finish_reason in ("error", "timeout") or int(rec.status or 0) >= 500,
                "refused": rec.finish_reason == "refused", "decode_avg": lane["decode_avg"]}

    def _prune(self, now_p: float) -> None:
        while self._done and now_p - self._done[0][0] > STATS_S:
            self._done.popleft()

    # ----------------------------------------------------------------- the document
    def snapshot(self, since: float | None = None) -> dict:
        """The lanes, the sums and the series (all of it, or the samples after `since`)."""
        now_p = self.perf()
        built = self._built
        if built is None or now_p - built[0] > BUILD_S:
            t0 = time.perf_counter()
            with self.reg._lock:
                live = list(self.reg._live.values())
            with self._lock:
                self._prune(now_p)
                doc = self._build(live, now_p)
            self._built = built = (now_p, doc)
            self.builds += 1
            us = (time.perf_counter() - t0) * 1e6
            self.build_us["last"] = _r(us, 0)
            self.build_us["max"] = max(self.build_us["max"], _r(us, 0))
        doc = dict(built[1])
        with self._lock:
            ser = [{"t": s["t"], "by": s["by"], "in_flight": s["in_flight"]}
                   for s in self._series if since is None or s["t"] > since]
        doc["series"] = ser
        doc["build_us"] = dict(self.build_us)
        return doc

    def _build(self, live, now_p: float) -> dict:
        lanes = []
        for rec in live:
            try:
                lanes.append(self._lane(rec, now_p))
            except act.READ_ERRORS:
                continue
        lanes.sort(key=lambda x: x["elapsed_ms"] or 0, reverse=True)   # oldest first
        recent = [dict(l, ended_ms=_r((now_p - p) * 1e3, 0)) for p, l, _s in self._done
                  if now_p - p <= DONE_KEEP_S]
        stats = [s for _p, _l, s in self._done]
        last_min = [s for p, _l, s in self._done if now_p - p <= COST_MIN_S]
        backends: dict[str, dict] = {}

        def row(bid, kind):
            b = backends.get(bid)
            if b is None:
                b = backends[bid] = {"id": bid, "kind": kind, "models": [], "in_flight": 0,
                                     "decoding": 0, "waiting": 0, "decode_tps": None,
                                     "decode_unreported": 0, "ttft_p50": None, "ttft_p90": None,
                                     "cache_hit": None, "cache_reported": 0, "finished": 0,
                                     "errors": 0, "cost_per_min": None, "cost_window": None,
                                     "tokens_window": 0, "frames": None}
            return b
        for l in lanes:
            b = row(l["backend"], l["kind"])
            if l["model"] and l["model"] not in b["models"]:
                b["models"].append(l["model"])
            b["in_flight"] += 1
            if l["state"] in ("queued", "waiting", "prefilling"):
                b["waiting"] += 1
            elif l["state"] not in ("done", "replaying"):
                b["decoding"] += 1
                if l["decode_now"] is None:
                    b["decode_unreported"] += 1
                else:
                    b["decode_tps"] = (b["decode_tps"] or 0.0) + l["decode_now"]
        for s in stats:
            row(s["backend"], s["kind"])
        for bid, b in backends.items():
            mine = [s for s in stats if s["backend"] == bid]
            b["finished"] = len(mine)
            b["errors"] = sum(1 for s in mine if s["error"])
            tt = sorted(s["ttft_ms"] for s in mine if s["ttft_ms"] is not None)
            b["ttft_p50"], b["ttft_p90"] = _pct(tt, 0.5), _pct(tt, 0.9)
            rep = [s for s in mine if s["prompt"] and s["cached"] is not None]
            b["cache_reported"] = len(rep)
            if rep:
                b["cache_hit"] = _r(sum(s["cached"] for s in rep) / sum(s["prompt"] for s in rep), 4)
            costs = [s["cost"] for s in mine if s["cost"] is not None]
            b["cost_window"] = sum(costs) if costs else None
            lm = [s["cost"] for s in last_min if s["backend"] == bid and s["cost"] is not None]
            b["cost_per_min"] = sum(lm) if lm else None
            b["tokens_window"] = sum(s["tokens"] or 0 for s in mine)
            if b["decode_tps"] is not None:
                b["decode_tps"] = _r(b["decode_tps"], 1)
            if b["kind"] != "tandemllm" or bid != ENGINE_ID:
                fcs = [fc for (fb, _m), fc in self._frames.items() if fb == bid]
                if fcs:
                    b["frames"] = {"verified": all(fc.verified for fc in fcs),
                                   "tools_verified": all(fc.tools_verified for fc in fcs),
                                   "models": len(fcs),
                                   "checked": sum(len(fc.results) for fc in fcs),
                                   "passed": sum(int(sum(fc.results)) for fc in fcs)}
        dec = [l for l in lanes if l["state"] not in ("queued", "waiting", "prefilling", "done",
                                                      "replaying")]
        tt = sorted(s["ttft_ms"] for s in stats if s["ttft_ms"] is not None)
        rep = [s for s in stats if s["prompt"] and s["cached"] is not None]
        costs = [s["cost"] for s in stats if s["cost"] is not None]
        lm = [s["cost"] for s in last_min if s["cost"] is not None]
        now_rates = [l["decode_now"] for l in dec if l["decode_now"] is not None]
        prefill_rates = [l["prefill"]["tps"] for l in lanes if l["state"] == "prefilling"
                         and l.get("prefill") and l["prefill"].get("tps") is not None]
        partial = [l["cost_micros"] for l in lanes if l["cost_micros"] is not None]
        totals = {
            "in_flight": len(lanes), "decoding": len(dec),
            "waiting": sum(1 for l in lanes if l["state"] in ("queued", "waiting", "prefilling")),
            "decode_tps": _r(sum(now_rates), 1) if now_rates else None,
            "decode_reported": len(now_rates),
            "decode_unreported": len(dec) - len(now_rates),
            "prefill_tps": _r(sum(prefill_rates), 1) if prefill_rates else None,
            "ttft_p50": _pct(tt, 0.5), "ttft_p90": _pct(tt, 0.9), "ttft_n": len(tt),
            "cache_hit": (_r(sum(s["cached"] for s in rep) / sum(s["prompt"] for s in rep), 4)
                          if rep else None),
            "cache_reported": len(rep),
            "prompt_tokens_window": sum(s["prompt"] for s in rep) if rep else None,
            "cached_tokens_window": sum(s["cached"] for s in rep) if rep else None,
            "cost_per_min": sum(lm) if lm else None,
            "cost_window": sum(costs) if costs else None,
            "cost_in_flight": sum(partial) if partial else None,
            "finished": len(stats), "errors": sum(1 for s in stats if s["error"]),
            "refused": sum(1 for s in stats if s["refused"]),
            "tokens_window": sum(s["tokens"] or 0 for s in stats),
            "backends": len(backends),
        }
        order = sorted(backends.values(), key=lambda b: (b["id"] != ENGINE_ID, b["id"]))
        return {"contract_version": CONTRACT, "at": _r(self.clock(), 3),
                "windows": {"now_s": NOW_S, "done_keep_s": DONE_KEEP_S, "stats_s": STATS_S,
                            "cost_min_s": COST_MIN_S},
                "lanes": lanes, "recent": recent, "backends": order, "totals": totals}


def encode(doc: dict) -> bytes:
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode()
