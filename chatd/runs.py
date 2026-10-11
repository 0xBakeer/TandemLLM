"""One run: the model call that answers one user message, on its own thread (SRV-345).

A run belongs to chatd, not to a browser: its thread holds the stream to the engine, so the engine
never sees a client leave when a tab closes. Phase 1 has no tools, so a run is one model call
(the agent loop of SRV-351 makes it 1 to 50). States: queued (waiting for one of the
QSE_CHAT_MAX_RUNS slots, or for the engine), running, completed, cancelled, failed, interrupted.

WHAT STREAMS. `reasoning_content` (or `reasoning`) deltas go to a thinking part; `content` goes to a
text part, and a `<think>...</think>` block inside it (an engine with `--reasoning-format tag`, or
any server that leaves the tags in) goes to a thinking part too. A part opens with `part.added` and
grows with `part.delta`; the deltas are coalesced to one event per part every 50 ms by the
service's ticker (a 10,000-token answer is about 400 events, not 10,000 rows).

STOP, SEND NOW, CANCEL. `stop(mode)` closes the socket to the engine (the engine books the request
`abandoned` and frees its slot within one block); the text so far stays, the message ends
`stopped`. `now` is Send now (the queue drains right after), `cancel` an edit, a regenerate or a
delete taking over (no drain), `shutdown` chatd stopping (the run ends `interrupted`).

RETRY AND CONTINUE. A refused or failed call (connection refused, 429, 5xx, a stream that drops
without a finish) is retried with backoff 1, 2, 4 ... 60 s for QSE_CHAT_RETRY_S (10 minutes), then
the run ends `interrupted`. A retry after text came continues the answer: the partial text goes back
as the last assistant message with `continue_final_message`. A 4xx is not retried: the run fails
with the engine's message.
"""

from __future__ import annotations

import re
import threading
import time

from chatd.engine import EngineError
from chatd.ids import iso, now_ms, ulid

RETRY_WINDOW_S = 600.0
BACKOFF_MAX_S = 60.0
FLUSH_S = 0.05
WARMUP_TOKENS = 50            # decode rates skip the first 50 tokens (the dashboard's NOW rule)
_WAITING = re.compile(r"waiting.*?position\s+(\d+)", re.I)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class Run:
    def __init__(self, svc, conv, *, user_message_id: str, message_id: str, max_steps: int,
                 started_by: str | None, continue_final: bool = False):
        self.svc = svc
        self.conv = conv
        self.id = ulid()
        self.user_message_id = user_message_id
        self.message_id = message_id
        self.started_ms = now_ms()
        self.started_by = started_by
        self.continue_final = continue_final
        self.max_steps = max_steps
        self.state = "queued"
        self.phase = "queued"
        self.now = "Queued"
        self.step = 0
        self.queue_position: int | None = None
        self.prefill: dict | None = None
        self.request_id: str | None = None
        self.stop_mode: str | None = None
        self.call = None
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        # the parts as they stream (the conversation's lock guards these and `pending`)
        self.open_kind: str | None = None
        self.open_index: int | None = None
        self.pending = ""
        self.last_flush = 0.0
        self.think_started: float | None = None
        self.in_tag = False
        self.carry = ""
        self.trim = False             # the newlines right after a tag are the template's, not text
        # counts for the figures when the live row is not there
        self.tokens = 0
        self.thinking_tokens = 0
        self.t_sent: float | None = None
        self.t_first: float | None = None
        self.marks: list[tuple[float, int]] = []
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.metrics: dict = {"decode_tps": None, "tokens_per_round": None, "acceptance": None,
                              "thinking_budget": None}
        self.tps_history: list = []
        self.error: str | None = None

    # ------------------------------------------------------------------ the contract's Run
    def as_dict(self, viewer: str | None = None) -> dict:
        by = self.started_by if self.started_by and self.started_by != viewer else None
        return {
            "id": self.id, "conversation_id": self.conv.id, "message_id": self.message_id,
            "started_at": iso(self.started_ms), "started_by": by,
            "phase": self.phase, "now": self.now, "step": self.step, "max_steps": self.max_steps,
            "queue_position": self.queue_position,
            "tool": None, "prefill": self.prefill,
            "decode_tps": self.metrics["decode_tps"],
            "tokens_per_round": self.metrics["tokens_per_round"],
            "acceptance": self.metrics["acceptance"],
            "thinking": {"used": self.thinking_tokens, "budget": self.metrics["thinking_budget"]},
            "tokens": self.tokens,
            "tps_history": list(self.tps_history[-120:]),
        }

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.thread = threading.Thread(target=self._main, name=f"chatd-run-{self.id[-6:]}",
                                       daemon=True)
        self.thread.start()

    def stop(self, mode: str = "stop") -> None:
        if self.stop_mode is None:
            self.stop_mode = mode
        self.wake.set()
        call = self.call
        if call is not None:
            call.close()

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    # ------------------------------------------------------------------ phases
    def set_phase(self, phase: str, now: str, **extra) -> None:
        with self.conv.lock:
            if phase == self.phase and now == self.now and not extra:
                return
            self.phase, self.now = phase, now
            ev = {"type": "run.phase", "run_id": self.id, "phase": phase, "now": now,
                  "step": self.step}
            if "queue_position" in extra:
                self.queue_position = extra["queue_position"]
                ev["queue_position"] = self.queue_position
            elif phase != "queued":
                self.queue_position = None
            if phase != "prefill":
                self.prefill = None
            self.svc.emit(self.conv, ev, run_id=self.id)
            self.svc.summary_changed(self.conv, now=now)

    # ------------------------------------------------------------------ the thread
    def _main(self) -> None:
        outcome = "failed"
        try:
            outcome = self._run()
        except Exception as exc:                                    # noqa: BLE001
            self.error = f"chatd: {type(exc).__name__}"
            self.svc.log(f"[run] {self.id} crashed: {type(exc).__name__}")
            outcome = "failed"
        finally:
            self.svc.slots.release_if(self)
            self.svc.finish_run(self, outcome)

    def _run(self) -> str:
        # one of the QSE_CHAT_MAX_RUNS slots (the engine serves at most 4; opencode keeps one)
        while not self.svc.slots.acquire(self, timeout=1.0):
            if self.stop_mode:
                return self._stopped()
            self.set_phase("queued", "Waiting for a free run slot")
        self.state = "running"
        deadline = time.monotonic() + self.svc.retry_window_s
        backoff = 1.0
        while True:
            if self.stop_mode:
                return self._stopped()
            body, headers = self.svc.request_for(self)
            self.set_phase("queued", "Sending to the engine")
            try:
                self.t_sent = time.monotonic()
                self.call = self.svc.engine.stream(body, headers)
            except EngineError as exc:
                self.call = None
                if not exc.retryable:
                    self.error = exc.args[0]
                    return "failed"
                if time.monotonic() + backoff > deadline:
                    self.error = exc.args[0]
                    return "interrupted"
                self.set_phase("queued", f"Waiting for the engine ({exc.args[0]}), retry in "
                                         f"{int(backoff)} s")
                self.wake.wait(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
                continue
            if self.stop_mode:                       # Stop came while the call opened
                self.call.close()
                return self._stopped()
            self.step = max(1, self.step)
            self.set_phase("prefill", "Reading the prompt")
            outcome = self._consume()
            self.call = None
            if outcome in ("completed", "failed"):
                return outcome
            if self.stop_mode:
                return self._stopped()
            # the stream broke with no finish: the engine went away. Retry, continuing the text.
            if time.monotonic() + backoff > deadline:
                self.error = "the engine stream ended early"
                return "interrupted"
            self.svc.flush(self)
            self.continue_final = True
            self.set_phase("queued", f"The engine stream ended early, retry in {int(backoff)} s")
            self.wake.wait(backoff)
            backoff = min(BACKOFF_MAX_S, backoff * 2)

    def _stopped(self) -> str:
        return "interrupted" if self.stop_mode == "shutdown" else "cancelled"

    def _consume(self) -> str:
        finished = False
        for kind, obj in self.call.events():
            if kind == "comment":
                m = _WAITING.search(obj or "")
                if m:
                    pos = int(m.group(1))
                    self.set_phase("queued", f"In the engine's line, {pos} ahead",
                                   queue_position=pos)
                continue
            if kind in ("done", "end"):
                break
            if kind == "broken":
                return "broken"
            if kind != "data" or not isinstance(obj, dict):
                continue
            if obj.get("error") and not obj.get("choices"):
                err = obj["error"]
                self.error = str(err.get("message") if isinstance(err, dict) else err)[:300]
                return "failed"
            if self.request_id is None and obj.get("id"):
                self.request_id = str(obj["id"])
            if isinstance(obj.get("usage"), dict):
                self.usage = {"usage": obj["usage"], "timings": obj.get("timings") or {},
                              "metrics": obj.get("metrics") or {},
                              "tandem": (obj["usage"].get("tandem") or obj.get("tandem"))}
            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                d = ch.get("delta") or {}
                r = d.get("reasoning_content") or d.get("reasoning")
                if isinstance(r, str) and r:
                    self._add("thinking", r)
                c = d.get("content")
                if isinstance(c, str) and c:
                    self._content(c)
                if ch.get("finish_reason"):
                    self.finish_reason = str(ch["finish_reason"])
                    finished = True
                    if self.phase != "finishing":
                        self.svc.flush(self)
                        self.set_phase("finishing", "Finishing")
            if self.stop_mode:
                return "broken"
        if self.stop_mode:
            return "broken"
        if self.carry:
            self._add("thinking" if self.in_tag else "text", self.carry)
            self.carry = ""
        return "completed" if finished or self.usage is not None else "broken"

    # ------------------------------------------------------------------ the parts
    def _content(self, text: str) -> None:
        """Content with `<think>` tags split into thinking and text; a tag cut across two chunks
        waits in `carry` until it is whole."""
        s = self.carry + text
        self.carry = ""
        while s:
            tag = THINK_CLOSE if self.in_tag else THINK_OPEN
            i = s.find(tag)
            if i >= 0:
                head = s[:i].lstrip("\n") if self.trim else s[:i]
                if head:
                    self.trim = False
                    self._add("thinking" if self.in_tag else "text", head)
                self.in_tag = not self.in_tag
                if not self.in_tag:
                    self.svc.close_thinking(self)
                self.trim = True
                s = s[i + len(tag):]
                continue
            keep = 0
            for k in range(min(len(tag) - 1, len(s)), 0, -1):
                if tag.startswith(s[-k:]):
                    keep = k
                    break
            body = s[:len(s) - keep] if keep else s
            if self.trim and body:
                body = body.lstrip("\n")
                self.trim = not body
            if body:
                self._add("thinking" if self.in_tag else "text", body)
            self.carry = s[len(s) - keep:] if keep else ""
            break

    def _add(self, kind: str, text: str) -> None:
        now = time.monotonic()
        if self.t_first is None:
            self.t_first = now
        self.tokens += 1
        if kind == "thinking":
            self.thinking_tokens += 1
        self.marks.append((now, self.tokens))
        if len(self.marks) > 400:
            del self.marks[:200]
        if kind == "thinking" and self.phase != "thinking":
            self.set_phase("thinking", "Thinking")
        elif kind == "text" and self.phase not in ("writing", "finishing"):
            if text.strip() or self.phase == "thinking":
                self.set_phase("writing", "Writing")
        self.svc.append(self, kind, text)

    # ------------------------------------------------------------------ figures
    def local_decode_tps(self) -> float | None:
        """Tokens over the last 2 s, after the first WARMUP_TOKENS; None when idle (the NOW rule).
        A token here is a delta chunk: an estimate, used only when the live row is missing."""
        if self.tokens <= WARMUP_TOKENS or len(self.marks) < 2:
            return None
        now = time.monotonic()
        t1, n1 = self.marks[-1]
        if now - t1 > 2.0:
            return None
        older = [(t, n) for t, n in self.marks if now - t <= 2.0 and n > WARMUP_TOKENS]
        if len(older) < 2:
            return None
        (t0, n0) = older[0]
        return round((n1 - n0) / (t1 - t0), 1) if t1 > t0 else None

    def stats(self, finish: str) -> dict:
        """The message's TurnStats: the usage frame's figures when it came, else chatd's own."""
        u = (self.usage or {}).get("usage") or {}
        t = (self.usage or {}).get("timings") or {}
        mt = (self.usage or {}).get("metrics") or {}
        spec = mt.get("speculative_decoding") or {}
        ptd = u.get("prompt_tokens_details") or {}
        ctd = u.get("completion_tokens_details") or {}
        ttft = t.get("ttft_ms") if t.get("ttft_ms") is not None else mt.get(
            "time_to_first_token_ms")
        if ttft is None and self.t_first is not None and self.t_sent is not None:
            ttft = round((self.t_first - self.t_sent) * 1000, 1)
        total = t.get("total_ms")
        if total is None and self.t_sent is not None:
            total = round((time.monotonic() - self.t_sent) * 1000, 1)
        out = {
            "finish": finish,
            "ttft_ms": ttft,
            "decode_tps": t.get("predicted_per_second") or None,
            "total_ms": total,
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
            "completion_tokens": int(u.get("completion_tokens") or self.tokens),
            "reasoning_tokens": int(ctd.get("reasoning_tokens") or self.thinking_tokens),
            "cached_tokens": int(ptd.get("cached_tokens") or 0),
            "tokens_per_round": t.get("tokens_per_block") or spec.get("mean_acceptance_length"),
            "acceptance": spec.get("draft_acceptance_rate"),
            "cache_source": t.get("cache_source"),
            "prefill_tps": t.get("prompt_per_second") or None,
            "cost_usd": None,
        }
        if ptd.get("hidden_tokens"):
            out["hidden_tokens"] = int(ptd["hidden_tokens"])
        return out
