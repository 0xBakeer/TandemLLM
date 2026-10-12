"""One run: the agent loop that answers one user message, on its own thread (SRV-345, SRV-351).

A run belongs to chatd, not to a browser: its thread holds the stream to the engine, so the engine
never sees a client leave when a tab closes. States: queued (waiting for one of the
QSE_CHAT_MAX_RUNS slots, or for the engine), running, completed, cancelled, failed, interrupted.

THE LOOP (Claude Code's; plan note Chat 7). One step is one model call:
1. the request: the system block, the tools in fixed order, the branch, the run's earlier steps
   with their tool results, the tail. Before it the context guard (chatd/compaction.py) may
   compact the conversation, and caps `max_tokens` to what the window has left;
2. the stream: thinking and text into parts, tool calls into `tool_call` parts as they are
   written; every part carries its `step`, which is how the next request is rebuilt;
3. a step that ends with tool calls: each is checked against the permission rules (a prompt
   waits as long as it must), read-only calls run in parallel (at most 4), writes and commands in
   order, and the results go back to the model in the next step. A message queued meanwhile (a
   steer) ends the run here, and the next run carries it: the model reads it right after the
   results;
4. a step without calls ends the run: `stop` completes it, `length` gets one continuation, an
   empty turn one retry with a hidden nudge (a second empty turn is an error the user sees).
Limits: `max_steps` model calls (50; the run ends `interrupted` with a notice and Continue goes
on), the same call with the same arguments three times in a row ends the run with a notice.

WHAT STREAMS. `reasoning_content` (or `reasoning`) deltas go to a thinking part; `content` goes to a
text part, and a `<think>...</think>` block inside it goes to a thinking part too. Parts are
coalesced to one event per part every 50 ms by the service's ticker.

STOP, SEND NOW, CANCEL. `stop(mode)` closes the socket to the engine (the engine books the request
`abandoned` and frees its slot within one block) and signals a running tool to end; the text so far
stays, the message ends `stopped`, a tool that was running ends `cancelled`.

RETRY AND CONTINUE. A refused or failed call (connection refused, 429, 5xx, a stream that drops
without a finish) is retried with backoff 1, 2, 4 ... 60 s for QSE_CHAT_RETRY_S (10 minutes), then
the run ends `interrupted`. A retry after text came continues the answer with
`continue_final_message`. A 4xx is not retried: the run fails with the engine's message.
"""

from __future__ import annotations

import collections
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
NUDGE = "(Your last turn was empty. Answer the user now, or call a tool.)"


class GuardError(Exception):
    """The request cannot go: it does not fit the window even after a compaction, or its user
    message is empty."""


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
        self.tool: dict | None = None
        self.request_id: str | None = None
        self.stop_mode: str | None = None
        self.call = None
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        # the parts as they stream (the conversation's lock guards these and `pending`)
        self.open_kind: str | None = None
        self.open_index: int | None = None
        self.pend: dict = {}                   # (part index or call id, field) -> text to flush
        self.out_sent: dict[str, int] = {}     # call id -> output characters sent in events
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
        self.notice: str | None = None         # why the run ended early (step cap, a loop)
        self.remark: str | None = None         # a note for the end of the answer (a gone pin)
        # the loop
        self.cancel = threading.Event()        # a running tool ends on it
        self.step_parts: list[int] = []        # this step's tool_call part indexes, in order
        self.tc_parts: dict[int, int] = {}     # this step's tool_calls[].index -> part index
        self.length_continued = False
        self.empty_retries = 0
        self.nudge = False
        self.recent_calls: collections.deque = collections.deque(maxlen=3)
        self.request_chars = 0
        self.request_est = 0
        self.compacted = False
        self.resume_same_step = False          # Continue of a cut text goes on in its own step
        self.acc = {"completion": 0, "reasoning": 0}
        self.step_usage: list[dict] = []       # per model call: prompt, cached, cache source
        self.snapshotted = False
        self.waiting: dict[str, dict] = {}     # call id -> {"event", "answer"} while a prompt waits

    # ------------------------------------------------------------------ the contract's Run
    def as_dict(self, viewer: str | None = None) -> dict:
        by = self.started_by if self.started_by and self.started_by != viewer else None
        return {
            "id": self.id, "conversation_id": self.conv.id, "message_id": self.message_id,
            "started_at": iso(self.started_ms), "started_by": by,
            "phase": self.phase, "now": self.now, "step": self.step, "max_steps": self.max_steps,
            "queue_position": self.queue_position,
            "tool": self.tool, "prefill": self.prefill,
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
        self.cancel.set()
        for w in list(self.waiting.values()):
            w["event"].set()
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
            if "tool" in extra:
                self.tool = extra["tool"]
                ev["tool"] = self.tool
            elif phase not in ("tool", "permission"):
                self.tool = None
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
        while True:
            if self.stop_mode:
                return self._stopped()
            if self.step >= self.max_steps:
                self.notice = (f"The run reached its limit of {self.max_steps} model calls. "
                               f"Continue goes on from here.")
                return "interrupted"
            outcome = self._call()
            if outcome != "completed":
                return outcome
            if self.step_parts:
                if self.svc.run_tools(self, list(self.step_parts)) == "cancelled":
                    return self._stopped()
                if self.notice:                     # the same call three times in a row
                    return "completed"
                if self.svc.queued(self.conv):      # a steer: the next run carries it
                    return "completed"
                continue
            if self.finish_reason == "length" and not self.length_continued:
                self.length_continued, self.continue_final = True, True
                continue
            if self.svc.step_empty(self):
                if self.empty_retries == 0:
                    self.empty_retries, self.nudge = 1, True
                    continue
                self.error = "The model answered with nothing, twice."
                return "failed"
            return "completed"

    def _call(self) -> str:
        """One model call with its retries: completed, failed, interrupted or cancelled."""
        deadline = time.monotonic() + self.svc.retry_window_s
        backoff = 1.0
        if self.resume_same_step:
            self.resume_same_step = False
        else:
            self.step += 1
        self.step_parts, self.tc_parts = [], {}
        self.finish_reason, self.usage = None, None
        while True:
            if self.stop_mode:
                return self._stopped()
            try:
                body, headers = self.svc.request_for(self)
            except GuardError as exc:
                self.error = str(exc)
                return "failed"
            self.nudge = False
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
            self.set_phase("prefill", "Reading the prompt")
            outcome = self._consume()
            self.call = None
            if outcome in ("completed", "failed"):
                self.continue_final = False
                return outcome
            if self.stop_mode:
                return self._stopped()
            # the stream broke with no finish: the engine went away. Retry, continuing the text.
            if time.monotonic() + backoff > deadline:
                self.error = "the engine stream ended early"
                return "interrupted"
            self.svc.flush(self)
            if self.step_parts:                      # a half-written call is not continued
                self.svc.drop_step(self)
            else:
                self.continue_final = True
            self.set_phase("queued", f"The engine stream ended early, retry in {int(backoff)} s")
            self.wake.wait(backoff)
            backoff = min(BACKOFF_MAX_S, backoff * 2)


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
                u = obj["usage"]
                self.usage = {"usage": u, "timings": obj.get("timings") or {},
                              "metrics": obj.get("metrics") or {},
                              "tandem": (u.get("tandem") or obj.get("tandem"))}
                t = obj.get("timings") or {}
                self.step_usage.append({"step": self.step,
                                        "prompt_tokens": int(u.get("prompt_tokens") or 0),
                                        "cached_tokens": int((u.get("prompt_tokens_details") or {})
                                                             .get("cached_tokens") or 0),
                                        "cache_source": t.get("cache_source"),
                                        "ttft_ms": t.get("ttft_ms")})
                self.acc["completion"] += int(u.get("completion_tokens") or 0)
                self.acc["reasoning"] += int((u.get("completion_tokens_details") or {})
                                             .get("reasoning_tokens") or 0)
                self.svc.learn_ratio(self.conv, int(u.get("prompt_tokens") or 0),
                                     self.request_chars)
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
                for tc in d.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        self._tool_delta(tc)
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

    def _stopped(self) -> str:
        return "interrupted" if self.stop_mode == "shutdown" else "cancelled"


    # ------------------------------------------------------------------ the parts
    def _tool_delta(self, tc: dict) -> None:
        """A tool call as the model writes it: the part opens with its name, then its
        arguments grow (`part.delta` field `input`)."""
        idx = int(tc.get("index") or 0)
        fn = tc.get("function") or {}
        if idx not in self.tc_parts:
            if self.carry:
                self._add("thinking" if self.in_tag else "text", self.carry)
                self.carry = ""
            if self.phase not in ("writing", "finishing"):
                self.set_phase("writing", f"Calling {fn.get('name') or 'a tool'}")
            k = self.svc.add_call(self, str(tc.get("id") or f"call_{ulid()[-12:]}"),
                                  str(fn.get("name") or ""))
            self.tc_parts[idx] = k
            self.step_parts.append(k)
        k = self.tc_parts[idx]
        if fn.get("name") and not self.svc.call_part(self, k).get("name"):
            self.svc.call_part(self, k)["name"] = str(fn["name"])
        args = fn.get("arguments")
        if isinstance(args, str) and args:
            if self.t_first is None:
                self.t_first = time.monotonic()
            self.tokens += 1
            self.marks.append((time.monotonic(), self.tokens))
            self.svc.append_input(self, k, args)

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
            "completion_tokens": int(self.acc["completion"] or u.get("completion_tokens") or
                                     self.tokens),
            "reasoning_tokens": int(self.acc["reasoning"] or ctd.get("reasoning_tokens") or
                                    self.thinking_tokens),
            "steps": self.step,
            "step_usage": self.step_usage[-50:],
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
