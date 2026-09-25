"""One record per request: the single source of its usage, its timings and its ledger row.

Open WebUI shows nothing under an answer unless the stream carries `usage`, and it shows speed only
when it carries llama.cpp's `timings` (SRV-27; the evidence, with Open WebUI 0.11.3 file:line, is
in the Memo note "Usage & speed metrics — design (2026-09-24)" section 1). The numbers were all in
the server already -- the `[req]` line printed most of them -- but in four places and computed
four ways. `RequestRecord` is filled as the request runs and everything that reports on a request
reads it: the usage JSON here, the Prometheus observations (SRV-9) and the ledger row (SRV-28).

THE TIME POINTS, all `time.perf_counter()`:

    t_arrival   the handler entered `_complete`
    t_lock      the engine lock was acquired
    t_first     the first token reached the handler (the prefill ran inside that `next()`)
    t_last      the last token reached the handler
    t_end       the request was logged, just before its finish chunk is written

so `queue_ms = t_lock - t_arrival`, `prompt_ms = t_first - t_lock` and `ttft_ms` is exactly their
sum. `predicted_per_second` is `(completion_tokens - 1) / predicted_ms`: the atlas row's divisor
and the `[req]` line's `committed/decode_ms`. llama.cpp divides by `predicted_n` instead; the
first token is the prefill's, not decode's, so this engine does not.

EXACTNESS. `completion_tokens` counts the ids the engine committed -- the generator yields
committed tokens only, so a drafted-and-rejected token can never be counted -- with the EOS
included, as the non-streamed body always counted it. Reasoning tokens are the committed ids up to
and including the one that closes the block (the special id or the literal text, as ThinkBudget
watches both), all of them if it never closed; the tokens a forced close writes are reasoning.

Nothing here imports torch.
"""

from __future__ import annotations

import time

# The timings keys, in order: llama.cpp's names and meanings first (tools/server/server-common.cpp
# in llama.cpp), then this engine's own. Open WebUI prints every one of them in its tooltip.
LLAMA_KEYS = ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
              "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
              "draft_n", "draft_n_accepted")
ENGINE_KEYS = ("ttft_ms", "queue_ms", "total_ms", "blocks", "tokens_per_block", "reasoning_n",
               "cache_source")


def _ms(a: float | None, b: float | None) -> float | None:
    return (b - a) * 1e3 if a is not None and b is not None else None


def _r(x: float | None, nd: int = 2) -> float:
    return round(float(x), nd) if x is not None else 0.0


def special_id(tokenizer, text: str) -> int | None:
    """The id of `text` when the tokenizer holds it as ONE token, else None.

    engine/spec.py's `_special_id` falls back to the last id of the text's own tokens, which is
    right for a budget that only needs something to watch and wrong for a count: on a tokenizer
    without the special token it would end the reasoning block at the first `>`.
    """
    i = tokenizer.convert_tokens_to_ids(text)
    if isinstance(i, int) and i >= 0 and i != getattr(tokenizer, "unk_token_id", None):
        return i
    return None


def reasoning_count(ids: list[int], end_id: int | None, end_text: list[int] | None) -> int:
    """Committed ids through the one that closes the reasoning block; all of them if none does.

    The model closes the block with the special `</think>` id or, on prompts that look like raw
    text, with the literal characters (engine/spec.py's ThinkBudget watches both, for the same
    reason), so both end it here.
    """
    seq = [int(t) for t in (end_text or [])]
    k = len(seq)
    for i, t in enumerate(ids):
        if end_id is not None and t == end_id:
            return i + 1
        if k and i + 1 >= k and [int(x) for x in ids[i + 1 - k:i + 1]] == seq:
            return i + 1
    return len(ids)


class RequestRecord:
    """Everything one request did, filled in as it happens.

    The handler sets the fields it knows; `track()` stamps the token times; `absorb_*` read the
    loop's own counters. The derived values are properties, so a record that was refused before
    the lock -- no prompt, no tokens -- still renders, with zeros where nothing happened.
    """

    def __init__(self, request_id: str, endpoint: str, stream: bool,
                 t_arrival: float | None = None, ts: float | None = None):
        self.request_id = request_id
        self.endpoint = endpoint                  # "chat" | "completions"
        self.stream = bool(stream)
        self.ts = time.time() if ts is None else ts
        self.t_arrival = time.perf_counter() if t_arrival is None else t_arrival
        self.t_lock = self.t_first = self.t_last = self.t_end = None
        self.model = ""
        self.status = 200
        self.finish_reason: str | None = None
        self.error_type: str | None = None
        self.prompt_tokens: int | None = None
        self.cached_tokens = 0
        self.prompt_n: int | None = None          # forwarded; None = the prompt's whole length
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.blocks = 0
        self.draft_n = 0
        self.draft_accepted = 0
        self.cache_source = "none"
        self.tool_calls = 0
        self.thinking = False
        self.max_tokens: int | None = None
        self.client_id = "anon"
        self.client_kind = "other"
        self.temperature: float | None = None
        # SRV-34: tokens handed to the handler so far, read once a second by server/live.py.
        # Equals `completion_tokens` at the end; the one per-token cost of the live view.
        self.n_live = 0

    # ----------------------------------------------------------------- filling it in
    def track(self, source):
        """Pass the token source through, stamping the first and the last token as they arrive.

        One integer add per token beside the two stamps (SRV-34): no lock, no allocation."""
        for t in source:
            now = time.perf_counter()
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            self.n_live += 1
            yield t

    def lock_acquired(self) -> None:
        self.t_lock = time.perf_counter()

    def absorb_prefill(self, info: dict | None) -> None:
        """`STATE["last_prefill"]` of this request's own prefill: tokens restored and forwarded."""
        if not info:
            return
        self.cached_tokens = int(info.get("reused") or 0)
        self.prompt_n = int(info.get("forwarded") or 0)
        if self.cached_tokens:
            self.cache_source = str(info.get("kind") or "prefix")

    def absorb_response_cache(self) -> None:
        """A replay: every prompt token came out of a cache and none was forwarded."""
        self.cache_source = "response"
        self.cached_tokens = int(self.prompt_tokens or 0)
        self.prompt_n = 0

    def absorb_blocks(self, bs) -> None:
        """The decode loop's `BlockStats`: forwards paid, and the first-miss histogram.

        `accept` maps a block's draft depth to {accepted: count}, so the tokens proposed and kept
        are exact sums over it. A step the drafter declined is a block with no draft.
        """
        if bs is None:
            return
        self.blocks = int(bs.blocks)
        self.draft_n = sum(d * n for d, h in bs.accept.items() for n in h.values())
        self.draft_accepted = sum(a * n for h in bs.accept.values() for a, n in h.items())

    def end(self) -> None:
        if self.t_end is None:
            self.t_end = time.perf_counter()

    # ----------------------------------------------------------------- derived
    @property
    def queue_ms(self) -> float | None:
        return _ms(self.t_arrival, self.t_lock)

    @property
    def prompt_ms(self) -> float | None:
        return _ms(self.t_lock, self.t_first)

    @property
    def ttft_ms(self) -> float | None:
        q, p = self.queue_ms, self.prompt_ms
        return q + p if q is not None and p is not None else None

    @property
    def predicted_ms(self) -> float | None:
        return _ms(self.t_first, self.t_last)

    @property
    def total_ms(self) -> float | None:
        return _ms(self.t_arrival, self.t_end)

    @property
    def forwarded(self) -> int:
        if self.prompt_n is not None:
            return self.prompt_n
        return int(self.prompt_tokens or 0) - self.cached_tokens

    @property
    def committed(self) -> int:
        """Decode's tokens: the first one is the prefill's."""
        return max(0, self.completion_tokens - 1)

    @property
    def decode_tps(self) -> float | None:
        ms = self.predicted_ms
        return self.committed / (ms / 1e3) if self.committed and ms else None

    @property
    def prefill_tps(self) -> float | None:
        ms = self.prompt_ms
        return self.forwarded / (ms / 1e3) if self.forwarded and ms else None

    @property
    def tokens_per_block(self) -> float | None:
        return self.committed / self.blocks if self.blocks else None

    @property
    def draft_acceptance(self) -> float | None:
        return self.draft_accepted / self.draft_n if self.draft_n else None

    # ----------------------------------------------------------------- the three objects
    def usage(self) -> dict:
        """vLLM's usage, the details included."""
        p, c = int(self.prompt_tokens or 0), int(self.completion_tokens)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                "prompt_tokens_details": {"cached_tokens": int(self.cached_tokens)},
                "completion_tokens_details": {"reasoning_tokens": int(self.reasoning_tokens)}}

    def timings(self) -> dict:
        """llama.cpp's `timings`, the one speed shape Open WebUI displays, plus the engine's keys."""
        n = self.forwarded
        pms = self.prompt_ms or 0.0
        dms = self.predicted_ms or 0.0
        return {
            "cache_n": int(self.cached_tokens),
            "prompt_n": int(n),
            "prompt_ms": _r(pms),
            "prompt_per_token_ms": _r(pms / n if n else 0.0),
            "prompt_per_second": _r(self.prefill_tps),
            "predicted_n": int(self.completion_tokens),
            "predicted_ms": _r(dms),
            "predicted_per_token_ms": _r(dms / self.committed if self.committed else 0.0),
            "predicted_per_second": _r(self.decode_tps),
            "draft_n": int(self.draft_n),
            "draft_n_accepted": int(self.draft_accepted),
            "ttft_ms": _r(self.ttft_ms),
            "queue_ms": _r(self.queue_ms),
            "total_ms": _r(self.total_ms),
            "blocks": int(self.blocks),
            "tokens_per_block": _r(self.tokens_per_block),
            "reasoning_n": int(self.reasoning_tokens),
            "cache_source": self.cache_source,
        }

    def metrics(self) -> dict:
        """vLLM 0.29's per-response `metrics` object. Open WebUI does not read it; vLLM-aware
        clients do."""
        total = self.total_ms or 0.0
        return {
            "time_to_first_token_ms": _r(self.ttft_ms),
            "generation_time_ms": _r(self.predicted_ms),
            "queue_time_ms": _r(self.queue_ms),
            "mean_itl_ms": _r((self.predicted_ms or 0.0) / self.committed
                              if self.committed else 0.0),
            "tokens_per_second": _r(self.completion_tokens / (total / 1e3) if total else 0.0),
            "speculative_decoding": {
                "mean_acceptance_length": _r(self.tokens_per_block),
                "draft_acceptance_rate": _r(self.draft_acceptance, 4),
            },
        }

    def fields(self) -> dict:
        """`usage`, `timings` and `metrics` together, for the one chunk or body that carries them."""
        return {"usage": self.usage(), "timings": self.timings(), "metrics": self.metrics()}

    def row(self, engine_version: str = "", code_sha: str = "") -> dict:
        """The ledger row (SRV-28): counts and times, nothing a person wrote or read.

        A request that never reached the engine -- refused at the queue, rejected with a 400 --
        has null token and speed fields rather than zeros, so a sum over the ledger is honest and
        a percentile over it does not see a refusal as an infinitely slow request.
        """
        ran = self.prompt_tokens is not None
        decoded = ran and self.t_first is not None

        def r2(x):
            return round(x, 2) if x is not None else None

        return {
            "ts_ms": int(self.ts * 1000), "request_id": self.request_id, "model": self.model,
            "client_id": self.client_id, "client_kind": self.client_kind,
            "endpoint": self.endpoint, "stream": int(self.stream), "status": int(self.status),
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens if ran else None,
            "cached_tokens": self.cached_tokens if ran else None,
            "completion_tokens": self.completion_tokens if ran else None,
            "reasoning_tokens": self.reasoning_tokens if ran else None,
            "queue_ms": r2(self.queue_ms), "prompt_ms": r2(self.prompt_ms),
            "ttft_ms": r2(self.ttft_ms), "decode_ms": r2(self.predicted_ms),
            "total_ms": r2(self.total_ms),
            "decode_tps": r2(self.decode_tps) if decoded else None,
            "prefill_tps": r2(self.prefill_tps) if decoded else None,
            "blocks": self.blocks if ran else None,
            "draft_tokens": self.draft_n if ran else None,
            "draft_accepted": self.draft_accepted if ran else None,
            "tool_calls": int(self.tool_calls), "thinking": int(self.thinking),
            "cache_source": self.cache_source if ran else None,
            "max_tokens": self.max_tokens, "error_type": self.error_type,
            "engine_version": engine_version, "code_sha": code_sha,
        }


def placement(body: dict, stream: bool, default_on: bool) -> str:
    """Where a response's usage goes: `body`, `separate`, `finish` or `none`. Exactly one place.

    Open WebUI's merge ADDS the token counts of every chunk that carries usage (utils/response.py
    `merge_usage`), so a stream that carried it twice would show doubled counts. It asks for
    `include_usage` only when a model's "Usage" capability is ticked, which it is not on the base
    models -- hence the default: with no `stream_options` the finish chunk carries it (a chunk
    with a non-empty `choices`, so a client that indexes `choices[0]` is safe), with
    `include_usage: true` the separate `choices: []` chunk does, as vLLM sends it, and an explicit
    `include_usage: false` gets nothing.
    """
    if not stream:
        return "body"
    opts = body.get("stream_options")
    asked = opts.get("include_usage") if isinstance(opts, dict) else None
    if asked is not None:
        return "separate" if asked else "none"
    return "finish" if default_on else "none"
