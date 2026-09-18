"""An OpenAI-compatible server over this engine, standard library only.

The engine holds one sequence: one recurrent state per linear-attention layer, one KV buffer, one
drafter cache. So the server is honest about that -- requests are serialised behind a lock, and
`/v1/models` and the usage counts are the only places it pretends to be a fleet. Concurrency
numbers from this server are queueing numbers, and the docs say so.

What it exists for: the bench harness this engine is measured against speaks
`POST /v1/chat/completions` with `stream: true` and `stream_options: {"include_usage": true}`,
reads time-to-first-token from the first content delta and inter-token latency from the gaps
between deltas, and takes the token counts from the usage frame. Any of those missing turns a
measured row into an estimated one, so all three are produced exactly.

Streaming emits one chunk per token rather than per buffer flush, because the gaps between chunks
are the measurement.

    python server/app.py --port 8000 --drafter mtp --depth 3
    QWEN38_NVFP4=~/nvfp4/mlp-clip.safetensors python server/app.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.spec import Relax, ThinkBudget  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.penalty import PatternStop, PenaltySpec, PenaltyState  # noqa: E402
from engine.sample import Sampler  # noqa: E402
from server.stream import Detokenizer, Reasoning, opens_think, split_full  # noqa: E402
from server import metrics  # noqa: E402

STATE: dict = {}
LOCK = threading.Lock()

# What the client is told when the repetition guard ends a stream (SRV-11). The finish reason is
# "stop" because that is the whole OpenAI vocabulary; this marker is the engine's own voice and
# says plainly that the cut was the engine's, so an incomplete answer is never presented as a
# finished one. Token accounting and the caches never see it.
GUARD_MARKER = "\n\n[engine: repetition guard stopped the output here]"


def _guard_headers(pstop) -> tuple:
    if pstop is not None and pstop.hit:
        return (("X-Engine-Stop", f"pattern-stop({pstop.label})"),)
    return ()

# The engine holds ONE sequence, so requests are serialised behind `LOCK` and everything else here
# is about being honest to a caller who arrives while it is held. An unbounded wait is not honesty:
# a client that queued behind four long generations gets an answer some minutes after it stopped
# caring, having held a socket the whole time. So the depth is bounded and the wait has a limit,
# and both refusals say `Retry-After`.
QUEUE = threading.Lock()
INFLIGHT = {"waiting": 0, "running": 0, "served": 0, "refused": 0, "errors": 0,
            "timeouts": 0, "abandoned": 0}


class Deadline:
    """A wall-clock cap on one generation, checked between blocks.

    Between blocks and not inside them: a block is one verify and is not interruptible, so the
    granularity of this is about 130 ms, which is the granularity the engine has. `hit` is what the
    stream reports as `finish_reason`, because a generation that ran out of time and one that ran
    out of tokens are different things to the caller and both used to be "length".
    """

    __slots__ = ("t_end", "hit")

    def __init__(self, seconds: float):
        self.t_end = time.monotonic() + float(seconds) if seconds else 0.0
        self.hit = False

    def expired(self) -> bool:
        if self.t_end and time.monotonic() > self.t_end:
            self.hit = True
        return self.hit


# ------------------------------------------------------------------ generation
def generate_stream(prompt: torch.Tensor, max_new: int, eos: set[int], think=None,
                    conv_id: str | None = None, deadline: "Deadline | None" = None,
                    pen: "PenaltyState | None" = None, pstop: "PatternStop | None" = None,
                    sampler: "Sampler | None" = None):
    """Yield token ids as they are decided, speculation included.

    Same loop as `engine.spec.generate_spec`, rewritten as a generator so a token reaches the
    socket at the moment it is accepted rather than at the end of the block. A verified block
    produces several tokens at once and they go out together; that is what the engine does, and
    smoothing it would make the inter-token latency a fiction.

    The prefill is `engine.cache.prefill`, which may resume from a state the store already holds.
    Nothing downstream of it knows or cares: it restores the same bytes a forward would have
    written and returns the same logits.
    """
    eng = STATE["engine"]
    drafter = STATE["drafter"]
    k = STATE["k"]
    ctx = prompt.tolist()
    if pen is not None:
        pen.seed(ctx)
    # The list the engine's own positions index into, published for `_remember`. It is the SAME
    # object, appended to as tokens are committed, so `ctx[:eng.kv.length]` is by construction the
    # prefix the engine really forwarded -- including when a caller abandons this generator half
    # way through a verified block, where `pos` has already advanced past what has been appended
    # and the slice comes out short. Reconstructing it from the tokens the caller collected would
    # be an argument about that invariant instead of a use of it.
    STATE["last_ctx"] = ctx
    if think is not None:
        think.start(ctx)
    with torch.no_grad():
        if drafter is not None:
            # A drafter that keeps per-request policy state clears it in `reset()`, so this is the
            # last moment the PREVIOUS request's choices can be read. Printed here rather than at
            # the end of the generation because the stream returns from four places inside its
            # loop and none of them is an exit worth wrapping for a log line.
            if STATE.get("verbose") and hasattr(drafter, "report"):
                print(f"[drafter] {drafter.report()}", flush=True)
            drafter.reset()
            if hasattr(drafter, "prime"):
                drafter.prime(ctx)
        t_pre = time.perf_counter()
        logits, reused, forwarded = cache.prefill(
            eng, drafter, ctx, prompt.device, store=STATE.get("state_store"),
            chunk=STATE.get("prefix_chunk", 0), conv_id=conv_id,
            checkpoint=bool(STATE.get("prefix_cache")))
        STATE["last_prefill"] = {"reused": reused, "forwarded": forwarded,
                                 "ms": (time.perf_counter() - t_pre) * 1e3}
        pos = prompt.numel()
        if pen is not None:
            pen.mask = bool(think is not None and think.inside)
            pen.apply_single(logits[0, -1])
        tok = sampler(logits[0, -1]) if sampler is not None and sampler.on \
            else int(logits[0, -1].argmax())
        n_out = 1
        ctx.append(tok)
        if pen is not None:
            pen.commit([tok])
        if drafter is not None:
            drafter.observe([tok])
        yield tok
        if tok in eos:
            return
        # A sampled request decodes without a drafter: speculative verification is exact only
        # under greedy until rejection sampling is built (ENG-19, half two).
        tree_mode = (STATE.get("tree") and drafter is not None
                     and hasattr(drafter, "propose_tree")
                     and not (sampler is not None and sampler.on))
        while n_out < max_new:
            if deadline is not None and deadline.expired():
                return
            if tree_mode:
                # The tree path. It is the same loop with three lines changed: the drafter hands
                # back a shape rather than a list, the accept is a walk down that shape instead of
                # a prefix comparison, and the commit takes the path rather than a length. The
                # reason it is worth the branch is in notes/SPEED-LEDGER.md under "tree verify":
                # the step costs the same for two rows as for sixteen.
                tree = drafter.propose_tree(ctx, min(k, max_new - n_out))
                # ENG-16: the KV write counts NODES (anchor included) while the budget above
                # counts output tokens, and a drafter may return more nodes than it was handed.
                # A DFS pre-order prefix is a valid tree, so cutting at the row bound only
                # drops candidates.
                if tree is not None:
                    tree = tree.truncate(eng.max_len - pos)
                if tree is None or tree.n_draft == 0:
                    draft = []
                else:
                    block = torch.tensor(tree.tokens, device=prompt.device)
                    tvt = time.perf_counter()
                    lg = eng.forward_tree(block, tree.parents, start=pos)
                    if pen is not None:
                        pen.mask = bool(think is not None and think.inside)
                        pen.apply_tree(lg, tree)
                    picks_t = lg.argmax(-1).tolist()
                    on_verify = getattr(drafter, "on_verify", None)
                    if on_verify is not None:
                        on_verify(tree.n_draft + 1, (time.perf_counter() - tvt) * 1e3)
                    path, new = eng.accept_tree(tree, picks_t)
                    eng.commit_tree(path)
                    if hasattr(drafter, "sync"):
                        sel = torch.tensor(path, device=prompt.device)
                        toks = [int(tree.tokens[i]) for i in path]
                        if getattr(drafter, "wants_rows", False):
                            drafter.sync(toks, eng.hidden_post_norm[0, sel], pos, rows=path)
                        else:
                            drafter.sync(toks, eng.hidden_post_norm[0, sel], pos)
                    pos += len(path)
                    drafter.observe(new)
                    if pen is not None:
                        pen.commit(new)
                    stop_now = pstop is not None and pstop.observe(new)
                    for t in new:
                        ctx.append(t)
                        n_out += 1
                        yield t
                        if t in eos or n_out >= max_new:
                            return
                    if stop_now:
                        # The repeating block was yielded first: the client sees what was written,
                        # then the stream ends with finish_reason stop and a [req] annotation.
                        return
                    tok = ctx[-1]
                    if think is not None:
                        think.observe(new)
                        if think.hit:
                            for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                                  pen=pen, pstop=pstop, sampler=sampler):
                                n_out += 1
                                yield t
                                if n_out >= max_new:
                                    return
                            pos += 1 + len(think.close_ids)
                            tok = ctx[-1]
                    continue
            else:
                draft = (drafter.propose(ctx, min(k, max_new - n_out))
                         if drafter is not None and not (sampler is not None and sampler.on)
                         else [])
                # ENG-16: the block's KV write is `1 + len(draft)` rows (the anchor's own row is
                # one of them) while every clamp above counts output tokens, and a drafter may
                # return more than it was handed. Cap on rows here, where the forward is paid.
                draft = draft[:max(0, eng.max_len - pos - 1)]
            if not draft:
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                pos += 1
                if pen is not None:
                    pen.mask = bool(think is not None and think.inside)
                    pen.apply_single(logits[0, -1])
                tok = sampler(logits[0, -1]) if sampler is not None and sampler.on \
                    else int(logits[0, -1].argmax())
                ctx.append(tok)
                n_out += 1
                if pen is not None:
                    pen.commit([tok])
                if pstop is not None and pstop.observe([tok]):
                    yield tok
                    return
                if drafter is not None:
                    drafter.observe([tok])
                yield tok
                if tok in eos:
                    return
                if think is not None:
                    think.observe([tok])
                continue
            block = torch.tensor([tok] + draft, device=prompt.device)
            tv = time.perf_counter()
            lg = eng.forward_block(block, start=pos)
            if pen is not None:
                pen.mask = bool(think is not None and think.inside)
                pen.apply_chain(lg, draft)
            picks = lg.argmax(-1).tolist()
            # the same hook the bench loop has: a drafter that prices block widths learns what a
            # width costs from the loop that pays for it (engine/lenrouter.py)
            on_verify = getattr(drafter, "on_verify", None)
            if on_verify is not None:
                on_verify(len(draft) + 1, (time.perf_counter() - tv) * 1e3)
            relax = STATE["relax"]
            n = 0
            for i, d in enumerate(draft):
                if picks[i] == d:
                    n += 1
                    continue
                if relax.on and relax.accepts(lg[i], d, picks[i]):
                    n += 1
                    continue
                break
            new = draft[:n] + [picks[n]]
            if pen is not None:
                pen.commit(new)
            stop_now = pstop is not None and pstop.observe(new)
            if n < len(draft):
                eng.rollback_to(n + 1)
            if drafter is not None and hasattr(drafter, "sync"):
                drafter.sync([int(x) for x in block[:n + 1]], eng.hidden_post_norm[0, :n + 1], pos)
            pos += n + 1
            if drafter is not None:
                drafter.observe(new)
            for t in new:
                ctx.append(t)
                n_out += 1
                yield t
                if t in eos or n_out >= max_new:
                    return
            if stop_now:
                return
            tok = ctx[-1]
            if think is not None:
                think.observe(new)
                if think.hit:
                    for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                          pen=pen, pstop=pstop, sampler=sampler):
                        n_out += 1
                        yield t
                        if n_out >= max_new:
                            return
                    pos += 1 + len(think.close_ids)
                    tok = ctx[-1]


def _force_close(eng, drafter, think, ctx, pos, device, pen=None, pstop=None, sampler=None):
    """Close the reasoning block for the model and take the first token of its answer.

    The forced tokens are run through the engine exactly as generated ones are -- one forward at
    `pos` -- so the KV, the recurrent state and the drafter's context all carry them, and the
    answer that follows is conditioned on a block that really does end where it appears to.
    """
    # `ctx[-1]` is the last committed token and its own forward has not happened yet -- that is
    # the loop's invariant, `len(ctx) == pos + 1`, and it is why the next verify block starts with
    # it. The forced pass has to carry it, or the closing phrase would be written over its position.
    print(f"[think] closed the reasoning block: reason={think.reason or 'budget'} "
          f"at {think.n} tokens", flush=True)
    closing = list(think.close_ids)
    forced = [int(ctx[-1])] + closing
    lg = eng.forward(torch.tensor(forced, device=device), start=pos, last_only=True)
    if drafter is not None and hasattr(drafter, "sync"):
        drafter.sync(forced, eng.hidden_post_norm[0], pos)
    if drafter is not None:
        drafter.observe(closing)
    ctx.extend(closing)
    if pen is not None:
        # The forced tokens join the history -- they are real context -- but no penalty was
        # consulted in choosing them, which is the point of forcing. The token AFTER them is a
        # genuine decision, so it is penalized against a history that includes them.
        pen.commit(closing)
        pen.apply_single(lg[0, -1])
    think.observe(closing)
    if pstop is not None and pstop.observe(closing):
        for t in closing:
            yield t
        return
    for t in closing:
        yield t
    nxt = sampler(lg[0, -1]) if sampler is not None and sampler.on else int(lg[0, -1].argmax())
    ctx.append(nxt)
    if pen is not None:
        pen.commit([nxt])
    if drafter is not None:
        drafter.observe([nxt])
    yield nxt


def _remember(prompt_ids: list[int], out_ids: list[int], conv_id: str | None) -> None:
    """After a turn: keep the state it ended in, and add its tokens to the suffix store.

    The loop's invariant at the end of a generation is `kv.length == len(ctx) - 1` -- the last
    token has been decided and not forwarded -- so what is snapshotted is the prefix that really
    was forwarded, and `generate_stream` publishes that very list rather than one rebuilt here. The next turn's prompt begins with all of it plus the chat template's own glue,
    so it resumes here and pays for the glue and the new message rather than for the conversation.
    """
    eng, drafter = STATE["engine"], STATE["drafter"]
    store = STATE.get("state_store")
    committed = STATE.get("last_ctx") or []
    if store is not None and STATE.get("session_cache") and eng.kv.length:
        # `StateStore.put` declines when the snapshot is longer than the tokens it is given, which
        # is exactly the abandoned-mid-block case above -- and, before phase 9, was also the
        # ordinary case, because the loop advances `pos` by the whole accepted path and appends to
        # `ctx` one token at a time as it yields them. Every generation that ends inside a block
        # ends with `kv.length` ahead of `len(ctx)`, and every generation ends inside a block, so
        # the store was declining EVERY put and the session cache had never stored anything.
        # `puts: 0, hits: 0, misses: 40` after forty requests is what that looks like from
        # outside, and nothing else in the server complains.
        before = store.stats["puts"]
        store.put(committed, cache.capture(eng, drafter), conv_id)
        if STATE.get("verbose") and store.stats["puts"] == before:
            print(f"[cache] put declined: kv.length={eng.kv.length} ctx={len(committed)}",
                  flush=True)
        # `declined_short` in /v1/cache/stats is the same event counted, for when nobody is
        # reading the log.
    suffix = STATE.get("suffix_store")
    if suffix is not None:
        full = list(prompt_ids) + list(out_ids)
        suffix.append(full if STATE.get("suffix_scope") == "all" else out_ids)


def conversation_id(body: dict, headers) -> str | None:
    """A label for the conversation, if the client offers one. It is never load-bearing.

    The state store matches on the token prefix and checks it element for element, so a wrong or
    missing conversation id costs a cache hit and can never produce a wrong one. What the id buys
    is a readable `/v1/cache/stats` and, when two conversations share a prefix, a way to tell which
    entry belongs to which.
    """
    for key in ("conversation_id", "session_id", "user"):
        v = body.get(key)
        if isinstance(v, str) and v:
            return v[:128]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("conversation_id") or meta.get("session_id")
        if isinstance(v, str) and v:
            return v[:128]
    try:
        v = headers.get("X-Conversation-Id")
    except Exception:
        v = None
    return v[:128] if isinstance(v, str) and v else None


def build_prompt(body: dict) -> tuple[torch.Tensor, str, bool]:
    """The prompt ids, what kind of request it was, and whether it ends inside `<think>`.

    The third value is the whole of bug 1. `add_generation_prompt=True` with thinking enabled ends
    the rendered prompt with `<|im_start|>assistant\n<think>\n`, so the OPENING tag is part of the
    prompt and the model never generates it: the only tag in the output is `</think>`, and a client
    that folds a reasoning block on a matched pair sees an unmatched closer and folds nothing.
    Whoever renders the prompt is the only code that can know this, so it is answered here and the
    server puts the tag back.
    """
    tok = STATE["tok"]
    if "messages" in body:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs.setdefault("enable_thinking", True)
        # The template resolves `reasoning_effort` to xhigh unless told otherwise, and xhigh is a
        # paragraph of instructions telling the model to check its assumptions and consider
        # alternatives. A server default is the cheapest way to make it think less, because it
        # changes what the model is asked for rather than cutting it off part-way.
        effort = body.get("reasoning_effort") or STATE.get("reasoning_effort")
        if effort:
            kwargs.setdefault("reasoning_effort", effort)
        enc = tok.apply_chat_template(body["messages"], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, **kwargs)
        ids = _as_ids(enc)
        # Decoded from the ids rather than rendered a second time, so what is inspected is exactly
        # the token sequence the engine is about to be given.
        return ids, "chat", opens_think(tok.decode(ids.tolist(), skip_special_tokens=False))
    text = body.get("prompt")
    if isinstance(text, list):
        text = text[0]
    return _as_ids(tok(text or "", return_tensors="pt")), "text", False


def _as_ids(enc) -> torch.Tensor:
    """One 1-D int tensor of token ids, whatever shape the tokeniser handed back.

    The chat-template call returns a mapping, a plain tensor or a fast-tokeniser Encoding
    depending on the version and the arguments, and getting this wrong is a 500 on every request.
    """
    obj = enc
    for key in ("input_ids",):
        if hasattr(obj, key):
            obj = getattr(obj, key)
            break
        if isinstance(obj, dict) or hasattr(obj, "keys"):
            try:
                obj = obj[key]
                break
            except Exception:
                pass
    if not torch.is_tensor(obj):
        if hasattr(obj, "ids"):
            obj = torch.tensor(obj.ids, dtype=torch.long)
        else:
            obj = torch.tensor(obj, dtype=torch.long)
    while obj.dim() > 1:
        obj = obj[0]
    return obj.to(STATE["device"])


def eos_ids(body: dict) -> set[int]:
    tok = STATE["tok"]
    out = set()
    for t in (tok.eos_token_id, STATE["cfg_eos"]):
        if isinstance(t, int):
            out.add(t)
        elif isinstance(t, (list, tuple)):
            out.update(int(x) for x in t)
    return out


# ------------------------------------------------------------------ HTTP
def _chunk(cid: str, model: str, created: int, delta: dict, finish=None, usage=None,
           error=None) -> str:
    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [] if delta is None and finish is None else
            [{"index": 0, "delta": delta or {}, "finish_reason": finish, "logprobs": None}]}
    if usage is not None:
        body["usage"] = usage
    if error is not None:
        # Not in the OpenAI schema, and deliberately alongside a real `finish_reason` rather than
        # instead of one: a client that only knows the schema still sees the stream end, and a
        # client that looks can find out why.
        body["error"] = error
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


def _log_request(cid: str, n_prompt: int, n_out: int, finish: str, t0: float, *,
                 stream: bool, exc: BaseException | None = None,
                 pen: PenaltySpec | None = None, pattern: str | None = None) -> None:
    """One line per generation, always, whatever happened to it.

    The server used to log the HTTP status and nothing else, so an answer that stopped at the
    default token limit and an answer that stopped because the engine raised looked the same from
    the outside -- a 200 and a short reply. Everything needed to tell those apart is here.
    """
    ms = (time.perf_counter() - t0) * 1e3
    rate = (n_out - 1) / (ms / 1e3) if n_out > 1 and ms > 0 else 0.0
    with QUEUE:
        if finish == "error":
            INFLIGHT["errors"] += 1
        elif finish == "timeout":
            INFLIGHT["timeouts"] += 1
        elif finish == "abandoned":
            INFLIGHT["abandoned"] += 1
    tail = f"  !! {type(exc).__name__}: {exc}" if exc is not None else ""
    pen_s = (f" pen=({pen.rep:g},{pen.presence:g},{pen.freq:g},n={pen.no_repeat})"
             if pen is not None and pen.on else "")
    pat_s = f" pattern-stop({pattern})" if pattern else ""
    print(f"[req] {cid} {'stream' if stream else 'json'} prompt={n_prompt} "
          f"completion={n_out} finish={finish} {ms:.0f} ms {rate:.2f} tok/s{pen_s}{pat_s}{tail}",
          flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen38-spark-engine"

    def log_message(self, fmt, *args):
        if STATE.get("verbose"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -------------------------------------------------------------- helpers
    def _send_bytes(self, code: int, raw: bytes, extra_headers: tuple = ()) -> None:
        """Write one response, tolerating a client that hung up first (SRV-10).

        A health checker that disconnects mid-write used to raise BrokenPipeError out of here
        and into socketserver's handle_error, which prints a full traceback that reads as a
        server crash. The request is gone either way; swallow it and mark the connection closed.
        """
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in extra_headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, code: int, payload: dict, extra_headers: tuple = ()) -> None:
        self._send_bytes(code, json.dumps(payload, ensure_ascii=False).encode(), extra_headers)

    def _busy(self, code: int, message: str, retry: int = 5) -> None:
        """A refusal the caller can act on: a status, a reason, and when to come back."""
        self._send_bytes(code, json.dumps({"error": {"message": message, "type": "server_busy",
                                                     "code": code}}).encode(),
                         extra_headers=(("Retry-After", str(retry)),))

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # -------------------------------------------------------------- routes
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/health", "/healthz", "/v1/health"):
            # What a watchdog needs to decide whether to restart, and what a person needs to
            # decide whether it is wedged or merely busy. `draining` is the difference between
            # "not taking work" and "broken", and a restarter that cannot tell them apart will
            # kill a server in the middle of a graceful shutdown.
            return self._json(503 if STATE.get("draining") else 200, {
                "status": "draining" if STATE.get("draining") else "ok",
                "model": STATE.get("model"),
                "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
                "engine_busy": LOCK.locked(),
                "inflight": dict(INFLIGHT),
                "max_queue": STATE.get("max_queue"),
                "request_timeout_s": STATE.get("request_timeout"),
                "max_len": STATE.get("max_len"),
                "sampling": {"temperature": STATE.get("temperature", 0.0),
                             "top_p": STATE.get("top_p", 1.0),
                             "top_k": STATE.get("top_k", 0)},
                "penalty": {"rep": STATE["pen_spec"].rep,
                            "presence": STATE["pen_spec"].presence,
                            "freq": STATE["pen_spec"].freq,
                            "no_repeat": STATE["pen_spec"].no_repeat},
                "default_max_tokens": STATE.get("default_max_tokens"),
                "cache": cache_stats(),
                "memory": _memory(),
            })
        if path == "/metrics":
            # Prometheus, in its own text format. Everything it reports is collected in
            # server/metrics.py, which wraps this module rather than editing it; see its header.
            return metrics.serve(self)
        if path == "/v1/cache/stats":
            return self._json(200, cache_stats())
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": STATE["model"], "object": "model", "created": STATE["started"],
                 "owned_by": "local"}]})
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read()
        except Exception as exc:
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        if path == "/v1/cache/clear":
            # Measuring a warm number against a cold one needs a way back to cold that is not a
            # server restart, because a restart also throws away the Triton autotuning and the
            # first row would pay for the compiler -- the trap the phase-6 table was thrown away
            # for. This clears the caches and nothing else.
            with LOCK:
                for name in ("state_store", "response_cache"):
                    obj = STATE.get(name)
                    if obj is not None:
                        obj.clear() if hasattr(obj, "clear") else None
            return self._json(200, cache_stats())
        if path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        try:
            return self._complete(body, chat=path.endswith("chat/completions"))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            if self._streamed:
                # The response is an event stream whose headers are long gone. A JSON error body
                # written here would be appended to it as garbage, and the client would see the
                # truncation without the reason -- which is precisely bug 2's symptom.
                return
            try:
                return self._json(500, {"error": {"message": str(exc),
                                                  "type": "internal_error"}})
            except Exception:
                return

    _streamed = False

    def _complete(self, body: dict, chat: bool) -> None:
        t_req = time.perf_counter()
        # Sampling (ENG-19). Greedy is the exact path and stays the default; a request that asks
        # for sampling gets real sampling from engine/sample.py, on the single-token path (no
        # drafter) until rejection sampling lands. Seeded requests reproduce exactly.
        try:
            sampler = Sampler(
                temperature=float(body["temperature"]) if body.get("temperature") is not None
                else float(STATE.get("temperature", 0.0)),
                top_p=float(body["top_p"]) if body.get("top_p") is not None
                else float(STATE.get("top_p", 1.0)),
                top_k=int(body["top_k"]) if body.get("top_k") is not None
                else int(STATE.get("top_k", 0)),
                seed=body.get("seed"))
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": {"message": f"bad sampling parameter: {exc}",
                                              "type": "invalid_request_error",
                                              "param": "temperature"}})
        if sampler.temperature > 2.0:
            return self._json(400, {"error": {
                "message": f"temperature must be in [0, 2], got {sampler.temperature}",
                "type": "invalid_request_error", "param": "temperature"}})
        # BUG 2, the first half. This used to default to 256 tokens, and a client that does not
        # send `max_tokens` -- Open WebUI does not -- got an answer that stopped in the middle of
        # a sentence with `finish_reason: "length"` and no other sign that anything had happened.
        # Two hundred and fifty-six tokens is about forty lines of code, which is exactly where
        # the Mario page stopped. A serving default belongs at the context length, not at a
        # number small enough to truncate an ordinary answer.
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens")
                      or STATE.get("default_max_tokens") or 8192)
        budget = body.get("max_reasoning_tokens")
        if budget is None:
            budget = body.get("thinking_budget")
        if budget is None:
            budget = STATE.get("think_budget") or 0
        budget = int(budget or 0)
        # ENG-18: the budget and the stall close work on BOTH verify paths now -- the forced close
        # runs a chain-shaped forward of the closing phrase, which the tree path writes like any
        # other block. ENG-21: `think` is constructed even without a budget, because the stall
        # detector is the signal that replaces a cut.
        stream = bool(body.get("stream"))
        try:
            fmt = str(body.get("reasoning_format") or STATE.get("reasoning_format") or "tags")
            Reasoning(fmt)                       # validate before anything is generated
        except ValueError as exc:
            return self._json(400, {"error": {"message": str(exc),
                                              "type": "invalid_request_error",
                                              "param": "reasoning_format"}})
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        # Anti-repetition penalties (ENG-17). Deterministic on the target's logits, so greedy and
        # speculative decoding stay identical under the rule -- see engine/penalty.py. The
        # per-request names are the OpenAI ones plus the HF one; defaults come from the server
        # flags. NOTE: clients that already send presence_penalty (Open WebUI's roster rows do)
        # change from ignored to honoured the day this ships -- that is the fix, and it is said
        # out loud in LIMITATIONS.md.
        base: PenaltySpec = STATE["pen_spec"]

        def _p(name: str, default: float) -> float:
            v = body.get(name)
            return float(v) if v is not None else default

        try:
            pen_spec = PenaltySpec(rep=_p("repetition_penalty", base.rep),
                                   presence=_p("presence_penalty", base.presence),
                                   freq=_p("frequency_penalty", base.freq),
                                   no_repeat=int(_p("no_repeat_ngram_size", base.no_repeat)))
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": {"message": f"bad penalty parameter: {exc}",
                                              "type": "invalid_request_error",
                                              "param": "repetition_penalty"}})
        pen = (PenaltyState(pen_spec, STATE["engine"].cfg.vocab_size, "cuda")
               if pen_spec.on else None)
        pstop = PatternStop(*STATE["pattern_stop"]) if STATE.get("pattern_stop") else None
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        model = body.get("model") or STATE["model"]
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        tok = STATE["tok"]

        conv_id = conversation_id(body, self.headers)

        if STATE.get("draining"):
            return self._busy(503, "the server is shutting down", retry=30)
        with QUEUE:
            if INFLIGHT["waiting"] >= int(STATE.get("max_queue", 8)):
                INFLIGHT["refused"] += 1
                return self._busy(503, f"{INFLIGHT['waiting']} requests are already queued and "
                                       f"this engine serves one at a time", retry=5)
            INFLIGHT["waiting"] += 1
        try:
            got = LOCK.acquire(timeout=float(STATE.get("queue_timeout", 120.0)))
        finally:
            # Off the waiting list whatever happened, including an exception in `acquire` itself.
            # A counter that leaks on the error path turns the queue bound into a slowly closing
            # door, and the symptom -- 503s on an idle server, hours later -- would be read as a
            # leak somewhere else entirely.
            with QUEUE:
                INFLIGHT["waiting"] -= 1
        if not got:
            with QUEUE:
                INFLIGHT["refused"] += 1
            return self._busy(429, "timed out waiting for the engine", retry=10)
        with QUEUE:
            INFLIGHT["running"] += 1
        deadline = Deadline(float(STATE.get("request_timeout", 0.0)))
        try:
            prompt, _, in_think = build_prompt(body)
            eos = eos_ids(body)
            n_prompt = int(prompt.numel())
            # BUG 2, the second half. The KV buffer is `--max-len` long and the recurrent state is
            # indexed by absolute position, so a generation that runs past the end of it does not
            # degrade -- it raises, part way through a stream whose headers have already gone out.
            # The request is clamped to what the engine can hold, and refused outright when the
            # prompt alone does not fit, which is a 400 the client can read rather than a truncated
            # answer it cannot.
            room = int(STATE["max_len"]) - n_prompt - 1
            if room <= 0:
                return self._json(400, {"error": {
                    "message": f"prompt is {n_prompt} tokens and the context is "
                               f"{STATE['max_len']}; nothing is left to generate",
                    "type": "invalid_request_error", "param": "messages"}})
            max_new = max(1, min(max_new, room))
            think = ThinkBudget(tok, budget, stall=bool(STATE.get("think_stall", True)))
            prompt_ids = prompt.tolist()

            # The exact-prompt response cache. Greedy decoding is a function of (prompt, params),
            # so an identical request has an identical answer and this is memoisation rather than
            # an approximation. Under a relaxed accept rule the engine is not answering the
            # greedy question at all, and that is the one setting where the key would be lying
            # about what produced the value -- so the cache is not consulted.
            rcache = STATE.get("response_cache")
            rkey, cached_ids = None, None
            if rcache is not None and not STATE["relax"].on and not sampler.on:
                # A sampled answer is not a function of (prompt, params) in any replayable sense
                # without the RNG stream; do not memoise it.
                # The penalty values are part of the question being memoised: greedy under
                # penalties is a different function, and a key without them would replay an
                # answer a different setting produced (ENG-17's cache-key fix).
                rkey = cache.ResponseCache.key(
                    prompt_ids, max_new=max_new, budget=budget, stops=tuple(stops),
                    eos=tuple(sorted(eos)), tree=bool(STATE.get("tree")), pen=pen_spec.key())
                cached_ids = rcache.get(rkey)
            source = (iter(list(cached_ids)) if cached_ids is not None
                      else generate_stream(prompt, max_new, eos, think, conv_id, deadline,
                                          pen=pen, pstop=pstop, sampler=sampler))

            if not stream:
                ids = []
                for t in source:
                    ids.append(t)
                if cached_ids is None:
                    _remember(prompt_ids, ids, conv_id)
                    if rkey is not None:
                        rcache.put(rkey, ids, prompt_ids)
                text = tok.decode(ids, skip_special_tokens=True)
                if pstop is not None and pstop.hit:
                    finish = "stop"
                else:
                    finish = "stop" if (ids and ids[-1] in eos) else (
                        "timeout" if deadline.hit else "length")
                text, cut = _apply_stops(text, stops)
                if cut:
                    finish = "stop"
                if pstop is not None and pstop.hit:
                    text += GUARD_MARKER
                usage = {"prompt_tokens": n_prompt, "completion_tokens": len(ids),
                         "total_tokens": n_prompt + len(ids)}
                _log_request(cid, n_prompt, len(ids), finish, t_req, stream=False, pen=pen_spec,
                             pattern=(pstop.label if pstop is not None and pstop.hit else None))
                if chat:
                    content, reasoning = split_full(text, fmt, in_think=in_think)
                    message = {"role": "assistant", "content": content}
                    if reasoning is not None:
                        message["reasoning_content"] = reasoning
                    payload = {"id": cid, "object": "chat.completion", "created": created,
                               "model": model, "usage": usage,
                               "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                                            "message": message}]}
                else:
                    payload = {"id": cid, "object": "text_completion", "created": created,
                               "model": model, "usage": usage,
                               "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                                            "text": text}]}
                return self._json(200, payload, extra_headers=_guard_headers(pstop))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            w = self.wfile
            self._streamed = True
            # One piece of text at a time, and every character in it final: `Detokenizer` holds a
            # code point back until all of its bytes have arrived, which is bug 3. `Reasoning` says
            # which field each piece belongs in, which is the second half of bug 1.
            det = Detokenizer(lambda seq: tok.decode(seq, skip_special_tokens=True))
            split = Reasoning(fmt, in_think=in_think)

            def send(pairs) -> None:
                for field, piece in pairs:
                    if not piece:
                        continue
                    if chat:
                        key = "reasoning_content" if field == "reasoning" else "content"
                        w.write(_chunk(cid, model, created, {key: piece}).encode())
                    else:
                        w.write(_text_chunk(cid, model, created, piece).encode())
                    w.flush()

            if chat:
                w.write(_chunk(cid, model, created, {"role": "assistant", "content": ""}).encode())
                w.flush()
            # BUG 1. The prompt ended inside `<think>`, so the opening tag is already spent and the
            # model will only ever write the closing one. Put it back, as the first content delta,
            # before any generated text goes out.
            if in_think and fmt in ("tags", "both"):
                send([("content", "<think>\n")])
            ids: list[int] = []
            finish = "length"
            failed: BaseException | None = None
            # reset per stream; the non-streamed branch returns before this point
            cut = False
            try:
                for t in source:
                    ids.append(t)
                    if t in eos:
                        finish = "stop"
                        break
                    piece = det.push(ids)
                    if not piece:
                        continue                   # a byte-level token that is not a character yet
                    cut_at = _stop_index(det.emitted, stops)
                    if cut_at is not None:
                        send(split.push(piece[: max(0, cut_at - (len(det.emitted) - len(piece)))]))
                        finish, cut = "stop", True
                        break
                    send(split.push(piece))
                if not cut:
                    send(split.push(det.flush(ids)))
                if pstop is not None and pstop.hit:
                    send([("content", GUARD_MARKER)])
                send(split.finish())
                if pstop is not None and pstop.hit and finish == "length":
                    finish = "stop"
                elif deadline.hit and finish == "length":
                    finish = "timeout"
            except (BrokenPipeError, ConnectionResetError):
                # The reader hung up -- a closed pipe and a reset connection are the same event
                # seen from two kernels, and neither is this server's fault. There is nothing to
                # report and nowhere to report it.
                _log_request(cid, n_prompt, len(ids), "abandoned", t_req, stream=True, pen=pen_spec,
                             pattern=(pstop.label if pstop is not None and pstop.hit else None))
                raise
            except Exception as exc:                                  # noqa: BLE001
                # BUG 2, the third half. The headers of a stream go out before the first token, so
                # an exception raised half way through used to unwind into `do_POST`, which tried
                # to send a 500 -- a JSON body, with its own status line, appended to a live
                # event stream. Every client on earth reads that as a stream that simply stopped.
                # It is logged here, and the stream is CLOSED PROPERLY: the partial text the reader
                # already has, then a finish reason that says what happened.
                import traceback
                failed = exc
                finish = "error"
                traceback.print_exc()
            if cached_ids is None and failed is None:
                _remember(prompt_ids, ids, conv_id)
                if rkey is not None:
                    rcache.put(rkey, ids, prompt_ids)
            _log_request(cid, n_prompt, len(ids), finish, t_req, stream=True, exc=failed, pen=pen_spec,
                         pattern=(pstop.label if pstop is not None and pstop.hit else None))
            try:
                if failed is not None and chat:
                    w.write(_chunk(cid, model, created, {}, finish=finish,
                                   error={"message": str(failed),
                                          "type": type(failed).__name__}).encode())
                else:
                    w.write((_chunk(cid, model, created, {}, finish=finish) if chat
                             else _text_chunk(cid, model, created, "", finish=finish)).encode())
                if want_usage:
                    n_out = len(ids)
                    w.write(_chunk(cid, model, created, None, usage={
                        "prompt_tokens": n_prompt, "completion_tokens": n_out,
                        "total_tokens": n_prompt + n_out}).encode())
                w.write(b"data: [DONE]\n\n")
                w.flush()
            except BrokenPipeError:
                pass
        finally:
            LOCK.release()
            with QUEUE:
                INFLIGHT["running"] -= 1
                INFLIGHT["served"] += 1


def _memory() -> dict:
    """What a soak run watches for a leak.

    On this board the GPU and the host share one pool and `nvidia-smi` reports N/A for used
    memory, so the numbers that mean anything are the allocator's own -- `allocated` is what the
    engine is holding and `reserved` is what it has taken from the driver and not given back. A
    leak shows as `allocated` climbing across requests; fragmentation shows as `reserved` climbing
    while `allocated` does not.
    """
    out = {}
    try:
        out["allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        out["reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
        out["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = str(exc)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_gb"] = round(int(line.split()[1]) / 2**20, 3)
                    break
    except Exception:
        pass
    return out


def cache_stats() -> dict:
    """What `/v1/cache/stats` answers: what is held, what it costs, and what it bought.

    The byte figures are the real ones -- every snapshot is asked for its own `nbytes` and the
    parts are broken out -- because the interesting question about a state cache on a 121 GiB board
    is not whether it hits, it is what a hit costs to keep. On this model the recurrent state is
    ~150 MB per snapshot whatever the prefix length, and the KV is ~0.8 MB a token; a budget is a
    number of conversations long before it is a number of tokens long.
    """
    eng = STATE.get("engine")
    out = {"model": STATE.get("model"),
           "session_cache": bool(STATE.get("session_cache")),
           "prefix_cache": bool(STATE.get("prefix_cache")),
           "prefix_chunk": STATE.get("prefix_chunk", 0),
           "last_prefill": STATE.get("last_prefill")}
    store = STATE.get("state_store")
    out["state_store"] = store.report() if store is not None else None
    rcache = STATE.get("response_cache")
    out["response_cache"] = rcache.report() if rcache is not None else None
    suffix = STATE.get("suffix_store")
    out["suffix_store"] = suffix.report() if suffix is not None else None
    if eng is not None:
        cfg = eng.cfg
        kv_per_token = (len(cfg.attention_layers) * cfg.num_key_value_heads * cfg.head_dim * 2 * 2)
        out["snapshot_cost"] = {
            "recurrent_bytes": eng.state.S.numel() * 4,
            "conv_bytes": eng.state.conv.numel() * eng.state.conv.element_size(),
            "kv_bytes_per_token": kv_per_token,
        }
    return out


def _text_chunk(cid, model, created, piece, finish=None) -> str:
    body = {"id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": piece, "finish_reason": finish, "logprobs": None}]}
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


def _stop_index(text: str, stops: list[str]) -> int | None:
    hits = [text.find(s) for s in stops if s and text.find(s) >= 0]
    return min(hits) if hits else None


def _apply_stops(text: str, stops: list[str]) -> tuple[str, bool]:
    i = _stop_index(text, stops)
    return (text[:i], True) if i is not None else (text, False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--served-model", default="qwen3.8-27b-spark-engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=8192,
                    help="context length. The KV buffer is allocated for all of it up front, so "
                         "this is a memory decision as much as a capability one: 64 KiB a token "
                         "of engine KV plus 20 KiB a token per drafter arm — 27.9 GB of KV and "
                         "53.4 GB of whole stack at 262,144 (measured 2026-09-18, "
                         "tools/mem_audit.py)")
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="deterministic repetition penalty on the target's logits, before the "
                         "argmax (1.0 = off, today's output). Exact under speculation because "
                         "acceptance is argmax equality; see engine/penalty.py. Per request: "
                         "send `repetition_penalty` in the body")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="flat subtraction from tokens already in the history (0 = off). "
                         "Per request: `presence_penalty` in the body — OpenAI clients and "
                         "Open WebUI already send it, so the day this ships those requests "
                         "change from ignored to honoured")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="subtraction per occurrence, on tokens already in the history (0 = off). "
                         "Per request: `frequency_penalty` in the body")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="server default sampling temperature (0 = greedy, the exact path). "
                         "A request that sends temperature > 0 samples -- via engine/sample.py -- "
                         "and decodes without a drafter until rejection sampling exists (ENG-19)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="ENG-20: forbid the token that would complete an n-gram already seen "
                         "(HF's no_repeat_ngram_size; 0 = off, must be >= 2). Deterministic, so "
                         "speculation stays exact; per request: `no_repeat_ngram_size`")
    ap.add_argument("--pattern-stop", default="",
                    help="end a generation that repeats one short pattern forever, as MAX:MIN:COUNT "
                         "(e.g. 64:1:8 -- a block of 1..64 tokens repeated 8 times ends the "
                         "generation with finish_reason stop). The backstop for a greedy loop the "
                         "penalties cannot break, vLLM's RepetitionDetectionParams; empty = off")
    ap.add_argument("--default-max-tokens", type=int, default=8192,
                    help="what a request that does not send max_tokens gets. It used to be 256 "
                         "and a long answer stopped in the middle of a line")
    ap.add_argument("--reasoning-format", default="tags",
                    choices=("tags", "reasoning_content", "both"),
                    help="how the reasoning block reaches the client: inside <think></think> in "
                         "content (default, and what Open WebUI folds on), as OpenAI-style "
                         "reasoning_content deltas, or both. Overridable per request")
    ap.add_argument("--drafter", default="mtp",
                    choices=("mtp", "router", "merged", "dflash2", "lenrouter", "none"))
    ap.add_argument("--tree", action="store_true",
                    help="verify a draft TREE per step instead of a chain, where the drafter "
                         "builds one; see notes/SPEED-LEDGER.md, section 'tree verify'")
    ap.add_argument("--budget", type=int, default=16, help="nodes per tree, anchor included")
    ap.add_argument("--df2-temp", type=float, default=1.0)
    ap.add_argument("--corpus", default=os.environ.get("QWEN38_CORPUS", ""),
                    help="suffix store for the lookup drafter, used by --drafter merged")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--k", type=int, default=0, help="verify block size; 0 = the drafter's depth")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None,
                    help="e4m3 lm_head from tools/quant_head.py; halves the head's 2.54 GB in "
                         "both the verify pass and the block drafter's top-k read")
    ap.add_argument("--dflash2-blocks", type=int, default=1,
                    help="chained 8-wide draft blocks; 1 proposes 7 tokens, 2 proposes 14")
    ap.add_argument("--dflash2-path", default="greedy", choices=("greedy", "viterbi"))
    ap.add_argument("--dflash2-ckpt", default=None)
    ap.add_argument("--dflash2-ckpt16", default=None,
                    help="the sixteen-wide drafter, for --drafter lenrouter. The router holds both "
                         "checkpoints and picks the block length per step; see engine/lenrouter.py")
    ap.add_argument("--len-fixed", type=int, default=0,
                    help="pin --drafter lenrouter to one block width (8 or 16). 0 lets the "
                         "policy choose, which since phase 9 means --len-latch: one decision a "
                         "request. 8 and 16 are how the fixed baselines in RESULTS.md are "
                         "measured through the same code")
    ap.add_argument("--len-latch", action=argparse.BooleanOptionalAction, default=True,
                    help="decide the block width ONCE a request -- four wide blocks, up to four "
                         "narrow probes, then no more switching. ON by default since phase 9, "
                         "because switching arms costs 3-18 %% of acceptance depending on the "
                         "workload whether or not the switch was a good idea (SPEED-LEDGER 00:50). "
                         "--no-len-latch restores the per-block router; needs --len-fixed 0 to do "
                         "anything either way")
    ap.add_argument("--len-explore", type=int, default=32,
                    help="blocks between forced wide probes when nothing suggests one")
    ap.add_argument("--drop-idle", action=argparse.BooleanOptionalAction, default=False,
                    help="release the arm that LOSES the latch for the rest of the request: no "
                         "tap, no sync, and nothing of it in the state snapshot. It stops about "
                         "1.5 ms a block of keeping a cache current for a drafter that will not "
                         "draft again, and takes a third off every snapshot, which is a third "
                         "more entries inside the same cache budget. REFUSES to start with "
                         "--no-len-latch, since the release only happens when the latch closes. "
                         "Off by default until the phase-10 measurement says otherwise")
    ap.add_argument("--relax-tau", type=float, default=1.0,
                    help="LOSSY. Accept a drafted token whose probability is at least this fraction "
                         "of the argmax's. 1.0 is the lossless rule and the default")
    ap.add_argument("--relax-rank", type=int, default=1,
                    help="LOSSY. Accept a drafted token among the target's top-r. 1 is lossless")
    ap.add_argument("--think-stall", dest="think_stall", action="store_true", default=True,
                    help="ENG-21: while the reasoning block is open, watch for looping (a short "
                         "pattern repeated) or novelty collapse and SIGNAL the model by closing "
                         "the block with the vendor's phrase -- it concludes and answers instead "
                         "of the stream being cut. --no-think-stall disables it")
    ap.add_argument("--no-think-stall", dest="think_stall", action="store_false",
                    help="disable the stall close (the budget still applies)")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="default cap on reasoning tokens per request, 0 = uncapped. A request may "
                         "override it with `max_reasoning_tokens`. When the cap is reached the "
                         "engine closes the reasoning block itself -- see engine/spec.py's "
                         "ThinkBudget -- which CHANGES THE ANSWER and is not a speed trick")
    ap.add_argument("--reasoning-effort", default=None, choices=("low", "medium", "xhigh"),
                    help="default `reasoning_effort` for the chat template. The template's own "
                         "default is xhigh, which is a paragraph asking the model to validate "
                         "assumptions and weigh alternatives before answering")
    ap.add_argument("--cache-budget-gb", type=float, default=40.0,
                    help="RAM budget for the state cache, in GiB. One snapshot is the 48 recurrent "
                         "states (~150 MB on this model, whatever the prefix length) plus the KV "
                         "of the prefix, so this is a number of CONVERSATIONS, not of tokens. "
                         "0 turns the state cache off entirely")
    ap.add_argument("--no-session-cache", action="store_true",
                    help="do not keep a conversation's state after its turn. On by default: the "
                         "next turn then re-reads the whole conversation through 64 layers")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="do not checkpoint a prefill at chunk boundaries. On by default, which "
                         "is what makes a shared system prompt free from the second request on")
    ap.add_argument("--prefix-chunk", type=int, default=1024,
                    help="tokens between prefill checkpoints, and the forward size of EVERY "
                         "prefill while the prefix cache is on -- the two have to agree or a warm "
                         "prefill is not the same arithmetic as a cold one. A CHUNK COSTS A WHOLE "
                         "16.35 GB WEIGHT READ: a 1,724-token prompt is 1.14x at 1024 and 1.57x "
                         "at 256 (SPEED-LEDGER, track D). 1024 is the default because a prompt "
                         "nobody shares pays that and gets nothing. Drop it to 256 if you serve "
                         "one system prompt to many different tails, where the finer grid wins "
                         "back far more than it costs")
    ap.add_argument("--response-cache", action="store_true",
                    help="OPT-IN. Answer an identical (prompt, params) request from memory. "
                         "Greedy decoding is a function so this is exact, but a server that "
                         "answers from a dictionary must never be what a benchmark measures")
    ap.add_argument("--response-cache-mb", type=float, default=256.0)
    ap.add_argument("--response-cache-ttl", type=float, default=3600.0)
    ap.add_argument("--suffix-store", default=os.environ.get(
        "QWEN38_SUFFIX_STORE", "~/.qwen38-spark-engine/suffix"),
        help="directory for the persistent suffix store of what this engine has read and written, "
             "which the lookup drafter reads as a second corpus. Token ids only, never text, "
             "outside this repository, mode 0700. Empty string turns it off")
    ap.add_argument("--suffix-store-mb", type=float, default=192.0,
                    help="cap on the store, in MiB of int32 token ids (192 MiB = 48 M tokens). "
                         "Over the cap the oldest half is forgotten at the next document boundary")
    ap.add_argument("--suffix-store-scope", default="all", choices=("all", "outputs"),
                    help="`all` remembers prompts and answers, `outputs` only what the engine "
                         "wrote. Both stay on the box")
    ap.add_argument("--request-timeout", type=float, default=900.0,
                    help="wall-clock cap on one generation, checked between blocks; 0 disables. "
                         "A request that hits it finishes with reason 'timeout', which is a "
                         "different thing from 'length' and the caller should be told which")
    ap.add_argument("--max-queue", type=int, default=8,
                    help="requests allowed to wait for the engine before new ones get a 503 with "
                         "Retry-After. This engine serves one sequence at a time")
    ap.add_argument("--queue-timeout", type=float, default=120.0,
                    help="how long a request waits for the engine before a 429")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    t0 = time.time()
    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=a.drafter in ("none", "dflash2", "merged", "lenrouter"),
                nvfp4=a.nvfp4,
                fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    drafter = None
    if a.drafter == "mtp":
        from engine.drafters.mtp import MTPDrafter
        drafter = MTPDrafter(eng, max_len=a.max_len, depth=a.depth, draft_head=a.draft_head)
    elif a.drafter == "router":
        from engine.router import RouterDrafter
        drafter = RouterDrafter(eng, max_len=a.max_len, depth=a.depth)
    elif a.drafter == "merged":
        # The configuration the 11:18 gate passed on, plus the tree: the block drafter priced
        # against the lookup drafter every step, both putting candidates in one verify call.
        from engine.drafters.dflash2 import DFlash2Drafter
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, VERIFY_MS, VERIFY_MS_NVFP4
        table = VERIFY_MS_NVFP4 if a.nvfp4 or os.environ.get("QWEN38_NVFP4") else VERIFY_MS
        head = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, max_len=a.max_len,
                              path=a.dflash2_path, draft_head=a.draft_head)
        head.tree_temp = a.df2_temp
        head._build()
        # `--budget` counts nodes INCLUDING the anchor, because that is what the measured curve is
        # keyed by and where its cliff is: 16 nodes cost 164.4 ms and 17 cost 172. So the drafters
        # get one fewer.
        ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16,
                          node_budget=a.budget - 1, branch_top_k=3, min_expected=0.2,
                          alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                          verify_base_ms=table[min(table)],
                          verify_per_node_ms=(table[max(table)] - table[min(table)])
                          / (max(table) - min(table)))
        drafter = MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                               node_budget=a.budget - 1, mtp_ms_per_token=0.0,
                               head_fixed_ms=35.0, adaptive_depth=False,
                               rollback_ms=6.2, verify_ms_table=table)
        a.depth = a.budget - 1
    elif a.drafter == "dflash2":
        from engine.drafters.dflash2 import DFlash2Drafter
        drafter = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=a.dflash2_blocks,
                                 path=a.dflash2_path, draft_head=a.draft_head,
                                 max_len=a.max_len)
        drafter._build()
        # The block width, not `--depth`, is what this drafter proposes per verify pass.
        a.depth = (drafter.cfg.block_size - 1) * a.dflash2_blocks
    elif a.drafter == "lenrouter":
        from engine.drafters.dflash2 import DFlash2Drafter
        from engine.lenrouter import LengthRouter
        if not a.dflash2_ckpt16:
            raise SystemExit("--drafter lenrouter needs --dflash2-ckpt16")
        small = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, path=a.dflash2_path,
                               draft_head=a.draft_head, max_len=a.max_len, block=8)
        small._build()
        large = DFlash2Drafter(eng, a.dflash2_ckpt16, blocks=1, path=a.dflash2_path,
                               draft_head=a.draft_head, max_len=a.max_len, block=16)
        large._build()
        if a.tree:
            # The combined configuration: each arm is the lookup drafter's tree merged with that
            # arm's own lattice, and the length router chooses the node budget. One NgramDrafter
            # for both arms -- its suffix index is updated in `observe`, and two arms each
            # observing every block would index every token twice.
            from engine.drafters.ngram import NgramDrafter
            from engine.router import MergedRouter
            tree_table = {8: 121.7, 16: 129.2, 32: 163.2}
            ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16,
                              node_budget=large.cfg.block_size - 1, branch_top_k=3,
                              min_expected=0.2, alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                              verify_base_ms=tree_table[8],
                              verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
            arms = [MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                                 node_budget=head.cfg.block_size - 1, mtp_ms_per_token=0.0,
                                 head_fixed_ms=27.0, adaptive_depth=False, rollback_ms=6.4,
                                 verify_ms_table=dict(tree_table), tree_ms_table=dict(tree_table))
                    for head in (small, large)]
            drafter = LengthRouter(arms[0], arms[1], fixed=a.len_fixed,
                                   explore_period=a.len_explore, tree=True, ngram=ng,
                                   latch=a.len_latch, drop_idle=a.drop_idle)
        else:
            drafter = LengthRouter(small, large, fixed=a.len_fixed,
                                   explore_period=a.len_explore, latch=a.len_latch,
                                   width_trim=not a.len_latch, drop_idle=a.drop_idle)
        # The router may propose the wide block on any step, so the loop's cap has to be the wide
        # one; asking it for fewer would silently pin it to the narrow length.
        a.depth = large.cfg.block_size - 1
    gen_cfg = os.path.join(cfg.path, "generation_config.json")
    cfg_eos = None
    if os.path.isfile(gen_cfg):
        cfg_eos = json.load(open(gen_cfg)).get("eos_token_id")
    relax = Relax(a.relax_tau, a.relax_rank)

    # --- the serving-time caches (engine/cache.py) ---------------------------------------------
    session_on = not a.no_session_cache
    prefix_on = not a.no_prefix_cache
    budget = int(a.cache_budget_gb * (1 << 30))
    if (session_on or prefix_on) and not cache.drafter_is_cacheable(drafter):
        # This drafter's cache is indexed by absolute position and it cannot hand it over. A
        # restored prefix would leave a permanent hole in it and the engine would decode at one
        # token a block, which is a far worse trade than a cold prefill.
        print(f"[cache] drafter {a.drafter} cannot snapshot its own cache; state cache OFF")
        session_on = prefix_on = False
    store = cache.StateStore(budget, chunk=a.prefix_chunk) if (budget and
                                                               (session_on or prefix_on)) else None
    rcache = (cache.ResponseCache(int(a.response_cache_mb * (1 << 20)), a.response_cache_ttl)
              if a.response_cache else None)
    suffix = None
    if a.suffix_store:
        suffix = cache.PersistentSuffixStore(
            a.suffix_store, max_tokens=int(a.suffix_store_mb * (1 << 20)) // 4).open()
        reader = next((d for d in (drafter, getattr(drafter, "ngram", None),
                                   getattr(drafter, "engram", None))
                       if hasattr(d, "add_store")), None)
        if reader is not None:
            reader.add_store(suffix)
        else:
            print(f"[cache] drafter {a.drafter} has no lookup store; the suffix store is being "
                  f"written but nothing reads it")

    STATE.update(engine=eng, tok=tok, drafter=drafter, k=a.k or a.depth, device="cuda",
                 tree=bool(a.tree), relax=relax,
                 model=a.served_model, started=int(time.time()), verbose=a.verbose,
                 max_len=int(a.max_len), default_max_tokens=int(a.default_max_tokens),
                 reasoning_format=a.reasoning_format, request_timeout=float(a.request_timeout),
                 max_queue=int(a.max_queue), queue_timeout=float(a.queue_timeout),
                 draining=False,
                 cfg_eos=cfg_eos, think_budget=a.think_budget, think_stall=a.think_stall,
                 reasoning_effort=a.reasoning_effort,
                 pen_spec=PenaltySpec(a.rep_penalty, a.presence_penalty,
                                     a.frequency_penalty, a.no_repeat_ngram),
                 temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                 pattern_stop=(tuple(int(x) for x in a.pattern_stop.split(":"))
                               if a.pattern_stop else None),
                 state_store=store, session_cache=session_on, prefix_cache=prefix_on,
                 prefix_chunk=a.prefix_chunk if prefix_on else 0,
                 response_cache=rcache, suffix_store=suffix, suffix_scope=a.suffix_store_scope)
    print(f"[server] {w.report()}")
    if relax.on:
        print(f"[server] LOSSY ACCEPT RULE ON: tau={relax.tau} rank={relax.rank}. Output is not "
              f"greedy and not reproducible against any lossless run. See LIMITATIONS.md")
    print(f"[cache] session={'on' if session_on else 'off'} "
          f"prefix={'on' if prefix_on else 'off'}/{a.prefix_chunk} "
          f"budget={a.cache_budget_gb:.0f} GiB "
          f"response={'on' if rcache else 'off'} "
          f"suffix={(suffix.report()['tokens'] if suffix else 0)} tokens")
    print(f"[server] drafter={a.drafter} depth={a.depth} k={STATE['k']} "
          f"nvfp4={w.nvfp4_source or 'off'} fp8_head={'on' if w.fp8_head_source else 'off'} "
          f" loaded in {time.time() - t0:.1f}s")
    # one warm request, so the first measured one is not paying for Triton autotuning
    with torch.no_grad():
        list(generate_stream(tok("warm up the kernels", return_tensors="pt").input_ids[0].cuda(),
                             4, set()))
    # After the warm-up, so the request that pays for Triton autotuning is not in the histograms.
    metrics.install(sys.modules[__name__])
    print(f"[server] listening on http://{a.host}:{a.port}  model {a.served_model}", flush=True)
    print(f"[server] queue max {a.max_queue} wait {a.queue_timeout:.0f}s "
          f"request timeout {a.request_timeout:.0f}s", flush=True)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)

    def _drain(signum, _frame):
        """SIGTERM/SIGINT: stop taking work, let the generation in flight finish, then exit.

        A hard kill in the middle of a verify leaves the caller with a truncated stream and the
        watchdog with no way to tell a crash from a deploy. `draining` makes new requests a 503
        with `Retry-After`, `/health` says `draining` rather than `ok`, and `shutdown()` returns
        once the handler threads are done -- it has to run off the serving thread or it deadlocks
        against the loop it is stopping.
        """
        if STATE.get("draining"):
            return
        STATE["draining"] = True
        print(f"[server] signal {signum}: draining, {INFLIGHT['running']} running, "
              f"{INFLIGHT['waiting']} queued", flush=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _drain)
    signal.signal(signal.SIGINT, _drain)
    try:
        httpd.serve_forever()
    finally:
        # The engine holds one sequence and the drafters hold caches indexed by absolute position;
        # neither survives the process, so there is nothing to persist. What is worth printing is
        # the tally, because a soak run reads it from the last line of the log.
        print(f"[server] stopped. {INFLIGHT}", flush=True)


if __name__ == "__main__":
    main()
