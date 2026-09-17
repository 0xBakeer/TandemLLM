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

STATE: dict = {}
LOCK = threading.Lock()


# ------------------------------------------------------------------ generation
def generate_stream(prompt: torch.Tensor, max_new: int, eos: set[int], think=None,
                    conv_id: str | None = None):
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
        tok = int(logits[0, -1].argmax())
        n_out = 1
        ctx.append(tok)
        if drafter is not None:
            drafter.observe([tok])
        yield tok
        if tok in eos:
            return
        tree_mode = STATE.get("tree") and drafter is not None and hasattr(drafter, "propose_tree")
        while n_out < max_new:
            if tree_mode:
                # The tree path. It is the same loop with three lines changed: the drafter hands
                # back a shape rather than a list, the accept is a walk down that shape instead of
                # a prefix comparison, and the commit takes the path rather than a length. The
                # reason it is worth the branch is in notes/SPEED-LEDGER.md under "tree verify":
                # the step costs the same for two rows as for sixteen.
                tree = drafter.propose_tree(ctx, min(k, max_new - n_out))
                if tree is None or tree.n_draft == 0:
                    draft = []
                else:
                    block = torch.tensor(tree.tokens, device=prompt.device)
                    tvt = time.perf_counter()
                    lg = eng.forward_tree(block, tree.parents, start=pos)
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
                    for t in new:
                        ctx.append(t)
                        n_out += 1
                        yield t
                        if t in eos or n_out >= max_new:
                            return
                    tok = ctx[-1]
                    if think is not None:
                        think.observe(new)
                    continue
            else:
                draft = drafter.propose(ctx, min(k, max_new - n_out)) if drafter is not None else []
            if not draft:
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                pos += 1
                tok = int(logits[0, -1].argmax())
                ctx.append(tok)
                n_out += 1
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
            tok = ctx[-1]
            if think is not None:
                think.observe(new)
                if think.hit:
                    for t in _force_close(eng, drafter, think, ctx, pos, prompt.device):
                        n_out += 1
                        yield t
                        if n_out >= max_new:
                            return
                    pos += 1 + len(think.close_ids)
                    tok = ctx[-1]


def _force_close(eng, drafter, think, ctx, pos, device):
    """Close the reasoning block for the model and take the first token of its answer.

    The forced tokens are run through the engine exactly as generated ones are -- one forward at
    `pos` -- so the KV, the recurrent state and the drafter's context all carry them, and the answer
    that follows is conditioned on a block that really does end where it appears to.
    """
    # `ctx[-1]` is the last committed token and its own forward has not happened yet -- that is the
    # loop's invariant, `len(ctx) == pos + 1`, and it is why the next verify block starts with it.
    # The forced pass has to carry it, or the closing phrase would be written over its position.
    closing = list(think.close_ids)
    forced = [int(ctx[-1])] + closing
    lg = eng.forward(torch.tensor(forced, device=device), start=pos, last_only=True)
    if drafter is not None and hasattr(drafter, "sync"):
        drafter.sync(forced, eng.hidden_post_norm[0], pos)
    if drafter is not None:
        drafter.observe(closing)
    ctx.extend(closing)
    think.observe(closing)
    for t in closing:
        yield t
    nxt = int(lg[0, -1].argmax())
    ctx.append(nxt)
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
        # is exactly the abandoned-mid-block case above.
        store.put(committed, cache.capture(eng, drafter), conv_id)
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


def build_prompt(body: dict) -> tuple[torch.Tensor, str]:
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
        return _as_ids(enc), "chat"
    text = body.get("prompt")
    if isinstance(text, list):
        text = text[0]
    return _as_ids(tok(text or "", return_tensors="pt")), "text"


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
def _chunk(cid: str, model: str, created: int, delta: dict, finish=None, usage=None) -> str:
    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [] if delta is None and finish is None else
            [{"index": 0, "delta": delta or {}, "finish_reason": finish, "logprobs": None}]}
    if usage is not None:
        body["usage"] = usage
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen38-spark-engine"

    def log_message(self, fmt, *args):
        if STATE.get("verbose"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -------------------------------------------------------------- helpers
    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # -------------------------------------------------------------- routes
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/health", "/healthz", "/v1/health"):
            return self._json(200, {"status": "ok"})
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
        except BrokenPipeError:
            return
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            try:
                return self._json(500, {"error": {"message": str(exc),
                                                  "type": "internal_error"}})
            except Exception:
                return

    def _complete(self, body: dict, chat: bool) -> None:
        temperature = float(body.get("temperature") or 0.0)
        if temperature > 0:
            # This engine decodes greedily, and its speculative path is exact only under greedy
            # verification. Serving a sampled answer from a greedy engine would be a quiet lie,
            # so the request is refused rather than silently answered at temperature 0.
            return self._json(400, {"error": {
                "message": "this engine serves greedy decoding only; send temperature=0",
                "type": "invalid_request_error", "param": "temperature"}})
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
        budget = body.get("max_reasoning_tokens")
        if budget is None:
            budget = body.get("thinking_budget")
        if budget is None:
            budget = STATE.get("think_budget") or 0
        budget = int(budget or 0)
        if budget and STATE.get("tree"):
            return self._json(400, {"error": {
                "message": "a reasoning budget needs the chain verify path; restart without --tree",
                "type": "invalid_request_error", "param": "max_reasoning_tokens"}})
        stream = bool(body.get("stream"))
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        model = body.get("model") or STATE["model"]
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        tok = STATE["tok"]

        conv_id = conversation_id(body, self.headers)

        with LOCK:
            prompt, _ = build_prompt(body)
            eos = eos_ids(body)
            n_prompt = int(prompt.numel())
            think = ThinkBudget(tok, budget) if budget else None
            prompt_ids = prompt.tolist()

            # The exact-prompt response cache. Greedy decoding is a function of (prompt, params),
            # so an identical request has an identical answer and this is memoisation rather than
            # an approximation. Under a relaxed accept rule the engine is not answering the
            # greedy question at all, and that is the one setting where the key would be lying
            # about what produced the value -- so the cache is not consulted.
            rcache = STATE.get("response_cache")
            rkey, cached_ids = None, None
            if rcache is not None and not STATE["relax"].on:
                rkey = cache.ResponseCache.key(
                    prompt_ids, max_new=max_new, budget=budget, stops=tuple(stops),
                    eos=tuple(sorted(eos)), tree=bool(STATE.get("tree")))
                cached_ids = rcache.get(rkey)
            source = (iter(list(cached_ids)) if cached_ids is not None
                      else generate_stream(prompt, max_new, eos, think, conv_id))

            if not stream:
                ids = []
                for t in source:
                    ids.append(t)
                if cached_ids is None:
                    _remember(prompt_ids, ids, conv_id)
                    if rkey is not None:
                        rcache.put(rkey, ids, prompt_ids)
                text = tok.decode(ids, skip_special_tokens=True)
                finish = "stop" if (ids and ids[-1] in eos) else "length"
                text, cut = _apply_stops(text, stops)
                if cut:
                    finish = "stop"
                usage = {"prompt_tokens": n_prompt, "completion_tokens": len(ids),
                         "total_tokens": n_prompt + len(ids)}
                if chat:
                    payload = {"id": cid, "object": "chat.completion", "created": created,
                               "model": model, "usage": usage,
                               "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                                            "message": {"role": "assistant", "content": text}}]}
                else:
                    payload = {"id": cid, "object": "text_completion", "created": created,
                               "model": model, "usage": usage,
                               "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                                            "text": text}]}
                return self._json(200, payload)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            w = self.wfile
            if chat:
                w.write(_chunk(cid, model, created, {"role": "assistant", "content": ""}).encode())
                w.flush()
            ids: list[int] = []
            emitted = ""
            finish = "length"
            for t in source:
                ids.append(t)
                if t in eos:
                    finish = "stop"
                    break
                text = tok.decode(ids, skip_special_tokens=True)
                piece = text[len(emitted):]
                if not piece:
                    continue                       # a byte-level token that is not a character yet
                emitted = text
                cut_at = _stop_index(emitted, stops)
                if cut_at is not None:
                    piece = piece[: max(0, cut_at - (len(emitted) - len(piece)))]
                    if piece:
                        w.write((_chunk(cid, model, created, {"content": piece}) if chat
                                 else _text_chunk(cid, model, created, piece)).encode())
                        w.flush()
                    finish = "stop"
                    break
                w.write((_chunk(cid, model, created, {"content": piece}) if chat
                         else _text_chunk(cid, model, created, piece)).encode())
                w.flush()
            if cached_ids is None:
                _remember(prompt_ids, ids, conv_id)
                if rkey is not None:
                    rcache.put(rkey, ids, prompt_ids)
            w.write((_chunk(cid, model, created, {}, finish=finish) if chat
                     else _text_chunk(cid, model, created, "", finish=finish)).encode())
            if want_usage:
                n_out = len(ids)
                w.write(_chunk(cid, model, created, None, usage={
                    "prompt_tokens": n_prompt, "completion_tokens": n_out,
                    "total_tokens": n_prompt + n_out}).encode())
            w.write(b"data: [DONE]\n\n")
            w.flush()


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
    ap.add_argument("--max-len", type=int, default=8192)
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
                    help="pin --drafter lenrouter to one block width (8 or 16). 0 routes. This is "
                         "how the fixed-length baselines are measured through the routed code")
    ap.add_argument("--len-explore", type=int, default=32,
                    help="blocks between forced wide probes when nothing suggests one")
    ap.add_argument("--relax-tau", type=float, default=1.0,
                    help="LOSSY. Accept a drafted token whose probability is at least this fraction "
                         "of the argmax's. 1.0 is the lossless rule and the default")
    ap.add_argument("--relax-rank", type=int, default=1,
                    help="LOSSY. Accept a drafted token among the target's top-r. 1 is lossless")
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
                                   explore_period=a.len_explore, tree=True, ngram=ng)
        else:
            drafter = LengthRouter(small, large, fixed=a.len_fixed,
                                   explore_period=a.len_explore)
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
                 cfg_eos=cfg_eos, think_budget=a.think_budget,
                 reasoning_effort=a.reasoning_effort,
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
    print(f"[server] listening on http://{a.host}:{a.port}  model {a.served_model}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
