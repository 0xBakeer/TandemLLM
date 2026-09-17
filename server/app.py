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
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

STATE: dict = {}
LOCK = threading.Lock()


# ------------------------------------------------------------------ generation
def generate_stream(prompt: torch.Tensor, max_new: int, eos: set[int]):
    """Yield token ids as they are decided, speculation included.

    Same loop as `engine.spec.generate_spec`, rewritten as a generator so a token reaches the
    socket at the moment it is accepted rather than at the end of the block. A verified block
    produces several tokens at once and they go out together; that is what the engine does, and
    smoothing it would make the inter-token latency a fiction.
    """
    eng = STATE["engine"]
    drafter = STATE["drafter"]
    k = STATE["k"]
    eng.reset()
    ctx = prompt.tolist()
    with torch.no_grad():
        if drafter is not None:
            drafter.reset()
            if hasattr(drafter, "prime"):
                drafter.prime(ctx)
        logits = eng.forward(prompt, start=0, last_only=True)
        if drafter is not None and hasattr(drafter, "sync"):
            drafter.sync(ctx, eng.hidden_post_norm[0], 0)
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
                    lg = eng.forward_tree(block, tree.parents, start=pos)
                    path, new = eng.accept_tree(tree, lg.argmax(-1).tolist())
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
                continue
            block = torch.tensor([tok] + draft, device=prompt.device)
            lg = eng.forward_block(block, start=pos)
            picks = lg.argmax(-1).tolist()
            n = 0
            for i, d in enumerate(draft):
                if picks[i] != d:
                    break
                n += 1
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


def build_prompt(body: dict) -> tuple[torch.Tensor, str]:
    tok = STATE["tok"]
    if "messages" in body:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs.setdefault("enable_thinking", True)
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
        stream = bool(body.get("stream"))
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        model = body.get("model") or STATE["model"]
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        tok = STATE["tok"]

        with LOCK:
            prompt, _ = build_prompt(body)
            eos = eos_ids(body)
            n_prompt = int(prompt.numel())
            if not stream:
                ids = []
                for t in generate_stream(prompt, max_new, eos):
                    ids.append(t)
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
            for t in generate_stream(prompt, max_new, eos):
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
            w.write((_chunk(cid, model, created, {}, finish=finish) if chat
                     else _text_chunk(cid, model, created, "", finish=finish)).encode())
            if want_usage:
                n_out = len(ids)
                w.write(_chunk(cid, model, created, None, usage={
                    "prompt_tokens": n_prompt, "completion_tokens": n_out,
                    "total_tokens": n_prompt + n_out}).encode())
            w.write(b"data: [DONE]\n\n")
            w.flush()


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
                    choices=("mtp", "router", "merged", "dflash2", "none"))
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
    ap.add_argument("--dflash2-blocks", type=int, default=1,
                    help="chained 8-wide draft blocks; 1 proposes 7 tokens, 2 proposes 14")
    ap.add_argument("--dflash2-path", default="greedy", choices=("greedy", "viterbi"))
    ap.add_argument("--dflash2-ckpt", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    t0 = time.time()
    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=a.drafter in ("none", "dflash2", "merged"), nvfp4=a.nvfp4)
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
    gen_cfg = os.path.join(cfg.path, "generation_config.json")
    cfg_eos = None
    if os.path.isfile(gen_cfg):
        cfg_eos = json.load(open(gen_cfg)).get("eos_token_id")
    STATE.update(engine=eng, tok=tok, drafter=drafter, k=a.k or a.depth, device="cuda",
                 tree=bool(a.tree),
                 model=a.served_model, started=int(time.time()), verbose=a.verbose,
                 cfg_eos=cfg_eos)
    print(f"[server] {w.report()}")
    print(f"[server] drafter={a.drafter} depth={a.depth} k={STATE['k']} "
          f"nvfp4={w.nvfp4_source or 'off'}  loaded in {time.time() - t0:.1f}s")
    # one warm request, so the first measured one is not paying for Triton autotuning
    with torch.no_grad():
        list(generate_stream(tok("warm up the kernels", return_tensors="pt").input_ids[0].cuda(),
                             4, set()))
    print(f"[server] listening on http://{a.host}:{a.port}  model {a.served_model}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
