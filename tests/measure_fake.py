"""A small in-process engine for tools/measure's CPU tests: /health, /v1/models, /v1/tokenize and a
streamed or plain /v1/completions, with the counts /health gives (requests running and decoding, served)
and a token a few milliseconds apart, so a request of N at once is seen by a 0.2 s poll. A token is a
character (a tokenizer-less server: tools/measure renders prompts as plain text).

    eng = FakeEngine(lanes=8, max_len=65536, at_once_cap=None, mode="raw")
    eng.base ... eng.stop()

`at_once_cap` makes /health report at most that many running and decoding (a server that does not run
N together); `mode` is the X-Tandem header it answers with ("raw": the engine)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeEngine:
    def __init__(self, lanes: int = 8, max_len: int = 65536, at_once_cap: int | None = None, mode: str = "raw",
                 tok_s: float = 800.0, prefill_tok_s: float = 2e6, pack: str = "fake-pack", version: str = "9.9.9"):
        self.lanes, self.max_len, self.cap, self.mode = lanes, max_len, at_once_cap, mode
        self.dt, self.prefill = 1.0 / tok_s, prefill_tok_s
        self.pack, self.version = pack, version
        self.lock = threading.Lock()
        self.running = self.decoding = self.served = 0
        self.paths: list[str] = []
        eng = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj, code=200, headers=None):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                eng.paths.append(self.path)
                if self.path == "/health":
                    return self._json(eng.health())
                if self.path == "/v1/models":
                    return self._json({"data": [{"id": "TandemLLM", "tandem": {"kind": "engine", "label": "Fake Model"}}]})
                self._json({"error": "no"}, 404)

            def do_POST(self):
                eng.paths.append(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/v1/tokenize":
                    text = body.get("text") or body.get("prompt") or ""
                    if body.get("messages"):
                        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                                       for m in body["messages"]) + "<|im_start|>assistant\n"
                    out = {"tokens": len(text)}
                    if body.get("ids"):
                        out["ids"] = [ord(c) % 50000 for c in text]
                    return self._json(out)
                if self.path == "/v1/completions":
                    return eng.complete(self, body)
                self._json({"error": "no"}, 404)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.th = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.th.start()

    def health(self) -> dict:
        with self.lock:
            run, dec = self.running, self.decoding
        if self.cap is not None:
            run, dec = min(run, self.cap), min(dec, self.cap)
        return {"status": "ok", "version": f"{self.version}+{self.pack}", "model": "TandemLLM", "max_len": self.max_len,
                "max_queue": self.lanes, "weights": {"name": self.pack, "drafters": "mtp", "format": "fake"},
                "parallel": {"n": self.lanes, "requested": self.lanes, "max_running": self.lanes, "running": run,
                             "decoding": dec, "slots": [{"max_len": self.max_len}] * self.lanes},
                "inflight": {"running": run, "waiting": 0, "served": self.served},
                "cache": {"response_cache": False, "suffix_store": False, "prefix_cache": True, "prefix_chunk": 4096},
                "memory": {"ceiling_gb": 1.0, "peak_gb": 2.0, "reserved_gb": 0.5, "over_peak_admits": 0}}

    def words(self, seed: int, n: int) -> list[str]:
        h = hashlib.sha256(str(seed).encode()).hexdigest()
        return [f"w{h[i % 64]}" + ("." if i % 9 == 8 else "") + " " for i in range(n)]

    def complete(self, h, body: dict) -> None:
        prompt = body.get("prompt") or ""
        n_out = int(body.get("max_tokens") or 16)
        greedy = float(body.get("temperature", 1.0)) == 0.0
        seed = 7 if greedy else int(body.get("seed") or 0)
        pieces = self.words(seed + len(prompt) % 3, n_out)
        with self.lock:
            self.running += 1
        try:
            # the gate's fake pack asks for a slow prefill in its burst (tools/gate/packs.yaml burst_word)
            time.sleep(0.6 if "FAKE_SLOW_PREFILL" in prompt else min(len(prompt) / self.prefill, 0.5))
            with self.lock:
                self.decoding += 1
            usage = {"prompt_tokens": len(prompt), "completion_tokens": n_out,
                     "prompt_tokens_details": {"cached_tokens": 0}}
            timings = {"draft_n": 0, "predicted_per_second": 1 / self.dt, "prompt_per_second": self.prefill,
                       "tokens_per_block": 1.0, "blocks": n_out}
            rid = "cmpl-" + hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()[:12]
            if not body.get("stream"):
                time.sleep(self.dt * n_out)
                text = "".join(pieces)
                c = {"text": text, "finish_reason": "length"}
                if body.get("logprobs"):
                    c["logprobs"] = {"tokens": pieces, "top_logprobs": [{p: -0.1, "x": -2.0} for p in pieces]}
                h._json({"id": rid, "choices": [c], "usage": usage, "timings": timings},
                        headers={"X-Tandem": self.mode})
                return
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("X-Tandem", self.mode)
            h.send_header("Connection", "close")
            h.end_headers()
            for i, p in enumerate(pieces):
                time.sleep(self.dt)
                ev = {"id": rid, "choices": [{"text": p, "finish_reason": "length" if i == n_out - 1 else None}]}
                h.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                h.wfile.flush()
            h.wfile.write(f"data: {json.dumps({'id': rid, 'choices': [], 'usage': usage, 'timings': timings})}\n\n".encode())
            h.wfile.write(b"data: [DONE]\n\n")
            h.wfile.flush()
            h.close_connection = True
        finally:
            with self.lock:
                self.running -= 1
                self.decoding = max(0, self.decoding - 1)
                self.served += 1

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
