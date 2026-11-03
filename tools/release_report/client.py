"""The report's requests: raw `/v1/completions` only (`X-Tandem: raw`), with the prompt rendered from
the model's chat template on the client (tools/parallel_card.py `Client.render`, checked id for id
against `/v1/tokenize`). Every response's `X-Tandem` header is read back: a response the chat layer
served is marked, and its row is INVALID.

A stream is recorded event by event (time and text), with the response's id, its usage and the
engine's own `timings` from the usage chunk."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request

from tools import parallel_card as pc
from tools.gate.client import GateClient

#: the only generation endpoint the report calls
ENDPOINT = "/v1/completions"
RAW_HEADERS = {"X-Tandem": "raw"}
GREEDY = {"temperature": 0.0}


def raw_served(headers) -> tuple[bool, str]:
    """Whether a response says it was served raw: `X-Tandem: raw`, the mode the request ran in (the
    server sends it on every completion so a bench client can assert it, docs/server.md). A conductor
    that resolved a profile names it in `X-Tandem-Profile` beside a raw mode; the mode decides."""
    mode = (headers.get("X-Tandem") or "").strip().lower() if headers is not None else ""
    prof = (headers.get("X-Tandem-Profile") or "").strip() if headers is not None else ""
    why = f"X-Tandem: {mode or 'missing'}" + (f", X-Tandem-Profile: {prof}" if prof else "")
    return mode == "raw", why


def _post(url: str, body: dict, timeout: float = 120.0, tries: int = 6) -> dict:
    """A JSON POST; HTTP 429 (the tokenizer's rate limit) waits and tries again."""
    for k in range(tries):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or k == tries - 1:
                raise
            time.sleep(10.0)
    raise RuntimeError("unreachable")


class ReportClient(GateClient):
    """parallel_card's client (the template rendering and its id check) with recorded streams."""

    def __init__(self, base: str, model: str, tok=None):
        super().__init__(base, model, dict(GREEDY), dict(RAW_HEADERS), tok=tok)
        self.calls: list[str] = []            # every endpoint called, for the checks

    def _ids(self, body: dict) -> list:
        self.calls.append("/v1/tokenize")
        return [int(t) for t in _post(self.base + "/v1/tokenize", {**body, "ids": True})["ids"]]

    def count(self, text: str) -> int:
        """Tokens of a text as `/v1/completions` reads it: the model's tokenizer here, else the server."""
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=False)["input_ids"]) + len(self._lead or [])
        self.calls.append("/v1/tokenize")
        return int(_post(self.base + "/v1/tokenize", {"text": text})["tokens"])

    def piece_tokens(self, piece: str) -> int:
        if self.tok is not None:
            return len(self.tok(piece, add_special_tokens=False)["input_ids"])
        return max(1, len(piece))

    def _body(self, prompt: str, max_tokens: int, stream: bool, seed: int, sampling: dict | None,
              cache_prompt: bool, logprobs: int | None, logit_bias: dict | None) -> dict:
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "stream": stream,
                "seed": seed, "cache_prompt": cache_prompt, **(sampling or GREEDY)}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if logprobs:
            body["logprobs"] = logprobs
        if logit_bias:
            body["logit_bias"] = logit_bias
        return body

    def stream_rec(self, prompt: str, max_tokens: int, seed: int = 0, sampling: dict | None = None,
                   cache_prompt: bool = False, logit_bias: dict | None = None,
                   stop: threading.Event | None = None, t_send: float | None = None,
                   timeout: float = 7200) -> dict:
        """One streamed raw completion: its id, the raw check, every text event's time, usage and the
        engine's timings. `stop` set ends the read (the client leaves)."""
        body = self._body(prompt, max_tokens, True, seed, sampling, cache_prompt, None, logit_bias)
        req = urllib.request.Request(self.base + ENDPOINT, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **self.headers})
        self.calls.append(ENDPOINT)
        t0 = time.monotonic() if t_send is None else t_send
        rec: dict = {"id": None, "raw": False, "raw_why": "no response", "events": [], "usage": {},
                     "timings": {}, "finish": None, "error": None, "t_send": t0, "left": False}
        text: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rec["raw"], rec["raw_why"] = raw_served(r.headers)
                for line in r:
                    if stop is not None and stop.is_set():
                        rec["left"] = True
                        break
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    rec["id"] = rec["id"] or ev.get("id")
                    if isinstance(ev.get("usage"), dict):
                        rec["usage"] = ev["usage"]
                    if isinstance(ev.get("timings"), dict):
                        rec["timings"] = ev["timings"]
                    if isinstance(ev.get("error"), dict):
                        rec["error"] = str(ev["error"].get("message"))[:300]
                    for c in ev.get("choices") or []:
                        if c.get("finish_reason") is not None:
                            rec["finish"] = c["finish_reason"]
                        piece = c.get("text") if isinstance(c.get("text"), str) else ""
                        if piece:
                            rec["events"].append((round(time.monotonic() - t0, 4), piece))
                            text.append(piece)
        except urllib.error.HTTPError as exc:
            rec["error"] = f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}"
        except OSError as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["e2e_s"] = round(time.monotonic() - t0, 4)
        rec["text"] = "".join(text)
        return summarize(rec, self)

    def complete_rec(self, prompt: str, max_tokens: int, seed: int = 0, logprobs: int | None = None,
                     cache_prompt: bool = False, timeout: float = 7200) -> dict:
        """One raw completion without a stream: text, finish, usage, timings, the raw check, and with
        `logprobs` the token pieces and the top-2 gap at each position."""
        body = self._body(prompt, max_tokens, False, seed, GREEDY, cache_prompt, logprobs, None)
        req = urllib.request.Request(self.base + ENDPOINT, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **self.headers})
        self.calls.append(ENDPOINT)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw, why = raw_served(r.headers)
                out = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return {"error": f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}",
                    "raw": False, "raw_why": "error"}
        except OSError as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "raw": False, "raw_why": "error"}
        c = (out.get("choices") or [{}])[0]
        text = c.get("text") or ""
        rec = {"id": out.get("id"), "raw": raw, "raw_why": why, "text": text,
               "sha256": hashlib.sha256(text.encode()).hexdigest(), "finish": c.get("finish_reason"),
               "usage": out.get("usage") or {}, "timings": out.get("timings") or {},
               "e2e_s": round(time.monotonic() - t0, 3)}
        lp = c.get("logprobs") or {}
        if lp.get("tokens") is not None:
            rec["pieces"] = list(lp["tokens"])
            gaps = {}
            for i, top in enumerate(lp.get("top_logprobs") or []):
                v = sorted((top or {}).values(), reverse=True)
                if len(v) >= 2:
                    gaps[str(i)] = round(v[0] - v[1], 5)
            rec["gaps"] = gaps
        return rec


def summarize(rec: dict, cl: ReportClient) -> dict:
    """A stream's numbers: TTFT, the client's decode rate (completion tokens after the first over the
    first to the last text), the prefill rate, each event's tokens, the best 1 s window, and the
    engine's own drafting counts."""
    u, t = rec["usage"], rec["timings"]
    comp = int(u.get("completion_tokens") or 0)
    ev = rec["events"]
    first = ev[0][0] if ev else None
    last = ev[-1][0] if ev else None
    rec["prompt_tokens"] = u.get("prompt_tokens")
    rec["completion_tokens"] = comp
    rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or t.get("cache_n")
    rec["ttft_s"] = first
    win = (last - first) if first is not None and last is not None else 0.0
    rec["decode_tok_s"] = round((comp - 1) / win, 2) if comp > 1 and win > 0 else None
    rec["prefill_tok_s"] = round(rec["prompt_tokens"] / first, 1) if first and rec["prompt_tokens"] else None
    # tokens per event: the pieces' own counts, scaled to the usage's completion tokens
    counts = [cl.piece_tokens(p) for _, p in ev]
    tot = sum(counts)
    scale = (comp / tot) if tot and comp else 1.0
    rec["ev_tokens"] = [round(c * scale, 3) for c in counts]
    rec["peak_1s_tok_s"] = peak_window(ev, rec["ev_tokens"], 1.0)
    draft_n = t.get("draft_n")
    rec["engine"] = {"blocks": t.get("blocks"), "tokens_per_block": t.get("tokens_per_block"),
                     "draft_n": draft_n, "draft_n_accepted": t.get("draft_n_accepted"),
                     "drafting": (draft_n > 0) if isinstance(draft_n, (int, float)) else None,
                     "decode_tok_s": t.get("predicted_per_second"),
                     "prefill_tok_s": t.get("prompt_per_second"), "queue_ms": t.get("queue_ms"),
                     "cache_n": t.get("cache_n")}
    return rec


def peak_window(events: list, tokens: list, width: float) -> float | None:
    """The most tokens in any `width`-second window after the first token, as tok/s (None for an
    answer shorter than the window). The first event's tokens are the prefill's, not counted."""
    if len(events) < 3 or events[-1][0] - events[0][0] < width:
        return None
    ts = [t for t, _ in events]
    best, j, acc = 0.0, 1, 0.0
    for i in range(1, len(ts)):
        acc += tokens[i]
        while ts[i] - ts[j] > width:
            acc -= tokens[j]
            j += 1
        best = max(best, acc)
    return round(best / width, 1)


def tokens_between(rec: dict, t_a: float, t_b: float) -> float:
    """A stream's tokens with event times in [t_a, t_b) (absolute monotonic times)."""
    t0 = rec["t_send"]
    return sum(n for (t, _), n in zip(rec["events"], rec["ev_tokens"]) if t_a <= t0 + t < t_b)


def health(base: str) -> dict:
    return pc.health(base)


def get_json(url: str, timeout: float = 10.0) -> dict:
    try:
        return pc._get(url, timeout)
    except Exception as exc:                                         # noqa: BLE001
        return {"error": str(exc)}


def busy(h: dict) -> str | None:
    """Why the server is busy now (requests running or waiting), or None."""
    inf, p = h.get("inflight") or {}, h.get("parallel") or {}
    dec = p.get("decoding")
    n_dec = dec if isinstance(dec, int) else len(dec or [])
    parts = [f"{k} {v}" for k, v in (("running", inf.get("running") or p.get("running")),
                                     ("waiting", inf.get("waiting")), ("decoding", n_dec)) if v]
    return ", ".join(parts) or None


def end_ids(tok) -> list[int]:
    from tools.gate.suites.api import end_ids as _e
    return _e(tok)


class Proof(pc.Poller):
    """/health every `period` s during a workload: slots decoding at once, requests running and waiting."""

    def __init__(self, base: str, period: float = 0.2):
        super().__init__(base)
        self.period = period

    def _loop(self):
        while not self.stop.is_set():
            h = pc.health(self.base)
            p, inf = h.get("parallel") or {}, h.get("inflight") or {}
            d = p.get("decoding")
            self.samples.append({"decoding": d if isinstance(d, int) else len(d or []) if d is not None else None,
                                 "running": p.get("running", inf.get("running")),
                                 "waiting": inf.get("waiting")})
            self.stop.wait(self.period)

    def summary(self) -> dict:
        s = super().summary()
        w = [x["waiting"] for x in self.samples if x.get("waiting") is not None]
        s["max_waiting"] = max(w) if w else 0
        return s


def together(cl: ReportClient, prompts: list[str], max_tokens: int, seeds: list[int] | None = None,
             cache_prompt: bool = False, logit_bias: dict | None = None, period: float = 0.2) -> dict:
    """`len(prompts)` streams released at once (a barrier), /health sampled meanwhile."""
    n = len(prompts)
    seeds = seeds or [100 + i for i in range(n)]
    gate = threading.Barrier(n + 1)
    res: dict = {}

    def go(i):
        gate.wait()
        res[i] = cl.stream_rec(prompts[i], max_tokens, seeds[i], cache_prompt=cache_prompt,
                               logit_bias=logit_bias, t_send=t_go[0])
    t_go = [0.0]
    ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    with Proof(cl.base, period) as pol:
        t_go[0] = time.monotonic()
        gate.wait()
        for t in ths:
            t.join()
    streams = [res[i] for i in range(n) if i in res]
    return {"streams": streams, "proof": pol.summary(), "t0": t_go[0],
            "wall_s": round(max((s["t_send"] + s["e2e_s"] for s in streams), default=t_go[0]) - t_go[0], 3)}
