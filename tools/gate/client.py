"""The requests the suites send, on tools/parallel_card.py's client: raw `/v1/completions` with
`X-Tandem: raw` (the engine, not the chat layer), prompts rendered from the model's chat template and
checked id for id against `/v1/tokenize`, and `/health` sampled during every round."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools import parallel_card as pc

REPO = Path(__file__).resolve().parent.parent.parent
PROMPTS = REPO / "bench" / "card-prompts-v1.json"
#: the sampling of the card's rows (parallel_card.py); a fixed seed a request makes them repeatable
CARD_SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
GREEDY = {"temperature": 0.0}
RAW = {"X-Tandem": "raw"}


class GateClient(pc.Client):
    """parallel_card's Client; without the model's tokenizer (the fake engine) a prompt is its
    messages as plain text, which `/v1/completions` reads as it is."""

    def render(self, messages: list, template: bool = False) -> str:
        if self.tok is None:
            return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) \
                + "<|im_start|>assistant\n"
        return super().render(messages, template)

    def complete(self, prompt: str, max_tokens: int, seed: int = 0, logprobs: int | None = None,
                 sampling: dict | None = None, timeout: float = 3600) -> dict:
        """One non-streamed completion: its text, finish reason, usage, and (with `logprobs`) the
        token pieces and the gap between the two best log probabilities at each position."""
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "seed": seed,
                "stream": False, **(sampling or GREEDY), **self.extra}
        if logprobs:
            body["logprobs"] = logprobs
        req = urllib.request.Request(self.base + pc.ENDPOINT, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **self.headers})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return {"error": f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}"}
        except OSError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        c = (out.get("choices") or [{}])[0]
        text = c.get("text") or ""
        rec = {"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "finish": c.get("finish_reason"), "usage": out.get("usage") or {},
               "e2e_s": round(time.monotonic() - t0, 4)}
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


def client(base: str, model: str, tok=None, sampling: dict | None = None) -> GateClient:
    return GateClient(base, model, sampling or CARD_SAMPLING, dict(RAW), tok=tok)


def prompts() -> dict:
    return pc.load_prompts(PROMPTS)


def filler_text(pj: dict, words: int) -> str:
    return " ".join(pc.filler_words(pj, words)[:words])


class HealthPoller(pc.Poller):
    """parallel_card's Poller, with the memory and error counters a suite also reads."""

    def _loop(self):
        while not self.stop.is_set():
            h = pc.health(self.base)
            p, m, inf = h.get("parallel") or {}, h.get("memory") or {}, h.get("inflight") or {}
            d = p.get("decoding")
            self.samples.append({"t": time.monotonic(),
                                 "decoding": d if isinstance(d, int) else len(d or []) if d is not None else None,
                                 "running": p.get("running"),
                                 "reserved_gb": m.get("reserved_gb"), "peak_gb": m.get("peak_gb"),
                                 "over_peak_admits": m.get("over_peak_admits"),
                                 "errors": inf.get("errors"), "abandoned": inf.get("abandoned")})
            self.stop.wait(self.period)

    def __init__(self, base: str, period: float = 0.2):
        super().__init__(base)
        self.period = period

    def memory(self) -> dict:
        res = [s["reserved_gb"] for s in self.samples if s.get("reserved_gb") is not None]
        peak = next((s["peak_gb"] for s in self.samples if s.get("peak_gb") is not None), None)
        over = [s["over_peak_admits"] for s in self.samples if s.get("over_peak_admits") is not None]
        return {"reserved_max_gb": max(res) if res else None, "peak_gb": peak,
                "over_peak_admits": (over[-1] - over[0]) if len(over) >= 2 else 0}


def released_together(cl: GateClient, prompts_: list[str], max_tokens: int, seeds: list[int],
                      poll: float = 0.2) -> dict:
    """`len(prompts_)` streamed requests released at once (a barrier, as parallel_card's rounds),
    with /health sampled meanwhile: each stream's record and the proof summary."""
    n = len(prompts_)
    gate = threading.Barrier(n + 1)
    res: dict = {}
    errs: list = []

    def go(i):
        try:
            gate.wait()
            res[i] = cl.stream(prompts_[i], max_tokens, seeds[i])
        except BaseException as exc:                                  # noqa: BLE001
            errs.append(repr(exc)[:300])

    ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    with HealthPoller(cl.base, poll) as pol:
        gate.wait()
        for t in ths:
            t.join()
    return {"streams": [res[i] for i in sorted(res)], "errors": errs, "proof": pol.summary(),
            "memory": pol.memory()}
