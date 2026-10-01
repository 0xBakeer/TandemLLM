"""The two models that answer "which of these is it": arbiter and Qwen itself.

Both answer a `choice` over options `{id: description}` and return an `Answer`; the router's
cascade (`router.py`) does not care which one spoke. Neither is ever awaited past its timeout,
and every failure comes back as an `Answer` with `error` set, never as an exception, so the
cascade fails open by construction.

ARBITER (role `decider`): the Jev contract `POST /v1/systemone` with one `choice` question.
Model-agnostic by arbiter's own hard rule: the limits come from `GET /v1/models` (read at first
use and every 10 minutes; the client's fallbacks 512/192/20 when the server does not say), and
the digest is sized to `context_tokens - question_tokens` at 3.5 characters a token, the estimate
arbiter's own client uses. No checkpoint name or Laya limit is written here.

QWEN (role `classifier`): the engine's own `/v1/chat/completions` on the RAW model name with
`X-Tandem: raw`, thinking off (`enable_thinking: false`), greedy, and
`structured_outputs: {"choice": [ids]}`, so the grammar lets it write only an option id. The
confidence is the probability of the path it wrote, `exp(sum(logprob))` over the answer's tokens,
under the constrained distribution. It is a short request unrelated to the conversation, so the
engine treats it as a guest: the rows it overwrites are stashed and restored (docs/caches.md).
`endpoint: self` means this server's own port. It must be called before the calling request takes
the engine lock, or the two would wait on each other.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request

DEFAULT_LIMITS = {"context_tokens": 512, "question_tokens": 192, "max_options": 20}
CHARS_PER_TOKEN = 3.5
LIMITS_TTL_S = 600.0
USER_AGENT = "tandem-conductor/1"


class Answer:
    __slots__ = ("label", "confidence", "probs", "ms", "error", "model", "stage")

    def __init__(self, stage: str, label=None, confidence=0.0, probs=None, ms=0.0, error=None,
                 model=None):
        self.stage, self.label, self.confidence = stage, label, float(confidence or 0.0)
        self.probs = probs or {}
        self.ms, self.error, self.model = round(float(ms), 2), error, model

    def as_dict(self) -> dict:
        return {"stage": self.stage, "label": self.label, "confidence": round(self.confidence, 4),
                "probs": {k: round(float(v), 4) for k, v in self.probs.items()}, "ms": self.ms,
                "error": self.error, "model": self.model}


def _post(url: str, body: dict, timeout_s: float, headers: dict | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read() or b"{}")


def _get(url: str, timeout_s: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read() or b"{}")


def _why(exc: BaseException) -> str:
    """A failure reason for the journal: timeout, unreachable or http_<code>."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    reason = getattr(exc, "reason", exc)
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError) or \
            "timed out" in str(reason):
        return "timeout"
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        return "unreachable"
    return "error"


class Arbiter:
    """The decider role over HTTP. Safe to share between request threads."""

    def __init__(self, endpoint: str, model: str = "auto", timeout_ms: int = 400,
                 api_key: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model, self.timeout_ms, self.api_key = model or "auto", int(timeout_ms), api_key
        self._limits: dict | None = None
        self._limits_at = 0.0
        self._lock = threading.Lock()

    def limits(self) -> dict:
        """The limits of the model `self.model` resolves to; the fallbacks when not reported."""
        now = time.monotonic()
        if self._limits is not None and now - self._limits_at < LIMITS_TTL_S:
            return self._limits
        with self._lock:
            got = dict(DEFAULT_LIMITS)
            try:
                data = _get(self.endpoint + "/v1/models", max(0.2, self.timeout_ms / 1000))
                want = "jev-latest" if self.model in ("auto", "default", "jev", "") else self.model
                for m in data.get("data") or []:
                    if m.get("id") == want or want in (m.get("aliases") or []):
                        for k, v in (m.get("limits") or {}).items():
                            if k in got and isinstance(v, int) and v > 0:
                                got[k] = v
            except Exception:                                      # noqa: BLE001
                pass
            self._limits, self._limits_at = got, now
            return got

    def state_chars(self) -> int:
        lim = self.limits()
        return max(200, int((lim["context_tokens"] - lim["question_tokens"]) * CHARS_PER_TOKEN))

    def choice(self, state: str, question: str, options: dict[str, str]) -> Answer:
        t0 = time.perf_counter()
        lim = self.limits()
        opts = dict(list(options.items())[:lim["max_options"]])
        body = {"state": state[:self.state_chars()], "model": self.model,
                "questions": {"q": {"type": "choice", "instructions": question,
                                    "criteria": opts}}}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        try:
            r = _post(self.endpoint + "/v1/systemone", body, self.timeout_ms / 1000, headers)
            a = (r.get("answers") or {}).get("q") or {}
            probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
            label = a.get("choice")
            conf = probs.get(label, a.get("confidence", 0.0)) if label else 0.0
            return Answer("arbiter", label, conf, probs, (time.perf_counter() - t0) * 1e3,
                          model=r.get("model"))
        except Exception as exc:                                   # noqa: BLE001
            return Answer("arbiter", ms=(time.perf_counter() - t0) * 1e3,
                          error=f"arbiter_{_why(exc)}")

    def health(self) -> dict:
        t0 = time.perf_counter()
        try:
            r = _get(self.endpoint + "/readyz", max(0.3, self.timeout_ms / 1000))
            return {"up": r.get("status") == "ready", "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "detail": {k: r.get(k) for k in ("engine", "models", "device", "version")},
                    "limits": self.limits()}
        except Exception as exc:                                   # noqa: BLE001
            return {"up": False, "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "error": _why(exc)}


class Qwen:
    """The classifier role: the engine itself on the raw model name, thinking off."""

    def __init__(self, endpoint: str, model: str, timeout_ms: int = 8000):
        self.endpoint = endpoint.rstrip("/")
        self.model, self.timeout_ms = model, int(timeout_ms)

    def choice(self, state: str, prompt: str, options: dict[str, str],
               max_chars: int = 2000) -> Answer:
        t0 = time.perf_counter()
        ids = list(options)
        body = {"model": self.model, "stream": False, "temperature": 0, "max_tokens": 24,
                "messages": [{"role": "system", "content": prompt},
                             {"role": "user", "content": state[:max_chars]}],
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"choice": ids},
                "logprobs": True, "top_logprobs": 3}
        try:
            r = _post(self.endpoint + "/v1/chat/completions", body, self.timeout_ms / 1000,
                      {"X-Tandem": "raw"})
            ch = (r.get("choices") or [{}])[0]
            label = str((ch.get("message") or {}).get("content") or "").strip()
            toks = ((ch.get("logprobs") or {}).get("content")) or []
            if toks:
                conf = math.exp(sum(float(t.get("logprob", 0.0)) for t in toks))
            else:
                conf = 0.5
            if label not in ids:
                return Answer("qwen", None, 0.0, {}, (time.perf_counter() - t0) * 1e3,
                              error="qwen_unknown_label", model=r.get("model"))
            return Answer("qwen", label, conf, {label: conf}, (time.perf_counter() - t0) * 1e3,
                          model=r.get("model"))
        except Exception as exc:                                   # noqa: BLE001
            return Answer("qwen", ms=(time.perf_counter() - t0) * 1e3,
                          error=f"qwen_{_why(exc)}")

    def health(self) -> dict:
        t0 = time.perf_counter()
        try:
            r = _get(self.endpoint + "/v1/models", 2.0)
            names = [m.get("id") for m in r.get("data") or []]
            return {"up": self.model in names, "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "detail": {"models": names}}
        except Exception as exc:                                   # noqa: BLE001
            return {"up": False, "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "error": _why(exc)}
