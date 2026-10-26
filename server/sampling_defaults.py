"""SPD-245: the filters a sampled request runs with when it names none.

A request that sends `temperature` alone used to sample the whole vocabulary (top_p 1.0, top_k 0),
which is not what the checkpoint was tuned for: its card (generation_config.json, read once at start
into `STATE["card_sampling"]`, server/conductor/loopescape.py `card_sampling`) recommends a top_k and
top_p (Qwen3.8-27B and its fine-tunes: top_k 20, top_p 0.95). The order, first one that says:

  1. the request: any of top_p / top_k / min_p in the body, and the fields it leaves out take the
     server's values below (the client chose its filters);
  2. the server: `--top-p` / `--top-k` given at start (ops/start.sh TOP_P / TOP_K);
  3. the card: its top_p and top_k, min_p 0;
  4. none: top_p 1.0, top_k 0, min_p 0 (the whole vocabulary).

Only the filters move; the temperature is the request's or the server's as before, a greedy
request is untouched, and the logits are not touched at all.
"""

from __future__ import annotations

FIELDS = ("top_p", "top_k", "min_p")
PLAIN = {"top_p": 1.0, "top_k": 0, "min_p": 0.0}


def _num(key: str, v):
    return int(v) if key == "top_k" else float(v)


def resolve(body: dict | None, server: dict | None = None, card: dict | None = None) -> dict:
    """{top_p, top_k, min_p, source} for one request. `server`: the start flags that were given
    ({field: value}, a field left out or None was not given); `card`: `STATE["card_sampling"]` or None.
    `source`: "request", "server", "card" or "none". Raises ValueError / TypeError on a bad field
    (the handler answers 400)."""
    body = body or {}
    server = {k: v for k, v in (server or {}).items() if k in FIELDS and v is not None}
    sent = {k: body[k] for k in FIELDS if body.get(k) is not None}
    if sent:
        base = dict(PLAIN, **server)
        out = {k: _num(k, sent.get(k, base[k])) for k in FIELDS}
        out["source"] = "request"
        return out
    if server:
        out = {k: _num(k, server.get(k, PLAIN[k])) for k in FIELDS}
        out["source"] = "server"
        return out
    if card:
        return {"top_p": float(card.get("top_p", 1.0)), "top_k": int(card.get("top_k", 0) or 0),
                "min_p": 0.0, "source": "card"}
    return dict(PLAIN, source="none")
