# Backend kinds — design (T10)

## Where we stand in 0.11

A backend already carries a `kind` (`proxy/config.py` `KINDS`: generic, tandemllm,
vllm, sglang, openai, deepseek; `TANDEM_BACKEND_<ID>_KIND` env, sqlite column), and
`proxy/engines.py` already has per-kind metric TABLES — but only `vllm` and `sglang`
are first-class: `engines.py` TABLES/poller/`figures()`/`parse_info()` know exactly
those two, everything else runs as `generic` (plain relay, no L1 stats, no KV usage).
The relay model itself is "any URL is a backend" (`proxy/relay.py`); autodetect
(`proxy/autodetect.py`) fingerprints ports but its llama.cpp/tabby/ollama hits come
from tie-break probes and PORT_HINTS, and the result is never fed back into backend
kinds.

Note: `ops/engine_status.py` is NOT an engine-kind mechanism — it is the
deploy-hold responder (status file + 503 page). Its name is a red herring for this
task; kind-level "what is loaded" answers come from each engine's own endpoints.

## Goal

Make `llama.cpp`, `tabby` (TabbyAPI) and `ollama` first-class kinds: detected,
capability-flagged, status-queried, error-normalized — without breaking the
generic relay. A backend with kind `generic` or `exl3` still works for every URL
today; kinds only ADD knowledge.

## Detection order (explicit > fingerprint > generic)

1. **Explicit config** — `TANDEM_BACKEND_<ID>_KIND=llama_cpp` (or the row in
   sqlite) wins outright. Never probed, never overridden. Same rule as today.
2. **Fingerprint** — `proxy/kinds.py::detect(url, fetch)` probes, in evidence
   order:
   - `GET /props` → dict with `model_path`/`model_type`/`n_ctx` → `llama_cpp`
     (llama.cpp server's server-state endpoint; an OpenAI-only shim does not
     serve it). `GET /api/version` with `{version}` → `ollama` (its native API
     root). `GET /v1/api/props` → dict with model-info keys → `tabby`.
   - If the existing autodetect already named a family from `/metrics`
     (`tabby_`/`exl3_` prefixes) or tie-breaks, that answer is reused —
     `kinds.detect` refines only the `openai`-shaped leftovers, to avoid
     doubling probes. (Implemented in `autodetect.one()`: known families map
     directly; `openai` gets one refined probe pass.)
3. **Generic** — nothing answered: kind `generic` (alias `exl3/gguf-http` for
   EXL3/GGUF-over-HTTP servers that speak plain OpenAI; same behavior, a label
   the dashboard may show). No probing of unknown ports beyond what autodetect
   already does.

Every probe is timeout-bounded (autodetect's PROBE_TIMEOUT) and fetch-injectable;
a failure means "not that kind", never an error.

## Per-kind capability flags (KindProfile)

| flag | llama_cpp | tabby | ollama | generic |
|---|---|---|---|---|
| `openai_chat` | yes (/v1/chat/completions) | yes (/v1/chat/completions) | yes (/v1/completions+chat) | yes |
| `native_completion` | yes — /completion, non-OpenAI body | no | yes — /api/chat, different envelope | no |
| `tokenizer_stats` | yes — /tokenize, /detokenize | to-verify (no public endpoint known) | no (token counts only inside usage) | no |
| `usage_fields` | prompt_eval_count/eval_count in native; standard in /v1 | standard OpenAI usage | prompt_eval_count, eval_count (native), standard in /v1 | standard |
| `/metrics` | yes, llama.cpp emits Prometheus since mid-2024 (prefix `llama_` — **to-verify**: exact family names vary by build) | yes, `tabby:` prefix (already in autodetect METRIC_FAMILIES) | **no** — no metrics endpoint at all; status from /api/ps | no |
| `engine_status` | /props gives n_ctx (context capacity); KV usage **to-verify** (newer builds expose llama:kv_cache_usage...) | /v1/api/props gives model + max_seq_len; KV usage not public — **to-verify** | /api/ps gives running models, size_vram, context per slot | none |

## engine_status integration ("what is loaded / KV or context usage")

`proxy/kinds.py::status(url, fetch)` per kind, returning
`{"loaded": bool, "models": [...], "ctx_tokens": int|None, "kv_usage": float|None,
"detail": {...raw-lite...}}`:

- **llama_cpp**: `GET /props` → `{model_path, n_ctx, ...}` → loaded + ctx_tokens.
  KV usage: to-verify — if `/metrics` carries a kv-cache family, `kv_usage` maps
  from it, else None.
- **tabby**: `GET /v1/api/props` → model_info (model, org, max_seq_len) →
  loaded + ctx_tokens from max_seq_len. KV usage: not public, to-verify.
- **ollama**: `GET /api/ps` → running slots (`models[].name, size_vram,
  context_window?` — field names to-verify across versions); loaded models list,
  vram-based, no KV fraction. `/api/version` for the version string.
- **vllm / sglang**: left as-is — `engines.py` TABLES already answer this better
  (mapped metric pages, hist deltas); kinds.py only carries their names so the
  registry is complete and callers have one entry point.

## Error-surface normalization

What a 4xx means, per kind, so the proxy can say something true in its own error
frame:

- **llama.cpp**: 404 on /v1/chat/completions with a plain-text body means the
  build predates the OpenAI shim — surface "no OpenAI endpoint, use /completion"
  rather than "model not found". 500 mid-stream with `{"error"}` is the native
  shape; text may be plain, not JSON. 400 on long prompts = context overflow
  (n_ctx), not a malformed request.
- **tabby**: proper OpenAI error JSON (HTTPException bodies); 403 = wrong API
  key (TabbyAPI requires one by default) — map to auth, not to server-down.
  503 = model still loading.
- **ollama**: native /api errors are `{"error": "..."}` with 200 sometimes
  (older builds streamed errors in-band) — normalize to status 500 semantics;
  the OpenAI-compat shim gives real 4xx. "model not found" errors mean the name
  lacks its tag (`llama3` vs `llama3:latest`) — a client fix, not a backend fault.
- **generic**: pass through untouched. The relay keeps its current behavior
  byte-for-byte; normalization is a read-only lens (`kinds.normalize_error(kind,
  code, body) -> (code, message, is_auth, is_overloaded)`), not a rewrite of the
  relay path. Prototype slice does not wire it into relay.py yet.

## What stays generic

- Relay, routing rules, harness switches, secrets, inflight caps, timeouts:
  untouched. A kind changes what we KNOW, not how we forward.
- Unknown/other engines (koboldcpp, exl3 servers, anything): `generic` with the
  existing autodetect family label; no behavior change.
- vllm/sglang stats: still engines.py's job; kinds.py delegates.

## Prototype slice (this commit)

`proxy/kinds.py` (pure, fetch-injectable: fetch(url) -> (code, text)|None,
fetch_json(url) -> obj|None — same seam autodetect already uses), wired into
`autodetect.scan` so `/v1/autodetect` entries gain a `kind` field. Route/relay
behavior unchanged. `tests/test_kinds.py` drives detection precedence, status
extraction and unknown→generic with real-world-shaped fake responses.
