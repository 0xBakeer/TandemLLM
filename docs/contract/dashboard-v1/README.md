# Dashboard API contract v1

These files are the truth for `/v1/dashboard/*`. Every response the server sends validates against its schema here, and both sides check that. The backend test suite validates real responses, and the dashboard's mock engine (`dashboard/mock/`) validates its fixtures and every mock response (`npm run test:contract` in `dashboard/`). A change to a field bumps `contract_version` and gets a line in the change list below.

| file | endpoint |
|-|-|
| `summary.schema.json` | `GET /v1/dashboard/summary?tz=` |
| `usage.schema.json` | `GET /v1/dashboard/usage?from=&to=&bucket=day\|hour&tz=&model=&client=` |
| `requests.schema.json` | `GET /v1/dashboard/requests?limit=&before=&model=&client=&finish=` |
| `system.schema.json` | `GET /v1/dashboard/system` |
| `logs-line.schema.json` | the `data:` payload of one `event: log` on `GET /v1/dashboard/logs?follow=1` |
| `logs-json.schema.json` | `GET /v1/dashboard/logs?follow=0` |
| `gap.schema.json` | the `data:` payload of one `event: gap` (lines a slow subscriber missed) |
| `session.schema.json` | `GET /v1/dashboard/session` (200 body) |
| (none: Prometheus text) | `GET /v1/dashboard/metrics`, the `/metrics` page under the dashboard's access rule |
| `live.schema.json` | `GET /v1/dashboard/live?follow=0`, and the `data:` of one `event: live` on `follow=1` (events after the first omit `history`) |
| `error.schema.json` | every 401, 400, 404 and 429 body |
| `tandem-eval.schema.json` | not an endpoint: the report `tools/tandem_eval.py` writes (`~/pub-results/tandem-eval/<night>/report.json`) and keeps on the box for the Conductor view's Compare screen (`POST /v1/conductor/evals`, listed by `GET /v1/conductor/evals`) |

Each `*.example.json` is a valid instance with consistent numbers (`tandem-eval.example.json` is made up, not a measurement; its `harness` says so). The request example is one finished request: 1,959 prompt tokens of which 1,536 came from the cache, 412 completion tokens of which 230 were reasoning, and a TTFT of 2,240.1 ms (0.4 ms in the queue, 2,239.7 ms of prompt).

The schemas use the JSON Schema 2020-12 subset that the backend's standard-library validator supports: `type`, `required`, `properties`, `additionalProperties`, `items`, `enum`, nullable through type arrays, and `minimum`. `additionalProperties` is `false` everywhere except `flags.args` and `caches`. Those two are the engine's own objects, the parsed command line and the `/v1/cache/stats` body.

## Rules the schemas cannot express

Both sides test these instead.

- `usage.buckets` is dense: one bucket per local day (or hour) of the range, zero-filled, with `null` percentiles. `start` is RFC 3339 with the zone's own offset, so its first ten characters are the local date.
- Percentiles are exact over the rows in range. Requests without a decode (errors, refusals, response-cache replays) are left out of the speed percentiles and counted in the token sums.
- `top_days` is the top 10 days by `total_tokens`, for day buckets only.
- `bucket=hour` allows at most 31 days and answers 400 above that.
- The log stream sends `event: log` with `id: <seq>`, a `: ping` comment every 15 s, and `event: gap` with `{"dropped": N}` for a slow subscriber. It resumes from `Last-Event-ID` or `since` and takes at most 4 subscribers.
- The live stream sends `event: live` after every sample, the first one with `history` (up to 300 one-second samples). It takes at most 4 subscribers and drops a closed reader within a second. While a request runs and a stream is open it sends `QSE_LIVE_HZ` events a second (default 4), otherwise one a second, and a `: ping` every 15 s. The server encodes each tick once for all open streams.

## Changes

- v1.0, first version: nine schemas, every one except `live`. A log line's `source` is an open set of bracket tags, so the schema types it as a string. `engine.drafter` and `engine.reasoning_effort` are `null` for an engine without a drafter or without a default effort, and `ledger.retention_days` is `null` when the usage ledger is off.
- v1.0, `live` added: the requests in flight and the totals, once a second. A new endpoint and no changed field, so `contract_version` stayed 1.0.
- `live` 1.1, additive: `seq` (also the SSE `id:`), an `engine` block (`idle`, `busy`, `waiting_for_client`, `draining`, `starting`, with KV, memory and store figures), a per-request `activity` (state, label, prefill progress, decode figures, tool, reasoning, client, continues, stop) and `timeline` (the last 16 transitions), `recent` (the last 20 finished requests of the last 15 minutes, each with its stop sentence) and `sampler` (the tick's own cost). `client.kind` gains `opencode` here and in `usage` and `requests`. No 1.0 field changed. With `--live-activity off` the new fields are `null`. `tests/fixtures/live-1.1-box.json` holds real messages recorded from a running engine.
- `live` 1.1, VIS-40, additive: the live rates without the warm-up. `windows` reports the settings (`QSE_LIVE_DECODE_WARMUP_TOKENS` 50, `QSE_LIVE_PREFILL_WARMUP_CHUNKS` 1, `QSE_LIVE_NOW_WINDOW_S` 2, `QSE_LIVE_MOSTLY_CACHED_PCT` 10, `QSE_LIVE_MOSTLY_CACHED_TOKENS` 4096). `now` gains `prefill_live_tps` and `prefill_live_count` (the running prefills' rate after their first chunk, over the chunk marks of the last 2 s), `decode_live_tps` and `decode_warming` (decode from each request's 50th token). A sample gains `decode_live_tps` and `prefill_live_tps`, a request `decode_live_now_tps`, `decode_live_avg_tps`, `decode_warmup` and `mostly_cached`, and `activity.prefill` gains `tps_live`, `tps_live_avg` and `warmup`. Every earlier field keeps its meaning: `now.prefill_tps` is still the last finished request's end-to-end rate, and `timings`, `usage`, the ledger and `/metrics` do not change. The new fields are optional in the schema, so an older server's messages still validate, and the dashboard falls back to the figures it had.
- `live` 1.2, SRV-115, additive: `activity.conductor`, the session monitor on a request it watches (`mode`, `acts`, `hidden_tokens`, `gave_up`, `supervising` while a hidden forward runs and for a second after it, the `action` it runs, the newest decision `last` and action `last_action` with how long ago), null for any other request. The activity's `state` keeps its enum; its `label` says "Supervising: E1 continuing an empty turn (...)" meanwhile. A 1.1 reader ignores the field; the schema accepts both versions, so 1.1 messages still validate.
- `session` 1.1: the body gains `contract_version` and `login`. `login` is `false` when the server runs with the dashboard login off (`--dashboard-login off`, the default); then `expires_at` is `null`, there is no sign-in or sign-out (`POST` and `DELETE` answer 404), and the page opens without its token screen. With the login on, the body is the 1.0 body plus the two fields. The other endpoints stay at their versions.
- `GET /v1/dashboard/metrics` added: the same page as `/metrics`, under the dashboard's access rule, so the Performance view's 5-minute figures work with the login off. `/metrics` keeps its own rule. A new endpoint and no changed field.
- `tandem-eval` 1.0, OPS-43: the eval harness's report. Per run the gates and the minute of each PASS, tokens, the cached share, empty stops, the continuations the watch-only driver would have sent, the engine's requests and the conductor's acts; per suite and arm the `rows` the Compare screen reads (`arm` is `raw` or `tandem`, `indicative` below n = 10); per suite the `comparison` (pass rate not lower, minutes to PASS not higher); the raw engine's never-worse row for the build and the `verdict`.
