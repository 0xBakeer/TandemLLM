# Dashboard API contract v1

These files are the truth for `/v1/dashboard/*`. Every response the server sends validates against its schema here, and both sides check that. The backend test suite validates real responses, and the dashboard's mock harness (`dashboard/mock/`) validates its fixtures and every mock response (`npm run test:contract` in `dashboard/`). A change to a field bumps `contract_version` and gets a line in the change list below.

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
| `live.schema.json` | `GET /v1/dashboard/live?follow=0`, and the `data:` of one `event: live` on `follow=1` (events after the first omit `history`) |
| `error.schema.json` | every 401, 400, 404 and 429 body |

Each `*.example.json` is a valid instance with consistent numbers. The request example is one finished request: 1,959 prompt tokens of which 1,536 came from the cache, 412 completion tokens of which 230 were reasoning, and a TTFT of 2,240.1 ms (0.4 ms in the queue, 2,239.7 ms of prompt).

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

- v1.0, first version: summary, usage, requests, system, logs, gap, session and error. A log line's `source` is an open set of bracket tags, so the schema types it as a string. `engine.drafter` and `engine.reasoning_effort` are `null` for an engine without a drafter or without a default effort, and `ledger.retention_days` is `null` when the usage ledger is off.
- v1.0, `live` added: the requests in flight and the totals, once a second. A new endpoint and no changed field, so `contract_version` stayed 1.0.
- `live` 1.1, additive: `seq` (also the SSE `id:`), an `engine` block (`idle`, `busy`, `waiting_for_client`, `draining`, `starting`, with KV, memory and store figures), a per-request `activity` (state, label, prefill progress, decode figures, tool, reasoning, client, continues, stop) and `timeline` (the last 16 transitions), `recent` (the last 20 finished requests of the last 15 minutes, each with its stop sentence) and `sampler` (the tick's own cost). `client.kind` gains `opencode` here and in `usage` and `requests`. No 1.0 field changed. With `--live-activity off` the new fields are `null`. `tests/fixtures/live-1.1-box.json` holds real messages recorded from a running engine.
