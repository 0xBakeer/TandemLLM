# Dashboard

The engine's own web app: a year of token usage (Usage), live and historical speed
(Performance), an LM Studio-style developer view with the live log (Dev), a chat playground
against the engine with tools, parameters and the per-response figures (Playground), and the
box's state (System). Served by the engine at `/dashboard/` from the committed `dist/`; no Node
on the box.

Built against the **dashboard API contract v1** — `docs/contract/dashboard-v1/` (schemas +
examples; the prose is the Memo note "Usage & speed metrics — design (2026-09-24)" §3) — first on
the mock harness, then on the real backend (SRV-27..SRV-31). Design: Memo "Dashboard — design
(2026-09-24)"; tickets VIS-2, VIS-13..VIS-18. The Playground: Memo "Dashboard Playground — design
(2026-09-25)"; tickets VIS-19..VIS-22.

## Stack

Vite + TypeScript + Lit (light-DOM components, one stylesheet), hand-drawn SVG charts (glowing
ribbons, stacked bars, scatter, the year heatmap, the timing bar), no CDN, no external fonts,
pinned dependencies. Build: one `index.html`, one hashed JS, one CSS, relative asset paths;
~43 KB gzip total (budget 300 KB, `npm run test:size`).

```
src/app.ts             the shell: session gate, top bar + status pill, rail / bottom tabs, hash routing, theme
src/views/*.ts         usage, performance, dev, playground, system
src/playground/*.ts    the Playground's parts: request builder, stream reducer, params table, tools,
                       presets, markdown, exports (all unit-tested), and the setup column component
src/charts/*.ts        ribbon, bars, scatter, heatmap, timing-bar (+ svg.ts scales/paths)
src/lib/prom.ts        Prometheus text parser + live-strip arithmetic (unit-tested)
src/lib/sse.ts         SSE framing + resumable stream (Last-Event-ID, back-off), OpenAI stream reader
src/lib/heatmap.ts     year grid: 53 × 7, Monday first, quantile levels, streaks
src/lib/time.ts        local days/hours in an IANA zone, DST-safe
src/api/client.ts      same-origin fetch, 401 → sign-in, offline → status pill
src/styles/tokens.css  the design tokens (dark instrument readout + light theme; validated dataviz palette)
src/styles/playground.css  the Playground's stylesheet (the bench, roles, readout, setup column / phone sheet)
mock/                  VIS-13: seeded year of usage, aggregation per the contract, SSE log, /metrics;
                       mock/chat.ts: the scripted chat (thinking, reasoning formats, tools, stops, budgets)
tests/                 vitest (unit + contract + build hygiene)
e2e/                   Playwright (VIS-18), Chromium + WebKit, desktop 1440×900 + phone 390×844
screenshots/           every view, desktop + phone, dark + light (from `npm run e2e:shots`)
```

## Commands (run in `dashboard/`)

| command | what |
|-|-|
| `npm run dev:mock` | the app on <http://localhost:5173/dashboard/> against the mock engine (token `mock`) |
| `npm run build` | type-check + build to `dist/` (commit it: the rsync deploy carries it) |
| `npm run preview:mock` | the built app on :4173 with the mock engine |
| `npm test` | unit tests, contract validation, dist hygiene (131 tests) |
| `npm run test:contract` | every contract example and every mock response against the schemas |
| `npm run test:size` | gzip size of `dist/` against the 300 KB budget |
| `npm run e2e` | Playwright against `preview:mock` (builds first); `npm run e2e:shots` refreshes `screenshots/` |

### Mock harness

`?mock=<mode>` on the page URL (or `POST /__mock/mode {"mode": …}`) flips the failure switch:
`ok`, `401` (session expired), `expire` (the next data call is a 401), `500` (summary fails),
`empty` (a three-day-old ledger), `slow` (3 s answers), `offline` (connections reset),
`drop[:seconds]` (the log stream closes every N s), `busy` (503 + `Retry-After`, queue 7/8,
15 GB disk, dropped ledger rows), `nospec` (no speculation families in `/metrics`), `nogpu`.
`POST /__mock/request` finishes a request now; `POST /__mock/log {"count": N}` injects lines.
The mock engine keeps serving on its own (a request every ~15 s, log lines in between, memory
drifting), so every live surface moves. Nothing from `mock/` is in `dist/` (checked by
`tests/build.test.ts`).

### Against the real engine

```
PW_BASE=http://<engine>:8000/dashboard/ PW_TOKEN=<admin token> npx playwright test
```

Mock-only scenarios (failure switches, injected lines) skip themselves; the fake-engine tier
of VIS-18 (`server/app.py --fake-engine`) and the box tier (`ops/hold.sh`, port 8011) run the
same suites. `VITE_GRAFANA_URL=https://…` at build time enables "Open in Grafana".

### Playground (VIS-19..22)

`#/playground`: a multi-turn streaming chat against the public `POST /v1/chat/completions`
(tagged `X-Requested-With: qse-dashboard`, so the ledger says `dashboard`). The transcript shows
the reasoning as a collapsible block (all three `reasoning_format`s are understood), renders
Markdown, and carries a readout line per assistant turn (TTFT · decode tok/s · tok/block ·
finish). Messages are editable (content and role; "Save & resend" on a user message truncates
what follows), Stop aborts the stream, Regenerate resends. The setup column: system prompt;
parameters with the server's defaults from `/v1/dashboard/system` next to each field and a reset
— **only changed fields are sent**; tools as JSON schema (validated, three templates),
`tool_choice`; presets in `localStorage` (`qse.playground.presets`, import/export as JSON). A
`tool_calls` answer renders as cards with a result box each; "Send results" appends the
`role: "tool"` messages and continues. The readout panel shows the last turn's `usage` /
`timings` / `metrics`; "JSON" shows the last request and response; "Raw request" shows exactly
what the next Send posts (copy as curl). Export the conversation as JSON or Markdown. The draft
(setup + transcript) autosaves under `qse.playground.draft`. On a phone the setup column is a
bottom sheet behind "Setup". Nothing is stored on the server.

Mock chat (`mock/chat.ts`): a user turn with tools → a streamed call to the first (or the named)
tool with argument deltas and `finish_reason: tool_calls`; a `tool` turn → an answer that quotes
the result; otherwise a Markdown answer with a code block; `max_tokens` → `length`; `stop`
strings honoured; `?mock=slow` streams at 150 ms per token (for the Stop scenario);
`?mock=busy` → 503 + `Retry-After`.

## Design notes

Dark-first instrument readout with a light theme (`prefers-color-scheme` until a choice is
remembered). Cobalt is the data colour, signal orange marks now / selection / warnings; the four
token classes (input, cached input, output, reasoning) are cobalt / aqua / orange / violet and the
cache sources cobalt / orange / aqua — the dataviz palette, validated for colour-vision deficiency
and contrast in both themes. Every figure carries its unit; tabular numerals; one y axis per chart;
a legend whenever two series share a plot; every chart has a keyboard-reachable tooltip; the
heatmap cells are a roving-tabindex grid. Responsive from 360 px (bottom tab bar ≤ 768 px) to
2560 px; reduced motion respected.
