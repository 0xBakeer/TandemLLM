# Dashboard

The engine's own web app: a year of token usage (Usage), live and historical speed
(Performance, with the one-second Live panel: every request in flight and the totals), an LM
Studio-style developer view with the live log (Dev), a chat playground
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
src/lib/activity.ts      the words for contract 1.1 (VIS-24): the Now line, activity cells, timelines, stops, stream health
src/views/live-panel.ts  the Live panel (VIS-23): /v1/dashboard/live over SSE, two figures with
                       sparklines, the counts line, one row (desktop) or card (phone) per request
src/lib/live.ts        the five-minute ring of one-second samples, gaps by timestamp, row order (unit-tested)
src/charts/spark.ts    the sparkline under a figure: a glow ribbon for a stream, dots for events
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
| `npm run e2e` | Playwright against `preview:mock` (builds first); `npm run e2e:shots` refreshes `screenshots/` (`live-*.png` come from `e2e/live-shots.spec.ts`) |

### Mock harness

`?mock=<mode>` on the page URL (or `POST /__mock/mode {"mode": …}`) flips the failure switch:
`ok`, `401` (session expired), `expire` (the next data call is a 401), `500` (summary fails),
`empty` (a three-day-old ledger), `slow` (3 s answers), `offline` (connections reset),
`drop[:seconds]` (the log stream closes every N s), `busy` (503 + `Retry-After`, queue 7/8,
15 GB disk, dropped ledger rows), `nospec` (no speculation families in `/metrics`), `nogpu`.
`POST /__mock/request` finishes a request now; `POST /__mock/log {"count": N}` injects lines;
`POST /__mock/live {"scenario": …}` seeds the Live panel: `busy` (one decoding, one prefilling,
one queued and two finished requests plus four minutes of history; `e2e/live.spec.ts`), and the
contract 1.1 scenarios of VIS-24 (`e2e/activity.spec.ts`): `agent-turn` (an opencode-shaped turn:
a chunked 48,210-token prefill with progress, thinking, writing, a `write_file` call whose
arguments grow, finishing, `tool_calls`; then the engine waits for the client for 6 s and the next
request arrives, continuing the first), `abandoned` (a 60,014-token prefill whose client leaves
after 3.6 s), `stops` (one finished request per stop reason in `recent`), `constrained` (a JSON
schema answer and a forced tool call, one of them queued), `loop` (VIS-30, `e2e/layout-shift.spec.ts`:
seven finished opencode turns with long "continues" notes and one turn in flight walking a
single-call prefill, thinking, writing and a `todowrite` call), `clear`. `at: <ms>` starts a scenario
that far in (the screenshots pick a moment that way); `contract: "1.0"` answers like a SRV-34
server; `activity: false` is the kill switch (1.1 with the new fields null); `draining: true`.
While a request is in flight the mock stream sends four events a second, as the engine does.
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

### The Live panel (VIS-23)

The top of `#/performance`. `GET /v1/dashboard/live?follow=1` (SRV-34) sends `event: live` once a
second; the first event carries five minutes of `history`, the later ones the newest `sample`. Two
figures: decode tok/s of everything decoding over the last 2 s, and the latest prefill's tok/s
("prefilling N tokens" while one runs), each with a sparkline of the last five minutes (a ribbon
for decode, dots for prefills, the newest point in orange, gaps where the sampler slept). A line
of counts (in flight, queued, prefilling, decoding, done in the last minute, served, errors,
refused). One row per request: phase (glyph + word, never colour alone), id, client, model and
temperature, prompt (cached), tokens so far, TTFT, prefill tok/s, decode now / average, tokens per
block, elapsed; a finished request stays 30 s with its final numbers, the same numbers its
response's `timings` carried. Cards instead of rows at ≤ 1100 px (≤ 768 px before VIS-30); the
client column hides at ≤ 1280 px. The stream closes while the tab
is hidden. The three 5-minute `/metrics` figures (TTFT p50, tokens per block, acceptance) sit under
the panel. Design: Memo "Live speed panel — design (2026-09-26)".

The panel holds still under its 4 Hz updates (VIS-30, 2026-09-27): both tables use fixed column
widths, every cell is a fixed stack of one-line slots (an ellipsis cuts long text and the `title`
tooltip holds all of it; the continues note and the token split get two lines), a line that comes
and goes is always rendered, blank when empty, and every changing number uses tabular digits.
`e2e/layout-shift.spec.ts` streams the `loop` scenario for 8 s and fails on any box that moves by
more than 1 px, and on Chromium on a layout-shift score of 0.01 or more.

### Live activity (VIS-24, contract 1.1)

What the model is doing right now, from the 1.1 fields of `/v1/dashboard/live` (SRV-37, ENG-114,
SRV-39), at up to four events a second while a request runs. `src/lib/activity.ts` holds the
pure helpers (unit-tested against `tests/fixtures/live-1.1-box.json`, real messages recorded on
the box); `views/live-panel.ts` renders them. Top to bottom:

- **The Now line** above the figures: one glyph and one sentence for the engine, the server's
  `label` as sent ("Prefilling 36,864 of 48,210 (76 %)", "Thinking", "Calling tool write_file",
  "Waiting for client: running tool bash", "Idle", "Draining"), the live numbers beside it
  (prefill tok/s, cached tokens, ETA; decode tok/s and thinking or content tokens; KB of tool
  arguments), the time in that state on the right with the client kind, a thin progress bar while
  a chunked prefill runs, and a red "client disconnected N s ago" when the socket is gone while
  the engine still works. The glyph pulses only while tokens move; never under reduced motion.
- **Under the decode figure**: "4.4 tokens per round · 90.9 ms a round (last second)".
- **Each request**: the phase column becomes an activity cell (`◐ prefilling 76 %`,
  `◍ thinking 611 tok`, `● writing`, `⚒ write_file 18 KB`, `○ queued about 2nd · 5.3 s`,
  `↻ replaying`, `… finishing saving state`, `✓ stop`, `✗ abandoned`); the tokens cell splits into
  thinking · content (· tool); the id carries the red client flag, "continues chatcmpl-… after
  8.4 s (client ran bash)" (with "(inferred)" when the link is a guess) and a `JSON schema` /
  `tool_choice` tag; a timeline strip with one segment per state, width by time, the word inside
  when it fits, and a visually hidden ordered list ("+19.9 s thinking") for screen readers. On a
  phone the strip is that list, cut to the last six transitions.
- **Last 20 requests**: how each one ended, newest first: time, client, the path as a small
  strip, tokens, TTFT, decode tok/s, tool names, and the stop as glyph + word + the server's
  sentence ("abandoned by the client after 73 s of silent prefill, 0 tokens sent"). `abandoned`,
  `error`, `timeout` and `cancelled` get a red ✗ and a red edge, `refused` and `rejected` an
  amber !; never colour alone. A row opens the Dev tab's request detail (`#/dev?request=<id>`).
- **The stream's own health** in the head: "streaming · 4/s", "no update for 3.0 s" (amber past
  2 × interval + 1.5 s, red past 6 s, the numbers dim), "reconnecting since 4 s" with the last
  numbers kept and dimmed, "paused" while the tab is hidden. A silent page is never mistaken for
  an idle engine.
- **A 1.0 server** (no `engine`, no `activity`) gets VIS-23's panel: no Now line, no timeline, no
  Last 20. With `--live-activity off` (1.1, fields null) the Now line shows the engine's label and
  the rows their phase words.
- `#/performance?debug=live` logs every event's arrival to the console and shows the last 20 gaps
  in a small overlay (for the phone soak of OPS-25).

Tones: cobalt for prefilling, a lighter cobalt (`--think`) for thinking, signal orange for writing
and tool calls (orange means now), muted for queued, aqua for a replay. Phone: the Now line wraps
with the numbers on their own line; requests are cards with the activity as the title; the Last 20
rows stack with the sentence first (a 44 px target); no horizontal scroll at 360 px. The
sparklines keep one point a second at four events a second (`push` ignores a repeated `sample.t`).
Screenshots `screenshots/activity{,-prefill,-waiting,-abandoned,-stops}-{desktop,phone}-{dark,light}.png`
(`npm run e2e:shots`). The Playground view is its own chunk, loaded the first time `#/playground`
opens, so the app bundle stays under 60 KB gzip (45 KB after this change). Design: Memo "Live
activity design (2026-09-27)" §7.

## Design notes

Dark-first instrument readout with a light theme (`prefers-color-scheme` until a choice is
remembered). Cobalt is the data colour, signal orange marks now / selection / warnings; the four
token classes (input, cached input, output, reasoning) are cobalt / aqua / orange / violet and the
cache sources cobalt / orange / aqua — the dataviz palette, validated for colour-vision deficiency
and contrast in both themes. Every figure carries its unit; tabular numerals; one y axis per chart;
a legend whenever two series share a plot; every chart has a keyboard-reachable tooltip; the
heatmap cells are a roving-tabindex grid. Responsive from 360 px (bottom tab bar ≤ 768 px) to
2560 px; reduced motion respected.
