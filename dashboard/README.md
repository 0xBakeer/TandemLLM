# Dashboard

The engine's own web app, served by the engine at `/dashboard/` from the committed `dist/`. The box needs no Node. [docs/dashboard.md](../docs/dashboard.md) describes what the views show. This page is for working on the app.

The app is built against the dashboard API contract in `docs/contract/dashboard-v1/` (schemas and examples). Every mock response validates against it, and so does every real response in the backend's tests.

## Stack

TypeScript with Lit components (light DOM, one stylesheet), built by Vite. Charts are hand-drawn SVG. Dependencies are pinned, and nothing loads from a CDN or a font service. The build is one `index.html`, one hashed JS file, one CSS file and the Playground as a chunk loaded on first use, all with relative paths. The part every tab loads is about 45 KB gzip, against a 300 KB budget (`npm run test:size`).

```
src/app.ts               the shell: session gate, top bar and status pill, rail or bottom tabs, hash routing, theme
src/views/*.ts           usage, performance, dev, playground, system
src/views/live-panel.ts  the Live panel: /v1/dashboard/live over SSE
src/lib/activity.ts      words and numbers for the live activity (unit-tested against recorded messages)
src/lib/live.ts          the five-minute ring of one-second samples, gaps by timestamp, row order
src/lib/prom.ts          Prometheus text parser and live-strip arithmetic
src/lib/sse.ts           SSE framing, resumable streams (Last-Event-ID, back-off), the OpenAI stream reader
src/lib/heatmap.ts       the year grid: 53 by 7, Monday first, quantile levels, streaks
src/lib/time.ts          local days and hours in an IANA zone, safe across DST
src/playground/*.ts      request builder, stream reducer, parameters, tools, presets, markdown, exports
src/charts/*.ts          ribbon, bars, scatter, heatmap, timing bar, sparkline
src/api/client.ts        same-origin fetch; a 401 opens sign-in, going offline shows in the status pill
src/styles/tokens.css    design tokens: dark instrument readout and a light theme
mock/                    the mock engine: a seeded year of usage, the contract's aggregation, SSE, /metrics, a scripted chat
tests/                   vitest: unit, contract and build checks
e2e/                     Playwright: Chromium and WebKit, desktop 1440 by 900 and phone 390 by 844
```

## Commands

Run these in `dashboard/`.

| command | what |
|-|-|
| `npm run dev:mock` | the app on `http://localhost:5173/dashboard/` against the mock engine (token `mock`) |
| `npm run build` | type-check and build `dist/`; commit the result, because the server serves it |
| `npm run preview:mock` | the built app on port 4173 with the mock engine |
| `npm test` | unit tests, contract validation and build checks (181 tests) |
| `npm run test:contract` | every contract example and every mock response against the schemas |
| `npm run test:size` | gzip size of `dist/` against the budget |
| `npm run e2e` | Playwright against `preview:mock` (builds first) |
| `npm run e2e:shots` | screenshots of every view into `screenshots/` (not versioned) |

## The mock engine

`?mock=<mode>` on the page URL, or `POST /__mock/mode {"mode": ...}`, switches the failure mode:

| mode | what the mock does |
|-|-|
| `ok` | normal service |
| `401`, `expire` | the session is gone, now or on the next data call |
| `500` | the summary fails |
| `empty` | a ledger only three days old |
| `slow` | 3 s answers |
| `offline` | connections reset |
| `drop[:seconds]` | the log stream closes every N seconds |
| `busy` | 503 with `Retry-After`, a queue of 7 of 8, a nearly full disk, dropped ledger rows |
| `nospec`, `nogpu` | no speculation metrics, no GPU figures |

`POST /__mock/request` finishes a request now, and `POST /__mock/log {"count": N}` injects log lines.

`POST /__mock/live {"scenario": ...}` seeds the Live panel:

| scenario | what it plays |
|-|-|
| `busy` | one request decoding, one prefilling, one queued, two finished, four minutes of history |
| `agent-turn` | an agent turn: a chunked 48,210-token prefill, thinking, writing, a `write_file` call whose arguments grow, then the wait for the client and the next request |
| `abandoned` | a 60,014-token prefill whose client leaves after 3.6 s |
| `stops` | one finished request per stop reason |
| `constrained` | a JSON-schema answer and a forced tool call |
| `loop` | seven finished agent turns and one in flight, for the layout-shift test |
| `clear` | nothing |

`at: <ms>` starts a scenario that far in. `contract: "1.0"` answers like an older server without the activity fields, `activity: false` sends them as `null`, and `draining: true` plays a server that is shutting down. While a request is in flight the mock sends four events a second, as the engine does. The mock also serves on its own (a request about every 15 s, log lines between them, memory drifting), so every live surface moves. Nothing from `mock/` ends up in `dist/`, and `tests/build.test.ts` checks that.

## Against a real engine

    PW_BASE=http://<engine>:8000/dashboard/ PW_TOKEN=<admin token> npx playwright test

Mock-only scenarios skip themselves. The same suites run against `server/app.py --fake-engine`, a real server with a scripted engine and no GPU. `VITE_GRAFANA_URL=<your Grafana URL>` at build time turns on the "Open in Grafana" links.

## The Playground

`#/playground` is a multi-turn streaming chat against the engine's public `POST /v1/chat/completions`. Its requests carry `X-Requested-With: qse-dashboard`, so the usage ledger files them under the dashboard.

- The transcript folds the reasoning into a block (it understands all three reasoning formats), renders Markdown, and shows a line per answer: time to first token, decode tok/s, tokens per round and the finish reason.
- Messages can be edited. "Save & resend" on a user message drops what follows it. Stop aborts the stream, and Regenerate sends again.
- The setup column holds the system prompt, the parameters with the server's defaults beside each field, the tools as JSON schema with three templates, `tool_choice`, and presets. Only changed fields are sent.
- A `tool_calls` answer shows one card per call, each with a box for the result. "Send results" appends the `role: "tool"` messages and continues.
- The readout shows the last answer's `usage`, `timings` and `metrics`. "Raw request" shows exactly what the next Send will post, and can copy it as curl.
- A conversation exports as JSON or Markdown. The draft and the presets stay in the browser's `localStorage`, and nothing is stored on the server.
- On a phone the setup column is a bottom sheet.

## The Live panel

The top of `#/performance`. It reads `GET /v1/dashboard/live?follow=1`. The first event carries five minutes of history, and later events carry the newest sample.

- A "Now" line with the server's label as sent ("Prefilling 36,864 of 48,210 (76 %)", "Calling tool write_file", "Waiting for client: running tool bash"), the live numbers beside it, the time in that state, a progress bar while a chunked prefill runs, and a red note when the client's socket is gone.
- Two figures, decode and prefill tok/s, each with a five-minute sparkline. Under the decode figure: tokens per round and milliseconds per round over the last second.
- One row per request (a card at 1100 px wide or less): an activity cell (for example "⚒ write_file 18 KB"), tokens split into thinking and content, a "continues" note for agent turns, and a timeline strip with one segment per state. A screen reader gets the timeline as an ordered list.
- The last 20 requests with the server's stop sentence. A bad stop gets a red mark and a refusal an amber one, always with a word, never color alone. A row opens the Dev view's request detail.
- The health of the stream itself: "streaming · 4/s", "no update for 3.0 s", "reconnecting since 4 s", with the last numbers kept and dimmed.

The panel holds still while it updates four times a second. Tables have fixed column widths, every cell is a fixed stack of one-line slots, a line that comes and goes is always rendered, and changing numbers use tabular digits. `e2e/layout-shift.spec.ts` streams the `loop` scenario for 8 s and fails if any box moves by more than 1 px.

## Design notes

The app is dark first, with a light theme that follows `prefers-color-scheme` until the user picks one. Cobalt is the data color, and signal orange marks "now", selection and warnings. The four token classes (input, cached input, output, reasoning) have fixed colors from a palette checked for color-vision deficiency and contrast in both themes. Every figure carries its unit and uses tabular digits. Each chart has one y axis, a legend when two series share it, and a tooltip reachable by keyboard. The layout works from 360 px (a bottom tab bar up to 768 px) to 2560 px, and it respects reduced motion.
