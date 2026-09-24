# Dashboard

The engine's own web app: a year of token usage (Usage), live and historical speed
(Performance), an LM Studio-style developer view with the live log (Dev), and the box's state
(System). Served by the engine at `/dashboard/` from the committed `dist/`; no Node on the box.

Built against the **dashboard API contract v1** — `docs/contract/dashboard-v1/` (schemas +
examples; the prose is the Memo note "Usage & speed metrics — design (2026-09-24)" §3) — first on
the mock harness, then on the real backend (SRV-27..SRV-31). Design: Memo "Dashboard — design
(2026-09-24)"; tickets VIS-2, VIS-13..VIS-18.

## Stack

Vite + TypeScript + Lit (light-DOM components, one stylesheet), hand-drawn SVG charts (glowing
ribbons, stacked bars, scatter, the year heatmap, the timing bar), no CDN, no external fonts,
pinned dependencies. Build: one `index.html`, one hashed JS, one CSS, relative asset paths;
~43 KB gzip total (budget 300 KB, `npm run test:size`).

```
src/app.ts             the shell: session gate, top bar + status pill, rail / bottom tabs, hash routing, theme
src/views/*.ts         usage, performance, dev, system
src/charts/*.ts        ribbon, bars, scatter, heatmap, timing-bar (+ svg.ts scales/paths)
src/lib/prom.ts        Prometheus text parser + live-strip arithmetic (unit-tested)
src/lib/sse.ts         SSE framing + resumable stream (Last-Event-ID, back-off), OpenAI stream reader
src/lib/heatmap.ts     year grid: 53 × 7, Monday first, quantile levels, streaks
src/lib/time.ts        local days/hours in an IANA zone, DST-safe
src/api/client.ts      same-origin fetch, 401 → sign-in, offline → status pill
src/styles/tokens.css  the design tokens (dark instrument readout + light theme; validated dataviz palette)
mock/                  VIS-13: seeded year of usage, aggregation per the contract, SSE log, /metrics, chat stream
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
| `npm test` | unit tests, contract validation, dist hygiene (76 tests) |
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

## Design notes

Dark-first instrument readout with a light theme (`prefers-color-scheme` until a choice is
remembered). Cobalt is the data colour, signal orange marks now / selection / warnings; the four
token classes (input, cached input, output, reasoning) are cobalt / aqua / orange / violet and the
cache sources cobalt / orange / aqua — the dataviz palette, validated for colour-vision deficiency
and contrast in both themes. Every figure carries its unit; tabular numerals; one y axis per chart;
a legend whenever two series share a plot; every chart has a keyboard-reachable tooltip; the
heatmap cells are a roving-tabindex grid. Responsive from 360 px (bottom tab bar ≤ 768 px) to
2560 px; reduced motion respected.
