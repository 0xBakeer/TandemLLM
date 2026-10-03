# Package library v1 (Fable, 2026-10-03)

One skill per folder, `<name>/SKILL.md`, opencode SKILL.md format (frontmatter `name`, `description`). The body before `## Reference` is the injected part (cap 350 tokens); the `## Reference` section (cap 600) is injected only when the skill is selected. Tokens are chars/3.9 estimates; measure with the engine tokenizer before install. Retrieval: match the task against `description` (the fit question over the class's candidates), then filter by class fit. The testing package and the five test-kind skills (`testing`, `test-web`, `test-cli`, `test-api`, `test-data`, `test-lib`), `tools-opencode`, `tools-hermes` and `tandem-harness` live one level up and are not repeated here.

Count: 44 skills. Body total 14859 tokens, reference total 18840, all 33699.

| name | description | body | ref | classes |
|-|-|-|-|-|
| **Web** | | | | |
| web-layout | Building an HTML/CSS page layout with flexbox and grid that holds at 1280 px desktop and 390 px phone width - headers, sidebars, cards, columns, spacing. | 341 | 246 | design_ui, code_edit |
| web-a11y | Accessibility basics for any web page or app - semantic elements, labels, keyboard focus, contrast, ARIA only when needed, screen-reader names. | 346 | 291 | design_ui, code_edit |
| dom-events | Vanilla JavaScript in the browser without a framework - DOM queries, event listeners, delegation, rendering state to HTML, and avoiding leaks. | 336 | 371 | code_edit, debug |
| canvas-2d | Drawing and animating on an HTML canvas with the 2D context - render loop, device pixel ratio, sprites, transforms, text, and keeping 60 fps. | 348 | 367 | drawing, code_edit, debug |
| threejs-scene | Building a WebGL scene with Three.js - renderer, camera, lights, meshes, controls, resize, animation loop, and performance (draw calls, instancing, disposal). | 340 | 569 | three_d, code_edit, debug |
| browser-game | Building a 2D browser game (platformer, shooter, puzzle) - game loop with fixed timestep, keyboard input, tile collisions, camera, HUD, lives and win state. | 343 | 511 | code_edit, debug, drawing |
| web-audio | Playing and synthesizing sound in the browser with the Web Audio API - AudioContext unlock on a gesture, oscillators, samples, gain envelopes, timing on the audio clock. | 338 | 357 | code_edit, debug |
| web-forms | HTML forms with client-side validation and submission - input types, required and pattern, custom messages, error display, submit via fetch, and server-side revalidation. | 340 | 472 | code_edit, design_ui |
| fetch-client | Calling a REST or JSON API from browser or Node JavaScript with fetch - status handling, JSON errors, timeouts with AbortController, retries, auth headers, loading and error state in the UI. | 344 | 471 | code_edit, debug |
| web-storage-offline | Persisting browser state with localStorage, sessionStorage and IndexedDB, and a minimal service worker for offline caching - versioned keys, try/catch, quotas, cache strategy. | 343 | 493 | code_edit, debug |
| **Frameworks** | | | | |
| react-basics | Writing React components with hooks - useState, useEffect, derived values, lists with keys, forms, lifting state, and the common performance and re-render pitfalls. | 346 | 439 | code_edit, debug, design_ui |
| svelte-basics | Building a small app with Svelte 5 - runes ($state, $derived, $effect), components and props, events, each blocks with keys, stores versus runes, and the Vite project setup. | 327 | 392 | code_edit, design_ui |
| ts-vite-build | Setting up TypeScript in a frontend or Node project with Vite or esbuild - tsconfig strict mode, types for props and API data, type narrowing, building and type-checking before done. | 349 | 361 | code_edit, debug |
| **Backend** | | | | |
| fastapi-service | Writing a Python HTTP API with FastAPI (or Flask) - pydantic models, routers, validation errors, status codes, dependencies, settings from the environment, running with uvicorn and checking routes with curl. | 332 | 466 | code_edit, debug |
| express-service | Writing a Node.js HTTP API with Express - routers, JSON bodies, validation, error middleware, status codes, async handlers, environment config, and probing routes with curl. | 332 | 508 | code_edit, debug |
| auth-basics | Adding login and authorization to a web app or API - password hashing, sessions or short-lived tokens, cookies with the right flags, protecting routes, and keeping secrets out of code and git. | 349 | 445 | code_edit, plan |
| sql-schema-migrations | Designing a relational schema in SQLite or PostgreSQL and changing it with migrations - tables, keys, constraints, timestamps, versioned migration files, applying and rolling back safely. | 339 | 495 | code_edit, plan |
| sql-queries-indexes | Writing correct and fast SQL queries - joins, aggregates, parameters against injection, pagination, N+1 avoidance, and choosing indexes with EXPLAIN in SQLite and PostgreSQL. | 323 | 389 | code_edit, debug |
| background-jobs | Running work outside the request - queues, workers, scheduled tasks and retries in Python or Node, with idempotent jobs, timeouts and a way to see what ran. | 330 | 537 | code_edit, plan |
| websockets | Realtime two-way messaging between browser and server with WebSockets (FastAPI or Node ws) - message schema, reconnect, heartbeats, broadcast, and testing with a scripted client. | 334 | 567 | code_edit, debug |
| **Python** | | | | |
| python-cli | Writing a Python command-line tool with argparse or typer - subcommands, flags, stdin and stdout, exit codes, error messages, --dump-state, and a main() that tests can call. | 324 | 540 | code_edit |
| python-env-packaging | Setting up a Python project - virtual environment with venv or uv, pyproject.toml, pinned dependencies, src layout, entry points, and installing it in editable mode. | 342 | 352 | code_edit, debug |
| pandas-cleaning | Loading and cleaning tabular data with pandas - reading CSV and Excel, types, missing values, duplicates, dates, string normalization, joins, and proving row counts at every step. | 324 | 468 | code_edit, research |
| matplotlib-plots | Making clear charts with matplotlib (line, bar, scatter, histogram, multi-panel) saved to PNG or SVG - figure sizing, labels, units, legends, log axes, and checking the output image. | 325 | 397 | code_edit, drawing, research |
| numpy-performance | Making numeric Python fast with NumPy - vectorizing loops, broadcasting, dtypes and memory, avoiding copies, and measuring with timeit before and after. | 309 | 434 | code_edit, debug |
| python-async | Using asyncio in Python - async functions, gather with limits, timeouts, cancellation, running blocking code in a thread, async HTTP with httpx, and avoiding the event loop stalls. | 318 | 432 | code_edit, debug |
| pytest-practice | Writing and running Python tests with pytest - test layout, fixtures, parametrize, tmp_path, monkeypatch and mocking, testing a CLI or API, and reading the summary line. | 323 | 442 | code_edit, debug |
| **Systems** | | | | |
| bash-scripting | Writing shell scripts that fail loudly and handle paths safely - set -euo pipefail, quoting, arrays, functions, arguments, temp files with trap cleanup, and testing the error paths. | 342 | 378 | code_edit, debug |
| git-workflow | Day-to-day git for a coding task - small commits with clear messages, a branch per task, reading status and diff before committing, resolving merge conflicts, and never committing secrets or generated files. | 349 | 357 | code_edit, plan |
| docker-basics | Writing a Dockerfile and compose file for a service, reading container logs, and running an app in a container - layers, caching, non-root user, ports, volumes, health checks, and the project rule on where images get built. | 344 | 315 | code_edit, plan |
| processes-ports | Starting servers and long-running commands from an agent shell - background with logs, waiting for a port, health checks, finding and killing stale processes, timeouts, and cleaning up at the end. | 332 | 413 | code_edit, debug |
| logging-observability | Adding useful logs, metrics and a state endpoint to a service or script - structured log lines, levels, request ids, timing, log files the runtime can tail, and counters that answer "what happened". | 347 | 452 | code_edit, debug |
| profiling | Finding why a program is slow before changing it - timing the real case, Python cProfile and py-spy, Node --cpu-prof, browser performance traces, picking the one hot spot, and proving the gain with numbers. | 346 | 351 | debug, code_edit |
| security-basics | Basic security for web apps, APIs and scripts - input validation, SQL and command injection, XSS, path traversal, secrets handling, dependency checks, and the quick probes that prove a hole is closed. | 345 | 461 | code_edit, plan |
| regex | Writing and testing regular expressions in Python and JavaScript - anchors, groups, character classes, greedy versus lazy, multiline flags, named groups, and testing against positive and negative examples before use. | 335 | 377 | code_edit, debug |
| files-and-formats | Reading and writing files and data formats correctly - text encodings (UTF-8, BOM, latin-1), newlines, binary versus text, JSON, YAML, CSV with quoting, large files streamed, atomic writes, and paths with pathlib. | 342 | 485 | code_edit, debug |
| datetime-timezones | Handling dates, times and time zones in Python and JavaScript - storing UTC, aware datetimes, parsing and formatting ISO-8601, converting for display, durations, DST pitfalls and date arithmetic. | 343 | 509 | code_edit, debug |
| **Practice** | | | | |
| read-codebase | Getting oriented in an unfamiliar repository before changing it - project rules files, entry points, build and test commands, the module map, data flow for one feature, and writing down what was learned. | 348 | 398 | plan, code_edit, research |
| refactor-safely | Changing the structure of existing code without changing its behavior - characterization tests first, small verified steps, renames and extractions, keeping the public API, and proving nothing changed. | 340 | 474 | code_edit, plan |
| write-readme | Writing or updating a README for a project or tool - what it is in two lines, install and run commands that were actually executed, configuration, a usage example with real output, and what is not covered. | 345 | 212 | code_edit, summarize |
| debug-method | A method for fixing a bug when the cause is unknown - reproduce with the smallest input, read the state before the symptom, bisect by halves (input, code, commits), name one cause, change one thing, re-run the same reproduction. | 348 | 436 | debug |
| plan-feature | Planning a multi-file feature before building it - clarify the outcome, list files and steps in build order, a test for each step, the risks, an estimate, and when to stop planning and start. | 346 | 390 | plan |
| algorithm-choice | Choosing a data structure or algorithm for a task - sizing the input, complexity targets, hash maps versus sorted structures, sets, heaps, graphs (BFS, DFS, Dijkstra), dynamic programming, and checking with a brute-force oracle. | 343 | 504 | code_edit, plan |
| numerical-precision | Avoiding floating-point mistakes in code that computes money, statistics, physics or geometry - when to use integers or Decimal, comparing with tolerances, summation order, catastrophic cancellation, float32 limits, and rounding for display. | 319 | 516 | code_edit, debug |

## Class coverage

- `code_edit` (42): web-layout, web-a11y, dom-events, canvas-2d, threejs-scene, browser-game, web-audio, web-forms, fetch-client, web-storage-offline, react-basics, svelte-basics, ts-vite-build, fastapi-service, express-service, auth-basics, sql-schema-migrations, sql-queries-indexes, background-jobs, websockets, python-cli, python-env-packaging, pandas-cleaning, matplotlib-plots, numpy-performance, python-async, pytest-practice, bash-scripting, git-workflow, docker-basics, processes-ports, logging-observability, profiling, security-basics, regex, files-and-formats, datetime-timezones, read-codebase, refactor-safely, write-readme, algorithm-choice, numerical-precision
- `debug` (26): dom-events, canvas-2d, threejs-scene, browser-game, web-audio, fetch-client, web-storage-offline, react-basics, ts-vite-build, fastapi-service, express-service, sql-queries-indexes, websockets, python-env-packaging, numpy-performance, python-async, pytest-practice, bash-scripting, processes-ports, logging-observability, profiling, regex, files-and-formats, datetime-timezones, debug-method, numerical-precision
- `design_ui` (5): web-layout, web-a11y, web-forms, react-basics, svelte-basics
- `drawing` (3): canvas-2d, browser-game, matplotlib-plots
- `three_d` (1): threejs-scene
- `plan` (10): auth-basics, sql-schema-migrations, background-jobs, git-workflow, docker-basics, security-basics, read-codebase, refactor-safely, plan-feature, algorithm-choice
- `research` (3): pandas-cleaning, matplotlib-plots, read-codebase
- `summarize` (1): write-readme
- `chat` (0): none; the coding package and these skills are not injected on this class

## Conventions every skill follows

- Imperative rules (4 to 6), then "What goes wrong" (3 to 5 items that happened), a "Check:" line, a "Done when" list of 3 items that each need evidence (a run, a test summary line, a screenshot at 1280 and 390, a log line, a status code).
- The `## Reference` holds snippets and commands only; some carry "More rules" and "More pitfalls" moved there to keep the body under the cap.
- State hooks (`window.__state()`, `/__state`, `--dump-state`, `./.tandem/*.log`) use the same names as the testing package.
- No client is named that the request did not come from; browser steps say "the browser tool"; shell steps assume a tool timeout and background servers.
