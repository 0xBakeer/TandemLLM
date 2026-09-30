# Changelog

Releases of TandemLLM. Speed numbers are the benchmark row `serve-single-i256-o256-v1` (one request at a time, 256 prompt tokens, 256 generated, thinking off, greedy), mean tok/s with the lookup store off unless noted. [docs/measurement.md](docs/measurement.md) explains the row. Every release produces the same text as the one before it, except where an entry says otherwise.

## Unreleased

- Image input: `image_url` parts (https and data: URLs) in chat requests. The checkpoint's vision tower runs as stored (BF16), its rows go into the prompt with the model's three-axis rotary, and every cache keys on the image's content. Bad images get a 400 that names the part; `--vision off` turns it off. Text requests are unchanged bit for bit.

## 0.2.1, 2026-09-29

Tagged `v0.2.1`. No decode path changed: the engine, the drafters, the router and the kernels are the ones of `v0.2.0-staircut`, so the generated text is the same. The paper's numbers stay those of `v0.2.0-staircut` (merge `c111fa7`).

- The dashboard login is off by default. `--dashboard-login on|off`, default from `QSE_DASHBOARD_LOGIN`, and `ops/serve.env` sets `off`. Off: `/v1/dashboard/*` answers without a token, even without a `secrets.env`, and the page opens straight into its views, with no sign-in page and no sign-out button. Anyone who can reach the port can then read the dashboard; [docs/dashboard.md](docs/dashboard.md#access) lists what it shows. On: the admin token or a session, as in 0.2.0, with 404 without an admin token and a 429 after five failed sign-ins a minute. To turn it on, set `QSE_DASHBOARD_LOGIN=on` (in `~/TandemLLM/run/local.env` on an installer setup), create the token with `ops/make-secrets.sh` and restart. A misspelt value stops the start.
- What the login does not change: `/metrics`, the full `/health` and `GET /v1/cache/stats` keep their rules, and `POST /v1/cache/clear` still needs the admin token as a bearer header. The box's own tools still need no token.
- Dashboard contract: `GET /v1/dashboard/session` is at 1.1 and says `login: true` or `false`; `GET /v1/dashboard/metrics` is the `/metrics` page under the dashboard's rule, which the Performance view now reads. The contract's [README](docs/contract/dashboard-v1/README.md) lists both.
- The balanced profile (`ops/serve-balanced.env`) now loads the published NVFP4 set's MLP overlay and head (`~/nvfp4-bf16/mlp.safetensors`, `~/nvfp4-bf16/head-fp8.safetensors`), not the first FP8-derived files. The FP8 and balanced profiles inherit StairCut's verify prices, measured on the full-NVFP4 profile; the router corrects their level online, but the staircase's shape for these profiles is not measured, and neither is their speed on this release.
- The 0.2.0 entry's agreement counts are corrected to the paper's classification: of 30 prompts, 17 identical up to the end of the answer, 13 that take the other token at a near tie inside the answer, 0 other differences.
- Docs: the installer's default release is `v0.2.1`, and the README, the installer's closing message, [docs/server.md](docs/server.md) and [docs/dashboard.md](docs/dashboard.md) describe the login as it is now. The paper is published on Zenodo, and the README and `CITATION.cff` cite it by its DOI, [10.5281/zenodo.23045275](https://doi.org/10.5281/zenodo.23045275).

## 0.2.0-staircut, 2026-09-29

Tagged `v0.2.0-staircut`: StairCut, from branch router-eng169, merged at `c111fa7`. The figures below are the release gate's same-hour runs. The paper's measurements of this release, on the published NVFP4 set, are in the [README](README.md#paper), and its classification of the 30-prompt agreement check is the one in [docs/exactness.md](docs/exactness.md).

- StairCut: `QWEN38_LEN_SWITCH=1 QWEN38_LEN_MODE=wide` cuts every round's tree (the wide drafter's lattice, its chain, the lookup tree, their merge) to the node count with the most expected tokens per millisecond on the measured verify staircase, priced by context class (`QWEN38_STAIR_TABLES`). The lookup's continuation rate on 8-token matches is a held-out prior counted online (`QWEN38_STAIR_RHO`), and `QWEN38_STAIR_SKIP=1` skips the head draft on a copy run. All flags are off in the code; `ops/serve.env` turns them on. With them off, every text is byte-identical to rc10. [docs/speculative-decoding.md](docs/speculative-decoding.md) has the rule.
- On the published NVFP4 set, the benchmark row reads 49.33 and 50.47 tok/s with the store off (50.37 and 50.60 with a clean store). Fixed block 8 reads 47.44 and 46.82 (47.71 and 46.80), and fixed block 16 reads 44.65 and 44.95 (43.80 and 44.05), all in the same hour.
- Not every text is identical. Of 30 prompts checked against plain greedy decoding, 17 are identical up to the end of the answer: 13 over all 256 tokens, and 4 more that differ only past the end of the answer. The other 13 take the other token at a near tie inside the answer, with a gap of 0.04 to 0.76 ulp, and continue from there. None differs otherwise. [docs/exactness.md](docs/exactness.md) states the rule.
- The router learns online: draft cost per context class, calibration, copy counts. A server keeps what one request teaches for the next. On the teacher-forced bench (25 workloads), a router rebuilt for every request averages 1.4 % below one that is kept.
- New tools: `tools/router_replay.py` replays router policies on recorded lattices, and `tools/forced_bench.py` measures configurations on one reference text.

## 0.1.0-rc10, 2026-09-28

Not tagged yet.

- The FP8 profile (`ops/serve-fp8.env`): every projection is the checkpoint's own FP8, and the head is its BF16 head. It ran at 28.4 tok/s on the benchmark row. A per-shape FP8 launch table keeps every tile bit-identical from 1 to 32 rows.
- The balanced profile (`ops/serve-balanced.env`): NVFP4 MLPs, the checkpoint's FP8 for the recurrent and attention layers, and the FP8 head. Not measured yet.
- `--fp8-head build` builds the FP8 head at load, and `--price-table` loads the router's verify prices for a weight set.
- First seams for other models: one registry for every engine setting, the checkpoint layout in one place, one `Linear` interface for every weight format, BF16 checkpoints, and drafters that declare what they read from the target. A test pins the exact bytes of a tiny model's run.
- The dashboard's Performance page no longer shifts while the Live panel updates.
- The product name is now TandemLLM (working name) in the README, the dashboard and the HTTP server header. Runtime names are unchanged.
- The NVFP4 profile is unchanged: its gate against the same-hour rc9 found nothing worse, and 13 plain requests came back byte-identical.

## 0.1.0-rc9, 2026-09-27

Tagged. It contains rc7 and rc8.

- Live activity. `/v1/dashboard/live` (contract 1.1) says what each request is doing now: queued, prefilling with progress and an estimate of the time left, thinking, writing, calling a named tool, waiting for the client between two agent turns. It also says how each of the last 20 requests ended, in one sentence. The dashboard's Live panel shows all of it.
- The Playground view loads on first use, and the bundle every tab downloads went from 56 KB to 45 KB gzip.
- The stream writer now sends every tick's event. Before this fix, it skipped an event when one landed between its read and its wait.
- 44.09 tok/s (44.14 with a clean store), against 44.16 for rc8 in the same hour.

## 0.1.0-rc8, 2026-09-27

- The resident prefix. The KV of the last prefill stays in place, with recurrent-state anchors every 1,024 tokens, so the next turn of a growing agent conversation resumes instead of re-reading everything. Resumes are bit-identical to a cold prefill. Requests at 77,000 to 79,000 tokens answered in 2.7 to 16.7 s, where rc7 re-read everything past 13,312 tokens.
- The prefill watch. After every prefill chunk the server checks whether the client is still there, and stops the prefill when it has gone. A retry resumes from the rows already done. A streamed prefill longer than 5 s sends `: prefill done/total` comments.
- 44.10 tok/s, against 44.09 for rc7 in the same hour.

## 0.1.0-rc7, 2026-09-27

- Tool calls are read in three forms (strict XML, lenient XML, JSON) and stream as argument deltas.
- Tool-call values carry the types of the request's own schema: `150` instead of `"150"`. On an 8-task agent run, a client that validates arguments refused 58 of 80 calls before, and 0 of 26 after.
- `tool_choice` works in all four modes, and `required` or a named function is enforced by a token mask.
- Structured outputs: `response_format` (`json_object`, `json_schema`) and regex or choice constraints, compiled to byte-level automata and exact under speculation.
- More of the OpenAI API: `logprobs`, `n`, `min_p`, `logit_bias`, `parallel_tool_calls`. Every OpenAI field is either honoured, accepted without effect, or refused by name.
- The Live panel and `GET /v1/dashboard/live`.
- `tools/client_smoke.py`: the request shapes real clients send, checked before every deploy.
- 43.89 tok/s, against 43.86 for rc6 in the same hour.

## 0.1.0-rc6, 2026-09-26

- `tools/block_ab.py`: an in-process, alternated A/B for changes that keep the output bit-identical.
- A faster 17 to 32-row tile table and a faster FP8 head GEMM, both bit-identical.
- 44.34 tok/s (45.04 with a clean store), 92 ms a round, against 43.36 for rc5 in the same hour.

## 0.1.0-rc5, 2026-09-26

- Sampled requests keep the draft tree (`--sampled-tree det`): at temperature 0.7, 28 % more tokens a round and 19 to 21 % more tok/s, exact in distribution.
- The stream decodes a window of recent tokens instead of the whole answer, 11.9 % faster over the last 1,000 tokens of an 8,000-token answer.
- The skinny kernel reads its scales in contiguous runs.
- About 43.4 to 44.0 tok/s, 92 to 93 ms a round.

## 0.1.0-rc4, 2026-09-25

The first release measured in two store views. Earlier rows ran with a lookup store that could hold the benchmark's own text, and this release replaces them.

- Kernels: the skinny W4A16 kernel for 1 to 32 rows, decode attention that reads the KV once, the block drafter in NVFP4, the verify and the draft from CUDA graphs, and the commit folded into the next verify.
- Trees: a 16-node tree for the narrow drafter, 24 nodes for the wide one after 32 committed tokens, verify up to 32 rows, and a deep chain for text that repeats itself. Tokens a round went from 3.3 to 4.0.
- Sampling with seeds: a seeded request writes the same text whatever the drafters did.
- Usage and timings on every response, the usage ledger, the dashboard at `/dashboard/`, Prometheus metrics, access tokens.
- Long context: the page cache of the weights is dropped after load, which keeps at least 50 GB free from 8k to 128k tokens of context.
- 41.58 tok/s (41.93 with a clean store), 423 ms to the first token, against 25.74 tok/s for rc3's settings on the same row.

## 0.1.0-rc3, 2026-09-20

Not tagged. Serving defaults for long, real work instead of a 256-token row:

- The full 262,144-token context, 32,768 default output tokens, a 3,600 s request timeout and a 240 s queue timeout.
- The state cache budget went from 24 GiB to 8 GiB, because large snapshots on top of the page cache ran the board out of memory.
- Loop guards for the thinking phase that close the reasoning block instead of cutting the stream.
- Anti-repetition penalties, off by default, with per-request values honoured.

## 0.1.0-rc2, 2026-09-18

- The drafter that loses the per-request width decision is released and stops syncing.
- Operations: a lock and a hold script for measurements, and a watchdog that respects a hold.

## 0.1.0-rc1, 2026-09-18

The first release candidate.

- The engine: the checkpoint's FP8 read in place, NVFP4 weights from a gated clip-search quantiser, the FP8 head, and fused kernels for the recurrent layers.
- Speculation: the block drafter, the lookup drafter, tree verify through the recurrent layers, and a block width decided once per request by the length router.
- The server: an OpenAI-compatible API with streaming and reasoning formats, a bounded queue, request timeouts, health, draining, a watchdog and a service unit.
- Serving caches: session and prefix state, a persistent suffix store, and an opt-in response cache.
