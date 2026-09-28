# Changelog

Releases of TandemLLM. Speed numbers are the benchmark row `serve-single-i256-o256-v1` (one request at a time, 256 prompt tokens, 256 generated, thinking off, greedy), mean tok/s with the lookup store off unless noted. [docs/measurement.md](docs/measurement.md) explains the row. Every release produces the same text as the one before it, except where an entry says otherwise.

## Unreleased (branch vision)

- Image input: `image_url` parts (https and data: URLs) in chat requests. The checkpoint's vision tower runs as stored (BF16), its rows go into the prompt with the model's three-axis rotary, and every cache keys on the image's content. Bad images get a 400 that names the part; `--vision off` turns it off. Text requests are unchanged bit for bit. Gate against rc9 pending.

## 0.1.0-rc10, 2026-09-28

Not tagged yet.

- The FP8 profile (`ops/serve-fp8.env`): every projection is the checkpoint's own FP8, and the head is its BF16 head. It runs at 28.4 tok/s, against 15.57 for vLLM 0.27.1 with MTP on the same checkpoint. A per-shape FP8 launch table keeps every tile bit-identical from 1 to 32 rows.
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
