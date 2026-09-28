# Measurement

Speed claims for inference engines are easy to inflate by accident. Our lookup store once held the benchmark's own prompts and answers, and it added about 21 % to the benchmark row without the engine getting any faster. This page describes how TandemLLM measures itself, from the benchmark row to the release gate and the comparison with other engines.

## The row

The benchmark row is `serve-single-i256-o256-v1`: one request at a time, 256 prompt tokens, 256 generated tokens, thinking off, temperature 0, seed 42, 50 requests after 3 warm-ups. The bench runner and the prompt set are the same ones used for every engine this row compares against.

Each request's speed is computed from its own record:

    tok/s = (completion_tokens - 1) / (end-to-end seconds - time to first token)

The first token belongs to the prefill, so it is subtracted, and the decode time is the request's wall clock after its first token. A row reports the mean, p50, p90 and max over the 50 requests, with the median time to first token.

Two readings of a row are easy to get wrong.

- The gaps between tokens. A verified round writes several tokens into the socket at once, so half the gaps are microseconds and the other half a whole round. The median inter-token latency therefore says nothing about round time.
- Mean against median. Some prompts ask the model to repeat text that is already in the context, and the drafters commit 15 tokens a round on them. That lifts the mean. The median reflects fresh text.

## Two views of every row

A row runs twice, in two views of the lookup store:

- Store off. The engine's persistent suffix store is off, so nothing the engine has seen before can help it.
- Clean store. The store is on but holds no text of the benchmark.

A row with the benchmark's own text in the store measures the store, not the engine. Benchmarks also run with `--suffix-store-readonly` when they read a store of real traffic, so they cannot write themselves into it.

## Noise, and what a row can decide

The length router decides each request's block width from timing taken inside that request. Two runs of the same engine can make different decisions on the same prompts. Rows taken from three separate server processes spread 10.4 % on the mean and 6.7 % on the median. Three rows through one process spread 5.9 % and 3.1 %.

`tools/row3.py` runs N rows per configuration through one server, recomputes each from its own records, and reports the median with the spread of the runs. `--compare` calls a difference resolved only when it is larger than the noise and the run ranges of the two sides do not overlap. A change worth less than about 3 % of the median cannot be settled by the row at any number of repeats.

Speed is a quotient: tokens a round over time a round. A change can trade one factor for the other and leave the quotient flat. The served loop therefore counts its own rounds, and every row reports tokens per round and milliseconds per round next to tok/s. Both are gated. A kernel change must not move tokens per round, and a drafter change must move it up.

Every report names the code it measured with `code_sha256`, a hash over every file of the four code folders. The same commit hashes the same everywhere, so a report can be matched to a commit.

## Same hour, both sides

Nobody compares a candidate with a number from another day. It runs against the current release in the same session on the same box, alternated: base, candidate, base, candidate. A board drifts over hours (its temperature, its memory, whatever else runs), and alternation spreads that drift over both sides.

The adoption rule is "never worse". A candidate ships only if nothing resolves worse against the same-hour base, in both store views: not the mean, p50, p90 or max, not time to first token, and not tokens or milliseconds per round.

## The block A/B, for changes that keep the bits

Most kernel changes are worth 0.5 to 3 ms of a 92 ms round, below what the row can resolve. For a change whose output is bit-identical, only time can differ. `tools/block_ab.py` loads the served engine once, switches each state's flags inside the process, warms each state, freezes the router's learned costs, and alternates base and candidate over three workloads for N pairs. Every run's tokens must equal the base's first run, and the rule is applied to milliseconds per round. Two states of the same code read "not resolved" with a spread of 0.35 to 0.87 %. Turning the verify graphs off resolved 3.15 ms worse on every workload.

## The release gate

`ops/gate.sh` is the whole protocol in one command, run on the box inside a hold ([operations.md](operations.md)). It stops at the first failing step:

1. Every CPU test, `tests/test_*.py`, in a clean environment.
2. Every GPU test, `tests/gpu/test_*.py`.
3. Identity: the served configuration with the new flags off must be bit-identical to the build it replaces (`tools/flagoff_identity.py`).
4. Lossless: `tools/verify_spec.py` with the served flags and the candidate's ([exactness.md](exactness.md)).
5. Optionally, the block A/B above.
6. Rows: two store-off rows and one clean-store row with the candidate's flags.
7. Compare: `tools/gatecheck.py` against the base reports, with the never-worse rule.

It writes a dated report with every command and its output to `results/gate/<label>/report.md`. `results/` is not under version control.

Rows are not enough on their own. A release candidate once passed every row and broke a real client's request shape. Before a deploy the gate is followed by:

- `tools/client_smoke.py`: the request shapes real clients send, through the official `openai` client. It covers an agent client's tools with arguments validated against their JSON schemas, a 12k-token prompt with penalties and tools, every `tool_choice` mode, streamed tool-call deltas, `n=2`, and prompts from 8k to over 100k tokens. It passes with zero failures.
- `tools/api_text_check.py`: 13 plain requests byte-identical to the current release.
- A real agent session with its own tool calls, all of which must go through.

## Comparing with other engines

The rule is simple: compare against the other engine's best configuration for the same weights and workload, not its default. For vLLM that means sweeping its speculative methods (MTP at 2 and 3 tokens, n-gram) and taking the fastest. Its run without speculation is context only.

| engine and weights | mean tok/s | role |
|-|-|-|
| vLLM 0.27.1, FP8, no speculation | 7.72 | context |
| vLLM 0.27.1, FP8, n-gram | 8.56 | context |
| vLLM 0.27.1, FP8, MTP 2 | 15.04 | sweep point |
| vLLM 0.27.1, FP8, MTP 3 | 15.57 | vLLM's best for FP8 |
| TandemLLM, FP8 profile | 28.4 | 1.82x vLLM's best |
| vLLM 0.27.1, NVFP4, best of MTP 2, MTP 3 and n-gram | 25.71 | vLLM's best for NVFP4 |
| TandemLLM, NVFP4 profile | about 44 | 1.71x vLLM's best |

vLLM ran as its 0.27.1 container with default settings except the memory share (0.82 for FP8, 0.78 for NVFP4, to keep 15 GiB free). Its prefix caching stayed at vLLM's default. vLLM 0.27.1 cannot load the DFlash2 drafter: the drafter's config declares `DFlash2DraftModel`, and vLLM's registry maps only its native `dflash` method. The NVFP4 rows read different weights. vLLM reads a published NVFP4 export, and TandemLLM reads its own quantised set ([quantisation.md](quantisation.md)).

SGLang and llama.cpp are the other baselines worth running on this board. Their rows will be added with the same rule.

## Traps

Each of these cost a wrong number once.

- A standalone kernel timing is a ranking, never a budget. The same configuration read 0.254 ms in one process and 0.627 ms in another.
- The first configuration in a fresh process pays for autotuning. It once read a verify of 260.2 ms against 115.5 ms for every later one, and the policy under test "won" by 35 %.
- A bandwidth-bound engine measured beside another engine measures the other engine. Two engines halve the bandwidth each one gets. Take the board lock.
- A speculative decoder judged on a short generation hides bugs. One drafter bug read 81 % acceptance at 48 tokens and 24 % at 128.
- Acceptance with the reasoning block open differs from acceptance with it closed, by up to 72 % on one prompt. Measure in the serving regime.
- A kernel probe whose working set fits in cache measures the cache. 266 GB/s on a 273 GB/s board is not a real reading.
- Two runs that agree to six decimals have not agreed. They are the same run: once, a test server had bound its port beside another.
- A quiet log is not a hung process. Python block-buffers into a pipe, so use `python -u`.

## Limits

The published numbers come from one board and one workload shape. Long prompts and parallel requests are not in the headline table yet. The quality gates are held-out loss, confident argmax agreement and free generation. No task benchmark suite has been run, so this repository makes no claim about task accuracy.
