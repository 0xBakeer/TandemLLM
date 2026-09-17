# Metrics

The server answers `GET /metrics` in the Prometheus text format. Nothing has to be installed for
it: `prometheus_client` is not on the board and the interpreter there has no pip, so
`server/metrics.py` writes the format out by hand.

```bash
curl -s localhost:8000/metrics | head -40
```

Names carry a `qse_` prefix and follow vLLM's names where the quantity is the same one, so a
dashboard written against vLLM needs a prefix change and not a rewrite. Units are in the name, and
they are seconds and bytes. The contract version is a label of `qse_engine_info`; it is `0.1.0` and
it moves when a name, a label or a unit changes.

Added 2026-09-18, phase 9. This file is append-only: a later version adds a dated section rather
than editing this one.

## Scraping it

```yaml
# prometheus.yml
scrape_configs:
  - job_name: qwen38-spark-engine
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

A scrape costs about 180 lines and under a millisecond. Every gauge is read when the scrape asks,
from the server's own objects, so a scrape never holds the engine lock and never waits for a
generation. A gauge whose source raises is left out of the page instead of failing it.

## What each metric is

### Requests

| metric | type | labels | what it counts |
|-|-|-|-|
| `qse_requests_total` | counter | `finish_reason` | generations that ended, by how they ended: `stop`, `length`, `timeout`, `abandoned`, `error` |
| `qse_request_success_total` | counter | | the `stop` and `length` rows of the line above |
| `qse_requests_refused_total` | counter | `reason` | requests turned away with a `Retry-After` before the engine saw them |
| `qse_requests_running` | gauge | | generations the engine is working on. This engine holds one sequence, so it is 0 or 1 |
| `qse_requests_waiting` | gauge | | callers holding a socket and waiting for the engine lock |
| `qse_prompt_tokens_total` | counter | | prompt tokens accepted, tokens served out of a cache included |
| `qse_generation_tokens_total` | counter | | tokens the engine wrote |
| `qse_errors_total` | counter | `type` | exceptions that ended a generation, by exception class |

`finish_reason` is the server's own set and it is worth reading carefully. `timeout` means the
wall clock in `--request-timeout` ran out and `length` means the caller's token budget did;
`abandoned` means the client hung up, which is not an error and is not counted as one.

### Latency

| metric | type | what it measures |
|-|-|-|
| `qse_time_to_first_token_seconds` | histogram | request arrival to its first token, queue wait included |
| `qse_time_per_output_token_seconds` | histogram | the gap between two tokens of one generation |
| `qse_e2e_request_latency_seconds` | histogram | request arrival to the end of its stream |

Read the inter-token histogram knowing what produces it. A verified block writes several tokens
into the socket at once, so about half of these gaps are microseconds and the other half are a
whole block, and the median is not a rate. The atlas row reports
`(completion_tokens - 1) / (e2e - ttft)` per request for that reason, and
`rate(qse_generation_tokens_total[1m])` is the closest thing here to the same number.

Buckets are chosen from what this engine does rather than copied: 761 ms is the median time to
first token on a 256-token prompt, 526 ms is a warm one on a 1,724-token prompt out of the prefix
cache, and 2.0 ms is a response-cache hit.

### Speculation

| metric | type | labels | what it counts |
|-|-|-|-|
| `qse_spec_decode_num_drafts_total` | counter | | blocks the drafter proposed |
| `qse_spec_decode_num_draft_tokens_total` | counter | | tokens proposed, the anchor excluded |
| `qse_spec_decode_num_accepted_tokens_total` | counter | | drafted tokens the target agreed with |
| `qse_spec_accept_per_block` | histogram | | tokens one block committed, the target's own token included |
| `qse_draft_width_chosen_total` | counter | `width` | blocks proposed at each width, the anchor included |
| `qse_verify_seconds` | histogram | | one verify pass, as the decode loop paid for it |
| `qse_draft_seconds` | histogram | | one call to the drafter, lookup and block drafter together |

The draft acceptance rate is `accepted / draft_tokens`. Tokens per block is
`qse_spec_accept_per_block_sum / _count`, and it is the number the whole engine is tuned on: the
row moves when it moves. On the five-workload bench at width 16 it reads 2.60 on fresh prose and
14.86 on a quotation, which is the spread a single average hides.

`width` is 8 or 16 on this build, counting the anchor: the length router holds two drafters, one
fine-tuned at each block length, and picks one. A run that shows `width="8"` on most blocks is the
latch deciding the narrow arm wins, which happens on fresh prose and nowhere else.

`qse_verify_seconds` is the same measurement `engine/lenrouter.py` prices its own decisions on. It
comes from the loop that pays for it rather than from a profiler, because
`tools/profile_block.py` once read a rollback at 23.4 ms that the serving loop reads at 6.4.

### Caches

| metric | type | labels | what it counts |
|-|-|-|-|
| `qse_cache_hits_total` | counter | `cache` | prefix lookups answered from a cache |
| `qse_cache_misses_total` | counter | `cache` | lookups that found nothing |
| `qse_cache_evictions_total` | counter | `cache` | entries dropped to stay inside the byte budget |
| `qse_cache_collisions_total` | counter | `cache` | stored prefixes whose 64-bit hash matched and whose tokens did not |
| `qse_cache_bytes` | gauge | `cache` | bytes held |
| `qse_cache_entries` | gauge | `cache` | entries held |
| `qse_prefill_tokens_reused_total` | counter | | prompt tokens restored from a snapshot |
| `qse_prefill_tokens_forwarded_total` | counter | | prompt tokens run through the 64 layers |

`cache` is `state`, `response` or `suffix`, and one thing about that label needs saying. The
session cache and the prefix cache are **one object**: `engine/cache.py::StateStore` holds a
snapshot per token boundary and a session boundary is a boundary like any other, so the store
cannot tell you which of the two a hit belonged to and neither can this page. The pair of numbers
that answers the question the label was meant to answer is
`qse_prefill_tokens_reused_total` against `qse_prefill_tokens_forwarded_total`: on a shared system
prompt of 1,724 tokens the warm request reuses 1,536 and forwards 188.

The suffix store reports its size and nothing else. Its lookups happen inside the drafter, tens of
thousands of times per generation, and counting them would cost more than they cost.

A cache that is switched off reports no rows at all rather than zeroes, so a panel that goes blank
is telling you the flag is off.

### The board

| metric | type | what it reads |
|-|-|-|
| `qse_gpu_memory_used_bytes` | gauge | `torch.cuda.memory_allocated()`, what the engine holds |
| `qse_gpu_memory_reserved_bytes` | gauge | what the allocator took from the driver and has not given back |
| `qse_unified_memory_free_bytes` | gauge | free bytes in the board's one pool |
| `qse_engine_uptime_seconds` | gauge | seconds since the server started |
| `qse_engine_info` | gauge | always 1; the labels are the configuration |

GB10 shares one memory pool between the GPU and the host and `nvidia-smi` reports N/A for used
memory, so these three are the only honest numbers about memory on this board. A leak shows as
`used` climbing across requests; fragmentation shows as `reserved` climbing while `used` does not.

`qse_engine_info` carries `version`, `model`, `drafter`, `width`, `tree`, `nvfp4`, `fp8_head`,
`max_len` and `caches`. Join on it when you compare two runs, because a tok/s number without the
weight set it was produced by is not comparable to anything.

## An example scrape

From a run of the atlas row, with the histogram buckets cut down to the interesting ones:

```
# HELP qse_requests_total generations that finished, by the reason they finished
# TYPE qse_requests_total counter
qse_requests_total{finish_reason="abandoned"} 1
qse_requests_total{finish_reason="length"} 50
# TYPE qse_request_success_total counter
qse_request_success_total 50
# TYPE qse_generation_tokens_total counter
qse_generation_tokens_total 12800
# TYPE qse_time_to_first_token_seconds histogram
qse_time_to_first_token_seconds_bucket{le="0.75"} 12
qse_time_to_first_token_seconds_bucket{le="1"} 50
qse_time_to_first_token_seconds_bucket{le="+Inf"} 50
qse_time_to_first_token_seconds_sum 38.07
qse_time_to_first_token_seconds_count 50
# TYPE qse_spec_decode_num_accepted_tokens_total counter
qse_spec_decode_num_accepted_tokens_total 6338
# TYPE qse_spec_decode_num_draft_tokens_total counter
qse_spec_decode_num_draft_tokens_total 24000
# TYPE qse_spec_decode_num_drafts_total counter
qse_spec_decode_num_drafts_total 1600
# TYPE qse_spec_accept_per_block histogram
qse_spec_accept_per_block_bucket{le="1"} 174
qse_spec_accept_per_block_bucket{le="4"} 1022
qse_spec_accept_per_block_bucket{le="16"} 1600
qse_spec_accept_per_block_bucket{le="+Inf"} 1600
qse_spec_accept_per_block_sum 7938
qse_spec_accept_per_block_count 1600
# TYPE qse_draft_width_chosen_total counter
qse_draft_width_chosen_total{width="16"} 1600
# TYPE qse_verify_seconds histogram
qse_verify_seconds_bucket{le="0.1"} 898
qse_verify_seconds_bucket{le="0.11"} 1600
qse_verify_seconds_sum 159.43
qse_verify_seconds_count 1600
# TYPE qse_requests_running gauge
qse_requests_running 1
# TYPE qse_requests_waiting gauge
qse_requests_waiting 0
# TYPE qse_cache_bytes gauge
qse_cache_bytes{cache="state"} 4541000000
qse_cache_bytes{cache="suffix"} 943052
# TYPE qse_prefill_tokens_reused_total counter
qse_prefill_tokens_reused_total 1536
# TYPE qse_prefill_tokens_forwarded_total counter
qse_prefill_tokens_forwarded_total 188
# TYPE qse_engine_info gauge
qse_engine_info{version="0.1.0",model="qwen38-spark-engine",drafter="LengthRouter",width="15",tree="1",nvfp4="mlp-clip,gdn-clip,attn-clip",fp8_head="1",max_len="32768",caches="session,prefix,suffix"} 1
```

Reading it: 1,600 blocks proposed 24,000 tokens and the target kept 6,338 of them, a draft
acceptance of 26.4 %, and 7,938 tokens came out of 1,600 blocks, so a block committed 4.96. The
verify sum over its count is 99.6 ms, which is the width-16 verify the length router is priced
against.

## Panels for a dashboard

Twelve panels, in the order they answer a question. Every query is PromQL against the names above.

#### 1. Tokens a second, served

The headline.
```promql
rate(qse_generation_tokens_total[1m])
```

#### 2. Tokens per block

The one number the whole engine is tuned on.
```promql
rate(qse_spec_accept_per_block_sum[5m]) / rate(qse_spec_accept_per_block_count[5m])
```

#### 3. Draft acceptance

Falls before tokens per block does, and it is the leading indicator of a
drafter that has lost the workload.
```promql
rate(qse_spec_decode_num_accepted_tokens_total[5m])
  / rate(qse_spec_decode_num_draft_tokens_total[5m])
```

#### 4. The block, split

Two series on one axis, in milliseconds. Their sum is the block, and the
gap between that sum and the reciprocal of panel 1 is the commit, the sync and the idle.
```promql
1000 * rate(qse_verify_seconds_sum[5m]) / rate(qse_verify_seconds_count[5m])
1000 * rate(qse_draft_seconds_sum[5m]) / rate(qse_draft_seconds_count[5m])
```

#### 5. Time to first token, p50 and p99

```promql
histogram_quantile(0.5,  sum by (le) (rate(qse_time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(qse_time_to_first_token_seconds_bucket[5m])))
```

#### 6. Queue depth

`qse_requests_waiting`, as a graph rather than a number. A server that is
never above 0 has no queueing problem whatever its rate looks like.

#### 7. Refusals a minute

`rate(qse_requests_refused_total[5m])`. Anything above zero means
callers are being sent away, and the queue bound is `--max-queue`, 8 by default.

#### 8. Finish reasons

`sum by (finish_reason) (rate(qse_requests_total[5m]))`, stacked.
`length` climbing means callers are hitting a token cap; `timeout` climbing means generations are
running past `--request-timeout`.

#### 9. Errors

`sum by (type) (rate(qse_errors_total[5m]))`. Alert on any of it.

#### 10. Prompt tokens reused

The caches, as the one ratio that says what they bought.
```promql
rate(qse_prefill_tokens_reused_total[10m])
  / (rate(qse_prefill_tokens_reused_total[10m]) + rate(qse_prefill_tokens_forwarded_total[10m]))
```

#### 11. Cache bytes against the budget

`qse_cache_bytes` by `cache`, with
`--cache-budget-gb` drawn as a threshold. Watch `qse_cache_evictions_total` beside it: eviction
starting is the moment the budget became the binding constraint.

#### 12. Memory

`qse_gpu_memory_used_bytes` and `qse_gpu_memory_reserved_bytes` on one axis,
`qse_unified_memory_free_bytes` on another. This is the soak-test panel: used climbing across
requests is a leak, reserved climbing while used does not is fragmentation.

Two alerts are worth having before any dashboard: `rate(qse_errors_total[5m]) > 0`, and
`qse_requests_waiting > 6` for five minutes, which says the queue is about to start refusing.

## What is deliberately not here

- **Batch size, KV utilisation, preemptions, swaps.** vLLM reports them because it has a
  scheduler. This engine holds one sequence and has none of those states.
  `notes/PLAN-PARALLELISM.md` says what would have to be built first, and it adds the metrics it
  would need to its own phases.
- **Per-model and per-user labels.** One process serves one model, and a user label on a counter
  is personal data on a monitoring endpoint. The label set here has a fixed, small cardinality:
  five finish reasons, two widths, three caches, and one exception class per distinct failure.
- **Rollbacks.** The loop rolls back on a partial accept and the count follows from
  `drafts - (blocks that accepted everything)`, which no counter here can separate without a hook
  inside the accept loop. It is in `engine/spec.py::DecodeStats` for the bench, where it belongs.
- **A pull of the drafter's own report.** `engine/lenrouter.py::report()` prints the latch, the
  ceiling and the calibration factor once a generation, and those are per-request beliefs rather
  than server state. They are in the log line, not on this page.

## Where the hooks are, and why they are not in the loop

`server/metrics.py` wraps four things and `server/app.py` carries an import, a route and one
`install()` call:

| wrapped | gives |
|-|-|
| `generate_stream` | time to first token, the gaps between tokens |
| `_log_request` | finish reasons, token counts, end-to-end latency, errors |
| `Handler._complete` | the arrival time, so the queue wait is inside time to first token |
| the drafter's `propose` / `propose_tree` / `observe` / `on_verify` | every speculation counter |

The alternative was a counter next to each of those calls inside the generation loop. Phase 9 was
editing that loop for three serving bugs at the same time this was written, and a hook scattered
through it would have collided with every one of those edits. The wrappers cost two
`perf_counter()` calls and a few dictionary increments per block, against a block of 133 ms.

`tests/test_metrics.py` is 26 tests, no torch and no board. Half of them are the exposition format,
parsed the way a scraper parses it; the other half are the semantics, and the one worth knowing
about asserts that an `observe` that follows no proposal is not counted as a block. The loop calls
`observe` after a prefill, after a step the drafter declined, and for the tokens a reasoning budget
forces, and counting those would put the accepted-tokens counter above the drafted one.
