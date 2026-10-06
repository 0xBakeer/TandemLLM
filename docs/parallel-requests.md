# Parallel requests

The engine serves one request at a time unless it is told otherwise. With `--parallel N` it holds N sequences over the one copy of the weights and decodes up to N requests at once. Two or more decoding requests share each verify pass, so the weights are read once for all of them. It is off by default, because the gain is modest at today's kernels and every slot costs memory.

How it works inside (slots, the GPU turn, lockstep rounds, admission) is in [server.md](server.md#parallel-requests-0338-test---parallel-n-off-by-default). This page is about using it.

## Switching it on

In `ops/serve.env` (or `local.env`):

```bash
QSE_PARALLEL=2            # 1 (the default) is one sequence; N builds N - 1 side slots
QSE_SIDE_MAX_LEN=65536    # a side slot's window in tokens; slot 0 keeps the full 262,144
QSE_PARALLEL_FLOOR_GB=15  # admission keeps MemAvailable at or above this
```

then restart the service. The same settings as flags: `--parallel N`, `--side-max-len`, `--parallel-floor-gb`.

| flag | default | what it does |
|-|-|-|
| `--parallel N` | 1 | sequence slots; N - 1 side slots beside the engine |
| `--side-max-len` | 65,536 | a side slot's window; a request whose prompt plus `max_tokens` does not fit waits for slot 0 |
| `--parallel-floor-gb` | 15 | a request in flight beside others waits while its worst case would push MemAvailable under this; a side slot is built only if it fits above it |
| `--parallel-batch` | on | decoding slots verify their trees in one joint pass (off: they take turns) |
| `--parallel-draft` | on | the decoding slots' draft calls in one packed pass over the drafter |
| `--parallel-cut` | share | `share`: each slot's tree cut to a fixed share (16 rows at two slots, 10 at three, 8 at four); `joint`: the round cuts all trees together, best nodes first |
| `--parallel-max-rows` | 32 | rows of a joint pass at most, the row-invariant kernels' range |
| `--side-precapture` | off | capture a side slot's verify graphs at start (about a minute a slot) instead of at first use |

Tested up to 4 slots.

## What a slot costs

A side slot holds its own KV, recurrent state and verify graphs: about 104 KiB a token of window plus 7.5 GiB, most of it the pools of its 121 verify graphs. An 8,192-token slot took 6.9 to 7.5 GiB on the box; a 65,536-token slot is about 14 GiB (arithmetic). On a 128 GB DGX Spark next to the engine's own caches, `--parallel 2` with the default window fits; four slots need shorter windows (the measurements below used 8,192).

## Which request goes where

Slot 0 is the engine as it always was: the full window, the resident conversation, the disk tier and the side lane. A request takes a side slot only when slot 0 is busy, its prompt plus `max_tokens` fits the side window, it has no images, and it does not continue slot 0's resident conversation (that turn waits and resumes in place). Everything else about the request is served by the same code: the conductor, the monitor, tool calls, structured outputs, streaming.

## What was measured

On one DGX Spark, Qwen3.8-27B NVFP4, store off, thinking off, greedy, single runs (needs verification):

| | total tok/s |
|-|-:|
| one request (median of 4 long-answer prompts, 384 tokens) | 48.8 |
| two at once (median of the two pairs) | 67.4 (1.38x) |
| four at once | 69.2 (1.42x) |
| four at once, `--parallel-cut joint` | 76.5 (1.57x) |

One request at a time costs nothing with the flag on: in one hour, A/B/A/B, 49.29 and 49.75 tok/s without it, 50.59 with `--parallel 2` (the atlas row, `--no-learn-cost`).

Every two- and four-request window showed that many slots decoding at once and joint passes of that many blocks; none queued. A joint pass of two blocks takes 94 to 98 ms on the GPU, of four 114 to 119 ms.

After the merge into 0.4.0 (2026-10-06), one check per shape (single runs, `--parallel 4 --side-max-len 8192`, the same four prompts): one at a time 48.35 tok/s (median), two at once 65.83 in total (1.36x; 59 and 80 two-way joint rounds, 2 slots decoding at once in both pairs), four at once 67.00 (1.39x; 45 four-way and 55 three-way rounds, 4 slots decoding at once), no round timed out. With the flag off, one request at a time: 49.96 tok/s before the merge and 50.39 after (93.4 and 93.5 ms a block; the atlas row, `--no-learn-cost`, one run each).

## The text

Beside other requests, a request writes what it writes alone up to a near tie. A joint round verifies the request's tree cut to its share, a different tree shape than alone, and the shape of the verified tree can move a token whose two best candidates score almost the same ([exactness.md](exactness.md)). On 30 prompts (prose, code, chat, 256 tokens), 17 of 30 were identical to their alone text at two requests and 15 of 30 at four, every first difference at a top-2 gap of 0.0019 to 0.118 nats. Two runs of one request alone can also part at a near tie, for the same reason ([exactness.md](exactness.md#two-runs-of-the-same-request)). The rows of a joint projection are the bits they are alone (`tools/parallel_verify_check.py`).

## Seeing it

`/health` and the dashboard's System page carry `parallel` (the slots, the GPU turn, the rounds by size, the admission) only when side slots exist. The Parallel view shows each engine lane's slot. `tools/parallel_bench.py` measures c1, c2 and c4 totals, each request against its alone text, and proves from `/health` that the requests decoded together; a level whose windows never show c slots decoding at once is reported as invalid.

## Not yet

Side slots take no images, do not resume slot 0's resident prefix, and have none of slot 0's caches. Two requests reach about 1.4x, not 2x: a joint pass of 32 rows costs more than one of 16 at today's kernels.
