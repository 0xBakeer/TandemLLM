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
| `--side-graphs` | on | `off`: a side slot holds no verify graphs and verifies eager when it decodes alone (about 1.5 GiB instead of 7.5 beyond its window) |
| `--side-slot-const-gb` | 7.5, or 1.5 with side graphs off | a side slot's bytes beyond its window in the build's estimate (`QSE_SIDE_SLOT_CONST_GB`) |

Tested up to 4 slots.

## What a slot costs

A side slot holds its own KV, recurrent state and verify graphs: about 104 KiB a token of window plus 7.5 GiB, most of it the pools of its 121 verify graphs. An 8,192-token slot took 6.9 to 7.5 GiB on the box; a 65,536-token slot is about 14 GiB (arithmetic). On a 128 GB DGX Spark next to the engine's own caches, `--parallel 2` with the default window fits; four slots need shorter windows (the measurements below used 8,192).

## Beside the small models

A box that also runs the decision models (Laya) and the narrator has less room for side slots. The plan for a 128 GB DGX Spark serving the 27B at its 262,144-token window, Laya-english, the narrator (Qwen3.5-2B) and the shadow reader (measured on 0.3.42-test, 2026-10-06):

| | GiB |
|-|-:|
| unified memory | 121.6 |
| the engine between requests (caches full, the 70 GiB ceiling) and its 77 GiB peak | +8.6 at the peak |
| Laya-english inside the engine | 1.6 |
| the narrator (Qwen3.5-2B) | 4.2 |
| the shadow reader (Qwen3.5-0.8B) and its session states (cap 4) | 1.5 + up to 4 |
| MemAvailable at the worst moment with one sequence | about 22.7 |

Four slots of 8,192 tokens (about 25 GiB with their graphs) do not fit beside that. Two do, with one side slot of 65,536 tokens and `--side-graphs off` (about 8 GiB). The window matters more than the number of slots: a request takes a side slot only when its prompt, its `max_tokens` and the conductor's margin (64 tokens, 8,192 more when the conductor shapes the request) fit. Over seven days on that box, opencode's prompts were 103k tokens at the median with `max_tokens` 65,536 (they always need slot 0); Open WebUI's were 19k with `max_tokens` 32,768, and a 65,536-token side window takes 57 % of them, an 8,192-token one none.

Side slots are built while the engine loads, before `ops/start.sh` asks the residency planner for Laya and the narrator. The planner then sees what the side slot took, so the `residency` document's claims must be the models' measured sizes (the reference box: decider 3.0, narrator 4.5 GiB, margin 5); with the old declared ones (8.2 and 6.0, margin 6) both are refused. A model the planner refuses is journaled in `residency.jsonl` and the engine serves without it.

While two requests run, Laya answers only when slot 0 is idle: a decision forward takes the engine lock (stage A). A request that takes a side slot arrives while slot 0 is busy, so its router question passes Laya's 400 ms and goes to the next stage, as it does for any request that arrives while the engine is busy with one sequence. The narrator serves both requests; a second request asking at the same moment gets the template line for that step.

## The memory policy and the side slots

The side slots' buffers are the engine's for good, so the engine's memory policy (engine/cache.py `Budget`) raises its ceiling and peak by what they took when they are built (7.65 GiB for one 65,536-token slot without graphs). From 0.3.44-test to 0.3.46-test the first request's residency document set the two limits again to their configured figures (70 and 77 GiB) and the raise was gone: the caches had no room left (live on 0.3.46-test: 131 anchors, 2 guest stashes, a park and a pin refused in one evening), and a guest whose stash was refused cut the resident rows at its start, so the next turn of a 41,474-token Open WebUI conversation read 41,749 tokens again (56 s). Since 0.3.47-test the slots' bytes are kept apart (`Budget.fixed`, in `/health` `cache.memory.fixed`) and added to every value the limits get. Replayed on the live engine with one build (n = 1 each): a 32,875-token conversation and two new user messages read 0 % from the cache at a 70 GiB ceiling (67.5 s and 70.7 s) and 99.9 % and 99.3 % at 78 GiB (6.4 s and 9.4 s).

## Which request goes where

Slot 0 is the engine as it always was: the full window, the resident conversation, the disk tier and the side lane. A request takes a side slot only when slot 0 is busy, its prompt plus `max_tokens` fits the side window, it has no images, and it does not continue slot 0's resident conversation (that turn waits and resumes in place). One kind of request takes an idle side slot even when slot 0 is free (0.3.47-test, SRV-207): a guest of slot 0's conversation, that is, a short request (at most `--resident-guest-max` tokens and under a quarter of the conversation) that shares less than one prefill chunk with it, while slot 0 holds at least two chunks. The router's own questions before a new user message (the class, the package: 150 to 450 tokens) and a client's title, tags and follow-up tasks are such requests. On slot 0 a guest writes over the rows the conversation's next turn resumes from and only the guest stash puts them back; on a side slot it costs the same prefill and touches nothing of slot 0. A guest short enough for the side lane takes the idle side slot too; a side-lane request that is no guest is still served at slot 0's safe points. With every side slot busy it runs on slot 0 with the stash, as before (`QWEN38_SIDE_GUESTS=off` restores that for every guest). Everything else about the request is served by the same code: the conductor, the monitor, tool calls, structured outputs, streaming.

"Continues slot 0's conversation" is judged on the client's tokens (0.3.46-test, SRV-197): at the end of every request on slot 0 whose rows stay resident, the server keeps that prompt as the client's messages render it (no router note, no replayed span, no generated row). A new request whose rendered prompt starts with all of it, within one prefill chunk, and which is at least a chunk long, waits for slot 0, also when slot 0 is held only for a moment by TandemLLM's own work (the shadow reader, a summary being prepared), which gives way to it. Until 0.3.45-test the test compared the request with the resident rows themselves; under a conductor plan those rows also hold the router's notes and the end of the last generation, so no agent turn ever counted as a continuation, and a turn that met slot 0 held by the shadow reader took the side slot and read its whole conversation again (an opencode session's cached share fell from about 0.97 to 0.4-0.68). `QWEN38_SIDE_CONTINUE=resident` restores the old test, for an A/B. And TandemLLM's own work never sends a client's request to a side slot: while slot 0 is held only by it (the shadow reader, a summary being prepared, the memory writer, the router's question), a request keeps waiting for slot 0, which that work gives up at its next safe point, for up to 15 s; a side slot is for a second client's request. Two long agent sessions at once mostly take turns on slot 0, each resuming its own conversation (the resident one in place, the other from the park or the disk): a side slot keeps no conversation, so a long agent turn there reads its whole prompt again, and the GPU it shares slows slot 0 too. Measured with two opencode sessions started together (n = 1 each, needs verification): taking turns on slot 0, both done in 301 s with cached shares 0.90 and 1.00; with one session sent to the side slot whenever slot 0 was busy, 490 s, that session at 0.26 (17 of 18 turns on the side slot, 379 s of prefill); one after the other, 193 s. A side slot pays off for requests slot 0 holds nothing for: a new conversation, a short chat, a script. The `[parallel]` log line names the slot, the numbers it was chosen on and who held slot 0; the `[req]` line and the usage ledger (schema 7, `slot`) record the slot of every request.

Sharing a client's head is not continuing a conversation (0.3.49-test, SRV-223). Open WebUI's system prompt and tool block render to about 6,700 tokens before the first user message, opencode's to about 7,400, the same for every chat of that client. While slot 0 held a conversation shorter than that head plus one chunk, a new chat of the same client shared all but a chunk of it and counted as its next turn: in the UX2 run's W18 a second Open WebUI tab asked '17 x 23' while the first wrote an essay, shared 6,675 of slot 0's 7,096 rows, and waited 35.8 s in the queue while the side slot was idle. Both tests (the client view and the resident rows) now also require the shared part to reach past the end of the request's first user message (`<|im_start|>user ... <|im_end|>` in the rendered ids, the template's own marks, not its words). A new chat then takes the idle side slot; the `[parallel]` line says `new conversation (shared head N)`. The next turn of the same chat shares its first message and everything after it and still waits for slot 0. `QWEN38_SIDE_CONV=off` is the old rule.

A side slot reads slot 0's disk tier (0.3.49-test, SRV-231): its prefill asks the tier as slot 0's does and reads a hit into its own buffers, so a new chat there starts from the client's head on the NVMe instead of prefilling it (server/diskcache.py `SideReader`). It writes nothing to the tier, counts no line and fences none of slot 0's write-behind jobs; a restore shares the tier's staging slots and locks with slot 0's. `/health` `parallel.side_disk` counts its lookups, hits and restored tokens; `QWEN38_SIDE_DISK=off` is 0.3.48-test's slot without the tier. Measured on one build with the tier switched off and on (a new Open WebUI chat on the side slot, n = 2 each): TTFT 9.6 and 9.8 s without it, 1.9 and 1.9 s with it.

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

Side slots take no images, do not resume slot 0's resident prefix, and have none of slot 0's caches in memory (they read its disk tier, nothing else). Two requests reach about 1.4x, not 2x: a joint pass of 32 rows costs more than one of 16 at today's kernels.
