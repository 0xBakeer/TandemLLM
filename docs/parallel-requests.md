# Parallel requests

The engine serves one request at a time unless it is told otherwise. With `--parallel N` it holds N sequences over the one copy of the weights and decodes up to N requests at once. Two or more decoding requests share each verify pass, so the weights are read once for all of them. It is off by default, because the gain is modest at today's kernels and every slot costs memory.

How it works inside (slots, the GPU turn, lockstep rounds, admission) is in [server.md](server.md), section Parallel requests. This page is about using it.

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
| `--parallel-floor-gb` | 15 | a request in flight beside others waits while its own prefill workspace, the others' and what the caches may still grow into (together at most the engine's room to `--mem-peak-gib`) would push MemAvailable under this; a side slot is built only if it fits above it. The wait is queue time (`wait_mem_ms` in the ledger) and the client sees a waiting line |
| `--parallel-batch` | on | decoding slots verify their trees in one joint pass (off: they take turns) |
| `--parallel-draft` | on | the decoding slots' draft calls in one packed pass over the drafter |
| `--parallel-cut` | share | `share`: each slot's tree cut to a fixed share (16 rows at two slots, 10 at three, 8 at four); `joint`: the round cuts all trees together, best nodes first |
| `--parallel-max-rows` | 32 | rows of a joint pass at most, the row-invariant kernels' range |
| `--side-precapture` | off | capture a side slot's verify graphs at start (about a minute a slot) instead of at first use |
| `--side-graphs` | on | `off`: a side slot holds no verify graphs and verifies eager when it decodes alone (about 1.5 GiB instead of 7.5 beyond its window) |
| `--side-slot-const-gb` | 7.5, or 1.5 with side graphs off | a side slot's bytes beyond its window in the build's estimate (`QSE_SIDE_SLOT_CONST_GB`) |

Tested up to 4 slots.

The shipped `ops/serve.env` keeps one sequence (`QSE_PARALLEL=1`). The box the measurements on these pages come from runs two (`QSE_PARALLEL=2`) with `QSE_SIDE_GRAPHS=off` and `QSE_SIDE_SLOT_CONST_GB=1.5`; every cache and memory setting it uses is the shipped one.

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

On that box one decider of three and the shadow reader stay unloaded, and that is the planner doing its job (0.4.1). The numbers from `residency.jsonl` on 0.3.51-test: MemAvailable 28.97 GiB, the engine at 76.4 GiB of a declared peak of 84.65 (77 plus the side slot's 7.65), so 8.25 GiB of growth is still to come; the shadow reader claims 6.5 GiB and would leave 14.22 GiB, under the 15 GiB floor itself; a second decider (3.0 GiB) would leave 17.72, above the floor and under floor plus the 4 GiB margin. The growth is not counted twice. The parallel admission checks MemAvailable only while two requests are in flight; a request alone is admitted by the engine's own peak (`server/slots.py` `Admission.reserve`), so a single 262k prefill can take those 8.25 GiB whatever else has loaded, and the planner's growth is the one thing that keeps the floor then. The Deciders screen says so in words: "not loaded: kept free for requests", with the numbers, and names the shadow reader when the last refusal was its. Room for them comes from a lower declared peak or a smaller margin, both edits in the `residency` document that trade cache room or safety for a model, not from the planner.

While two requests run, Laya answers only when slot 0 is idle: a decision forward takes the engine lock (stage A). A request that takes a side slot arrives while slot 0 is busy, so its router question passes Laya's 400 ms and goes to the next stage, as it does for any request that arrives while the engine is busy with one sequence. The narrator serves both requests; a second request asking at the same moment gets the template line for that step.

## The memory policy and the side slots

The side slots' buffers are the engine's for good, so the engine's memory policy (engine/cache.py `Budget`) raises its ceiling and peak by what they took when they are built (7.65 GiB for one 65,536-token slot without graphs). From 0.3.44-test to 0.3.46-test the first request's residency document set the two limits again to their configured figures (70 and 77 GiB) and the raise was gone: the caches had no room left (live on 0.3.46-test: 131 anchors, 2 guest stashes, a park and a pin refused in one evening), and a guest whose stash was refused cut the resident rows at its start, so the next turn of a 41,474-token Open WebUI conversation read 41,749 tokens again (56 s). Since 0.3.47-test the slots' bytes are kept apart (`Budget.fixed`, in `/health` `cache.memory.fixed`) and added to every value the limits get. Replayed on the live engine with one build (n = 1 each): a 32,875-token conversation and two new user messages read 0 % from the cache at a 70 GiB ceiling (67.5 s and 70.7 s) and 99.9 % and 99.3 % at 78 GiB (6.4 s and 9.4 s).

## The shared KV pool

`QSE_KV_POOL=on` (`--kv-pool on`, SRV-307, 2026-10-10). Without it every slot allocates its whole
window at load: 104.25 KiB a token, 106,752 bytes (64 KiB of engine attention KV for the 16 full-attention layers,
2 × 20 KiB of the two drafter arms' context KV, 256 B of rotary table). Four slots of 262,144 tokens
would be ~104 GiB before the first request; the box could afford one long window and three short ones.

With the pool, `engine/kvpool.py` reserves each KV tensor's full size as virtual addresses
(`cuMemAddressReserve`) and maps 2 MiB physical pages (`cuMemCreate`/`cuMemMap`, the allocation
granularity on GB10) only for the rows a conversation writes. The tensor has the static buffer's shape,
dtype and strides, so every kernel, every view and every captured CUDA graph is unchanged, and the
results are bit-identical by construction (`tests/gpu/test_kvpool_identity_gpu.py`: logits, KV rows, the
recurrent state and a greedy spec-tree decode equal at 8k, 32k and 128k).

- **Commit.** Every engine forward commits `start + rows` plus a margin (`QSE_KV_MARGIN_ROWS`, 256)
  before it runs; a generation commits its prompt first; every copy into the KV (a store restore, the
  pin, the park, the guest stash, the disk tier, the NVMe spill, a side slot's carry) commits its rows
  before it writes them. A fresh page is zeroed, as `torch.zeros` was.
- **Give back.** When a request ends, its slot keeps the rows its conversation still holds (the KV
  length, the resident prefix's valid rows, a side slot's carried end) and the pages past them go back.
- **Under pressure** a commit that would take MemAvailable under `QSE_KV_HARD_FLOOR_GB` (10) first asks
  the memory policy's levels (store, anchors, end, stash, park, pin) and the allocator's free blocks,
  then the small models (0.6.1, `server/rolemem.py`: the shadow reader, then the narrator; see below),
  then the conversations idle side slots keep (oldest first), then slot 0's idle conversation (to the
  NVMe first when the spill is on); only if none of that is enough is the commit refused. A request
  that does not fit the admission takes the same levels back, the caches left out, before it waits.
- **The small models under pressure** (0.6.1). An unloaded role gets a memory hold: its callers get
  `unloaded_for_memory` at once (no narrator lines, the template line instead; the shadow reader's
  journal-only answers stop) and nothing loads it on demand. It loads again through the residency
  planner, one role at a time and the narrator before the shadow reader, once the headroom
  (MemAvailable less the caches' growth and the requests' reservations) has stayed at
  `QSE_PARALLEL_FLOOR_GB` + its claim + `QSE_ROLE_RELOAD_MARGIN_GB` (4) for `QSE_ROLE_RELOAD_S` (60 s)
  with no request waiting and no reclaim in that time. A role unloaded again within `QSE_ROLE_THRASH_S`
  (600 s) of its reload waits twice as long next time (up to `QSE_ROLE_MAX_HOLD_S`). The Laya deciders
  stay loaded unless `QSE_ROLE_UNLOAD_LAYA=on`: they answer the router's questions, and their fallback,
  Qwen's own answer, can choose another class, another package or another effort, which changes what the model is
  given, and costs an engine turn and guest KV rows exactly while memory is short. `/health` `roles`
  and the System view's memory panel show each role as `loaded`, `unloaded_for_memory` or
  `reloading`; every unload and reload is a `[roles]` log line and a row of the residency journal.
  Qwen's output does not depend on the roles (their own memory pools and stream):
  `tests/gpu/test_roles_dynamic_gpu.py` checks tokens, blocks and prefill logits loaded, out and
  reloaded.
- **Admission** reserves, beside a request's prefill workspace, the KV pages its prompt and `max_tokens`
  may still commit on its slot, credited as its slot commits them. A request that does not fit waits
  (SRV-305): its stream carries "waiting: memory is almost full" and the rest of that line every 15 s (Open WebUI's status
  line, opencode's reasoning; the states lines are stripped from a turn sent back), the header
  `X-Tandem-Wait: memory`, the Parallel view shows the lane as `waiting_memory` and counts
  `totals.waiting_memory`; a streamed request waits up to `--request-timeout`, a non-streamed one gets a
  503 `memory_wait_timeout` with `Retry-After` after `--queue-timeout`.
- **The memory policy** no longer holds slot 0's window: its ceiling and peak drop by it (26 GiB at
  262,144), so the residency planner sees the uncommitted window as free and loads the narrator and the
  Laya decider beside four slots. `/health` `kv_pool` shows the committed pages by slot and by kind
  (`committed_by_kind_gib`: engine KV, drafter KV). The System view's memory split counts the KV as
  these committed pages, outside the allocator, never as a paged tensor's size (that is its reserved
  address range, `kv_reserved_va`, 26.6 GiB a 262,144-row slot and no memory), and adds `outside`
  (the CUDA context, the host side, other programs) so that the bar sums to MemTotal - MemAvailable.

## Which request goes where

Slot 0 is the engine as it always was: the full window, the resident conversation, the disk tier and the side lane. A request takes a side slot only when slot 0 is busy, its prompt plus `max_tokens` fits the side window, it has no images, and it does not continue slot 0's resident conversation (that turn waits and resumes in place). One kind of request takes an idle side slot even when slot 0 is free (0.3.47-test, SRV-207): a guest of slot 0's conversation, that is, a short request (at most `--resident-guest-max` tokens and under a quarter of the conversation) that shares less than one prefill chunk with it, while slot 0 holds at least two chunks. The router's own questions before a new user message (the class, the package: 150 to 450 tokens) and a client's title, tags and follow-up tasks are such requests. On slot 0 a guest writes over the rows the conversation's next turn resumes from and only the guest stash puts them back; on a side slot it costs the same prefill and touches nothing of slot 0. A guest short enough for the side lane takes the idle side slot too; a side-lane request that is no guest is still served at slot 0's safe points. With every side slot busy it runs on slot 0 with the stash, as before (`QWEN38_SIDE_GUESTS=off` restores that for every guest). Everything else about the request is served by the same code: the conductor, the monitor, tool calls, structured outputs, streaming.

"Continues slot 0's conversation" is judged on the client's tokens (0.3.46-test, SRV-197): at the end of every request on slot 0 whose rows stay resident, the server keeps that prompt as the client's messages render it (no router note, no replayed span, no generated row). A new request whose rendered prompt starts with all of it, within one prefill chunk, and which is at least a chunk long, waits for slot 0, also when slot 0 is held only for a moment by TandemLLM's own work (the shadow reader, a summary being prepared), which gives way to it. Until 0.3.45-test the test compared the request with the resident rows themselves; under a conductor plan those rows also hold the router's notes and the end of the last generation, so no agent turn ever counted as a continuation, and a turn that met slot 0 held by the shadow reader took the side slot and read its whole conversation again (an opencode session's cached share fell from about 0.97 to 0.4-0.68). `QWEN38_SIDE_CONTINUE=resident` restores the old test, for an A/B. And TandemLLM's own work never sends a client's request to a side slot: while slot 0 is held only by it (the shadow reader, a summary being prepared, the memory writer, the router's question), a request keeps waiting for slot 0, which that work gives up at its next safe point, for up to 15 s; a side slot is for a second client's request. Two long agent sessions at once mostly take turns on slot 0, each resuming its own conversation (the resident one in place, the other from the park or the disk): a side slot keeps no conversation, so a long agent turn there reads its whole prompt again, and the GPU it shares slows slot 0 too. Measured with two opencode sessions started together (n = 1 each, needs verification): taking turns on slot 0, both done in 301 s with cached shares 0.90 and 1.00; with one session sent to the side slot whenever slot 0 was busy, 490 s, that session at 0.26 (17 of 18 turns on the side slot, 379 s of prefill); one after the other, 193 s. A side slot pays off for requests slot 0 holds nothing for: a new conversation, a short chat, a script. The `[parallel]` log line names the slot, the numbers it was chosen on and who held slot 0; the `[req]` line and the usage ledger (schema 7, `slot`) record the slot of every request.

Sharing a client's head is not continuing a conversation (0.3.49-test, SRV-223). Open WebUI's system prompt and tool block render to about 6,700 tokens before the first user message, opencode's to about 7,400, the same for every chat of that client. While slot 0 held a conversation shorter than that head plus one chunk, a new chat of the same client shared all but a chunk of it and counted as its next turn: in the UX2 run's W18 a second Open WebUI tab asked '17 × 23' while the first wrote an essay, shared 6,675 of slot 0's 7,096 rows, and waited 35.8 s in the queue while the side slot was idle. Both tests (the client view and the resident rows) now also require the shared part to reach past the end of the request's first user message (`<|im_start|>user ... <|im_end|>` in the rendered ids, the template's own marks, not its words). A new chat then takes the idle side slot; the `[parallel]` line says `new conversation (shared head N)`. The next turn of the same chat shares its first message and everything after it and still waits for slot 0. `QWEN38_SIDE_CONV=off` is the old rule.

A conversation that ran on a side slot is carried into slot 0 (0.4.1, SRV-213). A new chat or agent session that starts beside a long conversation on slot 0 is a guest of it by size and runs on the idle side slot; once it outgrows the guest limit its next turn runs on slot 0, and until 0.4.1 it read everything past its client's head on the NVMe again there: 8.7k tokens in 18.3 s in UX2's O07, cached 0.44 and 17.0 s to the first token in UX1's W11 beside an 80,000-token conversation. At the end of every request on a side slot the server notes what the slot's rows hold. The conversation's next turn on slot 0 moves slot 0's conversation out the way a request of another session does (to the NVMe when it is longer than the park holds, else to the park), copies the side slot's rows and its state at the end into slot 0, about 104 KiB a token, and resumes at that end as at any end anchor, the hidden spans and Open WebUI's reasoning-less copy included. The ledger's `cache_source` is `side`. A short request unrelated to slot 0's conversation (the router's question before the next user message, a title task) leaves such a side slot alone for 10 minutes while slot 0 is free, and runs on slot 0 with the stash. `QWEN38_SIDE_CARRY=off` is 0.4.0's re-read. `tests/test_side_carry.py` checks that the carried turn writes the same greedy tokens as the conversation run on slot 0 throughout.

A side slot keeps its conversation (0.4.8-test, FX16). On 2026-10-08 two Open WebUI chats ran at once: a 64,000-token tool loop on slot 0 and a new chat on the side slot, decoding beside it. The new chat's next turn found slot 0 idle and was carried into it, which moved the long chat to the NVMe (83,160 tokens, 4.1 s); from then on neither chat could take the side slot, because the window was compared with the prompt plus the whole served `max_tokens` of 32,768 (any prompt past about 24,000 tokens was "too long"), and the two took turns on slot 0 with waits of 31 to 180 s while the side slot stood idle. Three changes. A side slot takes a request whose window leaves it at least `QWEN38_SIDE_MIN_ROOM` rows of output (16,384, the `high` thinking budget), with `max_tokens` cut to the room; the `[parallel]` line says `max_tokens N -> M (the slot's window)`. A side slot has a resident prefix of its own (`QWEN38_SIDE_RESIDENT_GB` of anchors, `QWEN38_SIDE_STASH_GB` of stash for a guest, the end anchor and the end alignment as slot 0's; no pin, park, disk tier or spill): the next turn of the conversation it holds is judged on the client's tokens against the last prompt served there and names the slot as its home (`[parallel] ... continues its conversation here (home)`), takes it before anything else, waits up to `QWEN38_SIDE_HOME_WAIT_S` while another request has it, and resumes at the end anchor there, slot 0's conversation untouched; the router's question in between is a guest of it and leaves it in place through the stash. The carry into slot 0 is the fallback (the slot busy for too long, the prompt outgrown): it copies the end anchor's own state pair, not the live state a guest wrote. And a new conversation beside one a side slot could never hold takes an idle side slot before slot 0 (`QWEN38_SIDE_NEW_FIRST`), as a guest does, so the long conversation is not evicted for it. `tests/test_side_home.py` holds that resumed turn on the side slot to the end anchor's gate.

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

## On Flash-Next

The same flags work on Flash-Next since ENG-241, and its profiles turn them on (`QSE_PARALLEL=3`, `QSE_KV_POOL=on` in `ops/profiles/_flashnext.env`, 1d4af64d). On 3.05, on fresh servers, greedy: three requests arriving together took 27.18 s against 27.03 s one after the other, the same total throughput. The first tokens came at 2.2 to 6.1 s instead of up to about 20 s for the last of three, and every answer equalled its single-stream answer. One request alone decoded at 45.78 tok/s with three slots and the pool, against 45.79 with one slot, with the same text. At the profile's default window (262,144) with two 65,536-token side slots, a 260,800-token prompt kept MemAvailable at 31.6 GiB or more (the floor is 15). So the slots buy time to first token for the second and third request, not throughput.

The limits, the serve.env defaults unless the profile sets them:

| setting | value |
|-|-|
| sequence slots (`QSE_PARALLEL`) | 3 |
| side window (`QSE_SIDE_MAX_LEN`) | 65,536, capped by `MAX_LEN` (40,960 on the reference box's :8000) |
| admission floor (`QSE_PARALLEL_FLOOR_GB`) | 15 GiB |
| joint verify | up to 32 rows |
| turn quantum, batch timeout | 50 ms, 2 ms |
| KV pool | on; hard floor 10 GiB (`QSE_KV_HARD_FLOOR_GB`), margin 256 rows, page 2 MiB |
| memory ceiling, peak, reserve slack | 66 GiB, 77 GiB, 2 GiB |

A side slot took 0.20 to 0.22 GiB allocated (0.38 to 0.40 GiB of MemAvailable) at load. Not built yet: CUDA graphs for the joint pass, and commits and drafts in one turn a round.

- KV pool (`--kv-pool on`): the K, V and indexer-key buffers of the 12 attention layers and of the MTP layer are reserved as addresses and committed as rows are written. On the server it committed 0.13, 0.91 and 4.01 GiB at 1k, 32k and 140k tokens of a 7.31 GiB window, and the answers equalled plain decode.
- Side slots: `side_slot` builds a slot over the same engine, with its own state, graphs, MTP head and drafter (StairCut from the same prices, the lookup over slot 0's corpus).
- Joint verify: `verify_multi` runs several states' blocks in one eager pass. The projections, the experts and the head run once over all rows; the norms, the mixers, the GDN recurrence and attention run per state. In one process the joint pass equalled each state's lone verify bit for bit (3 states, 12 rounds of chains and trees, 15 to 18 rows).

Measured on 3.05 with `--parallel 3 --kv-pool on` (2026-10-12, one run): three different 2,048-token requests of 384 new tokens each decoded at once, each answer equal to its single-stream answer. The server reported 3 decoding at once and ran 150 joint rounds, 106 of three and 44 of two. Together the three finished in 23.6 s by the wall clock, prefill included. Alone they took 10.2, 7.6 and 8.7 s, so one after the other 26.5 s. Beside each other each decoded at 19 to 26 tok/s, against 47.8, 67.4 and 55.3 tok/s alone. That first run predates the like-for-like check above, which timed both orders on fresh servers.

## Seeing it

`/health` and the dashboard's System page carry `parallel` (the slots, the GPU turn, the rounds by size, the admission) only when side slots exist. The Parallel view shows each engine lane's slot. `tools/parallel_bench.py` measures c1, c2 and c4 totals, each request against its alone text, and proves from `/health` that the requests decoded together; a level whose windows never show c slots decoding at once is reported as invalid.

## Not yet

Side slots take no images, do not resume slot 0's resident prefix, and have none of slot 0's caches in memory (they read its disk tier, nothing else). A conversation's turns that stay on a side slot read their prompt again each turn from the disk tier's head; only the move to slot 0 carries the rows (SRV-201 is the side slot keeping its own conversation). Two requests reach about 1.4x, not 2x: a joint pass of 32 rows costs more than one of 16 at today's kernels.
