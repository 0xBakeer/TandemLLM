# Caches

A long prose prompt sent twice took 44.4, 93.7 and 208.3 s to its first token at 32k, 64k and 128k tokens from cold, and 0.3, 0.6 and 1.1 s the second time, resumed from the resident prefix (the paper, one request each, published NVFP4 set). A prefill is the most expensive thing the engine does, and the caches exist so that it runs once per token of a conversation instead of once per turn.

Every cache here restores exact state, and none of them is meant to change the output. [exactness.md](exactness.md) says precisely what "exact" means for each, and what has been measured.

## What a state is

To continue a sequence at token L, the engine needs:

- the KV rows of the 16 attention layers for tokens 0 to L (64 KB a token);
- the 48 recurrent states and their convolution tails (about 147 MiB, whatever L is);
- the drafters' own position-indexed KV (about 40 KB a token for both drafters).

The recurrent part is the hard one. A KV row depends only on the tokens up to it, so rows written for an earlier request stay valid. The recurrent state has absorbed every token and cannot be rewound, so a cache must have taken a copy at L.

## The caches

| cache | what it keeps | where | default |
|-|-|-|-|
| resident prefix | the KV the last prefill wrote, in place, plus recurrent-state anchors every 1,024 tokens | device memory | on, 4 GiB of anchors |
| state store | whole state snapshots at prompt boundaries (session cache) and chunk boundaries (prefix cache) | device memory | on, 8 GiB in the served profile |
| persistent suffix store | token ids the engine read and wrote, with a suffix array | disk, outside the repository | on |
| response cache | the tokens of an identical greedy request | memory | off |

## Resident prefix

An agent client such as opencode sends one conversation that only grows: each turn's prompt is the previous prompt, plus the previous answer, plus a tool result. At 190k tokens a whole snapshot would be about 20 GB, so the state store cannot keep one.

`engine/cache.py::ResidentPrefix` uses the observation that the previous prompt's KV is still in the engine's own buffer, row for row. It leaves the KV in place and clones only the recurrent state and convolution tails (146.8 MiB) at every 1,024-token boundary of the prefill. These copies are called anchors. A new request that shares the first c tokens of the resident prompt resumes at the largest anchor at or below c. The engine copies the recurrent state back, sets the KV length, tells the drafters their caches are valid up to there, and prefills the rest.

![Resident prefix and anchors](media/caches_anchors.gif)

| first turn: cold prefill | next turn: resumed from the anchors |
|-|-|
| ![Cold prefill of a ~42k-token opencode request](media/cold_prefill.gif) | ![The next turn resumes at 99 % of the prefill](media/warm_prefill.gif) |

*The first turn of a ~42k-token opencode session prefills everything; the next turn resumes from the anchors and its prefill bar starts near the end. Recorded against the served engine (published NVFP4 weights, StairCut on); sped-up parts are marked in the clip.*

Resuming this way computes the same state as a cold prefill on the same grid. Anchors come only from prefill chunks, never from rows a verify round wrote, and the resumed request forwards the rest of its prompt on the same 1,024-token grid a cold prefill uses. `tests/test_resident.py` checks, on a small random model, that the resumed logits, recurrent state, convolution tails and KV rows equal a cold prefill's bit for bit. In the paper's long-context check on the real model, the cold and the resumed request wrote the same 256 tokens at 32k and 128k and different ones at 64k; that difference is not explained.

A short unrelated request between two turns, such as a title request, would overwrite rows the conversation needs. Such a guest request (much shorter than the resident prompt, and at most `--resident-guest-max` tokens, 8,192) gets the rows it can reach copied aside first, into a 1 GiB stash (`--resident-stash-gb`, the guest's rows and a 512-row margin), and they are copied back before the next request. The 2.5 GiB anchor budget (`--resident-gb`) holds 17 anchors, and the newest 4 are never evicted, because the next turn resumes near the end.

Grid anchors make every turn re-read the previous prompt from its last 1,024-token line (on one night of opencode traffic, 507 tokens a turn at the median for 197 new ones). Resuming at a finer line was built and measured: a forward that starts at a 64-row line inside a grid chunk does not reproduce the rows of the cold forward over that chunk on the served kernels (logits, states and KV of 4 resumed turns at 49k to 202k tokens all differed from a cold prefill's), so it is not in the engine.

A request too long to be a guest replaces the resident prompt. opencode's compaction request is one: no tools, no system text, the history as one user message, sharing nothing with the conversation, and the turn after it starts with the same tool block and system text as every turn before. Before such a request cuts the rows, the head of the old prompt is pinned: its KV rows and up to 8 anchors spread over it (the anchors every 4,096 tokens up to 32,768, which eviction keeps for this) are copied aside, as long a head as `--resident-pin-gb` holds (4 GiB with its anchors, about 33,000 tokens; its own budget, no longer taken out of the state store's). A later request that shares more with the pinned head than with the rows in place gets them copied back and resumes at the largest pinned anchor it shares. `tests/test_resident_pin.py` checks the resume against a cold prefill bit for bit.

Two sessions used in turn (A, then B, then A) share their head and nothing after it, and each switch re-read the other session past the head. When a request of another session (the body's conversation id, else opencode's `x-session-id`; without either, a prompt that parts from the rows in place) replaces the resident prompt, the rows it is about to overwrite, from where the two part up to the replaced prompt's newest anchor that fits, are parked with its newest 4 anchors (`--resident-park-gb`, 4 GiB, about 35,000 tokens past the head). A later request that shares more with the parked prompt than with the rows in place gets them back: the parked rows and the rows in place are swapped a block at a time, so B is parked while A comes back. The shared head has to be in place. A parked prompt is not pinned as well, and a compaction drops its session's parked rows instead of parking them. In a hold on 2026-10-01, A came back after B with 41,984 of 53,209 tokens reused and 11,225 re-read in 16.4 s, where it would have re-read about 45,000. `tests/test_resident_park.py` checks every swap against a cold prefill bit for bit.

### After a long generation

The rows a generation writes are never anchored, and the resident prefix holds the turn's prompt, not its answer: the next turn shares at most that prompt with it and resumes at its last anchor, so it re-reads the whole previous output. A session snapshot would hold the state after the generation, but it costs about 151 MB plus 104 KB a token, and a long session's is over the served entry cap (`STORE_ENTRY_GB=1`, about 7,000 tokens) and the whole store (`CACHE_GB=2`). Measured on 2026-10-02 (rrr2, opencode): turn 1 prefilled 34,038 tokens and thought 32,773, closed by the D2 backstop with the hidden phrase; turn 2's 66,964-token prompt resumed at 33,792 = 33 × 1,024 and re-read 33,172 tokens in about 44 s. Every one of that session's 15 turns, and the 18:08 session's, resumed at the previous prompt's last 1,024 line, whether the monitor had hidden a phrase or not, so the hidden phrase is not the cause, and replaying it at its position in the next prompt would not move the resume point. It would matter only once a state after the generation exists: a client copy without the phrase parts from it before its end, the copy with it hits (`tests/test_reread_after_think.py` shows both on the random model). Two ways to keep that state, neither built (a decision): one anchor at the end of the generation, the recurrent state only (147 MiB; the KV rows are in place), resumed under the greedy gate the session cache already uses, with the hidden phrase replayed so the client's copy matches the KV; or anchors every 1,024 generated tokens, which turns a 33k re-read into one of at most 1,024 tokens but takes a recurrent-state copy every 40 s of decoding.

## State store

`engine/cache.py::StateStore` keeps whole snapshots, keyed by the tokens that produced them. It serves two uses with one object.

The session cache keeps a conversation's state after its turn ends. The next turn starts with all those tokens and forwards only the new message. This state was written by verify rounds rather than prefill chunks, so it is the state that produced the previous answer, and not the state a cold re-read would compute. It is held to the greedy gate (same tokens) rather than bit equality.

The prefix cache takes a snapshot at the longest `--prefix-chunk` boundary of a prefill that fits the per-entry cap (`--store-boundaries`, default 1; 0 takes one at every boundary, where one cold prefill's nested snapshots fill the budget and evict everything else). A request whose prompt starts with a prefix an earlier request already read resumes at the longest snapshot they share. A shared system prompt then costs nothing from the second request on. These resumes land on the chunk grid, and `tests/test_cache.py` checks, on a small random model, that their logits equal a cold prefill's bit for bit.

A generation usually stops part way through a verified block's accepted path (at the end token, a stop or its budget), so the engine has forwarded tokens the turn never committed and the session snapshot has no key. `--session-trim on` cuts the block's commit, which is still pending when the commit is folded into the next verify, back to the committed tokens before the snapshot. It is off by default because a session resume is held to the greedy gate, not to bit equality with a cold prefill. The log line of a declined put names the reason, and `declined_big` and `declined_short` count them.

A snapshot is about 151 MB of recurrent state plus about 104 KB a token of target and drafter KV. The budget therefore counts conversations long before it counts tokens. The served budget is 2 GiB (`CACHE_GB`), and a single snapshot may take at most 1 GiB (`--store-entry-gb`, about 7,000 tokens): under opencode the store serves short chats only, and longer sessions are the resident prefix's. Room is made before a snapshot is cloned (`StateStore.reserve`), not after: at 24 GiB, a large snapshot cloned before eviction ran on top of 53 GB of engine and 40 GB of page cache, and it took the board to zero free memory.

Every hit re-checks the stored tokens element by element before restoring anything. A hash is a hint here, never an answer, because a 64-bit collision would answer one request with another request's state.

## Images in a prompt

An image's prompt rows are placeholder tokens, and the placeholder is the same token for every image. A cache keyed on the tokens would answer a request with another image's state. That is why `cache.prefill` takes `key_ids`: the prompt with each image's rows replaced by an id taken from a digest of the preprocessed patches and the grid. The state store, the resident prefix, the session state and the response cache all compare those ids, so the same URL with new pixels misses and the same pixels under another URL hit. The tower's output is cached by the same digest (`--image-cache-mb`, 512), so a conversation that sends an image on every turn, as Open WebUI does, encodes it once. The persistent suffix store never receives placeholder rows.

## The chunk size is a real trade

With the prefix cache on, every prefill runs in chunks of `--prefix-chunk` tokens, so that warm and cold runs do the same arithmetic. A chunk costs a full read of the weights. On a 1,724-token prompt a cold prefill took 2,508 ms in one piece, 2,858 ms in chunks of 1,024 and 3,940 ms in chunks of 256. The default is 1,024, because a prompt nobody shares pays the cost and gets nothing back. Drop it to 256 when many requests share one long system prompt.

## Persistent suffix store

`engine/cache.py::PersistentSuffixStore` is an append-only log of token ids that this engine has read and written, with a suffix array over it. The lookup drafter reads it beside the corpus ([speculative-decoding.md](speculative-decoding.md)), so text from an earlier session can be drafted at a 0.2 ms lookup.

It holds token ids and never text. Its directory is created with mode 0700, outside the repository, and capped by `--suffix-store-mb` (192 MiB, about 48 million tokens). Over the cap it forgets the oldest half at the next document boundary. `--suffix-store-scope outputs` keeps only what the engine wrote. The store records the tokenizer's sha256 and refuses to open with a different tokenizer; a store written before the fingerprint existed gets it recorded the next time it is opened for writing.

A request adds its prompt and its output, minus what one of the last four requests already added: a prompt that starts with an earlier prompt appends only from the point where they part, with the 8 tokens before that point in front, so every n-gram across the join is still in the store. An agent conversation sends the same growing prompt every turn; appending it whole wrote about 200 copies of one conversation in a night (397.5 million tokens for about 4.8 million new ones), rebuilt the suffix array after every turn and trimmed away the store's older content. `/health` reports the rebuilds' total time (`rebuild_ms_total`) beside the last one's.

A store that has seen a benchmark's prompts inflates that benchmark. On one release candidate it added about 21 % to the benchmark row. Benchmarks therefore run with the store off, or with a clean store that never held their text, and `--suffix-store-readonly` keeps a benchmark from writing itself into a store of real traffic ([measurement.md](measurement.md)).

## Response cache

`--response-cache` answers an identical greedy request (same prompt, same parameters) from memory. Greedy output is a function of its input, so this is memoisation and not an approximation. It is off by default, so a benchmark can never be pointed at a dictionary by accident.

## Memory

All caches share the board's 128 GB with everything else, the weights and the KV buffer first. At the 262,144-token window the engine holds 57 GiB after its load with empty caches; what grew from there to 75 and 83 GiB on 2026-10-01 was the caches at their old caps (14.5 GiB), a long prefill's workspace, and the allocator's reserve, which kept every freed block. `engine/cache.py::Budget` holds them to three limits:

- `--mem-ceiling-gib` (70): the engine between requests, allocated plus reserve. A cache grows only while the engine stays under the ceiling less the slack, and a more valuable cache may take the room of the cheaper ones (only what nothing uses).
- `--mem-peak-gib` (77): the engine during a prefill. Before a prefill, and before the response has started, the engine's reserve goes back to the driver and then the caches, cheapest first (state store, body anchors, guest stash, parked session, pinned head), until the prefill's estimated workspace fits; a prefill that still does not fit gets a 503 with Retry-After instead of running the board out of memory. The residency planner subtracts the engine's growth to this peak before any role's claim ([operations.md](operations.md)).
- `--reserve-slack-gib` (2): the free blocks the allocator may keep between requests; over it, they go back to the driver after the request, never inside one (0.3 to 54 ms for 1 to 4.8 GiB).

In a hold on 2026-10-01 the engine held 67.25 GiB with every cache filled the way an opencode night fills them, and a 261,611-token prompt peaked at 68.35 GiB allocated with nothing given back. Every limit decides only where a prefill starts; a cache given back is the cold path. The System view's Memory plan panel shows each limit with what it holds now and edits them (the conductor's `residency` document); `qse_memory_bytes` and `qse_memory_shed_total` are on `/metrics`. The operating rule stays: keep at least 15 GiB available at the worst case, and write down the worst-case bytes of a change before raising a budget.

## Routes and knobs

`GET /v1/cache/stats` reports every cache's counters. `POST /v1/cache/clear` empties them, and needs the admin token.

| knob | what |
|-|-|
| `--cache-budget-gb` (`CACHE_GB`) | the state store's budget |
| `--no-session-cache`, `--no-prefix-cache` | turn off one use of the state store |
| `--prefix-chunk` | the prefill chunk size and snapshot grid, default 1,024 |
| `--resident-gb`, `--resident-stash-gb`, `--resident-guest-max`, `--resident-tail` | the resident prefix's anchors, guest stash, longest guest and protected tail |
| `--resident-pin-gb`, `--resident-park-gb` | the pinned head, the parked session |
| `--store-entry-gb` | one state store snapshot at most |
| `--mem-ceiling-gib`, `--mem-peak-gib`, `--reserve-slack-gib` | the engine between requests, during a prefill, and the allocator's reserve |
| `--store-boundaries`, `--session-trim` | the state store's boundaries per prefill, the session snapshot's trim |
| `--prompt-token-cache` | millions of tokens of tokenised chat messages kept between requests |
| `--suffix-store`, `--suffix-store-mb`, `--suffix-store-scope`, `--suffix-store-readonly` | the persistent suffix store |
| `--response-cache`, `--response-cache-mb`, `--response-cache-ttl` | the response cache |

A cache tier on NVMe (state chunks on disk with a time to live and a 60 GB cap) is designed and not built. See the [roadmap](roadmap.md).
