# Caches

An agent turn at 190,000 tokens of context took 334 s to prefill from cold. With the resident prefix, the next turn of the same conversation started in 1.6 to 3.1 s (single runs, needs verification). A prefill is the most expensive thing the engine does, and the caches exist so that it runs once per token of a conversation instead of once per turn.

Every cache here restores exact state. None of them may change a token of output, and [exactness.md](exactness.md) says precisely what "exact" means for each.

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

Resuming this way is bit-identical to a cold prefill. Anchors come only from prefill chunks, never from rows a verify round wrote, and the resumed request forwards the rest of its prompt on the same 1,024-token grid a cold prefill uses. `tests/test_resident.py` checks every restored tensor for bit equality.

A short unrelated request between two turns, such as a title request, would overwrite rows the conversation needs. Such a guest request (much shorter than the resident prompt) gets the rows it can reach copied aside first, up to a 2 GiB stash (about 20,000 rows), and they are copied back before the next request. The 4 GiB anchor budget holds 27 anchors, and the newest 4 are never evicted, because the next turn resumes near the end.

## State store

`engine/cache.py::StateStore` keeps whole snapshots, keyed by the tokens that produced them. It serves two uses with one object.

The session cache keeps a conversation's state after its turn ends. The next turn starts with all those tokens and forwards only the new message. This state was written by verify rounds rather than prefill chunks, so it is the state that produced the previous answer, and not the state a cold re-read would compute. It is held to the greedy gate (same tokens) rather than bit equality.

The prefix cache takes a snapshot every `--prefix-chunk` tokens during a prefill. A request whose prompt starts with a prefix an earlier request already read resumes at the longest snapshot they share. A shared system prompt then costs nothing from the second request on. These resumes land on the chunk grid and are bit-identical.

A snapshot is about 151 MB of recurrent state plus about 104 KB a token of target and drafter KV. The budget therefore counts conversations long before it counts tokens. The served budget is 8 GiB, and a single snapshot may take at most a quarter of it. That limit is not a guess. At 24 GiB, a large snapshot cloned before eviction ran on top of 53 GB of engine and 40 GB of page cache, and it took the board to zero free memory.

Every hit re-checks the stored tokens element by element before restoring anything. A hash is a hint here, never an answer, because a 64-bit collision would answer one request with another request's state.

## Images in a prompt

An image's prompt rows are placeholder tokens, and the placeholder is the same token for every image. A cache keyed on the tokens would answer a request with another image's state. That is why `cache.prefill` takes `key_ids`: the prompt with each image's rows replaced by an id taken from a digest of the preprocessed patches and the grid. The state store, the resident prefix, the session state and the response cache all compare those ids, so the same URL with new pixels misses and the same pixels under another URL hit. The tower's output is cached by the same digest (`--image-cache-mb`, 512), so a conversation that sends an image on every turn, as Open WebUI does, encodes it once. The persistent suffix store never receives placeholder rows.

## The chunk size is a real trade

With the prefix cache on, every prefill runs in chunks of `--prefix-chunk` tokens, so that warm and cold runs do the same arithmetic. A chunk costs a full read of the weights. On a 1,724-token prompt a cold prefill took 2,508 ms in one piece, 2,858 ms in chunks of 1,024 and 3,940 ms in chunks of 256. The default is 1,024, because a prompt nobody shares pays the cost and gets nothing back. Drop it to 256 when many requests share one long system prompt.

## Persistent suffix store

`engine/cache.py::PersistentSuffixStore` is an append-only log of token ids that this engine has read and written, with a suffix array over it. The lookup drafter reads it beside the corpus ([speculative-decoding.md](speculative-decoding.md)), so text from an earlier session can be drafted at a 0.2 ms lookup.

It holds token ids and never text. Its directory is created with mode 0700, outside the repository, and capped by `--suffix-store-mb` (192 MiB, about 48 million tokens). Over the cap it forgets the oldest half at the next document boundary. `--suffix-store-scope outputs` keeps only what the engine wrote. The store records the tokenizer's sha256 and refuses to open with a different tokenizer.

A store that has seen a benchmark's prompts inflates that benchmark. On one release candidate it added about 21 % to the benchmark row. Benchmarks therefore run with the store off, or with a clean store that never held their text, and `--suffix-store-readonly` keeps a benchmark from writing itself into a store of real traffic ([measurement.md](measurement.md)).

## Response cache

`--response-cache` answers an identical greedy request (same prompt, same parameters) from memory. Greedy output is a function of its input, so this is memoisation and not an approximation. It is off by default, so a benchmark can never be pointed at a dictionary by accident.

## Memory

All caches share the board's 128 GB with everything else, the weights and the KV buffer first. The operating rule is to keep at least 15 GiB available at the worst case. Before raising a cache budget, write down the worst-case bytes of the change and check them against that floor ([operations.md](operations.md)).

## Routes and knobs

`GET /v1/cache/stats` reports every cache's counters. `POST /v1/cache/clear` empties them, and needs the admin token.

| knob | what |
|-|-|
| `--cache-budget-gb` (`CACHE_GB`) | the state store's budget |
| `--no-session-cache`, `--no-prefix-cache` | turn off one use of the state store |
| `--prefix-chunk` | the prefill chunk size and snapshot grid, default 1,024 |
| `--resident-gb`, `--resident-stash-gb`, `--resident-tail` | the resident prefix's anchors, guest stash and protected tail |
| `--suffix-store`, `--suffix-store-mb`, `--suffix-store-scope`, `--suffix-store-readonly` | the persistent suffix store |
| `--response-cache`, `--response-cache-mb`, `--response-cache-ttl` | the response cache |

A cache tier on NVMe (state chunks on disk with a time to live and a 60 GB cap) is designed and not built. See the [roadmap](roadmap.md).
