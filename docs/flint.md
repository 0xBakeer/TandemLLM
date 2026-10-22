# Flint

Recursive's Flint is a mixture-of-experts model for coding agents: 39.8B parameters, of which about 2.5B run for each token. Support ships in release 0.14.0. TandemLLM serves it from our EXL3 packs at 4.0 and 6.0 bits a weight or from Recursive's BF16 checkpoint, and drafts with the lookup, since the checkpoint has no draft layer.

Code: `engine/models/flint/` (registry name `flint`, selected by `FlintForCausalLM` in `config.json`). Setup Guide entry: [models/flint/model.yaml](../models/flint/model.yaml). Measurements: `models/flint/measurements/2026-10-22-summary.json`.

## The model

| part | value |
|-|-|
| layers | 49: 2 dense, 47 mixture-of-experts |
| attention | 32 query heads, 4 KV heads of 128; q and k RMS-normed per head; a learned sink logit per query head (`softmax_offset`) |
| sliding layers | 35 layers see the last 513 positions, the row included, with rotary (theta 10,000); the engine keeps each in a ring of 640 rows |
| full layers | 14 layers see every position and use no positional encoding; KV 28,672 bytes a token, 7.0 GiB at 262,144 tokens |
| MLP | layers 0 and 1: dense SwiGLU, 6,144 wide. Layers 2 to 48: 256 SwiGLU experts, 512 wide, 6 a token by sigmoid score with a per-expert bias, plus one shared expert 1,024 wide |
| vocabulary | 150,144 tokens; four end tokens |
| context | 262,144 tokens |

A decode token reads 1.34 GB of the 4.0 pack and 2.00 GB of the 6.0 pack (`PackIndex.decode_bytes`).

Flint's router is sensitive to rounding. Its sigmoid scores sit close together, so one BF16 rounding of the residual stream changes the chosen experts for 2 % of tokens at layer 2 and for 46 % at layer 46. Two BF16 forwards of the same weights differ by KL 0.034 and agree on 92.9 % of top-1 on 8 rows of 1,024 tokens, and an fp32 forward against a BF16 one differs by KL 0.18. Read every KL on this page against that floor.

## Profiles

DGX Spark, 2026-10-22. Decode step: `tools/flint_check.py bench`, one request, 512 greedy tokens, from its graph. Server: one request, 512 greedy tokens of code, thinking off, plain / lookup. Quality: the converter's `model_diff` against BF16 on 50 rows of 2,048 tokens of held-out docs (written after the model's training cutoff), and this engine against the converter's BF16 forward on 8 rows of 1,024 tokens of the same docs.

| profile | on disk | weights at load | decode step, tok/s | server code, tok/s | docs KL | docs top-1 | engine KL / top-1 |
|-|-|-|-|-|-|-|-|
| `flint-exl3-4.0` (default) | 20.6 GB | 19.25 GiB | 133.5 to 134.2 | 133.2 / 139.8 | 0.194 | 79.7 % | 0.160 / 80.5 % |
| `flint-exl3-6.0` | 30.5 GB | 28.49 GiB | 97.4 to 97.7 | 97.2 / 103.2 | 0.070 | 90.8 % | 0.052 / 91.6 % |
| `flint-bf16` | 80.4 GB | 75.53 GiB | 42.3 to 42.5 | 42.4 / 50.9 | 0 | 100 % | 0.034 / 92.9 % |

4.0 is the fast size and 6.0 the one to take when quality matters more than speed: 6.0 loses KL 0.070 on the docs and 4.0 loses 0.194, and 4.0 decodes 38 % faster. The BF16 row's engine column is the rounding floor: two BF16 forwards against each other. The packs are public, [0xBakeer/TandemLLM-Flint-EXL3-4.0](https://huggingface.co/0xBakeer/TandemLLM-Flint-EXL3-4.0) and [0xBakeer/TandemLLM-Flint-EXL3-6.0](https://huggingface.co/0xBakeer/TandemLLM-Flint-EXL3-6.0); each card has its code and wikitext-2 rows. The BF16 profile reads [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) at revision `2a247be0` and keeps its window at 131,072 tokens so the Chat tab's sandbox keeps room beside it.

The EXL3 packs store every routed expert's up projection divided by 128, and `quantization_config.expert_interm_div` says so: the experts' |silu(gate) x up| reaches 260,000 in the top layers, past fp16. The engine multiplies the routing weights back.

## Reasoning, effort and tools

Flint decides on each turn whether to reason, and opens its own `<think>` block as its first token; its template has no thinking switch. Thinking off puts an empty block after the generation prompt, so the answer starts at once (SRV-685); on a hard ask the model may still work through it in the visible answer. An effort level or a token budget closes the block at its cap with the closing phrase; on the server, effort `low` closed a hard ask's block at 1,045 tokens and a 64-token budget at 89, both counted from the model's own `<think>`.

Tool calls come in Flint's own form, `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` (`tool_format` argkey, SRV-683). The server types each value by the tool's schema and writes the calls back the same way, so a turn after a tool call resumes from the end anchor.

## How the engine runs it

- The row kernels compute each row on its own, so a verify block's rows equal the decode step's bit for bit; the load-time self-check compares 24 rows.
- A sliding layer's keys live in a ring of 640 rows: the window's 512 earlier rows and the rows a verify block writes past the length, rounded up to 64. Its attention reads two 512-key chunks. A full layer reads its cache by position. The sink adds one term to the softmax's denominator.
- The decode step runs the shared expert on a second CUDA stream beside the router and the routed experts, and its Triton kernels launch as programmatic dependents (`FLINT_SIDE_STREAM`, `FLINT_PDL`, both on). Same kernels, same bits: 4.0 went from 122.4 to 134.2 tok/s.
- A prompt chunk runs on the flash kernel with the 4 KV heads as they are; the sink scales its output by sigmoid(lse - sink) in one pass. The routed experts run as one grouped GEMM over the chunk, 64 pairs an item.
- The server captures the verify graphs of widths 2 to 17 of the first context class at start, so the first requests do not pay for them.

## Speed

One request through the server, 512 greedy tokens (the game file 2,048), thinking off, one fresh server a row, one run a cell, tok/s. Every drafted text is the plain one.

| profile | | prose | code | chat | edit | copy | game file |
|-|-|-|-|-|-|-|-|
| `flint-exl3-4.0` | plain | 136.6 | 133.2 | 133.3 | 128.2 | 128.2 | 129.3 |
| | lookup | 135.4 | 139.8 | 134.0 | 142.3 | 480.0 | 149.5 |
| `flint-exl3-6.0` | plain | 98.4 | 97.2 | 97.1 | 94.3 | 94.3 | 95.0 |
| | lookup | 97.1 | 103.2 | 98.5 | 107.3 | 353.4 | 114.1 |
| `flint-bf16` | plain | 42.6 | 42.4 | 42.4 | 41.9 | 41.9 | 41.9 |
| | lookup | 42.2 | 50.9 | 42.9 | 48.4 | 144.1 | 42.5 |

Prose with the lookup is 0.5 to 0.9 % under plain on 4.0 over three alternated pairs: it drafts 7 of 340 rounds, which together cost about what they save, and each plain round carries 21 µs of host work for the drafter.

Long prompts, 4.0, in the engine, 64 greedy tokens at each point:

| context | 4k | 16k | 64k | 128k | 256k |
|-|-|-|-|-|-|
| plain decode, tok/s | 123.0 | 101.7 | 61.0 | 39.5 | 23.0 |
| prompt reading, tok/s, from the start | | 3,925 | 3,561 | 2,767 | 1,865 |

At 128k a decode step spends 17.8 of its 25.3 ms in the 14 full layers' attention: it reads 3.76 GB of keys and values at about 211 GB/s. A 4,096-token chunk at 128k spends 1,424 of its 2,180 ms there, on the flash kernel at about 87 TFLOPS. A 4,096-token prompt reads at 5,375 tok/s on 4.0, 5,045 on 6.0 and 4,319 on BF16.

## The e4m3 KV cache (`FLINT_KV_FP8`, off)

With `FLINT_KV_FP8=1` the 14 full layers keep each key and value as e4m3 codes with one fp32 scale a (head, token): 16,416 bytes a token instead of 28,672. The output is no longer the exact model's. The knob is off by default, and the exactness checks hold only with it off. The manifest names it `flint-kvfp8`, so its cache entries are never read by the exact cache.

Cost, 4.0, against the exact cache (`tools/flint_check.py kvfp8`). Each prompt is teacher-forced through 16-row verify blocks:

| prompt | KL mean | KL median | top-1 agree | perplexity, exact / e4m3 |
|-|-|-|-|-|
| converter bundle, from token 128 | 0.050 | 0.009 | 92.4 % | 26.99 / 26.94 |
| 32,768 tokens of one source file, last 1,024 | 0.070 | 0.005 | 93.8 % | 7.165 / 7.465 |
| 122,880 tokens of the same file, last 1,024 | 0.050 | 0.002 | 95.7 % | 4.053 / 4.132 |

Speed, 4.0, in the engine, 64 greedy tokens at each point:

| context | 64k | 128k | 256k |
|-|-|-|-|
| plain decode, exact cache, tok/s | 60.6 | 39.3 | 23.05 |
| plain decode, e4m3 cache, tok/s | 79.4 | 57.1 | 36.6 |
| prompt reading from the start, exact / e4m3, tok/s | 3,532 / 3,368 | 2,741 / 2,554 | 1,848 / 1,634 |

Decode reads 57 % of the bytes and runs 31 to 59 % faster. Prompt reading is 5 to 12 % slower, since every chunk decodes the cached rows to BF16. The state for a 263,168-token window drops from 26.5 to 23.0 GiB. We ran each row once. We also tried per-channel key factors (powers of two, exact on the scores) on the same prompts. They helped some rows and hurt others, and the medians did not move, so we do not ship them.

## Drafting

Flint has no draft layer, so the drafter is the lookup alone: a continuation of the request's own text, the corpus and the suffix store, cut by StairCut from the profile's price file (`ops/prices/flint/`). The price file holds the measured cost of a verify block by size for each context class, and the plain step's cost for each class. After a round with no draft, the next 1, 2, 4, then 8 rounds take the plain step without looking (`FLINT_ANCHOR_BACKOFF`, `FLINT_LOOK_BACKOFF`). A verify block reads the KV once, so the lookup pays more at long context: behind a 124,000-token prompt of code, 4.0 ran a copy ask at 79.5 tok/s against 41.3 plain and an edit ask at 129.7 against 42.4, with the same text.

Verify blocks of 1 / 2 / 4 / 8 / 16 rows on 4.0, ms: 7.84 / 9.71 / 13.13 / 17.93 / 25.64 at 2k, 16.27 / 19.24 / 23.59 / 33.39 / 52.93 at 64k, 25.12 / 28.55 / 34.56 / 48.37 / 83.43 at 128k.

## Not supported yet

- A draft head. A head we trained on the model's own outputs reached 21 % depth-1 acceptance on held-out text; StairCut's prices say it needs about 80 % for 1.4 times plain at 2k and 55 % at 128k.
- NVFP4. Our round-to-nearest NVFP4 pack lost KL 0.63 against BF16, too much to ship.

## Checks

- `tests/test_flint.py`: the config, the reference forward against a plain transcription of the model, the row kernels against their references, the prompt attention with sinks and the window, and a tiny random Flint through the engine against the reference and against itself (decode, verify blocks, commit).
- `tools/flint_check.py`: `engine` and `cmp` against a converter bundle, `bench`, `staircase` and `prices` for the price files.
- The converter's forward was checked against a plain fp32 implementation of the published model before the conversion: per layer within fp16 noise on a tiny random Flint, and all 49 real layers at cosine 0.99998 or better.
