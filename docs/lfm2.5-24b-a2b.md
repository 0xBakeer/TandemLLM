# LFM2.5-24B-A2B

LiquidAI's LFM2.5-24B-A2B is the third model TandemLLM runs, beside Qwen3.8-27B and Qwen3.8-Flash-Next. It has 24B parameters, and about 2B of them run for each token: a mixture of 64 experts with 4 active, in 38 of its 40 layers. It is a chat model first. On this engine it is the fast one, and its profiles trade quality for speed down to 2.5 bits a weight.

The engine's code for it is `engine/models/lfm2_moe/` (registry name `lfm2_moe`, selected by `Lfm2MoeForCausalLM` in the checkpoint's `config.json`). The Setup Guide's entry is [models/lfm2.5-24b-a2b/model.yaml](../models/lfm2.5-24b-a2b/model.yaml).

## The model

| part | what it is |
|-|-|
| layers | 40: 30 short-convolution layers and 10 attention layers (indices 2, 6, 10, ..., 38) |
| short convolution | `in_proj` from 2,048 to 3 x 2,048 (B, C, x); y = C * conv(B * x) over 3 taps, one filter a channel; `out_proj`. A sequence keeps the last 2 inputs B * x of each layer: 240 KiB |
| attention | 32 query heads, 8 KV heads of 64, q and k RMS-normed per head, rotary over the whole head (theta 1e7). KV: 20,480 bytes a token, 2.5 GiB at the full 131,072-token context |
| MLP | layers 0 and 1 dense SwiGLU (11,776 wide); layers 2 to 39 a mixture of 64 SwiGLU experts (1,536 wide), 4 a token, no shared expert |
| router | the sigmoid of each expert's logit; a per-expert bias is added to choose the 4, and the 4 chosen sigmoids are renormalised |
| head | tied to the 128,000-token embedding |

A decode token reads 4.9 GB of the BF16 weights (the pack index's `decode_bytes`: experts 2.87 GB, convolution projections 1.01, head 0.52, dense MLPs 0.29, attention 0.21). At the Spark's ~270 GB/s that is a ceiling of about 55 tokens a second in BF16, and four times that at 4 bits a weight. The weights are small next to the number of kernels a token runs (about 400), so the whole decode step is one CUDA graph.

## Profiles

| profile | weights | on disk | weights on the device |
|-|-|-|-|
| `lfm25-bf16` | Liquid's BF16 checkpoint, the quality reference | 47.9 GB | 44.95 GiB |
| `lfm25-exl3-4.0` | our EXL3 pack at 4.0 bits a weight (exllamav3 1.6.0, the mul1 codebook, the head at 6 bits) | 12.64 GB | 12.05 GiB |
| `lfm25-exl3-3.0` | the same at 3.0 bits a weight (every tensor at 3 bits, the head at 6) | 9.68 GB | 9.4 GiB |
| `lfm25-exl3-2.5`, `-3.5`, `-4.5`, `-6.0` | our EXL3 packs at those rates | 8.1 to 18.3 GB, from the parameter count until each pack is measured | |

The default is `lfm25-exl3-4.0`, and it stays provisional until every pack is measured. The 3.0 pack decodes 9 % faster (158.7 against 145.8 tok/s, the decode step from its graph) for about twice the KL divergence on chat text (0.195 against 0.098 in the conversion's own exllamav3 comparison), so 4.0 keeps the default for now.

On 8 chat prompts, 128 tokens each, the 4.0 pack's top token agrees with transformers' BF16 model at 94.7 % of the 1,024 positions, with a KL divergence of 0.033 (the engine teacher-forced on the BF16 run's tokens, `tools/lfm_check.py cmp`). That is the pack with exllamav3's default calibration, which the profile serves; a trace-calibrated 4.0 pack gave 95.6 % and 0.027 on the same tokens.

The EXL3 packs are in a private Hugging Face repository and are not public yet. Below 4 bits a weight, exllamav3 1.6.0 gives some tensors half-integer rates (1.5, 2.5 or 3.5 bits; a tile is 16 K + 8 int16 wide, and an odd weight of a tile takes one bit more than an even one). The engine's EXL3 kernels decode them as exllamav3 does (`tools/exl3_linear.py`, `dq8_frac`; the kernels name them 15, 25 and 35), in the decode step, a verify block and the prefill. The pack reader stops by name on a rate the kernels do not take.

Switch to a profile with `ops/use-profile.sh lfm25-exl3-4.0`, or choose the model in the Setup Guide (`#/setup`).

## Chat, reasoning and tools

The chat template is the checkpoint's own (`chat_template.jinja`). It opens a reasoning block (`<think>`) at the end of every prompt, and the server splits the reasoning from the answer as it does for Qwen ([server.md](server.md#reasoning)). A client that turns thinking off (`chat_template_kwargs: {"enable_thinking": false}`) gets the block closed right after the generation prompt, as the Qwen template renders thinking off, so the answer starts at once.

The model writes tool calls as a Python list between `<|tool_call_start|>` and `<|tool_call_end|>`:

    <|tool_call_start|>[get_weather(city='Berlin', days=3)]<|tool_call_end|>

`server/toolcall.py` (`PythonicCalls`) rewrites each such block into the parser's JSON form as the text arrives, so a client gets OpenAI `tool_calls`. Values are Python literals (strings, numbers, lists, dicts, True, False, None, and the JSON spellings true, false and null), read with `ast.literal_eval` and never run. A block that is not a list of calls with keyword arguments goes out as the text it was.

Liquid's recommended sampling is the server's default for this model when a request names none: temperature 0.1, top_k 50, repetition penalty 1.05 (`TEMPERATURE`, `TOP_K`, `REP_PENALTY` in the profile).

The Setup Guide serves this model with the conductor off by default ([server.md](server.md)). On the 4.0 pack, four short chat asks (who are you, a multiplication, three tips, a time-zone question in German) with the conductor's notes in the prompt (1,241 to 2,714 hidden tokens) thought 856 to 1,437 tokens, and 99 to 902 without them. With thinking off and the notes in, the model reasoned in the answer instead of answering; without them it answered at once. The conductor can still be turned on in the Setup Guide.

## How the engine runs it

The decode step and every row of a verify block run the same kernels a row at a time (`engine/models/lfm2_moe/kernels.py`): the norms, the short convolution over the row's own window, q and k with their rotary, attention over the row's visible keys, the router. EXL3 projections run on `tools/fn_kernels.dense` and the experts on its MoE bank kernels; BF16 projections on a 16-row GEMM. So a row of a verify block has the bits of the decode step at that position, and a greedy run with speculation writes the plain greedy text. A load-time self-check verifies a recorded decode as one block and compares every row bit for bit; a difference turns speculation off.

A verify block writes nothing into the convolution state. The commit takes the new state from the last two inputs along the accepted path, which are copies of the block's own rows, and moves the path's KV rows down. A rollback is exact by construction.

A prompt runs in chunks of 4,096 tokens on batched kernels (cuBLAS or the EXL3 library GEMM, the experts grouped by expert, SDPA with a lower-right causal mask). Its next token agrees with the row path; its bits do not.

## Speed

One request, greedy, the decode step from its CUDA graph, measured on one DGX Spark (`tools/lfm_check.py bench`, 256 tokens after a short prompt):

| profile | tok/s | ms a token | bytes a token | GB/s |
|-|-|-|-|-|
| `lfm25-exl3-3.0` | 158.7 | 6.30 | 1.04 GB | 165 |
| `lfm25-exl3-4.0` | 145.8 | 6.86 | 1.31 GB | 191 |
| `lfm25-bf16` | 44.7 | 22.4 | 4.92 GB | 220 |

The 4.0 row runs with the dense splits of `ops/lfm25-dense-splits.json` (`tools/fn_tiles.py --dense` at the pack's shapes); without them the step ran at 136.1 tok/s. The 3.0 pack has no splits of its own yet. The graph saves 5 % against launching the step's 460 kernels one by one. An 8,192-token prompt prefills at 7,360 tok/s on 4.0. The MoE kernels' warps and stages (4 or 8, 2 to 4) moved the 4.0 step by less than 1.3 %.

Through the server (`tools/fn_lookup_ab.py`, one fresh server a run, 512 greedy tokens, thinking off), the Setup Guide's measured rows:

| profile | prose | chat | code |
|-|-|-|-|
| `lfm25-exl3-4.0`, the lookup | 139.4 | 142.6 | 145.3 |
| `lfm25-exl3-4.0`, plain | 143.7 | 142.3 | 143.4 |
| `lfm25-bf16`, plain | 43.4 | 43.3 | 43.3 |

A verify block costs about 3 ms a row more on 4.0, because each row routes to its own experts and a block reads the union of them (`tools/lfm_check.py staircase`, chain blocks from their graphs, the median of 20 at 1k tokens of context, before the dense splits):

| rows | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|-|-|-|-|-|-|-|-|-|
| ms | 7.42 | 10.62 | 13.79 | 16.86 | 21.40 | 27.47 | 36.17 | 45.88 |
| experts read, 38 layers | 152 | 259 | 333 | 391 | 491 | 563 | 664 | 735 |

At 8k tokens of context the same blocks cost 0.5 to 3.5 ms more. A block pays only when it is accepted for about 1.4 tokens a row it adds.

## Speculation

The model has no draft layer, and Liquid will publish a drafter for it. Until then the server drafts with the lookup alone (the request's own text and the suffix store), through the shared tree drafter with no head (`engine/methods/mtp_drafter.py` `NoHead`) and cut by StairCut, priced from `ops/prices/lfm2.5-24b-a2b/lfm25-exl3-4.0.dgx-spark.json` (the staircase above). With no head, each round's own lookup match decides through the cut's calibrated acceptance: a round with no match is the plain step and is priced as one, and StairCut's class-wide skip, which a head that drafts every round needs, is not consulted.

On 4.0 with the store off, so that no earlier answer is in it, an edit ask (a file back with a name renamed) ran at 222.0 tok/s and 3.43 tokens a block, and a copy ask (a text back unchanged) at 292.2 tok/s and 10.65 tokens a block, against 141.5 and 141.6 plain; prose, chat and code ran as plain (the table above), and every text equalled the plain run's. Before the no-head pricing, the same five asks in one server never drafted: the rounds of the first three, which matched little, had priced drafting above the plain step. Through the server on 4.0, `tools/fn_server_check.py --compare` gave the same answer with the lookup and without it for all 20 requests (greedy, thinking, streaming, every `tool_choice`, JSON, penalties with `logit_bias`, logprobs, seeds, a stop string), and the largest logprob difference was 0.0.

## Parallel requests and caches

The profiles serve three sequence slots over the one copy of the weights. A slot that takes a plain step joins the other decoding slots' round as one row (`joint_plain`), and when every decoding slot does, the round is one joint decode step from a graph (`JointDecodeGraph`), so the round reads the weights once. On 4.0 with the lookup (store off), 1, 2 and 3 requests at once decoded 139.9, 169.1 and 177.8 tok/s in total (`tools/lfm_parallel.py`, 256 tokens each); the health report saw all of them decoding together, 253 rounds of the three were joint passes, and every text equalled its request alone.

The resident prefix and the state store keep the state through the adapter: 30 convolution states (240 KiB) as sections, K and V of the 10 attention layers as views, on the prefill's 4,096-token grid. `tools/fn_cache_check.py` sent a sequence of conversations to a server with its caches on and each final request to a server with them off: every answer was the same, the logprobs equal, with the prefix resumed at 4,096 and 12,288 tokens. The session spill was checked too: a stop drained the resident conversation (13,650 tokens, 0.19 GB in 309 ms), the next start read it back in 163 ms, and the next turn resumed at 12,288 tokens with the cold server's answer. The disk tier's own restore was not reached in that check, so the profiles keep the tier, and with it the spill, off.

## What is not there yet

- A draft head: Liquid's drafter plugs in through the adapter's `draft_heads()`, in place of `NoHead`.
- The disk tier's restore on this model, and with it the session spill in the profiles.
- Measured rows for the 2.5, 3.5, 4.5 and 6.0 packs, and dense splits for packs other than 4.0.
- No vision: the model reads text only.

## The checks

| check | where | result |
|-|-|-|
| the reference forward against transformers' `Lfm2MoeForCausalLM`, a tiny random model in fp32 | `tests/test_lfm2_moe_ref.py` | prefill and 3 decode steps within 2.1e-5 |
| a verify tree's rows against the sequential decode of each row's path; commit and rollback | `tests/test_lfm2_moe_ref.py`, `tests/test_lfm2_moe_engine.py` | equal (the engine: bit for bit) |
| the BF16 engine against transformers in BF16, 8 chat prompts x 128 tokens teacher-forced | `tools/lfm_check.py hf` | top token 98.05 % (1,004 of 1,024); 19 of the 20 differences near ties; KL 0.0064 |
| the 4.0 pack against exllamav3's own forward, the same tokens | `tools/lfm_check.py exl3` | top token 97.85 %; all 22 differences near ties |
| the trellis decode against exllamav3's `LinearEXL3` | `tools/lfm_check.py xcheck` | bit for bit on every tensor checked |
| a verify block's rows against the decode steps', at every server start | the load-time self-check | 24 rows bit for bit |
| the pack reader on synthetic BF16 and EXL3 files | `tests/test_lfm2_moe_pack.py` | pass |
| the row kernels against their torch twins, and their row invariance | `tests/test_lfm2_moe_kernels.py`, `tests/gpu/test_lfm2_moe_gpu.py` | pass |
| the Pythonic tool calls | `tests/test_toolcall_pythonic.py` | pass |
| half-integer rates: the reference decode against exllamav3's `LinearEXL3`, the kernels against the reference | `tests/gpu/test_lfm2_moe_gpu.py` | bit for bit at 1.5 to 8 bits; the kernels within 3e-3, rows invariant up to 32 |
| the shared kernels at whole rates against the tree before the half-rate change | the same random blocks through both trees | 51 of 51 outputs bit for bit |

A near tie is a position where the reference's two best tokens are within 8 bf16 steps of its largest logit ([exactness.md](exactness.md)).
