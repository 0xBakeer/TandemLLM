# NemotronH (Nemotron 3.5 Lightning, Bolt)

NVIDIA's NemotronH hybrid is the fourth model family TandemLLM runs: Mamba2, NoPE attention and mixture-of-experts blocks with an MTP head. The engine is built and measured on Nemotron 3.5 Lightning 30B-A3B (BF16, the test model) for Nemotron 3.5 Bolt 74B-A9B, which shares its block types and is released as a GGUF with IQ2_XXS experts.

Code: `engine/models/nemotron_h/` (registry name `nemotron_h`, selected by `NemotronHForCausalLM`, `NemotronHPuzzleForCausalLM` and the Omni architectures `NemotronH_*Omni_Reasoning_V3`). Ladder tool: `tools/nh_check.py`. IQ2_XXS: `tools/iq2_linear.py`.

## The model

| part | Lightning 30B-A3B | Puzzle 75B-A9B (Bolt's trunk design) |
|-|-|-|
| layers | 52: 23 Mamba2, 23 MoE, 6 attention | 88: 40 Mamba2, 40 MoE, 8 attention |
| hidden | 2,688 | 4,096 |
| Mamba2 | 64 heads of 64, state 128, 8 groups, conv 4 taps with bias | 128 heads of 64, state 96, 8 groups |
| attention | 32 query heads, 2 KV heads of 128, no positional encoding | the same |
| MoE | 128 experts, top 6, relu^2 experts 1,856 wide, shared expert 3,712 | 512 experts, top 4..18 and width 1,280..2,688 a layer, latent 1,024, shared 5,376 |
| router | sigmoid, correction bias for the choice, top k / (sum + 1e-20) x 2.5 | the same, x 5.0 |
| MTP | one block pair, attention + MoE | the same, top 22, width 2,688 |
| vocabulary, context | 131,072, 262,144 | the same |

The config reader (`config.py`) builds one layer table from `hybrid_override_pattern`, `layers_block_type` or `block_configs` (per-block `num_experts_per_tok`, `moe_intermediate_size` and `n_routed_experts`), and from an Omni checkpoint's `llm_config`. An unknown field stops the load with its name.

A decode token reads 6.55 GB of Lightning's BF16 weights: experts 2.75 GB, Mamba2 projections 1.78, shared experts 0.92, head 0.70, attention 0.28 (`PackIndex.decode_bytes`); 41.7 tok/s at 273 GB/s is the bytes ceiling.

## Profile

`nemotron-lightning-bf16` (`ops/profiles/nemotron-lightning-bf16.env`): the BF16 checkpoint at revision a9904d24, 65.8 GB, 59.04 GiB at load without the MTP head; 262,144 tokens; 3 slots; the lookup cut by StairCut. The checkpoint's MTP head joins the lookup with `QSE_NH_HEAD=mtp` (off in the profile, see [Speculation](#speculation)). Setup Guide entry: [models/nemotron-3.5-lightning-30b-a3b/model.yaml](../models/nemotron-3.5-lightning-30b-a3b/model.yaml), a test model (order 90, not recommended). Raw data: `models/nemotron-3.5-lightning-30b-a3b/measurements/0.9.1-summary.json`.

DGX Spark, 2026-10-15, one fresh server a run, the GPU to itself, max_len 32,768, conductor raw, suffix store off.

| measurement | plain | lookup | MTP head (depth 3) + lookup |
|-|-|-|-|
| decode step from its graph (`nh_check.py bench`, 256 tokens) | 33.63 tok/s (29.73 ms) | | |
| prose / chat / code, 512 greedy tokens, thinking off (`fn_lookup_ab.py`) | 34.14 / 33.77 / 33.97 | 33.08 / 34.13 / 36.21 | 34.17 / 34.16 / 34.15 (no drafted round: the cut chose the plain step every turn) |
| edit / copy asks | 33.85 / 33.74 | 105.77 / 140.74 (9.83 / 14.19 tokens a block) | 34.01 / 34.06 |
| `fn_server_check.py` set (thinking on, tools, JSON), per request | 32.3-33.9 tok/s | 1.0-1.28 tokens a block | 34.2-53.3 tok/s, 1.0-4.3 tokens a block |
| 1 / 2 / 3 requests at once, 256 tokens each, total (`lfm_parallel.py`, lookup drafter) | | 33.73 / 44.73 / 50.65 tok/s, 2 and 3 slots decoding together (254 joint rounds), each text its request's alone | |

- Same weights in llama.cpp (5ecbe1ac1, `llama-bench -ngl 99 -p 0 -n 256`, the BF16 GGUF of ggml-org at 8a08a1c8, same board): 33.31 tok/s against the engine's 33.63.
- Every answer of the lookup and the MTP runs is the plain run's text. Through `fn_server_check.py`, 15 of 20 MTP answers equal the plain ones; the other 5 differ only from the reasoning budget's close on, which lands at the end of the verified block (2 to 3 tokens later than the plain step's), one of the output changes docs/exactness.md lists.
- The MTP head runs eagerly (no graph) and reads the 0.70 GB head once a depth; with thinking off StairCut priced its rounds above the plain step and took plain steps. Not the profile's default until its rounds are captured and priced.
- Prefill (`nh_check.py staircase`, one expert GEMM pair an expert): 545 tok/s at 1,024 tokens, 2,537 at 8,192. Two batched GEMMs padded to the busiest expert measured 359 and 1,248 and were dropped.

## Arithmetic

The reference forward (`ref.py`) follows transformers' NemotronH and the CUDA path the checkpoints were run with (mamba_ssm's `selective_state_update`, `causal_conv1d_update`, `rmsnorm_fn`):

- RMS norm in fp32 times the plain weight; the gated norm `norm_before_gate=False`: y silu(z) in fp32, RMS over groups of 512 (inner / 8), times the weight, one rounding.
- dt = softplus(dt + dt_bias), no clamp (`time_step_limit` [0, inf]; transformers' torch path clamps at `time_step_min`, its CUDA path does not).
- The SSM state is fp32 (`mamba_ssm_cache_dtype`), y = S C + D x rounded to bf16.
- Router: fp32 logits, sigmoid, the bias for the choice only, ties to the lower id.
- Experts: bf16(relu(bf16(up x))^2), down; the weighted sum in fp32 in ascending expert id, one rounding.

`tests/test_nemotron_h_ref.py` holds it to transformers in fp32 within 2e-5 on three tiny models (Lightning-like, latent, Puzzle per-layer). transformers' multi-chunk torch SSD disagrees with its own one-chunk result by 3e-2 on these weights; the references run one chunk a sequence.

## How the engine runs it

- Row path: every projection on 16-row tiles, the Mamba2 conv over the row's window, the SSM update walked from the committed state along the row's path in fp32, the gated norm, attention over fixed key chunks, the router, relu^2 experts and the ordered combine: a row of a verify block is the decode step's row bit for bit.
- Verify writes nothing but its KV rows. A commit walks each SSM state along the kept path with the decode step's own kernel (the same products in the same order) and takes the conv state from the path's last three inputs: the committed state is the decode steps' state bit for bit.
- Prefill: 4,096-token chunks, the chunked SSD form in fp32 matmuls (chunk 128), SDPA, one GEMM pair an expert. The next token agrees with the row path; the bits do not.
- CUDA graphs: the decode step per attention class, verify blocks, commits, joint decode steps of the side slots.
- Load-time self-check: a 24-token decode replayed as one verify block, every row and the committed states compared bit for bit; a difference turns speculation off.
- Caches: sections `mamba.conv.<L>` (bf16 [3, conv_dim]) and `mamba.S.<L>` (fp32 [heads, 64, state]), views K and V. Lightning: 46.8 MiB of sections a sequence, 6,144 bytes of KV a token.

## Weights

| source | reader | notes |
|-|-|-|
| BF16 safetensors | `pack.py`, `load.py` | `backbone.` and an Omni wrapper's `language_model.` stripped, tower tensors skipped; experts stacked; page cache dropped while loading |
| GGUF (`nemotron_h_moe`) | `gguf.py`, `gguf_load.py` | config from the metadata (per-layer expert counts from the router's rows); IQ2_XXS weights and expert banks kept packed; Q8_0 dequantised at load; other types refused by name |

IQ2_XXS: 256 weights in 66 bytes, w = d (0.5 + s) / 4 x grid x sign. The twin equals gguf-py's dequant bit for bit; the GEMM and the expert bank decode blocks on the fly and are row-invariant (`tests/test_iq2_kernels.py`). IQ2_XXS needs K a multiple of 256: Lightning's 2,688 and 1,856 are not, so its community IQ2 GGUFs hold IQ4_NL / Q5_0 experts. Bolt's Puzzle widths 1,280..2,048 and latent 1,024 are multiples; its 2,688-wide down projections are not.

## Speculation

- The lookup (the request's text, the corpus, the suffix store) cut by StairCut through the shared tree drafter, from the staircase's prices (`ops/prices/nemotron-3.5-lightning-30b-a3b/`).
- The checkpoint's MTP head (`mtp.py`, `--draft-head mtp`): one native depth, deeper drafts iterate the head on its own final normed row and the drafted token; its attention KV is a cache view. On Lightning BF16 (`nh_check.py mtp`, 8 chat prompts, 1,000 positions, depth 3): the head's draft equals the trunk's greedy token at depth 1, 2 and 3 in 91.9, 80.6 and 69.5 % of positions; mean accepted run 2.38 of 3.
- Verify cost (chain, `nh_check.py staircase`, 1k context): 1, 2, 4, 8 and 16 rows take 29.5, 37.0, 51.6, 74.8 and 103.4 ms.

## Checks

| check | result |
|-|-|
| config: the four published configs | `tests/test_nemotron_h_config.py` |
| reference vs transformers fp32 | within 2e-5, three tiny models |
| kernels: chain rows = decode steps, commit = decode state | bit for bit, GB10 and the CPU interpreter |
| engine: verify = decode, tree, moved-path commit, rollback, joint passes, graphs | 8 tests, GB10 |
| GGUF: tiny IQ2_XXS / Q8_0 file through the engine | self-check bit for bit on the packed kernels |
| Lightning BF16 against transformers BF16, teacher-forced (`nh_check.py hf --shared`, 8 chat prompts x 128 tokens) | top-1 98.63 % of 1,024 positions; all 14 differences near ties (margin under 8 bf16 steps), none other; mean relative logit difference 0.0075 |
| Lightning BF16 load-time self-check | PASS, 24 rows and the committed states |
| server: plain vs lookup, 20 requests (greedy, streaming, tools, JSON, penalties, seeds, stop) | every answer the same |
| caches (`fn_cache_check.py`): two sessions parked and restored, a 10k document twice, against cold answers | every answer the same token for token; logprobs within 0.07 where the end anchor restored (the tail's prefill chunks differ from the cold prompt's) |
