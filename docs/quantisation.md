# Quantisation and profiles

Four-bit weights take a decode step from 26.93 GB to 15.01 GB, and on this board bytes are time. This page covers the formats the engine reads, the quantiser that writes the 4-bit copies, the gate that decides whether they ship, and the three profiles built from them.

## What the checkpoint stores

Qwen ships the model as block-scaled FP8. Every large projection is a pair of tensors: `X.weight` holds e4m3 codes of shape `[N, K]`, and `X.weight_scale_inv` holds one BF16 scale per 128 by 128 block. A weight is `code * scale[n // 128, k // 128]`.

`engine/loader.py` keeps that pair as an `FP8Block` and never expands it in memory, because a decode step pays for exactly these bytes. `tools/fp8_linear.py` multiplies by it directly. Its tile is 128 by 128 on purpose: every step along K covers one scale block, so the scale is one number applied to the fp32 sum.

Some tensors stay BF16 in the checkpoint: the output head, the embedding, every norm, the convolution weights and the two small projections `in_proj_a` and `in_proj_b`.

## NVFP4

NVFP4 stores a weight in three tensors, held by `tools/nvfp4_linear.py::NVFP4Block`:

| tensor | type | shape | meaning |
|-|-|-|-|
| `weight` | uint8 | `[N, K/2]` | two e2m1 codes a byte, low nibble first |
| `weight_scale` | e4m3 | `[N, K/16]` | one scale per 16 weights of a row |
| `weight_scale_2` | fp32 | one number | the scale of the scales |

A weight is `e2m1(code) * weight_scale[n, k // 16] * weight_scale_2`. The e2m1 grid has eight sizes (0, 0.5, 1, 1.5, 2, 3, 4 and 6) and a sign. A weight costs 0.5625 bytes against 1.0 in FP8, and one layer's MLP drops from 267.4 MB to 150.4 MB.

A group of 16 is what makes 4 bits usable. An FP8 block shares one scale across 16,384 weights, while an NVFP4 group shares it across 16, so the grid follows a row's local range.

Our kernel is W4A16: it unpacks the 4-bit codes to 16-bit numbers in registers and multiplies on the ordinary tensor cores. On this board NVFP4 is a storage format and not a compute format. It saves bytes, and the arithmetic stays the same.

## The clip search

Rounding each group to nearest fails the quality gate on code by 0.007 nats. The same weights with a clip search pass it with 0.014 to spare.

`tools/quant_nvfp4.py` tries five scale multipliers per group (1.00, 0.95, 0.90, 0.85, 0.80). A smaller multiplier clips the group's largest weight and gives the rest a finer grid. It keeps the multiplier with the smallest error as the model sees it:

    sum over k of  act_ms[k] * (w[k] - dequant(quant(w[k])))^2

`act_ms[k]` is the mean square of input channel `k` over a calibration text. A channel the model never uses can take a coarse grid at no cost, and a channel it leans on cannot. The statistics come from running `bench/calib.txt` (63 KB, two thirds Python and one third package documentation, 8,192 tokens) through the engine and recording the input of every projection. Statistics for all 64 layers take 41 s, and quantising takes 120 s.

## Which tensors, and what they cost

`tools/quant_nvfp4.py` has three targets, one output file each:

| target | projections | count | bytes saved a step |
|-|-|-|-|
| `mlp` | `gate_proj`, `up_proj`, `down_proj` | 192 | 7.49 GB |
| `gdn` | `in_proj_qkv`, `in_proj_z`, `out_proj` | 144 | 2.43 GB |
| `attn` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 64 | 0.74 GB |

Our gate scores each set against the FP8 checkpoint on held-out text, as a change in loss per token:

| weights | prose | code | code argmax, confident positions |
|-|-|-|-|
| FP8 as shipped | 1.5366 nats | 0.7900 nats | |
| NVFP4 MLPs | +0.0131 | +0.0356 | 0.9754 |
| NVFP4 MLPs and GDN | +0.0186 | +0.0436 | 0.9694 |
| NVFP4 MLPs and attention | +0.0185 | +0.0429 | 0.9748 |
| NVFP4 everywhere | +0.0211 | +0.0493 | 0.9688 |

Its bar is +0.05 nats on the worse of the two texts, and all four pass. All projections at NVFP4 pass by 0.0007 nats on a code file of 2,047 tokens, which is a thin margin. Re-run the gate before quantising anything more.

Code costs about three times what prose costs in every row. One published NVFP4 export of this model, read through the same loader (its MLPs only), scored +0.0576 on code and fails the same bar.

## From the BF16 release

The weight set above came from Qwen's FP8 release, so every weight in it was rounded twice: to e4m3 on a 128 by 128 grid by Qwen, then to NVFP4 by us. `tools/quant_nvfp4.py` also reads the BF16 release `Qwen/Qwen3.8-27B`, whose projections are plain bf16 matrices in 18 shards. `Source` finds each projection through the checkpoint's index in either layout and skips the vision tower and the MTP layer.

`quant_nvfp4.py build` loads the BF16 model once (about 50 GB) and does everything in that process:

1. It runs a calibration text through the engine. `tools/calib_corpus.py` writes it: `bench/calib.txt` plus Python standard-library modules, English Wikipedia and German Wikipedia, 149,682 tokens in the build we shipped. Any document that shares a run of 12 words with a held-out file is dropped.
2. It records each projection's mean square input, which the clip search weights by, and the full second moment of each input, `H = X^T X`. At fp32 the second moments are 101 GB for 64 layers, so the build takes them in passes of about 13 layers (`--h-budget-gb 22`).
3. It quantises every projection twice: with the clip search, and with GPTQ on the NVFP4 grid. GPTQ rounds one input column at a time and pushes each column's error onto the columns not yet rounded, weighted by the inverse of `H`. At the first column of each group of 16 the clip search picks the group's scale from the weights as they stand. The format and the kernel do not change; only the stored codes do.

On the calibration inputs, GPTQ leaves 0.0024 relative output error against 0.0044 for the clip search. On held-out text it is also the better set. Against the BF16 model, on the same tokens:

| weights | prose | code (8,002 tokens) | more code | more prose | GB a step |
|-|-|-|-|-|-|
| FP8 release, BF16 head | +0.0079 | +0.0032 | +0.0050 | +0.0028 | 26.93 |
| NVFP4 everywhere, from FP8, clip | +0.0304 | +0.0220 | +0.0530 | +0.0286 | 15.01 |
| NVFP4 everywhere, from BF16, clip | +0.0267 | +0.0198 | +0.0373 | +0.0233 | 15.01 |
| NVFP4 everywhere, from BF16, GPTQ | +0.0234 | +0.0184 | +0.0272 | +0.0174 | 15.01 |
| NVFP4 MLPs, from FP8, clip | +0.0239 | +0.0144 | +0.0372 | +0.0197 | 18.17 |
| NVFP4 MLPs, from BF16, GPTQ | +0.0136 | +0.0144 | +0.0245 | +0.0159 | 18.17 |

"More code" and "more prose" are 10,500 and 11,999 tokens that neither the calibration nor anything else reads (`calib_corpus.py --eval-chars`). `quality_gate.py --plan` scores weight sets over one checkpoint against a reference loaded from another, which is how these rows compare against BF16 while the NVFP4 files load over the FP8 release as they do in serving.

## Mixing NVFP4 and FP8

`tools/quant_sensitivity.py measure` swaps one piece of one layer to NVFP4 at a time and measures how far the next-token distribution moves (KL over the reference's top 256 tokens, on 8,192 held-out tokens). A piece is a whole fusion group: the MLP's gate and up projections, its down projection, the GDN input projections, the GDN output projection, the attention q, k and v, the attention output. Per byte kept, the GDN and attention output projections cost the most, and layers 35 to 41 are the most sensitive. `quant_sensitivity.py mix --keep-gb N` keeps the pieces with the highest cost per byte in FP8 until N GB a step are spent and writes the rest into one file.

| weights | prose | code | more code | more prose | GB a step |
|-|-|-|-|-|-|
| mix, 0.99 GB kept in FP8 | +0.0144 | +0.0145 | +0.0211 | +0.0174 | 16.00 |
| mix, 3.16 GB kept in FP8 | +0.0090 | +0.0084 | +0.0185 | +0.0132 | 18.17 |

The second mix reads the same bytes as NVFP4 MLPs alone and beats it on all four texts. The per-piece costs do not add: the 256 pieces sum to 3.4 times the cost of all of them at once. A mix is a ranking, and the gate decides what it costs.

## The gate

`tools/quality_gate.py` reports three things, and a weight set ships only when all three hold.

1. Held-out loss, teacher-forced, as a change against the FP8 engine on the same tokens in the same process. The held-out texts are `bench/heldout_code.txt`, a Python file the calibration text does not contain, and `bench/heldout_prose.txt`, English prose on unrelated topics.
2. Argmax agreement with the FP8 engine, overall and at the positions where FP8 is confident (top-1 ahead of top-2 by at least 1.0 in the logits). Overall agreement on the full set is 0.9101. On a flat distribution over 248,320 tokens the top choice is decided by rounding, so the confident number (0.9688) is the one to watch.
3. A free generation of 900 greedy tokens on five prompts the calibration never saw, with repetition statistics. On the full set the share of unique 4-grams was 0.939 to 1.000, and every generation stopped at its own end.

Test 3 matters most. Teacher-forced loss never lets an error grow, so it cannot see a model that drifts off its own path. On an earlier engine on this board, loss rated a broken configuration better than a healthy one.

## The FP8 head

As shipped, the output head is 248,320 by 5,120 in BF16, 2.54 GB, and every verify reads all of it. `tools/quant_head.py build` writes it as e4m3 with one fp32 scale per vocabulary row (1.27 GB), searching the multipliers 1.0, 0.95 and 0.90 of each row's largest weight. One scale per row, because rows are tokens and their sizes differ more than a block scale shared by 128 neighbouring ids can follow.

This head decides what the engine writes, so it took the full gate: prose +0.0015 nats, code -0.0009, and agreement 1.0000 on all 2,618 confident positions. Its kernel returns fp32 logits at every row count. `--fp8-head build` builds the same head at load, without the file.

## What stays at full precision

The 48 recurrent states stay fp32. They are 151 MB, read and written once a step, about 1.5 ms of traffic. A recurrent state is a sum over the whole sequence carried across every step, so its rounding error grows with every step, and a weight's does not. The convolution weights, `A_log`, `dt_bias`, every norm and the drafters' own layers also keep their shipped precision.

## Profiles

A profile is a `serve.env` file. The three that ship differ only in their weights. Drafters, routers, caches and the exactness gate are the same in all of them.

| profile | file | MLP | GDN and attention | head | bytes a step | single request |
|-|-|-|-|-|-|-|
| NVFP4 | `ops/serve.env` | NVFP4 | NVFP4 | e4m3 | 15.0 GB | about 44 tok/s |
| balanced | `ops/serve-balanced.env` | NVFP4 | checkpoint FP8 | e4m3 | about 18.2 GB | not measured yet |
| FP8 | `ops/serve-fp8.env` | checkpoint FP8 | checkpoint FP8 | checkpoint BF16 | 26.9 GB | 28.4 tok/s |

An FP8 profile reads nothing but the vendor's files, so its quality is the checkpoint's. Balanced keeps the recurrent and attention layers on the vendor's weights and quantises only the MLPs, which carry 70 % of a verify's projection bytes.

Profiles and bytes interact with speculation. At one token a step, the full NVFP4 set is worth 14 % over NVFP4 MLPs alone. On a 16-node tree it is worth 0.2 %, because a round that commits 13 tokens has already spread its weight read over 13 tokens. Bytes and accepted tokens substitute for each other.

## Knobs

| knob | what |
|-|-|
| `--nvfp4 a,b,c` or `QWEN38_NVFP4` | the NVFP4 files to load, one per target; empty means the checkpoint's FP8 |
| `--fp8-head FILE`, `--fp8-head build` or `QWEN38_FP8_HEAD` | the e4m3 head; empty means the checkpoint's BF16 head |
| `tools/quant_nvfp4.py quant --mode rtn\|clip --targets --stats` | the quantiser |
| `tools/quant_head.py build --ratios` | the per-row scale search for the head |
| `tools/quant_nvfp4.py build --model <BF16> --corpus --methods clip,gptq --h-budget-gb` | statistics, clip and GPTQ from one load |
| `tools/calib_corpus.py --out DIR --eval-chars` | the calibration, sensitivity and wide-gate texts |
| `tools/quant_sensitivity.py measure` and `mix --keep-gb` | the per-piece costs and the FP8/NVFP4 mix file |
| `tools/quality_gate.py --plan plan.json --corpus name=path` | weight sets against a reference checkpoint, on extra texts |

## Limits

Every gate number here comes from one held-out file per domain, 2,048 tokens each. No task benchmark ran against the quantised engine, so this page makes no claim about task accuracy. The clip search assumes that input channels are uncorrelated, which is a first-order model.
