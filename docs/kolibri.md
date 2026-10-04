# Kolibri-1 on TandemLLM (stopped)

Status: stopped on 2026-10-04. This branch is an archive. It is not maintained and will not be merged into `main`. We stopped because the model wrote poor code in our own coding test, and the block drafter we trained for it did not reach its acceptance gate, so the speed-up we were building toward did not exist.

Everything we built for Aleph Alpha's Kolibri-1 is here: the engine (`engine/kolibri/`), the decode and verify kernels, lossless speculative decoding with StairCut, the OpenAI-compatible serving, the NVFP4 quantiser and its gate (`tools/kolibri_quant.py`, `tools/kolibri_final.py`), and the drafter training tools (`tools/kd_*.py`). The weights are [0xBakeer/TandemLLM-Kolibri-1-NVFP4](https://huggingface.co/0xBakeer/TandemLLM-Kolibri-1-NVFP4), 44.9 GB, and they load only with this branch.

## How to run it

We built and measured it on one NVIDIA DGX Spark (GB10, 128 GB shared by CPU and GPU, 273 GB/s). Install the packages from the Quick start in the [README](../README.md), then:

```bash
git clone -b kolibri-experimental https://github.com/0xBakeer/TandemLLM.git && cd TandemLLM
hf download 0xBakeer/TandemLLM-Kolibri-1-NVFP4 --local-dir ./Kolibri-1-NVFP4
python server/app.py --kolibri --port 8001 --served-model Kolibri-1 --max-len 65536 \
    --kolibri-set ./Kolibri-1-NVFP4 --kolibri-tokenizer ./Kolibri-1-NVFP4
```

Once the log prints `[kolibri] warm-up done`, the server listens. Speculation is on by default; `--kolibri-spec off` turns it off. A self-check at every load keeps it off anyway unless each verified row is bit-equal to the plain decode step. The lookup reads a corpus in Kolibri's token ids: build one with `tools/build_corpus.py --tokenizer ./Kolibri-1-NVFP4 --out ~/.kolibri-engine/corpus`. `KOLIBRI_KV_FP8=1` stores the KV cache in FP8 (e4m3, a scale per head and token); BF16 is the default. `reasoning_effort` takes `none`, `low`, `medium` or `high`.

`tools/kolibri_serve_check.py` runs client checks against a server. `python -m pytest tests/test_kolibri_*.py tests/test_kd_train.py` runs the CPU tests on a tiny random Kolibri. The GPU checks are `tools/kolibri_engine_check.py`, `tools/kolibri_spec_check.py` and `tests/gpu/test_kolibri_gpu.py`.

## Measured results

All numbers come from one DGX Spark, one stream, greedy. Rows marked n = 1 are single runs and only indicative. We did not test the cause of any gap below unless the text says so.

### Quantisation

Our set keeps the routed experts and the q and o projections of the 40 sliding-window layers in NVFP4. It keeps k and v, all attention of the 10 full-attention layers and the shared expert in FP8 with fp32 block scales. Its gate is the mean NLL difference per token, pooled per domain over about 26k tokens (chat: 5,111 generated tokens), with a bar of +0.05 nats.

| Domain | vs the FP8 release | vs BF16 |
|-|-|-|
| English prose | +0.035 | +0.021 |
| Code | +0.012 | 0.000 |
| German | +0.025 | +0.022 |
| Chat | +0.005 | +0.004 |

Everything in NVFP4 failed that bar twice. The first build lost +0.082 on the worst held-out text against BF16, and the best of four attention-only requantisations still lost +0.061. A decode token reads 2.44 GB of weights at 1k context; with all attention in FP8 it would read 2.91 GB.

### Correctness and speed

Against a plain PyTorch forward on the same dequantised weights, the engine picks the same argmax at 99.35 % of confident positions on the prefill path and 99.45 to 99.75 % on the decode path (about 2,000 positions; bar 99 %).

| Build | Decode tok/s at 1k | 8k | 32k |
|-|-|-|-|
| First working build (FP8 attention) | 39.4 | 39.0 | 35.7 |
| Fused attention, one-row kernels | 70.8 | 67.8 | 58.5 |
| Final: lane-accumulator kernels, PDL, L2 prefetch | 80.4 | 76.4 | 65.1 |

At 1k the byte ceiling is 109 to 112 tok/s, so the final build reaches 72 to 74 % of it. A prompt prefills at about 2,170 tok/s up to 32k and 1,510 tok/s on a 185k-token document. Short prompts get their first token in 0.13 to 0.24 s. For the full 262,144-token window the KV cache is 5.4 GB in BF16, because only the 10 full-attention layers keep every row; the 40 sliding layers share a 52 MB ring. FP8 KV decoded 37.2 tok/s at 131k against 32.8 for BF16 (n = 1, on an early build with the FP8-attention set), with argmax agreement 99.45 % against 99.60 %.

### Lossless speculation

Drafts come from a suffix lookup, and StairCut cuts each draft tree to the size with the most expected tokens per millisecond. Every class below produced the same tokens with speculation on and off. Kolibri is a mixture of experts, so each extra verify row brings its own experts and the verify cost rises with every row:

| Rows verified at 1k context | 1 (plain decode) | 2 | 4 | 8 | 16 | 32 |
|-|-|-|-|-|-|-|
| Time, ms | 12.5 | 16.5 | 23.2 | 35.0 | 57.9 | 101.9 |

| Text (384 tokens, n = 1) | Speed-up |
|-|-|
| Edit of a 1,112-token file | 3.27 to 3.39x |
| Verbatim copy | 3.21 to 3.33x |
| Chat with tools | 1.11x (1.16x counting the repeated tail after the answer) |
| Code | 1.04 to 1.06x |
| Prose | 0.99x |
| German | 0.97x |

Speculation pays only on edits and copies, where it gives more than 3x. On fresh text it costs 1 to 3 %.

### The block drafter

To speed up fresh text we trained a DSpark-style block drafter (5 layers, 452M parameters, taps from 5 layers of Kolibri) from scratch on 7,136 answers Kolibri wrote itself. After 3,900 steps it committed 1.47 tokens a round on 595 held-out answers at a chain of 3, against a gate of 2.0. A review of the trainer found no bug: the drafter had seen 6.6M answer tokens once, about a hundredth of what published drafters train on, and the curve was still rising slowly. Two things we measured on the way. The taps differ wildly in scale: the median row RMS grows from 0.53 after layer 1 to 28 after layer 49, and one channel reaches 805. And the engine's decode-step taps differ from the prefill taps the trainer uses, by 5 % relative RMS at layer 49 (on an RTX PRO 6000). `kd_train.py --tap-norm` addresses the first; it never ran on a GPU.

### Coding

One test, n = 1: through opencode, "create a mario game on browser" gave a minimal platformer with logic bugs. Kolibri thought for about 200 to 315 tokens a turn, and 987 at reasoning effort high. Aleph Alpha's report lists LiveCodeBench v6 85.9, HumanEval+ 92.7 and SWE-bench Verified 66.4 at effort high. We did not reproduce those numbers, and we did not test what causes the gap.
