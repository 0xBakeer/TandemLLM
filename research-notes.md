# Inference-engine research digest (working notes)

Scope: TandemLLM (EXL3/NVFP4/FP8 on sm_89 Ada vs sm_121a DGX Spark) vs Strata, vLLM, SGLang,
TensorRT-LLM, llama.cpp, exllamav3, FlashInfer, TensorFold. Focus: quantized decode/prefill
kernel design, MoE routing, speculation, determinism.

## Strata (Niko1221/Strata) — the closest architectural cousin

Runs Qwen3.8-Flash-Next (~125B, 48 layers × 512 experts, 10/token) on a 12 GB consumer card
(RTX 5070, sm_120) + 64 GB RAM. Measured: ~87-93 tok/s decode (Q2_0), 1.3-2.6k tok/s prefill.

**Architecture**: GPU holds attention/mixers/router/shared-experts/head/MTP-draft/KV + an
*adaptive expert cache* (~4.5k slots, learns hot experts; ~72% hit vs ~50% static profile).
Uncached experts are computed **in place on the CPU** from pinned RAM (AVX-512 VNNI for Q2_0,
ggml for i-quants) concurrently with the GPU's cached ones. SSD streams a 28.8 GB n-gram
table for speculation.

**What transfers / what to copy**:
- **Determinism is a first-class contract they also fight for.** Their verify-window
  determinism issues mirror our row-invariance exactly (issue #152): the CPU computes an
  expert alone vs in a multi-token kernel and rounding differs → same prompt can differ.
  Their answer is *opt-in determinism* (`STRATA_IQ_MT_MIN`, `--pcie-frac 0`,
  `--adapt-swaps 0`, `--prompt-cache 0`) trading 1-3% speed. TandemLLM's answer is stricter:
  bitwise row-invariance as the default (decode == verify always) — we already encode it as
  "split config is a function of weight shape only, never of M" — the same rule, enforced
  rather than opted into. Both agree the mechanism is: one fixed accumulation order per
  weight across all M in the verify band.
- **`cudaHostRegister` under WSL2** (`/dev/dxg` detection): the WSL2 driver caps
  `cudaHostAlloc` pinning at ~1 GiB — they fall back to `cudaHostRegister` for the KV host
  copy. Applies directly to us since we develop under WSL2: any future KV/expert spill or
  host-pinned staging must handle this.
- **KV quantization as a lever we haven't pulled**: 8-bit KV default; `q4_0` /
  `k8v4` hybrid KV with a **Hadamard rotation before 4-bit rounding** — the same rotation
  primitive our EXL3 path already fuses. On a 16 GB card, KV q4/k8v4 is how they buy
  128K-262K context; for us it's the path to >64K on the 4080.
- **Negative result worth remembering**: overlapping CPU/GPU expert halves inside a layer
  was ~7% SLOWER — shared experts across a window's tokens matter more than the overlap;
  dense weights would be read twice. Same reasoning validates our grouped-prefill design
  (group pairs by expert, decode each trellis once) rather than pipelining per-token.
- **Quant format menu**: Q2_0 (custom int8-TC GPU + VNNI CPU), i-quants IQ2_XS/IQ3_*
  (llama.cpp dot products verbatim — *keep the GGUF layout, transcribe the dequant*),
  K-quants as a separate opt-in lib (`STRATA_MMQ_KQ`), NVFP4 experts only on RTX 50
  (older RTX → experts at 8-bit), FP8 everywhere else. Lesson: they also concluded
  FP4 weights on pre-sm_120 hardware don't pay tensor-core-wise — consistent with our
  software-e2m1-decode choice being about *format support* not speed on Ada.
- **Thread-block clusters** (sm_90+) for block-selection + argmax: not for us (sm_89),
  but the *direction* — moving router/sort/sample into fused GPU kernels — matches our
  `_moe_combine_rot`/`_router` fusion work.

## Kernel-design cross-check vs the field

| Engine | Quant decode strategy | Relevance |
|---|---|---|
| llama.cpp | per-format dequant+dot, GGUF layout preserved, `dp4a`-style int paths | baseline correctness oracle; their K-quant table structure = our trellis LUT analog |
| exllamav3 | EXL3 trellis, `shf`+`dp4a` decode (sm_121a CUDA) | our direct source; we ported this to Triton on sm_89 |
| Strata | LUT gather for i-quants; VNNI int dot for Q2_0; NVFP4 only w/ HW cvt | confirms LUT-vs-arithmetic decode tradeoff; our `nvfp4_relu2` already uses the LUT style |
| TRT-LLM | Marlin-style repacked layouts, mma pipelines, warp-specialized | their answer to "gather-bound decode": repack ONCE into mma-friendly layout — worth revisiting if trellis stays the bottleneck |
| vLLM/SGLang | FlashInfer/cutlass backends, FP8/FP4 via Marlin/Machete, CUDA-graph decode | graph capture + per-shape single-config = same invariance contract we enforce |
| FlashInfer | FA-style attention, sampling kernels, JIT | attention isn't our bottleneck |
| TensorFold | (tensor-parallel folding research) | n/a for single-GPU |

## Where TandemLLM's Ada work now stands vs these

- EXL3 decode: dp4a+shf Triton port, direct-addressed loads, fused rot_in/rot_out/silu,
  K-split by shape — ~157 tok/s e2e on LFM2.5-24B @ RTX 4080; self-check bitwise PASS.
- NVFP4: software e2m1 decode (rebiased fp16 bits, no `cvt.e2m1x2`) — v1 kernel path,
  decode tile pinned for M≤32 (row-invariant); v2/skinny stay sm_120-only defaults.
- FP8: e4m3→bf16 cvt path already works on sm_89 (bitwise at M≤16).
- `nvfp4_relu2` (Nemotron MoE experts): LUT-gather decode — was always arch-neutral.

## Open items candidates (ranked by expected value)

1. KV q4/k8v4 with Hadamard rotation (context headroom on 16 GB).
2. cudaHostRegister fallback for any future host-pinned path (WSL2 dev box).
3. Marlin-style weight repack evaluation IF `_moe_trellis_split` remains the top kernel
   after the current fusion set — the gather-bound decode may want a different layout.
4. A/B the Strata-style "accept non-determinism for speed" — we already chose strictness;
   keep strict (speculation depends on it).

## Session: MoE/dense grouping + NVFP4-on-Ada (this round)

Wins landed:
- Row-grouped dense EXL3 GEMV (G=8, bitwise-identical): verify dense ~3-3.9x.
- Head path (kb=6, W2=48 direct) grouped: bitwise-exact, 6.44 -> 1.34ms at M=24.
- Grouped MoE trellis (G=2, fused argsort+group-build in one Triton kernel):
  _moe_groups_fused 14.7us vs 104us torch chain; bitwise row-invariant.
- Fused K-combine + rotation kernels (dense combine_rot, moe_combine_rotdown,
  moe_combine_rot): -3 launches/layer, no fp32 partial round-trip; bitwise-exact.
- Ada router: FK.gemv per-row fallback -> K.linear_rows (kills 2 _to_copy +
  cublas per layer; row-invariant fp32 out).
- Fused moe_combine Triton kernel replaces ~5 torch ops; fp32 sum then one
  bf16 round to match bits.
- NVFP4 on sm_89: software e2m1 decode (LUT-free affine+rebias map),
  bitwise-exact; v2/single-tile gating made arch-aware (M-invariant bits
  preserved for the verify gate). FP8 e4m3 path verified exact as-is.
- Prefill MoE: Python pair loop -> moe_grouped_ada (grouped trellis-GEMM,
  ~5x at M=1024; order-noise acceptable for prefill-only path).

Tested-and-rejected:
- w1 masked load (off<=16): WRONG in full kernel (same-file A/B), also slower.
- Manual prefetch/double-buffer in _moe_trellis_split: wrong + slower.
- Unrolled kt loop / direct-vs-gather variants: gather stays for W2<=32.
- G=4/TT=4 grouped MoE: bitwise-correct after fixing harness MUL1 constant,
  but slower (more programs beats ~1.5 pairs/expert sharing).
- TT<8 dense grouped: TT=8/G=8 optimal.
- int64 paired-word loads: w0i parity breaks alignment.

Caution: test-kernel replicas must import real constants — a hardcoded MUL1
in /tmp replicas produced false diffs (the w1-mask "wrongness" was partly a
harness bug; the same-file A/B still showed it changes bits and is slower).

## Session: prefill MoE tuning + flow audit (this round)

- `_moe_gtrellis` launch params: nw=4/ns=2 -> nw=8/ns=1. moe_grouped_ada at M=512
  ~2x (e2e prefill ~3.4k -> ~5.9k tok/s); decode path untouched.
- Rejected: BK16 K-widening (decode-bound, slower), BM>32 at M=4096 (fewer
  programs hurts occupancy), split-K (partial traffic 768MB@KS=4 swamps),
  tl.range(num_stages=2) EVERYWHERE (consistently slower on Ada - see Triton
  issues below; SW pipelining costs registers+smem for no gain here).
- Flow audit vs 760136f: all changes sit behind `_sm_supported`/cap gates;
  conductor, spec machinery, kernels.py untouched; on sm_121a original code
  runs verbatim. Ada-only additions: exl3_ada.py, moe_ada.py, cuda_arch.py.

## Source index (papers / repos / threads consulted)

### Quantized GEMM kernel design
- MARLIN (Frantar et al., PPoPP 2025): https://arxiv.org/abs/2408.11743
  async loads + pipelined mma; near-4x quant speedup to BS 16-32. Validates
  "batched quantized stays memory-bound" - our grouped verify bet.
- MARLIN repo: https://github.com/IST-DASLab/marlin
- Machete (Neural Magic/CUTLASS 3.5): https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
  Hopper wgmma + upconvert-overlap; repack layout described in CUTE.
- vLLM Machete integration: https://github.com/vllm-project/vllm/pull/7174
  and https://github.com/vllm-project/vllm/pull/7701
- vLLM kernel menu (Marlin/AllSpark/Conch/Exllama/Triton...): https://github.com/vllm-project/vllm/blob/7c2acd38/vllm/model_executor/kernels/linear/mixed_precision/__init__.py
- QTIP (Tseng et al., NeurIPS 2024): https://arxiv.org/abs/2406.11235
  trellis-coded quant our EXL3 derives from; bitshift trellis = why decode is
  funnel-shift + bytesum.
- qtip repo: https://github.com/Cornell-RelaxML/qtip (bitshift.py codebook)
- exllamav3: https://github.com/turboderp-org/exllamav3
  exl3.py format doc: .../blob/master/doc/exl3.md ; pack.cu bitstream layout.
- vLLM MPLinearKernel capability gating: get_min_capability() pattern -
  same shape as our `_sm_supported` gates.

### MoE / grouped GEMM
- vLLM fused_moe Triton: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
  sorted_token_ids + expert_ids + padded blocks - same grouping contract.
- vLLM PR #34846 (default moe configs): batch-aware BLOCK_M 16->128,
  BLOCK_K 64/128, explicit warps/stages, up to 2x on many-expert models.
  Matches our BM=32/nw=8/ns=1 landing.
- vLLM PR #44273 (H20 tuned configs): tuning space BM{16,32,64,128}
  BN{32..256} BK{64,128,256} GROUP{1,16,32,64} nw{4,8} ns{3,4,5}.
- vLLM PR #2913 (TRT MoE small-batch): skinny-M GEMMs need different
  kernels than big-batch - same split we keep (trellis split vs gtrellis).
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
  m_grouped masked/contiguous layouts; masked GEMM exists precisely for
  CUDA-graph decode with unknown expert counts - same constraint we handle
  with fixed-shape buffers + padding.
- vLLM DeepGEMM MoE PR #13932: DeepGemm wins large-batch grouped GEMM;
  Triton wins skinny - crossover reasoning applies to our prefill group size.
- Stream-K (Osama et al.): https://arxiv.org/abs/2301.03598 - work-centric
  K decomposition; relevant if a repack/GEMM path ever replaces trellis.

### Speculative decoding / drafters
- DFlash (z-lab, arXiv:2602.06036): https://arxiv.org/pdf/2602.06036v2
  block-diffusion draft, KV-injected target features, ~6x lossless; the
  engine already ships engine/drafters/dflash2.py - flow-native lever.
- dflash repo: http://github.com/z-lab/dflash ; speculators doc:
  https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md
- EAGLE-2 (EMNLP 2024): https://arxiv.org/html/2406.16858 - context-aware
  dynamic draft trees via calibrated draft confidence, +20-40% over EAGLE-1;
  expected-accepted objective = additive path products.
- EAGLE (ICML 2024): https://arxiv.org/html/2401.15077 - feature-uncertainty
  drafting; 3.2-4.5 tokens/forward.

### KV cache quantization
- KIVI (2bit KV): https://arxiv.org/html/2402.02750 - keys per-channel,
  values per-token; 2.6x memory, 2.35-3.47x throughput.
- KVLinC: https://arxiv.org/html/2510.05373 - Hadamard rotation + linear
  correction; keys keep channel-outliers -> channel-wise.
- QuaRot (referenced in KVLinC): Hadamard before quant - same rotation
  primitive our EXL3 path already computes (rot_in/rot_out).
- SGLang PR #36172: QK block-Hadamard before FP4 KV quant (--qkrot-block)
  +0.7-3.0 acc pts back on GSM8K/GPQA/MATH - concrete recipe for our KV-FP4.
- kv-hadamard eval: https://github.com/aryxnsdfs/kv-hadamard - INT2 V +
  Hadamard ~= KIVI INT4 quality; warns blind-PPL tests never hit the cache
  path (measure through use_cache=True).
- turboquant-vllm: https://github.com/thepradip/turboquant-vllm - PolarQuant
  +QJL+KIVI+Hadamard combination for q4 KV.

### Triton/Ada behavior (why our sweep results look the way they do)
- triton-lang/triton discussion #512: num_stages = smem pipeline depth
  (A100-era async-copy design).
- triton-lang/triton issue #4906 (mobicham): indirect/gathered loads much
  slower on Ampere/Hopper vs Ada 4090; shifter content matters; Ada handles
  bitpacked gathers unusually well - consistent with our direct-load wins
  and with gather-heavy decode being competitive on the 4080.
- ROCm Triton perf guide: stages help direct-load GEMMs, hurt compute-bound
  fused kernels; register-destination pipelining spills - explains every
  failed tl.range/num_stages attempt here.
- triton.Config docs: https://triton-lang.org/main/python-api/generated/triton.Config.html

### Model/architecture
- LFM2 tech report: https://arxiv.org/html/2511.23404 - hybrid gated
  short-conv + GQA; explains conv_rows in our step profile and why
  attention is a small fraction here.
- LFM2 blog: https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
- HF LFM2 modeling: https://huggingface.co/LiquidAI/LFM2-1.2B/raw/refs%2Fpr%2F2/modeling_lfm2.py
  - causal_conv1d_update pattern our conv path mirrors.

### Determinism / verification contract
- Strata issue #152 (verify-window determinism) - covered in Strata section.
- vLLM/DeepGEMM masked-grouped-GEMM-for-graph-decode (above) - same
  fixed-shape-buffer contract.
- Our own ENG-287 row-invariance doc note in exl3_linear._reference_rows.

### Community / forums (r/LocalLLaMA, GitHub issues, blogs)
- llama.cpp PR #30517 (am17an): MMVQ low-bit GEMV rework - big on Ampere
  (1.5-1.7x), ~1-8% on Ada/Blackwell -> Ada already handles bitpacked
  gathers well; our direct-load finding agrees.
- llama.cpp bc81d47: per-quant GEMV->GEMM (MMVQ->MMQ) crossover thresholds
  on batch size - same structure as our CHUNK_UNTIL/G dispatch.
- llama.cpp PR #7716 (JohannesGaessler): mmq/dmmv/mmvq refactor, per-arch
  constexpr config tables - the "one kernel, arch-tuned configs" pattern.
- exllamav3 v0.0.31->0.0.32 log: "GEMM: output Hadamard on finished tiles
  without gmem roundtrip" (df1a969), "exl3_moe: tune kernel shape, remove
  redundant barriers", "GEMM/MGEMM autotune with disk cache", "fuse Q,G",
  "skip torch.matmul in expert path" - upstream walks the same fusion path
  we did (rot into combine, sort fusion, no torch fallbacks in hot path).
- exllamav2 issue #153: near-full-VRAM perf collapse = CUDA context
  switching cost - relevant to our 15GB ceiling on a 16GB card + WSL.
- DevQuasar spec-decode study (Llama-3-70B + 8B draft, RTX 4080):
  draft_tokens=8 sweet spot, Q4_K_M mid-quality draft best acceptance;
  per-model tuning note if we touch drafter config (flow-preserving).
- r/LocalLLaMA exllamav2-vs-llama.cpp SD thread (17h4rqz): ~2x on 70B with
  TinyLlama draft; small-draft quality sets the acceptance ceiling.
- r/LocalLLaMA MTP reports: llama.cpp MTP can HALVE throughput when draft
  quality is low despite high accept rate (halved bandwidth/util) - verify
  cost must stay cheap; our grouped verify directly addresses this.
- llamabench.ai RTX 4080 board: Qwen3.8-27B offload 6.4-123 tok/s,
  draft-mtp top ~123 - context for our ~160 tok/s on LFM2.5-24B EXL3.
- "I Built a Quantized GEMV Kernel" (vijayprabhas9.github.io): q4_K
  dequant width -> thread-count mapping lesson (idle lanes); same class of
  issue as our G=2 masked lanes at ~1.5 pairs/expert.
- Bonsai llama.cpp GB10 commit (tangled.org): GB10-specific MMQ config
  table + prefetch - sm_121a gets its own tuning tables, same as ours.
