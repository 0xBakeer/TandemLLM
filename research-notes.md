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


## Quant-coverage test matrix (2026-10-19, RTX 4080 sm_89)

Support surface audited per arch:

| Arch | Loader | Quant types accepted |
|---|---|---|
| `qwen3_5` (Qwen3.8-27B) | safetensors + pack swap-ins | FP8 e4m3 (+128x128 scale), NVFP4 (load_nvfp4_mlp), EXL3 (load_exl3), BF16, FP8Head — **mixed in one checkpoint** |
| `lfm2_moe` | EXL3 pack + bf16 tensors | EXL3 any kb (2..8, frac via kb*10) + bf16 norms/embeds/conv — every pack is already mixed (body bpw + 6-bit head + bf16) |
| `qwen4_exp` (FlashNext) | EXL3 pack + Int8Rows | EXL3 + int8 mixer rows (FN_MIX) |
| `nemotron_h` | GGUF (nvidia Bolt releases) | IQ2_XXS->IQ2Linear/IQ2Bank, NVFP4->NVFP4Linear, Q8_0->Q8Linear, F32/F16/BF16->bf16; **all other GGML types refused by name** |

Test results on sm_89:

- **lfm25-exl3-2.5**: full engine, self-check PASS 24/24, ~160 tok/s warm (baseline)
- **lfm25-exl3-3.0**: full engine, loaded + generation OK, 92.8 tok/s cold (mtp drafter acc 21%)
- **lfm25-exl3-4.0**: full engine, loaded + generation OK, 58 tok/s cold, mtp acc 97% / mean-len 5.64
- **nemotron-bolt-iq2 GGUF** (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF, arch `nemotron_h_moe`,
  mix: Q8_0 x114 + NVFP4 x80 + IQ2_XXS x160 + F32 x489):
  kernel-level coverage — Q8Linear rel=3.1e-3, IQ2Linear rel=2.8e-3, NVFP4Linear matches after per-tensor
  scale (tiny QAD scale ~1.3e-4), IQ2Bank expert decode OK. Full-engine run needs >16GB VRAM (22GB file,
  loader is all-GPU) — out of reach on this box.
- **Bug found+fixed**: `iq2_linear.NVFP4Linear.matmul` gated skinny on `M<=SKINNY_MAX` only — on sm_89 it
  tried compiling the sm_121a inline extension and crashed. Now gated on `use_skinny(M)` (arch-aware).

### Gaps (infeasible on this box / unsupported)

- **OrcaCyber-Nano-1.0** (orcarouter): GGUF, Qwen-family arch — `config_from_gguf` only takes
  `nemotron_h`/`nemotron_h_moe`; refused by name. Also HF auto-gated. Supporting it = new GGUF arch
  adapter + likely new GGML types (its "Mix" files probably carry Q4_K/Q6_K families the reader rejects).
- **Qwen3.8-27B packs** (exl3-*, nvfp4): need the ~30GB FP8 base checkpoint; won't fit RAM/VRAM.
- **flashnext-***: 85GB packs. **super-vl-***: 75-97GB. **nemotron-bolt-exl3-***: 23.5GB+.
- **lfm25-bf16** (47.9GB), **lfm25-exl3-6.0** (18.6GB): exceed 16GB VRAM — loader is all-GPU.
- ~~GGUF types the reader refuses~~ — closed by the gguf_quants work below: every type the reader
  now accepts either runs a packed kernel or dequantizes at load. Still open: packed IQ-family and
  TQ/MXFP4 kernels (Tier B only today), and a Qwen-family GGUF arch adapter for files like
  OrcaCyber (the loader remains `nemotron_h`-scoped by design).

## GGUF/ggml quant coverage (2026-10-19 session)

tools/gguf_quants.py + tools/gguf_grids.py add the full ggml type table to the nemotron_h GGUF path.
Two tiers, mirroring how the codebase already treats formats:

- Tier A - packed + fused Triton kernels (PACKED_LINEAR): Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K,
  Q5_K, Q6_K - plus the pre-existing Q8_0 (Q8Linear), IQ2_XXS (IQ2Linear/IQ2Bank), NVFP4
  (NVFP4Linear) in tools/iq2_linear.py. All follow the Q8Linear contract: .shape,
  matmul(x, out=None, f32=False), bitwise row-invariant for M<=32, F.linear twin fallback above.
  Super-block scales are folded to fp16 once at init (6-bit Q4_K/Q5_K pack -> ds/dm fp16 [N,nb,8],
  Q2_K 4-bit pairs -> ds/dm fp16 [N,nb,16], Q3_K/Q6_K int6/int8 -> dl fp16 [N,nb,16]).
- Tier B - load-time dequant to bf16/fp32 via _dense -> gguf_quants.dequant: every other type in
  TYPES (Q8_1, Q8_K, IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0,
  MXFP4, BF16, I8-I64, F64). The reader BLOCKS table now lists all of them, so mixed-type and
  mixed-block-size files (QK=32 and QK=256 tensors in one GGUF) parse and dispatch per-tensor.

Correctness (RTX 4080, sm_89): every dequantizer bit-exact vs gguf-py 0.19.0 on identical random
blocks (23/23; Q8_1 and Q8_K have no gguf-py reference - ours follow the ggml source). Packed
kernels: matmul rel ~1.7-3.1e-3 vs fp32 twins (bf16 rounding noise), bitwise row-invariance,
twins oracle-exact.

Decode GEMV M=1, W [8192 x 4096], packed kernel vs bf16 twin (bytes 11-27MB vs 67MB):
Q4_0 1.5x, Q4_1 1.3x, Q5_0 1.1x, Q5_1 1.15x, Q2_K 1.8x, Q3_K 2.2x, Q4_K 2.9x, Q5_K 2.6x,
Q6_K 1.6x. The K formats win bigger because the 256-element super-block amortizes launch and
index overhead - consistent with llama.cpp putting its effort where the K-quants are.

### ggml-layout lessons applied (llama.cpp/ggml + gguf-py + Marlin lineage)

- The K-quant scale packing is a pure byte-shuffle; fold d/dmin*sc into fp16 once at load rather
  than decoding 6-bit fields per call - the kernel then reads one scalar per sub-block.
- Per-element scale gathers were the bottleneck, not the code bytes: Q2_K/Q3_K/Q6_K have two
  scales per 32-element sub-block, and replacing the [BN,32] fp16 gather with two scalar loads
  plus a t<16 select gave Q2_K 4.4x, Q3_K 2.9x, Q6_K 2.6x over the naive version - the same
  lesson as Marlin-style precomputed scale layouts (keep per-lane index math out of the loop).
- Ada/Triton launch sweet spot is again num_warps 1-2, num_stages 1-2 (measured per kernel; more
  stages hurt - consistent with the _moe_gtrellis result and Triton issue #4906).
- tools/gguf_grids.py lattice grids are generated programmatically from gguf-py grid constants
  (grid_hex/grid_map/grid_shape verbatim export) - no transcription, ggml MIT attribution kept.

## Update — full IQ lattice family fused (packed Tier A complete)

`tools/gguf_quants.py` now fuses **every IQ format**: IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
join IQ2_XXS (`tools/iq2_linear.py`), IQ4_NL and IQ4_XS. Tier A = 18 packed formats (also Q4_0,
Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K, NVFP4); Tier B dequant-on-load covers the rest of the ggml
table (Q8_1, Q8_K, TQ1_0, TQ2_0, MXFP4, integer scalars). Every dequantiser stays bit-exact vs
gguf-py 0.19.0.

Kernel recipe (extends the `_iq2_gemm` pattern in `tools/iq2_linear.py`):

- Decode a whole 256-weight block to a `[BN, 256]` tile and `tl.dot` once per block — 8-16x fewer
  dots than per-group steps, which was the dominant cost.
- Pack lattice-grid entries as one `int64` (8-elem grids) or `int32` (4-elem) load each
  (`GRID8` idiom), not 8 scalar gathers. IQ1 grids pack `magnitude + 1` since they are {-1,0,1}.
- Per-element scales fold to fp16 tables at init (IQ1_M's nibble-assembled `d * (2*sc+1)`
  becomes a `[N, nb, 16]` fp16 LUT; IQ4_NL/XS use their nonlinear LUTs directly).
- Ada launch configs re-swept: BN16/w4/s1 for the 16-group formats, BN32/w4/s1 for the
  8-group ones; measured M=1 GEMV at 4096x4096 lands ~0.7-0.9x of the bf16 twin — ALU-bound
  lattice decode vs the pure-bandwidth twin, while streaming 3-5x fewer bytes.
- Bitwise row invariance preserved: fp32 accumulate, fixed `[16, BN]` tiles, K in order.

Verified on the RTX 4080: `tests/test_gguf_quants.py` — packed-vs-twin matmul, bitwise row
invariance, twin-vs-oracle, and full-table dequant parity all pass for 17 PACKED_LINEAR entries.


## Optimization pass — packed GGUF kernels vs the RTX 4080 roofline

### Measurement
`roofline.py` (kept outside the tree at /home/user/tandem-opt): weight-stream GB/s of every packed
linear against a fixed reference of 620 GB/s, the best cuBLAS bf16 GEMV measured on this card (spec
716.8 GB/s). Each config rotates through enough independent weight copies to exceed 4x the 64 MB L2,
so the numbers are DRAM numbers rather than cache-warm ones (earlier same-weight microbenchmarks were
partly L2-resident). Widths M = 1 (decode), 9, 33 (ROWS_MAX + 1, the widest verify block).

### Diagnosis
- Only Q8_0 was near the roofline (80-89%). The others sat at 8-58%. Q8 shares the grid shape with the
  slow kernels, so the limiter was per-weight decode ALU work, not parallelism: int->float converts
  (quarter-rate on Ada), per-element scale multiplies, duplicate byte loads, global LUT gathers.
- IQ2Linear (IQ2_XXS, the Bolt production format) ran at 7-15% at M=1 and ~2% at M=9/33 because it
  reduces each row with CUDA-core `tl.sum`.
- Two correctness/perf bugs: the packed classes fell back to a numpy CPU dequant of the whole matrix
  for every prompt chunk (109 ms for one 4096x4096 Q4_K call at M=512), and a 33-row verify block
  (ROWS_MAX + 1) took the cuBLAS twin instead of the decode kernel, so decode and verify could produce
  different bits. The same cliff was in IQ2Linear / Q8Linear.

### Techniques applied, and where each came from
- **Scale after the dot.** Decode only the integer codes, dot them, then scale the [16, BN] partial
  per (sub)block, and fold mins/zero-points in through per-block sums of x. From llama.cpp's MMVQ/MMQ
  design (PR #2067, Johannes Gaessler, which quantizes x to q8_1 with a precomputed sum for exactly
  this) and from our own `_q8_gemm`.
- **Magic-number int->bf16.** OR the code into the mantissa of 128.0 and subtract, instead of I2F.
  From FasterTransformer, via Kim et al. 2022, "Who Says Elephants Can't Run" (arXiv 2211.10017);
  vLLM Marlin `dequant.h`; exllamav2 discussion #207; AWQ `dequantize.cuh`.
- **Word loads.** Load the packed bytes once as int32 words and unpack the planes with shifts. From
  llama.cpp PR #2067 (q8_1 sized so int8 pointers cast to int32).
- **Register LUTs.** The IQ4 codebooks via `tl.gather` instead of global gathers. From FLUTE (Guo et
  al., EMNLP Findings 2024, arXiv 2407.10960): LUT vectorization/duplication to avoid memory-bound
  lookups.
- **IQ lattice decode.** One load per grid entry, and the parity sign bit by popcount instead of the
  ksigns table. From llama.cpp PR #19624 (dfriehs, IQ2_XXS/XS/IQ3_XXS) and discussion #5152 (Peter
  Reid, IQ2_XS sign vectorization).
- **Per-shape configs, never per-M.** llama.cpp tunes MMVQ warps per arch and batch (PR #5394, PR
  #25479, PR #26079 with per-HW MMVQ->MMQ crossovers incl. Ada). We tune per (format, N, K) only,
  because a config that depends on M would break bitwise decode/verify equality.
- **Batch invariance as the constraint.** Thinking Machines, "Defeating Nondeterminism in LLM
  Inference" (2025): fixed reduction strategy per token, fixed split size rather than split count.
  LLM-42 (arXiv 2601.17768) notes the cost: batch-invariant GEMMs give up split-K. gemlite's GEMV
  RevSplit-K (mobiusml/dropbox) gets K-parallelism through atomics, which is non-deterministic, so we
  did not use it.
- **Bounded prefill workspace.** The prompt-chunk path now dequantizes on the GPU in N-chunks capped at
  256 MiB. Follows vllm-gguf-plugin PR #89 (estridell), which bounds the dequant+matmul fallback the
  same way.

### Results

Clean interleaved A/B (`ab_final.py`: the pre-change snapshot vs production in one process, the GPU
held busy for 0.4 s before every cell, old/new alternating per rep, 40 reps each). M = 1, three
shapes; % of 620 GB/s after the change:

| format | speedup at M=1 (4kx4k / 14kx4k / 4kx14k) | % of 620 after |
|---|---|---|
| IQ2_XXS (IQ2Linear, Ada) | 1.01 / 1.30 / 1.27; **3.5-4.7x at M=9** | 15-21% (was 4% at M=9) |
| Q4_1 | 1.18 / 1.37 / 1.03 | 22-50% |
| Q5_1 | 1.22 / 1.22 / 1.15 | 22-50% |
| Q5_0 | 1.04 / 1.22 / 1.01 | 23-47% |
| IQ4_NL | 1.19 / 1.16 / 1.13 | 19-49% |
| Q4_0 | 1.09 / 1.02 / 1.02 | 21-50% |
| Q6_K | 1.12 / 1.06 / 1.50 | 38-60% |
| Q2_K | 1.09 / 1.32 / 1.06 | 25-38% |
| Q3_K | 1.03 / 1.17 / 1.02 | 33-58% |
| Q4_K | 1.04-1.08 / 1.02 / 1.03 | 56-68% |
| IQ3_S | 1.18 / 1.00 / 1.00 | 16-21% |
| IQ3_XXS | 1.00 / 1.08 / 1.08 | 15-19% |
| IQ1_M | 1.00 / 1.00 / 1.07 | 9-12% |
| Q5_K, IQ4_XS, IQ2_XS, IQ2_S, IQ1_S | 1.00 (unchanged) | 8-76% |
| Q8_0 (reference) | 1.00 | 79-89% |

Prefill (one 4096x4096 call at M=512): **109 ms -> 0.35-0.43 ms** for every packed format (cuBLAS bf16
0.28 ms). The dequant reads the class's device tensors; no host bytes, no standing VRAM.

Verify-width fix: every packed linear, plus IQ2Linear and Q8Linear, now runs its row-invariant kernel
up to 33 rows. `test_row_invariance` checks every M = 1..33 bitwise against M = 1.

What didn't pay, measured (and kept out):
- Scale-after-dot on the K-quants and IQ lattice formats: the per-sub-block multiply after each MMA
  adds a serial dependency, and 16-element sub-blocks (Q2_K/Q3_K) doubled the iterations. 0.6-0.9x.
- Register `tl.gather` LUTs for IQ4: slower than plain L1-resident loads on Ada.
- Launch configs picked on noisy runs: three entries (Q5_K and Q6_K at 14336x4096, Q5_0 at 4096x4096)
  were 0.67-0.97x under the clean A/B and were removed.

Profiling (Nsight Compute counters are blocked in this WSL container: ERR_NVGPUCTRPERM). An ablation
stood in for it (loads kept with the decode math stripped, then decode kept with the loads made
L2-resident). Results: Q4_K/Q5_K/Q8 already sit at their load floor; IQ lattice formats are
decode/L1-gather bound (loads are 25-50% of runtime); the Q4_0 family is load-stream bound well below
DRAM speed, because of its 16-byte-per-row block layout. That points at an init-time repack of the
device layout (FLUTE-style restructuring), the clear next lever for the 32-block formats.

Measurement lesson for this card: an idle 4080 drops its memory clock to 405 MHz (11.2 GHz loaded).
Without a sustained warm-up the first cell after a pause reads 5-20x slow, and that confounded both
the baseline and early A/Bs. Heat the GPU before every timed cell.

### Update — plane-major repack for the 32-block nibble formats (1.8-2.2x)
The ablation pointed at the load stream for Q4_0/Q4_1/Q5_0/Q5_1, not the decode. Their kernels issued
byte-granular `[BN, 32]` gathers that loaded every byte twice, about one load instruction per weight.
At init each row's blocks are now regrouped 8 at a time into 32 int32 words: word j holds element j's
4-bit codes of blocks 8g..8g+7 at bits 4i ("P8 plane-major"). One loop step reads a contiguous
128-byte line a row, unpacks `[BN, 8, 32]` with shifts (the reshape to `[BN, 256]` is already the
contiguous K order, so x stays one `[16, 256]` load), folds the block scale in bf16 and does one K=256
`tl.dot`. Padded tail blocks have zero scale/min and x is zero-padded past K. The repacked tensor
replaces the original, so there is no standing VRAM cost; the CPU twin still reads `_raw`.

Warm interleaved A/B against the previous commit, 5 shapes (incl. 5120x17408 / 17408x5120) x M=1/9/33:

| format | geomean M=1 | worst cell | % of 620 GB/s at M=1 (was) |
|---|---|---|---|
| Q4_0 | 2.21x | 1.24x | 51-71% (21-50%) |
| Q4_1 | 1.89x | 1.03x | 47-63% (22-50%) |
| Q5_0 | 1.98x | 1.12x | 49-68% (23-47%) |
| Q5_1 | 1.79x | 1.00x | 43-61% (22-50%) |

The same repack lost on Q4_K/Q5_K (0.84x: they already read 32-byte runs, and the plane decode adds
register pressure) and on IQ4_NL/IQ4_XS (~0.5x: a per-element global LUT gather on the wider tile).
Precedent: FLUTE's offline restructuring of the quantized weight matrix (Guo et al., arXiv 2407.10960).

### Update — register LUTs and SWAR sign decode for the IQ family
- **IQ4_NL / IQ4_XS (2.0x / 2.15x geomean at M=1).** P8 plane-major layout, plus the 16-entry codebook
  held as four int32 words of int8 values and selected in registers (sign-extended `(v ^ 0x80) - 0x80`).
  This replaces the per-element global gather that made P8 lose in stage 4. Idea from llama.cpp's
  `get_int_from_table_16` (byte-permute table lookup for IQ4_NL in MMVQ/MMQ).
- **IQ2_XS / IQ2_S / IQ3_XXS (1.48x / 1.62x / 1.33x), IQ2_XXS on Ada (1.08x, bitwise identical to the
  previous Ada kernel).** SWAR sign application on the packed grid entry before unpacking. The sign
  bits become a byte mask with `t = (s * 0x01010101) & 0x08040201;
  M = (((t + 0x7F7F7F7F) | t) >> 7 & 0x01010101) * 0xFF`, then `(g ^ M) + (M & 0x01010101)` negates
  the marked bytes. That is valid because these grids' magnitudes are never 0, so no borrow crosses a
  byte. The parity sign bit is computed per entry, not per element. Precedent: llama.cpp discussion
  #5152 (Peter Reid's `__vcmpne4`/`__vsub4` sign vectorization for IQ2_XS) and PR #19624 (popcount
  signs, whole-entry loads).
- **IQ1_S / IQ1_M (1.32x / 1.47x).** Their grids contain 0, so SWAR doesn't apply. Gains came from the
  magic-number decode and wider loads.
- **IQ3_S rejected (0.89x geomean, 0.72x worst).** Joining its 4-bit sign fields cost more than it saved.
  Its kernel is unchanged.

All measured with the warm interleaved A/B over 5 shapes x M=1/9/33; every kept format's worst cell is
>= 1.00x.

### Cumulative — original code vs now (warm interleaved A/B, 5 shapes, M = 1 and 9)
Shapes 4096x4096, 14336x4096, 4096x14336, 5120x17408, 17408x5120. IQ2 = IQ2Linear (IQ2_XXS, Ada
path); its M=9 figure includes the split-K rewrite. Cells at 0.97-0.99x are unchanged kernels
within run-to-run noise.

| format | speedup M=1 (geomean, range) | M=9 geomean | % of 620 GB/s at M=1: before -> after |
|---|---|---|---|
| Q40 | 2.29x (1.50-2.77) | 2.35x | 21-49% -> 50-74% |
| Q41 | 2.07x (1.74-2.41) | 2.11x | 21-37% -> 47-63% |
| Q50 | 2.05x (1.75-2.40) | 2.16x | 22-39% -> 47-68% |
| Q51 | 2.01x (1.48-2.59) | 2.12x | 19-41% -> 43-61% |
| Q2K | 1.09x (0.99-1.31) | 1.09x | 20-29% -> 20-38% |
| Q3K | 1.05x (1.01-1.17) | 1.05x | 29-50% -> 29-58% |
| Q4K | 1.01x (0.99-1.03) | 1.02x | 48-67% -> 48-68% |
| Q5K | 0.99x (0.99-1.00) | 1.00x | 39-76% -> 38-76% |
| Q6K | 1.14x (0.97-1.52) | 1.15x | 33-57% -> 38-61% |
| IQ2XS | 1.49x (1.39-1.62) | 1.48x | 12-15% -> 17-24% |
| IQ3XXS | 1.37x (1.32-1.43) | 1.42x | 12-18% -> 16-24% |
| IQ1S | 1.33x (1.14-1.50) | 1.33x | 8-10% -> 9-14% |
| IQ4NL | 2.22x (1.47-2.72) | 2.27x | 17-42% -> 43-62% |
| IQ3S | 1.03x (1.00-1.19) | 1.01x | 15-21% -> 15-21% |
| IQ2S | 1.61x (1.37-2.00) | 1.26x | 9-15% -> 17-25% |
| IQ4XS | 2.17x (1.44-2.56) | 2.20x | 17-40% -> 41-58% |
| IQ1M | 1.49x (1.27-1.73) | 1.47x | 8-12% -> 13-19% |
| Q8 | 0.99x (0.98-1.01) | 1.00x | 79-89% -> 78-88% |
| IQ2 | 1.26x (0.98-1.54) | 5.00x | 14-15% -> 14-23% |

Plus the two fixes that aren't kernel speed: prefill 109 ms -> ~0.35-0.45 ms per 4096x4096
chunk, and a 33-row verify block now takes the decode kernel (bitwise row invariance).
