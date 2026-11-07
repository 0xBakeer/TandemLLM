# TandemLLM kernel gauntlet — Ada/sm_89 optimization pass

Thirteen-commit optimization package on `gauntlet-opt`. Real-pack gate: LFM2.5-24B-A2B
EXL3 2.5bpw on an RTX 4080 (`sm_89`), `self_check` bit-exact. Decode-step CUDA time
**4659us -> ~3255us (-30%)**; verify+commit at M=16 **23842us -> 19378us (-19%)**
with per-commit cache writes **190us -> 2.6us (~73x)**. Original conductor/speculative
flow untouched — every change is inside kernel dispatch and CUDA kernels; the same
rows go through the same conductor, drafting, verification and acceptance paths.

## Commits

| commit | change |
|---|---|
| `45b5a914` | Kernels: hoist repeated packed-field loads in the Q2_K/Q3_K GEMMs |
| `0f0343d9` | MoE: masked-load + gather split kernels behind `QWEN38_MOE_GATHER` |
| `4d8c4edf` | MoE masked kernels: single u64 pair-gather + rotate decode |
| `7e443418` | MoE: masked+u64 split kernels become the default on wmax<=32 banks |
| `61c8a0c6` | MoE group build: `tl.sort` + scan kernels replace serial rank sort |
| `b3d1dd5c` | EXL3 non-inline path: write segment outputs into caller `out` rows |
| `3eaf0a44` | `fn_kernels` inline extension: enable on `sm_89` with cluster/PDL compiled out |
| `aff80b8d` | `fn_kernels`: arch-tuned MoE launch defaults — Ada takes 4 stages / 8 warps |
| `e6228406` | `fn_kernels`: emulate the K-split cluster below `sm_90` — partials via global memory |
| `1f86d380` | `fn_kernels`: the emulated K-split reduces in the writer's launch — last rank's ticket |
| `6350ac83` | `lfm2_moe`: the MoE site's combine folds into its `add_rms` — one launch |
| `fbd7d834` | `lfm2_moe`: the verify commit as two kernels — ~130 launches down to 2 |
| `9f649a0b` | `fn_kernels`: `MOE_GROUP` on for `moe_experts` at M>=4 — verify MoE -36% |
| `cbf44a40` | `fn_kernels`: `dense` routes M in [4, 32] to `dense_rows` — verify dense -78% |
| `1014fe0f` | `fn_kernels`: the grouped MoE kernels get their own stage count — ST2 on Ada |

## Highlights

- **K-split cluster emulation below `sm_90`** (`e6228406`, `1f86d380`). `sm_90+`
  kernels reduce rank partials via DSMEM inside a cluster launch. Ada has no
  clusters, so rank partials go through global memory — initially a separate
  reducer launch, then a single writer launch where the last-arriving CTA per
  output tile reduces (ticket counter, `mix_tickets` convention). Fixed
  rank-major/warp-minor order -> bitwise identical to the DSMEM path. 1 MiB
  ticket-counter floor keeps captured graph pointers stable.
- **Re-derived Ada split rule** (`1f86d380`). `dense_split()` now computes legal
  S pre-`sm_90`: `KT % (16*S) == 0`, per-rank K-depth floors, base-NB saturation
  gate (vocab head `128000x2048x6` stays S1) and a medium-grid taper protecting
  M=32 `dense` verify. `FN_DENSE_SPLIT_FORCE=1` still returns table entries
  verbatim for sweeps. Measured optima: `2048x11776x25`->S4, `23552x2048x25`->S2,
  `2048x8192x35`->S8.
- **Inline CUDA on `sm_89`** (`3eaf0a44`). The `fn_kernels` torch extension now
  builds on Ada with cluster/DSMEM/PDL blocks behind `__CUDA_ARCH__ >= 900`
  guards and a lazy build-verdict fallback to Triton.
- **Arch-tuned MoE launch defaults** (`aff80b8d`). Ada: `FN_MOE_ST=4`,
  `FN_MOE_WARPS=8`; GB10 keeps its tuned 3/4. `dense` gains `FN_DENSE_ST=4` on
  Ada (bit-neutral pipeline depth, 7-14% per shape).
- **MoE masked/gather + group build** (`0f0343d9`..`61c8a0c6`): u64 pair-gather
  decode path and `tl.sort`+scan expert grouping; unselected experts are never
  read.
- **Fused verify commit** (`fbd7d834`). `commit_dev` ran ~130 tiny torch ops per
  verify (30 conv `cat+index_select+copy_`, 10 attn K/V `index_select`+
  `index_copy_`) plus ~80 temp allocs. `_commit_conv` + `_commit_kv` do it in two
  launches: each program gathers its source rows into a register tile before any
  store (index_select's snapshot semantics by construction — correct for any
  index order, no aliasing-order reliance). Per-layer tensors are addressed
  through cached int64 `data_ptr` arrays (refilled in place → captured graphs
  replay fresh addresses); bf16/contiguous gates keep the verbatim torch path
  otherwise. **190µs → 2.6µs a commit under replay (~73×); 1448µs → 48µs eager.**
- **Combine + residual-norm fusion** (`6350ac83`). Each of the 38 decode MoE
  sites ran `moe_combine` then `_add_rms` — two launches, a combine result
  round-trip. `combine_add_rms` does both in one Triton program a row: stable
  ascending-id slot order, the active backend's own arithmetic (`fmaf` on the
  `sm_89` CUDA path, `mul.rn`+add on the Triton fallback), and every bf16
  rounding point of the two-kernel sequence — bit-identical (verified M=1..16,
  S=4/8). Gated `M<=16`, `2<=S<=16` pow2, pow2 N, contiguous int32 ids; anything
  else falls back. ~0.6-0.9us a call → ~25-35us a decode step.
- **Grouped expert kernels on the verify path** (`9f649a0b`). The ENG-241 group
  kernels shipped compiled but disabled (`MOE_GROUP = False`). At M>1 every
  (row, slot) pair got its own CTA, so an expert picked by k rows had its
  packed tiles decoded k times. With the flag on, same-expert pairs share one
  CTA of up to 8 rows that decodes the tiles once — and same-expert runs are
  structural, not statistical (the router appends a shared-expert slot every
  row hits). Measured per verify+commit: **M=16 -4.5ms (-19%), M=8 -2.1ms
  (-15%), M=4 -0.6ms (-8%)**; M=2 regresses +6% (groups of ~1 cannot amortize
  the serial `moe_groups` prep) so the gate is `M>=4`, not `M>1`. Bit-identical
  per row by construction (same `RowGemm` base and reduction order), tested:
  grouped-vs-ungrouped `torch.equal` at M in {4,8,16,32}, MOE_GR boundary runs
  8/9/16/17, all-same-expert, companion-permutation invariance. The launch is
  a fixed P-sized grid with a device-side `NG[0]` early-out — no host readback,
  already capture-safe (`test_moe_graph_capture` passes). Blast radius: the
  flag is module-global, so flashnext's `moe_parts` row path also routes
  grouped at M>=4 — bit-identical by the same argument (the same kernels ship
  grouped-by-default in `exl3_relu2`).
- **`dense` → `dense_rows` reroute at M∈[4,32]** (`cbf44a40`). SPD-125's
  row-group kernel shipped compiled and tested but unadopted — `dense`
  launched per-row programs at every M, so a verify block's M rows
  re-decoded every weight tile M times. `dense_rows` decodes each tile
  once per ≤16-row group, the one-row call's exact bytes per row. Real
  pack at M=16: dense column ~10.0ms → ~2.1ms. Gate: bf16 + contiguous +
  M∈[4,32], with measured exceptions at M≤4 (3072x2048 loses -9%;
  unmeasured <2048-col blocks kept out); `DENSE_ROWS_MIN_M>=33` opts out.
- **Grouped MoE stage split** (`1014fe0f`). Group CTAs carry far more smem
  per stage than pair CTAs — measured ST2 > ST4 for `moe_*_group` while
  pairs regress below ST4. `g_moe_st_group`/`MOE_PICK_GROUP`/`FN_MOE_ST_GROUP`
  split stages only; warps stay shared (warp count sets the partial-sum
  order — the decode path's bits). Ada default ST2/W8: verify M=16
  12311 → 10692us.
- **Warp-parallel `moe_groups` + sort removal** (`378b232c`). The group walk
  was one thread over P pairs plus the stable sort it consumed (≈400us at
  M=16). One warp now does a shared-memory histogram → lane-0 base scan →
  chunked `__match_any` index-rank scatter: the serial kernel's exact output,
  deterministically. Since grouping is per-expert, a stable sort never
  changed the partition — the grouped path now passes an empty order and the
  sort is gone entirely. `exl3_relu2`'s vendored copy gains the `E` arg and
  the histogram smem. Verify M=16 10692 → **10182us**.
- **`dense_rows` stage split** (`66d09f5f`). The row-group kernel shared
  `g_dense_st` (ST4, tuned on the M=1 column); on the verify shapes it serves
  it measured faster at ST3. `g_dense_st_rows`/`FN_DENSE_ST_ROWS` — stages
  only (pipeline depth), S and warps stay shared for the partial-sum order.
  Verify M=16 10182 → **9957us**.
- **`moe_ada` probe at bank load** (`ca8b73c7`). Probing keyed only on
  `fn_kernels`'s arch verdict left `fn_prefill_kernels`'s narrower dispatch
  able to reach `moe_ada.ok()` unprobed on sm_89 (a `load-order` raise in
  `test_dense_and_experts_at_half_rates`). Probe when either verdict is
  unsupported; idempotent, still skipped where both native paths serve.
- **Falsified sweeps** (measured, reverted): a 16-warp `dense` variant won only
  in a narrow NB∈[92,128] band absent from this pack and needed >96KiB smem at
  ST4/kb6; a fused `gemv`+router ticket kernel was bit-exact but ~1us *slower*
  (64 same-address atomics + a serialized last-CTA tail cost more than the saved
  launch); `dense_rows` (S,warps) re-tunes — W4/W8 wins up to -46% at M=16 but
  the pair sets the partial-sum order, contract-pinned to `dense`'s values;
  prefill `_moe_gtrellis` knobs — num_stages>1 loses 60-250% (decode is ALU,
  not smem-staged), BM64 wins a ~M=512-1024 band but loses at M>=2048, BM128
  loses everywhere. Kept out; noted so they aren't re-tried blindly.

## Perf ledger (clean-clock real-pack profiles)

| kernel | before (us) | after (us) |
|---|---|---|
| `dense<25,4,8>` x70 | 1035 | ~817 |
| `dense<25,4,4>` x12 | 270 | ~175 |
| `dense<6,4,4>` vocab | ~338 | ~310 |
| MoE gate/up+down | ~1788 | ~1473 (>=95% DRAM roofline) |
| `moe_combine`+`_add_rms` x38 | ~160 | ~130 (fused; ~0.6-0.9us/call) |
| **decode step total** | **4659** | **3284** (fused combine est. ~3255) |

Verify path (rows_step at M>1 + commit), real pack:

| scope | before (us) | after (us) |
|---|---|---|
| verify+commit, M=16 | 23842 | **9957 (-58%)** |
| verify+commit, M=8 | 13601 | ~11500 grouped; lower with rows |
| verify+commit, M=4 | 8054 | ~7420 grouped; lower with rows |
| `moe_gu`/`moe_down` x38, M=16 | 12966 | 7089 grouped+ST2 (-45%) |
| `dense` column x83, M=16 | ~10027 | ~2143 rows (-79%) |
| `commit_dev` per commit | 190 (replay) / 1448 (eager) | 2.6 / 48 |

Decode-path remaining measured headroom is ~50-100us total (`_add_rms`
launch-bound fusion, w2 `exl3_gemv_split`) — those columns are at their
rooflines. Prefill profile (2048-token real pack, for completeness): `_moe_gtrellis`
dominates (~53%, the moe_ada Ada path at its measured-optimal BM32/W8/ST1
config — swept and falsified above), `_exl3_gemv_split_g` ~15%, cublas/cutlass
GEMMs ~13%, rot/combine/attn the rest. The decode-once-slab and row-grid
tradeoffs there are already engineered (ENG-213, CHUNK_UNTIL).

Verify-path remaining measured headroom: `moe_gu_group`/`moe_down_group`
(~7ms at M=16) sit at the decode-once-per-group weight-traffic roofline
(MOE_GR=16 would need >200KiB smem — infeasible). `dense_rows`'s (S,warps)
are contract-pinned to `dense`'s (they set the partial-sum order — a
W4/W8 re-tune falsified on that ground); stages were the only free knob
and are now split. The residual ~1ms is launch-bound micro-ops already
fused or previously falsified.

## Upstream merge (59d9d543, on `gauntlet-upstream`)

Upstream/main moved 72 commits past our base; five files overlapped — all in the
grouped-MoE machinery their SPD-249 work retooled in parallel (runtime GR cap,
grid-row levers, a staged-rows XS path, streamed s_group/gud kernels, a
1024-thread scanned group builder behind `FN_MOE_GPAR`).

- Ordered `fn_moe_groups` calls keep upstream's exact semantics: the serial
  walked-order kernel, or the scanned kernel under the lever — identical
  groups either way, their parity test included.
- Order-free calls (our grouped path never sorts now) take
  `moe_groups_flat_kernel`: the one-warp histogram+match-any builder extended
  with the runtime `GR` cap and expert bound — the same per-expert partition a
  stable sort would walk to, at any P (their scanned kernel caps at 1024).
- `MOE_PICK_GROUP_LV` applies our `g_moe_st_group` over their `LV` template;
  smem sizes rows/red by `g_moe_gr` (theirs) and stages by `g_moe_st_group` (ours).
- `relu2` reverted to upstream's `nh_relu2_groups` exactly; `fn_moe_groups`
  carries the `E` bound only for the flat path's histogram smem.

## Verification

- `Lfm2MoeEngine.self_check` on the real pack: `{'ok': True, 'rows': 8, 'first_diff': None}` — repeated across every iteration and again on the merge.
- `tests/gpu/test_fn_kernels_gpu.py`: 19/19 — upstream's scanned-groups, streamed-rows and grid-rows tests plus M=1..32 bit-equal vs one-row calls, forced S2/S4/S8, grouped, fp32 head, CUDA-graph capture/replay, ticket reuse, dense_rows dispatch gating.
- `tests/gpu/test_lfm2_moe_gpu.py`: 7/7 — dense/experts at half rates, MoE bank row invariance, decode-step graph, oracle independence.
- `tests/test_moe_gather.py`: 11/11 — adds grouped-vs-ungrouped bitwise, MOE_GR boundary runs, all-same-expert, companion-permutation invariance; EXL3 outpath/rate/layout/probe/tile/GEMV suites all green.
- `tests/gpu/test_fn_joint_graph_gpu.py`: not buildable here — `fn_seq_kernels` targets `compute_121a` (GB10-only, pre-existing environment limit).
- Bitwise row invariance is a hard contract: a row computed alone equals the same row in any batch; ticket-reduce preserves the exact partial-sum order.

## Test plan

- [x] self_check on LFM2.5-24B-A2B EXL3 2.5bpw, M=1 and M=8
- [x] full `test_fn_kernels_gpu` suite + related EXL3 suites
- [x] CUDA-graph capture + replay incl. expert-ID changes
- [x] clock-state-controlled profiling (idle 210MHz vs sustained 2775MHz — comparisons normalized)
- [ ] GB10 regression (kernels untouched on the `>= sm_90` path; `part=nullptr` still selects cluster/DSMEM there)

Generated with [Devin](https://devin.ai)
