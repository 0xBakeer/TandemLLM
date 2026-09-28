# Kernels

Decode on this board is a memory-read problem. Every round must read the projection weights once (about 13.7 GB of the 15 GB at NVFP4), and the GB10 GPU reads memory at about 235 to 240 GB/s in practice. A kernel is good here when it streams those bytes near that rate at the row counts a verify uses, which is 1 to 32 rows.

This page covers the matrix-product kernels, the recurrent-layer kernels, the head, and the one property every kernel must keep: a row's result may not depend on the other rows in the batch.

## Row invariance

Take a token verified alone and the same token verified as row 5 of a 16-row tree. Its logits must be the same bits in both cases. The exactness gate compares speculative output against one-token-at-a-time output, and that comparison only holds if the arithmetic of a row ignores its neighbours.

TandemLLM gets this by fixing the reduction order. Every tile table is keyed by the weight's shape (N, K), never by the row count, so the sum over K runs in the same order at 1 row and at 32. No kernel uses atomics, whose order would depend on the scheduler. Where K is split across warps, the partial sums are added in warp order. `tools/fp8_probe.py --sweep` keeps only FP8 tile choices that are bit-identical to the default at 1 to 32 rows, and the NVFP4 tables follow the same rule.

Row invariance is also what makes a tree verify, and later parallel requests, exact at all. [exactness.md](exactness.md) has the full contract.

## NVFP4 matrix products

The weights are 4-bit and the activations 16-bit (W4A16). The engine never uses the FP4 tensor-core path. On this chip an FP4 operand still has to travel through shared memory and a decode step before a tensor core sees it, and the chip has 99 KB of shared memory per SM. A 4-bit activation path would also need its own quantiser and quality gate, to save at most the 8 % of traffic the activations are.

Three kernels share one stored format and pick by row count:

| rows | kernel | file |
|-|-|-|
| 1 to 32 (decode and verify) | the skinny kernel, CUDA | `tools/nvfp4_skinny.py` |
| 33 to about 1,024 (prefill chunks) | Triton, v2 | `tools/nvfp4_linear_v2.py` |
| above that | expand to BF16 once, then the library GEMM | `tools/nvfp4_linear.py` |

The skinny kernel rests on one fact about the hardware. The instruction `cvt.rn.f16x2.e2m1x2` turns one packed byte into one register holding two 16-bit values, and that register is exactly the operand layout of an `mma.m16n8k16` tensor-core instruction. A thread loads 16 bytes of a weight row with one vector load and decodes each byte straight into an operand register, so nothing goes through shared memory. The group scale is decoded by `cvt.rn.f16x2.e4m3x2` and multiplied onto the weight in fp16. That product is exact: e2m1 times e4m3 needs six significant bits, and fp16 has eleven. The weights are the B operand (8 rows an instruction) and the activations the A operand (16 rows an instruction). A 16-row verify therefore costs one instruction per weight tile, and 17 to 32 rows cost two.

It builds on first use with `torch.utils.cpp_extension` for `sm_121a`, because the e2m1 converter is specific to that chip. A tile table per shape (`ops/skinny-tiles.json` up to 16 rows, `ops/skinny-tiles-wide.json` for 17 to 32) holds the launch settings the sweep picked.

## FP8 matrix products

`tools/fp8_linear.py` reads the checkpoint's block-scaled FP8 as stored: e4m3 codes with one BF16 scale per 128 by 128 block. The tile is 128 wide along K so that each step covers one scale block. For the plain-FP8 profile, `ops/fp8-tiles.json` holds one launch setting per shape for every row count up to 32. The profile then walks and verifies with a single program shape per weight.

## The recurrent layers

A Gated DeltaNet layer updates a 128 by 128 state per head at every token:

    S <- S * exp(g)
    delta <- (v - S^T k) * beta
    S <- S + k (x) delta
    out <- S^T q

Each shape of this work has its own kernel:

| work | kernel | what it does |
|-|-|-|
| one token | `tools/gdn_kernels.py` | loads a state tile once, applies the whole step in registers, stores once |
| a verify chain | `tools/gdn_verify_kernels.py` | walks up to 16 tokens over one loaded tile and returns the factors a rollback needs |
| a verify tree | `tools/gdn_tree_kernels.py`, `tools/gdn_verify_kernels.py` | walks the tree in depth-first order, rebuilding each node's parent state from a stack of factors along its path |
| a commit | `tools/gdn_commit_kernels.py` | writes the accepted path's state as one low-rank update |
| a prefill | `tools/gdn_prefill_kernels.py` | the chunked form, with the chunk-local work kept in registers |

Trees rely on depth-first order. When the walk reaches a node at depth d, the last node it visited at each shallower depth is that node's ancestor. A stack of one set of factors per depth therefore holds exactly the node's own path, and the kernel rebuilds the parent's state from the round's entry state. [speculative-decoding.md](speculative-decoding.md) explains the identity behind this.

In the served build the commit happens inside the next verify. The accepted rows are recorded, and the next verify applies them to each state tile as it loads it. That halves the state traffic of a round, from 604 MB to 302 MB.

## Head, norms, attention

Two kernels serve the output head. `tools/head_gemv.py` handles one row, and a tensor-core GEMM handles more than one. At one row a GEMV reads fastest. At eight rows the same GEMV does eight times the scalar work without a tensor core, and a verify once ran 2.5 times slower because of that. Both kernels return fp32 logits.

Each RMS norm is one kernel instead of eight small ones, written with the reference's own rounding points (`tools/norm_kernels.py`). The gated norm of the recurrent layers rounds to BF16 between the weight multiply and the gate, and doing that multiply in fp32 changes the answer.

Attention uses the library's fused attention with a causal flag, a lower-right causal mask for chunks that start past position 0, and a decode kernel for short query counts.

## Graphs

Every verify runs from CUDA graphs (`engine/verify_graph.py`), one per shape of round: chain or tree, row count, context class, flags. Positions and lengths live in device buffers, so one captured graph serves every round of that shape. Graphs took the launch gaps inside a verify from about 4.5 ms to about 1.5 ms. The drafter's forward runs from graphs too (`engine/drafters/draft_graph.py`). Graphed and eager runs agree bit for bit, and the tests check that.

## How a kernel gets in

Every kernel has a `check()` against the PyTorch function it replaces, and it answers to that before anyone times it. Only the engine's own round time decides whether it ships, measured with the flag flipped inside one process ([measurement.md](measurement.md)). A standalone timing is a ranking at best. The same NVFP4 configuration once read 0.254 ms in one process and 0.627 ms in another, and the two processes differed only in what they had allocated before.

Time a kernel over a working set larger than the chip's caches. `tools/cold_bw.py` does. A warm probe once reported 97.6 % of peak memory bandwidth on a GEMM, which is not possible.

## Knobs

Each kernel path has a `QWEN38_*` switch, and `ops/serve.env` sets the served combination. `engine/settings.py` lists every switch with its default, and `python -c "from engine.settings import SETTINGS; print(SETTINGS.describe())"` prints the values in effect.

| knob | what |
|-|-|
| `QWEN38_NVFP4_SKINNY`, `QWEN38_SKINNY_TILES`, `QWEN38_SKINNY_TILES_WIDE` | the skinny kernel and its tile tables |
| `QWEN38_FP8_TILES` | the FP8 per-shape launch table |
| `QWEN38_FUSED_GDNVERIFY`, `QWEN38_GDNV_WY*` | the fused verify recurrence and its variants |
| `QWEN38_COMMIT_IN_VERIFY` | fold the commit into the next verify |
| `QWEN38_VERIFY_GRAPH`, `QWEN38_DRAFT_GRAPH` | CUDA graphs for the verify and the drafter |
| `QWEN38_FUSED_GDNPREFILL`, `QWEN38_NVFP4_PREFILL_V2` | the prefill paths |
