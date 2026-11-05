# `tools/gpu`: one adapter per GPU

Every choice the engine makes because of the GPU it runs on is a question to this package. Each GPU
has one adapter file that answers those questions. Supporting a new GPU means adding its file, and it
never changes another GPU's answers.

```
tools/gpu/
  __init__.py   profile(device): the adapter for the device's compute capability, picked once a device
  base.py       GpuProfile: every question, answered for a GPU without an adapter and for the CPU
  gb10.py       sm_121a (DGX Spark): the reference GPU, every inline extension
  ada.py        sm_89: the base's answers until its kernels land; the place an Ada port fills in
```

## How it works

Kernel code asks the profile instead of reading the capability:

```python
from tools import gpu as _gpu

_gpu.arch_flags("fn_kernels")              # nvcc flags of an inline extension on this GPU
_gpu.profile().moe_launch()                # (stages, warps) of the fused MoE kernels
_gpu.profile(x.device).skinny_rows()       # rows the NVFP4 register-fed kernel takes, 0 for none
```

`profile(device)` reads the capability and the name once a device and keeps the answer. The first
adapter in `ADAPTERS` that claims them answers. When none does (an unknown GPU, or the CPU), `base.py`
answers. The NVFP4 projections then take the Triton kernels, and BF16 runs where the capability has it.
The base builds no inline CUDA extension. A module whose only path is one raises `NoInlineBuild` at
its first build, and the error names the GPU. `profile_for(cap, name)` gives the same adapters
without torch.

The CUDA sources keep their per-architecture code. The adapter picks the extension, its arch flags and
its launch shape.

## The questions

| question | asked by | `gb10` | base |
|---|---|---|---|
| `arch_flags(kind)` | every inline extension's build (`INLINE`) | `-gencode=arch=compute_121a,code=sm_121a` | `NoInlineBuild` |
| `inline(kind)` | modules that check before building | yes, every kind | no |
| `pdl()` | `fn_kernels`, `fn_seq_kernels`, `exl3_linear`, the 27B's fragment kernels | yes | no |
| `cluster()` | `fn_kernels.dense`, `fn_kernels.dense_rows`, `exl3_linear` | yes | from `sm_90` |
| `e2m1()` | `nvfp4_linear`, `nvfp4_linear_v2` | yes | from `sm_120` |
| `moe_launch()` | `fn_kernels` | `QWEN38_FN_MOE_ST`, `QWEN38_FN_MOE_WARPS` (3, 4) | `NoInlineBuild` |
| `dense_split(N, Kin, kb, table)` | `fn_kernels.dense_split` | the table, else a cluster of up to 8 CTAs below 96 column blocks, 4 warps | `NoInlineBuild` |
| `skinny_rows()` | `nvfp4_skinny.use_skinny`, the GGUF NVFP4 linear | 32 | 0 |
| `bf16()` | the small CUDA models (`engine/decision/laya`) | yes | from `sm_80` |
| `device_key()` | per-device files (`engine/methods/device.py`, price tables) | `dgx-spark` | the GPU's name with dashes, `cpu` |
| `unified_memory` | readers of the memory model | yes | no |

`key_matches(key, name)` tells the setup page whether a preset measured under `key` belongs to a GPU of
that name. `tools/gate/hw` sizes the gate for a box. Each of its profiles names the adapter it sizes
for (`gpu_adapter`), and this package says which adapter a box has.

## Adding a GPU

1. Write `tools/gpu/<gpu>.py` with a `GpuProfile` subclass: its `id`, the `arches` (and `names`) it
   claims, and the answers it changes. Leave the rest to the base.
2. Add it to `ADAPTERS` in `__init__.py`.
3. Read its knobs in its own file only, and register them in `engine/settings.py` with the file as
   their owner. Two GPUs never share a knob or a default.
4. Add its test to `tests/test_gpu_adapters.py`, with the capability patched and its answers checked.
5. If it builds inline extensions, add the GPU tests that hold its kernels to the reference.

## Rules

- No capability check, arch flag or per-GPU default lives outside this folder.
  `tests/test_gpu_adapters.py` searches the tree, and its allow-list names each exception and the
  reason for it.
- Each adapter reads only its own knobs.
- Two tests pin the `gb10` answers: `tests/test_gpu_adapters.py`, and the model fingerprints in
  `tests/test_adapter_guards.py`, which hash this folder with the kernels that import it.
- An unknown GPU gets the base and still serves every path that has a Triton or torch kernel.

## Sites

Moved: each now asks the profile.

| site | choice |
|---|---|
| `tools/fn_kernels.py`, `fn_prefill_kernels.py`, `fn_seq_kernels.py`, `exl3_linear.py`, `exl3_tall.py`, `exl3_relu2.py`, `nvfp4_skinny.py`, `l2_prefetch_bench.py`, `engine/models/qwen3_5/frag.py` | the arch flags of their inline builds |
| `tools/fn_kernels.py`, `fn_seq_kernels.py`, `exl3_linear.py`, `engine/models/qwen3_5/frag.py` | programmatic dependent launch compiled in |
| `tools/fn_kernels.py` | the MoE kernels' stages and warps (their knobs now read in `gb10.py`), the dense kernels' K split |
| `tools/nvfp4_skinny.py` `use_skinny`, `tools/iq2_linear.py` `NVFP4Linear.matmul` | the NVFP4 register-fed kernel's rows |
| `engine/decision/laya/model.py` | BF16 or FP16 autocast |
| `engine/methods/device.py` `device_id` | the key of per-device files |
| `server/setup/device.py` `preset` | whether a measured preset belongs to this GPU |
| `tools/gate/hw` (and `tools/release_report`, which reads it) | which sizing profile a box gets, and the torch probe |
| `install.sh` | the skinny kernel's switch off the Spark, now the adapter's answer |

Left, with the reason:

| site | why |
|---|---|
| `tools/csrc/**/*.cu` `__CUDA_ARCH__` | the per-architecture code paths inside the CUDA sources; the adapter picks the build |
| `install.sh` GPU check (`compute_cap` 12.1) | it runs before the repository is on disk; it sizes the install, not the engine |
| `proxy/hostcollect.py`, `proxy/scrape.py` (a GB10 reports one memory pool) | the standalone proxy ships without `tools/`; it reports any backend host's memory, which is not an engine choice |
| `tools/release_gate_setup.sh` | a stub `nvidia-smi` that plays a GB10 for the install check |
| `tools/exl3/common.sh` (`TORCH_CUDA_ARCH_LIST`, the capability in its log) | it builds the third-party exllamav3 on a quantization box, not this engine's kernels |
| `engine/keyed_noise.py` (SM count) | it repeats ATen's launch arithmetic from the device's properties, the same rule on every GPU |
| tile and split tables (`ops/*-tiles*.json`, `ops/*-splits.json`) | measured on the GB10 and named by the served profiles; an adapter that builds the kernels names its own |
| `ops/profiles/*.env`, `ops/serve.env` | model and serving settings; none depends on the GPU |
