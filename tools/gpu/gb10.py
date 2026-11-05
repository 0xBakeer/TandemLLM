"""NVIDIA GB10 (DGX Spark, sm_121a): the reference GPU, 128 GB shared between the CPU and the GPU.

Every inline extension is built for sm_121a (`-gencode=arch=compute_121a,code=sm_121a`: the e2m1
converter of the NVFP4 kernel is an arch-specific instruction, and an explicit arch flag stops torch
adding its own), with programmatic dependent launch. The launch shapes are the ones the served
profiles were measured and gated with: the MoE kernels' stages and warps from `QWEN38_FN_MOE_ST` (3)
and `QWEN38_FN_MOE_WARPS` (4), the dense kernels' K split from the measured table, else a cluster of
up to 8 CTAs while the output has fewer than 96 column blocks, 4 warps a CTA. NVFP4 projections of
1..32 rows may take the register-fed kernel. These answers are pinned by tests/test_gpu_adapters.py
and by the model fingerprints (tests/test_adapter_guards.py)."""

from __future__ import annotations

from tools.gpu.base import INLINE, GpuProfile, _known

#: the arch flags of every inline extension on this GPU
ARCH = ("-gencode=arch=compute_121a,code=sm_121a",)


class GB10(GpuProfile):
    id = "gb10"
    about = "NVIDIA GB10 (DGX Spark, sm_121a): unified memory, every inline extension"
    arches = ((12, 1),)
    names = ("GB10",)
    key = "dgx-spark"
    unified_memory = True

    def arch_flags(self, kind: str) -> list[str]:
        _known(kind)
        return list(ARCH)

    def inline(self, kind: str) -> bool:
        return kind in INLINE

    def pdl(self) -> bool:
        return True

    def moe_launch(self) -> tuple[int, int]:
        from engine.settings import SETTINGS as _S
        return int(_S.get("FN_MOE_ST") or "3"), int(_S.get("FN_MOE_WARPS") or "4")

    def dense_split(self, N: int, Kin: int, kb: int, table: dict) -> tuple[int, int]:
        got = table.get((N, Kin, kb))
        if got:
            return got
        S = 1
        while (N // 128) * S < 96 and S < 8 and (Kin // 16) % (8 * S * 2) == 0:
            S *= 2
        return S, 4

    def skinny_rows(self) -> int:
        return 32

    def bf16(self) -> bool:
        return True
