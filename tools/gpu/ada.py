"""NVIDIA Ada (sm_89: RTX 40xx, RTX 6000 Ada, L4, L40): a discrete card with its own memory.

No inline extension is built for sm_89 yet, so every answer is the base's: the NVFP4 projections take
the Triton kernels, bf16 autocast, and a module whose only path is an inline extension raises
`NoInlineBuild`. An Ada port answers here, and only here:

  arch_flags(kind)    the extensions that build for sm_89 (`-gencode=arch=compute_89,code=sm_89`); the
                      NVFP4 register-fed kernel stays out (its e2m1 converter needs sm_100a or later)
  pdl()               False: sm_89 has no programmatic dependent launch, so it is compiled out
  moe_launch()        the MoE kernels' stages and warps, from this adapter's own knobs and defaults
  dense_split()       the dense kernels' K split; without clusters (sm_90) the split is emulated, with
                      its own rule for when a split pays
  skinny_rows()       0
  load-time probes    the checks that a kernel gives the reference's bits on this card, run once a
                      device by this adapter; a kernel error after a passing probe still raises

Its knobs are read in this file only, so an Ada default never changes another GPU's answer."""

from __future__ import annotations

from tools.gpu.base import GpuProfile, NoInlineBuild, _known

#: sm_89 gencode for the extensions this adapter builds
ARCH = ("-gencode=arch=compute_89,code=sm_89",)

#: the inline extensions with sm_89 kernels: the dense/MoE/prefill set. exl3_linear's csrc builds
#: here but its fused down-silu and slab paths are not proven row-invariant on sm_89, so the Ada
#: EXL3 blocks take the Triton probes (tools/exl3_ada) -- the dispatch asks `inline` first
KINDS = ("fn_kernels", "fn_prefill_kernels")


class Ada(GpuProfile):
    id = "ada"
    about = "NVIDIA Ada (sm_89), discrete memory: fn_kernels and fn_prefill_kernels build for it"
    arches = ((8, 9),)
    names = ("RTX 40", "RTX 6000 ADA", "L4", "L40")

    def arch_flags(self, kind: str) -> list[str]:
        _known(kind)
        if kind not in KINDS:
            raise NoInlineBuild(f"{kind}: no inline CUDA build for {self.label()}")
        return list(ARCH)

    def moe_launch(self) -> tuple[int, int]:
        """(cp.async stages, warps): 4 and 8 measured on the RTX 4080."""
        from engine.settings import SETTINGS as _S
        return int(_S.get("FN_MOE_ST") or "4"), int(_S.get("FN_MOE_WARPS") or "8")

    def dense_split(self, N: int, Kin: int, kb: int, table: dict) -> tuple[int, int]:
        """No distributed shared memory among the CTAs: a split launches plain CTAs whose last-arriving
        rank reduces the partials in place (a ticket, one launch). The exchange is nearly free, so a
        split pays off wherever a rank still holds a real K slice -- and stops paying once the work
        is already spread thin: 2048+ elems of K below it (a verify-width grid is already saturated
        there and splitting can lose), the column grid under ~5 waves of CTAs to begin with, 32+
        K-tiles a rank, the rank slice whole 128 blocks, a grid not already past ~2K CTAs. On a
        column grid already past half a wave the deeper splits want ~1024 K elems a rank to keep
        paying (a wide projection's verify rows go through `dense` at M up to 32, where an over-deep
        split costs more than it buys); past a 4-way split the deeper split also wants a grid still
        short of a couple of waves or ~2048 K elems a rank."""
        from engine.settings import SETTINGS as _S
        got = table.get((N, Kin, kb))
        if got is None:
            S = 1
            while (N // 128) * S < 96 and S < 8 and (Kin // 16) % (8 * S * 2) == 0:
                S *= 2
            got = (S, 4)
        if _S.get("FN_DENSE_SPLIT_FORCE") == "1" and (N, Kin, kb) in table:
            return got                           # tools/fn_tiles sweeps: the table entry verbatim
        KT, S = Kin // 16, 1
        if KT >= 128 and N // 128 <= 384:
            while (S < 8 and KT % (16 * S) == 0 and KT // (2 * S) >= 32
                   and (N // 128) * S * 2 <= 2048
                   and (S < 2 or N // 128 <= 64 or KT // (2 * S) >= 64)
                   and (S < 4 or (N // 128) * S * 2 <= 192 or KT // (2 * S) >= 128)):
                S *= 2
        return S, got[1]
