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

from tools.gpu.base import GpuProfile


class Ada(GpuProfile):
    id = "ada"
    about = "NVIDIA Ada (sm_89), discrete memory: the base's paths until its kernels land"
    arches = ((8, 9),)
