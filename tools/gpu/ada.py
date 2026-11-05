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
    about = "NVIDIA Ada (sm_89), discrete memory: Triton EXL3 and MoE kernels with load-time verdicts"
    arches = ((8, 9),)

    # The Ada kernels are Triton, so this adapter builds no inline extension (arch_flags keeps
    # the base's NoInlineBuild) and answers no launch shapes. What it owns is the load-time
    # probes and verdicts of tools/exl3_ada.py and tools/moe_ada.py -- the dispatch reaches
    # those kernels only through these answers, so a False verdict always selects reference.
    def exl3_probe(self, device, kb: int) -> None:
        from tools import exl3_ada
        exl3_ada.probe(device, kb)

    def exl3_rot_ok(self, device) -> bool:
        from tools import exl3_ada
        return exl3_ada.rot_ok(device)

    def exl3_decode_ok(self, device, kb: int) -> bool:
        from tools import exl3_ada
        return exl3_ada.decode_ok(device, kb)

    def exl3_gemv_ok(self, device, kb: int) -> bool:
        from tools import exl3_ada
        return exl3_ada.gemv_ok(device, kb)

    def moe_probe(self, device) -> None:
        from tools import moe_ada
        moe_ada.probe(device)

    def moe_ok(self, device) -> bool:
        # A False verdict -- or no probe at all, e.g. a BF16-expert model that never loaded a
        # MoEBank -- selects the torch path; it is not a load-order error here.
        try:
            from tools import moe_ada
            return moe_ada.ok(device)
        except RuntimeError:
            return False
