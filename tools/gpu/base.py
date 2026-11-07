"""The questions the engine asks about a GPU, answered for a GPU without an adapter and for the CPU.

Every per-GPU choice is one of these questions. An adapter (tools/gpu/<gpu>.py) overrides the answers
it knows; this class answers the rest with the paths every CUDA GPU and the CPU have: no inline CUDA
extension is built, the NVFP4 projections take the Triton kernels, and bf16 is used where the
capability has it. A module whose only path is an inline extension raises `NoInlineBuild` at its
first build, naming the GPU, instead of building for an architecture the GPU cannot run.

    arch_flags(kind)        the nvcc arch flags inline extension `kind` builds with (`INLINE`)
    inline(kind)            whether `kind` builds and runs here
    pdl()                   programmatic dependent launch compiled into the extensions
    moe_launch()            (cp.async stages, warps a CTA splits K over) of the fused MoE kernels
    dense_split(N, Kin, kb, table)   (K split S, warps) of an EXL3 projection in the dense kernels
    skinny_rows()           rows the NVFP4 register-fed kernel takes (0: never)
    bf16()                  bf16 autocast for the small CUDA models (else fp16)
    e2m1()                  the NVFP4 kernels' e2m1 hardware converter (else a software decode)
    fp8_rows(N, K, bn)      (N tile, K tile, warps, stages) of the FP8 row kernel
    fp8_row_tile(M)         rows one program of the FP8 row kernel covers
    device_key()            the key of the per-device files measured on this GPU (price tables)
    unified_memory          the GPU shares the host's memory
"""

from __future__ import annotations

import re

#: the inline CUDA extensions, by the name their module asks `arch_flags` with
INLINE = ("exl3_linear", "exl3_tall", "exl3_relu2", "exl3_frag", "fn_kernels", "fn_prefill_kernels",
          "fn_seq_kernels", "nvfp4_skinny", "l2_prefetch")


class NoInlineBuild(RuntimeError):
    """An inline CUDA extension asked for on a GPU whose adapter does not build it."""


class GpuProfile:
    id = "base"
    about = "a GPU without an adapter, or the CPU: the Triton and torch paths, no inline extensions"
    #: compute capabilities (major, minor) and name parts (upper case) the adapter claims
    arches: tuple = ()
    names: tuple = ()
    #: the key of the per-device files measured on this GPU; None: the GPU's name
    key: str | None = None
    unified_memory = False

    def __init__(self, cap: tuple | None = None, name: str | None = None):
        self.cap = tuple(cap) if cap is not None else None
        self.name = name

    @classmethod
    def matches(cls, cap: tuple | None, name: str | None = None) -> bool:
        up = (name or "").upper()
        return (cap is not None and tuple(cap) in cls.arches) or any(n in up for n in cls.names)

    def label(self) -> str:
        if self.cap is None:
            return self.name or "the CPU"
        return f"{self.name or 'GPU'} (sm_{self.cap[0]}{self.cap[1]})"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}: {self.label()}>"

    # -- identity -------------------------------------------------------------------------
    def device_key(self) -> str:
        """The key of the per-device files measured on this GPU: the adapter's, else the GPU's name in
        lower case with dashes, "cpu" without one."""
        if self.key:
            return self.key
        if not self.name:
            return "cpu"
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    # -- builds -----------------------------------------------------------------------------
    def arch_flags(self, kind: str) -> list[str]:
        """The nvcc arch flags inline extension `kind` builds with here."""
        _known(kind)
        raise NoInlineBuild(f"{kind}: no inline CUDA build for {self.label()}; no adapter in tools/gpu "
                            f"builds it here (tools/gpu/README.md)")

    def inline(self, kind: str) -> bool:
        try:
            self.arch_flags(kind)
        except NoInlineBuild:
            return False
        return True

    def pdl(self) -> bool:
        """Programmatic dependent launch (griddepcontrol, sm_90 and later) compiled into the extensions."""
        return False

    def cluster(self) -> bool:
        """Threadblock clusters with distributed shared memory (sm_90 and later): the dense kernels'
        K split reduces the ranks through DSMEM there; below it the split is emulated -- plain CTAs
        and a python-side partials buffer (tools/fn_kernels.py `dense`)."""
        return self.cap is not None and self.cap >= (9, 0)

    def e2m1(self) -> bool:
        """The hardware e2m1 (NVFP4) converters, cvt.*.e2m1x2 (sm_100a and the sm_120 family)."""
        return self.cap is not None and self.cap >= (12, 0)

    # -- launch shapes ----------------------------------------------------------------------
    def moe_launch(self) -> tuple[int, int]:
        """(cp.async stages, warps a CTA splits its K range over) of the fused MoE kernels
        (tools/fn_kernels.py). The warps set the order of the partial sums."""
        raise NoInlineBuild(f"fn_kernels: no MoE launch shape for {self.label()}")

    def dense_split(self, N: int, Kin: int, kb: int, table: dict) -> tuple[int, int]:
        """(K split S, warps a CTA) of an EXL3 projection [N, Kin] at kb bits in the dense kernels
        (tools/fn_kernels.py `dense`): the measured `table` entry, else the adapter's rule. Keyed by
        the shape only: one summation order for every row count."""
        raise NoInlineBuild(f"fn_kernels: no dense split for {self.label()}")

    # -- paths ------------------------------------------------------------------------------
    def skinny_rows(self) -> int:
        """Rows up to which an NVFP4 projection takes the register-fed kernel (tools/nvfp4_skinny.py);
        0: never, the Triton kernels serve every row count."""
        return 0

    def bf16(self) -> bool:
        """bf16 autocast for the small CUDA models (engine/decision/laya); fp16 below sm_80."""
        return self.cap is not None and self.cap[0] >= 8

    # -- load-time probes and verdicts ------------------------------------------------------
    # An adapter that ships Triton kernels (tools/exl3_ada.py, tools/moe_ada.py) owns their
    # probes: the probe runs once per device at weight load, the verdicts gate every dispatch.
    # The base answers no kernels: a False verdict always selects the reference path.
    def exl3_probe(self, device, kb: int) -> None:
        """Probe this adapter's EXL3 kernels at rate `kb`; records the verdicts the dispatch reads."""

    def exl3_rot_ok(self, device) -> bool:
        """The rotation-kernel verdict on `device`."""
        return False

    def exl3_decode_ok(self, device, kb: int) -> bool:
        """The standalone decode-kernel verdict on `device` at rate `kb`."""
        return False

    def exl3_gemv_ok(self, device, kb: int) -> bool:
        """The projection-kernel verdict on `device` at rate `kb`."""
        return False

    def moe_probe(self, device) -> None:
        """Probe this adapter's MoE kernels once per device."""

    def moe_ok(self, device) -> bool:
        """The MoE-kernel verdict on `device`; False where no probe ever ran."""
        return False

    def e2m1(self) -> bool:
        """The NVFP4 e2m1 hardware converter (sm_100 and later); software e2m1 decode elsewhere."""
        return self.cap is not None and self.cap[0] >= 10

    # -- the FP8 row kernel (tools/nvfp4_relu2.py `_fp8_rows`) ---------------------------------
    def fp8_rows(self, N: int, K: int, bn: int | None = None) -> tuple:
        """(N tile, K tile, warps, stages) of the FP8 row kernel on a projection [N, K]; `bn` is the
        caller's N tile, stages None leaves Triton's default. The N tile is fixed by the shape, never
        by the row count: narrow enough to fill the board at one row."""
        return bn or (32 if N <= 8192 else 64), 128 if K % 128 == 0 else 64, 4, None

    def fp8_row_tile(self, M: int) -> int:
        """Rows one program of the FP8 row kernel covers in a call of M rows."""
        return 16


def _known(kind: str) -> None:
    if kind not in INLINE:
        raise ValueError(f"unknown inline extension {kind!r}: one of {', '.join(INLINE)}")
