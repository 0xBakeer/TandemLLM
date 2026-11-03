"""The questions the gate asks about a GPU, with conservative answers; each GPU's adapter
(tools/gate/hw/<gpu>.py) overrides what it knows. Nothing outside these files holds a GPU-specific
size, limit or setting.

A profile answers:

    lanes()              requests at once the server is started with (QSE_PARALLEL)
    decode_levels()      the decode suite's concurrency levels above 1
    prefill_depths(lv)   the solo prefill prompts, in tokens, at gate level `lv`
    burst()              (requests, tokens) of the first-token burst
    long_ctx(lv)         the longest prompt the long suite sends
    wide_chunk_ctx()     the prompt of the wide prefill chunk case
    ceiling_gib(), peak_gib()   the memory ceiling and peak flags each arm starts with
    arm_env()            the environment every arm's server gets for this GPU
    pack()               the reference pack (models/<pack>, tools/gate/packs.yaml)
    min_free_gib()       memory that must be free before an arm starts, and free_gib() what is
    mem_floor_gib()      MemAvailable under which the gate kills its own server (unified memory)
    rounds, warmups      measured and warm-up rounds a speed row takes
    min_effect_pct       the smallest resolved slowdown that counts as WORSE
    plain_variant        whether lossless starts a third, no-drafter head server (spec vs plain)
    stability_minutes, drift_max_pct   the stability suite's length and its decode drift limit
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Fingerprint:
    """What the box is: from nvidia-smi and the engine's torch (tools/gate/hw/__init__.py)."""

    name: str = "unknown"
    memory_gib: float | None = None
    cc: str | None = None
    driver: str | None = None
    cuda: str | None = None
    torch: str | None = None
    count: int = 0
    unified: bool = False

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class Profile:
    id = "base"
    #: a short description for the report and `python -m tools.gate hw`
    about = "conservative sizes for a GPU without an adapter"
    rounds = 3
    warmups = 1
    min_effect_pct = 3.0
    plain_variant = True
    stability_minutes = 30
    drift_max_pct = 10.0
    decode_tokens = 512
    c1_tokens = 256

    def __init__(self, fp: Fingerprint):
        self.fp = fp

    @classmethod
    def matches(cls, fp: Fingerprint) -> bool:
        return False

    # -- sizes ------------------------------------------------------------------------------
    def mem(self) -> float:
        return float(self.fp.memory_gib or 16.0)

    def lanes(self) -> int:
        return 2

    def decode_levels(self) -> list[int]:
        return [self.lanes()]

    def prefill_depths(self, level: str) -> list[int]:
        return [2048]

    def burst(self) -> tuple[int, int]:
        return (self.lanes(), 2048)

    def long_ctx(self, level: str) -> int:
        return 8192

    def wide_chunk_ctx(self) -> int:
        return 8192

    # -- memory -----------------------------------------------------------------------------
    def ceiling_gib(self) -> float:
        return round(self.mem() * 0.80, 1)

    def peak_gib(self) -> float:
        return round(self.mem() * 0.90, 1)

    def min_free_gib(self) -> float:
        return round(self.mem() * 0.85, 1)

    def mem_floor_gib(self) -> float | None:
        return None

    def free_gib(self) -> float | None:
        """Memory free for a new server now (None: this profile cannot tell)."""
        return None

    def arm_env(self) -> dict:
        return {"QSE_PARALLEL": str(self.lanes()), "MEM_CEILING_GIB": f"{self.ceiling_gib():g}",
                "MEM_PEAK_GIB": f"{self.peak_gib():g}"}

    # -- what runs --------------------------------------------------------------------------
    def pack(self) -> str | None:
        return None

    def describe(self) -> dict:
        return {"profile": self.id, "about": self.about, "fingerprint": self.fp.as_dict(),
                "lanes": self.lanes(), "decode_levels": self.decode_levels(),
                "prefill_depths": {lv: self.prefill_depths(lv) for lv in ("quick", "full")},
                "burst": list(self.burst()),
                "long_ctx": {lv: self.long_ctx(lv) for lv in ("quick", "full")},
                "wide_chunk_ctx": self.wide_chunk_ctx(),
                "ceiling_gib": self.ceiling_gib(), "peak_gib": self.peak_gib(),
                "min_free_gib": self.min_free_gib(), "mem_floor_gib": self.mem_floor_gib(),
                "arm_env": self.arm_env(), "pack": self.pack(), "rounds": self.rounds,
                "warmups": self.warmups, "min_effect_pct": self.min_effect_pct,
                "plain_variant": self.plain_variant,
                "stability_minutes": self.stability_minutes, "drift_max_pct": self.drift_max_pct}
