"""A GPU without an adapter: the base's conservative sizes, scaled by its memory, and no reference
pack (the run names one with --pack)."""

from __future__ import annotations

from tools.gate.hw.base import Profile


class Generic(Profile):
    id = "generic"
    about = "a GPU without an adapter: small sizes from its memory; pass --pack"

    def lanes(self) -> int:
        return 4 if self.mem() >= 40 else 2

    def prefill_depths(self, level: str) -> list[int]:
        return [2048, 8192] if self.mem() >= 40 else [2048]

    def long_ctx(self, level: str) -> int:
        return 16384 if self.mem() >= 40 else 8192
