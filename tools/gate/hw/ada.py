"""NVIDIA Ada (sm_89: RTX 40xx, RTX 6000 Ada, L4, L40): a discrete card with its own memory.

The sizes scale with the card's memory and are a starting point, not measured on an Ada card yet:
4 lanes from 40 GiB (else 2), the long prompt 32k from 40 GiB (else 16k), the memory ceiling 3 GiB
and the peak 1.5 GiB under the card's total, and 90 % of it free before a server starts. The
reference pack is LFM2.5 EXL3, the family Ada contributors serve: 4.0 bits from 24 GiB, else 2.5."""

from __future__ import annotations

import subprocess

from tools.gate.hw.base import Fingerprint, Profile


class Ada(Profile):
    id = "ada"
    about = "NVIDIA Ada (sm_89), discrete memory; sizes not yet checked on an Ada card"

    @classmethod
    def matches(cls, fp: Fingerprint) -> bool:
        return fp.cc == "8.9"

    def big(self) -> bool:
        return self.mem() >= 40

    def lanes(self) -> int:
        return 4 if self.big() else 2

    def prefill_depths(self, level: str) -> list[int]:
        if level == "full" and self.big():
            return [2048, 8192, 32768]
        return [2048, 8192]

    def burst(self) -> tuple[int, int]:
        return (self.lanes(), 2048)

    def long_ctx(self, level: str) -> int:
        return 32768 if self.big() else 16384

    def wide_chunk_ctx(self) -> int:
        return 16384

    def ceiling_gib(self) -> float:
        return round(self.mem() - 3.0, 1)

    def peak_gib(self) -> float:
        return round(self.mem() - 1.5, 1)

    def min_free_gib(self) -> float:
        return round(self.mem() * 0.9, 1)

    def free_gib(self) -> float | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            return round(float(out.splitlines()[0]) / 1024, 1)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def pack(self) -> str:
        return "lfm25-exl3-4.0" if self.mem() >= 24 else "lfm25-exl3-2.5"
