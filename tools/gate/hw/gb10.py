"""NVIDIA GB10 (DGX Spark, sm_121): 128 GB shared between the CPU and the GPU, the reference box.

The sizes are the box's own: 8 lanes, the memory ceiling and peak its served Flash-Next setup runs
with (66 / 77 GiB), the gate's server killed below 10 GiB MemAvailable (tools/row3.py MemGuard: a
board out of memory stops answering), and 70 GiB free before a server starts (the one-engine rule:
a second engine beside another wedges the box)."""

from __future__ import annotations

from tools.gate.hw.base import Profile


class GB10(Profile):
    id = "gb10"
    about = "NVIDIA GB10 (DGX Spark): unified memory, the reference box"
    gpu_adapter = "gb10"
    stability_minutes = 30

    def mem(self) -> float:
        return float(self.fp.memory_gib or 119.7)

    def lanes(self) -> int:
        return 8

    def decode_levels(self) -> list[int]:
        return [8]

    def prefill_depths(self, level: str) -> list[int]:
        return [2048, 8192, 32768, 65536] if level == "full" else [2048, 8192]

    def burst(self) -> tuple[int, int]:
        return (8, 2048)

    def long_ctx(self, level: str) -> int:
        return 131072 if level in ("quick+long", "full") else 65536

    def wide_chunk_ctx(self) -> int:
        return 20480

    def ceiling_gib(self) -> float:
        return 66.0

    def peak_gib(self) -> float:
        return 77.0

    def min_free_gib(self) -> float:
        return 70.0

    def mem_floor_gib(self) -> float:
        return 10.0

    def free_gib(self) -> float | None:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
        except OSError:
            return None
        return None

    def arm_env(self) -> dict:
        return {**super().arm_env(), "QSE_PARALLEL_FLOOR_GB": "10"}

    def pack(self) -> str:
        return "flashnext-3.05"
