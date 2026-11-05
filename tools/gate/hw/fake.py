"""The fake engine's box (`--hw fake`, pack `fake`): small sizes, no GPU, quick rounds. It checks the
gate itself (tests/test_gate_e2e.py), never an engine."""

from __future__ import annotations

from tools.gate.hw.base import Fingerprint, Profile


class Fake(Profile):
    id = "fake"
    about = "no GPU: the fake engine (server/fake_engine.py), for the gate's own tests"
    plain_variant = False
    stability_minutes = 1
    # rounds of 0.6-1 s at the fake's 400 tok/s: longer than the 0.2 s /health poll that proves them
    decode_tokens = 400
    c1_tokens = 256
    # the fake decodes on the CPU, so a shared CI runner with other test files beside it moves its
    # speed by several percent between two identical servers (an A/A row resolved 8.2 % worse with
    # 6.7 % noise); the planted slowdowns the tests check halve the speed, far above this
    min_effect_pct = 20.0

    def __init__(self, fp: Fingerprint | None = None):
        super().__init__(fp or Fingerprint(name="fake", memory_gib=0.0))

    def lanes(self) -> int:
        return 4

    def prefill_depths(self, level: str) -> list[int]:
        return [256, 1024]

    def burst(self) -> tuple[int, int]:
        return (4, 1024)        # the fake's prefill sleeps up to 0.5 s: long enough to see 4 running

    def long_ctx(self, level: str) -> int:
        return 2048

    def wide_chunk_ctx(self) -> int:
        return 1024

    def ceiling_gib(self) -> float:
        return 0.0

    def peak_gib(self) -> float:
        return 0.0

    def min_free_gib(self) -> float:
        return 0.0

    def arm_env(self) -> dict:
        return {"QSE_PARALLEL": str(self.lanes())}

    def pack(self) -> str:
        return "fake"
