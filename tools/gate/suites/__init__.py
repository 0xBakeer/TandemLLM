"""The gate's suites. Each module measures one arm (`measure(ctx)`, the server running) and turns the
arms' raw results into rows with a verdict (`rows(raw, plan)`); `raw` maps an arm key ("base",
"head", "head:plain", "base:wide", ...) to the list of that arm's raw results, one per server.

Every request goes to the engine (`/v1/completions`, `X-Tandem: raw`); every concurrent workload is
proved from the server's /health while it runs; every speed row takes at least two measured rounds a
side (tools/gate/verdict.py)."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class Ctx:
    side: str
    variant: str
    base: str
    cl: object
    health: dict
    knobs: dict
    profile: object
    level: str
    pack: dict
    out: Path
    pj: dict
    log: object = print

    def p(self, *x):
        self.log(f"[{self.side}{'' if self.variant == 'main' else ':' + self.variant}]", *x)


@dataclasses.dataclass
class Plan:
    level: str
    suites: list
    kernels: list
    profile: object
    pack: dict
    pairs: int = 1


def first(raw: dict, key: str):
    xs = raw.get(key) or []
    return xs[0] if xs else None


def pooled(raw: dict, key: str) -> list:
    return list(raw.get(key) or [])


ORDER = ("kernels", "lossless", "decode", "prefill", "long", "api", "stability")


def module(name: str):
    import importlib
    return importlib.import_module(f"tools.gate.suites.{name}")
