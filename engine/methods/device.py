"""The device a per-device file is keyed by (price files now; tile tables with ENG-251)."""

from __future__ import annotations

from tools import gpu as _gpu


def device_id(name: str | None = None) -> str:
    """The key of the GPU's adapter (tools/gpu `device_key`): the DGX Spark's GB10 is "dgx-spark"; any
    other GPU is its name in lower case with dashes ("nvidia-rtx-pro-6000" ...); no GPU is "cpu".
    `name`: the GPU's name (default: device 0's)."""
    return (_gpu.profile(0) if name is None else _gpu.by_name(name)).device_key()
