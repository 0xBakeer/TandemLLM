"""Which devices the inline CUDA extensions (the `sm_121a` sources) build and run for.

Every `load_inline` module in this package compiles only for the GB10 (sm_121a). On any
other GPU the public entry points take their torch reference paths instead -- the same
bits, slower -- so an unsupported GPU still serves. When a module grows kernels for
another architecture, add that capability to `_SUPPORTED` beside its code path.

The driver is asked once per device per process (`capability` caches the pair); an
arch-dependent dispatch is then a dict lookup, not a query a call.
"""

from __future__ import annotations

import torch

#: compute capabilities the inline extensions cover (the GB10's sm_121a)
_SUPPORTED = {(12, 1)}
_cache: dict[int, tuple | None] = {}


def capability(device: torch.device | str | int | None = None) -> tuple | None:
    """(major, minor) of `device` (a CUDA device index, device or string; the current
    device when None), or None for cpu, a machine without CUDA, or a failed query.
    Cached per device index."""
    if device is None:
        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
    else:
        dev = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
        if dev.type != "cuda":
            return None
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _cache:
        try:
            _cache[idx] = tuple(torch.cuda.get_device_capability(idx))
        except Exception:
            _cache[idx] = None
    return _cache[idx]


def inline_supported(device: torch.device | str | int | None = None) -> bool:
    """True when the inline extensions can run on `device`. False for cpu."""
    cap = capability(device)
    return cap is not None and cap in _SUPPORTED


def _reset_for_tests() -> None:
    """Drop the cache so a test can mock `torch.cuda.get_device_capability` afresh."""
    _cache.clear()
