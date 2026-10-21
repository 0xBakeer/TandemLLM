"""Which devices the inline CUDA extensions (the `sm_121a` sources) build and run for.

Every `load_inline` module in this package compiles only for the GB10 (sm_121a). On any
other GPU the public entry points take their torch reference paths instead -- the same
bits, slower -- so an unsupported GPU still serves. When a module grows kernels for
another architecture, add that capability to `_SUPPORTED` beside its code path.
"""

from __future__ import annotations

import torch

#: compute capabilities the inline extensions cover (the GB10's sm_121a)
_SUPPORTED = {(12, 1)}
_cache: dict[int, bool] = {}


def inline_supported(device: torch.device | str | int | None = None) -> bool:
    """True when the inline extensions can run on `device` (a CUDA device index,
    device or string; the current device when None). False for cpu."""
    if device is None:
        if not torch.cuda.is_available():
            return False
        idx = torch.cuda.current_device()
    else:
        dev = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
        if dev.type != "cuda":
            return False
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    ok = _cache.get(idx)
    if ok is None:
        ok = torch.cuda.get_device_capability(idx) in _SUPPORTED
        _cache[idx] = ok
    return ok
