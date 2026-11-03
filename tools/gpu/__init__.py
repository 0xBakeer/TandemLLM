"""The GPU's adapter: one class a GPU answers every per-GPU question the engine asks (README.md).

    profile(device=None)    the adapter for a device's compute capability, picked once a device
                            (the current device when None; the base for the CPU)
    profile_for(cap, name)  the adapter for a capability and a GPU name, without torch (tests, tools)
    by_name(name)           the adapter a GPU name claims (nvidia-smi rows, setup presets)
    arch_flags(kind)        the nvcc arch flags inline extension `kind` builds with on this device
    capability(device)      (major, minor) of a CUDA device, None for the CPU: the one lookup

Importing this package does not import torch; `profile` and `capability` do when first asked.
"""

from __future__ import annotations

from tools.gpu.ada import Ada
from tools.gpu.base import INLINE, GpuProfile, NoInlineBuild
from tools.gpu.gb10 import GB10

__all__ = ["ADAPTERS", "INLINE", "GpuProfile", "NoInlineBuild", "PROBE", "adapter_for", "arch_flags",
           "by_name", "capability", "key_matches", "parse_cc", "profile", "profile_for"]

#: adapters in the order they are asked; the base answers for any GPU none claims, and for the CPU
ADAPTERS: tuple[type[GpuProfile], ...] = (GB10, Ada)

#: what a box's GPU is, printed as JSON by the engine's interpreter (tools/gate/hw `detect`)
PROBE = (
    "import json, torch\n"
    "d = {'torch': torch.__version__, 'cuda': torch.version.cuda}\n"
    "if torch.cuda.is_available():\n"
    "    p = torch.cuda.get_device_properties(0)\n"
    "    d.update(name=p.name, memory_gib=round(p.total_memory / 2**30, 1),\n"
    "             cc=f'{p.major}.{p.minor}', count=torch.cuda.device_count())\n"
    "print(json.dumps(d))\n")

_BY_KEY: dict = {}      # (cap, name) -> profile
_BY_DEV: dict = {}      # the device argument -> profile
_CAP: dict = {}         # device index -> (major, minor)


def adapter_for(cap: tuple | None = None, name: str | None = None) -> type[GpuProfile]:
    """The first adapter that claims the capability or the name, else the base."""
    for cls in ADAPTERS:
        if cls.matches(cap, name):
            return cls
    return GpuProfile


def profile_for(cap: tuple | None = None, name: str | None = None) -> GpuProfile:
    key = (tuple(cap) if cap is not None else None, name)
    got = _BY_KEY.get(key)
    if got is None:
        got = _BY_KEY[key] = adapter_for(*key)(*key)
    return got


def by_name(name: str | None) -> GpuProfile:
    return profile_for(None, name)


def key_matches(key: str, name: str | None) -> bool:
    """Whether a GPU named `name` may use the files measured under device key `key`: an adapter's own
    key only on a GPU that adapter claims; any other key (a GPU without an adapter) on any GPU."""
    owner = next((cls for cls in ADAPTERS if cls.key == key), None)
    return owner is None or owner.matches(None, name)


def parse_cc(cc: str | None) -> tuple | None:
    """"12.1" -> (12, 1); None for a missing or unreadable one."""
    try:
        major, minor = str(cc).strip().split(".")
        return int(major), int(minor)
    except (AttributeError, ValueError):
        return None


def _index(device):
    import torch
    if device is None:
        return torch.cuda.current_device() if torch.cuda.is_available() else None
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


def capability(device=None) -> tuple | None:
    """(major, minor) of `device` (an index, a torch.device or a string; the current device when
    None), None for the CPU or a machine without CUDA. Asked of the driver once a device."""
    idx = _index(device)
    if idx is None:
        return None
    if idx not in _CAP:
        import torch
        _CAP[idx] = tuple(torch.cuda.get_device_capability(idx))
    return _CAP[idx]


def profile(device=None) -> GpuProfile:
    """The adapter for `device`, picked once and kept (the current device's when None)."""
    try:
        return _BY_DEV[device]
    except (KeyError, TypeError):
        pass
    idx = _index(device)
    if idx is None:
        got = profile_for(None, None)
    else:
        import torch
        got = profile_for(capability(idx), torch.cuda.get_device_name(idx))
    try:
        _BY_DEV[device] = got
    except TypeError:
        pass
    return got


def arch_flags(kind: str, device=None) -> list[str]:
    """The nvcc arch flags of inline extension `kind` on `device`; `NoInlineBuild` where its adapter
    does not build it."""
    return profile(device).arch_flags(kind)


def _reset_for_tests() -> None:
    _BY_KEY.clear()
    _BY_DEV.clear()
    _CAP.clear()
