"""Where the engine's memory goes (OPS-41): the load ledger and the resident split.

On the GB10 the GPU and the CPU share one pool, the engine holds about 80 GiB of it, and the
residency planner (server/conductor/residency.py) has to know what that is before a role may load
beside it. Two sources, both read-only:

  * the load ledger: `mark(stage)` after each load stage in `server/app.py::_load` -- the
    allocator's `memory_allocated` / `memory_reserved` and MemAvailable at that moment. The
    difference between two marks is what that stage took (weights, the engine's own buffers with
    the KV pool, vision, drafters, the warm-up's workspace, the verify graphs). A CUDA context and
    a library's own workspace are outside the allocator and show only in MemAvailable.
  * the exact parts: `split(state)` asks the objects that hold the big tensors for their
    `nbytes` -- the weights, the KV pool, the recurrent (GDN) state, the drafters' context caches,
    the vision tower -- and the caches for their reports (state store, resident prefix). What the
    allocator holds beyond the named parts is `other` (scratch, graph pools, small buffers), and
    `reserve` is what the allocator keeps from the driver without a tensor in it.

Nothing here allocates on the device, takes the engine lock, or runs during a request: the marks
are a host call between load stages, and `split` walks attributes when the dashboard asks.
"""

from __future__ import annotations

import time

GIB = float(1 << 30)


def mem_available() -> int | None:
    """MemAvailable in bytes, None where /proc does not exist."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class LoadLedger:
    """The allocator and MemAvailable after each load stage."""

    def __init__(self, probe=None, avail=mem_available, clock=time.time):
        self.probe = probe or _torch_probe
        self.avail, self.clock = avail, clock
        self.marks: list[dict] = []

    def mark(self, stage: str) -> dict:
        alloc, reserved = self.probe()
        m = {"stage": stage, "allocated": alloc, "reserved": reserved,
             "mem_available": self.avail(), "t": round(self.clock(), 3)}
        self.marks.append(m)
        return m

    def stages(self) -> list[dict]:
        """One row per stage after the first: what it added to the allocator (allocated and
        reserved) and what it took from MemAvailable, in bytes (None where a reading is missing)."""
        out = []
        for a, b in zip(self.marks, self.marks[1:]):
            def d(k, sign=1):
                return None if a[k] is None or b[k] is None else sign * (b[k] - a[k])
            out.append({"stage": b["stage"], "allocated": d("allocated"),
                        "reserved": d("reserved"), "mem_available_drop": d("mem_available", -1),
                        "s": round(b["t"] - a["t"], 1)})
        return out

    def report(self) -> dict:
        return {"marks": list(self.marks), "stages": self.stages()}


def _torch_probe() -> tuple[int | None, int | None]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
    except Exception:                                              # noqa: BLE001
        pass
    return None, None


def _nbytes(*ts) -> int:
    return sum(int(t.numel() * t.element_size()) for t in ts if t is not None)


def drafter_caches(drafter, depth: int = 4) -> list:
    """Every drafter object below `drafter` that holds a context cache (`_ck`, `_cv`): the length
    router's arms, a merged router's head, the drafter itself. Each once."""
    seen, out, todo = set(), [], [(drafter, 0)]
    while todo:
        obj, k = todo.pop()
        if obj is None or id(obj) in seen or k > depth:
            continue
        seen.add(id(obj))
        if getattr(obj, "_ck", None) is not None:
            out.append(obj)
        mod = type(obj).__module__ or ""
        if not mod.startswith("engine."):
            continue
        for v in list(vars(obj).values()) if hasattr(obj, "__dict__") else []:
            if isinstance(v, (list, tuple)):
                todo.extend((x, k + 1) for x in v)
            elif type(v).__module__.startswith("engine."):
                todo.append((v, k + 1))
    return out


def split(state: dict, probe=None, avail=mem_available) -> dict:
    """The resident set, in bytes, from the objects that hold it. `state` is server/app.py's
    STATE; a part the server does not have (no vision, no store, the fake engine) is left out,
    and every part is a reading, never an estimate."""
    alloc, reserved = (probe or _torch_probe)()
    parts: dict[str, int] = {}

    def put(name, n):
        if n:
            parts[name] = int(n)
    eng = state.get("engine")
    w = getattr(eng, "w", None)
    put("weights", getattr(w, "bytes_fp8", 0) + getattr(w, "bytes_other", 0))
    kv = getattr(eng, "kv", None)
    put("kv_pool", _nbytes(*(getattr(kv, k, None) for k in ("k", "v", "ks", "vs"))))
    st = getattr(eng, "state", None)
    put("gdn_state", _nbytes(*(getattr(st, k, None) for k in ("S", "conv", "_spare"))))
    caches = drafter_caches(state.get("drafter"))
    put("drafter_kv", sum(_nbytes(d._ck, getattr(d, "_cv", None)) for d in caches))
    put("vision", getattr(state.get("vision"), "nbytes", 0))
    put("state_store", getattr(state.get("state_store"), "bytes", 0))
    res = state.get("resident")
    put("resident_prefix", (res.report().get("bytes") or 0) if hasattr(res, "report") else 0)
    named = sum(parts.values())
    out = {"allocated": alloc, "reserved": reserved, "parts": parts,
           "other": (alloc - named) if alloc is not None else None,
           "reserve": (reserved - alloc) if alloc is not None and reserved is not None else None,
           "mem_available": avail(), "max_len": state.get("max_len"),
           "drafter_caches": len(caches)}
    led = state.get("load_ledger")
    if led is not None:
        out["load"] = led.stages()
    return out
