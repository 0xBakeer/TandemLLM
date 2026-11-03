"""One lock between a CUDA graph capture and the CUDA work of threads that are not the engine's.

A capture runs in CUDA's global mode: while the engine thread captures a graph (the drafter's at
its first use for a context class, a verify graph), a CUDA call another thread makes that a
capture cannot hold -- a copy and its synchronize, a pinned allocation -- fails with
`cudaErrorStreamCaptureUnsupported` and invalidates the capture, whose request then fails. On
2026-10-02 at 21:09 (the first request after the restart) the disk tier's writer
copied a session head's rows while the drafter captured: the writer stopped after one of the
head's 15 chunks, so the head was never readable, and the request ended in an error from the
caching allocator (`info.status != cudaStreamCaptureStatusInvalidated`).

Every capture holds `LOCK` (engine/drafters/draft_graph.py, engine/verify_graph.py), and every
thread that calls CUDA outside the engine's own lock holds it around those calls (the disk
tier's writer, the probe snapshots' writer). The decision models and the narrator run under the
engine's lock, which a capture also holds, and need nothing more. A capture is rare and a
background copy takes milliseconds, so neither side waits long. `tests/test_capture_lock.py`
holds every capture site to it; `tools/capture_race_check.py` checks it on the GPU.

SPD-257: under expandable segments (PYTORCH_CUDA_ALLOC_CONF, the Flash-Next profile) every new segment of the
caching allocator reserves address space of about 1.125 times the device's memory (130 GiB a segment measured on the GB10), and every
graph captured without a shared pool owns a pool of its own. On a test server on 2026-11-01 the address space grew
from 59 to 109 TiB in 20 minutes as 299 graph pools were captured (447 to 839 segments), against a 48-bit limit of
256 TiB (about 2,000 such segments), and nothing caps the graphs' shapes. A capture therefore holds `LOCK` with the allocator on plain segments
(`expandable_segments:False`, the rest of the setting as it was) and the setting goes back when it lets go: the
graphs' pools are what they were without the setting, and the default pool keeps its expandable segments.
`GUARD` is the same lock without that switch, for the CUDA work off the engine's thread.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading

#: the lock itself. A capture holds it through `LOCK`; CUDA work off the engine's thread that must not land inside a
#: capture (the disk tier's writer, a side slot's spill save, the probe writer, the GPU health probe, the live view's
#: layer marks, `off_thread`) holds `GUARD`, the same lock without the allocator switch (SPD-257)
_RAW = threading.RLock()
GUARD = _RAW

#: SPD-257: the allocator setting a capture runs under (`_plain_segments`): read once from PYTORCH_CUDA_ALLOC_CONF;
#: `on` only when that asks for expandable segments; `toggles` and the last `error` for /health and the tests
ALLOC: dict = {"on": None, "conf": "", "plain": "", "toggles": 0, "errors": 0, "error": None}

#: SPD-257: the graph captures and resets so far, one for each outermost hold of `LOCK` (counted when it lets go,
#: so a walk taken during a capture is taken again after it), each `give_back` (a graph owner's graphs reset) and
#: each `graphs_reset` (graphs reset outside both: an eviction, a flush). The graphs' private pools change then:
#: the memory policy walks the allocator's segments again only when this moved (`engine.cache.TorchPools`)
GEN: dict = {"captures": 0}
_GEN_LK = threading.Lock()


def _captured() -> None:
    with _GEN_LK:
        GEN["captures"] += 1


def graphs_reset() -> None:
    """SPD-257: a graph owner reset graphs outside a capture and `give_back` (a cache's eviction, a flush): their
    pools gave blocks back, and the memory policy's next reading walks the allocator again (`GEN`)."""
    _captured()


def alloc_expandable() -> bool:
    """SPD-257: the engine's allocator runs with expandable segments (PYTORCH_CUDA_ALLOC_CONF asks for them, read
    once) and no switch of the setting has failed. The memory policy reads the allocator's pools only then
    (server/app.py `_load_caches`); a capture switches to plain segments only then."""
    if ALLOC["on"] is None:
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or ""
        plain = re.sub(r"expandable_segments\s*:\s*true", "expandable_segments:False", conf, flags=re.I)
        ALLOC.update(conf=conf, plain=plain, on=plain != conf)
    return bool(ALLOC["on"])


def _set_allocator(setting: str) -> None:
    import torch
    fn = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if fn is None:                                       # a torch before the accelerator-wide setting
        fn = torch.cuda.memory._set_allocator_settings
    fn(setting)


def _plain_segments(capture: bool) -> None:
    """SPD-257: the allocator on plain segments for a capture (`capture`), back on the configured setting after it.
    A setting torch refuses turns the switch off for good, on the plain side when it was the way back."""
    if not alloc_expandable():
        return
    try:
        _set_allocator(ALLOC["plain"] if capture else ALLOC["conf"])
        ALLOC["toggles"] += 1
    except Exception as exc:                                       # noqa: BLE001
        ALLOC.update(on=False, errors=ALLOC["errors"] + 1, error=f"{type(exc).__name__}: {exc}")


class _CaptureLock:
    """`LOCK`: the one lock (`_RAW`, reentrant) for a graph capture, which also puts the allocator on plain segments
    while the outermost hold lasts (`_plain_segments`) and counts the capture when it lets go (`GEN`). Only its
    holder touches `_depth`."""

    def __init__(self, raw):
        self._raw = raw
        self._depth = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not self._raw.acquire(blocking, timeout):
            return False
        self._depth += 1
        if self._depth == 1:
            _plain_segments(True)
        return True

    def release(self) -> None:
        try:
            if self._depth == 1:
                _plain_segments(False)
                _captured()
        finally:
            self._depth -= 1
            self._raw.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False

    def _is_owned(self) -> bool:
        return self._raw._is_owned()


LOCK = _CaptureLock(_RAW)


# ------------------------------------------------------------------ SPD-249: the guard for CUDA work off the engine's thread
#
# An idle graph capture (server/app.py `graphs_once`) runs on the server's idle thread under the engine's lock; a
# request thread or an HTTP reader that touches CUDA without the engine's lock (a state-store restore into a side
# slot, a shed, a snapshot, a /health reader, the disk tier's writer) and lands inside that capture breaks it, and a
# later round meets an illegal memory access. Such a thread enters `off_thread(site)`: it waits while a capture holds
# LOCK and holds it while it runs. `check(site)` at the top of such a function is the debug assertion
# (QSE_CAPTURE_GUARD_DEBUG=1; the tests set it): entered without the guard, it raises.

#: the debug assertion on (tests, a debug run on the box); off it costs one attribute read
DEBUG = os.environ.get("QSE_CAPTURE_GUARD_DEBUG") == "1"
#: entries a site, for the tests and /health
SITES: dict = {}


class CaptureGuardError(RuntimeError):
    """CUDA work off the engine's thread entered without `off_thread` (QSE_CAPTURE_GUARD_DEBUG=1)."""


@contextlib.contextmanager
def off_thread(site: str):
    """CUDA work of a thread that does not run the engine's rounds and captures: no CUDA call of it lands inside a
    capture (it waits for an open one, and none opens while it runs)."""
    with GUARD:
        SITES[site] = SITES.get(site, 0) + 1
        yield


def held() -> bool:
    """The calling thread holds LOCK (inside `off_thread` or a capture)."""
    return GUARD._is_owned()


def check(site: str) -> None:
    """The debug assertion at a CUDA site that may run off the engine's thread: it must run under `off_thread` (or a
    capture's own LOCK). Off the debug switch it does nothing."""
    if DEBUG and not held():
        raise CaptureGuardError(f"{site}: CUDA work off the engine's thread outside capture_lock.off_thread")


# ------------------------------------------------------------------ SPD-223: a capture stream of one's own
#
# torch's `CUDAGraph.reset()` (an eviction, `graphs.clear()`, a graph's destructor) drops the cuBLAS workspace
# torch keeps for the graph's CAPTURE STREAM (`at::cuda::clearCublasWorkspacesForStream`). Every graph captured
# on that stream bakes that workspace into its GEMM nodes. `torch.cuda.Stream()` hands out one of a fixed pool of
# streams round robin, so the lone draft graphs, every packed draft set, the joint graphs and the verify graphs
# shared streams: an evicted packed set took the workspace of a kept graph on the same stream, whose next replay
# stored a split-K GEMM's partials into a block given back (an illegal memory access at a draft replay, C4/C8
# greedy, once the bench's lane sets outran the 24-set caches). A leased stream belongs to one graph owner until
# it gives it back, after its own graphs are reset.

_LEASE_LK = threading.Lock()
_FREE: list = []
LEASES = {"made": 0, "out": 0}


def _new_stream():
    import ctypes

    import torch
    from engine import kvpool
    drv = kvpool.Driver.get(torch.cuda.current_device())
    if drv is None:
        return torch.cuda.Stream()
    drv._bind()
    h = ctypes.c_void_p()
    drv.lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    drv._check(drv.lib.cuStreamCreate(ctypes.byref(h), 1), "cuStreamCreate")      # CU_STREAM_NON_BLOCKING
    return torch.cuda.ExternalStream(h.value)


def lease_stream():
    """A capture stream no other graph owner holds (a returned one, else a new non-blocking stream)."""
    with _LEASE_LK:
        if _FREE:
            LEASES["out"] += 1
            return _FREE.pop()
    s = _new_stream()
    with _LEASE_LK:
        LEASES["made"] += 1
        LEASES["out"] += 1
    return s


def give_back(stream) -> None:
    """The owner's graphs on `stream` are reset (so is torch's workspace for it): the stream may serve another.
    SPD-257: their pools changed with them (`GEN`)."""
    if stream is None:
        return
    _captured()
    with _LEASE_LK:
        _FREE.append(stream)
        LEASES["out"] -= 1
