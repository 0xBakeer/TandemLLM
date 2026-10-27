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
"""

from __future__ import annotations

import threading

LOCK = threading.RLock()


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
    """The owner's graphs on `stream` are reset (so is torch's workspace for it): the stream may serve another."""
    if stream is None:
        return
    with _LEASE_LK:
        _FREE.append(stream)
        LEASES["out"] -= 1
