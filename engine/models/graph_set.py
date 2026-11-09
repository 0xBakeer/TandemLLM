"""SRV-795: one slot's CUDA graphs on a capture stream of its own, reset all at once.

A slot adapter captures its graphs lazily (a decode step's for each context class, a verify block's for each
width and class, a commit's for each path, its draft head's for each feed), and every graph captured without a
shared pool owns a private pool. Nothing reset them: on 2026-11-08 a server up 29 h held 3,905 graphs' pools,
33.0 GB reserved (21.2 GB of it free and given back by no shed or `empty`), and refused requests for memory.

A reset of one graph drops torch's cuBLAS workspace of its capture stream, which every graph captured on that
stream baked in (engine/capture_lock.py, SPD-223), so a slot's graphs are never reset one by one. The owner
captures all of them on one stream it leases (`capture_lock.lease_stream`) at its first capture and keeps for its
life (one a slot: no other owner ever captures on it, and nothing waits for it to come back), and `drop` resets
every graph captured on it at once. The server drops the graph sets of idle slots, the least recently used first,
under memory pressure and over a cap (server/app.py `_drop_slot_graphs`).

    gs = GraphSet()
    torch.cuda.graph(g, stream=gs.stream())      # every capture of the slot
    gs.used()                                    # a round on the slot (the LRU order)
    gs.drop(graphs)                              # each CUDAGraph reset; the caller empties the allocator
"""

from __future__ import annotations

import threading
import time

from engine import capture_lock


class GraphSet:
    def __init__(self):
        self._stream = None
        #: an idle capture's step holds it, so does a drop: a drop never lands inside a step another thread runs
        self.lock = threading.Lock()
        self.t_used = 0.0                   # time.monotonic() of the slot's last round, 0: none yet
        #: a capture since the last drop: the set holds graphs a drop would reset
        self.holding = False
        #: dropped since the slot's last round: the idle captures leave it alone until it serves again
        self.dropped = False
        self.stats = {"drops": 0, "graphs_dropped": 0}

    def stream(self):
        """The capture stream of this slot's graphs, leased at its first capture and kept for its life."""
        if self._stream is None:
            self._stream = capture_lock.lease_stream()
        self.holding = True
        return self._stream

    def used(self) -> None:
        self.t_used = time.monotonic()
        self.dropped = False

    def drop(self, graphs) -> int:
        """Reset every CUDAGraph in `graphs` (None skipped: a graph object whose capture never started). The caller
        holds `lock`, no capture runs (`capture_lock.GUARD`) and no replay of the slot is queued; the device is
        synchronized first. The stream stays the slot's. The caller gives the pools' blocks back (`empty`) and then
        counts the reset (`capture_lock.graphs_reset`), so the memory policy's next walk sees them gone. Returns the
        graphs reset."""
        import torch
        graphs = [g for g in graphs if g is not None]
        if graphs and torch.cuda.is_available():
            torch.cuda.synchronize()
        for g in graphs:
            g.reset()
        self.holding = False
        self.dropped = True
        self.stats["drops"] += 1
        self.stats["graphs_dropped"] += len(graphs)
        return len(graphs)


def lru(holders: list) -> list:
    """`holders` [(key, graph set)] with the sets that hold graphs (a capture since their last drop), the least
    recently used first."""
    return sorted((h for h in holders if h[1] is not None and h[1].holding), key=lambda h: h[1].t_used)
