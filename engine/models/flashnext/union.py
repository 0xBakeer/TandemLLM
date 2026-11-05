"""SPD-249: the distinct routed experts of a joint pass, per layer (the union of every row's top-k).

A pass's expert bytes are U x one expert's bytes a layer, so U decides how far a round of 8 lanes is from its
memory floor. Every `every`-th joint pass marks its rows' experts in a [layers, experts] mask (one scatter a
layer on the device), counts them after the pass and copies the counts to the host without a sync; the
next sampled pass reads them once the copy is done (on the runner's thread: `report` makes no CUDA call).
Measuring only: no row's bits move.
"""

from __future__ import annotations

import torch


class ExpertUnion:
    def __init__(self, layers: int, experts: int, every: int, device="cuda"):
        self.layers, self.experts, self.every = int(layers), int(experts), max(0, int(every))
        self.device = torch.device(device)
        self.passes = 0                     # joint passes seen
        self.by_lanes: dict = {}            # lanes -> {"passes", "rows", "u", "u_layer" [layers]}
        self._mask = None
        self._pending = None                # (host counts, event or None, rows, lanes)

    @property
    def on(self) -> bool:
        return self.every > 0

    def begin(self):
        """At a joint pass's start: its mask (zeroed) when this pass is sampled, else None."""
        self.harvest()
        if not self.on:
            return None
        self.passes += 1
        if (self.passes - 1) % self.every or self._pending is not None:
            return None
        if self._mask is None:
            self._mask = torch.zeros(self.layers, self.experts, dtype=torch.int32, device=self.device)
        else:
            self._mask.zero_()
        return self._mask

    def note(self, mask, layer: int, ids: torch.Tensor) -> None:
        """Layer `layer`'s routed ids [rows, k + 1] (the shared slot last, left out)."""
        if mask is not None:
            mask[layer].scatter_(0, ids[:, :-1].reshape(-1).long(), 1)

    def end(self, mask, rows: int, lanes: int) -> None:
        """After the pass: the counts a layer, copied to the host without waiting."""
        if mask is None:
            return
        counts = mask.sum(1, dtype=torch.int32)
        if counts.is_cuda:
            host = torch.empty(self.layers, dtype=torch.int32, pin_memory=True)
            host.copy_(counts, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        else:
            host, ev = counts.clone(), None
        self._pending = (host, ev, int(rows), int(lanes))

    def harvest(self) -> None:
        p = self._pending
        if p is None or (p[1] is not None and not p[1].query()):
            return
        host, _, rows, lanes = p
        self._pending = None
        u = [int(x) for x in host.tolist()]
        s = self.by_lanes.setdefault(lanes, {"passes": 0, "rows": 0, "u": 0, "u_layer": [0] * self.layers})
        s["passes"] += 1
        s["rows"] += rows
        s["u"] += sum(u)
        s["u_layer"] = [a + b for a, b in zip(s["u_layer"], u)]

    def report(self) -> dict | None:
        """{lanes: {passes, rows_mean, u_mean (a layer), u_min / u_max (layer means), u_layer}}; None when off.
        Reads only what the runner's thread harvested: /health asks from another thread, and a CUDA call there
        (an event query) while the idle thread captures a graph breaks the capture."""
        if not self.on:
            return None
        out = {}
        for lanes, s in sorted(self.by_lanes.items()):
            n = max(1, s["passes"])
            per = [round(v / n, 1) for v in s["u_layer"]]
            out[lanes] = {"passes": s["passes"], "rows_mean": round(s["rows"] / n, 1),
                          "u_mean": round(s["u"] / (n * self.layers), 1), "u_min": min(per), "u_max": max(per),
                          "u_layer": per}
        return {"every": self.every, "passes": self.passes, "by_lanes": out}
