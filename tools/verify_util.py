#!/usr/bin/env python3
"""SPD-E5: the joint verify pass's GPU time and achieved weight bandwidth at E slots x T rows (graph replays).

Builds E engines over one copy of the weights, prefills each, and replays the joint pass (the joint graphs) `n`
times inside an NVTX range "verify" (for nsys --gpu-metrics-devices). Prints the median pass ms and the
weights' bytes over it (GB/s), so the headroom a second stream could use is a number, not a guess.

    nsys profile --gpu-metrics-devices=0 -t cuda,nvtx -o /tmp/vu python tools/verify_util.py --slots 8 --rows 8
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5 import joint_graphs as JG
    from engine.models.qwen3_5.adapter import Qwen38Engine
    from engine.tree import DraftTree
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True)
    wbytes = sum(getattr(b, "nbytes", 0) for b in list(w.q.values()) + list(w.t.values()))
    rng = random.Random(3)
    g = torch.Generator().manual_seed(5)
    E, T = a.slots, a.rows
    engs = [Qwen38Engine(cfg, w, max_len=4096) for _ in range(E)]
    pos, trees, blocks = [], [], []
    with torch.no_grad():
        for e in engs:
            p = torch.randint(1000, 100000, (rng.randint(400, 900),), generator=g).cuda()
            e.forward(p, start=0, last_only=True)
            pos.append(p.numel())
            parents, path = [-1], [0]
            for i in range(1, T):
                path = path[:rng.randint(1, len(path))]
                parents.append(path[-1])
                path.append(i)
            t = DraftTree([int(p[-1])] + [rng.randint(1000, 100000) for _ in range(T - 1)], parents)
            trees.append(t)
            blocks.append(torch.tensor(t.tokens).cuda())
        items = [(e, b, t.parents, s) for e, b, t, s in zip(engs, blocks, trees, pos)]
        for _ in range(JG.AFTER + 3):                     # capture, then warm replays
            JG.forward_tree_multi(items)
            for e in engs:
                e.discard_pending() if hasattr(e, "discard_pending") else None
        torch.cuda.synchronize()
        ts = []
        torch.cuda.nvtx.range_push("verify")
        for _ in range(a.n):
            t0 = time.perf_counter()
            JG.forward_tree_multi(items)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
            for e in engs:
                e.discard_pending() if hasattr(e, "discard_pending") else None
        torch.cuda.nvtx.range_pop()
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"VERIFY_UTIL E={E} T={T} rows={E * T}: pass median {med:.1f} ms (p10 {ts[len(ts) // 10]:.1f}); "
          f"weights {wbytes / 1e9:.2f} GB -> {wbytes / 1e9 / (med / 1e3):.0f} GB/s achieved on the weights "
          f"(stats {JG.graphs_for(engs[0]).report().get('replayed')} replays)", flush=True)


if __name__ == "__main__":
    main()
