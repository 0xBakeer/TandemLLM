#!/usr/bin/env python3
"""SPD-E5 A2: can the next group's packed draft hide under a verify pass? V, D and V || D by CUDA events.

E engines (one copy of the weights) with their DFlash2 drafters, each with a prefilled context and one T-row tree:
the joint verify pass from its CUDA graph (engine/models/qwen3_5/joint_graphs.py) and the packed draft from its
graph (engine/drafters/draft_graph.py). Timed, `n` replays each: V alone, D alone, and V on one stream with D on
another, launched together. overlap = V + D - T(V || D): what a second stream could hide at most.

    python tools/overlap_check.py --slots 8 --rows 8 --ckpt <drafter checkpoint>

The served environment (ops/serve.env), QWEN38_JOINT_GRAPHS=1; the server stopped or memory beside it.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ctx", type=int, default=700)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    from engine.config import load_config
    from engine.drafters import dflash2
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.drafters.draft_graph import packed_draft
    from engine.loader import Weights
    from engine.models.qwen3_5 import joint_graphs as JG
    from engine.models.qwen3_5.adapter import Qwen38Engine
    from engine.tree import DraftTree
    dflash2.HOST_WALK = True
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True)
    rng = random.Random(3)
    g = torch.Generator().manual_seed(5)
    E, T = a.slots, a.rows
    engs, ds, items = [], [], []
    with torch.no_grad():
        for i in range(E):
            eng = Qwen38Engine(cfg, w, max_len=4096)
            d = DFlash2Drafter(eng, a.ckpt, blocks=1, path="greedy", max_len=4096, block=16)
            d.attach()
            ids = torch.randint(1000, 100000, (a.ctx + 13 * i,), generator=g).cuda()
            eng.forward(ids, start=0, last_only=True)
            d.sync(ids.tolist(), eng.hidden_post_norm, 0)
            parents, path = [-1], [0]
            for k in range(1, T):
                path = path[:rng.randint(1, len(path))]
                parents.append(path[-1])
                path.append(k)
            t = DraftTree([int(ids[-1])] + [rng.randint(1000, 100000) for _ in range(T - 1)], parents)
            items.append((eng, torch.tensor(t.tokens).cuda(), t.parents, ids.numel()))
            engs.append(eng)
            ds.append(d)
        reqs = [(d, 1234, it[3] - 1) for d, it in zip(ds, items)]

        def V():
            JG.forward_tree_multi(items)
            for e in engs:
                e.discard_pending()

        def D():
            packed_draft(reqs, graph=True)
        for _ in range(JG.AFTER + 3):
            V()
            D()
        torch.cuda.synchronize()
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

        def timed(fn_pairs):
            out = []
            for _ in range(a.n):
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                for s, fn in fn_pairs:
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        fn()
                for s, _ in fn_pairs:
                    torch.cuda.current_stream().wait_stream(s)
                en.record()
                torch.cuda.synchronize()
                out.append(st.elapsed_time(en))
            return _median(out)
        tv = timed([(s1, V)])
        td = timed([(s2, D)])
        tb = timed([(s1, V), (s2, D)])
    print(f"OVERLAP E={E} T={T}: V {tv:.1f} ms, D {td:.1f} ms, V||D {tb:.1f} ms -> overlap {tv + td - tb:.1f} ms "
          f"(medians of {a.n}; verify replays {JG.graphs_for(engs[0]).report().get('replayed')})", flush=True)


if __name__ == "__main__":
    main()
