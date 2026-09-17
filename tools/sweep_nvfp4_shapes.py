"""Tile choice for the NVFP4 projections that are not the MLP, measured in the engine.

`tools/nvfp4_linear.py::_CONFIG` carries a measured tile for the two MLP shapes and a fallback for
everything else. Phase 4 put the GDN and attention projections into the same kernel, and the
fallback is not their tile: at 12:29 the GDN mixer read 64.9 MB of NVFP4 in 0.479 ms, which is
135 GB/s, against the MLP kernel's 211 on its own shapes.

The timing is the engine's own mixer, not a standalone kernel, for the reason recorded at 13:40 of
phase 2: the same configuration measured 0.254 ms and 0.627 ms in two processes that differed only
in what they had allocated before. One process here, one set of weights, the configuration flipped
between runs.

    python tools/sweep_nvfp4_shapes.py --rows 1,8
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from tools.nvfp4_linear import NVFP4Block, set_config  # noqa: E402


def timeit(fn, n: int = 40, warm: int = 8) -> float:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", default="1,8", help="row counts to tune for")
    ap.add_argument("--block-n", default="16,32,64")
    ap.add_argument("--split-k", default="1,2,4,8,16")
    ap.add_argument("--warps", default="1,2,4")
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {w.report()}")
    eng = Qwen38Engine(cfg, w, max_len=512, device=a.device)

    gdn_l, att_l = cfg.linear_layers[0], cfg.attention_layers[0]
    shapes = {}
    for group, layer in (("gdn", gdn_l), ("attn", att_l)):
        shapes[group] = sorted({(b.N, b.K) for name, b in w.q.items()
                                if name.startswith(f"layers.{layer}.")
                                and ".mlp." not in name and isinstance(b, NVFP4Block)})
        print(f"[shapes] {group}: {shapes[group]}")
    if not shapes["gdn"] and not shapes["attn"]:
        raise SystemExit("no NVFP4 projections outside the MLP; nothing to tune")

    grid = list(itertools.product([int(x) for x in a.block_n.split(",")],
                                  [int(x) for x in a.split_k.split(",")],
                                  [int(x) for x in a.warps.split(",")]))
    for M in [int(x) for x in a.rows.split(",")]:
        h = torch.randn(1, M, cfg.hidden_size, device=a.device, dtype=torch.bfloat16) * 0.02
        pos = torch.arange(0, M, device=a.device)
        for group, fn in (("gdn", lambda: eng.linear_attention(h, f"layers.{gdn_l}", gdn_l, True)),
                          ("attn", lambda: eng.attention(h, f"layers.{att_l}", att_l, 0, pos))):
            if not shapes[group]:
                continue
            nbytes = sum(w.q[n].nbytes for n in w.q
                         if n.startswith(f"layers.{gdn_l if group == 'gdn' else att_l}.")
                         and ".mlp." not in n)
            eng.state.primed = True
            best = None
            rows = []
            for bn, sk, nw in grid:
                for N, K in shapes[group]:
                    set_config(N, K, "decode", {"block_n": bn, "split_k": sk,
                                                "num_warps": nw, "num_stages": 3})
                try:
                    ms = timeit(fn) * 1e3
                except Exception as exc:                         # pragma: no cover
                    rows.append((float("inf"), f"bn={bn:3d} sk={sk:2d} w={nw}  failed: {exc}"))
                    continue
                rows.append((ms, f"bn={bn:3d} sk={sk:2d} w={nw}  {ms:7.3f} ms  "
                                 f"{nbytes / ms / 1e6:6.1f} GB/s"))
                if best is None or ms < best[0]:
                    best = (ms, bn, sk, nw)
            rows.sort()
            print(f"\n[{group}] M={M}  {nbytes / 1e6:.1f} MB of quantised weight in the mixer")
            for _, line in rows[:6]:
                print("   ", line)
            print(f"    best {best[1]}/{best[2]}/{best[3]} at {best[0]:.3f} ms")


if __name__ == "__main__":
    main()
