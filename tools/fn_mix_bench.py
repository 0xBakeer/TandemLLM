"""SPD-239: GB/s of Flash-Next's mixer kernels on the served shapes, weights cold in L2.

A ring of `--layers` distinct mixer weights (down|inject [R + hc, hc H] int8 with a scale a group of 128,
up [hc H, R] int8 with a scale a group of 64: `pack.mixer_weight`'s int8 form) is allocated; one CUDA
graph runs the down|inject GEMV of every weight in turn, another the mix_up of every weight, so each call
reads weights the previous one did not touch (the kernels' own launch chaining kept). Bytes are a
weight's int8 rows and scales. Prints one JSON line per M.

    python tools/fn_mix_bench.py --rows 1 2 3 4 5 8 16 24 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import fn_kernels as K  # noqa: E402

H, R, HC = 2560, 320, 4


def ring_ms(calls, reps: int) -> float:
    """ms a call: every call of the ring in one graph, replayed `reps` times."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / len(calls)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 16, 24, 32])
    ap.add_argument("--layers", type=int, default=96)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5)
    downs = [Int8Rows.of((torch.randn(R + HC, HC * H, generator=g, device=dev) * 0.02).half(), 128)
             for _ in range(a.layers)]
    ups = [Int8Rows.of((torch.randn(HC * H, R, generator=g, device=dev) * 0.05).half(), 64) for _ in range(a.layers)]
    nb_d, nb_u = downs[0].nbytes, ups[0].nbytes
    for M in a.rows:
        n = (torch.randn(M, HC * H, generator=g, device=dev)).to(torch.bfloat16)
        d = torch.empty(M, R + HC, dtype=torch.float32, device=dev)
        x = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        dms = ring_ms([lambda w=w: K.gemv(n, w, out=d) for w in downs], a.reps)
        ums = ring_ms([lambda w=w: K.mix_up(d, n, w, HC, out=x) for w in ups], a.reps)
        print(json.dumps({"M": M, "gemv_us": round(dms * 1e3, 1), "gemv_gbs": round(nb_d / dms / 1e6, 1),
                          "mix_up_us": round(ums * 1e3, 1), "mix_up_gbs": round(nb_u / ums / 1e6, 1),
                          "pass_ms_96": round(96 * (dms + ums), 2)}), flush=True)


if __name__ == "__main__":
    main()
