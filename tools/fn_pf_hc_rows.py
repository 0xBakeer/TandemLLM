"""SPD-249 P2: at which row counts the library fp32 GEMM of the prefill's up mixer sums like a plain fma chain.

    PYTHONPATH=<repo> python tools/fn_pf_hc_rows.py --rows 1-4096 --json out.json

For the up mixer's call (`mm_rows(a, Wu)`: a fp32 [M, 320] @ Wu fp32 [10240, 320]^T, both contiguous) and the down
mixer's (n fp32 [M, 10240] @ Wd fp32 [324, 10240]^T) at every row count asked, the library's product against one fp32
fma chain over k = 0 .. K-1 from 0 per output (tools/csrc/fn_pf_hc/hc.cu `pf_chain`), bit for bit. Prints the row
counts where they differ and the library kernels seen. A measurement of this build's library, nothing more.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import fn_pf_hc as HC  # noqa: E402


def rows_arg(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def kernels(f) -> list[str]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        f()
        torch.cuda.synchronize()
    return sorted({e.name[:70] for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1-4096")
    ap.add_argument("--shapes", default="up,down")
    ap.add_argument("--kernels-at", default="1,8,16,32,33,34,35,64,128,512,1024,2048,4096")
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    shapes = {"up": (10240, 320), "down": (324, 10240)}
    res = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
           "library": HC.library_id(), "shapes": {}}
    for name in a.shapes.split(","):
        N, K = shapes[name]
        w = torch.randn(N, K, device=dev) * 1e-2
        differ, seen = [], {}
        for M in rows_arg(a.rows):
            x = torch.randn(M, K, device=dev).to(torch.bfloat16).float()
            ref = x @ w.T
            got = HC.chain(x, w)
            if not torch.equal(ref.view(torch.int32), got.view(torch.int32)):
                differ.append(M)
        for M in [int(v) for v in a.kernels_at.split(",")]:
            x = torch.randn(M, K, device=dev)
            seen[M] = kernels(lambda: x @ w.T)
        res["shapes"][name] = {"N": N, "K": K, "rows": a.rows, "differ": differ, "kernels": seen}
        print(f"[pf-hc-rows] {name}: {len(differ)} row counts differ from the plain chain: {differ[:80]}", flush=True)
        for M, k in seen.items():
            print(f"[pf-hc-rows] {name} M={M}: {k}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
