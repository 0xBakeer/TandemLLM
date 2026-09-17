"""The same kernel on a COLD working set, which is the only way this engine ever reads a weight.

`tools/nvfp4_wide_probe.py` times one projection in a loop. One projection is 17 to 50 MB and this
board has a system-level cache in that neighbourhood, so the loop measures a kernel reading a
weight that is already resident -- and it reported 216-266 GB/s, which is 79-98 % of a 273 GB/s
board and should have been the giveaway. The verify pass reads 15 GB once; nothing is resident.

This walks a chain of DISTINCT weights whose total exceeds any cache, so every byte is a cold DRAM
read, and it reports the rate at each M and for each candidate tile. It answers two questions the
warm probe cannot: what this board actually gives a W4A16 kernel on cold bytes, and whether the M
dependence the warm probe found survives.
"""
import argparse, os, sys, time
import torch
sys.path.insert(0, "/home/user/qwen38-spark-engine")
from tools.nvfp4_linear import nvfp4_matmul, pick_config  # noqa: E402
from tools.quant_nvfp4 import quantize_clipped  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=17408)
ap.add_argument("--K", type=int, default=5120)
ap.add_argument("--gb", type=float, default=3.0, help="working set, GB")
ap.add_argument("--rows", default="1,2,4,8,14,16")
ap.add_argument("--reps", type=int, default=6)
a = ap.parse_args()

one = quantize_clipped(torch.randn(a.N, a.K, device="cuda", dtype=torch.float32) * 0.02, None)
nb = one.nbytes
n = max(2, int(a.gb * 1e9 / nb))
ws = [one] + [quantize_clipped(torch.randn(a.N, a.K, device="cuda", dtype=torch.float32) * 0.02,
                               None) for _ in range(n - 1)]
total = n * nb
print(f"N={a.N} K={a.K}  {nb / 1e6:.1f} MB each x {n} = {total / 1e9:.2f} GB working set")
print(f"{'M':>4} {'tile':>28} {'ms/chain':>10} {'GB/s':>8}")

DECODE = {"block_m": 16, "block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3}
for M in [int(x) for x in a.rows.split(",")]:
    x = torch.randn(M, a.K, device="cuda", dtype=torch.bfloat16)
    cands = [("decode m16 n16 k8 w1", DECODE)]
    blk = pick_config(a.N, a.K, M)
    if blk != DECODE:
        cands.append((f"block m{blk['block_m']} n{blk['block_n']} "
                      f"k{blk['split_k']} w{blk['num_warps']}", blk))
    for name, cfg in cands:
        for w in ws:                                   # warm the instruction cache, not the data
            nvfp4_matmul(x, w, **cfg)
        best = 1e9
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for w in ws:
                nvfp4_matmul(x, w, **cfg)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        print(f"{M:4d} {name:>28} {best * 1e3:10.2f} {total / best / 1e9:8.1f}", flush=True)
