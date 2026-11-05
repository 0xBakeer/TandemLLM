"""On the board: the device the engine runs on picks its adapter (tools/gpu), and the inline builds use
its answers.

Run on the box under the lock (it needs CUDA). On the DGX Spark the adapter is gb10: sm_121a flags for
every inline extension, PDL, 32 skinny rows; the fused kernels build with them and the MoE launch
shape is the knobs' (tests/test_gpu_adapters.py holds the same answers on the CPU).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools import gpu  # noqa: E402
from tools.gpu import INLINE  # noqa: E402


def test_the_device_picks_its_adapter():
    cap = tuple(torch.cuda.get_device_capability(0))
    p = gpu.profile(torch.device("cuda", 0))
    assert p is gpu.profile() and p is gpu.profile(0)
    assert p.cap == cap and type(p) is gpu.adapter_for(cap, torch.cuda.get_device_name(0))
    if cap == (12, 1):
        assert p.id == "gb10" and p.skinny_rows() == 32 and p.pdl()
        assert all(p.arch_flags(k) == ["-gencode=arch=compute_121a,code=sm_121a"] for k in INLINE)
    return f"{p!r}"


def test_the_fused_kernels_build_with_its_answers():
    p = gpu.profile()
    if not p.inline("fn_kernels"):
        return f"skipped: {p.id} builds no fn_kernels"
    from tools import fn_kernels as FK
    mod = FK._module()
    assert mod is FK._module()
    st, warps = p.moe_launch()
    assert (st, warps) == (int(os.environ.get("QWEN38_FN_MOE_ST", "3")), int(os.environ.get("QWEN38_FN_MOE_WARPS", "4")))
    assert FK.dense_split(2048, 4096, 4) == p.dense_split(2048, 4096, 4, FK.DENSE_SPLIT)
    return f"fn_kernels built for {p.id}, MoE launch ({st}, {warps})"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
