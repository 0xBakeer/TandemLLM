"""The GPU adapters (tools/gpu): one adapter a GPU, picked by its capability, and no per-GPU choice
outside them.

  1  picking       each adapter claims its capability (and name); an unknown GPU and the CPU get the base
  2  gb10          the GB10's answers are the ones the engine gave before the adapters, exactly
  3  base, ada     no inline build, the Triton paths; Ada answers like the base until its kernels land
  4  once          `profile` asks the driver once a device

CPU. Run: python tests/test_gpu_adapters.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools import gpu  # noqa: E402
from tools.gpu import INLINE, NoInlineBuild  # noqa: E402

GB10 = ((12, 1), "NVIDIA GB10")
ADA = ((8, 9), "NVIDIA GeForce RTX 4090")
OTHERS = [((9, 0), "NVIDIA H100 80GB HBM3"), ((12, 0), "NVIDIA RTX PRO 6000 Blackwell"),
          ((7, 5), "Tesla T4"), (None, None)]


def _main_dense_split(N, Kin, kb, table):
    """tools/fn_kernels.py `dense_split` before the adapters (077c2ea7), verbatim."""
    got = table.get((N, Kin, kb))
    if got:
        return got
    S = 1
    while (N // 128) * S < 96 and S < 8 and (Kin // 16) % (8 * S * 2) == 0:
        S *= 2
    return S, 4


# ------------------------------------------------------------------ 1: picking
def test_each_adapter_is_picked_by_its_capability():
    assert gpu.profile_for(*GB10).id == "gb10"
    assert gpu.profile_for((12, 1)).id == "gb10", "the capability alone"
    assert gpu.by_name("NVIDIA GB10").id == "gb10", "the name alone (nvidia-smi rows)"
    assert gpu.profile_for(*ADA).id == "ada" and gpu.profile_for((8, 9)).id == "ada"
    for cap, name in OTHERS:
        assert gpu.profile_for(cap, name).id == "base", (cap, name)
    assert gpu.profile_for(*GB10) is gpu.profile_for(*GB10), "one profile a (capability, name)"
    assert [a.id for a in gpu.ADAPTERS] == ["gb10", "ada"]
    assert gpu.parse_cc("12.1") == (12, 1) and gpu.parse_cc(None) is None and gpu.parse_cc("x") is None
    return "sm_121 -> gb10, sm_89 -> ada, sm_90 / sm_120 / sm_75 / the CPU -> base"


# ------------------------------------------------------------------ 2: gb10
def test_gb10_gives_mains_answers():
    p = gpu.profile_for(*GB10)
    for kind in INLINE:
        assert p.arch_flags(kind) == ["-gencode=arch=compute_121a,code=sm_121a"], kind
        assert p.inline(kind), kind
    assert p.pdl() and p.skinny_rows() == 32 and p.bf16() and p.unified_memory
    assert p.device_key() == "dgx-spark"
    try:
        p.arch_flags("no_such_extension")
        raise AssertionError("an unknown extension name was answered")
    except ValueError:
        pass
    return f"{len(INLINE)} extensions on sm_121a, PDL, 32 skinny rows, bf16, dgx-spark, unified"


def test_gb10_moe_launch_reads_its_knobs_with_mains_defaults():
    p = gpu.profile_for(*GB10)
    old = {k: os.environ.pop(k, None) for k in ("QWEN38_FN_MOE_ST", "QWEN38_FN_MOE_WARPS")}
    try:
        assert p.moe_launch() == (3, 4)
        os.environ.update(QWEN38_FN_MOE_ST="2", QWEN38_FN_MOE_WARPS="8")
        assert p.moe_launch() == (2, 8)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return "(3, 4) by default, the knobs read live"


def test_gb10_dense_split_is_mains():
    p = gpu.profile_for(*GB10)
    table = {(2048, 4096, 4): (2, 8), (6144, 2048, 3): (1, 4)}
    n = 0
    for N in (128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 151936):
        for Kin in (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 17408):
            for kb in (2, 3, 4, 6, 8, 25, 35):
                for t in (table, {}):
                    assert p.dense_split(N, Kin, kb, t) == _main_dense_split(N, Kin, kb, t), (N, Kin, kb)
                    n += 1
    return f"{n} shapes, table and rule, equal to main's"


# ------------------------------------------------------------------ 3: base, ada
def _base_like(p):
    for kind in INLINE:
        try:
            p.arch_flags(kind)
            raise AssertionError(f"{p.id} built {kind}")
        except NoInlineBuild as e:
            assert kind in str(e)
        assert not p.inline(kind)
    for q in (p.moe_launch, lambda: p.dense_split(2048, 4096, 4, {})):
        try:
            q()
            raise AssertionError(f"{p.id} answered a launch shape of a kernel it does not build")
        except NoInlineBuild:
            pass
    assert not p.pdl() and p.skinny_rows() == 0 and not p.unified_memory


def test_an_unknown_gpu_and_the_cpu_get_the_base():
    for cap, name in OTHERS:
        p = gpu.profile_for(cap, name)
        _base_like(p)
        assert p.bf16() == (cap is not None and cap[0] >= 8), cap
    assert gpu.profile_for(None, None).device_key() == "cpu"
    assert gpu.by_name("NVIDIA RTX PRO 6000 Blackwell").device_key() == "nvidia-rtx-pro-6000-blackwell"
    return "no inline build, no skinny rows, no PDL; bf16 from sm_80; the name as the device key"


def test_ada_answers_like_the_base_on_main():
    p = gpu.profile_for(*ADA)
    _base_like(p)
    assert p.bf16() and p.device_key() == "nvidia-geforce-rtx-4090"
    return "sm_89: the base's answers, bf16"


# ------------------------------------------------------------------ 4: once
def test_profile_asks_the_driver_once_a_device():
    import torch
    calls = []
    saved = {k: getattr(torch.cuda, k) for k in ("is_available", "current_device",
                                                   "get_device_capability", "get_device_name")}
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda i=None: calls.append(("cap", i)) or (12, 1)
    torch.cuda.get_device_name = lambda i=None: calls.append(("name", i)) or "NVIDIA GB10"
    gpu._reset_for_tests()
    try:
        a = gpu.profile()
        b = gpu.profile(torch.device("cuda", 0))
        c = gpu.profile(0)
        for _ in range(100):
            assert gpu.profile() is a
        assert a.id == "gb10" and a is b is c
        assert gpu.capability(0) == (12, 1)
        assert sum(1 for k, _ in calls if k == "cap") == 1, calls
        assert gpu.profile("cpu").id == "base"
    finally:
        for k, v in saved.items():
            setattr(torch.cuda, k, v)
        gpu._reset_for_tests()
    return f"{len(calls)} driver calls for 103 asks on one device"


def test_the_package_imports_without_torch():
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r)\n"
            "from tools import gpu\n"
            "assert gpu.profile_for((12, 1)).id == 'gb10' and gpu.by_name('NVIDIA GB10').unified_memory\n"
            % ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return "tools.gpu imports with torch blocked"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
