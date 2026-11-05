"""The GPU adapters (tools/gpu): one adapter a GPU, picked by its capability, and no per-GPU choice
outside them.

  1  picking       each adapter claims its capability (and name); an unknown GPU and the CPU get the base
  2  gb10          the GB10's answers are the ones the engine gave before the adapters, exactly
  3  base, ada     no inline build, the Triton paths; Ada answers like the base until its kernels land
  4  once          `profile` asks the driver once a device
  5  users         gate/hw, device_id, the setup presets and the fingerprints read the adapters
  6  grep          no capability check, arch flag or GPU name test outside tools/gpu (allow-list below)

CPU. Run: python tests/test_gpu_adapters.py
"""

from __future__ import annotations

import os
import re
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
    from engine.settings import KNOBS
    assert KNOBS["FN_MOE_ST"] == ("3", "tools/gpu/gb10.py")
    assert KNOBS["FN_MOE_WARPS"] == ("4", "tools/gpu/gb10.py")
    return "(3, 4) by default, the knobs read live, owned by gb10.py"


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


def test_ada_owns_the_triton_probe_verdicts():
    """sm_89 runs no inline extension, so its kernels are the Triton ones in tools/exl3_ada.py
    and tools/moe_ada.py, reached through the adapter's probe and verdict answers. Off a real
    probe -- the CPU profile_for -- every verdict says no kernel, so the reference serves."""
    p = gpu.profile_for(*ADA)
    dev = "cpu"
    assert not p.e2m1() and not p.moe_ok(dev)              # no probe ever ran -> False, not a raise
    for q in (p.exl3_rot_ok, lambda d: p.exl3_decode_ok(d, 5), lambda d: p.exl3_gemv_ok(d, 5)):
        try:
            q(dev)
            raise AssertionError("an unprobed exl3 verdict did not raise")
        except RuntimeError:
            pass                                         # the dispatch's raise-unprobed contract
    p.exl3_probe(dev, 5)
    p.moe_probe(dev)                                     # cpu devices take no probe
    return "exl3 verdicts raise unprobed; moe_ok falls back; e2m1 False"


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


# ------------------------------------------------------------------ 5: users
def test_gate_hw_matches_through_tools_gpu():
    from tools.gate.hw import pick
    from tools.gate.hw.base import Fingerprint
    assert pick(Fingerprint(name="NVIDIA GB10", cc="12.1")).id == "gb10"
    assert pick(Fingerprint(name="NVIDIA GB10")).id == "gb10"
    assert pick(Fingerprint(name="unknown", cc="12.1")).id == "gb10"
    assert pick(Fingerprint(name="NVIDIA GeForce RTX 4090", cc="8.9", memory_gib=24.0)).id == "ada"
    assert pick(Fingerprint(name="NVIDIA H100", cc="9.0", memory_gib=80.0)).id == "generic"
    from tools.gate import hw
    assert hw.TORCH_PROBE is gpu.PROBE
    return "gb10 by capability or name, ada by capability, generic otherwise; tools/gpu's probe"


def test_device_keys_and_presets():
    from engine.methods.device import device_id
    assert device_id("NVIDIA GB10") == "dgx-spark"
    assert device_id("NVIDIA RTX PRO 6000 Blackwell") == "nvidia-rtx-pro-6000-blackwell"
    assert gpu.key_matches("dgx-spark", "NVIDIA GB10")
    assert not gpu.key_matches("dgx-spark", "NVIDIA GeForce RTX 4090")
    assert gpu.key_matches("nvidia-geforce-rtx-4090", "NVIDIA GB10"), "a key no adapter owns"
    from server.setup.device import preset
    m = {"devices": [{"id": "dgx-spark", "total_gib": 119.7}]}
    assert preset(m, 119.5, "NVIDIA GB10") == "dgx-spark"
    assert preset(m, 119.5, None) == "dgx-spark"
    assert preset(m, 119.5, "NVIDIA RTX PRO 6000 Blackwell") is None
    return "dgx-spark only on a GB10; any other preset by memory alone"


def test_the_fingerprints_read_the_adapters():
    from engine import disk_cache as dc
    got = {os.path.relpath(p, ROOT) for p in dc._tools_modules(ROOT, [os.path.join(ROOT, "tools", "fn_kernels.py")])}
    want = {os.path.join("tools", "gpu", f) for f in os.listdir(os.path.join(ROOT, "tools", "gpu")) if f.endswith(".py")}
    assert want <= got, sorted(want - got)
    return f"tools/fn_kernels.py reaches {len(want)} adapter files"


# ------------------------------------------------------------------ 6: grep
#: what a per-GPU choice looks like in Python or shell
PATTERNS = [r"get_device_capability\s*\(", r"-gencode", r"arch=compute_\d", r"TORCH_CUDA_ARCH_LIST",
            r"__CUDA_ARCH__", r"\(\s*12\s*,\s*1\s*\)", r"\(\s*8\s*,\s*9\s*\)", r"[\"']12\.1[\"']",
            r"[\"']8\.9[\"']", r"[\"']GB10[\"']\s+in\b", r"\.major\s*[<>=]", r"\bcc\s*=="]
#: files under the scan that may keep a GPU check, each with its reason
ALLOWED = {
    # the installer checks the GPU before the repository is on disk, to size the install
    "install.sh": "runs before the repository is on disk",
    # builds the third-party exllamav3 on a quantization box and logs its capability
    "tools/exl3/common.sh": "the third-party exllamav3 build, not this engine's kernels",
    # the standalone proxy ships without tools/; it reports a GB10 backend's memory as unified
    "proxy/hostcollect.py": "the standalone proxy has no tools/; a monitoring view, not an engine choice",
    "proxy/scrape.py": "the standalone proxy has no tools/; a monitoring view, not an engine choice",
    # a stub nvidia-smi that plays a GB10 for the release gate's install check
    "tools/release_gate_setup.sh": "a stub nvidia-smi, not a choice",
}
SCAN = ("engine", "tools", "server", "proxy", "chatd", "ops", "bench")


def _scanned():
    out = ["install.sh"]
    for top in SCAN:
        for d, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "node_modules")]
            for f in files:
                if f.endswith((".py", ".sh", ".env")):
                    out.append(os.path.relpath(os.path.join(d, f), ROOT))
    return sorted(out)


def test_no_gpu_choice_outside_tools_gpu():
    rx = re.compile("|".join(PATTERNS))
    bad, seen = [], set()
    for rel in _scanned():
        if rel.startswith(os.path.join("tools", "gpu") + os.sep):
            continue
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                code = line.split("#", 1)[0] if rel.endswith((".sh", ".env")) else line
                if line.lstrip().startswith("#") or not rx.search(code):
                    continue
                if rel in ALLOWED:
                    seen.add(rel)
                    continue
                bad.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not bad, "a GPU choice outside tools/gpu:\n" + "\n".join(bad)
    assert seen == set(ALLOWED), f"an allow-list entry no longer needed: {sorted(set(ALLOWED) - seen)}"
    return f"{len(_scanned())} files, {len(ALLOWED)} allowed with reasons"


def test_the_package_imports_without_torch():
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r)\n"
            "from tools import gpu\n"
            "assert gpu.profile_for((12, 1)).id == 'gb10' and gpu.by_name('NVIDIA GB10').unified_memory\n"
            "from tools.gate.hw import pick\n" % ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return "tools.gpu and tools.gate.hw import with torch blocked"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
