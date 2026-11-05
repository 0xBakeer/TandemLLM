"""tools/gate/hw: the box's profile from faked nvidia-smi and torch answers, each adapter's answers,
and no GPU-specific setting outside the adapters. On a CPU.

Run: python tests/test_gate_hw.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.gate import hw  # noqa: E402
from tools.gate.hw.base import Fingerprint  # noqa: E402
from tools.gate.runner import load_packs  # noqa: E402

MEMINFO = "MemTotal:       125513680 kB\nMemAvailable:   47000000 kB\n"


def fake_run(smi: str | None, torch: dict | None):
    """A subprocess.run that answers nvidia-smi and the torch probe."""
    def run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            if smi is None:
                raise FileNotFoundError("nvidia-smi")
            return subprocess.CompletedProcess(cmd, 0, smi, "")
        if torch is None:
            return subprocess.CompletedProcess(cmd, 1, "", "no torch")
        return subprocess.CompletedProcess(cmd, 0, json.dumps(torch) + "\n", "")
    return run


def test_gb10_is_detected_with_unified_memory():
    fp = hw.detect(run=fake_run("NVIDIA GB10, [N/A], 580.173.02, 12.1\n",
                                {"torch": "2.13.0+cu130", "cuda": "13.0", "name": "NVIDIA GB10",
                                 "memory_gib": 119.7, "cc": "12.1"}), meminfo=MEMINFO)
    assert fp.name == "NVIDIA GB10" and fp.unified and fp.memory_gib == 119.7, fp
    assert fp.driver == "580.173.02" and fp.cc == "12.1" and fp.torch == "2.13.0+cu130" and fp.cuda == "13.0"
    p = hw.pick(fp)
    assert p.id == "gb10" and p.lanes() == 8 and p.decode_levels() == [8]
    assert p.pack() == "flashnext-3.05" and p.ceiling_gib() == 66.0 and p.peak_gib() == 77.0
    assert p.prefill_depths("quick") == [2048, 8192] and 65536 in p.prefill_depths("full")
    assert p.long_ctx("quick") == 65536 and p.long_ctx("quick+long") == 131072
    assert p.min_free_gib() == 70.0 and p.mem_floor_gib() == 10.0
    assert p.arm_env() == {"QSE_PARALLEL": "8", "MEM_CEILING_GIB": "66", "MEM_PEAK_GIB": "77",
                           "QSE_PARALLEL_FLOOR_GB": "10"}


def test_ada_sizes_follow_the_cards_memory():
    small = hw.pick(hw.detect(run=fake_run("NVIDIA GeForce RTX 4080, 16376, 575.51, 8.9\n",
                                           {"torch": "2.13.0", "cuda": "13.0"})))
    assert small.id == "ada" and small.fp.memory_gib == 16.0 and not small.fp.unified
    assert small.lanes() == 2 and small.decode_levels() == [2] and small.long_ctx("quick") == 16384
    assert small.pack() == "lfm25-exl3-2.5" and small.ceiling_gib() == 13.0
    big = hw.pick(Fingerprint(name="NVIDIA RTX 6000 Ada Generation", memory_gib=48.0, cc="8.9"))
    assert big.lanes() == 4 and big.pack() == "lfm25-exl3-4.0" and 32768 in big.prefill_depths("full")
    assert big.burst() == (4, 2048) and big.mem_floor_gib() is None


def test_an_unknown_gpu_gets_the_generic_profile_and_no_pack():
    p = hw.pick(hw.detect(run=fake_run("NVIDIA A100-SXM4-80GB, 81920, 550.54, 8.0\n", None)))
    assert p.id == "generic" and p.fp.memory_gib == 80.0 and p.fp.torch is None
    assert p.lanes() == 4 and p.pack() is None
    none = hw.pick(hw.detect(run=fake_run(None, None)))
    assert none.id == "generic" and none.fp.name == "unknown" and none.lanes() == 2
    # torch alone (no nvidia-smi) still names the card
    t = hw.detect(run=fake_run(None, {"torch": "2.13", "cuda": "13.0", "name": "NVIDIA L4",
                                      "memory_gib": 22.0, "cc": "8.9", "count": 1}))
    assert hw.pick(t).id == "ada" and t.memory_gib == 22.0


def test_named_profiles_and_the_fake():
    fp = Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True)
    assert hw.profile("generic", fp=fp).id == "generic"
    assert hw.profile("gb10", fp=fp).id == "gb10"
    assert hw.profile("fake").id == "fake" and hw.profile("fake").pack() == "fake"
    try:
        hw.profile("tpu", fp=fp)
        raise AssertionError("an unknown profile was accepted")
    except SystemExit as exc:
        assert "gb10" in str(exc)


def test_every_profile_answers_every_question():
    packs = load_packs()
    fps = [Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True),
           Fingerprint(name="RTX 4080", memory_gib=16.0, cc="8.9"),
           Fingerprint(name="A100", memory_gib=80.0, cc="8.0")]
    for p in [hw.pick(f) for f in fps] + [hw.profile("fake")]:
        d = p.describe()
        assert d["lanes"] >= 1 and d["decode_levels"] and d["prefill_depths"]["quick"]
        assert d["long_ctx"]["quick"] >= max(d["prefill_depths"]["quick"]), p.id
        assert d["pack"] is None or d["pack"] in packs, (p.id, d["pack"])
        assert d["arm_env"]["QSE_PARALLEL"] == str(d["lanes"])
        assert d["rounds"] >= 2, "the noise rule needs two runs a side"


def test_no_gpu_specific_setting_outside_the_adapters():
    pat = re.compile(r"MEM_CEILING_GIB|MEM_PEAK_GIB|QSE_PARALLEL_FLOOR|\bGB10\b|sm_\d|\b8\.9\b|12\.1")
    gate = os.path.join(ROOT, "tools", "gate")
    for root, _, files in os.walk(gate):
        if os.path.basename(root) == "hw" or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith((".py", ".yaml")):
                text = open(os.path.join(root, f)).read()
                assert not pat.search(text), (f, pat.search(text).group(0))


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
