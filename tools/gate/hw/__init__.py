"""The box's GPU profile: detected once from nvidia-smi and the engine's torch, then the first adapter
that claims it, else the generic one (tools/gate/hw/base.py lists the questions).

    profile("auto")    detect and pick
    profile("gb10")    a named adapter, on the detected fingerprint
    profile("fake")    the fake engine's box (no GPU)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tools.gate.hw.ada import Ada
from tools.gate.hw.base import Fingerprint, Profile
from tools.gate.hw.fake import Fake
from tools.gate.hw.gb10 import GB10
from tools.gate.hw.generic import Generic

#: adapters in the order they are asked; the generic one answers for any GPU none claims
ADAPTERS: list[type[Profile]] = [GB10, Ada]

TORCH_PROBE = (
    "import json, torch\n"
    "d = {'torch': torch.__version__, 'cuda': torch.version.cuda}\n"
    "if torch.cuda.is_available():\n"
    "    p = torch.cuda.get_device_properties(0)\n"
    "    d.update(name=p.name, memory_gib=round(p.total_memory / 2**30, 1),\n"
    "             cc=f'{p.major}.{p.minor}', count=torch.cuda.device_count())\n"
    "print(json.dumps(d))\n")


def _run(cmd, run=subprocess.run, timeout=60) -> str | None:
    try:
        r = run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def meminfo_gib(text: str | None = None) -> float | None:
    try:
        text = text if text is not None else open("/proc/meminfo").read()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 2**20, 1)
    return None


def detect(python: str | None = None, run=subprocess.run, meminfo: str | None = None) -> Fingerprint:
    """The fingerprint from `nvidia-smi` (name, memory, driver, compute capability) and the engine's
    torch (`python`, default this interpreter: version, CUDA, and the memory torch sees). A GPU that
    shares the host's memory (nvidia-smi says [N/A]) takes MemTotal and is marked unified."""
    fp = Fingerprint()
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"], run)
    if out:
        lines = [ln for ln in out.splitlines() if ln.strip()]
        fp.count = len(lines)
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 4:
            fp.name, mem, fp.driver, fp.cc = parts[0], parts[1], parts[2], parts[3]
            try:
                fp.memory_gib = round(float(mem) / 1024, 1)
            except ValueError:
                fp.unified = True
    out = _run([python or sys.executable, "-c", TORCH_PROBE], run, timeout=120)
    if out:
        try:
            t = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            t = {}
        fp.torch, fp.cuda = t.get("torch"), t.get("cuda")
        if fp.name == "unknown" and t.get("name"):
            fp.name, fp.cc, fp.count = t["name"], t.get("cc"), t.get("count", 1)
        if fp.memory_gib is None and not fp.unified and t.get("memory_gib"):
            fp.memory_gib = t["memory_gib"]
    if fp.unified:
        fp.memory_gib = meminfo_gib(meminfo)
    return fp


def pick(fp: Fingerprint) -> Profile:
    for cls in ADAPTERS:
        if cls.matches(fp):
            return cls(fp)
    return Generic(fp)


def profile(hw: str = "auto", fp: Fingerprint | None = None, python: str | None = None) -> Profile:
    if hw == "fake":
        return Fake()
    fp = fp or detect(python)
    if hw == "auto":
        return pick(fp)
    for cls in ADAPTERS + [Generic]:
        if cls.id == hw:
            return cls(fp)
    raise SystemExit(f"[gate] no hardware profile {hw!r}: auto, fake, "
                     + ", ".join(c.id for c in ADAPTERS + [Generic]))


def no_gpu() -> bool:
    return os.environ.get("CUDA_VISIBLE_DEVICES") == "" or _run(["nvidia-smi", "-L"]) is None
