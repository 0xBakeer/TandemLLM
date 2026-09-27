"""ENG-123/124: every QWEN38_* knob the engine reads goes through engine/settings.py.

Pins the seam, not a behaviour: (1) no module of the engine's scope (engine/, server/, the kernel
and linear modules of tools/) reads a QWEN38_* variable any other way than `SETTINGS.get`; (2) every
name read is in the registry and every registry entry is read somewhere; (3) `SETTINGS` views the
live environment (so a value set before an import, or before a per-call read, is seen exactly as the
old `os.environ.get` saw it); (4) a second EngineSettings over another mapping or another prefix is
independent of the process's.
"""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.settings import KNOBS, SETTINGS, EngineSettings  # noqa: E402

KERNEL = re.compile(r"_kernels|fp8_linear|nvfp4_linear|nvfp4_skinny|nvfp4_verify_tiles|head_gemv|"
                    r"attn_|small_linear|loop_sync")
# reads that are a tool's own __main__ bench knobs, not the engine's
ALLOWED = {("tools/gdn_wy_kernels.py", "QWEN38_WY_GRID"), ("tools/gdn_wy_kernels.py", "QWEN38_WY_SIZES")}


def scope() -> list[str]:
    fs = glob.glob(os.path.join(ROOT, "engine", "**", "*.py"), recursive=True)
    fs += glob.glob(os.path.join(ROOT, "server", "*.py"))
    fs += [f for f in glob.glob(os.path.join(ROOT, "tools", "*.py")) if KERNEL.search(os.path.basename(f))]
    return sorted(f for f in fs if not f.endswith("engine/settings.py"))


def test_no_stray_reads():
    raw = re.compile(r"os\.(environ|getenv)[^\n]*QWEN38_[A-Z0-9_]+|QWEN38_[A-Z0-9_]+[\"']\s*in\s+os\.environ")
    bad = []
    for f in scope():
        rel = os.path.relpath(f, ROOT)
        for i, line in enumerate(open(f), 1):
            if raw.search(line):
                name = re.search(r"QWEN38_[A-Z0-9_]+", line).group(0)
                if (rel, name) not in ALLOWED:
                    bad.append(f"{rel}:{i}: {line.strip()[:90]}")
    assert not bad, "QWEN38_* read around engine/settings.py:\n" + "\n".join(bad)
    return f"{len(scope())} modules, every QWEN38_* read through SETTINGS"


def test_registry_matches_reads():
    used = set()
    for f in scope():
        used |= set(re.findall(r"_S\.get\(\s*\"([A-Z0-9_]+)\"", open(f).read()))
    computed = {"TREE_NODES", "TREE_NODES_NARROW", "SKINNY_TILES_B", "SKINNY_TILES_C", "SKINNY_TILES_WIDE_B"}
    missing = used - set(KNOBS)
    unused = set(KNOBS) - used - computed
    assert not missing, f"read but not registered: {sorted(missing)}"
    assert not unused, f"registered but never read: {sorted(unused)}"
    for k, (d, where) in KNOBS.items():
        assert d is None or isinstance(d, str), (k, d)
        assert os.path.isfile(os.path.join(ROOT, where)), (k, where)
    return f"{len(KNOBS)} knobs, all read, all registered"


def test_live_environment():
    os.environ.pop("QWEN38_FUSE_PROJ", None)
    assert SETTINGS.get("FUSE_PROJ") == KNOBS["FUSE_PROJ"][0]
    os.environ["QWEN38_FUSE_PROJ"] = "1"
    try:
        assert SETTINGS.get("FUSE_PROJ") == "1" and SETTINGS.is_set("FUSE_PROJ")
    finally:
        os.environ.pop("QWEN38_FUSE_PROJ")
    try:
        SETTINGS.get("NOT_A_KNOB")
    except KeyError:
        pass
    else:
        raise AssertionError("an unregistered knob was read")
    return "reads the live environment; an unknown name is an error"


def test_second_configuration_and_prefix():
    other = EngineSettings(env={"QSE_FUSE_PROJ": "1", "QSE_VERIFY_ROWS": "32"}, prefix="QSE_")
    assert other.get("FUSE_PROJ") == "1" and other.get("VERIFY_ROWS") == "32"
    assert other.get("TREE_MS") == KNOBS["TREE_MS"][0]
    d = other.describe()
    assert d["FUSE_PROJ"]["set"] and not d["TREE_MS"]["set"] and len(d) == len(KNOBS)
    return "a mapping with prefix QSE_ is its own configuration; describe() lists every knob"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
