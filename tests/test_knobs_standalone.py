"""#108: the standalone proxy (`tandem-dash`, its image) ships `proxy` and `server` without the engine, so
neither may need the engine to import.

  1  server/knobs.py's defaults are engine/settings.py's for every knob the conductor and the proxy read,
     and every such read is one of them                      test_the_defaults_are_the_registry
  2  with `engine` and `torch` missing, every module of proxy/, server/conductor/ and its monitor imports,
     and the knobs read the environment with those defaults   test_every_module_imports_without_the_engine
  3  no module of theirs imports the engine at its top but inside a guard   test_no_bare_engine_import

CPU. Run: python tests/test_knobs_standalone.py
"""

from __future__ import annotations

import ast
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DIRS = ("proxy", "server/conductor", "server/conductor/monitor")


def _files():
    return sorted(p for d in DIRS for p in glob.glob(os.path.join(ROOT, d, "*.py")))


def test_the_defaults_are_the_registry():
    from engine.settings import KNOBS
    from server import knobs
    assert knobs.DEFAULTS == {k: KNOBS[k][0] for k in knobs.DEFAULTS}, \
        {k: (v, KNOBS[k][0]) for k, v in knobs.DEFAULTS.items() if KNOBS[k][0] != v}
    read = set()
    for p in _files():
        read |= set(re.findall(r'_S\.get\("([A-Z0-9_]+)"', open(p, encoding="utf-8").read()))
    assert read and read <= set(knobs.DEFAULTS), sorted(read - set(knobs.DEFAULTS))
    return f"{len(knobs.DEFAULTS)} knobs, {len(read)} read"


BLOCKED = r'''
import importlib, os, pkgutil, sys
class Block:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in ("engine", "torch"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import proxy, server.conductor, server.conductor.monitor
bad = []
mods = [(p.__name__, m.name) for p in (proxy, server.conductor, server.conductor.monitor)
        for m in pkgutil.iter_modules(p.__path__) if m.name != "__main__"]
for pkg, name in mods:
    try:
        importlib.import_module(f"{pkg}.{name}")
    except Exception as e:
        bad.append(f"{pkg}.{name}: {type(e).__name__}: {e}")
from server import knobs
assert type(knobs.SETTINGS).__name__ == "StandaloneSettings", knobs.SETTINGS
os.environ["QWEN38_TOOL_FIT"] = "off"
assert knobs.SETTINGS.get("TOOL_FIT") == "off" and knobs.SETTINGS.get("NOTE_DEDUP") == "parts"
try:
    knobs.SETTINGS.get("NOT_A_KNOB")
except KeyError:
    pass
else:
    bad.append("an unknown knob was read")
print(len(mods), "modules")
print("\n".join(bad))
sys.exit(1 if bad else 0)
'''


def test_every_module_imports_without_the_engine():
    env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
    p = subprocess.run([sys.executable, "-c", BLOCKED, ROOT], capture_output=True, text=True, timeout=300,
                       env=env, cwd=ROOT)
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    return p.stdout.splitlines()[0]


def _guarded(tree) -> set[int]:
    """Line numbers of imports inside a try at module level."""
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    out.add(sub.lineno)
    return out


def test_no_bare_engine_import():
    bad = []
    for p in _files():
        tree = ast.parse(open(p, encoding="utf-8").read())
        ok = _guarded(tree)
        for node in tree.body:
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n == "engine" or n.startswith("engine.") for n in names) and node.lineno not in ok:
                bad.append(f"{os.path.relpath(p, ROOT)}:{node.lineno}")
    assert not bad, bad


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
