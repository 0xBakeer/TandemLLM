"""tools/gate/map.yaml and tools/gate/pathmap.py: which gate a diff needs, on a CPU.

Run: python tests/test_gate_map.py
"""

from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.gate import LEVEL_SUITES  # noqa: E402
from tools.gate import pathmap  # noqa: E402


def need(*files):
    return pathmap.required(list(files))


def test_globs():
    m = pathmap.match
    assert m("docs/a/b.md", "docs/**") and m("docs/x.md", "docs/**")
    assert m("README.md", "*.md") and not m("docs/x.md", "*.md")
    assert m("engine/cache.py", "engine/{cache,kvpool}.py")
    assert not m("engine/models/cache.py", "engine/{cache,kvpool}.py")
    assert m("engine/models/qwen3_5/round/x.py", "engine/models/qwen3_5/round/**")
    assert m("a/b/c/kernels.py", "**/kernels.py") and m("kernels.py", "**/kernels.py")
    assert m("tools/csrc/fn_kernels/x.cu", "tools/csrc/fn_*/**")
    assert pathmap.braces("a{b,c}d{e,f}") == ["abde", "abdf", "acde", "acdf"]


def test_docs_and_dashboard_need_nothing():
    r = need("docs/exactness.md", "README.md", "dashboard/src/views/usage.ts", "CHANGELOG.md",
             "models/flint/community/x.json", "tests/test_cache.py", "chatd/agent.py")
    assert r["level"] == "none" and r["suites"] == [] and r["kernels"] == [], r
    assert need("tools/gate/runner.py", "tools/row3.py", "tools/parallel_card.py")["level"] == "none"


def test_engine_and_models_need_quick():
    r = need("engine/models/flashnext/engine.py")
    assert r["level"] == "quick", r
    assert r["suites"] == list(LEVEL_SUITES["quick"])
    assert "tests/gpu/test_fn_kernels_gpu.py" in r["kernels"]
    assert need("tools/nvfp4_skinny.py")["level"] == "quick"
    assert need("ops/profiles/flashnext-3.05.env")["level"] == "quick"


def test_memory_scheduler_server_need_long():
    for f in ("engine/kvpool.py", "server/slots.py", "server/round_runner.py",
              "engine/models/flashnext/serving.py", "ops/serve.env", "install.sh"):
        assert need(f)["level"] == "quick+long", f


def test_release_needs_full():
    r = need("VERSION", "docs/x.md")
    assert r["level"] == "full" and "stability" in r["suites"], r


def test_highest_level_wins_and_unmapped_is_quick():
    r = need("docs/a.md", "engine/kvpool.py", "engine/spec.py")
    assert r["level"] == "quick+long"
    assert r["files"]["docs/a.md"]["level"] == "none"
    u = need("somewhere/new.py")
    assert u["level"] == "quick" and u["files"]["somewhere/new.py"]["rule"] == "default"
    assert need()["level"] == "none"


def test_every_named_gpu_test_exists_and_every_gpu_test_is_named():
    m = pathmap.load()
    named = {t for k in m["kernels"] for t in k["tests"]}
    for t in named:
        assert os.path.isfile(os.path.join(ROOT, t)), t
    have = {os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "tests/gpu/test_*.py"))}
    assert have - named == set(), sorted(have - named)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
