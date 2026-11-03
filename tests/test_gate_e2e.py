"""`python -m tools.gate run` end to end on the fake engine (no model, no GPU): base and head trees
from git, one fresh fake server per arm, the suites, the report. An A/A run passes with no resolved
difference; a head whose writer is planted at half speed is WORSE on decode; a head whose answers
are planted in title case is CHANGED_TEXT on lossless. Needs the engine's CPU packages
(torch) and a git checkout; skips otherwise.

Run: python tests/test_gate_e2e.py
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

FAKE = "server/fake_engine.py"


def _ready() -> str | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "no torch (the fake engine's server needs CPU torch)"
    if subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True).returncode != 0:
        return "not a git checkout"
    return None


def patch(old: str, new: str) -> str:
    """A patch of server/fake_engine.py replacing `old` (once) with `new`, as `git apply` reads it."""
    src = open(os.path.join(ROOT, FAKE)).read()
    assert src.count(old) == 1, old
    out = "".join(difflib.unified_diff(src.splitlines(True), src.replace(old, new).splitlines(True),
                                       f"a/{FAKE}", f"b/{FAKE}"))
    fd, path = tempfile.mkstemp(suffix=".patch")
    with os.fdopen(fd, "w") as fh:
        fh.write(out)
    return path


def gate(*args) -> dict:
    from ports import free_port
    out = tempfile.mkdtemp(prefix="gate-e2e-")
    cmd = [sys.executable, "-m", "tools.gate", "run", "--pack", "fake", "--hw", "fake",
           "--base", "HEAD", "--head", "HEAD", "--suite", "quick", "--port", str(free_port()),
           "--out", out, *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert os.path.exists(os.path.join(out, "gate-report.json")), r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.load(open(os.path.join(out, "gate-report.json")))
    rep["_md"] = open(os.path.join(out, "gate-report.md")).read()
    rep["_rc"] = r.returncode
    return rep


def by_id(rep: dict) -> dict:
    return {r["id"]: r for r in rep["rows"]}


def test_a_a_passes_with_no_resolved_difference():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    from tools.gate.schema import validate
    rep = gate()
    assert validate({k: v for k, v in rep.items() if not k.startswith("_")}) == []
    rows = by_id(rep)
    assert rep["verdict"] == "PASS", [(r["id"], r["verdict"], r["why"]) for r in rep["rows"]
                                      if r["verdict"] not in ("PASS", "SKIPPED")]
    assert not [r for r in rep["rows"] if r["verdict"] in ("INVALID", "FAIL")]
    assert rows["decode.c4"]["proof"]["head_at_once"] == [4], rows["decode.c4"]
    assert rows["lossless.card-code.greedy"]["verdict"] == "PASS"
    assert rows["api.leave"]["verdict"] == "PASS", rows["api.leave"]
    assert rows["long.max.needle"]["verdict"] in ("PASS", "SKIPPED")
    assert rep["knobs"]["head"]["lanes"] == 4 and rep["knobs"]["base"]["max_running"] == 4
    assert rep["base"]["tree_sha256"] == rep["head"]["tree_sha256"]
    assert rep["planted"] is None and "## Quality gate: PASS (quick)" in rep["_md"]


def test_a_planted_slowdown_is_worse():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    p = patch('tps = max(1.0, float(getattr(a, "fake_tps", 200.0)))',
              'tps = max(1.0, float(getattr(a, "fake_tps", 200.0))) / 2')
    rep = gate("--only", "decode", "--head-patch", p, "--note", "the writer at half speed")
    rows = by_id(rep)
    assert rows["decode.c1"]["verdict"] == "WORSE", rows["decode.c1"]
    assert rows["decode.c1"]["delta_pct"] < -30, rows["decode.c1"]
    assert rep["verdict"] == "WORSE" and rep["planted"]["note"] == "the writer at half speed"
    assert rep["base"]["tree_sha256"] != rep["head"]["tree_sha256"]


def test_a_planted_text_change_is_changed_text():
    why = _ready()
    if why:
        print(f"skipped: {why}")
        return
    p = patch('answer = " ".join(words).capitalize() + "."',
              'answer = " ".join(words).title() + "."')
    rep = gate("--only", "lossless", "--head-patch", p)
    changed = [r for r in rep["rows"] if r["verdict"] == "CHANGED_TEXT"]
    assert changed and rep["verdict"] == "CHANGED_TEXT", [(r["id"], r["verdict"]) for r in rep["rows"]]
    assert all(r["suite"] == "lossless" for r in changed)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
