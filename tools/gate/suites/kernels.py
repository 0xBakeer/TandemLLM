"""kernels: the tests/gpu files the diff's paths map to (tools/gate/map.yaml), each in a process of
its own in the arm's tree with no server running. The head's tree runs every file; the base's runs
only the files that fail on the head, so a failure main shares is SKIPPED, not blamed on the change."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from tools.gate.verdict import case_row, skipped

NAME = "kernels"


def run_file(tree: Path, f: str, python: str, timeout: float = 1800) -> dict:
    env = dict(os.environ, PYTHONPATH=str(tree), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.monotonic()
    try:
        p = subprocess.run([python, "-u", f], cwd=tree, env=env, capture_output=True, text=True,
                           timeout=timeout)
        rc, log = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        rc, log = 124, f"timed out after {timeout:.0f} s"
    last = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
    return {"rc": rc, "s": round(time.monotonic() - t0, 1), "last": last[0][:200],
            "skipped": rc == 0 and "skipped" in log.lower() and " passed" not in log}


def run_all(tree: Path, files: list[str], python: str, log=print) -> dict:
    out = {}
    for f in files:
        out[f] = run_file(tree, f, python)
        log(f"[kernels] {f}: exit {out[f]['rc']} in {out[f]['s']} s ({out[f]['last']})")
    return out


def rows(head: dict, base: dict, files: list[str], why_none: str | None = None) -> list[dict]:
    if why_none:
        return [skipped(f"kernels.{os.path.basename(f)[:-3]}", NAME, f, why_none) for f in files] or \
            [skipped("kernels.none", NAME, "GPU tests", why_none)]
    if not files:
        return [skipped("kernels.none", NAME, "GPU tests", "no GPU test maps to the changed paths")]
    out = []
    for f in files:
        h = head[f]
        rid = f"kernels.{os.path.basename(f)[:-3]}"
        if h["skipped"]:
            out.append(skipped(rid, NAME, f, f"the test skipped itself: {h['last']}"))
            continue
        b = base.get(f)
        out.append(case_row(rid, NAME, f, None if b is None else b["rc"] == 0, h["rc"] == 0,
                            f"exit {h['rc']} in {h['s']} s, {h['last']}"))
    return out
