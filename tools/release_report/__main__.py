"""python -m tools.release_report {run,plan,render,publish} (tools/release_report/__init__.py)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def cmd_render(argv) -> int:
    from tools.release_report import render
    ap = argparse.ArgumentParser(prog="python -m tools.release_report render")
    ap.add_argument("data")
    ap.add_argument("--out", default=None, help="default: report.html beside data.json")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.data).read_text())
    html = render.page(data)
    out = Path(a.out or Path(a.data).with_name("report.html"))
    out.write_text(html)
    print(f"[report] {out} ({len(html.encode()) / 1024:.0f} KB)")
    return 0


def release_version(data: dict) -> str:
    return re.split(r"[+\s]", str(data.get("version") or ""))[0]


def cmd_publish(argv) -> int:
    """The maintainer's upload of a checked page to the GitHub release (it measures nothing)."""
    from tools.release_report import scrub
    ap = argparse.ArgumentParser(prog="python -m tools.release_report publish",
                                 description="upload DIR/report.html to the GitHub release of --tag")
    ap.add_argument("dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo", default="0xBakeer/TandemLLM")
    ap.add_argument("--name", default=None, help="the asset's file name (default: "
                    "tandemllm-<version>-report-<model>.html; name the pack too for two packs of one model)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    data = json.loads((d / "data.json").read_text())
    html = (d / "report.html").read_text()
    ver = release_version(data)
    if a.tag.lstrip("v") != ver:
        print(f"[report] the page is of {ver}, the tag is {a.tag}: refused", file=sys.stderr)
        return 2
    bad = scrub.leaks(html, data.get("private"))
    if bad:
        print("[report] the page still names the box: " + "; ".join(bad[:8]), file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="report-"))
    asset = tmp / (a.name or asset_name(data, ver))
    shutil.copy(d / "report.html", asset)
    cmd = ["gh", "release", "upload", a.tag, str(asset), "--clobber", "--repo", a.repo]
    print("[report] " + " ".join(cmd))
    if a.dry_run:
        return 0
    return subprocess.run(cmd).returncode


def asset_name(data: dict, ver: str) -> str:
    """The release asset's name: the version and the model, so each model's page sits beside the
    others on one release (Qwen3.8-Flash-Next -> qwen3.8-flash-next)."""
    model = re.sub(r"[^a-z0-9.]+", "-", str(data.get("model") or "model").lower()).strip("-.")
    return f"tandemllm-{ver}-report-{model}.html"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        from tools import release_report
        print(release_report.__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        from tools.release_report import runner
        return runner.main(rest)
    if cmd == "plan":
        from tools.release_report import runner
        return runner.plan_main(rest)
    if cmd == "render":
        return cmd_render(rest)
    if cmd == "publish":
        return cmd_publish(rest)
    print(f"unknown command {cmd!r}: run, plan, render, publish", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
