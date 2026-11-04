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
    ver = release_version(data)
    page = d / "report.html"
    if not page.exists():                         # a page written under its asset name
        page = d / asset_name(data, ver)
    html = page.read_text()
    if a.tag.lstrip("v") != ver:
        print(f"[report] the page is of {ver}, the tag is {a.tag}: refused", file=sys.stderr)
        return 2
    bad = scrub.leaks(html, data.get("private"))
    if bad:
        print("[report] the page still names the box: " + "; ".join(bad[:8]), file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="report-"))
    asset = tmp / (a.name or asset_name(data, ver))
    shutil.copy(page, asset)
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


def combine(datas: list[dict], labels: list[str], model: str | None = None) -> dict:
    """Several packs' data.json of one release and one model as one data.json, for one page (render shows
    an overview of the packs side by side, then each pack's sections). Each pack keeps its own rows, runs
    and version string; the private terms are pooled for the scrub."""
    vers = {release_version(d) for d in datas}
    if len(vers) != 1:
        raise SystemExit(f"[report] the packs are of different releases ({', '.join(sorted(vers))}): refused")
    private: dict = {}
    packs = []
    for d, label in zip(datas, labels):
        if d.get("packs"):
            raise SystemExit("[report] combine takes one pack's data.json each")
        for k, v in (d.get("private") or {}).items():
            private[k] = sorted(set(private.get(k) or []) | set(v))
        packs.append(dict({k: v for k, v in d.items() if k != "private"}, label=label))
    first = datas[0]
    return {"kind": "tandem-benchmark-report", "schema": first.get("schema"), "tool": first.get("tool"),
            "version": vers.pop(), "model": model or first.get("model"), "hardware": first.get("hardware"),
            "packs": packs, "private": private}


def cmd_combine(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.release_report combine",
                                 description="several packs' data.json (one release, one model) as one data.json")
    ap.add_argument("data", nargs="+", help="each pack's data.json")
    ap.add_argument("--out", required=True, help="the combined data.json")
    ap.add_argument("--label", action="append", default=None,
                    help="a pack's name on the page, once per data.json in order (default: its weights)")
    ap.add_argument("--model", default=None, help="the model's name on the page (default: the first pack's)")
    a = ap.parse_args(argv)
    datas = [json.loads(Path(x).read_text()) for x in a.data]
    labels = a.label or [str(d.get("weights")) for d in datas]
    if len(labels) != len(datas):
        ap.error("one --label per data.json")
    out = combine(datas, labels, a.model)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"[report] {a.out}: {len(datas)} packs of {out['model']} {out['version']}")
    return 0


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
    if cmd == "combine":
        return cmd_combine(rest)
    print(f"unknown command {cmd!r}: run, plan, render, combine, publish", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
