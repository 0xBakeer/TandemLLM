"""python -m tools.gate {run,map,hw,explain,render,check} (tools/gate/__init__.py)."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.gate import MIN_REASON, NEEDS_REASON  # noqa: E402


def cmd_map(argv) -> int:
    from tools.gate import pathmap
    ap = argparse.ArgumentParser(prog="python -m tools.gate map")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("files", nargs="*", help="these paths instead of the diff")
    a = ap.parse_args(argv)
    files = a.files or pathmap.changed_files(a.base, a.head, REPO)
    print(json.dumps(pathmap.required(files), indent=1))
    return 0


def cmd_hw(argv) -> int:
    from tools.gate import hw
    ap = argparse.ArgumentParser(prog="python -m tools.gate hw")
    ap.add_argument("--hw", default="auto")
    ap.add_argument("--python", default=sys.executable, help="the engine's Python (its torch)")
    a = ap.parse_args(argv)
    p = hw.profile(a.hw, python=a.python)
    print(json.dumps(p.describe(), indent=1))
    return 0


def cmd_explain(argv) -> int:
    from tools.gate import render
    ap = argparse.ArgumentParser(prog="python -m tools.gate explain",
                                 description="a written reason for a WORSE or CHANGED_TEXT row")
    ap.add_argument("report")
    ap.add_argument("row")
    ap.add_argument("reason")
    a = ap.parse_args(argv)
    with open(a.report) as fh:
        rep = json.load(fh)
    rows = {r["id"]: r for r in rep["rows"]}
    if a.row not in rows:
        print(f"no row {a.row!r}; rows: {', '.join(rows)}", file=sys.stderr)
        return 2
    if rows[a.row]["verdict"] not in NEEDS_REASON:
        print(f"{a.row} is {rows[a.row]['verdict']}: only WORSE and CHANGED_TEXT rows take a reason",
              file=sys.stderr)
        return 2
    if len(a.reason.strip()) < MIN_REASON:
        print(f"a reason needs at least {MIN_REASON} characters", file=sys.stderr)
        return 2
    rep.setdefault("reasons", {})[a.row] = a.reason.strip()
    write(rep, os.path.dirname(os.path.abspath(a.report)))
    print(f"[gate] {a.row}: reason written; paste gate-report.md again")
    return 0


def write(rep: dict, out: str) -> tuple[str, str]:
    from tools.gate import render
    os.makedirs(out, exist_ok=True)
    jp, mp = os.path.join(out, "gate-report.json"), os.path.join(out, "gate-report.md")
    with open(jp, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
        fh.write("\n")
    with open(mp, "w") as fh:
        fh.write(render.markdown(rep))
    return jp, mp


def cmd_render(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate render")
    ap.add_argument("report")
    a = ap.parse_args(argv)
    with open(a.report) as fh:
        rep = json.load(fh)
    from tools.gate import render
    sys.stdout.write(render.markdown(rep))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmds = {"map": cmd_map, "hw": cmd_hw, "explain": cmd_explain, "render": cmd_render}
    if not argv or argv[0] in ("-h", "--help"):
        from tools import gate
        print(gate.__doc__)
        return 0
    if argv[0] == "check":
        from tools.gate import check
        return check.main(argv[1:])
    if argv[0] == "run":
        from tools.gate import runner
        return runner.main(argv[1:])
    if argv[0] not in cmds:
        print(f"unknown command {argv[0]!r}: run, map, hw, explain, render, check", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
