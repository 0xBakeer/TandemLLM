"""python -m tools.measure {rows,plan,run,merge,render,index} (tools/measure/__init__.py)."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def cmd_rows(argv) -> int:
    from tools.measure import rows as rowsmod
    ap = argparse.ArgumentParser(prog="python -m tools.measure rows")
    ap.add_argument("--view", choices=("report", "card", "gate"), default=None)
    a = ap.parse_args(argv)
    rows, _ = rowsmod.load()
    sel = rowsmod.for_view(rows, a.view) if a.view else list(rows.values())
    for r in sel:
        print(f"{r.id:28} {r.server or '-':11} {r.est:5.1f} min  {','.join(r.views):18} {r.measures}")
    return 0


def cmd_render(argv) -> int:
    from tools.measure import store as storemod, views
    if not argv or argv[0] not in ("report", "card", "gate"):
        print("python -m tools.measure render report|card|gate ...", file=sys.stderr)
        return 2
    what, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"python -m tools.measure render {what}")
    if what == "gate":
        ap.add_argument("--base", required=True, help="the base side's record store")
        ap.add_argument("--head", required=True, help="the head side's record store")
        ap.add_argument("--build", default=None)
        ap.add_argument("--pack", default=None)
        ap.add_argument("--hw", default=None)
        ap.add_argument("--json", default=None, help="also write the rows as JSON")
        a = ap.parse_args(rest)
        rows, md = views.render_gate(a.base, a.head, a.build, a.pack, a.hw)
        sys.stdout.write(md)
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(rows, fh, indent=1)
        return 0
    ap.add_argument("--out", required=True, help="the record store")
    ap.add_argument("--build", default=None)
    ap.add_argument("--pack", action="append", default=None, help="a pack; several: one page of one model")
    ap.add_argument("--box", default=None)
    if what == "report":
        ap.add_argument("--label", action="append", default=None, help="a pack's name on the page, in --pack order")
        ap.add_argument("--model", default=None)
        ap.add_argument("--html", required=True, help="the page to write")
        ap.add_argument("--data", default=None, help="also write the assembled data.json")
        a = ap.parse_args(rest)
        st = storemod.Store(a.out)
        html = views.render_report(st, a.build, a.pack, a.label, a.model, a.html, a.data, a.box)
        print(f"[measure] {a.html} ({len(html.encode()) / 1024:.0f} KB)")
        return 0
    ap.add_argument("--model-id", required=True, help="the model's id (models/<id>)")
    ap.add_argument("--profile", required=True, help="the weight set (ops/profiles name or community/...)")
    ap.add_argument("--env-name", default="measure", help="the setup's name on the row")
    ap.add_argument("--commit", default=None, help="the served commit (default: the records' source)")
    ap.add_argument("--write", default=None, help="write the card tool's raw file here (default: print the row)")
    a = ap.parse_args(rest)
    st = storemod.Store(a.out)
    b, p = views.pick(st, a.build, (a.pack or [None])[0])
    raw = views.card_raw(st, b, p, a.model_id, a.profile, a.env_name, a.box, a.commit)
    from tools import parallel_card as pc
    if a.write:
        with open(a.write, "w") as fh:
            json.dump(raw, fh, indent=1)
            fh.write("\n")
        print(f"[measure] {a.write}")
    sys.stdout.write(pc.row_yaml(raw["row"]))
    for rid, src in raw["sources"].items():
        print(f"#   {rid}: {src}")
    return 0 if raw["valid"] else 1


def cmd_index(argv) -> int:
    from tools.measure import store as storemod
    ap = argparse.ArgumentParser(prog="python -m tools.measure index")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    idx = storemod.Store(a.out).write_index()
    print(f"[measure] {len(idx)} keys")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        from tools import measure
        print(measure.__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        from tools.measure import run
        return run.main(rest)
    if cmd == "plan":
        from tools.measure import plan
        return plan.main(rest)
    if cmd == "merge":
        from tools.measure import merge
        return merge.main(rest)
    if cmd == "compare":
        from tools.measure import compare
        return compare.main(rest)
    cmds = {"rows": cmd_rows, "render": cmd_render, "index": cmd_index}
    if cmd not in cmds:
        print(f"unknown command {cmd!r}: rows, plan, run, merge, render, compare, index", file=sys.stderr)
        return 2
    return cmds[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
