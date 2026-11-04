"""`python -m tools.measure plan`: for a view (or `--rows`) and one or more packs, every row -> the record
that holds it (with its source) or the hold that will measure it, with a time estimate.

Rows to measure are grouped by server (rows.yaml `server`: one fresh server start a group, its rows in
the list's order) and the groups packed into holds of at most HOLD_MAX_MIN minutes on the box (the
measuring, the server starts, and live's stop and restore). It refuses (exit 2) when a row would be
measured twice: a row asked by id that has a record, a warm row whose earlier rows have records (it
would measure them again), or an executor that measures a row with a record beside the one asked."""

from __future__ import annotations

import argparse
import os
import sys

from tools.measure import HOLD_MAX_MIN, HOLD_OVERHEAD_MIN, SERVER_START_MIN, rows as rowsmod, store as storemod

#: executors that measure all their rows at once (they cannot leave one out)
UNSPLIT = {"card.decode": "decode.*.c1", "gate.lossless": "lossless.set", "gate.stability": "stability.30m"}


def source_label(rec: dict) -> str:
    s = rec.get("source") or {}
    tail = os.path.basename(str(s.get("path") or ""))
    if s.get("how") == "imported":
        return f"{s.get('tool')}, {s.get('date')}, {s.get('path')}"
    return f"measured {s.get('date')} ({s.get('tool')}, {tail})"


def make(st: storemod.Store, packs: list[str], build: str, want: list, box: str | None, scale: dict,
         explicit: bool, force: bool = False) -> dict:
    rows, groups = rowsmod.load()
    lines, todo, refusals = [], [], []
    for pack in packs:
        have = {r.id: st.primary(r.id, build, pack, box) for r in want}
        miss = [r for r in want if force or not have[r.id]]
        for r in want:
            rec = have[r.id]
            if rec and not force:
                also = len([x for x in st.find(r.id, build, pack, box) if not x["_primary"]])
                lines.append((pack, r.id, "record", source_label(rec) + (f" (+{also} more)" if also else "")
                              + ("" if rec.get("valid") or rec.get("skipped") else " — INVALID: " + str(rec.get("why"))[:80])))
                if explicit:
                    refusals.append(f"{pack} {r.id}: has a record ({source_label(rec)}); asked to measure it again")
        # a warm row needs its earlier rows on the same server: their records would be measured again
        for r in miss:
            for pat in r.after:
                done = [x.id for x in rows.values() if rowsmod.matches(x.id, [pat]) and x.server == r.server and have.get(x.id)]
                if done and not force:
                    refusals.append(f"{pack} {r.id}: runs after {', '.join(done[:3])} on one server, and they have "
                                    "records: measuring it would measure them again")
            if r.exec in UNSPLIT:
                sib = [x.id for x in rows.values() if x.exec == r.exec and x.id != r.id and have.get(x.id)]
                if sib and not force:
                    refusals.append(f"{pack} {r.id}: its code ({r.exec}) also measures {', '.join(sib)}, which has "
                                    "a record")
        miss_ids = {r.id for r in miss}
        for r in want:
            if r.id in miss_ids:
                todo.append((pack, r))
    # server groups: (pack, group) -> rows in order
    gl: dict = {}
    for pack, r in todo:
        gl.setdefault((pack, r.server), []).append(r)
    cap = HOLD_MAX_MIN - HOLD_OVERHEAD_MIN
    items = []
    for (pack, g), rs in gl.items():
        sc = scale.get(pack, 1.0)
        part, mins = [], 0.0
        for r in rs:                          # a group over a hold: consecutive parts, each a server start
            m = r.est * sc
            if part and mins + m + SERVER_START_MIN > cap:
                items.append((pack, g, part, round(mins + SERVER_START_MIN, 1)))
                part, mins = [], 0.0
            part.append(r)
            mins += m
        if part:
            items.append((pack, g, part, round(mins + SERVER_START_MIN, 1)))
    split = {(p, g) for p, g, _, _ in items if sum(1 for x in items if x[:2] == (p, g)) > 1}
    for p, g in split:
        if any(r.after for r in gl[(p, g)]):
            refusals.append(f"{p} group {g} does not fit one hold and has warm rows that need the whole group "
                            "on one server: shorten it or raise the estimate scale knowingly")
    holds: list = []
    for it in sorted(items, key=lambda x: -x[3]):          # first fit, longest first
        for h in holds:
            if h["minutes"] + it[3] <= cap:
                h["items"].append(it)
                h["minutes"] = round(h["minutes"] + it[3], 1)
                break
        else:
            holds.append({"items": [it], "minutes": it[3]})
    for h in holds:                                       # within a hold: the plan's group order
        order = list(groups)
        h["items"].sort(key=lambda it: (packs.index(it[0]), order.index(it[1]) if it[1] in order else 99))
    return {"lines": lines, "todo": todo, "holds": holds, "refusals": refusals, "groups": groups, "cap": cap}


def render(pl: dict, view: str, build: str, packs: list[str], box: str | None, scale: dict) -> str:
    out = [f"# tools/measure plan: view {view} · build {build} · {', '.join(packs)} · box {box or 'any'}", ""]
    out.append("| pack | row | state | source / hold |")
    out.append("|---|---|---|---|")
    where = {}
    for i, h in enumerate(pl["holds"], 1):
        for pack, g, rs, _ in h["items"]:
            for r in rs:
                where[(pack, r.id)] = f"hold {i}, server {g}"
    rows_out = [(p, rid, st, src) for p, rid, st, src in pl["lines"]]
    rows_out += [(p, r.id, "measure", where.get((p, r.id), "?") + f" (~{r.est * scale.get(p, 1.0):.1f} min)")
                 for p, r in pl["todo"]]
    order, _ = rowsmod.load()
    ids = list(order)
    rows_out.sort(key=lambda x: (packs.index(x[0]), ids.index(x[1]) if x[1] in ids else 999))
    for p, rid, st, src in rows_out:
        out.append(f"| {p} | {rid} | {st} | {src} |")
    out.append("")
    n_srv = sum(len(h["items"]) for h in pl["holds"])
    tot = round(sum(h["minutes"] for h in pl["holds"]), 1)
    out.append(f"## Holds: {len(pl['holds'])} (each <= {HOLD_MAX_MIN} min: <= {pl['cap']:g} min measuring and "
               f"server starts + {HOLD_OVERHEAD_MIN:g} min live stop, restore and idle check); "
               f"{n_srv} server starts; ~{tot} min on the GPU")
    for i, h in enumerate(pl["holds"], 1):
        out.append(f"\n### Hold {i}: ~{h['minutes']} min + {HOLD_OVERHEAD_MIN:g}")
        for pack, g, rs, m in h["items"]:
            doc = (pl["groups"].get(g) or {}).get("doc", "")
            out.append(f"- {pack}, server `{g}` ({doc}; ~{m} min incl. a {SERVER_START_MIN:g} min start): "
                       + ", ".join(r.id for r in rs))
    if scale and any(v != 1.0 for v in scale.values()):
        out.append("\nEstimate scale (row minutes x): " + ", ".join(f"{k} {v:g}" for k, v in scale.items()))
    if pl["refusals"]:
        out.append("\n## REFUSED: a row would be measured twice")
        out += [f"- {x}" for x in pl["refusals"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.measure plan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the record store")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--view", choices=("report", "card", "gate"))
    sel.add_argument("--rows", help="row ids and globs (asking for a row with a record is refused)")
    ap.add_argument("--pack", action="append", required=True, help="a pack (repeat for several on one page)")
    ap.add_argument("--build", required=True)
    ap.add_argument("--box", default=None, help="the box profile of the key (gb10); default any")
    ap.add_argument("--scale", action="append", default=[], metavar="PACK=X",
                    help="row minutes x X for a pack (the estimates are the reference box's Flash-Next)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--write", default=None, help="also write the plan to this file")
    a = ap.parse_args(argv)
    st = storemod.Store(os.path.expanduser(a.out))
    rows, _ = rowsmod.load()
    want = rowsmod.select(rows, a.rows) if a.rows else rowsmod.for_view(rows, a.view)
    scale = {k: float(v) for k, v in (x.split("=", 1) for x in a.scale)}
    pl = make(st, a.pack, a.build, want, a.box, scale, explicit=bool(a.rows), force=a.force)
    text = render(pl, a.view or "rows", a.build, a.pack, a.box, scale)
    sys.stdout.write(text)
    if a.write:
        with open(os.path.expanduser(a.write), "w") as fh:
            fh.write(text)
    return 2 if pl["refusals"] else 0
