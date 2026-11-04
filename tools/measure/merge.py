"""`python -m tools.measure merge`: records that exist already, imported into the store, never measured again.

    merge --out DIR --report data.json [--rawdir DIR] [--label fn1] [--build B]   a report's rows (and the
                                                                                  card tool's raw files)
    merge --out DIR --gate gate-report.json [--gate-raw raw.json] [--side head]   a gate arm's rows

Each imported record keeps its source (the tool, the date, the path given here, the commit) and the run
it came from (its knobs, box and times), so a view shows where every number comes from. A row another
source holds already for the same key stays that source's; the import is listed beside it. A row of the
source that maps to no row id is named and left out."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from tools.measure import store as storemod

KSIZE = {1024: "1k", 2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k", 56320: "55k",
         57344: "57k", 65536: "64k", 98304: "96k", 131072: "128k", 196608: "192k", 245760: "240k"}


def report_row_id(r: dict) -> str | None:
    """A report section row (tools/release_report/sections) -> its row id, or None."""
    s, k = r["section"], r["key"]
    parts = k.split("/")
    if r.get("skipped"):
        return None
    if s == "a":
        n = int(parts[0])
        return f"length.{KSIZE.get(n, n)}"
    if s == "i":
        n = int(parts[1])
        return f"length.{KSIZE.get(n, n)}" if parts[0] == "fresh" else f"warm.{KSIZE.get(n, n)}"
    if s == "c" and parts[0] == "2k":
        return f"conc.2k.c{parts[1]}"
    if s == "c" and parts[0] == "32k":
        return f"conc.32k.c{parts[1]}"
    if s == "d":
        return f"shared.{parts[0]}"
    if s == "e":
        return {"load": "load.64k", "return": "return.96k"}.get(parts[0])
    if s == "f":
        return f"longdecode.57k.{parts[0]}"
    if s == "g" and len(parts) == 3 and parts[1] == "levels":
        return f"conc.card.c{parts[2]}" if parts[0] == "fixed" else f"conc.card.default.c{parts[2]}"
    if s == "g" and len(parts) == 3 and parts[1] == "decode":
        return f"decode.{parts[2]}.c1"
    if s == "g" and len(parts) == 3 and parts[1] == "prefill":
        return f"prefill.{KSIZE.get(int(parts[2]), parts[2])}"
    if s == "h" and parts[0] == "needle" and int(parts[1]) == 32768:
        return "needle.32k"
    return None


def run_of_row(r: dict, runs: list[dict]) -> dict:
    """The run a data.json row came from: the last run that measured its section (data.json keeps a
    section's last run); for section i the fresh rows' run (i first on its server) or the warm rows'."""
    s = r["section"]
    cand = [x for x in runs if s in (x.get("sections") or []) and (x.get("variant") or "spec") == "spec"] or \
        [x for x in runs if s in (x.get("sections") or [])]
    if s == "i":
        fresh = r["key"].startswith("fresh/")
        cand = [x for x in cand if ((x.get("sections") or [None])[0] == "i") == fresh] or cand
    return cand[-1] if cand else (runs[-1] if runs else {})


def card_raw(rawdir: Path | None, rid: str) -> dict | None:
    """The card tool's own raw file's part for a card row (raw/card-default.json, raw/card-fixed.json)."""
    if not rawdir:
        return None
    m = re.match(r"conc\.card\.(default\.)?c(\d+)$", rid)
    if m:
        f = rawdir / f"card-{'default' if m.group(1) else 'fixed'}.json"
        return (json.loads(f.read_text()).get("levels") or {}).get(m.group(2)) if f.exists() else None
    f = rawdir / "card-default.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text())
    m = re.match(r"decode\.(code|prose)\.c1$", rid)
    if m and raw.get("decode"):
        d = raw["decode"]
        return {"runs": d["runs"][m.group(1)], "median": d["median"][m.group(1)],
                "prompt_tokens": d["prompt_tokens"][m.group(1)]}
    m = re.match(r"prefill\.(\w+)$", rid)
    if m and raw.get("prefill"):
        tokens = {v: k for k, v in KSIZE.items()}.get(m.group(1))
        dd = raw["prefill"]["depths"].get(str(tokens))
        return {"runs": dd["runs"], "tokenize_offset": raw["prefill"].get("tokenize_offset")} if dd else None
    return None


def card_setup(rawdir: Path | None) -> dict | None:
    """The card raw file's setup and health record (the card view's row needs them)."""
    if not rawdir or not (rawdir / "card-default.json").exists():
        return None
    raw = json.loads((rawdir / "card-default.json").read_text())
    return {k: raw.get(k) for k in ("setup", "health_start", "health_end", "args", "model", "profile", "build",
                                    "commit", "date", "levels_method")}


def import_report(st: storemod.Store, path: str, rawdir: str | None = None, label: str | None = None,
                  build: str | None = None, commit: str | None = None, log=print) -> dict:
    data = json.loads(Path(path).read_text())
    runs = data.get("runs") or []
    b = build or storemod.build_of(data.get("version"))
    pack = data.get("weights") or "unknown"
    rd = Path(rawdir) if rawdir else None
    recs: dict = {}
    left = []
    for r in data.get("rows") or []:
        rid = report_row_id(r)
        if rid is None:
            if not r.get("skipped"):
                left.append(f"{r['section']}/{r['key']}")
            continue
        run = run_of_row(r, runs)
        knobs = run.get("knobs") or {}
        rec = recs.get(rid)
        if rec is None:
            rec = recs[rid] = {
                "row": rid, "build": b, "pack": pack, "box": run.get("hw_profile") or "unknown",
                "knobs_hash": storemod.knobs_hash(knobs), "knobs": storemod.limit_knobs(knobs),
                "version": data.get("version"), "valid": True, "why": None, "skipped": None, "values": {},
                "rows": [], "raw": card_raw(rd, rid),
                "source": {"tool": f"tools/release_report {data.get('tool')}", "how": "imported",
                           "date": str(run.get("started") or "")[:10], "path": path,
                           "run_started": run.get("started"), "served_at_start": run.get("served_at_start")}}
        rec["rows"].append(r)
        if not r.get("valid"):
            rec["valid"], rec["why"] = False, r.get("why")
    for rec in recs.values():
        rec["values"] = values_of(rec["row"], rec["rows"])
        raw = rec.get("raw") or {}
        if rec["row"].startswith("decode.") and raw.get("prompt_tokens"):
            rec["values"]["prompt_tokens"] = raw["prompt_tokens"]
        if commit:
            rec["source"]["commit"] = commit
    if not recs:
        raise SystemExit(f"[measure] {path}: no row maps to a row id")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = label or Path(path).parent.name
    run = {"id": f"import-{b}-{pack}-{name}-{stamp}", "how": "imported", "tool": "tools/measure merge",
           "from": path, "from_kind": "tools/release_report data.json", "origin_runs": runs,
           "card": card_setup(rd), "version": data.get("version"), "model": data.get("model"),
           "weights": data.get("weights"), "hardware": data.get("hardware"), "private": data.get("private"),
           "started": min((str(x.get("started")) for x in runs if x.get("started")), default=None),
           "commit": commit, "unmapped": left}
    p = st.add_run(run, recs, imported=True)
    log(f"[measure] {path}: {len(recs)} rows into {p}" + (f"; not row ids, left out: {', '.join(left)}" if left else ""))
    return {"path": str(p), "rows": sorted(recs), "unmapped": left}


def values_of(rid: str, rows: list[dict]) -> dict:
    from statistics import median
    ok = [r for r in rows if r.get("valid")]

    def med(k):
        xs = [r["v"].get(k) for r in ok if r["v"].get(k) is not None]
        return median(xs) if xs else None
    if rid.startswith(("length.", "warm.", "longdecode.")):
        return {"decode_tok_s": med("decode_tok_s"), "ttft_s": med("ttft_s")}
    if rid.startswith("conc.2k."):
        return {"agg_tok_s": med("agg_tok_s"), "decode_tok_s": med("decode_tok_s")}
    if rid.startswith("conc.32k."):
        return {"ttft_max_s": med("ttft_max_s")}
    if rid.startswith("conc.card."):
        return {k: med(k) for k in ("wall_tok_s", "decode_tok_s", "ttft_s")}
    if rid.startswith("decode."):
        return {"decode_tok_s": med("decode_tok_s")}
    if rid.startswith("prefill."):
        return {"tok_s": med("tok_s"), "ttft_s": med("ttft_s"), "prompt_tokens": med("prompt_tokens")}
    return dict(ok[0]["v"]) if ok else {}


# --- the gate -----------------------------------------------------------------------------------------

GATE_ROWS = (
    (r"^decode\.c1$", "decode.c1.short"), (r"^decode\.c1-1k$", "decode.c1.1k"), (r"^decode\.c8(\.wall)?$", "conc.card.c8"),
    (r"^prefill\.(\d+k)$", "prefill.{0}"), (r"^prefill\.burst8x2k$", "burst.8x2k"),
    (r"^long\.max\.", "long.131k.needle"), (r"^long\.wide\.", "long.wide"),
    (r"^lossless\.plain\.", "lossless.plain"), (r"^lossless\.(?!plain)", "lossless.set"),
    (r"^stability\.", "stability.30m"),
)


def gate_row_id(gid: str) -> str | None:
    for pat, rid in GATE_ROWS:
        m = re.match(pat, gid)
        if m:
            return rid.format(*m.groups())
    return None


def gate_sub(rid: str, suite_raw: dict) -> dict | None:
    """One server's suite raw cut to a row (what the gate's verdict reads for that row)."""
    if rid in ("decode.c1.short", "decode.c1.1k", "conc.card.c8"):
        sid = {"decode.c1.short": "c1", "decode.c1.1k": "c1-1k", "conc.card.c8": "c8"}[rid]
        return {sid: suite_raw[sid]} if sid in suite_raw else None
    if rid.startswith("prefill."):
        tokens = {v: k for k, v in KSIZE.items()}[rid.split(".")[1]]
        d = (suite_raw.get("depths") or {}).get(str(tokens))
        return {"depths": {str(tokens): d}, "tokenize_offset": suite_raw.get("tokenize_offset")} if d else None
    if rid == "burst.8x2k":
        return {"burst": suite_raw.get("burst")}
    return suite_raw


SUITE_OF = {"decode.c1.short": "decode", "decode.c1.1k": "decode", "conc.card.c8": "decode", "burst.8x2k": "prefill",
            "long.131k.needle": "long", "long.wide": "long", "lossless.set": "lossless", "lossless.plain": "lossless",
            "stability.30m": "stability"}


def arm_of(rid: str, side: str) -> str:
    return {"long.wide": f"{side}:wide", "lossless.plain": f"{side}:plain"}.get(rid, side)


def import_gate(st: storemod.Store, path: str, raw_path: str | None = None, side: str = "head",
                build: str | None = None, label: str | None = None, log=print) -> dict:
    rep = json.loads(Path(path).read_text())
    raw = (json.loads(Path(raw_path).read_text()).get("raw") or {}) if raw_path else {}
    knobs = (rep.get("knobs") or {}).get(side) or {}
    b = build or storemod.build_of(knobs.get("version"))
    pack = (rep.get("pack") or {}).get("id") or knobs.get("weights") or "unknown"
    box = (rep.get("hardware") or {}).get("profile") or "unknown"
    sha = (rep.get(side) or {}).get("sha")
    recs: dict = {}
    for gr in rep.get("rows") or []:
        rid = gate_row_id(gr["id"])
        if rid is None:
            continue
        rec = recs.get(rid)
        if rec is None:
            suite = SUITE_OF.get(rid) or rid.split(".")[0]
            arm = arm_of(rid, side)
            servers = [gate_sub(rid, x) for x in (raw.get(suite) or {}).get(arm) or []]
            rec = recs[rid] = {
                "row": rid, "build": b, "pack": pack, "box": box, "knobs_hash": storemod.knobs_hash(knobs),
                "knobs": storemod.limit_knobs(knobs), "version": knobs.get("version"), "valid": True, "why": None,
                "skipped": None, "values": {}, "rows": [], "gate_rows": [],
                "raw": {"servers": [x for x in servers if x is not None]} if servers else None,
                "source": {"tool": f"tools/gate {(rep.get('tool') or {}).get('version')}", "how": "imported",
                           "date": str(rep.get("created_at") or "")[:10], "path": path, "commit": sha,
                           "side": side, "level": rep.get("level"), "pairs": rep.get("pairs")}}
        rec["gate_rows"].append(gr)
        if gr["verdict"] == "INVALID":
            rec["valid"], rec["why"] = False, gr["why"]
        if gr.get(side) and isinstance(gr[side], dict) and gr[side].get("n"):
            rec["values"][gr["id"]] = gr[side]
    for rid, rec in recs.items():
        rec["rows"] = gate_report_rows(rid, rec)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = {"id": f"import-{b}-{pack}-{label or 'gate-' + (sha or 'x')[:8]}-{stamp}", "how": "imported",
           "tool": "tools/measure merge", "from": path, "from_raw": raw_path, "from_kind": "tools/gate gate-report.json",
           "side": side, "commit": sha, "started": rep.get("created_at"), "knobs": knobs, "level": rep.get("level"),
           "hardware": rep.get("hardware"), "profile": (rep.get("profile") or {}).get("id"),
           "version": knobs.get("version"), "weights": knobs.get("weights")}
    p = st.add_run(run, recs, imported=True)
    log(f"[measure] {path} ({side} {(sha or '')[:8]}): {len(recs)} rows into {p}")
    return {"path": str(p), "rows": sorted(recs)}


def gate_report_rows(rid: str, rec: dict) -> list[dict]:
    """The report's section rows a gate record gives: the needle of the longest prompt (h), the
    spec-vs-plain texts (h), the burst (c), the card's c8 level (g, fixed prompts)."""
    gr = {x["id"]: x for x in rec["gate_rows"]}
    servers = (rec.get("raw") or {}).get("servers") or []
    out = []
    if rid == "long.131k.needle":
        m = (servers[0] if servers else {}).get("max") or {}
        nd = gr.get("long.max.needle") or {}
        out.append({"section": "h", "key": f"needle/{m.get('tokens_asked', 131072)}",
                    "v": {"found": nd.get("verdict") == "PASS" if nd else m.get("needle"),
                          "answer": m.get("answer"), "prompt_tokens": m.get("prompt_tokens")},
                    "cond": {"target": m.get("tokens_asked", 131072), "shape": "the gate's long case (card filler)"},
                    "valid": rec["valid"], "why": rec["why"]})
    if rid == "lossless.plain":
        for k, x in gr.items():
            pid = k.split(".")[2]
            out.append({"section": "h", "key": f"specplain/{pid}", "v": {"same": x["verdict"] == "PASS",
                                                                       "why": x["why"]},
                        "cond": {"set": "the gate's lossless set (greedy)"}, "valid": x["verdict"] != "INVALID",
                        "why": None if x["verdict"] != "INVALID" else x["why"]})
    if rid == "burst.8x2k":
        x = gr.get("prefill.burst8x2k") or {}
        side = (rec["source"].get("side") or "head")
        out.append({"section": "c", "key": "burst/8x2k", "v": {"ttft_median_s": (x.get(side) or {}).get("median"),
                                                              "n": 8, "depth": 2048, "runs": (x.get(side) or {}).get("runs")},
                    "cond": {"rounds": (x.get(side) or {}).get("n")}, "valid": rec["valid"], "why": rec["why"]})
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m tools.measure merge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the record store")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--report", help="a tools/release_report data.json")
    src.add_argument("--gate", help="a tools/gate gate-report.json")
    ap.add_argument("--rawdir", default=None, help="--report: its raw/ folder (the card tool's raw files)")
    ap.add_argument("--gate-raw", default=None, help="--gate: its raw.json (each arm's suite results)")
    ap.add_argument("--side", default="head", choices=("head", "base"))
    ap.add_argument("--build", default=None, help="the build of the key (default: the source's version)")
    ap.add_argument("--label", default=None, help="a name for the import's run file")
    ap.add_argument("--commit", default=None, help="--report: the commit the server ran (data.json names the version)")
    a = ap.parse_args(argv)
    st = storemod.Store(os.path.expanduser(a.out))
    if a.report:
        import_report(st, a.report, a.rawdir, a.label, a.build, a.commit)
    else:
        import_gate(st, a.gate, a.gate_raw, a.side, a.build, a.label)
    return 0
