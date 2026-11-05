"""`python -m tools.measure compare --out DIR --pack P --old 0.15.1 --new 0.15.2`: every report row both builds
have a record of, old / new / change, and better / worse / same against the noise band: the larger of 3 %
(what a row can settle, docs/measurement.md) and either side's spread between its own repeats. A row with
one run a side has only the 3 % floor; the table says so. Then the 0.15.1 page's surprises, each fixed,
still there or not measured, from the new records. Measures nothing."""

from __future__ import annotations

import statistics

from tools.measure import rows as rowsmod, store as storemod

FLOOR = 3.0

#: row prefix -> (metric, label, higher is better)
METRICS = (
    ("length.", [("decode_tok_s", "decode tok/s", True), ("ttft_s", "first token s", False)]),
    ("warm.", [("decode_tok_s", "decode tok/s", True), ("ttft_s", "first token s", False)]),
    ("conc.2k.", [("decode_tok_s", "decode tok/s", True), ("agg_tok_s", "with prefill tok/s", True)]),
    ("conc.32k.", [("ttft_max_s", "slowest first token s", False)]),
    ("conc.card.default.", [("wall_tok_s", "wall tok/s", True), ("ttft_s", "first token s", False)]),
    ("conc.card.", [("wall_tok_s", "wall tok/s", True), ("ttft_s", "first token s", False)]),
    ("decode.", [("decode_tok_s", "decode tok/s", True)]),
    ("prefill.", [("tok_s", "prefill tok/s", True)]),
    ("shared.", [("ttft_last_s", "last chat's first token s", False)]),
    ("load.", [("ttft_s", "first token s", False), ("decoders_during_tok_s", "decoders during tok/s", True)]),
    ("return.", [("turn3_ttft_s", "turn 3 first token s", False)]),
    ("longdecode.", [("decode_tok_s", "decode tok/s", True)]),
)


def metrics_of(rid: str):
    for pre, ms in METRICS:
        if rid.startswith(pre):
            return ms
    return []


def runs(rec: dict | None, key: str) -> list[float]:
    """A metric's values in a record: one a run/round (its section rows, valid only)."""
    if not rec:
        return []
    out = [r["v"].get(key) for r in rec.get("rows") or [] if r.get("valid")]
    out = [float(x) for x in out if isinstance(x, (int, float))]
    if not out and isinstance((rec.get("values") or {}).get(key), (int, float)):
        out = [float(rec["values"][key])]
    return out


def spread(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return 100.0 * (max(xs) - min(xs)) / m if m else 0.0


def compare(st: storemod.Store, pack: str, old: str, new: str, box: str | None = None,
            bands: dict | None = None) -> list[dict]:
    """`bands`: {row glob: %} a known run-to-run band for rows with one value a side (two runs of the
    same rows on one server elsewhere), used when it is above the floor."""
    rows, _ = rowsmod.load()
    out = []
    for r in rowsmod.for_view(rows, "report"):
        a, b = st.primary(r.id, old, pack, box), st.primary(r.id, new, pack, box)
        for key, label, up in metrics_of(r.id):
            xa, xb = runs(a, key), runs(b, key)
            ent = {"row": r.id, "metric": label, "higher_better": up, "old_n": len(xa), "new_n": len(xb)}
            if not xa or not xb:
                ent["verdict"] = "not comparable: " + ("no old record" if not xa and not a else "no new record"
                                                      if not xb and not b else "no valid value on one side")
                out.append(ent)
                continue
            ma, mb = statistics.median(xa), statistics.median(xb)
            ch = 100.0 * (mb - ma) / ma if ma else 0.0
            known = max([v for k, v in (bands or {}).items() if rowsmod.matches(r.id, [k])], default=0.0)
            noise = max(FLOOR, spread(xa), spread(xb), known)
            if abs(ch) <= noise:
                v = "same"
            else:
                v = "better" if (ch > 0) == up else "worse"
            ent.update(old=round(ma, 3), new=round(mb, 3), change_pct=round(ch, 1), noise_pct=round(noise, 1), verdict=v,
                       old_src=(a.get("source") or {}).get("path"), new_src=(b.get("source") or {}).get("path"))
            out.append(ent)
    return out


def drafting(rec: dict | None) -> list:
    if not rec:
        return []
    return [((x.get("engine") or [{}])[0] or {}).get("drafting") for x in rec.get("rows") or [] if x.get("valid")]


def surprises(st: storemod.Store, pack: str, new: str, box: str | None = None) -> list[tuple[str, str, str]]:
    """The 0.15.1 page's six surprises against the new records: (surprise, state, evidence)."""
    g = lambda rid: st.primary(rid, new, pack, box)            # noqa: E731
    med = lambda xs: statistics.median(xs) if xs else None      # noqa: E731
    out = []
    # 1 drafting stops at 32k
    lens = [k for k in ("1k", "4k", "16k", "32k", "55k", "64k", "96k", "128k", "192k", "240k") if g(f"length.{k}")]
    if lens:
        d = {k: drafting(g(f"length.{k}")) for k in lens}
        on = [k for k in lens if any(d[k])]
        off = [k for k in lens if d[k] and not any(d[k])]
        state = "still there" if off and on and rowsmod.TOKENS[max(on, key=rowsmod.TOKENS.get)] <= 32768 else \
            ("fixed" if not off else "changed")
        out.append(("Drafting stops at 32k", state, f"drafting at {', '.join(on) or 'none'}; not at {', '.join(off) or 'none'}"))
    else:
        out.append(("Drafting stops at 32k", "not measured", "no length rows"))
    # 2 warm 128k 0.60x
    w, f = g("warm.128k"), g("length.128k")
    if w and f and w.get("valid"):
        wv, fv = med(runs(w, "decode_tok_s")), med(runs(f, "decode_tok_s"))
        r = wv / fv if wv and fv else None
        out.append(("Warm 128k at 0.60x of fresh", "fixed" if r and r >= 0.9 else "still there",
                    f"warm {wv} vs fresh {fv} tok/s = {r:.2f}x" if r else "no value"))
    else:
        out.append(("Warm 128k at 0.60x of fresh", "not measured" if not w else "invalid row",
                    (w or {}).get("why") or "no warm.128k record"))
    # 3 warm 192k memory kill
    w = g("warm.192k")
    out.append(("Warm 192k killed at the memory floor", "not measured" if not w else
                ("fixed" if w.get("valid") else "still there"),
                (f"decode {med(runs(w, 'decode_tok_s'))} tok/s, first token {med(runs(w, 'ttft_s'))} s" if w and w.get("valid")
                 else (w or {}).get("why") or "no warm.192k record")))
    # 4 shared prefix reuse
    cached = []
    for n in (5, 8):
        s = g(f"shared.{n}")
        for x in (s or {}).get("rows") or []:
            cached += list(x["v"].get("cached_tokens") or [])
    out.append(("Chats on one new system prompt reuse nothing", "not measured" if not cached else
                ("still there" if not any(cached) else "fixed"),
                f"cached tokens per chat: {cached}" if cached else "no shared rows"))
    # 5 decoders during a long prefill
    ld = g("load.64k")
    v = ((ld or {}).get("rows") or [{}])[0].get("v") if ld else None
    out.append(("Decoders crawl (~1.4 tok/s) during a 64k prefill", "not measured" if not v else
                ("still there" if (v.get("decoders_during_tok_s") or 0) < 10 else "fixed"),
                f"before {v.get('decoders_before_tok_s')}, during {v.get('decoders_during_tok_s')} tok/s; "
                f"its first token {v.get('ttft_s')} s" if v else "no load row"))
    # 6 prefill-only vs TTFT gap at 32k
    p, ln = g("prefill.32k"), g("length.32k")
    if p and ln:
        po = med(runs(p, "tok_s"))
        lt = med([x["v"].get("prefill_tok_s") for x in ln.get("rows") or [] if x.get("valid") and x["v"].get("prefill_tok_s")])
        gap = 100 * (po - lt) / po if po and lt else None
        out.append(("A 512-token request's first token ~13 % below prefill alone at 32k",
                    "still there" if gap and gap > 5 else "fixed", f"prefill alone {po} tok/s, a 512-token request {lt} "
                    f"({gap:.1f} % below)" if gap is not None else "no value"))
    else:
        out.append(("A 512-token request's first token ~13 % below prefill alone at 32k", "not measured",
                    "needs prefill.32k and length.32k"))
    return out


def markdown(cmp: list[dict], sur: list, pack: str, old: str, new: str) -> str:
    out = [f"# {pack}: {new} against {old}", "",
           f"Noise band of a row: the larger of {FLOOR:g} % and either side's spread between its own repeats "
           "(max - min over the median). same = the change is inside the band. n = runs/rounds a side.", ""]
    cnt = {k: sum(1 for c in cmp if c["verdict"] == k) for k in ("better", "worse", "same")}
    out.append(f"**{cnt['better']} better, {cnt['worse']} worse, {cnt['same']} same**, "
               f"{sum(1 for c in cmp if c['verdict'].startswith('not'))} not comparable.\n")
    out.append("| row | metric | old | new | change | band | n old/new | verdict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for c in cmp:
        if c["verdict"].startswith("not"):
            out.append(f"| {c['row']} | {c['metric']} | | | | | {c['old_n']}/{c['new_n']} | {c['verdict']} |")
        else:
            v = c["verdict"] if c["verdict"] == "same" else f"**{c['verdict']}**"
            out.append(f"| {c['row']} | {c['metric']} | {c['old']:g} | {c['new']:g} | {c['change_pct']:+.1f} % | "
                       f"{c['noise_pct']:.1f} % | {c['old_n']}/{c['new_n']} | {v} |")
    out += ["", f"## The {old} surprises", "", "| surprise | state | evidence |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in sur]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    import argparse
    import os
    ap = argparse.ArgumentParser(prog="python -m tools.measure compare", description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--box", default=None)
    ap.add_argument("--write", default=None)
    ap.add_argument("--band", action="append", default=[], metavar="GLOB=PCT",
                    help="a known run-to-run band for rows of one value a side")
    a = ap.parse_args(argv)
    st = storemod.Store(os.path.expanduser(a.out))
    bands = {k: float(v) for k, v in (x.split("=", 1) for x in a.band)}
    md = markdown(compare(st, a.pack, a.old, a.new, a.box, bands), surprises(st, a.pack, a.new, a.box), a.pack,
                  a.old, a.new)
    if bands:
        md += "\nKnown run-to-run bands used: " + ", ".join(f"{k} {v:g} %" for k, v in bands.items()) + "\n"
    if a.write:
        with open(os.path.expanduser(a.write), "w") as fh:
            fh.write(md)
    print(md)
    return 0
