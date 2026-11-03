"""data.json -> one self-contained HTML page (no external asset, inline SVG charts, light and dark).

The page leads with the engine: tiles of the headline numbers, a verdict list stated plainly (good and
bad, regressions included, causes left unknown unless the data holds an A/B), then one section per
measurement: a chart drawn from the rows, the rows as a table, and the method in one line. Every string
that comes from the data passes tools/release_report/scrub.py, and the page is refused if it still names
the box (a home path, a host or user name, an address).

The same page serves any model and any GPU: sections without rows say why."""

from __future__ import annotations

import html as H
import json
import re
import statistics

from tools.release_report import BARE, SECTIONS, scrub
from tools.release_report.sections.fidelity import PROMPTS, compare


class Leak(RuntimeError):
    pass


def esc(x) -> str:
    return H.escape(str(x), quote=True)


def num(x, nd=1, unit="") -> str:
    if x is None:
        return "–"
    if isinstance(x, (int, float)):
        s = f"{x:,.{nd}f}" if nd else f"{round(x):,}"
        return s + unit
    return esc(x)


def kfmt(n) -> str:
    try:
        return f"{round(int(n) / 1024)}k"
    except (TypeError, ValueError):
        return str(n)


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def rows_of(data: dict, section: str) -> list[dict]:
    return [r for r in data.get("rows") or [] if r.get("section") == section]


def valid(rs):
    return [r for r in rs if r.get("valid") and not r.get("skipped")]


# ---- charts -------------------------------------------------------------------------------------

W, PAD_L, PAD_R, PAD_T, PAD_B = 720, 56, 20, 16, 44


def nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    import math
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= v:
            return m * e
    return 10 * e


def ticks(top: float, n: int = 4) -> list[float]:
    return [top * i / n for i in range(n + 1)]


def xy_chart(cats: list[str], series: list[dict], ylabel: str, height: int = 260, ymax=None,
             aria: str = "") -> str:
    """Categories along x (equal steps), one or more series of points: each series {name, cls, pts:
    [(cat index, y, filled bool, tip)], line: [(cat index, y)]}."""
    h = height
    iw, ih = W - PAD_L - PAD_R, h - PAD_T - PAD_B
    ys = [p[1] for s in series for p in s["pts"] if p[1] is not None]
    top = ymax or nice_max(max(ys) * 1.08 if ys else 1)
    n = max(len(cats), 1)

    def X(i):
        return PAD_L + (iw * (i + 0.5) / n)

    def Y(v):
        return PAD_T + ih - ih * (v / top)
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="{esc(aria or ylabel)}">']
    for t in ticks(top):
        y = Y(t)
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">{t:,.0f}</text>')
    for i, c in enumerate(cats):
        out.append(f'<text class="tick" x="{X(i):.1f}" y="{h - PAD_B + 18}" text-anchor="middle">{esc(c)}</text>')
    out.append(f'<text class="axis" x="{PAD_L}" y="{h - 6}">{esc(ylabel)}</text>')
    for s in series:
        line = [(X(i), Y(v)) for i, v in s.get("line") or [] if v is not None]
        if len(line) >= 2:
            d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(line))
            out.append(f'<path class="ln {s["cls"]}" d="{d}"/>')
        off = s.get("dx", 0)
        for i, v, filled, tip in s["pts"]:
            if v is None:
                continue
            cls = f'pt {s["cls"]}' + ("" if filled else " hollow")
            out.append(f'<circle class="{cls}" cx="{X(i) + off:.1f}" cy="{Y(v):.1f}" r="4.5">'
                       f'<title>{esc(tip)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def bar_chart(groups: list[tuple[str, list[tuple[str, float | None, str, str]]]], unit: str,
              aria: str = "") -> str:
    """Horizontal bars: groups of (label, [(series name, value, cls, tip)])."""
    rows = sum(len(g[1]) for g in groups)
    bh, gap, ggap, left = 14, 2, 14, 168
    h = PAD_T + rows * (bh + gap) + len(groups) * ggap + 30
    vals = [v for _, bs in groups for _, v, _, _ in bs if v is not None]
    top = nice_max(max(vals) * 1.05) if vals else 1
    iw = W - left - PAD_R - 64
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="{esc(aria or unit)}">']
    for t in ticks(top):
        x = left + iw * t / top
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T - 4}" y2="{h - 26}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - 10}" text-anchor="middle">{t:,.0f}</text>')
    y = PAD_T
    for label, bs in groups:
        out.append(f'<text class="lab" x="{left - 10}" y="{y + bh - 2}" text-anchor="end">{esc(label)}</text>')
        for name, v, cls, tip in bs:
            if v is None:
                out.append(f'<text class="tick" x="{left + 4}" y="{y + bh - 3}">{esc(tip)}</text>')
            else:
                w = max(2.0, iw * v / top)
                out.append(f'<path class="bar {cls}" d="M{left},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bh - 8} '
                           f'a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z"><title>{esc(tip)}</title></path>')
                out.append(f'<text class="val" x="{left + w + 6:.1f}" y="{y + bh - 3}">{num(v)}</text>')
            y += bh + gap
        y += ggap
    out.append(f'<text class="axis" x="{left}" y="{h - 10}" text-anchor="end" dx="-10">{esc(unit)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items: list[tuple[str, str, bool]]) -> str:
    out = ['<div class="legend">']
    for name, cls, filled in items:
        out.append(f'<span><svg width="14" height="14" viewBox="0 0 14 14"><circle class="pt {cls}'
                   f'{"" if filled else " hollow"}" cx="7" cy="7" r="4.5"/></svg>{esc(name)}</span>')
    out.append("</div>")
    return "".join(out)


def table(head: list[str], body: list[list], cls: str = "") -> str:
    out = [f'<div class="tw"><table class="{cls}"><thead><tr>']
    out += [f"<th>{esc(h)}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def bad_cell(r: dict) -> str:
    return f'<span class="inv" title="{esc(r.get("why"))}">invalid</span>'


def skips(rs: list[dict]) -> str:
    sk = [r for r in rs if r.get("skipped")]
    if not sk:
        return ""
    return '<ul class="skips">' + "".join(f"<li><span>{esc(r['key'])}</span> {esc(r['skipped'])}</li>"
                                           for r in sk) + "</ul>"


def invalids(rs: list[dict]) -> str:
    bad = [r for r in rs if not r.get("valid") and not r.get("skipped")]
    if not bad:
        return ""
    return '<ul class="skips inv-list">' + "".join(
        f"<li><span>{esc(r['key'])}</span> invalid: {esc(r.get('why'))}</li>" for r in bad) + "</ul>"


# ---- sections -----------------------------------------------------------------------------------

def by_length(data):
    rs = valid(rows_of(data, "a"))
    out: dict = {}
    for r in rs:
        out.setdefault(int(r["cond"]["target"]), []).append(r)
    return dict(sorted(out.items()))


def drafting_of(r):
    e = (r.get("engine") or [{}])[0] or {}
    lg = (r.get("log") or [{}])[0] or {}
    d = e.get("drafting")
    return lg.get("drafting", d)


def sec_a(data) -> tuple[str, str]:
    L = by_length(data)
    if not L:
        return "", "no rows"
    cats = [kfmt(n) for n in L]
    pts, line = [], []
    for i, (n, rs) in enumerate(L.items()):
        for r in rs:
            v = r["v"]
            pts.append((i, v.get("decode_tok_s"), bool(drafting_of(r)),
                        f"{kfmt(n)} prompt ({num(v.get('prompt_tokens'), 0)} tokens): {num(v.get('decode_tok_s'))} tok/s"
                        f", drafting {'yes' if drafting_of(r) else 'no'}"))
        line.append((i, med([r["v"].get("decode_tok_s") for r in rs])))
    chart = xy_chart(cats, [{"name": "decode", "cls": "s1", "pts": pts, "line": line}], "decode tok/s, one request",
                     aria="Decode tokens per second by prompt length")
    body = []
    for n, rs in L.items():
        e = [((r.get("engine") or [{}])[0] or {}) for r in rs]
        lg = [((r.get("log") or [{}])[0] or {}) for r in rs]
        slots = sorted({x.get("slot") for x in lg if x.get("slot") is not None})
        body.append([kfmt(n), num(med([r["v"].get("prompt_tokens") for r in rs]), 0),
                     num(med([r["v"].get("decode_tok_s") for r in rs])),
                     " / ".join(num(r["v"].get("decode_tok_s")) for r in rs),
                     "yes" if all(drafting_of(r) for r in rs) else ("no" if not any(drafting_of(r) for r in rs) else "mixed"),
                     num(med([x.get("tokens_per_block") for x in e]), 2),
                     num(med([x.get("decode_tok_s") for x in e])),
                     ", ".join(str(s) for s in slots) or "–", str(len(rs))])
    tbl = table(["prompt", "prompt tokens", "decode tok/s (median)", "runs", "drafting", "tokens / pass",
                 "engine decode tok/s", "slot", "n"], body, "num")
    leg = legend([("drafting (engine count)", "s1", True), ("no drafting", "s1", False)])
    return leg + chart + tbl + invalids(rows_of(data, "a")) + skips(rows_of(data, "a")), ""


def sec_b(data):
    L = by_length(data)
    if not L:
        return "", "no rows"
    cats = [kfmt(n) for n in L]
    pts, line = [], []
    for i, (n, rs) in enumerate(L.items()):
        for r in rs:
            pts.append((i, r["v"].get("prefill_tok_s"), True,
                        f"{kfmt(n)}: {num(r['v'].get('prefill_tok_s'), 0)} tok/s, first token {num(r['v'].get('ttft_s'))} s"))
        line.append((i, med([r["v"].get("prefill_tok_s") for r in rs])))
    chart = xy_chart(cats, [{"name": "prefill", "cls": "s1", "pts": pts, "line": line}], "prefill tok/s",
                     aria="Prefill tokens per second by prompt length")
    body = [[kfmt(n), num(med([r["v"].get("prompt_tokens") for r in rs]), 0),
             num(med([r["v"].get("prefill_tok_s") for r in rs]), 0), num(med([r["v"].get("ttft_s") for r in rs]), 2) + " s"]
            for n, rs in L.items()]
    return chart + table(["prompt", "prompt tokens", "prefill tok/s", "first token"], body, "num"), ""


def sec_c(data):
    rs = rows_of(data, "c")
    two = [r for r in rs if r["key"].startswith("2k/")]
    t32 = [r for r in rs if r["key"].startswith("32k/")]
    if not two and not t32:
        return skips(rs), "no rows"
    ns = sorted({r["v"]["n"] for r in two if r.get("v")})
    pts, line, body = [], [], []
    for i, n in enumerate(ns):
        lv = [r for r in two if r["v"]["n"] == n]
        ok = valid(lv)
        for r in lv:
            pts.append((i, r["v"].get("agg_tok_s") if r.get("valid") else None, True,
                        f"{n} at once: {num(r['v'].get('agg_tok_s'))} tok/s aggregate, wall {num(r['v'].get('wall_s'))} s"))
        line.append((i, med([r["v"].get("agg_tok_s") for r in ok])))
        p = (lv[-1].get("cond") or {}).get("proof") or {}
        body.append([str(n), num(med([r["v"].get("agg_tok_s") for r in ok])) if ok else bad_cell(lv[-1]),
                     num(med([r["v"].get("decode_tok_s") for r in ok])) if ok else "–",
                     num(med([r["v"].get("wall_s") for r in ok])) + " s" if ok else "–",
                     num(med([r["v"].get("ttft_max_s") for r in ok]), 2) + " s" if ok else "–",
                     f'{p.get("max_decoding_at_once", "–")} decoding / {p.get("max_running", "–")} running',
                     f"{len(ok)} of {len(lv)}"])
    out = ""
    if ns:
        out += xy_chart([str(n) for n in ns], [{"name": "aggregate", "cls": "s1", "pts": pts, "line": line}],
                        "aggregate tok/s, ~2k prompts, N released together", aria="Aggregate tokens per second by requests at once")
        out += table(["at once", "aggregate tok/s", "decode tok/s", "wall", "slowest first token",
                      "proof (most seen)", "proved rounds"], body, "num")
    if t32:
        groups = []
        for r in sorted(t32, key=lambda r: r["v"].get("n") or 0):
            v = r["v"]
            if r.get("valid"):
                groups.append((f"{v['n']} at once", [("first", v.get("ttft_min_s"), "s2", f"fastest {num(v.get('ttft_min_s'))} s"),
                                                     ("slowest", v.get("ttft_max_s"), "s1", f"slowest {num(v.get('ttft_max_s'))} s")]))
            else:
                groups.append((f"{v.get('n')} at once", [("", None, "s1", f"invalid: {r.get('why')}")]))
        out += '<h3>~32k prompts: first token of the fastest and the slowest request</h3>'
        out += legend([("fastest first token", "s2", True), ("slowest first token", "s1", True)])
        out += bar_chart(groups, "seconds", aria="First token times at 32k prompts")
    return out + invalids(rs) + skips(rs), ""


def sec_d(data):
    rs = rows_of(data, "d")
    ok = valid(rs)
    if not ok:
        return invalids(rs) + skips(rs), "no valid rows"
    groups = [(f"{r['v']['chats']} chats", [("first", r["v"].get("ttft_first_s"), "s2", f"first chat {num(r['v'].get('ttft_first_s'))} s"),
                                             ("last", r["v"].get("ttft_last_s"), "s1", f"last chat {num(r['v'].get('ttft_last_s'))} s")])
              for r in ok]
    body = [[str(r["v"]["chats"]), num(r["v"].get("prompt_tokens"), 0), num(r["v"].get("ttft_first_s")) + " s",
             num(r["v"].get("ttft_last_s")) + " s", esc(", ".join(f"{c:,}" for c in r["v"].get("cached_tokens") or []))]
            for r in ok]
    return (legend([("first chat's first token", "s2", True), ("last chat's first token", "s1", True)])
            + bar_chart(groups, "seconds", aria="First token of chats on one new system prompt")
            + table(["chats", "prompt tokens", "first chat", "last chat", "tokens from the cache, per chat"], body, "num")
            + invalids(rs) + skips(rs)), ""


def sec_e(data):
    rs = rows_of(data, "e")
    out = ""
    ld = next((r for r in rs if r["key"] == "load"), None)
    if ld:
        v = ld["v"]
        if ld.get("valid"):
            out += "<h3>A long prompt arriving while two chats decode</h3>"
            out += bar_chart([("two decoders", [("before", v.get("decoders_before_tok_s"), "s2", "before the prompt arrived"),
                                                ("during", v.get("decoders_during_tok_s"), "s1", "during its prefill")])],
                             "combined tok/s of the two decoding chats", aria="Decoders' speed before and during a long prefill")
            out += table(["arriving prompt", "its first token", "decoders before", "decoders during"],
                         [[num(v.get("prompt_tokens"), 0) + " tokens", num(v.get("ttft_s")) + " s",
                           num(v.get("decoders_before_tok_s")) + " tok/s", num(v.get("decoders_during_tok_s")) + " tok/s"]], "num")
    rt = next((r for r in rs if r["key"] == "return"), None)
    if rt and rt.get("valid"):
        v = rt["v"]
        out += "<h3>A conversation that comes back</h3>"
        out += bar_chart([("turn 1, cold", [("", v.get("turn1_ttft_s"), "s1", "turn 1")]),
                          ("turn 2, right after", [("", v.get("turn2_ttft_s"), "s1", "turn 2")]),
                          ("three other long prompts", [("", v.get("flood_ttft_median_s"), "s3", "median of the three")]),
                          ("turn 3, after them", [("", v.get("turn3_ttft_s"), "s1", "turn 3")])],
                         "seconds to the first token", aria="First token of a returning conversation")
        out += table(["conversation", "turn 1", "turn 2", "turn 2 from cache", "other prompts (median)", "turn 3",
                      "turn 3 from cache"],
                     [[num(v.get("prompt_tokens"), 0) + " tokens", num(v.get("turn1_ttft_s")) + " s",
                       num(v.get("turn2_ttft_s"), 2) + " s", num(v.get("turn2_cached"), 0),
                       num(v.get("flood_ttft_median_s")) + " s", num(v.get("turn3_ttft_s"), 2) + " s",
                       num(v.get("turn3_cached"), 0)]], "num")
    return out + invalids(rs) + skips(rs), "" if out else "no valid rows"


def sec_f(data):
    rs = rows_of(data, "f")
    ok = valid(rs)
    if not ok:
        return invalids(rs) + skips(rs), "no valid rows"
    tasks = []
    for r in ok:
        if r["cond"]["task"] not in tasks:
            tasks.append(r["cond"]["task"])
    groups = []
    for t in tasks:
        tr = [r for r in ok if r["cond"]["task"] == t]
        groups.append((t, [("mean", med([r["v"].get("decode_tok_s") for r in tr]), "s1", "mean over the answer"),
                           ("peak", med([r["v"].get("peak_1s_tok_s") for r in tr]), "s2", "best one-second window")]))
    body = [[esc(r["cond"]["task"]), str(r["cond"]["run"] + 1), num(r["v"].get("prompt_tokens"), 0),
             num(r["v"].get("ttft_s")) + " s", num(r["v"].get("decode_tok_s")), num(r["v"].get("peak_1s_tok_s")),
             num(r["v"].get("completion_tokens"), 0), "yes" if drafting_of(r) else "no"] for r in ok]
    return (legend([("mean decode", "s1", True), ("best 1 s window", "s2", True)])
            + bar_chart(groups, "tok/s", aria="Decode after a long prompt, by task")
            + table(["task", "run", "prompt tokens", "first token", "decode tok/s", "best 1 s", "answer tokens",
                     "drafting"], body, "num") + invalids(rs) + skips(rs)), ""


def sec_g(data):
    rs = rows_of(data, "g")
    if not rs:
        return "", "no rows"
    out = ""
    lv = {tag: {r["key"].split("/")[-1]: r for r in rs if r["key"].startswith(f"{tag}/levels/")}
          for tag in ("default", "fixed")}
    ns = sorted({int(n) for d in lv.values() for n in d}, key=int)
    if ns:
        series = []
        for tag, cls, dx in (("default", "s1", -5), ("fixed", "s2", 5)):
            d = lv[tag]
            pts = [(i, d[str(n)]["v"].get("wall_tok_s") if str(n) in d and d[str(n)].get("valid") else None, True,
                    f"{n} at once, {tag}: {num(d[str(n)]['v'].get('wall_tok_s')) if str(n) in d else '–'} tok/s")
                   for i, n in enumerate(ns)]
            series.append({"name": tag, "cls": cls, "pts": pts, "dx": dx,
                           "line": [(i, p[1]) for i, p in enumerate(pts)]})
        out += legend([("default arguments", "s1", True), ("--fixed-prompts", "s2", True)])
        out += xy_chart([str(n) for n in ns], series, "wall tok/s, the card tool's levels",
                        aria="Card tool wall tokens per second by requests at once")
        body = []
        for n in ns:
            cells = [str(n)]
            for tag in ("default", "fixed"):
                r = lv[tag].get(str(n))
                if r is None:
                    cells += ["–", "–"]
                elif not r.get("valid"):
                    cells += [bad_cell(r), "–"]
                else:
                    cells += [num(r["v"].get("wall_tok_s")), num(r["v"].get("ttft_s"), 2) + " s"]
            body.append(cells)
        out += table(["at once", "wall tok/s", "first token", "wall tok/s, fixed", "first token, fixed"], body, "num")
    other = [r for r in rs if "/decode/" in r["key"] or "/prefill/" in r["key"]]
    if other:
        body = []
        for r in other:
            if "/decode/" in r["key"]:
                body.append([f"decode, {esc(r['key'].split('/')[-1])}", num(r["v"].get("decode_tok_s")) + " tok/s"])
            else:
                body.append([f"prefill {kfmt(r['key'].split('/')[-1])}", num(r["v"].get("tok_s"), 0) + " tok/s, first token "
                             + num(r["v"].get("ttft_s"), 2) + " s"])
        out += table(["the card's other rows (default arguments)", "value"], body, "num")
    return out + invalids(rs), ""


def sec_h(data):
    rs = rows_of(data, "h")
    cmp_ = compare(rs)
    out = ""
    needles = [r for r in rs if r["key"].startswith("needle/")]
    if needles:
        out += '<p class="needles">' + " · ".join(
            f"needle at {kfmt(r['cond']['target'])}: " + ("found" if r["v"].get("found") else "not found")
            if r.get("valid") else f"needle at {kfmt(r['cond']['target'])}: invalid" for r in needles) + "</p>"
    if cmp_:
        same = sum(1 for c in cmp_ if c["same"])
        tie = sum(1 for c in cmp_ if not c["same"] and c["near_tie"])
        out += (f'<p class="big-line"><strong>{same} of {len(cmp_)}</strong> greedy texts identical with speculation on and off'
                + (f"; {tie} part at a near tie" if tie else "") + "</p>")
        spec = {r["key"].split("/", 1)[1]: r for r in rs if r["key"].startswith("spec/")}
        body = []
        for c in cmp_:
            t = (spec.get(c["prompt"]) or {}).get("v", {}).get("text") or ""
            state = "identical" if c["same"] else (f"parts at token {c['at']}, near tie (gap {c['gap']:.3f})" if c["near_tie"]
                                                    else f"parts at token {c['at']}" + (f", gap {c['gap']:.3f}" if c["gap"] is not None else ""))
            body.append([esc(c["prompt"]), esc(state), f'<span class="txt">{esc(t[:220])}{"…" if len(t) > 220 else ""}</span>'])
        out += table(["prompt", "speculative vs plain", "answer (start)"], body)
    else:
        own = [r for r in rs if r["key"].split("/", 1)[0] in ("spec", "plain") and r.get("valid")]
        if own:
            out += f"<p>{len(own)} greedy texts recorded; no second run with speculation off to compare against.</p>"
    return out + invalids(rs) + skips(rs), "" if out else "no rows"


def sec_i(data):
    rs = rows_of(data, "i")
    ok = valid(rs)
    if not ok:
        return invalids(rs) + skips(rs), "not run"
    L = by_length(data)
    groups, body = [], []
    for r in ok:
        n = int(r["cond"]["target"])
        early = med([x["v"].get("decode_tok_s") for x in L.get(n, [])])
        groups.append((kfmt(n), [("earlier", early, "s2", "section a, earlier in the server's life"),
                                 ("warm", r["v"].get("decode_tok_s"), "s1", "after the rest of the suite")]))
        body.append([kfmt(n), num(early), num(r["v"].get("decode_tok_s")),
                     "yes" if drafting_of(r) else "no", num(r["cond"].get("served_before"), 0)])
    return (legend([("earlier (section a)", "s2", True), ("after the rest of the suite", "s1", True)])
            + bar_chart(groups, "decode tok/s, one request", aria="Long prompt decode, earlier and warm")
            + table(["prompt", "earlier tok/s", "warm tok/s", "drafting (warm)", "requests served before"], body, "num")
            + invalids(rs) + skips(rs)), ""


def sec_j(data):
    rs = rows_of(data, "j")
    ok = valid(rs)
    if not ok:
        why = (rs[0].get("skipped") or rs[0].get("why")) if rs else "not run"
        return f'<p class="muted">{esc(why)}</p>', "not run"
    v = ok[0]["v"]
    lo, hi = v.get("ci95") or (None, None)
    return table(["correct", "score", "95 % interval", "errors", "unparsed", "capped"],
                 [[f'{v["correct"]} / {v["items"]}', num(100 * v["score"], 1, " %"),
                   f"{num(100 * lo, 1)} – {num(100 * hi, 1)} %", str(v["errors"]), str(v["unparsed"]),
                   str(v["capped"])]], "num"), ""


def sec_k(data):
    runs = data.get("runs") or []
    if not runs:
        return "", "no runs"
    last = runs[-1]
    knobs = last.get("knobs") or {}
    keep = ("lanes", "max_running", "max_len", "windows", "max_queue", "prefill_chunk", "prefix_cache", "prefix_chunk",
            "cache_gb", "ceiling_gb", "peak_gb", "suffix_store", "response_cache", "drafter", "corrector_loaded",
            "self_check", "admission_floor_gib")
    rows = [[esc(k), esc(json.dumps(knobs.get(k)) if isinstance(knobs.get(k), (list, dict, bool)) or knobs.get(k) is None
                     else knobs.get(k))] for k in keep if k in knobs]
    env = knobs.get("env") or {}
    if env:
        rows.append(["environment", '<span class="txt">' + esc(", ".join(f"{k}={v}" for k, v in sorted(env.items()))) + "</span>"])
    flags = knobs.get("flags") or {}
    if flags:
        rows.append(["engine flags", '<span class="txt">' + esc(" ".join(f"{k} {v}" if v is not True else k
                                                                         for k, v in flags.items())) + "</span>"])
    cor = last.get("corrector")
    if cor:
        rows.append(["draft corrector", '<span class="txt">' + esc(json.dumps({k: cor.get(k) for k in
                                                                            ("loaded", "file", "sha256") if k in cor})) + "</span>"])
    gpu = last.get("gpu") or {}
    rows = [["version", esc(last.get("version"))], ["weights", esc(f"{last.get('weights')} ({last.get('format')})")],
            ["weights repository", esc(f"{last.get('weights_repo') or '–'} @ {last.get('weights_revision') or '–'}")],
            ["drafters", esc(last.get("drafters") or "none")],
            ["GPU", esc(f"{gpu.get('name')} · {gpu.get('memory_gib')} GiB · compute {gpu.get('cc')} · driver "
                        f"{gpu.get('driver')} · CUDA {gpu.get('cuda')} · torch {gpu.get('torch')}")],
            ["prompt text", esc(f"{(last.get('prose') or {}).get('source')} (sha256 {str((last.get('prose') or {}).get('sha256'))[:16]}"
                                f"{', repeated' if (last.get('prose') or {}).get('repeats') else ''})")]] + rows
    runs_body = []
    for r in runs:
        b = r.get("box") or {}
        runs_body.append([esc(r.get("started", "")[:16].replace("T", " ")), esc(", ".join(r.get("sections") or [])),
                          esc(r.get("variant")), esc(r.get("mode")), num(r.get("minutes")) + " min",
                          num(b.get("clock_mhz_median_load"), 0) + " MHz", num(b.get("power_w_median_load")) + " W",
                          num(b.get("temp_c_max"), 0) + " °C", num(b.get("mem_avail_gib_min")) + " GiB"])
    return (table(["setting", "value"], rows, "kv")
            + "<h3>Runs</h3>"
            + table(["started", "sections", "variant", "mode", "duration", "GPU clock (median under load)",
                     "power (median under load)", "max temperature", "lowest free memory"], runs_body, "num")), ""


METHODS = {
    "a": "One request at a time, real prose with a fresh nonce first, greedy, thinking off, 512-token answers. "
         "Decode = completion tokens after the first over the time from the first to the last token; the median of the runs. "
         "Drafting and tokens per pass are the engine's own counts for each request.",
    "b": "The same requests: prefill = prompt tokens over the time to the first token.",
    "c": "N requests released together, each with its own ~2k-token prompt and a 512-token answer; aggregate = all "
         "completion tokens over the wall time, prefill included. A round counts only when the server's decode set held N "
         "requests at once and no more ran; a level without such a round is invalid. Below: N prompts of ~32k tokens, 16 "
         "tokens out, the first token of the fastest and the slowest; a request left in the queue makes the row invalid.",
    "d": "Chats released together on one new ~28k-token system prompt, each with its own short question, prompt cache on.",
    "e": "Two chats decode long answers; a ~64k prompt arrives. Then a ~97k-token conversation: turn 1 cold, turn 2 right "
         "after, three other ~60k prompts, turn 3. Prompt cache on, as a chat client sends it.",
    "f": "One request with a ~56k-token prompt, greedy, up to 1,024 tokens, three tasks: a summary, code, and a copy of "
         "text in the prompt. Peak = the most tokens in any one second after the first token.",
    "g": "The repository's tools/parallel_card.py against the same server: its default arguments, and its levels again "
         "with --fixed-prompts (the same prompts and seeds every round).",
    "h": "Eight fixed prompts, greedy, thinking off, 512 tokens, with speculation on and on a second server with it off "
         "(--drafter none): identical, or parting at a near tie (top-2 gap ≤ 0.125 nats). Needles: a passphrase in the "
         "middle of ~32k and ~128k tokens of prose.",
    "i": "The longest prompts again, one request each, after the rest of the suite on the same server, beside section a's "
         "rows of the same length.",
    "j": "A multiple-choice set graded by the letter, no judge, 95 % Wilson interval. Run only on request.",
    "k": "Every limit the server ran with, read from the running server, and the box during each run.",
}
RENDER = {"a": sec_a, "b": sec_b, "c": sec_c, "d": sec_d, "e": sec_e, "f": sec_f, "g": sec_g, "h": sec_h,
          "i": sec_i, "j": sec_j, "k": sec_k}


# ---- headline and verdict -----------------------------------------------------------------------

def tiles(data) -> list[tuple[str, str, str]]:
    out = []
    L = by_length(data)
    if L:
        n0 = min(L)
        out.append((num(med([r["v"].get("decode_tok_s") for r in L[n0]])), "tok/s", f"one request, {kfmt(n0)} prompt"))
        nl = max(L)
        out.append((num(med([r["v"].get("decode_tok_s") for r in L[nl]])), "tok/s", f"one request, {kfmt(nl)} prompt"))
        n32 = min(L, key=lambda n: abs(n - 32768))
        out.append((num(med([r["v"].get("prefill_tok_s") for r in L[n32]]), 0), "tok/s", f"prefill at {kfmt(n32)}"))
    two = [r for r in valid(rows_of(data, "c")) if r["key"].startswith("2k/")]
    if two:
        top = max(r["v"]["n"] for r in two)
        out.append((num(med([r["v"].get("agg_tok_s") for r in two if r["v"]["n"] == top])), "tok/s",
                    f"{top} requests at once, aggregate"))
    g = [r for r in valid(rows_of(data, "g")) if r["key"].startswith("fixed/levels/")]
    if g:
        r = max(g, key=lambda r: int(r["key"].split("/")[-1]))
        out.append((num(r["v"].get("wall_tok_s")), "tok/s", f"card tool, {r['key'].split('/')[-1]} at once, fixed prompts"))
    cmp_ = compare(rows_of(data, "h"))
    if cmp_:
        same = sum(1 for c in cmp_ if c["same"] or c["near_tie"])
        out.append((f"{same} / {len(cmp_)}", "", "greedy texts, speculation on = off"))
    return out


def findings(data) -> list[tuple[str, str]]:
    """The verdict: what the rows show, plainly. A cause is named only where the data holds an A/B."""
    out: list = []
    L = by_length(data)
    if L:
        meds = {n: med([r["v"].get("decode_tok_s") for r in rs]) for n, rs in L.items()}
        dr = {n: [drafting_of(r) for r in rs] for n, rs in L.items()}
        n0, nl = min(L), max(L)
        out.append(("note", f"One request decodes at {num(meds[n0])} tok/s with a {kfmt(n0)} prompt and "
                            f"{num(meds[nl])} tok/s with a {kfmt(nl)} prompt."))
        drafted = [n for n in L if all(dr[n])]
        plain = [n for n in L if not any(dr[n])]
        if drafted and plain and max(drafted) < min(plain):
            pm = med([meds[n] for n in plain])
            out.append(("bad", f"Drafting runs up to {kfmt(max(drafted))} prompts and stops from {kfmt(min(plain))} on "
                               f"(the engine's own counts); without it decode sits near {num(pm)} tok/s. The cause is not "
                               f"isolated."))
        elif drafted and not plain:
            out.append(("good", f"The engine drafted at every prompt length measured, up to {kfmt(nl)}."))
        pf = {n: med([r["v"].get("prefill_tok_s") for r in rs]) for n, rs in L.items()}
        out.append(("note", f"Prefill reads {num(max(pf.values()), 0)} tok/s at best and {num(pf[nl], 0)} tok/s at {kfmt(nl)}; "
                            f"a {kfmt(nl)} prompt waits {num(med([r['v'].get('ttft_s') for r in L[nl]]), 0)} s for its first token."))
    c = rows_of(data, "c")
    two = [r for r in c if r["key"].startswith("2k/")]
    if two:
        ok = valid(two)
        ns = sorted({r["v"]["n"] for r in ok})
        bad_ns = sorted({r["v"]["n"] for r in two} - set(ns))
        if ns:
            a1 = med([r["v"]["agg_tok_s"] for r in ok if r["v"]["n"] == ns[0]])
            an = med([r["v"]["agg_tok_s"] for r in ok if r["v"]["n"] == ns[-1]])
            if a1 and an:
                out.append(("good" if an > 1.5 * a1 else "bad",
                            f"{ns[-1]} requests at once total {num(an)} tok/s against {num(a1)} for {ns[0]} "
                            f"({an / a1:.1f} times), prefill included."))
        if bad_ns:
            out.append(("bad", "No proof that " + ", ".join(map(str, bad_ns)) + " requests decoded at once: those levels "
                               "are invalid and show no number."))
    d = valid(rows_of(data, "d"))
    if d:
        r = max(d, key=lambda r: r["v"]["chats"])
        out.append(("note", f"{r['v']['chats']} chats arriving together on one new {kfmt(r['v'].get('prompt_tokens'))} system prompt "
                            f"wait {num(r['v'].get('ttft_first_s'))} to {num(r['v'].get('ttft_last_s'))} s for a first token."))
    e = {r["key"]: r for r in valid(rows_of(data, "e"))}
    if "return" in e:
        v = e["return"]["v"]
        out.append(("good" if (v.get("turn3_ttft_s") or 99) < 0.2 * (v.get("turn1_ttft_s") or 1) else "note",
                    f"A {kfmt(v.get('prompt_tokens'))} conversation comes back in {num(v.get('turn3_ttft_s'), 2)} s after three "
                    f"other long prompts (cold: {num(v.get('turn1_ttft_s'))} s)."))
    if "load" in e:
        v = e["load"]["v"]
        out.append(("note", f"While a {kfmt(v.get('prompt_tokens'))} prompt prefills, two decoding chats drop from "
                            f"{num(v.get('decoders_before_tok_s'))} to {num(v.get('decoders_during_tok_s'))} tok/s combined; its "
                            f"first token comes after {num(v.get('ttft_s'))} s."))
    w = valid(rows_of(data, "i"))
    for r in w:
        n = int(r["cond"]["target"])
        early = med([x["v"].get("decode_tok_s") for x in L.get(n, [])])
        wv = r["v"].get("decode_tok_s")
        if early and wv:
            ratio = wv / early
            out.append(("bad" if ratio < 0.8 else "good",
                        f"A {kfmt(n)} prompt decodes at {num(wv)} tok/s after the rest of the suite against {num(early)} "
                        f"earlier in the same server's life ({ratio:.2f} times)" + (". The cause is not isolated." if ratio < 0.8 else ".")))
    cmp_ = compare(rows_of(data, "h"))
    if cmp_:
        same = sum(1 for c in cmp_ if c["same"])
        tie = sum(1 for c in cmp_ if not c["same"] and c["near_tie"])
        wide = len(cmp_) - same - tie
        out.append(("good" if wide == 0 else "bad",
                    f"Speculation leaves the greedy text unchanged on {same} of {len(cmp_)} prompts"
                    + (f"; {tie} part at a near tie" if tie else "") + (f"; {wide} part above a near tie" if wide else "") + "."))
    nd = valid([r for r in rows_of(data, "h") if r["key"].startswith("needle/")])
    if nd:
        found = [r for r in nd if r["v"].get("found")]
        out.append(("good" if len(found) == len(nd) else "bad",
                    f"Needles found: {len(found)} of {len(nd)} (" + ", ".join(kfmt(r["cond"]["target"]) for r in nd) + ")."))
    inv = [r for r in data.get("rows") or [] if not r.get("valid") and not r.get("skipped")]
    if inv:
        out.append(("bad", f"{len(inv)} measured rows are invalid (a missing proof of concurrency, a failed request or a "
                           "response not served raw); they show their reason and no number."))
    for x in data.get("notes") or []:
        out.append(("note", str(x)))
    return out


# ---- page ---------------------------------------------------------------------------------------

CSS = """
:root{--bg:#f6f7f8;--surface:#ffffff;--ink:#14181d;--ink2:#4a535e;--ink3:#77818d;--rule:rgba(20,24,29,.09);
--s1:#2a78d6;--s2:#eb6834;--s3:#77818d;--good:#0b7a0b;--bad:#b42f2f;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15191e;--surface:#1c2127;--ink:#e9edf2;--ink2:#a9b3bf;
--ink3:#7d8794;--rule:rgba(233,237,242,.09);--s1:#3987e5;--s2:#d95926;--s3:#7d8794;--good:#3fb950;--bad:#e5534b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15191e;--surface:#1c2127;--ink:#e9edf2;--ink2:#a9b3bf;--ink3:#7d8794;--rule:rgba(233,237,242,.09);
--s1:#3987e5;--s2:#d95926;--s3:#7d8794;--good:#3fb950;--bad:#e5534b;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
.wrap{max-width:1040px;margin:0 auto;padding-block:40px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:40px}
header{display:grid;gap:12px}
h1{font-size:clamp(26px,4vw,36px);line-height:1.15;font-weight:600;margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font-size:20px;font-weight:600;margin:0;text-wrap:balance}
h3{font-size:15px;font-weight:600;margin:20px 0 4px;color:var(--ink)}
.bare{font-family:var(--mono);font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--rule);
border-radius:8px;padding:8px 12px;justify-self:start}
.meta{color:var(--ink2);font-size:14px;margin:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:12px;padding:16px;display:grid;gap:4px;min-width:0}
.tile .fig{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}
.tile .fig small{font-size:13px;font-weight:400;color:var(--ink3);margin-left:4px}
.tile .cap{font-size:13px;color:var(--ink2)}
.verdict{display:grid;gap:8px;margin:0;padding:0;list-style:none}
.verdict li{display:grid;grid-template-columns:76px 1fr;gap:12px;align-items:baseline}
.tag{font-family:var(--mono);font-size:12px;color:var(--ink3)}
.tag.good{color:var(--good)}.tag.bad{color:var(--bad)}
section{display:grid;gap:12px;min-width:0;border-top:1px solid var(--rule);padding-top:24px}
.method{color:var(--ink2);font-size:14px;margin:0;max-width:78ch}
.chart{width:100%;height:auto;display:block;max-width:100%}
.grid{stroke:var(--rule);stroke-width:1}
.tick,.axis,.val,.lab{font-family:var(--mono);font-size:12px;fill:var(--ink3)}
.lab{fill:var(--ink2)}.val{fill:var(--ink2)}
.ln{fill:none;stroke-width:2}.ln.s1{stroke:var(--s1)}.ln.s2{stroke:var(--s2)}
.pt{stroke:var(--surface);stroke-width:2}.pt.s1{fill:var(--s1)}.pt.s2{fill:var(--s2)}.pt.s3{fill:var(--s3)}
.pt.hollow{fill:var(--surface);stroke-width:2}.pt.hollow.s1{stroke:var(--s1)}.pt.hollow.s2{stroke:var(--s2)}
.bar.s1{fill:var(--s1)}.bar.s2{fill:var(--s2)}.bar.s3{fill:var(--s3)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:500;color:var(--ink2);white-space:nowrap}
tr:last-child td{border-bottom:0}
table.num td{font-variant-numeric:tabular-nums;white-space:nowrap}
table.kv td:first-child{color:var(--ink2);white-space:nowrap;width:1%}
.txt{font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-word;color:var(--ink2)}
.inv{font-family:var(--mono);font-size:12px;color:var(--bad)}
.skips{margin:0;padding-left:18px;color:var(--ink3);font-size:13px}
.skips span{font-family:var(--mono)}
.inv-list{color:var(--bad)}
.needles,.big-line{margin:0}
.muted{color:var(--ink3);margin:0}
footer{color:var(--ink3);font-size:13px;border-top:1px solid var(--rule);padding-top:16px}
@media (max-width:560px){.verdict li{grid-template-columns:1fr;gap:2px}.tile .fig{font-size:24px}}
"""


def page(data: dict) -> str:
    private = data.get("private") or {}
    pub = scrub.deep({k: v for k, v in data.items() if k != "private"}, private)
    ver = str(pub.get("version") or "").split("+")[0]
    model = pub.get("model") or "model"
    title = f"TandemLLM {ver} · {model}"
    runs = pub.get("runs") or []
    last = runs[-1] if runs else {}
    dates = sorted({str(r.get("started", ""))[:10] for r in runs if r.get("started")})
    head = (f"{esc(pub.get('hardware') or 'one GPU')} · weights {esc(pub.get('weights'))}"
            + (f" ({esc(last.get('weights_repo'))})" if last.get("weights_repo") else "")
            + f" · measured {esc(' and '.join(dates) or '–')} · version {esc(pub.get('version'))}")
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
             f'content="width=device-width,initial-scale=1,viewport-fit=cover"><title>{esc(title)}</title>'
             f'<meta name="claude:viewport-fit-sides" content="cover"><style>{CSS}</style></head><body><div class="wrap">']
    parts.append(f'<header><h1>{esc(title)}</h1><p class="meta">{head}</p><div class="bare">{esc(BARE)}</div></header>')
    tl = tiles(pub)
    if tl:
        parts.append('<div class="tiles">' + "".join(
            f'<div class="tile"><div class="fig">{f}<small>{esc(u)}</small></div><div class="cap">{esc(c)}</div></div>'
            for f, u, c in tl) + "</div>")
    fd = findings(pub)
    if fd:
        parts.append('<section><h2>Verdict</h2><ul class="verdict">' + "".join(
            f'<li><span class="tag {k}">{ {"good": "good", "bad": "watch", "note": "measured"}[k] }</span><span>{esc(t)}</span></li>'
            for k, t in fd) + "</ul></section>")
    for sid, (name, _) in SECTIONS.items():
        body, why = RENDER[sid](pub)
        parts.append(f'<section id="s-{sid}"><h2>{esc(name)}</h2><p class="method">{esc(METHODS[sid])}</p>'
                     + (body or f'<p class="muted">{esc(why or "not measured in this report")}</p>') + "</section>")
    parts.append(f'<footer>Made with tools/release_report (tool {esc(pub.get("tool"))}) from data.json; every number on '
                 f'this page comes from one run on one box, two or three requests a cell. '
                 f'<a href="https://github.com/0xBakeer/TandemLLM">TandemLLM</a></footer></div></body></html>')
    out = "".join(parts)
    bad = scrub.leaks(out, private)
    if bad:
        raise Leak("the page still names the box: " + "; ".join(bad[:8]))
    return out
