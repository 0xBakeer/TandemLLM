"""The code that measures each row (rows.yaml `exec`). An executor takes the rows of one batch (consecutive
rows of the same executor) and returns, by row id, the payload of its record:

    {"rows": [...], "raw": ..., "values": {...}, "valid": bool, "why": str | None, "skipped": str | None}

`rows` are the report's section rows (tools/release_report/sections: the page reads them), `raw` the
card's or the gate's own structure (the card and gate views read it), `values` the row's headline
numbers. Nothing here measures by itself: each executor calls the measuring code where it lives (the
report's sections, tools/parallel_card.py, the gate's suites) with only the rows asked."""

from __future__ import annotations

import argparse
import dataclasses
import statistics

from tools import parallel_card as pc

#: a server that has served at most this many requests is fresh (the run's probe)
FRESH = 2


@dataclasses.dataclass
class MCtx:
    base: str
    cl: object                 # tools/measure/client.ReportClient (greedy, the report's rows)
    gcl: object                # tools/gate/client.GateClient (the card's sampling: card and gate rows)
    health: dict
    knobs: dict
    plan: object               # tools/release_report/plan.Plan sized from this server
    prose: object
    pj: dict                   # bench/card-prompts-v1.json
    profile: object            # tools/gate/hw profile
    pack: dict                 # the gate pack (burst_word, proof kind)
    out: object
    log: object = print
    variant: str = "spec"
    args: object = None
    proof_kind: str = "decoding"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def payload(rows=None, raw=None, values=None, valid=True, why=None, skipped=None) -> dict:
    return {"rows": rows or [], "raw": raw, "values": values or {}, "valid": bool(valid) and not skipped,
            "why": why or skipped, "skipped": skipped}


def from_section(section_rows: list[dict], values=None) -> dict:
    """A row's payload from its section rows: invalid when one of them is (with its reason), skipped when
    the plan left it out."""
    sk = [r for r in section_rows if r.get("skipped")]
    if sk and len(sk) == len(section_rows):
        return payload(section_rows, skipped=sk[0]["skipped"])
    bad = [r for r in section_rows if not r.get("valid") and not r.get("skipped")]
    return payload(section_rows, values=values, valid=not bad, why=bad[0]["why"] if bad else None)


# --- the report's sections ------------------------------------------------------------------------

def section_ctx(mx: MCtx, **plan_fields):
    """The report's section context with a plan cut down to these rows (every other shape empty)."""
    from tools.release_report.sections import Ctx
    empty = dict(lengths=[], runs={}, conc_2k=[], conc_32k=[], chats=[], load_prompt=None, return_prompt=None,
                 flood_prompt=None, long_prompt=None, long_tasks=[], needles=[], warm=[])
    empty.update(plan_fields)
    # the executors say a row's own skip; the sections get none
    pl = dataclasses.replace(mx.plan, skipped={}, **empty)
    a = mx.args if mx.args is not None else argparse.Namespace()
    a.proof_kind = mx.proof_kind
    return Ctx(mx.base, mx.cl, mx.health, pl, mx.prose, mx.out, mx.log, mx.variant, a)


def plan_fits(mx: MCtx, key: str) -> str | None:
    """Why the plan sized from this server leaves a shape out, or None."""
    return mx.plan.skipped.get(key)


def ex_lengths(mx: MCtx, rows) -> dict:
    from tools.release_report.plan import kfmt
    from tools.release_report.sections import lengths
    out, todo = {}, []
    for r in rows:
        n = r.tokens
        why = plan_fits(mx, f"a.{kfmt(n)}")
        if why or n not in mx.plan.lengths:
            out[r.id] = payload(skipped=why or f"{n} tokens is not in the plan sized from this server")
        else:
            todo.append((r, n))
    if todo:
        ctx = section_ctx(mx, lengths=[n for _, n in todo], runs={n: mx.plan.runs[n] for _, n in todo})
        got = lengths.measure(ctx)
        for r, n in todo:
            mine = [x for x in got if x.get("cond", {}).get("target") == n]
            out[r.id] = from_section(mine, {"decode_tok_s": med([x["v"].get("decode_tok_s") for x in mine]),
                                            "ttft_s": med([x["v"].get("ttft_s") for x in mine]),
                                            "drafting": [((x.get("engine") or [{}])[0]).get("drafting") for x in mine]})
    return out


def ex_needle(mx: MCtx, rows) -> dict:
    from tools.release_report.plan import kfmt
    from tools.release_report.sections import fidelity
    out = {}
    for r in rows:
        n = r.tokens
        why = plan_fits(mx, f"h.needle.{kfmt(n)}")
        if why:
            out[r.id] = payload(skipped=why)
            continue
        got = fidelity.needles(section_ctx(mx, needles=[n]))
        mine = [x for x in got if x["key"] == f"needle/{n}"]
        out[r.id] = from_section(mine, {"found": mine[0]["v"].get("found") if mine else None})
    return out


def _fresh_or_warm(mx: MCtx, rows, want: str) -> dict:
    from tools.release_report.plan import kfmt
    from tools.release_report.sections import warm
    out, todo = {}, []
    for r in rows:
        n = r.tokens
        why = plan_fits(mx, f"i.{kfmt(n)}")
        (out.__setitem__(r.id, payload(skipped=why)) if why else todo.append((r, n)))
    if not todo:
        return out
    got = warm.measure(section_ctx(mx, warm=[n for _, n in todo]))
    for r, n in todo:
        mine = [x for x in got if x.get("cond", {}).get("target") == n]
        p = from_section(mine, {"decode_tok_s": med([x["v"].get("decode_tok_s") for x in mine]),
                                "ttft_s": med([x["v"].get("ttft_s") for x in mine])})
        state = mine[0]["key"].split("/")[0] if mine else None
        if p["valid"] and state != want:
            p.update(valid=False, why=f"the server was {state}, this row needs it {want} "
                                      f"(served {mine[0]['cond'].get('served_before')} before)")
        out[r.id] = p
    return out


def ex_fresh_long(mx: MCtx, rows) -> dict:
    return _fresh_or_warm(mx, rows, "fresh")


def ex_warm(mx: MCtx, rows) -> dict:
    return _fresh_or_warm(mx, rows, "warm")


def ex_conc2k(mx: MCtx, rows) -> dict:
    from tools.release_report.sections import concurrency
    ns = [r.n for r in rows if r.n in mx.plan.conc_2k]
    out = {r.id: payload(skipped=f"{r.n} at once is over the server's slot limit {mx.plan.lanes}")
           for r in rows if r.n not in ns}
    if ns:
        got = concurrency.measure(section_ctx(mx, conc_2k=ns, conc_32k=[], conc_rounds=mx.plan.conc_rounds))
        for r in rows:
            if r.n in ns:
                mine = [x for x in got if x["key"].startswith(f"2k/{r.n}/")]
                out[r.id] = from_section(mine, {"agg_tok_s": med([x["v"].get("agg_tok_s") for x in mine]),
                                                "decode_tok_s": med([x["v"].get("decode_tok_s") for x in mine])})
    return out


def ex_conc32k(mx: MCtx, rows) -> dict:
    from tools.release_report.sections import concurrency
    ns = [r.n for r in rows if r.n in mx.plan.conc_32k]
    out = {r.id: payload(skipped=mx.plan.skipped.get(f"c.32k.{r.n}") or f"{r.n} at once is not in the plan")
           for r in rows if r.n not in ns}
    if ns:
        got = concurrency.measure(section_ctx(mx, conc_2k=[], conc_32k=ns))
        for r in rows:
            if r.n in ns:
                mine = [x for x in got if x["key"] == f"32k/{r.n}"]
                out[r.id] = from_section(mine, {"ttft_max_s": mine[0]["v"].get("ttft_max_s") if mine else None})
    return out


def ex_shared(mx: MCtx, rows) -> dict:
    from tools.release_report.sections import shared
    if not mx.plan.shared_tokens:
        return {r.id: payload(skipped=mx.plan.skipped.get("d.prompt") or "no room for the shared prompt")
                for r in rows}
    ns = [r.n for r in rows if r.n <= mx.plan.lanes]
    out = {r.id: payload(skipped=f"{r.n} chats is over the slot limit {mx.plan.lanes}") for r in rows if r.n not in ns}
    if ns:
        got = shared.measure(section_ctx(mx, chats=ns))
        for r in rows:
            if r.n in ns:
                mine = [x for x in got if x["key"] == str(r.n)]
                out[r.id] = from_section(mine, {"ttft_last_s": mine[0]["v"].get("ttft_last_s") if mine else None,
                                                "cached_tokens": mine[0]["v"].get("cached_tokens") if mine else None})
    return out


def _load_part(mx: MCtx, rows, part: str) -> dict:
    from tools.release_report.sections import load
    pl = mx.plan
    if part == "load":
        if not pl.load_prompt:
            return {r.id: payload(skipped=pl.skipped.get("e.load") or "not in the plan") for r in rows}
        got = load.measure(section_ctx(mx, load_prompt=pl.load_prompt))
    else:
        if not pl.return_prompt:
            return {r.id: payload(skipped=pl.skipped.get("e.return") or "not in the plan") for r in rows}
        got = load.measure(section_ctx(mx, return_prompt=pl.return_prompt, flood_prompt=pl.flood_prompt))
    mine = [x for x in got if x["key"] == part]
    return {r.id: from_section(mine, dict(mine[0]["v"]) if mine else {}) for r in rows}


def ex_load(mx: MCtx, rows) -> dict:
    return _load_part(mx, rows, "load")


def ex_return(mx: MCtx, rows) -> dict:
    return _load_part(mx, rows, "return")


def ex_longdecode(mx: MCtx, rows) -> dict:
    from tools.release_report.sections import longdecode
    if not mx.plan.long_prompt:
        return {r.id: payload(skipped=mx.plan.skipped.get("f") or "not in the plan") for r in rows}
    tasks = [r.param["w"] for r in rows]
    got = longdecode.measure(section_ctx(mx, long_prompt=mx.plan.long_prompt, long_tasks=tasks,
                                         long_runs=mx.plan.long_runs))
    out = {}
    for r in rows:
        mine = [x for x in got if x["key"].startswith(r.param["w"] + "/")]
        out[r.id] = from_section(mine, {"decode_tok_s": med([x["v"].get("decode_tok_s") for x in mine])})
    return out


# --- the card tool's parts (tools/parallel_card.py) -------------------------------------------------

def _not_raw(streams: list[dict]) -> str | None:
    bad = [s for s in streams if s.get("x_tandem") != "raw"]
    if bad:
        return (f"{len(bad)} response(s) not served raw (X-Tandem: {bad[0].get('x_tandem') or 'missing'}): "
                "not the bare engine")
    return None


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("t_send", "t_first", "t_last", "t_end")}


def ex_card_decode(mx: MCtx, rows) -> dict:
    a = argparse.Namespace(warmups=1, decode_runs=3, max_tokens=512)
    pc.wait_idle(mx.base, 300)
    raw = pc.run_decode(mx.gcl, mx.pj, a, mx.log)
    why = _not_raw(raw["warmups"] + raw["runs"]["code"] + raw["runs"]["prose"])
    out = {}
    for r in rows:
        w = r.param["w"]
        v = {"decode_tok_s": raw["median"][w], "prompt_tokens": raw["prompt_tokens"][w]}
        g = {"section": "g", "key": f"default/decode/{w}", "v": {"decode_tok_s": raw["median"][w]},
             "cond": {"args": "default"}, "valid": why is None and v["decode_tok_s"] is not None, "why": why}
        out[r.id] = payload([g], {"runs": [_strip(x) for x in raw["runs"][w]], "median": raw["median"][w],
                                  "prompt_tokens": raw["prompt_tokens"][w]}, v,
                            valid=g["valid"], why=why or (None if v["decode_tok_s"] else "no decode window"))
    return out


def ex_card_prefill(mx: MCtx, rows) -> dict:
    depths = [r.tokens for r in rows]
    a = argparse.Namespace(prefill_depths=",".join(map(str, depths)), prefill_runs=3)
    pc.wait_idle(mx.base, 300)
    pre = pc.run_prefill(mx.gcl, mx.pj, a, mx.log)
    out = {}
    for r in rows:
        d = pre["depths"][str(r.tokens)]
        runs = [_strip(x) for x in d["runs"]]
        why = _not_raw(d["runs"])
        hit = [x for x in runs if x.get("cached_tokens")]
        why = why or (f"{len(hit)} request(s) were served from a cache: not a cold prefill" if hit else None)
        v = {"tok_s": d["tok_s"], "ttft_s": d["ttft_s"], "prompt_tokens": d["prompt_tokens"]}
        g = {"section": "g", "key": f"default/prefill/{r.tokens}", "v": v, "cond": {"args": "default"},
             "valid": why is None, "why": why}
        out[r.id] = payload([g], {"runs": runs, "tokenize_offset": pre["tokenize_offset"]}, v, valid=why is None,
                            why=why)
    return out


def level_values(lv: dict, kind: str = "decoding") -> dict:
    rs = lv.get("rounds") or []
    return {"wall_tok_s": med([x.get("wall_tok_s") for x in rs]), "decode_tok_s": med([x.get("decode_tok_s") for x in rs]),
            "ttft_s": med([s.get("ttft_s") for x in rs for s in x.get("streams") or []]),
            "at_once": min((x["proof"]["max_decoding_at_once" if kind == "decoding" else "max_running"]
                            for x in rs), default=0), "rounds": len(rs),
            "tokens_per_request": sorted({s.get("completion_tokens") for x in rs for s in x.get("streams") or []})}


def gate_ctx(mx: MCtx, side: str = "head", level: str = "full"):
    from tools.gate.suites import Ctx
    return Ctx(side, "main" if mx.variant in ("spec", "main") else mx.variant, mx.base, mx.gcl, mx.health,
               mx.knobs, mx.profile, level, mx.pack, mx.out, mx.pj, mx.log)


def _levels(mx: MCtx, rows, fixed: bool, tag: str) -> dict:
    """N card prose requests at once, warm-up then 3 proved rounds: fixed prompts through the gate's
    decode rounds (its proof rule, also the fake engine's `running` kind), default arguments through
    tools/parallel_card.py's levels."""
    from tools.gate.suites import decode as gdecode
    lim = (mx.health.get("parallel") or {}).get("max_running") or 1
    out = {}
    ns = [r.n for r in rows if r.n <= lim]
    for r in rows:
        if r.n > lim:
            out[r.id] = payload(skipped=f"the server's live slot limit is {lim}, under {r.n} at once")
    lvs: dict = {}
    if ns and fixed:
        gctx = gate_ctx(mx)
        for n in ns:
            lvs[n] = gdecode.rounds_of(gctx, n, mx.pj, 512)
    elif ns:
        a = argparse.Namespace(levels=",".join(map(str, ns)), fixed_prompts=False, levels_thinking="off",
                               warm_rounds=1, rounds=3, retries=2, max_tokens=512)
        got = pc.run_levels(mx.gcl, mx.pj, a, mx.log)
        for n in ns:
            lv = got[str(n)]
            lv["proof_kind"] = "decoding"
            lvs[n] = lv
    for r in rows:
        if r.n not in lvs:
            continue
        lv = lvs[r.n]
        for x in lv["rounds"] + lv["invalid"] + lv["warm"]:
            for s in x.get("streams") or []:
                for k in ("t_send", "t_first", "t_last", "t_end"):
                    s.pop(k, None)
        v = level_values(lv, lv.get("proof_kind", "decoding"))
        streams = [s for x in lv["rounds"] for s in x.get("streams") or []]
        why = _not_raw(streams)
        if not why and len(lv["rounds"]) < 3:
            kind = lv.get("proof_kind", "decoding")
            seen = max((x["proof"]["max_decoding_at_once" if kind == "decoding" else "max_running"]
                        for x in lv["invalid"]), default=0)
            why = f"{len(lv['rounds'])} of 3 rounds proved {r.n} {kind} at once (the most seen {seen})"
        g = {"section": "g", "key": f"{tag}/levels/{r.n}", "v": {k: v[k] for k in ("wall_tok_s", "decode_tok_s",
                                                                                 "ttft_s", "at_once", "rounds")},
             "cond": {"args": tag, "rounds": 3, "proof_kind": lv.get("proof_kind")}, "valid": why is None, "why": why}
        out[r.id] = payload([g], lv, v, valid=why is None, why=why)
    return out


def ex_levels_fixed(mx: MCtx, rows) -> dict:
    return _levels(mx, rows, True, "fixed")


def ex_levels_default(mx: MCtx, rows) -> dict:
    return _levels(mx, rows, False, "default")


# --- the gate's suites ---------------------------------------------------------------------------

def _gate_check(raw) -> str | None:
    """Every request record in a gate raw structure answered raw."""
    seen = []

    def walk(x):
        if isinstance(x, dict):
            if "x_tandem" in x:
                seen.append(x["x_tandem"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(raw)
    bad = [m for m in seen if m != "raw" and not (isinstance(m, list) and m == ["raw"])]
    return f"{len(bad)} response(s) not served raw: not the bare engine" if bad else None


def ex_gate_decode(mx: MCtx, rows) -> dict:
    from tools.gate.suites import decode as gdecode
    gctx = gate_ctx(mx)
    shapes = {sid: (n, pj, mt) for sid, n, pj, mt in gdecode.shapes(gctx)}
    out = {}
    for r in rows:
        sid = {"decode.c1.short": "c1", "decode.c1.1k": "c1-1k"}[r.id]
        n, pj, mt = shapes[sid]
        lv = gdecode.rounds_of(gctx, n, pj, mt)
        why = _gate_check(lv["rounds"]) or (None if len(lv["rounds"]) >= gctx.profile.rounds else
                                            f"{len(lv['rounds'])} of {gctx.profile.rounds} rounds proved")
        out[r.id] = payload([], {sid: lv}, {"decode_tok_s": med([x.get("decode_tok_s") for x in lv["rounds"]])},
                            valid=why is None, why=why)
    return out


def ex_gate_burst(mx: MCtx, rows) -> dict:
    from tools.gate.suites import prefill as gprefill
    b = gprefill.burst(gate_ctx(mx))
    rs = [x for x in b["rounds"] if not x["warm"]]
    inv = [x for x in rs if not x["valid"]]
    why = (f"{len(inv)} round(s) without {b['n']} running at once or with errors" if inv else None) \
        or ("a burst request was served from a cache" if any(x["cached"] for x in rs) else None) \
        or (None if all(x.get("x_tandem") == ["raw"] for x in b["rounds"]) else "a response not served raw")
    v = {"ttft_median_s": med([x["ttft_median_s"] for x in rs]), "n": b["n"], "depth": b["depth"]}
    row = {"section": "c", "key": f"burst/{b['n']}x{b['depth'] // 1024}k", "v": v,
           "cond": {"rounds": len(rs), "proof": {"max_running": min((x["proof"]["max_running"] for x in rs), default=0)}},
           "valid": why is None, "why": why}
    return {r.id: payload([row], {"burst": b}, v, valid=why is None, why=why) for r in rows}


def ex_gate_long(mx: MCtx, rows, wide: bool = False) -> dict:
    from tools.gate.suites import long as glong
    gctx = gate_ctx(mx)
    got = glong.measure_wide(gctx) if wide else glong.measure(gctx)
    rec = got["wide" if wide else "max"]
    why = None if rec["ok"] else f"the request failed: {rec.get('error')}"
    v = {k: rec.get(k) for k in ("ok", "needle", "e2e_s", "prompt_tokens", "reserved_max_gb", "peak_gb", "answer")}
    h = {"section": "h", "key": f"needle/{rec['tokens_asked']}", "v": {"found": rec["needle"], "answer": rec["answer"],
                                                                      "prompt_tokens": rec["prompt_tokens"]},
         "cond": {"target": rec["tokens_asked"], "shape": "the gate's long case (card filler)"},
         "valid": why is None, "why": why}
    return {r.id: payload([] if wide else [h], got, v, valid=why is None, why=why) for r in rows}


def ex_gate_long_wide(mx: MCtx, rows) -> dict:
    return ex_gate_long(mx, rows, wide=True)


def ex_gate_lossless(mx: MCtx, rows, plain: bool = False) -> dict:
    from tools.gate.suites import lossless
    gctx = gate_ctx(mx)
    got = lossless.measure_plain(gctx) if plain else lossless.measure(gctx)
    errs = [k for k, x in got.items() if x.get("error")]
    why = _gate_check(got) or (f"{len(errs)} request(s) failed: {errs[0]}" if errs else None)
    return {r.id: payload([], got, {"texts": {k: x.get("text_sha256") for k, x in got.items()}},
                          valid=why is None, why=why) for r in rows}


def ex_gate_lossless_plain(mx: MCtx, rows) -> dict:
    return ex_gate_lossless(mx, rows, plain=True)


def ex_gate_stability(mx: MCtx, rows) -> dict:
    from tools.gate.suites import stability
    got = stability.measure(gate_ctx(mx))
    return {r.id: payload([], got, {k: got.get(k) for k in ("failed", "server_errors", "reserved_max_gb", "drift_pct")},
                          valid=got.get("rc") == 0, why=None if got.get("rc") == 0 else f"soak exit {got.get('rc')}")
            for r in rows}


EXECS = {
    "report.lengths": ex_lengths, "report.needle": ex_needle, "report.fresh_long": ex_fresh_long,
    "report.warm": ex_warm, "report.conc2k": ex_conc2k, "report.conc32k": ex_conc32k,
    "report.shared": ex_shared, "report.load": ex_load, "report.return": ex_return,
    "report.longdecode": ex_longdecode,
    "card.decode": ex_card_decode, "card.prefill": ex_card_prefill,
    "card.levels.fixed": ex_levels_fixed, "card.levels.default": ex_levels_default,
    "gate.decode": ex_gate_decode, "gate.burst": ex_gate_burst, "gate.long": ex_gate_long,
    "gate.long_wide": ex_gate_long_wide, "gate.lossless": ex_gate_lossless,
    "gate.lossless_plain": ex_gate_lossless_plain, "gate.stability": ex_gate_stability,
}

#: rows that measure N requests at once: the record must carry the server's proof
AT_ONCE = ("conc.", "shared.", "burst.")


def proof_problem(rid: str, n: int | None, p: dict, kind: str = "decoding") -> str | None:
    """The runner's own check of a row of N at once, on top of the executor's: a valid record must carry
    the server's proof (/health during the run) that N ran together; for the decode rows (`conc.`) that N
    decoded together (or ran, on a server without a decode phase)."""
    if not p["valid"] or not rid.startswith(AT_ONCE):
        return None
    proofs = []

    def walk(x):
        if isinstance(x, dict):
            if "max_running" in x and ("max_decoding_at_once" in x or "samples" in x or "kind" in x):
                proofs.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(p.get("rows"))
    raw = p.get("raw")
    if isinstance(raw, dict):
        walk(raw.get("rounds"))
        walk(raw.get("burst"))
    if not proofs:
        return "no proof of the requests at once was recorded (/health during the run)"
    decode_rows = rid.startswith(("conc.card.", "conc.2k."))
    if n:
        key = "max_decoding_at_once" if decode_rows and kind == "decoding" else "max_running"
        counted = [x for x in proofs if key in x]
        if not counted or min(x[key] for x in counted) < n:
            seen = min((x.get(key, 0) for x in counted), default=0)
            what = "decoding" if key == "max_decoding_at_once" else "running"
            return f"the server's {what} count reached {seen} of {n} at once in a counted round"
    return None
