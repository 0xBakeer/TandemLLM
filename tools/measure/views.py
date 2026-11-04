"""The three views of the record store; none measures anything.

  report  the HTML page (tools/measure/page.py): a data.json assembled from the records of one pack (or
          several packs of one model on one page), every row with its source
  card    a weight card's numbers (decode code/prose, prefill 2k/8k/32k, 1/2/4/8 at once) in the shape
          tools/parallel_card.py writes: its raw file and `row` (the model.yaml entry)
  gate    the gate's verdict table from two record stores (base and head), with the gate's own rules
          (tools/gate/verdict.py: the 3 % settle floor, row3 noise, INVALID)"""

from __future__ import annotations

import json
import os
import statistics

from tools import parallel_card as pc
from tools.measure import rows as rowsmod, store as storemod
from tools.measure.plan import source_label


def pick(st: storemod.Store, build: str | None, pack: str | None) -> tuple[str, str]:
    """The build and pack of a store that holds one of each, else the ones given."""
    keys = [k.split("|") for k in st.index()]
    builds = sorted({k[1] for k in keys})
    b = build or (builds[0] if len(builds) == 1 else None)
    if not b:
        raise SystemExit(f"[measure] the store holds builds {builds}: pass --build")
    ps = sorted({k[2] for k in keys if k[1] == b})
    p = pack or (ps[0] if len(ps) == 1 else None)
    if not p:
        raise SystemExit(f"[measure] the store holds packs {ps} for {b}: pass --pack")
    return b, p


# --- report -------------------------------------------------------------------------------------------

def _page_run(run: dict, sections: list[str]) -> dict:
    """A run as the page's section k lists it."""
    keep = ("started", "ended", "minutes", "variant", "version", "model", "weights", "weights_repo",
            "weights_revision", "format", "drafters", "corrector", "hardware", "gpu", "hw_profile", "knobs",
            "health_end", "served_at_start", "uptime_at_start_s", "box", "probe", "prose", "plan")
    out = {k: run.get(k) for k in keep if k in run}
    out["sections"] = sections
    out["mode"] = run.get("mode") or run.get("how")
    out["variant"] = out.get("variant") or "spec"
    return out


def report_data(st: storemod.Store, build: str, pack: str, box: str | None = None) -> dict:
    rows, _ = rowsmod.load()
    want = rowsmod.for_view(rows, "report")
    data_rows: list = []
    runs: dict = {}
    sources: list = []
    missing: list = []
    private: dict = {}
    first = None
    for r in want:
        rec = st.primary(r.id, build, pack, box)
        if rec is None:
            missing.append(r.id)
            continue
        first = first or rec
        label = source_label(rec)
        also = [x for x in st.find(r.id, build, pack, box) if not x["_primary"]]
        sources.append({"row": r.id, "source": label, "valid": rec.get("valid"), "why": rec.get("why"),
                        "skipped": rec.get("skipped"), "also": [source_label(x) for x in also]})
        secs = set()
        for x in rec.get("rows") or []:
            data_rows.append(dict(x, source=label, row=r.id))
            secs.add(x["section"])
            if r.id in ("length.128k", "length.192k") and x["section"] == "i":
                # one measurement, two sections: the fresh long prompt is also section a's row of its length
                data_rows.append(dict(x, section="a", key=f"{x['cond']['target']}/0", source=label, row=r.id,
                                      cond=dict(x["cond"], run=0)))
                secs.add("a")
        run = rec.get("_run") or {}
        for k, v in (run.get("private") or {}).items():
            private[k] = sorted(set(private.get(k) or []) | set(v or []))
        if run.get("how") == "imported" and run.get("origin_runs") is not None:
            st_ = (rec.get("source") or {}).get("run_started")
            for o in run["origin_runs"]:
                if o.get("started") == st_:
                    rid = f"{run['id']}@{st_}"
                    runs.setdefault(rid, dict(o, mode=f"imported ({o.get('mode') or 'url'})"))
        elif run.get("how") == "imported":
            rid = run["id"]
            if rid not in runs:
                runs[rid] = {"started": run.get("started"), "sections": [], "variant": "spec",
                             "mode": f"imported from {run.get('from_kind')}", "knobs": run.get("knobs") or {},
                             "version": run.get("version"), "weights": run.get("weights"), "box": {}}
            runs[rid]["sections"] = sorted(set(runs[rid]["sections"]) | secs)
        else:
            rid = run.get("id")
            if rid not in runs:
                runs[rid] = _page_run(run, [])
            runs[rid]["sections"] = sorted(set(runs[rid]["sections"]) | secs)
    if first is None:
        raise SystemExit(f"[measure] no record of a report row for {build} / {pack}")
    meta = next((x.get("_run") for x in (st.primary(r.id, build, pack, box) for r in want) if x and
                 (x.get("_run") or {}).get("model")), {}) or {}
    ordered = sorted(runs.values(), key=lambda x: str(x.get("started") or ""))
    return {"kind": "tandem-benchmark-report", "schema": 1, "tool": "measure 1.0",
            "version": first.get("version") or build, "model": meta.get("model"), "weights": pack,
            "hardware": meta.get("hardware") or "one DGX Spark (GB10)", "rows": data_rows, "runs": ordered,
            "sources": sources, "missing": missing, "private": private}


def render_report(st: storemod.Store, build: str | None, packs: list[str] | None, labels: list[str] | None,
                  model: str | None, out_html: str, data_out: str | None = None, box: str | None = None) -> str:
    from tools.measure import page
    if packs and len(packs) > 1:
        from tools.release_report.__main__ import combine
        datas = [report_data(st, build, p, box) for p in packs]
        data = combine(datas, labels or packs, model)
    else:
        b, p = pick(st, build, (packs or [None])[0])
        data = report_data(st, b, p, box)
        if model:
            data["model"] = model
    if data_out:
        with open(os.path.expanduser(data_out), "w") as fh:
            json.dump(data, fh, indent=1, default=str)
    html = page.page(data)
    with open(os.path.expanduser(out_html), "w") as fh:
        fh.write(html)
    return html


# --- card ---------------------------------------------------------------------------------------------

CARD_ROWS = ("decode.code.c1", "decode.prose.c1", "prefill.2k", "prefill.8k", "prefill.32k",
             "conc.card.c1", "conc.card.c2", "conc.card.c4", "conc.card.c8")


def card_raw(st: storemod.Store, build: str, pack: str, model_id: str, profile: str, env_name: str,
             box: str | None = None, commit: str | None = None) -> dict:
    """The card tool's raw file, assembled from the records (decode, prefill and the fixed-prompt levels),
    with `row` as tools/parallel_card.py builds it and every number's source."""
    recs = {rid: st.primary(rid, build, pack, box) for rid in CARD_ROWS}
    missing = [k for k, v in recs.items() if v is None]
    if missing:
        raise SystemExit(f"[measure] no record for {', '.join(missing)} ({build} / {pack}): the card needs them")
    bad = [k for k, v in recs.items() if not v.get("valid")]
    run = {}
    setup = None
    for v in recs.values():
        r = v.get("_run") or {}
        if (r.get("card") or {}).get("setup"):
            setup, run = r["card"], r
            break
        if r.get("health_start") and not run:
            run = r
    if setup is None:
        hs = run.get("health_start") or {}
        setup = {"setup": {"env": env_name, "lanes": hs.get("lanes"), "window": max(hs.get("windows") or [hs.get("max_len") or 0]),
                           "max_tokens": 512, "suffix_store": hs.get("suffix_store"), "endpoint": pc.ENDPOINT,
                           "response_cache": hs.get("response_cache"),
                           "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": pc.SEED_FIXED},
                           "corrector": hs.get("corrector"),
                           "prompts": {"set": (run.get("prompts") or {}).get("set"),
                                       "sha256": (run.get("prompts") or {}).get("sha256") or "",
                                       "filler_sha256": (run.get("prompts") or {}).get("filler_sha256"),
                                       "prompt_tokens": {}}},
                 "health_start": hs, "health_end": run.get("health_end")}
    s = json.loads(json.dumps(setup["setup"]))
    s["env"] = env_name
    s["sampling"]["seed"] = pc.SEED_FIXED
    pt = {w: (recs[f"decode.{w}.c1"].get("values") or {}).get("prompt_tokens") for w in ("code", "prose")}
    s["prompts"]["prompt_tokens"] = {w: int(v) for w, v in pt.items() if v}
    dec = {w: (recs[f"decode.{w}.c1"].get("values") or {}).get("decode_tok_s") for w in ("code", "prose")}
    pre = {}
    for k, t in (("2k", 2048), ("8k", 8192), ("32k", 32768)):
        v = recs[f"prefill.{k}"].get("values") or {}
        runs = (recs[f"prefill.{k}"].get("raw") or {}).get("runs") or []
        ptok = v.get("prompt_tokens") or (statistics.median([x["prompt_tokens"] for x in runs if x.get("prompt_tokens")])
                                          if any(x.get("prompt_tokens") for x in runs) else None)
        pre[str(t)] = {"ttft_s": v.get("ttft_s"), "tok_s": v.get("tok_s"), "prompt_tokens": ptok}
    lv = {}
    for n in (1, 2, 4, 8):
        rec = recs[f"conc.card.c{n}"]
        v = rec.get("values") or {}
        raw = rec.get("raw") or {}
        rounds = raw.get("rounds") if isinstance(raw, dict) and isinstance(raw.get("rounds"), list) else \
            [{}] * int(v.get("rounds") or (raw.get("rounds") if isinstance(raw.get("rounds"), int) else 3))
        at_once = v.get("at_once") or (raw.get("at_once") if isinstance(raw, dict) else None) or n
        lv[str(n)] = {"wall_tok_s": v.get("wall_tok_s"), "decode_tok_s": v.get("decode_tok_s"), "ttft_s": v.get("ttft_s"),
                      "at_once": at_once, "rounds": rounds, "valid": rec.get("valid")}
    first = recs["decode.code.c1"]
    commit = commit or (first.get("source") or {}).get("commit") or run.get("commit") or "unknown"
    date = (first.get("source") or {}).get("date") or ""
    raw = {"schema": pc.SCHEMA, "tool": "tools/measure (card view)", "model": model_id, "profile": profile,
           "build": build, "commit": str(commit)[:8],
           "date": date, "args": {"rounds": 3, "fixed_prompts": True, "levels": "1,2,4,8", "max_tokens": 512},
           "endpoint": pc.ENDPOINT, "levels_method": pc.LEVELS_FIXED, "setup": s,
           "health_start": setup.get("health_start"), "health_end": setup.get("health_end"),
           "decode": {"median": dec, "prompt_tokens": pt}, "prefill": {"depths": pre}, "levels": lv,
           "raw_path": f"models/{model_id}/measurements/parallel-{date}-{profile.split('/')[-1]}.json",
           "valid": not bad and all(x["valid"] for x in lv.values()),
           "sources": {rid: source_label(v) for rid, v in recs.items()}}
    if bad:
        raw["invalid_rows"] = bad
    raw["row"] = pc.build_row(raw)
    return raw


# --- gate ---------------------------------------------------------------------------------------------

GATE_SUITES = ("lossless", "decode", "prefill", "long", "stability")


def gate_servers(rec: dict) -> list[dict]:
    """A record's gate structure, one dict per server (an import holds the source's servers)."""
    from tools.measure.merge import gate_sub
    raw = rec.get("raw")
    if isinstance(raw, dict) and "servers" in raw:
        return raw["servers"]
    rid = rec["row"]
    if rid.startswith("conc.card.c"):
        return [{f"c{rid.split('.c')[-1]}": raw}]
    if rid.startswith("prefill."):
        t = rowsmod.TOKENS[rid.split(".")[1]]
        return [{"depths": {str(t): {"runs": raw.get("runs")}}, "tokenize_offset": raw.get("tokenize_offset")}]
    return [gate_sub(rid, raw)] if raw is not None else []


def gate_raw(st: storemod.Store, build: str, pack: str, side: str) -> dict:
    """{suite: {arm key: [one dict a server]}} for one side, as the gate's runner holds it."""
    from tools.measure.merge import SUITE_OF, arm_of
    rows, _ = rowsmod.load()
    out: dict = {}
    for r in rowsmod.for_view(rows, "gate"):
        rec = st.primary(r.id, build, pack)
        if rec is None:
            continue
        suite = SUITE_OF.get(r.id) or ("prefill" if r.id.startswith("prefill.") else r.id.split(".")[0])
        arm = arm_of(r.id, side)
        lst = out.setdefault(suite, {}).setdefault(arm, [])
        for i, srv in enumerate(gate_servers(rec)):
            while len(lst) <= i:
                lst.append({})
            for k, v in (srv or {}).items():
                if k == "depths":
                    lst[i].setdefault("depths", {}).update(v)
                else:
                    lst[i][k] = v
    return out


def render_gate(base_dir: str, head_dir: str, build: str | None = None, pack: str | None = None,
                hw: str | None = None) -> tuple[list[dict], str]:
    from tools.gate import hw as hwmod
    from tools.gate.render import table
    from tools.gate.suites import Plan, module
    from tools.gate.verdict import overall
    bs, hs = storemod.Store(base_dir), storemod.Store(head_dir)
    bb, bp = pick(bs, None, pack)
    hb, hp = pick(hs, build, pack)
    box = next(iter(k.split("|")[3] for k in hs.index()), "gb10")
    prof = hwmod.profile(hw or box)
    raw: dict = {}
    for side, s, b, p in (("base", bs, bb, bp), ("head", hs, hb, hp)):
        for suite, arms in gate_raw(s, b, p, side).items():
            raw.setdefault(suite, {}).update(arms)
    suites = [x for x in GATE_SUITES if x in raw]
    level = "full" if "stability" in raw else ("quick+long" if any(":wide" in k for k in raw.get("long", {})) else "quick")
    plan = Plan(level, suites, [], prof, {"plain_env": {"x": "1"}} if any(":plain" in k for k in raw.get("lossless", {}))
                else {}, 1)
    out = []
    for s in suites:
        out += module(s).rows(raw[s], plan)
    return out, f"verdict {overall(out)}\n\n" + table(out) + "\n"
