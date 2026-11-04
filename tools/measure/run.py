"""`python -m tools.measure run`: measure the rows asked (`--rows` or a view's) that the record store at
`--out` does not hold for this server's key (build, pack, box, knobs hash), against a running server.

Before anything is measured it refuses, as the report did: a server without status ok, with the
response cache on, busy (`--wait-idle`), or answering a probe without `X-Tandem: raw`. It reads the
server's knobs from its own process (a local server) and the GPU's profile, then measures row after
row in the row list's order, one batch per executor, and writes one run file. A row whose records
answered without `X-Tandem: raw`, or a row of N at once without the server's proof that N ran (decoded)
together, is INVALID. A run takes the rows of one server group (rows.yaml `server`): a warm row needs
the rows it follows in the same run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from tools import parallel_card as pc
from tools.measure import VERSION, client as cmod, execs, rows as rowsmod, store as storemod
from tools.measure.box import BoxSampler
from tools.measure.server import (Refused, bare_refusal, hardware_label, is_fake, label_of, local_pid, model_card,
                                  probe, tokenizer_for, wait_idle)

REPO = Path(__file__).resolve().parent.parent.parent


def tool_commit() -> str | None:
    """The measuring code's commit: git, or a `git archive` tree's .tool-commit file."""
    try:
        got = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=8", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        got = ""
    f = REPO / ".tool-commit"
    return got or (f.read_text().strip() if f.exists() else None)


def server_key(h: dict, knobs: dict, prof, a) -> dict:
    w = h.get("weights") or {}
    return {"build": a.build or storemod.build_of(h.get("version")),
            "pack": a.pack or w.get("name") or "unknown", "box": prof.id,
            "knobs_hash": storemod.knobs_hash(knobs)}


def wanted(a, rows: dict) -> list:
    if a.rows:
        return rowsmod.select(rows, a.rows)
    return rowsmod.for_view(rows, a.view)


def check_order(todo: list, rows: dict, groups: dict, a) -> None:
    """One server group a run; a warm row after the rows it follows, in this run."""
    gs = {r.server for r in todo if r.server}
    if len(gs) > 1 and not a.any_group:
        raise SystemExit(f"[measure] refused: the rows are of {len(gs)} server groups ({', '.join(sorted(gs))}); "
                         "a run measures one server's rows (rows.yaml `server`); run each group on its own "
                         "server, or pass --any-group")
    ids = [r.id for r in todo]
    for r in todo:
        for pat in r.after:
            before = [x for x in rows.values() if rowsmod.matches(x.id, [pat]) and x.server == r.server]
            missing = [x.id for x in before if x.id not in ids[:ids.index(r.id)]]
            if missing and not a.any_order:
                raise SystemExit(f"[measure] refused: {r.id} is measured after {', '.join(missing[:4])}"
                                 f"{' ...' if len(missing) > 4 else ''} on the same server, and they are not in "
                                 "this run before it (--any-order to measure it all the same)")


def batches(todo: list) -> list[tuple[str, list]]:
    out: list = []
    for r in todo:
        if out and out[-1][0] == r.exec:
            out[-1][1].append(r)
        else:
            out.append((r.exec, [r]))
    return out


def measure(a, log=print) -> dict:
    from tools.gate import hw as hwmod, runner as grunner, server as gserver
    from tools.gate.client import client as gclient, prompts as gprompts
    from tools.release_report import plan as planmod
    from tools.measure.prompts import Prose
    st = storemod.Store(a.out)
    rows, groups = rowsmod.load()
    want = wanted(a, rows)
    base = a.url.rstrip("/")
    h = wait_idle(base, a.wait_idle)
    why = bare_refusal(h)
    if why:
        raise Refused(why)
    prof = hwmod.profile(a.hw, python=a.python)
    env, argv = {}, []
    pid = local_pid(base)
    if pid:
        env, argv = gserver.environ_of(pid), gserver.argv_of(pid)
    knobs = gserver.knobs(h, env, argv)
    if not pid:
        log("[measure] note: not a local server; the knobs are /health's only (environment and flags unknown)")
    key = server_key(h, knobs, prof, a)
    present = {r.id: st.primary(r.id, key["build"], key["pack"], key["box"], key["knobs_hash"]) for r in want}
    other = {r.id: st.find(r.id, key["build"], key["pack"], key["box"]) for r in want if not present[r.id]}
    todo = [r for r in want if a.force or not present[r.id]]
    log(f"[measure] server {h.get('version')} · pack {key['pack']} · box {key['box']} · knobs {key['knobs_hash']} "
        f"(lanes {knobs.get('lanes')}, max_running {knobs.get('max_running')}, max_len {knobs.get('max_len')}, "
        f"drafter {knobs.get('drafter')}, ceiling/peak {knobs.get('ceiling_gb')}/{knobs.get('peak_gb')} GB)")
    for r in want:
        p = present[r.id]
        if p and not a.force:
            log(f"[measure] {r.id}: has a record ({p['source'].get('tool')}, {p['source'].get('date')}, "
                f"{p['_path']}): not measured")
        elif other.get(r.id):
            log(f"[measure] {r.id}: records only under other knobs ({', '.join(x['knobs_hash'] for x in other[r.id])})")
    moved = [r.id for r in todo if other.get(r.id) and not a.force]
    if moved and not a.new_knobs:
        raise SystemExit(f"[measure] refused: {', '.join(moved[:6])} have records under other knobs; this server's "
                         f"knobs ({key['knobs_hash']}) differ from theirs. Check the server's settings, or pass "
                         "--new-knobs to measure them for these knobs")
    if not todo:
        log("[measure] nothing to measure: every row asked has a record")
        return {"measured": 0}
    check_order(todo, rows, groups, a)
    if a.dry:
        log("[measure] would measure: " + ", ".join(r.id for r in todo))
        return {"measured": 0, "todo": [r.id for r in todo]}
    tok = tokenizer_for(a, h)
    proof_kind = "running" if is_fake(h) else "decoding"
    card = model_card(base, h)
    cl = cmod.ReportClient(base, h.get("model") or "TandemLLM", tok=tok)
    gcl = gclient(base, h.get("model") or "TandemLLM", tok)
    prose = Prose(a.prose, tok)
    pl = planmod.make(h, card, prof, max_prompt=a.max_prompt)
    variant = a.variant or ("spec" if pl.drafter else "plain")
    pr = probe(cl, prose)
    packs = grunner.load_packs()
    pack = dict(packs.get(a.gate_pack or "", {}) or {}, id=a.gate_pack or key["pack"])
    if is_fake(h):
        pack.setdefault("proof", "running")
    a.proof_kind = proof_kind
    a.model_id = ((card.get("tandem") or {}).get("model") or (h.get("weights") or {}).get("name") or "model")
    a.no_drafter = not pl.drafter
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    mx = execs.MCtx(base, cl, gcl, h, knobs, pl, prose, gprompts(), prof, pack, out, log, variant, a, proof_kind)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.monotonic()
    got: dict = {}
    timing: dict = {}
    group = sorted({r.server for r in todo if r.server}) or ["-"]
    log(f"[measure] measuring {len(todo)} rows on group {', '.join(group)}: " + ", ".join(r.id for r in todo))
    with BoxSampler() as sampler:
        for ex, rs in batches(todo):
            ts = time.monotonic()
            pc.wait_idle(base, 300)
            res = execs.EXECS[ex](mx, rs)
            for r in rs:
                p = res[r.id]
                bad = execs.proof_problem(r.id, r.n, p, proof_kind)
                if bad:
                    p.update(valid=False, why=bad)
                got[r.id] = p
                log(f"[measure] {r.id}: " + ("skipped: " + p["skipped"] if p.get("skipped") else
                                            ("ok " if p["valid"] else "INVALID ") + str(p.get("why") or p.get("values"))[:200]))
            timing[ex] = round((time.monotonic() - ts) / 60, 2)
        h_end = cmod.health(base)
    if not set(cl.calls) <= {cmod.ENDPOINT, "/v1/tokenize"}:
        raise SystemExit(f"[measure] internal error: a call outside the raw path: {sorted(set(cl.calls))}")
    if a.engine_log and os.path.exists(os.path.expanduser(a.engine_log)):
        from tools.measure import logjoin
        log_ids = logjoin.parse(Path(os.path.expanduser(a.engine_log)).read_text(errors="replace"))
        for p in got.values():
            logjoin.enrich(p["rows"], log_ids)
    from tools.measure import scrub
    w = h.get("weights") or {}
    priv = scrub.private_terms()
    priv["allow"] = [prof.fp.name or "", "DGX Spark" if prof.id == "gb10" else ""]
    run_id = f"{key['build']}-{key['pack']}-{'+'.join(group)}-{stamp}"
    run = {"id": run_id, "how": "measured", "tool": f"tools/measure {VERSION}", "tool_commit": tool_commit(),
           "commit": a.commit, "started": started, "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "minutes": round((time.monotonic() - t0) / 60, 1), "exec_minutes": timing, "group": group,
           "variant": variant, "version": h.get("version"), "model": label_of(card, h), "weights": w.get("name"),
           "weights_repo": w.get("hf_repo"), "weights_revision": w.get("hf_revision"), "format": w.get("format"),
           "drafters": w.get("drafters"), "corrector": w.get("corrector"), "hardware": hardware_label(prof),
           "gpu": prof.fp.as_dict(), "hw_profile": prof.id, "knobs": knobs, "key": key,
           "health_start": pc.health_record(h), "health_end": pc.health_record(h_end),
           "served_at_start": (h.get("inflight") or {}).get("served"), "uptime_at_start_s": h.get("uptime_s"),
           "box": sampler.summary(), "probe": pr, "prose": prose.describe(), "plan": pl.as_dict(),
           "prompts": {"set": mx.pj.get("version"), "sha256": mx.pj.get("_sha256"),
                       "filler_sha256": mx.pj.get("_filler_sha256")},
           "proof_kind": proof_kind, "mode": "url", "private": priv}
    recs = {}
    for r in todo:
        p = got[r.id]
        recs[r.id] = {"row": r.id, **key, "knobs": storemod.limit_knobs(knobs), "version": h.get("version"),
                      "valid": p["valid"], "why": p.get("why"), "skipped": p.get("skipped"), "values": p["values"],
                      "rows": p["rows"], "raw": p["raw"],
                      "source": {"tool": f"tools/measure {VERSION}", "how": "measured", "date": started[:10],
                                 "path": f"runs/{run_id}.json", "commit": a.commit}}
    path = st.add_run(run, recs, force=a.force)
    n_bad = sum(1 for x in recs.values() if not x["valid"] and not x.get("skipped"))
    log(f"[measure] {len(recs)} rows ({n_bad} invalid) into {path} in {run['minutes']} min")
    return {"measured": len(recs), "invalid": n_bad, "path": str(path)}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.measure run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True, help="the record store (runs/ and index.json)")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--rows", help="row ids and globs, comma separated (python -m tools.measure rows)")
    sel.add_argument("--view", choices=("report", "card", "gate"), help="every row this view uses")
    ap.add_argument("--build", default=None, help="the build of the key (default: /health's version, as 0.15.2)")
    ap.add_argument("--pack", default=None, help="the pack of the key (default: /health's weights name)")
    ap.add_argument("--commit", default=None, help="the served commit, recorded beside")
    ap.add_argument("--force", action="store_true", help="measure rows that have a record again")
    ap.add_argument("--dry", action="store_true", help="say what would be measured, measure nothing")
    ap.add_argument("--new-knobs", action="store_true", help="measure rows that have records only under other knobs")
    ap.add_argument("--any-group", action="store_true", help="rows of several server groups on this one server")
    ap.add_argument("--any-order", action="store_true", help="a warm row without the rows it follows")
    ap.add_argument("--variant", choices=("spec", "plain", "wide"), default=None)
    ap.add_argument("--gate-pack", default=None, help="the gate pack (tools/gate/packs.yaml) for the gate's rows")
    ap.add_argument("--prose", default="repo", help="repo, wikitext2 (local cache) or a text file")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-prompt", type=int, default=None)
    ap.add_argument("--wait-idle", type=float, default=0.0)
    ap.add_argument("--engine-log", default=None)
    ap.add_argument("--hw", default="auto")
    ap.add_argument("--python", default=sys.executable)
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        r = measure(a)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 1 if r.get("invalid") else 0
