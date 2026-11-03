"""`python -m tools.release_report run`: the report's measurements against a running server (`--url`,
the default http://127.0.0.1:8000), or a server it starts itself (`--ref TAG --pack ID`, launch mode).

Before anything is measured, the run refuses a server that:
  - does not answer /health with status ok;
  - has the response cache on (/health `cache.response_cache`): an identical greedy request could be
    answered from it rather than by the engine;
  - is busy: requests running, waiting or decoding (`--wait-idle S` waits that long first);
  - answers a probe request without `X-Tandem: raw` (the chat layer in front of /v1/completions).
A loaded conductor is recorded, not refused: its router returns a raw request, and any request to
/v1/completions, unchanged; every response's `X-Tandem: raw` is checked all the same.
It then prints the plan (sized from /health and the GPU) with a time estimate from the probe's rates.
It never changes a server setting and calls no admin route. Raw files are written under `--out` only;
data.json there gathers every run's rows, so a report can be measured in parts (`--only`)."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from tools import parallel_card as pc
from tools.release_report import RUN_ORDER, SCHEMA, SECTIONS, VERSION
from tools.release_report import box as boxmod
from tools.release_report import client as cmod
from tools.release_report import logjoin, plan as planmod, scrub
from tools.release_report.prompts import Prose, sized
from tools.release_report.sections import Ctx

REPO = Path(__file__).resolve().parent.parent.parent


class Refused(SystemExit):
    def __init__(self, why: str):
        super().__init__(f"[report] refused: {why}")
        self.why = why


def bare_refusal(h: dict) -> str | None:
    """Why this server's raw requests would not reach the bare engine alone, or None."""
    if h.get("status") != "ok":
        return f"/health says status {h.get('status')!r}" + (f" ({h['error']})" if h.get("error") else "")
    if (h.get("cache") or {}).get("response_cache"):
        return ("the response cache is on: an identical greedy request could be answered from it. Start the "
                "server with RESPONSE_CACHE=off, or use launch mode")
    return None


def model_card(base: str, h: dict) -> dict:
    d = cmod.get_json(base.rstrip("/") + "/v1/models")
    for m in d.get("data") or []:
        if m.get("id") == h.get("model") or (m.get("tandem") or {}).get("kind") == "engine":
            return m
    return {}


def tokenizer_for(a, h: dict):
    path = a.tokenizer or (h.get("weights") or {}).get("weights")
    if path and os.path.isdir(path):
        try:
            return pc.load_tokenizer(path)
        except Exception as exc:                                     # noqa: BLE001
            raise Refused(f"the model's tokenizer did not load from the weights folder ({type(exc).__name__}); "
                          "run with the engine's Python, or pass --tokenizer DIR")
    if is_fake(h):
        return None
    raise Refused("the model's tokenizer is needed to render prompts from its chat template, and the weights "
                  "folder /health names is not on this machine: pass --tokenizer DIR")


def is_fake(h: dict) -> bool:
    """A server that names no weights folder: the fake engine (server/fake_engine.py), whose prompts are
    plain text."""
    return not (h.get("weights") or {}).get("weights")


def wait_idle(base: str, wait_s: float) -> dict:
    t0 = time.monotonic()
    while True:
        h = cmod.health(base)
        why = cmod.busy(h)
        if not why:
            return h
        if time.monotonic() - t0 >= wait_s:
            raise Refused(f"the server is busy ({why}); a report measures an idle server. Try again when it is "
                          "idle, or pass --wait-idle SECONDS")
        time.sleep(2.0)


def probe(cl, prose) -> dict:
    """One ~2k request, 64 tokens: the raw check and the rates the estimate uses."""
    p, _ = sized(cl, prose, 2048, "\n\nSummarise the text above.", tag="probe")
    r = cl.stream_rec(p, 64, seed=1)
    if r.get("error"):
        raise Refused(f"the probe request failed: {r['error']}")
    if not r.get("raw"):
        raise Refused(f"the probe was not served raw ({r['raw_why']}): the chat layer is in front of "
                      "/v1/completions on this server")
    return {"prefill_tok_s": r.get("prefill_tok_s") or 1000.0, "decode_tok_s": r.get("decode_tok_s") or 30.0,
            "ttft_s": r.get("ttft_s"), "prompt_tokens": r.get("prompt_tokens")}


def sections_of(only: list[str] | None, hle: bool) -> list[str]:
    want = only or [s for s in RUN_ORDER if s != "j" or hle]
    want = ["a" if s == "b" else s for s in want]
    bad = [s for s in want if s not in SECTIONS or s == "k"]
    if bad:
        raise SystemExit(f"[report] unknown section(s) {bad}: measured sections are {', '.join(RUN_ORDER)} "
                         "(b comes with a)")
    seen, out = set(), []
    for s in want:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def print_plan(pl, est: dict, sections: list[str], h: dict, hw, log) -> None:
    log(f"[report] server {h.get('version')} · {h.get('model')} · weights {(h.get('weights') or {}).get('name')}")
    log(f"[report] GPU {hw.fp.name} ({hw.id}), {hw.fp.memory_gib} GiB; slot limit {pl.lanes}; window "
        f"{pl.window:,} ({pl.window_why}); drafter {'yes' if pl.drafter else 'no'}")
    log(f"[report] a/b lengths: {', '.join(planmod.kfmt(n) for n in pl.lengths)} "
        f"(runs {', '.join(str(pl.runs[n]) for n in pl.lengths)})")
    log(f"[report] c: 2k x {pl.conc_2k} ({pl.conc_rounds} rounds); 32k x {pl.conc_32k}; d: chats {pl.chats}; "
        f"e: load {pl.load_prompt}, return {pl.return_prompt}; f: {pl.long_prompt} {pl.long_tasks} x{pl.long_runs}; "
        f"h: needles {pl.needles}; i: {pl.warm}")
    for k, why in pl.skipped.items():
        log(f"[report] skipped {k}: {why}")
    log(f"[report] sections {', '.join(sections)}; estimate (minutes): "
        + ", ".join(f"{k} {v}" for k, v in est.items()))
    if not pl.quick:
        log("[report] --quick runs fewer lengths and one round each, about 20 minutes on a fast box")


def merge(out: Path, rows: list[dict], run: dict, sections: list[str], variant: str, new: bool) -> dict:
    """data.json with this run's rows in place of the earlier rows of the same sections (and variant)."""
    p = out / "data.json"
    data = json.loads(p.read_text()) if p.exists() and not new else {}
    if data and (data.get("version") != run["version"] or data.get("weights") != run["weights"]):
        raise SystemExit(f"[report] {p} holds {data.get('version')} / {data.get('weights')}, this server is "
                         f"{run['version']} / {run['weights']}: use another --out, or --new to start over")

    states_i = {r["key"].split("/")[0] for r in rows if r["section"] == "i"} | {"skip"}

    def replaced(r):
        if r["section"] not in sections:
            return False
        if r["section"] == "h" and variant == "plain" and sections == ["h"]:
            return r["key"].startswith("plain/")             # a plain arm adds its texts only
        if r["section"] == "h" and variant == "spec":
            return not r["key"].startswith("plain/")
        if r["section"] == "i":                              # fresh and warm rows come from two runs
            return r["key"].split("/")[0] in states_i
        return True
    data.update({"kind": "tandem-benchmark-report", "schema": SCHEMA, "tool": VERSION,
                 "version": run["version"], "model": run["model"], "weights": run["weights"],
                 "hardware": run["hardware"]})
    data["rows"] = [r for r in data.get("rows") or [] if not replaced(r)] + rows
    data.setdefault("runs", []).append(run)
    private = data.get("private") or {"hosts": [], "users": [], "homes": [], "allow": []}
    for k, v in run.pop("_private").items():
        private[k] = sorted(set(private.get(k) or []) | set(v))
    data["private"] = private
    p.write_text(json.dumps(data, indent=1, default=str) + "\n")
    return data


def measure(a, base: str, variant: str | None = None, launch: dict | None = None, log=print) -> dict:
    """Every check, the plan, and the sections against the server at `base`. Returns data.json."""
    from tools.gate import hw as hwmod
    out = Path(a.out).expanduser()
    h = wait_idle(base, a.wait_idle)
    why = bare_refusal(h)
    if why:
        raise Refused(why)
    card = model_card(base, h)
    # the conductor's router hands a raw request (and any request to /v1/completions) back unchanged;
    # every response is still checked for `X-Tandem: raw`, the probe first
    loaded = cmod.get_json(base.rstrip("/") + "/v1/models").get("loaded") or {}
    prof = hwmod.profile(a.hw, python=a.python)
    tok = tokenizer_for(a, h)
    # the proof of N at once: N slots in the decode set; the fake engine's writer has no decode phase,
    # so there it is N running (tools/gate/packs.yaml `proof`)
    a.proof_kind = "running" if is_fake(h) else "decoding"
    cl = cmod.ReportClient(base, h.get("model") or "TandemLLM", tok=tok)
    prose = Prose(a.prose, tok)
    pl = planmod.make(h, card, prof, quick=a.quick, max_prompt=a.max_prompt)
    explicit = variant or a.variant
    a.no_drafter = not explicit and not pl.drafter         # a model that runs no drafter at all
    variant = explicit or ("spec" if pl.drafter else "plain")
    sections = ["h"] if explicit == "plain" else sections_of(a.only, a.hle)
    pr = probe(cl, prose)
    est = planmod.estimate(pl, sections, pr["prefill_tok_s"], pr["decode_tok_s"])
    print_plan(pl, est, sections, h, prof, log)
    if a.plan_only:
        return {"plan": pl.as_dict(), "estimate": est, "probe": pr}
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    a.model_id = ((card.get("tandem") or {}).get("model") or (h.get("weights") or {}).get("name") or "model")
    ctx = Ctx(base, cl, h, pl, prose, out, log, variant, a)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows: list = []
    t0 = time.monotonic()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    timings = {}
    from importlib import import_module
    with boxmod.BoxSampler() as sampler:
        for s in sections:
            mod = import_module(f"tools.release_report.sections.{SECTIONS[s][1]}")
            ts = time.monotonic()
            pc.wait_idle(base, 300)
            got = mod.measure(ctx)
            timings[s] = round((time.monotonic() - ts) / 60, 1)
            log(f"[report] section {s} done in {timings[s]} min: {len(got)} rows, "
                f"{sum(1 for r in got if not r.get('valid') and not r.get('skipped'))} invalid")
            rows += got
            (out / "raw" / f"{s}-{variant}-{stamp}.json").write_text(json.dumps(got, indent=1, default=str))
        h_end = cmod.health(base)
    if not set(cl.calls) <= {cmod.ENDPOINT, "/v1/tokenize"}:
        raise SystemExit(f"[report] internal error: a call outside the raw path: {sorted(set(cl.calls))}")
    elog = (launch or {}).get("engine_log") or a.engine_log
    joined = 0
    if elog and os.path.exists(os.path.expanduser(elog)):
        joined = logjoin.enrich(rows, logjoin.parse(Path(os.path.expanduser(elog)).read_text(errors="replace")))
    from tools.gate.server import knobs as gate_knobs
    env, argv = (launch or {}).get("env") or {}, (launch or {}).get("argv") or []
    w = h.get("weights") or {}
    priv = scrub.private_terms()
    priv["allow"] = [prof.fp.name or "", "DGX Spark" if prof.id == "gb10" else ""]
    run = {"started": started, "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "minutes": round((time.monotonic() - t0) / 60, 1), "sections": sections, "variant": variant,
           "section_minutes": timings, "version": h.get("version"), "model": label_of(card, h),
           "weights": w.get("name"), "weights_repo": w.get("hf_repo"), "weights_revision": w.get("hf_revision"),
           "format": w.get("format"), "drafters": w.get("drafters"), "corrector": w.get("corrector"),
           "hardware": hardware_label(prof), "gpu": prof.fp.as_dict(), "hw_profile": prof.id,
           "knobs": gate_knobs(h, env, argv), "health_end": pc.health_record(h_end),
           "served_at_start": (h.get("inflight") or {}).get("served"), "uptime_at_start_s": h.get("uptime_s"),
           "box": sampler.summary(), "plan": pl.as_dict(), "estimate": est, "probe": pr,
           "prose": prose.describe(), "launch": {k: v for k, v in (launch or {}).items()
                                                  if k in ("ref", "sha", "pack", "port")} or None,
           "mode": "launch" if launch else "url", "engine_log_joined": joined,
           "capabilities": pl.capabilities, "conductor": {"health_mode": h.get("mode"),
                                                          "loaded": loaded}, "_private": priv}
    data = merge(out, rows, run, sections, variant, a.new)
    log(f"[report] {len(rows)} rows into {out / 'data.json'} ({run['minutes']} min)")
    return data


def label_of(card: dict, h: dict) -> str:
    """The model's name for the page: the card's label, unless it is only the served name (as
    "TandemLLM"), then the card's model id, then the weights' name."""
    t = card.get("tandem") or {}
    served = {str(h.get("model") or ""), "TandemLLM"}
    for x in (t.get("label"), t.get("model"), (h.get("weights") or {}).get("hf_repo"),
              (h.get("weights") or {}).get("name")):
        if x and str(x) not in served:
            return re.sub(r"[-_](exl3|gguf|nvfp4|fp8|awq|gptq|bf16)$", "", str(x).split("/")[-1], flags=re.I)
    return str(h.get("model") or "model")


def hardware_label(prof) -> str:
    fp = prof.fp
    if prof.id == "gb10":
        return "one DGX Spark (GB10)"
    if prof.id == "fake":
        return "no GPU (the fake engine)"
    n = f"{fp.count} x " if fp.count and fp.count > 1 else "one "
    return f"{n}{fp.name}" + (f" ({fp.memory_gib:g} GiB)" if fp.memory_gib else "")


def launch(a, log=print) -> None:
    """Launch mode: a `git archive` of --ref, its own config copy and state, a fresh server on a free port
    of 8100 and up with the conductor, response cache and chat service off (tools/gate/server.py), the
    measurements against it through the same path as --url, then the server stopped; with section h and a
    drafter, a second server without speculation for the plain texts."""
    from tools.gate import hw as hwmod
    from tools.gate import runner as grunner, server as gserver
    packs = grunner.load_packs()
    if a.pack not in packs:
        raise SystemExit(f"[report] no pack {a.pack!r}: one of {', '.join(packs)}")
    pack = dict(packs[a.pack], id=a.pack)
    prof = hwmod.profile(a.hw, python=a.python)
    sha = gserver.git(str(REPO), "rev-parse", f"{a.ref}^{{commit}}")
    out = Path(a.out).expanduser()
    work = out / "launch" / f"tree-{sha[:12]}"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    gserver.make_tree(str(REPO), sha, work)
    site = grunner.site_of(a)
    lch = gserver.launcher(pack, prof, a.python, site)
    if hasattr(lch, "prepare"):
        lch.prepare(work)
    sections = sections_of(a.only, a.hle)
    arms = [("main", sections)]
    if "h" in sections and pack.get("plain_env"):
        arms.append(("plain", ["h"]))
    with grunner.box_lock(a.lock):
        for variant, secs in arms:
            env = grunner.arm_env(prof, pack, "head", variant)
            port = gserver.free_port(a.port, set())
            arm = gserver.Arm("head", variant, work, port, out / "launch" / f"arm-{variant}", env, log)
            log(f"[report] launch: {a.ref} ({sha[:12]}) {variant} on :{port}")
            k = lch.start(arm)
            try:
                info = {"ref": a.ref, "sha": sha, "pack": a.pack, "port": port,
                        "engine_log": str(work / "logs" / "engine.log"),
                        "env": gserver.environ_of(arm.pid) if arm.pid else {},
                        "argv": gserver.argv_of(arm.pid) if arm.pid else []}
                sub = argparse_copy(a, only=secs)
                measure(sub, arm.base, variant="plain" if variant == "plain" else None, launch=info, log=log)
                a.new = False
            finally:
                lch.stop(arm)
            del k


def argparse_copy(a, **kw):
    import copy
    b = copy.copy(a)
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m tools.release_report run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the running server (default the local one)")
    ap.add_argument("--out", required=True, help="the report's folder: data.json and raw/")
    ap.add_argument("--only", nargs="+", default=None, help=f"these sections ({', '.join(RUN_ORDER)}; b comes with a)")
    ap.add_argument("--quick", action="store_true", help="fewer lengths and one round each (about 20 min)")
    ap.add_argument("--plan", dest="plan_only", action="store_true", help="print the plan and the estimate, measure nothing")
    ap.add_argument("--hle", action="store_true", help="section j, the multiple-choice set (about 2.5 h)")
    ap.add_argument("--hle-data", default=None, help="the multiple-choice items (JSON lines, a local file)")
    ap.add_argument("--hle-budget", type=int, default=131072, help="tokens an item may write")
    ap.add_argument("--prose", default="repo", help="repo, wikitext2 (local Hugging Face cache) or a text file")
    ap.add_argument("--tokenizer", default=None, help="the model's tokenizer folder (default the weights folder /health names)")
    ap.add_argument("--max-prompt", type=int, default=None, help="cap every prompt at this many tokens")
    ap.add_argument("--variant", choices=("spec", "plain"), default=None,
                    help="what the server runs: plain = speculation off (only section h is measured)")
    ap.add_argument("--wait-idle", type=float, default=0.0, help="wait this long for a busy server to go idle")
    ap.add_argument("--engine-log", default=None, help="the server's engine log (a local file), for each request's slot")
    ap.add_argument("--hw", default="auto")
    ap.add_argument("--python", default=sys.executable, help="the engine's Python, for the GPU's torch facts")
    ap.add_argument("--new", action="store_true", help="start data.json over")
    ap.add_argument("--ref", default=None, help="launch mode: start a fresh server of this tag or commit")
    ap.add_argument("--pack", default=None, help="launch mode: the weight pack (tools/gate/packs.yaml)")
    ap.add_argument("--site", default=None, help="launch mode: what this box adds beside a checkout (tools/gate)")
    ap.add_argument("--port", type=int, default=8100, help="launch mode: the first port to try (8100 and up)")
    ap.add_argument("--lock", default=None, help="launch mode: a lock file to hold while the servers run")
    a = ap.parse_args(argv)
    try:
        if a.ref:
            if not a.pack:
                ap.error("--ref needs --pack")
            launch(a)
        else:
            measure(a, a.url)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


def plan_main(argv=None) -> int:
    return main(list(argv or []) + ["--plan", "--out", os.devnull])
