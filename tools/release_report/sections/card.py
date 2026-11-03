"""g: the repository's own card tool, tools/parallel_card.py, against the same server: once with its
default arguments and once with `--fixed-prompts` (its levels only: the flag changes nothing else).
Its levels are capped at the server's slot limit. Its raw files go to the report's raw folder; the rows
are the tool's own (wall tok/s, TTFT and decode at 1/2/4/8 at once, decode by workload, prefill), and a
level the tool calls invalid stays invalid here."""

from __future__ import annotations

import json
import subprocess
import sys

from tools.release_report.sections import Ctx, row

REPO_TOOL = "tools/parallel_card.py"


def run_card(ctx: Ctx, tag: str, extra: list[str]) -> dict:
    from tools.release_report.prompts import REPO
    raw = ctx.out / "raw" / f"card-{tag}.json"
    env_file = ctx.out / "raw" / "card-env.txt"
    env_file.write_text("# the server's settings are read from /health; no setup file is passed\n")
    w = (ctx.health.get("weights") or {})
    levels = ",".join(str(n) for n in (1, 2, 4, 8) if n <= ctx.plan.lanes)
    cmd = [sys.executable, REPO_TOOL, "--base", ctx.base, "--model", ctx.cl.model,
           "--model-id", str(ctx.args.model_id), "--profile", str(w.get("name") or "unknown"),
           "--env-name", "release-report", "--env-file", str(env_file),
           "--commit", str(ctx.health.get("version")), "--out", str(raw), "--levels", levels, *extra]
    if w.get("weights"):
        cmd += ["--tokenizer", w["weights"]]
    if ctx.args.tokenizer:
        cmd += ["--tokenizer", ctx.args.tokenizer]
    with open(ctx.out / "raw" / f"card-{tag}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    if not raw.exists():
        return {"rc": rc, "error": (ctx.out / "raw" / f"card-{tag}.log").read_text()[-400:]}
    return {"rc": rc, "raw": json.loads(raw.read_text())}


def measure(ctx: Ctx) -> list[dict]:
    out = []
    for tag, extra in (("default", []), ("fixed", ["--fixed-prompts", "--parts", "levels"])):
        got = run_card(ctx, tag, extra)
        if "raw" not in got:
            out.append(row("g", tag, {}, {"args": tag}, valid=False, why=f"the card tool failed: {got['error']}"))
            continue
        rw = got["raw"]
        rr = rw.get("row") or {}
        for n, lv in (rr.get("levels") or {}).items():
            raw_lv = (rw.get("levels") or {}).get(str(n)) or {}
            ok = bool(raw_lv.get("valid"))
            out.append(row("g", f"{tag}/levels/{n}", lv, {"args": tag, "rounds": rr.get("rounds")},
                           valid=ok, why=None if ok else "the card tool found no proof of N at once"))
        for w_, v in (rr.get("decode") or {}).items():
            out.append(row("g", f"{tag}/decode/{w_}", {"decode_tok_s": v}, {"args": tag}))
        for d, v in (rr.get("prefill") or {}).items():
            out.append(row("g", f"{tag}/prefill/{d}", v, {"args": tag}))
        ctx.log(f"[g] {tag}: rc {got['rc']}, levels {json.dumps(rr.get('levels'))}")
    return out
