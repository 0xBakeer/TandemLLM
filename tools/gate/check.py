"""The `gate` status check (.github/workflows/gate.yml): does the pull request carry a report that
covers its diff, measured on its head, with nothing failing and a written reason for every WORSE
and CHANGED_TEXT row?

    python -m tools.gate check --base-sha SHA --head-sha SHA --texts texts.json [--summary FILE]

`texts.json` is {"body": "...", "comments": ["...", ...]}, the pull request's body and its comments
oldest first; the last marked report wins. The check reads git (the checkout needs the full history)
and nothing else, needs no secret, and writes nothing to the pull request.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from tools.gate import (BLOCKING, LEVEL_SUITES, MIN_REASON, NEEDS_REASON, SKIPPED, VERSION,
                        level_index)
from tools.gate import pathmap
from tools.gate.render import extract
from tools.gate.schema import validate
from tools.gate.verdict import unexplained

HOW = ("run `python -m tools.gate run` on your GPU and paste gate-report.md into the pull request "
       "(docs/quality-gate.md)")


class Git:
    """The two questions the check asks git (a fake in the tests)."""

    def __init__(self, repo: str = "."):
        self.repo = repo

    def changed(self, a: str, b: str) -> list[str] | None:
        try:
            return pathmap.changed_between(a, b, self.repo)
        except subprocess.CalledProcessError:
            return None

    def is_ancestor(self, a: str, b: str) -> bool:
        r = subprocess.run(["git", "-C", self.repo, "merge-base", "--is-ancestor", a, b],
                           capture_output=True)
        return r.returncode == 0


def check(files: list[str], base_sha: str, head_sha: str, texts: list[str], git,
          mapping: dict | None = None) -> dict:
    need = pathmap.required(files, mapping)
    out = {"ok": False, "level": need["level"], "suites": need["suites"],
           "kernels": need["kernels"], "problems": [], "notes": [], "report": None}
    bad = out["problems"]
    if need["level"] == "none":
        out["ok"] = True
        out["notes"].append("the diff needs no GPU report (level none)")
        return out
    rep, err = extract(texts)
    if err:
        bad.append(err)
        return out
    if rep is None:
        bad.append(f"no gate report found (level {need['level']} needed): {HOW}")
        return out
    errs = validate(rep)
    if errs:
        bad += [f"report schema: {e}" for e in errs[:20]]
        return out
    out["report"] = {"verdict": rep["verdict"], "level": rep["level"],
                     "head": rep["head"]["sha"], "hardware": rep["hardware"].get("name"),
                     "pack": rep["pack"].get("id")}
    if str(rep["tool"]["version"]).split(".")[0] != VERSION.split(".")[0]:
        bad.append(f"the report's tool version {rep['tool']['version']} is not {VERSION}'s major "
                   "version: run the gate of this pull request's tree")
    if rep.get("planted"):
        bad.append("the report comes from a dry run with a planted change")
    if rep["head"]["sha"] != head_sha:
        moved = git.changed(rep["head"]["sha"], head_sha)
        if moved is None:
            bad.append(f"stale: the report measured {rep['head']['sha'][:12]}, which is not in "
                       f"this pull request's history (head {head_sha[:12]})")
        else:
            lv = pathmap.required(moved, mapping)["level"] if moved else "none"
            if lv == "none":
                out["notes"].append(f"the head moved since the report ({rep['head']['sha'][:12]} -> "
                                    f"{head_sha[:12]}) by {len(moved)} file(s) that need no report")
            else:
                bad.append(f"stale: the report measured {rep['head']['sha'][:12]}; since then "
                           f"{len(moved)} file(s) changed that need level {lv} "
                           f"({', '.join(moved[:5])}{' ...' if len(moved) > 5 else ''}): {HOW}")
    if not git.is_ancestor(rep["base"]["sha"], base_sha):
        bad.append(f"the report's base {rep['base']['sha'][:12]} is not a commit of the target "
                   f"branch ({base_sha[:12]})")
    if level_index(rep["level"]) < level_index(need["level"]):
        bad.append(f"the report ran level {rep['level']}; the diff needs {need['level']}")
    rows = rep["rows"]
    for s in LEVEL_SUITES[need["level"]]:
        if s not in rep["suites"]:
            bad.append(f"suite {s} did not run")
            continue
        if s == "kernels":
            continue
        ran = [r for r in rows if r["suite"] == s and r["verdict"] != SKIPPED]
        if not ran:
            bad.append(f"suite {s} measured nothing (every row SKIPPED)")
    missing = [t for t in need["kernels"] if t not in rep["kernels"]]
    if missing:
        bad.append(f"GPU tests the diff maps to did not run: {', '.join(missing)}")
    for r in rows:
        if r["verdict"] in BLOCKING:
            bad.append(f"{r['verdict']} {r['id']}: {r['why']}")
    for rid in unexplained(rows, rep.get("reasons") or {}):
        v = next(r["verdict"] for r in rows if r["id"] == rid)
        bad.append(f"{v} {rid} has no written reason (`python -m tools.gate explain "
                   f"gate-report.json {rid} \"...\"`, at least {MIN_REASON} characters)")
    explained = [r["id"] for r in rows if r["verdict"] in NEEDS_REASON and r["id"] not in
                 unexplained(rows, rep.get("reasons") or {})]
    if explained:
        out["notes"].append("explained: " + ", ".join(explained) + " (a maintainer reads the reasons)")
    out["ok"] = not bad
    return out


def summary_md(res: dict) -> str:
    head = "PASS" if res["ok"] else "FAIL"
    lines = [f"### gate: {head}", "", f"- needed: level `{res['level']}`"
             + (f", GPU tests {', '.join(res['kernels'])}" if res["kernels"] else "")]
    if res.get("report"):
        r = res["report"]
        lines.append(f"- report: {r['verdict']} at level {r['level']}, head {r['head'][:12]}, "
                     f"{r['hardware']}, pack {r['pack']}")
    lines += [f"- problem: {p}" for p in res["problems"]]
    lines += [f"- note: {n}" for n in res["notes"]]
    lines.append("- `gate-verified` is a maintainer's label after the re-run on the reference box")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate check", description=__doc__.split("\n")[0])
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--head-sha", required=True)
    ap.add_argument("--texts", required=True, help='{"body": ..., "comments": [...]} as JSON')
    ap.add_argument("--repo", default=".")
    ap.add_argument("--summary", default=None, help="append the result as Markdown here")
    a = ap.parse_args(argv)
    with open(a.texts) as fh:
        t = json.load(fh)
    texts = [t.get("body") or ""] + [c or "" for c in t.get("comments") or []]
    files = pathmap.changed_files(a.base_sha, a.head_sha, a.repo)
    res = check(files, a.base_sha, a.head_sha, texts, Git(a.repo))
    md = summary_md(res)
    print(md)
    if a.summary:
        with open(a.summary, "a") as fh:
            fh.write(md)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
