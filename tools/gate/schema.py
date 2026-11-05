"""The report's schema, checked with the standard library (the CI check installs nothing else).

A report (gate-report.json):

    kind         "tandem-gate-report"
    schema       1
    tool         {"version", "code_sha256"}: the gate's version and a hash of tools/gate/
    created_at   ISO time, UTC
    level        none | quick | quick+long | full: what ran; `required` is what the diff needs
    suites       the suites that ran
    kernels      the tests/gpu files the kernels suite ran
    hardware     {"profile", "name", "memory_gib", "cc", "driver", "cuda", "torch"}
    pack         {"id", "profile", "weights_sha256"}
    base, head   {"ref", "sha", "tree_sha256"}
    knobs        {"base": {...}, "head": {...}}: every limit knob, read from each running server
    planted      null, or what a dry run planted in the head (the CI check refuses one)
    rows         [{"id", "suite", "metric", "verdict", "why", ...}]
    reasons      {row id: written reason} for WORSE and CHANGED_TEXT rows
    verdict      PASS | WORSE | CHANGED_TEXT | FAIL (tools/gate/verdict.py `overall`)
    wall_s       the run's wall time
"""

from __future__ import annotations

import re

from tools.gate import LEVELS, SCHEMA, SKIPPED, SUITES, VERDICTS
from tools.gate.verdict import overall

KIND = "tandem-gate-report"
SHA = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def validate(rep) -> list[str]:
    """Every way `rep` is not a well-formed report; empty when it is one."""
    if not isinstance(rep, dict):
        return ["the report is not a JSON object"]
    errs: list[str] = []

    def need(obj, key, typ, where):
        v = obj.get(key) if isinstance(obj, dict) else None
        if not isinstance(v, typ):
            errs.append(f"{where}{key}: missing or not {getattr(typ, '__name__', typ)}")
            return None
        return v

    if rep.get("kind") != KIND:
        errs.append(f"kind: not {KIND!r}")
    if rep.get("schema") != SCHEMA:
        errs.append(f"schema: {rep.get('schema')!r}, this check reads {SCHEMA}")
    tool = need(rep, "tool", dict, "")
    if tool is not None:
        need(tool, "version", str, "tool.")
    need(rep, "created_at", str, "")
    level = rep.get("level")
    if level not in LEVELS:
        errs.append(f"level: {level!r} is not one of {', '.join(LEVELS)}")
    suites = need(rep, "suites", list, "") or []
    for s in suites:
        if s not in SUITES:
            errs.append(f"suites: unknown suite {s!r}")
    need(rep, "kernels", list, "")
    hw = need(rep, "hardware", dict, "")
    if hw is not None:
        need(hw, "profile", str, "hardware.")
        need(hw, "name", str, "hardware.")
    pack = need(rep, "pack", dict, "")
    if pack is not None:
        need(pack, "id", str, "pack.")
    for side in ("base", "head"):
        s = need(rep, side, dict, "")
        if s is not None and not SHA.match(str(s.get("sha") or "")):
            errs.append(f"{side}.sha: not a 40-character commit hash")
        if s is not None and not HEX64.match(str(s.get("tree_sha256") or "")):
            errs.append(f"{side}.tree_sha256: not a sha256")
    knobs = need(rep, "knobs", dict, "")
    if knobs is not None and suites and suites != ["kernels"]:
        for side in ("base", "head"):
            if not isinstance(knobs.get(side), dict) or not knobs.get(side):
                errs.append(f"knobs.{side}: missing (every limit knob is recorded)")
    if "planted" not in rep:
        errs.append("planted: missing (null for a real run)")
    rows = need(rep, "rows", list, "") or []
    ids = set()
    for i, r in enumerate(rows):
        where = f"rows[{i}]"
        if not isinstance(r, dict):
            errs.append(f"{where}: not an object")
            continue
        for k in ("id", "suite", "metric", "verdict", "why"):
            if not isinstance(r.get(k), str):
                errs.append(f"{where}.{k}: missing or not a string")
        if r.get("verdict") not in VERDICTS:
            errs.append(f"{where}.verdict: {r.get('verdict')!r} is not one of {', '.join(VERDICTS)}")
        if r.get("suite") not in SUITES:
            errs.append(f"{where}.suite: unknown suite {r.get('suite')!r}")
        if r.get("verdict") == SKIPPED and not (r.get("why") or "").strip():
            errs.append(f"{where}: SKIPPED without a reason")
        if r.get("id") in ids:
            errs.append(f"{where}.id: {r.get('id')!r} twice")
        ids.add(r.get("id"))
    reasons = rep.get("reasons", {})
    if not isinstance(reasons, dict):
        errs.append("reasons: not an object")
    if rep.get("verdict") not in ("PASS", "WORSE", "CHANGED_TEXT", "FAIL"):
        errs.append(f"verdict: {rep.get('verdict')!r}")
    elif not errs and rep["verdict"] != overall(rows):
        errs.append(f"verdict: {rep['verdict']} but the rows say {overall(rows)}")
    return errs
