"""tools/gate: row verdicts, the noise rule, the report schema and its Markdown, on a CPU.

Run: python tests/test_gate_report.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.gate import MARKER, render, schema, verdict  # noqa: E402
from tools.gate.verdict import case_row, overall, speed_row, text_row  # noqa: E402

SHA_B, SHA_H = "a" * 40, "b" * 40


def report(rows, **kw) -> dict:
    rep = {"kind": "tandem-gate-report", "schema": 1, "tool": {"version": "1.0", "code_sha256": "0" * 64},
           "created_at": "2026-11-03T12:00:00Z", "level": "quick",
           "suites": ["lossless", "decode", "prefill", "long", "api", "kernels"], "kernels": [],
           "hardware": {"profile": "gb10", "name": "NVIDIA GB10", "memory_gib": 119.7, "cc": "12.1",
                        "driver": "580", "cuda": "13.0", "torch": "2.13"},
           "pack": {"id": "flashnext-3.05", "profile": "flashnext-3.05", "weights_sha256": "1" * 64},
           "base": {"ref": "origin/main", "sha": SHA_B, "tree_sha256": "2" * 64},
           "head": {"ref": "HEAD", "sha": SHA_H, "tree_sha256": "3" * 64},
           "knobs": {"base": {"lanes": 8}, "head": {"lanes": 8}}, "planted": None,
           "rows": rows, "reasons": {}, "verdict": overall(rows), "wall_s": 1500.0}
    rep.update(kw)
    return rep


def test_noise_rule_resolves_only_beyond_noise_with_disjoint_ranges():
    # the same numbers: not resolved
    r = speed_row("d", "decode", "c8", "tok/s", True, [150, 152, 151], [151, 150, 152])
    assert r["verdict"] == "PASS" and "not resolved" in r["why"], r
    # head lower, ranges disjoint, delta beyond noise: WORSE
    r = speed_row("d", "decode", "c8", "tok/s", True, [150, 152, 151], [120, 121, 119])
    assert r["verdict"] == "WORSE" and r["delta_pct"] < -19 and r["disjoint"], r
    # head lower but the ranges overlap: not resolved, PASS
    r = speed_row("d", "decode", "c8", "tok/s", True, [150, 160, 140], [139, 150, 145])
    assert r["verdict"] == "PASS", r
    # lower is better (first token): a higher head is worse
    r = speed_row("t", "prefill", "ttft", "s", False, [1.0, 1.01, 1.02], [1.5, 1.52, 1.51])
    assert r["verdict"] == "WORSE", r
    # resolved better passes and says so
    r = speed_row("d", "decode", "c8", "tok/s", True, [150, 152, 151], [170, 171, 172])
    assert r["verdict"] == "PASS" and r["why"] == "resolved better", r


def test_resolved_worse_under_the_floor_passes():
    r = speed_row("d", "decode", "c8", "tok/s", True, [100.0, 100.1, 100.2], [98.0, 98.1, 98.2],
                  min_effect_pct=3.0)
    assert r["verdict"] == "PASS" and "under the 3 %" in r["why"], r
    r = speed_row("d", "decode", "c8", "tok/s", True, [100.0, 100.1, 100.2], [98.0, 98.1, 98.2],
                  min_effect_pct=1.0)
    assert r["verdict"] == "WORSE", r


def test_fewer_than_two_runs_or_a_failed_proof_is_invalid():
    r = speed_row("d", "decode", "c8", "tok/s", True, [150], [151, 150])
    assert r["verdict"] == "INVALID" and "two or more" in r["why"], r
    r = speed_row("d", "decode", "c8", "tok/s", True, [150, 151], [151, 150],
                  invalid="at most 4 decoded at once, 8 sent")
    assert r["verdict"] == "INVALID" and "base" not in r, r


def test_text_rows():
    same = text_row("l", "lossless", "p1", {"ids": [1, 2, 3]}, {"ids": [1, 2, 3]})
    assert same["verdict"] == "PASS" and same["base_tokens"] == 3
    diff = text_row("l", "lossless", "p1", {"ids": [1, 2, 3], "gaps": {"1": 0.0012}},
                    {"ids": [1, 9, 3]})
    assert diff["verdict"] == "CHANGED_TEXT" and diff["first_diff"] == 1, diff
    assert "0.0012" in diff["why"]
    short = text_row("l", "lossless", "p1", {"ids": [1, 2, 3]}, {"ids": [1, 2]})
    assert short["verdict"] == "CHANGED_TEXT" and short["first_diff"] == 2
    err = text_row("l", "lossless", "p1", {"ids": [1]}, {"error": "HTTP 500"})
    assert err["verdict"] == "FAIL" and "head" in err["why"]


def test_case_rows():
    assert case_row("a", "api", "stream", True, True)["verdict"] == "PASS"
    assert case_row("a", "api", "stream", True, False)["verdict"] == "FAIL"
    assert case_row("a", "api", "stream", None, False)["verdict"] == "FAIL"
    s = case_row("a", "api", "stream", False, False)
    assert s["verdict"] == "SKIPPED" and "base as well" in s["why"]


def test_overall():
    p = {"verdict": "PASS"}
    assert overall([p]) == "PASS" and overall([]) == "PASS"
    assert overall([p, {"verdict": "SKIPPED"}]) == "PASS"
    assert overall([p, {"verdict": "CHANGED_TEXT"}]) == "CHANGED_TEXT"
    assert overall([{"verdict": "CHANGED_TEXT"}, {"verdict": "WORSE"}]) == "WORSE"
    assert overall([{"verdict": "WORSE"}, {"verdict": "INVALID"}]) == "FAIL"
    assert overall([{"verdict": "FAIL"}]) == "FAIL"


def test_schema_accepts_a_good_report_and_names_each_fault():
    rows = [speed_row("decode.c8", "decode", "c8 decode", "tok/s", True, [150, 151, 152], [150, 151, 152])]
    good = report(rows)
    assert schema.validate(good) == [], schema.validate(good)
    for mutate, want in (
        (lambda r: r.update(kind="x"), "kind"),
        (lambda r: r["head"].update(sha="abc"), "head.sha"),
        (lambda r: r.update(level="medium"), "level"),
        (lambda r: r["rows"][0].update(verdict="OK"), "verdict"),
        (lambda r: r.update(verdict="FAIL"), "the rows say PASS"),
        (lambda r: r.pop("planted"), "planted"),
        (lambda r: r["knobs"].update(head={}), "knobs.head"),
        (lambda r: r["rows"].append(dict(r["rows"][0])), "twice"),
        (lambda r: r.update(suites=["decode", "speedy"]), "unknown suite"),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        errs = schema.validate(bad)
        assert any(want in e for e in errs), (want, errs)
    assert schema.validate([]) == ["the report is not a JSON object"]


def test_markdown_holds_the_table_and_the_json_round_trips():
    rows = [speed_row("decode.c8", "decode", "c8 decode", "tok/s", True, [150, 151, 152], [120, 121, 122]),
            text_row("lossless.p1", "lossless", "prompt 1", {"ids": [1, 2]}, {"ids": [1, 2]}),
            verdict.skipped("api.tool", "api", "tool call", "the pack has no tool template")]
    rep = report(rows)
    md = render.markdown(rep)
    assert "| decode | c8 decode (tok/s) |" in md and "**WORSE**" in md, md
    assert MARKER in md and "## Quality gate: WORSE (quick)" in md
    got, err = render.extract(["no report here", md])
    assert err is None and got == json.loads(json.dumps(rep, sort_keys=True)), err
    # the last report wins; a broken last block is an error
    rep2 = report(rows[1:])
    got, err = render.extract([md, render.markdown(rep2)])
    assert got["verdict"] == "PASS"
    broken = MARKER + "\n```json\n{not json\n```\n"
    got, err = render.extract([md, broken])
    assert err and "does not parse" in err


def test_markdown_stays_under_a_comment():
    rows = [speed_row(f"decode.r{i}", "decode", f"row {i}", "tok/s", True, [150, 151, 152],
                      [150, 151, 152]) for i in range(60)]
    assert len(render.markdown(report(rows))) < render.LIMIT


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
