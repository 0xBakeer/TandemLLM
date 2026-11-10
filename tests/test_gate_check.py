"""tools/gate/check.py: the `gate` status check on good, stale, failing and missing reports, on a CPU.

Run: python tests/test_gate_check.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from tools.gate import render  # noqa: E402
from tools.gate.check import check  # noqa: E402
from tools.gate.verdict import overall, speed_row, text_row  # noqa: E402
from test_gate_report import SHA_B, SHA_H, report  # noqa: E402

MAIN = "c" * 40


class FakeGit:
    def __init__(self, changed=None, ancestors=(SHA_B,)):
        self.changed_map, self.ancestors = changed or {}, set(ancestors)

    def changed(self, a, b):
        return self.changed_map.get((a, b))

    def is_ancestor(self, a, b):
        return a in self.ancestors


def rows_ok():
    return [text_row("lossless.p1", "lossless", "prompt 1", {"ids": [1]}, {"ids": [1]}),
            speed_row("decode.c8", "decode", "c8", "tok/s", True, [150, 151, 152], [150, 152, 151]),
            speed_row("prefill.2k", "prefill", "2k", "tok/s", True, [5000, 5010, 5020], [5005, 5000, 5015]),
            speed_row("long.max", "long", "64k", "s", False, [20, 20.1, 20.2], [20.1, 20, 20.2]),
            text_row("api.stream", "api", "stream", {"ids": [1]}, {"ids": [1]})]


ENGINE = ["engine/models/flashnext/engine.py"]


def run_check(rep, files=ENGINE, head=SHA_H, git=None, extra=()):
    texts = ["PR body", *extra] + ([render.markdown(rep)] if rep else [])
    return check(files, MAIN, head, texts, git or FakeGit())


def test_docs_only_needs_no_report():
    res = run_check(None, files=["docs/a.md", "README.md"])
    assert res["ok"] and res["level"] == "none", res


def test_good_report_passes():
    rep = report(rows_ok(), kernels=["tests/gpu/test_fn_kernels_gpu.py", "tests/gpu/test_fn_prefill_kernels_gpu.py",
                                     "tests/gpu/test_fn_seq_kernels_gpu.py", "tests/gpu/test_fn_attn_multi_gpu.py",
                                     "tests/gpu/test_fn_gemv_nb_gpu.py", "tests/gpu/test_fn_mixer_mma_gpu.py",
                                     "tests/gpu/test_fn_joint_graph_gpu.py", "tests/gpu/test_fn_joint_prefill_gpu.py",
                                     "tests/gpu/test_fn_attn_rows_big_gpu.py", "tests/gpu/test_fn_attn_stages_gpu.py",
                                     "tests/gpu/test_fn_pf_hc_gpu.py", "tests/gpu/test_fn_pf_nocopy_gpu.py",
                                     "tests/gpu/test_fn_pf_window_gpu.py", "tests/gpu/test_fn_prefill_mm_gpu.py",
                                     "tests/gpu/test_fn_pack_graph_gpu.py", "tests/gpu/test_fn_route_gpu.py"])
    res = run_check(rep)
    assert res["ok"], res["problems"]


def test_missing_report_fails_with_the_command():
    res = run_check(None)
    assert not res["ok"] and "no gate report found" in res["problems"][0]
    assert "python -m tools.gate run" in res["problems"][0]


def test_unparseable_report_fails():
    res = run_check(None, extra=[render.MARKER + "\n```json\n{oops\n```"])
    assert not res["ok"] and "does not parse" in res["problems"][0], res


def test_stale_report():
    rep = report(rows_ok())
    later = "d" * 40
    # the head moved by an engine file: stale
    git = FakeGit(changed={(SHA_H, later): ["engine/kvpool.py"]})
    res = run_check(rep, head=later, git=git)
    assert not res["ok"] and any("stale" in p and "quick+long" in p for p in res["problems"]), res
    # the head moved by docs only: still good
    git = FakeGit(changed={(SHA_H, later): ["docs/x.md"]})
    res = run_check(rep, head=later, git=git, files=ENGINE)
    assert not any("stale" in p for p in res["problems"]), res
    assert any("need no report" in n for n in res["notes"])
    # a head that is not in the history at all
    res = run_check(rep, head=later, git=FakeGit())
    assert any("not in this pull request's history" in p for p in res["problems"]), res


def test_base_must_be_a_target_branch_commit():
    res = run_check(report(rows_ok()), git=FakeGit(ancestors=()))
    assert any("is not a commit of the target branch" in p for p in res["problems"]), res


def test_failing_rows_and_unexplained_changes():
    rows = rows_ok()
    rows.append(speed_row("decode.c1", "decode", "c1", "tok/s", True, [60, 61, 62], [40, 41, 42]))
    rows.append(text_row("lossless.p2", "lossless", "prompt 2", {"ids": [1, 2]}, {"ids": [1, 3]}))
    rows.append(speed_row("decode.c4", "decode", "c4", "tok/s", True, [1, 2], [1, 2],
                          invalid="at most 2 decoded at once, 4 sent"))
    rep = report(rows)
    res = run_check(rep)
    probs = "\n".join(res["problems"])
    assert "INVALID decode.c4" in probs
    assert "WORSE decode.c1 has no written reason" in probs
    assert "CHANGED_TEXT lossless.p2 has no written reason" in probs
    # reasons for the two, and no invalid row: passes (a maintainer reads the reasons)
    rows = [r for r in rows if r["id"] != "decode.c4"]
    rep = report(rows, reasons={"decode.c1": "the c1 path now runs the checked kernel, 30 % slower",
                                "lossless.p2": "a near tie at position 1 flips with the new reduction order"},
                 kernels=["tests/gpu/test_fn_kernels_gpu.py", "tests/gpu/test_fn_prefill_kernels_gpu.py",
                          "tests/gpu/test_fn_seq_kernels_gpu.py", "tests/gpu/test_fn_attn_multi_gpu.py",
                          "tests/gpu/test_fn_gemv_nb_gpu.py", "tests/gpu/test_fn_mixer_mma_gpu.py",
                          "tests/gpu/test_fn_joint_graph_gpu.py", "tests/gpu/test_fn_joint_prefill_gpu.py",
                          "tests/gpu/test_fn_attn_rows_big_gpu.py", "tests/gpu/test_fn_attn_stages_gpu.py",
                          "tests/gpu/test_fn_pf_hc_gpu.py", "tests/gpu/test_fn_pf_nocopy_gpu.py",
                          "tests/gpu/test_fn_pf_window_gpu.py", "tests/gpu/test_fn_prefill_mm_gpu.py",
                          "tests/gpu/test_fn_pack_graph_gpu.py", "tests/gpu/test_fn_route_gpu.py"])
    res = run_check(rep)
    assert res["ok"], res["problems"]
    assert any("explained" in n for n in res["notes"])
    # a too-short reason does not count
    rep["reasons"]["decode.c1"] = "slower"
    assert not run_check(rep)["ok"]


def test_level_suites_kernels_and_planted():
    rep = report(rows_ok())
    res = run_check(rep, files=["server/slots.py"])
    assert any("ran level quick; the diff needs quick+long" in p for p in res["problems"]), res
    rep = report([r for r in rows_ok() if r["suite"] != "long"],
                 suites=["lossless", "decode", "prefill", "api", "kernels"])
    res = run_check(rep)
    assert any("suite long did not run" in p for p in res["problems"]), res
    rep = report(rows_ok())
    res = run_check(rep)
    assert any("GPU tests the diff maps to did not run" in p for p in res["problems"]), res
    rep = report(rows_ok(), planted={"note": "a sleep in decode"})
    res = run_check(rep, files=["engine/spec.py"])
    assert any("planted" in p for p in res["problems"]), res


def test_wrong_tool_major_and_bad_schema():
    rep = report(rows_ok())
    rep["tool"]["version"] = "2.0"
    assert any("major version" in p for p in run_check(rep, files=["engine/spec.py"])["problems"])
    rep = report(rows_ok())
    rep["rows"][0]["verdict"] = "GOOD"
    res = run_check(rep)
    assert any(p.startswith("report schema:") for p in res["problems"]), res


def test_the_command_line_on_a_real_repo():
    """`python -m tools.gate check` against a throwaway git repository: docs-only passes, an
    engine change without a report fails, and the summary file gets the result."""
    d = tempfile.mkdtemp(prefix="gate-check-")
    g = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    g("init", "-q")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "t")
    os.makedirs(os.path.join(d, "docs"))
    open(os.path.join(d, "docs/a.md"), "w").write("a\n")
    g("add", "-A")
    g("commit", "-qm", "base")
    base = g("rev-parse", "HEAD")
    open(os.path.join(d, "docs/a.md"), "w").write("b\n")
    g("commit", "-qam", "docs")
    head = g("rev-parse", "HEAD")
    texts = os.path.join(d, "texts.json")
    json.dump({"body": "x", "comments": []}, open(texts, "w"))
    summ = os.path.join(d, "summary.md")
    cmd = [sys.executable, "-m", "tools.gate", "check", "--base-sha", base, "--head-sha", head,
           "--texts", texts, "--repo", d, "--summary", summ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    os.makedirs(os.path.join(d, "engine"))
    open(os.path.join(d, "engine/x.py"), "w").write("x = 1\n")
    g("add", "-A")
    g("commit", "-qm", "engine")
    cmd[cmd.index(head)] = g("rev-parse", "HEAD")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "no gate report found" in r.stdout, r.stdout + r.stderr
    assert "### gate: FAIL" in open(summ).read()


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
