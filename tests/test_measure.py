"""tools/measure on a CPU against a small in-process engine (tests/measure_fake.py): a row with a record is
not measured again; merge keeps every imported row's source and the page shows it; the three views
(report page, card row, gate verdict) render from the records; the plan refuses a row that would be
measured twice; a row of N at once without the server's proof of N is INVALID; a server whose answers
lack `X-Tandem: raw` is refused.

Run: python tests/test_measure.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from measure_fake import FakeEngine  # noqa: E402
from tools.measure import plan as planmod, run as runmod, store as storemod, views  # noqa: E402
from tools.measure import merge as mergemod, rows as rowsmod  # noqa: E402
from tools.measure.server import Refused  # noqa: E402


def _run(base: str, out: str, rows: str, *extra) -> int:
    return runmod.main(["--url", base, "--out", out, "--rows", rows, "--hw", "fake", *extra])


def _tmp() -> str:
    return tempfile.mkdtemp(prefix="measure-")


def test_rows_are_unique_and_every_executor_exists():
    from tools.measure.execs import EXECS
    rows, groups = rowsmod.load()
    assert len(rows) == len(set(rows))
    for r in rows.values():
        assert r.exec in EXECS or r.exec == "every-run", r.id
        assert r.server is None or r.server in groups, r.id
    # the card's levels and the 2k requests at once are two rows, never one measurement
    assert rows["conc.card.c8"].exec != rows["conc.2k.c8"].exec
    assert rows["prefill.32k"].exec != rows["length.32k"].exec
    assert [r.id for r in rowsmod.for_view(rows, "gate") if r.id.startswith("conc.card")] == ["conc.card.c8"]


def test_a_row_with_a_record_is_not_measured_again():
    e = FakeEngine()
    out = _tmp()
    try:
        assert _run(e.base, out, "length.1k,length.4k") == 0
        st = storemod.Store(out)
        assert len(st.run_files()) == 1
        n0 = len([p for p in e.paths if p == "/v1/completions"])
        assert _run(e.base, out, "length.1k,length.4k") == 0          # nothing new: no run file, no request
        assert len(st.run_files()) == 1
        assert len([p for p in e.paths if p == "/v1/completions"]) == n0
        rec = st.primary("length.1k", "9.9.9", "fake-pack", "fake")
        assert rec["valid"] and len(rec["rows"]) == 3 and rec["source"]["how"] == "measured"
        assert rec["rows"][0]["section"] == "a" and rec["values"]["decode_tok_s"] > 0
        # --force measures it again; the new record becomes the key's, the old one stays beside it
        assert _run(e.base, out, "length.1k", "--force") == 0
        assert len(st.run_files()) == 2
        got = st.find("length.1k", "9.9.9", "fake-pack", "fake")
        assert [g["_primary"] for g in got] == [True, False] and got[0].get("forced")
        # a store add of the same key without force is refused
        try:
            st.add_run({"id": "again"}, {"length.4k": dict(rec, row="length.4k")})
            raise AssertionError("a duplicate record was taken")
        except storemod.Duplicate as exc:
            assert "length.4k" in str(exc)
    finally:
        e.stop()


def _report_data(tmp: Path) -> Path:
    """A report data.json with one length row, a card level and a fresh 128k row, as tools/release_report
    wrote them, with a planted home path."""
    knobs = {"lanes": 8, "max_running": 8, "max_len": 262144, "drafter": "mtp", "env": {"X": "/home/alice/x"}}
    runs = [{"started": "2026-11-04T15:07:56+0100", "sections": ["g"], "variant": "spec", "knobs": knobs,
             "hw_profile": "gb10", "served_at_start": 0, "mode": "url"},
            {"started": "2026-11-04T15:36:13+0100", "sections": ["i", "a"], "variant": "spec", "knobs": knobs,
             "hw_profile": "gb10", "served_at_start": 0, "mode": "url"}]
    rows = [{"section": "a", "key": "1024/0", "v": {"decode_tok_s": 60.0, "ttft_s": 1.2, "prompt_tokens": 1024},
             "cond": {"target": 1024, "run": 0}, "valid": True, "why": None},
            {"section": "g", "key": "fixed/levels/8", "v": {"wall_tok_s": 150.1, "decode_tok_s": 158.9, "ttft_s": 1.6,
                                                             "at_once": 8, "rounds": 3},
             "cond": {"args": "fixed"}, "valid": True, "why": None},
            {"section": "i", "key": "fresh/131072", "v": {"decode_tok_s": 37.1, "ttft_s": 182.6},
             "cond": {"target": 131072, "state": "fresh", "served_before": 1}, "valid": True, "why": None},
            {"section": "h", "key": "spec/code", "v": {"text": "x"}, "cond": {}, "valid": True, "why": None}]
    p = tmp / "data.json"
    p.write_text(json.dumps({"kind": "tandem-benchmark-report", "tool": "1.0", "version": "0.15.2+pk",
                             "model": "Model X", "weights": "pk", "hardware": "one DGX Spark (GB10)", "rows": rows,
                             "runs": runs, "private": {"homes": ["/home/alice"], "users": ["alice"], "hosts": ["boxy"]}}))
    return p


def _gate_report(tmp: Path) -> Path:
    knobs = {"version": "0.15.2+pk", "weights": "pk", "lanes": 8, "max_running": 8, "max_len": 262144,
             "drafter": "mtp"}
    side = {"n": 6, "median": 1131.8, "min": 1120.0, "max": 1140.0, "runs": [1131.8] * 6}
    rep = {"kind": "tandem-gate-report", "tool": {"version": "1.0"}, "created_at": "2026-11-04T13:15:53Z",
           "level": "full", "pack": {"id": "pk"}, "hardware": {"profile": "gb10"}, "pairs": 2,
           "head": {"sha": "e4547167f70f"}, "knobs": {"head": knobs},
           "rows": [{"id": "prefill.32k", "verdict": "PASS", "why": "x", "unit": "tok/s", "head": side},
                    {"id": "prefill.64k", "verdict": "PASS", "why": "x", "unit": "tok/s", "head": dict(side, median=997.1)},
                    {"id": "long.max.needle", "verdict": "PASS", "why": "passes: answered '7341'"},
                    {"id": "lossless.plain.code-1", "verdict": "PASS", "why": "the same text (193 tokens)"},
                    {"id": "api.health", "verdict": "PASS", "why": "x"}]}
    p = tmp / "gate-report.json"
    p.write_text(json.dumps(rep))
    return p


def test_merge_keeps_the_source_and_the_page_shows_it():
    tmp = Path(_tmp())
    out = str(tmp / "store")
    st = storemod.Store(out)
    got = mergemod.import_report(st, str(_report_data(tmp)), label="fn1", commit="6726bc14", log=lambda *x: None)
    assert got["unmapped"] == ["h/spec/code"]
    assert set(got["rows"]) == {"length.1k", "conc.card.c8", "length.128k"}
    rec = st.primary("length.1k", "0.15.2", "pk", "gb10")
    s = rec["source"]
    assert s["how"] == "imported" and s["tool"] == "tools/release_report 1.0" and s["date"] == "2026-11-04"
    assert s["path"].endswith("data.json") and s["commit"] == "6726bc14"
    assert st.primary("length.128k", "0.15.2", "pk", "gb10")["rows"][0]["section"] == "i"
    mergemod.import_gate(st, str(_gate_report(tmp)), log=lambda *x: None)
    assert st.primary("prefill.64k", "0.15.2", "pk", "gb10")["source"]["tool"] == "tools/gate 1.0"
    assert st.primary("lossless.plain", "0.15.2", "pk", "gb10")["rows"][0]["key"] == "specplain/code-1"
    assert st.primary("api.health", "0.15.2", "pk", "gb10") is None              # not a measurement
    # the same report imported again: the first stays the row's record, the second is listed beside it
    mergemod.import_report(st, str(_report_data(tmp)), label="again", log=lambda *x: None)
    both = st.find("length.1k", "0.15.2", "pk", "gb10")
    assert [b["_primary"] for b in both] == [True, False]
    html_p = tmp / "page.html"
    html = views.render_report(st, "0.15.2", ["pk"], None, None, str(html_p))
    assert "Where each row comes from" in html and "Rows from:" in html and "tools/gate 1.0" in html
    assert "needle at 128k (the gate&#x27;s long case (card filler)): found" in html or "needle at 128k" in html
    for planted in ("alice", "boxy", "/home/"):
        assert planted not in html, planted
    data = views.report_data(st, "0.15.2", "pk")
    assert {r["section"] for r in data["rows"] if r["row"] == "length.128k"} == {"i", "a"}
    assert "conc.2k.c8" in data["missing"]


def test_the_three_views_render_from_records():
    e = FakeEngine(tok_s=400.0)                 # 1.3 s answers: 8 at once overlap on a busy runner too
    base, head = _tmp(), _tmp()
    try:
        card = "decode.*.c1,prefill.2k,prefill.8k,prefill.32k,conc.card.c*"
        gate = "decode.c1.short,decode.c1.1k,burst.8x2k,lossless.set"
        assert _run(e.base, head, card, "--any-group") == 0
        assert _run(e.base, head, gate, "--gate-pack", "fake") == 0
        assert _run(e.base, base, "conc.card.c8,prefill.2k,prefill.8k," + gate, "--any-group", "--gate-pack", "fake") == 0
    finally:
        e.stop()
    st = storemod.Store(head)
    # card: tools/parallel_card.py's row, its numbers the records'
    raw = views.card_raw(st, "9.9.9", "fake-pack", "qwen3.8-27b", "exl3-4.0", "measure", commit="abcdef12")
    row = raw["row"]
    c8 = st.primary("conc.card.c8", "9.9.9", "fake-pack")["values"]
    assert row["levels"]["8"]["wall_tok_s"] == round(c8["wall_tok_s"], 1) and row["levels"]["8"]["at_once"] == 8
    assert row["levels"]["8"]["rounds"] == 3 and set(row["decode"]) == {"code", "prose"}
    assert set(row["prefill"]) == {"2048", "8192", "32768"} and row["commit"] == "abcdef12"
    assert raw["valid"] and all(v.startswith("measured") for v in raw["sources"].values())
    # gate: the verdict table from two stores, the gate's own rules
    rows, md = views.render_gate(base, head)
    ids = {r["id"]: r for r in rows}
    gate_ids = ("decode.c1", "decode.c1-1k", "decode.c8", "decode.c8.wall", "prefill.2k", "prefill.8k",
                "prefill.burst4x1k", "lossless.card-code.greedy")
    for rid in gate_ids:
        assert rid in ids, (rid, sorted(ids))
        assert ids[rid]["verdict"] in ("PASS", "SKIPPED", "WORSE"), (rid, ids[rid])
    assert ids["lossless.card-code.greedy"]["verdict"] == "PASS", ids["lossless.card-code.greedy"]
    assert md.startswith("verdict ") and "| decode |" in md
    # the speed rows above are two fake-engine runs on a shared runner, so their verdict is timing
    # (prefill.2k resolved 21.8 % worse once in CI); one store against itself passes every row
    rows_same, md_same = views.render_gate(head, head)
    same = {r["id"]: r for r in rows_same}
    for rid in gate_ids:
        assert same[rid]["verdict"] in ("PASS", "SKIPPED"), (rid, same[rid])
    assert md_same.startswith("verdict PASS"), md_same[:200]
    # the gate's runner writes each arm's suites as records (tools/measure/gatearm.py): the same table
    import types
    from tools.measure import gatearm
    gout = Path(_tmp())
    for side, d in (("base", base), ("head", head)):
        g = views.gate_raw(storemod.Store(d), "9.9.9", "fake-pack", side)
        arm = types.SimpleNamespace(side=side, variant="main", key=side, port=8100)
        knobs = st.primary("conc.card.c8", "9.9.9", "fake-pack")["_run"]["knobs"]
        gatearm.record_arm(gout, arm, knobs, {s: v[side][0] for s, v in g.items()}, "fake", "fake-pack", "abc",
                           log=lambda *x: None)
    rows2, md2 = views.render_gate(str(gout / "records" / "base"), str(gout / "records" / "head"))
    # the same rows; base vs head timed on a busy runner may say WORSE on a speed row (as above), so the PASS rule
    # is checked on head against itself
    assert [r["id"] for r in rows2] == [r["id"] for r in rows], md2[:400]
    md3 = views.render_gate(str(gout / "records" / "head"), str(gout / "records" / "head"))[1]
    assert md3.startswith("verdict PASS"), md3[:400]
    # report: a page from the measured records
    html = views.render_report(st, "9.9.9", ["fake-pack"], None, "Fake", os.path.join(head, "p.html"))
    assert 'id="s-g"' in html and "Rows from: measured" in html


def test_the_plan_refuses_a_row_measured_twice():
    tmp = Path(_tmp())
    st = storemod.Store(str(tmp / "store"))
    mergemod.import_report(st, str(_report_data(tmp)), log=lambda *x: None)
    rows, _ = rowsmod.load()
    # asked by id, a row with a record is refused
    pl = planmod.make(st, ["pk"], "0.15.2", rowsmod.select(rows, "length.1k,length.4k"), "gb10", {}, explicit=True)
    assert any("length.1k: has a record" in x for x in pl["refusals"]), pl["refusals"]
    assert planmod.main(["--out", str(tmp / "store"), "--rows", "length.1k", "--pack", "pk", "--build", "0.15.2"]) == 2
    # a view: rows with records are listed with their source, the rest packed into holds of <= 62 min
    pl = planmod.make(st, ["pk"], "0.15.2", rowsmod.for_view(rows, "report"), "gb10", {}, explicit=False)
    assert not pl["refusals"], pl["refusals"]
    assert ("pk", "length.1k", "record") == pl["lines"][[x[1] for x in pl["lines"]].index("length.1k")][:3]
    assert all(h["minutes"] <= pl["cap"] for h in pl["holds"])
    todo = [r.id for _, r in pl["todo"]]
    assert "length.1k" not in todo and "length.4k" in todo and len(todo) == len(set(todo))
    # a warm row whose earlier rows have records would measure them again: refused
    one = dict(st.primary("length.1k", "0.15.2", "pk", "gb10"))
    one.update(row="conc.32k.c1", rows=[])
    st.add_run({"id": "x"}, {"conc.32k.c1": {k: v for k, v in one.items() if not k.startswith("_")}})
    pl = planmod.make(st, ["pk"], "0.15.2", rowsmod.for_view(rows, "report"), "gb10", {}, explicit=False)
    assert any("warm.128k: runs after conc.32k.c1" in x for x in pl["refusals"]), pl["refusals"]


def test_n_at_once_needs_the_servers_proof():
    from tools.measure.execs import proof_problem
    e = FakeEngine(at_once_cap=2)                      # /health never shows more than 2 together
    out = _tmp()
    try:
        assert _run(e.base, out, "conc.2k.c2,conc.2k.c4") == 1         # an invalid row: exit 1
        st = storemod.Store(out)
        ok = st.primary("conc.2k.c2", "9.9.9", "fake-pack")
        bad = st.primary("conc.2k.c4", "9.9.9", "fake-pack")
        assert ok["valid"] and not bad["valid"] and "2 of 4" in bad["why"], bad["why"]
    finally:
        e.stop()
    # the runner's own rule beside the executor's: a valid row of N at once without a proof is INVALID
    p = {"valid": True, "rows": [{"section": "c", "cond": {}}], "raw": None}
    assert "no proof" in proof_problem("conc.2k.c4", 4, p)
    p = {"valid": True, "rows": [{"cond": {"proof": {"max_decoding_at_once": 3, "max_running": 4, "samples": 9}}}]}
    assert "3 of 4" in proof_problem("conc.2k.c4", 4, p, "decoding")
    assert proof_problem("conc.32k.c4", 4, p, "decoding") is None            # prefill rows: N running


def test_a_stopped_run_keeps_the_rows_it_finished():
    import signal
    import subprocess
    import time
    e = FakeEngine(tok_s=300.0)
    out = _tmp()
    try:
        p = subprocess.Popen([sys.executable, "-u", "-m", "tools.measure", "run", "--url", e.base, "--out", out,
                              "--rows", "length.1k,length.4k,length.16k", "--hw", "fake"], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(14.0)                          # 1k's three runs are done (~6 s each row), 16k is not
        p.send_signal(signal.SIGTERM)
        log = p.communicate(timeout=60)[0]
        assert p.returncode == 4, p.returncode                 # stopped: non-zero, its finished rows written
        st = storemod.Store(out)
        assert "stopped (signal 15)" in log, log[-800:]
        (f,) = st.run_files()
        run = json.loads(f.read_text())
        assert run["run"]["stopped"] and "length.16k" in run["run"]["not_measured"], run["run"]
        assert st.primary("length.1k", "9.9.9", "fake-pack")["valid"]
        assert st.primary("length.16k", "9.9.9", "fake-pack") is None
    finally:
        e.stop()


def test_a_server_that_is_not_raw_is_refused():
    import subprocess
    e = FakeEngine(mode="chat")
    try:
        # the command exits non-zero (3), so a hold script sees the refusal
        r = subprocess.run([sys.executable, "-m", "tools.measure", "run", "--url", e.base, "--out", _tmp(), "--rows",
                            "length.1k", "--hw", "fake"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert r.returncode == 3 and "refused" in r.stderr, (r.returncode, r.stderr[-300:])
        try:
            runmod.measure(runmod.parser().parse_args(["--url", e.base, "--out", _tmp(), "--rows", "length.1k",
                                                       "--hw", "fake"]), log=lambda *x: None)
            raise AssertionError("a chat-layer server was measured")
        except Refused as exc:
            assert "not served raw" in str(exc)
    finally:
        e.stop()


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
