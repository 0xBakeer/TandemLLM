"""tools/release_report on a CPU, no server: the plan sized from /health and the GPU, the raw-header
rule, the row schema, the public scrub, and the page rendered from a fixture data.json (with a planted
home path, user and host name that must not reach it).

Run: python tests/test_release_report.py
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.gate.hw.base import Fingerprint  # noqa: E402
from tools.gate.hw.gb10 import GB10  # noqa: E402
from tools.gate.hw.generic import Generic  # noqa: E402
from tools.release_report import BARE, SECTIONS, plan, render, scrub  # noqa: E402
from tools.release_report.client import raw_served  # noqa: E402
from tools.release_report.runner import bare_refusal, merge, sections_of  # noqa: E402
from tools.release_report.sections import check, row, skipped  # noqa: E402

PKG = os.path.join(ROOT, "tools", "release_report")


def health(lanes=8, window=262144, drafters="mtp (the pack's own layer)", **kw) -> dict:
    h = {"status": "ok", "version": "0.15.1+flashnext-3.05", "model": "TandemLLM", "max_len": window,
         "mode": "engine", "cache": {"response_cache": None},
         "weights": {"name": "flashnext-3.05", "weights": "/home/alice/packs/3.05", "drafters": drafters},
         "parallel": {"n": lanes, "max_running": lanes, "slots": [{"index": i, "max_len": window} for i in range(lanes)]},
         "inflight": {"running": 0, "waiting": 0, "served": 0}}
    h.update(kw)
    return h


def test_plans_follow_the_server_and_the_gpu():
    big = plan.make(health(), None, GB10(Fingerprint(name="NVIDIA GB10", memory_gib=119.7)))
    small = plan.make(health(lanes=2, window=20480), None, Generic(Fingerprint(name="RTX 4000", memory_gib=20)))
    assert big.lengths[-1] == 245760 and big.conc_2k == list(range(1, 9)) and big.conc_32k == list(range(1, 9))
    assert small.lengths == [1024, 4096, 16384] and small.conc_2k == [1, 2] and small.conc_32k == []
    assert "over the largest slot window" in small.skipped["a.32k"]
    assert small.load_prompt is None and "three requests" in small.skipped["e.load"]
    assert big.warm == [131072, 196608] and small.warm == []
    one = plan.make(health(lanes=1, window=65536), None, None)
    assert one.chats == [] and "one request at a time" in one.skipped["c"]
    q = plan.make(health(), None, GB10(Fingerprint(name="NVIDIA GB10")), quick=True)
    assert q.lengths == [1024, 16384, 65536] and all(v == 1 for v in q.runs.values())
    assert q.conc_2k == [1, 4, 8] and q.conc_32k == [1, 8] and q.warm == [] and q.needles == [32768]
    est = plan.estimate(q, ["a", "c", "d", "e", "f", "g", "h"], 950.0, 60.0)
    assert 12 < est["total"] < 26, est           # --quick: about 20 minutes on the reference box


def test_a_server_without_a_drafter_is_plain():
    assert plan.make(health(drafters="none"), None, None).drafter is False
    assert plan.make(health(drafters=None), None, None).drafter is False
    assert plan.make(health(), None, None).drafter is True


def test_raw_is_the_mode_header():
    assert raw_served({"X-Tandem": "raw"}) == (True, "X-Tandem: raw")
    assert raw_served({"X-Tandem": "raw", "X-Tandem-Profile": "chat"})[0] is True
    assert raw_served({"X-Tandem": "on", "X-Tandem-Profile": "chat"})[0] is False
    assert raw_served({})[0] is False
    ok = {"raw": True, "completion_tokens": 5}
    assert check([ok]) is None
    assert "not the bare engine" in check([ok, {"raw": False, "raw_why": "X-Tandem: missing", "completion_tokens": 5}])
    assert "failed" in check([{"error": "HTTP 500: x", "raw": True}])


def test_the_bare_check_refuses_a_response_cache_and_a_sick_server():
    assert bare_refusal(health()) is None
    assert bare_refusal(health(mode="chat")) is None           # a loaded conductor hands raw requests back
    assert "response cache" in bare_refusal(health(cache={"response_cache": {"entries": 0}}))
    assert "status" in bare_refusal({"status": "loading"})


def test_no_chat_endpoint_anywhere_in_the_report_code():
    for root, _, files in os.walk(PKG):
        for f in files:
            if f.endswith(".py"):
                text = open(os.path.join(root, f)).read()
                assert "chat/completions" not in text, f"{f} names the chat endpoint"
    from tools.release_report import client
    assert client.ENDPOINT == "/v1/completions" and client.RAW_HEADERS == {"X-Tandem": "raw"}


def test_sections_and_rows():
    assert sections_of(None, False) == ["a", "c", "d", "e", "f", "g", "h", "i"]
    assert sections_of(["b", "a", "i"], False) == ["a", "i"]
    assert "j" in sections_of(None, True)
    r = row("a", "1024/0", {"decode_tok_s": 60.0}, {"target": 1024})
    assert set(r) >= {"section", "key", "v", "cond", "valid", "why"} and r["valid"] is True
    s = skipped("i", "warm", "--quick")
    assert s["skipped"] == "--quick" and s["valid"] is False


PRIVATE = {"hosts": ["boxy-node7"], "users": ["alice"], "homes": ["/home/alice"], "allow": ["NVIDIA GB10", "DGX Spark"]}


def fixture() -> dict:
    """A data.json of every section, with private names planted in the strings a run records."""
    rows = []
    for n, d, dr in ((1024, 64.3, True), (16384, 61.7, True), (32768, 47.6, False), (131072, 35.2, False)):
        for k in range(2):
            rows.append(row("a", f"{n}/{k}", {"prompt_tokens": n - 20, "decode_tok_s": d + k, "ttft_s": n / 1000,
                                              "prefill_tok_s": 1000.0 - n / 1000},
                            {"target": n, "run": k}, engine=[{"drafting": dr, "tokens_per_block": 3.1 if dr else 1.0,
                                                              "decode_tok_s": d}]))
    for n in (1, 2, 4, 8):
        rows.append(row("c", f"2k/{n}/0", {"n": n, "agg_tok_s": 50 * n ** 0.6, "wall_s": 10.0, "decode_tok_s": 40,
                                           "ttft_max_s": 2.0 * n},
                        {"proof": {"max_decoding_at_once": n, "max_running": n}}))
    rows.append(row("c", "32k/2", {"n": 2, "ttft_min_s": 33.0, "ttft_max_s": 67.0}, {}))
    rows.append(row("c", "32k/8", {"n": 8}, {}, valid=False, why="only 6 of 8 ran at once"))
    rows.append(row("d", "5", {"chats": 5, "ttft_first_s": 120.3, "ttft_last_s": 126.5, "prompt_tokens": 28700,
                               "cached_tokens": [0, 0, 0, 0, 0]}, {}))
    rows.append(row("e", "load", {"ttft_s": 68.2, "prompt_tokens": 65500, "decoders_before_tok_s": 64.1,
                                  "decoders_during_tok_s": 1.2}, {}))
    rows.append(row("e", "return", {"prompt_tokens": 98000, "turn1_ttft_s": 122.8, "turn2_ttft_s": 0.98,
                                    "turn2_cached": 97000, "flood_ttft_median_s": 62.9, "turn3_ttft_s": 4.19,
                                    "turn3_cached": 96000}, {}))
    for t in ("summary", "code", "copy"):
        rows.append(row("f", f"{t}/0", {"prompt_tokens": 57700, "ttft_s": 61.7, "decode_tok_s": 36.0,
                                        "peak_1s_tok_s": 46.0, "completion_tokens": 1024}, {"task": t, "run": 0},
                        engine=[{"drafting": False}]))
    for tag in ("default", "fixed"):
        for n in (1, 2, 4, 8):
            rows.append(row("g", f"{tag}/levels/{n}", {"wall_tok_s": 40.0 * n ** 0.7, "ttft_s": 0.4 * n}, {}))
    rows.append(row("g", "default/decode/code", {"decode_tok_s": 80.2}, {}))
    rows.append(row("g", "default/prefill/2048", {"tok_s": 1088, "ttft_s": 1.9}, {}))
    for var in ("spec", "plain"):
        for pid in ("code", "logic"):
            rows.append(row("h", f"{var}/{pid}", {"text": f"answer to {pid}", "sha256": pid, "pieces": None},
                            {"variant": var}))
    rows.append(row("h", "needle/32768", {"found": True}, {"target": 32768}))
    rows.append(row("i", "warm/131072", {"decode_tok_s": 21.8}, {"target": 131072, "served_before": 140},
                    engine=[{"drafting": False}]))
    rows.append(row("i", "fresh/131072", {"decode_tok_s": 35.2}, {"target": 131072, "served_before": 1},
                    engine=[{"drafting": False}]))
    rows.append(skipped("j", "score", "not run (it takes about 2.5 hours: --hle --hle-data FILE)"))
    run = {"started": "2026-11-03T16:00:00+0100", "ended": "2026-11-03T17:00:00+0100", "minutes": 60.0,
           "sections": ["a", "c"], "variant": "spec", "version": "0.15.1+flashnext-3.05", "model": "Qwen3.8-Flash-Next",
           "weights": "flashnext-3.05", "weights_repo": "turboderp/Qwen3.8-Flash-Next-exl3", "format": "exl3",
           "hardware": "one DGX Spark (GB10)",
           "gpu": {"name": "NVIDIA GB10", "memory_gib": 119.7, "cc": "12.1", "driver": "580", "cuda": "13.0",
                   "torch": "2.13"},
           "knobs": {"lanes": 8, "max_len": 262144, "drafter": "mtp",
                     "env": {"QSE_WEIGHTS_DIR": "/home/alice/packs/3.05", "QWEN38_PROBE_DIR": "~/state/probe",
                             "QSE_NOTE": "served from boxy-node7 by alice at http://192.168.1.20:8000/x"},
                     "flags": {"--model": "/home/alice/packs/3.05", "--host": "boxy-node7.lan"}},
           "box": {"clock_mhz_median_load": 2184, "power_w_median_load": 41.4, "temp_c_max": 76,
                   "mem_avail_gib_min": 12.8},
           "prose": {"source": "/home/alice/texts/book.txt", "sha256": "ab" * 32, "repeats": False}}
    return {"kind": "tandem-benchmark-report", "schema": 1, "tool": "1.0", "version": "0.15.1+flashnext-3.05",
            "model": "Qwen3.8-Flash-Next", "weights": "flashnext-3.05", "hardware": "one DGX Spark (GB10)",
            "rows": rows, "runs": [run], "private": PRIVATE, "notes": ["checked on boxy-node7 in /home/alice/run"]}


def test_the_page_renders_every_section_and_names_no_box():
    data = fixture()
    html = render.page(data)
    assert "<title>TandemLLM 0.15.1 · Qwen3.8-Flash-Next</title>" in html
    assert BARE in html and html.count(BARE) == 1
    for sid, (name, _) in SECTIONS.items():
        assert f'id="s-{sid}"' in html and render.esc(name) in html
    for planted in ("alice", "boxy-node7", "/home/", "192.168.", "http://192", ".lan"):
        assert planted not in html, planted
    assert scrub.leaks(html, PRIVATE) == []
    assert "DGX Spark (GB10)" in html                     # the board's name is not a private name
    assert len(html.encode()) < 2 * 1024 * 1024
    assert re.search(r"<svg class=\"chart\"", html)
    assert "invalid" in html and "only 6 of 8 ran at once" in html     # an invalid row shows its reason
    assert "not run" in html                                             # section j without --hle


def test_the_verdict_states_what_the_rows_show():
    f = render.findings(fixture())
    text = " ".join(t for _, t in f)
    assert "Drafting runs up to 16k prompts and stops from 32k on" in text and "not isolated" in text
    assert any(k == "bad" and "after the rest of the suite" in t and "a fresh server" in t for k, t in f)
    assert "1 measured rows are invalid" in text


def test_scrub_and_leaks():
    s = scrub.text("model at /home/alice/packs/3.05 on boxy-node7, http://10.0.0.5:8000/v1", PRIVATE)
    assert "alice" not in s and "boxy-node7" not in s and "10.0.0.5" not in s and "3.05" in s
    assert scrub.leaks("<p>/Users/bob/x and 10.1.2.3 and gpu.corp</p>", {}) != []
    assert scrub.leaks("<p>one DGX Spark (GB10)</p>", {"hosts": ["dgx"], "allow": ["DGX Spark"]}) == []
    assert scrub.leaks("<p>ran on dgx</p>", {"hosts": ["dgx"]}) != []


def test_a_page_that_still_names_the_box_is_refused():
    data = fixture()
    data["model"] = "Qwen on boxy-node7"            # scrubbed, so it renders
    assert "boxy-node7" not in render.page(data)
    orig = render.scrub.deep
    try:
        render.scrub.deep = lambda x, p=None: x       # a scrub that does nothing: the leak check refuses
        try:
            render.page(data)
            raise AssertionError("a leaking page was written")
        except render.Leak as exc:
            assert "boxy-node7" in str(exc) or "path" in str(exc)
    finally:
        render.scrub.deep = orig


def test_data_json_merges_runs_and_keeps_one_version(tmp_path):
    run = {"version": "0.15.1", "weights": "w", "model": "m", "hardware": "h", "_private": {"hosts": ["a"]}}
    d1 = merge(tmp_path, [row("a", "1024/0", {"x": 1}), row("h", "spec/code", {})], dict(run), ["a", "h"], "spec", False)
    d2 = merge(tmp_path, [row("h", "plain/code", {})], dict(run), ["h"], "plain", False)
    keys = [r["key"] for r in d2["rows"]]
    assert keys == ["1024/0", "spec/code", "plain/code"] and len(d2["runs"]) == 2 and d1["private"]["hosts"] == ["a"]
    d3 = merge(tmp_path, [row("a", "1024/0", {"x": 2})], dict(run), ["a"], "spec", False)
    assert [r["v"].get("x") for r in d3["rows"] if r["section"] == "a"] == [2]
    merge(tmp_path, [row("i", "warm/131072", {})], dict(run), ["i"], "spec", False)
    d4 = merge(tmp_path, [row("i", "fresh/131072", {})], dict(run), ["i"], "spec", False)
    assert sorted(r["key"] for r in d4["rows"] if r["section"] == "i") == ["fresh/131072", "warm/131072"]
    try:
        merge(tmp_path, [], dict(run, version="0.15.2"), ["a"], "spec", False)
        raise AssertionError("two versions merged")
    except SystemExit as exc:
        assert "--new" in str(exc)
    json.loads((tmp_path / "data.json").read_text())


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
