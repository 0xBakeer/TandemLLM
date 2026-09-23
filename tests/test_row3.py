"""The decisive row's arithmetic, without a board.

`tools/row3.py` exists because a single row decided nothing in phase 9, and the thing it adds on
top of the row is arithmetic: one rate formula per request, a median across runs, a spread beside
the median, and a verdict that refuses to call a difference smaller than its own noise a result.
All four are wrong in ways that would be invisible on the board -- a warm-up counted as a request
moves the mean by a few per cent and looks like a change, and a verdict that reads `>=` where it
means `>` calls a tie a win. So they are tested here, on records made by hand.
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.row3 import across, row_stats, server_cmd                # noqa: E402


def _req(n, ttft_ms, decode_ms, warmup=False, status="ok"):
    return {"warmup": warmup, "status": status, "completion_tokens": n,
            "ttft_ms": ttft_ms, "e2e_ms": ttft_ms + decode_ms}


def _record(reqs, duration_s=100.0):
    return {"raw": {"payload": {"requests": reqs, "selected_iteration": 0,
                                "iterations": [{"duration_s": duration_s}]}}}


def test_formula_subtracts_the_prefill_token():
    """257 tokens in 2.56 s of decode is 100 tok/s only if the first token is not decode's."""
    st = row_stats(_record([_req(257, 800.0, 2560.0)]))
    assert abs(st["mean"] - 100.0) < 1e-9, st["mean"]
    assert st["ttft_p50_ms"] == 800.0


def test_warmups_and_failures_are_not_the_row():
    """Three warm-ups and a failure are in every atlas record; none of them is a measurement."""
    reqs = [_req(101, 900.0, 10_000.0, warmup=True),                # 10 tok/s
            _req(101, 900.0, 10_000.0, warmup=True),
            _req(101, 900.0, 10_000.0, status="error"),
            _req(101, 500.0, 1000.0),                               # 100 tok/s
            _req(101, 500.0, 2000.0)]                               # 50 tok/s
    st = row_stats(_record(reqs))
    assert st["n"] == 2, st
    assert abs(st["mean"] - 75.0) < 1e-9
    assert abs(st["p50"] - 75.0) < 1e-9
    assert abs(st["max"] - 100.0) < 1e-9


def test_a_request_with_one_token_is_not_a_rate():
    """`(n - 1) / decode` is zero over zero for a single-token answer; it is dropped, not counted."""
    st = row_stats(_record([_req(1, 700.0, 0.0), _req(11, 700.0, 1000.0)]))
    assert st["n"] == 1 and abs(st["mean"] - 10.0) < 1e-9


def test_percentiles_are_taken_on_the_sorted_rates():
    rates = [_req(n + 1, 600.0, 1000.0) for n in range(1, 11)]       # 1..10 tok/s
    st = row_stats(_record(rates))
    assert abs(st["p50"] - 5.5) < 1e-9
    assert abs(st["p90"] - 9.0) < 1e-9, st["p90"]
    assert abs(st["max"] - 10.0) < 1e-9


def test_median_and_spread_across_runs():
    """Phase 9's own three RC rows: the median is the middle one and the spread is what the first
    row would have been believed to have proved."""
    runs = [{"mean": 53.74}, {"mean": 59.06}, {"mean": 59.30}]
    got = across(runs, "mean")
    assert got["median"] == 59.06
    assert got["min"] == 53.74 and got["max"] == 59.30
    assert abs(got["spread_pct"] - 100.0 * (59.30 - 53.74) / 59.06) < 1e-9
    assert got["spread_pct"] > 9.0                                  # the 10 % that killed the A/B


def test_spread_of_one_run_is_zero_and_says_nothing():
    """One run has no spread, which is exactly the phase-9 mistake stated as arithmetic: the
    verdict function must not be allowed to read 0 % noise off a single row."""
    got = across([{"mean": 59.06}], "mean")
    assert got["spread_pct"] == 0.0 and got["median"] == 59.06


def test_a_missing_statistic_is_absent_rather_than_zero():
    assert across([{"mean": None}], "mean") == {}
    assert across([], "mean") == {}


def _verdict(b, o):
    """The verdict as `compare` computes it, over two lists of per-run values."""
    B, O = across([{"x": v} for v in b], "x"), across([{"x": v} for v in o], "x")
    delta = 100.0 * (O["median"] - B["median"]) / B["median"]
    noise = max(B["spread_pct"], O["spread_pct"])
    disjoint = O["min"] > B["max"] or O["max"] < B["min"]
    return "RESOLVED" if abs(delta) > noise and disjoint else "not resolved"


def test_the_verdict_is_strict():
    """The 0.2 % tree-mode difference against a 10 % spread must never read as a result, and
    neither may a difference the exact size of the noise."""
    assert _verdict([53.74, 59.06, 59.30], [53.80, 59.20, 59.40]) == "not resolved"
    assert _verdict([100.0, 100.0, 100.0], [110.0, 110.0, 110.0]) == "RESOLVED"
    # big delta, overlapping runs: one configuration did NOT beat the other every time
    assert _verdict([100.0, 100.0, 160.0], [110.0, 150.0, 150.0]) == "not resolved"
    # disjoint but only by less than the noise the runs themselves carry
    assert _verdict([100.0, 130.0, 140.0], [141.0, 150.0, 160.0]) == "not resolved"


def test_wall_comes_from_the_selected_iteration():
    rec = _record([_req(11, 600.0, 1000.0)], duration_s=261.9)
    rec["raw"]["payload"]["iterations"] = [{"duration_s": 999.0}, {"duration_s": 261.9}]
    rec["raw"]["payload"]["selected_iteration"] = 1
    assert row_stats(rec)["wall_s"] == 261.9


def _ns(**kw):
    from pathlib import Path
    from types import SimpleNamespace
    base = dict(python=Path("py"), port=8001, max_len=4096, len_fixed=0, budget=16,
                repo=Path("/r"), nvfp4="nv", head="hd", len_latch=True, server_arg=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_row_never_reads_the_persistent_suffix_store():
    """Every server on the board opens the same store, it keeps prompts and answers, and the row's
    prompts are fixed: left on, the row reads its own previous answers (2026-09-23 10:37)."""
    cmd = server_cmd(_ns())
    assert "--suffix-store=" in cmd
    assert "--no-session-cache" in cmd and "--no-prefix-cache" in cmd


def test_the_store_can_be_put_back_to_reproduce_an_old_row():
    assert "--suffix-store=" not in server_cmd(_ns(with_suffix_store=True))
    # and a --server-arg comes after it, so an explicit store path still wins
    cmd = server_cmd(_ns(server_arg=["--suffix-store=/tmp/s"]))
    assert cmd.index("--suffix-store=/tmp/s") > cmd.index("--suffix-store=")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
