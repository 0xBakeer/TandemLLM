"""CPU tests for tools/accept_hist.py and the histogram the length router writes for it.

The histogram travels as one token of a log line and is read back by a regex, so the one thing that
must hold is the round trip: what `LengthRouter.report()` writes is what `parse_log` reads, and the
cap rates are computed over the blocks that were actually counted.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.lenrouter import _hist_str  # noqa: E402
from tools.accept_hist import parse_hist, parse_log, summarize  # noqa: E402


def test_the_histogram_round_trips_through_the_log_token():
    h = {16: {16: 5, 4: 5}, 8: {2: 3, 8: 1}}
    tok = _hist_str(h)
    assert " " not in tok
    assert parse_hist(tok) == h
    assert parse_hist(_hist_str({})) == {}


def test_parse_log_reads_only_lines_that_carry_it():
    text = ("[req] something\n"
            "[drafter] lenrouter 8x0 (0.00 tok/block) widths {} commits - cap arm 0 depth 0\n"
            "[drafter] lenrouter 8x4 ... widths {8: 4} commits 8:2x3,8x1 cap arm 1 depth 2\n"
            "[drafter] an old line without the histogram\n")
    recs = parse_log(text)
    assert len(recs) == 2
    assert recs[1] == {"commits": {8: {2: 3, 8: 1}}, "cap_arm": 1, "cap_depth": 2}


def test_summarize_sums_requests_and_rates_over_counted_blocks():
    recs = [{"commits": {16: {16: 2, 4: 2}}, "cap_arm": 2, "cap_depth": 2},
            {"commits": {16: {16: 1}, 8: {3: 3}}, "cap_arm": 1, "cap_depth": 2}]
    s = summarize(recs)
    assert s["blocks"] == 8
    assert abs(s["mean"] - (16 * 3 + 4 * 2 + 3 * 3) / 8) < 1e-12
    assert s["arms"][16]["full"] == 3 and abs(s["arms"][16]["full_pct"] - 60.0) < 1e-9
    assert s["arms"][8]["full"] == 0
    assert abs(s["cap_arm_pct"] - 37.5) < 1e-9 and abs(s["cap_depth_pct"] - 50.0) < 1e-9


def test_summarize_of_nothing_is_zero_not_a_division_error():
    s = summarize([])
    assert s["blocks"] == 0 and s["mean"] == 0.0 and s["cap_arm_pct"] == 0.0


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
