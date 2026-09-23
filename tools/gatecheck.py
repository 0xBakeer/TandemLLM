"""The ship rule on two row3 reports, as an exit code (OPS-19).

    python tools/gatecheck.py BASE.json OTHER.json [--mode noworse|adopt]

`noworse` (the default): exit 1 if any statistic of OTHER is RESOLVED worse than BASE under
`tools/row3.py --compare`'s own rule -- mean, p50, p90, max, TTFT, wall, and since SPD-35 tokens a
block and ms a block. `adopt`: additionally exit 2 unless the mean is RESOLVED better. One line per
statistic and a PASS/FAIL line, so a gate log reads the same whoever ran it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.row3 import verdicts  # noqa: E402


def check(base: dict, other: dict, mode: str = "noworse") -> tuple[int, list[str]]:
    vs = verdicts(base, other)
    lines = []
    for v in vs:
        if v["verdict"] == "n/a":
            lines.append(f"  {v['stat']:<12} n/a")
        else:
            lines.append(f"  {v['stat']:<12} {v['base']:9.2f} -> {v['other']:9.2f} "
                         f"({v['delta']:+6.1f} %, noise {v['noise']:4.1f} %)  {v['verdict']}")
    worse = [v["stat"] for v in vs if v.get("worse")]
    mean = next(v for v in vs if v["stat"] == "mean")
    if worse:
        rc, why = 1, "RESOLVED-WORSE: " + ", ".join(worse)
    elif mode == "adopt" and not (mean.get("resolved") and not mean.get("worse")):
        rc, why = 2, "nothing resolved-worse, but the mean is not resolved better"
    else:
        rc, why = 0, "nothing resolved-worse" + (", mean resolved better" if mode == "adopt" else "")
    lines.append(f"[gate] {other['label']} vs {base['label']} ({mode}): "
                 f"{'PASS' if rc == 0 else 'FAIL'} -- {why}")
    return rc, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--mode", default="noworse", choices=("noworse", "adopt"))
    a = ap.parse_args()
    base, other = (json.load(open(p)) for p in (a.base, a.other))
    rc, lines = check(base, other, a.mode)
    print("\n".join(lines))
    sys.exit(rc)


if __name__ == "__main__":
    main()
