"""The quality gate for pull requests (docs/quality-gate.md).

    python -m tools.gate run [--suite auto|quick|quick+long|full] [--base origin/main] [--head HEAD]
                             [--pack ID] [--hw auto] [--out DIR]
    python -m tools.gate map [--base REF] [--head REF]          the level and suites a diff needs
    python -m tools.gate hw                                      the hardware profile this box gets
    python -m tools.gate explain REPORT.json ROW "reason"       a written reason for a WORSE or
                                                                 CHANGED_TEXT row
    python -m tools.gate check ...                               the CI check (.github/workflows/gate.yml)

One command on the contributor's GPU measures the base (main) and the head (the branch) on the same
box, one fresh server per arm, and writes gate-report.json and gate-report.md. The CI check refuses a
pull request whose report is missing, stale or failing.

Its measurements are tools/measure records too (OUT/records/base and OUT/records/head): python -m
tools.measure render gate --base OUT/records/base --head OUT/records/head prints the same verdict
table from them, and a release page imports them instead of measuring again (docs/measurement.md).
"""

from __future__ import annotations

#: the tool's version; a report names it, and the CI check refuses a report of another major version
VERSION = "1.0"
#: the report's schema version
SCHEMA = 1
#: the line that starts the report's JSON block in a pull request's body or comment
MARKER = "<!-- tandem-gate-report -->"

#: a row's verdicts
PASS, WORSE, CHANGED_TEXT, FAIL, INVALID, SKIPPED = (
    "PASS", "WORSE", "CHANGED_TEXT", "FAIL", "INVALID", "SKIPPED")
VERDICTS = (PASS, WORSE, CHANGED_TEXT, FAIL, INVALID, SKIPPED)
#: verdicts that block a merge outright, and the ones that need a written reason
BLOCKING = (FAIL, INVALID)
NEEDS_REASON = (WORSE, CHANGED_TEXT)
#: the shortest reason the CI check accepts for a WORSE or CHANGED_TEXT row
MIN_REASON = 20

#: gate levels, lowest first, and the suites each one runs
LEVELS = ("none", "quick", "quick+long", "full")
SUITES = ("lossless", "decode", "prefill", "long", "api", "kernels", "stability")
LEVEL_SUITES = {
    "none": (),
    "quick": ("lossless", "decode", "prefill", "long", "api", "kernels"),
    "quick+long": ("lossless", "decode", "prefill", "long", "api", "kernels"),
    "full": SUITES,
}


def level_index(level: str) -> int:
    return LEVELS.index(level)


def max_level(levels) -> str:
    return max(levels, key=level_index, default="none")
