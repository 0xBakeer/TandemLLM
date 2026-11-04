"""One measuring method (docs/measurement.md): one runner, one row list, a record store, and three views
of its records. The release report, the weight cards' numbers and the quality gate's verdict table are
views; none of them measures anything itself.

    python -m tools.measure rows [--view report|card|gate]                 the row list
    python -m tools.measure plan --view report --out DIR [--url URL]       rows -> record or hold
    python -m tools.measure run --url URL --rows a,b | --view V --out DIR  measure what DIR lacks
    python -m tools.measure merge --out DIR --gate gate-report.json | --report data.json
    python -m tools.measure render report|card --out DIR [--pack P] [--html F]
    python -m tools.measure render gate --base DIR --head DIR

A record is keyed by (row id, build, pack, box, knobs hash). `run` measures only the rows the store
does not hold for the server's key (`--force` measures again), on the bare engine: raw
/v1/completions with `X-Tandem: raw`, every response checked; a row of N requests at once carries the
server's own proof that N decoded together, else it is INVALID. An imported row keeps its source
(tool, date, path) and the page shows it.
"""

from __future__ import annotations

#: a record's schema version and the tool's version (a record names both)
SCHEMA = 1
VERSION = "1.0"

#: the limit knobs a record's key hashes (tools/gate/server.py `knobs` reads them from the server);
#: the version, environment and flags are recorded beside, not hashed
KNOB_KEYS = ("lanes", "max_running", "max_queue", "max_len", "windows", "prefill_chunk", "prefix_cache",
             "prefix_chunk", "cache_gb", "ceiling_gb", "peak_gb", "suffix_store", "response_cache",
             "drafter", "corrector_loaded", "admission_floor_gib")

#: one hold on the reference box: the measuring part, the server starts and live's stop and restore
HOLD_MAX_MIN = 62
SERVER_START_MIN = 3.0
HOLD_OVERHEAD_MIN = 9.0

VIEWS = ("report", "card", "gate")
