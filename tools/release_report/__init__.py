"""The benchmark report of a TandemLLM server (docs/benchmark-report.md): one self-contained HTML page
of what the bare engine does on this box, for any model the server serves.

    python -m tools.release_report run [--url http://127.0.0.1:8000] --out DIR [--quick] [--only S ...]
    python -m tools.release_report run --ref vX.Y.Z --pack ID --out DIR      launch mode: a fresh server
    python -m tools.release_report plan [--url URL] [--quick]                 what a run would do, and how long
    python -m tools.release_report render DIR/data.json --out report.html
    python -m tools.release_report publish DIR --tag vX.Y.Z                   the maintainer's upload

TandemLLM bare engine: raw /v1/completions, no conductor or chat layer. Every request goes to
`/v1/completions` with `X-Tandem: raw`, its prompt rendered from the model's chat template on the
client and checked id for id against `/v1/tokenize`; a response without `X-Tandem: raw` makes its row
INVALID. `run` and `render` talk to the local server only.
"""

from __future__ import annotations

#: data.json's schema version
SCHEMA = 1
#: the tool's version (data.json names it)
VERSION = "1.0"
#: the one line the page and the doc carry near the top
BARE = "TandemLLM bare engine: raw /v1/completions, no conductor or chat layer."

#: the sections, in the order a full run measures them and the page shows them:
#: id -> (title, module under tools/release_report/sections)
SECTIONS = {
    "a": ("Decode by prompt length", "lengths"),
    "b": ("Time to the first token", "lengths"),
    "c": ("Requests at once", "concurrency"),
    "d": ("Chats on one new system prompt", "shared"),
    "e": ("Latency under load and a returning conversation", "load"),
    "f": ("Decode after a long prompt", "longdecode"),
    "g": ("The card tool", "card"),
    "h": ("Output fidelity", "fidelity"),
    "i": ("Long prompts on a warm engine", "warm"),
    "j": ("Capability (multiple choice)", "capability"),
    "k": ("How it was run", None),
}
#: the order a full run measures them: the warm repeat last, after the rest of the suite
RUN_ORDER = ("a", "c", "d", "e", "f", "g", "h", "i", "j")
