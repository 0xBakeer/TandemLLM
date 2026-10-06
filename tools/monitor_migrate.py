"""Add the session monitor's new tasks to a box's saved `monitor` document (0.3.43-test).

A box that edited its monitor policy (the Conductor view's Monitor screen, or a migration) runs
its stored document, not ops/supervisor.yaml, so a task the file gains never reaches it. This
copies the named tasks (default: S5, the turn budget of SRV-179 and SRV-181) from the policy file
into the current document when the document lacks them, raises the document's `version` to the
file's when lower, and stores the result as a new version with a note; the old versions stay and
`restore` brings one back. A task the document has already keeps every value it has (the box's
own edit wins) and gains only the parameters the file has and it lacks (0.3.45-test: S5's verify
arm, `verify_s`, `verify_scratch_writes`, `verify_text`). Idempotent.

  python tools/monitor_migrate.py --doc ~/.qwen38-spark-engine/conductor/monitor/0002.json
  python tools/monitor_migrate.py --put http://127.0.0.1:8000 [--tasks S5] [--dry]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from packages_migrate import _call, _token  # noqa: E402
from server.conductor.monitor import policy as policy_mod  # noqa: E402

NOTE = ("S5 turn budget (SRV-179, SRV-181): the same problem stated 3 times, a build turn's setup "
        "before its first file, a chat turn past its minutes")
NOTE_7 = ("S5 verify (SRV-195, policy 7): the deliverable exists and the steps since its last change "
          "only built test tools -> one note to check once and report")


def migrate_doc(doc: dict, tasks=("S5",), source: dict | None = None) -> tuple[dict, list[str]]:
    """(the document with the missing tasks added, what changed)."""
    src = source or policy_mod.load().data
    new = json.loads(json.dumps(doc))
    changed = []
    for tid in tasks:
        if tid not in src["tasks"]:
            continue
        if tid in new["tasks"]:
            have = new["tasks"][tid]
            if have.get("rule") != src["tasks"][tid]["rule"]:
                continue                                 # the box's own task under that id
            added = [k for k in src["tasks"][tid] if k not in have]
            for k in added:
                have[k] = json.loads(json.dumps(src["tasks"][tid][k]))
            if added:
                changed.append(f"task {tid}: {', '.join(added)} added")
            continue
        new["tasks"][tid] = json.loads(json.dumps(src["tasks"][tid]))
        changed.append(f"task {tid} added ({src['tasks'][tid]['rule']}, {src['tasks'][tid]['mode']})")
    if changed and int(new.get("version", 0)) < int(src["version"]):
        changed.append(f"version {new.get('version')} -> {src['version']}")
        new["version"] = src["version"]
    if changed:
        policy_mod.Policy(new)                       # validates, or raises with the field
    return new, changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--doc", help="a monitor version file or document: print what would change")
    ap.add_argument("--put", help="a server's address: migrate its current monitor document")
    ap.add_argument("--tasks", default="S5")
    ap.add_argument("--token-file", default="~/.qwen38-spark-engine/secrets.env")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    tasks = tuple(t.strip() for t in a.tasks.split(",") if t.strip())
    if a.doc:
        with open(os.path.expanduser(a.doc)) as f:
            d = json.load(f)
        _, ch = migrate_doc(d.get("doc", d), tasks)
        print(json.dumps(ch, indent=1))
    if a.put:
        tok = _token(a.token_file)
        url = a.put.rstrip("/") + "/v1/conductor/monitor"
        cur = _call("GET", url, tok)
        new, ch = migrate_doc(cur["doc"], tasks)
        print(json.dumps(ch, indent=1))
        if not ch or a.dry:
            return 0
        note = NOTE_7 if any(": verify_" in c for c in ch) else NOTE
        got = _call("PUT", url, tok, {"doc": new, "note": note, "base_version": cur.get("version")})
        print(f"[monitor_migrate] stored monitor v{got.get('version')} (was v{cur.get('version')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
