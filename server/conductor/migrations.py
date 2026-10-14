"""Versioned changes to the conductor documents a box already stores, run once each at the engine's
start (after the shipped defaults are stored, server/conductor/shipped.py).

A shipped default reaches only a box that has no version of the document; a fix to a stored one
needs a migration. Each migration has an id, the document it reads, and a function that returns
the changed document, or None when there is nothing to change. A change is a new version of the
document (note `migration <id>`, author `migration`), so it is visible in the version list and
undone by a restore; the run is one log line and one journal row (`config`, decision
`migration`). `<conductor root>/migrations.json` records every id that ran, changed or not, so a
migration never runs twice: a switch the owner removes after it ran stays removed.

    2026-10-14-chat-app-memory   SRV-471: the chat-app profile's `memory` switch, "on" when the
                                 stored document has none (absent is off); an explicit value,
                                 "off" included, stays
    2026-10-14-chat-memory       SRV-471: the same for the chat, coding and hermes profiles
    2026-10-14-coding-memory
    2026-10-14-hermes-memory
"""

from __future__ import annotations

import datetime as _dt
import json
import os

STATE = "migrations.json"


def _memory_on(doc: dict) -> dict | None:
    if not isinstance(doc, dict) or "memory" in doc:
        return None
    return dict(doc, memory="on")


# (id, kind, name, change)
MIGRATIONS = [
    ("2026-10-14-chat-app-memory", "profiles", "chat-app", _memory_on),
    ("2026-10-14-chat-memory", "profiles", "chat", _memory_on),
    ("2026-10-14-coding-memory", "profiles", "coding", _memory_on),
    ("2026-10-14-hermes-memory", "profiles", "hermes", _memory_on),
]


def _state(root: str) -> dict:
    try:
        with open(os.path.join(root, STATE)) as f:
            got = json.load(f)
        return got if isinstance(got, dict) and isinstance(got.get("applied"), dict) else \
            {"applied": {}}
    except (OSError, ValueError):
        return {"applied": {}}


def _save(root: str, state: dict) -> None:
    os.makedirs(root, mode=0o700, exist_ok=True)
    path = os.path.join(root, STATE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def run(store, journal=None, migrations=None) -> list[dict]:
    """Every migration of `migrations` (default MIGRATIONS) that has not run on this store's
    root. Returns one record per migration run now: {id, kind, name, result, version}; result is
    `changed`, `unchanged` (the document already had it) or `absent` (no stored version; the
    shipped default, when stored later, has it). A read-only store runs nothing."""
    if getattr(store, "readonly", False):
        return []
    state = _state(store.root)
    done = []
    for mid, kind, name, change in (MIGRATIONS if migrations is None else migrations):
        if mid in state["applied"]:
            continue
        rec = {"id": mid, "kind": kind, "name": name, "result": "absent", "version": None}
        try:
            if store._pointer(store._dir(kind, name)) is not None:
                v, doc = store.get(kind, name)
                new = change(doc)
                if new is None:
                    rec.update(result="unchanged", version=v)
                else:
                    # the change leaves the tasks as they are, so the editor's check that an act
                    # task's role is up (offline at the start) does not apply to it
                    up, store.roles_up = store.roles_up, None
                    try:
                        rec.update(result="changed", version=store.put(
                            kind, name, new, note=f"migration {mid}", author="migration", base=v))
                    finally:
                        store.roles_up = up
        except Exception as exc:                                   # noqa: BLE001
            # not recorded as run: the next start tries again
            rec.update(result="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            print(f"[conductor] migration {mid} failed: {rec['error']}", flush=True)
            done.append(rec)
            continue
        state["applied"][mid] = {"at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), "result": rec["result"], "version": rec["version"]}
        _save(store.root, state)
        print(f"[conductor] migration {mid}: {kind}{'/' + name if name else ''} "
              f"{rec['result']}" + (f" (v{rec['version']})" if rec["version"] else ""), flush=True)
        if journal is not None:
            try:
                journal.note("config", kind, decision="migration",
                             detail={"id": mid, "name": name, "result": rec["result"],
                                     "version": rec["version"]})
            except Exception:                                      # noqa: BLE001
                pass
        done.append(rec)
    return done
