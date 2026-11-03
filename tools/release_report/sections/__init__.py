"""The report's sections. Each module's `measure(ctx)` sends its requests to the running server and
returns rows; a row is one measured case with its numbers (`v`), its conditions (`cond`), the
engine's own counts where the response carried them, and whether it counts (`valid`, `why`).

A row is INVALID when any of its responses was not served raw (no `X-Tandem: raw`), failed, or when
the proof of its concurrency is missing; an invalid row is shown with its reason, never as a number.
A case the plan left out is a row with `skipped` and the reason."""

from __future__ import annotations

import dataclasses
import statistics
from pathlib import Path


@dataclasses.dataclass
class Ctx:
    base: str
    cl: object
    health: dict
    plan: object
    prose: object
    out: Path
    log: object = print
    variant: str = "spec"
    args: object = None


def row(section: str, key: str, v: dict | None = None, cond: dict | None = None, valid: bool = True,
        why: str | None = None, engine: list | None = None, ids: list | None = None) -> dict:
    r = {"section": section, "key": key, "v": v or {}, "cond": cond or {}, "valid": bool(valid),
         "why": why}
    if engine is not None:
        r["engine"] = engine
    if ids:
        r["ids"] = [i for i in ids if i]
    return r


def skipped(section: str, key: str, why: str) -> dict:
    return {"section": section, "key": key, "skipped": why, "valid": False, "why": why, "v": {}, "cond": {}}


def check(recs: list[dict]) -> str | None:
    """Why these responses cannot count, or None: not raw, an error, or no tokens."""
    for r in recs:
        if r.get("error"):
            return f"a request failed: {r['error'][:200]}"
        if not r.get("raw"):
            return f"served through the chat layer ({r.get('raw_why')}): not the bare engine"
    for r in recs:
        if r.get("completion_tokens") == 0 and not r.get("left"):
            return "a request produced no tokens"
    return None


def engine_of(rec: dict) -> dict:
    return dict(rec.get("engine") or {})


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def skips_for(ctx: Ctx, prefix: str, section: str) -> list[dict]:
    return [skipped(section, k, why) for k, why in ctx.plan.skipped.items()
            if k == prefix or k.startswith(prefix + ".")]


def revalidate(rows: list[dict]) -> list[dict]:
    """The proof rule applied again from each row's recorded proof (it is deterministic from the data):
    a concurrency row whose server count shows more requests running than it sent overlapped with
    another request and is INVALID. Rows measured before the rule (2026-11-03) get it here."""
    for r in rows:
        p = (r.get("cond") or {}).get("proof") or {}
        n = (r.get("v") or {}).get("n") or (r.get("v") or {}).get("chats")
        if r.get("valid") and r.get("section") in ("c", "d") and n and p.get("max_running", 0) > n:
            r["valid"] = False
            r["why"] = (f"{p['max_running']} requests ran at once, more than the {n} sent: another request "
                        "overlapped")
    return rows
