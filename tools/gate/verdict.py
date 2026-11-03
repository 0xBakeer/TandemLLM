"""A row's verdict, and the report's.

Speed rows use tools/row3.py's rule (`resolve`): a difference is RESOLVED only when it is larger than
the run-to-run noise and the run ranges of the two sides do not overlap, with at least two runs a
side. A resolved difference smaller than the hardware profile's `min_effect_pct` is reported but
passes: docs/measurement.md puts what a row can settle at about 3 % of the median.

A row whose concurrency proof failed is INVALID and carries no number; so is a row with fewer than
two runs a side (its noise is unknown).
"""

from __future__ import annotations

import statistics

from tools.gate import (BLOCKING, CHANGED_TEXT, FAIL, INVALID, NEEDS_REASON, PASS, SKIPPED,
                        WORSE)
from tools.row3 import across, resolve


def summary(runs: list[float]) -> dict:
    """The median, range and spread of a side's runs (tools/row3.py's `across`), with the runs."""
    xs = [float(x) for x in runs if x is not None]
    if not xs:
        return {"n": 0, "runs": []}
    s = across([{"v": x} for x in xs], "v")
    return {"n": len(xs), "median": round(s["median"], 4), "min": round(s["min"], 4),
            "max": round(s["max"], 4), "spread_pct": round(s["spread_pct"], 3),
            "runs": [round(x, 4) for x in xs]}


def row(rid: str, suite: str, metric: str, verdict: str, why: str, **extra) -> dict:
    out = {"id": rid, "suite": suite, "metric": metric, "verdict": verdict, "why": why}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def speed_row(rid: str, suite: str, metric: str, unit: str, higher_is_better: bool,
              base_runs: list, head_runs: list, min_effect_pct: float = 3.0,
              invalid: str | None = None, proof: dict | None = None) -> dict:
    """A speed metric's row: base and head summaries, delta, noise, verdict."""
    better = "higher" if higher_is_better else "lower"
    b, h = summary(base_runs), summary(head_runs)
    common = {"unit": unit, "better": better, "base": b, "head": h, "proof": proof}
    if invalid:
        return row(rid, suite, metric, INVALID, invalid, unit=unit, better=better, proof=proof)
    if b["n"] < 2 or h["n"] < 2:
        return row(rid, suite, metric, INVALID,
                   f"{b['n']} base and {h['n']} head runs: two or more a side are needed to "
                   "know the noise", unit=unit, better=better, proof=proof)
    if b["median"] <= 0:
        return row(rid, suite, metric, INVALID, "the base median is not positive", **common)
    r = resolve(b, h, higher_is_better)
    delta, noise = round(r["delta"], 2), round(r["noise"], 2)
    if not r["resolved"]:
        verdict, why = PASS, "not resolved: the difference is within the run-to-run noise"
    elif not r["worse"]:
        verdict, why = PASS, "resolved better"
    elif abs(delta) < min_effect_pct:
        verdict, why = PASS, (f"resolved worse by {abs(delta):.2f} %, under the "
                              f"{min_effect_pct:g} % a row can settle")
    else:
        verdict, why = WORSE, f"resolved worse by {abs(delta):.2f} % (noise {noise:.2f} %)"
    return row(rid, suite, metric, verdict, why, delta_pct=delta, noise_pct=noise,
               disjoint=r["disjoint"], **common)


def first_difference(a: list, b: list) -> int | None:
    """The first index where two token lists differ, or None when they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def text_row(rid: str, suite: str, metric: str, base: dict, head: dict,
             what: str = "base and head") -> dict:
    """A lossless row: the two sides' greedy outputs must be the same token ids.

    `base` and `head` are {"ids": [...], "text_sha256": ..., "finish": ..., "gap": {pos: logprob
    gap}} for one prompt; `gap` (optional) is the top-2 logprob gap where a side recorded it."""
    if base.get("error") or head.get("error"):
        side = "base" if base.get("error") else "head"
        return row(rid, suite, metric, FAIL, f"the {side} request failed: "
                   f"{(base.get('error') or head.get('error'))[:200]}")
    bi, hi = base.get("ids") or [], head.get("ids") or []
    at = first_difference(bi, hi)
    tokens = {"base_tokens": len(bi), "head_tokens": len(hi)}
    if at is None:
        return row(rid, suite, metric, PASS, f"{what}: the same {len(bi)} token ids", **tokens)
    gap = (base.get("gaps") or {}).get(str(at))
    near = f", the base's top-2 gap there {gap:.4f} nats" if gap is not None else ""
    return row(rid, suite, metric, CHANGED_TEXT,
               f"{what}: the token ids differ from position {at}{near}",
               first_diff=at, base_at=_at(bi, at), head_at=_at(hi, at), gap_at_diff=gap, **tokens)


def _at(ids: list, i: int):
    return ids[i] if i < len(ids) else None


def case_row(rid: str, suite: str, metric: str, base_ok: bool | None, head_ok: bool,
             detail: str = "") -> dict:
    """A pass/fail check on both sides. A check the base fails too is not the change's: SKIPPED
    with that reason. `base_ok` None: the check runs on the head only."""
    detail = f": {detail}" if detail else ""
    if head_ok:
        return row(rid, suite, metric, PASS, "passes" + detail)
    if base_ok is False:
        return row(rid, suite, metric, SKIPPED, "fails on the base as well, so not this change's"
                   + detail)
    return row(rid, suite, metric, FAIL, ("fails on the head, passes on the base" if base_ok
                                          else "fails on the head") + detail)


def skipped(rid: str, suite: str, metric: str, why: str) -> dict:
    return row(rid, suite, metric, SKIPPED, why)


ORDER = (FAIL, INVALID, WORSE, CHANGED_TEXT, PASS)


def overall(rows: list[dict]) -> str:
    """The report's verdict: FAIL when any row is FAIL or INVALID, else WORSE, else CHANGED_TEXT,
    else PASS (SKIPPED rows do not count)."""
    seen = {r["verdict"] for r in rows}
    if seen & set(BLOCKING):
        return FAIL
    for v in (WORSE, CHANGED_TEXT):
        if v in seen:
            return v
    return PASS


def unexplained(rows: list[dict], reasons: dict) -> list[str]:
    """The WORSE / CHANGED_TEXT rows without a written reason."""
    from tools.gate import MIN_REASON
    return [r["id"] for r in rows if r["verdict"] in NEEDS_REASON
            and len((reasons.get(r["id"]) or "").strip()) < MIN_REASON]


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
