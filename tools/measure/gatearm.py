"""A gate arm's measurements as tools/measure records: after each server of `python -m tools.gate run`, its
suite results (lossless, decode, prefill, long, stability) go into DIR/records/<side>, one run file a server,
one record a row (rows.yaml), the shapes the gate's hardware profile ran recorded with them. Two servers of a
side (--pairs 2) are two records of a key, the second listed beside the first; `render gate --base
DIR/records/base --head DIR/records/head` gives the gate's verdict table from them."""

from __future__ import annotations

import time

from tools.measure import VERSION, store as storemod

#: a suite's rows in a gate arm (the decode suite's levels map to conc.card.c<N>)
SUITE_ROWS = {
    "lossless": {"main": ["lossless.set"], "plain": ["lossless.plain"]},
    "long": {"main": ["long.131k.needle"], "wide": ["long.wide"]},
    "stability": {"main": ["stability.30m"]},
}


def rows_of(suite: str, variant: str, raw: dict) -> dict:
    """{row id: the part of the suite's raw result that row reads}."""
    if suite == "decode":
        out = {}
        for sid in raw:
            rid = {"c1": "decode.c1.short", "c1-1k": "decode.c1.1k"}.get(sid, f"conc.card.{sid}")
            out[rid] = {sid: raw[sid]}
        return out
    if suite == "prefill":
        from tools.measure.merge import KSIZE
        out = {}
        for d in raw.get("depths") or {}:
            rid = f"prefill.{KSIZE[int(d)]}" if int(d) in KSIZE else f"prefill.{d}t"   # a profile's own depth
            out[rid] = {"depths": {d: raw["depths"][d]}, "tokenize_offset": raw.get("tokenize_offset")}
        out["burst.8x2k"] = {"burst": raw.get("burst")}
        return out
    return {rid: raw for rid in (SUITE_ROWS.get(suite) or {}).get(variant, [])}


def record_arm(out_dir, arm, knobs: dict, raws: dict, profile_id: str, pack_id: str, sha: str | None,
               log=print) -> str | None:
    """One server's suites as one run file of DIR/records/<side>. `raws`: {suite: that server's raw}."""
    from tools.measure.store import Store
    recs: dict = {}
    build = storemod.build_of(knobs.get("version"))
    kh = storemod.knobs_hash(knobs)
    for suite, raw in raws.items():
        for rid, sub in rows_of(suite, arm.variant, raw).items():
            if sub is None:
                continue
            recs[rid] = {"row": rid, "build": build, "pack": pack_id, "box": profile_id, "knobs_hash": kh,
                         "knobs": storemod.limit_knobs(knobs), "version": knobs.get("version"), "valid": True,
                         "why": None, "skipped": None, "values": {}, "rows": [],
                         "raw": {"servers": [sub]}, "shape": f"the gate's {profile_id} profile, {suite} suite",
                         "source": {"tool": f"tools/gate via tools/measure {VERSION}", "how": "measured",
                                    "date": time.strftime("%Y-%m-%d"), "path": f"gate arm {arm.key}",
                                    "commit": sha, "side": arm.side}}
    if not recs:
        return None
    st = Store(out_dir / "records" / arm.side)
    run = {"id": f"gate-{arm.key.replace(':', '-')}-{time.strftime('%Y%m%d-%H%M%S')}-{arm.port}", "how": "measured",
           "tool": f"tools/gate via tools/measure {VERSION}", "commit": sha, "knobs": knobs, "variant": arm.variant,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hw_profile": profile_id}
    p = st.add_run(run, recs, imported=True)          # a second server of the side lists beside the first
    log(f"[gate] {arm.key}: {len(recs)} records into {p}")
    return str(p)
