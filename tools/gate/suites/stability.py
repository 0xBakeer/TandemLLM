"""stability (full): `stability_minutes` of tools/soak.py's mixed traffic on each side (several
languages, thinking on and off, multi-turn sessions, tool calls, sampled requests, clients that hang
up on purpose), with /health every 5 s. Rows: no failed request (the deliberate hang-ups aside) and
no new server error; the reserved memory under the declared peak with no admission over it; the
decode rate of the last third of the run against the first third, within `drift_max_pct`. The graph
pools and the collector's pauses, where /health reports them, are recorded beside."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

from tools import parallel_card as pc
from tools.gate.client import REPO, HealthPoller
from tools.gate.suites import Ctx, Plan, first
from tools.gate.verdict import case_row, skipped

NAME = "stability"


def pools(h: dict) -> dict:
    p = h.get("parallel") or {}
    jg = ((p.get("joint_stats") or {}).get("joint_graphs") or {})
    gc = (((p.get("round_runner") or {}).get("host_work") or {}).get("gc_stats")) or None
    return {"joint_graph_captures": jg.get("captures"), "kv_pool_committed_gib":
            (p.get("kv_pool") or {}).get("committed_gib"), "gc_stats": gc}


def measure(ctx: Ctx) -> dict:
    minutes = ctx.profile.stability_minutes
    lanes = (ctx.health.get("parallel") or {}).get("max_running") or 1
    js = ctx.out / f"soak-{ctx.side}.json"
    h0 = pc.health(ctx.base)
    cmd = [sys.executable, "-u", str(REPO / "tools" / "soak.py"), "--base-url", ctx.base + "/v1",
           "--minutes", str(minutes), "--concurrency", str(max(1, lanes - 1)), "--json-out", str(js)]
    t0 = time.monotonic()
    with HealthPoller(ctx.base, 5.0) as pol, open(ctx.out / f"soak-{ctx.side}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=minutes * 60 + 1800).returncode
    h1 = pc.health(ctx.base)
    try:
        recs = json.load(open(js)).get("requests") or []
    except (OSError, ValueError):
        recs = []
    done = [r for r in recs if not r.get("abandoned") and r.get("probe") is None]
    failed = [r for r in done if r.get("error") or (r.get("status") not in (200, None))]
    rates = sorted((r["t0"], r["tok_s"]) for r in done if (r.get("tok_s") or 0) > 0 and (r.get("tokens") or 0) >= 32)
    third = len(rates) // 3
    first_t = statistics.median([x for _, x in rates[:third]]) if third else None
    last_t = statistics.median([x for _, x in rates[-third:]]) if third else None
    e0, e1 = (h0.get("inflight") or {}).get("errors") or 0, (h1.get("inflight") or {}).get("errors") or 0
    out = {"rc": rc, "minutes": minutes, "wall_s": round(time.monotonic() - t0, 1), "requests": len(recs),
           "failed": len(failed), "failed_examples": [str(r.get("error"))[:120] for r in failed[:3]],
           "server_errors": e1 - e0, "drift_first": first_t, "drift_last": last_t,
           "drift_pct": round(100 * (last_t - first_t) / first_t, 2) if first_t and last_t else None,
           **pol.memory(), "pools_start": pools(h0), "pools_end": pools(h1)}
    ctx.p(f"stability: {out['requests']} requests, {out['failed']} failed, server errors +{out['server_errors']}, "
          f"drift {out['drift_pct']} %, reserved max {out['reserved_max_gb']} GB of {out['peak_gb']}")
    return out


def rows(raw: dict, plan: Plan) -> list[dict]:
    b, h = first(raw, "base"), first(raw, "head")
    if not h:
        return [skipped("stability.all", NAME, "mixed load", "the head did not run")]
    lim = plan.profile.drift_max_pct

    def clean(x):
        return x is not None and x["rc"] == 0 and x["requests"] > 0 and not x["failed"] and not x["server_errors"]

    def under(x):
        return x is not None and x["reserved_max_gb"] is not None and x["peak_gb"] is not None \
            and x["reserved_max_gb"] <= x["peak_gb"] and not x["over_peak_admits"]

    def steady(x):
        return x is not None and x["drift_pct"] is not None and x["drift_pct"] >= -lim

    m = h["minutes"]
    out = [case_row("stability.errors", NAME, f"{m} min mixed load: no failed request", b and clean(b), clean(h),
                    f"{h['requests']} requests, {h['failed']} failed, server errors +{h['server_errors']}"
                    + (f"; e.g. {h['failed_examples'][0]}" if h["failed_examples"] else ""))]
    if h["reserved_max_gb"] is None:
        out.append(skipped("stability.reserve", NAME, "reserve under the peak", "no reserved memory in /health"))
    else:
        out.append(case_row("stability.reserve", NAME, "reserve under the peak", b and under(b), under(h),
                            f"reserved max {h['reserved_max_gb']} GB, peak {h['peak_gb']} GB, over the "
                            f"peak {h['over_peak_admits']}; pools {json.dumps(h['pools_end'])[:160]}"))
    out.append(case_row("stability.drift", NAME, f"decode drift, last third vs first (limit -{lim:g} %)",
                        b and steady(b), steady(h),
                        f"{h['drift_first']} -> {h['drift_last']} tok/s ({h['drift_pct']} %)"))
    return out
