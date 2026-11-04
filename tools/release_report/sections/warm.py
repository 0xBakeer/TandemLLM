"""i: the longest prompts (~128k and ~192k, one request each) on a warm engine against a fresh one.
Run after the rest of the suite on the same server, the rows are `warm`; run first on a freshly started
server (at most FRESH requests served before, the probe included), they are `fresh`. Without a fresh
run the page sets the warm rows beside section a's rows of the same length. Each row says how many
requests the server had served before it."""

from __future__ import annotations

from tools.release_report.plan import ANSWER
from tools.measure.prompts import sized
from tools.release_report.sections import Ctx, check, engine_of, row, skips_for
from tools.release_report.sections.lengths import TASK

#: requests a server may have served before section i and still count as fresh (the run's probe)
FRESH = 2


def measure(ctx: Ctx) -> list[dict]:
    from tools import parallel_card as pc
    out = skips_for(ctx, "i", "i")
    h = pc.health(ctx.base)
    served = (h.get("inflight") or {}).get("served")
    state = "fresh" if served is not None and served <= FRESH else "warm"
    for n in ctx.plan.warm:
        prompt, _ = sized(ctx.cl, ctx.prose, n, TASK, tag=f"i{n}")
        rec = ctx.cl.stream_rec(prompt, ANSWER, seed=7)
        bad = check([rec])
        v = {k: rec.get(k) for k in ("prompt_tokens", "ttft_s", "decode_tok_s", "prefill_tok_s", "completion_tokens")}
        ctx.log(f"[i] {n} {state}: {v}, drafting {rec['engine'].get('drafting')}" + (f" INVALID {bad}" if bad else ""))
        out.append(row("i", f"{state}/{n}", v, {"target": n, "state": state, "served_before": served,
                                             "uptime_s": h.get("uptime_s"), "max_tokens": ANSWER},
                       valid=bad is None, why=bad, engine=[engine_of(rec)], ids=[rec.get("id")]))
    return out
