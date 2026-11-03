"""prefill: cold solo prompts at the profile's depths (2k, 8k; at full 32k and 64k where they fit),
with tools/parallel_card.py's prefill phase: counted with /v1/tokenize, one token out, a nonce line
each, `cache_prompt: false`; prefill tok/s = prompt tokens / time to first token. A request the
server served from a cache (cached tokens > 0) is not a cold prefill: the row is INVALID.

Then the burst: N cold prompts of the burst's size released together, one token out each; the
median first token of each round, warm-up discarded. The proof is the server's own count: at least
N requests running at once during the round."""

from __future__ import annotations

import argparse
import statistics

from tools import parallel_card as pc
from tools.gate.client import released_together
from tools.gate.suites import Ctx, Plan, pooled
from tools.gate.verdict import skipped, speed_row

NAME = "prefill"


def measure(ctx: Ctx) -> dict:
    depths = ctx.profile.prefill_depths(ctx.level)
    a = argparse.Namespace(prefill_depths=",".join(map(str, depths)), prefill_runs=ctx.profile.rounds)
    pc.wait_idle(ctx.base)
    pre = pc.run_prefill(ctx.cl, ctx.pj, a, lambda *x: ctx.p(*x))
    out = {"depths": {d: {"runs": [{k: r.get(k) for k in ("prompt_tokens", "ttft_s", "prefill_tok_s",
                                                            "cached_tokens", "planned_tokens")}
                                   for r in v["runs"]]} for d, v in pre["depths"].items()},
           "tokenize_offset": pre["tokenize_offset"]}
    n, depth = ctx.profile.burst()
    k, _ = pc.prefill_words(ctx.base, ctx.pj, depth, f"Burst {'0' * 32}.\n", pre["tokenize_offset"])
    words = pc.filler_words(ctx.pj, depth)
    rounds = []
    for r in range(ctx.profile.warmups + ctx.profile.rounds):
        prompts = [ctx.cl.render([{"role": "user", "content": f"Burst {pc.nonce('b', r, i)}.\n"
                                   + " ".join(words[:k])}]) for i in range(n)]
        pc.wait_idle(ctx.base)
        got = released_together(ctx.cl, prompts, 1, [300 + i for i in range(n)], poll=0.05)
        ttfts = [s["ttft_s"] for s in got["streams"] if s.get("ttft_s") is not None]
        rec = {"warm": r < ctx.profile.warmups, "ttft_median_s": statistics.median(ttfts) if ttfts else None,
               "errors": got["errors"], "proof": got["proof"], "n": n,
               "cached": sum(int(s.get("cached_tokens") or 0) for s in got["streams"]),
               "valid": not got["errors"] and len(ttfts) == n and got["proof"]["max_running"] >= n}
        rounds.append(rec)
        ctx.p(f"burst {n} x {depth}: TTFT median {rec['ttft_median_s']} s, running at once "
              f"{got['proof']['max_running']}, valid {rec['valid']}")
    out["burst"] = {"n": n, "depth": depth, "rounds": rounds}
    return out


def rows(raw: dict, plan: Plan) -> list[dict]:
    base, head = pooled(raw, "base"), pooled(raw, "head")
    if not base or not head:
        return [skipped("prefill.all", NAME, "all depths", "an arm did not run")]
    out = []
    eff = plan.profile.min_effect_pct
    for d in base[0]["depths"]:
        def runs(side):
            return [r for x in side for r in x["depths"].get(d, {}).get("runs", [])]
        b, h = runs(base), runs(head)
        hit = [r for r in b + h if r.get("cached_tokens")]
        bad = f"{len(hit)} request(s) were served from a cache: not a cold prefill" if hit else None
        tok = int(statistics.median([r["prompt_tokens"] for r in b + h if r.get("prompt_tokens")] or [0]))
        out.append(speed_row(f"prefill.{int(d) // 1024}k", NAME, f"solo {tok} tokens, prefill",
                             "tok/s", True, [r["prefill_tok_s"] for r in b], [r["prefill_tok_s"] for r in h],
                             eff, invalid=bad))
    bb = [r for x in base for r in x["burst"]["rounds"] if not r["warm"]]
    hb = [r for x in head for r in x["burst"]["rounds"] if not r["warm"]]
    n, depth = base[0]["burst"]["n"], base[0]["burst"]["depth"]
    bad = None
    for side, rs in (("base", bb), ("head", hb)):
        inv = [r for r in rs if not r["valid"]]
        if inv:
            bad = (f"{len(inv)} {side} burst round(s) without {n} running at once (most seen "
                   f"{max(r['proof']['max_running'] for r in inv)}) or with errors")
            break
        if any(r["cached"] for r in rs):
            bad = f"a {side} burst request was served from a cache"
            break
    out.append(speed_row(f"prefill.burst{n}x{depth // 1024}k", NAME,
                         f"{n} x {depth // 1024}k at once, first token median", "s", False,
                         [r["ttft_median_s"] for r in bb], [r["ttft_median_s"] for r in hb], eff,
                         invalid=bad, proof={"n": n}))
    return out
