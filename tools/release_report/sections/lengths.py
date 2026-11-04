"""a and b: one request at a time at each planned prompt length, shortest first. Real prose with a fresh
nonce first, greedy, thinking off, 512-token answers. a: decode = completion tokens after the first over
the streaming window (first to last text). b: prefill = prompt tokens (usage) over the time to first
token. Per request the engine's own counts: drafting (draft tokens > 0), tokens per pass
(`tokens_per_block`), its decode rate, and the slot from the engine log where one is given."""

from __future__ import annotations

from tools.release_report.plan import ANSWER, kfmt
from tools.measure.prompts import sized
from tools.release_report.sections import Ctx, check, engine_of, row, skips_for

TASK = "\n\nSummarise the text above in five paragraphs."


def measure(ctx: Ctx) -> list[dict]:
    out = skips_for(ctx, "a", "a")
    served0 = (ctx.health.get("inflight") or {}).get("served")
    for n in ctx.plan.lengths:
        for r in range(ctx.plan.runs[n]):
            prompt, _ = sized(ctx.cl, ctx.prose, n, TASK, tag=f"a{n}.{r}")
            rec = ctx.cl.stream_rec(prompt, ANSWER, seed=7 + r)
            bad = check([rec])
            v = {k: rec.get(k) for k in ("prompt_tokens", "completion_tokens", "ttft_s", "decode_tok_s",
                                         "prefill_tok_s", "cached_tokens", "peak_1s_tok_s", "finish")}
            ctx.log(f"[a] {kfmt(n)} run {r + 1}: prompt {v['prompt_tokens']}, TTFT {v['ttft_s']} s, decode "
                    f"{v['decode_tok_s']} tok/s, drafting {rec['engine'].get('drafting')}, tokens/pass "
                    f"{rec['engine'].get('tokens_per_block')}" + (f" INVALID {bad}" if bad else ""))
            if rec.get("cached_tokens"):
                bad = bad or f"{rec['cached_tokens']} prompt tokens came from a cache: not a cold prefill"
            out.append(row("a", f"{n}/{r}", v, {"target": n, "run": r, "max_tokens": ANSWER, "greedy": True,
                                                "thinking": False, "served_before": served0},
                           valid=bad is None, why=bad, engine=[engine_of(rec)], ids=[rec.get("id")]))
    return out
