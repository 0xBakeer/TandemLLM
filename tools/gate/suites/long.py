"""long: the longest prompt the hardware profile allows, with a needle in the middle: the request
completes (HTTP 200, no error), the answer names the needle, and the server's reserved memory stays
under its declared peak with no admission over the peak (/health every 0.5 s during the request).
At quick+long and full a second server of each side runs the wide prefill chunk case (the pack's
`wide_chunk_env`, a prompt of `wide_chunk_ctx` tokens) the same way.

The rows are checks on each side (a check the base fails too is SKIPPED as not this change's); the
time to first token is recorded but one request a side cannot resolve a speed difference."""

from __future__ import annotations

from tools import parallel_card as pc
from tools.gate.client import GREEDY, HealthPoller
from tools.gate.suites import Ctx, Plan, first
from tools.gate.verdict import case_row, row, skipped

NAME = "long"
NEEDLE = "The access code for the north gate is 7341."
ANSWER = "7341"
QUESTION = "\n\nWhat is the access code for the north gate? Answer with the number only."


def needle_prompt(ctx: Ctx, tokens: int) -> str:
    """A user turn of about `tokens` prompt tokens: filler, the needle in the middle, the question."""
    head = f"Long {pc.nonce('long', tokens)}.\n"
    offset = 0
    k, planned = pc.prefill_words(ctx.base, ctx.pj, tokens - 64, head, offset)
    words = pc.filler_words(ctx.pj, tokens)[:k]
    mid = len(words) // 2
    text = head + " ".join(words[:mid]) + "\n" + NEEDLE + "\n" + " ".join(words[mid:]) + QUESTION
    return ctx.cl.render([{"role": "user", "content": text}])


def one(ctx: Ctx, tokens: int) -> dict:
    pc.wait_idle(ctx.base)
    prompt = needle_prompt(ctx, tokens)
    with HealthPoller(ctx.base, 0.5) as pol:
        ctx.cl.extra["cache_prompt"] = False
        rec = ctx.cl.complete(prompt, 24, sampling=GREEDY)
    mem = pol.memory()
    ok = "error" not in rec
    out = {"tokens_asked": tokens, "ok": ok, "error": rec.get("error"),
           "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"),
           "e2e_s": rec.get("e2e_s"), "needle": ok and ANSWER in (rec.get("text") or ""),
           "answer": (rec.get("text") or "")[:80], **mem}
    ctx.p(f"long {tokens}: ok {ok}, prompt {out['prompt_tokens']} tokens, {out['e2e_s']} s, "
          f"needle {out['needle']}, reserved max {mem['reserved_max_gb']} GB of peak {mem['peak_gb']}")
    return out


def measure(ctx: Ctx) -> dict:
    return {"max": one(ctx, ctx.profile.long_ctx(ctx.level))}


def measure_wide(ctx: Ctx) -> dict:
    return {"wide": one(ctx, ctx.profile.wide_chunk_ctx())}


def checks(case: str, b: dict | None, h: dict, out: list) -> None:
    pre = f"long.{case}"
    what = f"{h.get('prompt_tokens') or h['tokens_asked']} tokens"
    if case == "wide":
        what += ", wide prefill chunk"
    out.append(case_row(f"{pre}.complete", NAME, f"{what}: completes", b and b["ok"], h["ok"],
                        h.get("error") or f"{h['e2e_s']} s" + (f" (base {b['e2e_s']} s)" if b else "")))
    if h["ok"]:
        out.append(case_row(f"{pre}.needle", NAME, f"{what}: needle answer", b and b["needle"],
                            h["needle"], f"answered {h['answer']!r}"))
    if h.get("reserved_max_gb") is None or h.get("peak_gb") is None:
        out.append(skipped(f"{pre}.reserve", NAME, f"{what}: reserve under the peak",
                           "the server reports no reserved memory or peak"))
        return
    under = lambda x: x is not None and x["reserved_max_gb"] is not None and x["peak_gb"] is not None \
        and x["reserved_max_gb"] <= x["peak_gb"] and not x["over_peak_admits"]  # noqa: E731
    out.append(case_row(f"{pre}.reserve", NAME, f"{what}: reserve under the peak",
                        under(b) if b else None, under(h),
                        f"reserved max {h['reserved_max_gb']} GB, peak {h['peak_gb']} GB, admitted over "
                        f"the peak {h['over_peak_admits']}"))


def rows(raw: dict, plan: Plan) -> list[dict]:
    b, h = first(raw, "base"), first(raw, "head")
    if not h:
        return [skipped("long.max", NAME, "longest prompt", "the head did not run")]
    out: list = []
    checks("max", b and b["max"], h["max"], out)
    if plan.level in ("quick+long", "full"):
        wb, wh = first(raw, "base:wide"), first(raw, "head:wide")
        if wh:
            checks("wide", wb and wb["wide"], wh["wide"], out)
        elif not plan.pack.get("wide_chunk_env"):
            out.append(skipped("long.wide", NAME, "wide prefill chunk",
                               "the pack names no wide-chunk setting"))
        else:
            out.append(row("long.wide", NAME, "wide prefill chunk", "FAIL", "the wide arm did not run"))
    return out
