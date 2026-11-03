"""d: chats that arrive together on one new system prompt (~28k tokens, a fresh nonce, the same for
every chat), each with its own short question, released at once with the prompt cache on (the case a
shared prefix helps). The first and the last chat's first token, and the prompt tokens each one found
in a cache. Proof: all N running at once."""

from __future__ import annotations

from tools import parallel_card as pc
from tools.release_report.client import together
from tools.release_report.prompts import NONCE_LINE, nonce
from tools.release_report.sections import Ctx, check, row, skips_for

QUESTIONS = ("Who are the people the text names first?", "List three places the text mentions.",
             "What year does the text give first?", "Give the text a short title.",
             "Which sentence of the text is the longest?", "What is the text's main subject?",
             "Name one number the text uses and what it counts.", "Is the text fiction? One word.")


def measure(ctx: Ctx) -> list[dict]:
    out = skips_for(ctx, "d", "d")
    for n in ctx.plan.chats:
        head = NONCE_LINE.format(nonce=nonce("d", n)) + "Answer questions about this document.\n\n"
        base = ctx.cl.count(ctx.cl.render([{"role": "system", "content": head},
                                           {"role": "user", "content": QUESTIONS[0]}])) \
            if ctx.cl.tok is not None else len(head) + 64
        system = head + ctx.prose.take(max(0, ctx.plan.shared_tokens - base))
        prompts = [ctx.cl.render([{"role": "system", "content": system},
                                  {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]) for i in range(n)]
        pc.wait_idle(ctx.base, 120)
        got = together(ctx.cl, prompts, 32, cache_prompt=True, period=0.5)
        s, p = got["streams"], got["proof"]
        bad = check(s)
        if bad is None and p["max_running"] < n:
            bad = f"only {p['max_running']} of {n} ran at once (the rest waited in the queue)"
        if bad is None and p["max_running"] > n:
            bad = f"{p['max_running']} requests ran at once, more than the {n} sent: another request overlapped"
        tt = sorted(x["ttft_s"] for x in s if x.get("ttft_s") is not None)
        v = {"chats": n, "ttft_first_s": tt[0] if tt else None, "ttft_last_s": tt[-1] if tt else None,
             "prompt_tokens": max((x.get("prompt_tokens") or 0 for x in s), default=None),
             "cached_tokens": [x.get("cached_tokens") or 0 for x in s]}
        ctx.log(f"[d] {n} chats: first tokens {v['ttft_first_s']} .. {v['ttft_last_s']} s, cached "
                f"{v['cached_tokens']}" + (f" INVALID {bad}" if bad else ""))
        out.append(row("d", f"{n}", v, {"system_tokens": ctx.plan.shared_tokens, "max_tokens": 32,
                                        "cache_prompt": True, "proof": p},
                       valid=bad is None, why=bad, ids=[x.get("id") for x in s]))
    return out
