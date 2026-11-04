"""c: N requests released together, N = 1 up to the server's live slot limit.

~2k prompts: each request its own prose, greedy, 512-token answers. Aggregate = all completion tokens
over the wall time from the release to the last end (prefill counts against it); decode = the tokens
after each request's first over the first first token to the last token. The proof: /health every
0.2 s; a round counts only when the server's decode set held N slots at once and no more than N ran.
A round without it is run again once; a level without a proved round is INVALID.

~32k prompts: the slowest (and fastest) first token of N released together, 16 tokens out. The proof is
N running at once: a request left waiting in the queue makes the row INVALID."""

from __future__ import annotations

from tools import parallel_card as pc
from tools.measure.client import together
from tools.release_report.plan import ANSWER
from tools.measure.prompts import sized
from tools.release_report.sections import Ctx, check, row, skips_for

TASK = "\n\nContinue the text above with a detailed essay of your own on the same subject."
TASK_32 = "\n\nIn one sentence, what is the text above about?"


def level(ctx: Ctx, n: int, rnd: int) -> dict:
    prompts = [sized(ctx.cl, ctx.prose, 2048, TASK, tag=f"c{n}.{rnd}.{i}")[0] for i in range(n)]
    pc.wait_idle(ctx.base, 120)                   # the last round's requests have left the server
    got = together(ctx.cl, prompts, ANSWER, seeds=[1000 + i for i in range(n)])
    s, p = got["streams"], got["proof"]
    bad = check(s)
    kind = getattr(ctx.args, "proof_kind", "decoding")
    seen = p["max_decoding_at_once"] if kind == "decoding" else p["max_running"]
    if bad is None and (seen != n or p["max_running"] > n):
        bad = (f"the server's decode set held at most {p['max_decoding_at_once']} of {n} at once "
               f"(running at most {p['max_running']})")
    p["kind"] = kind
    toks = sum(x["completion_tokens"] for x in s)
    firsts = [x["t_send"] + x["ttft_s"] for x in s if x.get("ttft_s") is not None]
    lasts = [x["t_send"] + x["events"][-1][0] for x in s if x.get("events")]
    after = sum(max(0, x["completion_tokens"] - 1) for x in s)
    dec = round(after / (max(lasts) - min(firsts)), 2) if firsts and lasts and max(lasts) > min(firsts) else None
    v = {"n": n, "round": rnd, "tokens": toks, "wall_s": got["wall_s"],
         "agg_tok_s": round(toks / got["wall_s"], 2) if got["wall_s"] else None, "decode_tok_s": dec,
         "ttft_max_s": max((x["ttft_s"] for x in s if x.get("ttft_s") is not None), default=None),
         "prompt_tokens": sorted({x.get("prompt_tokens") for x in s if x.get("prompt_tokens")})[:1]}
    return {"v": v, "proof": p, "bad": bad, "ids": [x.get("id") for x in s],
            "engine": [x.get("engine") for x in s]}


def measure(ctx: Ctx) -> list[dict]:
    out = skips_for(ctx, "c", "c")
    lim = (ctx.cl and (ctx.health.get("parallel") or {}).get("max_running")) or 1
    for n in ctx.plan.conc_2k:
        for rnd in range(ctx.plan.conc_rounds):
            got = level(ctx, n, rnd)
            if got["bad"] and "decode set" in got["bad"]:
                ctx.log(f"[c] 2k x{n} round {rnd + 1}: {got['bad']}; once more")
                got = level(ctx, n, rnd)
            v, p = got["v"], got["proof"]
            ctx.log(f"[c] 2k x{n} round {rnd + 1}: aggregate {v['agg_tok_s']} tok/s, wall {v['wall_s']} s, "
                    f"decoding at once {p['max_decoding_at_once']}, running {p['max_running']}"
                    + (f" INVALID {got['bad']}" if got["bad"] else ""))
            out.append(row("c", f"2k/{n}/{rnd}", v, {"prompt": 2048, "max_tokens": ANSWER, "greedy": True,
                                                     "slot_limit": lim, "proof": p},
                           valid=got["bad"] is None, why=got["bad"], engine=got["engine"], ids=got["ids"]))
    for n in ctx.plan.conc_32k:
        prompts = [sized(ctx.cl, ctx.prose, 32768, TASK_32, tag=f"c32.{n}.{i}")[0] for i in range(n)]
        pc.wait_idle(ctx.base, 120)
        got = together(ctx.cl, prompts, 16, seeds=[2000 + i for i in range(n)], period=0.5)
        s, p = got["streams"], got["proof"]
        bad = check(s)
        if bad is None and p["max_running"] < n:
            bad = f"only {p['max_running']} of {n} ran at once (the rest waited in the queue)"
        if bad is None and p["max_running"] > n:
            bad = f"{p['max_running']} requests ran at once, more than the {n} sent: another request overlapped"
        tt = [x["ttft_s"] for x in s if x.get("ttft_s") is not None]
        v = {"n": n, "ttft_max_s": max(tt) if tt else None, "ttft_min_s": min(tt) if tt else None,
             "prompt_tokens": max((x.get("prompt_tokens") or 0 for x in s), default=None),
             "max_waiting": p.get("max_waiting")}
        ctx.log(f"[c] 32k x{n}: first tokens {v['ttft_min_s']} .. {v['ttft_max_s']} s, running at once "
                f"{p['max_running']}" + (f" INVALID {bad}" if bad else ""))
        out.append(row("c", f"32k/{n}", v, {"prompt": 32768, "max_tokens": 16, "proof": p},
                       valid=bad is None, why=bad, ids=[x.get("id") for x in s]))
    return out
