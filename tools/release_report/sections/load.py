"""e: latency under load, and a conversation that comes back.

Under load: two chats decode long answers (their end tokens banned, so they keep decoding); after both
decode, a 64k prompt arrives. Its first token, and the two decoders' combined tok/s in the 8 s before
it arrived and from its arrival to its first token. The decoders then leave.

Returning conversation (prompt cache on, as a chat client sends it): turn 1 cold on a ~97k-token
document, turn 2 straight after (its prefix in the cache), then three other ~60k prompts, then turn 3:
its first token is the time the engine takes to get the conversation back."""

from __future__ import annotations

import threading
import time

from tools import parallel_card as pc
from tools.release_report.client import end_ids, tokens_between
from tools.release_report.prompts import NONCE_LINE, nonce, sized
from tools.release_report.sections import Ctx, check, med, row, skips_for

DECODE_TASK = "\n\nWrite the longest essay you can about the text above. Do not stop."
BEFORE_S = 8.0


def under_load(ctx: Ctx) -> dict:
    cl = ctx.cl
    bias = {str(i): -100 for i in end_ids(cl.tok)}
    stop = threading.Event()
    recs: dict = {}
    ps = [sized(cl, ctx.prose, 2048, DECODE_TASK, tag=f"e.dec{i}")[0] for i in range(2)]
    big, _ = sized(cl, ctx.prose, ctx.plan.load_prompt, "\n\nIn one sentence, what is the text above about?",
                   tag="e.load")

    def dec(i):
        recs[i] = cl.stream_rec(ps[i], 8192, seed=50 + i, logit_bias=bias or None, stop=stop)
    ths = [threading.Thread(target=dec, args=(i,)) for i in range(2)]
    for t in ths:
        t.start()
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 300:          # both decoding
        dc = (pc.health(ctx.base).get("parallel") or {}).get("decoding")
        if (dc if isinstance(dc, int) else len(dc or [])) >= 2:
            break
        time.sleep(0.25)
    time.sleep(BEFORE_S + 1.0)
    t_a = time.monotonic()
    rec = cl.stream_rec(big, 16, seed=60)
    t_b = t_a + (rec["ttft_s"] or rec["e2e_s"])
    stop.set()
    for t in ths:
        t.join(timeout=600)
    d = [recs.get(i) for i in range(2)]
    bad = check([rec] + [x for x in d if x]) if all(d) else "a decoder did not finish its stream"
    if bad is None and any(x["t_send"] + (x["events"][-1][0] if x["events"] else 0) < t_b - 0.5 for x in d):
        bad = "a decoder stopped before the long prompt's first token"
    before = sum(tokens_between(x, t_a - BEFORE_S, t_a) for x in d if x) / BEFORE_S
    during = sum(tokens_between(x, t_a, t_b) for x in d if x) / max(t_b - t_a, 1e-6)
    return {"v": {"ttft_s": rec["ttft_s"], "prompt_tokens": rec.get("prompt_tokens"),
                  "decoders_before_tok_s": round(before, 2), "decoders_during_tok_s": round(during, 2)},
            "bad": bad, "ids": [rec.get("id")] + [x.get("id") for x in d if x]}


def returning(ctx: Ctx) -> list[dict]:
    cl, out = ctx.cl, []
    system = "You answer questions about the document the user gives."
    head = NONCE_LINE.format(nonce=nonce("e.ret"))
    user1 = head + ctx.prose.take(max(0, ctx.plan.return_prompt - 64)) + "\n\nWhat is the document above about?"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user1}]
    recs = []

    def turn(tag, messages):
        r = cl.stream_rec(cl.render(messages), 64, seed=70, cache_prompt=True)
        recs.append(r)
        ctx.log(f"[e] {tag}: prompt {r.get('prompt_tokens')}, TTFT {r.get('ttft_s')} s, cached {r.get('cached_tokens')}")
        return r
    t1 = turn("turn 1 (cold)", msgs)
    msgs += [{"role": "assistant", "content": t1["text"]},
             {"role": "user", "content": "Name two people or places it mentions."}]
    t2 = turn("turn 2 (straight after)", msgs)
    flood = []
    for i in range(3):
        p, _ = sized(cl, ctx.prose, ctx.plan.flood_prompt, "\n\nIn one sentence, what is this about?",
                     tag=f"e.flood{i}")
        r = cl.stream_rec(p, 16, seed=80 + i, cache_prompt=True)
        recs.append(r)
        flood.append(r)
    msgs += [{"role": "assistant", "content": t2["text"]},
             {"role": "user", "content": "Give the document a title."}]
    t3 = turn("turn 3 (after three other long prompts)", msgs)
    bad = check(recs)
    v = {"prompt_tokens": t1.get("prompt_tokens"), "turn1_ttft_s": t1["ttft_s"], "turn2_ttft_s": t2["ttft_s"],
         "turn2_cached": t2.get("cached_tokens"), "flood_ttft_median_s": med([r["ttft_s"] for r in flood]),
         "flood_prompt_tokens": med([r.get("prompt_tokens") for r in flood]), "turn3_ttft_s": t3["ttft_s"],
         "turn3_cached": t3.get("cached_tokens")}
    out.append(row("e", "return", v, {"cache_prompt": True, "flood": 3, "max_tokens": 64},
                   valid=bad is None, why=bad, ids=[r.get("id") for r in recs]))
    return out


def measure(ctx: Ctx) -> list[dict]:
    out = skips_for(ctx, "e", "e")
    if ctx.plan.load_prompt:
        got = under_load(ctx)
        ctx.log(f"[e] 64k under load: {got['v']}" + (f" INVALID {got['bad']}" if got["bad"] else ""))
        out.append(row("e", "load", got["v"], {"decoders": 2, "prompt": ctx.plan.load_prompt,
                                               "before_window_s": BEFORE_S},
                       valid=got["bad"] is None, why=got["bad"], ids=got["ids"]))
        pc.wait_idle(ctx.base, 300)
    if ctx.plan.return_prompt:
        out += returning(ctx)
    return out
