"""j: capability on a multiple-choice set (`--hle --hle-data FILE`, a local JSON-lines file of
{"id", "question", "choices": {"A": ...}, "answer": "A"}). Off by default: the page says "not run".

Each item is one raw completion with the chat template's own default (thinking as the model ships
it), the sampling the server's card names, 8 at once, graded by the last "Answer: X" letter, no judge.
The score's 95 % interval is Wilson's."""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor

from tools.release_report.sections import Ctx, check, row, skipped

LETTER = re.compile(r"Answer:\s*\(?([A-J])\)?", re.I)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def measure(ctx: Ctx) -> list[dict]:
    a = ctx.args
    if not getattr(a, "hle", False) or not getattr(a, "hle_data", None):
        return [skipped("j", "score", "not run (it takes about 2.5 hours: --hle --hle-data FILE)")]
    items = [json.loads(x) for x in open(a.hle_data) if x.strip()]
    samp = (ctx.health.get("sampling") or {}).get("card") or {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    samp = {k: samp[k] for k in ("temperature", "top_p", "top_k") if k in samp}

    def one(it):
        q = it["question"] + "\n\n" + "\n".join(f"{k}. {v}" for k, v in it["choices"].items()) + \
            "\n\nEnd with a line 'Answer: X', X the letter of your choice."
        p = ctx.cl.render([{"role": "user", "content": q}], template=True) if ctx.cl.tok is not None else \
            ctx.cl.render([{"role": "user", "content": q}])
        body_s = dict(samp)
        rec = ctx.cl.stream_rec(p, int(a.hle_budget), seed=42, sampling=body_s)
        m = LETTER.findall(rec.get("text") or "")
        got = m[-1].upper() if m else None
        return {"id": it.get("id"), "answer": it["answer"], "got": got, "ok": got == it["answer"],
                "tokens": rec.get("completion_tokens"), "bad": check([rec]), "finish": rec.get("finish")}
    with ThreadPoolExecutor(min(8, ctx.plan.lanes)) as ex:
        res = list(ex.map(one, items))
    n, k = len(res), sum(r["ok"] for r in res)
    errs = [r for r in res if r["bad"]]
    v = {"correct": k, "items": n, "score": round(k / n, 4) if n else None, "ci95": wilson(k, n),
         "errors": len(errs), "unparsed": sum(1 for r in res if r["got"] is None),
         "capped": sum(1 for r in res if r["finish"] == "length")}
    return [row("j", "score", v, {"sampling": samp, "budget": a.hle_budget, "at_once": min(8, ctx.plan.lanes),
                                  "items_file": a.hle_data.rsplit("/", 1)[-1]},
                valid=not errs, why=f"{len(errs)} items failed or were not raw" if errs else None)]
