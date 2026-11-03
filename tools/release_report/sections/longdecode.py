"""f: decode after a long prompt (~56k tokens): one request at a time, greedy, thinking off, up to 1,024
tokens, three tasks whose drafts differ: a summary (fresh text), code (structured text) and a copy (text
already in the prompt). Mean = the whole answer's decode rate; peak = the most tokens in any one-second
window after the first token."""

from __future__ import annotations

from tools.release_report.plan import LONG_ANSWER
from tools.release_report.prompts import sized
from tools.release_report.sections import Ctx, check, engine_of, row, skips_for

TASKS = {
    "summary": "\n\nWrite a detailed summary of the text above, section by section.",
    "code": "\n\nWrite a Python module that splits the text above into paragraphs, counts the words of "
            "each, and prints the ten most common words with their counts. Include tests.",
    "copy": "\n\nCopy the first four paragraphs of the text above exactly as they are written.",
}


def measure(ctx: Ctx) -> list[dict]:
    out = skips_for(ctx, "f", "f")
    if not ctx.plan.long_prompt:
        return out
    for r in range(ctx.plan.long_runs):
        for task in ctx.plan.long_tasks:
            prompt, _ = sized(ctx.cl, ctx.prose, ctx.plan.long_prompt, TASKS[task], tag=f"f.{task}.{r}")
            rec = ctx.cl.stream_rec(prompt, LONG_ANSWER, seed=90 + r)
            bad = check([rec])
            v = {k: rec.get(k) for k in ("prompt_tokens", "ttft_s", "decode_tok_s", "peak_1s_tok_s",
                                         "completion_tokens", "finish")}
            ctx.log(f"[f] {task} run {r + 1}: {v}" + (f" INVALID {bad}" if bad else ""))
            out.append(row("f", f"{task}/{r}", v, {"task": task, "run": r, "max_tokens": LONG_ANSWER,
                                                   "greedy": True},
                           valid=bad is None, why=bad, engine=[engine_of(rec)], ids=[rec.get("id")]))
    return out
