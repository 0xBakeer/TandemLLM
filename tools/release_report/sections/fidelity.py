"""h: output fidelity. Eight fixed prompts (code, SQL, two maths, logic, a fact, a translation, a summary),
greedy, thinking off, 512 tokens, each sent once as served and once with the top-2 log probabilities
(their token pieces place a difference and give the gap there). The texts are kept, so a run on the same
weights with speculation off (`--drafter none`, a second server: `--variant plain`) is compared text for
text: equal, or a difference at a near tie (a top-2 gap of at most 0.125 nats in the plain run,
docs/exactness.md). A server that runs no drafter has only plain texts, and the comparison is skipped.

Needles: a passphrase in the middle of ~32k and ~128k tokens of prose; found or not."""

from __future__ import annotations

from tools.release_report.prompts import NONCE_LINE, nonce
from tools.release_report.sections import Ctx, check, row, skipped

NEAR_TIE = 0.125
PROMPTS = {
    "code": "Write a Python class LRUCache with get and put in O(1), using collections.OrderedDict. Code only.",
    "sql": "Write one SQL query that returns the three most recent orders of every customer from a table "
           "orders(id, customer_id, created_at). Query only.",
    "math_sum": "Find the sum of all positive integers n < 100 such that n^2 + 1 is divisible by 5. Show the steps.",
    "math_prob": "Two fair dice are rolled. Given that the sum is at least 9, what is the probability that "
                 "both dice show the same number? Explain briefly.",
    "logic": "Alice is older than Bob. Bob is older than Carol. Dave is older than Alice. Who is the second "
             "oldest? Answer with the name, then one sentence.",
    "fact": "In three sentences, what is the Chandrasekhar limit?",
    "translate": "Translate into French: \"We shipped it on Friday, and by Monday it had already broken twice.\"",
    "summarise": None,            # a passage of the run's prose, summarised in three bullet points
}
NEEDLE = "The passphrase for the archive room is {word}."
WORDS = ("amber-falcon-41", "cobalt-river-77")


def prompt_of(ctx: Ctx, pid: str) -> str:
    text = PROMPTS[pid]
    if text is None:
        text = ("Summarise this passage in three bullet points:\n\n"
                + (ctx.__dict__.setdefault("_passage", ctx.prose.take(600))))
    return ctx.cl.render([{"role": "user", "content": text}])


def needle(ctx: Ctx, tokens: int, word: str) -> dict:
    cl = ctx.cl
    q = "\n\nWhat is the passphrase for the archive room? Answer with the passphrase only."
    body = ctx.prose.take(max(0, tokens - 96))
    mid = len(body) // 2
    dot = body.rfind(". ", 0, mid)
    cut = dot + 2 if dot >= 0 else mid
    text = NONCE_LINE.format(nonce=nonce("h", tokens)) + body[:cut] + NEEDLE.format(word=word) + " " + body[cut:] + q
    rec = cl.complete_rec(cl.render([{"role": "user", "content": text}]), 24)
    return rec


def measure(ctx: Ctx) -> list[dict]:
    out = []
    v = ctx.variant
    for pid in PROMPTS:
        p = prompt_of(ctx, pid)
        rec = ctx.cl.complete_rec(p, 512, seed=4242)
        lp = ctx.cl.complete_rec(p, 512, seed=4242, logprobs=2) if not rec.get("error") else {}
        bad = check([rec] + ([lp] if lp else []))
        same_lp = lp.get("sha256") == rec.get("sha256")
        vals = {"text": rec.get("text"), "sha256": rec.get("sha256"), "finish": rec.get("finish"),
                "completion_tokens": (rec.get("usage") or {}).get("completion_tokens"),
                "pieces": lp.get("pieces") if same_lp else None, "gaps": lp.get("gaps") if same_lp else None,
                "logprobs_same_text": same_lp}
        ctx.log(f"[h] {pid} ({v}): {vals['completion_tokens']} tokens, sha {str(rec.get('sha256'))[:12]}, "
                f"logprobs run same {same_lp}" + (f" INVALID {bad}" if bad else ""))
        out.append(row("h", f"{v}/{pid}", vals, {"variant": v, "greedy": True, "max_tokens": 512,
                                                 "drafter": ctx.plan.drafter},
                       valid=bad is None, why=bad, ids=[rec.get("id"), lp.get("id")]))
    if getattr(ctx.args, "no_drafter", False):
        out.append(skipped("h", "compare", "the server runs no drafter: its texts are the plain ones, "
                                           "so there is no speculative text to compare"))
    if v == "spec" or getattr(ctx.args, "no_drafter", False):
        for n, word in zip(ctx.plan.needles, WORDS):
            rec = needle(ctx, n, word)
            bad = check([rec])
            found = word in (rec.get("text") or "")
            ctx.log(f"[h] needle {n}: found {found}, answer {(rec.get('text') or '')[:40]!r}"
                    + (f" INVALID {bad}" if bad else ""))
            out.append(row("h", f"needle/{n}", {"found": found, "answer": (rec.get("text") or "")[:80],
                                                "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens")},
                           {"target": n}, valid=bad is None, why=bad, ids=[rec.get("id")]))
        for k, why in ctx.plan.skipped.items():
            if k.startswith("h."):
                out.append(skipped("h", k[2:], why))
    return out


def compare(rows: list[dict]) -> list[dict]:
    """Spec against plain, prompt by prompt (both variants' rows in data.json)."""
    spec = {r["key"].split("/", 1)[1]: r for r in rows if r["key"].startswith("spec/") and r.get("valid")}
    plain = {r["key"].split("/", 1)[1]: r for r in rows if r["key"].startswith("plain/") and r.get("valid")}
    out = []
    for pid in PROMPTS:
        s, p = spec.get(pid), plain.get(pid)
        if not s or not p:
            continue
        sv, pv = s["v"], p["v"]
        if sv["sha256"] == pv["sha256"]:
            out.append({"prompt": pid, "same": True, "near_tie": False, "at": None, "gap": None})
            continue
        si, pi = sv.get("pieces") or [], pv.get("pieces") or []
        at = next((i for i, (a, b) in enumerate(zip(si, pi)) if a != b), min(len(si), len(pi))) if si and pi else None
        gap = (pv.get("gaps") or {}).get(str(at)) if at is not None else None
        out.append({"prompt": pid, "same": False, "near_tie": gap is not None and gap <= NEAR_TIE,
                    "at": at, "gap": gap})
    return out
