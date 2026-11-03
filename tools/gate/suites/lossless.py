"""lossless: a fixed prompt set, greedy and seeded, must give the base's text on the head
(docs/exactness.md). Any difference is CHANGED_TEXT: the pull request says why, or it is a bug.

Per prompt and mode the arm sends the request as served (no logprobs) and once more with the top-2
log probabilities, whose token pieces place a difference and give the gap there. A side whose two
servers (`--pairs 2`) wrote two texts for one request is not deterministic: that row is INVALID.

Spec vs plain (`plain_variant`): the head's greedy texts against the same head started with
speculation off. A difference passes only at a near tie, a top-2 gap of at most NEAR_TIE nats in
the plain run (one BF16 step of a logit up to 32, docs/exactness.md); anything wider FAILs."""

from __future__ import annotations

from tools.gate import CHANGED_TEXT, FAIL, PASS
from tools.gate.client import CARD_SAMPLING, GREEDY
from tools.gate.suites import Ctx, Plan
from tools.gate.verdict import first_difference, row, skipped, text_row

NAME = "lossless"
NEAR_TIE = 0.125
SEED = 4242


def prompt_set(pj: dict) -> list[tuple[str, list]]:
    """(id, messages): the card's two workloads, three held-out prose and two held-out code
    continuations, and a question."""
    sys_ = pj["system"]
    words = pj["_filler"].split(" ")
    from tools.gate.client import REPO
    code = (REPO / "bench" / "heldout_code.txt").read_text().splitlines()
    out = [("card-code", [{"role": "system", "content": sys_},
                          {"role": "user", "content": pj["workloads"]["code"]}]),
           ("card-prose", [{"role": "system", "content": sys_},
                           {"role": "user", "content": pj["workloads"]["prose"]}])]
    for i, at in enumerate((0, 1500, 3000)):
        out.append((f"prose-{i + 1}", [{"role": "user", "content": "Continue this text:\n\n"
                                        + " ".join(words[at:at + 220])}]))
    for i, at in enumerate((0, 120)):
        out.append((f"code-{i + 1}", [{"role": "user", "content": "Continue this code:\n\n"
                                       + "\n".join(code[at:at + 60])}]))
    out.append(("question", [{"role": "user", "content": "Explain in four sentences why a "
                              "bridge expands in summer, and what engineers do about it."}]))
    return out


def run_set(ctx: Ctx, modes=("greedy", "seeded")) -> dict:
    out: dict = {}
    n = ctx.profile.c1_tokens
    for pid, msgs in prompt_set(ctx.pj):
        prompt = ctx.cl.render(msgs)
        for mode in modes:
            samp = GREEDY if mode == "greedy" else CARD_SAMPLING
            rec = ctx.cl.complete(prompt, n, seed=SEED, sampling=samp)
            lp = ctx.cl.complete(prompt, n, seed=SEED, sampling=samp, logprobs=2) \
                if "error" not in rec else {}
            if lp.get("pieces") is not None and lp.get("text_sha256") == rec.get("text_sha256"):
                rec["ids"], rec["gaps"] = lp["pieces"], lp.get("gaps") or {}
            rec["lp_same"] = lp.get("text_sha256") == rec.get("text_sha256")
            rec.pop("text", None)
            out[f"{pid}/{mode}"] = rec
            ctx.p(f"lossless {pid}/{mode}: {rec.get('usage', {}).get('completion_tokens')} tokens, "
                  f"{rec.get('finish')}, sha {str(rec.get('text_sha256'))[:12]}"
                  + (f" ERROR {rec['error']}" if rec.get("error") else ""))
    return out


def measure(ctx: Ctx) -> dict:
    return run_set(ctx)


def measure_plain(ctx: Ctx) -> dict:
    return run_set(ctx, modes=("greedy",))


def _consistent(runs: list[dict], key: str) -> str | None:
    shas = {r[key].get("text_sha256") for r in runs if key in r and not r[key].get("error")}
    return None if len(shas) <= 1 else f"{len(shas)} different texts from this side's servers"


def rows(raw: dict, plan: Plan) -> list[dict]:
    base, head = raw.get("base") or [], raw.get("head") or []
    if not base or not head:
        return [skipped("lossless.all", NAME, "all prompts", "an arm did not run")]
    out = []
    for key in base[0]:
        rid = f"lossless.{key.replace('/', '.')}"
        bad = _consistent(base, key) or _consistent(head, key)
        if bad:
            out.append(row(rid, NAME, key, "INVALID", f"not deterministic: {bad}"))
            continue
        out.append(text_row(rid, NAME, key, base[0][key], head[0].get(key, {"error": "missing"})))
    plain = raw.get("head:plain") or []
    if plain:
        for key, rec in plain[0].items():
            out.append(spec_vs_plain(f"lossless.plain.{key.split('/')[0]}", key, head[0].get(key) or {}, rec))
    elif plan.profile.plain_variant and plan.pack.get("plain_env"):
        out.append(skipped("lossless.plain", NAME, "spec vs plain", "the plain arm did not run"))
    else:
        out.append(skipped("lossless.plain", NAME, "spec vs plain",
                           "no plain variant for this pack or hardware profile"))
    return out


def spec_vs_plain(rid: str, metric: str, spec: dict, plain: dict) -> dict:
    """The head with speculation against the head without: equal, or a near tie."""
    if spec.get("error") or plain.get("error"):
        return row(rid, NAME, metric + " spec vs plain", FAIL,
                   f"a request failed: {(spec.get('error') or plain.get('error'))[:200]}")
    if spec.get("text_sha256") == plain.get("text_sha256"):
        n = (spec.get("usage") or {}).get("completion_tokens")
        return row(rid, NAME, metric + " spec vs plain", PASS, f"the same text ({n} tokens)")
    si, pi = spec.get("ids") or [], plain.get("ids") or []
    at = first_difference(si, pi) if si and pi else None
    gap = (plain.get("gaps") or {}).get(str(at)) if at is not None else None
    if gap is not None and gap <= NEAR_TIE:
        return row(rid, NAME, metric + " spec vs plain", PASS,
                   f"differs from token {at} at a near tie (plain top-2 gap {gap:.4f} nats)",
                   first_diff=at, gap_at_diff=gap)
    where = f"from token {at}" if at is not None else "(no token pieces to place it)"
    return row(rid, NAME, metric + " spec vs plain", FAIL if gap is not None else CHANGED_TEXT,
               f"speculation changes the text {where}"
               + (f", plain top-2 gap {gap:.4f} nats > {NEAR_TIE}" if gap is not None else ""),
               first_diff=at, gap_at_diff=gap)
