"""decode: one request with a short and with a 1k prompt, then N requests at once at each of the
hardware profile's levels (c8, or c4 where it has fewer lanes), with tools/parallel_card.py's rounds:
fixed prompts and seeds (every round decodes the same tokens), warm-up rounds discarded, and /health
every 0.2 s as the proof that N decoded at once. A round without the proof is run again; a level
with fewer proved rounds than the profile's `rounds` is INVALID. Before a level the server's live
slot limit (`parallel.max_running`) must be at least N, or the level is INVALID and not run.

Rows: decode tok/s (completion tokens after the first, over the first to the last token) for each
shape, and the wall rate (TTFT included) at each level; median of the rounds, row3's noise rule."""

from __future__ import annotations

from tools import parallel_card as pc
from tools.gate.client import filler_text
from tools.gate.suites import Ctx, Plan, pooled
from tools.gate.verdict import skipped, speed_row

NAME = "decode"


def shapes(ctx: Ctx) -> list[tuple[str, int, dict, int]]:
    """(id, N, prompt set, max tokens) for each shape this profile runs."""
    pj = ctx.pj
    long_pj = dict(pj, workloads=dict(pj["workloads"], prose=filler_text(pj, 700)
                                      + "\n\nSummarise the text above in one paragraph."))
    out = [("c1", 1, pj, ctx.profile.c1_tokens), ("c1-1k", 1, long_pj, ctx.profile.c1_tokens)]
    out += [(f"c{n}", n, pj, ctx.profile.decode_tokens) for n in ctx.profile.decode_levels()]
    return out


def proved(r: dict, n: int, kind: str) -> bool:
    """A round's proof: N in the server's decode set (or N running, `kind` "running"), never more
    than N running (no request from elsewhere), every request answered."""
    p = r["proof"]
    seen = p["max_decoding_at_once"] if kind == "decoding" else p["max_running"]
    return not r.get("errors") and seen == n and p["max_running"] <= n and \
        all(s["completion_tokens"] > 0 for s in r.get("streams") or [])


def at_once(r: dict, kind: str) -> int:
    return r["proof"]["max_decoding_at_once"] if kind == "decoding" else r["proof"]["max_running"]


def rounds_of(ctx: Ctx, n: int, pj: dict, max_tokens: int) -> dict:
    kind = ctx.pack.get("proof", "decoding")
    lv = {"warm": [], "rounds": [], "invalid": [], "proof_kind": kind}
    pc.wait_idle(ctx.base)
    for w in range(ctx.profile.warmups):
        lv["warm"].append(pc.one_round(ctx.cl, pj, n, f"w{w}", max_tokens, fixed=True))
    tries, want = 0, ctx.profile.rounds
    while len(lv["rounds"]) < want and tries < want + 2:
        tries += 1
        r = pc.one_round(ctx.cl, pj, n, f"r{tries}", max_tokens, fixed=True)
        r["valid"] = proved(r, n, kind)
        (lv["rounds"] if r["valid"] else lv["invalid"]).append(r)
        ctx.p(f"decode c{n} round {tries}: decode {r.get('decode_tok_s')} tok/s, wall "
              f"{r.get('wall_tok_s')}, decoding at once {r['proof']['max_decoding_at_once']}, running "
              f"{r['proof']['max_running']}, valid {r['valid']} ({kind})")
    for r in lv["warm"] + lv["rounds"] + lv["invalid"]:
        for s in r.get("streams") or []:
            s.pop("t_send", None)
    return lv


def measure(ctx: Ctx) -> dict:
    out = {}
    limit = (ctx.health.get("parallel") or {}).get("max_running") or 1
    for sid, n, pj, mt in shapes(ctx):
        if n > limit:
            out[sid] = {"refused": f"the server's live slot limit is {limit} "
                                   f"({(ctx.health.get('parallel') or {}).get('max_running_reason')}), "
                                   f"under {n} requests at once"}
            ctx.p(f"decode {sid}: refused, {out[sid]['refused']}")
            continue
        out[sid] = rounds_of(ctx, n, pj, mt)
    return out


def _invalid(levels: list[dict], n: int, want: int) -> str | None:
    for lv in levels:
        if lv.get("refused"):
            return lv["refused"]
        if len(lv["rounds"]) < want:
            kind = lv.get("proof_kind", "decoding")
            seen = max((at_once(r, kind) for r in lv["invalid"]), default=0)
            return (f"{len(lv['rounds'])} of {want} rounds proved {n} "
                    f"{'decoding' if kind == 'decoding' else 'running'} at once (the most seen {seen})")
    return None


def rows(raw: dict, plan: Plan) -> list[dict]:
    base, head = pooled(raw, "base"), pooled(raw, "head")
    if not base or not head:
        return [skipped("decode.all", NAME, "all shapes", "an arm did not run")]
    out = []
    want = plan.profile.rounds
    for sid in base[0]:
        n = 1 if sid.startswith("c1") else int(sid[1:])
        b = [r[sid] for r in base if sid in r]
        h = [r[sid] for r in head if sid in r]
        bad = _invalid(b, n, want) or _invalid(h, n, want)
        kind = next((lv.get("proof_kind") for lv in b + h if lv.get("proof_kind")), "decoding")
        proof = {"n": n, "kind": kind,
                 "base_at_once": [min((at_once(x, kind) for x in lv.get("rounds") or []), default=0) for lv in b],
                 "head_at_once": [min((at_once(x, kind) for x in lv.get("rounds") or []), default=0) for lv in h]}
        metric = {"c1": "1 request, short prompt", "c1-1k": "1 request, 1k prompt"}.get(
            sid, f"{n} requests at once")
        runs = lambda lvs, k: [x[k] for lv in lvs for x in lv.get("rounds") or []]  # noqa: E731
        out.append(speed_row(f"decode.{sid}", NAME, f"{metric}, decode", "tok/s", True,
                             runs(b, "decode_tok_s"), runs(h, "decode_tok_s"),
                             plan.profile.min_effect_pct, invalid=bad, proof=proof))
        if n > 1:
            out.append(speed_row(f"decode.{sid}.wall", NAME, f"{metric}, wall", "tok/s", True,
                                 runs(b, "wall_tok_s"), runs(h, "wall_tok_s"),
                                 plan.profile.min_effect_pct, invalid=bad, proof=proof))
    return out
