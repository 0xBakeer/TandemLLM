"""What a run measures on this server, sized from what the server reports (/health, /v1/models) and
the box's GPU (tools/gate/hw): prompt lengths up to the largest slot window the server sized from its
memory, concurrency up to its live slot limit, each section's shapes, and a time estimate.

A length or shape that does not fit becomes a skipped row with the reason; nothing is sent for it."""

from __future__ import annotations

import dataclasses

#: the single-request ladder (section a/b), tokens
LADDER = (1024, 4096, 16384, 32768, 56320, 65536, 98304, 131072, 196608, 245760)
QUICK_LADDER = (1024, 16384, 65536)
#: an answer's tokens in a/b and c, f's answer, and the room a request keeps beside its prompt
ANSWER, LONG_ANSWER, MARGIN = 512, 1024, 256


def kfmt(n: int) -> str:
    return f"{round(n / 1024)}k"


@dataclasses.dataclass
class Plan:
    quick: bool
    window: int                    # the largest prompt + answer one request may take
    window_why: str
    lanes: int                     # the live slot limit
    lengths: list                  # a/b: prompt tokens
    runs: dict                     # a/b: runs per length
    skipped: dict                  # what is not run: key -> why
    conc_2k: list                  # c: N at once, ~2k prompts
    conc_rounds: int
    conc_32k: list                 # c: N at once, ~32k prompts
    chats: list                    # d: chats at once on one new system prompt
    shared_tokens: int             # d: the system prompt
    load_prompt: int | None        # e: the prompt that arrives under load
    return_prompt: int | None      # e: the returning conversation
    flood_prompt: int | None
    long_prompt: int | None        # f
    long_tasks: list
    long_runs: int
    needles: list                  # h
    warm: list                     # i
    drafter: bool
    capabilities: dict

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def window_of(health: dict) -> tuple[int, str]:
    p = health.get("parallel") or {}
    wins = [s.get("max_len") for s in p.get("slots") or [] if s.get("max_len")]
    if wins:
        return max(wins), "the largest slot window the server sized from its memory"
    return int(health.get("max_len") or 8192), "the server's max_len"


def hw_cap(profile) -> tuple[int | None, str]:
    """A prompt cap from the GPU where the server's own window cannot be trusted to fit: a GPU without
    an adapter and under 40 GiB, and the fake engine's box. GB10 and Ada adapters trust the server."""
    if profile is None:
        return None, ""
    if profile.id == "fake":
        return 16384, "the fake engine's box"
    if profile.id in ("generic", "base") and profile.mem() < 40:
        return 32768, f"a GPU without an adapter with {profile.mem():g} GiB"
    return None, ""


def drafts(health: dict) -> bool:
    w = health.get("weights") or {}
    d = str(w.get("drafters") or "").strip().lower()
    return bool(d) and not d.startswith(("none", "off", "-"))


def make(health: dict, card: dict | None, profile=None, quick: bool = False,
         max_prompt: int | None = None) -> Plan:
    win, why = window_of(health)
    cap, cap_why = hw_cap(profile)
    if max_prompt:
        cap, cap_why = (max_prompt, "--max-prompt") if cap is None or max_prompt < cap else (cap, cap_why)
    lanes = int(((health.get("parallel") or {}).get("max_running")) or 1)
    skipped: dict = {}

    def fits(tokens: int, answer: int, key: str) -> bool:
        if tokens + answer + MARGIN > win:
            skipped[key] = f"{tokens:,} + {answer} tokens is over {why} ({win:,})"
            return False
        if cap is not None and tokens > cap:
            skipped[key] = f"{tokens:,} tokens is over the cap for {cap_why} ({cap:,})"
            return False
        return True

    ladder = QUICK_LADDER if quick else LADDER
    lengths = [n for n in ladder if fits(n, ANSWER, f"a.{kfmt(n)}")]
    runs = {n: (1 if quick else (3 if n <= 16384 else 2)) for n in lengths}
    if quick:
        skipped["a.more"] = "--quick: three lengths, one run each"
    conc = list(range(1, lanes + 1)) if not quick else sorted({1, max(1, lanes // 2), lanes})
    c32 = [n for n in (conc if not quick else sorted({1, lanes})) if fits(32768, 32, f"c.32k.{n}")]
    if lanes < 2:
        skipped["c"] = "the server serves one request at a time (live slot limit 1)"
    chats = sorted({min(5, lanes), lanes}) if lanes >= 2 else []
    if not chats:
        skipped["d"] = "the server serves one request at a time"
    shared = (8192 if quick else 28672)
    shared = shared if fits(shared, 64, "d.prompt") else 0
    if quick:
        chats = [lanes] if lanes >= 2 else []
    load = (32768 if quick else 65536)
    load = load if lanes >= 3 and fits(load, 64, "e.load") else None
    if lanes < 3:
        skipped["e.load"] = "needs three requests at once (two decoding, one arriving)"
    ret = 98304 if fits(98304 + 1024, 64, "e.return") else None
    flood = 61440 if ret else None
    if quick:
        ret = flood = None
        skipped["e.return"] = "--quick"
    lp = 57344 if fits(57344, LONG_ANSWER, "f") else None
    tasks = ["summary"] if quick else ["summary", "code", "copy"]
    needles = [n for n in ((32768,) if quick else (32768, 131072)) if fits(n, 64, f"h.needle.{kfmt(n)}")]
    warm = [] if quick else [n for n in (131072, 196608) if fits(n, ANSWER, f"i.{kfmt(n)}")]
    if quick:
        skipped["i"] = "--quick"
    caps = ((card or {}).get("tandem") or {}).get("capabilities") or {}
    return Plan(quick=quick, window=win, window_why=why, lanes=lanes, lengths=lengths, runs=runs,
                skipped=skipped, conc_2k=conc if lanes >= 2 else [1], conc_rounds=1 if quick else 2,
                conc_32k=c32 if lanes >= 2 else [], chats=chats if shared else [], shared_tokens=shared,
                load_prompt=load, return_prompt=ret, flood_prompt=flood, long_prompt=lp,
                long_tasks=tasks, long_runs=1 if quick else 2, needles=needles, warm=warm,
                drafter=drafts(health), capabilities=caps)


def estimate(plan: Plan, sections, prefill_tok_s: float, decode_tok_s: float) -> dict:
    """Minutes a section takes at the probe's rates (a prefill is serial; N decoders share the
    engine at about 2.5 times one's rate at 8). A rough figure for the plan, never a result."""
    pf0 = max(prefill_tok_s, 1.0)
    dc = max(decode_tok_s, 1.0)

    class _PF:
        """Prefill time of n tokens: the probe's rate, slowing with the length (attention grows with it;
        on the reference box about 0.9x at 64k and 0.6x at 240k)."""
        def __rtruediv__(self, n):
            return n / (pf0 / (1 + n / 400_000))
    pf = _PF()

    def agg(n):
        return dc * min(n, 1 + 0.22 * (n - 1))
    s: dict = {}
    s["a"] = sum(r * (n / pf + ANSWER / dc) for n, r in plan.runs.items())
    s["c"] = sum(plan.conc_rounds * (n * 2048 / pf + n * ANSWER / agg(n)) for n in plan.conc_2k) + \
        sum(n * 32768 / pf for n in plan.conc_32k)
    s["d"] = sum(c * plan.shared_tokens / pf + 64 / dc for c in plan.chats)
    s["e"] = ((plan.load_prompt or 0) / pf + 30 if plan.load_prompt else 0) + \
        ((plan.return_prompt or 0) * 1.2 / pf + 3 * (plan.flood_prompt or 0) / pf if plan.return_prompt else 0)
    s["f"] = len(plan.long_tasks) * plan.long_runs * ((plan.long_prompt or 0) / pf + LONG_ANSWER / dc) \
        if plan.long_prompt else 0
    lv = [n for n in (1, 2, 4, 8) if n <= plan.lanes]
    levels = sum(4 * n * ANSWER / agg(n) for n in lv)            # 1 warm-up + 3 rounds a level
    s["g"] = levels if plan.quick else 2 * levels + 6 * 400 / dc + 3 * (2048 / pf + 8192 / pf + 32768 / pf)
    s["h"] = 16 * 512 / dc + sum(plan.needles) / pf
    s["i"] = sum(n / pf + ANSWER / dc for n in plan.warm)
    s["j"] = 0
    out = {k: round(v / 60, 1) for k, v in s.items() if k in sections}
    out["total"] = round(sum(out.values()) + 1, 1)
    return out
