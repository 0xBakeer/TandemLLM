"""TRN-7's free go/no-go: what a drafter at the target's own confidence would commit, slot by slot.

the operator's rule of 2026-09-24: Draft-OPD (on-policy distillation with a position-weighted loss) may be
paid for only if a free check projects at least +5 % tokens a round on prose/chat. The check has two
halves, and this tool computes both from the same text:

  * the drafter the engine serves, slot by slot: a_i = P(slot i accepted | slots < i were), the
    SPD-36 curve, read from row3 reports (`accept_curve`) or from `tools/accept_hist.py --curve`;
  * the target's own ceiling on the same text: a drafter that has learned the target's distribution
    and draws from it lands on the greedy token at position m with probability p1(m), the target's
    top-1 probability there (tools/entropy_ceiling.py). Walked through the loop exactly -- a block
    starts where the last one ended, so the anchors are the loop's own, not every position -- that
    gives c_i, the same conditional rate, and the tokens a round such a drafter commits.

The text is the atlas row's own fifty prompts (`prompts-mixed-v1`, buckets s/m, seed 42, 256
tokens): synthetic, authored in the atlas repository, never read by `tools/train_data.py` and not
drawn from the public datasets `tools/h100/prompts.py` trains on -- a distinct-prompt holdout by
construction (the 2026-09-17 leak lesson). The bench's prose and chat prompts can be added.

    # on the board, inside ops/hold.sh (it loads the target): greedy continuations + p1
    python tools/p1_slots.py record --out results/trn7/p1.npz --atlas ~/inf-atlas --bench prose,chat

    # anywhere: the projection against the served curves -- the row's per prompt from its row3
    # server logs (three runs of 3 warm-ups + 50 requests each), the bench's from accept_hist
    python tools/p1_slots.py project results/trn7/p1.npz \\
        --row-log results/row3/p1final-nostore/server.log --curves results/p0/curve-tree.json

A share `s` of the per-slot gap closed on slots 1..8 (the slots a position-weighted loss acts on)
gives a_i' = a_i + s (c_i - a_i); tokens a round T(a) = 1 + sum_k prod_{j<=k} a_j; the projection is
T(a')/T(a) - 1, and the share that reaches +5 % is printed beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- the arithmetic (CPU, tested)


def loop_ceiling(q: np.ndarray, width: int) -> dict:
    """The loop with a drafter that is right at position m with probability q[m], walked exactly.

    `q[m]` is the target's top-1 probability for generated token m (index 0 is the token the
    prefill produced, which is the first anchor and is never drafted). From an anchor at a, slot i
    drafts token a + i; the block commits the accepted run plus the target's own next token, and the
    next block's anchor is that token. P[a] is the probability that a block is anchored at a, so the
    expected counts are exact, not sampled. Returns rounds, tokens, and per slot the probability mass
    that offered it and accepted it (censored where the text ends, as the live curve is).
    """
    n = len(q)
    P = np.zeros(n + width + 2)
    P[0] = 1.0
    offered = np.zeros(width + 1)
    accepted = np.zeros(width + 1)
    rounds = 0.0
    for a in range(n - 1):
        pa = P[a]
        if pa == 0.0:
            continue
        rounds += pa
        surv = 1.0
        for i in range(1, width + 1):
            if a + i >= n:
                surv = 0.0          # the text ended: nothing past it is offered or committed
                break
            offered[i] += pa * surv
            P[a + i] += pa * surv * (1.0 - q[a + i])    # slot i missed: its target token anchors
            surv *= q[a + i]
            accepted[i] += pa * surv
        if surv > 0.0:
            P[a + width + 1] += pa * surv               # the whole width, then the bonus token
    return {"rounds": rounds, "tokens": float(n - 1), "offered": offered[1:],
            "accepted": accepted[1:]}


def combine(parts: list[dict]) -> dict:
    """Sum `loop_ceiling` over sequences: tokens a round and the conditional rate per slot."""
    rounds = sum(p["rounds"] for p in parts)
    tokens = sum(p["tokens"] for p in parts)
    off = np.sum([p["offered"] for p in parts], axis=0)
    acc = np.sum([p["accepted"] for p in parts], axis=0)
    rate = np.where(off > 0, acc / np.maximum(off, 1e-12), np.nan)
    return {"tokens_per_round": tokens / rounds if rounds else float("nan"), "rate": rate,
            "offered": off, "rounds": rounds, "tokens": tokens}


def tokens_per_round(rates) -> float:
    """1 + sum_k prod_{j<=k} a_j: the bonus token plus the expected accepted run of a curve."""
    t, surv = 1.0, 1.0
    for r in rates:
        if r is None or not np.isfinite(r):
            break
        surv *= float(r)
        t += surv
    return t


def project(live, ceiling, share: float, slots: range = range(1, 9)) -> list[float]:
    """The live curve with `share` of the gap to the ceiling closed on `slots` (1-based)."""
    out = []
    for i, a in enumerate(live, start=1):
        c = ceiling[i - 1] if i - 1 < len(ceiling) else np.nan
        if i in slots and np.isfinite(c) and c > a:
            a = a + share * (c - a)
        out.append(float(a))
    return out


def breakeven(live, ceiling, target: float = 0.05, slots: range = range(1, 9)) -> float | None:
    """The share of the per-slot gap that projects `target` more tokens a round (None: not even
    the whole gap does)."""
    base = tokens_per_round(live)
    if tokens_per_round(project(live, ceiling, 1.0, slots)) / base - 1 < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if tokens_per_round(project(live, ceiling, mid, slots)) / base - 1 < target:
            lo = mid
        else:
            hi = mid
    return hi


def row_hists(logs: list[str], plens: list[int], warm: int = 3) -> list[dict]:
    """Per row prompt, the served accept histogram summed over every run in `logs`.

    A row3 server log holds its runs back to back, each `warm` warm-up requests (the first prompts
    again) and then the prompts in order, so request j of a run is prompt j - warm. The prompt
    lengths the `[req]` lines print must equal the recorded ones: that is the check that the
    mapping and the template are the row's."""
    from tools import rowlog
    per = [dict() for _ in plens]
    k = warm + len(plens)
    for path in logs:
        reqs = rowlog.parse_requests(Path(path).read_text())
        if len(reqs) % k:
            raise SystemExit(f"{path}: {len(reqs)} requests is not a whole number of {k}-request runs")
        for r0 in range(0, len(reqs), k):
            for j, r in enumerate(reqs[r0 + warm:r0 + k]):
                if r["prompt"] != plens[j]:
                    raise SystemExit(f"{path}: request {r0 + warm + j} has {r['prompt']} prompt "
                                     f"tokens, the recorded prompt {j} has {plens[j]}")
                if r.get("blocks"):
                    rowlog.add_hist(per[j], r["accept"])
    return per


def hist_curve(hists: list[dict]) -> tuple[list[float], int]:
    from tools import rowlog
    acc: dict = {}
    for h in hists:
        rowlog.add_hist(acc, h)
    cv = [c for c in rowlog.curve(acc) if c.get("rate") is not None]
    blocks = sum(sum(v.values()) for v in acc.values())
    return [c["rate"] for c in cv], blocks


def curve_from_report(paths: list[str]) -> tuple[list[float], int]:
    """The served curve of one or more row3 reports, their per-slot counts summed."""
    acc: dict[int, list[float]] = {}
    for p in paths:
        for c in json.load(open(p))["accept_curve"]:
            if c.get("rate") is None:
                continue
            a = acc.setdefault(c["slot"], [0.0, 0.0])
            a[0] += c["rate"] * c["n"]
            a[1] += c["n"]
    slots = sorted(acc)
    return [acc[s][0] / acc[s][1] for s in slots], int(acc[slots[0]][1]) if slots else 0


# ---------------------------------------------------------------- the board (GPU)


def row_prompts(atlas: str, count: int = 50, seed: int = 42, target: int = 256) -> list[dict]:
    """The atlas row's prompts, picked the way its serving workload picks them."""
    sys.path.insert(0, os.path.join(atlas, "bench"))
    from atlas_bench.data import filter_prompt_rows, load_prompt_rows, sample_prompts
    from atlas_bench.registry import Registry
    reg = Registry(atlas)
    rows = filter_prompt_rows(load_prompt_rows(reg, "prompts-mixed-v1"), ["s", "m"])
    return [{"name": r.id, "topic": r.topic, "messages": r.messages}
            for r in sample_prompts(rows, count, seed=seed, target_tokens=target)]


def record(a) -> None:
    import torch
    from transformers import AutoTokenizer

    from engine.config import load_config
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    from engine.spec import generate_spec
    from tools.bench_decode import PROMPTS

    prompts = row_prompts(a.atlas) if a.atlas else []
    for name in [b for b in a.bench.split(",") if b]:
        prompts.append({"name": f"bench-{name}", "topic": f"bench-{name}",
                        "messages": [{"role": "user", "content": PROMPTS[name]}]})
    cfg = load_config(a.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eng = Qwen38Engine(cfg, Weights(cfg.path, device="cuda", skip_mtp=True), max_len=a.max_len,
                       device="cuda")
    dr = DFlash2Drafter(eng, a.ckpt, blocks=1, max_len=a.max_len, block=16)
    eos = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
           if isinstance(t, int)]
    names, topics, plens, q_all, agree_all = [], [], [], [], []
    for i, p in enumerate(prompts):
        enc = tok.apply_chat_template(p["messages"], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt",
                                      return_dict=True)
        ids = enc["input_ids"][0].to("cuda")
        # the greedy continuation, through the block drafter: lossless, and ~4x faster than plain
        out, _ = generate_spec(eng, ids, a.new, dr, 15, eos=eos)
        out = [t for t in out if t not in eos]
        full = ids.tolist() + out
        n0 = ids.numel()
        # the target's own top-1 probability at every generated position, teacher-forced
        eng.reset()
        q = np.zeros(len(out))
        agree = np.zeros(len(out), dtype=bool)
        with torch.no_grad():
            at = 0
            while at < len(full):
                piece = full[at:at + 256]
                logits = eng.forward(torch.tensor(piece, device="cuda"), start=at)[0].float()
                pr = torch.softmax(logits, dim=-1)
                top, arg = pr.max(dim=-1)
                for r in range(len(piece)):
                    m = at + r + 1 - n0         # row at+r predicts token at+r+1
                    if 0 <= m < len(out):
                        q[m] = float(top[r])
                        agree[m] = int(arg[r]) == out[m]
                at += len(piece)
        names.append(p["name"]); topics.append(p["topic"]); plens.append(n0)
        q_all.append(q); agree_all.append(agree)
        print(f"[p1] {i + 1}/{len(prompts)} {p['name']:<18} {p['topic']:<14} {len(out):4d} tok  "
              f"mean p1 {q.mean():.3f}  argmax == greedy {agree.mean():.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, names=np.array(names), topics=np.array(topics),
                        plens=np.array(plens), lens=np.array([len(q) for q in q_all]),
                        q=np.concatenate(q_all), agree=np.concatenate(agree_all))
    print(f"[p1] {a.out}: {len(names)} sequences, {sum(len(q) for q in q_all)} positions")


# ---------------------------------------------------------------- the projection (CPU)


def load_npz(path: str) -> list[dict]:
    d = np.load(path)
    out, at = [], 0
    for name, topic, pl, n in zip(d["names"], d["topics"], d["plens"], d["lens"]):
        out.append({"name": str(name), "topic": str(topic), "plen": int(pl), "q": d["q"][at:at + n],
                    "agree": d["agree"][at:at + n]})
        at += n
    return out


def show(label: str, seqs: list[dict], live: list[float], width: int, shares) -> dict:
    ceil = combine([loop_ceiling(s["q"], width) for s in seqs])
    q = np.concatenate([s["q"] for s in seqs])
    t_live, t_ceil = tokens_per_round(live), ceil["tokens_per_round"]
    k = min(len(live), width)
    print(f"\n{label}: {len(seqs)} sequences, {len(q)} positions, mean p1 {q.mean():.3f}, "
          f"median {np.median(q):.3f}, p1 < 0.5 {100 * (q < 0.5).mean():.1f} %, "
          f"argmax == greedy {np.concatenate([s['agree'] for s in seqs]).mean():.4f}")
    print("    slot            " + " ".join(f"{i:>6d}" for i in range(1, k + 1)))
    print("    live a_i        " + " ".join(f"{x:6.3f}" for x in live[:k]))
    print("    ceiling c_i     " + " ".join(f"{x:6.3f}" for x in ceil["rate"][:k]))
    print("    headroom        " + " ".join(f"{c - a:+6.3f}" for a, c in zip(live[:k], ceil["rate"][:k])))
    print(f"    tokens a round: live curve {t_live:.3f}; p1-drafter in the loop {t_ceil:.3f} "
          f"({100 * (t_ceil / t_live - 1):+.1f} %); the live curve with the ceiling's rates on "
          f"slots 1-8 {tokens_per_round(project(live, ceil['rate'], 1.0)):.3f}")
    proj = {}
    for s in shares:
        t = tokens_per_round(project(live, ceil["rate"], s))
        proj[s] = 100 * (t / t_live - 1)
    be = breakeven(live, ceil["rate"])
    print("    OPD share of the slot 1-8 gap -> tokens a round: "
          + ", ".join(f"{int(100 * s)} % {v:+.1f} %" for s, v in proj.items())
          + f"; +5 % needs {'more than the whole gap' if be is None else f'{100 * be:.0f} % of it'}")
    return {"label": label, "sequences": len(seqs), "positions": int(len(q)),
            "mean_p1": float(q.mean()), "live": live[:k], "ceiling": [float(x) for x in ceil["rate"][:k]],
            "tokens_live": t_live, "tokens_ceiling": t_ceil,
            "projection_pct": {str(s): v for s, v in proj.items()}, "breakeven_share": be}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--atlas", default="", help="an inference-atlas checkout: the row's prompts")
    r.add_argument("--bench", default="", help="bench prompts to add, e.g. prose,chat")
    r.add_argument("--ckpt", default=os.path.expanduser("~/qwen38-spark-engine/train/ft-b16"))
    r.add_argument("--new", type=int, default=256)
    r.add_argument("--max-len", type=int, default=4096)
    r.add_argument("--model", default=None)
    p = sub.add_parser("project")
    p.add_argument("npz")
    p.add_argument("--row", action="append", default=[], help="row3 reports: the row's live curve")
    p.add_argument("--row-log", action="append", default=[],
                   help="row3 server logs: the live curve per prompt, so it splits by topic")
    p.add_argument("--curves", default="", help="accept_hist --curve JSON: per-workload curves")
    p.add_argument("--width", type=int, default=15)
    p.add_argument("--shares", default="0.1,0.25,0.5")
    p.add_argument("--json", default="")
    a = ap.parse_args()
    if a.stage == "record":
        record(a)
        return
    seqs = load_npz(a.npz)
    shares = [float(x) for x in a.shares.split(",")]
    out = []
    row = [s for s in seqs if not s["topic"].startswith("bench-")]
    if a.row and row:
        live, n = curve_from_report(a.row)
        out.append(show(f"the row ({len(a.row)} reports, {n} blocks)", row, live, a.width, shares))
    if a.row_log and row:
        hists = row_hists(a.row_log, [s["plen"] for s in row])
        groups = [("the row, every prompt", lambda t: True),
                  ("the row, prose/chat topics (no code, no math)", lambda t: t not in ("code", "math")),
                  ("the row, code + math", lambda t: t in ("code", "math"))]
        for label, keep in groups:
            idx = [i for i, s in enumerate(row) if keep(s["topic"])]
            live, n = hist_curve([hists[i] for i in idx])
            out.append(show(f"{label} ({len(idx)} prompts, {n} blocks)", [row[i] for i in idx],
                            live, a.width, shares))
    if a.curves:
        cv = json.load(open(a.curves))
        for s in seqs:
            name = s["topic"].removeprefix("bench-")
            if s["topic"].startswith("bench-") and name in cv:
                live = [c["rate"] for c in cv[name]["curve"] if c.get("rate") is not None]
                out.append(show(f"bench {name}", [s], live, a.width, shares))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
