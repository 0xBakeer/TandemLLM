"""Free go/no-go for a Kolibri-1 block drafter: how many tokens a round can a drafter commit, and
what does that buy on our engine's verify cost ramp?

Inputs are files that already exist; nothing here runs a model.

  * `vllm.pt` from the vLLM cross-check (`tools/kolibri_vllm_check.py`): 34 chat prompts
    through Kolibri's template (English, German, code; thinking on and off), each answered greedily
    by vLLM with Aleph Alpha's plugin on the FP8 policy, then scored with top-5 prompt log-probs.
    The answer positions give p1(m), the policy's own top-1 probability at every token it wrote.

THE CEILING
  A drafter that has learned the target's distribution and proposes its argmax is right at position
  m with probability p1(m) when the request samples at T=1 (rejection sampling accepts the argmax
  with probability p1), and at least that often under greedy. Walked through the loop -- a round
  starts where the last one ended, slot i is tried only if slots < i were accepted -- this gives
  the tokens a round such a drafter commits with a chain of k slots, and the conditional rate per
  slot.

REALISTIC DRAFTERS
  The ceiling is not a drafter. Qwen3.8's served block drafter (z-lab's DFlash2, trained on far
  more data than we can afford) reached the ceiling at slot 1 and 0.82..0.91 of it at slots 2..7
  (QWEN_LIVE / QWEN_CEIL below). We apply those per-slot ratios to Kolibri's p1 ("qwen-grade"), and the same
  ratios times 0.9 and 0.8 for a drafter trained on a small budget.

SPEED
  Round time = draft time + verify time of k+1 rows. A byte ramp for Kolibri at NVFP4
  (consecutive-token expert unions measured on 3,069 tokens): 1 row 1.00, 2 rows 1.20, 4 rows
  1.53, 8 rows 2.06, 16 rows 2.88 times one token. Two time models: proportional (every row's
  bytes at the decode step's own efficiency) and affine (a fixed launch overhead per step plus the
  bytes at the floor rate). Gain = tokens a round x T1 / round time - 1, at the best k.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

# the byte ramp (times one decode token), consecutive-token expert unions, NVFP4 kvfullsh8-like.
RAMP = {1: 1.00, 2: 1.20, 4: 1.53, 8: 2.06, 16: 2.88, 32: 4.04}
# Qwen3.8 served DFlash2, per-slot live rate / ceiling rate (34,255 served blocks).
QWEN_LIVE = [0.774, 0.705, 0.642, 0.675, 0.703, 0.698, 0.707, 0.665, 0.722, 0.699, 0.719, 0.669,
             0.724, 0.635, 0.575]
QWEN_CEIL = [0.775, 0.775, 0.779, 0.801, 0.828, 0.825, 0.833, 0.841, 0.847, 0.846, 0.856, 0.864,
             0.865, 0.868, 0.874]


def set_ramp(spec: str) -> None:
    """--ramp-ms "1:MS,2:MS,4:MS,...": measured verify times by row count, kept as ratios to row 1."""
    pts = dict((int(a), float(b)) for a, b in (x.split(":") for x in spec.split(",")))
    RAMP.clear()
    RAMP.update({k: v / pts[1] for k, v in pts.items()})


def ramp(rows: int) -> float:
    ks = sorted(RAMP)
    if rows in RAMP:
        return RAMP[rows]
    for a, b in zip(ks, ks[1:]):
        if a < rows < b:   # interpolate in log(rows), the unions grow ~ log-linearly
            t = (math.log(rows) - math.log(a)) / (math.log(b) - math.log(a))
            return RAMP[a] + t * (RAMP[b] - RAMP[a])
    raise ValueError(rows)


def load_p1(path: str) -> list[tuple[str, np.ndarray]]:
    import torch
    out = []
    for s in torch.load(path, weights_only=False):
        g = s.get("gen_start")
        if g is None:
            continue
        lp = s["top_lp"][:, 0].numpy()
        ids = np.asarray(s["ids"])
        # row t holds the top-5 for the token at t+1; the answer tokens are ids[g:]
        p = np.exp(lp[g - 1: len(ids) - 1])
        top = s["top_ids"][g - 1: len(ids) - 1, 0].numpy()
        agree = float(np.mean(top == ids[g:]))
        p = p[np.isfinite(p)]
        out.append((s["name"], p, agree))
    return out


def walk(seqs: list[np.ndarray], k: int, eff: list[float], reps: int, rng) -> tuple[float, list[float]]:
    """Monte Carlo of the loop: tokens a round, and the conditional accept rate per slot."""
    tried = np.zeros(k)
    acc = np.zeros(k)
    toks = 0
    rounds = 0
    for _ in range(reps):
        for p in seqs:
            n = len(p)
            m = 0
            while m < n:
                rounds += 1
                j = 0
                while j < k and m + j < n:
                    tried[j] += 1
                    if rng.random() < min(1.0, eff[j] * p[m + j]):
                        acc[j] += 1
                        j += 1
                    else:
                        break
                step = j + 1   # accepted drafts plus the target's own token
                toks += min(step, n - m)
                m += step
    return toks / rounds, list(acc / np.maximum(tried, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True, help="crosscheck/vllm.pt")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--t1-ms", type=float, default=14.1, help="decode step at 1k context, ms")
    ap.add_argument("--overhead-ms", type=float, default=5.0,
                    help="affine model: fixed per-step time (launches); the rest scales with bytes")
    ap.add_argument("--draft-ms", type=float, default=3.0)
    ap.add_argument("--max-k", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ramp-ms", default=None, help="measured verify times by rows (tools/kolibri_spec_check.py prices)")
    a = ap.parse_args()
    if a.ramp_ms:
        set_ramp(a.ramp_ms)
        print(f"ramp from measurement: " + ", ".join(f"{k} rows {v:.2f}x" for k, v in sorted(RAMP.items())))
    rng = np.random.default_rng(0)

    data = load_p1(a.vllm)
    allp = np.concatenate([p for _, p, _ in data])
    agree = np.mean([g for _, _, g in data])
    print(f"answers: {len(data)} sequences, {len(allp)} positions, argmax == greedy {agree:.4f}")
    print(f"  p1 mean {allp.mean():.3f}  median {np.median(allp):.3f}  p1<0.5 {np.mean(allp < 0.5):.3f}"
          f"  p1>0.9 {np.mean(allp > 0.9):.3f}")
    groups = {"all": [p for _, p, _ in data]}
    for key in ("-en-", "-de-", "-code-"):
        groups[key.strip("-")] = [p for n, p, _ in data if key in n]
    for key in ("-on", "-off"):
        groups["think" + key] = [p for n, p, _ in data if n.endswith(key)]
    for g, ps in groups.items():
        if ps:
            c = np.concatenate(ps)
            print(f"  {g:5s} {len(ps):2d} seqs {len(c):5d} pos  p1 mean {c.mean():.3f}")

    ratio = [l / c for l, c in zip(QWEN_LIVE, QWEN_CEIL)]
    scen = {"ceiling": [1.0] * a.max_k,
            "qwen-grade": ratio[: a.max_k],
            "budget-0.9": [0.9 * r for r in ratio[: a.max_k]],
            "budget-0.8": [0.8 * r for r in ratio[: a.max_k]]}
    res = {"positions": int(len(allp)), "p1_mean": float(allp.mean()),
           "p1_median": float(np.median(allp)), "scenarios": {}}
    T1, ov = a.t1_ms, a.overhead_ms
    print(f"\ntime models: T1 {T1} ms; draft {a.draft_ms} ms; proportional T(R)=T1*ramp(R);"
          f" affine T(R)={ov}+({T1}-{ov})*ramp(R)")
    for name, eff in scen.items():
        rows = []
        for k in range(1, a.max_k + 1):
            tau, rates = walk(groups["all"], k, eff, a.reps, rng)
            tp = a.draft_ms + T1 * ramp(k + 1)
            ta = a.draft_ms + ov + (T1 - ov) * ramp(k + 1)
            rows.append({"k": k, "tau": tau, "rates": rates,
                         "gain_prop": tau * T1 / tp - 1, "gain_aff": tau * T1 / ta - 1})
        best_p = max(rows, key=lambda r: r["gain_prop"])
        best_a = max(rows, key=lambda r: r["gain_aff"])
        res["scenarios"][name] = {"rows": rows, "best_prop": best_p, "best_aff": best_a}
        r7 = rows[-1]
        print(f"\n{name}: slot rates at k={a.max_k}: " + " ".join(f"{x:.3f}" for x in r7["rates"]))
        print("   k   tau   gain(prop)  gain(affine)")
        for r in rows:
            print(f"  {r['k']:2d}  {r['tau']:.3f}   {r['gain_prop']:+.1%}      {r['gain_aff']:+.1%}")
        print(f"  best: proportional k={best_p['k']} {best_p['gain_prop']:+.1%};"
              f" affine k={best_a['k']} {best_a['gain_aff']:+.1%}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
