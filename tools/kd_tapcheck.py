"""Do the drafter's training taps equal what the engine's decode step hands a drafter?

    python tools/kd_tapcheck.py --set DIR [--steps 64] [--prompt-file F]

The trainer reads the residual after layers 1, 13, 25, 37 and 49 from its own prefill pass
(`tools/kd_train.target_pass`); at serving the engine's decode graph writes them to
`eng.dec_taps` (`KolibriEngine.set_taps`). This decodes `--steps` tokens greedily through
the captured decode step with the taps on, then runs `target_pass` over the same text and
compares every decoded position's taps in BF16 (what the drafter reads): the share of elements
bit-equal, within 1 and 2 BF16 ulps, the worst ulp distance, and the relative RMS difference per
tap. It passes when, in every tap, at least 99 % of the elements are within 1 ulp and the relative
RMS difference is under 1 %. Run it on the build that will serve, before a drafter is wired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = ("<|im_start|>user\nErkläre in drei Sätzen, wie ein Kühlschrank funktioniert, und schreibe "
          "dann eine Python-Funktion, die prüft, ob eine Zahl prim ist.<|im_end|>\n<|im_start|>assistant\n"
          "<think>\n\n</think>\n\n")


def ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BF16 ulp distance per element (same-sign ordering of the 16-bit patterns)."""
    def key(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (key(a) - key(b)).abs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--taps", default="1,13,25,37,49")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--drafter", default=None, help="also: the drafter's held-out walk with decode-step taps "
                    "against prefill taps, on --seqs greedy held-out answers")
    ap.add_argument("--held", default=None, help="glob of generation shards (held-out split)")
    ap.add_argument("--seqs", type=int, default=24)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from tools.kd_train import load_target, target_pass
    taps = [int(x) for x in a.taps.split(",")]
    eng = load_target(a.set, "cuda", 8192, 4096)
    eng.set_taps(taps)
    tokp = os.path.join(a.set, "tokenizer.json")
    text = open(a.prompt_file).read() if a.prompt_file else PROMPT
    if os.path.exists(tokp):
        ids = Tokenizer.from_file(tokp).encode(text, add_special_tokens=False).ids
    else:
        ids = [127904, 882, 198, 5000, 6000, 7000, 127906, 198, 127904, 78191, 198]
    H = eng.cfg.hidden
    with torch.inference_mode():
        eng.reset()
        _, lab, _, _ = target_pass(eng, torch.tensor(ids, device="cuda"), taps)
        toks = [int(lab[-1])]
        dec = []
        for _ in range(a.steps):
            lg = eng.decode(toks[-1])
            dec.append(eng.dec_taps.clone())          # taps of the row just decoded
            toks.append(int(lg.argmax()))
        full = ids + toks[:-1]
        eng.reset()
        tp, *_ = target_pass(eng, torch.tensor(full, device="cuda"), taps)
    D = torch.stack(dec)                               # [steps, T, H] fp32
    P = tp[len(ids):len(ids) + a.steps].view(a.steps, len(taps), H)   # bf16
    res, ok = {}, True
    for j, L in enumerate(taps):
        d = D[:, j].to(torch.bfloat16)
        p = P[:, j]
        u = ulps(d, p)
        rel = ((d.float() - p.float()).pow(2).mean().sqrt() / p.float().pow(2).mean().sqrt()).item()
        r = {"equal": (u == 0).float().mean().item(), "within_1ulp": (u <= 1).float().mean().item(),
             "within_2ulp": (u <= 2).float().mean().item(), "max_ulp": int(u.max()), "rel_rms": rel}
        r["pass"] = r["within_1ulp"] >= 0.99 and rel < 0.01
        ok &= r["pass"]
        res[L] = r
        print(f"[tapcheck] layer {L}: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                                      for k, v in r.items()}), flush=True)
    print(f"[tapcheck] {'PASS' if ok else 'FAIL'}: decode-step taps against the trainer's prefill taps "
          f"over {a.steps} decoded tokens", flush=True)
    report = {"pass": ok, "taps": res}
    if a.drafter:
        report["walk"] = drafter_walk(a, eng, taps)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


def drafter_walk(a, eng, taps):
    """The held-out walk of a trained drafter twice over the same greedy answers: once with the taps
    of the trainer's prefill pass (what the gate uses), once with the decode step's taps for every
    answer position (what serving hands it; prompt rows stay prefill, as a served prefill gives).
    Labels are the decode step's argmax in both, so only the tap source differs."""
    import json as _j
    from safetensors.torch import load_file
    from engine.drafters.dspark import DSparkConfig
    from tools.kd_train import acceptance, build_module, load_gen, target_pass
    raw = _j.load(open(os.path.join(a.drafter, "config.json")))
    w = {k: v.to("cuda") for k, v in load_file(os.path.join(a.drafter, "model.safetensors")).items()}
    cfg, m = build_module(raw, w)
    head = (eng.head.w.float() * eng.head.s[:, None]).to(torch.bfloat16)
    seqs = [s for s in sorted(load_gen(a.held, 6144, split="heldout"), key=lambda s: s.id)
            if s.mode == "greedy"][: a.seqs]
    H = eng.cfg.hidden
    given_pre, given_dec = {}, {}
    with torch.inference_mode():
        for s in seqs:
            ids = torch.tensor(s.ids, device="cuda")
            eng.reset()
            tp, lab, _, _ = target_pass(eng, ids, taps)
            eng.reset()
            tq, lq, _, _ = target_pass(eng, ids[: s.gen_start], taps)
            dec_t = [tq]
            dec_l = [lq]
            for t in s.ids[s.gen_start:]:
                lg = eng.decode(int(t))
                dec_t.append(eng.dec_taps.reshape(1, -1).to(torch.bfloat16).clone())
                dec_l.append(lg.argmax().view(1))
            td = torch.cat(dec_t)
            ld = torch.cat(dec_l)
            given_pre[s.id] = (tp, ld)
            given_dec[s.id] = (td, ld)
    out = {}
    for name, given in (("prefill_taps", given_pre), ("decode_taps", given_dec)):
        r = acceptance(m, cfg, eng, eng.emb, head, seqs, taps, "cuda", given=given)["ALL"]
        out[name] = {"chain3": r["chains"]["3"], "slot_rates": r["slot_rates"][:3], "seqs": r["seqs"]}
        print(f"[tapcheck] drafter walk, {name}: chain 3 {r['chains']['3']:.3f}, slots "
              + " ".join(f"{x:.3f}" for x in r["slot_rates"][:3] if x is not None), flush=True)
    return out


if __name__ == "__main__":
    main()
