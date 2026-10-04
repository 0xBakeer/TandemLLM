"""Statistics of the drafter's taps (the target's residual stream after the tapped layers).

    python tools/kd_tapstats.py --set DIR --gen 'GLOB' [--seqs 16]

Per tap: the RMS of a row, the per-channel RMS over all rows (median, max, and the share of the
row's energy in the top 8 channels), and how much a row changes along the sequence (cosine of
consecutive rows after removing the channel mean). A few channels with most of the energy and rows
that barely change are what a randomly started `fc` + RMSNorm cannot see past.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--taps", default="1,13,25,37,49")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from tools.kd_train import load_gen, load_target, target_pass
    taps = [int(x) for x in a.taps.split(",")]
    eng = load_target(a.set, "cuda", 8192, 4096)
    seqs = load_gen(a.gen, 6144, split="train")[: a.seqs]
    H = eng.cfg.hidden
    rows = []
    for s in seqs:
        eng.reset()
        t, *_ = target_pass(eng, torch.tensor(s.ids, device="cuda"), taps)
        rows.append(t.float())
    X = torch.cat(rows)                                   # [N, T*H]
    out = {}
    for j, L in enumerate(taps):
        x = X[:, j * H:(j + 1) * H]
        row_rms = x.pow(2).mean(-1).sqrt()
        ch_rms = x.pow(2).mean(0).sqrt()
        ch_mean = x.mean(0)
        top8 = ch_rms.pow(2).topk(8).values.sum() / ch_rms.pow(2).sum()
        xc = x - ch_mean
        cos_adj = torch.nn.functional.cosine_similarity(xc[1:], xc[:-1], dim=-1).mean()
        mean_share = ch_mean.pow(2).sum() / x.pow(2).mean(0).sum()
        out[L] = {"row_rms_median": row_rms.median().item(), "ch_rms_median": ch_rms.median().item(),
                  "ch_rms_max": ch_rms.max().item(), "top8_energy_share": top8.item(),
                  "mean_energy_share": mean_share.item(), "adjacent_cos_centered": cos_adj.item(),
                  "top_channels": ch_rms.topk(8).indices.tolist()}
        print(f"[taps] layer {L}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in out[L].items()}), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
