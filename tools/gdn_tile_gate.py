"""SPD-245: the 27B's GDN_PREP scan launch at one tiling (value block x warps) against another, one process, one
set of weights, beside the order move already accepted for that path as the yardstick.

A scan tiling moves no recurrent state bit on the measured shapes, but the chunk outputs' tf32 products may round
a few bf16 outputs the other way: a one-time order move, and this is its KL / top-1 check. Arms:

    ref    --ref (32x8, the shared default)
    cand   --tile (16x4, the 27B's default since SPD-245)
    yard   the GDN_PREP kernel off (the fused pair as it was before SPD-191): the order move that shipped

Per prompt class (prose, code, repeated prose) and length: the prompt prefilled in chunks of 1,024 (the served
grid), then --greedy tokens teacher-forced on ref's own greedy continuation; per arm against ref: the argmax
agreement over the positions, KL(ref || arm) at temperature 1 (max and mean), max |dlogit| in bf16 ulps of the
largest logit, and the free-running greedy agreement (tokens until the first split). A second ref run is the
run-to-run floor (it must be bit-equal).

    python tools/gdn_tile_gate.py --ref 32x8 --tile 16x4 --lens 2048,8192,32768 [--json out.json]

`--flag NAME` gates an engine flag instead of a tiling: cand = ref's tile with `engine.model.NAME` on, ref and
ref2 with it off (SPD-245: `--ref 16x4 --tile 16x4 --flag GDN_CHUNKED`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
import engine.model as M  # noqa: E402
import engine.models.qwen3_5.adapter  # noqa: E402,F401  (the cold verbs)

REPO = Path(__file__).resolve().parent.parent


def class_ids(tok, cls: str, n: int) -> list:
    """n prompt tokens of a class: held-out prose, held-out code, or prose's first 1,024 tokens repeated."""
    prose = tok.encode((REPO / "bench" / "heldout_prose.txt").read_text())
    src = {"prose": prose, "code": tok.encode((REPO / "bench" / "heldout_code.txt").read_text()),
           "repeat": prose[:1024]}[cls]
    ids = list(src)
    while len(ids) < n:
        ids += src
    return ids[:n]


FLAG = None


def arm(name: str, tile: tuple) -> None:
    M.GDN_PREP = name != "yard"
    M.GDN_PREP_BV, M.GDN_PREP_WARPS = tile
    if FLAG:
        setattr(M, FLAG, name == "cand")


def run(eng, ids: torch.Tensor, greedy: int, forced: list | None = None) -> tuple:
    """Last-row logits of the prefill and of `greedy` steps after it, and the tokens fed: the argmax
    chain, or `forced` (teacher forcing)."""
    eng.reset()
    n = ids.numel()
    with torch.no_grad():
        for i in range(0, n, 1024):
            lg = eng.forward(ids[i:i + 1024], start=i, last_only=True)
        out = [lg.float().reshape(-1).clone()]
        toks: list = []
        t = int(out[0].argmax()) if forced is None else forced[0]
        for j in range(greedy):
            toks.append(t)
            lg = eng.forward(torch.tensor([t], device=ids.device), start=n + j, last_only=True)
            out.append(lg.float().reshape(-1).clone())
            t = int(out[-1].argmax()) if forced is None else forced[min(j + 1, len(forced) - 1)]
    return torch.stack(out), toks


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    ulp = (a - b).abs().amax(-1) / (a.abs().amax(-1) * 2 ** -8)
    return {"positions": a.shape[0], "argmax_agree": int((a.argmax(-1) == b.argmax(-1)).sum()),
            "bitequal_positions": int((a == b).all(-1).sum()), "kl_max": kl.max().item(),
            "kl_mean": kl.mean().item(), "ulp_max": ulp.max().item(), "ulp_prefill": ulp[0].item()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="32x8", help="reference BVxWARPS")
    ap.add_argument("--tile", default="16x4", help="candidate BVxWARPS")
    ap.add_argument("--lens", default="2048,8192,32768")
    ap.add_argument("--classes", default="prose,code,repeat")
    ap.add_argument("--greedy", type=int, default=64)
    ap.add_argument("--json", default=None)
    ap.add_argument("--flag", default=None, help="an engine.model flag the candidate turns on")
    a = ap.parse_args()
    global FLAG
    FLAG = a.flag
    ref_tile = tuple(int(v) for v in a.ref.split("x"))
    tiles = {"ref": ref_tile, "cand": tuple(int(v) for v in a.tile.split("x")), "yard": ref_tile}
    lens = [int(v) for v in a.lens.split(",")]
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    eng = M.Qwen38Engine(cfg, w, max_len=max(lens) + a.greedy + 64, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    res = {"ref": list(ref_tile), "cand": list(tiles["cand"]), "flag": a.flag, "rows": {}}
    k = 0
    for cls in a.classes.split(","):
        for T in lens:
            ids = torch.tensor(class_ids(tok, cls, T), device="cuda", dtype=torch.long)
            t0 = time.time()
            arm("ref", ref_tile)
            ref_free, toks = run(eng, ids, a.greedy)
            order = ["cand", "yard", "ref2"] if k % 2 == 0 else ["ref2", "yard", "cand"]
            k += 1
            row = {}
            for name in order:
                arm(name, tiles.get(name, ref_tile))
                forced_logits, _ = run(eng, ids, a.greedy, toks)
                _, free = run(eng, ids, a.greedy)
                split = next((j for j, (x, y) in enumerate(zip(toks, free)) if x != y), len(toks))
                row[name] = dict(compare(ref_free, forced_logits), greedy_agree=split)
            row["s"] = round(time.time() - t0, 1)
            res["rows"][f"{cls}-{T}"] = row
            print(f"{cls}-{T}", json.dumps(row), flush=True)
    arm("ref", ref_tile)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
