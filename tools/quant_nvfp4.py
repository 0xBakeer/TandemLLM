"""Quantise the MLP of every layer from the checkpoint's FP8 blocks to NVFP4.

69.7 % of the bytes a decode step reads are `gate_proj`, `up_proj` and `down_proj`. At FP8 they are
267.4 MB per layer; at NVFP4 they are 150.4 MB, and the step's byte floor falls from 98.6 ms to
71.2 ms. Attention, the linear-attention mixers and `lm_head` stay as they are -- they are 30 % of
the bytes and the more delicate 30 %.

Two ways to choose the scales, both here, because the choice is what the quality gate measures:

  * `rtn` -- per group of 16, the scale is `amax / 6`, rounded to e4m3 and rounded up where rounding
    down would clip. No data, no calibration, one pass over the weights.
  * `clip` -- the same, then a short search over scale multipliers per group, minimising the error
    the *model* sees rather than the error the weight sees: each input channel is weighted by its
    mean square activation over a calibration corpus, so a channel the model never excites is
    allowed to be approximated worse than one it leans on. The statistics come from `stats`.

The two levels of scale are not interchangeable with one wider level. A 128x128 fp8 block covers
16,384 weights with a single scale; an NVFP4 group covers 16. That is the entire reason four bits
are usable here, and it is why the format carries the per-tensor `weight_scale_2` as well: the e4m3
group scales themselves need a scale to sit inside e4m3's range.

Usage:

    python tools/quant_nvfp4.py stats --out ~/nvfp4/stats.pt --corpus bench/calib.txt
    python tools/quant_nvfp4.py quant --mode rtn  --out ~/nvfp4/mlp-rtn.safetensors
    python tools/quant_nvfp4.py quant --mode clip --stats ~/nvfp4/stats.pt \
                                      --out ~/nvfp4/mlp-clip.safetensors
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import resolve_snapshot  # noqa: E402
from tools.fp8_linear import BLOCK as FP8_BLOCK  # noqa: E402
from tools.nvfp4_linear import (E2M1_MAX, E4M3_MAX, GROUP, NVFP4Block,  # noqa: E402
                                _round_e2m1, quantize_to_nvfp4)

MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
LM_PREFIX = "model.language_model."


# ------------------------------------------------------------------ reading the fp8 checkpoint
def fp8_dequant(codes: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own definition: `code * scale[n // 128, k // 128]`."""
    s = scale_inv.float()
    s = s.repeat_interleave(FP8_BLOCK, 0)[: codes.shape[0]]
    s = s.repeat_interleave(FP8_BLOCK, 1)[:, : codes.shape[1]]
    return codes.to(torch.float32) * s


def layer_files(snapshot: str) -> dict[int, str]:
    idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    out: dict[int, str] = {}
    for key, fname in idx.items():
        name = key[len(LM_PREFIX):] if key.startswith(LM_PREFIX) else key
        if not name.startswith("layers."):
            continue
        if ".mlp." not in name:
            continue
        out[int(name.split(".")[1])] = os.path.join(snapshot, fname)
    return out


def read_mlp(path: str, layer: int, proj: str, device: str) -> torch.Tensor:
    """The bf16 weight of one MLP projection, from the stored fp8 pair."""
    base = f"{LM_PREFIX}layers.{layer}.mlp.{proj}"
    with safe_open(path, framework="pt", device=device) as f:
        keys = set(f.keys())
        wk = f"{base}.weight"
        if wk not in keys:                                   # some shards drop the prefix
            base = f"layers.{layer}.mlp.{proj}"
            wk = f"{base}.weight"
        codes = f.get_tensor(wk)
        scale = f.get_tensor(f"{base}.weight_scale_inv")
    return fp8_dequant(codes, scale)


# ------------------------------------------------------------------ the clip search
def quantize_clipped(ref: torch.Tensor, act_ms: torch.Tensor | None,
                     ratios=(1.0, 0.95, 0.90, 0.85, 0.80), rows: int = 1024) -> NVFP4Block:
    """Per group of 16, pick the scale multiplier that minimises the activation-weighted error.

    `act_ms[k]` is the mean square of input channel `k` over the calibration corpus. The quantity
    minimised is `sum_k act_ms[k] * (w_k - dequant(q(w_k)))^2`, which is the contribution of this
    group to the squared error of the projection's output, to first order and under the assumption
    that the input channels are uncorrelated. Weighting by activation is the whole point: a channel
    the model never excites can be approximated badly for free.

    A smaller multiplier clips the group's outlier and represents the rest of the group more
    finely. Which side wins is data-dependent, so it is searched rather than argued.
    """
    N, K = ref.shape
    dev = ref.device
    amax_t = ref.abs().max().float()
    scale_2 = float((amax_t / (E2M1_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny))
    if act_ms is None:
        act_ms = torch.ones(K, device=dev, dtype=torch.float32)
    w_k = act_ms.to(dev).float().reshape(1, K // GROUP, GROUP)
    codes = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    scales = torch.empty(N, K // GROUP, dtype=torch.float8_e4m3fn, device=dev)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        x = ref[r0:r1].float().reshape(r1 - r0, K // GROUP, GROUP)
        gmax = x.abs().amax(dim=2, keepdim=True)
        best_err = None
        best_s8 = None
        for ratio in ratios:
            s = (gmax * ratio / E2M1_MAX) / scale_2
            s8 = s.clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn)
            eff = (s8.float() * scale_2).clamp_min(torch.finfo(torch.float32).tiny)
            q = _round_e2m1((x / eff).abs().clamp(0.0, E2M1_MAX)).to(torch.int64)
            grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
            deq = grid[q] * torch.sign(x) * eff
            err = (w_k * (x - deq).pow(2)).sum(dim=2, keepdim=True)
            if best_err is None:
                best_err, best_s8 = err, s8.float()
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s8 = torch.where(take, s8.float(), best_s8)
            del s, s8, eff, q, deq, err
        s8 = best_s8.to(torch.float8_e4m3fn)
        eff = (s8.float() * scale_2).clamp_min(torch.finfo(torch.float32).tiny)
        v = x / eff
        idx = _round_e2m1(v.abs().clamp(0.0, E2M1_MAX))
        nib = (idx | ((v < 0).to(torch.uint8) * 8)).reshape(r1 - r0, K)
        codes[r0:r1] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        scales[r0:r1] = s8.reshape(r1 - r0, K // GROUP)
        del x, gmax, best_err, best_s8, s8, eff, v, idx, nib
    return NVFP4Block(codes, scales, scale_2)


# ------------------------------------------------------------------ commands
def cmd_quant(args) -> None:
    snapshot = resolve_snapshot(args.model)
    files = layer_files(snapshot)
    stats = torch.load(args.stats, map_location="cpu") if args.stats else None
    if args.mode == "clip" and stats is None:
        raise SystemExit("--mode clip needs --stats; run `quant_nvfp4.py stats` first")
    out: dict[str, torch.Tensor] = {}
    fp8_bytes = 0
    nvfp4_bytes = 0
    t0 = time.time()
    layers = sorted(files)[: args.layers] if args.layers else sorted(files)
    for layer in layers:
        for proj in MLP_PROJ:
            ref = read_mlp(files[layer], layer, proj, args.device)
            fp8_bytes += ref.shape[0] * ref.shape[1] + (ref.shape[0] // FP8_BLOCK) * (
                ref.shape[1] // FP8_BLOCK) * 2
            if args.mode == "rtn":
                blk = quantize_to_nvfp4(ref.to(torch.bfloat16))
            else:
                key = f"layers.{layer}.mlp.{proj}"
                blk = quantize_clipped(ref.to(torch.bfloat16),
                                       stats.get(key, None) if stats else None)
            base = f"layers.{layer}.mlp.{proj}"
            out[f"{base}.weight"] = blk.w.cpu()
            out[f"{base}.weight_scale"] = blk.s.cpu()
            out[f"{base}.weight_scale_2"] = torch.tensor(blk.s2, dtype=torch.float32)
            nvfp4_bytes += blk.nbytes
            del ref, blk
        if layer % 8 == 0:
            torch.cuda.empty_cache()
            print(f"  layer {layer:2d}  {time.time() - t0:6.1f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out, metadata={"format": "nvfp4", "group": str(GROUP),
                                       "mode": args.mode, "source": snapshot})
    print(f"[{args.mode}] {len(layers)} layers  fp8 {fp8_bytes / 1e9:.2f} GB -> "
          f"nvfp4 {nvfp4_bytes / 1e9:.2f} GB  ({nvfp4_bytes / fp8_bytes:.3f}x)  -> {args.out}")


def cmd_stats(args) -> None:
    """Per-input-channel mean square activation for every MLP projection, over a corpus."""
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    cfg = load_config(args.model)
    w = Weights(cfg.path, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=args.chunk + 16)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    acc: dict[str, torch.Tensor] = {}
    count = {"n": 0}

    orig_mlp = Qwen38Engine.mlp

    def tap_mlp(self, h, p):
        from engine.model import linear
        gate = linear(h, self.w.proj(f"{p}.mlp.gate_proj"))
        up = linear(h, self.w.proj(f"{p}.mlp.up_proj"))
        mid = F.silu(gate) * up
        for key, act in ((f"{p}.mlp.gate_proj", h), (f"{p}.mlp.up_proj", h),
                         (f"{p}.mlp.down_proj", mid)):
            v = act.reshape(-1, act.shape[-1]).float().pow(2).sum(0)
            acc[key] = v if key not in acc else acc[key] + v
        return linear(mid, self.w.proj(f"{p}.mlp.down_proj"))

    Qwen38Engine.mlp = tap_mlp
    try:
        text = open(args.corpus).read()
        ids = tok(text, return_tensors="pt").input_ids[0]
        n = min(len(ids), args.tokens)
        for start in range(0, n, args.chunk):
            piece = ids[start:start + args.chunk]
            if piece.numel() < 32:
                break
            eng.reset()
            eng.forward(piece.to("cuda"), start=0, last_only=True)
            count["n"] += piece.numel()
            print(f"  {start + piece.numel():6d} / {n} tokens", flush=True)
    finally:
        Qwen38Engine.mlp = orig_mlp
    out = {k: (v / max(1, count["n"])).cpu() for k, v in acc.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"[stats] {len(out)} projections over {count['n']} tokens -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quant")
    q.add_argument("--model", default=None)
    q.add_argument("--mode", choices=("rtn", "clip"), default="rtn")
    q.add_argument("--stats", default=None)
    q.add_argument("--out", required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--layers", type=int, default=0, help="quantise only the first N layers")
    q.set_defaults(fn=cmd_quant)

    s = sub.add_parser("stats")
    s.add_argument("--model", default=None)
    s.add_argument("--corpus", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tokens", type=int, default=8192)
    s.add_argument("--chunk", type=int, default=1024)
    s.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
