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

# Which projections each target quantises. The exclusions are as load-bearing as the inclusions:
#   * `conv1d`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` and every norm stay bf16. The two small
#     `a`/`b` projections are 48x[48, 5120] -- 0.17 % of the bytes -- and every published recipe
#     that quantises this family keeps them in full precision by name.
#   * the recurrent STATE is fp32 and is not a weight. It is not touched here and must not be:
#     RESEARCH 2.5 measures a bf16 state costing 5.8 AIME points and an fp8 one 56.
#   * the drafter's own weights stay bf16. It proposes; the verify pass decides.
TARGETS: dict[str, tuple[str, ...]] = {
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "gdn": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"),
    "attn": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
}
TARGETS["all"] = TARGETS["mlp"] + TARGETS["gdn"] + TARGETS["attn"]


# ------------------------------------------------------------------ reading the fp8 checkpoint
def fp8_dequant(codes: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own definition: `code * scale[n // 128, k // 128]`."""
    s = scale_inv.float()
    s = s.repeat_interleave(FP8_BLOCK, 0)[: codes.shape[0]]
    s = s.repeat_interleave(FP8_BLOCK, 1)[:, : codes.shape[1]]
    return codes.to(torch.float32) * s


def layer_files(snapshot: str) -> dict[int, str]:
    """Which shard holds which layer. One shard per layer in this checkpoint."""
    idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    out: dict[int, str] = {}
    for key, fname in idx.items():
        name = key[len(LM_PREFIX):] if key.startswith(LM_PREFIX) else key
        if not name.startswith("layers."):
            continue
        out[int(name.split(".")[1])] = os.path.join(snapshot, fname)
    return out


def read_proj(path: str, layer: int, proj: str, device: str) -> torch.Tensor | None:
    """The bf16 weight of one projection, from the stored fp8 pair.

    `proj` is the part after the layer, e.g. `mlp.gate_proj` or `linear_attn.in_proj_qkv`. Returns
    None where the layer does not have it -- 48 of the 64 layers are Gated DeltaNet and have no
    `self_attn`, the other 16 have no `linear_attn`.
    """
    with safe_open(path, framework="pt", device=device) as f:
        keys = set(f.keys())
        for base in (f"{LM_PREFIX}layers.{layer}.{proj}", f"layers.{layer}.{proj}"):
            if f"{base}.weight" in keys:
                codes = f.get_tensor(f"{base}.weight")
                if f"{base}.weight_scale_inv" not in keys:
                    return None                              # bf16 already: conv1d, a/b, norms
                scale = f.get_tensor(f"{base}.weight_scale_inv")
                return fp8_dequant(codes, scale)
    return None


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
    missing_stats: list[str] = []
    t0 = time.time()
    layers = sorted(files)[: args.layers] if args.layers else sorted(files)
    projs = TARGETS[args.targets]
    for layer in layers:
        for proj in projs:
            ref = read_proj(files[layer], layer, proj, args.device)
            if ref is None:
                continue
            fp8_bytes += ref.shape[0] * ref.shape[1] + (ref.shape[0] // FP8_BLOCK) * (
                ref.shape[1] // FP8_BLOCK) * 2
            base = f"layers.{layer}.{proj}"
            if args.mode == "rtn":
                blk = quantize_to_nvfp4(ref.to(torch.bfloat16))
            else:
                act = stats.get(base, None) if stats else None
                if act is None:
                    missing_stats.append(base)
                blk = quantize_clipped(ref.to(torch.bfloat16), act)
            out[f"{base}.weight"] = blk.w.cpu()
            out[f"{base}.weight_scale"] = blk.s.cpu()
            out[f"{base}.weight_scale_2"] = torch.tensor(blk.s2, dtype=torch.float32)
            nvfp4_bytes += blk.nbytes
            del ref, blk
        if layer % 8 == 0:
            torch.cuda.empty_cache()
            print(f"  layer {layer:2d}  {time.time() - t0:6.1f} s", flush=True)
    if missing_stats:
        # Not fatal, and not silent: an unweighted clip search is a different quantiser from the
        # one the MLP gate passed on, and the gate has to know which it is looking at.
        print(f"[warn] {len(missing_stats)} projections had no activation statistics and were "
              f"clipped on unweighted error, e.g. {missing_stats[0]}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out, metadata={"format": "nvfp4", "group": str(GROUP),
                                       "mode": args.mode, "targets": args.targets,
                                       "source": snapshot})
    print(f"[{args.mode}/{args.targets}] {len(layers)} layers, {len(out) // 3} projections  "
          f"fp8 {fp8_bytes / 1e9:.2f} GB -> nvfp4 {nvfp4_bytes / 1e9:.2f} GB  "
          f"({nvfp4_bytes / fp8_bytes:.3f}x)  -> {args.out}")


def cmd_stats(args) -> None:
    """Per-input-channel mean square activation for every quantisable projection, over a corpus.

    The tap is on the pair `Weights.proj` / `engine.model.linear` rather than on the engine's
    methods, so it covers the linear-attention and attention projections without a second copy of
    either forward. Every quantised projection in this engine is reached as
    `linear(x, self.w.proj(name))`, with the lookup immediately before the call, so recording the
    name in `proj` and reading it in `linear` names the activation exactly. `Weights.norm` clears
    it, which is what keeps `in_proj_a` and `in_proj_b` -- fetched through `norm`, bf16, and
    deliberately never quantised -- from being credited with their neighbour's activations.
    """
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    import engine.model as model_mod
    from transformers import AutoTokenizer

    cfg = load_config(args.model)
    w = Weights(cfg.path, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=args.chunk + 16)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    acc: dict[str, torch.Tensor] = {}
    count = {"n": 0}
    wanted = set(TARGETS["all"])

    last: dict[str, str | None] = {"name": None}
    orig_proj, orig_norm, orig_linear = Weights.proj, Weights.norm, model_mod.linear

    def tap_proj(self, name):
        last["name"] = name
        return orig_proj(self, name)

    def tap_norm(self, name):
        last["name"] = None
        return orig_norm(self, name)

    def tap_linear(x, weight):
        name, last["name"] = last["name"], None
        if name is not None and name.startswith("layers.") and \
                name.split(".", 2)[2] in wanted:
            v = x.reshape(-1, x.shape[-1]).float().pow(2).sum(0)
            acc[name] = v if name not in acc else acc[name] + v
        return orig_linear(x, weight)

    Weights.proj = tap_proj
    Weights.norm = tap_norm
    model_mod.linear = tap_linear
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
        Weights.proj, Weights.norm = orig_proj, orig_norm
        model_mod.linear = orig_linear
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
    q.add_argument("--targets", choices=tuple(TARGETS), default="mlp",
                   help="which projections to quantise; see TARGETS for what each excludes")
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
