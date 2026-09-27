"""The plain-FP8 projection path at the verify's row counts: row identity and rate (SPD-59, SPD-61).

On plain FP8 every projection is an `FP8Block` and goes through `tools/fp8_linear.fp8_matmul`. Two
questions decide the next tickets, and neither needs the checkpoint -- random codes of the served
shapes have the same bytes to read and the same tile arithmetic:

  identity   a verify's rows must equal the rows the one-token walk computes (M = 1), or the
             speculative output can drift from plain greedy decode. `fp8_matmul` picks
             `block_m` 16 up to 16 rows and 64 above, so rows 17..32 may take another tile.
             For each served shape and M it counts the rows that differ from the same row alone,
             for the default choice, for every pinned `block_m`, and for each split-K.
  rate       ms and GB/s per shape and M for the default tile and split-K 2/4/8 (the variant
             nothing calls today, `fp8_linear.py`), so the skinny kernel's ticket is sized by a
             measurement (SPD-61's first step).

    python tools/fp8_probe.py --out results/fp8/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fp8_linear import BLOCK, FP8Block, FP8Group, fp8_matmul  # noqa: E402


def served_shapes(cfg) -> dict[str, tuple[int, int, int]]:
    """name -> (N, K, count per step) for every FP8 projection of the target."""
    H, I = cfg.hidden_size, cfg.intermediate_size
    nl, na = len(cfg.linear_layers), len(cfg.attention_layers)
    L = nl + na
    return {
        "mlp.gate/up": (I, H, 2 * L), "mlp.down": (H, I, L),
        "gdn.in_proj_qkv": (cfg.conv_dim, H, nl), "gdn.in_proj_z": (cfg.value_dim, H, nl),
        "gdn.out_proj": (H, cfg.value_dim, nl),
        "attn.q": (cfg.q_dim, H, na), "attn.k/v": (cfg.kv_dim, H, 2 * na), "attn.o": (H, cfg.attn_out_dim, na),
    }


def rand_block(N: int, K: int, g: torch.Generator) -> FP8Block:
    codes = (torch.randn(N, K, generator=g, device="cuda") * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = (torch.rand(N // BLOCK, K // BLOCK, generator=g, device="cuda") * 1e-3 + 1e-4).to(torch.bfloat16)
    return FP8Block(codes, scale)


def rows_differing(w: FP8Block, x: torch.Tensor, **kw) -> int:
    """Rows of an M-row call that differ (bitwise) from the same row computed alone at M = 1 on
    the WALK's path (the default arguments, as the one-token decode runs it)."""
    y = fp8_matmul(x, w, **kw)
    alone = torch.cat([fp8_matmul(x[i:i + 1], w, split_k=kw.get("split_k", 1)) for i in range(x.shape[0])])
    return int((y.view(torch.int16) != alone.view(torch.int16)).any(dim=1).sum())


def timed(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--rows", default="1,2,4,8,12,16,17,20,24,28,32")
    p.add_argument("--splits", default="2,4,8")
    p.add_argument("--block-m", default="16,32,64")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default="results/fp8/probe.json")
    p.add_argument("--group", action="store_true",
                   help="SPD-63: fused FP8Group vs its members (identity and ms) instead of the shape sweep")
    a = p.parse_args()
    if a.group:
        return group_probe(a)
    from engine.config import load_config
    cfg = load_config(a.model)
    shapes = served_shapes(cfg)
    rows = [int(r) for r in a.rows.split(",")]
    splits = [int(s) for s in a.splits.split(",")]
    bms = [int(b) for b in a.block_m.split(",")]
    g = torch.Generator(device="cuda").manual_seed(0)
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "device": torch.cuda.get_device_name(),
           "shapes": {k: list(v) for k, v in shapes.items()}, "identity": {}, "rate": {}}
    step_bytes = 0
    for name, (N, K, count) in shapes.items():
        w = rand_block(N, K, g)
        step_bytes += w.nbytes * count
        ident, rate = {}, {}
        for M in rows:
            x = (torch.randn(M, K, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            d = {"default": rows_differing(w, x)}
            for bm in bms:
                d[f"bm{bm}"] = rows_differing(w, x, block_m=bm)
            for s in splits:
                d[f"sk{s}"] = rows_differing(w, x, split_k=s, block_m=16)
            ident[M] = d
            r = {"default": timed(lambda: fp8_matmul(x, w), a.reps)}
            for bm in bms:
                r[f"bm{bm}"] = timed(lambda: fp8_matmul(x, w, block_m=bm), a.reps)
            for s in splits:
                r[f"sk{s}"] = timed(lambda: fp8_matmul(x, w, split_k=s, block_m=16), a.reps)
            rate[M] = {k: {"ms": round(v, 4), "GBps": round(w.nbytes / v / 1e6, 1)} for k, v in r.items()}
        out["identity"][name] = ident
        out["rate"][name] = rate
        worst = {M: max(v.values()) for M, v in ident.items()}
        print(f"[{name}] N={N} K={K} x{count}  rows differing (max over variants) {worst}")
        for M in (1, 16, 17, 32):
            if M in rate:
                best = min(rate[M], key=lambda k: rate[M][k]["ms"])
                print(f"    M={M:<3} default {rate[M]['default']['ms']:.3f} ms "
                      f"{rate[M]['default']['GBps']:.0f} GB/s; best {best} {rate[M][best]['ms']:.3f} ms "
                      f"{rate[M][best]['GBps']:.0f} GB/s; identity default={ident[M]['default']} "
                      + " ".join(f"{k}={v}" for k, v in ident[M].items() if k != "default"))
        del w
        torch.cuda.empty_cache()
    out["step_projection_GB"] = step_bytes / 1e9
    # the step's projection time at each M, default vs the best variant per shape (row-identical ones only)
    summ = {}
    for M in rows:
        dflt = sum(out["rate"][n][M]["default"]["ms"] * shapes[n][2] for n in shapes)
        best = 0.0
        for n in shapes:
            ok = [k for k, v in out["identity"][n][M].items() if v == 0 and k in out["rate"][n][M]]
            best += min(out["rate"][n][M][k]["ms"] for k in ok) * shapes[n][2] if ok else \
                out["rate"][n][M]["default"]["ms"] * shapes[n][2]
        summ[M] = {"default_ms": round(dflt, 2), "best_identical_ms": round(best, 2)}
    out["step"] = summ
    print(f"projection bytes a step {step_bytes / 1e9:.2f} GB")
    for M, v in summ.items():
        print(f"  M={M:<3} projections a step: default {v['default_ms']:.1f} ms, best row-identical "
              f"variant per shape {v['best_identical_ms']:.1f} ms")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {a.out}")


def group_probe(a) -> None:
    """Each served projection group fused (one launch) against its members launched one by one."""
    from engine.config import load_config
    cfg = load_config(a.model)
    H, I = cfg.hidden_size, cfg.intermediate_size
    nl, na = len(cfg.linear_layers), len(cfg.attention_layers)
    groups = {"mlp.gate_up": ([I, I], nl + na), "attn.qkv": ([cfg.q_dim, cfg.kv_dim, cfg.kv_dim], na),
              "gdn.qkvz": ([cfg.conv_dim, cfg.value_dim], nl)}
    rows = [int(r) for r in a.rows.split(",")]
    g = torch.Generator(device="cuda").manual_seed(0)
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "groups": {}}
    tot = {M: [0.0, 0.0] for M in rows}
    for name, (sizes, count) in groups.items():
        blocks = [rand_block(n, H, g) for n in sizes]
        res = {}
        for M in rows:
            x = (torch.randn(M, H, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            sep = torch.cat([fp8_matmul(x, b) for b in blocks], dim=-1)
            t_sep = timed(lambda: [fp8_matmul(x, b) for b in blocks], a.reps)
            res[M] = {"sep_ms": round(t_sep, 4)}
            res[M]["_sep"] = sep
        grp = FP8Group(blocks, [f"m{i}" for i in range(len(blocks))])
        for M in rows:
            x = (torch.randn(M, H, generator=torch.Generator(device="cuda").manual_seed(M), device="cuda") * 0.5).to(torch.bfloat16)
            sep = torch.cat([fp8_matmul(x, b) for b in blocks], dim=-1)   # members are views now
            fused = fp8_matmul(x, grp)
            bad = int((sep.view(torch.int16) != fused.view(torch.int16)).any(dim=1).sum())
            t_f = timed(lambda: fp8_matmul(x, grp), a.reps)
            res[M].pop("_sep")
            res[M].update({"fused_ms": round(t_f, 4), "rows_differing": bad})
            tot[M][0] += res[M]["sep_ms"] * count
            tot[M][1] += t_f * count
        out["groups"][name] = res
        print(f"[{name}] sizes {sizes} x{count}: " + "; ".join(
            f"M={M} {res[M]['sep_ms']:.3f}->{res[M]['fused_ms']:.3f} ms diff rows {res[M]['rows_differing']}"
            for M in rows if M in (1, 8, 16, 24, 32)))
        del blocks, grp
        torch.cuda.empty_cache()
    out["step"] = {M: {"separate_ms": round(v[0], 2), "fused_ms": round(v[1], 2)} for M, v in tot.items()}
    for M, v in out["step"].items():
        print(f"  M={M:<3} grouped projections a step: separate {v['separate_ms']:.1f} ms, fused {v['fused_ms']:.1f} ms")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
