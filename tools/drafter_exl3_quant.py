"""Quantise a DFlash2 drafter's projections to EXL3 with exllamav3's per-tensor quantiser (ENG-215).

Runs in an exllamav3 environment (on the box: ~/exl3k/venv). Reads the drafter's bf16 checkpoint
and the Hessians `tools/drafter_hessians.py` captured, calls `quantize_exl3(W^T, H_data, quant_args)`
for every projection (q/k/v share their H, gate/up theirs, as exllamav3 shares a Hessian between
linears of one input), and writes `<base>.trellis/.suh/.svh/.su/.sv/.mul1` per projection into one
safetensors file that `engine/drafters/dflash2.py` loads with QWEN38_DRAFT_EXL3=<file>.

--k sets the bits of every tensor; --k-mlp / --k-down override the MLP's (e.g. 2.5 bpw on average:
--k 3 --k-mlp 2 --k-down 3). The mul1 codebook only (what the engine's kernels decode).

Usage: ~/exl3k/venv/bin/python tools/drafter_exl3_quant.py --ckpt train/ft-b16 --hess DIR --k 3 --out F
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

PROJ = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv",
        "self_attn.o_proj": "o", "mlp.gate_proj": "gate_up", "mlp.up_proj": "gate_up",
        "mlp.down_proj": "down"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hess", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--k-mlp", type=int, default=0, help="gate/up bits (0 = --k)")
    ap.add_argument("--k-down", type=int, default=0, help="down bits (0 = --k-mlp or --k)")
    ap.add_argument("--k-fc", type=int, default=0, help="fc bits (0 = --k)")
    ap.add_argument("--exllamav3", default=os.path.expanduser("~/exl3k/exllamav3-160"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.exllamav3)
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    w = {}
    for path in sorted(glob.glob(os.path.join(a.ckpt, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                w[key] = f.get_tensor(key)
    prefix = "" if "fc.weight" in w else "model."
    n_layers = 1 + max(int(k.split(".")[1]) for k in w if k.startswith(prefix + "layers."))
    hess = {}

    def h_data(name: str) -> dict:
        if name not in hess:
            d = torch.load(os.path.join(a.hess, f"{name}.pt"))
            H = d["H"].cuda()
            hess[name] = {"H": H, "first_key": name, "count": int(d["count"]), "finalized": False,
                          "num_total": int(d["count"]) * H.shape[0],
                          "inf_nan": torch.zeros(2, dtype=torch.long, device="cuda"),
                          "device": torch.device("cuda")}
        return hess[name]

    jobs = []
    for i in range(n_layers):
        for proj, g in PROJ.items():
            k = a.k
            if proj.startswith("mlp."):
                k = a.k_mlp or a.k
                if proj == "mlp.down_proj":
                    k = a.k_down or k
            jobs.append((f"layers.{i}.{proj}", f"layers.{i}.{g}", k))
    jobs.append(("fc", "fc", a.k_fc or a.k))
    out: dict[str, torch.Tensor] = {}
    report = {}
    t0 = time.perf_counter()
    for idx, (base, hname, K) in enumerate(jobs):
        W = w[prefix + base + ".weight"].float()                    # [out, in]
        qa = {"seed": idx, "K": K, "devices": [0], "apply_out_scales": None, "mul1": True,
              "debug_dir": os.path.join(os.path.dirname(a.out) or ".", "debug")}
        _, proxy_err, tensors = quantize_exl3(W.T.contiguous().cuda(), h_data(hname), qa, False)
        for sub, t in tensors.items():
            if t is not None:
                out[f"{base}.{sub}"] = t.contiguous().cpu()
        report[base] = {"K": K, "proxy_err": float(proxy_err), "shape": list(W.shape)}
        print(f"[quant] {base:32s} K{K}  proxy err {float(proxy_err):.6f}  "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
        if hname != (jobs[idx + 1][1] if idx + 1 < len(jobs) else None):
            hess.pop(hname, None)                                   # its group is done
            torch.cuda.empty_cache()
    save_file(out, a.out, metadata={"source": a.ckpt, "format": "exl3-mul1",
                                    "report": json.dumps(report)})
    nb = sum(t.numel() * t.element_size() for k, t in out.items() if k.endswith(".trellis"))
    print(f"[quant] {len(jobs)} projections -> {a.out}  trellis {nb / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
