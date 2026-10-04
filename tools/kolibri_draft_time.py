"""Time one draft call of a DSpark block drafter for Kolibri-1, in a CUDA graph, on the GPU.

    python tools/kolibri_draft_time.py DRAFTER_DIR --set SET_DIR [--accepted 3] [--ctx 2048]

What a round of the loop asks of the drafter, with the shapes of the drafter `tools/kd_train.py` trains (5 layers of
hidden 2560, block 8, taps of 5 x 2560, sliding window 2048, Kolibri's embedding and e4m3 head):

  1. the newly committed positions' taps (accepted + 1 rows) through `fc` and `hidden_norm`, their
     K/V for the draft cache (`context_kv`), written into the cache;
  2. the block (the anchor and 7 mask rows) through the 5 layers over the last 2048 context rows;
  3. rows 1..7 through Kolibri's e4m3 head (tensor-core GEMM), argmax, the Markov bias of the
     previous slot (one read of `markov_w2`), argmax again.

Weights only need the right shapes: a pilot export times the same as a trained one. Reports the
median ms of a graph replay for BF16 layers and for NVFP4 layers (`quantise_projections`). Loads
only the target's `outside.safetensors` (embedding and head), so it runs in seconds.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--set", required=True, help="the NVFP4 set directory")
    ap.add_argument("--accepted", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--profile", action="store_true", help="per-kernel time of one call, eager")
    a = ap.parse_args()
    from engine.drafters.dflash2 import load_weights, quantise_projections
    from engine.drafters.dspark import DSparkConfig, DSparkModule
    from engine.kolibri.weights import load_outside
    from tools.head_gemv import head_matmul_fp8
    dev = torch.device("cuda")
    ck = os.path.expanduser(a.ckpt)
    raw = json.load(open(os.path.join(ck, "config.json")))
    cfg = DSparkConfig(raw)
    w = load_weights(ck, "cuda")
    emb, _, head = load_outside(os.path.expanduser(a.set), dev)
    L, nkv, hd, H = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    taps = len(cfg.target_layer_ids)
    W = a.ctx
    out = {"config": {"layers": L, "hidden": H, "block": cfg.block_size, "window": W,
                      "accepted": a.accepted}}
    for name, weights in (("bf16", w), ("nvfp4", quantise_projections(w, L))):
        m = DSparkModule(cfg, weights)
        K = torch.zeros(L, nkv, W, hd, dtype=torch.bfloat16, device=dev)
        V = torch.zeros_like(K)
        n_new = a.accepted + 1
        new_taps = torch.randn(n_new, taps * H, device=dev, dtype=torch.bfloat16)
        new_pos = torch.arange(W - n_new, W, device=dev)
        ctx_pos = torch.arange(0, W, device=dev)
        blk_pos = torch.arange(W, W + cfg.block_size, device=dev)
        blk_ids = torch.full((cfg.block_size,), int(raw["dflash_config"]["mask_token_id"]),
                             dtype=torch.long, device=dev)
        anchor = torch.tensor([100], dtype=torch.long, device=dev)

        def call():
            ctx_h = m.project_context(new_taps)
            kv = m.context_kv(ctx_h, new_pos)
            for i, (k, v) in enumerate(kv):
                K[i].index_copy_(1, new_pos, k)
                V[i].index_copy_(1, new_pos, v)
            blk_ids[0] = anchor[0]
            noise = emb[blk_ids]
            h = m.forward_block(noise, blk_pos, [(K[i], V[i]) for i in range(L)], ctx_pos)
            logits = head_matmul_fp8(h[1:], head)
            ids = logits.argmax(-1)
            prev = torch.cat([anchor, ids[:-1]])
            bias = m.markov_bias(m.markov_latent(prev)).float()
            return (logits + bias).argmax(-1)

        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            ts = []
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            if a.profile:
                from torch.profiler import ProfilerActivity, profile as tprof
                for _ in range(2):
                    call()
                torch.cuda.synchronize()
                with tprof(activities=[ProfilerActivity.CUDA]) as pr:
                    for _ in range(5):
                        call()
                    torch.cuda.synchronize()
                rows = sorted(((e.key[:70], e.count // 5, e.device_time_total / 5e3) for e in pr.key_averages()
                               if e.device_time_total > 0 and e.device_type.name == "CUDA"), key=lambda r: -r[2])
                for r in rows[:14]:
                    print(f"[draft-prof] {name} {r[0]:70s} n {r[1]:3d} {r[2]:7.3f} ms", flush=True)
        out[name] = {"draft_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3)}
        print(f"[draft] {name}: {out[name]}", flush=True)
        del g, m, K, V
        torch.cuda.empty_cache()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
