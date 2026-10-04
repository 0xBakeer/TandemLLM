"""Time one draft call of the Kolibri block drafter in a CUDA graph, for several module shapes and
two runtimes, so the drafter's size is chosen against the gate R = (draft + 4-row verify) / decode.

    python tools/kd_draft_time.py --set SET_DIR [--variants base,lean,lean-fc4,...] [--json out]

Random weights of the right shapes time the same as trained ones, so this needs no export: each
variant builds its drafter from a config. Only the target's `outside.safetensors` is loaded.

Two runtimes:

  base  the `tools/kolibri_draft_time.py` call: `DSparkModule.forward_block` over a
        2,048-row window with its sliding mask; K/V repeated to the 32 query heads and the block's
        K/V concatenated to the context's, every layer, every call.
  lean  the same arithmetic without the copies: the draft cache keeps the last W - 8 context rows
        plus 8 block slots in one buffer per layer, the block writes its own K/V into those slots,
        and attention is one SDPA call over the buffer with `enable_gqa` and no mask (every row is
        visible). The only difference from training is the window edge: training lets row j see
        rows down to p + j - W, here every row sees the same W - 8 rows. That is 8 rows of 2,048 at
        the far end of the window, noise for a drafter.

Variant names: `base` and `lean`, then `-fc4` (fc in NVFP4), `-L4` / `-L3` (layers), `-q16` (16 query
heads instead of 32), `-nomk` (no Markov head), `-h64k` (a gathered head of 64,000 rows instead of
128,000), combinable, e.g. `lean-fc4-L4-nomk`. The projections of the layers are NVFP4 throughout
(the engine's `quantise_projections`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT = ("base,lean,lean-fc4,lean-fc4-nomk,lean-fc4-L4,lean-fc4-L3,lean-fc4-q16,lean-fc4-h64k,"
           "lean-fc4-L4-q16,lean-fc4-L3-q16-nomk")


def variant_config(name: str) -> dict:
    from tools.kd_train import drafter_config
    parts = name.split("-")
    layers = 4 if "L4" in parts else 3 if "L3" in parts else 5
    heads = 16 if "q16" in parts else 32
    markov = 0 if "nomk" in parts else 256
    return drafter_config(layers, 6144, heads, 4, [1, 13, 25, 37, 49], 2048, 8, markov)


def lean_draft(m, cfg, w, emb, head, new_taps, new_pos, slots, K, V, blk_pos, blk_ids, anchor, fc):
    """One round: the new context rows into the cache, the block through the layers, the head,
    the Markov refinement. Same math as `DSparkModule` (project_context, context_kv, forward_block)."""
    from engine.drafters.dflash2 import _apply_rope, _lin, _rms
    from tools.head_gemv import head_matmul_fp8
    eps = cfg.rms_norm_eps
    hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    ctx_h = _rms(_lin(new_taps, fc), w["hidden_norm.weight"], eps)
    n = ctx_h.shape[0]
    cos, sin = m.rope(new_pos, hd, ctx_h.dtype)
    bs = blk_ids.numel()
    Wc = K.shape[2] - bs
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}.self_attn"
        k = _rms(_lin(ctx_h, w[f"{p}.k_proj.weight"]).view(n, nkv, hd), w[f"{p}.k_norm.weight"], eps)
        k = _apply_rope(k.transpose(0, 1)[None], cos, sin)[0]
        v = _lin(ctx_h, w[f"{p}.v_proj.weight"]).view(n, nkv, hd).transpose(0, 1)
        K[i].index_copy_(1, slots, k)
        V[i].index_copy_(1, slots, v)
    blk_ids[0] = anchor[0]
    h = emb[blk_ids]
    cb, sb = m.rope(blk_pos, hd, h.dtype)
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}"
        res = h
        x = _rms(h, w[f"{p}.input_layernorm.weight"], eps)
        q = _rms(_lin(x, w[f"{p}.self_attn.q_proj.weight"]).view(bs, nh, hd), w[f"{p}.self_attn.q_norm.weight"], eps)
        k = _rms(_lin(x, w[f"{p}.self_attn.k_proj.weight"]).view(bs, nkv, hd), w[f"{p}.self_attn.k_norm.weight"], eps)
        v = _lin(x, w[f"{p}.self_attn.v_proj.weight"]).view(bs, nkv, hd)
        q = _apply_rope(q.transpose(0, 1)[None], cb, sb)
        K[i, :, Wc:] = _apply_rope(k.transpose(0, 1)[None], cb, sb)[0]
        V[i, :, Wc:] = v.transpose(0, 1)
        o = F.scaled_dot_product_attention(q, K[i][None], V[i][None], enable_gqa=True)
        h = res + _lin(o.transpose(1, 2).reshape(bs, -1), w[f"{p}.self_attn.o_proj.weight"])
        res = h
        x = _rms(h, w[f"{p}.post_attention_layernorm.weight"], eps)
        x = _lin(F.silu(_lin(x, w[f"{p}.mlp.gate_proj.weight"])) * _lin(x, w[f"{p}.mlp.up_proj.weight"]),
                 w[f"{p}.mlp.down_proj.weight"])
        h = res + x
    h = _rms(h, w["norm.weight"], eps)
    logits = head_matmul_fp8(h[1:], head)
    ids = logits.argmax(-1)
    if cfg.markov_rank:
        prev = torch.cat([anchor, ids[:-1]])
        bias = m.markov_bias(m.markov_latent(prev)).float()
        ids = (logits + bias[:, : logits.shape[1]]).argmax(-1)   # a gathered head reads its rows only
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="the NVFP4 set directory")
    ap.add_argument("--variants", default=DEFAULT)
    ap.add_argument("--accepted", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from engine.drafters.dflash2 import quantise_projections
    from engine.drafters.dspark import DSparkConfig, DSparkModule
    from engine.kolibri.kernels import E4M3Head
    from engine.kolibri.weights import load_outside
    from tools.head_gemv import head_matmul_fp8
    from tools.kd_train import init_params
    from tools.quant_nvfp4 import quantize_clipped
    dev = torch.device("cuda")
    emb, _, head_full = load_outside(os.path.expanduser(a.set), dev)
    out = {}
    for name in a.variants.split(","):
        raw = variant_config(name)
        cfg = DSparkConfig(raw)
        params = init_params(cfg, "cuda", 0)
        w = {k: v.detach().to(torch.bfloat16) for k, v in params.items()}
        del params
        if cfg.markov_rank:
            w["markov_head.markov_w2.weight"].normal_(0, 0.02)
        w = quantise_projections(w, cfg.num_hidden_layers)
        m = DSparkModule(cfg, w)
        head = head_full
        if "h64k" in name.split("-"):
            head = E4M3Head(head_full.w[:64000].contiguous(), head_full.s[:64000].contiguous())
        fc = quantize_clipped(w["fc.weight"].float(), None) if "fc4" in name.split("-") else w["fc.weight"]
        L, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
        bs, W, n_new = cfg.block_size, a.ctx, a.accepted + 1
        taps = len(cfg.target_layer_ids) * cfg.hidden_size
        new_taps = torch.randn(n_new, taps, device=dev, dtype=torch.bfloat16)
        blk_ids = torch.full((bs,), int(raw["dflash_config"]["mask_token_id"]), dtype=torch.long, device=dev)
        anchor = torch.tensor([100], dtype=torch.long, device=dev)
        if name.startswith("base"):
            K = torch.zeros(L, nkv, W, hd, dtype=torch.bfloat16, device=dev)
            V = torch.zeros_like(K)
            new_pos = torch.arange(W - n_new, W, device=dev)
            ctx_pos = torch.arange(0, W, device=dev)
            blk_pos = torch.arange(W, W + bs, device=dev)

            def call():
                ctx_h = m.project_context(new_taps)
                for i, (k, v) in enumerate(m.context_kv(ctx_h, new_pos)):
                    K[i].index_copy_(1, new_pos, k)
                    V[i].index_copy_(1, new_pos, v)
                blk_ids[0] = anchor[0]
                h = m.forward_block(emb[blk_ids], blk_pos, [(K[i], V[i]) for i in range(L)], ctx_pos)
                logits = head_matmul_fp8(h[1:], head)
                ids = logits.argmax(-1)
                if cfg.markov_rank:
                    prev = torch.cat([anchor, ids[:-1]])
                    ids = (logits + m.markov_bias(m.markov_latent(prev)).float()).argmax(-1)
                return ids
        else:
            Wc = W - bs
            K = torch.zeros(L, nkv, Wc + bs, hd, dtype=torch.bfloat16, device=dev)
            V = torch.zeros_like(K)
            new_pos = torch.arange(Wc - n_new, Wc, device=dev)
            slots = new_pos % Wc
            blk_pos = torch.arange(Wc, Wc + bs, device=dev)

            def call():
                return lean_draft(m, cfg, w, emb, head, new_taps, new_pos, slots, K, V, blk_pos, blk_ids,
                                  anchor, fc)
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
        nb = sum(t.numel() * t.element_size() for t in w.values() if torch.is_tensor(t))
        out[name] = {"draft_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3),
                     "layers": L, "q_heads": cfg.num_attention_heads, "markov": cfg.markov_rank}
        print(f"[draft] {name:28s} {out[name]['draft_ms']:6.3f} ms (min {out[name]['min_ms']:.3f})", flush=True)
        del g, m, K, V, w
        torch.cuda.empty_cache()
    print(json.dumps(out))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
