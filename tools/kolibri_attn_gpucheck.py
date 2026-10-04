"""GPU parity of Kolibri-1's attention (engine/kolibri/attn.py) against the reference forward.

Two layers of the real checkpoint (a sliding one and a full one; FP8 dequantised to BF16 weights),
real token ids from a text, the input of each layer taken as the embedding through that layer's
`input_layernorm`. The reference is `tools/kolibri_ref.attn_block` in float32 over the whole
sequence. The engine's attention is fed the same rows four ways:

  prefill   chunks of 2048 (SDPA paths)
  mixed     a 1,000-row chunk, then 37-row and 5-row chunks (decode kernels below 64 rows), then
            single tokens through `attn_decode` with a device position (eager)
  graph     the same single tokens through ONE captured CUDA graph of `attn_decode`, replayed at
            every position (what KolibriEngine captures)
  verify    blocks of 8 rows through `attn_block`, half of each rejected (`truncate`) and rewritten

and every row is compared with the reference: max |d| and the relative error of each row.
Also runs `tools/kolibri_attn_kernels.check()` (ring kernel against float32 torch, and row
independence, bf16 and e4m3) and `tools/attn_kernels`' full-layer kernel at head dim 128.

    python tools/kolibri_attn_gpucheck.py --model FP8_RELEASE_DIR --text bench/heldout_prose.txt \
        [--tokens 3000] [--layers 3,4] [--out result.json]

GPU memory: about 2 GB at 3,000 tokens; about 4 GB at 20,000. Never beside a resident engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri import attn as ka  # noqa: E402
from tools import kolibri_ref as ref  # noqa: E402


class LW:
    def __init__(self, **t):
        self.__dict__.update(t)


def rel_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1).clamp_min(1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", action="append", required=True, help="repeat: the texts are joined")
    ap.add_argument("--prefix", type=int, default=1000,
                    help="rows prefilled before the short chunks, decode steps and verify blocks")
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--layers", default="3,4")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    c = json.load(open(os.path.join(a.model, "config.json")))
    from tokenizers import Tokenizer
    tokp = os.path.join(a.model, "tokenizer.json")
    tok = Tokenizer.from_file(tokp)
    ids = tok.encode("\n\n".join(open(t).read() for t in a.text),
                     add_special_tokens=False).ids[: a.tokens]
    N = len(ids)
    ck = ref.Ckpt(a.model)
    emb = ck.get("model.embed_tokens.weight", torch.bfloat16, dev)
    x0 = emb[torch.tensor(ids, device=dev)].float()
    del emb
    res = {"tokens": N, "layers": {}}

    # the kernels on their own first
    from tools import kolibri_attn_kernels as kk
    res["ring_kernel_bf16"] = kk.check(device="cuda")
    res["ring_kernel_fp8"] = kk.check(device="cuda", fp8=True)
    for r in res["ring_kernel_bf16"] + res["ring_kernel_fp8"]:
        print("[ring]", r, flush=True)

    layers = [int(x) for x in a.layers.split(",")]
    # the float32 reference in query blocks of 256: at 20k tokens of context a block's scores are
    # 48 x 256 x 20k x 4 bytes = 1 GB instead of 4 GB
    _core = ref.attn_core
    ref.attn_core = lambda *p, **k: _core(*p, **{**k, "qblock": 256})
    for L in layers:
        p = f"model.layers.{L}."
        sliding = c["layer_types"][L] == "sliding_attention"
        lw32 = ref.load_layer(ck, L, c, dev, torch.float32, experts=False)
        x = ref.rms(x0, lw32.n_in, c["rms_norm_eps"])                  # fp32 [N, H]
        t0 = time.time()
        want = ref.attn_block(x, lw32, c, sliding)                        # fp32 reference
        bf = lambda t: t.to(torch.bfloat16)                               # noqa: E731
        lw = LW(index=L, sliding=sliding, qkv=bf(torch.cat([lw32.q, lw32.k, lw32.v])), o=bf(lw32.o),
                q_norm=bf(lw32.qn), k_norm=bf(lw32.kn))
        xb = bf(x)
        # the reference again in BF16 weights and inputs, to separate rounding from bugs
        lwb = ref.LayerW(q=bf(lw32.q).float(), k=bf(lw32.k).float(), v=bf(lw32.v).float(),
                         o=bf(lw32.o).float(), qn=bf(lw32.qn).float(), kn=bf(lw32.kn).float())
        want_b = ref.attn_block(xb.float(), lwb, c, sliding)
        row = {"sliding": sliding, "ref_seconds": round(time.time() - t0, 2),
               "bf16_weights_vs_fp32_rel_max": rel_rows(want_b, want).max().item()}
        attn = ka.KolibriAttention(c)

        P = min(a.prefix, N - 700)

        def fresh():
            return attn.make_kv(N + 64, dev)

        # prefill in chunks of 2048
        kv = fresh()
        outs = []
        for s in range(0, N, 2048):
            outs.append(attn.attn_prefill(xb[s:s + 2048], lw, kv, s))
            kv.length = min(N, s + 2048)
        got = torch.cat(outs)
        rr = rel_rows(got, want)
        row["prefill"] = {"max_abs": (got.float() - want).abs().max().item(),
                          "rel_max": rr.max().item(), "rel_mean": rr.mean().item()}

        # mixed: chunks, short chunks (kernels), eager decode with a device position
        kv = fresh()
        outs, s = [], 0
        for T in [t for t in [2048] * (P // 2048) + [P % 2048, 37, 5] if t]:
            outs.append(attn.attn_prefill(xb[s:s + T], lw, kv, s))
            s += T
            kv.length = s
        n_dec = min(N - s, 600)
        D0 = P + 42
        for i in range(n_dec):
            pos = torch.tensor([s], dtype=torch.int32, device=dev)
            outs.append(attn.attn_decode(xb[s:s + 1], lw, kv, pos))
            s += 1
            kv.length = s
        got = torch.cat(outs)
        rr = rel_rows(got, want[:s])
        row["mixed"] = {"rows": s, "rel_max": rr.max().item(), "rel_mean": rr.mean().item(),
                        "decode_rel_max": rr[D0:].max().item() if s > D0 else None}
        mixed_tail = got[D0:s]

        # graph: the same decode steps through one captured graph
        kv = fresh()
        s = 0
        for T in [t for t in [2048] * (P // 2048) + [P % 2048, 37, 5] if t]:
            attn.attn_prefill(xb[s:s + T], lw, kv, s)
            s += T
            kv.length = s
        xin = torch.zeros(1, xb.shape[1], dtype=torch.bfloat16, device=dev)
        pin = torch.tensor([s], dtype=torch.int32, device=dev)
        xin.copy_(xb[s:s + 1])
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                attn.attn_decode(xin, lw, kv, pin)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yout = attn.attn_decode(xin, lw, kv, pin)
        gouts = []
        for i in range(n_dec):
            xin.copy_(xb[s:s + 1])
            pin.fill_(s)
            g.replay()
            gouts.append(yout.clone())
            s += 1
            kv.length = s
        gg = torch.cat(gouts)
        rr = rel_rows(gg, want[D0:s])
        row["graph"] = {"rows": n_dec, "rel_max": rr.max().item(),
                        "bit_equal_to_eager_decode": bool(torch.equal(gg, mixed_tail))}

        # verify blocks of 8 with half rejected, then rewritten
        kv = fresh()
        V0 = P
        for c0 in range(0, V0, 2048):
            attn.attn_prefill(xb[c0:min(V0, c0 + 2048)], lw, kv, c0)
        kv.length = V0
        s = V0
        outs = []
        junk = torch.randn_like(xb[:8])
        while s + 8 <= min(N, V0 + 400):
            blk = torch.cat([xb[s:s + 4], junk[4:]])
            o = attn.attn_block(blk, lw, kv, s)
            kv.length = s + 8
            kv.truncate(s + 4)
            outs.append(o[:4])
            s += 4
        got = torch.cat(outs)
        rr = rel_rows(got, want[V0:s])
        row["verify"] = {"rows": s - V0, "rel_max": rr.max().item()}
        # the block's rows against the same rows decoded alone (row independence)
        kv2 = fresh()
        for c0 in range(0, V0, 2048):
            attn.attn_prefill(xb[c0:min(V0, c0 + 2048)], lw, kv2, c0)
        kv2.length = V0
        one = []
        for i in range(V0, s):
            one.append(attn.attn_block(xb[i:i + 1], lw, kv2, i))
            kv2.length = i + 1
        one = torch.cat(one)
        row["verify"]["max_abs_vs_single_rows"] = (one.float() - got.float()).abs().max().item()
        res["layers"][L] = row
        print(f"[layer {L} {'sliding' if sliding else 'full'}] {json.dumps(row)}", flush=True)
        del kv, kv2, lw32
        torch.cuda.empty_cache()
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"[gpucheck] peak {res['peak_gb']:.2f} GB", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()


def bench(starts=(0, 8192, 32768), chunk=2048, reps=5, max_len=40000) -> list[dict]:
    """Milliseconds of one layer's attention (projections included) for a prefill chunk at a few
    starts and for one decode step there, sliding and full, with random BF16 weights of the real
    shapes. `python -c "from tools.kolibri_attn_gpucheck import bench; bench()"`."""
    dev = torch.device("cuda")
    c = {"hidden_size": 2560, "num_attention_heads": 48, "num_key_value_heads": 4, "head_dim": 128,
         "sliding_window": 513, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
         "layer_types": ["sliding_attention", "full_attention"]}
    attn = ka.KolibriAttention(c)
    kv = attn.make_kv(max_len, dev)
    bf = torch.bfloat16
    out = []
    for L, sliding in ((0, True), (1, False)):
        lw = LW(index=L, sliding=sliding,
                qkv=torch.randn(7168, 2560, device=dev, dtype=bf) / 50,
                o=torch.randn(2560, 6144, device=dev, dtype=bf) / 80,
                q_norm=torch.ones(128, device=dev, dtype=bf), k_norm=torch.ones(128, device=dev, dtype=bf))
        x = torch.randn(chunk, 2560, device=dev, dtype=bf)
        for s in starts:
            def pre():
                attn.attn_prefill(x, lw, kv, s)
            pre()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                pre()
            torch.cuda.synchronize()
            ms_p = (time.perf_counter() - t0) / reps * 1e3
            pos = torch.tensor([s + chunk], dtype=torch.int32, device=dev)

            def dec():
                attn.attn_decode(x[:1], lw, kv, pos)
            dec()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps * 4):
                dec()
            torch.cuda.synchronize()
            ms_d = (time.perf_counter() - t0) / (reps * 4) * 1e3
            row = {"sliding": sliding, "start": s, "prefill_chunk_ms": round(ms_p, 2),
                   "decode_ms_eager": round(ms_d, 3)}
            print("[bench]", row, flush=True)
            out.append(row)
    return out
