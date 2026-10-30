"""ENG-401: per-input-channel activation statistics of every Flint projection, for activation-weighted NVFP4 scales
(tools/flint_nvfp4.py --stats). The BF16 checkpoint runs on this engine over calibration text, prompt chunk by chunk;
each projection's input rows add their squares per channel (q/k/v and gate/up share theirs), the routed experts'
from the rows routed to them (gate/up: the layer's normed rows; down: the SwiGLU activation).

    PYTHONPATH=<repo> python tools/flint_nvfp4_stats.py --model <bf16 dir> --corpus a.txt,b.txt --tokens 65536 --out stats.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="comma-separated text files")
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import engine.models.flint.engine as E
    from engine.models.flint.load import load
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, W, experts, _ = load(a.model, "cuda", log=None)
    eng = E.FlintEngine(idx.cfg, W, experts, device="cuda", max_len=a.chunk + 64)
    names = {id(v): k for k, v in W.items() if torch.is_tensor(v)}
    acc: dict = {}

    def add(key, x):
        x = x.reshape(-1, x.shape[-1]).float()
        s, n = acc.get(key, (0.0, 0))
        acc[key] = (s + x.pow(2).sum(0), n + x.shape[0])

    lin0 = eng.lin

    def lin(x, w):
        k = names.get(id(w))
        if k is not None:
            add(k, x)
        return lin0(x, w)
    eng.lin = lin

    def experts_chunk(n, ids, li):
        e = eng.experts[li]
        T, S = ids.shape
        I = e.I2 // 2
        flat = ids.reshape(-1).long()
        yd = torch.empty(T * S, n.shape[1], dtype=torch.float32, device=n.device)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=e.E).tolist()
        o = 0
        for ex, cnt in enumerate(counts):
            if not cnt:
                continue
            pairs = order[o:o + cnt]
            o += cnt
            x = n.index_select(0, pairs // S)
            add(f"layers.{li}.mlp.experts.{ex}.gate_up", x)
            gu = F.linear(x, e.w13[ex])
            act = F.silu(gu[:, :I]) * gu[:, I:]
            add(f"layers.{li}.mlp.experts.{ex}.down", act)
            yd[pairs] = F.linear(act, e.w2[ex]).float()
        return yd
    eng._moe_bf16_chunk = experts_chunk
    text = "\n\n".join(open(f, encoding="utf-8").read() for f in a.corpus.split(","))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = max(1, len(ids) // a.tokens) if len(ids) > a.tokens * 2 else 1
    ids = ids[:a.tokens] if step == 1 else [t for i, t in enumerate(ids) if (i // a.chunk) % step == 0][:a.tokens]
    for c0 in range(0, len(ids), a.chunk):
        st = eng.new_state(a.chunk + 64)
        with torch.no_grad():
            eng.prefill(torch.tensor(ids[c0:c0 + a.chunk]), st, head="none")
        print(f"[stats] {c0 + a.chunk}/{len(ids)} tokens, {len(acc)} keys", flush=True)
    out = {k: (s / max(n, 1)).cpu() for k, (s, n) in acc.items()}
    out["_tokens"] = torch.tensor(len(ids))
    torch.save(out, a.out)
    print(f"[stats] {len(out) - 1} keys -> {a.out}")


if __name__ == "__main__":
    main()
