"""Greedy decode NVIDIA's Flash-Next FP4-Local checkpoint through the plain-torch reference (`ref.FlashNextRef`), weights
as stored (`engine/models/flashnext/nvidia.py`). Slow by design: the yardstick the engine's kernels are checked against.

    python tools/nv_ref_greedy.py /models/nvidia/Qwen3.8-Flash-Next-FP4-Local --tokens 64 --out /tmp/nv_ref.json

Writes {prompt, prompt_ids, ids, text, top2 margins} so another runtime's greedy output can be compared token by token.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from engine.models.flashnext import nvidia
from engine.models.flashnext.ref import FlashNextRef, FNState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number, with a short docstring.")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--think", action="store_true", help="thinking on (default off)")
    ap.add_argument("--out", default="/tmp/nv_ref.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.path)
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=a.think)
    ids = tok(text, add_special_tokens=False)["input_ids"]

    t0 = time.time()
    cfg, W, tables = nvidia.load_reference(a.path, device="cuda")
    print(f"[nv-ref] loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU", flush=True)
    ref = FlashNextRef(cfg, W, tables, dtype=torch.bfloat16, device="cuda")
    st = FNState()
    out, margins = [], []
    with torch.no_grad():
        t1 = time.time()
        logits = ref.forward(torch.tensor(ids, device="cuda"), st, head="last")
        print(f"[nv-ref] prefill {len(ids)} tokens in {time.time() - t1:.1f} s", flush=True)
        for i in range(a.tokens):
            top = torch.topk(logits[-1], 2)
            nxt = int(top.indices[0])
            out.append(nxt)
            margins.append(round(float(top.values[0] - top.values[1]), 4))
            if nxt in (cfg.eos_token_id if isinstance(cfg.eos_token_id, (list, tuple)) else [cfg.eos_token_id]):
                break
            logits = ref.forward(torch.tensor([nxt], device="cuda"), st, head="last")
        dt = time.time() - t1
    res = {"prompt": a.prompt, "think": a.think, "prompt_ids": ids, "ids": out, "text": tok.decode(out),
           "top2_margin": margins, "decode_tok_s": round(len(out) / dt, 2)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(res["text"])
    print(f"[nv-ref] {len(out)} tokens, {res['decode_tok_s']} tok/s, smallest top-2 margin {min(margins):.4f} -> {a.out}")


if __name__ == "__main__":
    main()
