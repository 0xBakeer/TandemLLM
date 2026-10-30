"""Greedy decode NVIDIA's Flash-Next FP4-Local checkpoint through the engine (`engine.FlashNextEngine`, weights from
`nvidia.load_engine`) and compare it token by token with a reference run (tools/nv_ref_greedy.py's JSON).

    python tools/nv_engine_greedy.py /models/nvidia/Qwen3.8-Flash-Next-FP4-Local --ref .nv_ref.json --mode step

Modes: `forward` the prompt and every token through `forward` (the trunk); `step` the prompt through `forward`, then the
decode row step (`stage` + `step`, what the decode graph holds); `pre` the prompt through the prefill kernels
(`forward_pre`), then the row step.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from engine.models.flashnext import nvidia
from engine.models.flashnext.engine import FlashNextEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", required=True, help="tools/nv_ref_greedy.py output (prompt ids and the reference's tokens)")
    ap.add_argument("--mode", choices=("forward", "step", "pre"), default="step")
    ap.add_argument("--tokens", type=int, default=None, help="default: as many as the reference")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    ids, want = ref["prompt_ids"], ref["ids"]
    n = a.tokens or len(want)

    t0 = time.time()
    cfg, W, banks, tables, timing = nvidia.load_engine(a.path, device="cuda")
    eng = FlashNextEngine(cfg, W, tables, banks, device="cuda", max_len=4096)
    print(f"[nv-eng] loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU",
          flush=True)
    st = eng.new_state(4096)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, (list, tuple)) else [cfg.eos_token_id]
    out, margins, times = [], [], []
    with torch.no_grad():
        t1 = time.time()
        prompt = torch.tensor(ids, device="cuda")
        logits = eng.forward_pre(prompt, st, head="last") if a.mode == "pre" else eng.forward(prompt, st, head="last")
        torch.cuda.synchronize()
        print(f"[nv-eng] prefill {len(ids)} tokens in {time.time() - t1:.2f} s ({a.mode})", flush=True)
        for i in range(n):
            top = torch.topk(logits[-1].float(), 2)
            nxt = int(top.indices[0])
            out.append(nxt)
            margins.append(round(float(top.values[0] - top.values[1]), 4))
            if nxt in eos or i + 1 == n:
                break
            t = time.time()
            if a.mode == "forward":
                logits = eng.forward(torch.tensor([nxt], device="cuda"), st, head="last")
            else:
                eng.stage(nxt, st)
                logits = eng.step(st)
                st.length += 1
            torch.cuda.synchronize()
            times.append(time.time() - t)
    same = 0
    while same < min(len(out), len(want)) and out[same] == want[same]:
        same += 1
    res = {"mode": a.mode, "ids": out, "matches_ref": same, "of": min(len(out), len(want)),
           "first_diff": None if same == min(len(out), len(want)) else
           {"at": same, "ours": out[same], "ref": want[same], "our_margin": margins[same],
            "ref_margin": ref["top2_margin"][same]},
           "decode_ms_median": round(1e3 * sorted(times)[len(times) // 2], 2) if times else None}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
