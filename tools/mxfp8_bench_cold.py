"""Cold-cache bandwidth of tools/mxfp8_skinny.py: run every GDN layer's in_proj_qkv once in sequence (~0.94 GB, far
past L2), as a decode step streams them. Also the per-shape K-split choices."""
import sys, time, torch
sys.path.insert(0, "/work/TandemLLM")
from engine.models.flashnext import nvidia as N
from tools import mxfp8_skinny as S

idx = N.HFIndex("/models/nvidia/Qwen3.8-Flash-Next-FP4-Local")
cfg_layers = [l for l in range(48) if N.PREFIX + f"layers.{l}.linear_attn.in_proj_qkv.weight" in idx]
for name, shape_note in (("linear_attn.in_proj_qkv", "GDN qkv"), ("linear_attn.out_proj", "GDN out")):
    ws = []
    for l in cfg_layers:
        p = N.projection(idx, N.PREFIX + f"layers.{l}.{name}", "cuda")
        ws.append((p.parts["w"], p.parts["s"]))
    K = ws[0][0].shape[1]
    nbytes = sum(w.numel() + s.numel() for w, s in ws)
    for wk, nt in [(w_, n_) for n_ in (1, 2, 4) for w_ in (2, 4, 8, 16) if not (n_ == 4 and w_ == 16)]:
        line = []
        for m in (1, 8, 16, 32):
            x = (torch.randn(m, K, device="cuda") * 0.5).to(torch.bfloat16)
            for w, s in ws: S.mm(x, w, s, wk=wk, nt=nt)
            torch.cuda.synchronize(); t = time.time(); reps = 5
            for _ in range(reps):
                for w, s in ws: S.mm(x, w, s, wk=wk, nt=nt)
            torch.cuda.synchronize(); dt = (time.time() - t) / reps
            line.append(f"{m:2d} rows {nbytes / dt / 1e9:5.0f} GB/s")
        print(f"{shape_note:8s} x{len(ws)} [{ws[0][0].shape[0]}x{K}] {nbytes/1e9:.2f} GB  wk={wk:2d} nt={nt}: " + " | ".join(line), flush=True)
