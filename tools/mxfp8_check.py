"""Check tools/mxfp8_skinny.py on a real FP4-Local MXFP8 weight: accuracy vs the reference (dequant -> bf16 -> F.linear),
row invariance (M rows == M one-row calls, bit for bit, M = 1..32), and read bandwidth."""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/work/TandemLLM")
from engine.models.flashnext import nvidia as N
from tools import mxfp8_skinny as S

idx = N.HFIndex(sys.argv[1] if len(sys.argv) > 1 else "/models/nvidia/Qwen3.8-Flash-Next-FP4-Local")
torch.manual_seed(0)
for name in ("layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.out_proj", "layers.3.self_attn.o_proj",
             "layers.0.mlp.shared_expert.down_proj"):
    p = N.projection(idx, N.PREFIX + name, "cuda")
    w, s = p.parts["w"], p.parts["s"]
    Nn, K = w.shape
    wref = p.dequant(torch.bfloat16)
    x = (torch.randn(32, K, device="cuda") * 0.5).to(torch.bfloat16)
    y = S.mm(x, w, s)
    yr = F.linear(x, wref)
    rel = ((y.float() - yr.float()).norm() / yr.float().norm()).item()
    inv = all(torch.equal(S.mm(x[:m], w, s), torch.cat([S.mm(x[i:i + 1], w, s) for i in range(m)])) for m in (2, 5, 16, 17, 32))
    # bandwidth at 1 and 16 rows
    res = []
    for m in (1, 16):
        xm = x[:m].contiguous()
        for _ in range(5): S.mm(xm, w, s)
        torch.cuda.synchronize(); t = time.time(); n = 200
        for _ in range(n): S.mm(xm, w, s)
        torch.cuda.synchronize(); dt = (time.time() - t) / n
        res.append(f"{m} rows {dt*1e6:7.1f} us {(w.numel() + s.numel()) / dt / 1e9:6.1f} GB/s")
    print(f"{name:40s} [{Nn}x{K}] wk={S.default_wk(K)} rel.err vs ref {rel:.2e} row-invariant {inv} | " + " | ".join(res), flush=True)
