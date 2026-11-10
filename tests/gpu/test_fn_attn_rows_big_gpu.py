"""On the board (SPD-249): one `attn_rows` call past 29,127 rows.

`attn_rows` keeps M * HK * 9 * 16 * D partial-accumulator elements (HK 2, D 256: 73,728 a row). That offset passes
2^31 at 29,128 rows, where an int32 offset wraps. Each case runs in its own process (a CUDA fault ends the context):
the rows of one call equal the same rows from calls of at most 8,192 rows, bit for bit, at

  * 29,127 rows (the last count an int32 offset holds), 29,128 and 32,768, `attn_rows`;
  * 32,768 rows over two lanes, `attn_rows_multi`.

Rows are dense with 1 .. 200 visible keys; every 64th row and the last 32 see 2,051 (all nine splits).
About 16 GB of GPU memory. Run on the box: python tests/gpu/test_fn_attn_rows_big_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DEV = "cuda"
HQ, HK, AD, CAP = 24, 2, 256, 4096
PIECE = 8192
CASES = [("rows", 29127), ("rows", 29128), ("rows", 32768), ("multi", 32768)]


def _inputs(M, seed):
    import torch
    from tools import fn_seq_kernels as SK
    g = torch.Generator(device=DEV).manual_seed(seed)
    kc = torch.randn(HK, CAP, AD, generator=g, device=DEV).to(torch.bfloat16)
    vc = torch.randn(HK, CAP, AD, generator=g, device=DEV).to(torch.bfloat16)
    q = torch.cat([torch.randn(min(PIECE, M - a), HQ, AD, generator=g, device=DEV).to(torch.bfloat16)
                   for a in range(0, M, PIECE)])
    t = torch.arange(M, device=DEV)
    n = 1 + t % 200
    n[(t % 64 == 63) | (t >= M - 32)] = 2051
    ctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    ctl[:, 0], ctl[:, 1], ctl[:, 2] = n.int(), CAP, 1                    # dense, no row past the block start
    anc = torch.zeros(M, SK.ANC, dtype=torch.int32, device=DEV)
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    return q, kc, vc, ctl, anc, sel


def _pieces(q, kc, vc, ctl, anc, sel):
    import torch
    from tools import fn_seq_kernels as SK
    return torch.cat([SK.attn_rows(q[a:a + PIECE], kc, vc, ctl[a:a + PIECE], anc[a:a + PIECE], sel[a:a + PIECE])
                      for a in range(0, q.shape[0], PIECE)])


def case(kind: str, M: int) -> str:
    import torch
    from tools import fn_seq_kernels as SK
    if kind == "rows":
        x = _inputs(M, M)
        want = _pieces(*x)
        got = SK.attn_rows(*x)
    else:
        h = M // 2 + 5
        a, b = _inputs(h, 1), _inputs(M - h, 2)
        want = torch.cat([_pieces(*a), _pieces(*b)])
        pt = torch.tensor([SK.attn_rows_entry(0, *a[1:]), SK.attn_rows_entry(h, *b[1:])], dtype=torch.int64, device=DEV)
        lane = (torch.arange(M, device=DEV) >= h).int()
        kc = a[1]
        got = SK.attn_rows_multi(torch.cat([a[0], b[0]]), pt, lane, (kc.stride(0), kc.stride(1), kc.stride(0),
                                                                    kc.stride(1)), HK)
    torch.cuda.synchronize()
    bad = (got != want).flatten(1).any(1).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} rows differ, first {bad[:4].tolist()}, last {bad[-4:].tolist()}"
    assert bool(torch.isfinite(got.float()).all())
    return f"{M} rows equal the same rows in calls of {PIECE}"


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--case":
        print(case(sys.argv[2], int(sys.argv[3])), flush=True)
        sys.exit(0)
    fails = 0
    for kind, M in CASES:
        r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--case", kind, str(M)],
                           capture_output=True, text=True)
        name = f"attn_{kind}_{M}"
        if r.returncode == 0:
            print(f"  {name:28s} ok  {r.stdout.strip().splitlines()[-1]}", flush=True)
        else:
            fails += 1
            tail = [ln for ln in (r.stdout + r.stderr).strip().splitlines() if ln.strip()][-1:]
            print(f"  {name:28s} FAIL exit {r.returncode}: {' '.join(tail)[:400]}", flush=True)
    print(f"{len(CASES) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
