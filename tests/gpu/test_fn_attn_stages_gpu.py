"""On the board (SPD-249 P6): `attn_rows` gives the same bits at every split-launch stage count.

QWEN38_FN_PF_ATTN_STAGES sets the pipeline stages of the attention split launch over a prompt chunk's rows. It is a
launch setting, so the output must not move: stages 1, 3 and 4 equal stages 2 (the default), bit for bit, on

  * dense rows with 1 .. 2,051 visible keys (every chunk boundary of 256 among them), 1 .. 4,096 rows a call;
  * QSA-selected rows (512 blocks of 4 and the open block) at 32,768 and 5,000 tokens of context, 4,096, 1,000, 33
    and 1 rows;
  * a prompt's first chunk: dense rows up to 2,051 tokens, selected rows after;
  * tree rows (ancestors past the block start, the decode shape);

and the split launch's time at 4,096 selected rows per stage count (printed, not asserted).

Run on the box: python tests/gpu/test_fn_attn_stages_gpu.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import fn_seq_kernels as SK  # noqa: E402

DEV = "cuda"
HQ, HK, AD = 24, 2, 256
STAGES = (1, 3, 4)


def _qkv(M, cap, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kc = torch.randn(HK, cap, AD, generator=g, device=DEV).to(torch.bfloat16)
    vc = torch.randn(HK, cap, AD, generator=g, device=DEV).to(torch.bfloat16)
    q = torch.randn(M, HQ, AD, generator=g, device=DEV).to(torch.bfloat16)
    return q, kc, vc, g


def _dense(M, seed, keys=None):
    """Rows that see their first n cache slots, no ancestors."""
    q, kc, vc, _ = _qkv(M, 4096, seed)
    t = torch.arange(M, device=DEV)
    n = (1 + (t * 37 + seed) % 2051) if keys is None else torch.tensor(keys, device=DEV)[t % len(keys)]
    ctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    ctl[:, 0], ctl[:, 1], ctl[:, 2] = n.int(), 4096, 1
    return (q, kc, vc, ctl, torch.zeros(M, SK.ANC, dtype=torch.int32, device=DEV),
            torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV))


def _selected(M, ctx, seed, dense_to=0):
    """The last M rows of a `ctx`-token prompt: a row's 512 blocks (ascending) and its open block; rows at up to
    `dense_to` tokens dense instead."""
    q, kc, vc, g = _qkv(M, ctx, seed)
    p = torch.arange(ctx - M, ctx, device=DEV)
    nblk = (p + 1) // 4
    sc = torch.rand(M, int(nblk.max()), generator=g, device=DEV)
    sc.masked_fill_(torch.arange(sc.shape[1], device=DEV)[None, :] >= nblk[:, None], -1.0)
    blk = sc.topk(SK.TOPK, dim=1).indices.sort(1).values
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    sel[:, :4 * SK.TOPK] = (blk[:, :, None] * 4 + torch.arange(4, device=DEV)).reshape(M, 4 * SK.TOPK).int()
    for j in range(3):
        pos = nblk * 4 + j
        sel[:, 4 * SK.TOPK + j] = torch.where(pos <= p, pos, torch.zeros_like(pos)).int()
    ctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    ctl[:, 0] = (4 * SK.TOPK + (p + 1 - nblk * 4)).int()
    ctl[:, 1] = ctx
    d = p + 1 <= dense_to
    ctl[d, 0], ctl[d, 2] = (p[d] + 1).int(), 1
    return q, kc, vc, ctl, torch.zeros(M, SK.ANC, dtype=torch.int32, device=DEV), sel.contiguous()


def _tree(L, seed):
    """Rows of a draft tree past a block start L: a row sees the L cached rows and its own path."""
    parents = [-1, 0, 0, 1, 1, 2, 3, 3, 4, 6, 6, 9, 10, 11, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22]
    M = len(parents)
    q, kc, vc, _ = _qkv(M, 4096, seed)
    ctl = torch.zeros(M, 4, dtype=torch.int32)
    anc = torch.zeros(M, SK.ANC, dtype=torch.int32)
    for i in range(M):
        pa, j = [], i
        while j >= 0:
            pa.append(j)
            j = parents[j]
        pa = pa[::-1]
        ctl[i, :3] = torch.tensor([L + len(pa), L, 1])
        anc[i, :len(pa)] = torch.tensor([L + a for a in pa])
    return q, kc, vc, ctl.to(DEV), anc.to(DEV), torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)


def _check(x, what):
    want = SK.attn_rows(*x, stages=2)
    assert bool(torch.isfinite(want.float()).all()), what
    assert torch.equal(SK.attn_rows(*x), want), f"{what}: the default is not stages 2"
    for s in STAGES:
        got = SK.attn_rows(*x, stages=s)
        bad = (got.view(torch.int16) != want.view(torch.int16)).flatten(1).any(1).nonzero().flatten()
        assert bad.numel() == 0, (f"{what}, stages {s}: {bad.numel()} of {want.shape[0]} rows differ, "
                                  f"first {bad[:4].tolist()}")


def test_dense_rows():
    n = 0
    for M in (1, 2, 3, 7, 15, 16, 17, 33, 64, 100, 257, 1000, 4096):
        _check(_dense(M, 100 + M), f"dense {M} rows")
        n += 1
    edges = [1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1279, 1280, 1281, 1535,
             1536, 1537, 1791, 1792, 1793, 2047, 2048, 2049, 2050, 2051]
    _check(_dense(len(edges) * 3, 7, keys=edges), "dense rows at the chunk edges")
    return f"{n + 1} calls, stages {STAGES} = 2"


def test_selected_rows():
    n = 0
    for ctx in (32768, 5000):
        for M in (4096, 1000, 33, 1):
            _check(_selected(M, ctx, ctx + M), f"selected {M} rows at {ctx}")
            n += 1
    return f"{n} calls, stages {STAGES} = 2"


def test_first_chunk_dense_then_selected():
    x = _selected(4096, 4096, 11, dense_to=2051)
    assert int(x[3][:, 2].sum()) == 2051
    _check(x, "a prompt's first chunk")
    x = _selected(3000, 3000, 12, dense_to=2051)
    _check(x, "a 3,000-token prompt")
    return f"2,051 dense and 2,045 selected rows in one call; 3,000 rows; stages {STAGES} = 2"


def test_tree_rows():
    for L in (5, 300, 1500, 2030):
        _check(_tree(L, L), f"tree rows at {L}")
    return f"24 tree rows at 4 block starts, stages {STAGES} = 2"


def test_times_at_4096_selected_rows():
    x = _selected(4096, 32768, 5)
    bufs = {}
    out = []
    for s in (2, 1, 3, 4):
        SK.attn_rows(*x, bufs=bufs, stages=s)
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(10):
            t0 = time.perf_counter()
            SK.attn_rows(*x, bufs=bufs, stages=s)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        out.append(f"stages {s} {best * 1e3:.3f} ms")
    return "attn_rows (split + combine), best of 10: " + "; ".join(out)


if __name__ == "__main__":
    fails = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            got = fn()
            print(f"  {name:44s} ok{'  ' + got if isinstance(got, str) else ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:44s} FAIL {type(e).__name__}: {str(e)[:600]}", flush=True)
    print(f"{len(tests) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
