"""On the board (ENG-287): the shared EXL3 kernels at LFM2.5-24B-A2B's shapes, and its engine on a tiny model.

  * tools/fn_kernels.dense, the projections up to 32 rows, at every shape the model multiplies by
    (in_proj 2048 -> 6144, q|k|v 2048 -> 3072, out_proj and the conv's out 2048 -> 2048, the dense
    MLP's w1|w3 2048 -> 23552 and w2 11776 -> 2048, a 16,384-column slice of the head in fp32): the
    torch reference within the bound below, and every row of an M-row call (M = 2..32) the M = 1
    call's row bit for bit;
  * the MoE bank kernels with LFM's experts (hidden 2048, expert 1536, 4 slots a row, no shared
    expert): the one-row output against the fixed-order torch reference, and row invariance with rows
    sharing experts;
  * the tiny-model engine tests of tests/test_lfm2_moe_engine.py and the row kernels of
    tests/test_lfm2_moe_kernels.py, here on the GPU.

Run on the box: python tests/gpu/test_lfm2_moe_gpu.py   (under flock ~/.qwen38-box.flock)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402

H, I = 2048, 1536
# the reference makes the kernels' roundings (rotated input to fp16, output to bf16) with fp32 sums in
# another order: the output within a few bf16 steps of the largest value
BOUND = 2 ** -6


def _rel(a, b) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def test_dense_at_lfm_shapes():
    out = []
    for N, Kin, kb, fp32 in ((6144, 2048, 3, False), (3072, 2048, 4, False), (2048, 2048, 2, False),
                             (23552, 2048, 4, False), (2048, 11776, 3, False), (16384, 2048, 6, True)):
        w = X.random_block(N, Kin, kb, seed=N + Kin)
        w.out_fp32 = fp32
        x = (torch.randn(32, Kin, device="cuda") * 0.5).to(torch.bfloat16)
        y = FK.dense(x, w)
        rel = _rel(y[:1], X.reference_matmul(x[:1], w))
        assert rel < BOUND, (N, Kin, kb, rel)
        for M in (2, 5, 16, 32):
            assert torch.equal(FK.dense(x[:M], w)[M - 1], FK.dense(x[M - 1:M], w)[0]), (N, Kin, M)
        out.append(f"{N}x{Kin}@K{kb} {rel:.1e}")
    return ", ".join(out)


def test_moe_bank_at_lfm_shapes():
    E = 8
    experts = [tuple(X.random_block(n, k, kb, seed=100 * e + j) for j, (n, k) in enumerate(((I, H), (I, H), (H, I))))
               for e, kb in zip(range(E), (2, 3, 4, 3, 2, 4, 3, 4))]
    bank = FK.MoEBank([tuple((b.t, b.suh[0], b.svh) for b in ex) for ex in experts], "cuda")
    x = (torch.randn(16, H, device="cuda") * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    ids = torch.stack([torch.randperm(E, generator=g)[:4] for _ in range(16)]).to(torch.int32).cuda()
    wts = torch.rand(16, 4, generator=g).cuda()
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16).float()
    yd = FK.moe_experts(x[:1], ids[:1], bank, order=torch.empty(0, dtype=torch.int32, device="cuda"))
    one = FK.moe_combine(yd, ids[:1], wts[:1])
    ref = FK.moe_reference(x[:1], ids[:1], wts[:1], experts)
    rel = _rel(one, ref)
    assert rel < BOUND, rel
    full = FK.moe_experts(x, ids, bank)
    for m in range(16):
        alone = FK.moe_experts(x[m:m + 1], ids[m:m + 1], bank, order=torch.empty(0, dtype=torch.int32, device="cuda"))
        assert torch.equal(full[4 * m:4 * m + 4], alone), m
    return f"one row vs reference {rel:.1e}; 16 rows each its one-row bits"


def test_tiny_engine_and_kernels_on_the_gpu():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_lfm2_moe_engine as E
    import test_lfm2_moe_kernels as Kt
    assert E.DEV == "cuda" and Kt.DEV == "cuda"
    n = 0
    for mod in (Kt, E):
        for name, fn in sorted(vars(mod).items()):
            if name.startswith("test_") and callable(fn):
                r = fn()
                assert not (r and "skipped" in r), name
                n += 1
    return f"{n} tests"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
