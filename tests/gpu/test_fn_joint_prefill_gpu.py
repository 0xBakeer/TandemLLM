"""On the board (SPD-249): the kernel facts Flash-Next's joint prefill (engine/models/flashnext/joint_prefill.py)
rests on, bit for bit.

  * `fn_prefill_kernels.moe_grouped` over the rows of several prompts at once (row counts 1, 33, 128, 205, 300, and
    eight of 205 as a burst of 8; skewed routing, so experts take several items) gives every pair the output its own
    prompt's call gives it: a pair's output does not depend on its group, its item or the other rows;
  * `mm_rows` with the fp32 weight cache on (int8 rows and fp16 weights, 1 to 300 rows) equals the uncached call;
  * the EXL3 slab GEMM (rows past EXL3_CHUNK_UNTIL) with the slab cache on equals the uncached call at the same rows,
    and a prompt's rows through a row slice of a joint input equal its own call (the input copied, as `rotate_in`).

The whole-model check is tools/fn_joint_check.py --joint-prefill (every prompt's logits, streams, KV and GDN state).
Run on the box: python tests/gpu/test_fn_joint_prefill_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

H, I = 2560, 640
DEV = "cuda"


def _x(M, K_, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, K_, generator=g, device=DEV) * scale).to(torch.bfloat16)


def _bank(n_exp: int, kbs, shared_k: int, seed: int = 0):
    blocks = []
    for e in range(n_exp + 1):
        kb = shared_k if e == n_exp else kbs[e % len(kbs)]
        blocks.append(tuple(X.random_block(n, k, kb, seed=seed + 3 * e + j)
                            for j, (n, k) in enumerate(((I, H), (I, H), (H, I)))))
    trip = [tuple((b.t, b.suh[0].contiguous(), b.svh) for b in eb) for eb in blocks]
    return K.MoEBank(trip, DEV)


def _ids(M, n_exp, k, seed):
    """Router-like ids [M, k + 1], the shared expert last; experts 0 and 1 in four rows of five (several items each)."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for m in range(M):
        rest = (torch.randperm(n_exp - 2, generator=g) + 2).tolist()
        rows.append((([0, 1] if m % 5 else []) + rest)[:k] + [n_exp])
    return torch.tensor(rows, dtype=torch.int32, device=DEV)


def test_moe_grouped_rows_are_their_prompts_own():
    out = []
    bank = _bank(24, (2, 3, 4), 5, seed=7)
    for lens in ((205, 33, 1, 128, 300), (205,) * 8):
        xs = [_x(M, H, 40 + j, 0.5) for j, M in enumerate(lens)]
        ids = [_ids(M, 24, 10, 90 + j) for j, M in enumerate(lens)]
        solo = [PK.moe_grouped(x, i, bank) for x, i in zip(xs, ids)]
        joint = PK.moe_grouped(torch.cat(xs), torch.cat(ids), bank)
        r0 = 0
        for M, y in zip(lens, solo):
            assert torch.equal(joint[r0:r0 + y.shape[0]], y), (lens, M)
            r0 += y.shape[0]
        out.append(f"{len(lens)} prompts, {sum(lens)} rows")
    return "every pair its prompt's own bits: " + "; ".join(out)


def test_mm_rows_weight_cache_keeps_the_bits():
    g = torch.Generator(device=DEV).manual_seed(5)
    w8 = Int8Rows.of((torch.randn(324, 4 * H, generator=g, device=DEV) * 0.02).half())
    w16 = (torch.randn(513, H, generator=g, device=DEV) * 0.02).half()
    cases = []
    for w, Kk in ((w8, 4 * H), (w16, H)):
        for M in (1, 33, 205, 300):
            x = _x(M, Kk, M)
            base = PK.mm_rows(x, w)
            PK.wf_cache_on()
            try:
                a, b = PK.mm_rows(x, w), PK.mm_rows(x, w)
            finally:
                PK.wf_cache_off()
            assert torch.equal(a, base) and torch.equal(b, base), (Kk, M)
            cases.append(M)
    return f"{len(cases)} cases equal"


def test_slab_cache_and_row_slices_keep_the_bits():
    w = X.random_block(4096, H, 3, seed=11)
    lens = (205, 129, 300)
    xs = [_x(M, H, 60 + j) for j, M in enumerate(lens)]
    solo = [X.exl3_matmul(x, w) for x in xs]
    joint_in = torch.cat(xs)
    X.slab_cache_on()
    try:
        r0, got = 0, []
        for x in xs:
            got.append(X.exl3_matmul(joint_in[r0:r0 + x.shape[0]], w))
            r0 += x.shape[0]
    finally:
        X.slab_cache_off()
    for M, a, b in zip(lens, solo, got):
        assert torch.equal(a, b), M
    return f"rows {lens} past EXL3_CHUNK_UNTIL {X.CHUNK_UNTIL}: slab cache + row slices equal"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<46} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<46} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
