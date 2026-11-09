"""On the board: the non-inline matmul path's segment results land in the caller's `out` row
slices (`_reference_rows`, `_reference_xh`, `_fallback_matmul`, `exl3_matmul`, `FK.dense`,
`_slab_matmul_ada`), and `rot_in_g` rotates every segment's suh in one launch -- each the
bit-for-bit bytes of the assemble-then-copy code it replaced. The oracle below is that old
assembly verbatim (fresh tensors per segment + torch.cat + the .to cast), so a PASS means the
plumbing changed nothing but where the same kernel stores land. Run on the box under the lock:
python tests/gpu/test_exl3_outpath_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X, exl3_ada as A  # noqa: E402
from tools import fn_kernels as FK
from tools import gpu as _gpu  # noqa: E402


def _old_segments(x: torch.Tensor, w: X.EXL3Block) -> torch.Tensor:
    """The pre-out= _reference_rows: rot_in a segment at a time, fresh `y` each, cat, cast."""
    ada = x.is_cuda and A.rot_ok(x.device)
    gok = x.is_cuda and A.gemv_ok(x.device, w.kb)
    H = X.hadamard128(x.device)
    M = x.shape[0]
    outs = []
    for g, n0, n1 in X.segments(w):
        if ada:
            xhs = A.rot_in(x, w.suh[g])
        else:
            xhs = torch.empty(M, w.K, dtype=torch.float16, device=x.device)
            for m in range(M):
                xs = x[m:m + 1].float() * w.suh[g].float()
                xhs[m] = (xs.view(1, -1, 128) @ H).view(w.K).to(torch.float16)
        if ada and gok:
            y = A.gemv_rot(xhs, w.t[:, n0 // 16:n1 // 16], w.svh[n0:n1], w.out_fp32)
        else:
            yp = (A.gemv(xhs, w.t[:, n0 // 16:n1 // 16]) if gok else
                  torch.cat([xhs[m:m + 1].float() @ w.inner(n0, n1).float()
                             for m in range(M)]))
            y = (A.rot_out(yp, w.svh[n0:n1], w.out_fp32) if ada else
                 torch.cat([(yp[m:m + 1].view(1, -1, 128) @ H).view(1, n1 - n0)
                            for m in range(M)]) * w.svh[n0:n1].float())
        outs.append(y)
    y = torch.cat(outs, 1)
    return y if w.out_fp32 else y.to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.dtype in (torch.bfloat16, torch.float16)
                               else torch.int32)


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


_SIZES = {2: (1024, 1024), 3: (512, 768, 768)}


def _block(N: int, kb: int, seed: int, G: int = 1) -> X.EXL3Block:
    if G == 1:
        return X.random_block(N, 2048, kb, device="cuda", seed=seed)
    parts = [X.random_block(s, 2048, kb, device="cuda", seed=seed * 31 + g)
             for g, s in enumerate(_SIZES[G])]
    return X.EXL3Block.group(parts, [f"m{g}" for g in range(G)])


def test_rot_in_g_is_the_per_segment_calls():
    for G in (1, 2, 3):
        x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
        suh = torch.randn(G, 2048, dtype=torch.float16, device="cuda")
        got = A.rot_in_g(x, suh)
        for g in range(G):
            assert _eq(got[g], A.rot_in(x, suh[g])), f"G={G} seg {g}"
    return "G 1..3: every segment's fp16 the lone rot_in's"


def test_reference_rows_writes_the_callers_rows():
    cases = 0
    for kb in (2, 25):
        for G in (1, 2, 3):
            w = _block(2048, kb, 100 + kb * 7 + G, G)
            for M in (1, 4, 33):
                x = torch.randn(M, 2048, dtype=torch.bfloat16, device="cuda")
                ref = _old_segments(x.clone(), w)
                assert _eq(X._reference_rows(x, w), ref), f"kb{kb} G{G} M{M} fresh"
                odt = torch.float32 if w.out_fp32 else torch.bfloat16
                out = torch.empty(M, w.N, dtype=odt, device="cuda")
                assert X._reference_rows(x, w, out=out) is out and _eq(out, ref), \
                    f"kb{kb} G{G} M{M} out="
                wide = torch.full((M, w.N + 128), float("nan"), dtype=odt, device="cuda")
                X._reference_rows(x, w, out=wide[:, 64:64 + w.N])
                assert _eq(wide[:, 64:64 + w.N], ref), f"kb{kb} G{G} M{M} strided"
                if not w.out_fp32:
                    o32 = torch.empty(M, w.N, dtype=torch.float32, device="cuda")
                    assert _eq(X._reference_rows(x, w, out=o32), ref.float()), \
                        f"kb{kb} G{G} M{M} fp32-out cast"
                cases += 1
    return f"{cases} block cells x M 1/4/33 x fresh/out/strided/fp32-out: identical"


def test_exl3_matmul_and_dense_pass_out_through():
    torch.manual_seed(0)
    cases = 0
    for kb in (3, 25):
        w = _block(2048, kb, 300 + kb, 2)
        for M in (1, 16):
            x = torch.randn(M, 2048, dtype=torch.bfloat16, device="cuda")
            ref = _old_segments(x.clone(), w)
            assert _eq(X.exl3_matmul(x, w), ref), f"matmul kb{kb} M{M}"
            if M <= 32 and kb in (2, 3, 4, 5, 6, 8, 15, 25, 35):
                out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16,
                                  device="cuda")
                # the inline kernels' bits are their own: out= must equal the same call's
                # fresh output; the Triton oracle only holds where dense falls back to it
                expect = ref if not _gpu.profile(x.device).inline("fn_kernels") else FK.dense(x, w)
                assert _eq(FK.dense(x, w, out=out), expect), f"dense kb{kb} M{M}"
            cases += 1
    return f"{cases} matmul/dense cells: the out= bytes are the cat's"


def test_reference_xh_and_down_silu_keep_their_bytes():
    torch.manual_seed(1)
    w = _block(2048, 25, 555, 2)
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    xh = A.rot_in_g(x, w.suh)
    ref = _old_segments(x.clone(), w)
    got = X._reference_xh(xh, w)
    assert _eq(got, ref), "_reference_xh fresh"
    out = torch.empty(4, w.N, dtype=torch.bfloat16, device="cuda")
    assert _eq(X._reference_xh(xh, w, out=out), ref), "_reference_xh out="
    return "pre-rotated input path: identical"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:42s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
