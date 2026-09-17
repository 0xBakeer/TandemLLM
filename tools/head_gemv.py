"""The vocabulary projection as an own kernel.

`lm_head` is [248,320 x 5,120] in bf16 -- 2.54 GB, 13 % of an NVFP4 step's read set and the single
largest tensor the engine touches. It is read once per verify pass and, when a drafter has no head
of its own, once per drafted token as well. The step breakdown charges it 14.5 ms, which is
176 GB/s: a third below what a plain streaming read reaches on this board.

At M = 1 this is a matrix-vector product and the library's general path is not built for it. The
kernel here is: one program per tile of `BN` vocabulary rows, each looping over the 5,120 reduction
elements in `BK` chunks with the activation held in registers. 248,320 / 64 = 3,880 programs
against 48 SMs, every one of them reading `BN x 5,120 x 2` contiguous bytes, which is what a
bandwidth-bound kernel wants.

SPLIT-K. Splitting the reduction is the standard answer when a GEMV cannot fill the machine, and
here it cannot be the answer -- 3,880 tiles already fill it many times over, and splitting would add
an atomic pass over a 993 KB output for nothing. What `--split-k` measures instead is the opposite
question: whether *fewer, longer* tiles beat more short ones. Both are swept by `bench()` and the
winner is a measurement, not an argument.

The logits are computed in fp32 and returned in fp32, which is what `argmax` and the drafter's
`topk` consume. Nothing downstream reads them in bf16.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _head_gemv(X, W, Y, M, N, K,
                   s_wn, s_wk,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        mn = rn < N
        rm = tl.arange(0, BM)
        mm = rm < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            x = tl.load(X + rm[:, None] * K + rk[None, :],
                        mask=mm[:, None] & mk[None, :], other=0.0).to(tl.float32)
            w = tl.load(W + rn[:, None] * s_wn + rk[None, :] * s_wk,
                        mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & mn[None, :])


def head_matmul(x: torch.Tensor, w: torch.Tensor, *, bn: int = 32, bk: int = 256,
                bm: int = 1) -> torch.Tensor:
    """`x @ w.T` for a bf16 head [N, K] and a short activation [M, K]. Returns fp32 [M, N]."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    flat = x.reshape(-1, x.shape[-1])
    M, K = flat.shape
    N = w.shape[0]
    if M > bm:
        raise ValueError(f"head_matmul was built for M <= {bm}, got {M}")
    flat = flat.contiguous()
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    _head_gemv[(triton.cdiv(N, bn),)](flat, w, y, M, N, K, w.stride(0), w.stride(1),
                                      BM=bm, BN=bn, BK=bk, num_warps=4)
    return y


# --------------------------------------------------------------------------- the gate

def check(N: int = 248320, K: int = 5120, device: str = "cuda", **kw) -> dict:
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w).float()
    got = head_matmul(x, w, **kw)
    d = (ref - got).abs()
    return {"absmax": float(d.max()), "scale": float(ref.abs().max()),
            "argmax_same": int(ref.argmax()) == int(got.argmax()),
            "top8_same": torch.equal(ref.topk(8).indices, got.topk(8).indices)}


def bench(N: int = 248320, K: int = 5120, device: str = "cuda", iters: int = 50,
          sweep=((64, 128), (64, 256), (128, 128), (128, 256), (256, 256), (32, 256))) -> list[str]:
    import time
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
    nbytes = N * K * 2
    out = []

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    ms = timed(lambda: torch.nn.functional.linear(x, w))
    out.append(f"torch F.linear        {ms:7.2f} ms  {nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s")
    for bn, bk in sweep:
        try:
            ms = timed(lambda: head_matmul(x, w, bn=bn, bk=bk))
        except Exception as exc:                               # pragma: no cover
            out.append(f"triton bn={bn:3d} bk={bk:3d}  failed: {exc}")
            continue
        out.append(f"triton bn={bn:3d} bk={bk:3d}   {ms:7.2f} ms  "
                   f"{nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s")
    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(check())
    for line in bench():
        print(line)
