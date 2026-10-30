"""The EXL3 prefill slab product with its output rotation in the epilogue (SPD-175).

`tools/exl3_linear.py` `_slab_matmul` runs a prefill projection as the library GEMM of the rotated
input against a decoded fp16 slab, kept in fp32, and then `rot_out_kernel`: a 128-point Walsh-Hadamard
over each 128-column block of every row, the 1/sqrt(128) scale and the per-column scale svh, rounded
to bf16. At 8k that is the fp32 product written and read back once per projection (about 1 GB a
layer) and a sixth of the prefill's non-GEMM time.

Here the product and the rotation are one Triton kernel. Its main loop is the plain tl.dot loop over
K in ascending order with an fp32 accumulator, which gives the library GEMM's bits on every slab
shape and row count measured on GB10 (2026-10-23, 5 tilings x 6 shapes x M in 700..8192, all equal);
its epilogue is `wht128`'s butterfly in the same stage order (bit 0 first), the same fp32 operations
in the same order, and the same rounding. So the output is the shipped path's, bit for bit, and the
row's bits do not depend on M (the K order never does).

The 27B's own module: only engine/model.py imports it (behind QWEN38_SLAB_TRITON and the window knobs),
and it reads tools/exl3_linear.py without changing it, so no other family's rows or fingerprint move.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

RSQRT128 = tl.constexpr(0.08838834764831845) if HAVE_TRITON else 0.08838834764831845


if HAVE_TRITON:

    @triton.jit
    def _wht_stage(x, R: tl.constexpr, H: tl.constexpr):
        """One butterfly stage over [R * 2 * H] values viewed as (R, 2, H): low + high, low - high."""
        x = tl.reshape(x, (R, 2, H))
        x = tl.permute(x, (0, 2, 1))
        a, b = tl.split(x)
        x = tl.join(a + b, a - b)
        return tl.permute(x, (0, 2, 1))

    @triton.jit
    def _wht128_rows(x, BM: tl.constexpr, BN: tl.constexpr):
        """The unnormalised natural-order Walsh-Hadamard over each 128-column block of x [BM, BN]:
        stage s pairs column i with i ^ 2^s, low + high and low - high, bit 0 first (`wht128`)."""
        R: tl.constexpr = BM * (BN // 128)
        x = tl.reshape(x, (R * 64, 2))
        a, b = tl.split(x)
        x = tl.join(a + b, a - b)
        x = _wht_stage(x, R * 32, 2)
        x = _wht_stage(x, R * 16, 4)
        x = _wht_stage(x, R * 8, 8)
        x = _wht_stage(x, R * 4, 16)
        x = _wht_stage(x, R * 2, 32)
        x = _wht_stage(x, R, 64)
        return tl.reshape(x, (BM, BN))

    @triton.jit
    def _slab_rot(A, B, SV, C, M, N, K, s_am, s_bk, s_cm,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                  OUT_F32: tl.constexpr):
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (GROUP * npn)
        fm = gid * GROUP
        gs = min(npm - fm, GROUP)
        pm = fm + (pid % gs)
        pn = (pid % (GROUP * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm[:, None] < M
        a = A + rm[:, None] * s_am + rk[None, :]
        b = B + rk[:, None] * s_bk + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(a, mask=mm, other=0.0)
            y = tl.load(b)
            acc = tl.dot(x, y, acc)
            a += BK
            b += BK * s_bk
        v = _wht128_rows(acc, BM, BN)
        sv = tl.load(SV + rn).to(tl.float32)
        r = v * RSQRT128 * sv[None, :]
        if OUT_F32:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], r, mask=mm)
        else:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], r.to(tl.bfloat16), mask=mm)


if HAVE_TRITON:

    @triton.jit
    def _slab_rot_sliced(A, B, SV, C, SL, M, N, s_am, s_bk, s_cm,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                         S: tl.constexpr, WK: tl.constexpr, OUT_F32: tl.constexpr):
        """`_slab_rot` with the skinny GEMM's summation order (tools/exl3_linear.py `gemm_kernel`): K
        in the S x WK slices of `pick` (SL: their [k0, k1) in 16-row K tiles), each slice a chain of
        k16 products from zero in K order, the slices of a rank added in warp order, the ranks in rank
        order; then the same output rotation. So a row gets the skinny kernel's bits, from a decoded slab."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (GROUP * npn)
        fm = gid * GROUP
        gs = min(npm - fm, GROUP)
        pm = fm + (pid % gs)
        pn = (pid % (GROUP * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm[:, None] < M
        tot = tl.zeros((BM, BN), dtype=tl.float32)
        for r in tl.static_range(S):
            part = tl.zeros((BM, BN), dtype=tl.float32)
            for w in tl.static_range(WK):
                k0 = tl.load(SL + 2 * (r * WK + w)) * 16
                k1 = tl.load(SL + 2 * (r * WK + w) + 1) * 16
                a = A + rm[:, None] * s_am + (k0 + rk)[None, :]
                b = B + (k0 + rk)[:, None] * s_bk + rn[None, :]
                acc = tl.zeros((BM, BN), dtype=tl.float32)
                for _ in range(k0, k1, BK):
                    x = tl.load(a, mask=mm, other=0.0)
                    y = tl.load(b)
                    acc = tl.dot(x, y, acc)
                    a += BK
                    b += BK * s_bk
                part += acc
            tot += part
        v = _wht128_rows(tot, BM, BN)
        sv = tl.load(SV + rn).to(tl.float32)
        o = v * RSQRT128 * sv[None, :]
        if OUT_F32:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], o, mask=mm)
        else:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], o.to(tl.bfloat16), mask=mm)


if HAVE_TRITON:

    @triton.jit
    def _slab_rot_s1(A, B, SV, C, M, N, KT, L, s_am, s_bk, s_cm,
                     BM: tl.constexpr, BN: tl.constexpr, Q: tl.constexpr, GROUP: tl.constexpr,
                     OUT_F32: tl.constexpr):
        """`_slab_rot_sliced` for S = 1 (one rank: every 27B shape): one pass over K, the chain of a
        slice of L K tiles (the last one shorter) added to the running sum at its end and begun again
        from zero. Two accumulator tiles instead of three, one software-pipelined loop."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (GROUP * npn)
        fm = gid * GROUP
        gs = min(npm - fm, GROUP)
        pm = fm + (pid % gs)
        pn = (pid % (GROUP * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, 16 * Q)
        mm = rm[:, None] < M
        a = A + rm[:, None] * s_am + rk[None, :]
        b = B + rk[:, None] * s_bk + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        part = tl.zeros((BM, BN), dtype=tl.float32)
        for kt in range(0, KT, Q):
            x = tl.load(a, mask=mm, other=0.0)
            y = tl.load(b)
            acc = tl.dot(x, y, acc)
            a += 16 * Q
            b += 16 * Q * s_bk
            if ((kt + Q) % L == 0) | (kt + Q == KT):
                part += acc
                acc = tl.zeros((BM, BN), dtype=tl.float32)
        v = _wht128_rows(part, BM, BN)
        sv = tl.load(SV + rn).to(tl.float32)
        o = v * RSQRT128 * sv[None, :]
        if OUT_F32:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], o, mask=mm)
        else:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], o.to(tl.bfloat16), mask=mm)


if HAVE_TRITON:

    @triton.jit
    def _add_norm_rot(R, X, W, H, Y, SUH, XH, M, N, EPS: tl.constexpr, BLOCK: tl.constexpr,
                      G: tl.constexpr, STORE_Y: tl.constexpr):
        """`_add_rms_norm` (tools/norm_kernels.py) and then `rot_in_kernel` of its bf16 output for each
        of the G members' sign vectors: the same operations in the same order, one launch. STORE_Y:
        the norm's output written too (SPD-245: a caller whose next GEMM reads the rotated rows alone
        skips the store; the rows' bits do not depend on it)."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        r = tl.load(R + row * N + cols, mask=m, other=0.0).to(tl.float32)
        a = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
        h = (r + a).to(H.dtype.element_ty)
        tl.store(H + row * N + cols, h, mask=m)
        x = h.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x) / N + EPS)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = (x * rstd * (1.0 + w)).to(Y.dtype.element_ty)
        if STORE_Y:
            tl.store(Y + row * N + cols, y, mask=m)
        yf = y.to(tl.float32)
        for g in tl.static_range(G):
            su = tl.load(SUH + g * N + cols, mask=m, other=0.0).to(tl.float32)
            v = tl.reshape(yf * su, (BLOCK // 128, 128))
            v = _wht128_rows(v, BLOCK // 128, 128)
            o = tl.reshape((v * RSQRT128).to(tl.float16), (BLOCK,))
            tl.store(XH + (g * M + row) * N + cols, o, mask=m)


def add_norm_rot(res: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, suh: torch.Tensor,
                 out: tuple) -> torch.Tensor:
    """`tools.norm_kernels.add_rms_norm(res, x, weight, eps, out=out)` and the rotated input of a
    projection whose input signs are suh [G, N] (`tools.exl3_linear.rotate_in` of the norm's output):
    returns xh [G, M, N] fp16. `out` is (h, y); y None skips the norm output's store (SPD-245: the
    MLP slab reads xh alone, so the 84 MB a layer at 8,192 rows need not be written)."""
    n = x.shape[-1]
    r = res.reshape(-1, n)
    a = x.reshape(-1, n).contiguous()
    h = out[0].view(-1, n)
    y = h if out[1] is None else out[1].view(-1, n)
    M = r.shape[0]
    G = suh.shape[0]
    xh = torch.empty(G, M, n, dtype=torch.float16, device=x.device)
    blk = 1
    while blk < n:
        blk *= 2
    _add_norm_rot[(M,)](r, a, weight, h, y, suh, xh, M, n, EPS=eps, BLOCK=blk, G=G,
                        STORE_Y=out[1] is not None, num_warps=8 if n >= 4096 else 4)
    return xh


def mlp_slab_pairs(rows: int, gu, wd) -> bool:
    """Whether `mlp_slab` takes these weights at this many rows (its own check, for a caller that
    decides before the norm whether the normed rows are read at all)."""
    from tools import exl3_linear as X
    F = wd.K
    return bool(gu.sizes and len(gu.sizes) == 2 and gu.sizes[0] == gu.sizes[1] == F
                and wd.suh.shape[0] == 1 and rows > X.CHUNK_UNTIL and X._mm_out_fp32() is not False)


#: (BM, BN, BK, warps, stages). The tiling never changes a row's bits (every tiling gives the library
#: GEMM's), so it is picked for speed alone: 128 x 256 x 64 is the fastest or within 4 % on every 27B slab
#: shape at 700-8,192 rows (2026-10-23 sweep, raw slab-rot.json), except the narrow 1,024-column members
#: and the shortest calls, where 128 x 128 x 32 wins
WIDE = (128, 256, 64, 8, 3)
NARROW = (128, 128, 32, 4, 4)


def pick(M: int, N: int) -> tuple:
    return WIDE if (N % 256 == 0 and N >= 2048 and M > 768) else NARROW


def slab_rot(xh: torch.Tensor, wi: torch.Tensor, svh: torch.Tensor, out: torch.Tensor,
             tile: tuple | None = None) -> torch.Tensor:
    """out[M, n] = rot_out(xh[M, K] @ wi[K, n]) with the column scales svh[n]; xh, wi fp16
    (contiguous rows), out bf16 or fp32 (any row stride, unit column stride). n % 128 == 0."""
    M, K = xh.shape
    N = wi.shape[1]
    assert N % 128 == 0 and xh.stride(1) == 1 and wi.stride(1) == 1 and out.stride(1) == 1
    bm, bn, bk, w, st = tile or pick(M, N)
    assert K % bk == 0 and N % bn == 0
    grid = (triton.cdiv(M, bm) * (N // bn),)
    _slab_rot[grid](xh, wi, svh, out, M, N, K, xh.stride(0), wi.stride(0), out.stride(0),
                    BM=bm, BN=bn, BK=bk, GROUP=8, OUT_F32=out.dtype == torch.float32,
                    num_warps=w, num_stages=st)
    return out


_SLICES: dict = {}
#: (BM, BN, warps, stages) of the one-rank sliced product, by K; the tiling never moves a bit (each slice's
#: chain and the slices' sum are per element), so it is picked for speed alone: 128 x 128, 8 warps is the
#: fastest or within 5 % on the 27B's shapes at 512 and 4,096 rows (2026-10-26 sweep of 7 tilings), with a
#: fourth stage for the long-K down projection
def sliced_tile(K: int) -> tuple:
    return (128, 128, 8, 4 if K > 8192 else 3)


def skinny_slices(N: int, Kin: int, kb: int, dev) -> tuple:
    """(S, WK, bounds [S * WK, 2] int32 on `dev`, K tiles a step): the K slices `gemm_kernel` sums a
    column block of this shape in (`pick`'s S and wk: ranks, then warps, both contiguous), and the
    largest step of 1, 2 or 4 K tiles that divides every slice."""
    from tools import exl3_linear as X
    key = (N, Kin, kb, str(dev))
    got = _SLICES.get(key)
    if got is None:
        t = X.pick(N, Kin, kb)
        S, WK, KT = int(t["S"]), int(t["wk"]), Kin // 16
        per_r = (KT + S - 1) // S
        sl = []
        for r in range(S):
            r0 = min(KT, r * per_r)
            r1 = min(KT, r0 + per_r)
            per_w = (r1 - r0 + WK - 1) // WK
            for w in range(WK):
                k0 = min(r1, r0 + w * per_w)
                sl.append((k0, min(r1, k0 + per_w)))
        q = next(q for q in (4, 2, 1) if all((b - a) % q == 0 for a, b in sl))
        got = _SLICES[key] = (S, WK, torch.tensor(sl, dtype=torch.int32, device=dev).view(-1), q)
    return got


def slab_rot_sliced(xh: torch.Tensor, wi: torch.Tensor, svh: torch.Tensor, out: torch.Tensor, w,
                    tile: tuple | None = None) -> torch.Tensor:
    """`slab_rot` in the skinny kernel's summation order for weight `w` (the block or group the slab is
    a column range of): out[M, n] gets the bits `tools.exl3_linear.exl3_matmul` gives these rows on the
    skinny kernel (M <= EXL3_CHUNK_UNTIL there), from the decoded slab wi."""
    M, K = xh.shape
    N = wi.shape[1]
    assert N % 128 == 0 and xh.stride(1) == 1 and wi.stride(1) == 1 and out.stride(1) == 1
    S, WK, sl, q = skinny_slices(w.N, w.K, w.kb, xh.device)
    if S == 1:
        KT = K // 16
        L = (KT + WK - 1) // WK                    # gemm_kernel's per_w: every slice but the last
        if L % q == 0:
            bm, bn, nw, st = tile or sliced_tile(K)
            grid = (triton.cdiv(M, bm) * (N // bn),)
            _slab_rot_s1[grid](xh, wi, svh, out, M, N, KT, L, xh.stride(0), wi.stride(0), out.stride(0),
                               BM=bm, BN=bn, Q=q, GROUP=8, OUT_F32=out.dtype == torch.float32,
                               num_warps=nw, num_stages=st)
            return out
    bm, bn, _, nw, st = pick(M, N)
    grid = (triton.cdiv(M, bm) * (N // bn),)
    _slab_rot_sliced[grid](xh, wi, svh, out, sl, M, N, xh.stride(0), wi.stride(0), out.stride(0),
                           BM=bm, BN=bn, BK=16 * q, GROUP=8, S=S, WK=WK, OUT_F32=out.dtype == torch.float32,
                           num_warps=nw, num_stages=st)
    return out


def skinny_matmul(x: torch.Tensor | None, w, *, xh: torch.Tensor | None = None, M: int | None = None,
                  segs: list | None = None) -> torch.Tensor:
    """SPD-245: `exl3_matmul`'s skinny-kernel bits at any row count, through decoded slabs: the rotated
    input (unless `xh`), then per slab (`segs`: (member, n0, n1, Wi) decoded once by the caller, else
    decoded here) the sliced product + rot_out. For rows of several prompts in one call: each row gets
    what its own <= EXL3_CHUNK_UNTIL-row call gives."""
    from tools import exl3_linear as X
    if xh is None:
        M = x.shape[0]
        if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
            x = x.contiguous()
        xh = X.rotate_in(x, w)
    out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=xh.device)
    segs = segs or [(g, n0, n1, None) for g, n0, n1 in X.segments(w, X.SLAB_N)]
    for g, n0, n1, wi in segs:
        slab_rot_sliced(xh[g], wi if wi is not None else w.inner(n0, n1), w.svh[n0:n1], out[:, n0:n1], w)
    return out


# ------------------------------------------------------------------- the 27B's prefill entry points
# Only engine/model.py imports this module (the 27B's arithmetic manifest); the shared EXL3 path in
# tools/exl3_linear.py is read, never changed, so no other family's rows or fingerprint move.

def slab_matmul(x: torch.Tensor | None, w, *, xh: torch.Tensor | None = None, M: int | None = None,
                triton_path: bool = True) -> torch.Tensor:
    """`tools.exl3_linear.exl3_matmul` for a call on the slab path (more than EXL3_CHUNK_UNTIL rows): the
    rotated input (made here unless `xh` is given), then per slab the Triton product + rot_out
    (`triton_path`), or the library one."""
    from tools import exl3_linear as X
    if xh is None:
        M = x.shape[0]
        if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
            x = x.contiguous()
        xh = X.rotate_in(x, w)
    if not triton_path:
        return X.exl3_matmul(None, w, xh=xh, M=M, dev=xh.device)
    out = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=xh.device)
    segs = X._slabs(w) or [(g, n0, n1, None) for g, n0, n1 in X.segments(w, X.SLAB_N)]
    for g, n0, n1, wi in segs:
        slab_rot(xh[g], wi if wi is not None else w.inner(n0, n1), w.svh[n0:n1], out[:, n0:n1])
    return out


def mlp_slab(x: torch.Tensor, gu, wd, *, xh: torch.Tensor | None = None,
             triton_path: bool = True) -> torch.Tensor | None:
    """`tools.exl3_linear.exl3_mlp_slab` with the rotated input given (`xh`, from `add_norm_rot`) and,
    with `triton_path`, gate|up through `slab_rot` into bf16, exl3_rot_in_silu (the bridge's bits) and the
    down projection through `slab_matmul`. None when the shapes do not pair (the caller's usual path)."""
    from tools import exl3_linear as X
    F = wd.K
    if not mlp_slab_pairs(x.shape[0], gu, wd):
        return None
    mod = X._module()
    M = x.shape[0]
    if xh is None:
        xh = X.rotate_in(x, gu)
    segs = X._slabs(gu) or [(g, n0, n1, None) for g, n0, n1 in X.segments(gu, X.SLAB_N)]
    xd = torch.empty(1, M, F, dtype=torch.float16, device=x.device)
    if triton_path:
        y = torch.empty(M, 2 * F, dtype=torch.bfloat16, device=x.device)
        for g, n0, n1, wi in segs:
            slab_rot(xh[g], wi if wi is not None else gu.inner(n0, n1), gu.svh[n0:n1], y[:, n0:n1])
        mod.exl3_rot_in_silu(y, wd.suh, xd)
        del y
        return slab_matmul(None, wd, xh=xd, M=M)
    gs = [t for t in segs if t[0] == 0]
    us = [t for t in segs if t[0] == 1]
    if len(gs) != len(us):
        return None
    for (_, a0, a1, wg), (_, b0, b1, wu) in zip(gs, us):
        if b0 - F != a0 or b1 - a1 != b0 - a0:
            return None
        yg = X._mm_fp32(xh[0], wg if wg is not None else gu.inner(a0, a1))
        yu = X._mm_fp32(xh[1], wu if wu is not None else gu.inner(b0, b1))
        mod.exl3_mlp_bridge(yg, yu, gu.svh[a0:a1], gu.svh[b0:b1], wd.suh[0, a0:a1].contiguous(),
                            xd[0, :, a0:a1])
        del yg, yu
    return X.exl3_matmul(None, wd, xh=xd, M=M, dev=x.device)


def add_rms_norm_out(res: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, out: tuple):
    """`tools.norm_kernels.add_rms_norm` into the given contiguous (h, y) (h may be `res`: a row is read
    before it is written): its kernel, its launch."""
    from tools import norm_kernels as N
    n = x.shape[-1]
    r = res.reshape(-1, n)
    a = x.reshape(-1, n).contiguous()
    h, y = (t.view(-1, n) for t in out)
    N._add_rms_norm[(r.shape[0],)](r, a, weight, h, y, n, EPS=eps, BLOCK=N._block(n), PDL=N._pdl(),
                                   num_warps=8 if n >= 4096 else 4)
    return out


def rms_norm_gated_out(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float,
                       out: torch.Tensor) -> torch.Tensor:
    """`tools.norm_kernels.rms_norm_gated` into `out`: its kernel, its launch."""
    from tools import norm_kernels as N
    n = x.shape[-1]
    flat = x.reshape(-1, n).contiguous()
    g, s_zt, zh = N.gate_rows(gate, n, flat.shape[0])
    y = out.view(-1, n)
    N._rms_norm_gated[(flat.shape[0],)](flat, g, weight, y, n, s_zt, zh, EPS=eps, BLOCK=N._block(n),
                                        PDL=N._pdl(), num_warps=8 if n >= 4096 else 4)
    return out
