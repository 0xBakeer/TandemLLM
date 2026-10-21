"""Batched MoE kernels for Ada (sm_89) and other archs without the sm_121a extensions.

`fn_kernels`'s moe_gu / moe_down / moe_combine arithmetic, in Triton, launched once for every
(row, slot) pair of a call. A pair's computation touches only its own row, its own expert's
pointers (loaded from the bank's device tables, no host sync) and a fixed reduction order, so the
bits a row produces do not depend on M or on the other pairs: the decode/verify row-invariance
contract (ENG-287).

  rot_in    xh[p, j] = Hadamard(x[row(p)] * suh[e, j]) / sqrt(128) -> fp16, one program per
            (pair, matrix, 128-block); the butterfly is the Sylvester ordering scaled at the end.
  down_in   xd[p] = Hadamard(bf16(bf16(silu(g)) * u) * suh[e, 2]) / sqrt(128) -> fp16.
  trellis   yp[p, j, nb*128:] = sum over K of xh[p, j] . W[e, j][:, block]: the EXL3 decode of
            exl3_ada._exl3_gemv_kernel fused with the dot, 128 output columns a program.
  rot_out   y[p, j] = Hadamard(yp) / sqrt(128) * svh[e, j], bf16 for the gate|up buffer, fp32 for
            the down output (`moe_experts`'s yd).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tools.exl3_ada import MUL1, _k_bias, _k_inv

RSQRT128 = tl.constexpr(0.08838834764831845)


@triton.jit
def _fwht(v):
    """The 128-point Walsh-Hadamard of a [128] fp32 vector, unscaled: v -> H_sylvester . v.
    Seven butterfly stages in registers (reshape / trans / split / join)."""
    for i in tl.static_range(7):
        v3 = tl.reshape(v, (64 >> i, 2, 1 << i))
        a, b = tl.split(tl.trans(v3, 0, 2, 1))
        v = tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 2, 1), (128,))
    return v


@triton.jit
def _moe_rot_in(X, IDS, SU, XH, S, K, J: tl.constexpr):
    p = tl.program_id(0)
    j = tl.program_id(1)
    kb128 = tl.program_id(2)
    e = tl.load(IDS + p)
    su = tl.load(SU + e * 3 + j).to(tl.pointer_type(tl.float16))
    off = tl.arange(0, 128)
    x = tl.load(X + (p // S) * K + kb128 * 128 + off).to(tl.float32)
    suh = tl.load(su + kb128 * 128 + off).to(tl.float32)
    xh = (_fwht(x * suh) * RSQRT128).to(tl.float16)
    tl.store(XH + (p * J + j) * K + kb128 * 128 + off, xh)


@triton.jit
def _moe_down_in(YGU, IDS, SU, XD, I):
    p = tl.program_id(0)
    kb128 = tl.program_id(1)
    e = tl.load(IDS + p)
    su = tl.load(SU + e * 3 + 2).to(tl.pointer_type(tl.float16))
    off = tl.arange(0, 128)
    g = tl.load(YGU + (p * 2 + 0) * I + kb128 * 128 + off).to(tl.float32)
    u = tl.load(YGU + (p * 2 + 1) * I + kb128 * 128 + off).to(tl.float32)
    a = (g / (1 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u
    a = a.to(tl.bfloat16).to(tl.float32)
    suh = tl.load(su + kb128 * 128 + off).to(tl.float32)
    xd = (_fwht(a * suh) * RSQRT128).to(tl.float16)
    tl.store(XD + p * I + kb128 * 128 + off, xd)


@triton.jit
def _moe_trellis(XH, IDS, TR, KBt, PINV, YP,
                 K, KT, NT, J: tl.constexpr, TJ: tl.constexpr,
                 K_INV, K_BIAS):
    """grid (P, J, NT // 8): a program's 128 output columns of one (pair, matrix). TJ: the
    bank projection index -- the grid's own j for gate|up (TJ = -1), 2 for the down call."""
    p = tl.program_id(0)
    j = tl.program_id(1)
    nb = tl.program_id(2)
    e = tl.load(IDS + p)
    tj = j if TJ < 0 else TJ
    T = tl.load(TR + e * 3 + tj).to(tl.pointer_type(tl.int32))
    kb = tl.load(KBt + e * 3 + tj)
    W2v = tl.where(kb < 10, kb * 8, (kb // 10) * 8 + 4)
    nbits = W2v * 32

    s = tl.arange(0, 256)
    pinv = tl.load(PINV + s)
    pos = pinv + 1
    end = tl.where(kb < 10, pos * kb, pos * (kb // 10) + pos // 2)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + nbits, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2v, 0, w0i + 1)

    t8 = tl.arange(0, 8)
    wrd = tl.arange(0, 64)                              # W2v <= 64 for every rate (kb 8 = 64 words)
    xh_off = tl.arange(0, 16)
    acc = tl.zeros([8, 16], dtype=tl.float32)
    for kt in range(KT):
        base = (kt * NT + nb * 8) * W2v
        w = tl.load(T + base + t8[:, None] * W2v + wrd[None, :],
                    mask=(wrd < W2v)[None, :], other=0)                   # [8, 32]
        w0 = tl.gather(w, tl.broadcast_to(w0i[None, :], (8, 256)), axis=1)
        w1 = tl.gather(w, tl.broadcast_to(w1i[None, :], (8, 256)), axis=1)
        sw = (off > 16)[None, :]
        lo = tl.where(sw, w1, w0)
        hi = tl.where(sw, w0, w1)
        sh = tl.where(sw, 48 - off[None, :], 16 - off[None, :])
        v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",
                                          "=r,r,r,r", [lo, hi, sh],
                                          dtype=tl.int32, is_pure=True, pack=1) & 0xFFFF
        s1024 = tl.inline_asm_elementwise(
            "dp4a.u32.u32 $0, $1, 0x01010101, 1024;",
            "=r,r", [v * MUL1], dtype=tl.int32, is_pure=True, pack=1)
        val = (s1024.to(tl.float32) * K_INV + K_BIAS).to(tl.float16).to(tl.float32)
        xh = tl.load(XH + (p * J + j) * K + kt * 16 + xh_off).to(tl.float32)      # [16]
        acc += tl.sum(tl.reshape(val, (8, 16, 16)) * xh[None, :, None], axis=1)
    tl.store(YP + (p * J + j) * (NT * 16) + nb * 128 + tl.arange(0, 128),
             tl.reshape(acc, (128,)))


@triton.jit
def _moe_trellis_split(XH, IDS, TR, KBt, PINV, YPP,
                       K, KT, NT, NBS, KS, J: tl.constexpr, TJ: tl.constexpr,
                       K_INV, K_BIAS):
    """`_moe_trellis` with the K range split inside grid axis 2: program (p, j, nb + ks * NBS)
    sums the kt of [ks * KC, (ks + 1) * KC) in order into the fp32 partial YPP[p, j, ks].
    NBS and KS come from the weights' shapes only, so a row's bits do not depend on M."""
    p = tl.program_id(0)
    j = tl.program_id(1)
    nb = tl.program_id(2) % NBS
    ks = tl.program_id(2) // NBS
    e = tl.load(IDS + p)
    tj = j if TJ < 0 else TJ
    T = tl.load(TR + e * 3 + tj).to(tl.pointer_type(tl.int32))
    kb = tl.load(KBt + e * 3 + tj)
    W2v = tl.where(kb < 10, kb * 8, (kb // 10) * 8 + 4)
    nbits = W2v * 32

    s = tl.arange(0, 256)
    pinv = tl.load(PINV + s)
    pos = pinv + 1
    end = tl.where(kb < 10, pos * kb, pos * (kb // 10) + pos // 2)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + nbits, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2v, 0, w0i + 1)

    t8 = tl.arange(0, 8)
    xh_off = tl.arange(0, 16)
    KC = (KT + KS - 1) // KS
    kt0 = ks * KC
    kt1 = tl.minimum(kt0 + KC, KT)
    acc = tl.zeros([8, 16], dtype=tl.float32)
    for kt in range(kt0, kt1):
        base = (kt * NT + nb * 8) * W2v
        # the slot's two window words read in place: a tile's bitstring is one cache line,
        # so the gathers all hit L1; cheaper than a [8, 64] load + register shuffle
        w0 = tl.load(T + base + t8[:, None] * W2v + w0i[None, :])
        w1 = tl.load(T + base + t8[:, None] * W2v + w1i[None, :])
        sw = (off > 16)[None, :]
        lo = tl.where(sw, w1, w0)
        hi = tl.where(sw, w0, w1)
        sh = tl.where(sw, 48 - off[None, :], 16 - off[None, :])
        v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",
                                          "=r,r,r,r", [lo, hi, sh],
                                          dtype=tl.int32, is_pure=True, pack=1) & 0xFFFF
        s1024 = tl.inline_asm_elementwise(
            "dp4a.u32.u32 $0, $1, 0x01010101, 1024;",
            "=r,r", [v * MUL1], dtype=tl.int32, is_pure=True, pack=1)
        val = (s1024.to(tl.float32) * K_INV + K_BIAS).to(tl.float16).to(tl.float32)
        xh = tl.load(XH + (p * J + j) * K + kt * 16 + xh_off).to(tl.float32)
        acc += tl.sum(tl.reshape(val, (8, 16, 16)) * xh[None, :, None], axis=1)
    tl.store(YPP + ((p * J + j) * KS + ks) * (NT * 16) + nb * 128 + tl.arange(0, 128),
             tl.reshape(acc, (128,)))


@triton.jit
def _moe_gtrellis(XH, ORDER, STARTS, COUNTS, TR, KBt, PINV, YP,
                  K, KT, NT, NBS, XSTR, XJSTR, YSTR, YJSTR,
                  J: tl.constexpr, TJ: tl.constexpr, BM: tl.constexpr,
                  K_INV, K_BIAS):
    """Grouped trellis GEMM for prefill: grid (E, ceil(P/BM), J*NBS). Program (e, mb, j*NBS+nb)
    decodes the [16k, 128n] trellis slice of projection (e, tj) once per kt and tl.dots it with
    the BM pairs of expert e that block mb covers (order/starts/counts group pairs by expert).
    No fp16 weight slab is materialised; a pair-block belongs to one expert, so the decode is
    shared BM-fold. Output in the pairs' own order (YP[pair])."""
    e = tl.program_id(0)
    mb = tl.program_id(1)
    nbj = tl.program_id(2)
    j = nbj // NBS
    nb = nbj % NBS
    c0 = tl.load(STARTS + e)
    cnt = tl.load(COUNTS + e)
    if mb * BM >= cnt:
        return
    mo = tl.arange(0, BM)
    pmask = mo < cnt - mb * BM
    prow = tl.load(ORDER + c0 + mb * BM + mo, mask=pmask, other=0)
    tj = j if TJ < 0 else TJ
    T = tl.load(TR + e * 3 + tj).to(tl.pointer_type(tl.int32))
    kb = tl.load(KBt + e * 3 + tj)
    W2v = tl.where(kb < 10, kb * 8, (kb // 10) * 8 + 4)
    nbits = W2v * 32

    s = tl.arange(0, 256)
    pinv = tl.load(PINV + s)
    pos = pinv + 1
    end = tl.where(kb < 10, pos * kb, pos * (kb // 10) + pos // 2)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + nbits, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2v, 0, w0i + 1)

    t8 = tl.arange(0, 8)
    xo = tl.arange(0, 16)
    acc = tl.zeros([BM, 128], dtype=tl.float32)
    for kt in range(KT):
        base = (kt * NT + nb * 8) * W2v
        w0 = tl.load(T + base + t8[:, None] * W2v + w0i[None, :])
        w1 = tl.load(T + base + t8[:, None] * W2v + w1i[None, :])
        sw = (off > 16)[None, :]
        lo = tl.where(sw, w1, w0)
        hi = tl.where(sw, w0, w1)
        sh = tl.where(sw, 48 - off[None, :], 16 - off[None, :])
        v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",
                                          "=r,r,r,r", [lo, hi, sh],
                                          dtype=tl.int32, is_pure=True, pack=1) & 0xFFFF
        s1024 = tl.inline_asm_elementwise(
            "dp4a.u32.u32 $0, $1, 0x01010101, 1024;",
            "=r,r", [v * MUL1], dtype=tl.int32, is_pure=True, pack=1)
        val = (s1024.to(tl.float32) * K_INV + K_BIAS).to(tl.float16)
        # val[t, s]: tile t, s = k * 16 + n_local -> w[k, t * 16 + n_local] as [16, 128]
        w2d = tl.reshape(tl.permute(tl.reshape(val, (8, 16, 16)), (1, 0, 2)), (16, 128))
        xh = tl.load(XH + prow[:, None] * XSTR + j * XJSTR + kt * 16 + xo[None, :],
                     mask=pmask[:, None], other=0)
        acc += tl.dot(xh, w2d)
    n_off = nb * 128 + tl.arange(0, 128)
    tl.store(YP + prow[:, None] * YSTR + j * YJSTR + n_off[None, :],
             acc, mask=pmask[:, None])


@triton.jit
def _moe_combine(YPP, YP, N_OUT, KS: tl.constexpr):
    """YP[pj, n] = the partials' sum in ks order -- a fixed sequence for every output element."""
    pj = tl.program_id(0)
    nb = tl.program_id(1)
    off = nb * 128 + tl.arange(0, 128)
    acc = tl.zeros([128], dtype=tl.float32)
    for ks in tl.static_range(KS):
        acc += tl.load(YPP + (pj * KS + ks) * N_OUT + off)
    tl.store(YP + pj * N_OUT + off, acc)


@triton.jit
def _moe_rot_out_down(YGU_P, IDS, SV, SU, XD, I):
    """`_moe_rot_out` on the (g, u) partials + `_moe_down_in`: the bf16-rounded g/u feed silu,
    the product rotates with the down suh -- all in registers, identical rounding chain."""
    p = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(IDS + p)
    svg = tl.load(SV + e * 3 + 0).to(tl.pointer_type(tl.float16))
    svu = tl.load(SV + e * 3 + 1).to(tl.pointer_type(tl.float16))
    sud = tl.load(SU + e * 3 + 2).to(tl.pointer_type(tl.float16))
    off = tl.arange(0, 128)
    yg = (_fwht(tl.load(YGU_P + (p * 2 + 0) * I + nb * 128 + off)) * RSQRT128
          * tl.load(svg + nb * 128 + off).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    yu = (_fwht(tl.load(YGU_P + (p * 2 + 1) * I + nb * 128 + off)) * RSQRT128
          * tl.load(svu + nb * 128 + off).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    a = (yg / (1 + tl.exp(-yg))).to(tl.bfloat16).to(tl.float32) * yu
    a = a.to(tl.bfloat16).to(tl.float32)
    suh = tl.load(sud + nb * 128 + off).to(tl.float32)
    xd = (_fwht(a * suh) * RSQRT128).to(tl.float16)
    tl.store(XD + p * I + nb * 128 + off, xd)



@triton.jit
def _moe_combine_rotdown(YPP, IDS, SV, SU, XD, I, KS: tl.constexpr):
    """`_moe_combine` + `_moe_rot_out_down`: sums the gate|up K-partials in order, rotates them,
    and does the silu / bf16 chain / down-input rotation -- `ygu_p` is never materialised."""
    p = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(IDS + p)
    svg = tl.load(SV + e * 3 + 0).to(tl.pointer_type(tl.float16))
    svu = tl.load(SV + e * 3 + 1).to(tl.pointer_type(tl.float16))
    sud = tl.load(SU + e * 3 + 2).to(tl.pointer_type(tl.float16))
    off = nb * 128 + tl.arange(0, 128)
    g = tl.zeros([128], dtype=tl.float32)
    u = tl.zeros([128], dtype=tl.float32)
    for ks in tl.static_range(KS):
        g += tl.load(YPP + (p * 2 * KS + ks) * I + off)
        u += tl.load(YPP + ((p * 2 + 1) * KS + ks) * I + off)
    yg = (_fwht(g) * RSQRT128 * tl.load(svg + off).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    yu = (_fwht(u) * RSQRT128 * tl.load(svu + off).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    a = (yg / (1 + tl.exp(-yg))).to(tl.bfloat16).to(tl.float32) * yu
    a = a.to(tl.bfloat16).to(tl.float32)
    suh = tl.load(sud + off).to(tl.float32)
    xd = (_fwht(a * suh) * RSQRT128).to(tl.float16)
    tl.store(XD + p * I + off, xd)


@triton.jit
def _moe_combine_rot(YPP, IDS, SV, YD, N_OUT, KS: tl.constexpr, TJ: tl.constexpr):
    """`_moe_combine` + `_moe_rot_out`: the K-partials sum then fwht * svh, fp32 out; `yd_p`
    is never materialised."""
    p = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(IDS + p)
    sv = tl.load(SV + e * 3 + TJ).to(tl.pointer_type(tl.float16))
    off = nb * 128 + tl.arange(0, 128)
    acc = tl.zeros([128], dtype=tl.float32)
    for ks in tl.static_range(KS):
        acc += tl.load(YPP + (p * KS + ks) * N_OUT + off)
    y = _fwht(acc) * RSQRT128 * tl.load(sv + off).to(tl.float32)
    tl.store(YD + p * N_OUT + off, y)


@triton.jit
def _moe_rot_out(YP, IDS, SV, Y, N_OUT, J: tl.constexpr, TJ: tl.constexpr, BF16: tl.constexpr):
    p = tl.program_id(0)
    j = tl.program_id(1)
    nb = tl.program_id(2)
    e = tl.load(IDS + p)
    tj = j if TJ < 0 else TJ
    sv = tl.load(SV + e * 3 + tj).to(tl.pointer_type(tl.float16))
    off = tl.arange(0, 128)
    yp = tl.load(YP + (p * J + j) * N_OUT + nb * 128 + off)
    svh = tl.load(sv + nb * 128 + off).to(tl.float32)
    y = _fwht(yp) * RSQRT128 * svh
    if BF16:
        tl.store(Y + (p * J + j) * N_OUT + nb * 128 + off, y.to(tl.bfloat16))
    else:
        tl.store(Y + (p * J + j) * N_OUT + nb * 128 + off, y)


_PINV: dict = {}


def _pinv(device) -> torch.Tensor:
    """inverse of the tensor-core perm: slot (r * 16 + c) -> string position."""
    key = str(device)
    if key not in _PINV:
        from tools.exl3_linear import _tile_perm_dev
        perm = _tile_perm_dev(device)
        inv = torch.empty(256, dtype=torch.long, device=device)
        inv[perm] = torch.arange(256, device=device)
        _PINV[key] = inv.to(torch.int32).contiguous()
    return _PINV[key]


_OK: bool | None = None


def moe_experts(x: torch.Tensor, ids: torch.Tensor, bank, yd: torch.Tensor | None = None):
    """`fn_kernels.moe_experts` for archs without the sm_121a module: yd [M*S, H] fp32, every
    (row, slot) pair's unweighted down output in the pairs' own order."""
    M, S = ids.shape
    P = M * S
    H, I = bank.H, bank.I
    dev = x.device
    flat = ids.reshape(-1).to(torch.int32).contiguous()
    pinv = _pinv(dev)

    if not x.is_contiguous():
        x = x.contiguous()
    xh = torch.empty(P, 2, H, dtype=torch.float16, device=dev)
    _moe_rot_in[(P, 2, H // 128)](x, flat, bank.su, xh, S, H, J=2, num_warps=4)

    # K splits until ~768 programs exist for one pair; a function of the projections' shapes
    # only, so a pair's summation order (and bits) never depends on M (ENG-287)
    ks_gu = min(max(768 // (2 * max(I // 128, 1)), 1), min(32, H // 16))
    grouped = 16 <= P <= 128
    if grouped:
        order = torch.empty(P, dtype=torch.int32, device=dev)
        es = torch.empty(P, dtype=torch.int32, device=dev)
        lead = torch.empty(P, dtype=torch.int32, device=dev)
        glen = torch.empty(P, dtype=torch.int32, device=dev)
        gc = torch.empty(1, dtype=torch.int32, device=dev)
        _moe_groups_fused[(1,)](flat, order, es, lead, glen, gc, P, G=2, num_warps=1)
    if ks_gu > 1:
        ygu_pp = torch.empty(P, 2 * ks_gu, I, dtype=torch.float32, device=dev)
        if grouped:
            _moe_trellis_split_g[(P, 2, (I // 128) * ks_gu)](
                xh, order, es, lead, glen, gc, bank.tr, bank.kb, pinv, ygu_pp,
                H, H // 16, I // 16, I // 128, ks_gu, J=2, TJ=-1,
                K_INV=_k_inv(), K_BIAS=_k_bias(), G=2, num_warps=1, num_stages=1)
        else:
            _moe_trellis_split[(P, 2, (I // 128) * ks_gu)](
                xh, flat, bank.tr, bank.kb, pinv, ygu_pp,
                H, H // 16, I // 16, I // 128, ks_gu, J=2, TJ=-1,
                K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=1, num_stages=1)
        xd = torch.empty(P, I, dtype=torch.float16, device=dev)
        _moe_combine_rotdown[(P, I // 128)](ygu_pp, flat, bank.sv, bank.su, xd, I,
                                           KS=ks_gu, num_warps=4)
        del ygu_pp
    else:
        ygu_p = torch.empty(P, 2, I, dtype=torch.float32, device=dev)
        _moe_trellis[(P, 2, I // 128)](xh, flat, bank.tr, bank.kb, pinv, ygu_p,
                                      H, H // 16, I // 16, J=2, TJ=-1,
                                      K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=2, num_stages=1)
        xd = torch.empty(P, I, dtype=torch.float16, device=dev)
        _moe_rot_out_down[(P, I // 128)](ygu_p, flat, bank.sv, bank.su, xd, I, num_warps=4)
        del ygu_p
    del xh

    if yd is None:
        yd = torch.empty(P, H, dtype=torch.float32, device=dev)
    ks_dn = min(max(768 // max(H // 128, 1), 1), min(32, I // 16))
    if ks_dn > 1:
        yd_pp = torch.empty(P, ks_dn, H, dtype=torch.float32, device=dev)
        if grouped:
            _moe_trellis_split_g[(P, 1, (H // 128) * ks_dn)](
                xd, order, es, lead, glen, gc, bank.tr, bank.kb, pinv, yd_pp,
                I, I // 16, H // 16, H // 128, ks_dn, J=1, TJ=2,
                K_INV=_k_inv(), K_BIAS=_k_bias(), G=2, num_warps=1, num_stages=1)
        else:
            _moe_trellis_split[(P, 1, (H // 128) * ks_dn)](
                xd, flat, bank.tr, bank.kb, pinv, yd_pp,
                I, I // 16, H // 16, H // 128, ks_dn, J=1, TJ=2,
                K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=1, num_stages=1)
        _moe_combine_rot[(P, H // 128)](yd_pp, flat, bank.sv, yd.view(P, H), H,
                                       KS=ks_dn, TJ=2, num_warps=4)
        del yd_pp
    else:
        yd_p = torch.empty(P, 1, H, dtype=torch.float32, device=dev)
        _moe_trellis[(P, 1, H // 128)](xd, flat, bank.tr, bank.kb, pinv, yd_p,
                                      I, I // 16, H // 16, J=1, TJ=2,
                                      K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=2, num_stages=1)
        _moe_rot_out[(P, 1, H // 128)](yd_p, flat, bank.sv, yd.view(P, 1, H), H,
                                      J=1, TJ=2, BF16=False, num_warps=4)
    del xd
    return yd


def moe_grouped_ada(x: torch.Tensor, ids: torch.Tensor, bank) -> torch.Tensor:
    """`fn_prefill_kernels.moe_grouped` for archs without the sm_121a grouped GEMM: every pair's
    rotations run batched (`_moe_rot_in`, `_moe_rot_out_down`), each used expert's gate/up and
    down trellises decode once to fp16 slabs and the pairs assigned to it take the library GEMM
    (fp32 products, as `moe_grouped`). Output yd [M*S, H] fp32 in pair order."""
    from tools import exl3_ada  # noqa: F401  (decode kept for the reference path)
    M, S = ids.shape
    P = M * S
    dev = x.device
    H, I = bank.H, bank.I
    flat = ids.reshape(-1).to(torch.int32).contiguous()
    pinv = _pinv(dev)
    BM = 32

    # every pair's rotated input, batched (same kernel as the decode path)
    if not x.is_contiguous():
        x = x.contiguous()
    xh = torch.empty(P, 2, H, dtype=torch.float16, device=dev)
    _moe_rot_in[(P, 2, H // 128)](x, flat, bank.su, xh, S, H, J=2, num_warps=4)

    # pairs grouped by expert: each (expert, pair-block, n-block) program decodes the trellis
    # once for its BM pairs (in-kernel decode + tl.dot, no fp16 slab materialised)
    order = torch.argsort(flat, stable=True).to(torch.int32).contiguous()
    counts = torch.bincount(flat[order.long()].to(torch.long), minlength=bank.n).to(torch.int32).contiguous()
    starts = (counts.cumsum(0) - counts).to(torch.int32).contiguous()
    mbs = (P + BM - 1) // BM

    ygu_p = torch.empty(P, 2, I, dtype=torch.float32, device=dev)
    _moe_gtrellis[(bank.n, mbs, 2 * (I // 128))](
        xh, order, starts, counts, bank.tr, bank.kb, pinv, ygu_p,
        H, H // 16, I // 16, I // 128, 2 * H, H, 2 * I, I, J=2, TJ=-1, BM=BM,
        K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=8, num_stages=1)
    del xh

    # silu(g)*u + the down input rotation, batched (the decode path's fused kernel)
    xd = torch.empty(P, I, dtype=torch.float16, device=dev)
    _moe_rot_out_down[(P, I // 128)](ygu_p, flat, bank.sv, bank.su, xd, I, num_warps=4)
    del ygu_p

    yd_p = torch.empty(P, H, dtype=torch.float32, device=dev)
    _moe_gtrellis[(bank.n, mbs, H // 128)](
        xd, order, starts, counts, bank.tr, bank.kb, pinv, yd_p,
        I, I // 16, H // 16, H // 128, I, 0, H, 0, J=1, TJ=2, BM=BM,
        K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=8, num_stages=1)
    del xd

    yd = torch.empty(P, H, dtype=torch.float32, device=dev)
    _moe_rot_out[(P, 1, H // 128)](yd_p, flat, bank.sv, yd.view(P, 1, H), H,
                                  J=1, TJ=2, BF16=False, num_warps=4)
    return yd


@triton.jit
def _moe_groups_fused(FLAT, ORD, ES, LEAD, LEN, GC, P, G: tl.constexpr):
    """The whole pair->group build in one launch: stable-sort the P pair expert ids (rank sort,
    P <= 128 fits one program), write the sorted order and ids, then mark the group leads (a new
    expert run or every G inside a run), compact them into LEAD/LEN, and store the count.
    Identical output to argsort+index_select+_moe_groups_build."""
    r = tl.arange(0, 128)
    im = r < P
    e = tl.load(FLAT + r, mask=im, other=0x3FFFFFFF)
    # stable rank: elements sorting before i are (ej < ei) or (ej == ei and j < i)
    ej = e[None, :]
    ei = e[:, None]
    before = (ej < ei) | ((ej == ei) & (tl.arange(0, 128)[None, :] < r[:, None]))
    rank = tl.sum(before.to(tl.int32), axis=1)
    tl.store(ORD + rank, r, mask=im)
    tl.store(ES + rank, e, mask=im)
    # sorted expert at position q = the e_i whose rank == q
    esv = tl.sum(tl.where(rank[None, :] == r[:, None], e[None, :], 0), axis=1)
    prev = tl.sum(tl.where(rank[None, :] == r[:, None] - 1, e[None, :], 0), axis=1)
    prev = tl.where(r == 0, -1, prev)
    run0 = tl.min(tl.where(esv[None, :] == esv[:, None], r[None, :], P), axis=1)
    is_lead = im & ((r == 0) | (esv != prev) | ((r - run0) % G == 0))
    gidx = tl.cumsum(is_lead.to(tl.int32), axis=0) - 1
    tl.store(LEAD + gidx, r, mask=is_lead)
    nxt = tl.min(tl.where(is_lead[None, :] & (r[None, :] > r[:, None]), r[None, :], P), axis=1)
    tl.store(LEN + gidx, nxt - r, mask=is_lead)
    tl.store(GC, tl.sum(is_lead.to(tl.int32)))


@triton.jit
def _moe_groups_build(E_SORT, LEAD, LEN, GC, P, G: tl.constexpr):
    """Scan the expert-sorted pairs once: LEAD[g] is the sorted position where group g starts,
    LEN[g] its size (<= G, a run of one expert chunked), GC[0] the group count. One program and
    fixed shapes, so it is safe inside a captured graph."""
    g = 0
    n = 0
    prev = -1
    for i in range(P):
        e = tl.load(E_SORT + i)
        if i == 0 or e != prev or n == G:
            if i > 0:
                tl.store(LEN + g, n)
                g += 1
            tl.store(LEAD + g, i)
            n = 0
            prev = e
        n += 1
    tl.store(LEN + g, n)
    tl.store(GC, g + 1)


@triton.jit
def _moe_trellis_split_g(XH, ORD, ES, LEAD, LEN, GC, TR, KBt, PINV, YPP,
                         K, KT, NT, NBS, KS, J: tl.constexpr, TJ: tl.constexpr,
                         K_INV, K_BIAS, G: tl.constexpr):
    """`_moe_trellis_split` where program g decodes its tile once for every pair of its group
    (<= G pairs that share the expert). acc is [8, 16, G]; the per-pair tree sum over the 16
    elements keeps the ungrouped kernel's order, so a pair's bits do not depend on how many
    partners share its expert."""
    gi = tl.program_id(0)
    j = tl.program_id(1)
    nb = tl.program_id(2) % NBS
    ks = tl.program_id(2) // NBS
    ng = tl.load(GC)
    if gi >= ng:
        return
    lead = tl.load(LEAD + gi)
    ln = tl.load(LEN + gi)
    tj = j if TJ < 0 else TJ
    e = tl.load(ES + lead)
    T = tl.load(TR + e * 3 + tj).to(tl.pointer_type(tl.int32))
    kb = tl.load(KBt + e * 3 + tj)
    W2v = tl.where(kb < 10, kb * 8, (kb // 10) * 8 + 4)
    nbits = W2v * 32

    s = tl.arange(0, 256)
    pinv = tl.load(PINV + s)
    pos = pinv + 1
    end = tl.where(kb < 10, pos * kb, pos * (kb // 10) + pos // 2)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + nbits, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2v, 0, w0i + 1)

    t8 = tl.arange(0, 8)
    ga = tl.arange(0, G)
    gm = ga < ln
    pair = tl.load(ORD + lead + ga, mask=gm, other=0)          # [G] pair ids
    xh_off = tl.arange(0, 16)
    KC = (KT + KS - 1) // KS
    kt0 = ks * KC
    kt1 = tl.minimum(kt0 + KC, KT)
    acc = tl.zeros([8, 16, G], dtype=tl.float32)
    for kt in range(kt0, kt1):
        base = (kt * NT + nb * 8) * W2v
        w0 = tl.load(T + base + t8[:, None] * W2v + w0i[None, :])
        w1 = tl.load(T + base + t8[:, None] * W2v + w1i[None, :])
        sw = (off > 16)[None, :]
        lo = tl.where(sw, w1, w0)
        hi = tl.where(sw, w0, w1)
        sh = tl.where(sw, 48 - off[None, :], 16 - off[None, :])
        v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",
                                          "=r,r,r,r", [lo, hi, sh],
                                          dtype=tl.int32, is_pure=True, pack=1) & 0xFFFF
        s1024 = tl.inline_asm_elementwise(
            "dp4a.u32.u32 $0, $1, 0x01010101, 1024;",
            "=r,r", [v * MUL1], dtype=tl.int32, is_pure=True, pack=1)
        val = (s1024.to(tl.float32) * K_INV + K_BIAS).to(tl.float16).to(tl.float32)
        xh = tl.load(XH + (pair[None, :] * J + j) * K + kt * 16 + xh_off[:, None],
                     mask=gm[None, :], other=0.0).to(tl.float32)   # [16, G]
        acc += tl.sum(tl.reshape(val, (8, 16, 16, 1)) * xh[None, :, None, :], axis=1)
    cols = nb * 128 + tl.arange(0, 128)
    for g in tl.static_range(G):
        yp = tl.reshape(tl.sum(acc * (ga[None, None, :] == g), axis=2), (128,))
        pg = tl.sum(tl.where(ga == g, pair, 0), 0)
        tl.store(YPP + ((pg * J + j) * KS + ks) * (NT * 16) + cols,
                 yp, mask=(g < ln) & (cols >= 0))


def try_moe_experts(x: torch.Tensor, ids: torch.Tensor, bank):
    """`moe_experts` gated on a one-time agreement check against the torch reference; None when
    the kernels are unavailable or disagree."""
    global _OK
    if _OK is None:
        try:
            from tools import fn_prefill_kernels as PK
            ref = PK.moe_grouped_reference(x, ids, bank)
            got = moe_experts(x, ids, bank)
            _OK = bool(torch.allclose(ref, got, atol=4e-2, rtol=4e-2))
        except Exception:
            _OK = False
    if not _OK:
        return None
    try:
        return moe_experts(x, ids, bank)
    except Exception:
        _OK = False
        return None


@triton.jit
def _moe_combine_ada(YD, IDS, WTS, OUT, H, S: tl.constexpr):
    """ in one launch: the slots of row m in ascending expert id (stable), each yd
    pair row weighted and summed fp32, one bf16 rounding. Grid (M, H/128)."""
    m = tl.program_id(0)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    sl = tl.arange(0, S)
    e = tl.load(IDS + m * S + sl)
    # stable rank: r slots sort before i when (e_r < e_i) or (e_r == e_i and r < i)
    rank = tl.sum(((e[None, :] < e[:, None]) | ((e[None, :] == e[:, None]) & (sl[None, :] < sl[:, None]))).to(tl.int32), axis=1)
    acc = tl.zeros([128], dtype=tl.float32)
    for k in tl.static_range(S):
        slot = tl.sum(tl.where(rank == k, sl, 0), 0)
        w = tl.load(WTS + m * S + slot)
        y = tl.load(YD + (m * S + slot) * H + cols)
        # the product rounds to fp32 before it sums, as the reference's two ops do; written in
        # asm so ptxas cannot contract it into the accumulate
        acc += tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [w, y],
                                         dtype=tl.float32, is_pure=True, pack=1)
    tl.store(OUT + m * H + cols, acc.to(tl.bfloat16))


def try_moe_combine(yd: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor,
                    out: torch.Tensor | None = None):
    """ on the Ada kernels; None when yd/ids shapes do not fit."""
    M, S = ids.shape
    H = yd.shape[1]
    if yd.shape[0] != M * S or H % 128 or yd.dtype != torch.float32:
        return None
    if out is None:
        out = torch.empty(M, H, dtype=torch.bfloat16, device=yd.device)
    _moe_combine_ada[(M, H // 128)](yd, ids.contiguous(), wts.contiguous(), out, H,
                                  S=S, num_warps=4)
    return out
