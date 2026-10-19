"""EXL3 kernels for Ada (sm_89) and any arch the sm_121a inline extensions do not cover.

`decode_trellis`'s arithmetic, in Triton: each 16x16 tile's bit string (W int16 words, the
MSB-first packing the CUDA kernels and `decode_trellis` share) is decoded by a two-word gather
per weight -- a weight's 16-bit window ends at `end(p)` and spans at most two int32 words. The
mul1 codebook transform and the fp16 affine are elementwise. Bit for bit the torch reference:
integer arithmetic throughout, the fp32 affine rounded once to fp16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: 0x83DCD12D as a signed int32: the wraparound product's low 32 bits are the unsigned one's
MUL1 = tl.constexpr(-2082680531)
RSQRT128 = tl.constexpr(0.08838834764831845)


@triton.jit
def _fwht(v):
    """The 128-point Walsh-Hadamard of a [128] fp32 vector, unscaled: v -> H_sylvester . v.
    Seven butterfly stages in registers (reshape / trans / split / join); a fixed order, so a
    row's rotated bits do not depend on M or on any other row."""
    for i in tl.static_range(7):
        v3 = tl.reshape(v, (64 >> i, 2, 1 << i))
        a, b = tl.split(tl.trans(v3, 0, 2, 1))
        v = tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 2, 1), (128,))
    return v


@triton.jit
def _exl3_rot_in_kernel(X, SUH, XH, X_STRIDE, K):
    """XH[m, b] = fp16(H(fp32(X[m, b]) * fp32(SUH[b])) / sqrt(128)) for 128-block b: `rot_in`'s
    arithmetic, one program a (row, block), so a row's bits do not depend on M."""
    m = tl.program_id(0)
    b = tl.program_id(1)
    off = b * 128 + tl.arange(0, 128)
    x = tl.load(X + m * X_STRIDE + off).to(tl.float32)
    suh = tl.load(SUH + off).to(tl.float32)
    xh = (_fwht(x * suh) * RSQRT128).to(tl.float16)
    tl.store(XH + m * K + off, xh)


@triton.jit
def _exl3_rot_in_silu_kernel(YGU, SUH, XH, Y_STRIDE, I):
    """`rot_in` of the MLP's down projection input fused with its activation: per row and
    128-block, xh = fp16(H(bf16(bf16(silu(g)) * u) * suh) / sqrt(128)) -- bit for bit the torch
    silu, the bf16 products and the rotation (the double rounding of `rot_in_silu_kernel`)."""
    m = tl.program_id(0)
    b = tl.program_id(1)
    off = b * 128 + tl.arange(0, 128)
    g = tl.load(YGU + m * Y_STRIDE + off).to(tl.float32)
    u = tl.load(YGU + m * Y_STRIDE + I + off).to(tl.float32)
    a = (g / (1 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u
    a = a.to(tl.bfloat16).to(tl.float32)
    suh = tl.load(SUH + off).to(tl.float32)
    xh = (_fwht(a * suh) * RSQRT128).to(tl.float16)
    tl.store(XH + m * I + off, xh)


@triton.jit
def _exl3_rot_out_kernel(YP, SVH, Y, N_STRIDE, BF16: tl.constexpr):
    """Y[m, b] = H(YP[m, b]) / sqrt(128) * fp32(SVH[b]): `rot_out`'s arithmetic to bf16 (or fp32
    for the vocabulary head), one program a (row, 128-block)."""
    m = tl.program_id(0)
    b = tl.program_id(1)
    off = b * 128 + tl.arange(0, 128)
    yp = tl.load(YP + m * N_STRIDE + off)
    svh = tl.load(SVH + off).to(tl.float32)
    y = _fwht(yp) * RSQRT128 * svh
    if BF16:
        tl.store(Y + m * N_STRIDE + off, y.to(tl.bfloat16))
    else:
        tl.store(Y + m * N_STRIDE + off, y)


@triton.jit
def _exl3_decode_kernel(T, OUT, PERM,
                        NT_IN, N_TILES, OUT_STRIDE,
                        W2: tl.constexpr, KB: tl.constexpr,
                        HALF: tl.constexpr, TILES_N: tl.constexpr,
                        K_INV, K_BIAS):
    """One program: TILES_N adjacent n-tiles of one K-tile row (256 * TILES_N weights).

    T: int32 words of the trellis [KT, NT_IN, W2] (the int16 tensor viewed pairwise).
    OUT: fp16 [KT*16, N_TILES*16], the decoded Wi's columns n0*16:.
    PERM: int32 [256], position -> slot inside the tile (exllamav3's tensor_core_perm).
    K_INV / K_BIAS: fp32, decode_trellis's fp16(0x1eee) and fp16(0xc931) as fp32.
    """
    kt = tl.program_id(0)
    nt_out0 = tl.program_id(1) * TILES_N
    p = tl.arange(0, 256)

    # window end (exclusive) per position: integer rates (p+1)*kb; a half-integer rate's
    # odd weights take one bit more
    if HALF:
        end = (p + 1) * (KB // 10) + (p + 1) // 2
    else:
        end = (p + 1) * KB
    a0 = end - 16
    # the tail-biting wrap: a negative start reads the string's end (word-aligned, W2 * 32)
    a = tl.where(a0 < 0, a0 + W2 * 32, a0)
    off = a % 32                                 # bit offset inside the first word
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2, 0, w0i + 1)

    perm = tl.load(PERM + p)                     # [256]
    r = perm // 16
    c = perm % 16

    for j in tl.static_range(TILES_N):
        nt_out = nt_out0 + j
        if nt_out < N_TILES:
            base = (kt * NT_IN + nt_out) * W2    # nt0 == 0: the wrapper slices columns itself
            w0 = tl.load(T + base + w0i)         # gathered; ~W2 distinct words, L1-resident
            w1 = tl.load(T + base + w1i)
            sw = off > 16
            lo = tl.where(sw, w1, w0)
            hi = tl.where(sw, w0, w1)
            sh = tl.where(sw, 48 - off, 16 - off)
            v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",
                                              "=r,r,r,r", [lo, hi, sh],
                                              dtype=tl.int32, is_pure=True, pack=1) & 0xFFFF
            s1024 = tl.inline_asm_elementwise(
                "dp4a.u32.u32 $0, $1, 0x01010101, 1024;",
                "=r,r", [v * MUL1], dtype=tl.int32, is_pure=True, pack=1)
            val = (s1024.to(tl.float32) * K_INV + K_BIAS).to(tl.float16)

            tl.store(OUT + (kt * 16 + r) * OUT_STRIDE + nt_out * 16 + c, val)


@triton.jit
def _exl3_gemv_kernel(XH, T, OUT, END_S,
                      KSTR, NB, XH_STRIDE, OUT_STRIDE, KT,
                      W2: tl.constexpr, W2P: tl.constexpr,
                      K_INV, K_BIAS):
    """Y[m, tile] = sum over the K tiles of x.h[m, k] . W[k, n]: the decode of `_exl3_decode_kernel`
    fused with the GEMV. One program owns one (row, 16-column tile): its K reduction runs kt =
    0..KT-1 in order inside the program, so a row's bits do not depend on M or on any other row
    (the decode/verify row-invariance contract). `s` enumerates the 256 slots of a tile in (r, c)
    order -- END_S holds the slot's window end -- so the decoded values are already a [16, 16]
    matrix; no register gather."""
    m = tl.program_id(0)
    nb = tl.program_id(1)
    s = tl.arange(0, 256)
    end = tl.load(END_S + s)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + W2 * 32, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2, 0, w0i + 1)
    xh_off = tl.arange(0, 16)
    wrd = tl.arange(0, W2P)                             # W2P = next_pow2(W2): covers every rate
                                                        # (W2 is 16..64 int32 for kb 2..8, <= 28 for halves)
    acc = tl.zeros([16], dtype=tl.float32)
    for kt in range(KT):
        base = kt * KSTR + nb * W2
        w = tl.load(T + base + wrd, mask=wrd < W2, other=0)               # [32]
        w0 = tl.gather(w, w0i, axis=0)
        w1 = tl.gather(w, w1i, axis=0)
        single = (w0 >> tl.minimum(tl.maximum(16 - off, 0), 31)) & 0xFFFF
        split = ((w0 & ((1 << tl.minimum(tl.maximum(32 - off, 0), 31)) - 1))
                 << tl.minimum(tl.maximum(off - 16, 0), 31)) | \
                 ((w1 >> tl.minimum(tl.maximum(48 - off, 0), 31))
                  & ((1 << tl.minimum(tl.maximum(off - 16, 0), 31)) - 1))
        sw = off > 16
        lo = tl.where(sw, w1, w0)
        hi = tl.where(sw, w0, w1)
        sh = tl.where(sw, 48 - off, 16 - off)
        v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",
                                      "=r,r,r,r", [lo, hi, sh],
                                      dtype=tl.int32, is_pure=True, pack=1) & 0xFFFF
        s1024 = tl.inline_asm_elementwise(
            "dp4a.u32.u32 $0, $1, 0x01010101, 1024;",
            "=r,r", [v * MUL1], dtype=tl.int32, is_pure=True, pack=1)
        val = (s1024.to(tl.float32) * K_INV + K_BIAS).to(tl.float16).to(tl.float32)
        xh = tl.load(XH + m * XH_STRIDE + kt * 16 + xh_off).to(tl.float32)   # [16]
        acc += tl.sum(tl.reshape(val, (16, 16)) * xh[:, None], axis=0)
    tl.store(OUT + m * OUT_STRIDE + nb * 16 + xh_off, acc)


@triton.jit
def _exl3_gemv_split_kernel(XH, T, P_OUT, END_S,
                            NT, KSTR, NB, XH_STRIDE, KT, KS,
                            W2: tl.constexpr, W2P: tl.constexpr,
                            K_INV, K_BIAS):
    """`_exl3_gemv_kernel` with the K range split over a third grid axis: program (m, nb, ks)
    sums the kt in [ks * KC, (ks + 1) * KC) in order into the fp32 partial P_OUT[m, ks, nb*16:].
    KS and the chunk boundaries depend on the trellis's shape only, so a row's bits do not
    depend on M or on any other row."""
    m = tl.program_id(0)
    nb = tl.program_id(1) * 8                            # first of the program's 8 n-tiles
    ks = tl.program_id(2)
    s = tl.arange(0, 256)
    end = tl.load(END_S + s)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + W2 * 32, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2, 0, w0i + 1)
    xh_off = tl.arange(0, 16)
    t8 = tl.arange(0, 8)
    wrd = tl.arange(0, W2P)
    KC = (KT + KS - 1) // KS
    kt0 = ks * KC
    kt1 = tl.minimum(kt0 + KC, KT)
    nv = (nb + t8) < NT                                   # valid tiles of the program's 8
    acc = tl.zeros([8, 16], dtype=tl.float32)
    for kt in range(kt0, kt1):
        base = kt * KSTR + nb * W2
        w = tl.load(T + base + t8[:, None] * W2 + wrd[None, :],
                    mask=(wrd < W2)[None, :] & nv[:, None], other=0)   # [8, W2P]
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
        xh = tl.load(XH + m * XH_STRIDE + kt * 16 + xh_off).to(tl.float32)
        acc += tl.sum(tl.reshape(val, (8, 16, 16)) * xh[None, :, None], axis=1)
    tl.store(P_OUT + (m * KS + ks) * NB * 16 + nb * 16 + t8[:, None] * 16 + xh_off[None, :],
             acc, mask=nv[:, None])


@triton.jit
def _exl3_gemv_split_direct_kernel(XH, T, P_OUT, END_S,
                                   NT, KSTR, NB, XH_STRIDE, KT, KS,
                                   W2: tl.constexpr, K_INV, K_BIAS):
    """`_exl3_gemv_split_kernel` reading the window words in place: for W2 > 32 the masked
    [8, W2P] load + register gather costs more than two addressed [8, 256] loads (a tile's
    bitstring is a cache line, so they all hit L1). Same summation order -- same bits."""
    m = tl.program_id(0)
    nb = tl.program_id(1) * 8
    ks = tl.program_id(2)
    s = tl.arange(0, 256)
    end = tl.load(END_S + s)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + W2 * 32, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2, 0, w0i + 1)
    xh_off = tl.arange(0, 16)
    t8 = tl.arange(0, 8)
    KC = (KT + KS - 1) // KS
    kt0 = ks * KC
    kt1 = tl.minimum(kt0 + KC, KT)
    nv = (nb + t8) < NT
    acc = tl.zeros([8, 16], dtype=tl.float32)
    for kt in range(kt0, kt1):
        base = kt * KSTR + nb * W2
        w0 = tl.load(T + base + t8[:, None] * W2 + w0i[None, :], mask=nv[:, None], other=0)
        w1 = tl.load(T + base + t8[:, None] * W2 + w1i[None, :], mask=nv[:, None], other=0)
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
        xh = tl.load(XH + m * XH_STRIDE + kt * 16 + xh_off).to(tl.float32)
        acc += tl.sum(tl.reshape(val, (8, 16, 16)) * xh[None, :, None], axis=1)
    tl.store(P_OUT + (m * KS + ks) * NB * 16 + nb * 16 + t8[:, None] * 16 + xh_off[None, :],
             acc, mask=nv[:, None])


@triton.jit
def _exl3_gemv_combine_kernel(P_OUT, OUT, NB16, KS: tl.constexpr):
    """OUT[m, n] = sum over ks of the partials in ks order. One program owns (m, 256 columns):
    the addition sequence for an output element is fixed whatever M is."""
    m = tl.program_id(0)
    nb = tl.program_id(1)
    off = nb * 256 + tl.arange(0, 256)
    acc = tl.zeros([256], dtype=tl.float32)
    for ks in tl.static_range(KS):
        acc += tl.load(P_OUT + (m * KS + ks) * NB16 + off)
    tl.store(OUT + m * NB16 + off, acc)


@triton.jit
def _exl3_gemv_combine_rot_kernel(P_OUT, SVH, OUT, NB16,
                                  KS: tl.constexpr, BF16: tl.constexpr):
    """`_exl3_gemv_combine_kernel` + `rot_out`: OUT[m, n] = H(sum_ks P_OUT[m, ks, n]) / sqrt(128)
    * SVH[n], bf16 or fp32. The 256-column program covers two 128-blocks; the addition sequence
    is the combine's, the rotation's arithmetic is unchanged, so the bits are identical."""
    m = tl.program_id(0)
    nb = tl.program_id(1)
    off = nb * 256 + tl.arange(0, 256)
    acc = tl.zeros([256], dtype=tl.float32)
    for ks in tl.static_range(KS):
        acc += tl.load(P_OUT + (m * KS + ks) * NB16 + off)
    svh = tl.load(SVH + off).to(tl.float32)
    h2 = tl.reshape(acc, (2, 128))
    s2 = tl.reshape(svh, (2, 128))
    a, b = tl.split(tl.trans(h2, 1, 0))
    sa, sb = tl.split(tl.trans(s2, 1, 0))
    y0 = _fwht(a) * RSQRT128 * sa
    y1 = _fwht(b) * RSQRT128 * sb
    y = tl.reshape(tl.trans(tl.join(y0, y1), 1, 0), (256,))
    if BF16:
        tl.store(OUT + m * NB16 + off, y.to(tl.bfloat16))
    else:
        tl.store(OUT + m * NB16 + off, y)


_END_S: dict = {}


def _end_slot(kb: int, device) -> torch.Tensor:
    """window end per slot (r * 16 + c): decode_trellis's `end` gathered by the inverse of the
    tensor-core perm, so the kernel's lane index reads the tile in (r, c) order."""
    key = (kb, str(device))
    if key not in _END_S:
        from tools.exl3_linear import _tile_perm_dev
        p = torch.arange(256, device=device)
        end = (p + 1) * kb if kb < 10 else (p + 1) * (kb // 10) + (p + 1) // 2
        perm = _tile_perm_dev(device)
        pinv = torch.empty(256, dtype=torch.long, device=device)
        pinv[perm] = p
        _END_S[key] = end[pinv].to(torch.int32).contiguous()
    return _END_S[key]


@triton.jit
def _exl3_gemv_split_g_kernel(XH, T, P_OUT, END_S,
                              NT, KSTR, NB, XH_STRIDE, KT, KS, M,
                              W2: tl.constexpr, W2P: tl.constexpr,
                              K_INV, K_BIAS, G: tl.constexpr):
    """`_exl3_gemv_split_kernel` where a program decodes its tile once for G rows: acc is
    [8, 16, G] and each row's tree sum keeps the ungrouped order, so a row's bits do not
    depend on its row-group mates."""
    mg = tl.program_id(0)
    nb = tl.program_id(1) * 8
    ks = tl.program_id(2)
    s = tl.arange(0, 256)
    end = tl.load(END_S + s)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + W2 * 32, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2, 0, w0i + 1)
    xh_off = tl.arange(0, 16)
    t8 = tl.arange(0, 8)
    wrd = tl.arange(0, W2P)
    ga = tl.arange(0, G)
    row = mg * G + ga
    rm = row < M
    KC = (KT + KS - 1) // KS
    kt0 = ks * KC
    kt1 = tl.minimum(kt0 + KC, KT)
    nv = (nb + t8) < NT
    acc = tl.zeros([8, 16, G], dtype=tl.float32)
    for kt in range(kt0, kt1):
        base = kt * KSTR + nb * W2
        w = tl.load(T + base + t8[:, None] * W2 + wrd[None, :],
                    mask=(wrd < W2)[None, :] & nv[:, None], other=0)   # [8, W2P]
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
        xh = tl.load(XH + row[None, :] * XH_STRIDE + kt * 16 + xh_off[:, None],
                     mask=rm[None, :], other=0.0).to(tl.float32)   # [16, G]
        acc += tl.sum(tl.reshape(val, (8, 16, 16, 1)) * xh[None, :, None, :], axis=1)
    for g in tl.static_range(G):
        yp = tl.sum(acc * (ga[None, None, :] == g), axis=2)          # [8, 16]
        rw = mg * G + g
        tl.store(P_OUT + (rw * KS + ks) * NB * 16 + nb * 16 + t8[:, None] * 16 + xh_off[None, :],
                 yp, mask=nv[:, None] & (rw < M))


@triton.jit
def _exl3_gemv_split_direct_g_kernel(XH, T, P_OUT, END_S,
                                     NT, KSTR, NB, XH_STRIDE, KT, KS, M,
                                     W2: tl.constexpr, K_INV, K_BIAS, G: tl.constexpr):
    """`_exl3_gemv_split_direct_kernel` grouped over G rows: addressed window loads (as the
    ungrouped direct variant, same summation order)."""
    mg = tl.program_id(0)
    nb = tl.program_id(1) * 8
    ks = tl.program_id(2)
    s = tl.arange(0, 256)
    end = tl.load(END_S + s)
    a0 = end - 16
    a = tl.where(a0 < 0, a0 + W2 * 32, a0)
    off = a % 32
    w0i = a // 32
    w1i = tl.where(w0i + 1 >= W2, 0, w0i + 1)
    xh_off = tl.arange(0, 16)
    t8 = tl.arange(0, 8)
    ga = tl.arange(0, G)
    row = mg * G + ga
    rm = row < M
    KC = (KT + KS - 1) // KS
    kt0 = ks * KC
    kt1 = tl.minimum(kt0 + KC, KT)
    nv = (nb + t8) < NT
    acc = tl.zeros([8, 16, G], dtype=tl.float32)
    for kt in range(kt0, kt1):
        base = kt * KSTR + nb * W2
        w0 = tl.load(T + base + t8[:, None] * W2 + w0i[None, :], mask=nv[:, None], other=0)
        w1 = tl.load(T + base + t8[:, None] * W2 + w1i[None, :], mask=nv[:, None], other=0)
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
        xh = tl.load(XH + row[None, :] * XH_STRIDE + kt * 16 + xh_off[:, None],
                     mask=rm[None, :], other=0.0).to(tl.float32)   # [16, G]
        acc += tl.sum(tl.reshape(val, (8, 16, 16, 1)) * xh[None, :, None, :], axis=1)
    for g in tl.static_range(G):
        yp = tl.sum(acc * (ga[None, None, :] == g), axis=2)
        rw = mg * G + g
        tl.store(P_OUT + (rw * KS + ks) * NB * 16 + nb * 16 + t8[:, None] * 16 + xh_off[None, :],
                 yp, mask=nv[:, None] & (rw < M))


def gemv_rot(xh: torch.Tensor, t: torch.Tensor, svh: torch.Tensor,
             out_fp32: bool) -> torch.Tensor:
    """`gemv` + `rot_out` in the combine launch when K is split (the `out` fp32 round-trip
    disappears); falls back to gemv + rot_out for the single-program path. Bit-identical to
    `rot_out(gemv(xh, t), svh, out_fp32)`."""
    from tools.exl3_linear import kb_of
    KT, NTs, W = t.shape
    M, K = xh.shape
    kb = kb_of(W)
    tw = t.view(torch.int32) if t.stride(2) == 1 and t.data_ptr() % 4 == 0 \
        else t.contiguous().view(torch.int32)
    kstr = tw.stride(0)
    n = NTs * 16
    nb8 = (NTs + 7) // 8
    ks = min(max(KT // 16, (8 * 304) // max(nb8, 1)), 32)
    out = torch.empty(M, n, dtype=torch.float32 if out_fp32 else torch.bfloat16,
                      device=xh.device)
    if ks > 1:
        part = torch.empty(M, ks, n, dtype=torch.float32, device=xh.device)
        if M > 1:
            G = 8 if M >= 8 else (4 if M >= 4 else 2)
            mg = (M + G - 1) // G
            if W // 2 > 32:
                _exl3_gemv_split_direct_g_kernel[(mg, nb8, ks)](
                    xh, tw, part, _end_slot(kb, xh.device), NTs, kstr, NTs, K, KT, ks, M,
                    W2=W // 2, K_INV=_k_inv(), K_BIAS=_k_bias(), G=G,
                    num_warps=1, num_stages=1)
            else:
                _exl3_gemv_split_g_kernel[(mg, nb8, ks)](
                    xh, tw, part, _end_slot(kb, xh.device), NTs, kstr, NTs, K, KT, ks, M,
                    W2=W // 2, W2P=triton.next_power_of_2(W // 2),
                    K_INV=_k_inv(), K_BIAS=_k_bias(), G=G,
                    num_warps=1, num_stages=4)
        elif W // 2 > 32:
            _exl3_gemv_split_direct_kernel[(M, nb8, ks)](
                xh, tw, part, _end_slot(kb, xh.device), NTs, kstr, NTs, K, KT, ks,
                W2=W // 2, K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=1, num_stages=1)
        else:
            _exl3_gemv_split_kernel[(M, nb8, ks)](xh, tw, part, _end_slot(kb, xh.device),
                                                NTs, kstr, NTs, K, KT, ks,
                                                W2=W // 2, W2P=triton.next_power_of_2(W // 2),
                                                K_INV=_k_inv(), K_BIAS=_k_bias(),
                                                num_warps=1, num_stages=4)
        _exl3_gemv_combine_rot_kernel[(M, (n + 255) // 256)](
            part, svh, out, n, KS=ks, BF16=not out_fp32, num_warps=4)
    else:
        yp = gemv(xh, t)
        return rot_out(yp, svh, out_fp32)
    return out


def gemv(xh: torch.Tensor, t: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """xh [M, K] fp16 @ W[K, n]^T decoded from trellis slice t [KT, n/16, W] int16 -> [M, n] fp32,
    the reference's products up to summation order (every row's own order is fixed)."""
    from tools.exl3_linear import kb_of
    KT, NTs, W = t.shape
    M, K = xh.shape
    assert K == KT * 16, (K, KT * 16)
    kb = kb_of(W)
    # read the trellis (possibly a column slice) in place: the kt row stride carries the copy
    # a `.contiguous()` would otherwise make for a sliced view
    tw = t.view(torch.int32) if t.stride(2) == 1 and t.data_ptr() % 4 == 0 \
        else t.contiguous().view(torch.int32)
    kstr = tw.stride(0)                                  # int32 a kt row
    n = NTs * 16
    if out is None:
        out = torch.empty(M, n, dtype=torch.float32, device=xh.device)
    # split K across programs until ~4 waves of 76-SM worth of tiles exist; a function of the
    # trellis's shape only, so the summation order (and therefore the bits) never depends on M
    nb8 = (NTs + 7) // 8
    ks = min(max(KT // 16, (8 * 304) // max(nb8, 1)), 32)
    if ks > 1:
        part = torch.empty(M, ks, n, dtype=torch.float32, device=xh.device)
        if M > 1:
            G = 8 if M >= 8 else (4 if M >= 4 else 2)
            mg = (M + G - 1) // G
            if W // 2 > 32:
                _exl3_gemv_split_direct_g_kernel[(mg, nb8, ks)](
                    xh, tw, part, _end_slot(kb, xh.device), NTs, kstr, NTs, K, KT, ks, M,
                    W2=W // 2, K_INV=_k_inv(), K_BIAS=_k_bias(), G=G,
                    num_warps=1, num_stages=1)
            else:
                _exl3_gemv_split_g_kernel[(mg, nb8, ks)](
                    xh, tw, part, _end_slot(kb, xh.device), NTs, kstr, NTs, K, KT, ks, M,
                    W2=W // 2, W2P=triton.next_power_of_2(W // 2),
                    K_INV=_k_inv(), K_BIAS=_k_bias(), G=G,
                    num_warps=1, num_stages=4)
        elif W // 2 > 32:
            _exl3_gemv_split_direct_kernel[(M, nb8, ks)](
                xh, tw, part, _end_slot(kb, xh.device), NTs, kstr, NTs, K, KT, ks,
                W2=W // 2, K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=1, num_stages=1)
        else:
            _exl3_gemv_split_kernel[(M, nb8, ks)](xh, tw, part, _end_slot(kb, xh.device),
                                                NTs, kstr, NTs, K, KT, ks,
                                                W2=W // 2, W2P=triton.next_power_of_2(W // 2),
                                                K_INV=_k_inv(), K_BIAS=_k_bias(),
                                                num_warps=1, num_stages=4)
        _exl3_gemv_combine_kernel[(M, (n + 255) // 256)](part, out, n, KS=ks, num_warps=4)
    else:
        _exl3_gemv_kernel[(M, NTs)](xh, tw, out, _end_slot(kb, xh.device),
                                    kstr, NTs, K, n, KT,
                                    W2=W // 2, W2P=triton.next_power_of_2(W // 2),
                                    K_INV=_k_inv(), K_BIAS=_k_bias(), num_warps=4)
    return out


_GEMV_OK: dict = {}


def try_gemv(xh: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
    """`gemv` gated on a one-time check against the torch decode+product; None when the kernel is
    unavailable or disagrees (the caller takes the reference path). The probe is per rate: a
    kernel good at one bit width is not trusted at another."""
    if not xh.is_cuda:
        return None
    from tools.exl3_linear import kb_of
    kb = kb_of(t.shape[2])
    ok = _GEMV_OK.get(kb)
    if ok is None:
        try:
            from tools.exl3_linear import decode_trellis
            ref = xh[:1].float() @ decode_trellis(t).float()
            ok = bool(torch.allclose(ref, gemv(xh[:1].contiguous(), t), atol=2e-3))
        except Exception:
            ok = False
        _GEMV_OK[kb] = ok
    if not ok:
        return None
    try:
        return gemv(xh, t)
    except Exception:
        _GEMV_OK[kb] = False
        return None


def try_gemv_rot(xh: torch.Tensor, t: torch.Tensor, svh: torch.Tensor,
                 out_fp32: bool) -> torch.Tensor | None:
    """`gemv_rot` gated on the same per-rate probe as `try_gemv` (the rotation half of the
    fused combine is the same elementwise arithmetic, so the gemv probe carries it)."""
    if not xh.is_cuda:
        return None
    from tools.exl3_linear import kb_of
    kb = kb_of(t.shape[2])
    ok = _GEMV_OK.get(kb)
    if ok is None:
        try:
            from tools.exl3_linear import decode_trellis
            ref = xh[:1].float() @ decode_trellis(t).float()
            ok = bool(torch.allclose(ref, gemv(xh[:1].contiguous(), t), atol=2e-3))
        except Exception:
            ok = False
        _GEMV_OK[kb] = ok
    if not ok:
        return None
    try:
        return gemv_rot(xh, t, svh, out_fp32)
    except Exception:
        _GEMV_OK[kb] = False
        return None


_KI = _KB_ = None


def _k_inv() -> float:
    global _KI
    if _KI is None:
        _KI = float(torch.tensor(0x1eee, dtype=torch.int16).view(torch.float16).float())
    return _KI


def _k_bias() -> float:
    global _KB_
    if _KB_ is None:
        _KB_ = float(torch.tensor(0xc931 - 0x10000, dtype=torch.int16).view(torch.float16).float())
    return _KB_


_OK: bool | None = None


def decode(t: torch.Tensor, nt0: int = 0, nt1: int | None = None) -> torch.Tensor:
    """The trellis decode the engine calls: `decode_trellis_triton` once a probe proves it bit for
    bit `decode_trellis`'s on this build (any failure -- compile, launch, a single bit -- and the
    torch reference runs instead, permanently)."""
    global _OK
    if not t.is_cuda:
        from tools.exl3_linear import decode_trellis
        return decode_trellis(t, nt0, nt1)
    if _OK is None:
        try:
            from tools.exl3_linear import decode_trellis
            _OK = bool(torch.equal(decode_trellis_triton(t, nt0, nt1), decode_trellis(t, nt0, nt1)))
        except Exception:
            _OK = False
    if _OK:
        try:
            return decode_trellis_triton(t, nt0, nt1)
        except Exception:
            _OK = False
    from tools.exl3_linear import decode_trellis
    return decode_trellis(t, nt0, nt1)


def decode_trellis_triton(t: torch.Tensor, nt0: int = 0, nt1: int | None = None) -> torch.Tensor:
    """`decode_trellis` on the GPU: trellis [KT, NT, W] int16 -> Wi fp16 [KT*16, (nt1-nt0)*16]."""
    from tools.exl3_linear import _tile_perm_dev, kb_of
    KT, NT, W = t.shape
    kb = kb_of(W)
    nt1 = NT if nt1 is None else nt1
    n = nt1 - nt0
    W2 = W // 2
    ts = t[:, nt0:nt1]
    tw = ts.view(torch.int32) if ts.stride(2) == 1 and ts.data_ptr() % 4 == 0 \
        else ts.contiguous().view(torch.int32)            # the kt stride reads a slice in place
    out = torch.empty(KT * 16, n * 16, dtype=torch.float16, device=t.device)
    k_inv = float(torch.tensor(0x1eee, dtype=torch.int16).view(torch.float16).float())
    k_bias = float(torch.tensor(0xc931 - 0x10000, dtype=torch.int16).view(torch.float16).float())
    perm = _tile_perm_dev(t.device)
    if perm.dtype != torch.int32:
        perm = perm.to(torch.int32)
    TILES_N = 4
    grid = (KT, (n + TILES_N - 1) // TILES_N)
    _exl3_decode_kernel[grid](tw, out, perm,
                              tw.stride(0) // W2, n, n * 16,
                              W2=W2, KB=kb, HALF=kb >= 10, TILES_N=TILES_N,
                              K_INV=k_inv, K_BIAS=k_bias, num_warps=4)
    return out


def rot_in(x: torch.Tensor, suh: torch.Tensor) -> torch.Tensor:
    """xh = fp16(H(fp32(x) * fp32(suh)) / sqrt(128)) for every row of x, one launch; each
    row's rotation is a program's, so a row's bits do not depend on M."""
    if x.stride(1) != 1:
        x = x.contiguous()
    M, K = x.shape
    xh = torch.empty(M, K, dtype=torch.float16, device=x.device)
    _exl3_rot_in_kernel[(M, K // 128)](x, suh, xh, x.stride(0), K, num_warps=1)
    return xh




def rot_in_silu(ygu: torch.Tensor, suh: torch.Tensor) -> torch.Tensor:
    """xh of a [gate | up] bf16 buffer: silu, the bf16 product and the input rotation in one
    launch, each row's bits a program's (row-invariant)."""
    M, N2 = ygu.shape
    I = N2 // 2
    if ygu.stride(1) != 1:
        ygu = ygu.contiguous()
    xh = torch.empty(M, I, dtype=torch.float16, device=ygu.device)
    _exl3_rot_in_silu_kernel[(M, I // 128)](ygu, suh, xh, ygu.stride(0), I, num_warps=1)
    return xh


def rot_out(yp: torch.Tensor, svh: torch.Tensor, out_fp32: bool) -> torch.Tensor:
    """y = H(yp) / sqrt(128) * fp32(svh), bf16 or fp32 out, one launch for every row."""
    M, N = yp.shape
    if yp.stride(1) != 1 or yp.stride(0) != N:
        yp = yp.contiguous()
    y = torch.empty(M, N, dtype=torch.float32 if out_fp32 else torch.bfloat16, device=yp.device)
    _exl3_rot_out_kernel[(M, N // 128)](yp, svh, y, N, BF16=not out_fp32, num_warps=1)
    return y


_ROT_OK: bool | None = None


def rot_ok(device) -> bool:
    """One-time probe: rot_in/rot_out against the torch rotation they replace; False
    permanently on any compile or value failure (the caller keeps the torch path)."""
    global _ROT_OK
    if _ROT_OK is None:
        try:
            from tools.exl3_linear import hadamard128
            H = hadamard128(device)
            x = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
            suh = torch.randn(256, dtype=torch.float16, device=device)
            ref = ((x.float() * suh.float()).view(4, 2, 128) @ H).view(4, 256).to(torch.float16)
            yp = torch.randn(4, 256, dtype=torch.float32, device=device)
            svh = torch.randn(256, dtype=torch.float16, device=device)
            refy = (yp.view(4, 2, 128) @ H).view(4, 256) * svh.float()
            _ROT_OK = bool(torch.allclose(ref.float(), rot_in(x, suh).float(), atol=1e-3)
                           and torch.allclose(refy, rot_out(yp, svh, True), atol=1e-3))
        except Exception:
            _ROT_OK = False
    return _ROT_OK
