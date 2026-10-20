"""GGUF (ggml) quantization coverage for the NemotronH GGUF path, beyond the file's original
Q8_0 / IQ2_XXS / NVFP4 set. Two tiers, chosen per tensor by `gguf_load`:

- packed GPU linears with a fused Triton dequant+dot for the hot formats (Q4_0, Q4_K, Q6_K —
  the bulk of published GGUF weights): the same contract as `iq2_linear.Q8Linear` — row-invariant
  at any M (verify-safe), a cuBLAS twin for M > ROWS_MAX.
- `_dequant` numpy decoders for every other ggml type, so a mixed file can always load: the
  caller gets a dense bf16 tensor and the engine's ordinary dense path applies (one kernel per
  type found in a release becomes "one decoder per type in the file, packed where it pays").

The block layouts and decode order are ggml's (llama.cpp `ggml-quants.c`, MIT), cross-checked
against gguf-py's `quants.py` (MIT); the IQ lattice grids live in `tools/gguf_grids.py`. NVFP4
and Q8_0 keep their existing dedicated modules. Q1_0 (type 41) has no settled ggml decode and
no known checkpoints — it stays a clean refusal.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

ROWS_MAX = 32   # iq2_linear's contract: rows up to this use the packed kernel (row-invariant)
ROWS_KERN = ROWS_MAX + 1          # rows up to this still run the packed kernel (the verify block)
PREFILL_W_BYTES = 256 << 20       # a prefill matmul decodes the weight to bf16 a slice this big


# ============================================================ numpy decoders (the Tier-B path and twins)
# Each takes packed blocks uint8 [n, type_size] and returns fp32 [n, block_values], in ggml's
# element order — the same transforms gguf-py `dequantize_blocks` performs.

def _f16(b):
    return b.view(np.float16).astype(np.float32)


def _nibbles(qs, width):
    """Two nibbles a byte, lo first: [n, w] -> [n, 2w]."""
    return np.concatenate([qs & np.uint8(0x0F), qs >> np.uint8(4)], axis=-1)


def _p8(codes):
    """codes [N, nb, 32] u8 -> plane-major words [N, G, 32] int32: word j of group g packs the
    4-bit codes of element j of blocks 8g..8g+7 at bits 4i (blocks zero-padded to a multiple
    of 8), so one loop step reads one contiguous 128-byte line a row."""
    N, nb, _ = codes.shape
    nbp = (nb + 7) // 8 * 8
    if nbp != nb:
        codes = np.pad(codes, ((0, 0), (0, nbp - nb), (0, 0)))
    c = codes.reshape(N, -1, 8, 32).astype(np.uint32)
    return (c << (4 * np.arange(8, dtype=np.uint32))[None, :, None]).sum(2).astype(np.uint32).view(np.int32)


def _padn(a, nbp):
    """Pad a [N, nb] row-aligned array with zeros out to nbp columns."""
    return np.pad(a, ((0, 0), (0, nbp - a.shape[1]))) if a.shape[1] != nbp else a


def dq_f32(b):  return b.view(np.float32)
def dq_f16(b):  return _f16(b)
def dq_bf16(b): return (b.view(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def dq_i8(b):  return b.view(np.int8).astype(np.float32)
def dq_i16(b): return b.view(np.int16).astype(np.float32)
def dq_i32(b): return b.view(np.int32).astype(np.float32)
def dq_i64(b): return b.view(np.int64).astype(np.float32)
def dq_f64(b): return b.view(np.float64).astype(np.float32)


def dq_q4_0(b):
    n = b.shape[0]
    d, qs = np.hsplit(b, [2])
    q = _nibbles(qs, 16).astype(np.int8) - np.int8(8)
    return _f16(d) * q.astype(np.float32)


def dq_q4_1(b):
    d, rest = np.hsplit(b, [2])
    m, qs = np.hsplit(rest, [2])
    return _f16(d) * _nibbles(qs, 16).astype(np.float32) + _f16(m)


def dq_q5_0(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qh, qs = np.hsplit(rest, [4])
    bits = (qh.view(np.uint32).reshape(n, 1) >> np.arange(32, dtype=np.uint32)[None, :]) & np.uint32(1)
    q = (_nibbles(qs, 16) | bits.astype(np.uint8) << np.uint8(4)).astype(np.int8) - np.int8(16)
    return _f16(d) * q.astype(np.float32)


def dq_q5_1(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    m, rest = np.hsplit(rest, [2])
    qh, qs = np.hsplit(rest, [4])
    bits = (qh.view(np.uint32).reshape(n, 1) >> np.arange(32, dtype=np.uint32)[None, :]) & np.uint32(1)
    q = (_nibbles(qs, 16) | bits.astype(np.uint8) << np.uint8(4)).astype(np.float32)
    return _f16(d) * q + _f16(m)


def dq_q8_0(b):
    d, q = np.hsplit(b, [2])
    return _f16(d) * q.view(np.int8).astype(np.float32)


def dq_q8_1(b):
    # block_q8_1: f32 pair (d, s); s = d * sum(q) is a dot-product auxiliary, not part of the weight.
    ds, q = np.hsplit(b, [8])
    return ds[:, :4].view(np.float32) * q.view(np.int8).astype(np.float32)


def _q4k_scales(sc12):
    """Q4_K/Q5_K's 12-byte (6-bit) scale+min pack -> (sc[n,8], m[n,8]) uint8."""
    n = sc12.shape[0]
    s = sc12.reshape(n, 3, 4)
    d_, m_, md = np.split(s, 3, axis=-2)
    sc = np.concatenate([d_ & np.uint8(0x3F), (md & np.uint8(0x0F)) | ((d_ >> np.uint8(2)) & np.uint8(0x30))], -1)
    mn = np.concatenate([m_ & np.uint8(0x3F), (md >> np.uint8(4)) | ((m_ >> np.uint8(2)) & np.uint8(0x30))], -1)
    return sc.reshape(n, 8), mn.reshape(n, 8)


def dq_q2_k(b):
    n = b.shape[0]
    sc, rest = np.hsplit(b, [16])
    qs, rest = np.hsplit(rest, [64])
    d, dmin = np.hsplit(rest, [2])
    dl = (_f16(d) * (sc & np.uint8(0x0F)).astype(np.float32))[..., None]      # [n,16,1]
    ml = (_f16(dmin) * (sc >> np.uint8(4)).astype(np.float32))[..., None]
    q = (qs.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & np.uint8(3)
    return (dl * q.reshape(n, 16, 16).astype(np.float32) - ml).reshape(n, -1)


def dq_q3_k(b):
    n = b.shape[0]
    hmask, rest = np.hsplit(b, [32])
    qs, rest = np.hsplit(rest, [64])
    sc, d = np.hsplit(rest, [12])
    ls, hs = np.hsplit(sc, [8])
    ls = (ls.reshape(n, 1, 8) >> np.array([0, 4], np.uint8)[None, :, None]).reshape(n, 16)
    hs = (hs.reshape(n, 1, 4) >> np.array([0, 2, 4, 6], np.uint8)[None, :, None]).reshape(n, 16)
    sc16 = ((ls & np.uint8(0x0F)) | ((hs & np.uint8(0x03)) << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = (_f16(d) * sc16.astype(np.float32)).reshape(n, 16, 1)
    ql = (qs.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & np.uint8(3)
    qh = (hmask.reshape(n, -1, 1, 32) >> np.arange(8, dtype=np.uint8)[None, None, :, None]) & np.uint8(1)
    q = (ql.reshape(n, 16, 16).astype(np.int8) - ((qh.reshape(n, 16, 16) ^ np.uint8(1)) << np.uint8(2)).astype(np.int8))
    return (dl * q.astype(np.float32)).reshape(n, -1)


def dq_q4_k(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    dmin, rest = np.hsplit(rest, [2])
    sc12, qs = np.hsplit(rest, [12])
    sc, m = _q4k_scales(sc12)
    dl = (_f16(d) * sc.astype(np.float32)).reshape(n, 8, 1)
    ml = (_f16(dmin) * m.astype(np.float32)).reshape(n, 8, 1)
    q = (qs.reshape(n, -1, 1, 32) >> np.array([0, 4], np.uint8)[None, None, :, None]) & np.uint8(0x0F)
    return (dl * q.reshape(n, 8, 32).astype(np.float32) - ml).reshape(n, -1)


def dq_q5_k(b):
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    dmin, rest = np.hsplit(rest, [2])
    sc12, rest = np.hsplit(rest, [12])
    qh, qs = np.hsplit(rest, [32])
    sc, m = _q4k_scales(sc12)
    dl = (_f16(d) * sc.astype(np.float32)).reshape(n, 8, 1)
    ml = (_f16(dmin) * m.astype(np.float32)).reshape(n, 8, 1)
    ql = (qs.reshape(n, -1, 1, 32) >> np.array([0, 4], np.uint8)[None, None, :, None]) & np.uint8(0x0F)
    hb = (qh.reshape(n, -1, 1, 32) >> np.arange(8, dtype=np.uint8)[None, None, :, None]) & np.uint8(1)
    q = (ql.reshape(n, 8, 32) | (hb.reshape(n, 8, 32) << np.uint8(4)))
    return (dl * q.astype(np.float32) - ml).reshape(n, -1)


def dq_q6_k(b):
    n = b.shape[0]
    ql, rest = np.hsplit(b, [128])
    qh, rest = np.hsplit(rest, [64])
    sc, d = np.hsplit(rest, [16])
    dl = (_f16(d) * sc.view(np.int8).astype(np.float32)).reshape(n, 16, 1)
    lo = (ql.reshape(n, -1, 1, 64) >> np.array([0, 4], np.uint8)[None, None, :, None]) & np.uint8(0x0F)
    hb = (qh.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & np.uint8(3)
    q = (lo.reshape(n, -1, 32) | (hb.reshape(n, -1, 32) << np.uint8(4))).astype(np.int8) - np.int8(32)
    return (dl * q.reshape(n, 16, 16).astype(np.float32)).reshape(n, -1)


def dq_q8_k(b):
    # block_q8_K: f32 d, int8 qs[256], int16 bsums[16] (dot aux; unused by the dequant)
    d, rest = np.hsplit(b, [4])
    return d.view(np.float32) * rest[:, :256].view(np.int8).astype(np.float32)


def _ksigns_apply(signs_idx, n):
    """7-bit sign index -> ±1 [n, groups, 4 or 8]: ksigns_iq2xs (bit j set = magnitude j negative)."""
    from tools.iq2_linear import KSIGNS
    s = KSIGNS[signs_idx.astype(np.int64)]
    bits = (s.reshape(n, -1, 1) >> np.arange(8, dtype=np.uint8)[None, None, :]) & np.uint8(1)
    return np.where(bits == 0, np.float32(1), np.float32(-1))


def dq_iq2_xxs(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, qs = np.hsplit(b, [2])
    qs = qs.view(np.uint32).reshape(n, -1, 2)
    db = (_f16(d) * (0.5 + (qs[..., 1] >> np.uint32(28)).astype(np.float32)) * 0.25)[..., None, None]
    si = ((qs[..., 1].reshape(n, -1, 1) >> np.array([0, 7, 14, 21], np.uint32)[None, None, :]) & np.uint32(0x7F))
    signs = _ksigns_apply(si, n).reshape(n, -1, 4, 8)
    grid = GR.IQ2_XXS_GRID.reshape(256, 8)[qs[..., 0].copy().view(np.uint8).reshape(n, -1)]
    grid = grid.reshape(n, -1, 4, 8)
    return (db * grid * signs).reshape(n, -1)


def dq_iq2_xs(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qs, sc = np.hsplit(rest, [64])
    qs = qs.view(np.uint16)
    s = ((sc.reshape(n, -1, 1) >> np.array([0, 4], np.uint8)[None, None, :]) & np.uint8(0x0F)).reshape(n, -1)
    db = (_f16(d) * (0.5 + s.astype(np.float32)) * 0.25)[..., None, None]
    signs = _ksigns_apply((qs >> np.uint16(9)).astype(np.int64).reshape(n, -1), n)
    grid = GR.IQ2_XS_GRID.reshape(512, 8)[(qs & np.uint16(511)).reshape(n, -1)].reshape(n, -1, 2, 8)
    return (db * grid * signs.reshape(n, -1, 2, 8)).reshape(n, -1)


def dq_iq2_s(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qs, rest = np.hsplit(rest, [32])
    sg, rest = np.hsplit(rest, [32])
    qh, sc = np.hsplit(rest, [8])
    s = ((sc.reshape(n, -1, 1) >> np.array([0, 4], np.uint8)[None, None, :]) & np.uint8(0x0F)).reshape(n, -1)
    db = (_f16(d) * (0.5 + s.astype(np.float32)) * 0.25)[..., None, None]
    signs = ((sg.reshape(n, -1, 1) >> np.arange(8, dtype=np.uint8)[None, None, :]) & np.uint8(1))
    signs = np.where(signs == 0, np.float32(1), np.float32(-1)).reshape(n, -1, 2, 8)
    hb = (qh.reshape(n, -1, 1) >> np.array([0, 2, 4, 6], np.uint8)[None, None, :]) & np.uint8(3)
    idx = qs.astype(np.uint16) | (hb.reshape(n, -1) << np.uint16(8))
    grid = GR.IQ2_S_GRID.reshape(1024, 8)[idx].reshape(n, -1, 2, 8)
    return (db * grid * signs).reshape(n, -1)


def dq_iq3_xxs(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qs, sc = np.hsplit(rest, [64])
    sc = sc.view(np.uint32)
    db = (_f16(d) * (0.5 + (sc >> np.uint32(28)).astype(np.float32)) * 0.5)[..., None, None]
    si = ((sc.reshape(n, -1, 1) >> np.array([0, 7, 14, 21], np.uint32)[None, None, :]) & np.uint32(0x7F))
    signs = _ksigns_apply(si, n).reshape(n, -1, 4, 8)
    # each qs byte addresses a 4-magnitude grid entry; two entries per 8-elem group
    grid = GR.IQ3_XXS_GRID.reshape(256, 4)[qs.reshape(n, -1)].reshape(n, -1, 2, 4)
    grid = grid.reshape(n, -1, 4, 8)
    return (db * grid * signs).reshape(n, -1)


def dq_iq3_s(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qs, rest = np.hsplit(rest, [64])
    qh, rest = np.hsplit(rest, [8])
    sg, sc = np.hsplit(rest, [32])
    s = ((sc.reshape(n, -1, 1) >> np.array([0, 4], np.uint8)[None, None, :]) & np.uint8(0x0F)).reshape(n, -1)
    db = (_f16(d) * (1 + 2 * s.astype(np.float32)))[..., None, None]
    signs = ((sg.reshape(n, -1, 1) >> np.arange(8, dtype=np.uint8)[None, None, :]) & np.uint8(1))
    signs = np.where(signs == 0, np.float32(1), np.float32(-1)).reshape(n, -1, 4, 8)
    hb = (qh.reshape(n, -1, 1) >> np.arange(8, dtype=np.uint8)[None, None, :]) & np.uint8(1)
    idx = qs.astype(np.uint16) | (hb.reshape(n, -1) << np.uint16(8))
    grid = GR.IQ3_S_GRID.reshape(512, 4)[idx].reshape(n, -1, 2, 4)
    grid = grid.reshape(n, -1, 4, 8)
    return (db * grid * signs).reshape(n, -1)


def dq_iq1_s(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    qs, qh = np.hsplit(rest, [32])
    qh = qh.view(np.uint16)
    dl = (_f16(d) * (2 * ((qh >> np.uint16(12)) & np.uint16(7)).astype(np.float32) + 1)).reshape(n, -1, 1, 1)
    delta = np.where((qh & np.uint16(0x8000)) == 0, GR.IQ1_DELTA, -GR.IQ1_DELTA).reshape(n, -1, 1, 1)
    hb = (qh.reshape(n, -1, 1) >> np.array([0, 3, 6, 9], np.uint16)[None, None, :]) & np.uint16(7)
    idx = qs.astype(np.uint16) | (hb.reshape(n, -1) << np.uint16(8))
    grid = GR.IQ1_GRID.reshape(2048, 8)[idx].reshape(n, -1, 4, 8)
    return (dl * (grid + delta)).reshape(n, -1)


def dq_iq1_m(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    qs, rest = np.hsplit(b, [32])
    qh, sc = np.hsplit(rest, [16])
    scw = sc.view(np.uint16).reshape(n, 4)
    d = ((scw & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], np.uint16)[None, :])
    d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
    d = d.view(np.uint16).view(np.float16).astype(np.float32).reshape(n, 1)
    sl = (sc.view(np.uint16).reshape(n, -1, 1) >> np.array([0, 3, 6, 9], np.uint16)[None, None, :]) & np.uint16(7)
    dl = (d * (2 * sl.reshape(n, -1).astype(np.float32) + 1)).reshape(n, -1, 2, 1, 1)
    hb = (qh.reshape(n, -1, 1) >> np.array([0, 4], np.uint8)[None, None, :])
    idx = qs.astype(np.uint16) | ((hb & np.uint8(7)).astype(np.uint16) << np.uint16(8)).reshape(n, -1)
    delta = np.where((hb & np.uint8(0x08)) == 0, GR.IQ1_DELTA, -GR.IQ1_DELTA)
    grid = GR.IQ1_GRID.reshape(2048, 8)[idx].reshape(n, -1, 2, 2, 8)
    return (dl * (grid + delta.reshape(n, -1, 2, 2, 1))).reshape(n, -1)


def dq_iq4_nl(b):
    from tools import gguf_grids as GR
    d, qs = np.hsplit(b, [2])
    return _f16(d) * GR.IQ4_NL_VALUES[_nibbles(qs, 16)].astype(np.float32)


def dq_iq4_xs(b):
    from tools import gguf_grids as GR
    n = b.shape[0]
    d, rest = np.hsplit(b, [2])
    sh, rest = np.hsplit(rest, [2])
    sl, qs = np.hsplit(rest, [4])
    sl = ((sl.reshape(n, -1, 1) >> np.array([0, 4], np.uint8)[None, None, :]) & np.uint8(0x0F)).reshape(n, -1)
    sh = ((sh.view(np.uint16).reshape(n, 1, -1) >> (2 * np.arange(8, dtype=np.uint16))[None, :, None]))
    sc = (sl | ((sh.reshape(n, -1).astype(np.uint8) & np.uint8(3)) << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = (_f16(d) * sc.astype(np.float32)).reshape(n, -1, 1)
    q = _nibbles(qs.reshape(n, 8, 16), 16).reshape(n, 8, 32)
    return (dl * GR.IQ4_NL_VALUES[q].astype(np.float32)).reshape(n, -1)


def dq_tq1_0(b):
    # packed base-3 digits: each byte carries 5 values in a /243 fraction (ggml tq1_0)
    n = b.shape[0]
    qs, rest = np.hsplit(b, [48])
    qh, d = np.hsplit(rest, [4])
    w = np.array([1, 3, 9, 27, 81], np.uint8)
    qs0 = (qs[:, :32].reshape(n, -1, 1, 32) * w[None, None, :, None]).reshape(n, -1)
    qs1 = (qs[:, 32:].reshape(n, -1, 1, 16) * w[None, None, :, None]).reshape(n, -1)
    qh_ = (qh.reshape(n, -1, 1, 4) * w[:4][None, None, :, None]).reshape(n, -1)
    q = np.concatenate([qs0, qs1, qh_], -1)
    q = ((q.astype(np.uint16) * np.uint16(3)) >> np.uint16(8)).astype(np.int8) - np.int8(1)
    return _f16(d) * q.astype(np.float32)


def dq_tq2_0(b):
    qs, d = np.hsplit(b, [64])
    q = ((qs.reshape(b.shape[0], -1, 1, 32) >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & np.uint8(3))
    return _f16(d) * (q.reshape(b.shape[0], -1).astype(np.int8) - np.int8(1)).astype(np.float32)


def dq_mxfp4(b):
    from tools import gguf_grids as GR
    e, qs = np.hsplit(b, [1])
    e = e.reshape(-1)
    bits = np.where(e < np.uint8(2), np.uint32(0x00200000) << e.astype(np.uint32),
                    (e.astype(np.uint32) - np.uint32(1)) << np.uint32(23))
    d = bits.view(np.float32).reshape(-1, 1)
    q = _nibbles(qs, 16).view(np.int8)
    return d * GR.MXFP4_VALUES[q.astype(np.int64)]


def dq_nvfp4(b, scale: float = 1.0):
    from tools.iq2_linear import dequant_nvfp4
    n = b.shape[0]
    w = dequant_nvfp4(b.reshape(-1), n, 64)  # returns [n, 64] fp32
    return w.numpy() * scale


#: ggml type id -> (values a block, bytes a block, decoder). Decoder = packed-uint8 [n, bytes]
#: -> fp32 [n, values]; the few without one stay refusals (see module docstring).
TYPES = {
    0:  (1, 4, lambda b: dq_f32(b).reshape(b.shape[0], -1)),
    1:  (1, 2, dq_f16),
    2:  (32, 18, dq_q4_0),
    3:  (32, 20, dq_q4_1),
    6:  (32, 22, dq_q5_0),
    7:  (32, 24, dq_q5_1),
    8:  (32, 34, dq_q8_0),
    9:  (32, 40, dq_q8_1),
    10: (256, 84, dq_q2_k),
    11: (256, 110, dq_q3_k),
    12: (256, 144, dq_q4_k),
    13: (256, 176, dq_q5_k),
    14: (256, 210, dq_q6_k),
    15: (256, 292, dq_q8_k),
    16: (256, 66, dq_iq2_xxs),
    17: (256, 74, dq_iq2_xs),
    18: (256, 98, dq_iq3_xxs),
    19: (256, 50, dq_iq1_s),
    20: (32, 18, dq_iq4_nl),
    21: (256, 110, dq_iq3_s),
    22: (256, 82, dq_iq2_s),
    23: (256, 136, dq_iq4_xs),
    24: (1, 1, dq_i8),
    25: (1, 2, dq_i16),
    26: (1, 4, dq_i32),
    27: (1, 8, dq_i64),
    28: (1, 8, dq_f64),
    29: (256, 56, dq_iq1_m),
    30: (1, 2, dq_bf16),
    34: (256, 54, dq_tq1_0),
    35: (256, 66, dq_tq2_0),
    39: (32, 17, dq_mxfp4),
    40: (64, 36, dq_nvfp4),
}


def dequant(tid: int, raw: np.ndarray, shape) -> torch.Tensor:
    """Any supported ggml type's packed `raw` -> fp32 tensor of `shape` (the dequant twin used by the
    dense path and by the packed linears' large-M fallback)."""
    qk, ts, fn = TYPES[tid]
    n_elem = int(np.prod(shape))
    if n_elem % qk or raw.size * 1.0 != n_elem * ts / qk:
        raise ValueError(f"{n_elem} values do not fill whole blocks of {qk} for type {tid} ({raw.size} B)")
    w = fn(np.ascontiguousarray(raw).reshape(-1, ts)).reshape(-1)
    return torch.from_numpy(np.ascontiguousarray(w.astype(np.float32))).reshape(shape)


if HAVE_TRITON:

    @triton.jit
    def _q40_gemm(X, s_x, D, QP, Y, s_y, M, N, NG, BN: tl.constexpr, F32: tl.constexpr):
        # Q4_0, plane-major: each group of 8 blocks is 32 int32 words a row (one 128 B line); word j
        # packs element j's 4-bit codes of blocks 8g..8g+7 at bits 4i. Scales zero-pad blocks
        # contribute nothing (x zero-padded by the caller past K). w = (q-8)*d decoded to bf16,
        # one K=256 tl.dot a group, fp32 accumulate, K in order -> same row-invariance contract.
        pm, pn = tl.program_id(0), tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        i8 = tl.arange(0, 8)
        e = tl.arange(0, 256)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for g in range(0, NG):
            w32 = tl.load(QP + (cols[:, None] * NG + g) * 32 + t[None, :],
                          mask=cm[:, None], other=0)
            c = (w32[:, None, :] >> (4 * i8)[None, :, None]) & 0xF
            d8 = tl.load(D + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            cb = ((c | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True).to(tl.float32)
            w = ((cb - 136.0) * d8[:, :, None]).to(tl.bfloat16)
            w = tl.reshape(w, (BN, 256))
            x = tl.load(X + rows[:, None] * s_x + g * 256 + e[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out,
                 mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _q4k_gemm(X, s_x, QS, DS, DM, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # Q4_K: 256 weights a super-block, eight 32-wide sub-blocks with a 6-bit scale and min each
        # (decoded once at load into DS/DM fp16 [N, nb, 8]). Same row-invariance contract as _q40_gemm.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            for s in tl.static_range(8):
                byte = (s // 2) * 32 + t
                q = tl.load(QS + (cols[:, None] * NB + kb) * 128 + byte[None, :], mask=cm[:, None], other=0)
                code = (q >> ((s % 2) * 4)) & 0xF
                ds = tl.load(DS + (cols * NB + kb) * 8 + s, mask=cm, other=0.0)
                dm = tl.load(DM + (cols * NB + kb) * 8 + s, mask=cm, other=0.0)
                # magic int->bf16: bf16(0x4300|c) == 128+c
                cb = ((code.to(tl.int32) | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True) - 128.0
                w = cb * ds.to(tl.bfloat16)[:, None] - dm.to(tl.bfloat16)[:, None]
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q6k_gemm(X, s_x, QL, QH, DL, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # Q6_K: 256 weights a super-block; ql[128] low nibbles, qh[64] high pairs, 16 int8 scales folded
        # into DL fp16 [N, nb, 16] at load. Sub s (32 elems, h = s//4 half, c = s%4 chunk):
        #   code = ((ql[64h + 32(c&1) + t] >> 4*(c>>1)) & 0xF) | (((qh[32h + t] >> 2c) & 3) << 4) - 32
        #   scale index s*2 + t//16 — same contract as _q40_gemm.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            for s in tl.static_range(8):
                h = s // 4
                c = s % 4
                ql = tl.load(QL + (cols[:, None] * NB + kb) * 128 + h * 64 + (c % 2) * 32 + t[None, :],
                             mask=cm[:, None], other=0)
                qh = tl.load(QH + (cols[:, None] * NB + kb) * 64 + h * 32 + t[None, :],
                             mask=cm[:, None], other=0)
                code = ((ql >> (4 * (c // 2))) & 0xF) | (((qh >> (2 * c)) & 3) << 4)
                base = (cols * NB + kb) * 16 + s * 2
                dl0 = tl.load(DL + base, mask=cm, other=0.0)
                dl1 = tl.load(DL + base + 1, mask=cm, other=0.0)
                # magic int->bf16: bf16(0x4300|c) == 128+c; -160 gives the signed c-32 code
                c6 = ((code | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
                dl = tl.where((t // 16)[None, :] == 1, dl1[:, None], dl0[:, None]).to(tl.bfloat16)
                w = (c6 - 160.0) * dl
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q41_gemm(X, s_x, D, M_, QP, Y, s_y, M, N, NG, BN: tl.constexpr, F32: tl.constexpr):
        # Q4_1: Q4_0's plane-major pack with Q4_1's min: w = q*d + m decoded to bf16, one dot a group.
        # Same layout and contract as _q40_gemm.
        pm, pn = tl.program_id(0), tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        i8 = tl.arange(0, 8)
        e = tl.arange(0, 256)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for g in range(0, NG):
            w32 = tl.load(QP + (cols[:, None] * NG + g) * 32 + t[None, :],
                          mask=cm[:, None], other=0)
            c = (w32[:, None, :] >> (4 * i8)[None, :, None]) & 0xF
            d8 = tl.load(D + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            m8 = tl.load(M_ + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            cb = ((c | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True).to(tl.float32)
            w = ((cb - 128.0) * d8[:, :, None] + m8[:, :, None]).to(tl.bfloat16)
            w = tl.reshape(w, (BN, 256))
            x = tl.load(X + rows[:, None] * s_x + g * 256 + e[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out,
                 mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _q50_gemm(X, s_x, D, QH, QP, Y, s_y, M, N, NG, BN: tl.constexpr, F32: tl.constexpr):
        # Q5_0: plane-major nibbles + a per-block u32 of fifth bits (bit j = element j, zero-padded
        # like the codes): w = (code-16)*d to bf16, one dot a group. Same contract as _q40_gemm.
        pm, pn = tl.program_id(0), tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        i8 = tl.arange(0, 8)
        e = tl.arange(0, 256)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for g in range(0, NG):
            w32 = tl.load(QP + (cols[:, None] * NG + g) * 32 + t[None, :],
                          mask=cm[:, None], other=0)
            c = (w32[:, None, :] >> (4 * i8)[None, :, None]) & 0xF
            qh8 = tl.load(QH + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0)
            c5 = c | ((qh8[:, :, None] >> t[None, None, :]) & 1) << 4
            d8 = tl.load(D + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            cb = ((c5 | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True).to(tl.float32)
            w = ((cb - 144.0) * d8[:, :, None]).to(tl.bfloat16)
            w = tl.reshape(w, (BN, 256))
            x = tl.load(X + rows[:, None] * s_x + g * 256 + e[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out,
                 mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _q51_gemm(X, s_x, D, M_, QH, QP, Y, s_y, M, N, NG, BN: tl.constexpr, F32: tl.constexpr):
        # Q5_1: Q5_0's codes with Q4_1's scale+min: w = code*d + m to bf16, one dot a group.
        # Same contract as _q40_gemm.
        pm, pn = tl.program_id(0), tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        i8 = tl.arange(0, 8)
        e = tl.arange(0, 256)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for g in range(0, NG):
            w32 = tl.load(QP + (cols[:, None] * NG + g) * 32 + t[None, :],
                          mask=cm[:, None], other=0)
            c = (w32[:, None, :] >> (4 * i8)[None, :, None]) & 0xF
            qh8 = tl.load(QH + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0)
            c5 = c | ((qh8[:, :, None] >> t[None, None, :]) & 1) << 4
            d8 = tl.load(D + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            m8 = tl.load(M_ + (cols[:, None] * NG + g) * 8 + i8[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            cb = ((c5 | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True).to(tl.float32)
            w = ((cb - 128.0) * d8[:, :, None] + m8[:, :, None]).to(tl.bfloat16)
            w = tl.reshape(w, (BN, 256))
            x = tl.load(X + rows[:, None] * s_x + g * 256 + e[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out,
                 mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _q2k_gemm(X, s_x, QS, DS, DM, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # Q2_K: 256 weights a super-block; qs[64] hold the 2-bit codes of sub s at byte
        # (s//4)*32 + t, shift 2*(s%4); scales are 16 4-bit pairs folded to DS/DM fp16 [N, nb, 16]
        # at load (index s*2 + t//16). Same contract as _q40_gemm.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            for s in tl.static_range(8):
                q = tl.load(QS + (cols[:, None] * NB + kb) * 64 + (s // 4) * 32 + t[None, :],
                            mask=cm[:, None], other=0)
                code = (q >> (2 * (s % 4))) & 3
                base = (cols * NB + kb) * 16 + s * 2
                ds0 = tl.load(DS + base, mask=cm, other=0.0)
                ds1 = tl.load(DS + base + 1, mask=cm, other=0.0)
                dm0 = tl.load(DM + base, mask=cm, other=0.0)
                dm1 = tl.load(DM + base + 1, mask=cm, other=0.0)
                hi = (t // 16)[None, :] == 1
                # magic int->bf16: bf16(0x4300|c) == 128+c
                cb = ((code.to(tl.int32) | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True) - 128.0
                w = (cb * tl.where(hi, ds1[:, None], ds0[:, None]).to(tl.bfloat16)
                     - tl.where(hi, dm1[:, None], dm0[:, None]).to(tl.bfloat16))
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q3k_gemm(X, s_x, QS, HM, DL, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # Q3_K: 2-bit codes like Q2_K plus one hmask bit per element (0 -> subtract 4);
        # 16 signed 6-bit scales folded to DL fp16 [N, nb, 16] at load. Same contract.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            for s in tl.static_range(8):
                q = tl.load(QS + (cols[:, None] * NB + kb) * 64 + (s // 4) * 32 + t[None, :],
                            mask=cm[:, None], other=0)
                hm = tl.load(HM + (cols[:, None] * NB + kb) * 32 + t[None, :], mask=cm[:, None], other=0)
                code = ((q >> (2 * (s % 4))) & 3).to(tl.int32)
                base = (cols * NB + kb) * 16 + s * 2
                dl0 = tl.load(DL + base, mask=cm, other=0.0)
                dl1 = tl.load(DL + base + 1, mask=cm, other=0.0)
                # magic int->bf16: bf16(0x4300|c) == 128+c; -(132-4*hbit) gives c - 4*(1-hbit)
                cb = ((code | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
                hb = ((hm >> s) & 1).to(tl.bfloat16)
                dl = tl.where((t // 16)[None, :] == 1, dl1[:, None], dl0[:, None]).to(tl.bfloat16)
                w = (cb - (132.0 - 4.0 * hb)) * dl
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q5k_gemm(X, s_x, QS, QH, DS, DM, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # Q5_K: Q4_K's nibbles + one high bit per element from qh[32] (bit s of byte t in sub s);
        # DS/DM fp16 [N, nb, 8] predecoded from the 6-bit pack at load. Same contract.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            for s in tl.static_range(8):
                byte = (s // 2) * 32 + t
                q = tl.load(QS + (cols[:, None] * NB + kb) * 128 + byte[None, :], mask=cm[:, None], other=0)
                hb = tl.load(QH + (cols[:, None] * NB + kb) * 32 + t[None, :], mask=cm[:, None], other=0)
                code = ((q >> ((s % 2) * 4)) & 0xF) | (((hb >> s) & 1) << 4)
                ds = tl.load(DS + (cols * NB + kb) * 8 + s, mask=cm, other=0.0).to(tl.float32)
                dm = tl.load(DM + (cols * NB + kb) * 8 + s, mask=cm, other=0.0).to(tl.float32)
                w = (code.to(tl.float32) * ds[:, None] - dm[:, None]).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _iq4nl_gemm(X, s_x, D, LUT, Q, Y, s_y, M, N, K, BN: tl.constexpr, SB: tl.constexpr,
                  F32: tl.constexpr):
        # IQ4_NL: Q4_0's layout (16 nibble bytes + fp16 d a 32-weight block) but the codes index a
        # fixed 16-entry value table instead of q-8. Same contract as _q40_gemm.
        # codes index the LUT (exact ints) -> w raw in the dot, d after: acc += d*part
        pm, pn = tl.program_id(0), tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        nb = K // 32
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, nb, SB):
            for j in tl.static_range(SB):
                b = kb + j
                if b < nb:
                    q = tl.load(Q + cols[:, None] * (nb * 16) + b * 16 + (t & 15)[None, :],
                                mask=cm[:, None], other=0)
                    nib = (q >> ((t // 16) * 4).to(tl.int32)[None, :]) & 0xF
                    w = tl.load(LUT + nib).to(tl.bfloat16)
                    d = tl.load(D + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                    x = tl.load(X + rows[:, None] * s_x + b * 32 + t[None, :],
                                mask=rows[:, None] < M, other=0.0)
                    acc += tl.dot(x, tl.trans(w)) * d[None, :]
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq4xs_gemm(X, s_x, QS, DL, LUT, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # IQ4_XS: 256 a super-block, Q4_K's nibble layout but the codes index the IQ4_NL value
        # table and each 32-elem sub-block has a signed 6-bit scale folded into DL fp16 [N, nb, 8]
        # at load (d * ((sl & 0xF | sh_2bit << 4) - 32)). Same contract as _q40_gemm.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            for s in tl.static_range(8):
                q = tl.load(QS + (cols[:, None] * NB + kb) * 128 + s * 16 + (t & 15)[None, :],
                            mask=cm[:, None], other=0)
                nib = (q >> ((t // 16) * 4).to(tl.int32)[None, :]) & 0xF
                v = tl.load(LUT + nib)
                dl = tl.load(DL + (cols * NB + kb) * 8 + s, mask=cm, other=0.0).to(tl.float32)
                w = (v * dl[:, None]).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq2xs_gemm(X, s_x, D, QS, SC, GR, KS, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # IQ2_XS: 256 a block; 32 u16 — low 9 bits index the 512-entry 8-elem grid, top 7 bits the
        # even-parity sign table; a 4-bit scale a 16-elem group. w = d*(0.5+s)*0.25*grid*sign.
        # The whole block is decoded to [BN, 256] and dotted once — the hot path's contract.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        brow = cols * NB
        t = tl.arange(0, 256)
        e = tl.arange(0, 32)
        j8 = tl.arange(0, 8)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            b = brow + kb
            lo = tl.load(QS + b[:, None] * 64 + 2 * e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            hi = tl.load(QS + b[:, None] * 64 + 2 * e[None, :] + 1, mask=cm[:, None], other=0).to(tl.int32)
            u = lo | (hi << 8)                                                            # [BN,32]
            ge = tl.load(GR + (u & 511))                                                        # [BN,32] i64
            gr = ((ge[:, :, None] >> (8 * j8[None, None, :])) & 0xFF).to(tl.float32)              # [BN,32,8]
            ks = tl.load(KS + (u >> 9)).to(tl.int32)                                      # [BN,32]
            neg = ((ks[:, :, None] >> j8[None, None, :]) & 1) == 1                          # [BN,32,8]
            scb = tl.load(SC + b[:, None] * 8 + (e // 4)[None, :], mask=cm[:, None], other=0).to(tl.int32)
            sc4 = (scb >> (4 * ((e // 2) % 2))[None, :]) & 0xF                              # [BN,32] group scale a entry pair
            d = tl.load(D + b, mask=cm, other=0.0).to(tl.float32)                           # [BN]
            db = (d[:, None] * (0.5 + sc4.to(tl.float32)) * 0.25)[:, :, None]               # [BN,32,1]
            w = tl.reshape(tl.where(neg, -1.0, 1.0) * gr * db, (BN, 256)).to(tl.bfloat16)
            x = tl.load(X + rows[:, None] * s_x + kb * 256 + t[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq3xxs_gemm(X, s_x, D, QS, SC, GR, KS, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # IQ3_XXS: 256 a block; 64 qs bytes index the 256-entry 4-elem grid; eight u32 sc words,
        # each four 7-bit sign indices and a 4-bit scale. w = d*(0.5+s)*0.5*grid*sign.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        brow = cols * NB
        t = tl.arange(0, 256)
        e = tl.arange(0, 64)
        l4 = tl.arange(0, 4)
        e4 = tl.arange(0, 4)
        g8 = tl.arange(0, 8)
        j8 = tl.arange(0, 8)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            b = brow + kb
            idx = tl.load(QS + b[:, None] * 64 + e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            ge = tl.load(GR + idx).to(tl.int64)                                                   # [BN,64] i32
            gr = ((ge[:, :, None] >> (8 * e4[None, None, :])) & 0xFF).to(tl.float32)              # [BN,64,4]
            scw = tl.load(SC + b[:, None] * 8 + g8[None, :], mask=cm[:, None], other=0)       # [BN,8] u32
            si = (scw[:, :, None] >> (7 * l4[None, None, :])) & 0x7F                          # [BN,8,4]
            ks = tl.load(KS + si).to(tl.int32)
            neg = ((ks[:, :, :, None] >> j8[None, None, None, :]) & 1) == 1                   # [BN,8,4,8]
            d = tl.load(D + b, mask=cm, other=0.0).to(tl.float32)
            db = (d[:, None] * (0.5 + ((scw >> 28) & 0xF).to(tl.float32)) * 0.5)[:, :, None]  # [BN,8,1]
            w = tl.reshape(tl.where(neg, -1.0, 1.0), (BN, 8, 32))
            w = (w * tl.reshape(gr, (BN, 8, 32)) * db)
            w = tl.reshape(w, (BN, 256)).to(tl.bfloat16)
            x = tl.load(X + rows[:, None] * s_x + kb * 256 + t[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq3s_gemm(X, s_x, D, QS, QH, SG, SC, GR, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # IQ3_S: 256 a block; 64 qs bytes | one qh bit each index the 512-entry 4-elem grid; 32 sg
        # bytes sign each weight; eight 4-bit group scales. w = d*(1+2s)*grid*sign.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        brow = cols * NB
        t = tl.arange(0, 256)
        e = tl.arange(0, 64)
        g8 = tl.arange(0, 8)
        e4 = tl.arange(0, 4)
        j8 = tl.arange(0, 8)
        b32 = tl.arange(0, 32)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            b = brow + kb
            lo = tl.load(QS + b[:, None] * 64 + e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            hmb = tl.load(QH + b[:, None] * 8 + (e // 8)[None, :], mask=cm[:, None], other=0).to(tl.int32)
            idx = lo | (((hmb >> (e % 8)[None, :]) & 1) << 8)                                 # [BN,64]
            ge = tl.load(GR + idx).to(tl.int64)                                                   # [BN,64] i32
            gr = ((ge[:, :, None] >> (8 * e4[None, None, :])) & 0xFF).to(tl.float32)              # [BN,64,4]
            sgb = tl.load(SG + b[:, None] * 32 + b32[None, :], mask=cm[:, None], other=0).to(tl.int32)
            neg = ((sgb[:, :, None] >> j8[None, None, :]) & 1) == 1                           # [BN,32,8]
            scb = tl.load(SC + b[:, None] * 4 + (g8 // 2)[None, :], mask=cm[:, None], other=0).to(tl.int32)
            sc4 = (scb >> (4 * (g8 % 2))[None, :]) & 0xF                                      # [BN,8]
            d = tl.load(D + b, mask=cm, other=0.0).to(tl.float32)
            db = (d[:, None] * (1.0 + 2.0 * sc4.to(tl.float32)))[:, :, None]                  # [BN,8,1]
            w = tl.reshape(tl.where(neg, -1.0, 1.0), (BN, 8, 32))
            w = (w * tl.reshape(gr, (BN, 8, 32)) * db)
            w = tl.reshape(w, (BN, 256)).to(tl.bfloat16)
            x = tl.load(X + rows[:, None] * s_x + kb * 256 + t[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq2s_gemm(X, s_x, D, QS, QH, SG, SC, GR, Y, s_y, M, N, NB, BN: tl.constexpr, F32: tl.constexpr):
        # IQ2_S: 256 a block; 32 qs bytes | two qh bits each index the 1024-entry 8-elem grid; 32 sg
        # bytes sign each weight; a 4-bit scale a 16-elem group. w = d*(0.5+s)*0.25*grid*sign.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        brow = cols * NB
        t = tl.arange(0, 256)
        e = tl.arange(0, 32)
        j8 = tl.arange(0, 8)
        g8 = tl.arange(0, 8)
        c4 = tl.arange(0, 4)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            b = brow + kb
            lo = tl.load(QS + b[:, None] * 32 + e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            qhb = tl.load(QH + b[:, None] * 8 + g8[None, :], mask=cm[:, None], other=0).to(tl.int32)
            hbv = (qhb[:, :, None] >> (2 * c4[None, None, :])) & 3                            # [BN,8,4]
            idx = lo | (tl.reshape(hbv, (BN, 32)) << 8)                                       # [BN,32]
            ge = tl.load(GR + idx)                                                              # [BN,32] i64
            gr = ((ge[:, :, None] >> (8 * j8[None, None, :])) & 0xFF).to(tl.float32)              # [BN,32,8]
            sgb = tl.load(SG + b[:, None] * 32 + e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            neg = ((sgb[:, :, None] >> j8[None, None, :]) & 1) == 1                           # [BN,32,8]
            scb = tl.load(SC + b[:, None] * 8 + (e // 4)[None, :], mask=cm[:, None], other=0).to(tl.int32)
            sc4 = (scb >> (4 * ((e // 2) % 2))[None, :]) & 0xF                                # [BN,32]
            d = tl.load(D + b, mask=cm, other=0.0).to(tl.float32)
            db = (d[:, None] * (0.5 + sc4.to(tl.float32)) * 0.25)[:, :, None]                 # [BN,32,1]
            w = tl.reshape(tl.where(neg, -1.0, 1.0) * gr * db, (BN, 256)).to(tl.bfloat16)
            x = tl.load(X + rows[:, None] * s_x + kb * 256 + t[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq1s_gemm(X, s_x, D, QS, QH, GR, Y, s_y, M, N, NB, DELTA: tl.constexpr, BN: tl.constexpr,
                   F32: tl.constexpr):
        # IQ1_S: 256 a block; 32 qs bytes | 3 bits of the group's u16 qh word index the 2048-entry
        # 8-elem grid; qh bits 12..14 the group scale, bit 15 the constant delta's sign.
        # w = d*(2*sc+1) * (grid +- DELTA).
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        brow = cols * NB
        t = tl.arange(0, 256)
        e = tl.arange(0, 32)
        j8 = tl.arange(0, 8)
        g8 = tl.arange(0, 8)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            b = brow + kb
            lo = tl.load(QS + b[:, None] * 32 + e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            qw = tl.load(QH + b[:, None] * 8 + (e // 4)[None, :], mask=cm[:, None], other=0)    # [BN,32]
            idx = lo | (((qw >> (3 * (e % 4))[None, :]) & 7) << 8)                            # [BN,32]
            ge = tl.load(GR + idx)                                                              # [BN,32] i64
            gr = ((ge[:, :, None] >> (8 * j8[None, None, :])) & 0xFF).to(tl.float32) - 1.0        # [BN,32,8]
            qw8 = tl.load(QH + b[:, None] * 8 + g8[None, :], mask=cm[:, None], other=0)       # [BN,8]
            delta = tl.where((qw8 & 0x8000) == 0, DELTA, -DELTA)[:, :, None]                  # [BN,8,1]
            d = tl.load(D + b, mask=cm, other=0.0).to(tl.float32)
            dl = (d[:, None] * (2.0 * ((qw8 >> 12) & 7).to(tl.float32) + 1.0))[:, :, None]    # [BN,8,1]
            w = tl.reshape(gr, (BN, 8, 32)) + delta
            w = tl.reshape(w * dl, (BN, 256)).to(tl.bfloat16)
            x = tl.load(X + rows[:, None] * s_x + kb * 256 + t[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    @triton.jit
    def _iq1m_gemm(X, s_x, QS, QH, DL, GR, Y, s_y, M, N, NB, DELTA: tl.constexpr, BN: tl.constexpr,
                   F32: tl.constexpr):
        # IQ1_M: 256 a block; 32 qs bytes | a qh nibble's low three bits index the 2048-entry grid,
        # its high bit flips DELTA; the 16 scales folded into DL fp16 at init. w = dl*(grid+-DELTA).
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        brow = cols * NB
        t = tl.arange(0, 256)
        e = tl.arange(0, 32)
        j8 = tl.arange(0, 8)
        s16 = tl.arange(0, 16)
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, NB):
            b = brow + kb
            lo = tl.load(QS + b[:, None] * 32 + e[None, :], mask=cm[:, None], other=0).to(tl.int32)
            hmb = tl.load(QH + b[:, None] * 16 + (e // 2)[None, :], mask=cm[:, None], other=0).to(tl.int32)
            hb = (hmb >> (4 * (e % 2))[None, :]) & 0xF                                        # [BN,32]
            idx = lo | ((hb & 7) << 8)
            ge = tl.load(GR + idx)                                                              # [BN,32] i64
            gr = ((ge[:, :, None] >> (8 * j8[None, None, :])) & 0xFF).to(tl.float32) - 1.0        # [BN,32,8]
            delta = tl.where((hb & 8) == 0, DELTA, -DELTA)[:, :, None]                        # [BN,32,1]
            dl = tl.load(DL + b[:, None] * 16 + s16[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            w = tl.reshape(gr + delta, (BN, 16, 16))
            w = tl.reshape(w * dl[:, :, None], (BN, 256)).to(tl.bfloat16)
            x = tl.load(X + rows[:, None] * s_x + kb * 256 + t[None, :],
                        mask=rows[:, None] < M, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


    # GPU dequant for the prefill path (M > ROWS_KERN): each program decodes packed blocks from
    # the class's device tensors (the same fields its GEMM kernel reads — no host bytes, no extra
    # VRAM) into Y bf16. Scales folded to fp16 at init are used as-is, so the output is within
    # ~1 bf16 ulp of the fp32 `dq_*` twin rather than bitwise-equal to it.

    @triton.jit
    def _dq_q40(D, Q, Y, TOT, GB: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr):
        p = tl.program_id(0) * GB + tl.arange(0, GB)
        gm = p < TOT
        r = p // NB
        bi = p % NB
        t = tl.arange(0, 32)
        d = tl.load(D + r * NBP + bi, mask=gm, other=0.0).to(tl.float32)
        w32 = tl.load(Q + (r * (NBP // 8) + bi // 8)[:, None] * 32 + t[None, :], mask=gm[:, None], other=0)
        code = w32 >> (4 * (bi % 8))[:, None] & 0xF
        w = d[:, None] * (code.to(tl.float32) - 8.0)
        tl.store(Y + p[:, None] * 32 + t[None, :], w.to(tl.bfloat16), mask=gm[:, None])

    @triton.jit
    def _dq_q41(D, M_, Q, Y, TOT, GB: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr):
        p = tl.program_id(0) * GB + tl.arange(0, GB)
        gm = p < TOT
        r = p // NB
        bi = p % NB
        t = tl.arange(0, 32)
        d = tl.load(D + r * NBP + bi, mask=gm, other=0.0).to(tl.float32)
        w32 = tl.load(Q + (r * (NBP // 8) + bi // 8)[:, None] * 32 + t[None, :], mask=gm[:, None], other=0)
        code = w32 >> (4 * (bi % 8))[:, None] & 0xF
        m = tl.load(M_ + r * NBP + bi, mask=gm, other=0.0).to(tl.float32)
        w = d[:, None] * code.to(tl.float32) + m[:, None]
        tl.store(Y + p[:, None] * 32 + t[None, :], w.to(tl.bfloat16), mask=gm[:, None])

    @triton.jit
    def _dq_q50(D, QH, Q, Y, TOT, GB: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr):
        p = tl.program_id(0) * GB + tl.arange(0, GB)
        gm = p < TOT
        r = p // NB
        bi = p % NB
        t = tl.arange(0, 32)
        d = tl.load(D + r * NBP + bi, mask=gm, other=0.0).to(tl.float32)
        w32 = tl.load(Q + (r * (NBP // 8) + bi // 8)[:, None] * 32 + t[None, :], mask=gm[:, None], other=0)
        code = w32 >> (4 * (bi % 8))[:, None] & 0xF
        qh = tl.load(QH + r * NBP + bi, mask=gm, other=0)
        code = code | (((qh[:, None] >> t[None, :]) & 1) << 4)
        w = d[:, None] * (code - 16).to(tl.float32)
        tl.store(Y + p[:, None] * 32 + t[None, :], w.to(tl.bfloat16), mask=gm[:, None])

    @triton.jit
    def _dq_q51(D, M_, QH, Q, Y, TOT, GB: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr):
        p = tl.program_id(0) * GB + tl.arange(0, GB)
        gm = p < TOT
        r = p // NB
        bi = p % NB
        t = tl.arange(0, 32)
        d = tl.load(D + r * NBP + bi, mask=gm, other=0.0).to(tl.float32)
        w32 = tl.load(Q + (r * (NBP // 8) + bi // 8)[:, None] * 32 + t[None, :], mask=gm[:, None], other=0)
        code = w32 >> (4 * (bi % 8))[:, None] & 0xF
        qh = tl.load(QH + r * NBP + bi, mask=gm, other=0)
        code = code | (((qh[:, None] >> t[None, :]) & 1) << 4)
        m = tl.load(M_ + r * NBP + bi, mask=gm, other=0.0).to(tl.float32)
        w = d[:, None] * code.to(tl.float32) + m[:, None]
        tl.store(Y + p[:, None] * 32 + t[None, :], w.to(tl.bfloat16), mask=gm[:, None])

    @triton.jit
    def _dq_q2k(DS, DM, QS, Y):
        # per 256-block: folded fp16 scales/mins [16]; qs 64 bytes of 2-bit codes
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        qb = tl.load(QS + pid * 64 + (e // 128) * 32 + (e % 32)).to(tl.int32)
        code = ((qb >> (2 * ((e // 32) % 4))) & 3).to(tl.float32)
        d = tl.load(DS + pid * 16 + e // 16).to(tl.float32)
        m = tl.load(DM + pid * 16 + e // 16).to(tl.float32)
        tl.store(Y + pid * 256 + e, (d * code - m).to(tl.bfloat16))

    @triton.jit
    def _dq_q3k(DL, QS, HM, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        qb = tl.load(QS + pid * 64 + (e // 128) * 32 + (e % 32)).to(tl.int32)
        code = (qb >> (2 * ((e // 32) % 4))) & 3
        hbit = (tl.load(HM + pid * 32 + (e % 32)).to(tl.int32) >> (e // 32)) & 1
        q = code - 4 * (1 - hbit)
        dl = tl.load(DL + pid * 16 + e // 16).to(tl.float32)
        tl.store(Y + pid * 256 + e, (dl * q.to(tl.float32)).to(tl.bfloat16))

    @triton.jit
    def _dq_q4k(DS, DM, QS, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        s = e // 32
        qb = tl.load(QS + pid * 128 + (s // 2) * 32 + (e % 32)).to(tl.int32)
        code = ((qb >> (4 * (s % 2))) & 0xF).to(tl.float32)
        d = tl.load(DS + pid * 8 + s).to(tl.float32)
        m = tl.load(DM + pid * 8 + s).to(tl.float32)
        tl.store(Y + pid * 256 + e, (d * code - m).to(tl.bfloat16))

    @triton.jit
    def _dq_q5k(DS, DM, QS, QH, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        s = e // 32
        hbit = (tl.load(QH + pid * 32 + (e % 32)).to(tl.int32) >> s) & 1
        qb = tl.load(QS + pid * 128 + (s // 2) * 32 + (e % 32)).to(tl.int32)
        code = (((qb >> (4 * (s % 2))) & 0xF) | (hbit << 4)).to(tl.float32)
        d = tl.load(DS + pid * 8 + s).to(tl.float32)
        m = tl.load(DM + pid * 8 + s).to(tl.float32)
        tl.store(Y + pid * 256 + e, (d * code - m).to(tl.bfloat16))

    @triton.jit
    def _dq_q6k(DL, QL, QH, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        ql = tl.load(QL + pid * 128 + (e // 128) * 64 + (e % 64)).to(tl.int32)
        lo = (ql >> (4 * ((e // 64) % 2))) & 0xF
        qh = tl.load(QH + pid * 64 + (e // 128) * 32 + (e % 32)).to(tl.int32)
        hb = (qh >> (2 * ((e // 32) % 4))) & 3
        code = ((lo | (hb << 4)) - 32).to(tl.float32)
        dl = tl.load(DL + pid * 16 + e // 16).to(tl.float32)
        tl.store(Y + pid * 256 + e, (dl * code).to(tl.bfloat16))

    @triton.jit
    def _dq_iq4nl(D, Q, LUT, Y, TOT, GB: tl.constexpr):
        p = tl.program_id(0) * GB + tl.arange(0, GB)
        gm = p < TOT
        t = tl.arange(0, 32)
        d = tl.load(D + p, mask=gm, other=0.0).to(tl.float32)
        q = tl.load(Q + p[:, None] * 16 + (t & 15)[None, :], mask=gm[:, None], other=0).to(tl.int32)
        w = d[:, None] * tl.load(LUT + ((q >> ((t // 16) * 4)[None, :]) & 0xF))
        tl.store(Y + p[:, None] * 32 + t[None, :], w.to(tl.bfloat16), mask=gm[:, None])

    @triton.jit
    def _dq_iq4xs(QS, DL, LUT, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        s = e // 32
        ee = e % 32
        nib = (tl.load(QS + pid * 128 + s * 16 + (ee & 15)).to(tl.int32) >> (4 * (ee // 16))) & 0xF
        dl = tl.load(DL + pid * 8 + s).to(tl.float32)
        tl.store(Y + pid * 256 + e, (dl * tl.load(LUT + nib)).to(tl.bfloat16))

    @triton.jit
    def _dq_iq2xs(D, QS16, SC, GR, KS, Y):
        # sixteen 16-weight groups; QS16 = the block's 32 u16 grid|sign-index words
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        g = e // 16
        i2 = (e % 16) // 8
        j = e % 8
        u = tl.load(QS16 + pid * 32 + 2 * g + i2).to(tl.int32) & 0xFFFF
        mag = ((tl.load(GR + (u & 511)) >> (8 * j)) & 0xFF).to(tl.float32)
        neg = ((tl.load(KS + (u >> 9)) >> j) & 1) == 1
        s4 = (tl.load(SC + pid * 8 + (g // 2)).to(tl.int32) >> (4 * (g % 2))) & 0xF
        d = tl.load(D + pid).to(tl.float32)
        w = (d * (0.5 + s4.to(tl.float32))) * 0.25 * mag
        tl.store(Y + pid * 256 + e, tl.where(neg, -w, w).to(tl.bfloat16))

    @triton.jit
    def _dq_iq3xxs(D, QS, SC, GR, KS, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        s = e // 32
        ee = e % 32
        mag = ((tl.load(GR + tl.load(QS + pid * 64 + s * 8 + ee // 4).to(tl.int32)) >> (8 * (e % 4)))
               & 0xFF).to(tl.float32)
        scw = tl.load(SC + pid * 8 + s)
        ks = tl.load(KS + ((scw >> (7 * (ee // 8))) & 0x7F))
        neg = ((ks >> (e % 8)) & 1) == 1
        d = tl.load(D + pid).to(tl.float32)
        w = (d * (0.5 + ((scw >> 28) & 0xF).to(tl.float32))) * 0.5 * mag
        tl.store(Y + pid * 256 + e, tl.where(neg, -w, w).to(tl.bfloat16))

    @triton.jit
    def _dq_iq3s(D, QS, QH, SG, SC, GR, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        s = e // 32
        ee = e % 32
        lo = tl.load(QS + pid * 64 + s * 8 + ee // 4).to(tl.int32)
        hmb = tl.load(QH + pid * 8 + s).to(tl.int32)
        mag = ((tl.load(GR + (lo | (((hmb >> (ee // 4)) & 1) << 8))) >> (8 * (e % 4)))
               & 0xFF).to(tl.float32)
        neg = ((tl.load(SG + pid * 32 + s * 4 + ee // 8).to(tl.int32) >> (e % 8)) & 1) == 1
        s4 = (tl.load(SC + pid * 4 + (s // 2)).to(tl.int32) >> (4 * (s % 2))) & 0xF
        d = tl.load(D + pid).to(tl.float32)
        w = (d * (1.0 + 2.0 * s4.to(tl.float32))) * mag
        tl.store(Y + pid * 256 + e, tl.where(neg, -w, w).to(tl.bfloat16))

    @triton.jit
    def _dq_iq2s(D, QS, SG, QH, SC, GR, Y):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        g = e // 16
        i2 = (e % 16) // 8
        j = e % 8
        h = 2 * g + i2
        lo = tl.load(QS + pid * 32 + 2 * g + i2).to(tl.int32)
        hb = (tl.load(QH + pid * 8 + (h // 4)).to(tl.int32) >> (2 * (h % 4))) & 3
        mag = ((tl.load(GR + (lo | (hb << 8))) >> (8 * j)) & 0xFF).to(tl.float32)
        neg = ((tl.load(SG + pid * 32 + 2 * g + i2).to(tl.int32) >> j) & 1) == 1
        s4 = (tl.load(SC + pid * 8 + (g // 2)).to(tl.int32) >> (4 * (g % 2))) & 0xF
        d = tl.load(D + pid).to(tl.float32)
        w = (d * (0.5 + s4.to(tl.float32))) * 0.25 * mag
        tl.store(Y + pid * 256 + e, tl.where(neg, -w, w).to(tl.bfloat16))

    @triton.jit
    def _dq_iq1s(D, QS, QH, GR, Y, DELTA: tl.constexpr):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        s = e // 32
        l = (e % 32) // 8
        qh = tl.load(QH + pid * 8 + s)
        idx = tl.load(QS + pid * 32 + s * 4 + l).to(tl.int32) | (((qh >> (3 * l)) & 7) << 8)
        mag = ((tl.load(GR + idx) >> (8 * (e % 8))) & 0xFF).to(tl.float32) - 1.0
        d = tl.load(D + pid).to(tl.float32)
        dl = d * (2.0 * ((qh >> 12) & 7).to(tl.float32) + 1.0)
        delta = tl.where((qh & 0x8000) == 0, DELTA, -DELTA)
        tl.store(Y + pid * 256 + e, (dl * (mag + delta)).to(tl.bfloat16))

    @triton.jit
    def _dq_iq1m(QS, QH, DL, GR, Y, DELTA: tl.constexpr):
        pid = tl.program_id(0)
        e = tl.arange(0, 256)
        i = e // 8
        j = e % 8
        hb = (tl.load(QH + pid * 16 + (i // 2)).to(tl.int32) >> (4 * (i % 2))) & 0xF
        idx = tl.load(QS + pid * 32 + i).to(tl.int32) | ((hb & 7) << 8)
        mag = ((tl.load(GR + idx) >> (8 * j)) & 0xFF).to(tl.float32) - 1.0
        dl = tl.load(DL + pid * 16 + i // 2).to(tl.float32)
        delta = tl.where((hb & 8) == 0, DELTA, -DELTA)
        tl.store(Y + pid * 256 + e, (dl * (mag + delta)).to(tl.bfloat16))


#: tid -> (dequant kernel, ((attr, int16-view?)...), constexpr attr -> param, blocks/program):
#: _Packed.dequant_gpu launches one program per GB packed blocks of the n0:n1 row slice.
_DQ = {}
if HAVE_TRITON:
    _DQ = {2: (_dq_q40, (("d", 1, 0), ("q", 1, 0)), {"NB": "_nb", "NBP": "_nbp"}, 8),
           3: (_dq_q41, (("d", 1, 0), ("m", 1, 0), ("q", 1, 0)), {"NB": "_nb", "NBP": "_nbp"}, 8),
           6: (_dq_q50, (("d", 1, 0), ("qh", 1, 0), ("q", 1, 0)), {"NB": "_nb", "NBP": "_nbp"}, 8),
           7: (_dq_q51, (("d", 1, 0), ("m", 1, 0), ("qh", 1, 0), ("q", 1, 0)),
               {"NB": "_nb", "NBP": "_nbp"}, 8),
           10: (_dq_q2k, (("ds", 1, 0), ("dm", 1, 0), ("qs", 1, 0)), {}, 1),
           11: (_dq_q3k, (("dl", 1, 0), ("qs", 1, 0), ("hm", 1, 0)), {}, 1),
           12: (_dq_q4k, (("ds", 1, 0), ("dm", 1, 0), ("qs", 1, 0)), {}, 1),
           13: (_dq_q5k, (("ds", 1, 0), ("dm", 1, 0), ("qs", 1, 0), ("qh", 1, 0)), {}, 1),
           14: (_dq_q6k, (("dl", 1, 0), ("ql", 1, 0), ("qh", 1, 0)), {}, 1),
           17: (_dq_iq2xs, (("d", 1, 0), ("qs", 1, torch.int16), ("sc", 1, 0),
                            ("gr", 0, 0), ("ks", 0, 0)), {}, 1),
           18: (_dq_iq3xxs, (("d", 1, 0), ("qs", 1, 0), ("sc", 1, 0),
                             ("gr", 0, 0), ("ks", 0, 0)), {}, 1),
           19: (_dq_iq1s, (("d", 1, 0), ("qs", 1, 0), ("qh", 1, 0), ("gr", 0, 0)),
                {"DELTA": "_delta"}, 1),
           20: (_dq_iq4nl, (("d", 1, 0), ("q", 1, 0), ("lut", 0, 0)), {}, 8),
           21: (_dq_iq3s, (("d", 1, 0), ("qs", 1, 0), ("qh", 1, 0), ("sg", 1, 0),
                           ("sc", 1, 0), ("gr", 0, 0)), {}, 1),
           22: (_dq_iq2s, (("d", 1, 0), ("qs", 1, 0), ("sg", 1, 0), ("qh", 1, 0),
                           ("sc", 1, 0), ("gr", 0, 0)), {}, 1),
           23: (_dq_iq4xs, (("qs", 1, 0), ("dl", 1, 0), ("lut", 0, 0)), {}, 1),
           29: (_dq_iq1m, (("qs", 1, 0), ("qh", 1, 0), ("dl", 1, 0), ("gr", 0, 0)),
                {"DELTA": "_delta"}, 1)}


class _Packed:
    """Shared bits of the packed GGUF linears: matmul on the fused kernel up to ROWS_KERN rows,
    chunked GPU dequant + cuBLAS beyond. `_raw` stays on the host for `dequant`."""

    def _twin(self):
        return self.dequant().to(torch.bfloat16)

    def dequant(self) -> torch.Tensor:                    # fp32 [N, K] on CPU
        return dequant(self.tid, self._raw, (self.N, self.K))

    def dequant_gpu(self, n0: int = 0, n1: int | None = None) -> torch.Tensor:
        """Rows n0:n1 decoded to bf16 on the device from the class's device tensors (no host
        bytes); within ~1 bf16 ulp of `dequant()` since folded scales are stored fp16."""
        n1 = self.N if n1 is None else n1
        dev = next(v.device for v in vars(self).values() if isinstance(v, torch.Tensor))
        w = torch.empty(n1 - n0, self.K, dtype=torch.bfloat16, device=dev)
        kern, attrs, cst, gb = _DQ[self.tid]
        nbr = self.K // TYPES[self.tid][0]
        args = []
        for name, is_row, view in attrs:
            v = getattr(self, name)
            v = v.reshape(-1)[n0 * (v.numel() // self.N):
                              n1 * (v.numel() // self.N)] if is_row else v
            args.append(v.view(view) if view else v)
        nb = (n1 - n0) * nbr
        kw = dict(num_warps=4, **{k: getattr(self, v) for k, v in cst.items()})
        if gb > 1:
            kern[(triton.cdiv(nb, gb),)](*args, w, nb, GB=gb, **kw)
        else:
            kern[(nb,)](*args, w, **kw)
        return w

    def _prefill(self, x2: torch.Tensor, f32: bool) -> torch.Tensor:
        """A prompt chunk (M > ROWS_KERN): rows are independent, so the weight is decoded to bf16
        a slice of rows at a time (a slice stays under PREFILL_W_BYTES) and each goes through
        cuBLAS into its columns of y. Off-GPU (or no Triton) falls back to the fp32 twin."""
        if not x2.is_cuda or not HAVE_TRITON:
            y = torch.nn.functional.linear(x2, self._twin().to(x2.device))
            return y.float() if f32 else y
        xb = x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)
        y = torch.empty(x2.shape[0], self.N, dtype=torch.bfloat16, device=x2.device)
        cn = max(1, int(PREFILL_W_BYTES // (2 * self.K)))
        for n0 in range(0, self.N, cn):
            n1 = min(self.N, n0 + cn)
            y[:, n0:n1] = torch.nn.functional.linear(xb, self.dequant_gpu(n0, n1))
        return y.float() if f32 else y

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        if M > ROWS_KERN or not HAVE_TRITON:
            y = self._prefill(x2, f32)
            return (y if out is None else out.copy_(y)).view(*x.shape[:-1], self.N)
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        self._kernel(x2, out, M, f32)
        return out.view(*x.shape[:-1], self.N)


class Q40Linear(_Packed):
    """Packed Q4_0 [N, K] on the device: d fp16 [N, nbp] zero-padded to 8-block
    groups, codes plane-major int32 words [N, nbp/8, 32] (word j = element j's codes
    of blocks 8g..8g+7 at bits 4i)."""

    tid = 2

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 18)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self._nb, self._nbp = nb, (nb + 7) // 8 * 8
        self.d = torch.from_numpy(_padn(np.ascontiguousarray(b[..., :2]).view(np.float16)
                                        .reshape(N, nb), self._nbp)).to(device)
        self.q = torch.from_numpy(_p8(_nibbles(np.ascontiguousarray(b[..., 2:]), 16))).to(device)
        self.nbytes = N * nb * 18

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 4, 2))
        ng = self.q.shape[1]
        if ng * 256 != self.K:
            x2 = torch.cat([x2, x2.new_zeros(M, ng * 256 - self.K)], 1)  # pad block tail -> 0
        _q40_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.q, out,
               out.stride(0), M, self.N, ng, BN=bn, F32=f32, num_warps=wp, num_stages=st)


class Q4KLinear(_Packed):
    """Packed Q4_K [N, K] on the device: the 12-byte scale pack is decoded once at init into
    ds = d*sc, dm = dmin*m fp16 [N, K/256, 8]; qs keep the packed nibbles [N, K/256, 128]."""

    tid = 12

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 144)
        self._raw = b.copy()
        d = _f16(b[..., :2])                       # [N, nb, 1]
        dm = _f16(b[..., 2:4])
        sc, mn = _q4k_scales(b[..., 4:16].reshape(-1, 12))   # [(N*nb), 8] u8 each
        sc = sc.reshape(N, nb, 8)
        mn = mn.reshape(N, nb, 8)
        ds = d * sc.astype(np.float32)
        dmm = dm * mn.astype(np.float32)
        self.N, self.K, self.shape = N, K, (N, K)
        self.ds = torch.from_numpy(np.ascontiguousarray(ds.astype(np.float16))).to(device)
        self.dm = torch.from_numpy(np.ascontiguousarray(dmm.astype(np.float16))).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 16:])).to(device)  # [N, nb, 128]
        self.nbytes = N * nb * 144

    def _kernel(self, x2, out, M, f32):
        _q4k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.qs, self.ds,
                                                               self.dm, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=32, F32=f32,
                                                               num_warps=2, num_stages=3)


class Q6KLinear(_Packed):
    """Packed Q6_K [N, K]: dl = d*sc fp16 [N, K/256, 16] folded at init; ql [N, nb, 128] and
    qh [N, nb, 64] keep the packed codes."""

    tid = 14

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 210)
        self._raw = b.copy()
        self.ql = torch.from_numpy(np.ascontiguousarray(b[..., :128])).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 128:192])).to(device)
        d = _f16(b[..., 208:210])
        dl = d * b[..., 192:208].view(np.int8).astype(np.float32)   # [N, nb, 16]
        self.dl = torch.from_numpy(np.ascontiguousarray(dl.astype(np.float16))).to(device)
        self.N, self.K, self.shape = N, K, (N, K)
        self.nbytes = N * nb * 210

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 1, 2))
        _q6k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.ql, self.qh,
                                                               self.dl, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=bn, F32=f32,
                                                               num_warps=wp, num_stages=st)


class Q41Linear(_Packed):
    """Packed Q4_1 [N, K]: d/m fp16 [N, nbp] padded like Q4_0; plane-major codes
    [N, nbp/8, 32] int32."""

    tid = 3

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 20)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self._nb, self._nbp = nb, (nb + 7) // 8 * 8
        self.d = torch.from_numpy(_padn(np.ascontiguousarray(_f16(b[..., 0:2]))
                                        .astype(np.float16).reshape(N, nb), self._nbp)).to(device)
        self.m = torch.from_numpy(_padn(np.ascontiguousarray(_f16(b[..., 2:4]))
                                        .astype(np.float16).reshape(N, nb), self._nbp)).to(device)
        self.q = torch.from_numpy(_p8(_nibbles(np.ascontiguousarray(b[..., 4:20]), 16))).to(device)
        self.nbytes = N * nb * 20

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 4, 2))
        ng = self.q.shape[1]
        if ng * 256 != self.K:
            x2 = torch.cat([x2, x2.new_zeros(M, ng * 256 - self.K)], 1)  # pad block tail -> 0
        _q41_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.m, self.q, out,
               out.stride(0), M, self.N, ng, BN=bn, F32=f32, num_warps=wp, num_stages=st)


class Q50Linear(_Packed):
    """Packed Q5_0 [N, K]: d fp16 and qh u32 bits [N, nbp] padded; plane-major
    codes [N, nbp/8, 32] int32."""

    tid = 6

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 22)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self._nb, self._nbp = nb, (nb + 7) // 8 * 8
        self.d = torch.from_numpy(_padn(np.ascontiguousarray(_f16(b[..., 0:2]))
                                        .astype(np.float16).reshape(N, nb), self._nbp)).to(device)
        self.qh = torch.from_numpy(_padn(np.ascontiguousarray(b[..., 2:6]).view(np.int32)
                                         .reshape(N, nb), self._nbp)).to(device)
        self.q = torch.from_numpy(_p8(_nibbles(np.ascontiguousarray(b[..., 6:22]), 16))).to(device)
        self.nbytes = N * nb * 22

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 4, 2))
        ng = self.q.shape[1]
        if ng * 256 != self.K:
            x2 = torch.cat([x2, x2.new_zeros(M, ng * 256 - self.K)], 1)  # pad block tail -> 0
        _q50_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.qh, self.q, out,
               out.stride(0), M, self.N, ng, BN=bn, F32=f32, num_warps=wp, num_stages=st)


class Q51Linear(_Packed):
    """Packed Q5_1 [N, K]: d/m fp16 and qh u32 bits [N, nbp] padded; plane-major
    codes [N, nbp/8, 32] int32."""

    tid = 7

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 24)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self._nb, self._nbp = nb, (nb + 7) // 8 * 8
        self.d = torch.from_numpy(_padn(np.ascontiguousarray(_f16(b[..., 0:2]))
                                        .astype(np.float16).reshape(N, nb), self._nbp)).to(device)
        self.m = torch.from_numpy(_padn(np.ascontiguousarray(_f16(b[..., 2:4]))
                                        .astype(np.float16).reshape(N, nb), self._nbp)).to(device)
        self.qh = torch.from_numpy(_padn(np.ascontiguousarray(b[..., 4:8]).view(np.int32)
                                         .reshape(N, nb), self._nbp)).to(device)
        self.q = torch.from_numpy(_p8(_nibbles(np.ascontiguousarray(b[..., 8:24]), 16))).to(device)
        self.nbytes = N * nb * 24

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 4, 2))
        ng = self.q.shape[1]
        if ng * 256 != self.K:
            x2 = torch.cat([x2, x2.new_zeros(M, ng * 256 - self.K)], 1)  # pad block tail -> 0
        _q51_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.m, self.qh, self.q, out,
               out.stride(0), M, self.N, ng, BN=bn, F32=f32, num_warps=wp, num_stages=st)


class Q2KLinear(_Packed):
    """Packed Q2_K [N, K]: ds = d*(sc&0xF), dm = dmin*(sc>>4) fp16 [N, K/256, 16] folded at init;
    qs [N, K/256, 64] keep the packed 2-bit codes."""

    tid = 10

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 84)
        self._raw = b.copy()
        sc = b[..., :16]                                  # u8: lo nibble scale, hi nibble min
        d = _f16(b[..., 80:82]).astype(np.float32)
        dmin = _f16(b[..., 82:84]).astype(np.float32)
        ds = d * (sc & 0xF).astype(np.float32)
        dm = dmin * (sc >> 4).astype(np.float32)
        self.N, self.K, self.shape = N, K, (N, K)
        self.ds = torch.from_numpy(np.ascontiguousarray(ds.astype(np.float16))).to(device)
        self.dm = torch.from_numpy(np.ascontiguousarray(dm.astype(np.float16))).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 16:80])).to(device)
        self.nbytes = N * nb * 84

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 2, 3))
        _q2k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.qs, self.ds,
                                                               self.dm, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=bn, F32=f32,
                                                               num_warps=wp, num_stages=st)


class Q3KLinear(_Packed):
    """Packed Q3_K [N, K]: dl = d*sc int6→int8 fp16 [N, K/256, 16] folded at init;
    qs [N, K/256, 64] low two bits, hmask [N, K/256, 32] the sign bit per element."""

    tid = 11

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 110)
        self._raw = b.copy()
        a = b[..., 96:108]                                # [N, nb, 12] packed 6-bit scales
        flat = a.reshape(-1, 12)
        ls = (flat[:, :8].reshape(-1, 1, 8) >> np.array([0, 4], np.uint8)[None, :, None]).reshape(-1, 16)
        hs = (flat[:, 8:12].reshape(-1, 1, 4) >> np.array([0, 2, 4, 6], np.uint8)[None, :, None]).reshape(-1, 16)
        sc16 = ((ls & np.uint8(0x0F)) | ((hs & np.uint8(0x03)) << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = _f16(b[..., 108:110]).astype(np.float32).reshape(-1, 1) * sc16.astype(np.float32)
        self.N, self.K, self.shape = N, K, (N, K)
        self.dl = torch.from_numpy(np.ascontiguousarray(dl.astype(np.float16))).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 32:96])).to(device)
        self.hm = torch.from_numpy(np.ascontiguousarray(b[..., 0:32])).to(device)
        self.nbytes = N * nb * 110

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 2, 3))
        _q3k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.qs, self.hm,
                                                               self.dl, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=bn, F32=f32,
                                                               num_warps=wp, num_stages=st)


class Q5KLinear(_Packed):
    """Packed Q5_K [N, K]: ds/dm fp16 [N, K/256, 8] from the 6-bit pack at init;
    qs [N, K/256, 128] nibbles, qh [N, K/256, 32] the fifth bits."""

    tid = 13

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 176)
        self._raw = b.copy()
        d = _f16(b[..., :2]).astype(np.float32)
        dm = _f16(b[..., 2:4]).astype(np.float32)
        sc, mn = _q4k_scales(b[..., 4:16].reshape(-1, 12))
        ds = d.reshape(-1, 1) * sc.astype(np.float32)
        dmm = dm.reshape(-1, 1) * mn.astype(np.float32)
        self.N, self.K, self.shape = N, K, (N, K)
        self.ds = torch.from_numpy(np.ascontiguousarray(ds.reshape(N, nb, 8).astype(np.float16))).to(device)
        self.dm = torch.from_numpy(np.ascontiguousarray(dmm.reshape(N, nb, 8).astype(np.float16))).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 48:176])).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 16:48])).to(device)
        self.nbytes = N * nb * 176

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 2, 3))
        _q5k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.qs, self.qh,
                                                               self.ds, self.dm, out, out.stride(0),
                                                               M, self.N, self.K // 256, BN=bn,
                                                               F32=f32, num_warps=wp, num_stages=st)


class IQ4NLLinear(_Packed):
    """Packed IQ4_NL [N, K]: d fp16 [N, K/32], nibble bytes [N, K/32, 16] indexing a fixed
    16-entry nonlinear value table (kept on-device as fp32)."""

    tid = 20

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 18)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.q = torch.from_numpy(np.ascontiguousarray(b[..., 2:18])).to(device)
        self.lut = torch.from_numpy(np.ascontiguousarray(GR.IQ4_NL_VALUES.astype(np.float32))).to(device)
        self.nbytes = N * nb * 18

    def _kernel(self, x2, out, M, f32):
        _iq4nl_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.lut,
                                                                 self.q, out, out.stride(0), M, self.N,
                                                                 self.K, BN=32, SB=4, F32=f32,
                                                                 num_warps=2, num_stages=3)


class IQ4XSLinear(_Packed):
    """Packed IQ4_XS [N, K]: dl = d*sc fp16 [N, K/256, 8] folded at init (sc = the 6-bit field
    (4-bit lo, 2-bit hi) minus 32); qs [N, K/256, 128] index the IQ4_NL value table on-device."""

    tid = 23

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 136)
        self._raw = b.copy()
        d = _f16(b[..., 0:2]).astype(np.float32)
        sh = b[..., 2:4].reshape(-1, 2).view(np.uint16) >> (2 * np.arange(8, dtype=np.uint16))[None, :]
        sl = (b[..., 4:8].reshape(-1, 4, 1) >> np.array([0, 4], np.uint8)[None, None, :]).reshape(-1, 8)
        sc = ((sl & np.uint8(0x0F)) | ((sh & np.uint16(3)).astype(np.uint8) << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = d.reshape(-1, 1) * sc.astype(np.float32)
        self.N, self.K, self.shape = N, K, (N, K)
        self.dl = torch.from_numpy(np.ascontiguousarray(dl.reshape(N, nb, 8).astype(np.float16))).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 8:136])).to(device)
        self.lut = torch.from_numpy(np.ascontiguousarray(GR.IQ4_NL_VALUES.astype(np.float32))).to(device)
        self.nbytes = N * nb * 136

    def _kernel(self, x2, out, M, f32):
        _iq4xs_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.qs, self.dl,
                                                                 self.lut, out, out.stride(0), M, self.N,
                                                                 self.K // 256, BN=32, F32=f32,
                                                                 num_warps=2, num_stages=1)


class IQ2XSLinear(_Packed):
    """Packed IQ2_XS [N, K]: d fp16 [N, K/256], qs [N, nb, 64] (u16 grid|sign-index pairs),
    sc [N, nb, 8] (group nibbles); the 512-entry grid and the sign table live on-device."""

    tid = 17

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        from tools.iq2_linear import KSIGNS
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 74)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 2:66])).to(device)
        self.sc = torch.from_numpy(np.ascontiguousarray(b[..., 66:74])).to(device)
        self.gr = torch.from_numpy((GR.IQ2_XS_GRID.astype(np.int64) << (8 * np.arange(8))).sum(1)).to(device)
        self.ks = torch.from_numpy(np.ascontiguousarray(KSIGNS.astype(np.int32))).to(device)
        self.nbytes = N * nb * 74

    def _kernel(self, x2, out, M, f32):
        _iq2xs_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 16))](x2, x2.stride(0), self.d, self.qs,
                                                                 self.sc, self.gr, self.ks, out,
                                                                 out.stride(0), M, self.N,
                                                                 self.K // 256, BN=16, F32=f32,
                                                                 num_warps=4, num_stages=1)


class IQ3XXSLinear(_Packed):
    """Packed IQ3_XXS [N, K]: d fp16 [N, nb], qs [N, nb, 64] (u8 grid indices into the
    256-entry 4-elem grid), sc u32 [N, nb, 8] (sign indices + group scale)."""

    tid = 18

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        from tools.iq2_linear import KSIGNS
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 98)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 2:66])).to(device)
        self.sc = torch.from_numpy(np.ascontiguousarray(b[..., 66:98]).view(np.uint32).astype(np.int32)).to(device)
        self.gr = torch.from_numpy((GR.IQ3_XXS_GRID.astype(np.int32) << (8 * np.arange(4))).sum(1)).to(device)
        self.ks = torch.from_numpy(np.ascontiguousarray(KSIGNS.astype(np.int32))).to(device)
        self.nbytes = N * nb * 98

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 4, 1))
        _iq3xxs_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.qs,
                                                                  self.sc, self.gr, self.ks, out,
                                                                  out.stride(0), M, self.N,
                                                                  self.K // 256, BN=bn, F32=f32,
                                                                  num_warps=wp, num_stages=st)


class IQ3SLinear(_Packed):
    """Packed IQ3_S [N, K]: d fp16 [N, nb], qs [N, nb, 64] + qh [N, nb, 8] (9-bit grid indices into
    the 512-entry 4-elem grid), sg [N, nb, 32] (per-weight sign bits), sc [N, nb, 4] nibbles."""

    tid = 21

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 110)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 2:66])).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 66:74])).to(device)
        self.sg = torch.from_numpy(np.ascontiguousarray(b[..., 74:106])).to(device)
        self.sc = torch.from_numpy(np.ascontiguousarray(b[..., 106:110])).to(device)
        self.gr = torch.from_numpy((GR.IQ3_S_GRID.astype(np.int32) << (8 * np.arange(4))).sum(1)).to(device)
        self.nbytes = N * nb * 110

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (32, 4, 1))
        _iq3s_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.qs,
                                                                self.qh, self.sg, self.sc, self.gr,
                                                                out, out.stride(0), M, self.N,
                                                                self.K // 256, BN=bn, F32=f32,
                                                                num_warps=wp, num_stages=st)


class IQ2SLinear(_Packed):
    """Packed IQ2_S [N, K]: d fp16 [N, nb], qs [N, nb, 32] + qh [N, nb, 8] (10-bit grid indices
    into the 1024-entry 8-elem grid), sg [N, nb, 32] sign bits, sc [N, nb, 8] nibbles."""

    tid = 22

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 82)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 2:34])).to(device)
        self.sg = torch.from_numpy(np.ascontiguousarray(b[..., 34:66])).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 66:74])).to(device)
        self.sc = torch.from_numpy(np.ascontiguousarray(b[..., 74:82])).to(device)
        self.gr = torch.from_numpy((GR.IQ2_S_GRID.astype(np.int64) << (8 * np.arange(8))).sum(1)).to(device)
        self.nbytes = N * nb * 82

    def _kernel(self, x2, out, M, f32):
        _iq2s_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.qs,
                                                                self.qh, self.sg, self.sc, self.gr,
                                                                out, out.stride(0), M, self.N,
                                                                self.K // 256, BN=32, F32=f32,
                                                                num_warps=4, num_stages=1)


class IQ1SLinear(_Packed):
    """Packed IQ1_S [N, K]: d fp16 [N, nb], qs [N, nb, 32] + qh words [N, nb, 8] carrying grid
    hi-bits, group scale and delta sign; the 2048-entry grid on-device."""

    tid = 19

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 50)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 2:34])).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 34:50]).view(np.uint16).astype(np.int32)).to(device)
        self.gr = torch.from_numpy(((GR.IQ1_GRID.astype(np.int64) + 1) << (8 * np.arange(8))).sum(1)).to(device)
        self.nbytes = N * nb * 50
        self._delta = float(GR.IQ1_DELTA)

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (16, 4, 1))
        _iq1s_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.d, self.qs,
                                                                self.qh, self.gr, out,
                                                                out.stride(0), M, self.N,
                                                                self.K // 256, DELTA=self._delta,
                                                                BN=bn, F32=f32,
                                                                num_warps=wp, num_stages=st)


class IQ1MLinear(_Packed):
    """Packed IQ1_M [N, K]: qs [N, nb, 32] + qh [N, nb, 16] (grid hi-bits + delta bits), and the
    nibble-assembled fp16 d folded with the 3-bit scales into dl fp16 [N, nb, 16] at init."""

    tid = 29

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        from tools import gguf_grids as GR
        nb = K // 256
        b = np.ascontiguousarray(raw).reshape(N, nb, 56)
        self._raw = b.copy()
        scw = b[..., 48:56].reshape(-1, 8).view(np.uint16).reshape(-1, 4)
        dn = (scw >> np.uint16(12)) & np.uint16(0xF)                                          # [nb*N, 4]
        d = ((dn[:, 0] | (dn[:, 1] << 4) | (dn[:, 2] << 8) | (dn[:, 3] << 12)).astype(np.uint16)
             .view(np.float16).astype(np.float32))
        sl = ((scw.reshape(-1, 4, 1) >> np.array([0, 3, 6, 9], np.uint16)[None, None, :])
              & np.uint16(7)).reshape(-1, 16)
        dl = (d[:, None] * (2 * sl.astype(np.float32) + 1)).reshape(N, nb, 16)
        self.N, self.K, self.shape = N, K, (N, K)
        self.dl = torch.from_numpy(np.ascontiguousarray(dl.astype(np.float16))).to(device)
        self.qs = torch.from_numpy(np.ascontiguousarray(b[..., 0:32])).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 32:48])).to(device)
        self.gr = torch.from_numpy(((GR.IQ1_GRID.astype(np.int64) + 1) << (8 * np.arange(8))).sum(1)).to(device)
        self.nbytes = N * nb * 56
        self._delta = float(GR.IQ1_DELTA)

    def _kernel(self, x2, out, M, f32):
        bn, wp, st = _GEMM_CFG.get((self.tid, self.N, self.K), (16, 2, 1))
        _iq1m_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, bn))](x2, x2.stride(0), self.qs, self.qh,
                                                                self.dl, self.gr, out,
                                                                out.stride(0), M, self.N,
                                                                self.K // 256, DELTA=self._delta,
                                                                BN=bn, F32=f32,
                                                                num_warps=wp, num_stages=st)

# (tid, N, K) -> (BN, num_warps, num_stages) on Ada: interleaved-A/B winners over the default
# launch at M=1 that also held at M=33 (proto/sweep_s3.py + proto/ab_cfg.py). No entry -> the
# class's built-in default.
_GEMM_CFG = {
    (10, 14336, 4096): (32, 4, 1),   # Q2_K
    (11, 14336, 4096): (32, 2, 2),   # Q3_K
    (14, 4096, 14336): (16, 1, 2),   # Q6_K
    (18, 14336, 4096): (16, 4, 1),   # IQ3_XXS
    (18, 4096, 14336): (16, 2, 1),   # IQ3_XXS
    (21, 4096, 4096): (16, 4, 1),    # IQ3_S
    (19, 14336, 4096): (16, 4, 2),   # IQ1_S
    (29, 14336, 4096): (32, 4, 1),   # IQ1_M
    (29, 4096, 14336): (16, 1, 1),   # IQ1_M
}

#: 2-D tensors of these types load as packed GPU linears (Tier A); every other `TYPES` entry
#: dequantises to a dense tensor at load (Tier B).
PACKED_LINEAR = {2: Q40Linear, 3: Q41Linear, 6: Q50Linear, 7: Q51Linear,
                 10: Q2KLinear, 11: Q3KLinear, 12: Q4KLinear, 13: Q5KLinear, 14: Q6KLinear,
                 17: IQ2XSLinear, 18: IQ3XXSLinear, 19: IQ1SLinear, 20: IQ4NLLinear,
                 21: IQ3SLinear, 22: IQ2SLinear, 23: IQ4XSLinear, 29: IQ1MLinear}
