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


# ============================================================ numpy decoders (the Tier-B path and twins)
# Each takes packed blocks uint8 [n, type_size] and returns fp32 [n, block_values], in ggml's
# element order — the same transforms gguf-py `dequantize_blocks` performs.

def _f16(b):
    return b.view(np.float16).astype(np.float32)


def _nibbles(qs, width):
    """Two nibbles a byte, lo first: [n, w] -> [n, 2w]."""
    return np.concatenate([qs & np.uint8(0x0F), qs >> np.uint8(4)], axis=-1)


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
    def _q40_gemm(X, s_x, D, Q, Y, s_y, M, N, K, BN: tl.constexpr, SB: tl.constexpr, F32: tl.constexpr):
        # Q4_0: 32 weights a block (16 bytes of nibbles + fp16 d). 16-row x tiles, BN outputs, SB blocks
        # a step; w decoded to bf16 and dotted, fp32 accumulate, K in order -> a row's bits do not depend
        # on the other rows of its tile (the engine's verify contract).
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        nb = K // 32
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, nb, SB):
            for j in tl.static_range(SB):
                b = kb + j
                q = tl.load(Q + cols[:, None] * (nb * 16) + b * 16 + (t & 15)[None, :],
                            mask=cm[:, None], other=0)
                code = ((q >> ((t // 16) * 4).to(tl.int32)[None, :]) & 0xF).to(tl.float32) - 8.0
                d = tl.load(D + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                w = (code * d[:, None]).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + b * 32 + t[None, :], mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

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
                ds = tl.load(DS + (cols * NB + kb) * 8 + s, mask=cm, other=0.0).to(tl.float32)
                dm = tl.load(DM + (cols * NB + kb) * 8 + s, mask=cm, other=0.0).to(tl.float32)
                w = (code.to(tl.float32) * ds[:, None] - dm[:, None]).to(tl.bfloat16)
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
                code = (((ql >> (4 * (c // 2))) & 0xF) | (((qh >> (2 * c)) & 3) << 4)).to(tl.float32) - 32.0
                base = (cols * NB + kb) * 16 + s * 2
                dl0 = tl.load(DL + base, mask=cm, other=0.0).to(tl.float32)
                dl1 = tl.load(DL + base + 1, mask=cm, other=0.0).to(tl.float32)
                w = (code * tl.where((t // 16)[None, :] == 1, dl1[:, None], dl0[:, None])).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + kb * 256 + s * 32 + t[None, :],
                            mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q41_gemm(X, s_x, D, M_, Q, Y, s_y, M, N, K, BN: tl.constexpr, SB: tl.constexpr, F32: tl.constexpr):
        # Q4_1: Q4_0's nibble pack with an additive min instead of the -8 bias: w = q*d + m.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        nb = K // 32
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, nb, SB):
            for j in tl.static_range(SB):
                b = kb + j
                q = tl.load(Q + cols[:, None] * (nb * 16) + b * 16 + (t & 15)[None, :],
                            mask=cm[:, None], other=0)
                code = (q >> ((t // 16) * 4).to(tl.int32)[None, :]) & 0xF
                d = tl.load(D + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                m = tl.load(M_ + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                w = (code.to(tl.float32) * d[:, None] + m[:, None]).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + b * 32 + t[None, :], mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q50_gemm(X, s_x, D, QH, Q, Y, s_y, M, N, K, BN: tl.constexpr, SB: tl.constexpr, F32: tl.constexpr):
        # Q5_0: nibble + a fifth bit per element (qh, bit i for element i) - 16, times fp16 d.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        nb = K // 32
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, nb, SB):
            for j in tl.static_range(SB):
                b = kb + j
                q = tl.load(Q + cols[:, None] * (nb * 16) + b * 16 + (t & 15)[None, :],
                            mask=cm[:, None], other=0)
                hb = tl.load(QH + cols[:, None] * nb + b, mask=cm[:, None], other=0)
                code = (((q >> ((t // 16) * 4).to(tl.int32)[None, :]) & 0xF)
                        | (((hb >> t[None, :]) & 1) << 4)).to(tl.float32) - 16.0
                d = tl.load(D + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                w = (code * d[:, None]).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + b * 32 + t[None, :], mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

    @triton.jit
    def _q51_gemm(X, s_x, D, M_, QH, Q, Y, s_y, M, N, K, BN: tl.constexpr, SB: tl.constexpr, F32: tl.constexpr):
        # Q5_1: Q5_0's codes, unsigned, with Q4_1's scale+min: w = q*d + m.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        t = tl.arange(0, 32)
        nb = K // 32
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for kb in range(0, nb, SB):
            for j in tl.static_range(SB):
                b = kb + j
                q = tl.load(Q + cols[:, None] * (nb * 16) + b * 16 + (t & 15)[None, :],
                            mask=cm[:, None], other=0)
                hb = tl.load(QH + cols[:, None] * nb + b, mask=cm[:, None], other=0)
                code = ((q >> ((t // 16) * 4).to(tl.int32)[None, :]) & 0xF) | (((hb >> t[None, :]) & 1) << 4)
                d = tl.load(D + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                m = tl.load(M_ + cols * nb + b, mask=cm, other=0.0).to(tl.float32)
                w = (code.to(tl.float32) * d[:, None] + m[:, None]).to(tl.bfloat16)
                x = tl.load(X + rows[:, None] * s_x + b * 32 + t[None, :], mask=rows[:, None] < M, other=0.0)
                acc += tl.dot(x, tl.trans(w))
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])

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
                ds0 = tl.load(DS + base, mask=cm, other=0.0).to(tl.float32)
                ds1 = tl.load(DS + base + 1, mask=cm, other=0.0).to(tl.float32)
                dm0 = tl.load(DM + base, mask=cm, other=0.0).to(tl.float32)
                dm1 = tl.load(DM + base + 1, mask=cm, other=0.0).to(tl.float32)
                hi = (t // 16)[None, :] == 1
                w = (code.to(tl.float32) * tl.where(hi, ds1[:, None], ds0[:, None])
                     - tl.where(hi, dm1[:, None], dm0[:, None])).to(tl.bfloat16)
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
                code = ((q >> (2 * (s % 4))) & 3).to(tl.int32) - 4 * (1 - ((hm >> s) & 1))
                base = (cols * NB + kb) * 16 + s * 2
                dl0 = tl.load(DL + base, mask=cm, other=0.0).to(tl.float32)
                dl1 = tl.load(DL + base + 1, mask=cm, other=0.0).to(tl.float32)
                w = (code.to(tl.float32)
                     * tl.where((t // 16)[None, :] == 1, dl1[:, None], dl0[:, None])).to(tl.bfloat16)
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


class _Packed:
    """Shared bits of the packed GGUF linears: `matmul` on the fused kernel up to ROWS_MAX rows, the
    bf16 twin on cuBLAS beyond — Q8Linear's contract. `raw` stays on the host for `dequant`."""

    def _twin(self):
        return self.dequant().to(torch.bfloat16)

    def dequant(self) -> torch.Tensor:                    # fp32 [N, K] on CPU
        return dequant(self.tid, self._raw, (self.N, self.K))

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        if M > ROWS_MAX or not HAVE_TRITON:
            y = torch.nn.functional.linear(x2, self._twin().to(x2.device))
            y = y.float() if f32 else y
            return (y if out is None else out.copy_(y)).view(*x.shape[:-1], self.N)
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        self._kernel(x2, out, M, f32)
        return out.view(*x.shape[:-1], self.N)


class Q40Linear(_Packed):
    """Packed Q4_0 [N, K] on the device: d fp16 [N, K/32], nibble bytes [N, K/32, 16]."""

    tid = 2

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 18)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(N, nb)).to(device)
        self.q = torch.from_numpy(np.ascontiguousarray(b[..., 2:])).to(device)   # [N, nb, 16] u8
        self.nbytes = N * nb * 18

    def _kernel(self, x2, out, M, f32):
        _q40_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.q, out,
                                                               out.stride(0), M, self.N, self.K,
                                                               BN=32, SB=4, F32=f32, num_warps=2, num_stages=3)


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
        _q6k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.ql, self.qh,
                                                               self.dl, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=32, F32=f32,
                                                               num_warps=1, num_stages=2)


class Q41Linear(_Packed):
    """Packed Q4_1 [N, K]: d fp16 [N, K/32], m fp16 [N, K/32], nibbles [N, K/32, 16]."""

    tid = 3

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 20)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.m = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 2:4])).astype(np.float16).reshape(N, nb)).to(device)
        self.q = torch.from_numpy(np.ascontiguousarray(b[..., 4:20])).to(device)
        self.nbytes = N * nb * 20

    def _kernel(self, x2, out, M, f32):
        _q41_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.m,
                                                               self.q, out, out.stride(0), M, self.N,
                                                               self.K, BN=32, SB=4, F32=f32,
                                                               num_warps=2, num_stages=3)


class Q50Linear(_Packed):
    """Packed Q5_0 [N, K]: d fp16 [N, K/32], qh u32 bits [N, K/32] (int32), nibbles [N, K/32, 16]."""

    tid = 6

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 22)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 2:6]).view(np.int32).reshape(N, nb)).to(device)
        self.q = torch.from_numpy(np.ascontiguousarray(b[..., 6:22])).to(device)
        self.nbytes = N * nb * 22

    def _kernel(self, x2, out, M, f32):
        _q50_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.qh,
                                                               self.q, out, out.stride(0), M, self.N,
                                                               self.K, BN=32, SB=4, F32=f32,
                                                               num_warps=2, num_stages=3)


class Q51Linear(_Packed):
    """Packed Q5_1 [N, K]: d/m fp16 [N, K/32] each, qh u32 bits [N, K/32] (int32), nibbles [N, K/32, 16]."""

    tid = 7

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        nb = K // 32
        b = np.ascontiguousarray(raw).reshape(N, nb, 24)
        self._raw = b.copy()
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 0:2])).astype(np.float16).reshape(N, nb)).to(device)
        self.m = torch.from_numpy(np.ascontiguousarray(_f16(b[..., 2:4])).astype(np.float16).reshape(N, nb)).to(device)
        self.qh = torch.from_numpy(np.ascontiguousarray(b[..., 4:8]).view(np.int32).reshape(N, nb)).to(device)
        self.q = torch.from_numpy(np.ascontiguousarray(b[..., 8:24])).to(device)
        self.nbytes = N * nb * 24

    def _kernel(self, x2, out, M, f32):
        _q51_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.d, self.m,
                                                               self.qh, self.q, out, out.stride(0),
                                                               M, self.N, self.K, BN=32, SB=4,
                                                               F32=f32, num_warps=1, num_stages=1)


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
        _q2k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.qs, self.ds,
                                                               self.dm, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=32, F32=f32,
                                                               num_warps=2, num_stages=3)


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
        _q3k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.qs, self.hm,
                                                               self.dl, out, out.stride(0), M, self.N,
                                                               self.K // 256, BN=32, F32=f32,
                                                               num_warps=2, num_stages=3)


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
        _q5k_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, 32))](x2, x2.stride(0), self.qs, self.qh,
                                                               self.ds, self.dm, out, out.stride(0),
                                                               M, self.N, self.K // 256, BN=32,
                                                               F32=f32, num_warps=2, num_stages=3)


#: 2-D tensors of these types load as packed GPU linears (Tier A); every other `TYPES` entry
#: dequantises to a dense tensor at load (Tier B).
PACKED_LINEAR = {2: Q40Linear, 3: Q41Linear, 6: Q50Linear, 7: Q51Linear,
                 10: Q2KLinear, 11: Q3KLinear, 12: Q4KLinear, 13: Q5KLinear, 14: Q6KLinear}
