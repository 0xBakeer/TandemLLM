"""Attention for a decode step or a verify block, reading each cached key and value ONCE.

A verify block is at most seventeen query rows against a context that is a prompt long. Below
`GQA_FROM` rows the engine used to hand SDPA a `repeat_interleave`d cache: the four key/value heads
widened to the twenty-four query heads, which writes the whole context six times over and reads it
back, per layer, per step. At a few hundred tokens of context that is a 3.7 MB copy the SDPA kernel
liked having. At 32k it is 384 MB a tensor a layer, and the step reads
about 26 GB of it against 15 GB of weights.

This kernel indexes instead. One program owns one key/value head, one slice of the context and a
group of query rows, where a "row" is a (query head, block position) pair: the six query heads that
share a key/value head and the T positions of the block are flattened into up to 6 x 17 = 102 rows,
so each key and value tile is loaded once for all of them and fed to `tl.dot`. The context is split
across programs (flash-decoding) and a second, small kernel combines the slices.

The mask is the engine's: every column before `start` is visible to every row, and inside the block
row t sees column start + j where `block_mask[t, j]` -- the causal triangle for a chain, the
ancestor relation for a tree.

Optionally the cache is e4m3 with one fp32 scale per (head, token) for keys and one for values:
half the bytes of bf16. The scales fold into the kernel for free -- the key scale
multiplies a score column, the value scale multiplies a probability column -- so the dot products
run on the unscaled codes.

The arithmetic order differs from SDPA's, so the output is not bit-identical to the path it
replaces. It IS row-independent: a row's result does not depend on how many other rows share the
launch, which is what keeps the decode step and the verify block on the same numbers and the
losslessness gate meaningful. `check()` compares against a float32 reference.
"""

from __future__ import annotations

import math
import os
import sys

import torch

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402

# Three software-pipeline stages for the split-KV loop instead of two: the same arithmetic (the
# stage count only moves when a tile's loads are issued), 2-4 % faster from 8k to 128k of context
# (tools/attn_sweep.py, 2026-10-09). QWEN38_ATTN_STAGES3=0 restores two.
STAGES3 = _S.get("ATTN_STAGES3") == "1"

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

FP8_MAX = 448.0


if HAVE_TRITON:

    @triton.jit
    def _attn_split(Q, K, V, KS, VS, BMASK, OUT, PM, PL, PACC,
                    T, R, NKV, START, LC, CHUNK,
                    s_qh, s_qt, s_kh, s_kn, s_vh, s_vn, s_sh, s_oh, s_ot, LENP,
                    SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr, DIRECT: tl.constexpr,
                    DEVLEN: tl.constexpr = False):
        if DEVLEN:
            # the block's start and the context length from the device, so a captured
            # graph serves every position; splits past the length run no iteration and hand the
            # combine an empty (m = -inf, l = 0) slice, which it weighs exactly zero
            START = tl.load(LENP)
            LC = tl.load(LENP + 1)
        rg = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        rows = rg * BM + tl.arange(0, BM)                    # flattened (group, position)
        rmask = rows < R
        g = rows // T
        t = rows % T
        qh = kvh * REP + g
        od = tl.arange(0, D)
        q = tl.load(Q + qh[:, None] * s_qh + t[:, None] * s_qt + od[None, :],
                    mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, LC)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            k = tl.load(K + kvh * s_kh + cols[:, None] * s_kn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            if FP8:
                ks = tl.load(KS + kvh * s_sh + cols, mask=cmask, other=0.0)
                s = s * ks[None, :]
            s = s * SCALE
            inblk = cols - START
            bm = tl.load(BMASK + t[:, None] * T + inblk[None, :],
                         mask=rmask[:, None] & (inblk[None, :] >= 0) & cmask[None, :], other=0)
            vis = (cols[None, :] < START) | (bm != 0)
            vis = vis & cmask[None, :] & rmask[:, None]
            s = tl.where(vis, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + cols[:, None] * s_vn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            if FP8:
                vs = tl.load(VS + kvh * s_sh + cols, mask=cmask, other=0.0)
                p = p * vs[None, :]
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        if DIRECT:
            o = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
            tl.store(OUT + t[:, None] * s_ot + qh[:, None] * s_oh + od[None, :],
                     o.to(tl.bfloat16), mask=rmask[:, None])
        else:
            base = sp * (NKV * R) + kvh * R + rows
            tl.store(PM + base, m_i, mask=rmask)
            tl.store(PL + base, l_i, mask=rmask)
            tl.store(PACC + base[:, None] * D + od[None, :], acc, mask=rmask[:, None])

    @triton.jit
    def _attn_inv(Q, K, V, ANC, s_anc, PM, PL, PACC, T, R, NKV, START, CHUNK,
                  s_qh, s_qt, s_kh, s_kn, s_vh, s_vn, LENP,
                  SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  TP: tl.constexpr, DEVLEN: tl.constexpr):
        """SPD-201 (ATTN_INV): one token's query heads (BM rows, REP real) over the columns of its own positions
        0 .. its position, in blocks and splits cut by position: column j < START is the cache's row j, column
        START + d the token's ancestor at depth d (the tree's row, from the ancestor mask: a depth is a node's
        ancestor count less one). So a row's statements are the same wherever the last commit fell and whatever
        else the tree holds. The partial (m, l, acc) of each split goes to `_attn_combine`, as `_attn_split`'s."""
        if DEVLEN:
            START = tl.load(LENP)
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        g = tl.arange(0, BM)
        gmask = g < REP
        qh = kvh * REP + g
        od = tl.arange(0, D)
        jj = tl.arange(0, TP)
        jm = jj < T
        anc = tl.load(ANC + jj[:, None] * s_anc + jj[None, :], mask=jm[:, None] & jm[None, :], other=0) != 0
        dep = tl.sum(anc.to(tl.int32), axis=1) - 1                       # [TP] each node's depth
        arow = tl.load(ANC + t * s_anc + jj, mask=jm, other=0) != 0       # the token's ancestors
        dt = tl.sum(arow.to(tl.int32), axis=0) - 1
        lcv = START + dt + 1                                              # the columns the token sees
        # the chunk of the token's own length class (512 to 32,768 columns, then a 64th of the class), not of
        # the launch's: the same token is cut at the same columns whatever the block around it
        cls = 1024
        while cls < lcv:
            cls *= 2
        CHUNK = tl.maximum(512, cls // 64)
        q = tl.load(Q + qh[:, None] * s_qh + t * s_qt + od[None, :], mask=gmask[:, None], other=0.0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, lcv)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            dcol = cols - START
            hit = arow[None, :] & (dep[None, :] == dcol[:, None])         # [BN, TP]
            node = tl.sum(tl.where(hit, jj[None, :], 0), axis=1)
            phys = tl.where(cols < START, cols, START + node)
            k = tl.load(K + kvh * s_kh + phys[:, None] * s_kn + od[None, :], mask=cmask[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * SCALE
            sc = tl.where(cmask[None, :] & gmask[:, None], sc, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            pr = tl.exp(sc - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(pr, 1)
            v = tl.load(V + kvh * s_vh + phys[:, None] * s_vn + od[None, :], mask=cmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        r = g * T + t
        base = sp * (NKV * R) + kvh * R + r
        tl.store(PM + base, m_i, mask=gmask)
        tl.store(PL + base, l_i, mask=gmask)
        tl.store(PACC + base[:, None] * D + od[None, :], acc, mask=gmask[:, None])

    @triton.jit
    def _attn_combine(PM, PL, PACC, OUT, T, R, NKV, NS, s_oh, s_ot,
                      REP: tl.constexpr, D: tl.constexpr, NSP: tl.constexpr):
        kvh = tl.program_id(0)
        r = tl.program_id(1)
        g = r // T
        t = r % T
        qh = kvh * REP + g
        stride = NKV * R
        sps = tl.arange(0, NSP)
        smask = sps < NS
        idx = sps * stride + kvh * R + r
        m = tl.load(PM + idx, mask=smask, other=-float("inf"))
        l = tl.load(PL + idx, mask=smask, other=0.0)
        mx = tl.max(m, 0)
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        w = tl.where(smask, tl.exp(m - mx), 0.0)
        od = tl.arange(0, D)
        acc = tl.load(PACC + idx[:, None] * D + od[None, :], mask=smask[:, None], other=0.0)
        num = tl.sum(acc * w[:, None], 0)
        den = tl.sum(l * w, 0)
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + t * s_ot + qh * s_oh + od, o.to(tl.bfloat16))


#: SPD-201: a verify's attention row by row over its own positions (`_attn_inv`): a token's bits do not depend
#: on the tree around it or on where the last commit fell
INV = _S.get("ATTN_INV") == "1"


def attention_inv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start, block_mask: torch.Tensor,
                  chunk: int, ns: int, *, lenp: torch.Tensor | None = None, scale: float | None = None,
                  bn: int = 32, num_warps: int = 4, num_stages: int | None = None) -> torch.Tensor:
    """`decode_attention`'s contract through `_attn_inv` and `_attn_combine` (`lenp`: the start from the device,
    `decode_attention_dev`'s). bf16 cache."""
    _, hq, T, D = q.shape
    _, hkv, _, _ = k.shape
    rep = hq // hkv
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    num_stages = (3 if STAGES3 else 2) if num_stages is None else num_stages
    # splits enough for every token's own chunk: a 512-column chunk to 32,768 columns, past it 64 of them
    lc_max = ns * chunk
    ns = max(2, -(-lc_max // 512) if lc_max <= 32768 else 64)
    R = rep * T
    out = torch.empty(T, hq, D, dtype=torch.bfloat16, device=q.device)
    anc = block_mask.to(torch.int8).contiguous()
    q0, k0, v0 = q[0], k[0], v[0]
    pm = torch.empty(ns * R * hkv, dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    pacc = torch.empty(ns * R * hkv, D, dtype=torch.float32, device=q.device)
    _attn_inv[(T, hkv, ns)](q0, k0, v0, anc, anc.stride(0), pm, pl, pacc, T, R, hkv,
                            0 if lenp is not None else int(start), chunk,
                            q0.stride(0), q0.stride(1), k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
                            lenp if lenp is not None else q0,
                            SCALE=scale, REP=rep, D=D, BM=16, BN=bn, TP=max(16, triton.next_power_of_2(T)),
                            DEVLEN=lenp is not None, num_warps=num_warps, num_stages=num_stages)
    _attn_combine[(hkv, R)](pm, pl, pacc, out, T, R, hkv, ns, out.stride(1), out.stride(0),
                            REP=rep, D=D, NSP=triton.next_power_of_2(ns), num_warps=4)
    return out.unsqueeze(0).transpose(1, 2)


def pick_launch(T: int, rep: int, lc: int, *, min_chunk: int = 512, max_splits: int = 64,
                bm: int = 32):
    """Row-group size, number of row groups, the context chunk and the number of chunks.

    The chunk is a function of the context length alone, and a power of two times `min_chunk`, so
    the single-token step at context p and the verify block whose row sees p columns cut the context
    at the same places. Columns a row may not see add exact zeros, so a row's arithmetic is then the
    same whether it was decoded alone or verified in a block: the step and the verify agree bit for
    bit except where the chunk doubles (at 32k, 64k and 128k of context).
    """
    R = rep * T
    # One row-group height for every T. The first board check (2026-09-23 09:42) used 16 rows for
    # a decode step and 32 for a block, and a row's output then differed by one bf16 ulp between
    # the two at 300 tokens of context: a different `tl.dot` shape is a different reduction.
    groups = -(-R // bm)
    per = -(-lc // max_splits)
    chunk = max(min_chunk, 1 << (per - 1).bit_length())
    return bm, groups, -(-lc // chunk), chunk


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int,
                     block_mask: torch.Tensor, *, ks: torch.Tensor | None = None,
                     vs: torch.Tensor | None = None, scale: float | None = None,
                     bn: int = 32, num_warps: int = 4, num_stages: int | None = None, bm: int = 32,
                     max_splits: int = 64) -> torch.Tensor:
    """Softmax(q k^T * scale, masked) v, with q [1, Hq, T, D] and the cache k, v [1, Hkv, L, D].

    `L` is the context INCLUDING the block (start + T). `block_mask` is [T, T] bool: row t may see
    block column j. Returns [1, Hq, T, D] bf16, laid out so `.transpose(1, 2)` is contiguous.
    With `ks`/`vs` ([1, Hkv, >=L] fp32) the cache is e4m3 codes and these are its scales.
    """
    _, hq, T, D = q.shape
    _, hkv, lc, _ = k.shape
    rep = hq // hkv
    assert hq == 24 or hq == rep * hkv, (hq, hkv)
    fp8 = ks is not None
    if INV and not fp8 and rep <= 16:
        _, _, ns_, chunk_ = pick_launch(T, rep, lc, bm=bm, max_splits=max_splits)
        return attention_inv(q, k, v, start, block_mask, chunk_, ns_, scale=scale, bn=bn, num_warps=num_warps,
                             num_stages=num_stages)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    num_stages = (3 if STAGES3 else 2) if num_stages is None else num_stages
    out = torch.empty(T, hq, D, dtype=torch.bfloat16, device=q.device)
    bm, groups, ns, chunk = pick_launch(T, rep, lc, bm=bm, max_splits=max_splits)
    R = rep * T
    bmask = block_mask.to(torch.int8).contiguous()
    q0, k0, v0 = q[0], k[0], v[0]
    ks0 = ks[0] if fp8 else k0
    vs0 = vs[0] if fp8 else k0
    s_sh = ks0.stride(0) if fp8 else 0
    direct = ns == 1
    if direct:
        pm = pl = pacc = out
    else:
        pm = torch.empty(ns * R * hkv, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(ns * R * hkv, D, dtype=torch.float32, device=q.device)
    _attn_split[(groups, hkv, ns)](
        q0, k0, v0, ks0, vs0, bmask, out, pm, pl, pacc,
        T, R, hkv, start, lc, chunk,
        q0.stride(0), q0.stride(1), k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
        s_sh, out.stride(1), out.stride(0),
        q0, SCALE=scale, REP=rep, D=D, BM=bm, BN=bn, FP8=fp8, DIRECT=direct,
        num_warps=num_warps, num_stages=num_stages)
    if not direct:
        _attn_combine[(hkv, R)](pm, pl, pacc, out, T, R, hkv, ns, out.stride(1), out.stride(0),
                                REP=rep, D=D, NSP=triton.next_power_of_2(ns), num_warps=4)
    return out.unsqueeze(0).transpose(1, 2)


def dev_chunk(T: int, rep: int, max_lc: int) -> int:
    """The chunk a captured verify cuts the context at, for a context class `max_lc`.

    A class is a power of two from 1,024 and holds the lengths (max_lc / 2, max_lc]; over those the
    eager path's chunk is constant -- 512 up to 32,768, then max_lc / 64 -- so the graph cuts where
    the eager step does at every length it serves. Anything else is refused."""
    assert max_lc >= 1024 and max_lc & (max_lc - 1) == 0, f"not a context class: {max_lc}"
    chunk = pick_launch(T, rep, max_lc)[3]
    assert chunk == max(512, max_lc // 64), (max_lc, chunk)
    return chunk


def decode_attention_dev(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lenp: torch.Tensor,
                         block_mask: torch.Tensor, max_lc: int, *, scale: float | None = None,
                         bn: int = 32, num_warps: int = 4,
                         num_stages: int | None = None) -> torch.Tensor:
    """`decode_attention` with the start and the context length in a device tensor.

    `lenp` is int32 [start, start + T]; `k`, `v` are the whole cache buffers [1, Hkv, max_len, D];
    `max_lc` bounds the length for the launch. The chunk is `pick_launch`'s for `max_lc`, which is
    512 up to 32,768 tokens of context and max_lc / 64 past it -- the same chunk the eager path uses
    at any length in the class (`dev_chunk`), so a captured verify cuts the context where the eager
    one does. One split more or fewer
    adds exactly nothing; and where the eager path, at 512 tokens or fewer, divides in the kernel
    (`DIRECT`), the combine of one real split and empty ones produces the same bits. bf16 cache only.
    """
    _, hq, T, D = q.shape
    _, hkv, _, _ = k.shape
    rep = hq // hkv
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    num_stages = (3 if STAGES3 else 2) if num_stages is None else num_stages
    bm, groups, ns, chunk = pick_launch(T, rep, max_lc)
    dev_chunk(T, rep, max_lc)
    if INV and rep <= 16:
        return attention_inv(q, k, v, None, block_mask, chunk, ns, lenp=lenp, scale=scale, bn=bn,
                             num_warps=num_warps, num_stages=num_stages)
    ns = max(ns, 2)
    R = rep * T
    out = torch.empty(T, hq, D, dtype=torch.bfloat16, device=q.device)
    bmask = block_mask.to(torch.int8).contiguous()
    q0, k0, v0 = q[0], k[0], v[0]
    pm = torch.empty(ns * R * hkv, dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    pacc = torch.empty(ns * R * hkv, D, dtype=torch.float32, device=q.device)
    _attn_split[(groups, hkv, ns)](
        q0, k0, v0, k0, k0, bmask, out, pm, pl, pacc,
        T, R, hkv, 0, 0, chunk,
        q0.stride(0), q0.stride(1), k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
        0, out.stride(1), out.stride(0), lenp,
        SCALE=scale, REP=rep, D=D, BM=bm, BN=bn, FP8=False, DIRECT=False, DEVLEN=True,
        num_warps=num_warps, num_stages=num_stages)
    _attn_combine[(hkv, R)](pm, pl, pacc, out, T, R, hkv, ns, out.stride(1), out.stride(0),
                            REP=rep, D=D, NSP=triton.next_power_of_2(ns), num_warps=4)
    return out.unsqueeze(0).transpose(1, 2)


def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., D] bf16 -> e4m3 codes and one fp32 scale per vector (amax / 448)."""
    amax = x.float().abs().amax(-1).clamp_min(1e-12)
    s = amax / FP8_MAX
    codes = (x.float() / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, s


def reference(q, k, v, start, block_mask, ks=None, vs=None, scale=None):
    """The same attention in float32 torch, with the cache dequantised when it is e4m3."""
    _, hq, T, D = q.shape
    hkv, lc = k.shape[1], k.shape[2]
    kf = k.float() * (ks[..., :lc, None] if ks is not None else 1.0)
    vf = v.float() * (vs[..., :lc, None] if vs is not None else 1.0)
    rep = hq // hkv
    kf = kf.repeat_interleave(rep, dim=1)
    vf = vf.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = torch.einsum("bhtd,bhnd->bhtn", q.float(), kf) * scale
    vis = torch.zeros(T, lc, dtype=torch.bool, device=q.device)
    vis[:, :start] = True
    vis[:, start:start + T] = block_mask
    s = s.masked_fill(~vis, float("-inf"))
    return torch.einsum("bhtn,bhnd->bhtd", s.softmax(-1), vf)


def check(ctx=(1, 300, 4096, 33000), Ts=(1, 8, 16, 17), *, fp8: bool = False, seed: int = 0,
          device: str = "cuda") -> list[dict]:
    """Max |d| against the float32 reference, beside the bf16 SDPA path's own distance from it."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    out = []
    for L in ctx:
        for T in Ts:
            start = L
            lc = start + T
            q = torch.randn(1, 24, T, 256, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, 4, lc, 256, device=device, dtype=torch.bfloat16)
            v = torch.randn(1, 4, lc, 256, device=device, dtype=torch.bfloat16)
            # a tree mask with a branch, and the causal triangle on the diagonal
            bm = torch.ones(T, T, dtype=torch.bool, device=device).tril()
            if T >= 4:
                bm[T // 2:, 1:T // 2] = False
                bm[T // 2:, 0] = True
            ks = vs = None
            if fp8:
                k, ks = quantize_kv(k)
                v, vs = quantize_kv(v)
            ref = reference(q, k, v, start, bm, ks, vs)
            got = decode_attention(q, k, v, start, bm, ks=ks, vs=vs).float()
            row = {"ctx": L, "T": T, "max_abs": (got - ref).abs().max().item(),
                   "ref_absmax": ref.abs().max().item()}
            if not fp8:
                vis = torch.zeros(T, lc, dtype=torch.bool, device=device)
                vis[:, :start] = True
                vis[:, start:] = bm
                sd = F.scaled_dot_product_attention(q, k.repeat_interleave(6, 1),
                                                    v.repeat_interleave(6, 1), attn_mask=vis)
                row["sdpa_abs"] = (sd.float() - ref).abs().max().item()
            # row independence, on the chain triangle: row t verified in the block against the
            # same row decoded alone at context start + t + 1. Bit-identical by construction.
            if T > 1:
                tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
                blk = decode_attention(q, k, v, start, tri, ks=ks, vs=vs)
                one = torch.cat([decode_attention(
                    q[:, :, t:t + 1], k[:, :, :start + t + 1], v[:, :, :start + t + 1], start + t,
                    torch.ones(1, 1, dtype=torch.bool, device=device), ks=ks, vs=vs)
                    for t in range(T)], dim=2)
                row["row_indep_abs"] = (one.float() - blk.float()).abs().max().item()
            out.append(row)
    return out


def bench(ctx=(4096, 8192, 32768, 131072), Ts=(1, 16), *, layers: int = 16, reps: int = 5,
          device: str = "cuda") -> list[dict]:
    """Milliseconds for the sixteen attention layers of one step, three ways, on distinct buffers.

    Each layer gets its own cache, so the loop reads 16 layers' worth of KV cold, as a step does:
    the SDPA path as the engine ships it below 64 rows (`repeat_interleave`, then a boolean mask),
    this kernel on the bf16 cache, and this kernel on the e4m3 one.
    """
    import time as _t
    import torch.nn.functional as F
    out = []
    for L in ctx:
        for T in Ts:
            start = L - T
            q = torch.randn(1, 24, T, 256, device=device, dtype=torch.bfloat16)
            ks_ = [torch.randn(1, 4, L, 256, device=device, dtype=torch.bfloat16)
                   for _ in range(layers)]
            vs_ = [torch.randn(1, 4, L, 256, device=device, dtype=torch.bfloat16)
                   for _ in range(layers)]
            tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
            vis = torch.zeros(T, L, dtype=torch.bool, device=device)
            vis[:, :start] = True
            vis[:, start:] = tri

            def sdpa():
                for k, v in zip(ks_, vs_):
                    F.scaled_dot_product_attention(q, k.repeat_interleave(6, 1),
                                                   v.repeat_interleave(6, 1),
                                                   attn_mask=None if T == 1 else vis)

            def mine():
                for k, v in zip(ks_, vs_):
                    decode_attention(q, k, v, start, tri)

            q8 = [quantize_kv(k) for k in ks_]
            v8 = [quantize_kv(v) for v in vs_]

            def mine8():
                for (k, sk), (v, sv) in zip(q8, v8):
                    decode_attention(q, k, v, start, tri, ks=sk, vs=sv)

            row = {"ctx": L, "T": T}
            for name, fn in (("sdpa", sdpa), ("kernel", mine), ("kernel_fp8", mine8)):
                try:
                    fn()
                    torch.cuda.synchronize()
                    best = 1e9
                    for _ in range(reps):
                        t0 = _t.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        best = min(best, _t.perf_counter() - t0)
                    row[name] = best * 1e3
                except torch.OutOfMemoryError:
                    row[name] = float("nan")
                    torch.cuda.empty_cache()
            bytes_bf16 = layers * 2 * 4 * L * 256 * 2
            row["kv_gb"] = bytes_bf16 / 1e9
            print(f"ctx {L:>7} T {T:>2}  kv {row['kv_gb']:6.2f} GB   sdpa {row['sdpa']:8.2f} ms   "
                  f"kernel {row['kernel']:7.2f} ms ({bytes_bf16 / row['kernel'] / 1e6:6.1f} GB/s)   "
                  f"fp8 {row['kernel_fp8']:7.2f} ms", flush=True)
            out.append(row)
            del ks_, vs_, q8, v8
            torch.cuda.empty_cache()
    return out


def sweep(ctx=(32768, 131072), Ts=(1, 16), configs=None, *, layers: int = 16, reps: int = 5,
          device: str = "cuda") -> list[dict]:
    """The first question, before any new kernel: how far the launch of THIS kernel is from
    its best at long context. Milliseconds for the sixteen layers of one step, per launch config
    (row-group height, key tile, warps, stages, splits), on distinct buffers as `bench` runs them,
    and each config's max |d| from the float32 reference on one layer. A config is a candidate
    only if it holds for every T at a length (a row decoded alone and in a block must keep the
    same launch) -- the table says which."""
    import time as _t
    configs = configs or [dict(bm=32, bn=32, num_warps=4, num_stages=2, max_splits=64)]
    out = []
    for L in ctx:
        for T in Ts:
            start = L - T
            q = torch.randn(1, 24, T, 256, device=device, dtype=torch.bfloat16)
            ks_ = [torch.randn(1, 4, L, 256, device=device, dtype=torch.bfloat16)
                   for _ in range(layers)]
            vs_ = [torch.randn(1, 4, L, 256, device=device, dtype=torch.bfloat16)
                   for _ in range(layers)]
            tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
            ref = reference(q, ks_[0], vs_[0], start, tri)
            for cfg in configs:
                row = {"ctx": L, "T": T, **cfg}
                try:
                    err = (decode_attention(q, ks_[0], vs_[0], start, tri, **cfg).float()
                           - ref).abs().max().item()
                    for k, v in zip(ks_, vs_):
                        decode_attention(q, k, v, start, tri, **cfg)
                    torch.cuda.synchronize()
                    best = 1e9
                    for _ in range(reps):
                        t0 = _t.perf_counter()
                        for k, v in zip(ks_, vs_):
                            decode_attention(q, k, v, start, tri, **cfg)
                        torch.cuda.synchronize()
                        best = min(best, _t.perf_counter() - t0)
                    row.update(ms=best * 1e3, max_abs=err)
                except Exception as e:                            # a config that does not compile
                    row.update(ms=float("nan"), error=repr(e)[:160])
                print(f"ctx {L:>7} T {T:>2}  {cfg}  {row.get('ms', float('nan')):8.2f} ms  "
                      f"err {row.get('max_abs', float('nan')):.2e}", flush=True)
                out.append(row)
            del ks_, vs_
            torch.cuda.empty_cache()
    return out


def check_inv(seed: int = 0) -> list[str]:
    """SPD-201 on the board: six tokens' attention as (A) a 6-row chain verified at S, (B) three committed and a
    3-row chain at S + 3, (C) the six on a path of an 8-node tree beside two siblings -- every token's output bit
    for bit the same; the device-length launch equals the eager one; and against float32 torch."""
    global INV
    from engine.tree import DraftTree
    g = torch.Generator(device="cuda").manual_seed(seed)
    hq, hkv, D = 24, 4, 256
    out = []
    keep = INV
    INV = True
    try:
        for S in (37, 500, 511, 1200, 3000, 32765, 40000):
            L = S + 16 + 64
            kc = (torch.randn(1, hkv, L, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
            vc = (torch.randn(1, hkv, L, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
            tk = (torch.randn(8, hkv, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)   # the six + 2 siblings
            tv = (torch.randn(8, hkv, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
            tq = (torch.randn(8, hq, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)

            def run(start, rows, parents, dev=False):
                k, v = kc.clone(), vc.clone()
                n = len(rows)
                k[0, :, start:start + n] = tk[rows].transpose(0, 1)
                v[0, :, start:start + n] = tv[rows].transpose(0, 1)
                anc = torch.tensor(DraftTree(tokens=[0] * n, parents=parents).ancestor_mask(), dtype=torch.bool,
                                   device="cuda")
                q = tq[rows].transpose(0, 1)[None]
                if dev:
                    lenp = torch.tensor([start, start + n], dtype=torch.int32, device="cuda")
                    lc = 1024
                    while lc < start + n:
                        lc *= 2
                    return decode_attention_dev(q, k, v, lenp, anc, lc)[0].transpose(0, 1), k, v
                return decode_attention(q, k[..., :start + n, :], v[..., :start + n, :], start, anc)[0].transpose(0, 1), k, v
            oA, kA, vA = run(S, [0, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3, 4])
            # B: the first three committed at S .. S+2, then three verified at S + 3
            kc2, vc2 = kc.clone(), vc.clone()
            kc2[0, :, S:S + 3] = tk[:3].transpose(0, 1)
            vc2[0, :, S:S + 3] = tv[:3].transpose(0, 1)
            kcs, vcs = kc, vc
            kc, vc = kc2, vc2
            oB, _, _ = run(S + 3, [3, 4, 5], [-1, 0, 1])
            oBd, _, _ = run(S + 3, [3, 4, 5], [-1, 0, 1], dev=True)
            kc, vc = kcs, vcs
            oC, _, _ = run(S, [0, 1, 6, 2, 3, 7, 4, 5], [-1, 0, 1, 1, 3, 4, 4, 6])
            oAd, _, _ = run(S, [0, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3, 4], dev=True)
            assert torch.equal(oA[3:], oB), f"S={S}: chain vs committed + chain"
            assert torch.equal(oB, oBd) and torch.equal(oA, oAd), f"S={S}: eager vs device length"
            assert torch.equal(oA, oC[[0, 1, 3, 4, 6, 7]]), f"S={S}: chain vs tree path"
            # float32 reference for the chain
            kf = kA[0, :, :S + 6].float().repeat_interleave(hq // hkv, 0)
            vf = vA[0, :, :S + 6].float().repeat_interleave(hq // hkv, 0)
            qf = tq[:6].transpose(0, 1).float()
            sc = torch.einsum("htd,hnd->htn", qf, kf) / math.sqrt(D)
            pos = torch.arange(S + 6, device="cuda")
            sc = sc.masked_fill(pos[None, None, :] > (S + torch.arange(6, device="cuda"))[None, :, None], float("-inf"))
            ref = torch.einsum("htn,hnd->htd", sc.softmax(-1), vf).transpose(0, 1)
            err = ((oA.float() - ref).abs().max() / ref.abs().max()).item()
            out.append(f"context {S}: chain = committed + chain = tree path, eager = device length, bit for bit; "
                       f"max relative vs fp32 {err:.1e}")
    finally:
        INV = keep
    return out
