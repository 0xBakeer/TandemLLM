"""Verify a block of drafted rows with the decode step's own arithmetic, for lossless speculation.

A verify row is only as good as its bits. Greedy picks the argmax, and a near-tie flips on one ULP,
so "close to the decode step" is not lossless: over a few hundred tokens some answer changes. The
rule here is stronger: row i of a verify block gives the logits the decode step would give for
that token, bit for bit, at the position it would have, after the same ancestors.

The decode step is not row-invariant with the tiled kernels a block of rows would take: the FP8
projections, the routed experts and the head run one-row kernels without `tl.dot`, and the decode
attention splits the keys into slices sized by the row's own position. So each of those kernels
has a TWIN here: the decode kernel's body, unchanged, in a loop over the rows a program serves, so
each row runs the same instructions on the same tile shapes with the same warps (and the program's
weight tile comes from L1/L2 after the first row):

  decode kernel (engine/kolibri/kernels.py,         twin here
  tools/kolibri_attn_kernels.py)
  `_fp8_gemv_1row`                                  `_fp8_gemv_rows`
  `_moe_gate_up_1row`, `_moe_down_1row`             `_moe_gate_up_rows`, `_moe_down_rows`
                                                    (rows grouped by expert, `moe_plan`)
  `_head_gemv_fp8` at one row                       `_head_gemv_rows`
  `_qk_prep_dec`, `_dec_split`, `_dec_combine`      `_qk_prep_ver`, `_ver_split`, `_ver_combine`

The rest is row-invariant already and is called as it is: the NVFP4 dense kernel (one row and up
to 16 rows run the same 16-row tile; more rows go in chunks of 16), the router's logits GEMV and
top-k (one program a row), the fused norms (one program a row), the SwiGLU kernel and the MoE
combine (one program a row).

THE CACHE IS NOT WRITTEN. A verify row's k and v go to a side buffer. The attention twin reads a
key slot from the side buffer when the decode step would have found that row's ancestor there
(slot (p + d) % R in the ring, row p + d in a full layer, for d up to the row's depth), and from
the cache otherwise. So a rejected row costs nothing to undo, and a tree's siblings never compete
for a slot. `commit(nodes)` copies the accepted path's k/v into the cache, where the decode step
would have written them, and sets the length.

`selfcheck()` proves the claim on the board at load: it decodes a few tokens one by one, then
verifies the same tokens as a chain and as a tree, and compares every row's logits bit for bit.
Any difference keeps speculation off (it names the first row that differs). A decode kernel that
changes without its twin turns speculation off; it cannot make it wrong.

`ReplayVerifier` is the reference semantics on any device: each node decoded alone at its
position (DFS order, truncating back to the node's depth), the path decoded again at commit. The
CPU tests and the GPU check compare against it.
"""

from __future__ import annotations

import math
import os
import time

import torch

from engine.kolibri import kernels as KK

try:
    import triton
    import triton.language as tl
    from engine.kolibri.kernels import _nv_tile
    HAVE_TRITON = KK.HAVE_TRITON
except ImportError:                                              # pragma: no cover
    HAVE_TRITON = False

#: the largest block a verify takes (the router GEMV and the small MoE tiles cover up to 32 rows)
MAX_ROWS = 32
#: pairs of one expert a MoE twin program takes (more pairs of the same expert: another tile)
MOE_BR = int(os.environ.get("KOLIBRI_VERIFY_MOE_BR", "4"))


def _pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


# ------------------------------------------------------------------------------ twins: weights
# Each twin is its decode kernel's body, unchanged, inside a loop over the rows the program serves:
# the same shapes, the same warps, the same reductions, so the same bits; the program's weight tile
# comes from L1/L2 after the first row. (Holding several rows' accumulators at once and decoding
# the tile once is cheaper, but it changes the reduction's register layout, and the lane
# accumulators of the newer one-row kernels do not fit more than one row.)
if HAVE_TRITON:

    @triton.jit
    def _fp8_gemv_rows(X, W, S, Y, M, N, K, sx, sy, swn, ssn,
                       BN: tl.constexpr, BK: tl.constexpr):
        """`_fp8_gemv_1row` for rows 0..M-1."""
        pn = tl.program_id(0)
        rn = pn * BN + tl.arange(0, BN)
        s_row = (pn * BN) // 128
        for m in range(0, M):
            acc = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, K, BK):
                kk = k0 + tl.arange(0, BK)
                x = tl.load(X + m * sx + kk).to(tl.float32)
                w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
                part = tl.reshape(w * x[None, :], (BN, BK // 128, 128))
                sc = tl.load(S + s_row * ssn + k0 // 128 + tl.arange(0, BK // 128))
                acc += tl.sum(tl.sum(part, 2) * sc[None, :], 1)
            tl.store(Y + m * sy + rn, acc.to(tl.bfloat16))

    @triton.jit
    def _moe_gate_up_rows(X, GC, GS, G2, UC, US, U2, Hout, ORDER, TE, TS, TL,
                          K, I, KTOP, sxm, swe, swn, sse, ssn, shm,
                          BN: tl.constexpr, BK: tl.constexpr):
        """`_moe_gate_up_1row` for each (row, slot) pair of a tile routed to one expert."""
        t = tl.program_id(0)
        pn = tl.program_id(1)
        ln = tl.load(TL + t)
        if ln == 0:
            return
        e = tl.load(TE + t)
        st = tl.load(TS + t)
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < I
        for m in range(0, ln):
            pair = tl.load(ORDER + st + m)
            row = pair // KTOP
            ag = tl.zeros((BN,), dtype=tl.float32)
            au = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, K, BK):
                kk = k0 + tl.arange(0, BK)
                x = tl.load(X + row * sxm + kk, mask=kk < K, other=0.0).to(tl.float32)
                wg = _nv_tile(GC, GS, e, rn, nm, k0, K, swe, swn, sse, ssn, BN, BK).to(tl.float32)
                ag += tl.sum(wg * x[None, :], 1)
                wu = _nv_tile(UC, US, e, rn, nm, k0, K, swe, swn, sse, ssn, BN, BK).to(tl.float32)
                au += tl.sum(wu * x[None, :], 1)
            g = ag * tl.load(G2 + e)
            u = au * tl.load(U2 + e)
            h = g * tl.sigmoid(g) * u
            tl.store(Hout + pair * shm + rn, h.to(tl.bfloat16), mask=nm)

    @triton.jit
    def _moe_down_rows(Hin, DC, DS, D2, RW, P, ORDER, TE, TS, TL,
                       I, H, shm, swe, swn, sse, ssn, spm,
                       BN: tl.constexpr, BK: tl.constexpr):
        """`_moe_down_1row` for each pair of a tile routed to one expert."""
        t = tl.program_id(0)
        pn = tl.program_id(1)
        ln = tl.load(TL + t)
        if ln == 0:
            return
        e = tl.load(TE + t)
        st = tl.load(TS + t)
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < H
        for m in range(0, ln):
            j = tl.load(ORDER + st + m)
            acc = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, I, BK):
                kk = k0 + tl.arange(0, BK)
                x = tl.load(Hin + j * shm + kk, mask=kk < I, other=0.0).to(tl.float32)
                w = _nv_tile(DC, DS, e, rn, nm, k0, I, swe, swn, sse, ssn, BN, BK).to(tl.float32)
                acc += tl.sum(w * x[None, :], 1)
            rw = tl.load(RW + j).to(tl.float32)
            tl.store(P + j * spm + rn, acc * tl.load(D2 + e) * rw, mask=nm)

    @triton.jit
    def _head_gemv_rows(X, W, S, Y, M, N, K, s_wn, BN: tl.constexpr, BK: tl.constexpr):
        """`tools/head_gemv._head_gemv_fp8` at BM = 1, for rows 0..M-1."""
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        rm = tl.arange(0, 1)
        mm = rm < 1
        for m in range(0, M):
            acc = tl.zeros((1, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                rk = k0 + tl.arange(0, BK)
                x = tl.load(X + (m + rm)[:, None] * K + rk[None, :], mask=mm[:, None],
                            other=0.0).to(tl.float32)
                w = tl.load(W + rn[:, None] * s_wn + rk[None, :]).to(tl.float32)
                acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
            s = tl.load(S + rn).to(tl.float32)
            tl.store(Y + (m + rm)[:, None] * N + rn[None, :], acc * s[None, :], mask=mm[:, None])


#: rows on a grid axis (1, default): each program is the one-row kernel for one row, rows of the
#: same tile launched side by side so the tile's second read comes from L2; 0: the row loop inside
#: one program. Both are the one-row body unchanged.
PAR = os.environ.get("KOLIBRI_VERIFY_PAR", "1") != "0"

if HAVE_TRITON:

    @triton.jit
    def _fp8_gemv_par(X, W, S, Y, N, K, sx, sy, swn, ssn, BN: tl.constexpr, BK: tl.constexpr):
        """`_fp8_gemv_1row` for row program_id(0) (rows vary fastest)."""
        m = tl.program_id(0)
        pn = tl.program_id(1)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), dtype=tl.float32)
        s_row = (pn * BN) // 128
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(X + m * sx + kk).to(tl.float32)
            w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
            part = tl.reshape(w * x[None, :], (BN, BK // 128, 128))
            sc = tl.load(S + s_row * ssn + k0 // 128 + tl.arange(0, BK // 128))
            acc += tl.sum(tl.sum(part, 2) * sc[None, :], 1)
        tl.store(Y + m * sy + rn, acc.to(tl.bfloat16))

    @triton.jit
    def _moe_gate_up_par(X, IDS, GC, GS, G2, UC, US, U2, Hout, K, I, KTOP, sxm, swe, swn, sse, ssn,
                         shm, BN: tl.constexpr, BK: tl.constexpr):
        """`_moe_gate_up_1row` for pair j = program_id(0) (row j // KTOP, slot j % KTOP)."""
        j = tl.program_id(0)
        pn = tl.program_id(1)
        e = tl.load(IDS + j)
        X = X + (j // KTOP) * sxm
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < I
        ag = tl.zeros((BN,), dtype=tl.float32)
        au = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(X + kk, mask=kk < K, other=0.0).to(tl.float32)
            wg = _nv_tile(GC, GS, e, rn, nm, k0, K, swe, swn, sse, ssn, BN, BK).to(tl.float32)
            ag += tl.sum(wg * x[None, :], 1)
            wu = _nv_tile(UC, US, e, rn, nm, k0, K, swe, swn, sse, ssn, BN, BK).to(tl.float32)
            au += tl.sum(wu * x[None, :], 1)
        g = ag * tl.load(G2 + e)
        u = au * tl.load(U2 + e)
        h = g * tl.sigmoid(g) * u
        tl.store(Hout + j * shm + rn, h.to(tl.bfloat16), mask=nm)

    @triton.jit
    def _head_gemv_par(X, W, S, Y, N, K, s_wn, BN: tl.constexpr, BK: tl.constexpr):
        """`tools/head_gemv._head_gemv_fp8` at BM = 1 for row program_id(0)."""
        m = tl.program_id(0)
        pid = tl.program_id(1)
        X = X + m * K
        Y = Y + m * N
        rn = pid * BN + tl.arange(0, BN)
        rm = tl.arange(0, 1)
        mm = rm < 1
        acc = tl.zeros((1, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            x = tl.load(X + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
            w = tl.load(W + rn[:, None] * s_wn + rk[None, :]).to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        s = tl.load(S + rn).to(tl.float32)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc * s[None, :], mask=mm[:, None])


# ------------------------------------------------------------------------------ twins: attention
if HAVE_TRITON:

    @triton.jit
    def _qk_prep_ver(Y, QN, KN, POS0, DEPTH, OQ, SK, SV, s_yt, s_st, EPS, LOG2_THETA,
                     NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr):
        """`_qk_prep_dec` for row t at position POS0 + DEPTH[t]: q to OQ[t], k and v to the side
        buffer row t (not the cache)."""
        t = tl.program_id(0)
        h = tl.program_id(1)
        isq = h < NQ
        col = tl.where(isq, h * D, NQ * D + (h - NQ) * D)
        half = tl.arange(0, D // 2)
        x1 = tl.load(Y + t * s_yt + col + half).to(tl.float32)
        x2 = tl.load(Y + t * s_yt + col + D // 2 + half).to(tl.float32)
        ms = (tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D
        r = 1.0 / tl.sqrt(ms + EPS)
        hk = tl.maximum(h - NQ, 0)
        w1 = tl.where(isq, tl.load(QN + half).to(tl.float32), tl.load(KN + half).to(tl.float32))
        w2 = tl.where(isq, tl.load(QN + D // 2 + half).to(tl.float32),
                      tl.load(KN + D // 2 + half).to(tl.float32))
        y1 = (x1 * r * w1).to(tl.bfloat16).to(tl.float32)
        y2 = (x2 * r * w2).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS0) + tl.load(DEPTH + t)
        if ROPE:
            p = pos.to(tl.float32)
            inv = tl.exp2(-LOG2_THETA * (2.0 * half.to(tl.float32) / D))
            f = p * inv
            c = tl.cos(f)
            sn = tl.sin(f)
            o1 = y1 * c - y2 * sn
            o2 = y2 * c + y1 * sn
        else:
            o1 = y1
            o2 = y2
        if isq:
            tl.store(OQ + (t * NQ + h) * D + half, o1.to(tl.bfloat16))
            tl.store(OQ + (t * NQ + h) * D + D // 2 + half, o2.to(tl.bfloat16))
        else:
            kb = SK + t * s_st + hk * D
            tl.store(kb + half, o1.to(tl.bfloat16))
            tl.store(kb + D // 2 + half, o2.to(tl.bfloat16))
            od = tl.arange(0, D)
            vv = tl.load(Y + t * s_yt + (NQ + NK) * D + hk * D + od)
            tl.store(SV + t * s_st + hk * D + od, vv)

    @triton.jit
    def _ver_split(Q, K, V, KIDX, SK, SV, POS0, DEPTH, ANC, PM, PL, PACC,
                   s_kh, s_kn, s_vh, s_vn, s_st, s_at, RMOD,
                   SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, NS: tl.constexpr, CH: tl.constexpr, RING: tl.constexpr,
                   WIN: tl.constexpr, NQ: tl.constexpr):
        """`_dec_split` for verify row t: the keys the decode step would see at the row's position,
        its ancestors' k/v taken from the side buffer at the slots decode would have put them."""
        sp = tl.program_id(0)
        kvh = tl.program_id(1)
        t = tl.program_id(2)
        nkv = tl.num_programs(1)
        p0 = tl.load(POS0)
        dep = tl.load(DEPTH + t)
        pos = p0 + dep
        rows = tl.arange(0, BM)
        rmask = rows < REP
        od = tl.arange(0, D)
        q = tl.load(Q + (t * NQ + kvh * REP + rows)[:, None] * D + od[None, :], mask=rmask[:, None],
                    other=0.0)
        if RING:
            c0 = sp * CH
            c1 = c0 + CH
            pr = p0 % RMOD
        else:
            n = pos + 1
            per = tl.cdiv(tl.cdiv(n, NS), BN) * BN
            c0 = sp * per
            c1 = tl.minimum(c0 + per, n)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            if RING:
                dd = (cols - pr + RMOD) % RMOD
            else:
                dd = cols - p0
            side = cmask & (dd >= 0) & (dd <= dep)
            node = tl.load(ANC + t * s_at + dd, mask=side, other=0)
            kc = tl.load(K + kvh * s_kh + cols.to(tl.int64)[:, None] * s_kn + od[None, :],
                         mask=cmask[:, None] & ~side[:, None], other=0.0)
            ksd = tl.load(SK + node[:, None] * s_st + kvh * D + od[None, :], mask=side[:, None],
                          other=0.0)
            k = tl.where(side[:, None], ksd, kc)
            s = tl.dot(q, tl.trans(k)) * SCALE
            if RING:
                cid = tl.load(KIDX + cols, mask=cmask, other=-1)
                cid = tl.where(side, p0 + dd, cid)
                vis = cmask & (cid >= 0) & (cid <= pos) & (cid >= pos - (WIN - 1))
            else:
                vis = cmask
            s = tl.where(vis[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            vc = tl.load(V + kvh * s_vh + cols.to(tl.int64)[:, None] * s_vn + od[None, :],
                         mask=cmask[:, None] & ~side[:, None], other=0.0)
            vsd = tl.load(SV + node[:, None] * s_st + kvh * D + od[None, :], mask=side[:, None],
                          other=0.0)
            v = tl.where(side[:, None], vsd, vc)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = ((t * NS + sp) * nkv + kvh) * BM + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _ver_combine(PM, PL, PACC, OUT, NQ: tl.constexpr, REP: tl.constexpr, BM: tl.constexpr,
                     D: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr):
        """`_dec_combine` for verify row t."""
        qh = tl.program_id(0)
        t = tl.program_id(1)
        nkv = NQ // REP
        kvh = qh // REP
        r = qh % REP
        sps = tl.arange(0, NSP)
        smask = sps < NS
        idx = t * (NS * nkv * BM) + (sps * nkv + kvh) * BM + r
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
        tl.store(OUT + t * (NQ * D) + qh * D + od, o.to(tl.bfloat16))


# ------------------------------------------------------------------------------ the verify step
def tree_tables(parents: list[int]) -> tuple[list[int], list[list[int]]]:
    """Depth of every node and, per node, its ancestor at each depth (itself at its own depth,
    -1 below none). `parents` in DFS pre-order, node 0 the anchor with parent -1."""
    n = len(parents)
    depth = [0] * n
    anc = [[-1] * n for _ in range(n)]
    for i in range(n):
        p = parents[i]
        if i == 0:
            assert p == -1, parents
        else:
            assert 0 <= p < i, f"node {i}: parent {p} breaks pre-order"
            depth[i] = depth[p] + 1
            anc[i][: depth[i]] = anc[p][: depth[i]]
        anc[i][depth[i]] = i
    return depth, anc


class _Bufs:
    """The device inputs of one verify size: ids, the anchor position, depths, ancestor table."""

    def __init__(self, M: int, dev):
        self.M = M
        self.i32 = torch.zeros(1 + M + M * M, dtype=torch.int32, device=dev)
        self.pos0 = self.i32[:1]
        self.depth = self.i32[1:1 + M]
        self.anc = self.i32[1 + M:].view(M, M)
        self.ids = torch.zeros(M, dtype=torch.long, device=dev)
        self.host = torch.zeros(1 + M + M * M, dtype=torch.int32).pin_memory() \
            if dev.type == "cuda" else torch.zeros(1 + M + M * M, dtype=torch.int32)
        self.host_ids = torch.zeros(M, dtype=torch.long).pin_memory() \
            if dev.type == "cuda" else torch.zeros(M, dtype=torch.long)

    def load(self, p: int, tokens: list[int], depth: list[int], anc: list[list[int]]) -> None:
        M = self.M
        h = self.host
        h[0] = p
        h[1:1 + M] = torch.tensor(depth, dtype=torch.int32)
        h[1 + M:] = torch.tensor(anc, dtype=torch.int32).reshape(-1)
        self.host_ids[:] = torch.tensor(tokens, dtype=torch.long)
        self.i32.copy_(h)
        self.ids.copy_(self.host_ids)


class Verifier:
    """Verify blocks on the GPU with the decode step's arithmetic; one CUDA graph per row count."""

    def __init__(self, eng, max_rows: int = MAX_ROWS, graphs: bool = True, moe_br: int = MOE_BR):
        self.eng = eng
        c = eng.cfg
        kv = eng.kv
        self.max_rows = int(max_rows)
        self.moe_br = int(moe_br)
        self.dev = eng.device
        self.graphs_on = bool(graphs)
        self.R = kv.ring.R
        nring, nfull = len(kv.ring.slot), len(kv.slot)
        sh = (self.max_rows, c.nkv, c.hd)
        bf = torch.bfloat16
        self.rk = torch.zeros(nring, *sh, dtype=bf, device=self.dev)
        self.rv = torch.zeros_like(self.rk)
        self.fk = torch.zeros(nfull, *sh, dtype=bf, device=self.dev)
        self.fv = torch.zeros_like(self.fk)
        self._graphs: dict[int, tuple] = {}
        self.hidden = None
        self.taps = None
        self._taps = None
        self.set_taps()
        self.stats = {"verifies": 0, "rows": 0, "commits": 0, "graph_ms": 0.0, "captures": 0}
        self.ok, self.why = self.supported()

    # --- what this needs
    def supported(self) -> tuple[bool, str]:
        from engine.kolibri import attn as A
        eng = self.eng
        if not HAVE_TRITON or eng.device.type != "cuda":
            return False, "no Triton GPU"
        if eng.kv.fp8:
            return False, "FP8 KV (the verify twins read a BF16 cache)"
        if not (A.DEC_FUSED and A.FUSED_QK) or not getattr(eng, "use_graphs", False):
            return False, "the decode step is not the fused, captured one"
        if not isinstance(eng.attn, A.KolibriAttention):
            return False, "reference attention"
        if not (KK.FUSED and KK.ONEROW):
            return False, "KOLIBRI_FUSED / KOLIBRI_ONEROW off"
        if eng.head.K % 256:
            return False, "head width"
        return True, ""

    # --- the twins, per op
    def _lin(self, lin, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """`lin.matmul(x)` row by row as the decode step computes it, for M rows."""
        from engine.kolibri.weights import Concat
        M = x.shape[0]
        rows = getattr(lin, "matmul_rows", None)
        if rows is not None:                          # the kernel owner's own M-row entry point
            return rows(x) if out is None else rows(x, out=out)
        if isinstance(lin, Concat):
            y = torch.empty(M, lin.N, dtype=torch.bfloat16, device=x.device) if out is None else out
            n0 = 0
            for p in lin.parts:
                self._lin(p, x, out=y[:, n0:n0 + p.N])
                n0 += p.N
            return y
        if isinstance(lin, KK.FP8Linear):
            cfg = KK.FP8_1ROW.get((lin.N, lin.K))
            if cfg is not None:
                bn1, bk1, w1, st1 = cfg
                x2 = x.to(torch.bfloat16).contiguous()
                y = torch.empty(M, lin.N, dtype=torch.bfloat16, device=x.device) if out is None else out
                if PAR:
                    _fp8_gemv_par[(M, lin.N // bn1)](x2, lin.w, lin.s, y, lin.N, lin.K, x2.stride(0),
                                                     y.stride(0), lin.w.stride(0), lin.s.stride(0),
                                                     BN=bn1, BK=bk1, num_warps=w1, num_stages=st1)
                    return y
                _fp8_gemv_rows[(lin.N // bn1,)](x2, lin.w, lin.s, y, M, lin.N, lin.K, x2.stride(0),
                                                y.stride(0), lin.w.stride(0), lin.s.stride(0),
                                                BN=bn1, BK=bk1, num_warps=w1, num_stages=st1)
                return y
        # the tiled kernels are row-invariant within one 16-row tile: chunks of 16
        y = torch.empty(M, lin.N, dtype=torch.bfloat16, device=x.device) if out is None else out
        for r0 in range(0, M, 16):
            lin.matmul(x[r0:r0 + 16], out=y[r0:r0 + 16])
        return y

    def _moe(self, x: torch.Tensor, lw) -> torch.Tensor:
        """`KolibriEngine._moe` for M rows with the one-row kernels' arithmetic."""
        from engine.kolibri.model import FUSED_SHARED
        eng = self.eng
        whole = getattr(eng, "_moe_rows", None)
        if whole is not None:                         # the engine's own M-row MoE block
            return whole(x, lw)
        sid = eng.shared_id if lw.shared is None else None
        ids, w = KK.route_fused(x, lw.gate, lw.bias, eng.cfg.topk, sid, eng.cfg.norm_topk_prob)
        extra = None
        if lw.shared is not None:
            if not FUSED_SHARED:
                raise RuntimeError("KOLIBRI_FUSED_SHARED=0 is not mirrored by the verify step")
            sh = lw.shared
            act_rows = getattr(sh, "act_rows", None)
            extra = (act_rows(x) if act_rows is not None
                     else self._lin(sh.d, KK.swiglu(self._lin(sh.gu, x), sh.F)))
        if hasattr(KK, "moe_experts_rows"):           # the kernel owner's own M-row entry point
            return KK.moe_experts_rows(x, ids, w, lw.G, lw.U, lw.D, extra=extra)
        M, H = x.shape
        k = ids.shape[1]
        G, U, D = lw.G, lw.U, lw.D
        I = G.N
        x = x.to(torch.bfloat16).contiguous()
        rw = w.float().reshape(-1).contiguous()
        (bn, bk, warps, stages), (bnd, bkd, warpsd, stagesd) = KK.MOE_1ROW
        if PAR:
            # a program per (row, slot) pair: the one-row kernels' bodies, x at row j // k
            idf = ids.reshape(-1).contiguous()
            h = torch.empty(M * k, I, dtype=torch.bfloat16, device=x.device)
            _moe_gate_up_par[(M * k, triton.cdiv(I, bn))](
                x, idf, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I, k,
                x.stride(0), G.codes.stride(0), G.codes.stride(1), G.scale.stride(0),
                G.scale.stride(1), h.stride(0), BN=bn, BK=bk, num_warps=warps, num_stages=stages)
            p = torch.empty(M * k, H, dtype=torch.float32, device=x.device)
            KK._moe_down_1row[(M * k, triton.cdiv(H, bnd))](
                h, idf, D.codes, D.scale, D.scale_2, rw, p, I, H, h.stride(0), D.codes.stride(0),
                D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
                BN=bnd, BK=bkd, num_warps=warpsd, num_stages=stagesd)
        else:
            p = self._moe_tiles(x, ids, rw, G, U, D)
        y = torch.empty(M, H, dtype=torch.float32, device=x.device)
        ex = extra if extra is not None else p
        KK._moe_combine_kernel[(M, triton.cdiv(H, 256))](
            p, y, H, k, p.stride(0), y.stride(0), ex, ex.stride(0) if extra is not None else 0,
            BN=256, EXTRA=extra is not None, num_warps=4)
        return y

    def _moe_tiles(self, x, ids, rw, G, U, D):
        M, H = x.shape
        k = ids.shape[1]
        I = G.N
        br = self.moe_br
        order, te, ts, tln = KK.moe_plan(ids, G.E, br)
        T = te.numel()
        (bn, bk, warps, stages), (bnd, bkd, warpsd, stagesd) = KK.MOE_1ROW
        h = torch.empty(M * k, I, dtype=torch.bfloat16, device=x.device)
        _moe_gate_up_rows[(T, triton.cdiv(I, bn))](
            x, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, order, te, ts, tln,
            H, I, k, x.stride(0), G.codes.stride(0), G.codes.stride(1), G.scale.stride(0),
            G.scale.stride(1), h.stride(0), BN=bn, BK=bk, num_warps=warps, num_stages=stages)
        p = torch.empty(M * k, H, dtype=torch.float32, device=x.device)
        _moe_down_rows[(T, triton.cdiv(H, bnd))](
            h, D.codes, D.scale, D.scale_2, rw, p, order, te, ts, tln, I, H, h.stride(0),
            D.codes.stride(0), D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
            BN=bnd, BK=bkd, num_warps=warpsd, num_stages=stagesd)
        return p

    def _attn(self, x: torch.Tensor, lw, b: _Bufs) -> torch.Tensor:
        """`KolibriAttention.attn_decode` (the fused path) for M rows of a block."""
        from tools import kolibri_attn_kernels as AK
        s = self.eng.attn.spec
        kv = self.eng.kv
        M = x.shape[0]
        y = self._lin(lw.qkv, x)
        if lw.sliding:
            r = kv.ring
            i = r.slot[lw.index]
            kc, vc, idx = r.k[i][0], r.v[i][0], r.idx
            sk, sv = self.rk[i], self.rv[i]
        else:
            i = kv.slot[lw.index]
            kc, vc, idx = kv.k[i][0], kv.v[i][0], b.pos0
            sk, sv = self.fk[i], self.fv[i]
        nq, nk, d = s.nq, s.nkv, s.hd
        q = torch.empty(M, nq, d, dtype=torch.bfloat16, device=x.device)
        _qk_prep_ver[(M, nq + nk)](y, lw.q_norm, lw.k_norm, b.pos0, b.depth, q, sk, sv,
                                   y.stride(0), sk.stride(0), float(s.eps), math.log2(s.rope_theta),
                                   NQ=nq, NK=nk, D=d, ROPE=bool(lw.sliding), num_warps=1)
        rep = nq // nk
        bm = max(16, triton.next_power_of_2(rep))
        bn = AK.DEC_BN
        N = kc.shape[1]
        if lw.sliding:
            ch = math.gcd(N, AK.RING_CH)
            ch = ch if ch >= 16 else N
            ns = N // ch
        else:
            ns, ch = AK.FULL_NS, bn
        pm = torch.empty(M * ns * nk * bm, dtype=torch.float32, device=x.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(M * ns * nk * bm, d, dtype=torch.float32, device=x.device)
        _ver_split[(ns, nk, M)](q, kc, vc, idx, sk, sv, b.pos0, b.depth, b.anc, pm, pl, pacc,
                                kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
                                sk.stride(0), b.anc.stride(0), N,
                                SCALE=self.eng.attn.scale, REP=rep, D=d, BM=bm, BN=min(bn, ch),
                                NS=ns, CH=ch, RING=bool(lw.sliding), WIN=s.window, NQ=nq,
                                num_warps=AK.DEC_WARPS, num_stages=2)
        out = torch.empty(M, nq * d, dtype=torch.bfloat16, device=x.device)
        _ver_combine[(nq, M)](pm, pl, pacc, out, NQ=nq, REP=rep, BM=bm, D=d, NS=ns,
                              NSP=triton.next_power_of_2(ns), num_warps=4)
        return self._lin(lw.o, out)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        hd = self.eng.head
        if hasattr(hd, "logits_rows"):                # the kernel owner's own M-row entry point
            return hd.logits_rows(x)
        M = x.shape[0]
        xf = x.float().contiguous()
        y = torch.empty(M, hd.N, dtype=torch.float32, device=x.device)
        if PAR:
            _head_gemv_par[(M, triton.cdiv(hd.N, 32))](xf, hd.w, hd.s, y, hd.N, hd.K, hd.w.stride(0),
                                                       BN=32, BK=256, num_warps=4)
            return y
        _head_gemv_rows[(triton.cdiv(hd.N, 32),)](xf, hd.w, hd.s, y, M, hd.N, hd.K, hd.w.stride(0),
                                                  BN=32, BK=256, num_warps=4)
        return y

    def _forward(self, b: _Bufs) -> torch.Tensor:
        """`KolibriEngine._hidden(decode=True)` + the head, for the block in `b`."""
        eng = self.eng
        c = eng.cfg
        r = eng.emb[b.ids].float()
        x = KK.rms_fused(r, eng.layers[0].n_in, c.eps)
        n = len(eng.layers)
        for i, lw in enumerate(eng.layers):
            a = self._attn(x, lw, b)
            r, x2 = KK.add_rms2(r, a, lw.n_pa, lw.n_pal, c.eps)
            y = self._moe(x2, lw)
            if i + 1 < n:
                r, x = KK.add_rms2(r, y, lw.n_pf, eng.layers[i + 1].n_in, c.eps)
            else:
                r, x = KK.add_rms2(r, y, lw.n_pf, eng.final_norm, c.eps, torch.float32)
            j = getattr(eng, "_tap_at", {}).get(i)
            if j is not None:
                self._taps[j, :r.shape[0]].copy_(r)
        self.hidden = x
        return self._head(x)

    def set_taps(self) -> None:
        """Follow the engine's tap layers (`KolibriEngine.set_taps`): after each verify, `taps`
        [n_taps, n, H] fp32 holds the residual after those layers for the block's rows. The graphs
        are captured again."""
        n = len(getattr(self.eng, "tap_layers", ()))
        self._taps = (torch.zeros(n, self.max_rows, self.eng.cfg.hidden, dtype=torch.float32,
                                  device=self.dev) if n else None)
        self._graphs = {}

    # --- the public calls
    def _graph(self, M: int):
        g = self._graphs.get(M)
        if g is not None:
            return g
        b = _Bufs(M, self.dev)
        # a harmless block for the warm-up and the capture: anchor at the current length, chain
        p = self.eng.kv.length
        depth, anc = tree_tables([-1] + list(range(M - 1)))
        b.load(p, [0] * M, depth, anc)
        if not self.graphs_on:
            self._graphs[M] = (None, b, None, None)
            return self._graphs[M]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                self._forward(b)
        torch.cuda.current_stream().wait_stream(st)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._forward(b)
        self._graphs[M] = (graph, b, out, self.hidden)
        self.stats["captures"] += 1
        return self._graphs[M]

    def capture(self, sizes) -> float:
        """Capture the graphs for these row counts now (not in a client's request); seconds."""
        t0 = time.perf_counter()
        for M in sizes:
            if 2 <= M <= self.max_rows:
                self._graph(int(M))
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    @torch.inference_mode()
    def verify(self, tokens: list[int], parents: list[int]) -> torch.Tensor:
        """fp32 logits [n, V], row i = what the decode step gives for node i after its ancestors
        (node 0 at position kv.length). Writes no cache row; `commit` keeps a path."""
        n = len(tokens)
        assert 1 <= n <= self.max_rows, n
        p = self.eng.kv.length
        assert p + n <= self.eng.max_len, (p, n)
        depth, anc = tree_tables(parents)
        graph, b, out, hid = self._graph(max(n, 2))
        if b.M != n:                                     # one row: pad with a second (unused)
            depth, anc = tree_tables(list(parents) + [0])
            tokens = list(tokens) + [tokens[0]]
        b.load(p, tokens, depth, anc)
        self._last = (p, n, b.M)
        self.stats["verifies"] += 1
        self.stats["rows"] += n
        if graph is None:
            lg = self._forward(b)[:n]
            self.hidden = self.hidden[:n]
            self.taps = self._taps[:, :n] if self._taps is not None else None
            return lg
        graph.replay()
        # the final-normed hidden rows of this block (fp32 [n, H]) and the taps, for a drafter
        # that reads them; valid until the next verify
        self.hidden = hid[:n]
        self.taps = self._taps[:, :n] if self._taps is not None else None
        return out[:n]

    @torch.inference_mode()
    def commit(self, nodes: list[int]) -> None:
        """Keep the path `nodes` (node 0 first, each the next one's parent) of the last verify: their
        k/v go into the cache where the decode step writes them; kv.length = p + len(nodes)."""
        p, n, _ = self._last
        a = len(nodes)
        assert a >= 1 and nodes[0] == 0 and all(0 <= j < n for j in nodes), nodes
        kv = self.eng.kv
        dev = self.dev
        src = torch.tensor(nodes, dtype=torch.long, device=dev)
        pos = torch.arange(p, p + a, dtype=torch.long, device=dev)
        r = kv.ring
        slots = pos % r.R
        # ring [40, 1, nk, R, d] <- side [40, rows, nk, d]
        r.k[:, 0].index_copy_(2, slots, self.rk.index_select(1, src).transpose(1, 2))
        r.v[:, 0].index_copy_(2, slots, self.rv.index_select(1, src).transpose(1, 2))
        r.idx[slots] = pos.to(torch.int32)
        kv.k[:, 0].index_copy_(2, pos, self.fk.index_select(1, src).transpose(1, 2))
        kv.v[:, 0].index_copy_(2, pos, self.fv.index_select(1, src).transpose(1, 2))
        kv.length = p + a
        r.primed = True
        self.stats["commits"] += 1

    # --- the proof, on the board
    @torch.inference_mode()
    def selfcheck(self, ids: list[int], steps: int = 12, log=print) -> tuple[bool, str]:
        """Decode `steps` tokens after `ids` one by one; verify the same tokens as a chain and as a
        tree with siblings; every row's logits must be bit-equal. Leaves the engine reset."""
        if not self.ok:
            return False, self.why
        eng = self.eng
        eng.reset()
        eng.prefill(ids)
        p = eng.kv.length
        toks = [int(ids[-1])]
        eng.truncate(p - 1)
        rows = []
        t = toks[0]
        for _ in range(steps):
            lg = eng.decode(t).clone()
            rows.append(lg)
            t = int(lg.argmax())
            toks.append(t)
        toks = toks[:steps]                               # node i = toks[i] at position p-1+i
        eng.truncate(p - 1)
        lg = self.verify(toks, [-1] + list(range(steps - 1))).clone()
        for i in range(steps):
            if not torch.equal(lg[i], rows[i]):
                d = (lg[i] - rows[i]).abs().max().item()
                eng.reset()
                return False, f"chain row {i} differs from decode (max |d| {d:.3g})"
        # a tree in DFS pre-order: the chain's first half, a sibling of its last node, and a
        # second child of the anchor (other tokens, so they route to other experts)
        h = steps // 2
        V = eng.cfg.vocab
        tokens = toks[:h] + [(toks[h - 1] + 7) % V, (toks[1] + 11) % V]
        parents = [-1] + list(range(h - 1)) + [h - 2, 0]
        lt = self.verify(tokens, parents).clone()
        ref = ReplayVerifier(eng).verify(tokens, parents)
        for i in range(len(tokens)):
            if not torch.equal(lt[i], ref[i]):
                d = (lt[i] - ref[i]).abs().max().item()
                eng.reset()
                return False, f"tree node {i} differs from decode (max |d| {d:.3g})"
        # a commit then a decode equals the plain decode
        eng.truncate(p - 1)
        self.verify(toks[:4], [-1, 0, 1, 2])
        self.commit([0, 1, 2])
        nxt = eng.decode(toks[3]).clone()
        eng.reset()
        if not torch.equal(nxt, rows[3]):
            return False, "decode after a commit differs"
        log(f"[kolibri-spec] self-check: {steps}-row chain, {len(tokens)}-node tree and a commit "
            f"bit-equal to the decode step")
        return True, ""


class ReplayVerifier:
    """The reference semantics on any device: each node decoded alone at its own position after its
    ancestors (DFS order, cut back to the node's depth first), the path decoded again at commit.
    Exact by construction, `n` decode steps a verify; for the CPU tests and the GPU check."""

    max_rows = MAX_ROWS
    ok, why = True, ""

    def __init__(self, eng):
        self.eng = eng
        self.stats = {"verifies": 0, "rows": 0, "commits": 0}

    def capture(self, sizes) -> float:
        return 0.0

    @torch.inference_mode()
    def verify(self, tokens: list[int], parents: list[int]) -> torch.Tensor:
        eng = self.eng
        p = eng.kv.length
        depth, _ = tree_tables(parents)
        out = []
        for i, t in enumerate(tokens):
            eng.truncate(p + depth[i])
            out.append(eng.decode(int(t)).clone())
        eng.truncate(p)
        self._last = (p, list(tokens))
        self.stats["verifies"] += 1
        self.stats["rows"] += len(tokens)
        return torch.stack(out)

    @torch.inference_mode()
    def commit(self, nodes: list[int]) -> None:
        p, tokens = self._last
        eng = self.eng
        eng.truncate(p)
        for j in nodes:
            eng.decode(int(tokens[j]))
        self.stats["commits"] += 1


def make_verifier(eng, log=print, selfcheck_ids: list[int] | None = None):
    """The GPU verifier when the board can prove it exact, else None (speculation off)."""
    v = Verifier(eng)
    if not v.ok:
        log(f"[kolibri-spec] off: {v.why}")
        return None
    if selfcheck_ids is not None:
        ok, why = v.selfcheck(selfcheck_ids, log=log)
        if not ok:
            log(f"[kolibri-spec] off: self-check failed: {why}")
            return None
    return v
