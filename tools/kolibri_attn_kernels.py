"""Decode and verify attention for Kolibri-1's sliding-window layers, over the ring.

`tools/attn_kernels.py` reads a cache whose column c holds row c. A sliding layer keeps only the
last R rows (engine/kolibri/attn.py `SlidingRing`): slot s holds the row whose sequence index is
`idx[s]` (-1 empty), and slots are in no useful order. Attention does not care about the order of
its keys, only which ones a row may see, so this kernel walks the R slots as they lie and decides
visibility by index:

    committed  idx < start, idx >= lo[t]      (lo[t] = the row's position - (window - 1))
    the block  start <= idx < start + T, block_mask[t, idx - start]
    otherwise  invisible (an empty slot, or a rejected row past the length)

Everything else is the flash-decoding kernel of `tools/attn_kernels.py`: one program a (row group,
KV head, slice of slots), the 12 query heads that share a KV head and the T rows of the block
flattened into rows of one `tl.dot`, a second kernel combining the slices. The slices are cut at the
same slots whatever T is (R is fixed), so a row verified in a block gets the bits it gets decoded
alone: the same keys, in the same order, through the same tiles.

`ring_reference` is the same attention in float32 torch, for `check()` and the CPU tests.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _ring_split(Q, K, V, KS, VS, KIDX, QLO, BMASK, OUT, PM, PL, PACC,
                    T, R, NKV, START, LC, CHUNK,
                    s_qh, s_qt, s_kh, s_kn, s_vh, s_vn, s_sh, s_oh, s_ot,
                    SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr, DIRECT: tl.constexpr,
                    STARTP=None, DEVSTART: tl.constexpr = False):
        if DEVSTART:
            # the block's start from the device, so a captured decode step serves every position
            START = tl.load(STARTP)
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
        lo = tl.load(QLO + t, mask=rmask, other=0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, LC)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            cid = tl.load(KIDX + cols, mask=cmask, other=-1)
            k = tl.load(K + kvh * s_kh + cols[:, None] * s_kn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            if FP8:
                ks = tl.load(KS + kvh * s_sh + cols, mask=cmask, other=0.0)
                s = s * ks[None, :]
            s = s * SCALE
            inblk = cid - START
            isblk = (inblk >= 0) & (inblk < T)
            bm = tl.load(BMASK + t[:, None] * T + inblk[None, :],
                         mask=rmask[:, None] & isblk[None, :] & cmask[None, :], other=0)
            com = (cid >= 0) & (cid < START)
            vis = (com[None, :] & (cid[None, :] >= lo[:, None])) | (isblk[None, :] & (bm != 0))
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


def ring_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor,
                   start: int, qlo: torch.Tensor, block_mask: torch.Tensor, *,
                   ks: torch.Tensor | None = None, vs: torch.Tensor | None = None,
                   scale: float | None = None, bn: int = 32, num_warps: int = 4,
                   num_stages: int = 2, bm: int = 32,
                   startp: torch.Tensor | None = None) -> torch.Tensor:
    """q [1, Hq, T, D] bf16; the ring k, v [1, Hkv, R, D]; idx int32 [R]; qlo int32 [T];
    block_mask [T, T] bool. With ks/vs ([1, Hkv, R] fp32) the ring holds e4m3 codes. With
    `startp` (int32 [1] on the device) the start is read there and `start` is ignored: nothing
    about the launch depends on the position, so the call can be captured in a CUDA graph.
    Returns [1, Hq, T, D] bf16 laid out so `.transpose(1, 2)` is contiguous."""
    from tools.attn_kernels import _attn_combine, pick_launch
    _, hq, T, D = q.shape
    _, hkv, R_slots, _ = k.shape
    rep = hq // hkv
    fp8 = ks is not None
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    out = torch.empty(T, hq, D, dtype=torch.bfloat16, device=q.device)
    bm_, groups, ns, chunk = pick_launch(T, rep, R_slots, bm=bm)
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
    _ring_split[(groups, hkv, ns)](
        q0, k0, v0, ks0, vs0, idx, qlo.to(torch.int32).contiguous(), bmask, out, pm, pl, pacc,
        T, R, hkv, start, R_slots, chunk,
        q0.stride(0), q0.stride(1), k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
        s_sh, out.stride(1), out.stride(0),
        SCALE=scale, REP=rep, D=D, BM=bm_, BN=bn, FP8=fp8, DIRECT=direct,
        STARTP=startp if startp is not None else idx, DEVSTART=startp is not None,
        num_warps=num_warps, num_stages=num_stages)
    if not direct:
        _attn_combine[(hkv, R)](pm, pl, pacc, out, T, R, hkv, ns, out.stride(1), out.stride(0),
                                REP=rep, D=D, NSP=triton.next_power_of_2(ns), num_warps=4)
    return out.unsqueeze(0).transpose(1, 2)


def ring_reference(q, k, v, idx, start, qlo, block_mask, ks=None, vs=None, scale=None):
    """The same attention in float32 torch. Shapes as `ring_attention`."""
    _, hq, T, D = q.shape
    hkv = k.shape[1]
    kf = k.float() * (ks[..., None] if ks is not None else 1.0)
    vf = v.float() * (vs[..., None] if vs is not None else 1.0)
    rep = hq // hkv
    kf = kf.repeat_interleave(rep, dim=1)
    vf = vf.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = torch.einsum("bhtd,bhnd->bhtn", q.float(), kf) * scale
    cid = idx.long()
    com = (cid >= 0) & (cid < start)
    vis = com[None, :] & (cid[None, :] >= qlo.long()[:, None])
    j = cid - start
    blk = (j >= 0) & (j < T)
    jj = j.clamp(0, T - 1)
    bmv = block_mask[:, jj] & blk[None, :]
    vis = vis | bmv
    s = s.masked_fill(~vis, float("-inf"))
    return torch.einsum("bhtn,bhnd->bhtd", s.softmax(-1), vf)


def check(Ts=(1, 4, 16, 33), lengths=(5, 300, 513, 700, 5000), *, R=640, window=513,
          fp8: bool = False, seed: int = 0, device: str = "cuda") -> list[dict]:
    """Max |d| against the float32 reference, and row independence (a row in a block against the
    same row decoded alone), on a ring filled as a sequence of `length` rows would leave it."""
    from engine.kolibri.attn import quantize_kv
    torch.manual_seed(seed)
    out = []
    for L in lengths:
        for T in Ts:
            start = L
            total = start + T
            n = min(total, R)
            idx = torch.full((R,), -1, dtype=torch.int32, device=device)
            ar = torch.arange(total - n, total, device=device)
            idx[ar % R] = ar.to(torch.int32)
            q = torch.randn(1, 48, T, 128, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, 4, R, 128, device=device, dtype=torch.bfloat16)
            v = torch.randn(1, 4, R, 128, device=device, dtype=torch.bfloat16)
            ks = vs = None
            if fp8:
                k, ks = quantize_kv(k)
                v, vs = quantize_kv(v)
            tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
            qlo = torch.arange(start, start + T, device=device, dtype=torch.int32) - (window - 1)
            ref = ring_reference(q, k, v, idx, start, qlo, tri, ks, vs)
            got = ring_attention(q, k, v, idx, start, qlo, tri, ks=ks, vs=vs).float()
            row = {"len": L, "T": T, "max_abs": (got - ref).abs().max().item()}
            if T > 1:
                one = torch.cat([ring_attention(
                    q[:, :, t:t + 1], k, v, _idx_upto(idx, start + t + 1), start + t,
                    qlo[t:t + 1], torch.ones(1, 1, dtype=torch.bool, device=device),
                    ks=ks, vs=vs) for t in range(T)], dim=2)
                row["row_indep_abs"] = (one.float() - got).abs().max().item()
            out.append(row)
    return out


def _idx_upto(idx: torch.Tensor, n: int) -> torch.Tensor:
    """The ring's index table as a decode step at length n-1 sees it: rows >= n not written yet
    (they hold what was there before, which here is invisible either way)."""
    out = idx.clone()
    out[out >= n] = -1
    return out


# ------------------------------------------------------------------------------ q/k norm + RoPE
if HAVE_TRITON:

    @triton.jit
    def _qk_prep(Y, QN, KN, POS, OQ, OK, s_yt, EPS, LOG2_THETA,
                 NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr):
        """One program per (row, head) over the q heads then the k heads of the fused qkv output:
        RMSNorm in fp32 rounded to bf16 (as `rms_heads`), then NeoX RoPE in fp32 rounded to bf16
        (as `rope`), written contiguous."""
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
        if ROPE:
            p = tl.load(POS + t).to(tl.float32)
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
            base = OQ + (t * NQ + h) * D
        else:
            base = OK + (t * NK + hk) * D
        tl.store(base + half, o1.to(tl.bfloat16))
        tl.store(base + D // 2 + half, o2.to(tl.bfloat16))


def qk_prep(y: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor, positions: torch.Tensor,
            nq: int, nk: int, d: int, theta: float, eps: float, rope: bool):
    """y [T, (nq + 2 nk) d] bf16 (the fused qkv output) -> q [T, nq, d], k [T, nk, d] bf16, normed
    and (sliding layers) rotated; one launch for both. `positions` int [T] on the device."""
    T = y.shape[0]
    q = torch.empty(T, nq, d, dtype=torch.bfloat16, device=y.device)
    k = torch.empty(T, nk, d, dtype=torch.bfloat16, device=y.device)
    _qk_prep[(T, nq + nk)](y, q_norm, k_norm, positions, q, k, y.stride(0), float(eps),
                           math.log2(theta), NQ=nq, NK=nk, D=d, ROPE=bool(rope), num_warps=1)
    return q, k


def check_qk_prep(T=37, device="cuda") -> dict:
    """Max |d| of `qk_prep` against `rms_heads` + `rope` (engine/kolibri/attn.py)."""
    from engine.kolibri.attn import rms_heads, rope
    torch.manual_seed(0)
    y = torch.randn(T, 7168, device=device, dtype=torch.bfloat16) * 3
    qn = (1 + 0.1 * torch.randn(128, device=device)).to(torch.bfloat16)
    kn = (1 + 0.1 * torch.randn(128, device=device)).to(torch.bfloat16)
    pos = torch.arange(100000, 100000 + T, device=device)
    out = {}
    for r in (False, True):
        q, k = qk_prep(y, qn, kn, pos, 48, 4, 128, 10000.0, 1e-6, r)
        rq = rms_heads(y[:, :6144].reshape(T, 48, 128), qn, 1e-6)
        rk = rms_heads(y[:, 6144:6656].reshape(T, 4, 128), kn, 1e-6)
        if r:
            rq, rk = rope(rq, pos, 10000.0), rope(rk, pos, 10000.0)
        out["rope" if r else "norm"] = max((q.float() - rq.float()).abs().max().item(),
                                            (k.float() - rk.float()).abs().max().item())
    return out


# ------------------------------------------------------------------- the decode step, fused
# A decode step (one row, BF16 cache) in three launches a layer instead of about fourteen:
#
#   _qk_prep_dec   q/k norm + RoPE as `_qk_prep`, and the row's k and v written straight into the
#                  cache (the ring slot pos % R and its index entry, or full-layer row pos)
#   _dec_split     flash decoding over NS slices of the keys, every slice its own program
#   _dec_combine   the NS partial softmaxes merged, one program a query head
#
# The unfused path runs the ring kernel in 2 slices of 512 slots for 4 KV heads: 8 programs,
# each walking 16 tiles one after the other. Here the ring is cut into
# R / CH slices (fixed: the ring does not grow), and a full layer's 0..pos into NS slices whose
# width follows pos on the device, so every context length gets NS * 4 programs and the launch
# stays capturable. The row's own k/v are written by the launch before the one that reads them.
if HAVE_TRITON:

    @triton.jit
    def _qk_prep_dec(Y, QN, KN, POSP, OQ, KC, VC, IDX, s_kh, s_kn, s_vh, s_vn, RMOD, EPS,
                     LOG2_THETA, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                     ROPE: tl.constexpr, RING: tl.constexpr):
        """One program a head (q heads, then k heads) of the single row's fused qkv output. q and k
        exactly as `_qk_prep` computes them; a k program also copies its head's v."""
        h = tl.program_id(0)
        isq = h < NQ
        col = tl.where(isq, h * D, NQ * D + (h - NQ) * D)
        half = tl.arange(0, D // 2)
        x1 = tl.load(Y + col + half).to(tl.float32)
        x2 = tl.load(Y + col + D // 2 + half).to(tl.float32)
        ms = (tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D
        r = 1.0 / tl.sqrt(ms + EPS)
        hk = tl.maximum(h - NQ, 0)
        w1 = tl.where(isq, tl.load(QN + half).to(tl.float32), tl.load(KN + half).to(tl.float32))
        w2 = tl.where(isq, tl.load(QN + D // 2 + half).to(tl.float32),
                      tl.load(KN + D // 2 + half).to(tl.float32))
        y1 = (x1 * r * w1).to(tl.bfloat16).to(tl.float32)
        y2 = (x2 * r * w2).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POSP)
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
            tl.store(OQ + h * D + half, o1.to(tl.bfloat16))
            tl.store(OQ + h * D + D // 2 + half, o2.to(tl.bfloat16))
        else:
            if RING:
                slot = pos % RMOD
            else:
                slot = pos
            kb = KC + hk * s_kh + slot.to(tl.int64) * s_kn
            tl.store(kb + half, o1.to(tl.bfloat16))
            tl.store(kb + D // 2 + half, o2.to(tl.bfloat16))
            od = tl.arange(0, D)
            vv = tl.load(Y + (NQ + NK) * D + hk * D + od)
            tl.store(VC + hk * s_vh + slot.to(tl.int64) * s_vn + od, vv)
            if RING:
                if hk == 0:
                    tl.store(IDX + slot, pos)

    @triton.jit
    def _dec_split(Q, K, V, KIDX, POSP, PM, PL, PACC, s_kh, s_kn, s_vh, s_vn,
                   SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, NS: tl.constexpr, CH: tl.constexpr, RING: tl.constexpr,
                   WIN: tl.constexpr):
        sp = tl.program_id(0)
        kvh = tl.program_id(1)
        nkv = tl.num_programs(1)
        pos = tl.load(POSP)
        rows = tl.arange(0, BM)
        rmask = rows < REP
        od = tl.arange(0, D)
        q = tl.load(Q + (kvh * REP + rows)[:, None] * D + od[None, :], mask=rmask[:, None],
                    other=0.0)
        if RING:
            c0 = sp * CH
            c1 = c0 + CH
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
            k = tl.load(K + kvh * s_kh + cols.to(tl.int64)[:, None] * s_kn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * SCALE
            if RING:
                cid = tl.load(KIDX + cols, mask=cmask, other=-1)
                vis = cmask & (cid >= 0) & (cid <= pos) & (cid >= pos - (WIN - 1))
            else:
                vis = cmask
            s = tl.where(vis[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + cols.to(tl.int64)[:, None] * s_vn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = (sp * nkv + kvh) * BM + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _dec_combine(PM, PL, PACC, OUT, REP: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
                     NS: tl.constexpr, NSP: tl.constexpr):
        qh = tl.program_id(0)
        nkv = tl.num_programs(0) // REP
        kvh = qh // REP
        r = qh % REP
        sps = tl.arange(0, NSP)
        smask = sps < NS
        idx = (sps * nkv + kvh) * BM + r
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
        tl.store(OUT + qh * D + od, o.to(tl.bfloat16))


#: decode launch shapes: ring slice width (R / RING_CH programs a KV head), full-layer slices,
#: key tile, and the warps of the split kernel
RING_CH = 64
FULL_NS = 16
DEC_BN = 32
DEC_WARPS = 4


def decode_fused(y: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor, posp: torch.Tensor,
                 kc: torch.Tensor, vc: torch.Tensor, idx: torch.Tensor | None, *, nq: int, nk: int,
                 d: int, theta: float, eps: float, rope: bool, window: int, scale: float,
                 ring_ch: int | None = None, full_ns: int | None = None, bn: int | None = None,
                 num_warps: int | None = None) -> torch.Tensor:
    """One decode row: y [1, (nq + 2 nk) d] bf16, the fused qkv output; `posp` int32 [1] on the
    device (the row's position). `kc`, `vc` the layer's BF16 cache [1, nk, N, d]: a ring of N slots
    with its index table `idx` (sliding layers), or N = max_len rows (`idx` None). Writes the row's
    k, v (and idx) and returns the attention output [1, nq * d] bf16. Graph-capturable."""
    ring = idx is not None
    ring_ch = ring_ch or RING_CH
    full_ns = full_ns or FULL_NS
    bn = bn or DEC_BN
    num_warps = num_warps or DEC_WARPS
    dev = y.device
    k0, v0 = kc[0], vc[0]
    N = k0.shape[1]
    q = torch.empty(nq, d, dtype=torch.bfloat16, device=dev)
    _qk_prep_dec[(nq + nk,)](y, q_norm, k_norm, posp, q, k0, v0, idx if ring else posp,
                             k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
                             N, float(eps), math.log2(theta), NQ=nq, NK=nk, D=d, ROPE=bool(rope),
                             RING=ring, num_warps=1)
    rep = nq // nk
    bm = max(16, triton.next_power_of_2(rep))
    if ring:
        # slices of ring_ch slots when they tile the ring, else the whole ring in one slice
        ch = math.gcd(N, ring_ch)
        ch = ch if ch >= 16 else N
        ns = N // ch
    else:
        ns, ch = full_ns, bn
    pm = torch.empty(ns * nk * bm, dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    pacc = torch.empty(ns * nk * bm, d, dtype=torch.float32, device=dev)
    _dec_split[(ns, nk)](q, k0, v0, idx if ring else posp, posp, pm, pl, pacc,
                         k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
                         SCALE=scale, REP=rep, D=d, BM=bm, BN=min(bn, ch), NS=ns, CH=ch, RING=ring,
                         WIN=window, num_warps=num_warps, num_stages=2)
    out = torch.empty(1, nq * d, dtype=torch.bfloat16, device=dev)
    _dec_combine[(nq,)](pm, pl, pacc, out, REP=rep, BM=bm, D=d, NS=ns,
                        NSP=triton.next_power_of_2(ns), num_warps=4)
    return out
