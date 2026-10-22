"""Flint's row kernels: what Flint adds to LFM2's (engine/models/lfm2_moe/kernels.py, which keeps the
norms, q/k prep, BF16 projections and experts). Triton, one program a row (or a row's KV head and chunk),
no reduction across rows and no tile shape that depends on the row count, so a row of an M-row verify
block is the decode step's row bit for bit.

  attn_rows        softmax over [q k^T / sqrt(D) of the row's visible keys, the head's sink logit] times
                   v: the cache's positions before the block's start L, then the row's ancestors (`anc`);
                   on a sliding layer only the keys at positions > pos - W, read from the layer's ring
                   (slot = position mod R). Fixed chunks of CHUNK keys, one program each, combined in
                   chunk order with the sink's exp() in the denominator. A full layer launches the
                   context class's chunks from position 0 (an empty chunk adds exact zeros); a sliding
                   layer launches 2 chunks from the CHUNK-aligned start of its window, whatever the
                   context, so its cost does not grow with the context
  router           sigmoid of the bf16 logits in fp32 (as the published routing), an optional bias for the
                   choice, the top k (ties to the lower id), the chosen sigmoids over their sum + eps, fp32,
                   times the pack's expert scale (an EXL3 pack stores the experts' up projections divided by it)
  prompt_attention a prompt chunk's attention on SDPA's flash kernel with the KV heads as they are (no copy
                   per query head), the causal mask aligned to the chunk's end, the sliding window, and the
                   sink applied from the kernel's log-sum-exp: o * 1 / (1 + exp(sink - lse))

Each has a torch twin (`*_reference`) for the CPU tests.
"""

from __future__ import annotations

import torch

from engine.models.lfm2_moe.kernels import ANC, CHUNK, DOT32, HAVE_TRITON, class_of, classes  # noqa: F401

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from engine.models.lfm2_moe.kernels import _pdl_enter

#: SPD-145: the decode kernels launch as programmatic dependents (each waits for its predecessor, then lets
#: its successor launch at once: no launch gap, the next projection's weights stream early); the engine sets it
PDL = False

#: chunks a sliding layer's row reads: its window of W <= CHUNK + 1 keys spans at most two CHUNK-aligned chunks
SW_CHUNKS = 2


if HAVE_TRITON:

    @triton.jit
    def _attn_split(Q, K, V, CTL, ANCT, PM, PL, PACC, s_kh, s_kn, s_vh, s_vn, WIN, RING,
                    SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, CHUNK_: tl.constexpr, NA: tl.constexpr, NS: tl.constexpr,
                    SLIDING: tl.constexpr, DOT32_: tl.constexpr, PDL: tl.constexpr = False):
        # CTL[t] = (visible keys n = pos + 1, block start L): key c < L is cache position c, c >= L the
        # row's ancestor at depth c - L (ANCT[t, c - L], a cache slot = its position); a sliding layer
        # sees c >= n - WIN only and reads slot % RING
        z = 0
        if PDL:
            z = _pdl_enter()
        t = tl.program_id(0) + z
        kvh = tl.program_id(1) + z
        sp = tl.program_id(2) + z
        n = tl.load(CTL + t * 2)
        L = tl.load(CTL + t * 2 + 1)
        rows = tl.arange(0, BM)
        rmask = rows < REP
        qh = kvh * REP + rows
        od = tl.arange(0, D)
        q = tl.load(Q + (t * HQ + qh[:, None]) * D + od[None, :], mask=rmask[:, None], other=0.0)
        if SLIDING:
            lo = tl.maximum(n - WIN, 0)
            c0 = (lo // CHUNK_) * CHUNK_ + sp * CHUNK_
            cs = tl.maximum(c0, lo)
        else:
            c0 = sp * CHUNK_
            cs = c0
        c1 = tl.minimum(c0 + CHUNK_, n)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(cs, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            tail = cols >= L
            a = tl.load(ANCT + t * NA + (cols - L), mask=cmask & tail, other=0)
            slot = tl.where(tail, a, cols)
            if SLIDING:
                slot = slot % RING
            k = tl.load(K + kvh * s_kh + slot[:, None] * s_kn + od[None, :], mask=cmask[:, None], other=0.0)
            if DOT32_:
                s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * SCALE
            else:
                s = tl.dot(q, tl.trans(k)) * SCALE
            s = tl.where(cmask[None, :] & rmask[:, None], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + slot[:, None] * s_vn + od[None, :], mask=cmask[:, None], other=0.0)
            if DOT32_:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = ((t * HK + kvh) * NS + sp) * BM + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _attn_combine(PM, PL, PACC, SINK, OUT, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
                      HK: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr, PDL: tl.constexpr = False):
        z = 0
        if PDL:
            z = _pdl_enter()
        t = tl.program_id(0) + z
        qh = tl.program_id(1) + z
        kvh = qh // REP
        r = qh % REP
        od = tl.arange(0, D)
        b0 = (t * HK + kvh) * NS * BM + r
        sink = tl.load(SINK + qh)
        mx = sink
        for s in range(0, NS):
            mx = tl.maximum(mx, tl.load(PM + b0 + s * BM))
        num = tl.zeros([D], dtype=tl.float32)
        den = tl.exp(sink - mx)
        for s in range(0, NS):
            w = tl.exp(tl.load(PM + b0 + s * BM) - mx)
            num = num + tl.load(PACC + (b0 + s * BM) * D + od) * w
            den = den + tl.load(PL + b0 + s * BM) * w
        o = num / den
        tl.store(OUT + (t * HQ + qh) * D + od, o.to(tl.bfloat16))

    @triton.jit
    def _router(LOG, s_log, BIAS, IDS, WTS, eps, scale, E: tl.constexpr, KK: tl.constexpr, BIASED: tl.constexpr,
                PDL: tl.constexpr = False):
        z = 0
        if PDL:
            z = _pdl_enter()
        r = tl.program_id(0) + z
        o = tl.arange(0, E)
        lb = tl.load(LOG + r * s_log + o).to(tl.bfloat16).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-lb))
        if BIASED:
            sc = s + tl.load(BIAS + o)
        else:
            sc = s
        tot = 0.0
        for j in tl.static_range(KK):
            mx = tl.max(sc, 0)
            i = tl.min(tl.where(sc == mx, o, E), 0)          # the lower id on a tie
            tl.store(IDS + r * KK + j, i)
            wj = tl.sum(tl.where(o == i, s, 0.0), 0)
            tl.store(WTS + r * KK + j, wj)
            tot = tot + wj
            sc = tl.where(o == i, -float("inf"), sc)
        den = tot + eps
        for j in tl.static_range(KK):
            wj = tl.load(WTS + r * KK + j)
            tl.store(WTS + r * KK + j, wj / den * scale)


if HAVE_TRITON:

    @triton.jit
    def _glu_act(GU, s_gu, A, s_a, I: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
        z = 0
        if PDL:
            z = _pdl_enter()
        r = tl.program_id(0) + z
        c = (tl.program_id(1) + z) * BLOCK + tl.arange(0, BLOCK)
        m = c < I
        g = tl.load(GU + r * s_gu + c, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GU + r * s_gu + I + c, mask=m, other=0.0).to(tl.float32)
        sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        tl.store(A + r * s_a + c, (sg * u).to(tl.bfloat16), mask=m)

    @triton.jit
    def _add2_rms(H_IN, D1, D2, H_OUT, N_OUT, W, eps, N: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
        z = 0
        if PDL:
            z = _pdl_enter()
        r = tl.program_id(0) + z
        o = tl.arange(0, BLOCK)
        m = o < N
        d = (tl.load(D1 + r * N + o, mask=m, other=0.0).to(tl.float32)
             + tl.load(D2 + r * N + o, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16)
        h = (tl.load(H_IN + r * N + o, mask=m, other=0.0).to(tl.float32) + d.to(tl.float32)).to(tl.bfloat16)
        tl.store(H_OUT + r * N + o, h, mask=m)
        hf = h.to(tl.float32)
        var = tl.sum(hf * hf, 0) / N
        nrm = (hf * tl.rsqrt(var + eps)).to(tl.bfloat16)
        w = tl.load(W + o, mask=m, other=0.0)
        tl.store(N_OUT + r * N + o, (nrm.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16), mask=m)


if HAVE_TRITON:

    @triton.jit
    def _moe_gemm(X, s_x, ROWS, BEXP, W, S, S2, F4, F8, Y, s_y, OUTROW, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NVFP4: tl.constexpr, SCATTER: tl.constexpr,
                  F32OUT: tl.constexpr):
        # a block of BM rows of one expert (ROWS: the source row of each slot, -1 padding; BEXP: the block's
        # expert, -1 an empty block) times that expert's weight [N, K]^T, BN outputs
        b = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(BEXP + b)
        if e >= 0:
            slots = b * BM + tl.arange(0, BM)
            rows = tl.load(ROWS + slots)
            rm = rows >= 0
            n = nb * BN + tl.arange(0, BN)
            kk = tl.arange(0, BK)
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            e64 = e.to(tl.int64)
            for k0 in range(0, K, BK):
                x = tl.load(X + tl.maximum(rows, 0)[:, None].to(tl.int64) * s_x + k0 + kk[None, :], mask=rm[:, None],
                            other=0.0)
                if NVFP4:
                    j = tl.arange(0, BK // 2)
                    byte = tl.load(W + e64 * (N * (K // 2)) + n[:, None] * (K // 2) + k0 // 2 + j[None, :]).to(tl.int32)
                    lo = tl.load(F4 + (byte & 15))
                    hi = tl.load(F4 + (byte >> 4))
                    w = tl.reshape(tl.join(lo, hi), (BN, BK))
                    g = tl.arange(0, BK // 16)
                    sc = tl.load(F8 + tl.load(S + e64 * (N * (K // 16)) + n[:, None] * (K // 16) + k0 // 16 + g[None, :]).to(tl.int32))
                    w = tl.reshape(tl.reshape(w, (BN, BK // 16, 16)) * sc[:, :, None], (BN, BK))
                    w = (w * tl.load(S2 + e)).to(tl.bfloat16)
                else:
                    w = tl.load(W + e64 * (N * K) + n[:, None] * K + k0 + kk[None, :])
                acc = tl.dot(x, tl.trans(w), acc=acc)
            if SCATTER:
                out = tl.load(OUTROW + slots)
                tl.store(Y + tl.maximum(out, 0)[:, None].to(tl.int64) * s_y + n[None, :],
                         acc.to(tl.bfloat16).to(tl.float32), mask=(out >= 0)[:, None])
            else:
                if F32OUT:
                    tl.store(Y + slots[:, None].to(tl.int64) * s_y + n[None, :], acc)
                else:
                    tl.store(Y + slots[:, None].to(tl.int64) * s_y + n[None, :], acc.to(tl.bfloat16))


#: rows a block of the grouped expert GEMM takes
GROUP_BM = 64


def group_plan(ids: torch.Tensor, E: int, bm: int = GROUP_BM) -> dict:
    """The (row, slot) pairs of ids [T, S] sorted by expert into blocks of `bm` rows, on the device and with no
    host sync: `rows` [slots] the pair's source row (-1 padding), `pair` [slots] the pair's index (-1 padding),
    `bexp` [blocks] each block's expert (-1 past the last), sized for the worst case (every expert padded)."""
    T, S = ids.shape
    P = T * S
    dev = ids.device
    flat = ids.reshape(-1).long()
    counts = torch.bincount(flat, minlength=E)
    padded = (counts + bm - 1) // bm * bm
    start = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(P, device=dev) - first[flat[order]]
    dest = start[flat[order]] + rank
    nblk = (P + bm - 1) // bm + E
    rows = torch.full((nblk * bm,), -1, dtype=torch.int64, device=dev)
    pair = torch.full((nblk * bm,), -1, dtype=torch.int64, device=dev)
    rows[dest] = order // S
    pair[dest] = order
    bexp = torch.full((nblk,), -1, dtype=torch.int32, device=dev)
    blk_of_expert_end = torch.cumsum(padded, 0) // bm
    blk = torch.arange(nblk, device=dev)
    e_of_blk = torch.searchsorted(blk_of_expert_end, blk, right=True)
    bexp = torch.where(e_of_blk < E, e_of_blk, -1).to(torch.int32)
    return {"rows": rows, "pair": pair, "bexp": bexp, "nblk": nblk, "P": P}


#: the grouped GEMM's tiles (BM rows of one expert, BN outputs, BK along K) and launch shape
GROUP_CFG = {"BN": 64, "BK": 64, "warps": 4, "stages": 2}
#: from 1,024 tokens a chunk: 128-row blocks (4,096 tokens x 6 slots of 256 NVFP4 experts on the Spark: 10.1 ms a
#: layer against 15.5 with 64-row blocks; 32-row blocks 23.2)
GROUP_CFG_WIDE = {"BM": 128, "BN": 128, "BK": 64, "warps": 8, "stages": 3}


def moe_grouped_glu(x: torch.Tensor, ids: torch.Tensor, gu, dn, I: int, nvfp4: bool, E: int,
                    cfg: dict | None = None) -> torch.Tensor:
    """Every (row, slot) pair of a prompt chunk through its expert's SwiGLU in two grouped launches (no loop over
    experts on the host): x [T, H] bf16, ids [T, S] -> yd [T S, H] fp32 (bf16 values, unweighted, pair order).
    `gu` / `dn`: (W, S, S2) of an NVFP4 bank ([E, 2 I, H / 2] codes, e4m3 scale bytes, fp32 tensor scales) or the
    BF16 stacks ([E, 2 I, H], [E, H, I])."""
    T, H = x.shape
    S = ids.shape[1]
    cfg = cfg or (GROUP_CFG_WIDE if T >= 1024 else GROUP_CFG)
    BM = cfg.get("BM", GROUP_BM)
    pl = group_plan(ids, E, BM)
    dev = x.device
    f4 = fp4_table(dev) if nvfp4 else x
    f8 = e4m3_table(dev) if nvfp4 else x
    BN, BK, nw, ns = cfg["BN"], cfg["BK"], cfg["warps"], cfg["stages"]
    nslots = pl["nblk"] * BM
    ygu = torch.empty(nslots, 2 * I, dtype=torch.bfloat16, device=dev)
    W, Sc, S2 = (gu if nvfp4 else (gu, x, x))
    _moe_gemm[(pl["nblk"], 2 * I // BN)](x, x.stride(0), pl["rows"], pl["bexp"], W, Sc, S2, f4, f8, ygu, ygu.stride(0),
                                        pl["rows"], N=2 * I, K=H, BM=BM, BN=BN, BK=BK, NVFP4=nvfp4, SCATTER=False,
                                        F32OUT=False, num_warps=nw, num_stages=ns)
    a = glu_act(ygu, I)
    slots = torch.arange(nslots, device=dev)
    srows = torch.where(pl["rows"] >= 0, slots, -1)
    yd = torch.empty(T * S, H, dtype=torch.float32, device=dev)
    W, Sc, S2 = (dn if nvfp4 else (dn, x, x))
    _moe_gemm[(pl["nblk"], H // BN)](a, a.stride(0), srows, pl["bexp"], W, Sc, S2, f4, f8, yd, yd.stride(0),
                                    pl["pair"], N=H, K=I, BM=BM, BN=BN, BK=BK, NVFP4=nvfp4, SCATTER=True, F32OUT=True,
                                    num_warps=nw, num_stages=ns)
    return yd


def fp4_table(device):
    from tools.nvfp4_relu2 import fp4_table as f
    return f(device)


def e4m3_table(device):
    from tools.nvfp4_relu2 import e4m3_table as f
    return f(device)


def glu_act(gu: torch.Tensor, I: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """gu [M, 2 I] bf16 (gate | up) -> bf16(bf16(silu(gate)) * up) [M, I] in one launch: the bits of
    `F.silu(gu[:, :I]) * gu[:, I:]` in bf16."""
    M = gu.shape[0]
    if out is None:
        out = torch.empty(M, I, dtype=torch.bfloat16, device=gu.device)
    if not gu.is_cuda:
        return out.copy_(glu_act_reference(gu, I))
    _glu_act[(M, triton.cdiv(I, 1024))](gu, gu.stride(0), out, out.stride(0), I=I, BLOCK=1024, PDL=PDL, launch_pdl=PDL,
                                        num_warps=4)
    return out


def glu_act_reference(gu, I):
    return torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]


def add2_rms(h: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor, w: torch.Tensor, eps: float):
    """(h + bf16(d1 + d2), its norm) in one launch: the bits of LFM2's add_rms(h, d1 + d2, w) with the sum in bf16."""
    M, N = h.shape
    if not h.is_cuda:
        from engine.models.lfm2_moe.kernels import add_rms_reference
        return add_rms_reference(h, (d1.float() + d2.float()).to(torch.bfloat16), w, eps)
    hn, nn = torch.empty_like(h), torch.empty_like(h)
    _add2_rms[(M,)](h, d1, d2, hn, nn, w, eps, N=N, BLOCK=triton.next_power_of_2(N), PDL=PDL, launch_pdl=PDL,
                    num_warps=4)
    return hn, nn


def attn_rows(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, ctl: torch.Tensor, anc: torch.Tensor, ns: int,
              sink: torch.Tensor, window: int = 0, ring: int = 0, bufs: dict | None = None,
              out: torch.Tensor | None = None) -> torch.Tensor:
    """q [M, Hq, D] bf16, kc / vc [Hk, cap, D] bf16 (a ring of `ring` rows on a sliding layer), ctl int32
    [M, 2] (visible keys, block start), anc int32 [M, ANC] (the row's ancestors' positions by depth), `ns`
    CHUNK programs of a full layer (the class, `class_of`; a sliding layer takes SW_CHUNKS), sink fp32 [Hq],
    `window` the sliding layer's W (0: a full layer) -> o [M, Hq, D] bf16."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    REP = HQ // HK
    BM = 16
    sliding = window > 0
    if sliding:
        assert window <= CHUNK + 1 and ring >= window
        ns = SW_CHUNKS
    assert REP <= BM and D == kc.shape[2]
    b = bufs if bufs is not None else {}
    n = M * HK * ns * BM
    pm = b.get("pm")
    if pm is None or pm.shape[0] < n:
        if pm is not None and b.get("rows"):
            raise ValueError(f"{M} rows in partials sized for {b['rows']}")
        cap = max(M, int(b.get("rows") or M)) * HK * ns * BM
        pm = torch.empty(cap, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(cap, D, dtype=torch.float32, device=q.device)
        if bufs is not None:
            bufs.update(pm=pm, pl=pl, pacc=pacc)
    else:
        pl, pacc = b["pl"], b["pacc"]
    pm, pl, pacc = pm[:n], pl[:n], pacc[:n]
    if out is None:
        out = torch.empty(M, HQ, D, dtype=torch.bfloat16, device=q.device)
    _attn_split[(M, HK, ns)](q, kc, vc, ctl, anc, pm, pl, pacc, kc.stride(0), kc.stride(1), vc.stride(0),
                             vc.stride(1), int(window), int(max(ring, 1)), SCALE=D ** -0.5, REP=REP, D=D, HQ=HQ,
                             HK=HK, BM=BM, BN=32, CHUNK_=CHUNK, NA=anc.shape[1], NS=ns, SLIDING=sliding,
                             DOT32_=DOT32, PDL=PDL, launch_pdl=PDL, num_warps=4, num_stages=2)
    _attn_combine[(M, HQ)](pm, pl, pacc, sink, out, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, NS=ns, PDL=PDL, launch_pdl=PDL,
                           num_warps=1)
    return out


def attn_rows_reference(q, kc, vc, ctl, anc, sink, window: int = 0, ring: int = 0):
    M, HQ, D = q.shape
    rep = HQ // kc.shape[0]
    out = []
    for t in range(M):
        n, L = (int(x) for x in ctl[t, :2])
        lo = max(0, n - window) if window else 0
        pos = torch.arange(lo, n, device=q.device)
        slot = torch.where(pos >= L, anc[t].long()[(pos - L).clamp(min=0, max=anc.shape[1] - 1)], pos)
        if window:
            slot = slot % ring
        k = kc[:, slot].float().repeat_interleave(rep, 0)
        v = vc[:, slot].float().repeat_interleave(rep, 0)
        s = torch.einsum("hd,hnd->hn", q[t].float(), k) * D ** -0.5
        a = torch.softmax(torch.cat([s, sink.float()[:, None]], -1), -1)[:, :-1]
        out.append(torch.einsum("hn,hnd->hd", a, v))
    return torch.stack(out)


def router(logits: torch.Tensor, bias: torch.Tensor | None, k: int, eps: float, ids=None, wts=None,
           scale: float = 1.0):
    """logits fp32 [M, E] -> ids int32 [M, k], wts fp32 [M, k] (times `scale`: a pack whose experts' up
    projections are stored divided by it)."""
    M, E = logits.shape
    if ids is None:
        ids = torch.empty(M, k, dtype=torch.int32, device=logits.device)
        wts = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    _router[(M,)](logits, logits.stride(0), bias if bias is not None else logits, ids, wts, float(eps), float(scale), E=E, KK=k,
                  BIASED=bias is not None, PDL=PDL, launch_pdl=PDL, num_warps=1)
    return ids, wts


def router_reference(logits, bias, k: int, eps: float, scale: float = 1.0):
    s = logits.to(torch.bfloat16).float().sigmoid()
    sc = s + bias if bias is not None else s
    order = torch.sort(sc, dim=-1, descending=True, stable=True).indices[:, :k]
    w = torch.gather(s, 1, order)
    return order.to(torch.int32), w / (w.sum(-1, keepdim=True) + eps) * scale


#: whether the flash kernel ran here (None: not tried yet; False: the torch path)
FLASH = None


def prompt_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, sink: torch.Tensor,
                     window: int = 0) -> torch.Tensor:
    """A prompt chunk's attention. q [T, Hq, D] bf16 of positions [start + S - T, start + S) where k / v
    [Hk, S, D] hold positions [start, start + S) (the chunk's own rows last), sink fp32 [Hq], `window` W of
    a sliding layer (0: full) -> o [T, Hq, D] bf16. The flash kernel takes the KV heads as they are and gives
    the log-sum-exp the sink needs; where it is missing (the CPU, a device without it) the torch reference."""
    global FLASH
    T, HQ, D = q.shape
    S = k.shape[1]
    if q.is_cuda and FLASH is not False:
        try:
            out, lse, *_ = torch.ops.aten._flash_attention_forward(
                q.unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), None, None, T, S,
                0.0, True, False, scale=D ** -0.5, window_size_left=(window - 1) if window else -1,
                window_size_right=0 if window else -1)
            FLASH = True
            f = torch.sigmoid(lse[0].float() - sink.float()[:, None])           # [Hq, T] = 1 / (1 + e^(sink - lse))
            return (out[0].float() * f.transpose(0, 1)[:, :, None]).to(q.dtype)
        except (RuntimeError, TypeError):
            FLASH = False
    return prompt_attention_reference(q, k, v, sink, window).to(q.dtype)


def prompt_attention_reference(q, k, v, sink, window: int = 0, block: int = 256):
    T, HQ, D = q.shape
    HK, S = k.shape[0], k.shape[1]
    rep = HQ // HK
    out = torch.empty(T, HQ, D, dtype=torch.float32, device=q.device)
    kf = k.float().repeat_interleave(rep, 0)
    vf = v.float().repeat_interleave(rep, 0)
    j = torch.arange(S, device=q.device)[None, :]
    for a in range(0, T, block):
        b = min(a + block, T)
        i = torch.arange(a, b, device=q.device)[:, None] + (S - T)
        m = j <= i
        if window:
            m = m & (i - j < window)
        s = torch.einsum("thd,hsd->hts", q[a:b].float(), kf) * D ** -0.5
        s = s.masked_fill(~m[None], float("-inf"))
        s = torch.cat([s, sink.float()[:, None, None].expand(HQ, b - a, 1)], -1)
        p = torch.softmax(s, -1)[..., :-1]
        out[a:b] = torch.einsum("hts,hsd->thd", p, vf)
    return out


def sink_from_lse_reference(o: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """o [T, Hq, D] without the sink and its lse [Hq, T] -> with it (what prompt_attention does)."""
    return o * torch.sigmoid(lse - sink[:, None]).transpose(0, 1)[:, :, None]


def window_bounds(pos: int, window: int) -> int:
    """The first position a sliding row at `pos` sees."""
    return max(0, pos - window + 1)

