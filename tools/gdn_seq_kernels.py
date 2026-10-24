"""The GDN verify recurrence one node at a time along its path (SPD-201, QWEN38_GDN_SEQ).

The WY form (tools/gdn_wy_kernels.py) computes a block's rows from the block's entry state in a closed form, and
the commit adds the accepted path's rank-k product to that entry state. Both are exact algebra, but their
floating-point sums depend on how the tokens were grouped into blocks: the bits a position gets depend on the tree
it was drafted in and on where the last commit fell, which the traffic decides (a request alone drafts 16-row
trees, among eight 8-row ones). Here every state is the previous one advanced by one token, in one fixed order:

    y = S * exp(g)       kv = y^T k       d = (v - kv) * beta       S' = fma(k, d, y)       o = S'^T q

A node's state is its parent's advanced by its own token: the walk recomputes the path from the block's entry
state for each node (DFS pre-order, so a path is its nodes in index order), and keeps each node's d. The commit
replays `y = S * exp(g); S' = fma(k, d, y)` over the accepted rows with the walk's own k, d and g, so the committed
state is the walked state bit for bit, and the next block starts where a token-at-a-time decode would have. The
factors keep their buffers and their meaning for the fold and the trims: kk the normalised key, delta the node's
d, gc the node's own gate (not a path sum).

Two kernels a layer, through the grouped launch's offset table (one engine alone is a table of one, so a request's
rows are the same statements alone and among others):

  _seq_prep   grid (E * H,): the q, k and v convolution, the l2 norms, the gates (`_wy_prep`'s statements)
  _seq_walk   grid (E * H, Dv / BV): the pending commit replayed into the tile and written back, then every node in
              DFS order: one step from the node before when that is its parent, else along its path from the entry

QWEN38_GDN_SEQ_BYNODE=E: a launch of at most E blocks walks a program a node instead (`_seq_pend` first), the
same steps from the same entry, the same bits (`check`); slower on this board at every E measured, so 0.

`seq_commit` is the replay on its own (the engine's flush and rollback), the same statements.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

from engine.settings import SETTINGS as _S  # noqa: E402

#: on (1): tree verifies, the fold and the commits advance the GDN state one token at a time (this module)
ON = _S.get("GDN_SEQ") == "1"
#: the value block and warps of the state kernels: one fixed launch, the reductions' order with it
BV = int(os.environ.get("QWEN38_GDN_SEQ_BV", "16"))
WARPS = int(os.environ.get("QWEN38_GDN_SEQ_WARPS", "1"))
#: a launch of at most this many blocks walks node by node in parallel programs (the chain is a path's steps);
#: past it a program walks all its nodes (the work once). The same steps either way: the same bits
BY_NODE_E = int(os.environ.get("QWEN38_GDN_SEQ_BYNODE", "0"))

if HAVE_TRITON:
    from tools.gdn_wy_kernels import NOFF, _conv_rows, _qk_rows
    NOFF_ = tl.constexpr(NOFF)

    @triton.jit
    def _step(x, kj, vj, gj, bj):
        """One token: y = S exp(g), d = (v - y^T k) beta, S' = fma(k, d, y). Returns (S', d)."""
        y = x * tl.exp(gj)
        kv = tl.sum(y * kj[:, None], axis=0)
        d = (vj - kv) * bj
        return tl.fma(kj[:, None], d[None, :], y), d

    @triton.jit
    def _replay(s, kj, dj, gj):
        """`_step` with the node's d given: the commit of an accepted row."""
        y = s * tl.exp(gj)
        return tl.fma(kj[:, None], dj[None, :], y)

    @triton.jit
    def _seq_prep(GC, T, H, KK, kk_h, gc_h, QS, VS,
                  X, s_xt, CST, s_cs, CW, WIN, A, B_, s_a, s_b, ALOG, DTB, BETA_S, KEY_DIM,
                  TP: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, REP: tl.constexpr, EPS: tl.constexpr,
                  SCALE: tl.constexpr, WIDTH: tl.constexpr, OFF=None):
        """Per (engine, value head): the q, k and v convolution, the l2 norms, the gates."""
        e = tl.program_id(0) // H
        h = tl.program_id(0) % H
        ob = OFF + e * NOFF_
        T = tl.load(ob + 0).to(tl.int32)
        GC += tl.multiple_of(tl.load(ob + 1), 16)
        KK += tl.multiple_of(tl.load(ob + 3), 16)
        X += tl.multiple_of(tl.load(ob + 4), 16)
        CST += tl.multiple_of(tl.load(ob + 5), 16)
        WIN += tl.multiple_of(tl.load(ob + 6), 16)
        A += tl.multiple_of(tl.load(ob + 7), 16)
        B_ += tl.multiple_of(tl.load(ob + 8), 16)
        hs = e * H + h
        hk = h // REP
        tt = tl.arange(0, TP)
        mt = tt < T
        ok = tl.arange(0, DK)
        q, k = _qk_rows(X, X, 0, X, s_xt, CST, s_cs, CW, WIN, KEY_DIM, hk, ok, tt, mt, DK, True, True, WIDTH)
        q = q * tl.rsqrt(tl.sum(q * q, axis=1) + EPS)[:, None] * SCALE
        k = k * tl.rsqrt(tl.sum(k * k, axis=1) + EPS)[:, None]
        tl.store(KK + h * kk_h + tt[:, None] * DK + ok[None, :], k, mask=mt[:, None])
        tl.store(QS + (hs * TP + tt[:, None]) * DK + ok[None, :], q)
        od = tl.arange(0, DV)
        v = _conv_rows(X, s_xt, CST, s_cs, CW, WIN, 2 * KEY_DIM + h * DV + od, tt, mt, True, WIDTH)
        tl.store(VS + (hs * TP + tt[:, None]) * DV + od[None, :], v)
        a = tl.load(A + tt * s_a + h, mask=mt, other=0.0).to(tl.float32)
        b = tl.load(B_ + tt * s_b + h, mask=mt, other=0.0).to(tl.float32)
        x = a + tl.load(DTB + h).to(tl.float32)
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.where(mt, -tl.exp(tl.load(ALOG + h).to(tl.float32)) * sp, 0.0)
        beta = tl.where(mt, 1.0 / (1.0 + tl.exp(-b)), 0.0)
        tl.store(BETA_S + hs * TP + tt, beta)
        tl.store(GC + h * gc_h + tt, g, mask=mt)

    @triton.jit
    def _seq_pend(S, s_h, s_k, s_v, PK, PU, PG, PROWS, PN, pk_h, pk_t, pu_h, pu_t, pg_h,
                  DK: tl.constexpr, BV: tl.constexpr, OFF=None, NH=0):
        """Per (engine, value head, value block): the previous block's accepted rows replayed into the state."""
        H = NH
        e = tl.program_id(0) // H
        h = tl.program_id(0) % H
        ob = OFF + e * NOFF_
        S += tl.multiple_of(tl.load(ob + 9), 16)
        PK += tl.multiple_of(tl.load(ob + 12), 16)
        PU += tl.multiple_of(tl.load(ob + 13), 16)
        PG += tl.multiple_of(tl.load(ob + 14), 16)
        PROWS += tl.multiple_of(tl.load(ob + 15), 16)
        PN += tl.multiple_of(tl.load(ob + 16), 16)
        P = tl.load(PN)
        if P > 0:
            vb = tl.program_id(1)
            ok = tl.arange(0, DK)
            ov = vb * BV + tl.arange(0, BV)
            tile = h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
            s = tl.load(S + tile)
            for j in range(P):
                r = tl.load(PROWS + j)
                s = _replay(s, tl.load(PK + h * pk_h + r * pk_t + ok), tl.load(PU + h * pu_h + r * pu_t + ov),
                            tl.load(PG + h * pg_h + r))
            tl.store(S + tile, s)

    @triton.jit
    def _path_state(x, GC, gc_h, KK, kk_h, VS, BETA_S, ANC, s_anc, h, hs, n, ok, ov,
                    TP: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """Node n's state: the entry state x advanced along n's path (its ancestors, in index order), and its d."""
        d = tl.zeros([BV], dtype=tl.float32)
        for j in range(n + 1):
            if tl.load(ANC + n * s_anc + j) != 0:
                x, d = _step(x, tl.load(KK + h * kk_h + j * DK + ok), tl.load(VS + (hs * TP + j) * DV + ov),
                             tl.load(GC + h * gc_h + j), tl.load(BETA_S + hs * TP + j))
        return x, d

    @triton.jit
    def _seq_walk(GC, gc_h, S, OUT, DELTA, KK, kk_h, QS, VS, ANC, s_anc, s_h, s_k, s_v, u_h, BETA_S,
                  PK, PU, PG, PROWS, PN, pk_h, pk_t, pu_h, pu_t, pg_h,
                  TP: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                  BY_NODE: tl.constexpr, PEND: tl.constexpr, OFF=None, NH=0):
        """Each node's output and d from the committed entry state. BY_NODE: program axis 2 is the node (the
        pending commit replayed before, by `_seq_pend`); else a program replays the pending commit into its tile,
        writes it back and walks every node of its (engine, head, value block) in DFS order, a node whose parent is
        the node before it one step from that node's state, any other along its path from the entry."""
        H = NH
        e = tl.program_id(0) // H
        h = tl.program_id(0) % H
        ob = OFF + e * NOFF_
        T = tl.load(ob + 0).to(tl.int32)
        GC += tl.multiple_of(tl.load(ob + 1), 16)
        ANC += tl.multiple_of(tl.load(ob + 2), 16)
        s_anc = tl.load(ob + 17)
        KK += tl.multiple_of(tl.load(ob + 3), 16)
        S += tl.multiple_of(tl.load(ob + 9), 16)
        DELTA += tl.multiple_of(tl.load(ob + 10), 16)
        OUT += tl.multiple_of(tl.load(ob + 11), 16)
        hs = e * H + h
        vb = tl.program_id(1)
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        tile = h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        if BY_NODE:
            n = tl.program_id(2)
            if n < T:
                x, d = _path_state(tl.load(S + tile), GC, gc_h, KK, kk_h, VS, BETA_S, ANC, s_anc, h, hs, n, ok, ov,
                                   TP, DK, DV, BV)
                tl.store(DELTA + h * u_h + n * DV + ov, d)
                o = tl.sum(x * tl.load(QS + (hs * TP + n) * DK + ok)[:, None], axis=0)
                tl.store(OUT + (n * H + h) * DV + ov, o.to(OUT.dtype.element_ty))
        else:
            s = tl.load(S + tile)
            if PEND:
                PK += tl.multiple_of(tl.load(ob + 12), 16)
                PU += tl.multiple_of(tl.load(ob + 13), 16)
                PG += tl.multiple_of(tl.load(ob + 14), 16)
                PROWS += tl.multiple_of(tl.load(ob + 15), 16)
                PN += tl.multiple_of(tl.load(ob + 16), 16)
                P = tl.load(PN)
                if P > 0:
                    for j in range(P):
                        r = tl.load(PROWS + j)
                        s = _replay(s, tl.load(PK + h * pk_h + r * pk_t + ok), tl.load(PU + h * pu_h + r * pu_t + ov),
                                    tl.load(PG + h * pg_h + r))
                    tl.store(S + tile, s)
            xp = s
            for n in range(T):
                if n > 0 and tl.load(ANC + n * s_anc + n - 1) != 0:
                    # the node before is this node's parent (DFS pre-order): one step from its state
                    x, d = _step(xp, tl.load(KK + h * kk_h + n * DK + ok), tl.load(VS + (hs * TP + n) * DV + ov),
                                 tl.load(GC + h * gc_h + n), tl.load(BETA_S + hs * TP + n))
                else:
                    x, d = _path_state(s, GC, gc_h, KK, kk_h, VS, BETA_S, ANC, s_anc, h, hs, n, ok, ov,
                                       TP, DK, DV, BV)
                tl.store(DELTA + h * u_h + n * DV + ov, d)
                o = tl.sum(x * tl.load(QS + (hs * TP + n) * DK + ok)[:, None], axis=0)
                tl.store(OUT + (n * H + h) * DV + ov, o.to(OUT.dtype.element_ty))
                xp = x

    @triton.jit
    def _seq_commit(S, s_l, s_h, s_k, s_v, PK, PU, PG, pk_l, pk_h, pk_t, pu_l, pu_h, pu_t, pg_l, pg_h,
                    PROWS, P, H, DK: tl.constexpr, BV: tl.constexpr):
        """`_seq_pend`'s replay over rows PROWS[:P] of every layer's factors, in place."""
        lh = tl.program_id(0)
        l = lh // H
        h = lh % H
        vb = tl.program_id(1)
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        tile = l * s_l + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        s = tl.load(S + tile)
        for j in range(P):
            r = tl.load(PROWS + j)
            s = _replay(s, tl.load(PK + l * pk_l + h * pk_h + r * pk_t + ok),
                        tl.load(PU + l * pu_l + h * pu_h + r * pu_t + ov), tl.load(PG + l * pg_l + h * pg_h + r))
        tl.store(S + tile, s)


def seq_recurrence_grouped(items: list, *, key_heads: int, value_heads: int, head_k: int, head_v: int,
                           conv_w, a_log, dt_bias, key_dim: int, reuse: dict | None = None) -> list:
    """`tools/gdn_wy_kernels.py wy_recurrence_grouped`'s contract (the same items, offset table, factor buffers
    and returns), the recurrence one node at a time. A table of one engine is the lone pass."""
    from tools import gdn_wy_kernels as G
    H, dk, dv = value_heads, head_k, head_v
    E = len(items)
    i0 = items[0]
    dev = i0["state"].device
    f32 = torch.float32
    Ts = [it["mixed"].shape[0] for it in items]
    if max(Ts) > 32:
        raise ValueError(f"a sequential verify block of {max(Ts)} rows (32 at most)")
    # the tile's rows: a row's statements do not depend on it (every reduction is along a row or picks one row)
    tp = 16 if max(Ts) <= 16 else 32
    rows = [0]
    for t in Ts:
        rows.append(rows[-1] + t)
    qs = torch.empty(E * H, tp, dk, dtype=f32, device=dev)
    beta_s = torch.empty(E * H, tp, dtype=f32, device=dev)
    out = (torch.zeros if G.TABLE_REC is not None else torch.empty)(rows[-1], H, dv, dtype=i0["mixed"].dtype,
                                                                    device=dev)
    if i0.get("fac_out") is not None:
        facs = [it["fac_out"] for it in items]
    else:
        kk_all = torch.empty(E, H, tp, dk, dtype=f32, device=dev)
        de_all = torch.empty(E, H, tp, dv, dtype=f32, device=dev)
        gc_all = torch.empty(E, H, tp, dtype=f32, device=dev)
        facs = [(kk_all[e], de_all[e], gc_all[e]) for e in range(E)]
    kk0, de0, gc0 = facs[0]
    p0 = i0["pend"]
    rec = G.TABLE_REC
    if rec is not None:
        table = rec.record(G.grouped_offsets(items, facs, rows, H * dv), items, facs)
        reuse = None
    prev = reuse.get("table") if reuse is not None else None
    guard = (tuple(Ts), G._elem_off(items[-1]["mixed"], i0["mixed"]), G._elem_off(items[-1]["state"], i0["state"]),
             G._elem_off(facs[-1][0], kk0)) if reuse is not None else None
    hit = prev is not None and prev[0] == guard
    if rec is not None:
        pass
    elif hit:
        table = prev[1]
    elif E == 1:
        off = G.grouped_offsets(items, facs, rows, H * dv)
        key = ("seq", dev, tuple(off))
        table = G._ONE_TABLES.get(key)
        if table is None:
            table = G._ONE_TABLES[key] = torch.tensor(off, dtype=torch.int64).to(dev)
    else:
        from engine.model import h2d
        table = h2d(G.grouped_offsets(items, facs, rows, H * dv), torch.int64, dev)
    if reuse is not None and not hit:
        reuse["table"] = (guard, table)
    x, cst = i0["mixed"], i0["conv_state"]
    win = i0["window"]
    a8 = i0["anc"].view(torch.uint8)
    a_raw, b_raw = i0["a_raw"], i0["b_raw"]
    width = conv_w.shape[-1]
    vs = torch.empty(E * H, tp, dv, dtype=f32, device=dev)
    _seq_prep[(E * H,)](gc0, tp, H, kk0, kk0.stride(0), gc0.stride(0), qs, vs,
                        x, x.stride(0), cst, cst.stride(0), conv_w, win, a_raw, b_raw, a_raw.stride(0),
                        b_raw.stride(0), a_log, dt_bias, beta_s, key_dim,
                        TP=tp, DK=dk, DV=dv, REP=H // key_heads, EPS=1e-6, SCALE=dk ** -0.5, WIDTH=width,
                        OFF=table, num_warps=4)
    S = i0["state"]
    by_node = E <= BY_NODE_E
    if p0 is not None:
        pk, pu, pg, prows, pn = p0
        pargs = (pk, pu, pg, prows, pn, pk.stride(0), pk.stride(1), pu.stride(0), pu.stride(1), pg.stride(0))
        if by_node:
            _seq_pend[(E * H, dv // BV)](S, S.stride(0), S.stride(1), S.stride(2), *pargs,
                                         DK=dk, BV=BV, OFF=table, NH=H, num_warps=WARPS)
    else:
        pargs = (gc0, gc0, gc0, gc0, gc0, 0, 0, 0, 0, 0)
    grid = (E * H, dv // BV, tp) if by_node else (E * H, dv // BV)
    _seq_walk[grid](gc0, gc0.stride(0), S, out, de0, kk0, kk0.stride(0), qs, vs, a8, i0["anc"].stride(0),
                    S.stride(0), S.stride(1), S.stride(2), de0.stride(0), beta_s, *pargs,
                    TP=tp, DK=dk, DV=dv, BV=BV, BY_NODE=by_node, PEND=p0 is not None and not by_node, OFF=table,
                    NH=H, num_warps=WARPS)
    res = []
    for e, T in enumerate(Ts):
        kk, de, gc = facs[e]
        res.append((out[rows[e]:rows[e + 1]].view(1, T, H, dv),
                    (kk[None, :, :T], de[None, :, :T], gc[None, :, :T])))
    return res


def lone(mixed, conv_state, conv_w, a_raw, b_raw, a_log, dt_bias, state, *, key_dim, key_heads, value_heads,
         head_k, head_v, window, anc, pend=None, fac_out=None):
    """`verify_mixer`'s tree block through the sequential kernels, a table of one engine: (o [1, T, H, Dv],
    factors, None) as `verify_mixer` returns them."""
    item = dict(mixed=mixed, conv_state=conv_state.reshape(conv_state.shape[-2], -1), window=window, anc=anc,
                a_raw=a_raw, b_raw=b_raw, state=state.reshape(value_heads, head_k, head_v), pend=pend,
                fac_out=fac_out)
    (o, fac), = seq_recurrence_grouped([item], key_heads=key_heads, value_heads=value_heads, head_k=head_k,
                                       head_v=head_v, conv_w=conv_w, a_log=a_log, dt_bias=dt_bias, key_dim=key_dim)
    return o, fac, None


def seq_commit(S: torch.Tensor, kk: torch.Tensor, u: torch.Tensor, gc: torch.Tensor, rows: list) -> None:
    """The accepted rows of every layer's factors replayed into S in place. S [L, 1, H, Dk, Dv] (or [L, H, Dk,
    Dv]); kk [L, H, T, Dk], u [L, H, T, Dv], gc [L, H, T]."""
    if not rows:
        return
    Sv = S.reshape(S.shape[0], -1, S.shape[-2], S.shape[-1])
    L, H, dk, dv = Sv.shape
    from tools.gdn_commit_kernels import _rows_tensor
    pr = _rows_tensor(list(rows), S.device)
    _seq_commit[(L * H, dv // BV)](Sv, Sv.stride(0), Sv.stride(1), Sv.stride(2), Sv.stride(3),
                                   kk, u, gc, kk.stride(0), kk.stride(1), kk.stride(2), u.stride(0), u.stride(1),
                                   u.stride(2), gc.stride(0), gc.stride(1), pr, len(rows), H, DK=dk, BV=BV,
                                   num_warps=WARPS)


def reference_walk(S0, q, k, v, g, beta, parents):
    """The token-at-a-time recurrence in torch fp32 over a tree (S0 [H, Dk, Dv]; q, k normalised [T, H, Dk]; v
    [T, H, Dv]; g, beta [T, H]): per node its output [T, H, Dv] and state. For the checks."""
    T = len(parents)
    states, outs = [None] * T, []
    for n in range(T):
        Sp = S0 if parents[n] < 0 else states[parents[n]]
        y = Sp * torch.exp(g[n])[:, None, None]
        kv = torch.einsum("hkv,hk->hv", y, k[n])
        d = (v[n] - kv) * beta[n][:, None]
        states[n] = y + k[n][:, :, None] * d[:, None, :]
        outs.append(torch.einsum("hkv,hk->hv", states[n], q[n]))
    return torch.stack(outs), states


def check(seed: int = 0) -> list[str]:
    """`check_invariance` with the walk node by node and with every node in one program: the same bits."""
    global BY_NODE_E
    keep = BY_NODE_E
    out = []
    try:
        BY_NODE_E = 0
        a = check_invariance(seed, keep_outputs=True)
        BY_NODE_E = 99
        b = check_invariance(seed, keep_outputs=True)
    finally:
        BY_NODE_E = keep
    for (name, x), (_, y) in zip(a[1], b[1]):
        assert torch.equal(x, y), f"walk vs by-node: {name}"
    out.extend(a[0])
    out.append(f"the walk by node = the walk in one program: {len(a[1])} outputs and states bit-identical")
    return out


def bench(reps: int = 30) -> list[str]:
    """GPU us a layer of the sequential launches against the WY pair, at one block of 16 rows and eight of 8."""
    import random
    from torch.profiler import ProfilerActivity, profile
    from engine.tree import DraftTree
    from tools import gdn_wy_kernels as G
    rng = random.Random(0)
    gen = torch.Generator(device="cuda").manual_seed(0)
    H, dk, dv, kh, kd = G.KW["value_heads"], G.KW["head_k"], G.KW["head_v"], G.KW["key_heads"], G.KW["key_dim"]
    gk = dict(key_heads=kh, value_heads=H, head_k=dk, head_v=dv)
    out = []
    for E, T in ((1, 16), (2, 16), (4, 16), (8, 8)):
        shared = G._inputs(16, gen)
        sw = dict(conv_w=shared["conv_w"], a_log=shared["a_log"], dt_bias=shared["dt_bias"], key_dim=kd)
        its = []
        for _ in range(E):
            x = G._inputs(T, gen)
            par, pth = [-1], [0]
            for i in range(1, T):
                pth = pth[:rng.randint(1, len(pth))]
                par.append(pth[-1])
                pth.append(i)
            ta = G._tree_args(DraftTree(tokens=[0] * T, parents=par))
            pd = (torch.randn(H, 32, dk, device="cuda", generator=gen) * 0.1,
                  torch.randn(H, 32, dv, device="cuda", generator=gen) * 0.1,
                  -torch.rand(H, 32, device="cuda", generator=gen) * 0.05,
                  torch.arange(32, dtype=torch.int32, device="cuda"), torch.tensor([3], dtype=torch.int32, device="cuda"))
            its.append((x, ta, pd))
        mixed = torch.cat([x["mixed"] for x, *_ in its])
        ab = torch.cat([torch.cat([x["a_raw"], x["b_raw"]], -1) for x, *_ in its])

        def items():
            r0, res = 0, []
            for x, ta, pd in its:
                res.append(dict(mixed=mixed[r0:r0 + T], conv_state=x["conv_state"].reshape(x["conv_state"].shape[1], -1),
                                window=ta["window"], anc=ta["anc"], a_raw=ab[r0:r0 + T, :H], b_raw=ab[r0:r0 + T, H:],
                                state=x["state"].reshape(H, dk, dv).clone(), pend=pd,
                                fac_out=(torch.zeros(H, 32, dk, device="cuda"), torch.zeros(H, 32, dv, device="cuda"),
                                         torch.zeros(H, 32, device="cuda"))))
                r0 += T
            return res
        row = [f"E{E} T{T}"]
        for name, fn in (("WY", lambda it: G.wy_recurrence_grouped(it, **gk, **sw)),
                         ("seq", lambda it: seq_recurrence_grouped(it, **gk, **sw))):
            it = items()
            for _ in range(3):
                fn(it)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(reps):
                    fn(it)
                torch.cuda.synchronize()
            us = sum(e.device_time_total for e in pr.key_averages() if e.key.startswith("_")) / reps
            row.append(f"{name} {us:.0f} us (" + " ".join(f"{e.key[:10]}={e.device_time_total / reps:.0f}"
                                                         for e in pr.key_averages() if e.key.startswith("_")) + ")")
        out.append(", ".join(row))
    return out


def check_invariance(seed: int = 0, keep_outputs: bool = False):
    """The proof on the board: six tokens through (A) one 6-row chain, (B) two 3-row blocks with the first's commit
    pending in the second, (C) an 8-node tree whose path to the last token is the six with siblings beside them --
    the outputs of every token and the committed state bit for bit the same; (D) the three in one grouped launch
    equal each alone; (E) the replay on its own (`seq_commit`) equals the replay in the next block. And the chain
    against the float64 walk."""
    from engine.tree import DraftTree
    from tools import gdn_wy_kernels as G
    gen = torch.Generator(device="cuda").manual_seed(seed)
    H, dk, dv, kh, kd = G.KW["value_heads"], G.KW["head_k"], G.KW["head_v"], G.KW["key_heads"], G.KW["key_dim"]
    gk = dict(key_heads=kh, value_heads=H, head_k=dk, head_v=dv)
    x = G._inputs(8, gen)
    C = x["mixed"].shape[1]
    sw = dict(conv_w=x["conv_w"], a_log=x["a_log"], dt_bias=x["dt_bias"], key_dim=kd)
    raw, a, b = x["mixed"], x["a_raw"], x["b_raw"]                 # rows 0..5 the tokens, 6..7 two siblings
    cst0 = x["conv_state"].reshape(C, -1)
    S0 = x["state"].reshape(H, dk, dv)

    def fac():
        return (torch.zeros(H, 32, dk, device="cuda"), torch.zeros(H, 32, dv, device="cuda"),
                torch.zeros(H, 32, device="cuda"))

    def item(rows, parents, cst, S, pend):
        t = DraftTree(tokens=[0] * len(parents), parents=parents)
        ta = G._tree_args(t)
        return dict(mixed=raw[rows].contiguous(), conv_state=cst.clone(), window=ta["window"], anc=ta["anc"],
                    a_raw=a[rows].contiguous(), b_raw=b[rows].contiguous(), state=S, pend=pend, fac_out=fac())

    def run(its):
        return seq_recurrence_grouped(its, **gk, **sw)

    out = []
    # A: one chain of six, then its commit
    SA = S0.clone()
    iA = item(list(range(6)), [-1, 0, 1, 2, 3, 4], cst0, SA, None)
    (oA, fA), = run([iA])
    kA, dA, gA = iA["fac_out"]
    SA_end = SA.clone()
    seq_commit(SA_end[None], kA[None], dA[None], gA[None], list(range(6)))
    # B: three, then three with the first three's commit pending (the fold)
    SB = S0.clone()
    iB1 = item([0, 1, 2], [-1, 0, 1], cst0, SB, None)
    (oB1, _), = run([iB1])
    k1, d1, g1 = iB1["fac_out"]
    cst1 = torch.cat([cst0, raw[0:3].t()], dim=1)[:, -3:].contiguous()
    pend = (k1, d1, g1, torch.arange(3, dtype=torch.int32, device="cuda"),
            torch.tensor([3], dtype=torch.int32, device="cuda"))
    iB2 = item([3, 4, 5], [-1, 0, 1], cst1, SB, pend)
    (oB2, _), = run([iB2])
    SB_end = SB.clone()
    k2, d2, g2 = iB2["fac_out"]
    seq_commit(SB_end[None], k2[None], d2[None], g2[None], [0, 1, 2])
    assert torch.equal(oA[:, :3], oB1), "A vs B: the first block's outputs"
    assert torch.equal(oA[:, 3:], oB2), "A vs B: the outputs after the folded commit"
    assert torch.equal(SA_end, SB_end), "A vs B: the committed state"
    out.append("one 6-row block = two 3-row blocks with the commit folded: outputs and state bit-identical")
    # E: the replay alone (seq_commit) against the replay inside the next block (B2's entry)
    SE = S0.clone()
    seq_commit(SE[None], k1[None], d1[None], g1[None], [0, 1, 2])
    iE = item([3, 4, 5], [-1, 0, 1], cst1, SE, None)
    (oE, _), = run([iE])
    assert torch.equal(oE, oB2), "the replay alone vs in the next block"
    out.append("the commit replayed on its own = replayed inside the next verify: bit-identical")
    # C: the six on one path of an 8-node tree with two siblings
    rows_c = [0, 1, 6, 2, 3, 7, 4, 5]
    par_c = [-1, 0, 1, 1, 3, 4, 4, 6]
    SC = S0.clone()
    iC = item(rows_c, par_c, cst0, SC, None)
    (oC, _), = run([iC])
    path = [0, 1, 3, 4, 6, 7]
    assert torch.equal(oC[:, path], oA), "A vs C: a token's output beside siblings"
    kC, dC, gC = iC["fac_out"]
    SC_end = SC.clone()
    seq_commit(SC_end[None], kC[None], dC[None], gC[None], path)
    assert torch.equal(SC_end, SA_end), "A vs C: the committed state of the path"
    out.append("the same six on a path of an 8-node tree: outputs and committed state bit-identical")
    # D: the three engines in one launch
    # a joint pass hands every engine a pending commit, of 0 rows where there is none
    none = (k1, d1, g1, torch.zeros(3, dtype=torch.int32, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"))
    its = [item(list(range(6)), [-1, 0, 1, 2, 3, 4], cst0, S0.clone(), none),
           item([3, 4, 5], [-1, 0, 1], cst1, S0.clone(), pend),
           item(rows_c, par_c, cst0, S0.clone(), none)]
    mixed = torch.cat([it["mixed"] for it in its])
    ab = torch.cat([torch.cat([it["a_raw"], it["b_raw"]], -1) for it in its])
    r0 = 0
    for it in its:
        T = it["mixed"].shape[0]
        it["mixed"], it["a_raw"], it["b_raw"] = mixed[r0:r0 + T], ab[r0:r0 + T, :H], ab[r0:r0 + T, H:]
        r0 += T
    got = run(its)
    assert torch.equal(got[0][0], oA) and torch.equal(got[1][0], oB2) and torch.equal(got[2][0], oC), "grouped"
    out.append("the three in one grouped launch = each alone: bit-identical")
    # accuracy: the chain against the float64 walk
    conv = G.conv_rows(raw[:6], cst0, x["conv_w"], iA["window"])
    o64, _, _ = G.reference64(conv, a[:6], b[:6], x["a_log"], x["dt_bias"], S0, **G.KW)
    err = ((oA[0].double() - o64).abs().max() / o64.abs().max()).item()
    out.append(f"against the float64 walk: max relative {err:.2e}")
    if keep_outputs:
        return out, [("A", oA), ("B2", oB2), ("C", oC), ("SA", SA_end), ("SC", SC_end), ("D0", got[0][0]),
                     ("D2", got[2][0])]
    return out
