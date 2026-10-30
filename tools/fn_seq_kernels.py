"""Flash-Next's sequence kernels (ENG-227, ENG-147): the parts of a row that read OTHER positions.

A row of the trunk reads other positions in four places: the GDN recurrence (the state after the
row's parent), the GDN convolution (the three positions before it), the PLE convolution (the
positions 3, 6 and 9 before it) and attention (every visible key). In a verify block the positions
before a row are its ancestors, which are not the rows before it in a tree, and some of them sit in
the committed state rather than in the block. Every kernel here therefore takes, per row, WHERE its
predecessors are, and computes the row with an instruction sequence that does not depend on the
other rows or on where they are. The decode step is the same kernels at one row: a verified row is
the decode step's row bit for bit, which is the property lossless speculation needs (docs/kernels.md,
"Verify twins").

  gdn_pre_rows   the GDN depthwise causal conv (4 taps) and SiLU over the row's window, and the
                 gates g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b). The window's taps
                 come from rows of the block or from the conv state (`src`). One row (decode) can
                 also shift the state in place.
  gdn_rows       the gated delta rule, row after row. A CTA owns one value head and 16 value
                 columns of its [128, 128] state; lane l holds key rows l, l + 32, l + 64, l + 96 of
                 four columns, so both reductions over the key dimension are one lane's four values
                 in order and then a fixed xor tree inside a warp. Per row the tile comes from the
                 previous row (registers), the entry state, or a scratch slot where an earlier row
                 with a second child left it (`ctl`); the tile is written back to the state only
                 when asked (decode, commit). Parked rows (`pre`, the last commit's path) are walked
                 first and written back: the commit folded into the next call.
  ple_gate       PLE's gate: g = bf16(sum bf16(key query) / sqrt(H)) per (row, stream), its signed
                 square root, gv = bf16(bf16(sigmoid(g)) value) -- the reference's bf16 roundings.
  ple_conv       PLE's dilated depthwise conv (4 taps, dilation 3) over the normed gv of the row and
                 of the positions 3, 6, 9 before it, SiLU, + gv.
  attn_rows      softmax(q k^T / sqrt(D)) v of every query head of one row over the row's visible
                 keys, in VIRTUAL order: visible key c is position c (dense) or the c-th selected
                 position (QSA, ENG-147); a position before the block's start L is its cache slot, a
                 later one is the row's ancestor at that depth (`anc`). The virtual columns are cut
                 into fixed chunks of 256 (nine cover the 2,051 keys a query ever sees), each chunk a
                 program, combined in chunk order. A row's arithmetic is a function of its own query
                 and visible keys only.
  qsa_pool       a pooled indexer key: the mean of a block's 4 raw keys, the norm, the rope at the
                 block's first position (the shared pool, and a verify row's own blocks)
  qsa_scores     a row's block scores: relu(q_h . pooled) summed over the heads in order
  qsa_topk       the 512 best blocks of a row (ties to the lower block), ascending, then the tail:
                 one bitonic sort in shared memory a row

Built on first use for sm_121a (CUDA via torch.utils.cpp_extension) and Triton (attention).
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import native_build as nb  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

#: virtual columns a split program of `attn_rows` owns, and the splits: a query never sees more than
#: budget + compress_ratio - 1 = 2,051 keys (dense up to that context, 512 selected blocks of 4 and a
#: tail of at most 3 past it), so nine chunks of 256 hold every row at every context length
ATTN_CHUNK = 256
ATTN_SPLITS = 9
#: depths an ancestor table holds (a verify block is at most 32 rows)
ANC = 33
#: visible positions a row's QSA selection list holds (2,048 selected + a tail of at most 3, padded)
NSEL = 2064

_CUDA = nb.source(__file__, "seq.cu")

_CPP = nb.source(__file__, "seq.cpp")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        tag = nb.tag(_CPP, _CUDA)
        _MOD = nb.load(f"fn_seq_kernels_{tag}", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                       functions=["seq_set_pdl", "seq_commit_multi", "seq_gdn_rows_multi", "seq_gdn_pre", "seq_gdn_rows", "seq_ple_gate", "seq_ple_conv",
                                  "seq_qsa_pool", "seq_qsa_pool_multi", "seq_qsa_scores", "seq_qsa_topk"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
        _MOD.seq_set_pdl(int(_S.get("FN_PDL") == "1"))
    return _MOD


# ------------------------------------------------------------------------------------- commits
#: the int64 fields of a `commit_multi` entry
COMMIT_FIELDS = 16


def commit_multi(entries: torch.Tensor) -> None:
    """SPD-239: the copies of many commits in one launch, `entries` int64 [n, COMMIT_FIELDS] on the device: a row
    gather (kind 0) or a bf16 window shifted in place from a block's rows (kind 1); the layouts at the kernel."""
    _module().seq_commit_multi(entries)


def rows_entry(src: torch.Tensor, dst: torch.Tensor, idx: torch.Tensor, n: int) -> list:
    """A kind 0 entry: dst rows 0..n-1 = src rows idx (rows of 4-byte words, unit stride within a row)."""
    eb = src.element_size()
    words = src.shape[1] * eb // 4
    assert src.stride(1) == 1 and dst.stride(1) == 1 and (src.shape[1] * eb) % 4 == 0 and src.dtype == dst.dtype
    return [0, src.data_ptr(), src.stride(0) * eb // 4, dst.data_ptr(), dst.stride(0) * eb // 4, words,
            idx.data_ptr(), n] + [0] * (COMMIT_FIELDS - 8)


def window_entry(win: torch.Tensor, src: torch.Tensor, cols: list[int]) -> list:
    """A kind 1 entry: the bf16 window [1, C, w] gets cols (`_tail_cols`) of [its columns | src's rows]."""
    C, w = win.shape[-2], win.shape[-1]
    assert win.is_contiguous() and win.dtype == src.dtype == torch.bfloat16 and src.stride(1) == 1 and w <= 9
    assert len(cols) == w and src.shape[1] >= C
    return [1, win.data_ptr(), src.data_ptr(), src.stride(0), C, w, 0] + list(cols) + [0] * (COMMIT_FIELDS - 7 - w)


# ------------------------------------------------------------------------------------- GDN
def gdn_pre(x, conv_state, conv_w, src, ab, a_log, dt_bias, shift: bool = False, out=None):
    """x [M, >= C] bf16 (the qkv projection rows), conv_state [1, C, 3] bf16, conv_w [C, 4] bf16,
    src int32 [M, 3] (the taps at distance 3, 2, 1: a row, or -1 - state column), ab [M, 2 H] bf16,
    a_log / dt_bias fp32 [H] -> (y [M, C] bf16 = silu(conv), g [M, H] fp32, beta [M, H] fp32).
    `shift` (one row): the state advances in place, as the decode step's."""
    M = x.shape[0]
    C = conv_w.shape[0]
    H = a_log.numel()
    y = out if out is not None else torch.empty(M, C, dtype=torch.bfloat16, device=x.device)
    g = torch.empty(M, H, dtype=torch.float32, device=x.device)
    beta = torch.empty(M, H, dtype=torch.float32, device=x.device)
    _module().seq_gdn_pre(x, conv_state, conv_w, src, y, ab, a_log, dt_bias, g, beta, shift)
    return y, g, beta


def gdn_rows(qkv, key_dim: int, g, beta, S, scratch, ctl, rep: int, write: bool, out=None, pre=None):
    """The delta rule over M rows of qkv [M, 2 key_dim + H 128] bf16 (q | k | v after the conv),
    g / beta [M, H] fp32, S [1, H, 128, 128] fp32 (the entry state; written with `write`), scratch
    fp32 [slots, H, 128, 128], ctl int32 [M, 2] (load, save) -> out [M, H, 128] bf16. `pre` = (rows
    [P, C] bf16, g [P, H], beta [P, H], n int32 [1] on the device): n parked rows applied to S first
    and S written back (a commit folded into this call; n = 0 does nothing)."""
    M = qkv.shape[0]
    H = S.shape[-3]
    if out is None:
        out = torch.empty(M, H, 128, dtype=torch.bfloat16, device=qkv.device)
    pr, pg, pb, pn = pre if pre is not None else (None, None, None, None)
    _module().seq_gdn_rows(qkv, key_dim, 2 * key_dim, g, beta, S, scratch, ctl, out, rep, write, pr, pg, pb, pn)
    return out


def gdn_rows_entry(qkv, g, beta, S, scratch, ctl, out, write: bool, pre=None) -> list:
    """SPD-239: one state's `gdn_rows` call as a `gdn_rows_multi` entry (the same tensors and checks' shapes; the
    scratch is this state's own slots)."""
    M = qkv.shape[0]
    assert qkv.dtype == torch.bfloat16 and qkv.stride(1) == 1 and S.is_contiguous() and scratch.is_contiguous()
    assert ctl.dtype == torch.int32 and ctl.is_contiguous() and ctl.numel() == 2 * M and out.is_contiguous()
    assert g.is_contiguous() and beta.is_contiguous() and g.dtype == beta.dtype == torch.float32
    pr, pg, pb, pn = pre if pre is not None else (None, None, None, None)
    return [qkv.data_ptr(), qkv.stride(0), g.data_ptr(), beta.data_ptr(), S.data_ptr(), scratch.data_ptr(),
            ctl.data_ptr(), M, out.data_ptr(), pr.data_ptr() if pr is not None else 0,
            pr.stride(0) if pr is not None else 0, pg.data_ptr() if pg is not None else 0,
            pb.data_ptr() if pb is not None else 0, pn.data_ptr() if pn is not None else 0, int(write), 0]


def gdn_rows_multi(entries: torch.Tensor, key_dim: int, H: int, rep: int) -> None:
    """SPD-239: `gdn_rows` of several states in one launch (`entries` int64 [states, 16] on the device, one
    `gdn_rows_entry` a state): every CTA computes what the state's own call computes."""
    _module().seq_gdn_rows_multi(entries, key_dim, 2 * key_dim, H, rep)


def gdn_rows_reference(qkv, key_dim: int, g, beta, S, parents: list[int], rep: int):
    """The recurrence in torch over a tree (`parents`, -1 for the entry state), fp32: (out [M, H, 128]
    fp32, the state after each row [M, H, 128, 128])."""
    M = qkv.shape[0]
    H = S.shape[-3]
    q = qkv[:, :key_dim].float().view(M, -1, 128).repeat_interleave(rep, 1)
    k = qkv[:, key_dim:2 * key_dim].float().view(M, -1, 128).repeat_interleave(rep, 1)
    v = qkv[:, 2 * key_dim:2 * key_dim + H * 128].float().view(M, H, 128)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    states, outs = [], []
    for m in range(M):
        s = (S.view(H, 128, 128) if parents[m] < 0 else states[parents[m]]).clone()
        s = s * torch.exp(g[m]).view(H, 1, 1)
        kv = torch.einsum("hkv,hk->hv", s, k[m])
        delta = (v[m] - kv) * beta[m].view(H, 1)
        s = s + k[m].unsqueeze(-1) * delta.unsqueeze(1)
        outs.append(torch.einsum("hkv,hk->hv", s, q[m]))
        states.append(s)
    return torch.stack(outs), torch.stack(states)


def tree_ctl(parents: list[int]) -> tuple[list[tuple[int, int]], int]:
    """The (load, save) of every row of a DFS pre-order block for `gdn_rows`, and the scratch slots it
    uses: a row continues from the previous row's tile when that row is its parent, starts from the
    entry state when it is a root, and otherwise loads its parent's tile from the slot the parent
    saved it in (a parent saves when it has a child other than the next row)."""
    M = len(parents)
    need = [False] * M
    for i in range(1, M):
        p = parents[i]
        if p >= 0 and p != i - 1:
            need[p] = True
    slot, n = {}, 0
    for i in range(M):
        if need[i]:
            slot[i] = n
            n += 1
    out = []
    for i, p in enumerate(parents):
        if i == 0 or p < 0:
            ld = -2
        elif p == i - 1:
            ld = -1
        else:
            ld = slot[p]
        out.append((ld, slot.get(i, -1)))
    return out, n


def window_src(parents: list[int], dists: tuple[int, ...], state_len: int) -> list[list[int]]:
    """For every row of a block, where the position `d` before it lives, for each d in `dists`: the
    row of its ancestor at that distance (>= 0), or -1 - the column of the state ([.., state_len]
    columns, the last the newest) when the walk leaves the block."""
    out = []
    for i in range(len(parents)):
        anc = [i]
        while parents[anc[-1]] >= 0:
            anc.append(parents[anc[-1]])
        row = []
        for d in dists:
            if d < len(anc):
                row.append(anc[d])
            else:
                row.append(-1 - (state_len - (d - len(anc) + 1)))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------- PLE
def ple_gate(key, qry, val, hc: int):
    """key, qry [M, hc H] bf16 (normed), val [M, H] bf16 -> gv [M, hc H] bf16."""
    gv = torch.empty_like(key)
    _module().seq_ple_gate(key, qry, val, gv, hc)
    return gv


def ple_conv(gvn, gv, state, w_bf16, src):
    """gvn, gv [M, C] bf16, state [1, C, slen] bf16, w [C, 4] bf16, src int32 [M, 3] (distance 9, 6, 3)
    -> gv + silu(conv) [M, C] bf16."""
    y = torch.empty_like(gv)
    _module().seq_ple_conv(gvn, gv, state, w_bf16, src, y)
    return y


# ------------------------------------------------------------------------------------- QSA
#: blocks a row may own past the shared pool: a block that holds one of the round's positions is
#: pooled per row from the row's own path (a verify block is at most 32 rows past its start)
OWN = 9
#: the selection: budget / compress ratio blocks of 4 keys
TOPK = 512


def qsa_pool(ikc, slots, posb, kln, cos, sin, eps: float, out=None):
    """Pooled indexer keys: ikc [cap, 128] bf16 (raw keys), slots int32 [NB, 4] (the block's key
    slots), posb int32 [NB] (its first position), kln [128] (k_layernorm, 1 + w), cos / sin [L, ROT]
    -> [NB, 128] bf16 (mean, norm, rope at the block's first position)."""
    NB = posb.numel()
    if out is None:
        out = torch.empty(NB, 128, dtype=torch.bfloat16, device=ikc.device)
    _module().seq_qsa_pool(ikc, slots, posb, kln, cos, sin, out, eps)
    return out


def qsa_pool_entry(ikc, slots, posb, kln, pool, rows) -> list:
    """SPD-239: one (state, layer)'s `qsa_pool` + `pool.index_copy_(0, rows, ...)` as a `qsa_pool_multi` entry."""
    assert ikc.is_contiguous() and pool.is_contiguous() and slots.is_contiguous() and posb.is_contiguous()
    assert pool.shape[-1] == 128 and rows.dtype == torch.int64 and rows.is_contiguous()
    return [ikc.data_ptr(), slots.data_ptr(), posb.data_ptr(), kln.data_ptr(), pool.data_ptr(), rows.data_ptr()] + [0] * 10


def qsa_pool_multi(entries: torch.Tensor, NB: int, cos, sin, eps: float) -> None:
    """SPD-239: `qsa_pool` of many (state, layer) pairs, each block written at its pool row, in one launch."""
    _module().seq_qsa_pool_multi(entries, NB, cos, sin, eps)


def qsa_scores(iq, pool, own, qctl, NB: int, out=None):
    """Block scores of M rows: iq [M, NH, 128] bf16 (normed, rotated), pool [*, 128] (committed
    blocks), own [M, OWN, 128] (the row's blocks from qctl[:, 1] on), qctl int32 [M, 4] (complete
    blocks, first own block) -> fp32 [M, NB], -inf past a row's complete blocks."""
    M = iq.shape[0]
    if out is None:
        out = torch.empty(M, NB, dtype=torch.float32, device=iq.device)
    _module().seq_qsa_scores(iq, pool, own, qctl, out)
    return out


def qsa_select(sc, tail, sel):
    """The visible positions of each row from its block scores: the TOPK best blocks (ties to the
    lower block: a stable sort), in ascending order, 4 positions each, then the row's tail positions
    (tail int32 [M, 3], the positions past the last complete block; `actl` holds how many are real)
    -> written into sel int32 [M, >= 4 TOPK + 3]. Fixed shapes, no host sync. One kernel a call up to
    8,192 blocks (32k of context), torch sorts past it (and on the CPU); the same selection."""
    if sc.is_cuda and sc.shape[1] <= 8192:
        _module().seq_qsa_topk(sc, tail, sel, TOPK)
        return sel
    return qsa_select_torch(sc, tail, sel)


def qsa_select_torch(sc, tail, sel):
    """`qsa_select` in torch ops (the reference for the kernel; contexts past 32k)."""
    M = sc.shape[0]
    top = torch.sort(sc, dim=1, descending=True, stable=True).indices[:, :TOPK]
    blk = torch.sort(top, dim=1).values
    pos = (blk.unsqueeze(-1) * 4 + torch.arange(4, device=sc.device)).view(M, 4 * TOPK)
    sel[:, :4 * TOPK].copy_(pos)
    sel[:, 4 * TOPK:4 * TOPK + 3].copy_(tail)
    return sel


def qsa_reference(iq, keys, kln, cos, sin, p: int, eps: float):
    """The reference's selection for one query at position p over raw keys [>p, 128] (engine/models/
    flashnext/ref.py qsa_select's arithmetic in torch): the visible positions in ascending order."""
    ratio = 4
    nblk = (p + 1) // ratio
    if nblk <= TOPK:
        return torch.arange(p + 1, device=keys.device)
    g = keys[: nblk * ratio].view(nblk, ratio, -1).float().mean(1).to(torch.bfloat16).float()
    g = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps) * (1 + kln.float())).to(torch.bfloat16)
    starts = torch.arange(nblk, device=keys.device) * ratio
    c, s = cos[starts], sin[starts]
    d = c.shape[-1]
    xr = g[:, :d]
    rot = torch.cat([-xr[:, d // 2:], xr[:, :d // 2]], -1)
    g = torch.cat([((xr * c).to(torch.bfloat16) + (rot * s).to(torch.bfloat16)).to(torch.bfloat16), g[:, d:]], -1)
    sc = torch.relu(iq.float() @ g.float().T).sum(0) / 128 ** 0.5
    sel = torch.sort(torch.sort(sc, descending=True, stable=True).indices[:TOPK]).values
    tok = (sel.unsqueeze(-1) * ratio + torch.arange(ratio, device=keys.device)).flatten()
    return torch.cat([tok, torch.arange(nblk * ratio, p + 1, device=keys.device)])


# ------------------------------------------------------------------------------- attention
if HAVE_TRITON:

    @triton.jit
    def _attn_rows_split(Q, K, V, CTL, ANCT, SEL, PM, PL, PACC,
                         s_kh, s_kn, s_vh, s_vn,
                         SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
                         HK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CHUNK: tl.constexpr,
                         NA: tl.constexpr, NSEL: tl.constexpr, NS: tl.constexpr):
        # CTL[t] = (visible keys n, block start L, dense) int32
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(CTL + t * 4)
        L = tl.load(CTL + t * 4 + 1)
        dense = tl.load(CTL + t * 4 + 2)
        rows = tl.arange(0, BM)
        rmask = rows < REP
        qh = kvh * REP + rows
        od = tl.arange(0, D)
        q = tl.load(Q + (t * HQ + qh[:, None]) * D + od[None, :], mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, n)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            sel = tl.load(SEL + t * NSEL + cols, mask=cmask & (dense == 0), other=0)
            pos = tl.where(dense != 0, cols, sel)
            tail = pos >= L
            a = tl.load(ANCT + t * NA + (pos - L), mask=cmask & tail, other=0)
            slot = tl.where(tail, a, pos)
            k = tl.load(K + kvh * s_kh + slot[:, None] * s_kn + od[None, :], mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k))
            s = s * SCALE
            vis = cmask[None, :] & rmask[:, None]
            s = tl.where(vis, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + slot[:, None] * s_vn + od[None, :], mask=cmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = ((t * HK + kvh) * NS + sp) * BM + rows
        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base[:, None] * D + od[None, :], acc)

    @triton.jit
    def _attn_rows_combine(PM, PL, PACC, OUT, REP: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
                           HK: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr):
        t = tl.program_id(0)
        qh = tl.program_id(1)
        kvh = qh // REP
        r = qh % REP
        od = tl.arange(0, D)
        b0 = (t * HK + kvh) * NS * BM + r
        mx = tl.load(PM + b0)
        for s in tl.static_range(1, NS):
            mx = tl.maximum(mx, tl.load(PM + b0 + s * BM))
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        num = tl.zeros([D], dtype=tl.float32)
        den = 0.0
        # in chunk order: the chunks past a row's last key add exact zeros
        for s in tl.static_range(0, NS):
            w = tl.exp(tl.load(PM + b0 + s * BM) - mx)
            num = num + tl.load(PACC + (b0 + s * BM) * D + od) * w
            den = den + tl.load(PL + b0 + s * BM) * w
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + (t * HQ + qh) * D + od, o.to(tl.bfloat16))


def attn_rows(q, kc, vc, ctl, anc, sel, bufs: dict | None = None):
    """q [M, Hq, D] bf16 (normed, rotated); kc / vc [Hk, cap, D] bf16 (the caches); ctl int32 [M, 4]
    (visible keys, block start L, dense flag, -); anc int32 [M, ANC] (the cache slot of the row's
    ancestor at each depth, 0 the block's first row); sel int32 [M, NSEL] (QSA: the visible
    positions in ascending order; read only for rows not dense) -> o [M, Hq, D] bf16."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    REP = HQ // HK
    BM = 16
    assert REP <= BM and D == kc.shape[2]
    b = bufs if bufs is not None else {}
    pm = b.get("pm")
    if pm is None or pm.shape[0] != M * HK * ATTN_SPLITS * BM:
        pm = torch.empty(M * HK * ATTN_SPLITS * BM, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(pm.shape[0], D, dtype=torch.float32, device=q.device)
        if bufs is not None:
            bufs.update(pm=pm, pl=pl, pacc=pacc)
    else:
        pl, pacc = b["pl"], b["pacc"]
    out = torch.empty(M, HQ, D, dtype=torch.bfloat16, device=q.device)
    _attn_rows_split[(M, HK, ATTN_SPLITS)](
        q, kc, vc, ctl, anc, sel, pm, pl, pacc,
        kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
        SCALE=D ** -0.5, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, BN=32, CHUNK=ATTN_CHUNK,
        NA=anc.shape[1], NSEL=sel.shape[1], NS=ATTN_SPLITS, num_warps=4, num_stages=2)
    _attn_rows_combine[(M, HQ)](pm, pl, pacc, out, REP=REP, D=D, HQ=HQ, HK=HK, BM=BM, NS=ATTN_SPLITS,
                                num_warps=4)
    return out


def attn_rows_reference(q, kc, vc, ctl, anc, sel):
    """`attn_rows` in torch, fp32, from the same visible-key lists."""
    M, HQ, D = q.shape
    HK = kc.shape[0]
    rep = HQ // HK
    out = []
    for t in range(M):
        n, L, dense = (int(x) for x in ctl[t, :3])
        pos = torch.arange(n, device=q.device) if dense else sel[t, :n].long()
        slot = torch.where(pos >= L, anc[t].long()[(pos - L).clamp(min=0, max=anc.shape[1] - 1)], pos)
        k = kc[:, slot].float().repeat_interleave(rep, 0)
        v = vc[:, slot].float().repeat_interleave(rep, 0)
        s = torch.einsum("hd,hnd->hn", q[t].float(), k) * D ** -0.5
        out.append(torch.einsum("hn,hnd->hd", torch.softmax(s, -1), v))
    return torch.stack(out)
