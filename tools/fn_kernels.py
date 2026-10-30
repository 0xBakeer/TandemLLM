"""Flash-Next kernels on the board (ENG-223, ENG-141, ENG-143, ENG-149): CUDA, row-invariant.

Row invariance is the property speculation needs: a row computed alone (the one-token walk) and the
same row inside a verify block of up to 32 rows give the same bits. Every kernel here computes a row
(or a (row, expert) pair) with an instruction sequence that does not depend on the other rows: no
reduction runs across rows, no tile shape depends on M. The `_rows` twins of the convention
(docs/verify-twins.md) are the same entry points called with M > 1.

  gemv          y[M, N] = x[M, K] . W[N, K]^T in fp32, W int8 rows with an fp32 scale (the
                hyper-connection mixers, `pack.Int8Rows`) or fp16 (router, PLE projections, the GDN
                a/b projections). A group of LPR lanes owns one output row; each lane sums its
                strided K slice in K order, the group adds the lanes by a fixed xor tree, then the
                scale. 16-byte weight loads. A weight of few rows (mixer down, router, GDN a/b) splits
                each row's K over the warps of a CTA, the slices added in warp order (gemv_split).
  mix_up        the mixer's second half in one launch: a = bf16(silu(bf16(d) / hc)) (the down
                projection d from `gemv`), the up projection of a, and the stream mean
                x[j] = bf16(sum_s bf16(bf16(sigmoid(bf16(up_s . a))) * n[s H + j]) / hc) -- one warp
                a column j, its four lane groups the four streams.
  router        softmax over the bf16-rounded logits of E <= 512 experts in fp32, the top k with the lower
                index first on ties, weights renormalised in slot order and rounded to bf16 (as the
                reference's `.to(logits.dtype)`), and one more slot for the shared expert: id E,
                weight bf16(sigmoid(bf16(logit E))).
  moe_gu        gate and up of every (row, slot) pair: a CTA owns one pair, one matrix and one
                128-column block. It rotates the pair's input with the expert's own suh into shared
                memory (exactly `exl3_linear.rot_in_kernel`'s arithmetic), runs the trellis GEMV over
                its K range split in eight contiguous warp slices (one row of the m16n8k16 tile is
                real), adds the warps in warp order, and applies the output rotation and svh -> bf16.
  moe_down      the down projection of every pair: the CTA forms bf16(bf16(silu(g)) * u) of its
                pair and rotates it with the expert's suh (`rot_in_silu_kernel`'s arithmetic), the
                same GEMV body, fp32 out.
  moe_combine   out[m] = bf16(sum over the row's slots in ascending expert id of w * y): routed
                experts in id order, then the shared expert (id E), in fp32, rounded once.

The routed and the shared experts are one bank (`MoEBank`): expert E is the shared expert, so the
same kernels read it, at its own K. Only the experts a call selects are read: a pair's CTAs take
their expert's pointers from the bank by the id in the device buffer, nothing else. Pairs are
launched in ascending expert id (`order`), so rows that share an expert read its tiles back to back
(from L2 the second time); the arithmetic of a pair does not depend on where it is launched.

The trellis decode, the Hadamard and the mma helpers are tools/exl3_linear.py's (`_HELPERS`),
which ports the bit extraction and the codebook from exllamav3 (MIT, Copyright (c) 2025
Turboderp; the notice travels with that source). Built on first use for sm_121a.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import native_build as nb  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402
from tools import exl3_linear as X  # noqa: E402

# the helper half of exl3_linear's CUDA source: includes, the licence, the trellis decode, mma,
# cp.async and the 128-point Hadamard; it ends inside the anonymous namespace
_HELPERS = X._CUDA[: X._CUDA.index("constexpr float RSQRT128")]

_CUDA = nb.source(__file__, "kernels.cu")

_CPP = nb.source(__file__, "kernels.cpp")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        src = "#define EXL3_ST 3\n" + _HELPERS + _CUDA
        tag = nb.tag(_CPP, src)
        _MOD = nb.load(f"fn_kernels_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["fn_gemv", "fn_mix_up", "fn_router", "fn_moe_gu", "fn_moe_down",
                                  "fn_moe_combine", "fn_moe_groups", "fn_moe_gu_group", "fn_moe_down_group",
                                  "fn_hc_norm", "fn_inject_norm", "fn_gated_norm",
                                  "fn_set_pdl", "fn_set_gemv_split", "fn_set_gemv_nb", "fn_set_gemv_mc", "fn_set_mixer_mma", "fn_set_moe_st", "fn_set_moe_warps", "fn_attn_prep",
                                  "fn_gate_mul", "fn_dense", "fn_dense_rows", "fn_combine_inject_norm"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
        _MOD.fn_set_pdl(int(_S.get("FN_PDL") == "1"))
        _MOD.fn_set_gemv_split(int(_S.get("FN_GEMV_SPLIT") == "1"))
        _MOD.fn_set_gemv_nb(int(_S.get("FN_GEMV_NB") == "1"))
        _MOD.fn_set_gemv_mc(int(_S.get("FN_GEMV_MC") or 0))
        _MOD.fn_set_mixer_mma(int(_S.get("FN_MIXER_MMA") == "1"))
        _MOD.fn_set_moe_st(int(_S.get("FN_MOE_ST")))
        _MOD.fn_set_moe_warps(int(_S.get("FN_MOE_WARPS")))
    return _MOD


# ----------------------------------------------------------------------------------- weights
class MoEBank:
    """The experts of one layer as the kernels read them: E routed experts and the shared expert as
    expert E. Each projection is a trellis [Kin/16, N/16, 16 K] with suh [Kin] and svh [N] anywhere
    on the device; the bank holds their pointers ([E + 1, 3] for gate, up, down), their K and the
    tensors themselves (`keep`, so the pointers stay valid)."""

    def __init__(self, experts: list, device):
        """experts: [(gate, up, down)] of (trellis, suh, svh) device tensors, the shared expert last."""
        tr, kb, su, sv = [], [], [], []
        keep = []
        for proj in experts:
            for t, s_u, s_v in proj:
                assert t.dtype == torch.int16 and t.is_contiguous() and t.data_ptr() % 16 == 0
                assert s_u.dtype == torch.float16 and s_v.dtype == torch.float16
                tr.append(t.data_ptr())
                kb.append(X.kb_of(t.shape[2]))
                su.append(s_u.data_ptr())
                sv.append(s_v.data_ptr())
                keep += [t, s_u, s_v]
        self.keep = keep
        self.n = len(experts)                       # E + 1
        self.tr = torch.tensor(tr, dtype=torch.int64, device=device)
        self.kb = torch.tensor(kb, dtype=torch.int32, device=device)
        self.su = torch.tensor(su, dtype=torch.int64, device=device)
        self.sv = torch.tensor(sv, dtype=torch.int64, device=device)
        g = experts[0][0]
        self.H = g[0].shape[0] * 16
        self.I = g[0].shape[1] * 16
        # by tile width: the launch sizes its stages for the widest (ks[-1])
        self.ks = sorted(set(kb), key=X.tile_width)
        if not set(self.ks) <= set(MOE_K):
            raise ValueError(f"expert K {self.ks}: the MoE kernels take K 2..8, 1.5, 2.5 and 3.5 (15, 25, 35)")

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keep)

    def block(self, e: int, j: int):
        """Projection j (0 gate, 1 up, 2 down) of expert e as an EXL3Block (views; the reference
        checks dequantise it)."""
        t, s_u, s_v = self.keep[9 * e + 3 * j: 9 * e + 3 * j + 3]
        return X.EXL3Block(t, s_u, s_v)


def gemv(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] bf16 @ w^T -> fp32 [M, N]; w an fp16 tensor [N, K] or int8 rows (`pack.Int8Rows`)."""
    if hasattr(w, "q"):
        q, s = w.q, w.s
    else:
        q, s = w, None
    if out is None:
        out = torch.empty(x.shape[0], q.shape[0], dtype=torch.float32, device=x.device)
    _module().fn_gemv(x, q, s, out)
    return out


def gemv_bf16(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """`gemv` with the fp32 sums rounded to bf16 in the kernel (what `F.linear` returns in bf16)."""
    q, s = (w.q, w.s) if hasattr(w, "q") else (w, None)
    if out is None:
        out = torch.empty(x.shape[0], q.shape[0], dtype=torch.bfloat16, device=x.device)
    _module().fn_gemv(x, q, s, out)
    return out


def hc_norm(x: torch.Tensor, w: torch.Tensor, group: int, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """The reference's grouped RMS norm with weight (1 + w): x [M, n * group] bf16 -> bf16."""
    if out is None:
        out = torch.empty_like(x)
    _module().fn_hc_norm(x, w, out, group, eps)
    return out


def inject_norm(h0: torch.Tensor, out: torch.Tensor, d: torch.Tensor, R: int, w: torch.Tensor | None,
                hc: int, eps: float):
    """h = bf16(h0 + bf16(out * inj)) with inj = 2 sigmoid(bf16(d[:, R:]) / hc) per stream, and the
    next site's norm of h with weight (1 + w) (None: no norm, n is None). Returns (h, n)."""
    hn = torch.empty_like(h0)
    nn = torch.empty_like(h0) if w is not None else hn
    _module().fn_inject_norm(h0, out, d, R, w, hn, nn, hc, eps)
    return hn, (nn if w is not None else None)


def gated_norm(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, sigmoid: bool) -> torch.Tensor:
    """The GDN output norm over head rows of 128: bf16(bf16(w * bf16(norm(o))) * act(z)), plain w."""
    y = torch.empty_like(o)
    _module().fn_gated_norm(o.contiguous(), z.contiguous(), w, y, eps, sigmoid)
    return y


# (N, Kin, K) -> (cluster split S, warps); a shape not named takes `dense_split`'s default
DENSE_SPLIT: dict = {}


def dense_split(N: int, Kin: int, kb: int) -> tuple[int, int]:
    """The split of an EXL3 projection for `dense`: a cluster of S CTAs over K when the output has
    few 128-column blocks (keeping >= ~96 CTAs), 4 warps a CTA. Keyed by the shape only: one order
    for every row count."""
    got = DENSE_SPLIT.get((N, Kin, kb))
    if got:
        return got
    S = 1
    while (N // 128) * S < 96 and S < 8 and (Kin // 16) % (8 * S * 2) == 0:
        S *= 2
    return S, 4


def dense(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M <= 32, Kin] bf16 @ EXL3Block^T -> [M, N] (bf16, or fp32 for an `out_fp32` block): the input
    rotation fused, K over a cluster, programmatic dependent launch; each row its own programs, so an
    M-row call's rows are its one-row calls' bits."""
    S, warps = dense_split(w.tile_n, w.K, w.kb)
    if out is None:
        out = torch.empty(x.shape[0], w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=x.device)
    _module().fn_dense(x, w.t, w.suh, w.svh, out, w.bounds[0], w.bounds[1], S, warps)
    return out


def dense_rows(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """`dense` for a verify block (SPD-125): each trellis tile decoded once for up to 16 rows (rows
    of the m16n8k16 tile), the inputs rotated once a row first. Every output byte is the one-row
    `dense` call's on that row (same split, same K slices, warp and rank order); an opt-in for M > 1."""
    S, warps = dense_split(w.tile_n, w.K, w.kb)
    if out is None:
        out = torch.empty(x.shape[0], w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=x.device)
    xh = torch.empty(w.suh.shape[0], x.shape[0], w.K, dtype=torch.float16, device=x.device)
    _module().fn_dense_rows(x, w.t, w.suh, w.svh, xh, out, w.bounds[0], w.bounds[1], S, warps)
    return out


def attn_prep(qkv, iqk, pos, qw, kw, cos, sin, kc, vc, ikc, hq: int, hk: int, ioff: int, eps: float, slot=None,
              qln=None, iq=None):
    """The attention inputs of M rows: query heads normed (1 + w) and rotated at the row's position ->
    [M, Hq, D]; key heads the same and value heads as they are, into the caches kc/vc [Hk, cap, D] at
    the row's slot; the indexer key iqk[:, ioff:ioff + ihd] into ikc [cap, ihd]. qkv [M, Hq 2D + 2 Hk D],
    pos / slot int64 [M] on the device (slot None: the position). With `iq` ([M, heads, ihd] bf16) and
    `qln` (q_layernorm): the indexer query heads iqk[:, :heads ihd], normed (1 + w) and rotated, into iq."""
    D = kc.shape[2]
    qkv = qkv.reshape(-1, qkv.shape[-1])
    q = torch.empty(qkv.shape[0], hq, D, dtype=torch.bfloat16, device=qkv.device)
    _module().fn_attn_prep(qkv, None if iqk is None else iqk.reshape(-1, iqk.shape[-1]), pos,
                           pos if slot is None else slot, qw, kw, cos, sin, q, kc, vc, ikc, hq, hk, ioff, eps, qln, iq)
    return q


def gate_mul(o: torch.Tensor, qkv: torch.Tensor, D: int) -> torch.Tensor:
    """bf16(o * bf16(sigmoid(gate))) per row and query head, the gate beside each head in qkv's row:
    o [M, Hq, D] (any view of M Hq D contiguous values) -> [M, Hq D]."""
    qkv = qkv.reshape(-1, qkv.shape[-1])
    y = torch.empty(qkv.shape[0], o.numel() // qkv.shape[0], dtype=torch.bfloat16, device=o.device)
    _module().fn_gate_mul(o.contiguous(), qkv, y, D)
    return y


def mix_up(d: torch.Tensor, n: torch.Tensor, w_up, hc: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The mixer's second half: d [M, >=R] fp32 (the down projection), n [M, hc H] bf16 (the normed
    streams), w_up [hc H, R] fp16 or Int8Rows -> x [M, H] bf16."""
    q, s = (w_up.q, w_up.s) if hasattr(w_up, "q") else (w_up, None)
    H = q.shape[0] // hc
    if out is None:
        out = torch.empty(d.shape[0], H, dtype=torch.bfloat16, device=d.device)
    # a launch stages its rows' activations in shared memory: at most MIX_ROWS rows a launch
    for r0 in range(0, d.shape[0], MIX_ROWS):
        _module().fn_mix_up(d[r0:r0 + MIX_ROWS], n[r0:r0 + MIX_ROWS], q, s, out[r0:r0 + MIX_ROWS], hc)
    return out


MIX_ROWS = 64


def router(logits: torch.Tensor, E: int, k: int, ids=None, wts=None):
    """logits fp32 [M, E + 1] (the shared gate last) -> ids int32 [M, k + 1], wts fp32 [M, k + 1]."""
    M = logits.shape[0]
    if ids is None:
        ids = torch.empty(M, k + 1, dtype=torch.int32, device=logits.device)
        wts = torch.empty(M, k + 1, dtype=torch.float32, device=logits.device)
    _module().fn_router(logits, E, k, ids, wts)
    return ids, wts


def moe(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, bank: MoEBank, order: torch.Tensor | None = None,
        bufs: dict | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """The experts of every (row, slot) pair and their combine: x [M, H] bf16, ids/wts [M, S]
    (S = top-k + the shared slot) -> [M, H] bf16. `order` (int32 [M S]) is the launch order of the
    pairs; ascending expert id by default (`expert_order`). `bufs` holds preallocated
    intermediates for a graph."""
    M, S = ids.shape
    P = M * S
    mod = _module()
    if order is None:
        if M > 1:
            order = expert_order(ids)
        else:
            order = _EMPTY.get(x.device)
            if order is None:
                order = _EMPTY[x.device] = torch.empty(0, dtype=torch.int32, device=x.device)
    b = bufs if bufs is not None else {}
    ygu = b.get("ygu")
    if ygu is None or ygu.shape[0] != P:
        ygu = torch.empty(P, 2, bank.I, dtype=torch.bfloat16, device=x.device)
        yd = torch.empty(P, bank.H, dtype=torch.float32, device=x.device)
        if bufs is not None:
            bufs["ygu"], bufs["yd"] = ygu, yd
    else:
        yd = b["yd"]
    flat = ids.reshape(-1)
    mod.fn_moe_gu(x, flat, order, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1])
    mod.fn_moe_down(ygu, flat, order, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
    if out is None:
        out = torch.empty(M, bank.H, dtype=torch.bfloat16, device=x.device)
    mod.fn_moe_combine(yd, flat, wts.reshape(-1), out)
    return out


_EMPTY: dict = {}


def moe_experts(x: torch.Tensor, ids: torch.Tensor, bank: MoEBank, order: torch.Tensor | None = None) -> torch.Tensor:
    """`moe` without its combine: every pair's down-projection output yd [M S, H] fp32 (unweighted),
    for `combine_inject_norm`."""
    M, S = ids.shape
    if order is None:
        if M > 1:
            order = expert_order(ids)
        else:
            order = _EMPTY.get(x.device)
            if order is None:
                order = _EMPTY[x.device] = torch.empty(0, dtype=torch.int32, device=x.device)
    mod = _module()
    ygu = torch.empty(M * S, 2, bank.I, dtype=torch.bfloat16, device=x.device)
    yd = torch.empty(M * S, bank.H, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1)
    if MOE_GROUP and M > 1:
        # ENG-241: the pairs of one expert in groups of MOE_GR, its tiles decoded once a group (a
        # row's bits are its own: the self-check compares them with the decode step's single row)
        g = torch.empty(M * S, 2 + MOE_GR, dtype=torch.int32, device=x.device)
        ng = torch.empty(1, dtype=torch.int32, device=x.device)
        mod.fn_moe_groups(flat, order, g, ng)
        mod.fn_moe_gu_group(x, g, ng, S, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1])
        mod.fn_moe_down_group(ygu, g, ng, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
        return yd
    mod.fn_moe_gu(x, flat, order, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1])
    mod.fn_moe_down(ygu, flat, order, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
    return yd


#: ENG-241: a block's pairs of one expert share a CTA, up to MOE_GR of them (False: a CTA a pair,
#: the A/B of tools/fn_moe_group_check.py)
MOE_GROUP = False
MOE_GR = 8
#: the experts' bits the MoE kernels decode (the half-integer rates as ten times the rate)
MOE_K = (2, 3, 4, 5, 6, 7, 8, 15, 25, 35)


def moe_combine(yd: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """The combine alone: yd [M S, H] fp32 (every pair's expert output), ids / wts [M, S] -> [M, H] bf16
    (the slots of a row in ascending expert id, fp32, one rounding; `moe`'s last launch)."""
    M = ids.shape[0]
    if out is None:
        out = torch.empty(M, yd.shape[1], dtype=torch.bfloat16, device=yd.device)
    _module().fn_moe_combine(yd, ids.reshape(-1), wts.reshape(-1), out)
    return out


def combine_inject_norm(h0, yd, ids, wts, d, R: int, w, hc: int, eps: float):
    """`inject_norm` of the MoE's output, its combine done in the same launch (the bits of
    `moe_combine` followed by `inject_norm`). Returns (h, n)."""
    hn = torch.empty_like(h0)
    nn = torch.empty_like(h0) if w is not None else hn
    _module().fn_combine_inject_norm(h0, yd, ids.reshape(-1), wts.reshape(-1), d, R, w, hn, nn, hc, eps)
    return hn, (nn if w is not None else None)


def expert_order(ids: torch.Tensor) -> torch.Tensor:
    """The pairs in ascending expert id (stable), as int32; fixed shape, no host sync."""
    return torch.sort(ids.reshape(-1), stable=True).indices.to(torch.int32)


# ------------------------------------------------------------------------------- references
def moe_reference(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, experts: list) -> torch.Tensor:
    """The kernels' arithmetic in torch, per pair, fp32 GEMV in place of the tensor-core sums:
    experts[e] = (gate, up, down) EXL3Blocks. Equal to the kernels within fp32 summation order
    (the trellis decode, the rotations and every bf16 rounding are the same)."""
    from tools.exl3_linear import hadamard128
    M, S = ids.shape
    H = x.shape[1]
    out = torch.zeros(M, H, dtype=torch.float32, device=x.device)
    Hd = hadamard128(x.device)

    def rot(v, suh):
        n = v.shape[-1]
        return ((v.float() * suh.float()).view(-1, n // 128, 128) @ Hd).view(-1, n).to(torch.float16).float()

    def gemm(xh, w):
        yp = xh @ w.inner().float()
        n = yp.shape[-1]
        return (yp.view(-1, n // 128, 128) @ Hd).view(-1, n) * w.svh.float()

    for m in range(M):
        es = ids[m].tolist()
        for s in sorted(range(S), key=lambda s: es[s]):
            g, u, d = experts[es[s]]
            xr = x[m:m + 1]
            yg = gemm(rot(xr, g.suh[0]), g).to(torch.bfloat16).float()
            yu = gemm(rot(xr, u.suh[0]), u).to(torch.bfloat16).float()
            a = (F_silu(yg).to(torch.bfloat16).float() * yu).to(torch.bfloat16)
            yd = gemm(rot(a, d.suh[0]), d)
            out[m] = out[m] + float(wts[m, s]) * yd[0]
    return out.to(torch.bfloat16)


def F_silu(v: torch.Tensor) -> torch.Tensor:
    return v / (1.0 + torch.exp(-v))


def router_reference(logits: torch.Tensor, E: int, k: int):
    """The router kernel's arithmetic in torch (ids int32, wts fp32)."""
    lb = logits[:, :E].to(torch.bfloat16).float()
    p = torch.softmax(lb, dim=-1)
    v, i = torch.sort(p, dim=-1, descending=True, stable=True)
    v, i = v[:, :k], i[:, :k]
    tot = v[:, 0].clone()
    for j in range(1, k):
        tot = tot + v[:, j]
    w = (v / tot[:, None]).to(torch.bfloat16).float()
    sg = torch.sigmoid(logits[:, E].to(torch.bfloat16).float()).to(torch.bfloat16).float()
    ids = torch.cat([i, torch.full_like(i[:, :1], E)], 1).to(torch.int32)
    return ids, torch.cat([w, sg[:, None]], 1)
