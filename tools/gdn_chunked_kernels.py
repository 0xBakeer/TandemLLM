"""SPD-245: the GDN prefill recurrence of the GDN_PREP path at a 32-row chunk (QWEN38_GDN_CHUNKED).

`tools/gdn_prep_kernels.py chunk_prefill_normed` runs the chunked gated delta rule of
`tools/gdn_prefill_kernels.py` at 64 rows: `_intra` (the UT transform of each chunk, in parallel) and
`_scan` (the walk over the chunks with the fp32 state tile in registers). Both run at 255 registers
and spill (`_intra` 162, `_scan` 142 at 16 x 4): one `_intra` program a SM, two `_scan` programs, so
the 384 scan programs of a 1,024-row call run in four waves and a call costs ten times the latency
of its 16 chunk steps.

At 32 rows the same arithmetic fits: the UT transform (`_intra_c`, `_intra`'s statements with k read
again after the inverse) holds 217 registers and does not spill, its forward substitution is 31 steps
on a 32 x 32 tile instead of 63 on 64 x 64, and the walk's tiles are half the size. The walk
(`_scan_c`) is `_scan`'s statements in another order, with two changes of schedule:

  * programs value-block fastest: the value blocks of one head run side by side and read the chunk's
    k_cumdecay, q and k from memory once between them;
  * k_cumdecay and u_beta read after the q products, so q is dead before they load.

The chunk is a blocking choice of the same rule, not a different rule: per 1,024-row call the GDN
recurrence goes from 2.3 ms to 1.0 ms on the board, the error against an fp64 token-serial reference
stays where it is (output: the bf16 rounding; state 1.3e-5 relative), and the bits move (an order
move: tools/gdn_tile_gate.py --flag GDN_CHUNKED decides).

fp32 state; the gates through `tl.exp` as the shipped pair; the products at GDN_PREFILL_PREC /
_PREC_A as the shipped pair; the output in the input dtype; the final state [1, H, Dk, Dv] fp32.
"""

from __future__ import annotations

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402

import torch  # noqa: E402

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

#: the chunk, the walk's value block and warps, the UT transform's warps (measured on the 27B's shapes)
CHUNK = int(_S.get("GDN_CHUNKED_C"))
BV = int(_S.get("GDN_CHUNKED_BV"))
WARPS = int(_S.get("GDN_CHUNKED_WARPS"))
WARPS_INTRA = int(_S.get("GDN_CHUNKED_WARPS_INTRA"))


if HAVE_TRITON:
    from tools.gdn_prefill_kernels import _mm

    @triton.jit
    def _intra_c(K, V, G, BETA, UB, KCD,
                 T, s_kt, s_kh, s_vt, s_vh, s_gt, s_gh,
                 C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, PREC: tl.constexpr,
                 REP: tl.constexpr = 1):
        """`tools.gdn_prefill_kernels._intra`'s statements (forward substitution) for one (head, chunk);
        k is read again after the inverse instead of living through it, v read after it: the same
        values in fewer registers (217, no spills at 32 rows and 4 warps; 255 and 52 spills otherwise)."""
        h = tl.program_id(0)
        c = tl.program_id(1)
        i = tl.arange(0, C)
        rows = c * C + i
        m = rows < T
        ok = tl.arange(0, DK)
        ov = tl.arange(0, DV)
        kp = K + rows[:, None] * s_kt + (h // REP) * s_kh + ok[None, :]
        k = tl.load(kp, mask=m[:, None], other=0.0).to(tl.float32)
        g = tl.load(G + rows * s_gt + h * s_gh, mask=m, other=0.0).to(tl.float32)
        beta = tl.load(BETA + rows * s_gt + h * s_gh, mask=m, other=0.0).to(tl.float32)
        gc = tl.cumsum(g, axis=0)
        kb = k * beta[:, None]
        lower = i[:, None] >= i[None, :]
        decay = tl.where(lower, tl.exp(tl.where(lower, gc[:, None] - gc[None, :], 0.0)), 0.0)
        A = -_mm(kb, tl.trans(k), PREC) * decay
        A = tl.where(i[:, None] > i[None, :], A, 0.0)
        for r in tl.static_range(1, C):
            here = i == r
            row = tl.sum(tl.where(here[:, None], A, 0.0), axis=0)
            row = row + tl.sum(row[:, None] * A, axis=0) * (i < r)
            A = tl.where(here[:, None], row[None, :], A)
        inv = A + tl.where(i[:, None] == i[None, :], 1.0, 0.0)
        k = tl.load(kp, mask=m[:, None], other=0.0, volatile=True).to(tl.float32)
        kb = k * beta[:, None]
        v = tl.load(V + rows[:, None] * s_vt + h * s_vh + ov[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        vb = v * beta[:, None]
        ub = _mm(inv, vb, PREC)
        kcd = _mm(inv, kb * tl.exp(gc)[:, None], PREC)
        tl.store(UB + h * T * DV + rows[:, None] * DV + ov[None, :], ub, mask=m[:, None])
        tl.store(KCD + h * T * DK + rows[:, None] * DK + ok[None, :], kcd, mask=m[:, None])

    @triton.jit
    def _scan_c(Q, K, G, UB, KCD, OUT, S,
                T, NC, s_kt, s_kh, s_gt, s_gh, s_ot, s_oh,
                C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                SCALE: tl.constexpr, PREC: tl.constexpr, PREC_A: tl.constexpr, REP: tl.constexpr = 1):
        """One (value block, head), value block fastest: the chunks in order, the state tile in
        registers; `tools.gdn_prefill_kernels._scan`'s products, u_beta and k_cumdecay read late."""
        vbk = tl.program_id(0)
        h = tl.program_id(1)
        hq = h // REP
        i = tl.arange(0, C)
        ok = tl.arange(0, DK)
        ov = vbk * BV + tl.arange(0, BV)
        sp = S + h * DK * DV + ok[:, None] * DV + ov[None, :]
        s = tl.load(sp)
        for c in range(NC):
            rows = c * C + i
            m = rows < T
            q = tl.load(Q + rows[:, None] * s_kt + hq * s_kh + ok[None, :],
                        mask=m[:, None], other=0.0).to(tl.float32)
            k = tl.load(K + rows[:, None] * s_kt + hq * s_kh + ok[None, :],
                        mask=m[:, None], other=0.0).to(tl.float32)
            g = tl.load(G + rows * s_gt + h * s_gh, mask=m, other=0.0).to(tl.float32)
            q = q * SCALE
            gc = tl.cumsum(g, axis=0)
            lower = i[:, None] >= i[None, :]
            decay = tl.where(lower, tl.exp(tl.where(lower, gc[:, None] - gc[None, :], 0.0)), 0.0)
            a = _mm(q, tl.trans(k), PREC_A) * decay
            inter = _mm(q * tl.exp(gc)[:, None], s, PREC)
            u = tl.load(UB + h * T * DV + rows[:, None] * DV + ov[None, :], mask=m[:, None], other=0.0)
            kcd = tl.load(KCD + h * T * DK + rows[:, None] * DK + ok[None, :], mask=m[:, None], other=0.0)
            v_new = u - _mm(kcd, s, PREC)
            o = inter + _mm(a, v_new, PREC_A)
            tl.store(OUT + rows[:, None] * s_ot + h * s_oh + ov[None, :],
                     o.to(OUT.dtype.element_ty), mask=m[:, None])
            g_last = tl.sum(tl.where(i == C - 1, gc, 0.0))
            s = s * tl.exp(g_last) + _mm(tl.trans(k * tl.exp(g_last - gc)[:, None]), v_new, PREC)
        tl.store(sp, s)


def chunked_prefill_normed(q, k, v, g, beta, state=None, chunk_size: int | None = None,
                           bv: int | None = None, warps: int | None = None,
                           warps_intra: int | None = None):
    """`tools.gdn_prep_kernels.chunk_prefill_normed`'s contract (q and k l2-normalised at the key heads,
    [1, T, H, D] in; [1, T, H, Dv] and the final fp32 state [1, H, Dk, Dv] out) at this file's chunk."""
    from tools import gdn_prefill_kernels as G
    C = chunk_size or CHUNK
    bv = bv or BV
    warps = warps or WARPS
    wi = warps_intra or WARPS_INTRA
    B, T, Hk, Dk = q.shape
    H, Dv = v.shape[2], v.shape[-1]
    rep = H // Hk
    if B != 1 or C not in (16, 32, 64) or H % Hk or Dv % bv:
        raise ValueError("chunked_prefill_normed: one sequence, a chunk of 16 / 32 / 64, bv | Dv")
    qq, kk = q.reshape(T, Hk, Dk), k.reshape(T, Hk, Dk)
    vv = v.reshape(T, H, Dv).contiguous()
    gg = g.reshape(T, H).contiguous()
    bb = beta.reshape(T, H).contiguous()
    nc = (T + C - 1) // C
    ub = torch.empty(H, T, Dv, dtype=torch.float32, device=q.device)
    kcd = torch.empty(H, T, Dk, dtype=torch.float32, device=q.device)
    out = torch.empty(T, H, Dv, dtype=q.dtype, device=q.device)
    S = (torch.zeros(1, H, Dk, Dv, dtype=torch.float32, device=q.device)
         if state is None else state.clone().float().reshape(1, H, Dk, Dv))
    _intra_c[(H, nc)](kk, vv, gg, bb, ub, kcd, T, kk.stride(0), kk.stride(1), vv.stride(0), vv.stride(1),
                      gg.stride(0), gg.stride(1), C=C, DK=Dk, DV=Dv, PREC=G.PREC, REP=rep, num_warps=wi)
    _scan_c[(Dv // bv, H)](qq, kk, gg, ub, kcd, out, S, T, nc, qq.stride(0), qq.stride(1), gg.stride(0),
                           gg.stride(1), out.stride(0), out.stride(1), C=C, DK=Dk, DV=Dv, BV=bv,
                           SCALE=Dk ** -0.5, PREC=G.PREC_SCAN or G.PREC, PREC_A=G.PREC_A, REP=rep,
                           num_warps=warps, num_stages=G.STAGES_SCAN)
    return out.reshape(B, T, H, Dv), S.reshape(B, H, Dk, Dv)
