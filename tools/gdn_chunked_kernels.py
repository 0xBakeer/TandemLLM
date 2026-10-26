"""SPD-245: the GDN prefill recurrence of the GDN_PREP path, rescheduled (QWEN38_GDN_CHUNKED).

`tools/gdn_prep_kernels.py chunk_prefill_normed` runs the chunked gated delta rule of
`tools/gdn_prefill_kernels.py`: `_intra` (the UT transform of each 64-row chunk, in parallel) and `_scan`
(the walk over the chunks with the fp32 state tile in registers). Both run at 255 registers and spill
(`_intra` 162, `_scan` 142 at 16 x 4): one `_intra` program a SM, two `_scan` programs, so the 384 walk
programs of a 1,024-row call run in four waves and the call costs ten times the latency of its 16 chunk
steps. This file runs the same rule with less live state:

  * `_intra_c`: `_intra`'s statements, k read again after the inverse instead of living through it and
    the forward substitution as a loop instead of 63 unrolled steps: at 64 rows the same bits
    (tests/gpu/test_gdn_chunked_gpu.py), -15 % a call;
  * `_scan_c`: `_scan`'s statements with its programs value-block fastest (the value blocks of one head
    run side by side and read the chunk's k_cumdecay, q and k from memory once between them) and
    u_beta / k_cumdecay read after the q products (q is dead before they load): the state keeps its bits,
    about one output in 10^5 rounds one bf16 ulp the other way (an order move, KL-gated with
    tools/gdn_tile_gate.py --flag GDN_CHUNKED: top-1 64-65/65 on every row), -50 % a call.

Per 1,024-row call 2.2-2.5 ms -> 1.4-1.5 ms on the board. QWEN38_GDN_CHUNKED_C=32 runs the rule at 32-row
chunks (no spills in either kernel, 0.95-1.15 ms a call, the error against an fp64 token-serial
reference unchanged): every output's rounding moves, and the KL gate did not pass it (63/65 top-1 on
prose at 32K), so it is not the default.

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
                 REP: tl.constexpr = 1, DYN: tl.constexpr = False):
        """`tools.gdn_prefill_kernels._intra`'s statements (forward substitution) for one (head, chunk);
        k is read again after the inverse instead of living through it, v read after it, and with DYN the
        substitution is a loop: the same values in fewer registers."""
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
        if DYN:
            # the same substitution as a loop, not unrolled: the same statements a row
            for r in range(1, C):
                here = i == r
                row = tl.sum(tl.where(here[:, None], A, 0.0), axis=0)
                row = row + tl.sum(row[:, None] * A, axis=0) * (i < r)
                A = tl.where(here[:, None], row[None, :], A)
        else:
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
                           warps_intra: int | None = None, intra: str = "c", dyn: bool | None = None):
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
    if intra == "ship" and C == G.CHUNK:
        G._intra[(H, nc)](kk, vv, gg, bb, ub, kcd, T, kk.stride(0), kk.stride(1), vv.stride(0), vv.stride(1),
                          gg.stride(0), gg.stride(1), C=C, DK=Dk, DV=Dv, STEPS=G.STEPS, PREC=G.PREC,
                          SERIES=G.SERIES, REP=rep, num_warps=G.WARPS_INTRA)
    else:
        _intra_c[(H, nc)](kk, vv, gg, bb, ub, kcd, T, kk.stride(0), kk.stride(1), vv.stride(0),
                          vv.stride(1), gg.stride(0), gg.stride(1), C=C, DK=Dk, DV=Dv, PREC=G.PREC, REP=rep,
                          DYN=C >= 64 if dyn is None else dyn, num_warps=wi)
    _scan_c[(Dv // bv, H)](qq, kk, gg, ub, kcd, out, S, T, nc, qq.stride(0), qq.stride(1), gg.stride(0),
                           gg.stride(1), out.stride(0), out.stride(1), C=C, DK=Dk, DV=Dv, BV=bv,
                           SCALE=Dk ** -0.5, PREC=G.PREC_SCAN or G.PREC, PREC_A=G.PREC_A, REP=rep,
                           num_warps=warps, num_stages=G.STAGES_SCAN)
    return out.reshape(B, T, H, Dv), S.reshape(B, H, Dk, Dv)
