"""Flash-Next prefill kernels (ENG-225): a prompt chunk of many rows on the tensor cores.

The decode and verify kernels (tools/fn_kernels.py, tools/fn_seq_kernels.py) compute every row on its
own, so a verified row is the decode step's bit for bit. A prompt chunk has no such contract: it has
to agree with the reference forward on the next token (teacher-forced argmax), not bit for bit. So
the kernels here share work across rows:

  moe_grouped   the routed and shared experts of every (row, slot) pair of a chunk, grouped by expert:
                the pairs sorted by expert id (stable), each pair's input rotated with its expert's own
                suh (`pair_rot`), then one GEMM over the experts' row groups (`group_gemm`): a CTA owns
                one item -- up to 16 MT consecutive pairs of one expert -- one matrix and one 128-column
                block, decodes the expert's trellis tiles into the mma B registers once for all of its
                rows, sums its K range over 4 warps (added in warp order), and applies the output
                rotation and svh. gate|up land in sorted order (bf16), the down projection's input is
                bf16(silu(g)) * u rotated with the down suh (`pair_silu_rot`, moe_down_body's
                arithmetic), and the down products (fp32) land at each pair's original place, so the
                decode path's `combine_inject_norm` finishes the layer. Experts with no pair are not read.
                The item table is a function of the routing on the device (counts, starts, an inclusive
                scan of items; a CTA finds its item by binary search): no host sync.
  mm_rows       x @ W^T for int8-row or fp16 weights (the hyper-connection mixers, the router, PLE and
                GDN a/b): the weight dequantised to fp16 for the call, the library GEMM, fp32 out.
  mix_tail      the mixer's stream mean from the up projection's fp32 products (`mix_up_kernel`'s
                roundings: bf16(sigmoid(bf16(u))) * n per stream, summed in stream order, / hc).

The trellis decode and the mma helpers are tools/exl3_linear.py's (`_HELPERS`), which ports the bit
extraction and the codebook from exllamav3 (MIT, Copyright (c) 2025 Turboderp; the notice travels
with that source). Built on first use for sm_121a.
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

_HELPERS = X._CUDA[: X._CUDA.index("constexpr float RSQRT128")]

_CUDA = nb.source(__file__, "prefill.cu")

_CPP = nb.source(__file__, "prefill.cpp")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        src = "#define EXL3_ST 3\n" + _HELPERS + _CUDA
        tag = nb.tag(_CPP, src)
        _MOD = nb.load(f"fn_prefill_kernels_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["pre_pair_rot", "pre_pair_silu_rot", "pre_group_gemm", "pre_mix_tail"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
    return _MOD


# rows an item of the group GEMM takes (16 MT; 32 or 64): 32 measured faster at 2k and 4k rows a chunk
# (2026-10-11, tools/fn_moe_prefill_bench.py: 9.7 vs 10.7 ms, 15.8 vs 17.2 ms a layer)
GROUP_ROWS = 32


def group_plan(ids_flat: torch.Tensor, groups: int, rows: int = GROUP_ROWS) -> dict:
    """The pairs of a chunk grouped by expert, on the device, no host sync:

      order   int64 [P]      the pairs in ascending expert id (stable)
      eid     int32 [P]      the sorted pairs' experts
      cnt     int32 [G]      pairs an expert
      pstart  int32 [G]      an expert's first row in the padded layout (its rows padded to 16)
      prow    int32 [P]      each sorted pair's row in the padded layout
      iend    int32 [G]      the items of experts 0..g (an item = up to `rows` pairs)
      R       int            padded rows (an upper bound: P + 15 G, a multiple of 16)
      items   int            the item count's upper bound
    """
    P = ids_flat.numel()
    order = torch.sort(ids_flat, stable=True).indices
    eid = ids_flat.index_select(0, order).to(torch.int32)
    cnt = torch.bincount(ids_flat.long(), minlength=groups)
    first = torch.cumsum(cnt, 0) - cnt
    pad = (cnt + 15) // 16 * 16
    pstart = torch.cumsum(pad, 0) - pad
    e = eid.long()
    prow = (pstart.index_select(0, e) + torch.arange(P, device=ids_flat.device) - first.index_select(0, e)).to(torch.int32)
    iend = torch.cumsum((cnt + rows - 1) // rows, 0).to(torch.int32)
    return {"order": order, "eid": eid, "cnt": cnt.to(torch.int32), "pstart": pstart.to(torch.int32), "prow": prow,
            "iend": iend, "R": (P + 15 * groups + 15) // 16 * 16, "items": P // rows + groups}


def moe_grouped(x: torch.Tensor, ids: torch.Tensor, bank, rows: int = GROUP_ROWS) -> torch.Tensor:
    """Every (row, slot) pair's expert output, unweighted: x [M, H] bf16, ids int32 [M, S] (the router's,
    the shared expert as id E) -> yd [M S, H] fp32 in the pairs' own order (`combine_inject_norm`'s
    input, as `fn_kernels.moe_experts` returns it)."""
    M, S = ids.shape
    P = M * S
    mod = _module()
    g = group_plan(ids.reshape(-1), bank.n, rows)
    R, prow = g["R"], g["prow"]
    row = torch.div(g["order"], S, rounding_mode="floor").to(torch.int32)
    kbmax = bank.ks[-1]
    xh = torch.empty(2, R, bank.H, dtype=torch.float16, device=x.device)
    mod.pre_pair_rot(x, row, g["eid"], prow, bank.tr, bank.kb, bank.su, bank.sv, 0, xh)
    ygu = torch.empty(R, 2 * bank.I, dtype=torch.bfloat16, device=x.device)
    mod.pre_group_gemm(xh, g["cnt"], g["pstart"], g["iend"], bank.tr, bank.kb, bank.su, bank.sv, 0, bank.I, ygu, None,
                       g["items"], rows // 16, kbmax)
    del xh
    xd = torch.empty(1, R, bank.I, dtype=torch.float16, device=x.device)
    mod.pre_pair_silu_rot(ygu, g["eid"], prow, bank.tr, bank.kb, bank.su, bank.sv, xd)
    del ygu
    dst = torch.empty(R, dtype=torch.int32, device=x.device)
    dst.index_copy_(0, prow.long(), g["order"].to(torch.int32))
    yd = torch.empty(P, bank.H, dtype=torch.float32, device=x.device)
    mod.pre_group_gemm(xd, g["cnt"], g["pstart"], g["iend"], bank.tr, bank.kb, bank.su, bank.sv, 2, bank.H, yd, dst,
                       g["items"], rows // 16, kbmax)
    return yd


# the library GEMM of `mm_rows`: "fp32" = the exact weights (int8 times its scale in fp32) and fp32 sums
# (the decode path's GEMV arithmetic up to summation order), "fp16" = weights rounded to fp16 on the
# tensor cores (QWEN38_FN_PREFILL_MM). 2026-10-11: fp16 failed the agreement gate at 3k code (kernels vs
# decode path 95.6 % against the decode path's 97.2 % with the reference), fp32 passed every context
# and class at 1k/3k/8k, at 1,243 vs 1,441 tok/s (8k); two fp16 terms failed too and were slower.
# "tc" (SPD-227): the int8 weights on the tensor cores with nothing rounded -- x bf16 and q int8 are exact
# in bf16, their products exact, summed in fp32 over each scale group, the group's sum times its scale,
# the groups added in order (`mm_i8_tc`); fp16 weights as "fp32".
MM = _S.get("FN_PREFILL_MM")

_TC = None


def _tc_kernel():
    global _TC
    if _TC is None:
        import triton
        import triton.language as tl

        # one compile a weight shape: a chunk's row count (the last chunk of a prompt any number) is not specialised
        @triton.jit(do_not_specialize=["M", "N", "K", "NG", "sxm", "sqn", "ssn", "sym"])
        def mm_i8_kernel(X, Q, S, Y, M, N, K, NG, sxm, sqn, ssn, sym,
                         G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            pm, pn = tl.program_id(0), tl.program_id(1)
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            mm, mn = rm < M, rn < N
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for g in range(0, NG):
                part = tl.zeros((BM, BN), dtype=tl.float32)
                for k0 in range(0, G, BK):
                    k = g * G + k0 + rk
                    x = tl.load(X + rm[:, None] * sxm + k[None, :], mask=mm[:, None], other=0.0)
                    q = tl.load(Q + rn[None, :] * sqn + k[:, None], mask=mn[None, :], other=0)
                    part = tl.dot(x, q.to(tl.bfloat16), part)
                sc = tl.load(S + rn * ssn + g, mask=mn, other=0.0)
                acc += part * sc[None, :]
            tl.store(Y + rm[:, None] * sym + rn[None, :], acc, mask=mm[:, None] & mn[None, :])

        _TC = mm_i8_kernel
    return _TC


def mm_i8_tc(x: torch.Tensor, w) -> torch.Tensor:
    """x [M, K] bf16 @ (q s)^T for `pack.Int8Rows` w -> fp32 [M, N], nothing rounded but the fp32 sums: each
    scale group's products (exact in the tensor cores' bf16 inputs) summed in fp32, times the group's scale,
    the groups added in order (one group of K for a scale a row)."""
    import triton
    q, s = w.q, w.s
    M, K = x.shape
    N = q.shape[0]
    ng = 1 if s.dim() == 1 else s.shape[1]
    G = K // ng
    BK = 64 if G % 64 == 0 else 32
    if G % BK or x.stride(1) != 1:
        raise ValueError(f"mm_i8_tc: K {K} in {ng} groups of {G}")
    BM, BN = (128, 64) if N >= 2048 else (64, 64)
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    sv = s if s.dim() == 2 else s.view(N, 1)
    _tc_kernel()[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, q, sv, y, M, N, K, ng, x.stride(0), q.stride(0), sv.stride(0), y.stride(0),
        G=G, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return y


def mm_rows(x: torch.Tensor, w) -> torch.Tensor:
    """x [M, K] bf16 @ W^T -> fp32 [M, N] on the library GEMM: W int8 with fp32 scales (a row's or a
    group's, `pack.Int8Rows`) or fp16 [N, K]. `MM` fp32: the weights exact in fp32, fp32 sums; fp16: int8 rows
    dequantised to fp16 (q s in fp32, rounded once), x rounded to fp16, the tensor cores; tc: int8 weights
    through `mm_i8_tc`, fp16 ones as fp32."""
    if MM == "tc" and hasattr(w, "q") and x.dtype == torch.bfloat16:
        return mm_i8_tc(x.contiguous(), w)
    if MM in ("fp32", "tc"):
        wf = w.dequant() if hasattr(w, "q") else w.float()
        return x.float() @ wf.T
    if hasattr(w, "q"):
        N = w.q.shape[0]
        sc = w.s.view(N, -1, 1)                          # a scale a row, or a group of the row's weights
        wf = torch.empty(w.q.shape, dtype=torch.float16, device=w.q.device)
        torch.mul(w.q.view(N, sc.shape[1], -1), sc, out=wf.view(N, sc.shape[1], -1))
    else:
        wf = w if w.dtype == torch.float16 else w.half()
    return X._mm_fp32(x.half(), wf.T)


def qsa_scores_rows(iq: torch.Tensor, pool: torch.Tensor, nblk: torch.Tensor, NB: int, block: int = 1024) -> torch.Tensor:
    """The indexer's block scores of a chunk's rows on the library GEMM, the reference's arithmetic
    (`ref.qsa_mask`: fp32 products, relu, summed over the heads in order, / sqrt(head dim)): iq [M, NH, D]
    bf16, pool [blocks, D] bf16 (every complete block pooled; may hold fewer than NB rows), nblk int [M]
    (a row's complete blocks)
    -> fp32 [M, NB], -inf past a row's complete blocks. In blocks of `block` rows (the products of one
    block are [block NH, NB] fp32)."""
    M, NH, D = iq.shape
    nb = min(NB, pool.shape[0])              # the score width may pass the pool (a cache under NB blocks)
    pk = pool[:nb].float().T
    out = torch.full((M, NB), float("-inf"), dtype=torch.float32, device=iq.device)
    cols = torch.arange(nb, device=iq.device)
    for r0 in range(0, M, block):
        r1 = min(M, r0 + block)
        sc = torch.relu(iq[r0:r1].reshape(-1, D).float() @ pk).view(r1 - r0, NH, nb)
        acc = sc[:, 0]
        for h in range(1, NH):
            acc = acc + sc[:, h]
        out[r0:r1, :nb] = (acc / D ** 0.5).masked_fill(cols[None, :] >= nblk[r0:r1, None], float("-inf"))
    return out


def mix_tail(u: torch.Tensor, n: torch.Tensor, hc: int) -> torch.Tensor:
    """The mixer's stream mean from the up projection's products u [M, hc H] fp32 and the normed
    streams n [M, hc H] bf16 -> x [M, H] bf16."""
    x = torch.empty(n.shape[0], n.shape[1] // hc, dtype=torch.bfloat16, device=n.device)
    _module().pre_mix_tail(u, n, x, hc)
    return x


def moe_grouped_reference(x: torch.Tensor, ids: torch.Tensor, bank) -> torch.Tensor:
    """`moe_grouped` in torch, fp32 products (tests): each pair through its expert's dequantised
    matrices with the kernels' roundings (gate/up to bf16, the activation bf16(bf16(silu(g)) u))."""
    M, S = ids.shape
    out = torch.zeros(M * S, bank.H, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1).tolist()
    for p, e in enumerate(flat):
        xr = x[p // S:p // S + 1]
        g = X.reference_matmul(xr, bank.block(e, 0)).float()
        u = X.reference_matmul(xr, bank.block(e, 1)).float()
        a = (g / (1.0 + torch.exp(-g))).to(torch.bfloat16).float() * u
        out[p] = X.reference_matmul(a.to(torch.bfloat16), _fp32(bank.block(e, 2)))[0]
    return out


def _fp32(b):
    b.out_fp32 = True
    return b
