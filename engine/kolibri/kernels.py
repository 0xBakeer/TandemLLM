"""Kolibri-1's weight formats and the kernels that read them in place.

Formats (detected from the tensors, never from names):

  FP8 block   e4m3 codes [N, K] + one fp32 scale per 128x128 block [N/128, K/128] (Aleph Alpha's
              release). The scale stays fp32: the release's scales are `amax / 448` of the BF16
              block to the last bit.
              (`tools/fp8_linear.FP8Block` rounds its scale table to bf16; it is not used here.)
              The GEMM is `tools/fp8_linear._fp8_gemm_kernel` with SCALE_W=False: the codes go to
              bf16 exactly, products accumulate in fp32 and the fp32 block scale multiplies each
              128-wide K step's partial sum.
  NVFP4       uint8 codes [N, K/2] (low nibble = even K) + e4m3 scale [N, K/16] + fp32 scale_2 per
              tensor (the published set, `tools/kolibri_nvfp4.py`). The group scale is applied to the decoded
              weight (exact in bf16), scale_2 to the fp32 accumulator, per output row, so members of
              a fused group may carry different scale_2.
  e4m3 head   codes [V, K] + fp32 scale per row; `tools/head_gemv.FP8Head`, logits in fp32.

The routed experts are an `ExpertBank` per projection (codes [E, N, K/2], scale [E, N, K/16],
scale_2 [E]); the shared expert is appended as expert E, so one launch set computes the six
routed experts and the shared one: the router's ids get a seventh column E with weight 1.0, and
the combine sums the seven slots in slot order in fp32.

The MoE kernels are row-invariant (a row's result does not depend on the other rows of the
launch), with the combine writing fp32.

Every op has a plain-torch path for the CPU (tests on a tiny random Kolibri) that computes the
same arithmetic in fp32.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    HAVE_TRITON = True
except ImportError:                                              # pragma: no cover
    HAVE_TRITON = False

NV_GROUP = 16
FP8_BLOCK = 128
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
DOT32 = os.environ.get("TRITON_INTERPRET") == "1"


def fp4_unpack(codes: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> fp32 [..., K] e2m1 values, low nibble first."""
    grid = FP4_GRID.to(codes.device)
    lo = (codes & 0x0F).long()
    hi = (codes >> 4).long()
    vlo = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
    vhi = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
    return torch.stack([vlo, vhi], dim=-1).flatten(-2)


def nv_dequant(codes, scale, s2_rows) -> torch.Tensor:
    """fp32 [..., N, K]: e2m1 * group scale (exact, as the kernel's bf16 tile) * per-row scale_2."""
    v = fp4_unpack(codes) * scale.float().repeat_interleave(NV_GROUP, -1)
    return v.to(torch.bfloat16).float() * s2_rows.float()[..., None]


def block_m_for(M: int) -> int:
    return 16 if M <= 16 else (32 if M <= 32 else 64)


# ------------------------------------------------------------------------------ FP8 block, fp32 scales
class FP8Linear:
    """e4m3 codes [N, K] with fp32 128x128 block scales; `.matmul(x bf16 [M, K]) -> bf16 [M, N]`."""

    __slots__ = ("w", "s", "N", "K", "sizes")

    def __init__(self, codes: torch.Tensor, scale_inv: torch.Tensor, sizes=None):
        assert codes.dtype == torch.float8_e4m3fn and codes.dim() == 2, (codes.dtype, codes.shape)
        self.w = codes.contiguous()
        self.s = scale_inv.float().contiguous()      # fp32, never rounded
        self.N, self.K = codes.shape
        assert tuple(self.s.shape) == (-(-self.N // FP8_BLOCK), -(-self.K // FP8_BLOCK)), (self.s.shape, codes.shape)
        self.sizes = sizes or [self.N]

    @classmethod
    def cat(cls, parts: list["FP8Linear"]) -> "FP8Linear":
        """Members of the same K stacked along N (q|k|v): one launch, each member's columns equal
        to its own call (a program owns one N tile, every N is a multiple of 128)."""
        assert all(p.N % FP8_BLOCK == 0 and p.K == parts[0].K for p in parts)
        return cls(torch.cat([p.w for p in parts]), torch.cat([p.s for p in parts]), [p.N for p in parts])

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 4

    def dense(self) -> torch.Tensor:
        s = self.s.repeat_interleave(FP8_BLOCK, 0)[: self.N].repeat_interleave(FP8_BLOCK, 1)[:, : self.K]
        return self.w.float() * s

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """`out` (optional): a [M, N] bf16 view with unit stride along N to write into (a slice of
        a wider output, so a mixed-format qkv needs no concatenation launch)."""
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.K)
        if x2.device.type != "cuda" or not HAVE_TRITON:
            y = F.linear(x2.float(), self.dense()).to(torch.bfloat16)
            return y.view(*lead, self.N) if out is None else out.copy_(y)
        from tools.fp8_linear import _fp8_gemm_kernel
        assert self.N % FP8_BLOCK == 0 and self.K % FP8_BLOCK == 0
        x2 = x2.to(torch.bfloat16).contiguous()
        M = x2.shape[0]
        y = torch.empty(M, self.N, dtype=torch.bfloat16, device=x2.device) if out is None else out
        cfg = FP8_1ROW.get((self.N, self.K)) if M == 1 and ONEROW else None
        if cfg is not None:
            bn1, bk1, w1, st1 = cfg
            _fp8_gemv_1row[(self.N // bn1,)](x2, self.w, self.s, y, self.N, self.K,
                                             self.w.stride(0), self.s.stride(0), BN=bn1, BK=bk1,
                                             num_warps=w1, num_stages=st1)
            return y.view(*lead, self.N) if out is None else y
        bm = 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 128 else 128))
        # 64-wide N tiles when the 128-wide grid would leave the board's 48 SMs idle (o_proj: N=2560)
        bn = 64 if self.N // 128 < 96 else 128
        _fp8_gemm_kernel[(self.N // bn, triton.cdiv(M, bm))](
            x2, self.w, self.s, y, M, self.N, self.K,
            x2.stride(0), self.w.stride(0), self.s.stride(0), y.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=FP8_BLOCK, SCALE_W=False,
            num_warps=4, num_stages=2 if bm >= 128 else 3)
        return y.view(*lead, self.N) if out is None else y


# ------------------------------------------------------------------------------ NVFP4 dense
if HAVE_TRITON:

    @triton.jit
    def _dot(a, b, DOT32: tl.constexpr):
        if DOT32:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), out_dtype=tl.float32)
        return tl.dot(a, b, out_dtype=tl.float32)

    @triton.jit
    def _fp4_decode_tile(c, BN: tl.constexpr, BKH: tl.constexpr):
        """uint8 codes [BN, BKH] -> fp32 values [BN, 2*BKH], low nibble first (even K)."""
        lo = (c & 0x0F).to(tl.int32)
        hi = ((c >> 4) & 0x0F).to(tl.int32)
        vals = tl.join(lo, hi)
        n = tl.reshape(vals, (BN, 2 * BKH))
        s = (n >> 3) & 1
        e = (n >> 1) & 3
        m = (n & 1).to(tl.float32)
        mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * ((e + 126) << 23).to(tl.float32, bitcast=True))
        return tl.where(s == 1, -mag, mag)

    @triton.jit
    def _nv_tile(Wc, Ws, e, rn, nm, k0, K, swe, swn, sse, ssn, BN: tl.constexpr, BK: tl.constexpr):
        """Decoded bf16 weight tile [BN, BK] of expert e at K offset k0 (group scale applied)."""
        rkh = tl.arange(0, BK // 2)
        kmh = (k0 // 2 + rkh) < K // 2
        c = tl.load(Wc + e * swe + rn[:, None] * swn + (k0 // 2 + rkh)[None, :],
                    mask=nm[:, None] & kmh[None, :], other=0)
        rs = tl.arange(0, BK // 16)
        sc = tl.load(Ws + e * sse + rn[:, None] * ssn + (k0 // 16 + rs)[None, :],
                     mask=nm[:, None] & ((k0 + rs * 16) < K)[None, :], other=0.0).to(tl.float32)
        v = _fp4_decode_tile(c, BN, BK // 2)
        w = tl.reshape(v, (BN, BK // 16, 16)) * sc[:, :, None]
        return tl.reshape(w, (BN, BK)).to(tl.bfloat16)

    @triton.jit
    def _nv_dense_kernel(X, Wc, Ws, S2, Y, M, N, K, sxm, swn, ssn, sym,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT32: tl.constexpr):
        pn = tl.program_id(0)
        pm = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        mm = rm < M
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < N
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            x = tl.load(X + rm[:, None] * sxm + kk[None, :], mask=mm[:, None] & (kk < K)[None, :], other=0.0)
            w = _nv_tile(Wc, Ws, 0, rn, nm, k0, K, 0, swn, 0, ssn, BN, BK)
            acc += _dot(x, tl.trans(w), DOT32)
        s2 = tl.load(S2 + rn, mask=nm, other=0.0)
        tl.store(Y + rm[:, None] * sym + rn[None, :], (acc * s2[None, :]).to(tl.bfloat16),
                 mask=mm[:, None] & nm[None, :])


def nv_dense_tile(M: int, K: int) -> tuple[int, int, int, int]:
    """(BN, BK, warps, stages) for the NVFP4 dense kernel, picked on the GB10 at q (6144x2560) and
    o (2560x6144). Rows up to 32 share one shape (decode = verify rows)."""
    if M <= 32:
        return (64, 256, 8, 3) if K >= 4096 else (32, 256, 2, 2)
    return (128, 128, 4, 1) if K >= 4096 else (64, 128, 2, 1)


class NVFP4Linear:
    """NVFP4 codes [N, K/2] + e4m3 scale [N, K/16] + per-row fp32 scale_2 [N] (a fused group's
    members keep their own per-tensor scale_2). `.matmul(x bf16 [M, K]) -> bf16 [M, N]`."""

    __slots__ = ("w", "s", "s2", "N", "K", "sizes")

    def __init__(self, codes, scale, s2_rows, sizes=None):
        assert codes.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
        self.w, self.s = codes.contiguous(), scale.contiguous()
        self.N, self.K = codes.shape[0], codes.shape[1] * 2
        self.s2 = torch.as_tensor(s2_rows, dtype=torch.float32, device=codes.device).reshape(-1)
        if self.s2.numel() == 1:
            self.s2 = self.s2.expand(self.N).contiguous()
        assert self.s2.numel() == self.N
        self.sizes = sizes or [self.N]

    @classmethod
    def cat(cls, parts: list["NVFP4Linear"]) -> "NVFP4Linear":
        return cls(torch.cat([p.w for p in parts]), torch.cat([p.s for p in parts]),
                   torch.cat([p.s2 for p in parts]), [p.N for p in parts])

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() + self.s2.numel() * 4

    def dense(self) -> torch.Tensor:
        return nv_dequant(self.w, self.s, self.s2)

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """`out` as `FP8Linear.matmul`."""
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.K)
        if x2.device.type != "cuda" or not HAVE_TRITON:
            y = F.linear(x2.to(torch.bfloat16).float(), self.dense()).to(torch.bfloat16)
            return y.view(*lead, self.N) if out is None else out.copy_(y)
        x2 = x2.to(torch.bfloat16).contiguous()
        M = x2.shape[0]
        y = torch.empty(M, self.N, dtype=torch.bfloat16, device=x2.device) if out is None else out
        bm = block_m_for(M)
        bn, bk, warps, stages = nv_dense_tile(M, self.K)
        _nv_dense_kernel[(triton.cdiv(self.N, bn), triton.cdiv(M, bm))](
            x2, self.w, self.s, self.s2, y, M, self.N, self.K,
            x2.stride(0), self.w.stride(0), self.s.stride(0), y.stride(0),
            BM=bm, BN=bn, BK=bk, DOT32=DOT32, num_warps=warps, num_stages=stages)
        return y.view(*lead, self.N) if out is None else y


# ------------------------------------------------------------------------------ the experts
class ExpertBank:
    """One projection of a layer's experts: codes [E, N, K/2], scale [E, N, K/16], scale_2 [E]."""

    __slots__ = ("codes", "scale", "scale_2", "E", "N", "K")

    def __init__(self, codes, scale, scale_2):
        assert codes.dtype == torch.uint8 and codes.dim() == 3, (codes.dtype, codes.shape)
        assert scale.dtype == torch.float8_e4m3fn, scale.dtype
        self.codes, self.scale = codes.contiguous(), scale.contiguous()
        self.scale_2 = scale_2.float().reshape(-1).contiguous()
        self.E, self.N = codes.shape[:2]
        self.K = codes.shape[2] * 2
        assert tuple(self.scale.shape) == (self.E, self.N, self.K // NV_GROUP)
        assert self.scale_2.numel() == self.E

    @property
    def nbytes(self) -> int:
        return self.codes.numel() + self.scale.numel() + self.scale_2.numel() * 4

    def dense(self, e: int) -> torch.Tensor:
        return nv_dequant(self.codes[e], self.scale[e], self.scale_2[e].expand(self.N))


_ONE_ROW_PLAN: dict = {}

#: one-row (decode) kernels for the routed experts and the FP8 projections; 0: the tiled ones
ONEROW = os.environ.get("KOLIBRI_ONEROW", "1") != "0"
#: (BN, BK, warps, stages) of `_moe_gate_up_1row` and `_moe_down_1row`, picked on the GB10
MOE_1ROW = ((16, 512, 4, 3), (16, 128, 4, 3))
#: (BN, BK, warps, stages) of `_fp8_gemv_1row` per (N, K), picked on the GB10
FP8_1ROW = {(1024, 2560): (8, 512, 8, 3), (2560, 512): (4, 512, 2, 2),
            (7168, 2560): (4, 512, 8, 2), (2560, 6144): (32, 512, 8, 3)}


def moe_plan(topk_ids: torch.Tensor, n_experts: int, bm: int):
    """Device-side grouping of (row, slot) pairs by expert, no host sync.

    Returns (order, tile_expert, tile_start, tile_len): `order` lists pair indices (row * k + slot)
    sorted by expert and, within an expert, by pair index; tile t covers
    order[tile_start[t] : tile_start[t] + tile_len[t]], all routed to tile_expert[t]."""
    flat = topk_ids.reshape(-1).long()
    P = flat.numel()
    if topk_ids.shape[0] == 1:
        # one row: its experts are distinct, every pair is its own tile, in slot order (the tile
        # order does not matter: each tile writes only its own pair's rows). The constant tables
        # are made once (two launches fewer a layer in the decode graph).
        key = (P, flat.device)
        c = _ONE_ROW_PLAN.get(key)
        if c is None:
            c = _ONE_ROW_PLAN[key] = (torch.arange(P, dtype=torch.int32, device=flat.device),
                                      torch.ones(P, dtype=torch.int32, device=flat.device))
        return c[0], flat.to(torch.int32), c[0], c[1]
    srt = torch.sort(flat, stable=True)
    order = srt.indices
    counts = torch.zeros(n_experts, dtype=torch.long, device=flat.device).scatter_add_(
        0, flat, torch.ones_like(flat))
    nt = (counts + bm - 1) // bm
    cum = torch.cumsum(nt, 0)
    tile_off = cum - nt
    pair_off = torch.cumsum(counts, 0) - counts
    T = P // bm + min(n_experts, P) + 1
    t = torch.arange(T, device=flat.device)
    te = torch.searchsorted(cum, t, right=True)
    valid = te < n_experts
    tec = te.clamp(max=n_experts - 1)
    local = t - tile_off[tec]
    start = pair_off[tec] + local * bm
    ln = torch.clamp(counts[tec] - local * bm, min=0, max=bm)
    ln = torch.where(valid, ln, torch.zeros_like(ln))
    return order.to(torch.int32), tec.to(torch.int32), start.to(torch.int32), ln.to(torch.int32)


if HAVE_TRITON:

    @triton.jit
    def _moe_gate_up_kernel(X, GC, GS, G2, UC, US, U2, Hout, ORDER, TE, TS, TL,
                            K, I, KTOP, sxm, swe, swn, sse, ssn, shm,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT32: tl.constexpr):
        t = tl.program_id(0)
        pn = tl.program_id(1)
        ln = tl.load(TL + t)
        if ln == 0:
            return
        e = tl.load(TE + t)
        st = tl.load(TS + t)
        r = tl.arange(0, BM)
        rmask = r < ln
        pair = tl.load(ORDER + st + r, mask=rmask, other=0)
        row = pair // KTOP
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < I
        rk = tl.arange(0, BK)
        ag = tl.zeros((BM, BN), dtype=tl.float32)
        au = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            x = tl.load(X + row[:, None] * sxm + kk[None, :], mask=rmask[:, None] & (kk < K)[None, :], other=0.0)
            wg = _nv_tile(GC, GS, e, rn, nm, k0, K, swe, swn, sse, ssn, BN, BK)
            ag += _dot(x, tl.trans(wg), DOT32)
            wu = _nv_tile(UC, US, e, rn, nm, k0, K, swe, swn, sse, ssn, BN, BK)
            au += _dot(x, tl.trans(wu), DOT32)
        g = ag * tl.load(G2 + e)
        u = au * tl.load(U2 + e)
        h = g * tl.sigmoid(g) * u
        tl.store(Hout + pair[:, None] * shm + rn[None, :], h.to(tl.bfloat16), mask=rmask[:, None] & nm[None, :])

    @triton.jit
    def _moe_down_kernel(Hin, DC, DS, D2, RW, P, ORDER, TE, TS, TL,
                         I, H, shm, swe, swn, sse, ssn, spm,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT32: tl.constexpr):
        t = tl.program_id(0)
        pn = tl.program_id(1)
        ln = tl.load(TL + t)
        if ln == 0:
            return
        e = tl.load(TE + t)
        st = tl.load(TS + t)
        r = tl.arange(0, BM)
        rmask = r < ln
        pair = tl.load(ORDER + st + r, mask=rmask, other=0)
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < H
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, I, BK):
            kk = k0 + rk
            x = tl.load(Hin + pair[:, None] * shm + kk[None, :], mask=rmask[:, None] & (kk < I)[None, :], other=0.0)
            w = _nv_tile(DC, DS, e, rn, nm, k0, I, swe, swn, sse, ssn, BN, BK)
            acc += _dot(x, tl.trans(w), DOT32)
        rw = tl.load(RW + pair, mask=rmask, other=0.0).to(tl.float32)
        y = acc * tl.load(D2 + e) * rw[:, None]
        tl.store(P + pair[:, None] * spm + rn[None, :], y, mask=rmask[:, None] & nm[None, :])

    @triton.jit
    def _moe_gate_up_1row(X, IDS, GC, GS, G2, UC, US, U2, Hout, K, I, swe, swn, sse, ssn, shm,
                          BN: tl.constexpr, BK: tl.constexpr):
        """One decode row: program (slot j, BN columns of I), no tl.dot: the row's x times the
        decoded bf16 tile in fp32, the expert read straight from the router's ids."""
        j = tl.program_id(0)
        pn = tl.program_id(1)
        e = tl.load(IDS + j)
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
    def _moe_down_1row(Hin, IDS, DC, DS, D2, RW, P, I, H, shm, swe, swn, sse, ssn, spm,
                       BN: tl.constexpr, BK: tl.constexpr):
        j = tl.program_id(0)
        pn = tl.program_id(1)
        e = tl.load(IDS + j)
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < H
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, I, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(Hin + j * shm + kk, mask=kk < I, other=0.0).to(tl.float32)
            w = _nv_tile(DC, DS, e, rn, nm, k0, I, swe, swn, sse, ssn, BN, BK).to(tl.float32)
            acc += tl.sum(w * x[None, :], 1)
        rw = tl.load(RW + j).to(tl.float32)
        tl.store(P + j * spm + rn, acc * tl.load(D2 + e) * rw, mask=nm)

    @triton.jit
    def _fp8_gemv_1row(X, W, S, Y, N, K, swn, ssn, BN: tl.constexpr, BK: tl.constexpr):
        """One row of an FP8 block-scaled projection: BN output rows a program over the
        whole K, 128-wide scale blocks applied to the fp32 partial sums."""
        pn = tl.program_id(0)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), dtype=tl.float32)
        s_row = (pn * BN) // 128
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(X + kk).to(tl.float32)
            w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
            part = tl.reshape(w * x[None, :], (BN, BK // 128, 128))
            sc = tl.load(S + s_row * ssn + k0 // 128 + tl.arange(0, BK // 128))
            acc += tl.sum(tl.sum(part, 2) * sc[None, :], 1)
        tl.store(Y + rn, acc.to(tl.bfloat16))

    @triton.jit
    def _moe_combine_kernel(P, Y, H, KTOP, spm, sym, EX, sex, BN: tl.constexpr,
                            EXTRA: tl.constexpr):
        row = tl.program_id(0)
        pn = tl.program_id(1)
        rn = pn * BN + tl.arange(0, BN)
        nm = rn < H
        acc = tl.zeros((BN,), dtype=tl.float32)
        for j in range(0, KTOP):
            acc += tl.load(P + (row * KTOP + j) * spm + rn, mask=nm, other=0.0)
        if EXTRA:
            # the shared expert's bf16 output, added after the routed sum (`y + shared.float()`)
            acc = acc + tl.load(EX + row * sex + rn, mask=nm, other=0.0).to(tl.float32)
        tl.store(Y + row * sym + rn, acc, mask=nm)

    @triton.jit
    def _swiglu_kernel(GU, A, F, sg, sa, BF: tl.constexpr):
        """bf16 SiLU(g) * u from the fused gate|up output, with the arithmetic of torch's
        `silu(gu.float()[:, :F]) * gu.float()[:, F:]` (IEEE exp and division), rounded to bf16."""
        row = tl.program_id(0)
        pc = tl.program_id(1)
        c = pc * BF + tl.arange(0, BF)
        m = c < F
        g = tl.load(GU + row * sg + c, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GU + row * sg + F + c, mask=m, other=0.0).to(tl.float32)
        si = tl.math.div_rn(g, 1.0 + libdevice.exp(-g))
        tl.store(A + row * sa + c, (si * u).to(tl.bfloat16), mask=m)


def swiglu(gu: torch.Tensor, F_: int) -> torch.Tensor:
    """gu bf16 [M, 2F] -> bf16 [M, F]: one launch for `.float()`, silu, the product and the cast."""
    M = gu.shape[0]
    a = torch.empty(M, F_, dtype=torch.bfloat16, device=gu.device)
    _swiglu_kernel[(M, triton.cdiv(F_, 512))](gu, a, F_, gu.stride(0), a.stride(0), BF=512, num_warps=4)
    return a


# Launch shapes picked on the GB10 (sm_121) at the served shapes (385 experts, 2560/512):
# (BN, BK, warps, stages) for gate/up and for down. A prefill chunk takes the large shapes (with
# (64, 128, 4, 2) the two 64x64 accumulators spill registers). Rows up to SMALL_M (decode, verify)
# take the small shapes, so every such row runs the same program shape whatever the row count.
SMALL_M = 32
MOE_TILES = {"small": ((32, 128, 4, 3), (32, 32, 4, 1)), "large": ((128, 128, 8, 3), (128, 128, 4, 1))}


def moe_experts(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, G: ExpertBank, U: ExpertBank,
                D: ExpertBank, extra: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 [M, H] = sum over slots j of w[:, j] * D_e(SiLU(G_e x) * U_e x), e = ids[:, j], summed in
    slot order. x bf16 [M, H]; ids int [M, k]; w fp32 [M, k]."""
    M, H = x.shape
    k = ids.shape[1]
    if x.device.type != "cuda" or not HAVE_TRITON:
        y = moe_experts_torch(x, ids, w, G, U, D)
        return y if extra is None else y + extra.float()
    I = G.N
    if M == 1 and ONEROW and HAVE_TRITON:
        return _moe_experts_1row(x, ids, w, G, U, D, extra)
    bm = block_m_for(M)
    (bn, bk, warps, stages), (bnd, bkd, warpsd, stagesd) = MOE_TILES["small" if M <= SMALL_M else "large"]
    order, te, ts, tln = moe_plan(ids, G.E, bm)
    T = te.numel()
    x = x.to(torch.bfloat16).contiguous()
    h = torch.empty(M * k, I, dtype=torch.bfloat16, device=x.device)
    _moe_gate_up_kernel[(T, triton.cdiv(I, bn))](
        x, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, order, te, ts, tln,
        H, I, k, x.stride(0), G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1),
        h.stride(0), BM=bm, BN=bn, BK=bk, DOT32=DOT32, num_warps=warps, num_stages=stages)
    p = torch.empty(M * k, H, dtype=torch.float32, device=x.device)
    rw = w.float().reshape(-1).contiguous()
    _moe_down_kernel[(T, triton.cdiv(H, bnd))](
        h, D.codes, D.scale, D.scale_2, rw, p, order, te, ts, tln,
        I, H, h.stride(0), D.codes.stride(0), D.codes.stride(1), D.scale.stride(0), D.scale.stride(1),
        p.stride(0), BM=bm, BN=bnd, BK=bkd, DOT32=DOT32, num_warps=warpsd, num_stages=stagesd)
    y = torch.empty(M, H, dtype=torch.float32, device=x.device)
    if extra is None:
        _moe_combine_kernel[(M, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), p, 0,
                                                      BN=256, EXTRA=False, num_warps=4)
    else:
        _moe_combine_kernel[(M, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), extra,
                                                      extra.stride(0), BN=256, EXTRA=True,
                                                      num_warps=4)
    return y


def _moe_experts_1row(x, ids, w, G, U, D, extra):
    """`moe_experts` for one row: the experts straight from the router's ids (no plan), one-row
    kernels, the same combine."""
    H = x.shape[1]
    k = ids.shape[1]
    I = G.N
    x = x.to(torch.bfloat16).contiguous()
    ids = ids.reshape(-1).contiguous()
    rw = w.float().reshape(-1).contiguous()
    (bn, bk, warps, stages), (bnd, bkd, warpsd, stagesd) = MOE_1ROW
    h = torch.empty(k, I, dtype=torch.bfloat16, device=x.device)
    _moe_gate_up_1row[(k, triton.cdiv(I, bn))](
        x, ids, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
        G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
        BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    p = torch.empty(k, H, dtype=torch.float32, device=x.device)
    _moe_down_1row[(k, triton.cdiv(H, bnd))](
        h, ids, D.codes, D.scale, D.scale_2, rw, p, I, H, h.stride(0), D.codes.stride(0),
        D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
        BN=bnd, BK=bkd, num_warps=warpsd, num_stages=stagesd)
    y = torch.empty(1, H, dtype=torch.float32, device=x.device)
    ex = extra if extra is not None else p
    _moe_combine_kernel[(1, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), ex,
                                                  ex.stride(0) if extra is not None else 0,
                                                  BN=256, EXTRA=extra is not None, num_warps=4)
    return y


def moe_experts_torch(x, ids, w, G, U, D) -> torch.Tensor:
    """The kernels' arithmetic in torch (fp32 products of bf16-exact operands, scale_2 on the
    accumulator, SiLU(g)*u rounded to bf16 as the kernel stores it, slot-order sum)."""
    M, H = x.shape
    k = ids.shape[1]
    xb = x.to(torch.bfloat16).float()
    p = torch.zeros(M, k, H, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        xe = xb[rows]
        g = xe @ (fp4_unpack(G.codes[e]) * G.scale[e].float().repeat_interleave(NV_GROUP, -1)).to(torch.bfloat16).float().T * G.scale_2[e]
        u = xe @ (fp4_unpack(U.codes[e]) * U.scale[e].float().repeat_interleave(NV_GROUP, -1)).to(torch.bfloat16).float().T * U.scale_2[e]
        a = (g * torch.sigmoid(g) * u).to(torch.bfloat16).float()
        d = a @ (fp4_unpack(D.codes[e]) * D.scale[e].float().repeat_interleave(NV_GROUP, -1)).to(torch.bfloat16).float().T
        p[rows, slot] = d * D.scale_2[e] * w[rows, slot, None].float()
    return p.sum(1)


# ------------------------------------------------------------------------------ the router
def route(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, k: int, shared_id: int | None,
          renorm: bool = False):
    """The plugin's `sigmoid_logit_add_routing`: fp32 logits; the top k of logits + expert_bias;
    each chosen expert weighted by sigmoid(logit) without the bias, not renormalised.

    `gate` is the router [E, H] (BF16 as released, or fp32), x bf16 [M, H]. With
    `shared_id` a (k+1)-th column routes every row to the shared expert at weight 1.0.
    Returns (ids int64 [M, k(+1)], weights fp32 [M, k(+1)])."""
    logits = x.float() @ gate.float().T
    ids = torch.topk(logits + bias, k=k, dim=-1)[1]
    w = torch.sigmoid(logits.gather(1, ids))
    if renorm:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    if shared_id is not None:
        M = x.shape[0]
        ids = torch.cat([ids, torch.full((M, 1), shared_id, dtype=ids.dtype, device=ids.device)], 1)
        w = torch.cat([w, torch.ones(M, 1, dtype=w.dtype, device=w.device)], 1)
    return ids, w


# ------------------------------------------------------------------------------ norms
def rms(x: torch.Tensor, w: torch.Tensor, eps: float, out_dtype=torch.bfloat16) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * w in fp32, rounded once to `out_dtype` (the reference's RMSNorm)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(out_dtype)


# ------------------------------------------------------------------------------ head
class E4M3Head:
    """`lm_head` as e4m3 codes [V, K] + fp32 scale per row; fp32 logits."""

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor):
        self.w = codes.contiguous()
        self.s = scale.float().contiguous()
        self.N, self.K = codes.shape

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 4

    def logits(self, x: torch.Tensor, rows: int = 16384) -> torch.Tensor:
        """x [M, K] (bf16 or fp32) -> fp32 [M, V]. One row on the GPU takes the fp32-activation
        GEMV of `tools/head_gemv.py`; more rows a chunked fp32 product (prefill's last row is the
        only one serving needs; the gate reads all rows)."""
        x2 = x.reshape(-1, self.K)
        M = x2.shape[0]
        if x2.device.type == "cuda" and HAVE_TRITON and M == 1 and self.K % 256 == 0:   # the GEMV has no K mask
            from tools.head_gemv import _head_gemv_fp8
            xf = x2.float().contiguous()
            y = torch.empty(1, self.N, dtype=torch.float32, device=x2.device)
            _head_gemv_fp8[(triton.cdiv(self.N, 32),)](xf, self.w, self.s, y, 1, self.N, self.K,
                                                       self.w.stride(0), BM=1, BN=32, BK=256, num_warps=4)
            return y
        xf = x2.float()
        out = torch.empty(M, self.N, dtype=torch.float32, device=x2.device)
        for v0 in range(0, self.N, rows):
            Wv = self.w[v0:v0 + rows].float() * self.s[v0:v0 + rows, None]
            out[:, v0:v0 + rows] = xf @ Wv.T
        return out


# ------------------------------------------------------------------------------ fused glue (decode)
# Without them a decode step is hundreds of small launches (norms, casts, router glue). These
# kernels replace the norm/residual runs and the router's top-k glue with one
# program per row each, in the same arithmetic as the torch paths above (fp32 throughout, one
# rounding where the torch path rounds). `KOLIBRI_FUSED=0` takes the torch paths.
FUSED = os.environ.get("KOLIBRI_FUSED", "1") == "1"

if HAVE_TRITON:

    @triton.jit
    def _add_rms2_kernel(R, Y, W1, W2, RO, XO, H, eps, sr, sy, sro, sxo, BH: tl.constexpr):
        """r' = r + rms(y) * w1 (fp32); x = rms(r') * w2 (stored in XO's dtype)."""
        row = tl.program_id(0)
        c = tl.arange(0, BH)
        m = c < H
        y = tl.load(Y + row * sy + c, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R + row * sr + c, mask=m, other=0.0).to(tl.float32)
        w1 = tl.load(W1 + c, mask=m, other=0.0).to(tl.float32)
        w2 = tl.load(W2 + c, mask=m, other=0.0).to(tl.float32)
        ry = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
        r2 = r + (y * ry) * w1
        tl.store(RO + row * sro + c, r2, mask=m)
        rr = tl.rsqrt(tl.sum(r2 * r2, axis=0) / H + eps)
        tl.store(XO + row * sxo + c, ((r2 * rr) * w2).to(XO.dtype.element_ty), mask=m)

    @triton.jit
    def _rms_kernel(X, W, Y, H, eps, sx, sy, BH: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, BH)
        m = c < H
        x = tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
        tl.store(Y + row * sy + c, ((x * r) * w).to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _route_kernel(L, B, IDS, WTS, E, SHARED, sl, K: tl.constexpr, KO: tl.constexpr, BE: tl.constexpr):
        """Top K of logits + bias (lower index on ties), weight sigmoid(logit); slot K = SHARED at 1."""
        row = tl.program_id(0)
        r = tl.arange(0, BE)
        m = r < E
        lg = tl.load(L + row * sl + r, mask=m, other=0.0)
        s = tl.where(m, lg + tl.load(B + r, mask=m, other=0.0), float("-inf"))
        for j in tl.static_range(K):
            v = tl.max(s, axis=0)
            i = tl.min(tl.where(s == v, r, BE), axis=0)
            li = tl.sum(tl.where(r == i, lg, 0.0), axis=0)
            tl.store(IDS + row * KO + j, i.to(tl.int64))
            tl.store(WTS + row * KO + j, 1.0 / (1.0 + tl.exp(-li)))
            s = tl.where(r == i, float("-inf"), s)
        if KO > K:
            tl.store(IDS + row * KO + K, SHARED.to(tl.int64))
            tl.store(WTS + row * KO + K, 1.0)


if HAVE_TRITON:

    @triton.jit
    def _router_logits_kernel(X, Wg, Y, E, K, sx, sw, sy, BE: tl.constexpr, BK: tl.constexpr):
        """fp32 logits[row, e] = sum_k x[row, k] * gate[e, k]: bf16 operands widened to fp32 (exact
        products), fp32 accumulation; BE experts a program, so a one-row step reads the bf16 router
        once (2 MB) instead of an fp32 copy (4 MB)."""
        row = tl.program_id(1)
        pe = tl.program_id(0)
        re_ = pe * BE + tl.arange(0, BE)
        me = re_ < E
        acc = tl.zeros((BE,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            x = tl.load(X + row * sx + rk, mask=mk, other=0.0).to(tl.float32)
            w = tl.load(Wg + re_[:, None] * sw + rk[None, :], mask=me[:, None] & mk[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(Y + row * sy + re_, acc, mask=me)


def router_logits(x: torch.Tensor, gate_bf16: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    E = gate_bf16.shape[0]
    x = x.contiguous()
    y = torch.empty(M, E, dtype=torch.float32, device=x.device)
    # 2 experts a program, 1024-wide K steps (picked on the GB10)
    _router_logits_kernel[(triton.cdiv(E, 2), M)](x, gate_bf16, y, E, K, x.stride(0), gate_bf16.stride(0),
                                                  y.stride(0), BE=2, BK=1024, num_warps=4)
    return y


def add_rms2(r: torch.Tensor, y: torch.Tensor, w1, w2, eps: float, out_dtype=torch.bfloat16):
    """(r + rms(y) * w1, rms(that) * w2): the post-norm residual add and the next pre-norm."""
    if not (FUSED and HAVE_TRITON and r.is_cuda):
        r2 = r + rms(y, w1, eps, torch.float32)
        return r2, rms(r2, w2, eps, out_dtype)
    M, H = r.shape
    y = y.contiguous()
    ro = torch.empty(M, H, dtype=torch.float32, device=r.device)
    xo = torch.empty(M, H, dtype=out_dtype, device=r.device)
    _add_rms2_kernel[(M,)](r, y, w1, w2, ro, xo, H, eps, r.stride(0), y.stride(0), ro.stride(0), xo.stride(0),
                           BH=triton.next_power_of_2(H), num_warps=8)
    return ro, xo


def rms_fused(x: torch.Tensor, w, eps: float, out_dtype=torch.bfloat16):
    if not (FUSED and HAVE_TRITON and x.is_cuda):
        return rms(x, w, eps, out_dtype)
    M, H = x.shape
    x = x.contiguous()
    y = torch.empty(M, H, dtype=out_dtype, device=x.device)
    _rms_kernel[(M,)](x, w, y, H, eps, x.stride(0), y.stride(0), BH=triton.next_power_of_2(H), num_warps=8)
    return y


def route_fused(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, k: int, shared_id: int | None,
                renorm: bool = False):
    """`route`, with the top-k glue in one kernel (renorm=False only, as Kolibri's config)."""
    if renorm or not (FUSED and HAVE_TRITON and x.is_cuda):
        return route(x, gate.float() if gate.dtype != torch.float32 else gate, bias, k, shared_id, renorm)
    if gate.dtype == torch.bfloat16 and x.shape[0] <= 32:
        logits = router_logits(x, gate)
    else:
        logits = (x.float() @ gate.float().T).contiguous()
    M, E = logits.shape
    ko = k + (shared_id is not None)
    ids = torch.empty(M, ko, dtype=torch.int64, device=x.device)
    w = torch.empty(M, ko, dtype=torch.float32, device=x.device)
    _route_kernel[(M,)](logits, bias, ids, w, E, shared_id if shared_id is not None else 0, logits.stride(0),
                        K=k, KO=ko, BE=triton.next_power_of_2(E), num_warps=4)
    return ids, w
