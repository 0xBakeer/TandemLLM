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
    try:
        from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    except ImportError:                                          # pragma: no cover
        gdc_wait = gdc_launch_dependents = None
except ImportError:                                              # pragma: no cover
    HAVE_TRITON = False

NV_GROUP = 16
FP8_BLOCK = 128
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
DOT32 = os.environ.get("TRITON_INTERPRET") == "1"


#: programmatic dependent launch for the decode step's kernels. Each one-row kernel is
#: launched while the kernel before it still runs: it pulls its own weights toward L2 (they never
#: depend on the step), then waits (`griddepcontrol.wait`) for the previous kernel to finish before
#: it reads an activation or writes anything, and only then lets the next kernel launch. 0: plain
#: launches, every kernel exactly as before.
PDL = os.environ.get("KOLIBRI_PDL", "1") != "0"
#: L2 prefetch of a kernel's own weight lines before its wait, a bit per kernel family
#: (KOLIBRI_PDL_PF): 1 NVFP4 q/o, 2 FP8 projections, 4 routed experts, 8 head, 16 router; 0 none
PDL_PF = int(os.environ.get("KOLIBRI_PDL_PF", "31"))
PF_NV, PF_FP8, PF_MOE, PF_HEAD, PF_ROUTER = 1, 2, 4, 8, 16


def pdl_on(M: int) -> bool:
    return PDL and M == 1 and HAVE_TRITON and gdc_wait is not None and not DOT32   # DOT32: the interpreter


#: PDL (and the weight prefetch before the wait) for the verify rows too, 2 to 64 rows
#: (KOLIBRI_PDL_ROWS): the rows entry points and the small per-row kernels (norms, router, top-k,
#: SwiGLU). Timing only: every kernel waits before it reads an activation or writes.
PDL_ROWS = os.environ.get("KOLIBRI_PDL_ROWS", "1") != "0"


def pdl_rows(M: int) -> bool:
    return PDL_ROWS and 1 < M <= 64 and pdl_on(1)


def pdl_any(M: int) -> bool:
    return pdl_on(M) or pdl_rows(M)


if HAVE_TRITON:

    @triton.jit
    def _gdc(PDL: tl.constexpr):
        """Wait for the previous kernel (all its writes visible), then let the next one launch."""
        if PDL:
            gdc_wait()
            gdc_launch_dependents()

    @triton.jit
    def _l2_prefetch(ptrs):
        """`prefetch.global.L2` of each address (a hint: no data comes back, nothing waits)."""
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; // $0", "=r,l", [ptrs.to(tl.int64)],
                                  dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _pf_rows(base, rows_off, row_len, PF: tl.constexpr, ESZ: tl.constexpr = 1):
        """Prefetch PF 128-byte lines from each row start; offsets and `row_len` in elements of
        ESZ bytes (clipped to the row: lines repeat)."""
        if PF > 0:
            off = tl.minimum(tl.arange(0, PF) * (128 // ESZ), row_len - 1)
            _l2_prefetch(base + rows_off[:, None] + off[None, :])


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

    def matmul_rows(self, x: torch.Tensor, out: torch.Tensor | None = None, swiglu: bool = False):
        """Row i equal, bit for bit, to `matmul(x[i:i+1])` (verify rows): the one-row kernel's
        body once per row. swiglu: x is the fused gate|up output [M, 2K], as `swiglu_matmul`."""
        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        cfg = FP8_1ROW.get((self.N, self.K))
        y = torch.empty(M, self.N, dtype=torch.bfloat16, device=x2.device) if out is None else out
        if not (ONEROW and cfg is not None and x2.is_cuda and HAVE_TRITON) or (swiglu and not FUSE_SWIGLU):
            for i in range(M):
                xi = x2[i:i + 1]
                y[i:i + 1].copy_(self.matmul(swiglu_rows(xi, self.K) if swiglu else xi))
            return y
        bn1, bk1, w1, st1 = cfg
        x2 = x2.to(torch.bfloat16).contiguous()
        pdl = pdl_rows(M)
        if ROWS8 & ROWS8_FP8 and M > 1:
            rb = min(8, M)
            _fp8_rows8[(self.N // bn1, triton.cdiv(M, rb))](x2, self.w, self.s, y, M, self.N, self.K,
                                                            self.w.stride(0), self.s.stride(0),
                                                            x2.stride(0), y.stride(0), BN=bn1, BK=bk1,
                                                            RB=rb, SWI=swiglu, PDL=pdl,
                                                            PF=_lines(self.K, PF_FP8) if pdl else 0,
                                                            num_warps=w1, num_stages=st1, launch_pdl=pdl)
            return y
        _fp8_gemv_1row[(self.N // bn1,)](x2, self.w, self.s, y, self.N, self.K, self.w.stride(0),
                                         self.s.stride(0), BN=bn1, BK=bk1, SWI=swiglu, PDL=pdl,
                                         PF=_lines(self.K, PF_FP8) if pdl else 0,
                                         num_warps=w1, num_stages=st1, M=M, sxm=x2.stride(0),
                                         sym=y.stride(0), launch_pdl=pdl)
        return y

    def swiglu_matmul(self, gu: torch.Tensor) -> torch.Tensor | None:
        """One decode row: SiLU(gu[:K]) * gu[K:] (bf16) times this projection, in one launch.
        None when the one-row path does not apply (the caller runs `swiglu` + `matmul`)."""
        cfg = FP8_1ROW.get((self.N, self.K))
        if not (FUSE_SWIGLU and ONEROW and cfg is not None and gu.is_cuda and HAVE_TRITON
                and gu.shape[0] == 1 and gu.shape[-1] == 2 * self.K):
            return None
        bn1, bk1, w1, st1 = cfg
        pdl = pdl_on(1)
        gu = gu.contiguous()
        y = torch.empty(1, self.N, dtype=torch.bfloat16, device=gu.device)
        _fp8_gemv_1row[(self.N // bn1,)](gu, self.w, self.s, y, self.N, self.K, self.w.stride(0),
                                         self.s.stride(0), BN=bn1, BK=bk1, PDL=pdl,
                                         PF=_lines(self.K, PF_FP8) if pdl else 0, SWI=True,
                                         num_warps=w1, num_stages=st1, launch_pdl=pdl)
        return y

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
            pdl = pdl_on(M)
            _fp8_gemv_1row[(self.N // bn1,)](x2, self.w, self.s, y, self.N, self.K,
                                             self.w.stride(0), self.s.stride(0), BN=bn1, BK=bk1,
                                             PDL=pdl, PF=_lines(self.K, PF_FP8) if pdl else 0,
                                             num_warps=w1, num_stages=st1, launch_pdl=pdl)
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


#: one-row NVFP4 kernels: the GEMV for q/o, and the routed experts' one-row kernels, with a
#: cheap e2m1 decode and an accumulator per lane summed once at the end. 0: the older kernels.
NV1ROW = os.environ.get("KOLIBRI_NV1ROW", "1") != "0"
#: e2m1 decode: 0 = fp16 bit pattern (portable), 1 = `cvt.rn.f16x2.e2m1x2` (sm_100 and newer; the
#: default there: 1 to 3 % faster on the GB10, the same values)


def _asm_default() -> int:
    try:
        return int(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10)
    except Exception:                                                # noqa: BLE001
        return 0


NV_ASM = int(os.environ.get("KOLIBRI_NV_ASM", _asm_default()))

if HAVE_TRITON:

    @triton.jit
    def _e2m1_bits(c):
        """uint8 codes -> (low nibble, high nibble) as fp32 e2m1 * 2^-14. The nibble's sign goes to
        fp16 bit 15 and its exponent and mantissa to bits 11..9, so e > 0 reads as
        2^(e-15) * (1 + m/2) and e = 0 as the subnormal m * 2^-15: both e2m1 * 2^-14, exactly."""
        ci = c.to(tl.int32)
        lo = ((ci & 0x08) << 12) | ((ci & 0x07) << 9)
        hi = ((ci & 0x80) << 8) | ((ci & 0x70) << 5)
        lo = lo.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        hi = hi.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        return lo, hi

    @triton.jit
    def _e2m1_asm(c):
        """uint8 codes -> (low nibble, high nibble) as fp32 e2m1 values, by the hardware convert
        (one `cvt` per byte, then two `prmt` per four bytes to split the halves)."""
        lo, hi = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 a0, a1, a2, a3;
            .reg .b32 r0, r1, r2, r3;
            mov.b32 {a0, a1, a2, a3}, $4;
            cvt.rn.f16x2.e2m1x2 r0, a0;
            cvt.rn.f16x2.e2m1x2 r1, a1;
            cvt.rn.f16x2.e2m1x2 r2, a2;
            cvt.rn.f16x2.e2m1x2 r3, a3;
            prmt.b32 $0, r0, r1, 0x5410;
            prmt.b32 $1, r2, r3, 0x5410;
            prmt.b32 $2, r0, r1, 0x7632;
            prmt.b32 $3, r2, r3, 0x7632;
            }
            """,
            constraints="=r,=r,=r,=r,r",
            args=[c], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
        return lo.to(tl.float32), hi.to(tl.float32)

    @triton.jit
    def _x_pairs(X, k0, BKH: tl.constexpr, ASM: tl.constexpr):
        """x[k0 : k0 + 2*BKH] as (even, odd) fp32 halves; times 2^14 for the bit decode, so each
        product is e2m1 * x exactly."""
        xx = tl.load(X + k0 + tl.arange(0, 2 * BKH)).to(tl.float32)
        xe, xo = tl.split(tl.reshape(xx, (BKH, 2)))
        if ASM == 0:
            xe = xe * 16384.0
            xo = xo * 16384.0
        return xe, xo

    @triton.jit
    def _nv_acc(acc, c, xe, xo, Sp, srow, k0, BN: tl.constexpr, BKH: tl.constexpr,
                ASM: tl.constexpr, MODE: tl.constexpr):
        """acc += the BN rows' partial products over K = k0 .. k0 + 2*BKH; `Sp + srow` points at each
        row's e4m3 group scales. MODE 0: acc [BN, BKH], the scale gathered per byte; MODE 1:
        acc [BN, BKH/8], the 16-value groups summed before their scale."""
        if ASM == 1:
            lo, hi = _e2m1_asm(c)
        else:
            lo, hi = _e2m1_bits(c)
        p = lo * xe[None, :] + hi * xo[None, :]
        if MODE == 0:
            rb = tl.arange(0, BKH)
            sc = tl.load(Sp + srow[:, None] + (k0 // 16 + rb // 8)[None, :]).to(tl.float32)
            acc += p * sc
        else:
            G: tl.constexpr = BKH // 8
            sc = tl.load(Sp + srow[:, None] + (k0 // 16 + tl.arange(0, G))[None, :]).to(tl.float32)
            acc += tl.sum(tl.reshape(p, (BN, G, 8)), 2) * sc
        return acc

    @triton.jit
    def _nv_gemv_1row(X, Wc, Ws, S2, Y, N, K, swn, ssn, BN: tl.constexpr, BK: tl.constexpr,
                      ASM: tl.constexpr, MODE: tl.constexpr, PDL: tl.constexpr = False,
                      PF: tl.constexpr = 0, PFS: tl.constexpr = 0, M=1, sxm=0, sym=0):
        """One row of an NVFP4 projection: BN output rows a program over the whole K; no
        tl.dot, no per-step reduction. N % BN == 0 and K % BK == 0. PDL: PF code lines and PFS scale
        lines a row prefetched before the wait. M > 1 (verify, `matmul_rows`): the one-row body
        once per row, so every row gets the bits a one-row call gives it."""
        pn = tl.program_id(0)
        rn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pf_rows(Wc, rn * swn, K // 2, PF)
            _pf_rows(Ws, rn * ssn, K // 16, PFS)
        _gdc(PDL)
        BKH: tl.constexpr = BK // 2
        rb = tl.arange(0, BKH)
        for m in range(0, M):
            if MODE == 0:
                acc = tl.zeros((BN, BKH), dtype=tl.float32)
            else:
                acc = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            for k0 in range(0, K, BK):
                c = tl.load(Wc + rn[:, None] * swn + (k0 // 2 + rb)[None, :])
                xe, xo = _x_pairs(X + m * sxm, k0, BKH, ASM)
                acc = _nv_acc(acc, c, xe, xo, Ws, rn * ssn, k0, BN, BKH, ASM, MODE)
            y = tl.sum(acc, 1) * tl.load(S2 + rn)
            tl.store(Y + m * sym + rn, y.to(tl.bfloat16))


#: (BN, BK, warps, stages, mode) of `_nv_gemv_1row` per served (N, K), picked on the GB10 with cold
#: weights under graph timing
NV_1ROW = {(6144, 2560): (16, 512, 8, 3, 0), (2560, 6144): (16, 512, 8, 3, 1)}


def _lines(nbytes: int, bit: int, cap: int = 64) -> int:
    """Prefetch lines for a row of `nbytes` (a power of two, at most `cap`), 0 when PDL_PF is off."""
    if not PDL_PF & bit:
        return 0
    return min(cap, triton.next_power_of_2(max(1, -(-nbytes // 128))))


def nv_gemv_1row(lin: "NVFP4Linear", x2: torch.Tensor, y: torch.Tensor, cfg=None) -> torch.Tensor:
    bn, bk, warps, stages, mode = cfg or NV_1ROW[(lin.N, lin.K)]
    pdl = pdl_on(1)
    _nv_gemv_1row[(lin.N // bn,)](x2, lin.w, lin.s, lin.s2, y, lin.N, lin.K, lin.w.stride(0),
                                  lin.s.stride(0), BN=bn, BK=bk, ASM=NV_ASM, MODE=mode, PDL=pdl,
                                  PF=_lines(lin.K // 2, PF_NV) if pdl else 0,
                                  PFS=_lines(lin.K // 16, PF_NV) if pdl else 0,
                                  num_warps=warps, num_stages=stages, launch_pdl=pdl)
    return y


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

    def matmul_rows(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Row i equal, bit for bit, to `matmul(x[i:i+1])` (verify rows)."""
        x2 = x.reshape(-1, self.K)
        M = x2.shape[0]
        y = torch.empty(M, self.N, dtype=torch.bfloat16, device=x2.device) if out is None else out
        if not (NV1ROW and (self.N, self.K) in NV_1ROW and x2.is_cuda and HAVE_TRITON):
            if x2.is_cuda and HAVE_TRITON and not NV1ROW:
                # the tiled kernel gives a row the same bits in any launch of up to 16 rows (one
                # 16-row tile, the same program shape): chunks of 16
                for r0 in range(0, M, 16):
                    self.matmul(x2[r0:r0 + 16], out=y[r0:r0 + 16])
                return y
            for i in range(M):
                self.matmul(x2[i:i + 1], out=y[i:i + 1])
            return y
        bn, bk, warps, stages, mode = NV_1ROW[(self.N, self.K)]
        x2 = x2.to(torch.bfloat16).contiguous()
        pdl = pdl_rows(M)
        if ROWS8 & ROWS8_NV and M > 1:
            rb = min(8, M)
            _nv_rows8[(self.N // bn, triton.cdiv(M, rb))](x2, self.w, self.s, self.s2, y, M, self.N, self.K,
                                                          self.w.stride(0), self.s.stride(0), x2.stride(0),
                                                          y.stride(0), BN=bn, BK=bk, ASM=NV_ASM, MODE=mode,
                                                          RB=rb, PDL=pdl, PF=_lines(self.K // 2, PF_NV) if pdl else 0,
                                                          num_warps=warps, num_stages=stages, launch_pdl=pdl)
            return y
        _nv_gemv_1row[(self.N // bn,)](x2, self.w, self.s, self.s2, y, self.N, self.K, self.w.stride(0),
                                       self.s.stride(0), BN=bn, BK=bk, ASM=NV_ASM, MODE=mode, PDL=pdl,
                                       PF=_lines(self.K // 2, PF_NV) if pdl else 0,
                                       PFS=_lines(self.K // 16, PF_NV) if pdl else 0,
                                       num_warps=warps, num_stages=stages, M=M, sxm=x2.stride(0),
                                       sym=y.stride(0), launch_pdl=pdl)
        return y

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
        if M == 1 and NV1ROW and (self.N, self.K) in NV_1ROW:
            nv_gemv_1row(self, x2, y)
            return y.view(*lead, self.N) if out is None else y
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
#: decode fusions: the shared expert's SwiGLU inside its down projection (KOLIBRI_FUSE_SWIGLU),
#: the routed experts' combine inside the residual add and norm after them (KOLIBRI_FUSE_COMBINE)
FUSE_SWIGLU = os.environ.get("KOLIBRI_FUSE_SWIGLU", "1") != "0"
FUSE_COMBINE = os.environ.get("KOLIBRI_FUSE_COMBINE", "1") != "0"
#: one-row kernels for the routed experts with lane accumulators (KOLIBRI_MOE2=0 keeps the earlier
#: one-row kernels). (BN, BK, warps, stages, mode) for gate/up and down, picked on the GB10 with cold
#: experts. The FP8 one-row GEMV keeps a single accumulator: lanes made no difference there.
MOE2 = os.environ.get("KOLIBRI_MOE2", "1") != "0"
MOE_1ROW2 = ((16, 512, 4, 3, 1), (8, 256, 4, 3, 1))


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
    def _moe_gate_up_1row2(X, IDS, GC, GS, G2, UC, US, U2, Hout, K, I, swe, swn, sse, ssn, shm,
                           BN: tl.constexpr, BK: tl.constexpr, ASM: tl.constexpr, MODE: tl.constexpr,
                           PDL: tl.constexpr = False, PF: tl.constexpr = 0, IDS_EARLY: tl.constexpr = False,
                           KTOP=1, sxm=0, ORD=None, USE_ORD: tl.constexpr = False):
        """`_moe_gate_up_1row` with lane accumulators, one reduction at the end.
        I % BN == 0, K % BK == 0. IDS_EARLY: the router's ids are final before the wait (the kernel
        before this one passed its own wait after the router finished: the shared expert's down
        projection runs between them), so the expert's lines are prefetched before the wait too.
        USE_ORD (verify rows): program j runs pair ORD[j], the pairs sorted by expert, so pairs of
        one expert run side by side and its tile comes from L2 for all but the first."""
        j = tl.program_id(0)
        pn = tl.program_id(1)
        rn = pn * BN + tl.arange(0, BN)
        if PDL and IDS_EARLY:
            e = tl.load(IDS + j)
            _pf_rows(GC + e * swe, rn * swn, K // 2, PF)
            _pf_rows(UC + e * swe, rn * swn, K // 2, PF)
        _gdc(PDL)
        if USE_ORD:
            j = tl.load(ORD + j)
        e = tl.load(IDS + j)
        X = X + (j // KTOP) * sxm                 # rows (verify): program j is pair (row, slot)
        BKH: tl.constexpr = BK // 2
        if MODE == 0:
            ag = tl.zeros((BN, BKH), dtype=tl.float32)
            au = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        rb = tl.arange(0, BKH)
        cg = GC + e * swe + rn[:, None] * swn
        cu = UC + e * swe + rn[:, None] * swn
        srow = e * sse + rn * ssn
        for k0 in range(0, K, BK):
            xe, xo = _x_pairs(X, k0, BKH, ASM)
            c = tl.load(cg + (k0 // 2 + rb)[None, :])
            ag = _nv_acc(ag, c, xe, xo, GS, srow, k0, BN, BKH, ASM, MODE)
            c = tl.load(cu + (k0 // 2 + rb)[None, :])
            au = _nv_acc(au, c, xe, xo, US, srow, k0, BN, BKH, ASM, MODE)
        g = tl.sum(ag, 1) * tl.load(G2 + e)
        u = tl.sum(au, 1) * tl.load(U2 + e)
        h = g * tl.sigmoid(g) * u
        tl.store(Hout + j * shm + rn, h.to(tl.bfloat16))

    @triton.jit
    def _moe_down_1row2(Hin, IDS, DC, DS, D2, RW, P, I, H, shm, swe, swn, sse, ssn, spm,
                        BN: tl.constexpr, BK: tl.constexpr, ASM: tl.constexpr, MODE: tl.constexpr,
                        PDL: tl.constexpr = False, PF: tl.constexpr = 0, ORD=None,
                        USE_ORD: tl.constexpr = False):
        """Down projection of the six experts; its ids are final before the wait (gate/up, the
        kernel before it, read them after its own wait). USE_ORD as `_moe_gate_up_1row2`."""
        j = tl.program_id(0)
        if USE_ORD:
            j = tl.load(ORD + j)
        pn = tl.program_id(1)
        rn = pn * BN + tl.arange(0, BN)
        if PDL:
            e = tl.load(IDS + j)
            _pf_rows(DC + e * swe, rn * swn, I // 2, PF)
        _gdc(PDL)
        e = tl.load(IDS + j)
        BKH: tl.constexpr = BK // 2
        if MODE == 0:
            acc = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            acc = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        rb = tl.arange(0, BKH)
        for k0 in range(0, I, BK):
            xe, xo = _x_pairs(Hin + j * shm, k0, BKH, ASM)
            c = tl.load(DC + e * swe + rn[:, None] * swn + (k0 // 2 + rb)[None, :])
            acc = _nv_acc(acc, c, xe, xo, DS, e * sse + rn * ssn, k0, BN, BKH, ASM, MODE)
        rw = tl.load(RW + j).to(tl.float32)
        tl.store(P + j * spm + rn, tl.sum(acc, 1) * tl.load(D2 + e) * rw)

    @triton.jit
    def _fp8_gemv_1row(X, W, S, Y, N, K, swn, ssn, BN: tl.constexpr, BK: tl.constexpr,
                       PDL: tl.constexpr = False, PF: tl.constexpr = 0, SWI: tl.constexpr = False,
                       M=1, sxm=0, sym=0):
        """One row of an FP8 block-scaled projection: BN output rows a program over the
        whole K, 128-wide scale blocks applied to the fp32 partial sums. SWI: X is the fused
        gate|up output [2K] and the input is SiLU(g) * u rounded to bf16, in `_swiglu_kernel`'s
        arithmetic (the shared expert's down projection without the SwiGLU launch)."""
        pn = tl.program_id(0)
        rn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pf_rows(W, rn * swn, K, PF)
        _gdc(PDL)
        s_row = (pn * BN) // 128
        for m in range(0, M):
            acc = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, K, BK):
                kk = k0 + tl.arange(0, BK)
                if SWI:
                    g = tl.load(X + m * sxm + kk).to(tl.float32)
                    u = tl.load(X + m * sxm + K + kk).to(tl.float32)
                    x = (tl.math.div_rn(g, 1.0 + libdevice.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
                else:
                    x = tl.load(X + m * sxm + kk).to(tl.float32)
                w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
                part = tl.reshape(w * x[None, :], (BN, BK // 128, 128))
                sc = tl.load(S + s_row * ssn + k0 // 128 + tl.arange(0, BK // 128))
                acc += tl.sum(tl.sum(part, 2) * sc[None, :], 1)
            tl.store(Y + m * sym + rn, acc.to(tl.bfloat16))

    @triton.jit
    def _moe_combine_kernel(P, Y, H, KTOP, spm, sym, EX, sex, BN: tl.constexpr,
                            EXTRA: tl.constexpr, PDL: tl.constexpr = False):
        _gdc(PDL)
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
    def _swiglu_kernel(GU, A, F, sg, sa, BF: tl.constexpr, PDL: tl.constexpr = False):
        """bf16 SiLU(g) * u from the fused gate|up output, with the arithmetic of torch's
        `silu(gu.float()[:, :F]) * gu.float()[:, F:]` (IEEE exp and division), rounded to bf16."""
        _gdc(PDL)
        row = tl.program_id(0)
        pc = tl.program_id(1)
        c = pc * BF + tl.arange(0, BF)
        m = c < F
        g = tl.load(GU + row * sg + c, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GU + row * sg + F + c, mask=m, other=0.0).to(tl.float32)
        si = tl.math.div_rn(g, 1.0 + libdevice.exp(-g))
        tl.store(A + row * sa + c, (si * u).to(tl.bfloat16), mask=m)


# ------------------------------------------------------------------------------ verify rows
# Multi-row forms of the one-row decode kernels for the verify (engine/kolibri/verify.py): a weight tile is read once for
# all rows (or, for the experts, for all pairs that chose that expert), and each row keeps its own
# accumulator and the one-row kernel's order of products and sums, so every row is bit-equal to
# the one-row call (tests/test_kolibri_kern2.py). Unrolled for up to 8 rows a program.
if HAVE_TRITON:

    @triton.jit
    def _head_rows8(X, W, S, Y, M, N, K, swn, sxm, sym, BN: tl.constexpr, BK: tl.constexpr,
                    RB: tl.constexpr, PDL: tl.constexpr = False, PF: tl.constexpr = 0):
        """Up to RB (<= 8) rows a program against one weight tile (verify): each row keeps its own
        lane accumulator and runs `_head_gemv2`'s arithmetic in its order, so row i equals the
        one-row call bit for bit; the tile is read and widened once for all rows."""
        pn = tl.program_id(0)
        r0 = tl.program_id(1) * RB
        rn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pf_rows(W, rn * swn, K, PF)
        _gdc(PDL)
        a0 = tl.zeros((BN, BK), dtype=tl.float32)
        a1 = tl.zeros((BN, BK), dtype=tl.float32)
        a2 = tl.zeros((BN, BK), dtype=tl.float32)
        a3 = tl.zeros((BN, BK), dtype=tl.float32)
        a4 = tl.zeros((BN, BK), dtype=tl.float32)
        a5 = tl.zeros((BN, BK), dtype=tl.float32)
        a6 = tl.zeros((BN, BK), dtype=tl.float32)
        a7 = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
            if RB > 0:
                x0 = tl.load(X + (r0 + 0) * sxm + kk, mask=(kk < K) & (r0 + 0 < M), other=0.0)
                a0 += w * x0[None, :]
            if RB > 1:
                x1 = tl.load(X + (r0 + 1) * sxm + kk, mask=(kk < K) & (r0 + 1 < M), other=0.0)
                a1 += w * x1[None, :]
            if RB > 2:
                x2 = tl.load(X + (r0 + 2) * sxm + kk, mask=(kk < K) & (r0 + 2 < M), other=0.0)
                a2 += w * x2[None, :]
            if RB > 3:
                x3 = tl.load(X + (r0 + 3) * sxm + kk, mask=(kk < K) & (r0 + 3 < M), other=0.0)
                a3 += w * x3[None, :]
            if RB > 4:
                x4 = tl.load(X + (r0 + 4) * sxm + kk, mask=(kk < K) & (r0 + 4 < M), other=0.0)
                a4 += w * x4[None, :]
            if RB > 5:
                x5 = tl.load(X + (r0 + 5) * sxm + kk, mask=(kk < K) & (r0 + 5 < M), other=0.0)
                a5 += w * x5[None, :]
            if RB > 6:
                x6 = tl.load(X + (r0 + 6) * sxm + kk, mask=(kk < K) & (r0 + 6 < M), other=0.0)
                a6 += w * x6[None, :]
            if RB > 7:
                x7 = tl.load(X + (r0 + 7) * sxm + kk, mask=(kk < K) & (r0 + 7 < M), other=0.0)
                a7 += w * x7[None, :]
        s = tl.load(S + rn)
        if RB > 0:
            tl.store(Y + (r0 + 0) * sym + rn, tl.sum(a0, 1) * s, mask=(rn < N) & (r0 + 0 < M))
        if RB > 1:
            tl.store(Y + (r0 + 1) * sym + rn, tl.sum(a1, 1) * s, mask=(rn < N) & (r0 + 1 < M))
        if RB > 2:
            tl.store(Y + (r0 + 2) * sym + rn, tl.sum(a2, 1) * s, mask=(rn < N) & (r0 + 2 < M))
        if RB > 3:
            tl.store(Y + (r0 + 3) * sym + rn, tl.sum(a3, 1) * s, mask=(rn < N) & (r0 + 3 < M))
        if RB > 4:
            tl.store(Y + (r0 + 4) * sym + rn, tl.sum(a4, 1) * s, mask=(rn < N) & (r0 + 4 < M))
        if RB > 5:
            tl.store(Y + (r0 + 5) * sym + rn, tl.sum(a5, 1) * s, mask=(rn < N) & (r0 + 5 < M))
        if RB > 6:
            tl.store(Y + (r0 + 6) * sym + rn, tl.sum(a6, 1) * s, mask=(rn < N) & (r0 + 6 < M))
        if RB > 7:
            tl.store(Y + (r0 + 7) * sym + rn, tl.sum(a7, 1) * s, mask=(rn < N) & (r0 + 7 < M))

    @triton.jit
    def _fp8_rows8(X, W, S, Y, M, N, K, swn, ssn, sxm, sym, BN: tl.constexpr, BK: tl.constexpr,
                   RB: tl.constexpr, SWI: tl.constexpr, PDL: tl.constexpr = False, PF: tl.constexpr = 0):
        """`_fp8_gemv_1row` for up to RB (<= 8) rows a program, the tile read once (verify);
        each row's sums in the one-row kernel's order, so row i equals the one-row call bit for bit."""
        pn = tl.program_id(0)
        r0 = tl.program_id(1) * RB
        rn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pf_rows(W, rn * swn, K, PF)
        _gdc(PDL)
        s_row = (pn * BN) // 128
        a0 = tl.zeros((BN,), dtype=tl.float32)
        a1 = tl.zeros((BN,), dtype=tl.float32)
        a2 = tl.zeros((BN,), dtype=tl.float32)
        a3 = tl.zeros((BN,), dtype=tl.float32)
        a4 = tl.zeros((BN,), dtype=tl.float32)
        a5 = tl.zeros((BN,), dtype=tl.float32)
        a6 = tl.zeros((BN,), dtype=tl.float32)
        a7 = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
            sc = tl.load(S + s_row * ssn + k0 // 128 + tl.arange(0, BK // 128))
            if RB > 0:
                if SWI:
                    g0 = tl.load(X + (r0 + 0) * sxm + kk, mask=r0 + 0 < M, other=0.0).to(tl.float32)
                    u0 = tl.load(X + (r0 + 0) * sxm + K + kk, mask=r0 + 0 < M, other=0.0).to(tl.float32)
                    x0 = (tl.math.div_rn(g0, 1.0 + libdevice.exp(-g0)) * u0).to(tl.bfloat16).to(tl.float32)
                else:
                    x0 = tl.load(X + (r0 + 0) * sxm + kk, mask=r0 + 0 < M, other=0.0).to(tl.float32)
                p0 = tl.reshape(w * x0[None, :], (BN, BK // 128, 128))
                a0 += tl.sum(tl.sum(p0, 2) * sc[None, :], 1)
            if RB > 1:
                if SWI:
                    g1 = tl.load(X + (r0 + 1) * sxm + kk, mask=r0 + 1 < M, other=0.0).to(tl.float32)
                    u1 = tl.load(X + (r0 + 1) * sxm + K + kk, mask=r0 + 1 < M, other=0.0).to(tl.float32)
                    x1 = (tl.math.div_rn(g1, 1.0 + libdevice.exp(-g1)) * u1).to(tl.bfloat16).to(tl.float32)
                else:
                    x1 = tl.load(X + (r0 + 1) * sxm + kk, mask=r0 + 1 < M, other=0.0).to(tl.float32)
                p1 = tl.reshape(w * x1[None, :], (BN, BK // 128, 128))
                a1 += tl.sum(tl.sum(p1, 2) * sc[None, :], 1)
            if RB > 2:
                if SWI:
                    g2 = tl.load(X + (r0 + 2) * sxm + kk, mask=r0 + 2 < M, other=0.0).to(tl.float32)
                    u2 = tl.load(X + (r0 + 2) * sxm + K + kk, mask=r0 + 2 < M, other=0.0).to(tl.float32)
                    x2 = (tl.math.div_rn(g2, 1.0 + libdevice.exp(-g2)) * u2).to(tl.bfloat16).to(tl.float32)
                else:
                    x2 = tl.load(X + (r0 + 2) * sxm + kk, mask=r0 + 2 < M, other=0.0).to(tl.float32)
                p2 = tl.reshape(w * x2[None, :], (BN, BK // 128, 128))
                a2 += tl.sum(tl.sum(p2, 2) * sc[None, :], 1)
            if RB > 3:
                if SWI:
                    g3 = tl.load(X + (r0 + 3) * sxm + kk, mask=r0 + 3 < M, other=0.0).to(tl.float32)
                    u3 = tl.load(X + (r0 + 3) * sxm + K + kk, mask=r0 + 3 < M, other=0.0).to(tl.float32)
                    x3 = (tl.math.div_rn(g3, 1.0 + libdevice.exp(-g3)) * u3).to(tl.bfloat16).to(tl.float32)
                else:
                    x3 = tl.load(X + (r0 + 3) * sxm + kk, mask=r0 + 3 < M, other=0.0).to(tl.float32)
                p3 = tl.reshape(w * x3[None, :], (BN, BK // 128, 128))
                a3 += tl.sum(tl.sum(p3, 2) * sc[None, :], 1)
            if RB > 4:
                if SWI:
                    g4 = tl.load(X + (r0 + 4) * sxm + kk, mask=r0 + 4 < M, other=0.0).to(tl.float32)
                    u4 = tl.load(X + (r0 + 4) * sxm + K + kk, mask=r0 + 4 < M, other=0.0).to(tl.float32)
                    x4 = (tl.math.div_rn(g4, 1.0 + libdevice.exp(-g4)) * u4).to(tl.bfloat16).to(tl.float32)
                else:
                    x4 = tl.load(X + (r0 + 4) * sxm + kk, mask=r0 + 4 < M, other=0.0).to(tl.float32)
                p4 = tl.reshape(w * x4[None, :], (BN, BK // 128, 128))
                a4 += tl.sum(tl.sum(p4, 2) * sc[None, :], 1)
            if RB > 5:
                if SWI:
                    g5 = tl.load(X + (r0 + 5) * sxm + kk, mask=r0 + 5 < M, other=0.0).to(tl.float32)
                    u5 = tl.load(X + (r0 + 5) * sxm + K + kk, mask=r0 + 5 < M, other=0.0).to(tl.float32)
                    x5 = (tl.math.div_rn(g5, 1.0 + libdevice.exp(-g5)) * u5).to(tl.bfloat16).to(tl.float32)
                else:
                    x5 = tl.load(X + (r0 + 5) * sxm + kk, mask=r0 + 5 < M, other=0.0).to(tl.float32)
                p5 = tl.reshape(w * x5[None, :], (BN, BK // 128, 128))
                a5 += tl.sum(tl.sum(p5, 2) * sc[None, :], 1)
            if RB > 6:
                if SWI:
                    g6 = tl.load(X + (r0 + 6) * sxm + kk, mask=r0 + 6 < M, other=0.0).to(tl.float32)
                    u6 = tl.load(X + (r0 + 6) * sxm + K + kk, mask=r0 + 6 < M, other=0.0).to(tl.float32)
                    x6 = (tl.math.div_rn(g6, 1.0 + libdevice.exp(-g6)) * u6).to(tl.bfloat16).to(tl.float32)
                else:
                    x6 = tl.load(X + (r0 + 6) * sxm + kk, mask=r0 + 6 < M, other=0.0).to(tl.float32)
                p6 = tl.reshape(w * x6[None, :], (BN, BK // 128, 128))
                a6 += tl.sum(tl.sum(p6, 2) * sc[None, :], 1)
            if RB > 7:
                if SWI:
                    g7 = tl.load(X + (r0 + 7) * sxm + kk, mask=r0 + 7 < M, other=0.0).to(tl.float32)
                    u7 = tl.load(X + (r0 + 7) * sxm + K + kk, mask=r0 + 7 < M, other=0.0).to(tl.float32)
                    x7 = (tl.math.div_rn(g7, 1.0 + libdevice.exp(-g7)) * u7).to(tl.bfloat16).to(tl.float32)
                else:
                    x7 = tl.load(X + (r0 + 7) * sxm + kk, mask=r0 + 7 < M, other=0.0).to(tl.float32)
                p7 = tl.reshape(w * x7[None, :], (BN, BK // 128, 128))
                a7 += tl.sum(tl.sum(p7, 2) * sc[None, :], 1)
        if RB > 0:
            tl.store(Y + (r0 + 0) * sym + rn, a0.to(tl.bfloat16), mask=r0 + 0 < M)
        if RB > 1:
            tl.store(Y + (r0 + 1) * sym + rn, a1.to(tl.bfloat16), mask=r0 + 1 < M)
        if RB > 2:
            tl.store(Y + (r0 + 2) * sym + rn, a2.to(tl.bfloat16), mask=r0 + 2 < M)
        if RB > 3:
            tl.store(Y + (r0 + 3) * sym + rn, a3.to(tl.bfloat16), mask=r0 + 3 < M)
        if RB > 4:
            tl.store(Y + (r0 + 4) * sym + rn, a4.to(tl.bfloat16), mask=r0 + 4 < M)
        if RB > 5:
            tl.store(Y + (r0 + 5) * sym + rn, a5.to(tl.bfloat16), mask=r0 + 5 < M)
        if RB > 6:
            tl.store(Y + (r0 + 6) * sym + rn, a6.to(tl.bfloat16), mask=r0 + 6 < M)
        if RB > 7:
            tl.store(Y + (r0 + 7) * sym + rn, a7.to(tl.bfloat16), mask=r0 + 7 < M)

    @triton.jit
    def _nv_rowacc(acc, lo, hi, xe, xo, sc, BN: tl.constexpr, BKH: tl.constexpr, MODE: tl.constexpr):
        """`_nv_acc` after its decode and scale load: the same products, sums and scale for one row."""
        p = lo * xe[None, :] + hi * xo[None, :]
        if MODE == 0:
            acc += p * sc
        else:
            G: tl.constexpr = BKH // 8
            acc += tl.sum(tl.reshape(p, (BN, G, 8)), 2) * sc
        return acc

    @triton.jit
    def _nv_sc(Sp, srow, k0, BKH: tl.constexpr, MODE: tl.constexpr):
        """The e4m3 group scales `_nv_acc` loads for one K step."""
        if MODE == 0:
            rb = tl.arange(0, BKH)
            return tl.load(Sp + srow[:, None] + (k0 // 16 + rb // 8)[None, :]).to(tl.float32)
        else:
            return tl.load(Sp + srow[:, None] + (k0 // 16 + tl.arange(0, BKH // 8))[None, :]).to(tl.float32)

    @triton.jit
    def _nv_dec(c, ASM: tl.constexpr):
        if ASM == 1:
            lo, hi = _e2m1_asm(c)
        else:
            lo, hi = _e2m1_bits(c)
        return lo, hi

    @triton.jit
    def _moe_gate_up_grp8(X, GE, GP, GN, GC, GS, G2, UC, US, U2, Hout, K, I, swe, swn, sse, ssn, shm,
                          sxm, KTOP, BN: tl.constexpr, BK: tl.constexpr, ASM: tl.constexpr,
                          MODE: tl.constexpr, PDL: tl.constexpr = False):
        """Verify rows, grouped by expert: a program owns up to 8 (row, slot) pairs that chose
        the same expert and reads its gate/up tile once for all of them; each pair runs
        `_moe_gate_up_1row2`'s arithmetic in its order, so its h row equals the one-row call's."""
        _gdc(PDL)
        g = tl.program_id(0)
        pn = tl.program_id(1)
        cnt = tl.load(GN + g)
        if cnt == 0:
            return
        e = tl.load(GE + g)
        rn = pn * BN + tl.arange(0, BN)
        BKH: tl.constexpr = BK // 2
        rb = tl.arange(0, BKH)
        q0 = tl.load(GP + g * 8 + 0)
        x0p = X + (q0 // KTOP) * sxm
        q1 = tl.load(GP + g * 8 + 1)
        x1p = X + (q1 // KTOP) * sxm
        q2 = tl.load(GP + g * 8 + 2)
        x2p = X + (q2 // KTOP) * sxm
        q3 = tl.load(GP + g * 8 + 3)
        x3p = X + (q3 // KTOP) * sxm
        q4 = tl.load(GP + g * 8 + 4)
        x4p = X + (q4 // KTOP) * sxm
        q5 = tl.load(GP + g * 8 + 5)
        x5p = X + (q5 // KTOP) * sxm
        q6 = tl.load(GP + g * 8 + 6)
        x6p = X + (q6 // KTOP) * sxm
        q7 = tl.load(GP + g * 8 + 7)
        x7p = X + (q7 // KTOP) * sxm
        if MODE == 0:
            ag0 = tl.zeros((BN, BKH), dtype=tl.float32)
            au0 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag0 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au0 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag1 = tl.zeros((BN, BKH), dtype=tl.float32)
            au1 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag1 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au1 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag2 = tl.zeros((BN, BKH), dtype=tl.float32)
            au2 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag2 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au2 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag3 = tl.zeros((BN, BKH), dtype=tl.float32)
            au3 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag3 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au3 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag4 = tl.zeros((BN, BKH), dtype=tl.float32)
            au4 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag4 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au4 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag5 = tl.zeros((BN, BKH), dtype=tl.float32)
            au5 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag5 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au5 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag6 = tl.zeros((BN, BKH), dtype=tl.float32)
            au6 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag6 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au6 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            ag7 = tl.zeros((BN, BKH), dtype=tl.float32)
            au7 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            ag7 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
            au7 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        cg = GC + e * swe + rn[:, None] * swn
        cu = UC + e * swe + rn[:, None] * swn
        srow = e * sse + rn * ssn
        for k0 in range(0, K, BK):
            lg_, hg_ = _nv_dec(tl.load(cg + (k0 // 2 + rb)[None, :]), ASM)
            lu_, hu_ = _nv_dec(tl.load(cu + (k0 // 2 + rb)[None, :]), ASM)
            scg = _nv_sc(GS, srow, k0, BKH, MODE)
            scu = _nv_sc(US, srow, k0, BKH, MODE)
            if cnt > 0:
                xe, xo = _x_pairs(x0p, k0, BKH, ASM)
                ag0 = _nv_rowacc(ag0, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au0 = _nv_rowacc(au0, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 1:
                xe, xo = _x_pairs(x1p, k0, BKH, ASM)
                ag1 = _nv_rowacc(ag1, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au1 = _nv_rowacc(au1, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 2:
                xe, xo = _x_pairs(x2p, k0, BKH, ASM)
                ag2 = _nv_rowacc(ag2, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au2 = _nv_rowacc(au2, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 3:
                xe, xo = _x_pairs(x3p, k0, BKH, ASM)
                ag3 = _nv_rowacc(ag3, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au3 = _nv_rowacc(au3, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 4:
                xe, xo = _x_pairs(x4p, k0, BKH, ASM)
                ag4 = _nv_rowacc(ag4, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au4 = _nv_rowacc(au4, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 5:
                xe, xo = _x_pairs(x5p, k0, BKH, ASM)
                ag5 = _nv_rowacc(ag5, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au5 = _nv_rowacc(au5, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 6:
                xe, xo = _x_pairs(x6p, k0, BKH, ASM)
                ag6 = _nv_rowacc(ag6, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au6 = _nv_rowacc(au6, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
            if cnt > 7:
                xe, xo = _x_pairs(x7p, k0, BKH, ASM)
                ag7 = _nv_rowacc(ag7, lg_, hg_, xe, xo, scg, BN, BKH, MODE)
                au7 = _nv_rowacc(au7, lu_, hu_, xe, xo, scu, BN, BKH, MODE)
        g2 = tl.load(G2 + e)
        u2 = tl.load(U2 + e)
        if cnt > 0:
            gg = tl.sum(ag0, 1) * g2
            uu = tl.sum(au0, 1) * u2
            tl.store(Hout + q0 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 1:
            gg = tl.sum(ag1, 1) * g2
            uu = tl.sum(au1, 1) * u2
            tl.store(Hout + q1 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 2:
            gg = tl.sum(ag2, 1) * g2
            uu = tl.sum(au2, 1) * u2
            tl.store(Hout + q2 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 3:
            gg = tl.sum(ag3, 1) * g2
            uu = tl.sum(au3, 1) * u2
            tl.store(Hout + q3 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 4:
            gg = tl.sum(ag4, 1) * g2
            uu = tl.sum(au4, 1) * u2
            tl.store(Hout + q4 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 5:
            gg = tl.sum(ag5, 1) * g2
            uu = tl.sum(au5, 1) * u2
            tl.store(Hout + q5 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 6:
            gg = tl.sum(ag6, 1) * g2
            uu = tl.sum(au6, 1) * u2
            tl.store(Hout + q6 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))
        if cnt > 7:
            gg = tl.sum(ag7, 1) * g2
            uu = tl.sum(au7, 1) * u2
            tl.store(Hout + q7 * shm + rn, (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16))

    @triton.jit
    def _moe_down_grp8(Hin, GE, GP, GN, DC, DS, D2, RW, P, I, H, shm, swe, swn, sse, ssn, spm,
                       BN: tl.constexpr, BK: tl.constexpr, ASM: tl.constexpr, MODE: tl.constexpr,
                       PDL: tl.constexpr = False):
        """`_moe_down_1row2` for up to 8 pairs of one expert, the tile read once (verify)."""
        _gdc(PDL)
        g = tl.program_id(0)
        pn = tl.program_id(1)
        cnt = tl.load(GN + g)
        if cnt == 0:
            return
        e = tl.load(GE + g)
        rn = pn * BN + tl.arange(0, BN)
        BKH: tl.constexpr = BK // 2
        rb = tl.arange(0, BKH)
        q0 = tl.load(GP + g * 8 + 0)
        q1 = tl.load(GP + g * 8 + 1)
        q2 = tl.load(GP + g * 8 + 2)
        q3 = tl.load(GP + g * 8 + 3)
        q4 = tl.load(GP + g * 8 + 4)
        q5 = tl.load(GP + g * 8 + 5)
        q6 = tl.load(GP + g * 8 + 6)
        q7 = tl.load(GP + g * 8 + 7)
        if MODE == 0:
            a0 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a0 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a1 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a1 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a2 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a2 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a3 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a3 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a4 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a4 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a5 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a5 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a6 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a6 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a7 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a7 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        srow = e * sse + rn * ssn
        for k0 in range(0, I, BK):
            lo_, hi_ = _nv_dec(tl.load(DC + e * swe + rn[:, None] * swn + (k0 // 2 + rb)[None, :]), ASM)
            sc = _nv_sc(DS, srow, k0, BKH, MODE)
            if cnt > 0:
                xe, xo = _x_pairs(Hin + q0 * shm, k0, BKH, ASM)
                a0 = _nv_rowacc(a0, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 1:
                xe, xo = _x_pairs(Hin + q1 * shm, k0, BKH, ASM)
                a1 = _nv_rowacc(a1, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 2:
                xe, xo = _x_pairs(Hin + q2 * shm, k0, BKH, ASM)
                a2 = _nv_rowacc(a2, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 3:
                xe, xo = _x_pairs(Hin + q3 * shm, k0, BKH, ASM)
                a3 = _nv_rowacc(a3, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 4:
                xe, xo = _x_pairs(Hin + q4 * shm, k0, BKH, ASM)
                a4 = _nv_rowacc(a4, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 5:
                xe, xo = _x_pairs(Hin + q5 * shm, k0, BKH, ASM)
                a5 = _nv_rowacc(a5, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 6:
                xe, xo = _x_pairs(Hin + q6 * shm, k0, BKH, ASM)
                a6 = _nv_rowacc(a6, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if cnt > 7:
                xe, xo = _x_pairs(Hin + q7 * shm, k0, BKH, ASM)
                a7 = _nv_rowacc(a7, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
        d2 = tl.load(D2 + e)
        if cnt > 0:
            rw = tl.load(RW + q0).to(tl.float32)
            tl.store(P + q0 * spm + rn, tl.sum(a0, 1) * d2 * rw)
        if cnt > 1:
            rw = tl.load(RW + q1).to(tl.float32)
            tl.store(P + q1 * spm + rn, tl.sum(a1, 1) * d2 * rw)
        if cnt > 2:
            rw = tl.load(RW + q2).to(tl.float32)
            tl.store(P + q2 * spm + rn, tl.sum(a2, 1) * d2 * rw)
        if cnt > 3:
            rw = tl.load(RW + q3).to(tl.float32)
            tl.store(P + q3 * spm + rn, tl.sum(a3, 1) * d2 * rw)
        if cnt > 4:
            rw = tl.load(RW + q4).to(tl.float32)
            tl.store(P + q4 * spm + rn, tl.sum(a4, 1) * d2 * rw)
        if cnt > 5:
            rw = tl.load(RW + q5).to(tl.float32)
            tl.store(P + q5 * spm + rn, tl.sum(a5, 1) * d2 * rw)
        if cnt > 6:
            rw = tl.load(RW + q6).to(tl.float32)
            tl.store(P + q6 * spm + rn, tl.sum(a6, 1) * d2 * rw)
        if cnt > 7:
            rw = tl.load(RW + q7).to(tl.float32)
            tl.store(P + q7 * spm + rn, tl.sum(a7, 1) * d2 * rw)

    @triton.jit
    def _nv_rows8(X, Wc, Ws, S2, Y, M, N, K, swn, ssn, sxm, sym, BN: tl.constexpr, BK: tl.constexpr,
                  ASM: tl.constexpr, MODE: tl.constexpr, RB: tl.constexpr, PDL: tl.constexpr = False,
                  PF: tl.constexpr = 0):
        """`_nv_gemv_1row` for up to RB (<= 8) rows a program: the codes decoded once a K step, each
        row's products, sums and scales in the one-row kernel's order (verify)."""
        pn = tl.program_id(0)
        r0 = tl.program_id(1) * RB
        rn = pn * BN + tl.arange(0, BN)
        BKH: tl.constexpr = BK // 2
        rb = tl.arange(0, BKH)
        if PDL:
            _pf_rows(Wc, rn * swn, K // 2, PF)
        _gdc(PDL)
        if MODE == 0:
            a0 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a0 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a1 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a1 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a2 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a2 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a3 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a3 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a4 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a4 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a5 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a5 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a6 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a6 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        if MODE == 0:
            a7 = tl.zeros((BN, BKH), dtype=tl.float32)
        else:
            a7 = tl.zeros((BN, BKH // 8), dtype=tl.float32)
        for k0 in range(0, K, BK):
            lo_, hi_ = _nv_dec(tl.load(Wc + rn[:, None] * swn + (k0 // 2 + rb)[None, :]), ASM)
            sc = _nv_sc(Ws, rn * ssn, k0, BKH, MODE)
            if RB > 0:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 0, M - 1) * sxm, k0, BKH, ASM)
                a0 = _nv_rowacc(a0, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 1:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 1, M - 1) * sxm, k0, BKH, ASM)
                a1 = _nv_rowacc(a1, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 2:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 2, M - 1) * sxm, k0, BKH, ASM)
                a2 = _nv_rowacc(a2, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 3:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 3, M - 1) * sxm, k0, BKH, ASM)
                a3 = _nv_rowacc(a3, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 4:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 4, M - 1) * sxm, k0, BKH, ASM)
                a4 = _nv_rowacc(a4, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 5:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 5, M - 1) * sxm, k0, BKH, ASM)
                a5 = _nv_rowacc(a5, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 6:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 6, M - 1) * sxm, k0, BKH, ASM)
                a6 = _nv_rowacc(a6, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
            if RB > 7:
                xe, xo = _x_pairs(X + tl.minimum(r0 + 7, M - 1) * sxm, k0, BKH, ASM)
                a7 = _nv_rowacc(a7, lo_, hi_, xe, xo, sc, BN, BKH, MODE)
        s2 = tl.load(S2 + rn)
        if RB > 0:
            tl.store(Y + (r0 + 0) * sym + rn, (tl.sum(a0, 1) * s2).to(tl.bfloat16), mask=r0 + 0 < M)
        if RB > 1:
            tl.store(Y + (r0 + 1) * sym + rn, (tl.sum(a1, 1) * s2).to(tl.bfloat16), mask=r0 + 1 < M)
        if RB > 2:
            tl.store(Y + (r0 + 2) * sym + rn, (tl.sum(a2, 1) * s2).to(tl.bfloat16), mask=r0 + 2 < M)
        if RB > 3:
            tl.store(Y + (r0 + 3) * sym + rn, (tl.sum(a3, 1) * s2).to(tl.bfloat16), mask=r0 + 3 < M)
        if RB > 4:
            tl.store(Y + (r0 + 4) * sym + rn, (tl.sum(a4, 1) * s2).to(tl.bfloat16), mask=r0 + 4 < M)
        if RB > 5:
            tl.store(Y + (r0 + 5) * sym + rn, (tl.sum(a5, 1) * s2).to(tl.bfloat16), mask=r0 + 5 < M)
        if RB > 6:
            tl.store(Y + (r0 + 6) * sym + rn, (tl.sum(a6, 1) * s2).to(tl.bfloat16), mask=r0 + 6 < M)
        if RB > 7:
            tl.store(Y + (r0 + 7) * sym + rn, (tl.sum(a7, 1) * s2).to(tl.bfloat16), mask=r0 + 7 < M)


#: which verify-row entry points take the multi-row kernels (bits: 1 head, 2 FP8, 4 NVFP4, 8 experts
#: grouped); the rest take the row loop. All are bit-equal to the one-row calls. Default: FP8 only.
#: The multi-row FP8 kernel reads a tile once for all rows; for the head and the NVFP4 q/o the extra
#: accumulators cost more than re-reading the tile from L1, and grouping experts pays only when rows
#: share experts.
ROWS8 = int(os.environ.get("KOLIBRI_ROWS8", "2"))
ROWS8_HEAD, ROWS8_FP8, ROWS8_NV, ROWS8_MOE = 1, 2, 4, 8


def expert_groups(ids: torch.Tensor, n_experts: int, rb: int = 8):
    """(GE, GP, GN) for the grouped expert kernels, on the device, no host sync: pairs (row * k +
    slot) sorted by expert; position i heads a group of GN[i] <= rb pairs of expert GE[i], listed
    in GP[i]; GN is 0 elsewhere. A grid of len(pairs) programs covers every group."""
    flat = ids.reshape(-1).long()
    P = flat.numel()
    se, order = torch.sort(flat, stable=True)
    idx = torch.arange(P, device=flat.device)
    start = torch.ones(P, dtype=torch.bool, device=flat.device)
    start[1:] = se[1:] != se[:-1]
    first = torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), 0).values
    pos = idx - first
    counts = torch.zeros(n_experts, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    cnt = torch.clamp(counts[se] - pos, max=rb)
    gn = torch.where(pos % rb == 0, cnt, torch.zeros_like(cnt)).to(torch.int32)
    pad = torch.cat([order, torch.zeros(rb, dtype=order.dtype, device=flat.device)])
    gp = pad[idx[:, None] + torch.arange(rb, device=flat.device)[None, :]]
    gp = torch.where(torch.arange(rb, device=flat.device)[None, :] < gn[:, None], gp, torch.zeros_like(gp))
    return se.to(torch.int32).contiguous(), gp.to(torch.int32).contiguous(), gn.contiguous()


def swiglu_rows(gu: torch.Tensor, F_: int) -> torch.Tensor:
    """`swiglu` on the GPU, the torch form elsewhere (row-wise identical either way)."""
    if gu.is_cuda and HAVE_TRITON:
        return swiglu(gu, F_)
    g = gu.float()
    return (torch.nn.functional.silu(g[:, :F_]) * g[:, F_:]).to(torch.bfloat16)


if HAVE_TRITON:

    @triton.jit
    def _pair_order(IDS, ORD, P, PB: tl.constexpr, PDL: tl.constexpr = False):
        """ORD[rank] = pair for the P (row, slot) pairs sorted by (expert, pair): one program, the
        rank of each pair counted against all the others (P <= PB, at most a few hundred)."""
        _gdc(PDL)
        i = tl.arange(0, PB)
        m = i < P
        e = tl.load(IDS + i, mask=m, other=0).to(tl.int64)
        key = tl.where(m, e * PB + i, 9223372036854775807)
        rank = tl.sum((key[None, :] < key[:, None]).to(tl.int32), axis=1)
        tl.store(ORD + rank, i.to(tl.int32), mask=m)


#: verify rows: run the routed experts' pairs in expert order (KOLIBRI_MOE_ORDER), same arithmetic
MOE_ORDER = os.environ.get("KOLIBRI_MOE_ORDER", "1") != "0"


def moe_experts_rows(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, G: "ExpertBank",
                     U: "ExpertBank", D: "ExpertBank", extra: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 [M, H], row i equal, bit for bit, to `moe_experts(x[i:i+1], ids[i:i+1], w[i:i+1], ...,
    extra[i:i+1])` (verify rows): a program per (row, slot) pair running the one-row body."""
    M, H = x.shape
    k = ids.shape[1]
    I = G.N
    if not (x.is_cuda and HAVE_TRITON and ONEROW and MOE2 and moe_1row2_fits(I, H)):
        return torch.cat([moe_experts(x[i:i + 1], ids[i:i + 1], w[i:i + 1], G, U, D,
                                      None if extra is None else extra[i:i + 1]) for i in range(M)])
    x = x.to(torch.bfloat16).contiguous()
    idf = ids.reshape(-1).contiguous()
    rw = w.float().reshape(-1).contiguous()
    (bn, bk, warps, stages, mode), (bnd, bkd, warpsd, stagesd, moded) = MOE_1ROW2
    h = torch.empty(M * k, I, dtype=torch.bfloat16, device=x.device)
    p = torch.empty(M * k, H, dtype=torch.float32, device=x.device)
    pdl = pdl_rows(M)
    if ROWS8 & ROWS8_MOE and M > 1:
        ge, gp, gn = expert_groups(idf, G.E)
        _moe_gate_up_grp8[(M * k, I // bn)](
            x, ge, gp, gn, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
            G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
            x.stride(0), k, BN=bn, BK=bk, ASM=NV_ASM, MODE=mode, PDL=pdl, num_warps=warps,
            num_stages=stages, launch_pdl=pdl)
        _moe_down_grp8[(M * k, H // bnd)](
            h, ge, gp, gn, D.codes, D.scale, D.scale_2, rw, p, I, H, h.stride(0), D.codes.stride(0),
            D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
            BN=bnd, BK=bkd, ASM=NV_ASM, MODE=moded, PDL=pdl, num_warps=warpsd, num_stages=stagesd,
            launch_pdl=pdl)
        y = torch.empty(M, H, dtype=torch.float32, device=x.device)
        ex = extra.contiguous() if extra is not None else p
        _moe_combine_kernel[(M, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), ex,
                                                      ex.stride(0) if extra is not None else 0,
                                                      BN=256, EXTRA=extra is not None, num_warps=4)
        return y
    P = M * k
    ordr = None
    if MOE_ORDER and M > 1 and P <= 1024:
        ordr = torch.empty(P, dtype=torch.int32, device=x.device)
        _pair_order[(1,)](idf, ordr, P, PB=triton.next_power_of_2(P), PDL=pdl, num_warps=4,
                          launch_pdl=pdl)
    _moe_gate_up_1row2[(M * k, I // bn)](
        x, idf, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
        G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
        BN=bn, BK=bk, ASM=NV_ASM, MODE=mode, PDL=pdl, PF=_lines(H // 2, PF_MOE) if pdl else 0,
        num_warps=warps, num_stages=stages, KTOP=k, sxm=x.stride(0), ORD=ordr,
        USE_ORD=ordr is not None, launch_pdl=pdl)
    _moe_down_1row2[(M * k, H // bnd)](
        h, idf, D.codes, D.scale, D.scale_2, rw, p, I, H, h.stride(0), D.codes.stride(0),
        D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
        BN=bnd, BK=bkd, ASM=NV_ASM, MODE=moded, PDL=pdl, PF=_lines(I // 2, PF_MOE) if pdl else 0,
        num_warps=warpsd, num_stages=stagesd, ORD=ordr, USE_ORD=ordr is not None, launch_pdl=pdl)
    y = torch.empty(M, H, dtype=torch.float32, device=x.device)
    ex = extra.contiguous() if extra is not None else p
    _moe_combine_kernel[(M, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), ex,
                                                  ex.stride(0) if extra is not None else 0,
                                                  BN=256, EXTRA=extra is not None, PDL=pdl,
                                                  num_warps=4, launch_pdl=pdl)
    return y


def swiglu(gu: torch.Tensor, F_: int) -> torch.Tensor:
    """gu bf16 [M, 2F] -> bf16 [M, F]: one launch for `.float()`, silu, the product and the cast."""
    M = gu.shape[0]
    a = torch.empty(M, F_, dtype=torch.bfloat16, device=gu.device)
    pdl = pdl_any(M)
    _swiglu_kernel[(M, triton.cdiv(F_, 512))](gu, a, F_, gu.stride(0), a.stride(0), BF=512, PDL=pdl,
                                              num_warps=4, launch_pdl=pdl)
    return a


# Launch shapes picked on the GB10 (sm_121) at the served shapes (385 experts, 2560/512):
# (BN, BK, warps, stages) for gate/up and for down. A prefill chunk takes the large shapes (with
# (64, 128, 4, 2) the two 64x64 accumulators spill registers). Rows up to SMALL_M (decode, verify)
# take the small shapes, so every such row runs the same program shape whatever the row count.
SMALL_M = 32
MOE_TILES = {"small": ((32, 128, 4, 3), (32, 32, 4, 1)), "large": ((128, 128, 8, 3), (128, 128, 4, 1))}


def moe_experts(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, G: ExpertBank, U: ExpertBank,
                D: ExpertBank, extra: torch.Tensor | None = None, parts: bool = False):
    """fp32 [M, H] = sum over slots j of w[:, j] * D_e(SiLU(G_e x) * U_e x), e = ids[:, j], summed in
    slot order. x bf16 [M, H]; ids int [M, k]; w fp32 [M, k]."""
    M, H = x.shape
    k = ids.shape[1]
    if x.device.type != "cuda" or not HAVE_TRITON:
        y = moe_experts_torch(x, ids, w, G, U, D)
        return y if extra is None else y + extra.float()
    I = G.N
    if M == 1 and ONEROW and HAVE_TRITON:
        return _moe_experts_1row(x, ids, w, G, U, D, extra, parts)
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


def _moe_experts_1row(x, ids, w, G, U, D, extra, parts=False):
    """`moe_experts` for one row: the experts straight from the router's ids (no plan), one-row
    kernels, the same combine."""
    H = x.shape[1]
    k = ids.shape[1]
    I = G.N
    x = x.to(torch.bfloat16).contiguous()
    ids = ids.reshape(-1).contiguous()
    rw = w.float().reshape(-1).contiguous()
    h = torch.empty(k, I, dtype=torch.bfloat16, device=x.device)
    p = torch.empty(k, H, dtype=torch.float32, device=x.device)
    if MOE2 and moe_1row2_fits(I, H):
        moe_1row2(x, ids, rw, G, U, D, h, p, ids_early=extra is not None)
    else:
        _moe_1row_ke3(x, ids, rw, G, U, D, h, p)
    if parts:
        return MoEParts(p, extra)
    y = torch.empty(1, H, dtype=torch.float32, device=x.device)
    ex = extra if extra is not None else p
    pdl = pdl_on(1)
    _moe_combine_kernel[(1, triton.cdiv(H, 256))](p, y, H, k, p.stride(0), y.stride(0), ex,
                                                  ex.stride(0) if extra is not None else 0,
                                                  BN=256, EXTRA=extra is not None, PDL=pdl,
                                                  num_warps=4, launch_pdl=pdl)
    return y


def moe_1row2_fits(I: int, H: int, cfg=None) -> bool:
    """The one-row kernels have no masks: every tile must lie inside the bank."""
    (bn, bk, *_), (bnd, bkd, *_) = cfg or MOE_1ROW2
    return I % bn == 0 and H % bk == 0 and H % bnd == 0 and I % bkd == 0


def moe_1row2(x, ids, rw, G, U, D, h, p, cfg=None, ids_early: bool = False):
    """The one-row gate/up and down into h [k, I] and p [k, H]. `ids_early`: a PDL kernel that
    waited after the router ran sits between the router and this call (the FP8 shared expert)."""
    k, I = h.shape
    H = p.shape[1]
    (bn, bk, warps, stages, mode), (bnd, bkd, warpsd, stagesd, moded) = cfg or MOE_1ROW2
    pdl = pdl_on(1)
    _moe_gate_up_1row2[(k, I // bn)](
        x, ids, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
        G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
        BN=bn, BK=bk, ASM=NV_ASM, MODE=mode, PDL=pdl, PF=_lines(H // 2, PF_MOE) if pdl else 0,
        IDS_EARLY=pdl and ids_early, num_warps=warps, num_stages=stages, launch_pdl=pdl)
    _moe_down_1row2[(k, H // bnd)](
        h, ids, D.codes, D.scale, D.scale_2, rw, p, I, H, h.stride(0), D.codes.stride(0),
        D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
        BN=bnd, BK=bkd, ASM=NV_ASM, MODE=moded, PDL=pdl, PF=_lines(I // 2, PF_MOE) if pdl else 0,
        num_warps=warpsd, num_stages=stagesd, launch_pdl=pdl)


def _moe_1row_ke3(x, ids, rw, G, U, D, h, p):
    k, I = h.shape
    H = p.shape[1]
    (bn, bk, warps, stages), (bnd, bkd, warpsd, stagesd) = MOE_1ROW
    _moe_gate_up_1row[(k, triton.cdiv(I, bn))](
        x, ids, G.codes, G.scale, G.scale_2, U.codes, U.scale, U.scale_2, h, H, I,
        G.codes.stride(0), G.codes.stride(1), G.scale.stride(0), G.scale.stride(1), h.stride(0),
        BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    _moe_down_1row[(k, triton.cdiv(H, bnd))](
        h, ids, D.codes, D.scale, D.scale_2, rw, p, I, H, h.stride(0), D.codes.stride(0),
        D.codes.stride(1), D.scale.stride(0), D.scale.stride(1), p.stride(0),
        BN=bnd, BK=bkd, num_warps=warpsd, num_stages=stagesd)


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
#: the one-row head GEMV with a lane accumulator (0: tools/head_gemv's kernel); (BN, BK, warps, stages)
HEAD2 = os.environ.get("KOLIBRI_HEAD2", "1") != "0"
HEAD_1ROW = (8, 512, 8, 3)       # picked on the GB10

if HAVE_TRITON:

    @triton.jit
    def _head_gemv2(X, W, S, Y, N, K, swn, BN: tl.constexpr, BK: tl.constexpr,
                    PDL: tl.constexpr = False, PF: tl.constexpr = 0, M=1, sxm=0, sym=0):
        """fp32 logits of one fp32 row against e4m3 codes with an fp32 scale per vocabulary row;
        the scale multiplies the finished sum, as `_head_gemv_fp8`. N % BN == 0, K % BK == 0.
        M > 1: the one-row body once per row (`logits_rows`)."""
        pn = tl.program_id(0)
        rn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pf_rows(W, rn * swn, K, PF)
        _gdc(PDL)
        for m in range(0, M):
            acc = tl.zeros((BN, BK), dtype=tl.float32)
            for k0 in range(0, K, BK):
                kk = k0 + tl.arange(0, BK)
                x = tl.load(X + m * sxm + kk)
                w = tl.load(W + rn[:, None] * swn + kk[None, :]).to(tl.float32)
                acc += w * x[None, :]
            tl.store(Y + m * sym + rn, tl.sum(acc, 1) * tl.load(S + rn))


class E4M3Head:
    """`lm_head` as e4m3 codes [V, K] + fp32 scale per row; fp32 logits."""

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor):
        self.w = codes.contiguous()
        self.s = scale.float().contiguous()
        self.N, self.K = codes.shape

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() * 4

    def logits_rows(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 [M, V], row i equal, bit for bit, to `logits(x[i:i+1])` (verify rows)."""
        x2 = x.reshape(-1, self.K)
        M = x2.shape[0]
        bn2, bk2, w2, st2 = HEAD_1ROW
        if not (x2.is_cuda and HAVE_TRITON and HEAD2 and self.N % bn2 == 0 and self.K % bk2 == 0):
            return torch.cat([self.logits(x2[i:i + 1]) for i in range(M)])
        xf = x2.float().contiguous()
        y = torch.empty(M, self.N, dtype=torch.float32, device=x2.device)
        pdl = pdl_rows(M)
        if ROWS8 & ROWS8_HEAD and M > 1:
            rb = min(8, M)
            _head_rows8[(self.N // bn2, triton.cdiv(M, rb))](xf, self.w, self.s, y, M, self.N, self.K,
                                                             self.w.stride(0), xf.stride(0), y.stride(0),
                                                             BN=bn2, BK=bk2, RB=rb, PDL=pdl,
                                                             PF=_lines(self.K, PF_HEAD) if pdl else 0,
                                                             num_warps=w2, num_stages=st2, launch_pdl=pdl)
            return y
        _head_gemv2[(self.N // bn2,)](xf, self.w, self.s, y, self.N, self.K, self.w.stride(0),
                                      BN=bn2, BK=bk2, PDL=pdl, PF=_lines(self.K, PF_HEAD) if pdl else 0,
                                      num_warps=w2, num_stages=st2, M=M,
                                      sxm=xf.stride(0), sym=y.stride(0), launch_pdl=pdl)
        return y

    def logits(self, x: torch.Tensor, rows: int = 16384) -> torch.Tensor:
        """x [M, K] (bf16 or fp32) -> fp32 [M, V]. One row on the GPU takes the fp32-activation
        GEMV of `tools/head_gemv.py`; more rows a chunked fp32 product (prefill's last row is the
        only one serving needs; the gate reads all rows)."""
        x2 = x.reshape(-1, self.K)
        M = x2.shape[0]
        bn2, bk2, w2, st2 = HEAD_1ROW
        if (x2.device.type == "cuda" and HAVE_TRITON and M == 1 and HEAD2 and self.N % bn2 == 0
                and self.K % bk2 == 0):
            xf = x2.float().contiguous()
            y = torch.empty(1, self.N, dtype=torch.float32, device=x2.device)
            pdl = pdl_on(1)
            _head_gemv2[(self.N // bn2,)](xf, self.w, self.s, y, self.N, self.K, self.w.stride(0),
                                          BN=bn2, BK=bk2, PDL=pdl, PF=_lines(self.K, PF_HEAD) if pdl else 0,
                                          num_warps=w2, num_stages=st2, launch_pdl=pdl)
            return y
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
    def _add_rms2_kernel(R, Y, W1, W2, RO, XO, H, eps, sr, sy, sro, sxo, BH: tl.constexpr,
                         PDL: tl.constexpr = False, NP: tl.constexpr = 0, EX=None,
                         EXTRA: tl.constexpr = False):
        """r' = r + rms(y) * w1 (fp32); x = rms(r') * w2 (stored in XO's dtype). NP > 0 (one row,
        y is the routed experts' combine, Y holding their NP fp32 partial rows [NP, H] (summed
        in slot order) and EX the shared expert's bf16 output, as `_moe_combine_kernel`."""
        _gdc(PDL)
        row = tl.program_id(0)
        c = tl.arange(0, BH)
        m = c < H
        if NP > 0:
            y = tl.zeros((BH,), dtype=tl.float32)
            for j in range(0, NP):
                y += tl.load(Y + j * sy + c, mask=m, other=0.0)
            if EXTRA:
                y = y + tl.load(EX + c, mask=m, other=0.0).to(tl.float32)
        else:
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
    def _rms_kernel(X, W, Y, H, eps, sx, sy, BH: tl.constexpr, PDL: tl.constexpr = False):
        _gdc(PDL)
        row = tl.program_id(0)
        c = tl.arange(0, BH)
        m = c < H
        x = tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
        tl.store(Y + row * sy + c, ((x * r) * w).to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _route_kernel(L, B, IDS, WTS, E, SHARED, sl, K: tl.constexpr, KO: tl.constexpr, BE: tl.constexpr,
                      PDL: tl.constexpr = False):
        """Top K of logits + bias (lower index on ties), weight sigmoid(logit); slot K = SHARED at 1.
        (A one-warp int64-key top-k gives the same picks and was not faster.)"""
        _gdc(PDL)
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
    def _router_logits_kernel(X, Wg, Y, E, K, sx, sw, sy, BE: tl.constexpr, BK: tl.constexpr,
                              PDL: tl.constexpr = False, PF: tl.constexpr = 0):
        """fp32 logits[row, e] = sum_k x[row, k] * gate[e, k]: bf16 operands widened to fp32 (exact
        products), fp32 accumulation; BE experts a program, so a one-row step reads the bf16 router
        once (2 MB) instead of an fp32 copy (4 MB)."""
        row = tl.program_id(1)
        pe = tl.program_id(0)
        re_ = pe * BE + tl.arange(0, BE)
        me = re_ < E
        if PDL:
            _pf_rows(Wg, tl.minimum(re_, E - 1) * sw, K, PF, 2)
        _gdc(PDL)
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
    pdl = pdl_any(M)
    _router_logits_kernel[(triton.cdiv(E, 2), M)](x, gate_bf16, y, E, K, x.stride(0), gate_bf16.stride(0),
                                                  y.stride(0), BE=2, BK=1024, PDL=pdl,
                                                  PF=_lines(K * 2, PF_ROUTER) if pdl else 0, num_warps=4,
                                                  launch_pdl=pdl)
    return y


class MoEParts:
    """A one-row MoE output not yet combined: the routed slots' fp32 rows p [k, H] and the shared
    expert's bf16 row (or None); `add_rms2` sums them as `_moe_combine_kernel` would."""
    __slots__ = ("p", "extra")

    def __init__(self, p, extra):
        self.p, self.extra = p, extra

    def combine(self) -> torch.Tensor:
        k, H = self.p.shape
        y = torch.empty(1, H, dtype=torch.float32, device=self.p.device)
        ex = self.extra if self.extra is not None else self.p
        _moe_combine_kernel[(1, triton.cdiv(H, 256))](self.p, y, H, k, self.p.stride(0), y.stride(0), ex,
                                                      ex.stride(0) if self.extra is not None else 0,
                                                      BN=256, EXTRA=self.extra is not None, num_warps=4)
        return y


def add_rms2(r: torch.Tensor, y, w1, w2, eps: float, out_dtype=torch.bfloat16):
    """(r + rms(y) * w1, rms(that) * w2): the post-norm residual add and the next pre-norm. `y` may
    be `MoEParts` (one decode row): the combine happens inside the same launch."""
    if isinstance(y, MoEParts):
        if not (FUSED and HAVE_TRITON and r.is_cuda):
            y = y.combine()
        else:
            M, H = r.shape
            pdl = pdl_on(M)
            ro = torch.empty(M, H, dtype=torch.float32, device=r.device)
            xo = torch.empty(M, H, dtype=out_dtype, device=r.device)
            ex = y.extra if y.extra is not None else y.p
            _add_rms2_kernel[(1,)](r, y.p, w1, w2, ro, xo, H, eps, r.stride(0), y.p.stride(0), ro.stride(0),
                                   xo.stride(0), BH=triton.next_power_of_2(H), PDL=pdl, NP=y.p.shape[0],
                                   EX=ex, EXTRA=y.extra is not None, num_warps=8, launch_pdl=pdl)
            return ro, xo
    if not (FUSED and HAVE_TRITON and r.is_cuda):
        r2 = r + rms(y, w1, eps, torch.float32)
        return r2, rms(r2, w2, eps, out_dtype)
    M, H = r.shape
    y = y.contiguous()
    ro = torch.empty(M, H, dtype=torch.float32, device=r.device)
    xo = torch.empty(M, H, dtype=out_dtype, device=r.device)
    pdl = pdl_any(M)
    _add_rms2_kernel[(M,)](r, y, w1, w2, ro, xo, H, eps, r.stride(0), y.stride(0), ro.stride(0), xo.stride(0),
                           BH=triton.next_power_of_2(H), PDL=pdl, num_warps=8, launch_pdl=pdl)
    return ro, xo


def rms_fused(x: torch.Tensor, w, eps: float, out_dtype=torch.bfloat16):
    if not (FUSED and HAVE_TRITON and x.is_cuda):
        return rms(x, w, eps, out_dtype)
    M, H = x.shape
    x = x.contiguous()
    y = torch.empty(M, H, dtype=out_dtype, device=x.device)
    pdl = pdl_any(M)
    _rms_kernel[(M,)](x, w, y, H, eps, x.stride(0), y.stride(0), BH=triton.next_power_of_2(H), PDL=pdl,
                      num_warps=8, launch_pdl=pdl)
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
    pdl = pdl_any(M)
    _route_kernel[(M,)](logits, bias, ids, w, E, shared_id if shared_id is not None else 0, logits.stride(0),
                        K=k, KO=ko, BE=triton.next_power_of_2(E), PDL=pdl, num_warps=4, launch_pdl=pdl)
    return ids, w
