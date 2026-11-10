"""The e4m3 KV cache's coding (SPD-150), shared by the families that offer it (Flint's own copy stays in
engine/models/flint/kernels.py).

A cached key or value row of one KV head is kept as D e4m3 codes (stored as uint8) and one fp32 scale, the
row's absolute maximum over 448 (e4m3's largest finite value). A row is coded on its own, so a verify row's
codes are the decode step's for the same bf16 row. Readers multiply a code by its row's scale in fp32 and
round to bf16 before the product with the query, which is the bf16 cache's arithmetic on those values.

  quantize_kv   [..., D] bf16 -> (codes uint8, scales fp32), torch (the CPU twin and the prompt paths)
  kv_dequant    codes and scales -> bf16, torch
  kv_quant2     a call's K and V rows, written in bf16 to a staging buffer, coded into the caches at their
                slots in one launch (Triton; the torch twin on the CPU)
  bytes_per_token
                the bytes a token of one layer: codes and one fp32 scale a KV head, for K and V
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

#: e4m3's largest finite value
FP8_MAX = 448.0


def bytes_per_token(kv_heads: int, head_dim: int, fp8: bool) -> int:
    """K and V of one layer a token: bf16, or e4m3 codes and one fp32 scale a KV head."""
    return 2 * (kv_heads * head_dim + 4 * kv_heads) if fp8 else 2 * kv_heads * head_dim * 2


def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., D] bf16 -> e4m3 codes as uint8 and one fp32 scale a vector (amax / 448)."""
    amax = x.float().abs().amax(-1).clamp_min(1e-12)
    s = amax / FP8_MAX
    codes = (x.float() / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8), s


def kv_dequant(codes: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """[..., n, D] uint8 e4m3 codes and [..., n] fp32 scales -> bf16 (the value a reader multiplies by)."""
    return (codes.view(torch.float8_e4m3fn).float() * scales[..., None]).to(dtype)


if HAVE_TRITON:

    @triton.jit
    def _kv_quant2(KS_, VS_, s_sh, s_sn, IDX, KD, VD, s_dh, s_dn, KSC, VSC, s_ch, SLOT, D: tl.constexpr):
        # row m of head h: KS_[h, IDX[m]] and VS_[h, IDX[m]] bf16 -> KD / VD[h, SLOT[m]] e4m3 codes (as uint8) and
        # KSC / VSC[h, SLOT[m]] their scales, amax / 448 in fp32 (`quantize_kv`'s arithmetic)
        m = tl.program_id(0)
        h = tl.program_id(1)
        od = tl.arange(0, D)
        i = tl.load(IDX + m).to(tl.int64)
        sl = tl.load(SLOT + m).to(tl.int64)
        x = tl.load(KS_ + h * s_sh + i * s_sn + od).to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(x), 0), 1e-12) / 448.0
        c = tl.minimum(tl.maximum(x / sc, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(KD + h * s_dh + sl * s_dn + od, c.to(tl.uint8, bitcast=True))
        tl.store(KSC + h * s_ch + sl, sc)
        y = tl.load(VS_ + h * s_sh + i * s_sn + od).to(tl.float32)
        sv = tl.maximum(tl.max(tl.abs(y), 0), 1e-12) / 448.0
        cv = tl.minimum(tl.maximum(y / sv, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(VD + h * s_dh + sl * s_dn + od, cv.to(tl.uint8, bitcast=True))
        tl.store(VSC + h * s_ch + sl, sv)


def kv_quant2(ksrc: torch.Tensor, vsrc: torch.Tensor, idx: torch.Tensor, kdst: torch.Tensor, ks: torch.Tensor,
              vdst: torch.Tensor, vs: torch.Tensor, slot: torch.Tensor) -> None:
    """Rows `idx` of ksrc / vsrc [Hk, R, D] bf16 (the staging buffers, one layout) into kdst / vdst [Hk, cap, D]
    uint8 (e4m3 codes, one layout) and their scales ks / vs [Hk, cap] fp32 at `slot` (int64 [M] on the device),
    one launch; no host sync, so a CUDA graph holds it."""
    M = idx.numel()
    HK, _, D = ksrc.shape
    assert kdst.dtype == vdst.dtype == torch.uint8 and ks.dtype == vs.dtype == torch.float32
    if not ksrc.is_cuda or not HAVE_TRITON:
        i, s = idx.long(), slot.long()
        for src, dst, sc in ((ksrc, kdst, ks), (vsrc, vdst, vs)):
            codes, scale = quantize_kv(src[:, i])
            dst[:, s] = codes
            sc[:, s] = scale
        return
    assert ksrc.stride() == vsrc.stride() and kdst.stride() == vdst.stride() and ks.stride() == vs.stride()
    assert ksrc.stride(2) == 1 and kdst.stride(2) == 1 and ks.stride(1) == 1
    _kv_quant2[(M, HK)](ksrc, vsrc, ksrc.stride(0), ksrc.stride(1), idx, kdst, vdst, kdst.stride(0), kdst.stride(1),
                        ks, vs, ks.stride(0), slot, D=D, num_warps=1)
