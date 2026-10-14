"""GGUF IQ2_XXS (and Q8_0) weights: the dequant twin from the format, and a row-invariant GEMM that
decodes the blocks on the fly.

IQ2_XXS (ggml type 16, 2.0625 bits a weight): a block of 256 weights in 66 bytes, an fp16 scale `d` and
eight groups of 32 weights, each group two u32 words:

  q0   four bytes, each an index into the 256-entry E8 lattice grid (`iq2xxs_grid`: 8 magnitudes of
       8, 25 or 43 an entry)
  q1   bits 0..27: four 7-bit sign indices (`ksigns_iq2xs`: the 7 bits and an eighth that makes the
       parity even); bits 28..31: the group's 4-bit scale s

  w = d * (0.5 + s) * 0.25 * grid[q0 byte l][j] * (sign bit j of ksigns[q1 >> 7 l] ? -1 : 1)

in fp32 in that order (llama.cpp's `dequantize_row_iq2_xxs` and gguf-py's `IQ2_XXS.dequantize_blocks`;
the grid and sign tables are ggml's, MIT). `dequant_iq2_xxs` is the twin; tests/test_iq2_kernels.py
holds it to gguf-py bit for bit. Q8_0 (type 8): 32 int8 with an fp16 scale, w = d * q.

`IQ2Linear` is the Linear interface (`shape`, `nbytes`, `matmul`) over a packed [N, K] tensor (rows of
K / 256 blocks, as stored): `iq2_gemm` decodes a [BN, 32] slice of weights per step and multiplies it
in fp32 (IEEE, no TF32) into 16-row tiles of x, K in order, so a row of an M-row call is the one-row
call's row bit for bit (docs/kernels.md, "Verify twins").
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:          # noqa: BLE001
    HAVE_TRITON = False

QK = 256
BLOCK_BYTES = 66
# ggml's iq2xxs_grid, each entry's 8 magnitudes as 2-bit codes of (8, 25, 43), 16 bits an entry (hex,
# little-endian u16s): the packing gguf-py uses
_GRID_HEX = (
    b"00000200050008000a00110014002000220028002a0041004400500058006100"
    b"6400800082008a00a20001010401100115014001840198010002020222028202"
    b"010404041004210424044004420448046004810484049004a404000502050805"
    b"200546056905800591050906100640068406a406000805080808140828084108"
    b"440850085208880804094009020a140a01100410101021104010601084109010"
    b"951000110811201150115a118011241245120014081420142514491480141815"
    b"6215001616160118041810184018811800190519a019511a002002200a204420"
    b"6120802082202921482100220222012404241024402456240025412564259026"
    b"082820289428442a014004401040184021402440404048405640604081408440"
    b"9040004120416141804185410142104248425642684200440844204480449944"
    b"124524450046014804481048404845480049584961498249454a904a00500850"
    b"1150195020508050885004514251a4519152905492540a550156545600581158"
    b"195864584059085a046010604060686000615561186260620064056410651265"
    b"84654268008002800a8041808280048118814081118201840484108415844084"
    b"608400854685948509864086608602880489118a0490109024904090a1901691"
    b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)


def grid_table() -> np.ndarray:
    """The IQ2_XXS grid: float32 [256, 8] of 8, 25 or 43."""
    raw = np.frombuffer(_GRID_HEX, dtype=np.uint8).reshape(-1, 2)
    nib = np.where(raw > 0x40, raw + 9, raw) & 0x0F
    b = (nib[:, 0] << 4) | nib[:, 1]                        # 512 bytes, 4 codes a byte, low bits first
    codes = (b[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)[None, :]) & 3
    return np.array([8, 25, 43], dtype=np.float32)[codes.reshape(256, 8)]


def ksigns_table() -> np.ndarray:
    """ksigns_iq2xs: the 7 sign bits and the parity bit (bit 7) that makes the count even, uint8 [128]."""
    i = np.arange(128, dtype=np.uint8)
    par = np.array([bin(int(v)).count("1") & 1 for v in i], dtype=np.uint8)
    return i | (par << 7)


GRID = grid_table()
KSIGNS = ksigns_table()


def split_blocks(raw: np.ndarray, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """A packed [N, K] IQ2_XXS tensor (uint8, rows of K / 256 blocks) -> (d fp16 [N, nb], qs int32 [N, nb, 16])."""
    nb = K // QK
    b = np.ascontiguousarray(raw).reshape(N, nb, BLOCK_BYTES)
    d = np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(N, nb)
    qs = np.ascontiguousarray(b[..., 2:]).view(np.int32).reshape(N, nb, 16)
    return d, qs


def dequant_iq2_xxs(raw: np.ndarray, N: int, K: int) -> torch.Tensor:
    """The twin: a packed [N, K] IQ2_XXS tensor -> fp32 [N, K] (torch, CPU)."""
    d, qs = split_blocks(raw, N, K)
    d = torch.from_numpy(d.astype(np.float32))                          # [N, nb]
    q = torch.from_numpy(qs.view(np.uint32).astype(np.int64)).view(N, -1, 8, 2)   # [N, nb, 8 groups, (q0, q1)]
    q0, q1 = q[..., 0], q[..., 1]
    scale = (q1 >> 28) & 0xF
    db = d[..., None] * (0.5 + scale.to(torch.float32))                 # fp32: d * (0.5 + s)
    db = db * 0.25
    idx = (q0[..., None] >> torch.tensor([0, 8, 16, 24])) & 0xFF        # [N, nb, 8, 4]
    g = torch.from_numpy(GRID)[idx]                                     # [N, nb, 8, 4, 8]
    si = (q1[..., None] >> torch.tensor([0, 7, 14, 21])) & 0x7F
    ks = torch.from_numpy(KSIGNS.astype(np.int64))[si]                  # [N, nb, 8, 4]
    bit = (ks[..., None] >> torch.arange(8)) & 1
    sign = torch.where(bit == 1, torch.tensor(-1.0), torch.tensor(1.0))
    w = (db[..., None, None] * g) * sign
    return w.reshape(N, K)


def dequant_q8_0(raw: np.ndarray, N: int, K: int) -> torch.Tensor:
    """A packed [N, K] Q8_0 tensor -> fp32 [N, K]: w = d * q per block of 32."""
    b = np.ascontiguousarray(raw).reshape(N, K // 32, 34)
    d = torch.from_numpy(np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(N, K // 32).astype(np.float32))
    q = torch.from_numpy(np.ascontiguousarray(b[..., 2:]).view(np.int8).astype(np.float32))
    return (d[..., None] * q).reshape(N, K)


if HAVE_TRITON:

    @triton.jit
    def _iq2_gemm(X, s_x, D, QS, G, KS, Y, s_y, M, N, NB: tl.constexpr, BN: tl.constexpr, F32: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * 16 + tl.arange(0, 16)
        cols = pn * BN + tl.arange(0, BN)
        cm = cols < N
        k32 = tl.arange(0, 32)
        grp = k32 // 8
        el = k32 % 8
        acc = tl.zeros([16, BN], dtype=tl.float32)
        for b in range(0, NB):
            d = tl.load(D + cols * NB + b, mask=cm, other=0.0).to(tl.float32)            # [BN]
            for j in tl.static_range(8):
                q0 = tl.load(QS + (cols * NB + b) * 16 + 2 * j, mask=cm, other=0)
                q1 = tl.load(QS + (cols * NB + b) * 16 + 2 * j + 1, mask=cm, other=0)
                s = ((q1 >> 28) & 0xF).to(tl.float32)
                db = (d * (0.5 + s)) * 0.25                                               # [BN]
                idx = (q0[:, None] >> (8 * grp[None, :])) & 0xFF                          # [BN, 32]
                g = tl.load(G + idx * 8 + el[None, :])
                si = (q1[:, None] >> (7 * grp[None, :])) & 0x7F
                ks = tl.load(KS + si)
                neg = ((ks >> el[None, :]) & 1) == 1
                w = db[:, None] * g
                w = tl.where(neg, -w, w)                                                  # [BN, 32]
                k = b * 256 + j * 32 + k32
                x = tl.load(X + rows[:, None] * s_x + k[None, :], mask=rows[:, None] < M, other=0.0).to(tl.float32)
                acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        out = acc if F32 else acc.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * s_y + cols[None, :], out, mask=(rows[:, None] < M) & cm[None, :])


class IQ2Linear:
    """A packed IQ2_XXS weight [N, K] on the device: d fp16 [N, K / 256], qs int32 [N, K / 256, 16]."""

    def __init__(self, raw: np.ndarray, N: int, K: int, device="cuda"):
        if K % QK:
            raise ValueError(f"IQ2_XXS needs K a multiple of {QK}, not {K}")
        d, qs = split_blocks(raw, N, K)
        self.N, self.K, self.shape = N, K, (N, K)
        self.d = torch.from_numpy(d).to(device)
        self.qs = torch.from_numpy(qs).to(device)
        self.nbytes = N * (K // QK) * BLOCK_BYTES
        self._g = torch.from_numpy(GRID.reshape(-1)).to(device)
        self._ks = torch.from_numpy(KSIGNS.astype(np.int32)).to(device)

    def dequant(self) -> torch.Tensor:
        raw = np.concatenate([self.d.cpu().numpy().view(np.uint8).reshape(self.N, -1, 2),
                              self.qs.cpu().numpy().view(np.uint8).reshape(self.N, -1, 64)], -1)
        return dequant_iq2_xxs(raw.reshape(-1), self.N, self.K)

    def matmul(self, x: torch.Tensor, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
        x2 = x.reshape(-1, self.K).contiguous()
        M = x2.shape[0]
        if out is None:
            out = torch.empty(M, self.N, dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        BN = 32
        _iq2_gemm[(triton.cdiv(M, 16), triton.cdiv(self.N, BN))](x2, x2.stride(0), self.d, self.qs, self._g, self._ks, out,
                                                                 out.stride(0), M, self.N, NB=self.K // QK, BN=BN,
                                                                 F32=f32, num_warps=4)
        return out.view(*x.shape[:-1], self.N)


def matmul_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The twin of `IQ2Linear.matmul`: x (as fp32) @ w^T (the dequantised fp32 weight), fp32."""
    return x.float() @ w.float().T
