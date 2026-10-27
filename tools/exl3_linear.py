"""EXL3 weights (exllamav3's trellis format) behind the engine's Linear interface.

A quantised projection in an EXL3 checkpoint (exllamav3 v1.6.0, the converter the
TandemLLM-Qwen3.8-27B-EXL3-* sets were made with) is, for in_features Kin and out_features N:

    <name>.trellis  int16 [Kin/16, N/16, 16 K]   K bits a weight, K an integer 2..8 per tensor
    <name>.suh      fp16  [Kin]                  input channel factor (sign x scale)
    <name>.svh      fp16  [N]                    output channel factor (sign x scale)
    <name>.mul1     the codebook's multiplier    0x83DCD12D: the "mul1" procedural codebook

and the weight it stands for (exllamav3 `LinearEXL3.get_weight_tensor`, in [Kin, N] orientation) is

    W = diag(suh) . H . Wi . H . diag(svh)        H = the 128-point Sylvester Hadamard / sqrt(128),
                                                     block-diagonal over 128-wide blocks
    y = x @ W = ( H( H(x * suh) @ Wi ) ) * svh

where Wi is the trellis decoded. A 16x16 tile of Wi is one cyclic bit string of 256 K bits. The
weight at position p (0..255) is the 16-bit window ending at bit (p + 1) K (wrapping round the
tile; the bit string is MSB-first inside each little-endian uint32 word), put through the codebook:

    v = window * 0x83DCD12D (mod 2^32);  h = fp16 bits (0x6400 + the sum of v's four bytes)
    w = fma(h, fp16(0x1eee), fp16(0xc931))        one fp16 rounding

and position p = 8 l + j belongs to (row k, column n) of the tile with
    k = 2 (l % 4) + (j & 1) + 8 ((j >> 1) & 1),   n = l / 4 + 8 (j >> 2)
which is exactly the tensor core's B-operand fragment of `mma.m16n8k16` for lane l: positions
8l..8l+3 are the operand registers of the tile's first eight columns, 8l+4..8l+7 of the second.
So a lane decodes its own operand registers in place; nothing is shuffled.

The kernels here (ENG-199) are this engine's own, written for the GB10 (sm_121) and the verify
shapes M = 1..32; the bit extraction and the codebook decode are ported from exllamav3's
`exl3_dq.cuh` / `codebook.cuh` (MIT, Copyright (c) 2025 Turboderp; the notice is kept with the
CUDA source, tools/csrc/exl3_linear/exl3.cu). Arithmetic, per call of `exl3_matmul`:

  1. `rot_in`: xh = fp16( H(fp32(x) * fp32(suh)) )           one warp per row and 128-block
  2. `gemm`:   y' = xh @ Wi in fp32 on the tensor cores (fp16 x fp16 -> fp32), the weight decoded
               from the trellis into the B registers
  3. epilogue: y = fp32( H(y') ) * svh -> bf16 (or fp32 for the vocabulary head)

Determinism (the losslessness gate's row independence). Every row is computed by the same
instruction sequence whatever M is: rows 1..15 of an M = 1 call are zeros in the same mma, and
17..32 a second mma. A CTA owns 128 output columns (one Hadamard block) and a contiguous K range;
its warps split that range into contiguous slices, each summed in K order by the mma's own
accumulator, and the slices are added in warp order through shared memory; a cluster of S CTAs
splits K further and rank 0 adds the ranks' partials in rank order through distributed shared
memory, then applies the output rotation. (S, WK) are a function of the weight's shape only
(`pick`), never of M, so a row's bits do not depend on the block it is verified in. No atomics.

Prefill (M > 32): the trellis is decoded to an fp16 Wi slab by a reconstruct kernel, the product
runs on the library GEMM, the rotations as above (ENG-203).

Built on first use from tools/csrc/exl3_linear/ (exl3.cu, the bindings exl3.cpp) with
`torch.utils.cpp_extension.load_inline` for sm_121a, cached under ~/.cache/torch_extensions by source hash
(engine/native_build.py).
"""

from __future__ import annotations

import math
import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import native_build as nb  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch  # noqa: E402

MUL1 = 0x83DCD12D
KS = (2, 3, 4, 5, 6, 7, 8)               # every integer rate the recipes use (head 4/5/6/8)
# ENG-285: the half-integer rates of exllamav3 1.6.0's mul1 packs below 4 bpw (1.5, 2.5, 3.5), as the
# kernels name them: ten times the rate (a tile 16 K + 8 int16 wide)
HALF_KS = (15, 25, 35)


def kb_of(width: int) -> int:
    """The kernels' bits of a trellis whose tiles are `width` int16 wide: K for 16 K, ten times the rate
    for a half-integer one (16 K + 8 -> 10 K + 5)."""
    return width // 16 if width % 16 == 0 else 10 * (width // 16) + 5


def rate(kb: int) -> float:
    """Bits a weight of the kernels' bits `kb`."""
    return kb / 10 if kb >= 10 else float(kb)


def tile_width(kb: int) -> int:
    """int16 a 256-weight tile at the kernels' bits `kb`."""
    return 16 * (kb // 10) + 8 if kb >= 10 else 16 * kb
SKINNY_MAX = 32                           # rows the decode-in-registers kernel takes in one call
# rows up to which a call runs the skinny kernel in 32-row chunks instead of decoding a slab
# (one weight read per chunk against a 2-byte write and read of the slab per weight)
CHUNK_UNTIL = int(_S.get("EXL3_CHUNK_UNTIL"))
# 33..CHUNK_UNTIL rows in one launch whose 32-row chunks are adjacent CTAs (ENG-213); 0 = one launch a chunk
ROWGRID = _S.get("EXL3_ROWGRID") == "1"
# cp.async stages of the skinny GEMM's trellis pipeline (a warp's stages in flight); the
# summation order does not depend on it
STAGES = int(_S.get("EXL3_STAGES"))
#: SPD-245: the skinny GEMM launched with programmatic stream serialization: its CTAs start in the previous
#: kernel's tail and fetch their first trellis stages, then wait for it (griddepcontrol). No arithmetic moves.
PDL = _S.get("EXL3_PDL") == "1"
SLAB_N = 16384                            # columns a prefill slab decodes at once (fp16 Kin x SLAB_N)

_CUDA = nb.source(__file__, "exl3.cu")

_CPP = nb.source(__file__, "exl3.cpp")

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        src = f"#define EXL3_ST {STAGES}\n#define EXL3_PDL {int(PDL)}\n" + _CUDA
        tag = nb.tag(_CPP, src)
        _MOD = nb.load(f"qwen38_exl3_{tag}", cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["exl3_gemm", "exl3_rot_in", "exl3_rot_out", "exl3_recon",
                                  "exl3_rot_in_silu", "exl3_mlp_bridge"],
                       extra_cuda_cflags=["-O3", "-lineinfo",
                                          "-gencode=arch=compute_121a,code=sm_121a"],
                       verbose=False)
    return _MOD


# ------------------------------------------------------------------------------------- tiles
# (N, Kin, K bits) -> {"S": cluster split of K, "wk": warps splitting a CTA's K}. Both set the
# summation order, so they are keyed by the weight's shape and rate only; the column tiling
# ("cfg" for 1..16 rows, "cfg2" for 17..32) does not touch the order and may differ by row count.
_CONFIG: dict[tuple[int, int, int], dict] = {}
# QWEN38_EXL3_TILES unset: ops/exl3-tiles-fused.json, the fused group shapes' column tiling at the
# default order (bit-identical to default_tile, faster; 2026-10-09 A/B). "none" = default_tile only.
_TILES = _S.get("EXL3_TILES") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              "ops", "exl3-tiles-fused.json")
if _TILES != "none" and os.path.isfile(_TILES):
    import json as _json
    for _k, _v in _json.load(open(_TILES)).items():
        if _k.startswith("_"):                  # provenance
            continue
        _n, _kk, _kb = (int(v) for v in _k.split("x"))
        _CONFIG[(_n, _kk, _kb)] = {"S": int(_v["S"]), "wk": int(_v["wk"]),
                                   "cfg": int(_v.get("cfg", 0)), "cfg2": int(_v.get("cfg2", 2))}


def default_tile(N: int, Kin: int, kb: int) -> dict:
    """The order-defining split for a shape no table names. One CTA a 128-column block is the
    best measured split for every wide shape (2026-10-08 sweep: the cluster reduction costs more
    than the 40 CTAs of a 5120-wide output lose); only narrow outputs (k/v, 8 blocks) split K
    over a cluster, keeping at least eight K tiles a warp."""
    cbs = N // 128
    kt = Kin // 16
    S = 1
    while cbs * S < 32 and S < 8 and kt // (S * 2 * 8) >= 8:
        S *= 2
    wk = 8 if kt // (S * 8) >= 8 else 4
    return {"S": S, "wk": wk, "cfg": 0, "cfg2": 2}


def pick(N: int, Kin: int, kb: int) -> dict:
    return _CONFIG.get((N, Kin, kb)) or default_tile(N, Kin, kb)


def set_config(N: int, Kin: int, kb: int, cfg: dict) -> None:
    _CONFIG[(N, Kin, kb)] = dict(cfg)


# --------------------------------------------------------------------------------- the weight
class EXL3Block:
    """One EXL3 projection (or a fused group of them sharing the input and the rate).

    Engine orientation: `shape` is (N, Kin) like every Linear here; the trellis is exllamav3's
    [Kin/16, N/16, 16 K]. For a group, the members' trellises are concatenated along N tiles, their
    suh stacked [G, Kin] (each member rotates the input with its own factors), svh concatenated;
    `sizes` are the members' N, `bounds` the 128-column block where members 2 and 3 start.
    `out_fp32`: the vocabulary head returns fp32 logits, as the e4m3 head does.
    """

    __slots__ = ("t", "suh", "svh", "kb", "N", "K", "sizes", "names", "bounds", "out_fp32", "_bf16",
                 "tile_n")

    def __init__(self, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, *,
                 sizes: list[int] | None = None, names: list[str] | None = None,
                 out_fp32: bool = False):
        assert trellis.dtype == torch.int16 and trellis.dim() == 3, (trellis.dtype, trellis.shape)
        kb = kb_of(trellis.shape[2])
        assert trellis.shape[2] == tile_width(kb) and kb in KS + HALF_KS, \
            f"trellis {tuple(trellis.shape)}: K must be 2..8, 1.5, 2.5 or 3.5"
        # dense tiles; a group member is a view (its trellis rows are the group's, ldt apart)
        self.t = trellis if (trellis.stride(2) == 1 and trellis.stride(1) == trellis.shape[2]) \
            else trellis.contiguous()
        self.kb = kb
        self.K = trellis.shape[0] * 16
        self.N = trellis.shape[1] * 16
        assert self.K % 128 == 0 and self.N % 128 == 0, (self.K, self.N)
        suh = suh.to(torch.float16)
        self.suh = (suh if suh.dim() == 2 else suh.view(1, -1)).contiguous()
        self.svh = svh.to(torch.float16).contiguous().view(-1)
        assert self.suh.shape[1] == self.K and self.svh.numel() == self.N
        self.sizes = list(sizes) if sizes else None
        self.names = list(names) if names else None
        G = self.suh.shape[0]
        assert G <= 3 and (G == 1 or (self.sizes and len(self.sizes) == G)), "a group is 2..3 members"
        if self.sizes:
            assert sum(self.sizes) == self.N and all(s % 128 == 0 for s in self.sizes), self.sizes
            starts = [sum(self.sizes[:i]) // 128 for i in range(1, len(self.sizes))]
            starts += [1 << 30] * (2 - len(starts))
            self.bounds = tuple(starts[:2])
        else:
            self.bounds = (1 << 30, 1 << 30)
        self.out_fp32 = out_fp32
        self._bf16 = None
        # the N the tile table is read at: a group member reads its group's entry, so a member
        # used on its own sums K in the order the group's launch does (one order a weight)
        self.tile_n = self.N

    @classmethod
    def group(cls, blocks: list["EXL3Block"], names: list[str]) -> "EXL3Block":
        """One launch for projections that read the same activation at the same rate. Nothing is
        duplicated in the end: the members are dropped by the caller (`Weights.fuse_exl3_groups`)."""
        assert len({b.kb for b in blocks}) == 1 and len({b.K for b in blocks}) == 1
        t = torch.cat([b.t for b in blocks], dim=1)
        suh = torch.cat([b.suh for b in blocks], dim=0)
        svh = torch.cat([b.svh for b in blocks], dim=0)
        return cls(t, suh, svh, sizes=[b.N for b in blocks], names=names)

    def member(self, i: int) -> "EXL3Block":
        """Member i of a group as its own EXL3Block, a view into the group's tensors."""
        off = sum(self.sizes[:i])
        n = self.sizes[i]
        m = EXL3Block(self.t[:, off // 16:(off + n) // 16], self.suh[i], self.svh[off:off + n],
                      out_fp32=self.out_fp32)
        m.tile_n = self.N
        return m

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.t.numel() * 2 + self.suh.numel() * 2 + self.svh.numel() * 2

    # byte accounting reads a head as numel * element_size (FP8Head does the same)
    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1

    @property
    def device(self):
        return self.t.device

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """The Linear interface: `x[..., K] @ W^T` -> [..., N] (bf16; fp32 for the head)."""
        return exl3_matmul(x.reshape(-1, x.shape[-1]), self).view(*x.shape[:-1], self.N)

    def inner(self, n0: int = 0, n1: int | None = None) -> torch.Tensor:
        """Wi[:, n0:n1] in fp16 (the rotated basis), decoded on the device."""
        n1 = self.N if n1 is None else n1
        wi = torch.empty(self.K, n1 - n0, dtype=torch.float16, device=self.t.device)
        if self.t.is_cuda:
            _module().exl3_recon(self.t, wi, self.kb, n0)
        else:
            wi.copy_(decode_trellis(self.t, n0 // 16, n1 // 16))
        return wi

    def dequant(self) -> torch.Tensor:
        """The weight in the engine's [N, K] orientation, bf16, from fp32 rotations (tests, the
        drafter's NVFP4 head copy). Never on the decode path."""
        out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.t.device)
        H = hadamard128(self.t.device)
        for g, n0, n1 in segments(self):
            wi = self.inner(n0, n1).float()                                 # [K, n]
            w = (H @ wi.view(-1, 128, n1 - n0)).view(self.K, n1 - n0)      # H . Wi
            w = w * self.suh[g].float().view(-1, 1)
            w = (w.view(self.K, -1, 128) @ H).view(self.K, n1 - n0)         # . H
            w = w * self.svh[n0:n1].float().view(1, -1)
            out[n0:n1] = w.T.to(torch.bfloat16)
        return out

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            self._bf16 = self.dequant()
        return self._bf16


def segments(w: EXL3Block, step: int = 4096):
    """(member, n0, n1) column ranges of at most `step`, never across a member boundary."""
    starts = [0]
    for sz in (w.sizes or [w.N]):
        starts.append(starts[-1] + sz)
    for g in range(len(starts) - 1):
        for n0 in range(starts[g], starts[g + 1], step):
            yield g, n0, min(starts[g + 1], n0 + step)


def member_of(w: EXL3Block, col: int) -> int:
    if not w.sizes:
        return 0
    acc = 0
    for i, s in enumerate(w.sizes):
        acc += s
        if col < acc:
            return i
    raise IndexError(col)


_HAD: dict = {}


def hadamard128(device) -> torch.Tensor:
    """H / sqrt(128), H[i, j] = (-1)^popcount(i & j): Sylvester's construction, exllamav3's had_128."""
    key = str(device)
    if key not in _HAD:
        i = torch.arange(128)
        bits = (i.view(-1, 1) & i.view(1, -1))
        pc = torch.zeros_like(bits)
        for b in range(7):
            pc += (bits >> b) & 1
        _HAD[key] = ((1 - 2 * (pc & 1)).float() / math.sqrt(128)).to(device)
    return _HAD[key]


# ------------------------------------------------------------------- the torch reference decode
def _tile_perm() -> torch.Tensor:
    """position p (0..255) -> k * 16 + n inside a 16x16 tile (exllamav3 `tensor_core_perm`)."""
    p = torch.empty(256, dtype=torch.long)
    for lane in range(32):
        r0 = (lane % 4) * 2
        c0 = lane // 4
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for j in range(8):
            r = rows[j & 3]
            c = c0 + 8 * (j >> 2)
            p[lane * 8 + j] = r * 16 + c
    return p


def decode_trellis(t: torch.Tensor, nt0: int = 0, nt1: int | None = None) -> torch.Tensor:
    """Pure-torch decode of trellis tiles [KT, nt0:nt1] -> Wi fp16 [Kin, 16 (nt1 - nt0)].
    Bit for bit what the kernels decode (the gate checks it); slow, for tests and the CPU path."""
    KT, NT, W = t.shape
    kb = kb_of(W)
    nt1 = NT if nt1 is None else nt1
    tt = t[:, nt0:nt1].contiguous().view(torch.int32).long() & 0xFFFFFFFF   # [KT, n, W / 2] words
    n = nt1 - nt0
    words = tt.view(KT * n, W // 2)
    nbits = 16 * W
    # the bit string, MSB-first per word: bit b of the string is bit (31 - b % 32) of word b // 32
    shifts = 31 - torch.arange(32, device=t.device)
    bitstr = ((words.unsqueeze(-1) >> shifts) & 1).view(KT * n, nbits)            # [tiles, 256 kb]
    p = torch.arange(256, device=t.device)
    # exclusive end; a half-integer rate's odd weights take one bit more (exl3_linear dq8_frac)
    end = (p + 1) * kb if kb < 10 else (p + 1) * (kb // 10) + (p + 1) // 2
    idx = (end.view(-1, 1) - 16 + torch.arange(16, device=t.device).view(1, -1)) % nbits   # [256, 16]
    win = bitstr[:, idx]                                                             # [tiles, 256, 16]
    weights = (1 << (15 - torch.arange(16, device=t.device)))
    v = (win * weights).sum(-1)                                                      # 16-bit windows
    x = (v * MUL1) & 0xFFFFFFFF
    s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    h = (1024 + s).float()                                                           # fp16 0x6400 + s
    k_inv = torch.tensor(0x1eee, dtype=torch.int16).view(torch.float16).float()
    k_bias = torch.tensor(0xc931 - 0x10000, dtype=torch.int16).view(torch.float16).float()
    val = (h * k_inv + k_bias).to(torch.float16)                                     # exact in fp32
    perm = _tile_perm().to(t.device)
    tile = torch.empty(KT * n, 256, dtype=torch.float16, device=t.device)
    tile[:, perm] = val
    tile = tile.view(KT, n, 16, 16).permute(0, 2, 1, 3).reshape(KT * 16, n * 16)
    return tile


def reference_matmul(x: torch.Tensor, w: EXL3Block) -> torch.Tensor:
    """The product the kernels mean, in fp32 except for the two roundings the kernels make on
    purpose (the rotated input to fp16, the output to bf16/fp32): the correctness gate's yardstick."""
    H = hadamard128(x.device)
    M = x.shape[0]
    outs = []
    for g, n0, n1 in segments(w):
        xs = x.float() * w.suh[g].float()
        xh = (xs.view(M, -1, 128) @ H).view(M, w.K).to(torch.float16).float()
        yp = xh @ w.inner(n0, n1).float()
        y = (yp.view(M, -1, 128) @ H).view(M, n1 - n0) * w.svh[n0:n1].float()
        outs.append(y)
    y = torch.cat(outs, 1)
    return y if w.out_fp32 else y.to(torch.bfloat16)


# ---------------------------------------------------------------------------------- the product
def rotate_in(x: torch.Tensor, w: EXL3Block) -> torch.Tensor:
    G = w.suh.shape[0]
    xh = torch.empty(G, x.shape[0], w.K, dtype=torch.float16, device=x.device)
    _module().exl3_rot_in(x, w.suh, xh)
    return xh


# The MLP's silu(gate) * up fused into the down projection's input rotation (`exl3_down_silu`):
# the same bits as the two torch kernels and rot_in (tests/gpu/test_exl3_silu_gpu.py), two launches
# fewer a layer. 0 = the torch path.
SILU_FUSED = _S.get("EXL3_SILU_FUSED") == "1"
# A prefill MLP's gate|up products straight into the down projection's rotated input
# (`exl3_mlp_slab`): the same bits, the bf16 gate|up and the activation never written. 0 = off.
MLP_BRIDGE = _S.get("EXL3_MLP_BRIDGE") == "1"


def exl3_down_silu(y: torch.Tensor, w: EXL3Block) -> torch.Tensor:
    """down(silu(gate) * up) for y = [gate | up] (bf16 [M, 2 Kin]), w the down projection."""
    assert y.dtype == torch.bfloat16 and y.dim() == 2 and y.shape[1] == 2 * w.K and w.suh.shape[0] == 1
    if y.stride(1) != 1 or y.stride(0) % 4 or y.data_ptr() % 8:
        y = y.contiguous()
    xh = torch.empty(1, y.shape[0], w.K, dtype=torch.float16, device=y.device)
    _module().exl3_rot_in_silu(y, w.suh, xh)
    return exl3_matmul(None, w, xh=xh, M=y.shape[0], dev=y.device)


def exl3_mlp_slab(x: torch.Tensor, gu: EXL3Block, wd: EXL3Block) -> torch.Tensor | None:
    """A prefill MLP (rows past CHUNK_UNTIL) whose gate+up products go straight from the library
    GEMM's fp32 into the down projection's rotated input (`mlp_bridge_kernel`): the bits of
    rot_out -> silu * up -> rot_in without writing the bf16 gate|up and the activation. None when
    the shapes do not pair (the caller takes the usual path)."""
    F = wd.K
    if not (gu.sizes and len(gu.sizes) == 2 and gu.sizes[0] == gu.sizes[1] == F
            and wd.suh.shape[0] == 1 and x.shape[0] > CHUNK_UNTIL):
        return None
    if _mm_out_fp32() is False:
        return None
    mod = _module()
    M = x.shape[0]
    xh = rotate_in(x, gu)                                    # [2, M, K]
    segs = _slabs(gu) or [(g, n0, n1, None) for g, n0, n1 in segments(gu, SLAB_N)]
    gs = [t for t in segs if t[0] == 0]
    us = [t for t in segs if t[0] == 1]
    if len(gs) != len(us):
        return None
    xd = torch.empty(1, M, F, dtype=torch.float16, device=x.device)
    for (_, a0, a1, wg), (_, b0, b1, wu) in zip(gs, us):
        if b0 - F != a0 or b1 - a1 != b0 - a0:
            return None
        yg = _mm_fp32(xh[0], wg if wg is not None else gu.inner(a0, a1))
        yu = _mm_fp32(xh[1], wu if wu is not None else gu.inner(b0, b1))
        mod.exl3_mlp_bridge(yg, yu, gu.svh[a0:a1], gu.svh[b0:b1], wd.suh[0, a0:a1].contiguous(),
                            xd[0, :, a0:a1])
        del yg, yu
    return exl3_matmul(None, wd, xh=xd, M=M, dev=x.device)


def exl3_matmul(x: torch.Tensor | None, w: EXL3Block, *, out: torch.Tensor | None = None,
                cfg: int | None = None, S: int | None = None, wk: int | None = None,
                xh: torch.Tensor | None = None, M: int | None = None, dev=None) -> torch.Tensor:
    """Y[M, N] = x[M, K] @ W^T for an EXL3Block (or group). x bf16. `xh`: the input already
    rotated ([G, M, K] fp16, what `rotate_in(x, w)` returns) in place of x."""
    if xh is None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == w.K, (x.dtype, x.shape, w.shape)
        M = x.shape[0]
        if not x.is_cuda:
            return reference_matmul(x, w)
        if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
            x = x.contiguous()
            if x.data_ptr() % 8:
                x = x.clone()
        dev = x.device
    odt = torch.float32 if w.out_fp32 else torch.bfloat16
    if out is None:
        out = torch.empty(M, w.N, dtype=odt, device=dev)
    mod = _module()
    # the skinny GEMM decodes integer rates; a half-integer one (an LFM2.5 pack's, ENG-285) takes the
    # slab at any row count (its rows <= 32 run on tools/fn_kernels.dense)
    if M > CHUNK_UNTIL or w.kb >= 10:
        return _slab_matmul(x, w, out, xh=xh)
    tile = pick(w.tile_n, w.K, w.kb)
    S_ = tile["S"] if S is None else S
    wk_ = tile["wk"] if wk is None else wk
    if M > SKINNY_MAX and ROWGRID:
        # ENG-213 row grid: one rotation and ONE launch for every 32-row chunk, the chunks of a
        # column block adjacent CTAs (the second reads the trellis from L2); bit for bit the
        # chunked launches below
        c = cfg if cfg is not None else tile["cfg2"]
        mod.exl3_gemm(rotate_in(x, w) if xh is None else xh, w.t, w.svh, out, w.kb, c, wk_, S_,
                      w.bounds[0], w.bounds[1], (M + SKINNY_MAX - 1) // SKINNY_MAX)
        return out
    for r0 in range(0, M, SKINNY_MAX):
        r1 = min(M, r0 + SKINNY_MAX)
        if xh is None:
            xs = rotate_in(x[r0:r1], w)
        else:
            xs = xh[:, r0:r1].contiguous()
        c = cfg if cfg is not None else (tile["cfg"] if r1 - r0 <= 16 else tile["cfg2"])
        mod.exl3_gemm(xs, w.t, w.svh, out[r0:r1], w.kb, c, wk_, S_, w.bounds[0], w.bounds[1], 1)
    return out


# The layer-major prefill's slab cache (engine/model.py `prefill_window`): while it is on, a
# weight's decoded fp16 slabs are kept from its first prefill call to `slab_cache_clear`, so the
# chunks of a window that pass one layer's weights decode each slab once instead of once a chunk.
# The slab is the same fp16 tensor either way and every GEMM keeps its shape: bit for bit the
# uncached calls. Holds one layer's projections (~0.8 GB at 4.5 bpw) between clears.
_SLAB_CACHE: dict | None = None


def slab_cache_on() -> None:
    global _SLAB_CACHE
    _SLAB_CACHE = {}


def slab_cache_clear() -> None:
    if _SLAB_CACHE is not None:
        _SLAB_CACHE.clear()


def slab_cache_off() -> None:
    global _SLAB_CACHE
    _SLAB_CACHE = None


def _slabs(w: EXL3Block) -> list:
    """(member, n0, n1, Wi[:, n0:n1]) for every slab of `w`; cached while the cache is on."""
    c = _SLAB_CACHE
    if c is None:
        return None
    got = c.get(id(w))
    if got is None or got[0] is not w:
        got = c[id(w)] = (w, [(g, n0, n1, w.inner(n0, n1)) for g, n0, n1 in segments(w, SLAB_N)])
    return got[1]


def _slab_matmul(x: torch.Tensor, w: EXL3Block, out: torch.Tensor,
                 xh: torch.Tensor | None = None) -> torch.Tensor:
    """Prefill: decode a column slab of Wi to fp16, the library GEMM, the output rotation."""
    mod = _module()
    if xh is None:
        xh = rotate_in(x, w)                                # [G, M, K]
    cached = _slabs(w)
    if cached is not None:
        for g, n0, n1, wi in cached:
            yp = _mm_fp32(xh[g], wi)
            mod.exl3_rot_out(yp, w.svh[n0:n1], out[:, n0:n1])
            del yp
        return out
    for g, n0, n1 in segments(w, SLAB_N):
        wi = w.inner(n0, n1)
        yp = _mm_fp32(xh[g], wi)
        mod.exl3_rot_out(yp, w.svh[n0:n1], out[:, n0:n1])
        del wi, yp
    return out


_MM_OUT_DTYPE: bool | None = None


def _mm_out_fp32() -> bool | None:
    """Whether the library GEMM keeps the fp32 product (None before the first prefill asked)."""
    if _MM_OUT_DTYPE is None and torch.cuda.is_available():
        a = torch.zeros(16, 16, dtype=torch.float16, device="cuda")
        _mm_fp32(a, a)
    return _MM_OUT_DTYPE


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp16 x fp16 on the library GEMM, the product kept in fp32 when torch can (`out_dtype`),
    else rounded to fp16 as exllamav3's own prefill GEMM rounds it."""
    global _MM_OUT_DTYPE
    if _MM_OUT_DTYPE is None:
        try:
            torch.mm(a[:1, :16], b[:16, :16], out_dtype=torch.float32)
            _MM_OUT_DTYPE = True
        except Exception:  # noqa: BLE001
            _MM_OUT_DTYPE = False
    return torch.mm(a, b, out_dtype=torch.float32) if _MM_OUT_DTYPE else torch.mm(a, b)


# ------------------------------------------------------------------------------- reading files
def read_block(f, base: str, *, out_fp32: bool = False) -> EXL3Block:
    """One EXL3 linear from an open safetensors handle: `base.trellis`, `base.suh`/`base.svh` (or the
    packed sign fields `su`/`sv`), the mul1 codebook only."""
    keys = set(f.keys())
    t = f.get_tensor(f"{base}.trellis")
    if f"{base}.mcg" in keys:
        raise ValueError(f"{base}: the mcg codebook is not supported (mul1 only)")
    if f"{base}.mul1" in keys:
        m = f.get_tensor(f"{base}.mul1").view(torch.int32).item() & 0xFFFFFFFF
        if m != MUL1:
            raise ValueError(f"{base}: mul1 multiplier {m:#x} != {MUL1:#x}")
    else:
        raise ValueError(f"{base}: no mul1 tensor (3inst codebook is not supported)")
    suh = f.get_tensor(f"{base}.suh") if f"{base}.suh" in keys else unpack_signs(f.get_tensor(f"{base}.su"))
    svh = f.get_tensor(f"{base}.svh") if f"{base}.svh" in keys else unpack_signs(f.get_tensor(f"{base}.sv"))
    if f"{base}.bias" in keys:
        raise ValueError(f"{base}: a bias is not supported")
    return EXL3Block(t, suh, svh, out_fp32=out_fp32)


def unpack_signs(bits: torch.Tensor) -> torch.Tensor:
    """exllamav3 `unpack_bf`: int16 bit fields -> +-1 fp16 (bit set = -1), LSB first."""
    b = bits.view(torch.int16).to(torch.int32) & 0xFFFF
    masks = 1 << torch.arange(16, device=b.device)
    e = ((b.unsqueeze(-1) & masks) > 0).flatten()
    return (1.0 - e.to(torch.float16) * 2.0).contiguous()


# ----------------------------------------------------------------------------- gate and bench
def check_block(w: EXL3Block, rows=(1, 2, 4, 8, 16, 24, 32), seed: int = 0, scale: float = 1.0,
                x: torch.Tensor | None = None) -> dict:
    """Kernel against the torch reference on `w`, and row independence, every row count."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if x is None:
        x = (torch.randn(max(rows), w.K, device="cuda", generator=g) * scale).to(torch.bfloat16)
    ref = reference_matmul(x, w).float()
    worst, single = 0.0, None
    res = {}
    for M in rows:
        y = exl3_matmul(x[:M], w).float()
        d = (y - ref[:M]).abs().max().item()
        rel = d / max(ref[:M].abs().max().item(), 1e-30)
        worst = max(worst, rel)
        if single is None:
            single = y[:1].clone()
        indep = (y[:1] - single).abs().max().item()
        res[M] = {"max_abs": d, "rel": rel, "row0_vs_M1": indep}
    res["worst_rel"] = worst
    return res


def bench(w: EXL3Block, rows=(1, 8, 16, 32), reps: int = 20, cfg=None, S=None, wk=None) -> dict:
    """Warm (L2-resident or not, depending on the size) timing of one weight, CUDA events."""
    out = {}
    for M in rows:
        x = torch.randn(M, w.K, device="cuda").to(torch.bfloat16)
        for _ in range(3):
            exl3_matmul(x, w, cfg=cfg, S=S, wk=wk)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            exl3_matmul(x, w, cfg=cfg, S=S, wk=wk)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1000 / reps
        out[M] = {"us": us, "GBps": w.nbytes / us / 1e3}
    return out


def random_block(N: int, Kin: int, kb: int, device="cuda", seed: int = 0) -> EXL3Block:
    """Random trellis bits (every bit string is a valid code) and plausible channel factors."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randint(-32768, 32767, (Kin // 16, N // 16, tile_width(kb)), generator=g, dtype=torch.int32)
    t = t.to(torch.int16)
    suh = (torch.rand(Kin, generator=g) * 0.02 + 0.005) * (torch.randint(0, 2, (Kin,), generator=g) * 2 - 1)
    svh = (torch.rand(N, generator=g) * 0.5 + 0.5) * (torch.randint(0, 2, (N,), generator=g) * 2 - 1)
    return EXL3Block(t.to(device), suh.half().to(device), svh.half().to(device))
