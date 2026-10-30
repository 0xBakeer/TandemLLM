"""NVIDIA's Flash-Next checkpoints (`nvidia/Qwen3.8-Flash-Next-FP4-Local`) in the forms the reference multiplies by.

The checkpoint is a plain Hugging Face safetensors model (21 shards, `quantization_config.quant_method: modelopt`,
`MIXED_PRECISION`), weights as NVIDIA ships them:

  routed experts        NVFP4: `weight` U8 [N, K/2] (two E2M1 codes a byte, low nibble first), block scales e4m3 per 16
                        values, one fp32 `weight_scale_2`. Trunk scales are CSF-compressed (`weight_scale.nvfp4_csf_fixed`
                        + `.nvfp4_csf_exceptions`), the MTP layer's are stored plain (`weight_scale`).
  GDN / attention /     MXFP8: `weight` F8_E4M3 [N, K], `weight_scale` U8 [N, K/32] (E8M0: 2^(s - 127)).
  indexer / shared
  mixers, router, head, bf16 (or fp32 `hc_norm`), as stored.
  PLE, MTP non-experts
  n-gram table          NVFP4 rows: 128 shards `ngram_embedding.shard_k.weight` U8 [R, 80] + `.weight_scale` e4m3 [R, 10]
                        and one `ngram_embedding.weight_scale_2`; row r of the table is row r % R of shard r // R.

Everything here is lossless: CSF scales decode to exactly the stored e4m3 bytes, MXFP8 and NVFP4 widen exactly to fp32
before the cast to the compute dtype. `load_reference` feeds `ref.FlashNextRef`; `load_engine` feeds `engine.FlashNextEngine`
the same weights in its forms (fused launches, a stacked expert bank, the n-gram table as packed rows), every weight
object answering the engine's `mul_rows(x)` protocol.

CSF reader after vllm-project/vllm#60646 (Apache-2.0); codec `byte-window4-fixed-stream-u24-exceptions/1`.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

PREFIX = "model.language_model."
FIXED, EXC = ".nvfp4_csf_fixed", ".nvfp4_csf_exceptions"
_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8,
       "U32": np.uint32, "I64": np.int64, "I32": np.int32, "U16": np.uint16}
_TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
          "U8": torch.uint8, "U32": torch.uint32, "I64": torch.int64, "I32": torch.int32, "U16": torch.uint16}
_ITEM = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "U32": 4, "I64": 8, "I32": 4, "U16": 2}
#: E2M1 (FP4) code -> value; bit 3 is the sign
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def is_checkpoint(path: str) -> bool:
    """An NVIDIA ModelOpt Flash-Next checkpoint (HF safetensors + `quantization_config.quant_method: modelopt`), not an
    EXL3 pack."""
    try:
        c = json.load(open(os.path.join(path, "config.json")))
    except (OSError, ValueError):
        return False
    return (c.get("quantization_config") or {}).get("quant_method") == "modelopt" and \
        os.path.isfile(os.path.join(path, "model.safetensors.index.json"))


# ------------------------------------------------------------------------------ the index
@dataclass
class TensorInfo:
    name: str
    file: str
    offset: int
    nbytes: int
    dtype: str
    shape: tuple


class HFIndex:
    """Every tensor of a sharded safetensors model by name, from the shard headers (no tensor data read)."""

    def __init__(self, path: str):
        self.path = path
        wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.tensors: dict[str, TensorInfo] = {}
        for rel in sorted(set(wm.values())):
            f = os.path.join(path, rel)
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                head = json.loads(fh.read(n))
            base = 8 + n
            for name, h in head.items():
                if name == "__metadata__":
                    continue
                lo, hi = h["data_offsets"]
                self.tensors[name] = TensorInfo(name, f, base + lo, hi - lo, h["dtype"], tuple(h["shape"]))

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def numpy(self, name: str) -> np.ndarray:
        """A read-only memory map of the tensor's bytes, in its storage dtype."""
        t = self.tensors[name]
        a = np.memmap(t.file, dtype=_NP[t.dtype], mode="r", offset=t.offset, shape=t.shape if t.shape else (1,))
        return a if t.shape else a.reshape(())

    def array(self, name: str) -> np.ndarray:
        """The tensor's bytes read into memory (no map kept open: thousands of small tensors stay within the
        process's file limit)."""
        t = self.tensors[name]
        a = np.fromfile(t.file, dtype=_NP[t.dtype], count=t.nbytes // _ITEM[t.dtype], offset=t.offset)
        return a.reshape(t.shape) if t.shape else a.reshape(())

    def read(self, name: str, device="cpu") -> torch.Tensor:
        t = self.tensors[name]
        raw = np.fromfile(t.file, dtype=np.uint8, count=t.nbytes, offset=t.offset)
        return torch.from_numpy(raw).view(_TORCH[t.dtype]).reshape(t.shape).to(device)


# ------------------------------------------------------------------------------ formats
def decode_csf_fixed(fixed: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    """CSF `fixed` scales -> the plain e4m3 bytes [rows, cols]: per 16-row slab one base byte a row, then 4-bit offsets
    (low nibble first); each uint32 exception = 24-bit flat position | replacement byte << 24. Bases + offsets wrap
    mod 256 (the encoder's byte arithmetic)."""
    f = np.asarray(fixed)
    rows = f.shape[0] * 16
    cols = (f.shape[1] // 16 - 1) * 2
    bases = f[:, :16].reshape(rows, 1).astype(np.int16)
    packed = f[:, 16:].reshape(rows, cols // 2)
    out = (bases + np.stack((packed & 0xF, packed >> 4), -1).reshape(rows, cols)).reshape(-1)
    w = np.asarray(exceptions).view(np.uint32).astype(np.int64).reshape(-1)
    out[w & 0xFFFFFF] = w >> 24
    return out.astype(np.uint8).reshape(rows, cols)


_LUT: dict = {}


def _e2m1(device) -> torch.Tensor:
    """The E2M1 table on `device`, made once (a graph may hold the dequant: no copy inside it)."""
    k = str(device)
    if k not in _LUT:
        _LUT[k] = E2M1.to(device)
    return _LUT[k]


def nvfp4_dequant(w: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    """NVFP4 -> fp32 [..., N, K]: E2M1 codes [..., N, K/2] (low nibble = even column) x e4m3 block scale per 16 x
    fp32 global (a scalar, or one a leading index broadcast over [N, K])."""
    codes = torch.stack((w & 0xF, w >> 4), -1).reshape(*w.shape[:-1], -1).long()
    v = _e2m1(w.device)[codes]                                             # [..., N, K]
    s = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    s2 = scale2.float()
    s = s * (s2.reshape(()) if s2.numel() == 1 else s2.reshape(*s2.shape, *([1] * (s.dim() - s2.dim()))))
    return v * s[..., : v.shape[-1]]


def mxfp8_bf16(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP8 -> bf16 [N, K] without an fp32 copy: e4m3 -> bf16 is exact, and so is the product by 2^(E8M0 - 127) for
    every normal result (a power of two moves the exponent only)."""
    N, K = w.shape
    s = torch.exp2(scale.to(torch.int32).to(torch.float32) - 127).to(torch.bfloat16)
    return (w.view(torch.float8_e4m3fn).to(torch.bfloat16).view(N, K // 32, 32) * s[:, :K // 32, None]).view(N, K)


def mxfp8_dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP8 -> fp32 [N, K]: e4m3 values x 2^(E8M0 - 127) per 32 values."""
    s = torch.exp2(scale.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    return w.view(torch.float8_e4m3fn).float() * s[:, : w.shape[1]]


class NVWeight:
    """One quantised projection kept as stored on `device`; `dequant()` widens it exactly (fp32 math, then the
    compute dtype), `bf16_cached()` keeps that once (the reference's EXL3Block contract)."""

    def __init__(self, kind: str, parts: dict[str, torch.Tensor], cache: bool = True):
        self.kind, self.parts, self._cache, self._bf16 = kind, parts, cache, None
        self.shape = (parts["w"].shape[0], parts["w"].shape[1] * (2 if kind == "nvfp4" else 1))

    def dequant(self, dtype=torch.bfloat16) -> torch.Tensor:
        p = self.parts
        if self.kind == "nvfp4":
            return nvfp4_dequant(p["w"], p["s"], p["s2"]).to(dtype)
        return mxfp8_dequant(p["w"], p["s"]).to(dtype)

    def bf16_cached(self) -> torch.Tensor:
        if self._bf16 is None:
            w = self.dequant()
            if not self._cache:
                return w
            self._bf16 = w
        return self._bf16

    def mul_rows(self, x: torch.Tensor) -> torch.Tensor:
        """x [M, K] bf16 -> [M, N] bf16 (the engine's weight protocol). MXFP8 up to 32 rows: tools/mxfp8_skinny (the
        weight decoded in registers, row-invariant); more rows: the exact bf16 weight on the library GEMM."""
        if self.kind == "mxfp8" and x.shape[0] <= 32:
            from tools import mxfp8_skinny as S
            return S.mm(x, self.parts["w"], self.parts["s"])
        if self.kind == "mxfp8" and x.dtype == torch.bfloat16:
            return torch.nn.functional.linear(x, mxfp8_bf16(self.parts["w"], self.parts["s"]))
        return torch.nn.functional.linear(x, self.dequant(x.dtype))

    def rows(self, a: int, b: int) -> "NVWeight":
        """Output rows [a, b) as a weight of their own (views: a fused launch's member)."""
        p = self.parts
        return NVWeight(self.kind, {"w": p["w"][a:b], "s": p["s"][a:b], **({"s2": p["s2"]} if "s2" in p else {})},
                        self._cache)

    @staticmethod
    def cat(ws: list["NVWeight"]) -> "NVWeight":
        """MXFP8 weights of one K stacked by output rows: exact (the scales are a row's own)."""
        assert all(w.kind == "mxfp8" for w in ws)
        return NVWeight("mxfp8", {"w": torch.cat([w.parts["w"] for w in ws]).contiguous(),
                                  "s": torch.cat([w.parts["s"] for w in ws]).contiguous()})


def projection(idx: HFIndex, base: str, device, cache: bool = True):
    """The weight at `base` (no `.weight` suffix) in the form the reference takes: a tensor for bf16/fp32, an NVWeight
    for NVFP4 (CSF or plain scales) and MXFP8."""
    w = idx.tensors[base + ".weight"]
    if w.dtype in ("BF16", "F32", "F16"):
        return idx.read(base + ".weight", device)
    if w.dtype == "F8_E4M3":                                               # MXFP8
        return NVWeight("mxfp8", {"w": idx.read(base + ".weight", device).view(torch.uint8),
                                  "s": idx.read(base + ".weight_scale", device)}, cache)
    if w.dtype == "U8":                                                    # NVFP4
        if base + ".weight_scale" + FIXED in idx:
            s = decode_csf_fixed(idx.numpy(base + ".weight_scale" + FIXED), idx.numpy(base + ".weight_scale" + EXC))
            s = torch.from_numpy(s).to(device)
        else:
            s = idx.read(base + ".weight_scale", device).view(torch.uint8)
        return NVWeight("nvfp4", {"w": idx.read(base + ".weight", device), "s": s,
                                  "s2": idx.read(base + ".weight_scale_2", device)}, cache)
    raise ValueError(f"{base}: weight dtype {w.dtype} not handled")


# ------------------------------------------------------------------------------ the n-gram table
class NVFP4NGramTable:
    """The PLE n-gram table as stored: 128 NVFP4 shards, read row by row from the files (nothing resident).
    `gather(ids [T, heads])` -> [T, heads, D] in `dtype`."""

    def __init__(self, idx: HFIndex, prefix: str):
        self.idx, self.prefix = idx, prefix
        k = 0
        self.w, self.s = [], []
        while f"{prefix}shard_{k}.weight" in idx:
            self.w.append(idx.numpy(f"{prefix}shard_{k}.weight"))
            self.s.append(idx.numpy(f"{prefix}shard_{k}.weight_scale"))
            k += 1
        if not k:
            raise ValueError(f"{prefix}: no n-gram table shards")
        self.rows = self.w[0].shape[0]
        self.dim = self.w[0].shape[1] * 2
        self.s2 = float(idx.numpy(prefix + "weight_scale_2").reshape(-1)[0])
        self.total = sum(a.shape[0] for a in self.w)

    def gather(self, ids: torch.Tensor, dtype=torch.bfloat16, device=None) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        sh, row = flat // self.rows, flat % self.rows
        w = np.stack([self.w[a][b] for a, b in zip(sh, row)])
        s = np.stack([self.s[a][b] for a, b in zip(sh, row)])
        v = nvfp4_dequant(torch.from_numpy(w), torch.from_numpy(s), torch.tensor(self.s2))
        return v.reshape(*ids.shape, self.dim).to(dtype).to(device if device is not None else "cpu")


class NVFP4RowTable(NVFP4NGramTable):
    """The table in the engine's row interface: `rows_of(ids)` -> packed rows int16 [n, words] on the host (the 80 code
    bytes, then the 10 scale bytes, of each row), `decode_dev(packed, heads)` -> fp32 [n, D] on the device (fixed-shape
    torch: a graph can hold it). No head bias: `offsets_np` is one head."""

    def __init__(self, idx: HFIndex, prefix: str):
        super().__init__(idx, prefix)
        self.cb, self.sb = self.w[0].shape[1], self.s[0].shape[1]
        self.words = (self.cb + self.sb + 1) // 2
        self.offsets_np = np.zeros(1, dtype=np.int64)
        self._s2: dict = {}

    def rows_of(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        out = np.zeros((ids.size, self.words * 2), dtype=np.uint8)
        sh, row = ids // self.rows, ids % self.rows
        for k in np.unique(sh):
            m = sh == k
            out[m, :self.cb] = self.w[k][row[m]]
            out[m, self.cb:self.cb + self.sb] = self.s[k][row[m]]
        return out.view(np.int16)

    def decode_dev(self, packed: torch.Tensor, heads: torch.Tensor | None = None) -> torch.Tensor:
        dev = packed.device
        if str(dev) not in self._s2:
            self._s2[str(dev)] = torch.tensor(self.s2, dtype=torch.float32, device=dev)
        b = packed.reshape(packed.shape[0], -1).view(torch.uint8)
        return nvfp4_dequant(b[:, :self.cb], b[:, self.cb:self.cb + self.sb], self._s2[str(dev)])


class BF16Rows:
    """A bf16 weight the engine multiplies through `mul_rows` (fp32 out): tools/bf16_skinny on the tensor cores up to
    32 rows (row-invariant: a decode row and a verify block's row are the same bits), fp32 products above."""

    def __init__(self, w: torch.Tensor):
        self.w = w.contiguous()
        self.shape = self.w.shape

    def mul_rows(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 32:
            from tools import bf16_skinny as BS
            return BS.mm(x, self.w)
        # a prompt chunk: the library GEMM in bf16 (fp32 accumulate, the sums rounded once; prompt rows agree with
        # the row step on the next token, not bit for bit -- engine.forward_pre)
        return torch.nn.functional.linear(x, self.w).float()

    def dequant(self, dtype=torch.bfloat16) -> torch.Tensor:
        return self.w.to(dtype)


class BF16Lin(BF16Rows):
    """A bf16 projection read through `lin` (the MTP layer's attention, input fusion and shared expert): `BF16Rows`
    with the sums rounded once to bf16, as `F.linear` in bf16 returns them."""

    def mul_rows(self, x: torch.Tensor) -> torch.Tensor:
        return super().mul_rows(x).to(torch.bfloat16)


# ------------------------------------------------------------------------------ the expert bank
class NVBank:
    """One layer's routed experts stacked as stored (NVFP4: gate|up [E, 2I, H/2], down [E, H, I/2], plain e4m3 block
    scales, one fp32 global a projection) and its shared expert (MXFP8 in the trunk, bf16 in the MTP layer) as expert E:
    the engine's bank interface (`fn_kernels.moe_experts` and `fn_prefill_kernels.moe_grouped` hand a bank with
    `experts` to it). Up to tools/nvfp4_moe.PMAX pairs: the grouped NVFP4 kernels (tools/nvfp4_moe.py); more (a prompt
    chunk): `experts_torch`, fp32 products of the exact weights with the kernels' roundings (gate/up to bf16,
    a = bf16(bf16(silu(g)) u), down fp32), PAIRS pairs a launch. The shared expert: its gate|up as one weight."""

    PAIRS = 16

    def __init__(self, gu_w, gu_s, gu_s2, d_w, d_s, d_s2, shared: dict):
        self.gu_w, self.gu_s, self.gu_s2 = gu_w, gu_s, gu_s2          # [E, 2I, H/2], [E, 2I, H/16], [E, 2]
        self.d_w, self.d_s, self.d_s2 = d_w, d_s, d_s2                # [E, H, I/2], [E, H, I/16], [E]
        self.shared = shared                                          # {"gate", "up", "down"}: NVWeight or bf16
        g, u = shared["gate"], shared["up"]
        self.shared["gu"] = NVWeight.cat([g, u]) if isinstance(g, NVWeight) else BF16Lin(torch.cat([g.w, u.w]))
        self.n = gu_w.shape[0]
        self.I = gu_w.shape[1] // 2
        self.H = d_w.shape[1]

    def _sh(self, x: torch.Tensor) -> torch.Tensor:
        """The shared expert of the rows: fp32 [M, H] (gate|up one launch, the activation, down)."""
        from tools import nvfp4_moe as NM

        def lin(w, v):
            return w.mul_rows(v) if hasattr(w, "mul_rows") else torch.nn.functional.linear(v, w)
        return lin(self.shared["down"], NM.silu_mul(lin(self.shared["gu"], x)))

    #: one side stream for every bank: the shared expert's launches beside the router's and the routed experts'
    _side = None

    def begin(self, x: torch.Tensor) -> None:
        """The shared expert of the rows, started on a side stream before the router runs: it needs only x, and its
        ~5 MB stream through memory while the router, the plan and the routed experts' first launches (small,
        latency-bound) leave the bandwidth idle. `experts` joins it. A graph captures both streams."""
        if NVBank._side is None:
            NVBank._side = torch.cuda.Stream(device=x.device)
        cur, side = torch.cuda.current_stream(x.device), NVBank._side
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            self._ysh = self._sh(x)
        self._ysh_x = x

    def experts(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """x [M, H] bf16, ids int32 [M, S] (the shared expert as id E, the last slot) -> yd [M S, H] fp32,
        unweighted."""
        from tools import nvfp4_moe as NM
        M, S = ids.shape
        yd = NM.experts(x, ids, self)
        ysh = None
        if getattr(self, "_ysh_x", None) is x:
            torch.cuda.current_stream(x.device).wait_stream(NVBank._side)
            # no record_stream: the next layer's `begin` makes the side stream wait for this one (the copy included)
            # before it allocates again
            ysh, self._ysh, self._ysh_x = self._ysh, None, None
        yd.view(M, S, self.H)[:, S - 1].copy_(ysh if ysh is not None else self._sh(x))
        return yd

    def experts_torch(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """`experts` in torch, any number of pairs."""
        M, S = ids.shape
        flat = ids.reshape(-1).long()
        rows = torch.arange(M * S, device=x.device) // S
        yd = torch.empty(M * S, self.H, dtype=torch.float32, device=x.device)
        I = self.I
        for p0 in range(0, M * S, self.PAIRS):
            e = flat[p0:p0 + self.PAIRS].clamp(max=self.n - 1)                 # the shared slot: any routed id, replaced below
            xs = x.index_select(0, rows[p0:p0 + self.PAIRS]).float().unsqueeze(-1)   # [P, H, 1]
            gu = nvfp4_dequant(self.gu_w[e], self.gu_s[e], torch.ones(1, device=x.device))   # [P, 2I, H]
            gu = (gu @ xs).squeeze(-1)                                         # [P, 2I]
            gu = gu * self.gu_s2[e].repeat_interleave(I, dim=1)
            g, u = gu[:, :I].to(torch.bfloat16).float(), gu[:, I:].to(torch.bfloat16).float()
            a = (g / (1.0 + torch.exp(-g))).to(torch.bfloat16).float() * u
            a = a.to(torch.bfloat16).float().unsqueeze(-1)                     # [P, I, 1]
            dn = nvfp4_dequant(self.d_w[e], self.d_s[e], self.d_s2[e])         # [P, H, I]
            yd[p0:p0 + self.PAIRS] = (dn @ a).squeeze(-1)
        sh = flat == self.n
        ysh = self._sh(x)                                                      # [M, H]
        return torch.where(sh[:, None], ysh.index_select(0, rows), yd)


def load_bank(idx: HFIndex, p: str, cfg, device) -> NVBank:
    """The NVBank of the MLP at prefix `p` (`...layers.L.mlp.`): CSF scales decoded once on the host."""
    E = cfg.num_experts

    def scales(b):
        if b + ".weight_scale" + FIXED in idx:
            return decode_csf_fixed(idx.array(b + ".weight_scale" + FIXED), idx.array(b + ".weight_scale" + EXC))
        return idx.array(b + ".weight_scale")

    gu_w, gu_s, gu_s2, d_w, d_s, d_s2 = [], [], [], [], [], []
    for e in range(E):
        b = f"{p}experts.{e}."
        gu_w.append(np.concatenate([idx.array(b + "gate_proj.weight"), idx.array(b + "up_proj.weight")]))
        gu_s.append(np.concatenate([scales(b + "gate_proj"), scales(b + "up_proj")]))
        gu_s2.append([float(idx.array(b + "gate_proj.weight_scale_2").reshape(-1)[0]),
                      float(idx.array(b + "up_proj.weight_scale_2").reshape(-1)[0])])
        d_w.append(idx.array(b + "down_proj.weight"))
        d_s.append(scales(b + "down_proj"))
        d_s2.append(float(idx.array(b + "down_proj.weight_scale_2").reshape(-1)[0]))

    def dev(a, dt=None):
        t = torch.from_numpy(np.ascontiguousarray(np.stack(a) if isinstance(a, list) else a))
        return (t if dt is None else t.to(dt)).to(device)
    shared = {x: projection(idx, f"{p}shared_expert.{x}_proj", device) for x in ("gate", "up", "down")}
    shared = {x: BF16Lin(w) if isinstance(w, torch.Tensor) else w for x, w in shared.items()}
    return NVBank(dev(gu_w), dev(gu_s), dev(np.asarray(gu_s2, dtype=np.float32)), dev(d_w), dev(d_s),
                  dev(np.asarray(d_s2, dtype=np.float32)), shared)


# ------------------------------------------------------------------------------ the reference's weights
def canonical(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def load_reference(path: str, device="cuda", mtp: bool = False, log=print, layers=None):
    """(cfg, W, tables) for `ref.FlashNextRef` from an NVIDIA FP4-Local checkpoint: every projection as stored on
    `device` (NVWeight for NVFP4 / MXFP8), routed experts as `ref.ExpertBank`s of NVWeights, the n-gram table read
    from the files. `layers`: only these trunk layers (a debug model)."""
    from engine.models.flashnext.config import load_flashnext_config
    from engine.models.flashnext.ref import ExpertBank
    cfg = load_flashnext_config(path)
    idx = HFIndex(path)
    lay = set(range(cfg.num_hidden_layers) if layers is None else layers)
    W: dict = {}
    bases = set()
    for n in idx.tensors:
        if n.startswith("model.visual.") or (n.startswith("mtp.") and not mtp) or "ngram_embedding." in n:
            continue
        c = canonical(n)
        if c.startswith("layers.") and int(c.split(".")[1]) not in lay:
            continue
        if ".experts." in c:
            continue                                                       # below, as banks
        if n.endswith(".weight"):
            bases.add(n[: -len(".weight")])
        elif not any(n.endswith(x) for x in (".weight_scale", ".weight_scale_2", ".input_scale", FIXED, EXC)):
            W[c] = idx.read(n, device)                                     # A_log, dt_bias, hash constants, ...
    for b in sorted(bases):
        W[canonical(b) + ".weight"] = projection(idx, b, device)
    # routed experts: one bank per layer and projection, NVWeights not cached (dequantised per use)
    for li in sorted(lay) + (["mtp"] if mtp else []):
        p = "mtp.layers.0.mlp." if li == "mtp" else f"{PREFIX}layers.{li}.mlp."
        cp = canonical(p)
        for x in ("gate", "up", "down"):
            blocks = {e: projection(idx, f"{p}experts.{e}.{x}_proj", device, cache=False) for e in range(cfg.num_experts)}
            W[f"{cp}experts.{x}_proj"] = ExpertBank(blocks=blocks)
        if log and isinstance(li, int) and li % 8 == 7:
            log(f"[nv-load] experts through layer {li}")
    tables = {}
    for li in cfg.ple_layers:
        pre = f"{PREFIX}layers.{li}.ple.ple_embedding.ngram_embedding."
        if f"{pre}shard_0.weight" in idx:
            tables[li] = NVFP4NGramTable(idx, pre)
    return cfg, W, tables


# ------------------------------------------------------------------------------ the engine's weights
def load_engine(path: str, device="cuda", mtp: bool = False, log=print, layers=None):
    """(cfg, W, banks, tables, timing) for `engine.FlashNextEngine` from an NVIDIA FP4-Local checkpoint, in the forms
    `load.load` gives an EXL3 pack (its docstring): the reference's names, the fused launches (`qkvz` = qkv | z,
    `ab` = a | b, `qkv` = q | k | v: MXFP8 rows stacked, bf16 concatenated), mixers as `down_inject` + up + `rank` (bf16
    as stored), `mlp.router` = gate ; shared_expert_gate (bf16), the experts as NVBanks, the n-gram table as an
    NVFP4RowTable. Nothing is requantised: every weight is the stored bytes."""
    import time
    from engine.models.flashnext.config import load_flashnext_config
    from engine.models.flashnext.ref import MTP
    t0 = time.time()
    cfg = load_flashnext_config(path)
    idx = HFIndex(path)
    lay = list(range(cfg.num_hidden_layers) if layers is None else layers)
    W: dict = {}
    mixers = set()
    for n in idx.tensors:
        if n.startswith("model.visual.") or (n.startswith("mtp.") and not mtp) or "ngram_embedding." in n:
            continue
        c = canonical(n)
        if c.startswith("layers.") and int(c.split(".")[1]) not in lay:
            continue
        if ".experts." in c or ".shared_expert." in c:
            continue                                                       # in the bank
        if "_hyper_connection." in c or c.startswith(("hyper_connection_mixer.", "mtp.hyper_connection_mixer.")):
            if c.endswith((".input_mix_weight_down.weight", ".input_mix_weight_up.weight", ".block_inject_weight.weight")):
                mixers.add(n.rsplit(".", 2)[0])
                continue
        if n.endswith(".weight"):
            W[c] = projection(idx, n[: -len(".weight")], device)
        elif not any(n.endswith(x) for x in (".weight_scale", ".weight_scale_2", ".input_scale", FIXED, EXC)):
            W[c] = idx.read(n, device)
    for site in sorted(mixers):
        dn = idx.read(site + ".input_mix_weight_down.weight", device)
        ij = site + ".block_inject_weight.weight"
        di = torch.cat([dn, idx.read(ij, device)]) if ij in idx else dn
        cs = canonical(site)
        W[cs + ".down_inject"] = BF16Rows(di)
        W[cs + ".input_mix_weight_up.weight"] = idx.read(site + ".input_mix_weight_up.weight", device).contiguous()
        W[cs + ".rank"] = dn.shape[0]

    def fuse(names):
        ws = [W[n] for n in names]
        if all(isinstance(w, NVWeight) for w in ws):
            g = NVWeight.cat(ws)
            o = 0
            for n, w in zip(names, ws):
                W[n] = g.rows(o, o + w.shape[0])
                o += w.shape[0]
            return g
        return torch.cat(ws).contiguous()
    for li in lay + ([MTP] if mtp else []):
        p = "mtp.layers.0." if li == MTP else f"layers.{li}."
        if li != MTP and cfg.is_linear(li):
            a = p + "linear_attn."
            W[a + "qkvz"] = fuse([a + "in_proj_qkv.weight", a + "in_proj_z.weight"])
            W[a + "ab"] = fuse([a + "in_proj_a.weight", a + "in_proj_b.weight"])
        else:
            a = p + "self_attn."
            W[a + "qkv"] = fuse([a + f"{x}_proj.weight" for x in ("q", "k", "v")])
        W[p + "mlp.router"] = BF16Rows(torch.cat([W[p + "mlp.gate.weight"], W[p + "mlp.shared_expert_gate.weight"]]))
    # the bf16 projections the engine reads through `lin` (the MTP layer's): onto the tensor-core skinny kernel
    for k, v in list(W.items()):
        if (isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 and v.dim() == 2 and k.startswith("mtp.")
                and k.endswith(("_proj.weight", ".qkv", "fc_embedding.weight", "fc_hidden.weight"))):
            W[k] = BF16Lin(v)
    timing = {"dense_s": round(time.time() - t0, 2)}
    t1 = time.time()
    banks = {}
    for li in lay:
        banks[li] = load_bank(idx, f"{PREFIX}layers.{li}.mlp.", cfg, device)
        if log and li % 8 == 7:
            log(f"[nv-load] experts through layer {li}: {time.time() - t1:.1f} s")
    if mtp:
        banks[MTP] = load_bank(idx, "mtp.layers.0.mlp.", cfg, device)
    timing["experts_s"] = round(time.time() - t1, 2)
    tables = {}
    for li in cfg.ple_layers:
        pre = f"{PREFIX}layers.{li}.ple.ple_embedding.ngram_embedding."
        if f"{pre}shard_0.weight" in idx and li in lay:
            tables[li] = NVFP4RowTable(idx, pre)
    timing["total_s"] = round(time.time() - t0, 2)
    if log:
        log(f"[nv-load] {path}: {timing}")
    return cfg, W, banks, tables, timing
