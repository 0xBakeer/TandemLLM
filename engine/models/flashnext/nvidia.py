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
before the cast to the compute dtype. Phase 0 (`load_reference`) feeds `ref.FlashNextRef`; the engine's kernels come later.

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


def nvfp4_dequant(w: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    """NVFP4 -> fp32 [N, K]: E2M1 codes [N, K/2] (low nibble = even column) x e4m3 block scale per 16 x fp32 global."""
    lut = E2M1.to(w.device)
    codes = torch.stack((w & 0xF, w >> 4), -1).reshape(w.shape[0], -1).long()
    v = lut[codes]                                                         # [N, K]
    s = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1) * scale2.float().reshape(())
    return v * s[:, : v.shape[1]]


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
