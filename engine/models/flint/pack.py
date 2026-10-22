"""The Flint weights on disk: the BF16 checkpoint or an EXL3 pack, one index over both.

Both are safetensors shards with a `model.safetensors.index.json`. `PackIndex.open(dir)` reads every header
(nothing else) and puts every tensor in exactly one kind:

  expert   a routed expert's projection: `layers.L.mlp.experts.E.{gate,up,down}_proj.*`
  exl3     any other trellis-coded projection: `X.trellis` int16 [Kin/16, N/16, 16 K], `X.suh` / `X.svh`
           fp16 (or the packed sign fields `X.su` / `X.sv`), `X.mul1` (checked to be 0x83DCD12D)
  plain    everything else, as stored: a BF16 checkpoint's projections, the norms, the sink logits, the
           router's `gate.weight`, the embedding; an NVFP4 pack's projections (`X.weight` uint8, `X.weight_scale`
           e4m3, `X.weight_scale_2` fp32: tools/flint_nvfp4.py)

The layout and its checks are LFM2's (engine/models/lfm2_moe/pack.py); what differs is the names
(`projections()`, `plain_names()`, the expert pattern) and the decode bytes (KV rows by layer kind: a
sliding layer reads at most its window).
"""

from __future__ import annotations

import json
import os
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from engine.models.flint.config import FlintConfig, config_from_dict

PREFIX = "model."
MUL1 = 0x83DCD12D
EXPERT_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$")
#: the bits a weight the EXL3 kernels decode (tools/exl3_linear.py KS and HALF_KS)
RATES = (1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)
EXL3_SUFFIX = ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg")
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I16": torch.int16,
      "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8, "F8_E4M3": torch.float8_e4m3fn}
DSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1}


def safetensors_header(path: str) -> tuple[int, dict]:
    """(data start offset, header) of one safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return 8 + n, hdr


def canonical(key: str) -> str:
    return key[len(PREFIX):] if key.startswith(PREFIX) else key


@dataclass
class TensorInfo:
    name: str               # canonical
    key: str                # as stored
    file: str
    dtype: str
    shape: tuple
    offset: int             # absolute byte offset in the file
    nbytes: int
    kind: str = "plain"

    @property
    def k(self) -> float | None:
        """Bits a weight of a trellis tensor (x.5 for a half-integer rate: 16 K + 8 int16 a tile)."""
        if not self.name.endswith(".trellis"):
            return None
        w = self.shape[2]
        return w / 16 if w % 16 == 0 else (w - 8) / 16 + 0.5


def kind_of(name: str) -> str:
    if EXPERT_RE.match(name):
        return "expert"
    if name.rsplit(".", 1)[-1] in EXL3_SUFFIX:
        return "exl3"
    return "plain"


def projections(cfg: FlintConfig, experts: bool = True) -> list[tuple[str, int, int]]:
    """Every projection the model multiplies by: (base name, N, Kin), engine orientation."""
    c = cfg
    H = c.hidden_size
    out = []
    for li in range(c.num_hidden_layers):
        a = f"layers.{li}.self_attn."
        out += [(a + "q_proj", c.q_dim, H), (a + "k_proj", c.kv_dim, H), (a + "v_proj", c.kv_dim, H),
                (a + "o_proj", H, c.q_dim)]
        f = f"layers.{li}.mlp."
        if c.is_moe(li):
            if experts:
                I = c.moe_intermediate_size
                for e in range(c.num_experts):
                    out += [(f + f"experts.{e}.gate_proj", I, H), (f + f"experts.{e}.up_proj", I, H),
                            (f + f"experts.{e}.down_proj", H, I)]
            S = c.shared_expert_intermediate_size
            if S:
                out += [(f + "shared_expert.gate_proj", S, H), (f + "shared_expert.up_proj", S, H),
                        (f + "shared_expert.down_proj", H, S)]
        else:
            I = c.intermediate_size
            out += [(f + "gate_proj", I, H), (f + "up_proj", I, H), (f + "down_proj", H, I)]
    return out


def plain_names(cfg: FlintConfig) -> list[tuple[str, tuple]]:
    """Every plain tensor the model reads, with its shape (an optional router bias is not listed: the
    loader takes it when the checkpoint has one)."""
    c = cfg
    H = c.hidden_size
    out = [("embed_tokens.weight", (c.vocab_size, H)), ("norm.weight", (H,))]
    for li in range(c.num_hidden_layers):
        p = f"layers.{li}."
        out += [(p + "input_layernorm.weight", (H,)), (p + "post_attention_layernorm.weight", (H,)),
                (p + "self_attn.q_norm.weight", (c.head_dim,)), (p + "self_attn.k_norm.weight", (c.head_dim,)),
                (p + "self_attn.softmax_offset", (c.num_attention_heads,))]
        if c.is_moe(li):
            out.append((p + "mlp.gate.weight", (c.num_experts, H)))
    return out


# ------------------------------------------------------------------------------ the index
@dataclass
class PackIndex:
    path: str
    cfg: FlintConfig
    tensors: dict[str, TensorInfo]
    files: dict[str, int]                       # shard file -> size on disk
    quant: dict = field(default_factory=dict)   # the EXL3 pack's quantization_config ({} for BF16)
    formats: dict = field(default_factory=dict)  # projection base -> "bf16" | "exl3" (after audit)
    notes: list[str] = field(default_factory=list)

    @property
    def is_nvfp4(self) -> bool:
        return str(self.quant.get("quant_method", "")).lower() == "nvfp4"

    @property
    def is_exl3(self) -> bool:
        return str(self.quant.get("quant_method", "")).lower() == "exl3" or any(
            t.name.endswith(".trellis") for t in self.tensors.values())

    @classmethod
    def open(cls, path: str) -> "PackIndex":
        path = os.path.expanduser(path)
        with open(os.path.join(path, "config.json")) as f:
            raw = json.load(f)
        gen = None
        gp = os.path.join(path, "generation_config.json")
        if os.path.isfile(gp):
            with open(gp) as f:
                gen = json.load(f)
        cfg = config_from_dict(raw, path, gen)
        ip = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(ip):
            with open(ip) as f:
                wm = json.load(f)["weight_map"]
            names = sorted(set(wm.values()))
        elif os.path.isfile(os.path.join(path, "model.safetensors")):
            wm, names = None, ["model.safetensors"]
        else:
            raise FileNotFoundError(f"{path}: no model.safetensors.index.json and no model.safetensors")
        tensors: dict[str, TensorInfo] = {}
        files: dict[str, int] = {}
        for fn in names:
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"{fp}: the index names it, the pack does not have it")
            start, hdr = safetensors_header(fp)
            files[fn] = os.path.getsize(fp)
            listed = {k for k, v in wm.items() if v == fn} if wm is not None else set(hdr)
            for k in listed:
                if k not in hdr:
                    raise ValueError(f"{fn}: the index puts {k} here, the file's header does not have it")
            end = 0
            for k, t in hdr.items():
                b0, b1 = t["data_offsets"]
                end = max(end, b1)
                if k not in listed:
                    continue
                name = canonical(k)
                if name in tensors:
                    raise ValueError(f"{name} is in {tensors[name].file} and {fn}")
                info = TensorInfo(name, k, fp, t["dtype"], tuple(t["shape"]), start + b0, b1 - b0)
                if info.nbytes != int(np.prod(info.shape)) * DSIZE[info.dtype]:
                    raise ValueError(f"{fn}: {k} has {info.nbytes} bytes for {info.shape} {info.dtype}")
                info.kind = kind_of(name)
                tensors[name] = info
            if files[fn] != start + end:
                raise ValueError(f"{fp}: {files[fn]} bytes on disk, the header describes {start + end}"
                                 " (truncated or corrupt)")
        quant = raw.get("quantization_config") or {}
        return cls(path, cfg, tensors, files, quant)

    # --------------------------------------------------------------------------- checks
    def audit(self) -> list[str]:
        """Every tensor the layout expects, each projection as BF16 or as a whole EXL3 block whose
        shapes fit the config, mul1 codebook, integer K; the head is the pack's own or the embedding
        (tied). Raises naming the first problem; returns notes."""
        c = self.cfg
        formats = {}
        for base, N, Kin in projections(c) + [("lm_head", c.vocab_size, c.hidden_size)]:
            if base + ".weight" in self.tensors:
                t = self.tensors[base + ".weight"]
                if t.dtype == "U8" and base + ".weight_scale" in self.tensors:
                    # NVFP4: e2m1 codes two a byte, an e4m3 scale per 16 along K, an fp32 tensor scale
                    sc = self.tensors[base + ".weight_scale"]
                    if t.shape != (N, Kin // 2) or sc.shape != (N, Kin // 16) or base + ".weight_scale_2" not in self.tensors:
                        raise ValueError(f"{base}: NVFP4 {t.shape} / {sc.shape} does not fit {(N, Kin)}")
                    formats[base] = "nvfp4"
                    continue
                if t.shape != (N, Kin):
                    raise ValueError(f"{base}.weight {t.shape}: the config wants {(N, Kin)}")
                formats[base] = "bf16" if t.dtype == "BF16" else t.dtype.lower()
                continue
            if base + ".trellis" not in self.tensors:
                if base == "lm_head":
                    formats[base] = "tied"
                    continue
                raise ValueError(f"{base} is missing from the pack (no .weight, no .trellis)")
            t = self.tensors[base + ".trellis"]
            if base + ".mcg" in self.tensors:
                raise ValueError(f"{base}: the mcg codebook; the engine decodes mul1 only")
            if base + ".mul1" not in self.tensors:
                raise ValueError(f"{base}: no mul1 tensor (the 3inst codebook); the engine decodes mul1 only")
            k = t.k
            if k not in RATES:
                raise ValueError(f"{base}: {k} bits a weight (trellis {t.shape}); the kernels take 2..8, "
                                 "1.5, 2.5 and 3.5")
            if len(t.shape) != 3 or (t.shape[0] * 16, t.shape[1] * 16) != (Kin, N):
                raise ValueError(f"{base}.trellis {t.shape} does not fit {(N, Kin)}")
            for s, alt, n in (("suh", "su", Kin), ("svh", "sv", N)):
                if base + "." + s in self.tensors:
                    if self.tensors[base + "." + s].shape != (n,):
                        raise ValueError(f"{base}.{s} {self.tensors[base + '.' + s].shape} != ({n},)")
                elif base + "." + alt not in self.tensors:
                    raise ValueError(f"{base}: no {s} and no {alt}")
            formats[base] = "exl3"
        for name, shape in plain_names(c):
            t = self.tensors.get(name)
            if t is None:
                raise ValueError(f"{name} is missing from the pack")
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"{name} {t.shape}: the config wants {shape}")
        self.formats = formats
        kinds = Counter(formats.values())
        self.notes = [f"projections: {dict(kinds)}"]
        if self.tensors.get("lm_head.trellis") is None and formats["lm_head"] == "tied":
            self.notes.append("head tied to the embedding")
        return self.notes

    def check_mul1(self) -> None:
        """Every `.mul1` holds 0x83DCD12D (the codebook the kernels decode)."""
        for n in self.tensors:
            if n.endswith(".mul1"):
                v = int(self.read(n).view(-1)[0].item()) & 0xFFFFFFFF
                if v != MUL1:
                    raise ValueError(f"{n} = {v:#x}: only the mul1 codebook {MUL1:#x} is decoded")

    # --------------------------------------------------------------------------- summaries
    def k_histogram(self) -> dict[str, Counter]:
        """{group: Counter(K -> tensors)} of the trellis tensors: experts by projection, the others
        by family, the head."""
        out: dict[str, Counter] = defaultdict(Counter)
        for n, t in self.tensors.items():
            if not n.endswith(".trellis"):
                continue
            m = EXPERT_RE.match(n)
            if m:
                g = "experts." + m.group(3)
            elif n.startswith("lm_head"):
                g = "head"
            else:
                g = ".".join(p for p in n.split(".")[2:-1] if not p.isdigit())
            out[g][t.k] += 1
        return dict(out)

    def proj_bytes(self, base: str) -> int:
        """Bytes of one projection as stored (its weight, or its trellis and channel factors)."""
        if base + ".weight" in self.tensors:
            return self.tensors[base + ".weight"].nbytes + (self.tensors[base + ".weight_scale"].nbytes
                                                            if base + ".weight_scale" in self.tensors else 0)
        return sum(self.tensors[base + "." + s].nbytes for s in ("trellis", "suh", "svh", "su", "sv")
                   if base + "." + s in self.tensors)

    def expert_bytes(self, layer: int) -> list[int]:
        c = self.cfg
        return [sum(self.proj_bytes(f"layers.{layer}.mlp.experts.{e}.{w}") for w in ("gate_proj", "up_proj", "down_proj"))
                for e in range(c.num_experts)]

    def decode_bytes(self, context: int = 0) -> dict:
        """What one decode token reads (bytes), by part; `context`: the KV rows attention reads."""
        c = self.cfg
        parts: dict[str, float] = defaultdict(float)
        for li in c.moe_layers:
            eb = self.expert_bytes(li)
            parts["experts"] += c.num_experts_per_tok * sum(eb) / len(eb)
        for base, _, _ in projections(c, experts=False):
            fam = "attn_proj" if ".self_attn." in base else ("shared_expert" if ".shared_expert." in base else "dense_mlp")
            parts[fam] += self.proj_bytes(base)
        head = self.proj_bytes("lm_head") if ("lm_head.weight" in self.tensors or "lm_head.trellis" in self.tensors) \
            else self.tensors["embed_tokens.weight"].nbytes
        parts["head"] += head
        e = self.tensors["embed_tokens.weight"]
        parts["embed_row"] += e.shape[1] * DSIZE[e.dtype]
        for name, _ in plain_names(c):
            if name == "embed_tokens.weight":
                continue
            parts["router" if ".gate." in name else "plain_small"] += self.tensors[name].nbytes
        if context:
            # a full layer reads every row, a sliding layer its window
            row = 2 * c.kv_dim * 2
            parts["kv_read"] += row * (len(c.full_layers) * context + len(c.sliding_layers) * min(context, c.sliding_window))
        parts = {k: int(v) for k, v in parts.items()}
        return {"parts": parts, "total": sum(parts.values()), "context": context}

    def resident_bytes(self) -> dict:
        """What the engine holds on the device, by kind (the router and norms as bf16)."""
        out: dict[str, int] = defaultdict(int)
        for n, t in self.tensors.items():
            if n.endswith((".mul1", ".mcg")):
                continue
            out[t.kind] += t.nbytes
        out = dict(out)
        out["total"] = sum(out.values())
        return out

    # --------------------------------------------------------------------------- reading
    def read(self, name: str, device=None) -> torch.Tensor:
        """One tensor as stored (a CPU copy, or on `device`)."""
        t = self.tensors[name]
        arr = np.fromfile(t.file, dtype=np.uint8, count=t.nbytes, offset=t.offset)
        out = torch.from_numpy(arr).view(DT[t.dtype]).reshape(t.shape)
        return out.to(device) if device is not None else out

    def exl3(self, base: str, device=None, out_fp32: bool = False):
        """The EXL3 projection `base` as an EXL3Block (signs unpacked where the pack packed them)."""
        from tools.exl3_linear import EXL3Block, unpack_signs

        def part(s, alt):
            if base + "." + s in self.tensors:
                return self.read(base + "." + s, device)
            return unpack_signs(self.read(base + "." + alt)).to(device)
        return EXL3Block(self.read(base + ".trellis", device), part("suh", "su"), part("svh", "sv"), out_fp32=out_fp32)


def summary(idx: PackIndex) -> dict:
    """The audit record (ENG-285): counts by kind, K histogram, formats, resident and decode bytes."""
    return {
        "path": idx.path, "tensors": len(idx.tensors), "kinds": dict(Counter(t.kind for t in idx.tensors.values())),
        "files": idx.files, "quant": idx.quant,
        "formats": dict(Counter(idx.formats.values())),
        "k_histogram": {g: dict(sorted(c.items())) for g, c in sorted(idx.k_histogram().items())},
        "resident": idx.resident_bytes(),
        "decode_bytes": idx.decode_bytes(),
        "decode_bytes_8k": idx.decode_bytes(context=8192),
        "notes": idx.notes,
    }
