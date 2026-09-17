"""Read the checkpoint into device tensors, text model only, fp8 left as codes.

The checkpoint is one safetensors file per layer plus `outside.safetensors` (embeddings, head, final
norm, and the vision tower) and `mtp.safetensors`. The vision tensors are 0.92 GB of a 30.87 GB
checkpoint and this engine never evaluates them, so they are skipped by name.

Quantised projections arrive as a pair -- `X.weight` (fp8 e4m3) and `X.weight_scale_inv` (bf16, one
value per 128x128 block) -- and are kept as that pair in an `FP8Block`. Nothing is dequantised at
load time: the decode step is bandwidth-bound on exactly these bytes.
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fp8_linear import FP8Block  # noqa: E402
from tools.nvfp4_linear import NVFP4Block  # noqa: E402

LM_PREFIX = "model.language_model."
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


class Weights:
    """Every tensor the text model needs, addressed by its checkpoint name."""

    def __init__(self, snapshot: str, device: str = "cuda", *, skip_mtp: bool = False,
                 nvfp4: str | None = None):
        self.snapshot = snapshot
        self.device = device
        self.t: dict[str, torch.Tensor] = {}
        self.q: dict[str, FP8Block | NVFP4Block] = {}
        self.bytes_fp8 = 0
        self.bytes_other = 0
        self.nvfp4_source: str | None = None
        self._load(skip_mtp)
        nvfp4 = nvfp4 if nvfp4 is not None else os.environ.get("QWEN38_NVFP4")
        if nvfp4:
            self.load_nvfp4_mlp(os.path.expanduser(nvfp4))

    def _files(self, skip_mtp: bool) -> list[str]:
        out = []
        for name in sorted(os.listdir(self.snapshot)):
            if not name.endswith(".safetensors"):
                continue
            if name == "mtp.safetensors" and skip_mtp:
                continue
            out.append(os.path.join(self.snapshot, name))
        return out

    def _load(self, skip_mtp: bool) -> None:
        pending_scale: dict[str, torch.Tensor] = {}
        pending_code: dict[str, torch.Tensor] = {}
        for path in self._files(skip_mtp):
            with safe_open(path, framework="pt", device=self.device) as f:
                for key in f.keys():
                    if ".visual." in key or key.startswith("visual."):
                        continue
                    name = key[len(LM_PREFIX):] if key.startswith(LM_PREFIX) else key
                    if name.endswith(".weight_scale_inv"):
                        base = name[: -len(".weight_scale_inv")]
                        s = f.get_tensor(key)
                        if base in pending_code:
                            self._pair(base, pending_code.pop(base), s)
                        else:
                            pending_scale[base] = s
                        continue
                    t = f.get_tensor(key)
                    if t.dtype == torch.float8_e4m3fn:
                        base = name[: -len(".weight")]
                        if base in pending_scale:
                            self._pair(base, t, pending_scale.pop(base))
                        else:
                            pending_code[base] = t
                        continue
                    self.t[name] = t
                    self.bytes_other += t.numel() * t.element_size()
        if pending_code or pending_scale:
            raise RuntimeError(f"unpaired fp8 tensors: {sorted(pending_code) + sorted(pending_scale)}")

    def _pair(self, base: str, codes: torch.Tensor, scale: torch.Tensor) -> None:
        blk = FP8Block(codes, scale)
        self.q[base] = blk
        self.bytes_fp8 += blk.nbytes

    # --- accessors used by the model ---
    def norm(self, name: str) -> torch.Tensor:
        return self.t[name]

    def proj(self, name: str) -> FP8Block:
        return self.q[name]

    def report(self) -> str:
        total = self.bytes_fp8 + self.bytes_other
        return (f"{len(self.q)} quantised projections ({self.bytes_fp8 / 2**30:.2f} GiB), "
                f"{len(self.t)} plain tensors ({self.bytes_other / 2**30:.2f} GiB), "
                f"{total / 2**30:.2f} GiB resident")

    def decode_step_bytes(self, n_layers: int) -> dict[str, float]:
        """Bytes a single autoregressive step has to read, by group."""
        per_layer: dict[int, int] = {}
        for base, blk in self.q.items():
            if not base.startswith("layers."):
                continue
            idx = int(base.split(".")[1])
            per_layer[idx] = per_layer.get(idx, 0) + blk.nbytes
        for name, t in self.t.items():
            if not name.startswith("layers."):
                continue
            idx = int(name.split(".")[1])
            per_layer[idx] = per_layer.get(idx, 0) + t.numel() * t.element_size()
        layers = sum(v for k, v in per_layer.items() if k < n_layers)
        head = self.t["lm_head.weight"]
        return {
            "layers_GB": layers / 1e9,
            "lm_head_GB": head.numel() * head.element_size() / 1e9,
            "total_GB": (layers + head.numel() * head.element_size()) / 1e9,
        }

    # ------------------------------------------------------------------ NVFP4 MLPs
    def load_nvfp4_mlp(self, path: str) -> None:
        """Replace every layer's three MLP projections with their NVFP4 form.

        Two sources are accepted and normalised to the same three tensor names, because the whole
        point of the exercise is to compare them: a file written by `tools/quant_nvfp4.py`, whose
        keys are already `layers.N.mlp.X.<...>`, and a published NVFP4 snapshot directory, whose
        keys carry the checkpoint's `model.language_model.` prefix and whose other tensors -- the
        attention and linear-attention projections, which that checkpoint quantises per tensor and
        this engine reads per 128x128 block -- are deliberately not read.

        The fp8 codes of the replaced projections are dropped as each one is swapped, so the peak
        footprint is one extra projection, not a second copy of the MLPs.
        """
        files = []
        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.isfile(idx):
                import json
                names = sorted(set(json.load(open(idx))["weight_map"].values()))
                files = [os.path.join(path, n) for n in names]
            else:
                files = sorted(glob_safetensors(path))
        else:
            files = [path]
        found = 0
        freed = 0
        added = 0
        for fpath in files:
            with safe_open(fpath, framework="pt", device=self.device) as f:
                keys = [k for k in f.keys() if ".mlp." in k and (
                    k.endswith(".weight") or k.endswith(".weight_scale")
                    or k.endswith(".weight_scale_2"))]
                bases = sorted({k.rsplit(".", 1)[0] for k in keys
                                if k.rsplit(".", 1)[1] in ("weight", "weight_scale")})
                bases = [b for b in bases if b.split(".")[-1] in MLP_PROJ]
                for base in bases:
                    if f"{base}.weight_scale_2" not in f.keys():
                        continue          # not NVFP4: a plain fp8 or bf16 MLP, leave it alone
                    name = base[len(LM_PREFIX):] if base.startswith(LM_PREFIX) else base
                    if name not in self.q:
                        continue
                    old = self.q[name]
                    blk = NVFP4Block(f.get_tensor(f"{base}.weight"),
                                     f.get_tensor(f"{base}.weight_scale"),
                                     f.get_tensor(f"{base}.weight_scale_2").float().item())
                    assert blk.shape == old.shape, (name, blk.shape, old.shape)
                    freed += old.nbytes
                    added += blk.nbytes
                    self.q[name] = blk
                    old.w = None
                    old.s = None
                    found += 1
        if not found:
            raise RuntimeError(f"no NVFP4 MLP tensors found in {path}")
        torch.cuda.empty_cache()
        self.bytes_fp8 += added - freed
        self.nvfp4_source = path
        print(f"[nvfp4] {found} MLP projections from {path}: "
              f"{freed / 1e9:.2f} GB fp8 -> {added / 1e9:.2f} GB nvfp4")


def glob_safetensors(d: str) -> list[str]:
    import glob as _g
    return _g.glob(os.path.join(d, "*.safetensors"))
