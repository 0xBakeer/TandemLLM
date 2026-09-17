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

# Projections that read the SAME activation and do not read each other's output, so their launches
# can be one launch. See `tools/nvfp4_linear_v2.NVFP4Group` for why that is worth doing; the short
# version is that a verify runs one kernel at a time and the widest of them puts 272 programs on
# 48 SMs, so the board spends every launch's tail idle and there are eleven launches a layer.
#
# These three cover 9.2 GB of the verify's 13.7 and take the step from 400 launches to 256.
PROJ_GROUPS = (
    ("mlp.gate_up", ("mlp.gate_proj", "mlp.up_proj")),
    ("self_attn.qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("linear_attn.qkvz", ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")),
)


class Weights:
    """Every tensor the text model needs, addressed by its checkpoint name."""

    def __init__(self, snapshot: str, device: str = "cuda", *, skip_mtp: bool = False,
                 nvfp4: str | None = None, fp8_head: str | None = None):
        self.snapshot = snapshot
        self.device = device
        self.t: dict[str, torch.Tensor] = {}
        self.q: dict[str, FP8Block | NVFP4Block] = {}
        self.g: dict = {}
        self.bytes_fp8 = 0
        self.bytes_other = 0
        self.nvfp4_source: str | None = None
        self.fp8_head_source: str | None = None
        self._load(skip_mtp)
        nvfp4 = nvfp4 if nvfp4 is not None else os.environ.get("QWEN38_NVFP4")
        if nvfp4:
            # A comma-separated list, because the quality gate decides which GROUPS of projections
            # are quantised and that decision has to be expressible without re-running the
            # quantiser: one file per group, and the ones that passed are the ones that are named.
            for part in str(nvfp4).split(","):
                if part.strip():
                    self.load_nvfp4_mlp(os.path.expanduser(part.strip()))
        fp8_head = fp8_head if fp8_head is not None else os.environ.get("QWEN38_FP8_HEAD")
        if fp8_head:
            self.load_fp8_head(os.path.expanduser(fp8_head))
        from tools.nvfp4_linear_v2 import FUSE_PROJ
        if FUSE_PROJ:
            layers = [int(k.split(".")[1]) for k in self.q if k.startswith("layers.")]
            if layers:
                self.fuse_nvfp4_groups(max(layers) + 1)

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

    def group(self, name: str):
        """The fused form of a projection group, or None if it was not built for this layer."""
        return self.g.get(name)

    def fuse_nvfp4_groups(self, n_layers: int) -> int:
        """Lay each group's projections out as one weight, and leave the members as views of it.

        Only NVFP4 groups, and only complete ones: a layer whose attention projections were left in
        fp8 by the quality gate keeps its three launches, and the model falls back to them by
        finding no group. Nothing is duplicated -- `torch.cat` along dim 0 leaves each member's
        rows contiguous, so after the copy every member points into the fused buffer and its own
        storage is dropped. Peak cost is one extra copy of the largest group, 89 MB.
        """
        from tools.nvfp4_linear_v2 import NVFP4Group
        made, fused_bytes = 0, 0
        for layer in range(n_layers):
            p = f"layers.{layer}"
            for key, members in PROJ_GROUPS:
                names = [f"{p}.{m}" for m in members]
                blocks = [self.q.get(n) for n in names]
                if any(not isinstance(b, NVFP4Block) for b in blocks):
                    continue
                grp = NVFP4Group(blocks, names)
                self.g[f"{p}.{key}"] = grp
                fused_bytes += grp.nbytes
                made += 1
        torch.cuda.empty_cache()
        if made:
            print(f"[fuse] {made} projection groups, {fused_bytes / 1e9:.2f} GB in one launch each")
        return made

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
        # A directory is somebody else's published snapshot, and this engine reads only its MLPs:
        # its attention and linear-attention tensors are quantised per tensor where this one reads
        # per 128x128 block, so they are a different format wearing the same names. A single file
        # is one this repository wrote, and whatever is in it is what was asked for -- since 13:20
        # of phase 4 that can be the GDN and attention projections as well.
        mlp_only = os.path.isdir(path)
        for fpath in files:
            with safe_open(fpath, framework="pt", device=self.device) as f:
                keys = [k for k in f.keys() if (not mlp_only or ".mlp." in k) and (
                    k.endswith(".weight") or k.endswith(".weight_scale")
                    or k.endswith(".weight_scale_2"))]
                bases = sorted({k.rsplit(".", 1)[0] for k in keys
                                if k.rsplit(".", 1)[1] in ("weight", "weight_scale")})
                if mlp_only:
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
            raise RuntimeError(f"no NVFP4 tensors found in {path}")
        torch.cuda.empty_cache()
        self.bytes_fp8 += added - freed
        self.nvfp4_source = path if not self.nvfp4_source else f"{self.nvfp4_source},{path}"
        print(f"[nvfp4] {found} projections from {path}: "
              f"{freed / 1e9:.2f} GB fp8 -> {added / 1e9:.2f} GB nvfp4")

    # ------------------------------------------------------------------ fp8 lm_head
    def load_fp8_head(self, path: str) -> None:
        """Swap the bf16 `lm_head` for the e4m3 one written by `tools/quant_head.py`.

        The bf16 tensor is dropped as the codes arrive: the head is the single largest tensor the
        engine holds and keeping both would cost 3.8 GB for nothing. Everything downstream reaches
        the head through `Weights.norm("lm_head.weight")`, and `engine.model.head_logits`
        dispatches on the type, so nothing else has to know.
        """
        from safetensors import safe_open
        from tools.head_gemv import FP8Head
        with safe_open(path, framework="pt", device=self.device) as f:
            head = FP8Head(f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))
        old = self.t["lm_head.weight"]
        assert tuple(head.shape) == tuple(old.shape), (head.shape, old.shape)
        was = old.numel() * old.element_size()
        self.t["lm_head.weight"] = head
        del old
        torch.cuda.empty_cache()
        self.bytes_other += head.nbytes - was
        self.fp8_head_source = path
        print(f"[fp8-head] {was / 1e9:.2f} GB bf16 -> {head.nbytes / 1e9:.2f} GB e4m3 "
              f"(per-row scales) from {path}")


def glob_safetensors(d: str) -> list[str]:
    import glob as _g
    return _g.glob(os.path.join(d, "*.safetensors"))
