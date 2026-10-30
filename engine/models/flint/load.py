"""Load Flint weights onto the device, in the forms the engine multiplies by.

  W[name]          the reference's canonical names (engine/models/flint/ref.py): EXL3Blocks for the pack's
                   EXL3 projections, bf16 tensors for the rest (an fp16 copy the converter stored is cast
                   back to bf16: the norms, the router, the embedding); the sink logits in fp32
  W[fused]         what the engine launches once instead of two or three times:
                     layers.L.self_attn.qkv            q | k | v (one EXL3 group, or one bf16 weight)
                     layers.L.mlp.gate_up              gate | up of a dense layer
                     layers.L.mlp.shared_expert.gate_up   gate | up of the shared expert
  W["lm_head.weight"]  the pack's head (Flint's is untied)
  experts[L]       the routed experts of MoE layer L: LFM2's `ExpertsEXL3` (each projection's trellis,
                   suh, svh, for the bank kernels), `NVFP4GLUBank` (engine/models/flint/nvfp4.py) or `ExpertsBF16`
                   (stacked [E, 2 I, H] / [E, H, I])
  An NVFP4 pack's projections are tools/nvfp4_relu2.NVFP4Wide (the W4A16 kernels), fused groups one block.
"""

from __future__ import annotations

import os
import time

import torch

from engine.models.flint.pack import PackIndex, plain_names, projections
from engine.models.lfm2_moe.load import ExpertsBF16, ExpertsEXL3


def _plain(idx: PackIndex, name: str, device, dtype=torch.bfloat16) -> torch.Tensor:
    t = idx.read(name)
    return t.to(device=device, dtype=dtype if t.is_floating_point() else t.dtype).contiguous()


def _nvfp4_parts(idx: PackIndex, base: str):
    """(codes, e4m3 scales, tensor scale) of an NVFP4 projection, host tensors as stored."""
    return (idx.read(base + ".weight"), idx.read(base + ".weight_scale").view(torch.float8_e4m3fn),
            float(idx.read(base + ".weight_scale_2").float().view(-1)[0]))


def _nvfp4(idx: PackIndex, bases: list, device):
    """One NVFP4 projection, or several sharing their input and tensor scale as one launch (rows concatenated)."""
    from tools.nvfp4_linear import NVFP4Block
    from tools.nvfp4_relu2 import NVFP4Wide
    parts = [_nvfp4_parts(idx, b) for b in bases]
    s2 = {p[2] for p in parts}
    if len(s2) != 1:
        raise ValueError(f"{bases}: NVFP4 members with different tensor scales cannot share a launch")
    blk = NVFP4Block(torch.cat([p[0] for p in parts]).to(device), torch.cat([p[1] for p in parts]).to(device), s2.pop())
    # the dense layers' and the shared experts' SwiGLU activations stay inside fp16 (|silu(g) u| up to 6.8e3 on
    # 2,048-token text, the fp32 reference): no row scaling, which cost ~1 ms a decode step in small launches. The
    # routed experts (up to 2.6e5) run on NVFP4GLUBank, fp32 throughout
    return NVFP4Wide(blk, wide=False)


def _proj(idx: PackIndex, base: str, device, dtype, out_fp32: bool = False):
    if idx.formats.get(base) == "exl3":
        return idx.exl3(base, device, out_fp32=out_fp32)
    if idx.formats.get(base) == "nvfp4":
        return _nvfp4(idx, [base], device)
    return _plain(idx, base + ".weight", device, dtype)


def _fuse(W: dict, base: str, parts: tuple, names: list, fused: str, idx: PackIndex | None = None, device=None) -> None:
    from tools.exl3_linear import EXL3Block
    if idx is not None and all(idx.formats.get(base + x) == "nvfp4" for x in parts):
        W[base + fused] = _nvfp4(idx, [base + x for x in parts], device)
        for x in parts:                  # the engine reads the fused launch only
            W.pop(base + x + ".weight", None)
        return
    ws = [W[base + x + ".weight"] for x in parts]
    if all(isinstance(w, EXL3Block) for w in ws) and len({w.kb for w in ws}) == 1:
        g = EXL3Block.group(ws, names)
        W[base + fused] = g
        for i, x in enumerate(parts):
            W[base + x + ".weight"] = g.member(i)
    elif all(torch.is_tensor(w) for w in ws):
        W[base + fused] = torch.cat(ws).contiguous()


#: Flint's own splits of tools/fn_kernels.dense for its EXL3 shapes (tools/fn_tiles.py --dense; a shape not in the file
#: takes the kernel's default split): a split is a summation order, so the file is in the fingerprint (serving.TABLES)
DENSE_SPLITS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                            "ops", "flint-dense-splits.json")


def apply_splits(path: str = DENSE_SPLITS) -> int:
    from engine.models.lfm2_moe.load import apply_splits as _apply
    return _apply(path)


def load(path: str, device="cuda", dtype=torch.bfloat16, log=print, idx: PackIndex | None = None):
    """(idx, W, experts, timing) of the checkpoint or pack at `path`."""
    t0 = time.time()
    idx = idx or PackIndex.open(path)
    idx.audit()
    if idx.formats.get("lm_head") == "tied":
        raise ValueError(f"{path}: no lm_head; Flint's head is untied")
    if idx.is_exl3:
        idx.check_mul1()
    c = idx.cfg
    timing = {"index_s": round(time.time() - t0, 2), "notes": list(idx.notes)}
    if idx.is_exl3 and str(device).startswith("cuda"):
        timing["dense_splits"] = apply_splits()
    W: dict = {}
    t1 = time.time()
    fused_nvfp4 = {b for b, _, _ in projections(c, experts=False)
                   if idx.formats.get(b) == "nvfp4" and b.rsplit(".", 1)[1] in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")}
    for base, _, _ in projections(c, experts=False):
        if base not in fused_nvfp4:            # an NVFP4 member is read once, as its fused launch
            W[base + ".weight"] = _proj(idx, base, device, dtype)
    for name, _ in plain_names(c):
        W[name] = _plain(idx, name, device, torch.float32 if name.endswith("softmax_offset") else dtype)
    for li in c.moe_layers:
        b = f"layers.{li}.mlp.gate.expert_bias"
        if b in idx.tensors:
            W[b] = _plain(idx, b, device, torch.float32)
    W["lm_head.weight"] = _proj(idx, "lm_head", device, dtype, out_fp32=True)
    timing["dense_s"] = round(time.time() - t1, 2)
    for li in range(c.num_hidden_layers):
        p = f"layers.{li}."
        _fuse(W, p + "self_attn.", ("q_proj", "k_proj", "v_proj"), ["q", "k", "v"], "qkv", idx, device)
        if c.is_moe(li):
            if c.shared_expert_intermediate_size:
                _fuse(W, p + "mlp.shared_expert.", ("gate_proj", "up_proj"), ["gate", "up"], "gate_up", idx, device)
        else:
            _fuse(W, p + "mlp.", ("gate_proj", "up_proj"), ["gate", "up"], "gate_up", idx, device)
    t1 = time.time()
    experts: dict = {}
    for li in c.moe_layers:
        f = f"layers.{li}.mlp.experts."
        if idx.formats.get(f + "0.gate_proj") == "nvfp4":
            from engine.models.flint.nvfp4 import NVFP4GLUBank
            ex = []
            for e in range(c.num_experts):
                g, u, d = (_nvfp4_parts(idx, f + f"{e}.{w}") for w in ("gate_proj", "up_proj", "down_proj"))
                if g[2] != u[2]:
                    raise ValueError(f"{f}{e}: gate and up need one tensor scale")
                ex.append(((g[0], g[1]), (u[0], u[1]), g[2], (d[0], d[1]), d[2]))
            experts[li] = NVFP4GLUBank.stack(ex, device)
        elif idx.formats.get(f + "0.gate_proj") == "exl3":
            experts[li] = ExpertsEXL3([tuple(idx.exl3(f + f"{e}.{w}", device) for w in ("gate_proj", "up_proj", "down_proj"))
                                       for e in range(c.num_experts)])
        else:
            I, H = c.moe_intermediate_size, c.hidden_size
            w13 = torch.empty(c.num_experts, 2 * I, H, dtype=dtype, device=device)
            w2 = torch.empty(c.num_experts, H, I, dtype=dtype, device=device)
            for e in range(c.num_experts):
                w13[e, :I].copy_(idx.read(f + f"{e}.gate_proj.weight"))
                w13[e, I:].copy_(idx.read(f + f"{e}.up_proj.weight"))
                w2[e].copy_(idx.read(f + f"{e}.down_proj.weight"))
            experts[li] = ExpertsBF16(w13, w2)
        if log and li % 8 == 7:
            log(f"[flint-load] experts through layer {li}: {time.time() - t1:.1f} s")
    timing["experts_s"] = round(time.time() - t1, 2)
    timing["total_s"] = round(time.time() - t0, 2)
    if log:
        log(f"[flint-load] {path}: {timing}")
    return idx, W, experts, timing
