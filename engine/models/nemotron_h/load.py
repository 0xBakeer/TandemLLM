"""Load NemotronH weights onto the device, in the forms the engine multiplies by.

  W[name]          the reference's canonical names (engine/models/nemotron_h/ref.py) as bf16 tensors;
                   the router's correction bias, A_log, dt_bias and D as fp32 (the reference reads
                   them in fp32)
  W[fused]         layers.L.mixer.qkv: q | k | v of an attention layer in one weight (one launch)
  experts[L]       the routed experts of MoE layer L, stacked (`ExpertsRelu2`: up [E, I, X], down
                   [E, X, I], X the latent width or the hidden width); the MTP head's MoE block J
                   under ("mtp", J)

Each expert is copied into its stack slot straight from the file, so the host holds one expert at a
time beside the device copy.
"""

from __future__ import annotations

import time

import torch

from engine.models.nemotron_h.pack import PackIndex, expected

FP32 = ("gate.e_score_correction_bias", "mixer.A_log", "mixer.dt_bias", "mixer.D")


class ExpertsRelu2:
    """The routed experts of one MoE block, stacked: up [E, I, X], down [E, X, I], bf16."""

    def __init__(self, up: torch.Tensor, down: torch.Tensor):
        self.up, self.down = up, down
        self.E, self.I, self.X = up.shape

    @property
    def nbytes(self) -> int:
        return (self.up.numel() + self.down.numel()) * 2

    def expert(self, e: int):
        return self.up[e], self.down[e]


def _experts(idx: PackIndex, prefix: str, E: int, I: int, X: int, device, dtype) -> ExpertsRelu2:
    up = torch.empty(E, I, X, dtype=dtype, device=device)
    down = torch.empty(E, X, I, dtype=dtype, device=device)
    for e in range(E):
        up[e].copy_(idx.read(prefix + f"experts.{e}.up_proj.weight"))
        down[e].copy_(idx.read(prefix + f"experts.{e}.down_proj.weight"))
    return ExpertsRelu2(up, down)


def load(path: str, device="cuda", dtype=torch.bfloat16, mtp: bool = True, log=print, idx: PackIndex | None = None):
    """(idx, W, experts, timing) of the checkpoint at `path`; `mtp`: the MTP head's tensors too."""
    t0 = time.time()
    idx = idx or PackIndex.open(path)
    idx.audit()
    c = idx.cfg
    mtp = mtp and idx.has_mtp
    timing = {"index_s": round(time.time() - t0, 2), "notes": list(idx.notes)}
    W: dict = {}
    t1 = time.time()
    for name, _ in expected(c, mtp=mtp):
        if ".experts." in name:
            continue
        t = idx.read(name)
        want = torch.float32 if name.endswith(FP32) else dtype
        W[name] = t.to(device=device, dtype=want).contiguous()
    if c.tie_word_embeddings:
        W["lm_head.weight"] = W["embeddings.weight"]
    for li in c.attention_layers:
        m = f"layers.{li}.mixer."
        W[m + "qkv"] = torch.cat([W[m + f"{x}_proj.weight"] for x in "qkv"]).contiguous()
    timing["dense_s"] = round(time.time() - t1, 2)
    t1 = time.time()
    experts: dict = {}
    for n, li in enumerate(c.moe_layers):
        experts[li] = _experts(idx, f"layers.{li}.mixer.", c.moe_experts[li], c.moe_ffn[li], c.expert_in, device, dtype)
        if log and n % 8 == 7:
            log(f"[nh-load] experts through layer {li}: {time.time() - t1:.1f} s")
    if mtp:
        for j, (k, ffn, E) in c.mtp_moe.items():
            experts[("mtp", j)] = _experts(idx, f"mtp.layers.{j}.mixer.", E, ffn, c.expert_in, device, dtype)
        m = "mtp.layers.0.mixer."
        if m + "q_proj.weight" in W:
            W[m + "qkv"] = torch.cat([W[m + f"{x}_proj.weight"] for x in "qkv"]).contiguous()
    timing["experts_s"] = round(time.time() - t1, 2)
    timing["total_s"] = round(time.time() - t0, 2)
    timing["mtp"] = mtp
    if log:
        log(f"[nh-load] {path}: {timing}")
    return idx, W, experts, timing
