"""The reference forward: correct, readable, and slow enough to be trusted.

Everything faster in this engine is checked against this file. It follows the published
implementation operation for operation, including where a cast to fp32 happens and which of the two
RMS norm conventions a given norm uses, because those are the details that silently cost accuracy:

  * `rms_norm`      -- input / post-attention / final / q / k -- is `normalise(x) * (1 + weight)`
  * `rms_norm_gated` -- the linear-attention output norm      -- is `weight * normalise(x) * silu(z)`

The weights are never dequantised into memory. Projections read the checkpoint's fp8 codes through
`tools.fp8_linear`, which applies the 128x128 block scale to the fp32 accumulator.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import gdn  # noqa: E402
from engine.config import TextConfig  # noqa: E402
from engine.loader import Weights  # noqa: E402
from tools.fp8_linear import FP8Block, fp8_matmul  # noqa: E402


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dt)
    h = h * F.silu(gate.float())
    return h.to(dt)


def linear(x: torch.Tensor, w: FP8Block | torch.Tensor) -> torch.Tensor:
    """`x @ w^T` for either a stored fp8 block weight or a plain bf16 one."""
    if isinstance(w, FP8Block):
        flat = x.reshape(-1, x.shape[-1])
        return fp8_matmul(flat, w).view(*x.shape[:-1], w.N)
    return F.linear(x, w)


class KVCache:
    """One sequence, one contiguous buffer per attention layer, an append pointer per layer.

    Rolling a rejected speculative block back is the pointer moving back; nothing is copied.
    """

    def __init__(self, cfg: TextConfig, max_len: int, device: str, dtype=torch.bfloat16):
        n = len(cfg.attention_layers)
        self.slot = {l: i for i, l in enumerate(cfg.attention_layers)}
        self.k = torch.zeros(n, 1, cfg.num_key_value_heads, max_len, cfg.head_dim,
                             dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.length = 0
        self.max_len = max_len

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> tuple:
        i = self.slot[layer]
        t = k.shape[2]
        self.k[i, :, :, start:start + t] = k
        self.v[i, :, :, start:start + t] = v
        return self.k[i, :, :, :start + t], self.v[i, :, :, :start + t]


class GDNState:
    """The recurrent and convolution state of every linear-attention layer, one sequence."""

    def __init__(self, cfg: TextConfig, device: str):
        n = len(cfg.linear_layers)
        self.slot = {l: i for i, l in enumerate(cfg.linear_layers)}
        self.S = torch.zeros(n, 1, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
                             cfg.linear_value_head_dim, dtype=torch.float32, device=device)
        self.conv = torch.zeros(n, 1, cfg.conv_dim, cfg.linear_conv_kernel_dim - 1,
                                dtype=torch.bfloat16, device=device)
        self.primed = False

    def clone(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.S.clone(), self.conv.clone()

    def restore(self, saved: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.S.copy_(saved[0])
        self.conv.copy_(saved[1])

    @property
    def nbytes(self) -> int:
        return self.S.numel() * 4 + self.conv.numel() * 2


class Qwen38Engine:
    def __init__(self, cfg: TextConfig, w: Weights, max_len: int = 8192, device: str = "cuda"):
        self.cfg = cfg
        self.w = w
        self.device = device
        self.max_len = max_len
        self.kv = KVCache(cfg, max_len, device)
        self.state = GDNState(cfg, device)
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.tap = None  # set to a callable to receive every layer's hidden state

    # ---------------------------------------------------------------- rotary
    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Partial rotary over the first `rotary_dim` of each head.

        The published model carries an interleaved 3-axis mRoPE for image and video grids. For text
        the three position rows are identical, so the interleave rewrites each row with a copy of
        itself and the result is ordinary RoPE. This engine is text only, so that is what is built.
        """
        dim = self.cfg.rotary_dim
        inv = 1.0 / (self.cfg.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
        freqs = positions.float()[:, None] * inv[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.cat([-b, a], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                   sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = cos.shape[-1]
        c = cos[None, None, :, :]
        s = sin[None, None, :, :]
        qr, qp = q[..., :d], q[..., d:]
        kr, kp = k[..., :d], k[..., d:]
        q = torch.cat([qr * c + self._rotate_half(qr) * s, qp], dim=-1)
        k = torch.cat([kr * c + self._rotate_half(kr) * s, kp], dim=-1)
        return q, k

    # ---------------------------------------------------------------- blocks
    def mlp(self, h: torch.Tensor, p: str) -> torch.Tensor:
        gate = linear(h, self.w.proj(f"{p}.mlp.gate_proj"))
        up = linear(h, self.w.proj(f"{p}.mlp.up_proj"))
        return linear(F.silu(gate) * up, self.w.proj(f"{p}.mlp.down_proj"))

    def attention(self, h: torch.Tensor, p: str, layer: int, start: int,
                  positions: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h.shape
        qg = linear(h, self.w.proj(f"{p}.self_attn.q_proj")).view(B, T, cfg.num_attention_heads,
                                                                  cfg.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = rms_norm(q, self.w.norm(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        k = linear(h, self.w.proj(f"{p}.self_attn.k_proj")).view(B, T, cfg.num_key_value_heads,
                                                                 cfg.head_dim)
        k = rms_norm(k, self.w.norm(f"{p}.self_attn.k_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        v = linear(h, self.w.proj(f"{p}.self_attn.v_proj")).view(
            B, T, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = self.apply_rope(q, k, cos, sin)
        kk, vv = self.kv.append(layer, k, v, start)
        rep = cfg.num_attention_heads // cfg.num_key_value_heads
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
        if T == 1:
            o = F.scaled_dot_product_attention(q, kk, vv, is_causal=False)
        else:
            mask = torch.ones(T, kk.shape[2], dtype=torch.bool, device=h.device).tril(start)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return linear(o, self.w.proj(f"{p}.self_attn.o_proj"))

    def linear_attention(self, h: torch.Tensor, p: str, layer: int,
                         use_state: bool) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h.shape
        i = self.state.slot[layer]
        mixed = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_qkv")).transpose(1, 2)
        cw = self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1)
        if use_state:
            mixed = gdn.conv_update(mixed, self.state.conv[i], cw)
        else:
            mixed, new_conv = gdn.conv_prefill(mixed, cw)
            self.state.conv[i].copy_(new_conv)
        mixed = mixed.transpose(1, 2)
        q, k, v = mixed.split([cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1)
        q = q.view(B, T, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.view(B, T, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.view(B, T, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        z = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_z")).view(
            B, T, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        b = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_b.weight"))
        a = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_a.weight"))
        beta = b.sigmoid()
        A_log = self.w.norm(f"{p}.linear_attn.A_log")
        dt_bias = self.w.norm(f"{p}.linear_attn.dt_bias")
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        rep = cfg.num_v_per_k
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        if use_state and T == 1:
            o, _ = gdn.recurrent_gated_delta_rule(q, k, v, g, beta, self.state.S[i])
        else:
            o, S = gdn.chunk_gated_delta_rule(q, k, v, g, beta,
                                              self.state.S[i] if use_state else None)
            self.state.S[i].copy_(S)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        o = o.view(B, T, -1)
        return linear(o, self.w.proj(f"{p}.linear_attn.out_proj"))

    # ---------------------------------------------------------------- forward
    def forward(self, tokens: torch.Tensor, start: int = 0, *,
                last_only: bool = False) -> torch.Tensor:
        cfg = self.cfg
        T = tokens.numel()
        h = F.embedding(tokens.view(1, T), self.w.norm("embed_tokens.weight"))
        if self.tap is not None:
            self.tap(h[0].detach())
        positions = torch.arange(start, start + T, device=self.device)
        use_state = self.state.primed
        for layer in range(cfg.num_hidden_layers):
            p = f"layers.{layer}"
            res = h
            x = rms_norm(h, self.w.norm(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            if cfg.is_linear(layer):
                x = self.linear_attention(x, p, layer, use_state)
            else:
                x = self.attention(x, p, layer, start, positions)
            h = res + x
            res = h
            x = rms_norm(h, self.w.norm(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
            h = res + self.mlp(x, p)
            if self.tap is not None:
                self.tap(h[0].detach())
        self.state.primed = True
        self.kv.length = start + T
        h = rms_norm(h, self.w.norm("norm.weight"), cfg.rms_norm_eps)
        if last_only:
            h = h[:, -1:]
        return linear(h, self.w.norm("lm_head.weight"))

    def reset(self) -> None:
        self.state.S.zero_()
        self.state.conv.zero_()
        self.state.primed = False
        self.kv.length = 0
