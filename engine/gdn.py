"""The gated delta rule and its depthwise causal convolution.

`flash-linear-attention` and `causal_conv1d` are not installed on this board and the interpreter
has no pip, so both are written here. That is not a workaround: the decode path needs a recurrent
step it can capture in a CUDA graph, and block verification needs to run the recurrence over an
arbitrary prefix of a verified block starting from a saved state, and neither is something the
library kernels expose.

Both functions here follow the reference implementation exactly, including the fp32 casts, the
l2 normalisation of q and k, the `1/sqrt(head_dim)` scale applied after it, and the chunk size 64
of the chunked form. `state_shape` is the contract the rest of the engine holds them to.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def conv_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One or more steps of the depthwise causal conv, advancing `conv_state` in place.

    x           [B, C, T]      the new projections
    conv_state  [B, C, W - 1 + T'] the last W-1 inputs; updated to the last W-1 of the concatenation
    weight      [C, W]
    """
    width = weight.shape[-1]
    joined = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(joined[:, :, -(width - 1):] if width > 1 else joined[:, :, :0])
    out = F.conv1d(joined, weight.unsqueeze(1), None, padding=0, groups=x.shape[1])
    return F.silu(out[:, :, -x.shape[-1]:]).to(x.dtype)


def conv_prefill(x: torch.Tensor, weight: torch.Tensor,
                 conv_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The same convolution over a fresh sequence; returns the output and the new conv state."""
    width = weight.shape[-1]
    padded = F.pad(x, (width - 1, 0))
    out = F.silu(F.conv1d(padded, weight.unsqueeze(1), None, padding=0, groups=x.shape[1]))
    new_state = padded[:, :, -(width - 1):].contiguous() if width > 1 else padded[:, :, :0]
    return out.to(x.dtype), new_state


def recurrent_gated_delta_rule(query, key, value, g, beta, state):
    """Token-at-a-time. Shapes [B, T, H, D]; `state` [B, H, Dk, Dv] fp32, advanced in place."""
    dtype = query.dtype
    query = l2norm(query.float(), dim=-1)
    key = l2norm(key.float(), dim=-1)
    query = query * (query.shape[-1] ** -0.5)
    value = value.float()
    beta = beta.float()
    g = g.float()
    B, T, H, Dv = value.shape
    out = torch.empty(B, T, H, Dv, dtype=torch.float32, device=value.device)
    for i in range(T):
        q_t = query[:, i]              # [B, H, Dk]
        k_t = key[:, i]
        v_t = value[:, i]              # [B, H, Dv]
        state.mul_(g[:, i].exp()[:, :, None, None])
        kv = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv) * beta[:, i][:, :, None]
        state.add_(k_t[:, :, :, None] * delta[:, :, None, :])
        out[:, i] = torch.einsum("bhkv,bhk->bhv", state, q_t)
    return out.to(dtype), state


def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = 64,
                           output_final_state: bool = True):
    """Chunked form, for prefill and for verifying a block of drafted tokens.

    Shapes in [B, T, H, D], state [B, H, Dk, Dv] fp32. A copy of `state` is used, never the caller's
    tensor, so the entry state of a speculative block survives the verify pass unchanged.
    """
    dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g)]
    B, H, T, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    Tp = T + pad
    query = query * (Dk ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(B, H, -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = (torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=value.device)
         if state is None else state.clone().float())
    out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 1)
    for i in range(Tp // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        inter = (q_i * g[:, :, i, :, None].exp()) @ S
        out[:, :, i] = inter + a @ v_new
        S = (S * g[:, :, i, -1, None, None].exp()
             + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(dtype)
    return out, (S if output_final_state else None)
