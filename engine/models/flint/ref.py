"""The Flint forward in torch, operation for operation: the model the board code is held to.

`FlintRef` runs a block of rows through the model, where a block is a prompt chunk (a chain), one decode
token, or a draft tree in DFS pre-order with its parents. Every row sees the cache's rows [0, length) and
its own ancestors in the block, and `commit(path)` keeps a path: the path's KV rows moved down to positions
length.., the rest left past the length. The arithmetic is written from the published model description
(the checkpoint's model card and serving kit):

  attention   q, k, v; per-head RMS norm on q and k (plain `w`); NEOX rotary at the row's position on
              the sliding layers, none on the full layers (`no_rope_layer_indices`); scores q k^T /
              sqrt(D) over the visible keys -- on a sliding layer the keys at positions > pos - W, W the
              config's `sliding_window`, the row itself counted -- and one learned sink logit per query
              head (`softmax_offset`) as an extra softmax column with no value; o_proj
  MLP         dense SwiGLU (`mlp_only_layers`), else the routed experts: sigmoid of the router logits,
              the top k (an `expert_bias` for the choice only, none in the released checkpoint), the
              chosen sigmoids over their sum + 1e-20, each expert's output times its weight added in
              ascending expert id, plus the ungated shared expert
  norms       pre-norm residual blocks, the final norm, an untied head

with the dtype casts of the LFM2 reference (engine/models/lfm2_moe/ref.py): the RMS norm in fp32, cast
back, then times `w`; the rotary tables in the activations' dtype; the softmax in fp32. The tests hold it
to a plain fp32 implementation of the same description on a tiny random model (tests/test_flint_ref.py).

Weights are a dict of canonical names (the checkpoint's without its `model.` prefix):

  embed_tokens.weight, norm.weight, lm_head.weight
  layers.L.input_layernorm.weight, layers.L.post_attention_layernorm.weight
  layers.L.self_attn.{q,k,v,o}_proj.weight, .{q,k}_norm.weight [D], .softmax_offset [Hq]
  layers.L.mlp.{gate,up,down}_proj.weight                                    (dense layers)
  layers.L.mlp.gate.weight [E, H], .experts.E.{gate,up,down}_proj.weight,
  layers.L.mlp.shared_expert.{gate,up,down}_proj.weight                       (MoE layers)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from engine.models.lfm2_moe.ref import chain, tree_paths

__all__ = ["FlintRef", "FlintState", "chain", "tree_paths"]


@dataclass
class FlintState:
    """One sequence: K and V per layer [kv heads, cap, head_dim] (the reference keeps every row of
    every layer; the board engine keeps a ring on the sliding layers), the length."""
    cap: int
    length: int = 0
    k: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    ring: dict = field(default_factory=dict)   # the board engine's: layer -> ring rows (absent: the whole history)
    ks: dict = field(default_factory=dict)     # FLINT_KV_FP8: a full layer's e4m3 scales [kv heads, cap] fp32
    vs: dict = field(default_factory=dict)
    pages: object = None
    block: tuple | None = None

    def reset_inplace(self) -> None:
        self.length = 0


class FlintRef:
    def __init__(self, cfg, W: dict, dtype=torch.float32, device="cpu", experts: dict | None = None):
        self.cfg, self.W, self.dtype, self.device = cfg, W, dtype, torch.device(device)
        self.experts = experts or {}
        D = cfg.head_dim
        self.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        self._block = None

    # ------------------------------------------------------------------------ helpers
    @staticmethod
    def pre(li: int) -> str:
        return f"layers.{li}."

    def w(self, name: str) -> torch.Tensor:
        return self.W[name]

    def head_weight(self):
        return self.W["lm_head.weight"]

    def lin(self, x: torch.Tensor, w) -> torch.Tensor:
        if not torch.is_tensor(w):
            return w.matmul(x.to(torch.bfloat16)).to(x.dtype if not w.out_fp32 else torch.float32)
        return F.linear(x, w)

    def rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return w * h.to(dt)

    def rope(self, positions: torch.Tensor):
        f = positions.float()[:, None] * self.inv_freq[None, :].to(positions.device)
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def new_state(self, cap: int) -> FlintState:
        c = self.cfg
        st = FlintState(cap=cap)
        for li in range(c.num_hidden_layers):
            st.k[li] = torch.zeros(c.num_key_value_heads, cap, c.head_dim, dtype=self.dtype, device=self.device)
            st.v[li] = torch.zeros_like(st.k[li])
        return st

    # ------------------------------------------------------------------------ the layers
    def attention(self, h: torch.Tensor, li: int, st: FlintState, paths: list[list[int]],
                  positions: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = self.pre(li) + "self_attn."
        M, D, Hq, Hk = h.shape[0], c.head_dim, c.num_attention_heads, c.num_key_value_heads
        L = st.length
        q = self.rms(self.lin(h, self.w(p + "q_proj.weight")).view(M, Hq, D), self.w(p + "q_norm.weight"))
        k = self.rms(self.lin(h, self.w(p + "k_proj.weight")).view(M, Hk, D), self.w(p + "k_norm.weight"))
        v = self.lin(h, self.w(p + "v_proj.weight")).view(M, Hk, D)
        if c.has_rope(li):
            cos, sin = self.rope(positions)
            cos, sin = cos[:, None, :], sin[:, None, :]
            q = q * cos + self.rotate_half(q) * sin
            k = k * cos + self.rotate_half(k) * sin
        st.k[li][:, L:L + M] = k.transpose(0, 1)
        st.v[li][:, L:L + M] = v.transpose(0, 1)
        sink = self.w(p + "softmax_offset").float()                       # [Hq]
        W = c.sliding_window if c.is_sliding(li) else 0
        scale = D ** -0.5
        out = torch.empty(M, Hq, D, dtype=h.dtype, device=h.device)
        for i, pa in enumerate(paths):
            pos = L + len(pa) - 1
            lo = max(0, pos - W + 1) if W else 0
            # the visible keys by position: the cache's [lo, L), then the row's path (positions L..pos)
            slots = torch.cat([torch.arange(min(lo, L), L, device=h.device),
                               L + torch.tensor(pa, device=h.device)])
            kpos = torch.cat([torch.arange(min(lo, L), L, device=h.device),
                              L + torch.arange(len(pa), device=h.device)])
            keep = kpos >= lo
            slots = slots[keep]
            ks = st.k[li][:, slots].repeat_interleave(c.gqa, dim=0)      # [Hq, n, D]
            vs = st.v[li][:, slots].repeat_interleave(c.gqa, dim=0)
            s = torch.einsum("hd,hnd->hn", q[i], ks).float() * scale
            a = torch.softmax(torch.cat([s, sink[:, None]], -1), dim=-1)[:, :-1].to(h.dtype)
            out[i] = torch.einsum("hn,hnd->hd", a, vs)
        return self.lin(out.reshape(M, Hq * D), self.w(p + "o_proj.weight"))

    def glu(self, h: torch.Tensor, p: str) -> torch.Tensor:
        return self.lin(F.silu(self.lin(h, self.w(p + "gate_proj.weight"))) * self.lin(h, self.w(p + "up_proj.weight")),
                        self.w(p + "down_proj.weight"))

    def route(self, h: torch.Tensor, li: int):
        """(ids [M, k] int64, weights [M, k] fp32): the sigmoid in fp32, as the published routing."""
        c = self.cfg
        p = self.pre(li) + "mlp."
        s = self.lin(h, self.w(p + "gate.weight")).float().sigmoid()
        bias = self.W.get(p + "gate.expert_bias")
        _, ids = torch.topk(s + bias if bias is not None else s, k=c.num_experts_per_tok, dim=-1)
        wts = torch.gather(s, 1, ids)
        return ids, wts / (wts.sum(dim=-1, keepdim=True) + c.route_eps) * c.route_scale

    def expert(self, x: torch.Tensor, li: int, e: int) -> torch.Tensor:
        if li in self.experts:
            wg, wu, wd = self.experts[li].expert(e)
        else:
            p = f"{self.pre(li)}mlp.experts.{e}."
            wg, wu, wd = self.w(p + "gate_proj.weight"), self.w(p + "up_proj.weight"), self.w(p + "down_proj.weight")
        return self.lin(F.silu(self.lin(x, wg)) * self.lin(x, wu), wd)

    def moe(self, h: torch.Tensor, li: int) -> torch.Tensor:
        ids, wts = self.route(h, li)
        out = torch.zeros_like(h)
        for e in sorted(set(ids.view(-1).tolist())):
            rows, slot = torch.where(ids == e)
            y = self.expert(h[rows], li, e).float() * wts[rows, slot, None]
            out.index_add_(0, rows, y.to(out.dtype))
        if self.cfg.shared_expert_intermediate_size:
            out = out + self.glu(h, self.pre(li) + "mlp.shared_expert.")
        return out

    # ------------------------------------------------------------------------ a block
    @torch.no_grad()
    def rows(self, tokens: torch.Tensor, parents: list[int], st: FlintState) -> torch.Tensor:
        """Logits [M, vocab] (fp32) of a block at the state's length; nothing in `st` moves but the
        block's own KV rows past the length (`commit` keeps a path)."""
        c = self.cfg
        tokens = tokens.view(-1).to(self.device)
        M = tokens.numel()
        if st.length + M > st.cap:
            raise ValueError(f"{st.length + M} rows exceed the state's capacity {st.cap}")
        paths = tree_paths(parents)
        positions = st.length + torch.tensor([len(pa) - 1 for pa in paths], device=self.device)
        self._block = (list(parents), paths)
        h = self.W["embed_tokens.weight"][tokens].to(self.dtype)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            h = h + self.attention(self.rms(h, self.w(p + "input_layernorm.weight")), li, st, paths, positions)
            n = self.rms(h, self.w(p + "post_attention_layernorm.weight"))
            h = h + (self.moe(n, li) if c.is_moe(li) else self.glu(n, p + "mlp."))
        h = self.rms(h, self.w("norm.weight"))
        return self.lin(h, self.head_weight()).float()

    @torch.no_grad()
    def commit(self, st: FlintState, path: list[int]) -> None:
        parents, paths = self._block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        L, n = st.length, len(path)
        idx = torch.tensor(path, dtype=torch.long, device=self.device)
        if any(r != j for j, r in enumerate(path)):
            src, dst = L + idx, L + torch.arange(n, device=self.device)
            for li in st.k:
                st.k[li][:, dst] = st.k[li][:, src]
                st.v[li][:, dst] = st.v[li][:, src]
        st.length = L + n
        self._block = None

    def prefill(self, tokens: torch.Tensor, st: FlintState) -> torch.Tensor:
        n = tokens.numel()
        lg = self.rows(tokens, chain(n), st)
        self.commit(st, list(range(n)))
        return lg

    def decode(self, token: int, st: FlintState) -> torch.Tensor:
        return self.prefill(torch.tensor([int(token)]), st)
