"""LiLiCorr: NVIDIA's block drafter for Nemotron 3.5 Bolt and Super VL (for Bolt
nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, arXiv 2608.20530), as a draft head of the shared
tree drafter (engine/methods/mtp_drafter.py).

Written from the drafter's paper, its config.json and the tensors of its checkpoint
(`LiLiCorrDraftModel`, ModelOpt NVFP4 W4A16, group 16). A draft is three stages over the target's own
embedding and LM head; the tensor names below are the checkpoint's.

  context   one row g per committed target token: the residual rows entering the target layers
            `target_layer_ids` (Bolt: 1, 18, 35, 51, 68, 85; the engine's taps,
            `NemotronHEngine.set_taps`) concatenated, `fc`, RMSNorm `hidden_norm`. Every draft layer's
            context key is RoPE(k_norm(k_proj g)) and its value v_proj g, at the token's position, kept
            in a ring of `swa_window_size` positions
  block     the ids [anchor, mask x (block - 1)] at positions n .. n + block - 1, embedded with the
            target's table. A layer: RMSNorm; the grouped dynamic convolution's input side
            (dflash2._Conv); q, k, v with q_norm and k_norm and RoPE over the whole head; each block row
            attends to the context keys inside its window (query - key < window) and to every row of
            the block, with one learned logit a head (`attention_sink_bias`) added to the softmax's
            denominator; o_proj; the convolution's output side; the residual. The SiLU MLP the same way
            around its own convolution. The final norm, then the target's head on rows 1 .. block - 1
  lattice   a node per (slot, candidate): each slot's top `lilicorr_candidate_topk` tokens. A node's
            state is the token's embedding (projected to the lattice width), the slot's block row
            (`pass_hidden_proj`), an MLP over the LayerNorm of five features of the candidate
            (log p, p, log p - max log p, rank / (k - 1), rank == 0), and slot and rank embeddings;
            then `lilicorr_num_layers` pre-norm transformer layers over all nodes, whose attention
            adds a learned bias a head by the slot distance and another for the same slot; the output
            norm. The anchor's state is `context_proj` of the g before the anchor, `anchor_norm`ed.
            Each node's factor silu(W [x, a, x * a] + b) gives an outgoing and an incoming unit
            vector (`out_head`, `in_head`); slot 1 scores scale * <anchor_out_head(a), in>, slot s
            given slot s - 1's pick scale * <out_pick, in>
  walk      greedy from the left, on the device: slot 1's best, then each slot's best given the
            previous pick; each drafted token's confidence is the softmax of its slot's row of scores

The draft reads its context from the engine's tap ring (`NhState.ring`) by position, not from the rows
the tree drafter hands over (those are the trunk's final rows, which the lookup's bookkeeping counts):
a draft anchored at n computes g, K and V for the committed rows it has not read yet within the window
(n - window + 1 .. n - 1) and drops its ring when the engine's rows below the length stop being this
sequence's (`tap_gen`: a restart, a cache restore). Rows it never read change a draft, never the text:
every drafted token is verified by the trunk, so a draft's arithmetic (bf16, torch) never reaches it.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from engine.drafters.dflash2 import _Conv, _apply_rope, _lin, _rms, _rope_tables
from engine.methods import draft_meet

#: the files of a LiLiCorr checkpoint directory
FILES = ("config.json", "model.safetensors")


class LiLiCorrConfig:
    """The fields of the checkpoint's config.json a draft reads (the lattice's under `dflash_config`).
    A setting the draft does not implement is refused, never ignored."""

    def __init__(self, raw: dict):
        dc = raw.get("dflash_config") or {}
        arch = raw.get("architectures") or []
        if "LiLiCorrDraftModel" not in arch or not dc.get("lilicorr_enabled"):
            raise ValueError(f"not a LiLiCorr draft: architectures {arch}")
        if dc.get("projector_type", "lilicorr") != "lilicorr" or dc.get("causal", False):
            raise ValueError(f"LiLiCorr projector {dc.get('projector_type')!r}, causal {dc.get('causal')}: "
                             "unsupported")
        if raw.get("hidden_act", "silu") != "silu" or raw.get("rope_scaling"):
            raise ValueError("LiLiCorr: silu MLP and plain RoPE only")
        # the draft backbone
        self.hidden_size = int(raw["hidden_size"])
        self.num_hidden_layers = int(raw["num_hidden_layers"])
        self.num_attention_heads = int(raw["num_attention_heads"])
        self.num_key_value_heads = int(raw["num_key_value_heads"])
        self.head_dim = int(raw.get("head_dim") or self.hidden_size // self.num_attention_heads)
        self.intermediate_size = int(raw["intermediate_size"])
        self.rms_norm_eps = float(raw["rms_norm_eps"])
        self.rope_theta = float(raw.get("rope_theta", 10000.0))
        self.vocab_size = int(raw["vocab_size"])
        # the block, the taps, the window and the convolution
        self.block_size = int(raw.get("block_size") or dc["block_size"])
        self.mask_token_id = int(dc["mask_token_id"])
        self.target_layer_ids = [int(t) for t in dc["target_layer_ids"]]
        use_window = dc.get("use_swa", True)
        self.window = int(dc.get("swa_window_size") or raw.get("sliding_window")) if use_window else 0
        self.sinks = bool(dc.get("attention_sink_bias", raw.get("attention_sink_bias", False)))
        self.conv_kernel_size = int(dc.get("conv_kernel_size") or 0)
        self.conv_group_size = int(dc.get("conv_group_size") or 0)
        # the lattice
        self.topk = int(dc["lilicorr_candidate_topk"])
        self.lc_hidden = int(dc.get("lilicorr_hidden_size") or self.hidden_size)
        self.lc_layers = int(dc["lilicorr_num_layers"])
        self.lc_heads = int(dc["lilicorr_num_heads"])
        self.lc_mlp = int(self.lc_hidden * float(dc.get("lilicorr_mlp_ratio", 2.0)))
        self.factor_dim = int(dc.get("lilicorr_factor_dim") or self.lc_hidden)
        self.vector_eps = float(dc.get("lilicorr_vector_eps", 1e-4))
        self.logit_scale = float(dc.get("lilicorr_logit_scale", 1.0))
        # Super VL's drafter has an LM head of its own (NVFP4, the target's vocabulary); Bolt's uses the target's
        self.own_head = bool(raw.get("has_own_lm_head", False))
        if self.own_head and int(raw.get("draft_vocab_size") or self.vocab_size) != self.vocab_size:
            raise ValueError("LiLiCorr with its own head over a draft vocabulary: unsupported")
        if not self.window:
            raise ValueError("LiLiCorr without a sliding window: the context ring needs one")
        if not (1 <= self.topk <= 16) or self.topk & (self.topk - 1):
            raise ValueError(f"LiLiCorr candidate_topk {self.topk}: a power of two up to 16")
        if (self.conv_kernel_size == 0) != (self.conv_group_size == 0):
            raise ValueError("LiLiCorr convolution: kernel and group size together")

    @property
    def slots(self) -> int:
        """Drafted tokens a block: every row but the anchor's."""
        return self.block_size - 1

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size if self.conv_group_size else 0


def load_config(path: str) -> LiLiCorrConfig:
    with open(os.path.join(path, "config.json")) as f:
        return LiLiCorrConfig(json.load(f))


def missing_files(path: str | None) -> list[str]:
    if not path or not os.path.isdir(path):
        return [path or "(no path)"]
    return [name for name in FILES if not os.path.isfile(os.path.join(path, name))]


def load_weights(path: str, device: str = "cuda") -> dict:
    """The checkpoint's tensors: a ModelOpt NVFP4 linear (`weight` uint8, `weight_scale` e4m3,
    `weight_scale_2` fp32) as one `NVFP4Block` under its weight's name, every other tensor bf16."""
    from safetensors import safe_open

    from tools.nvfp4_linear import NVFP4Block
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu") as f:
        stored = {name: f.get_tensor(name) for name in f.keys()}
    out = {}
    for name, t in stored.items():
        if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue                                                      # read with their weight
        if t.dtype != torch.uint8:
            out[name] = t.to(device=device, dtype=torch.bfloat16)
            continue
        stem = name[:-len("weight")]
        scale, scale2 = stored.get(stem + "weight_scale"), stored.get(stem + "weight_scale_2")
        if scale is None or scale2 is None:
            raise ValueError(f"{name}: an NVFP4 weight without its scales")
        out[name] = NVFP4Block(t.to(device), scale.to(device), scale2.float())
    return out


#: the largest activation the NVFP4 kernels take as it is: they multiply in fp16 (max 65,504)
F16_SAFE = 2.0 ** 15


def _lin_wide(x: torch.Tensor, w) -> torch.Tensor:
    """`_lin` for an input that can leave fp16's range (the taps, the MLPs' inputs: the draft's
    activations reach 1e5 where the trunk's normed ones never do): an NVFP4 weight's kernel multiplies
    in fp16, so a row whose largest value is past F16_SAFE is scaled down by a power of two first and
    the product scaled back (both exact); a bf16 weight runs as it is."""
    if isinstance(w, torch.Tensor):
        return _lin(x, w)
    s = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).float() / F16_SAFE)).clamp_min(0.0))
    return (_lin((x / s).to(x.dtype), w) * s).to(x.dtype)


def _linb(x: torch.Tensor, W: dict, name: str) -> torch.Tensor:
    """The checkpoint's linear `name`, plus its bias when it has one."""
    y = _lin(x, W[name + ".weight"])
    bias = W.get(name + ".bias")
    return y if bias is None else y + bias


def _unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    """x over its L2 norm along the last axis (the norm floored at eps)."""
    return F.normalize(x, dim=-1, eps=eps)


class LiLiCorrModel:
    """A draft's arithmetic over the checkpoint's tensors `W`, the target's embedding `embed`
    [vocab, H] and LM head `head` (rows [M, H] -> logits [M, vocab], fp32). It knows nothing of the
    engine, so tests and tools drive it directly."""

    def __init__(self, cfg: LiLiCorrConfig, W: dict, embed: torch.Tensor, head):
        self.cfg, self.W, self.embed, self.head = cfg, W, embed, head
        L = cfg.num_hidden_layers
        self.convs = [self._convs_of(i) for i in range(L)]
        # a context row's K and V for every layer come out of one product with the stacked weights
        self.kv_w = torch.cat([W[f"layers.{i}.self_attn.{side}_proj.weight"]
                               for i in range(L) for side in ("k", "v")]).contiguous()
        self.sink = [W[f"layers.{i}.self_attn.attention_sink_bias"].float() if cfg.sinks else None
                     for i in range(L)]
        self.attn_bias = self._lattice_bias()
        k = cfg.topk
        self.rank_frac = (torch.arange(k, dtype=torch.float32) / max(k - 1, 1)).to(self.device)
        self.is_top1 = (torch.arange(k) == 0).float().to(self.device)
        # the factor input [x, a, x * a] is one weight of three column blocks; each block multiplies
        # its own part
        w3, h = W["lilicorr.factor_input_proj.weight"], cfg.lc_hidden
        self.factor_w = tuple(w3[:, j * h:(j + 1) * h].contiguous() for j in range(3))

    @property
    def device(self) -> torch.device:
        return self.embed.device

    def _convs_of(self, i: int):
        """Layer i's two convolutions (attention side, MLP side), or None when the draft has none."""
        c, W = self.cfg, self.W
        if not c.conv_kernel_size:
            return None
        return tuple(_Conv(W[f"layers.{i}.{side}.base_kernel"], W[f"layers.{i}.{side}.kernel_projection.weight"],
                           c.conv_kernel_size, c.num_groups, c.conv_group_size)
                     for side in ("attention_conv", "mlp_conv"))

    def _lattice_bias(self) -> torch.Tensor:
        """The lattice attention's additive bias [heads, nodes, nodes]: a head's learned value for the
        slot distance between the two nodes (`relative_slot_bias`, distance + block - 1 its index),
        plus `same_slot_bias` where both nodes are in one slot."""
        c, W = self.cfg, self.W
        node_slot = torch.arange(c.slots, device=self.device).repeat_interleave(c.topk)
        dist = node_slot[:, None] - node_slot[None, :]
        by_dist = W["lilicorr.relative_slot_bias"][:, dist + (c.block_size - 1)]
        same = (dist == 0).to(by_dist.dtype)
        return (by_dist + same[None] * W["lilicorr.same_slot_bias"].view(-1, 1, 1)).contiguous()

    # ---- context ---------------------------------------------------------------------------------
    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Target tap rows [n, taps, H] at `positions` -> (g [n, H], K, V [layers, kv heads, n, head_dim]),
        K normed and rotated, ready for the block."""
        c, W = self.cfg, self.W
        n, L, eps = taps.shape[0], c.num_hidden_layers, c.rms_norm_eps
        g = _rms(_lin_wide(taps.reshape(n, -1), W["fc.weight"]), W["hidden_norm.weight"], eps)
        kv = F.linear(g, self.kv_w).view(n, L, 2, c.num_key_value_heads, c.head_dim)
        keys = torch.stack([_rms(kv[:, i, 0], W[f"layers.{i}.self_attn.k_norm.weight"], eps)
                            for i in range(L)])                                      # [L, n, kv, D]
        cos, sin = _rope_tables(positions, c.head_dim, c.rope_theta, g.dtype)
        keys = _apply_rope(keys.transpose(1, 2), cos, sin)                            # [L, kv, n, D]
        return g, keys, kv[:, :, 1].permute(1, 2, 0, 3)

    # ---- block -----------------------------------------------------------------------------------
    def block(self, anchor: torch.Tensor, pos0: torch.Tensor, ck: torch.Tensor, cv: torch.Tensor,
              ctx_ok: torch.Tensor) -> torch.Tensor:
        """The rows [anchor, mask, ...] at positions pos0 .. pos0 + block - 1, attending to the context
        keys `ck` / values `cv` [layers, kv heads, C, D] where `ctx_ok` [block, C] (bool) allows it and to
        every block row. The final normed rows [block, H]."""
        c, W = self.cfg, self.W
        B, eps, dev = c.block_size, c.rms_norm_eps, self.device
        ids = torch.full((B,), c.mask_token_id, dtype=torch.long, device=dev)
        ids[:1] = anchor
        h = self.embed.index_select(0, ids)
        row = torch.arange(B, device=dev)
        cos, sin = _rope_tables(pos0 + row, c.head_dim, c.rope_theta, h.dtype)
        # additive mask over [context keys | block keys]: 0 where a row may look, -inf elsewhere
        allowed = torch.cat([ctx_ok, torch.ones(B, B, dtype=torch.bool, device=dev)], 1)
        mask = torch.zeros(allowed.shape, dtype=torch.float32, device=dev).masked_fill_(~allowed, float("-inf"))
        for i in range(c.num_hidden_layers):
            h = h + self._attention_part(i, h, row, cos, sin, ck[i], cv[i], mask)
            h = h + self._mlp_part(i, h, row)
        return _rms(h, W["norm.weight"], eps)

    def _attention_part(self, i, h, row, cos, sin, ck, cv, mask) -> torch.Tensor:
        """Layer i's attention sublayer on the block rows `h`, inside its convolution: its residual."""
        c, W, p = self.cfg, self.W, f"layers.{i}.self_attn"
        B, D, eps = h.shape[0], c.head_dim, c.rms_norm_eps
        conv = self.convs[i][0] if self.convs[i] is not None else None
        x = _rms(h, W[f"layers.{i}.input_layernorm.weight"], eps)
        out_coef = None
        if conv is not None:
            x, out_coef = conv.prepare(x, row)
        q = _lin(x, W[f"{p}.q_proj.weight"]).view(B, c.num_attention_heads, D)
        k = _lin(x, W[f"{p}.k_proj.weight"]).view(B, c.num_key_value_heads, D)
        v = _lin(x, W[f"{p}.v_proj.weight"]).view(B, c.num_key_value_heads, D)
        q = _apply_rope(_rms(q, W[f"{p}.q_norm.weight"], eps).transpose(0, 1)[None], cos, sin)[0]
        k = _apply_rope(_rms(k, W[f"{p}.k_norm.weight"], eps).transpose(0, 1)[None], cos, sin)[0]
        o = self._attend(q, torch.cat([ck, k], 1), torch.cat([cv, v.transpose(0, 1)], 1), mask, self.sink[i])
        o = _lin(o, W[f"{p}.o_proj.weight"])
        return o if conv is None else conv.finish(o, out_coef, row)

    def _mlp_part(self, i, h, row) -> torch.Tensor:
        """Layer i's SiLU MLP on the block rows `h`, inside its convolution: its residual."""
        W, p = self.W, f"layers.{i}.mlp"
        conv = self.convs[i][1] if self.convs[i] is not None else None
        x = _rms(h, W[f"layers.{i}.post_attention_layernorm.weight"], self.cfg.rms_norm_eps)
        out_coef = None
        if conv is not None:
            x, out_coef = conv.prepare(x, row)
        gate, up = _lin_wide(x, W[f"{p}.gate_proj.weight"]), _lin_wide(x, W[f"{p}.up_proj.weight"])
        y = _lin_wide(F.silu(gate) * up, W[f"{p}.down_proj.weight"])
        return y if conv is None else conv.finish(y, out_coef, row)

    def _attend(self, q, keys, vals, mask, sink) -> torch.Tensor:
        """Grouped-query attention in fp32: q [heads, B, D] over keys / vals [kv heads, K, D], the
        additive `mask` [B, K], and when `sink` [heads] is given one extra logit a head that only
        enlarges the softmax's denominator. -> [B, heads * D] in q's dtype."""
        nh, B, D = q.shape
        G, K = keys.shape[0], keys.shape[1]
        per = nh // G
        logits = torch.einsum("gqbd,gkd->gqbk", q.float().view(-1, per, B, D), keys.float()) / math.sqrt(D)
        logits = logits.reshape(nh, B, -1) + mask
        if sink is not None:
            logits = torch.cat([logits, sink.view(nh, 1, 1).expand(nh, B, 1)], -1)
        w = torch.softmax(logits, -1)[..., :K]
        o = torch.einsum("gqbk,gkd->gqbd", w.view(-1, per, B, K), vals.float())
        return o.reshape(nh, B, D).transpose(0, 1).reshape(B, nh * D).to(q.dtype)

    # ---- lattice ---------------------------------------------------------------------------------
    def candidates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each slot's top-k ids and their log-probabilities (descending) from its logits [slots, vocab]."""
        best, ids = torch.topk(logits.float(), self.cfg.topk, dim=-1)
        return ids, best - torch.logsumexp(logits.float(), -1, keepdim=True)

    def lattice(self, ids: torch.Tensor, logp: torch.Tensor, pass_hidden: torch.Tensor, anchor_g: torch.Tensor,
                anchor_ok: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """(start [k], pairs [slots - 1, k, k]) in logit units for one sequence. ids / logp [slots, k],
        pass_hidden [slots, H] (the block's final rows 1 ..), anchor_g [H] (g of the token before the
        anchor; `anchor_ok` 0 when that row is not this sequence's)."""
        c = self.cfg
        S, k = ids.shape[0], c.topk
        dt = self.W["lilicorr.slot_embedding"].dtype
        x = self._nodes(ids, logp, pass_hidden)                                       # [S * k, Hc]
        a = _linb(anchor_g.view(1, -1).to(dt), self.W, "lilicorr.context_proj") * anchor_ok
        x = self._encode(x, S * k).view(S, k, c.lc_hidden)
        return self._scores(x, a)

    def _nodes(self, ids, logp, pass_hidden) -> torch.Tensor:
        """The nodes' input states [S * k, Hc]: candidate token, block row, candidate features, slot
        and rank."""
        c, W = self.cfg, self.W
        S, k, Hc = ids.shape[0], c.topk, c.lc_hidden
        dt = W["lilicorr.slot_embedding"].dtype
        tok = self.embed.index_select(0, ids.reshape(-1)).to(dt)
        x = _linb(tok, W, "lilicorr.token_proj") if c.lc_hidden != c.hidden_size else tok
        x = x.view(S, k, Hc) + _linb(pass_hidden.to(dt), W, "lilicorr.pass_hidden_proj").unsqueeze(1)
        lp = logp.float()
        feats = torch.stack([lp, lp.exp(), lp - lp.max(-1, keepdim=True).values,
                             self.rank_frac.expand_as(lp), self.is_top1.expand_as(lp)], -1).to(dt)
        feats = F.layer_norm(feats, (5,), W["lilicorr.feature_norm.weight"], W["lilicorr.feature_norm.bias"])
        x = x + _linb(F.silu(_linb(feats, W, "lilicorr.feature_mlp.up_proj")), W, "lilicorr.feature_mlp.down_proj")
        x = x + W["lilicorr.slot_embedding"][0, 0, :S] + W["lilicorr.rank_embedding"][0, 0]
        return x.reshape(S * k, Hc)

    def _encode(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """The lattice's pre-norm transformer layers over the n nodes, then its output norm."""
        c, W = self.cfg, self.W
        eps, heads = c.rms_norm_eps, c.lc_heads
        hd = c.lc_hidden // heads
        bias = self.attn_bias[:, :n, :n]
        for j in range(c.lc_layers):
            p = f"lilicorr.layers.{j}"
            y = _rms(x, W[f"{p}.attn_norm.weight"], eps)
            qkv = F.linear(y, W[f"{p}.attn.in_proj_weight"], W[f"{p}.attn.in_proj_bias"])
            q, kk, v = (t.view(n, heads, hd).transpose(0, 1) for t in qkv.chunk(3, -1))
            o = F.scaled_dot_product_attention(q[None], kk[None], v[None], attn_mask=bias[None])[0]
            x = x + _linb(o.transpose(0, 1).reshape(n, c.lc_hidden), W, f"{p}.attn.out_proj")
            y = F.silu(_linb(_rms(x, W[f"{p}.mlp_norm.weight"], eps), W, f"{p}.mlp.up_proj"))
            x = x + _lin_wide(y, W[f"{p}.mlp.down_proj.weight"]) + W[f"{p}.mlp.down_proj.bias"]
        return _rms(x, W["lilicorr.output_norm.weight"], eps)

    def _scores(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The nodes' factor vectors and the scores: slot 1 against the anchor, slot s against slot s - 1."""
        c, W = self.cfg, self.W
        a = _rms(a, W["lilicorr.anchor_norm.weight"], c.rms_norm_eps)
        w_x, w_a, w_xa = self.factor_w
        pre = F.linear(x, w_x, W["lilicorr.factor_input_proj.bias"]) + F.linear(a, w_a) + F.linear(x * a, w_xa)
        factor = F.silu(pre)
        outgoing = _unit(_linb(factor, W, "lilicorr.out_head"), c.vector_eps)
        incoming = _unit(_linb(factor, W, "lilicorr.in_head"), c.vector_eps)
        anchor_out = _unit(_linb(a, W, "lilicorr.anchor_out_head"), c.vector_eps)[0]
        start = (incoming[0] @ anchor_out).float() * c.logit_scale
        pairs = torch.matmul(outgoing[:-1], incoming[1:].transpose(-1, -2)).float() * c.logit_scale
        return start, pairs

    @staticmethod
    def walk(ids: torch.Tensor, start: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """The greedy walk on the device: [slots, 4] fp32 rows (token, the softmax of its score over its
        slot's row, the row's second token, its softmax). Slot 1's row is `start`, every later slot's the
        row of `pairs` its predecessor's pick selects."""
        rows, pick = [start], start.argmax().view(1)
        for s in range(ids.shape[0] - 1):
            nxt = pairs[s].index_select(0, pick)[0]
            pick = nxt.argmax().view(1)
            rows.append(nxt)
        prob = torch.softmax(torch.stack(rows), -1)                                   # [S, k]
        two = torch.topk(prob, 2, -1)
        tok = ids.gather(1, two.indices)
        return torch.stack([tok[:, 0].float(), two.values[:, 0], tok[:, 1].float(), two.values[:, 1]], 1)

    def draft(self, anchor, pos0, ck, cv, ctx_ok, anchor_g, anchor_ok=1.0) -> torch.Tensor:
        """The block, the target's head on its rows 1 .., the candidates, the lattice and the walk:
        [slots, 4] on the device (`walk`)."""
        rows = self.block(anchor, pos0, ck, cv, ctx_ok)[1:]
        ids, logp = self.candidates(self.head(rows))
        start, pairs = self.lattice(ids, logp, rows, anchor_g, anchor_ok)
        return self.walk(ids, start, pairs)

    # ---- several sequences at once (SPD-233) -----------------------------------------------------
    def draft_multi(self, reqs: list) -> torch.Tensor:
        """`draft` of B sequences in one pass, reqs [(anchor, pos0, ck, cv, ctx_ok, anchor_g, anchor_ok)]
        as `draft` takes them: every projection, the head and the lattice's layers over all B blocks' rows
        at once (the weights read once), the attention and the walk per sequence on its own rows, context
        and mask. [B, slots, 4] (`walk` a sequence)."""
        c = self.cfg
        B, S = len(reqs), c.block_size - 1
        h = self.block_multi(reqs)
        rows = h.view(B, c.block_size, -1)[:, 1:]                                      # [B, S, H]
        ids, logp = self.candidates(self.head(rows.reshape(B * S, -1)))
        ids, logp = ids.view(B, S, -1), logp.view(B, S, -1)
        start, pairs = self.lattice_multi(ids, logp, rows, torch.stack([r[5] for r in reqs]),
                                          torch.stack([r[6].reshape(()) for r in reqs]))
        return self.walk_multi(ids, start, pairs)

    def block_multi(self, reqs: list) -> torch.Tensor:
        """`block` of B sequences: their rows stacked [B * block, H] (a sequence's rows adjacent, the
        convolutions' taps stopping at each block's first row), the final normed rows."""
        c, W = self.cfg, self.W
        B, blk, eps, dev = len(reqs), c.block_size, c.rms_norm_eps, self.device
        ids = torch.full((B, blk), c.mask_token_id, dtype=torch.long, device=dev)
        ids[:, 0] = torch.cat([r[0].view(1) for r in reqs])
        h = self.embed.index_select(0, ids.view(-1))
        rb = torch.arange(blk, device=dev)
        row = rb.repeat(B)
        pos = torch.stack([r[1].reshape(()) for r in reqs]).view(B, 1) + rb
        cos, sin = _rope_tables(pos.view(-1), c.head_dim, c.rope_theta, h.dtype)
        ones = torch.ones(blk, blk, dtype=torch.bool, device=dev)
        masks = torch.stack([torch.zeros(blk, r[4].shape[1] + blk, dtype=torch.float32, device=dev).masked_fill_(
            ~torch.cat([r[4], ones], 1), float("-inf")) for r in reqs])                  # [B, blk, C + blk]
        for i in range(c.num_hidden_layers):
            h = h + self._attention_multi(i, h, row, cos, sin, [r[2][i] for r in reqs], [r[3][i] for r in reqs],
                                          masks)
            h = h + self._mlp_part(i, h, row)
        return _rms(h, W["norm.weight"], eps)

    def _attention_multi(self, i, h, row, cos, sin, cks, cvs, masks) -> torch.Tensor:
        """`_attention_part` over B blocks' rows: the projections and the convolution on all rows, each
        block's attention over its own context and its own rows."""
        c, W, p = self.cfg, self.W, f"layers.{i}.self_attn"
        T, D, eps = h.shape[0], c.head_dim, c.rms_norm_eps
        B = len(cks)
        blk = T // B
        conv = self.convs[i][0] if self.convs[i] is not None else None
        x = _rms(h, W[f"layers.{i}.input_layernorm.weight"], eps)
        out_coef = None
        if conv is not None:
            x, out_coef = conv.prepare(x, row)
        q = _lin(x, W[f"{p}.q_proj.weight"]).view(T, c.num_attention_heads, D)
        k = _lin(x, W[f"{p}.k_proj.weight"]).view(T, c.num_key_value_heads, D)
        v = _lin(x, W[f"{p}.v_proj.weight"]).view(T, c.num_key_value_heads, D)
        q = _apply_rope(_rms(q, W[f"{p}.q_norm.weight"], eps).transpose(0, 1)[None], cos, sin)[0]   # [nh, T, D]
        k = _apply_rope(_rms(k, W[f"{p}.k_norm.weight"], eps).transpose(0, 1)[None], cos, sin)[0]   # [kv, T, D]
        vt = v.transpose(0, 1)
        keys = torch.stack([torch.cat([cks[b], k[:, b * blk:(b + 1) * blk]], 1) for b in range(B)])
        vals = torch.stack([torch.cat([cvs[b], vt[:, b * blk:(b + 1) * blk]], 1) for b in range(B)])
        qb = q.view(q.shape[0], B, blk, D).transpose(0, 1)                                   # [B, nh, blk, D]
        o = self._attend_multi(qb, keys, vals, masks, self.sink[i])
        o = _lin(o, W[f"{p}.o_proj.weight"])
        return o if conv is None else conv.finish(o, out_coef, row)

    @staticmethod
    def _attend_multi(q, keys, vals, masks, sink) -> torch.Tensor:
        """`_attend` of B blocks: q [B, heads, blk, D] over keys / vals [B, kv heads, K, D] with masks
        [B, blk, K]. -> [B * blk, heads * D]."""
        B, nh, blk, D = q.shape
        G, K = keys.shape[1], keys.shape[2]
        per = nh // G
        logits = torch.einsum("bgqtd,bgkd->bgqtk", q.float().reshape(B, G, per, blk, D), keys.float()) / math.sqrt(D)
        logits = logits.reshape(B, nh, blk, K) + masks[:, None]
        if sink is not None:
            logits = torch.cat([logits, sink.view(1, nh, 1, 1).expand(B, nh, blk, 1)], -1)
        w = torch.softmax(logits, -1)[..., :K]
        o = torch.einsum("bgqtk,bgkd->bgqtd", w.reshape(B, G, per, blk, K), vals.float())
        return o.reshape(B, nh, blk, D).transpose(1, 2).reshape(B * blk, nh * D).to(q.dtype)

    def lattice_multi(self, ids, logp, pass_hidden, anchor_g, anchor_ok) -> tuple[torch.Tensor, torch.Tensor]:
        """`lattice` of B sequences: ids / logp [B, slots, k], pass_hidden [B, slots, H], anchor_g [B, H],
        anchor_ok [B] -> (start [B, k], pairs [B, slots - 1, k, k])."""
        c, W = self.cfg, self.W
        B, S, k = ids.shape
        dt = W["lilicorr.slot_embedding"].dtype
        tok = self.embed.index_select(0, ids.reshape(-1)).to(dt)
        x = _linb(tok, W, "lilicorr.token_proj") if c.lc_hidden != c.hidden_size else tok
        x = x.view(B, S, k, c.lc_hidden) + _linb(pass_hidden.to(dt), W, "lilicorr.pass_hidden_proj").unsqueeze(2)
        lp = logp.float()
        feats = torch.stack([lp, lp.exp(), lp - lp.max(-1, keepdim=True).values,
                             self.rank_frac.expand_as(lp), self.is_top1.expand_as(lp)], -1).to(dt)
        feats = F.layer_norm(feats, (5,), W["lilicorr.feature_norm.weight"], W["lilicorr.feature_norm.bias"])
        x = x + _linb(F.silu(_linb(feats, W, "lilicorr.feature_mlp.up_proj")), W, "lilicorr.feature_mlp.down_proj")
        x = x + W["lilicorr.slot_embedding"][0, 0, :S] + W["lilicorr.rank_embedding"][0, 0]
        a = _linb(anchor_g.view(B, 1, -1).to(dt), W, "lilicorr.context_proj")
        a = a * anchor_ok.view(B, 1, 1).to(a.dtype)
        x = self._encode_multi(x.reshape(B, S * k, c.lc_hidden)).view(B, S, k, c.lc_hidden)
        return self._scores_multi(x, a)

    def _encode_multi(self, x: torch.Tensor) -> torch.Tensor:
        """`_encode` of B sequences' nodes x [B, n, Hc]."""
        c, W = self.cfg, self.W
        B, n = x.shape[0], x.shape[1]
        eps, heads = c.rms_norm_eps, c.lc_heads
        hd = c.lc_hidden // heads
        bias = self.attn_bias[:, :n, :n]
        for j in range(c.lc_layers):
            p = f"lilicorr.layers.{j}"
            y = _rms(x, W[f"{p}.attn_norm.weight"], eps)
            qkv = F.linear(y, W[f"{p}.attn.in_proj_weight"], W[f"{p}.attn.in_proj_bias"])
            q, kk, v = (t.reshape(B, n, heads, hd).transpose(1, 2) for t in qkv.chunk(3, -1))
            o = F.scaled_dot_product_attention(q, kk, v, attn_mask=bias[None])
            x = x + _linb(o.transpose(1, 2).reshape(B, n, c.lc_hidden), W, f"{p}.attn.out_proj")
            y = F.silu(_linb(_rms(x, W[f"{p}.mlp_norm.weight"], eps), W, f"{p}.mlp.up_proj"))
            x = x + _lin_wide(y, W[f"{p}.mlp.down_proj.weight"]) + W[f"{p}.mlp.down_proj.bias"]
        return _rms(x, W["lilicorr.output_norm.weight"], eps)

    def _scores_multi(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`_scores` of B sequences: x [B, slots, k, Hc], a [B, 1, Hc]."""
        c, W = self.cfg, self.W
        B = x.shape[0]
        a = _rms(a, W["lilicorr.anchor_norm.weight"], c.rms_norm_eps)
        w_x, w_a, w_xa = self.factor_w
        a4 = a.view(B, 1, 1, -1)
        pre = F.linear(x, w_x, W["lilicorr.factor_input_proj.bias"]) + F.linear(a4, w_a) + F.linear(x * a4, w_xa)
        factor = F.silu(pre)
        outgoing = _unit(_linb(factor, W, "lilicorr.out_head"), c.vector_eps)
        incoming = _unit(_linb(factor, W, "lilicorr.in_head"), c.vector_eps)
        anchor_out = _unit(_linb(a, W, "lilicorr.anchor_out_head"), c.vector_eps)[:, 0]           # [B, D2]
        start = (incoming[:, 0] @ anchor_out.unsqueeze(-1)).squeeze(-1).float() * c.logit_scale
        pairs = torch.matmul(outgoing[:, :-1], incoming[:, 1:].transpose(-1, -2)).float() * c.logit_scale
        return start, pairs

    @staticmethod
    def walk_multi(ids: torch.Tensor, start: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """`walk` of B sequences: ids [B, slots, k], start [B, k], pairs [B, slots - 1, k, k] -> [B, slots, 4]."""
        B = ids.shape[0]
        ar = torch.arange(B, device=ids.device)
        rows, pick = [start], start.argmax(-1)
        for s in range(ids.shape[1] - 1):
            nxt = pairs[:, s][ar, pick]
            pick = nxt.argmax(-1)
            rows.append(nxt)
        prob = torch.softmax(torch.stack(rows, 1), -1)                                # [B, S, k]
        two = torch.topk(prob, 2, -1)
        tok = ids.gather(2, two.indices)
        return torch.stack([tok[..., 0].float(), two.values[..., 0], tok[..., 1].float(), two.values[..., 1]], -1)


#: SPD-233: a joint round's LiLiCorr drafts in one packed pass (the round runner's draft meeting,
#: engine/methods/draft_meet.py): the lanes' context rows in one product, then one graph for the lanes' blocks,
#: the head over all their rows, the lattices and the walks, so the drafter's weights are read once a round, not
#: once a lane. Bolt EXL3 4.0, 8 lanes, 2026-10-26: the round's draft 80.9 -> 17.3 ms; served C8 T1 56.5 -> 75.5 /
#: 74.7 tok/s. "off": each lane drafts from its own graph
PACK = os.environ.get("QSE_NH_LILICORR_PACK", "on") != "off"
#: packed graphs kept (one a set of lanes, the least recently replayed reset first)
PACK_GRAPHS = int(os.environ.get("QSE_NH_LILICORR_PACK_GRAPHS", "12") or 12)


class LiLiCorrHead:
    """LiLiCorr as a draft head of the shared tree drafter (`MTPTreeDrafter`): `depth` 7, the rows it
    is handed are counted (the drafter's bookkeeping) and the context is read from the serving engine's
    tap ring. `srv`: the NemotronH serving engine (`st`, `kv.length`, `fn` with the taps set).

    The draft (the block, the head, the lattice, the walk) is one CUDA graph over the head's rings and
    a staged input (the anchor, its position, the anchor row's slot, which context slots each block row
    sees), captured at the first draft (the load-time gate's); the context's rows run eagerly before it
    (their count varies)."""

    def __init__(self, srv, model: LiLiCorrModel, graph: bool = True):
        self.srv, self.m = srv, model
        c = model.cfg
        self.depth = c.slots
        self.W = c.window
        dev, dt = model.device, model.embed.dtype
        self.k = torch.zeros(c.num_hidden_layers, c.num_key_value_heads, self.W, c.head_dim, dtype=dt, device=dev)
        self.v = torch.zeros_like(self.k)
        self.g = torch.zeros(self.W, c.hidden_size, dtype=dt, device=dev)
        self.tag = torch.full((self.W,), -1, dtype=torch.long)       # the position a ring slot holds (host)
        self.fed = 0                                                  # every position below is in the ring
        self.gen = None
        self.st = type("LiLiCorrState", (), {"length": 0})()
        self.captures = 0
        self._out = None
        self._n = 0
        self.stats = {"drafts": 0, "context_rows": 0}
        # the staged input: int64 (anchor, position, anchor row's slot), fp32 (the anchor row is this
        # sequence's), bool [block, window] (the context slots a block row sees)
        pin = dev.type == "cuda"
        self._h = {"i": torch.zeros(3, dtype=torch.long, pin_memory=pin), "f": torch.zeros(1, pin_memory=pin),
                   "ok": torch.zeros(c.block_size, self.W, dtype=torch.bool, pin_memory=pin)}
        self._d = {k: torch.zeros_like(v, device=dev) for k, v in self._h.items()}
        self._copied = torch.cuda.Event() if pin else None
        self.graph = None if not (graph and pin) else False           # False: not captured yet

    # --- the drafter's interface ------------------------------------------------------------------
    def reset(self) -> None:
        self.st.length = 0

    def prime(self, rows, nexts) -> None:
        self.st.length += len(nexts)

    def views(self) -> list:
        return []

    def resume(self, n: int) -> None:
        self.st.length = int(n)

    def capture_all(self) -> None:
        pass

    @torch.no_grad()
    def launch(self, rows, nexts: list[int]) -> None:
        """The draft anchored at the round's anchor (`nexts[-1]`, at the engine's length n): the
        context brought up to n - 1, the block, the lattice and the walk, left on the device for
        `read`."""
        self._n = len(nexts)
        if PACK and draft_meet.MEET is not None and draft_meet.MEET(self):
            # SPD-233: the round runner drafts this lane with the round's other lanes (`run_packed`), before the read
            self._pend = (int(nexts[-1]), int(self.srv.kv.length))
            return
        self._out = self.draft_at(int(nexts[-1]), int(self.srv.kv.length))

    @property
    def fn(self):
        """The engine whose heads draft together (the round runner packs the heads of one engine)."""
        return self.srv.fn

    def read(self) -> tuple[list, list]:
        self.st.length += self._n
        out = self._out.tolist()
        return [[int(r[0]), int(r[2])] for r in out], [[r[1], r[3]] for r in out]

    # --- the context and the draft ----------------------------------------------------------------
    def draft_at(self, anchor: int, n: int) -> torch.Tensor:
        """[slots, 4] on the device (`LiLiCorrModel.walk`) for `anchor` at position n."""
        self.feed(n)
        self._stage(anchor, n)
        if self.graph is None:
            return self._draft()
        if self.graph is False:
            self._capture()
        self.graph.replay()
        return self._gout

    def _stage(self, anchor: int, n: int) -> None:
        """The draft's staged input: the anchor, its position, the anchor row's slot, whether that row is
        this sequence's, the context slots each block row sees."""
        tag = self.tag
        ok = (tag >= max(n - self.W + 1, 0)) & (tag < n)              # a slot this draft may read at all
        # row i (position n + i) sees a context key at p when n + i - p < window
        pos = torch.arange(self.m.cfg.block_size)[:, None] + n
        prev = (n - 1) % self.W
        if self._copied is not None:
            self._copied.synchronize()                                # the last draft's copy has left
        h = self._h
        h["ok"].copy_(ok[None, :] & (pos - tag[None, :] < self.W))
        h["i"][0], h["i"][1], h["i"][2] = anchor, n, prev
        h["f"][0] = 1.0 if n > 0 and int(tag[prev]) == n - 1 else 0.0
        for k, t in h.items():
            self._d[k].copy_(t, non_blocking=True)
        if self._copied is not None:
            self._copied.record()
        self.stats["drafts"] += 1

    def _req(self) -> tuple:
        """The draft's inputs as `LiLiCorrModel.draft` takes them, from the staged buffers and the rings."""
        d = self._d
        return (d["i"][:1], d["i"][1], self.k, self.v, d["ok"], self.g.index_select(0, d["i"][2:])[0], d["f"][0])

    def _draft(self) -> torch.Tensor:
        return self.m.draft(*self._req())

    @staticmethod
    @torch.no_grad()
    def run_packed(heads: list) -> None:
        """SPD-233: the staged drafts of these heads (one engine, one checkpoint) in one pass: their context
        rows in one product (`feed_multi`), then one graph a set of heads (`_PackedGraph`); each head's
        draft is left for its `read`."""
        LiLiCorrHead.feed_multi(heads, [h._pend[1] for h in heads])
        for h in heads:
            h._stage(*h._pend)
        m = heads[0].m
        if not m.device.type == "cuda":
            out = m.draft_multi([h._req() for h in heads])
        else:
            reg = m.__dict__.setdefault("_packed", {})
            key = tuple(id(h) for h in heads)
            g = reg.pop(key, None)
            if g is None:
                g = _PackedGraph(tuple(heads))
                g.capture()
                for h in heads:
                    h.captures += 1             # this round's draft time is the capture's (the drafter skips it)
            reg[key] = g
            while len(reg) > PACK_GRAPHS:
                reg.pop(next(iter(reg))).release()
            g.graph.replay()
            out = g.out
        for b, h in enumerate(heads):
            h._out = out[b]

    @staticmethod
    def feed_multi(heads: list, ns: list) -> None:
        """`feed` of several heads over one checkpoint: every head's new context rows in one `context` call."""
        jobs = []
        for h, n in zip(heads, ns):
            st = h.srv.st
            if (id(st), st.tap_gen) != h.gen:
                h.gen = (id(st), st.tap_gen)
                h.tag.fill_(-1)
                h.fed = 0
            R = st.ring.shape[0]
            a = max(h.fed, n - h.W + 1, st.tap_from, n - R, 0)
            if a < n:
                jobs.append((h, a, n, st.ring, R))
            h.fed = max(h.fed, n)
        if not jobs:
            return
        m = heads[0].m
        dev = m.device
        taps = torch.cat([ring.index_select(0, (torch.arange(a, n) % R).to(dev)) for _, a, n, ring, R in jobs])
        g, k, v = m.context(taps, torch.cat([torch.arange(a, n) for _, a, n, _, _ in jobs]).to(dev))
        o = 0
        for h, a, n, _, _ in jobs:
            pos = torch.arange(a, n)
            slot = (pos % h.W).to(dev)
            h.g.index_copy_(0, slot, g[o:o + n - a])
            h.k.index_copy_(2, slot, k[:, :, o:o + n - a])
            h.v.index_copy_(2, slot, v[:, :, o:o + n - a])
            h.tag[pos % h.W] = pos
            h.stats["context_rows"] += n - a
            o += n - a

    def _capture(self) -> None:
        from engine import capture_lock
        # SPD-223: a capture stream of the head's own: the draft's cuBLAS products bake that stream's workspace,
        # which a reset of another owner's graph on a shared stream (a joint decode graph's eviction) would free
        s = self._stream = getattr(self, "_stream", None) or capture_lock.lease_stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._draft()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g, stream=s):
            self._gout = self._draft()
        torch.cuda.synchronize()
        self.graph = g
        self.captures += 1

    def feed(self, n: int) -> None:
        """g, K and V of the committed rows below n this ring does not hold yet, from the engine's tap
        ring: the rows within the window that the engine still has from this sequence."""
        st = self.srv.st
        if (id(st), st.tap_gen) != self.gen:
            self.gen = (id(st), st.tap_gen)
            self.tag.fill_(-1)
            self.fed = 0
        ring = st.ring
        R = ring.shape[0]
        a = max(self.fed, n - self.W + 1, st.tap_from, n - R, 0)
        if a < n:
            pos = torch.arange(a, n)
            dev = self.m.device
            g, k, v = self.m.context(ring.index_select(0, (pos % R).to(dev)), pos.to(dev))
            slot = (pos % self.W).to(dev)
            self.g.index_copy_(0, slot, g)
            self.k.index_copy_(2, slot, k)
            self.v.index_copy_(2, slot, v)
            self.tag[pos % self.W] = pos
            self.stats["context_rows"] += n - a
        self.fed = max(self.fed, n)


class _PackedGraph:
    """SPD-233: `LiLiCorrModel.draft_multi` of one set of heads captured once, on a capture stream of its own
    (engine/capture_lock.py, SPD-223); `out` [heads, slots, 4] after each replay."""

    def __init__(self, heads: tuple):
        from engine import capture_lock
        self.heads = heads
        self.stream = capture_lock.lease_stream()
        self.graph = None
        self.out = None

    def capture(self) -> None:
        from engine import capture_lock
        m = self.heads[0].m
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                m.draft_multi([h._req() for h in self.heads])
        torch.cuda.current_stream().wait_stream(self.stream)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g, stream=self.stream):
            self.out = m.draft_multi([h._req() for h in self.heads])
        torch.cuda.synchronize()
        self.graph = g

    def release(self) -> None:
        """The graph reset (its pool and the stream's workspace go with it), then the stream back."""
        from engine import capture_lock
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
            capture_lock.give_back(self.stream)


def build_head(srv, path: str, log=print) -> LiLiCorrHead:
    """The checkpoint at `path` over the serving engine's embedding and head (its taps set first:
    `NemotronHEngine.set_taps` before any state)."""
    cfg = load_config(path)
    eng = srv.fn
    if cfg.hidden_size != eng.cfg.hidden_size or cfg.vocab_size != eng.cfg.vocab_size:
        raise ValueError(f"LiLiCorr hidden {cfg.hidden_size} / vocab {cfg.vocab_size} against the target's "
                         f"{eng.cfg.hidden_size} / {eng.cfg.vocab_size}")
    if tuple(eng.taps) != tuple(cfg.target_layer_ids):
        raise ValueError(f"the engine taps {eng.taps}, the draft reads {cfg.target_layer_ids}")
    W = load_weights(path, str(eng.device))
    head = eng.head
    if cfg.own_head:
        if "lm_head.weight" not in W:
            raise ValueError("LiLiCorr config says has_own_lm_head, the checkpoint has no lm_head.weight")
        hw = W["lm_head.weight"]
        head = (lambda rows: F.linear(rows, hw).float()) if isinstance(hw, torch.Tensor) else \
            (lambda rows: _lin_wide(rows, hw).float())
    model = LiLiCorrModel(cfg, W, eng.W["embeddings.weight"], head)
    log(f"[lilicorr] {path}: {cfg.num_hidden_layers} layers, block {cfg.block_size}, window {cfg.window}, "
        f"taps {cfg.target_layer_ids}, lattice top {cfg.topk}, head {'its own' if cfg.own_head else 'the target'}'s")
    return LiLiCorrHead(srv, model)


def gate(eng, head: LiLiCorrHead, st, prompt: list[int], steps: int = 40, every: int = 4) -> dict:
    """The load-time agreement gate: the trunk's greedy decode of `steps` tokens after `prompt` (the
    engine's own decode step, taps on), then a draft at every `every`-th position against the tokens
    the trunk wrote: the mean run of drafted tokens equal to them (the tokens a greedy verify would
    keep)."""
    lg = eng.prefill(torch.tensor(prompt), st)
    seq = list(prompt) + [int(lg[0].argmax())]
    for _ in range(steps):
        eng.stage(seq[-1], st)
        out = eng.step(st)
        st.length += 1
        seq.append(int(out[0].argmax()))
    srv_st, head.srv = head.srv, type("GateSrv", (), {"st": st})()
    runs = []
    try:
        for t in range(len(prompt), len(prompt) + steps - head.depth, every):
            # the anchor seq[t] at position t; the trunk's next tokens seq[t + 1 ..]
            got = head.draft_at(seq[t], t)[:, 0].long().tolist()
            r = 0
            while r < len(got) and got[r] == seq[t + 1 + r]:
                r += 1
            runs.append(r)
    finally:
        head.srv = srv_st
        head.gen = None
    return {"positions": len(runs), "mean_run": round(sum(runs) / max(len(runs), 1), 3), "runs": runs}
