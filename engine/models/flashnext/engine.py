"""Flash-Next on the board (ENG-221): the reference forward with its weight reads on the kernels, and
a one-token decode step that reads its inputs from device buffers, so a CUDA graph can hold it.

`FlashNextRef` (ref.py) is the model, operation for operation, checked against `transformers`. This
class keeps its structure and routes what reads weights to kernels over the pack's formats:

  EXL3 projections and the head     tools/exl3_linear.exl3_matmul (q|k|v and qkv|z as one group)
  mixers (int8 rows or fp16)        tools/fn_kernels.gemv (down and inject in one weight) + mix_up
  router, routed and shared experts tools/fn_kernels.router + moe (one bank, the shared expert last)
  GDN a/b, PLE projections (fp16)   tools/fn_kernels.gemv
  GDN recurrence (decode)           tools/gdn_kernels.decode_pre + fused_decode_step (the 27B's)
  attention (decode, dense regime)  tools/attn_kernels.decode_attention_dev (length on the device)

`forward()` (the reference's) serves a prompt chunk of any length. `step()` is one token whose id,
position and PLE rows sit in device buffers (`self.g`); `DecodeGraph` captures it once and replays it
per token. Every part of `step()` computes its row with the arithmetic of the row-invariant kernels,
so the decode step is the reference a verify block (ENG-227) has to equal bit for bit.

The decode step covers the dense attention regime: a context of up to `indexer_budget + 3` tokens
(2,051), where QSA keeps every token. Past it the step takes the reference's attention (eager, slow,
exact); the selected-key decode of `flashnext` (5668e843) is ENG-147's remaining work.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from engine.models.flashnext.ref import MTP, FlashNextRef, FNState, apply_rope
from engine.models.flashnext.ple import decode_rows


class FlashNextEngine(FlashNextRef):

    def __init__(self, cfg, W: dict, tables: dict, banks: dict, device="cuda", max_len: int = 8192):
        super().__init__(cfg, W, tables, dtype=torch.bfloat16, device=device)
        self.banks = banks
        self.max_len = max_len
        cos, sin = FlashNextRef.rope(self, torch.arange(max_len))
        self.rope_cos, self.rope_sin = cos, sin                          # [max_len, rotary] bf16
        self.dense_limit = (cfg.indexer_budget + cfg.indexer_compress_ratio - 1) if cfg.has_indexer else 1 << 60
        self.g = {}                                                      # the step's device buffers
        self._moe_bufs: dict = {}

    # ------------------------------------------------------------------------- state
    def new_state(self, cap: int | None = None) -> FNState:
        """A state with every tensor the step writes allocated up front (a graph holds them)."""
        c = self.cfg
        cap = cap or self.max_len
        st = FNState(cap=cap)
        dev = self.device
        for li in range(c.num_hidden_layers):
            if c.is_linear(li):
                st.S[li] = torch.zeros(1, c.linear_num_value_heads, c.linear_key_head_dim,
                                       c.linear_value_head_dim, dtype=torch.float32, device=dev)
                st.conv[li] = torch.zeros(1, c.conv_dim, c.linear_conv_kernel_dim - 1, dtype=self.dtype, device=dev)
            else:
                st.k[li] = torch.zeros(c.num_key_value_heads, cap, c.head_dim, dtype=self.dtype, device=dev)
                st.v[li] = torch.zeros_like(st.k[li])
                if c.has_indexer:
                    st.ik[li] = torch.zeros(cap, c.indexer_head_dim, dtype=self.dtype, device=dev)
        for li in c.ple_layers:
            st.ple_conv[li] = torch.zeros(1, c.hc_hidden, (c.ple_conv_kernel_size - 1) * c.ngram_size,
                                          dtype=self.dtype, device=dev)
        return st

    def reset(self, st: FNState) -> None:
        st.reset_inplace()
        for d in (st.k, st.v, st.ik):
            for v in d.values():
                v.zero_()

    # ------------------------------------------------------------------------- primitives
    def lin(self, x, w):
        from tools import fn_kernels as K
        from tools.exl3_linear import EXL3Block, exl3_matmul
        if isinstance(w, EXL3Block):
            lead = x.shape[:-1]
            y = exl3_matmul(x.reshape(-1, x.shape[-1]).contiguous(), w)
            return y.view(*lead, w.N)
        if hasattr(w, "q") or (isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.dim() == 2):
            lead = x.shape[:-1]
            y = K.gemv(x.reshape(-1, x.shape[-1]).contiguous(), w)
            return y.to(x.dtype).view(*lead, -1)
        return super().lin(x, w)

    def rope(self, positions: torch.Tensor):
        p = positions.to(self.rope_cos.device).long()
        return self.rope_cos.index_select(0, p), self.rope_sin.index_select(0, p)

    def rope_at(self, positions: torch.Tensor):
        return self.rope(positions)

    def delta_rule(self, q, k, v, g, beta, S):
        if q.shape[1] > 64:
            from tools.gdn_prefill_kernels import CHUNK, fused_chunk_prefill, fused_prefill_refusal
            if not fused_prefill_refusal(CHUNK):
                return fused_chunk_prefill(q, k, v, g, beta, S, chunk_size=CHUNK)
        return super().delta_rule(q, k, v, g, beta, S)

    def mixer(self, h, p, combine=True):
        """norm, down (+ inject) in one GEMV, then mix_up: the reference's mixer with each bf16
        rounding where `lin` makes it."""
        from tools import fn_kernels as K
        c = self.cfg
        n = self.rms(h, self.w(p + ".hc_norm.weight"), c.rms_norm_eps, group=c.hidden_size)
        di = K.gemv(n, self.W[p + ".down_inject"])
        x = K.mix_up(di, n, self.W[p + ".input_mix_weight_up.weight"], c.hc_count)
        if not combine:
            return x
        R = self.W[p + ".rank"]
        inj = 2 * torch.sigmoid(di[:, R:].to(self.dtype) / c.hc_count)
        return x, h, inj

    def moe(self, x, li):
        from tools import fn_kernels as K
        c = self.cfg
        out = torch.empty_like(x)
        step = 2048
        for r0 in range(0, x.shape[0], step):
            xs = x[r0:r0 + step]
            logits = K.gemv(xs, self.W[self.pre(li) + "mlp.router"])
            ids, wts = K.router(logits, c.num_experts, c.num_experts_per_tok)
            if self.route_trace is not None:
                self.route_trace.append((li, ids[:, :-1].long()))
            K.moe(xs, ids, wts, self.banks[li], out=out[r0:r0 + step])
        return out

    # ------------------------------------------------------------------------- the decode step
    def alloc_step(self) -> None:
        """The step's input buffers: token, position, [start, start + 1], the PLE rows."""
        c, dev = self.cfg, self.device
        tab = next(iter(self.tables.values())) if self.tables else None
        self.g = {
            "tok": torch.zeros(1, dtype=torch.long, device=dev),
            "pos": torch.zeros(1, dtype=torch.long, device=dev),
            "lenp": torch.zeros(2, dtype=torch.int32, device=dev),
            "packed": torch.zeros(c.ngram_heads, tab.words if tab is not None else 1, dtype=torch.int16, device=dev),
            "heads": torch.zeros(c.ngram_heads, dtype=torch.long, device=dev),
            "mask": torch.ones(1, 1, dtype=torch.bool, device=dev),
            "logits": torch.zeros(1, c.vocab_size, dtype=torch.float32, device=dev),
        }
        if tab is not None:
            self.g["bias"] = tab.bias.to(dev)
            self.g["codebook"] = tab.codebook(dev)

    def stage(self, token: int, st: FNState) -> None:
        """Host work of one step, before the step (or its graph) runs: the PLE rows of `token`
        (hash with the state's context, gather from the table) and the position buffers."""
        g = self.g
        g["tok"].fill_(token)
        g["pos"].fill_(st.length)
        g["lenp"].copy_(torch.tensor([st.length, st.length + 1], dtype=torch.int32), non_blocking=True)
        for li in self.cfg.ple_layers:
            ids = self.ple_ids(torch.tensor([token]), li, st)[0]           # [heads]
            tab = self.tables[li]
            g["packed"].copy_(tab.gather_packed(ids), non_blocking=True)
            g["heads"].copy_(tab.head_of(ids), non_blocking=True)

    def step(self, st: FNState) -> torch.Tensor:
        """One token from the device buffers (`stage` first); the state advances in place except
        `st.length`, which the caller moves. Returns logits [1, vocab] fp32 (a static buffer)."""
        c, g = self.cfg, self.g
        h = self.W["embed_tokens.weight"].index_select(0, g["tok"]).repeat(1, c.hc_count)
        for li in range(c.num_hidden_layers):
            if li in c.ple_layers:
                tab = self.tables[li]
                emb = decode_rows(g["packed"], g["heads"], g["bias"], tab.K, g["codebook"])
                h = h + self.ple_apply(h, emb.to(self.dtype).reshape(1, -1), li, st)
            x, h0, inj = self.mixer(h, self.pre(li) + "attn_hyper_connection")
            out = self._gdn1(x, li, st) if c.is_linear(li) else self._attn1(x, li, st)
            h = self.inject(h0, out, inj)
            x, h0, inj = self.mixer(h, self.pre(li) + "mlp_hyper_connection")
            h = self.inject(h0, self.moe(x, li), inj)
        self.stack = h
        x = self.mixer(h, "hyper_connection_mixer", combine=False)
        from tools.exl3_linear import exl3_matmul
        exl3_matmul(x, self.W["lm_head.weight"], out=g["logits"])
        return g["logits"]

    def _gdn1(self, x, li, st):
        from tools import fn_kernels as K
        from tools.gdn_kernels import decode_pre, fused_decode_step
        c = self.cfg
        p = self.pre(li) + "linear_attn."
        qkvz = self.lin(x, self.W[p + "qkvz"])[0]
        mixed, z = qkvz[: c.conv_dim], qkvz[c.conv_dim:]
        ab = K.gemv(x, self.W[p + "ab"]).to(self.dtype)[0]
        Hv = c.linear_num_value_heads
        qkv, gg, beta = decode_pre(mixed, st.conv[li], self.W[p + "conv1d.weight"].squeeze(1),
                                   ab[:Hv], ab[Hv:], self.W[p + "A_log"], self.W[p + "dt_bias"])
        kd, vd = c.key_dim, c.value_dim
        q = qkv[:kd].view(1, 1, c.linear_num_key_heads, c.linear_key_head_dim)
        k = qkv[kd:2 * kd].view(1, 1, c.linear_num_key_heads, c.linear_key_head_dim)
        v = qkv[2 * kd:2 * kd + vd].view(1, 1, Hv, c.linear_value_head_dim)
        o = fused_decode_step(q, k, v, gg.view(1, 1, -1), beta.view(1, 1, -1), st.S[li],
                              rep=c.num_v_per_k).reshape(-1, c.linear_value_head_dim)
        hh = o.float()
        hh = hh * torch.rsqrt(hh.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
        hh = self.W[p + "norm.weight"] * hh.to(o.dtype)
        act = torch.sigmoid if c.output_gate_type == "sigmoid" else F.silu
        hh = (hh * act(z.view(-1, c.linear_value_head_dim).float())).to(o.dtype)
        return self.lin(hh.reshape(1, -1), self.W[p + "out_proj.weight"])

    def _attn1(self, x, li, st):
        """Attention of one token, dense regime, position and length from the device buffers."""
        from tools.attn_kernels import decode_attention_dev
        c, g = self.cfg, self.g
        p = self.pre(li) + "self_attn."
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        qkv = self.lin(x, self.W[p + "qkv"])[0]
        qg = qkv[: Hq * 2 * D].view(1, Hq, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(1, -1)
        q = self.rms(q.contiguous(), self.W[p + "q_norm.weight"], c.rms_norm_eps)
        k = self.rms(qkv[Hq * 2 * D: Hq * 2 * D + Hk * D].view(1, Hk, D), self.W[p + "k_norm.weight"], c.rms_norm_eps)
        v = qkv[Hq * 2 * D + Hk * D:].view(1, Hk, D)
        cos, sin = self.rope_cos.index_select(0, g["pos"]), self.rope_sin.index_select(0, g["pos"])
        q = apply_rope(q, cos[:, None], sin[:, None])
        k = apply_rope(k, cos[:, None], sin[:, None])
        st.k[li].index_copy_(1, g["pos"], k.transpose(0, 1).contiguous())
        st.v[li].index_copy_(1, g["pos"], v.transpose(0, 1).contiguous())
        if c.has_indexer:
            ip = p + "indexer."
            qk = self.lin(x, self.W[ip + "index_qk_proj.weight"])
            st.ik[li].index_copy_(0, g["pos"], qk[:, c.indexer_n_heads * c.indexer_head_dim:].contiguous())
        max_lc = self.g.get("max_lc", 2048)
        o = decode_attention_dev(q.transpose(0, 1).unsqueeze(0), st.k[li].unsqueeze(0), st.v[li].unsqueeze(0),
                                 g["lenp"], g["mask"], max_lc)
        o = o[0].transpose(0, 1).reshape(1, -1)
        return self.lin(o * torch.sigmoid(gate), self.W[p + "o_proj.weight"])

    # ------------------------------------------------------------------------- generation
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, st: FNState, chunk: int = 2048) -> torch.Tensor:
        """The prompt in chunks (the reference's batched forward); logits of the last row."""
        out = None
        for i in range(0, ids.numel(), chunk):
            last = i + chunk >= ids.numel()
            out = self.forward(ids[i:i + chunk].to(self.device), st, head="last" if last else "none")
        return out[-1:]

    @torch.no_grad()
    def decode_eager(self, token: int, st: FNState) -> torch.Tensor:
        """One token through `step` (no graph), or the reference's forward past the dense regime."""
        if st.length + 1 > self.dense_limit:
            return self.forward(torch.tensor([token], device=self.device), st)
        if not self.g:
            self.alloc_step()
        self.stage(token, st)
        out = self.step(st)
        st.length += 1
        return out


class DecodeGraph:
    """`FlashNextEngine.step` captured once for a state and a context class; `__call__(token)` stages
    the token (host), replays, and moves the length. Past `max_lc` or the dense regime it runs eager."""

    def __init__(self, eng: FlashNextEngine, st: FNState, max_lc: int = 2048):
        self.eng, self.st, self.max_lc = eng, st, max_lc
        if not eng.g:
            eng.alloc_step()
        eng.g["max_lc"] = max_lc
        self.graph = None

    def capture(self) -> None:
        eng, st = self.eng, self.st
        saved = {k: v.clone() for k, v in _state_tensors(st).items()}
        pctx = dict(st.ple_ctx)
        length = st.length
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.stage(0, st)
                eng.step(st)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            eng.step(st)
        torch.cuda.synchronize()
        for k, v in _state_tensors(st).items():         # the warm-up steps wrote the state: undo
            v.copy_(saved[k])
        st.ple_ctx.clear()
        st.ple_ctx.update(pctx)
        st.length = length

    @torch.no_grad()
    def __call__(self, token: int) -> torch.Tensor:
        eng, st = self.eng, self.st
        if st.length + 1 > min(self.max_lc, eng.dense_limit):
            return eng.decode_eager(token, st)
        if self.graph is None:
            self.capture()
        eng.stage(token, st)
        self.graph.replay()
        st.length += 1
        return eng.g["logits"]


def _state_tensors(st: FNState) -> dict:
    out = {}
    for name, d in (("S", st.S), ("conv", st.conv), ("k", st.k), ("v", st.v), ("ik", st.ik), ("pconv", st.ple_conv)):
        for li, t in d.items():
            out[f"{name}.{li}"] = t
    return out
