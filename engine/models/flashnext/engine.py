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

import numpy as np
import torch

from engine import capture_lock
from engine.settings import SETTINGS as _S
from engine.models.flashnext.ref import FlashNextRef, FNState
from engine.models.flashnext.ple import decode_rows


# rows up to which an EXL3 projection runs on tools/fn_kernels.dense (the kernel's limit);
# QWEN38_FN_DENSE=0: the library path of tools/exl3_linear for every row count (an A/B)
DENSE_MAX = 32 if _S.get("FN_DENSE") == "1" else 0


class FlashNextEngine(FlashNextRef):

    def __init__(self, cfg, W: dict, tables: dict, banks: dict, device="cuda", max_len: int = 8192):
        super().__init__(cfg, W, tables, dtype=torch.bfloat16, device=device)
        self.banks = banks
        self.max_len = max_len
        cos, sin = FlashNextRef.rope(self, torch.arange(max_len))
        self.rope_cos, self.rope_sin = cos, sin                          # [max_len, rotary] bf16
        self.dense_limit = (cfg.indexer_budget + cfg.indexer_compress_ratio - 1) if cfg.has_indexer else 1 << 60
        self.g = {}                                                      # the step's device buffers
        self._order1 = torch.empty(0, dtype=torch.int32, device=device)    # one row: launch order

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
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            # up to 32 rows (decode, verify): Flash-Next's dense kernel (rows invariant, PDL); a
            # prompt chunk: the library path of tools/exl3_linear
            y = K.dense(x2, w) if x2.shape[0] <= DENSE_MAX else exl3_matmul(x2, w)
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
            # one row: any launch order is the same arithmetic, so skip the sort
            order = self._order1 if xs.shape[0] == 1 else None
            K.moe(xs, ids, wts, self.banks[li], order=order, out=out[r0:r0 + step])
        return out

    def moe_parts(self, x, li):
        """The router and the experts of every (row, slot) pair, without the combine (the trunk
        folds it into the next inject): (yd [M S, H] fp32, ids, wts)."""
        from tools import fn_kernels as K
        c = self.cfg
        logits = K.gemv(x, self.W[self.pre(li) + "mlp.router"])
        ids, wts = K.router(logits, c.num_experts, c.num_experts_per_tok)
        if self.route_trace is not None:
            self.route_trace.append((li, ids[:, :-1].long()))
        yd = K.moe_experts(x, ids, self.banks[li], order=self._order1 if x.shape[0] == 1 else None)
        if self._moe_record is not None:
            self._moe_record.append((li, x.clone(), ids.clone(), wts.clone(), yd.clone()))
        return yd, ids, wts

    _moe_record = None
    spec_ok = False            # the load-time self-check passed (speculation may run)
    MOE_ROWS = 2048            # rows the MoE takes in one call (11 pairs a row on the grid's y axis)
    TREE_ROWS = (0, 1, 2, 3, 5, 6, 8, 11)
    CHAIN_WIDTHS = (2, 4, 8, 12, 16, 32)

    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 32, log=print) -> dict:
        """ENG-223's load-time check of the expert kernel's verify twins on real routing: decode
        `steps` tokens one at a time (the decode step), recording every layer's MoE input, ids,
        weights and every pair's expert output; then run each layer's rows again as chain calls of
        every verify width in CHAIN_WIDTHS (the first rows) and an 8-row tree-shaped call (rows
        TREE_ROWS, out of order), with the router recomputed from the recorded inputs, and compare
        every row bit for bit (the
        combine is per row and per column, so equal pairs give an equal sum). Any difference
        leaves `spec_ok` False and the log names the layer and the row."""
        from tools import fn_kernels as K
        c = self.cfg
        st = self.new_state(self.max_len)
        logits = self.prefill(torch.tensor(prompt), st)
        tok = int(logits[0].argmax())
        self._moe_record = rec = []
        try:
            for _ in range(steps):
                tok = int(self.decode_eager(tok, st)[0].argmax())
        finally:
            self._moe_record = None
        by_layer: dict = {}
        for li, x, ids, wts, out in rec:
            by_layer.setdefault(li, []).append((x, ids, wts, out))
        bad = None
        checked = 0
        S = c.num_experts_per_tok + 1
        for li, rows in sorted(by_layer.items()):
            X = torch.cat([r[0] for r in rows])
            I = torch.cat([r[1] for r in rows])
            Wt = torch.cat([r[2] for r in rows])
            Y = torch.cat([r[3] for r in rows]).view(len(rows), S, -1)        # each row's pairs
            ids, wts = K.router(K.gemv(X, self.W[self.pre(li) + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
            calls = [(f"chain{w}", list(range(w))) for w in self.CHAIN_WIDTHS if w <= len(rows)]
            calls.append(("tree", [t for t in self.TREE_ROWS[::-1] if t < len(rows)]))
            for name, sel in calls:
                sl = torch.tensor(sel, device=X.device)
                got = K.moe_experts(X[sl].contiguous(), I[sl].contiguous(), self.banks[li]).view(len(sel), S, -1)
                for j, t in enumerate(sel):
                    checked += 1
                    if bad is None and not (torch.equal(got[j], Y[t]) and torch.equal(ids[t], I[t]) and torch.equal(wts[t], Wt[t])):
                        bad = {"layer": li, "row": t, "call": name}
        self.spec_ok = bad is None
        res = {"ok": self.spec_ok, "steps": steps, "layers": len(by_layer), "rows_checked": checked, "first_diff": bad}
        if log:
            log(f"[fn-selfcheck] {'PASS' if self.spec_ok else 'FAIL: speculation stays off'}: {res}")
        return res

    # ------------------------------------------------------------------------- the trunk
    def _mix(self, n, p):
        """The mixer from its normed streams: (x [M, H], d [M, R + hc] fp32 = down and inject)."""
        from tools import fn_kernels as K
        di = K.gemv(n, self.W[p + ".down_inject"])
        return K.mix_up(di, n, self.W[p + ".input_mix_weight_up.weight"], self.cfg.hc_count), di

    def _site_norm(self, li: int):
        """The norm weight of layer li's first mixer (the final mixer's past the last layer)."""
        if li >= self.cfg.num_hidden_layers:
            return self.W["hyper_connection_mixer.hc_norm.weight"]
        return self.W[self.pre(li) + "attn_hyper_connection.hc_norm.weight"]

    def trunk(self, h, tokens, st, positions, step: bool = False):
        """Every layer: each inject writes the streams and, in the same launch, the next site's norm
        (not in front of a PLE layer, which changes the streams first). Returns the final mixer's
        normed streams n [M, hc H]; the streams themselves are `self.stack`."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        n = None
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            if li in c.ple_layers:
                h = h + (self._ple_step(h, li, st) if step else self.ple(h, tokens, li, st))
                n = None
            if n is None:
                n = K.hc_norm(h, self._site_norm(li), H, eps)
            x, di = self._mix(n, p + "attn_hyper_connection")
            if c.is_linear(li):
                out = self._gdn1(x, li, st) if step else self.gdn(x, li, st)
            else:
                out = self._attn1(x, li, st) if step else self.attention(x, li, st, positions)
            h, n = K.inject_norm(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                                 self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
            x, di = self._mix(n, p + "mlp_hyper_connection")
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            R = self.W[p + "mlp_hyper_connection.rank"]
            if x.shape[0] <= self.MOE_ROWS:
                yd, ids, wts = self.moe_parts(x, li)
                h, n = K.combine_inject_norm(h, yd, ids, wts, di, R, nxt, hc, eps)
            else:
                h, n = K.inject_norm(h, self.moe(x, li), di, R, nxt, hc, eps)
        self.stack = h
        if n is None:
            n = K.hc_norm(h, self._site_norm(c.num_hidden_layers), H, eps)
        return n

    def final(self, n):
        """The final mixer (no inject) from its normed streams, and the head: fp32 logits."""
        from tools import fn_kernels as K
        d = K.gemv(n, self.W["hyper_connection_mixer.down_inject"])
        x = K.mix_up(d, n, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], self.cfg.hc_count)
        return self.lin(x, self.W["lm_head.weight"]).float()

    @torch.no_grad()
    def forward(self, tokens, st, layers=None, trace=None, head="all"):
        """A prompt chunk (or any block) through the fused trunk; the reference's forward for a
        layer subset or a trace."""
        if layers is not None or trace is not None:
            return super().forward(tokens, st, layers=layers, trace=trace, head=head)
        tokens = tokens.view(-1).to(self.device)
        T = tokens.shape[0]
        positions = torch.arange(st.length, st.length + T, device=self.device)
        h = self.embed(tokens).repeat(1, self.cfg.hc_count)
        n = self.trunk(h, tokens, st, positions)
        st.length += T
        if head == "none":
            return None
        return self.final(n[-1:] if head == "last" else n)

    # ------------------------------------------------------------------------- the decode step
    def alloc_step(self) -> None:
        """The step's input buffers: token, position, [start, start + 1], the PLE rows."""
        c, dev = self.cfg, self.device
        tab = next(iter(self.tables.values())) if self.tables else None
        nh = c.ngram_heads
        words = tab.words if tab is not None else 1
        # one pinned host image and its device copy: [tok, pos, heads x nh] int64, then
        # [start, end] int32, then the packed rows int16 -- one H2D a step
        nb = 8 * (2 + nh) + 8 + 2 * nh * words
        self._hbuf = torch.zeros(nb, dtype=torch.uint8, pin_memory=True)
        self._copied = torch.cuda.Event()
        self._copied.record()
        self._dbuf = torch.zeros(nb, dtype=torch.uint8, device=dev)

        def views(b):
            i64 = b[: 8 * (2 + nh)].view(torch.int64)
            i32 = b[8 * (2 + nh): 8 * (2 + nh) + 8].view(torch.int32)
            rows = b[8 * (2 + nh) + 8:].view(torch.int16).view(nh, words)
            return i64, i32, rows
        hi64, hi32, hrows = views(self._hbuf)
        di64, di32, drows = views(self._dbuf)
        self._h = {"i64": hi64, "lenp": hi32, "rows": hrows}
        self.g = {
            "tok": di64[0:1], "pos": di64[1:2], "heads": di64[2:], "lenp": di32, "packed": drows,
            "mask": torch.ones(1, 1, dtype=torch.bool, device=dev),
            "logits": torch.zeros(1, c.vocab_size, dtype=torch.float32, device=dev),
            "next": torch.zeros(1, dtype=torch.long, device=dev),
        }
        self._next_h = torch.zeros(1, dtype=torch.long, pin_memory=True)
        if tab is not None:
            self.g["bias"] = tab.bias.to(dev)
            self.g["codebook"] = tab.codebook(dev)

    def stage(self, token: int, st: FNState) -> None:
        """Host work of one step, before the step (or its graph) runs: the PLE rows of `token`
        (hash with the state's context, gather from the table) and the position buffers."""
        # the pinned image is the previous step's copy source until that copy has run
        self._copied.synchronize()
        hb = self._h
        i64 = hb["i64"].numpy()
        i64[0], i64[1] = token, st.length
        hb["lenp"].numpy()[:] = (st.length, st.length + 1)
        for li in self.cfg.ple_layers:
            hsh, tab = self.hash[li], self.tables[li]
            ctx = st.ple_ctx.get(li)
            ctx = [self.cfg.eos_token_id] * hsh.context_len if ctx is None else ctx.view(-1).tolist()
            ids = hsh.ids_one(token, ctx)
            st.ple_ctx[li] = torch.tensor([(ctx + [token])[-hsh.context_len:]], dtype=torch.long)
            hb["rows"].numpy()[:] = tab.map[ids]
            i64[2:] = np.searchsorted(tab.offsets_np, ids, side="right") - 1
        self._dbuf.copy_(self._hbuf, non_blocking=True)
        self._copied.record()

    def step(self, st: FNState) -> torch.Tensor:
        """One token from the device buffers (`stage` first); the state advances in place except
        `st.length`, which the caller moves. Returns logits [1, vocab] fp32 (a static buffer)."""
        from tools import fn_kernels as K
        c, g = self.cfg, self.g
        h = self.W["embed_tokens.weight"].index_select(0, g["tok"]).repeat(1, c.hc_count)
        n = self.trunk(h, None, st, None, step=True)
        d = K.gemv(n, self.W["hyper_connection_mixer.down_inject"])
        x = K.mix_up(d, n, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], c.hc_count)
        if DENSE_MAX:
            K.dense(x, self.W["lm_head.weight"], out=g["logits"])
        else:
            from tools.exl3_linear import exl3_matmul
            exl3_matmul(x, self.W["lm_head.weight"], out=g["logits"])
        # the greedy token, and its copy into pinned host memory: the host reads it after a sync
        torch.argmax(g["logits"][0], dim=-1, keepdim=True, out=g["next"])
        self._next_h.copy_(g["next"], non_blocking=True)
        return g["logits"]

    def _ple_step(self, h, li, st):
        g, tab = self.g, self.tables[li]
        emb = decode_rows(g["packed"], g["heads"], g["bias"], tab.K, g["codebook"])
        return self.ple_apply(h, emb.to(self.dtype).reshape(1, -1), li, st)

    def _gdn1(self, x, li, st):
        from tools import fn_kernels as K
        from tools.gdn_kernels import decode_pre, fused_decode_step
        c = self.cfg
        p = self.pre(li) + "linear_attn."
        qkvz = self.lin(x, self.W[p + "qkvz"])[0]
        mixed, z = qkvz[: c.conv_dim], qkvz[c.conv_dim:]
        ab = K.gemv_bf16(x, self.W[p + "ab"])[0]
        Hv = c.linear_num_value_heads
        qkv, gg, beta = decode_pre(mixed, st.conv[li], self.W[p + "conv1d.weight"].squeeze(1),
                                   ab[:Hv], ab[Hv:], self.W[p + "A_log"], self.W[p + "dt_bias"])
        kd, vd = c.key_dim, c.value_dim
        q = qkv[:kd].view(1, 1, c.linear_num_key_heads, c.linear_key_head_dim)
        k = qkv[kd:2 * kd].view(1, 1, c.linear_num_key_heads, c.linear_key_head_dim)
        v = qkv[2 * kd:2 * kd + vd].view(1, 1, Hv, c.linear_value_head_dim)
        o = fused_decode_step(q, k, v, gg.view(1, 1, -1), beta.view(1, 1, -1), st.S[li],
                              rep=c.num_v_per_k).reshape(-1, c.linear_value_head_dim)
        hh = K.gated_norm(o, z.view(-1, c.linear_value_head_dim), self.W[p + "norm.weight"], c.rms_norm_eps,
                          c.output_gate_type == "sigmoid")
        return self.lin(hh.reshape(1, -1), self.W[p + "out_proj.weight"])

    def _attn1(self, x, li, st):
        """Attention of one token, dense regime, position and length from the device buffers:
        q|k|v and the indexer projection, one prep launch (norms, rope, cache writes), the
        attention, the output gate, o_proj."""
        from tools import fn_kernels as K
        from tools.attn_kernels import decode_attention_dev
        c, g = self.cfg, self.g
        p = self.pre(li) + "self_attn."
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        qkv = self.lin(x, self.W[p + "qkv"])
        iqk = self.lin(x, self.W[p + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
        q = K.attn_prep(qkv, iqk, g["pos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        self.rope_cos, self.rope_sin, st.k[li], st.v[li], st.ik.get(li), Hq, Hk,
                        c.indexer_n_heads * c.indexer_head_dim if c.has_indexer else 0, c.rms_norm_eps)
        max_lc = self.g.get("max_lc", 2048)
        o = decode_attention_dev(q.view(1, Hq, 1, D), st.k[li].unsqueeze(0), st.v[li].unsqueeze(0),
                                 g["lenp"], g["mask"], max_lc)
        y = K.gate_mul(o.transpose(1, 2), qkv, D)
        return self.lin(y, self.W[p + "o_proj.weight"])

    # ------------------------------------------------------------------------- MTP (ENG-153)
    @torch.no_grad()
    def mtp_feed(self, stack: torch.Tensor, next_tokens: torch.Tensor, st_m: FNState):
        """The MTP layer over committed rows: `stack` [T, hc H] (the trunk's streams at positions
        st_m.length..) and the tokens that follow them. Returns (logits [T, vocab] of the token after
        next, the MTP layer's own streams for a deeper draft). The reference's `mtp` on the engine's
        primitives (EXL3 projections, int8 mixers, the expert kernels); eager."""
        return self.mtp(stack, next_tokens.to(self.device), st_m)

    @torch.no_grad()
    def mtp_chain(self, stack_row: torch.Tensor, next_token: int, st_m: FNState, depth: int) -> list[int]:
        """A greedy draft chain of `depth` tokens from one committed row and its successor: the
        first feed is real and stays in the MTP layer's cache; the deeper feeds are drafts and are
        rolled back (the cache is kept by length)."""
        logits, h = self.mtp_feed(stack_row, torch.tensor([next_token]), st_m)
        keep = st_m.length
        out = [int(logits[-1].argmax())]
        for _ in range(depth - 1):
            logits, h = self.mtp_feed(h[-1:], torch.tensor([out[-1]]), st_m)
            out.append(int(logits[-1].argmax()))
        st_m.length = keep
        st_m.clamp_pool()
        return out

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
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.step(st)
        torch.cuda.synchronize()
        for k, v in _state_tensors(st).items():         # the warm-up steps wrote the state: undo
            v.copy_(saved[k])
        st.ple_ctx.clear()
        st.ple_ctx.update(pctx)
        st.length = length

    @torch.no_grad()
    def greedy(self, token: int) -> int:
        """One greedy step: the next token, read from the graph's own argmax (no extra launch)."""
        eng, st = self.eng, self.st
        if st.length + 1 > min(self.max_lc, eng.dense_limit) or self.graph is None:
            return int(self(token)[0].argmax())
        eng.stage(token, st)
        self.graph.replay()
        st.length += 1
        torch.cuda.current_stream().synchronize()
        return int(eng._next_h[0])

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
