"""Flint on the board: the reference forward with its weight reads on the kernels, a row step that serves
a decode token and a verify block of up to 32 rows with the same per-row arithmetic, and the step as a
CUDA graph. The structure is LFM2's board engine (engine/models/lfm2_moe/engine.py) without the short
convolutions and with Flint's attention:

  rows_step(st, M)   M rows whose tokens, positions, cache slots, attention control and ancestors sit in
                     device buffers (`stage` for a decode token, `stage_rows` for a draft block in DFS
                     pre-order), every part on a row-invariant kernel:
                       projections        tools/fn_kernels.dense_rows (EXL3, up to 32 rows) or LFM2's
                                          kernels.linear_rows (BF16)
                       norms, q/k prep    LFM2's kernels (a full layer's q/k prep reads a one-row identity
                                          rotary table at position 0: no positional encoding, the same bits)
                       attention          kernels.attn_rows: the sink in the softmax, a full layer over every
                                          visible key, a sliding layer over its window from its ring
                       router, experts    kernels.router (fp32 sigmoid), the EXL3 bank kernels of
                                          tools/fn_kernels (or LFM2's moe_bf16), fn_kernels.moe_combine; the
                                          shared expert as a dense SwiGLU, added to the routed sum
                     so the row of a verify block is the decode step's row at that position bit for bit, and a
                     greedy speculative run writes the plain greedy text.
  commit(st, path)   the path's KV rows moved down (the full layers' slots, the rings' slots), the length moved
  prefill(tokens)    a prompt in chunks of PREFILL_CHUNK on batched kernels (the EXL3 library GEMM, the grouped
                     EXL3 experts, kernels.prompt_attention): the next token agrees with the row path, the bits
                     do not

KV: a full layer keeps every row ([kv heads, cap, D], paged by the KV pool when it is on); a sliding layer
keeps a ring of RING rows (slot = position mod RING): its window of W - 1 earlier rows plus the rows a verify
block or a capture writes past the length, so its memory does not grow with the context.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import capture_lock
from engine.models.flint import kernels as K
from engine.models.flint.ref import FlintRef, FlintState
from engine.models.lfm2_moe import kernels as LK

#: rows a prompt chunk takes (the caches' anchor grid)
PREFILL_CHUNK = 4096
#: rows up to which a projection runs on the row-invariant kernels (their limit, the verify width)
ROWS_MAX = 32
#: rows a captured step, verify block or commit may write past the length (a block is at most 33 rows)
CAPTURE_ROWS = 64
#: SPD-144: a prompt chunk's BF16 and NVFP4 experts in two grouped launches (kernels.moe_grouped_glu); 0 = a loop
#: over the experts on the host (each one's rows through cuBLAS)
GROUPED = os.environ.get("QSE_FLINT_GROUPED", "1") != "0"
#: the bit widths tools/fn_kernels.dense decodes (the half-integer rates as ten times the rate)
DENSE_K = (2, 3, 4, 5, 6, 8, 15, 25, 35)


def ring_rows(window: int) -> int:
    """A sliding layer's ring: the window's W - 1 earlier rows, the CAPTURE_ROWS a block or a capture writes
    past the length, rounded up to 64 rows."""
    return -(-(window - 1 + CAPTURE_ROWS + 1) // 64) * 64


def _is_exl3(w) -> bool:
    return hasattr(w, "kb")


def _is_nvfp4_bank(e) -> bool:
    return hasattr(e, "moe_glu")


class _NoEvent:
    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _event(dev):
    return torch.cuda.Event() if dev.type == "cuda" else _NoEvent()


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.current_stream().synchronize()


class FlintEngine(FlintRef):

    def __init__(self, cfg, W: dict, experts: dict, device="cuda", max_len: int = 32768):
        super().__init__(cfg, W, dtype=torch.bfloat16, device=device, experts=experts)
        c = cfg
        self.max_len = max_len
        cos, sin = FlintRef.rope(self, torch.arange(max_len, device=self.device))
        self.rope_cos, self.rope_sin = cos.contiguous(), sin.contiguous()        # [max_len, D] bf16
        # a full layer's "rotary": one row, cos 1 and sin 0, read at position 0 (x * 1 + rot(x) * 0 = x exactly)
        self.id_cos = torch.ones(1, c.head_dim, dtype=torch.bfloat16, device=self.device)
        self.id_sin = torch.zeros(1, c.head_dim, dtype=torch.bfloat16, device=self.device)
        self.ring = ring_rows(c.sliding_window) if c.sliding_window else 0
        self.classes = K.classes(max_len)
        self._rb: dict = {}
        self.host_ms: dict = {}
        self._gate = {li: W[self.pre(li) + "mlp.gate.weight"].to(torch.bfloat16).contiguous() for li in c.moe_layers}
        # the router's 256 x 2048 weight on tools/fn_kernels.gemv where fp16 holds it exactly
        self._gate16 = {}
        for li, g in self._gate.items():
            h = g.to(torch.float16)
            if self.device.type == "cuda" and torch.equal(h.to(torch.bfloat16), g):
                self._gate16[li] = h.contiguous()
        self._bias = {li: W[self.pre(li) + "mlp.gate.expert_bias"].float().contiguous()
                      for li in c.moe_layers if self.pre(li) + "mlp.gate.expert_bias" in W}
        self._sink = {li: W[self.pre(li) + "self_attn.softmax_offset"].float().contiguous()
                      for li in range(c.num_hidden_layers)}
        self.exl3 = any(_is_exl3(w) for w in W.values())
        self.banks = {li: e.bank(self.device) for li, e in experts.items() if hasattr(e, "bank")} \
            if str(self.device).startswith("cuda") else {}

    # ------------------------------------------------------------------------- state
    def new_state(self, cap: int | None = None, paged: bool = False) -> FlintState:
        """A state with every tensor the step writes allocated up front (a graph holds them): K and V of the
        full layers [kv heads, cap, D] (reserved as addresses and committed as rows are written with the KV
        pool, `paged`), a ring [kv heads, RING, D] of each sliding layer."""
        from engine import kvpool as _kvp
        c = self.cfg
        cap = cap or self.max_len
        st = FlintState(cap=cap)
        if paged and _kvp.POOL is not None and cap >= _kvp.PAGED_MIN_ROWS and str(self.device).startswith("cuda"):
            st.pages = _kvp.PagedSet(_kvp.OWNER)
        for li in range(c.num_hidden_layers):
            sliding = c.is_sliding(li)
            rows = self.ring if sliding else cap
            if sliding:
                st.ring[li] = self.ring
            for kind, d in (("k", st.k), ("v", st.v)):
                shape = (c.num_key_value_heads, rows, c.head_dim)
                if st.pages is None or sliding:
                    d[li] = torch.zeros(*shape, dtype=self.dtype, device=self.device)
                else:
                    d[li] = st.pages.add(_kvp.PagedBuffer(shape, 1, self.dtype, self.device, owner=st.pages.owner,
                                                          name=f"flint.{kind}.{li}")).tensor
        return st

    def reset(self, st: FlintState) -> None:
        st.reset_inplace()

    # ------------------------------------------------------------------------- primitives
    def lin(self, x: torch.Tensor, w) -> torch.Tensor:
        from tools import fn_kernels as FK
        from tools.exl3_linear import exl3_matmul
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if _is_exl3(w):
            if not x2.is_cuda:
                y = torch.cat([exl3_matmul(x2[i:i + 1], w) for i in range(x2.shape[0])])
            elif x2.shape[0] <= ROWS_MAX and w.kb in DENSE_K:
                y = FK.dense_rows(x2, w)
            else:
                y = exl3_matmul(x2, w)
            return y.view(*x.shape[:-1], w.N)
        if not torch.is_tensor(w):
            return w.matmul(x2).view(*x.shape[:-1], w.N)      # NVFP4 (tools/nvfp4_relu2.NVFP4Wide)
        if x2.shape[0] <= ROWS_MAX:
            return LK.linear_rows(x2, w).view(*x.shape[:-1], w.shape[0])
        return F.linear(x2, w).view(*x.shape[:-1], w.shape[0])

    def _fused(self, x: torch.Tensor, p: str, name: str, parts: tuple) -> torch.Tensor:
        w = self.W.get(p + name)
        if w is not None:
            return self.lin(x, w)
        return torch.cat([self.lin(x, self.W[p + q + ".weight"]) for q in parts], -1)

    def head(self, n: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Logits [M, vocab] fp32 of the final normed rows (a static `out` in a graph)."""
        from tools import fn_kernels as FK
        from tools.exl3_linear import exl3_matmul
        w = self.head_weight()
        M = n.shape[0]
        if not torch.is_tensor(w) and not _is_exl3(w):
            return w.matmul(n, out=out, f32=True)                # NVFP4
        if _is_exl3(w):
            if not n.is_cuda:
                y = torch.cat([exl3_matmul(n[i:i + 1], w) for i in range(M)]).float()
                return y if out is None else out.copy_(y)
            if M <= ROWS_MAX and w.kb in DENSE_K:
                return FK.dense_rows(n, w, out=out)
            return exl3_matmul(n, w, out=out)
        if M <= ROWS_MAX:
            return LK.linear_rows(n, w, out=out, f32=True)
        y = F.linear(n, w).float()
        return y if out is None else out.copy_(y)

    def glu_rows(self, n: torch.Tensor, p: str, I: int) -> torch.Tensor:
        gu = self._fused(n, p, "gate_up", ("gate_proj", "up_proj"))
        return self.lin(K.glu_act(gu, I), self.W[p + "down_proj.weight"])

    def moe_rows(self, n: torch.Tensor, li: int) -> torch.Tensor:
        """The MoE of M <= 32 rows: the router, every (row, slot) pair through its expert, the combine in
        ascending expert id, plus the shared expert."""
        from tools import fn_kernels as FK
        c = self.cfg
        g16 = self._gate16.get(li)
        logits = FK.gemv(n, g16) if g16 is not None else LK.linear_rows(n, self._gate[li], f32=True)
        ids, wts = K.router(logits, self._bias.get(li), c.num_experts_per_tok, c.route_eps, scale=c.route_scale)
        if li in self.banks and _is_nvfp4_bank(self.banks[li]):
            yd = self.banks[li].moe_glu(n, ids)
        elif li in self.banks:
            yd = FK.moe_experts(n, ids, self.banks[li], order=self._order1 if n.shape[0] == 1 else None)
        else:
            e = self.experts[li]
            yd = LK.moe_bf16(n, ids, e.w13, e.w2)
        if self.route_trace is not None:
            self.route_trace.append((li, ids.long()))
        m = FK.moe_combine(yd, ids, wts) if self.device.type == "cuda" else LK.combine_reference(yd, ids, wts)
        if c.shared_expert_intermediate_size:
            # the shared expert's output beside the routed sum: the caller adds both in one launch (add2_rms)
            return m, self.glu_rows(n, self.pre(li) + "mlp.shared_expert.", c.shared_expert_intermediate_size)
        return m

    route_trace = None

    @property
    def _order1(self):
        if not hasattr(self, "_o1"):
            self._o1 = torch.empty(0, dtype=torch.int32, device=self.device)
        return self._o1

    # ------------------------------------------------------------------------- rows: decode and verify
    def _rows_buffers(self, M: int, key=None) -> dict:
        """The device inputs of an M-row call and their pinned host image (one H2D copy a call):

          tok, pos, slot, rslot, zero   int64 [M]: the token, its position, its cache slot (a full layer's),
                                        its ring slot (slot mod RING), zeros (the full layers' rotary row)
          ctl                           int32 [M, 2]: visible keys, block start
          anc                           int32 [M, ANC]: the position of the row's ancestor at each depth
        and the outputs (logits, greedy tokens and their pinned copy, the final normed rows)."""
        key = M if key is None else key
        rb = self._rb.get(key)
        if rb is not None:
            return rb
        c, dev = self.cfg, self.device
        n64 = 5 * M
        n32 = M * (2 + K.ANC)
        n32 += n32 % 2
        nb = 8 * n64 + 4 * n32

        def views(b):
            i64 = b[:8 * n64].view(torch.int64)
            i32 = b[8 * n64:].view(torch.int32)
            v = {name: i64[j * M:(j + 1) * M] for j, name in enumerate(("tok", "pos", "slot", "rslot", "zero"))}
            v["ctl"] = i32[:2 * M].view(M, 2)
            v["anc"] = i32[2 * M:2 * M + K.ANC * M].view(M, K.ANC)
            return v
        pin = dev.type == "cuda"
        hbuf = torch.zeros(nb, dtype=torch.uint8, pin_memory=pin)
        dbuf = torch.zeros(nb, dtype=torch.uint8, device=dev)
        copied = _event(dev)
        copied.record()
        rb = {"M": M, "hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied,
              "logits": torch.zeros(M, c.vocab_size, dtype=torch.float32, device=dev),
              "next": torch.zeros(M, dtype=torch.long, device=dev),
              "next_h": torch.zeros(M, dtype=torch.long, pin_memory=pin),
              "hidden": torch.zeros(M, c.hidden_size, dtype=self.dtype, device=dev),
              "attn": {}}
        self._rb[key] = rb
        return rb

    @staticmethod
    def _key(slot, M: int):
        return M if slot is None else (slot, M)

    def stage(self, token: int, st: FlintState, slot=None) -> None:
        """Host work of one decode step: the token, its position and slots, its attention control."""
        rb = self._rows_buffers(1, self._key(slot, 1))
        rb["copied"].synchronize()
        h, L = rb["h"], st.length
        h["tok"].numpy()[0] = token
        h["pos"].numpy()[0] = L
        h["slot"].numpy()[0] = L
        h["rslot"].numpy()[0] = L % max(self.ring, 1)
        h["ctl"].numpy()[0] = (L + 1, L)
        h["anc"].numpy()[0, 0] = L
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()

    def stage_rows(self, tokens: list[int], parents: list[int], st: FlintState, key=None) -> None:
        """Host work of a verify block: `tokens` in DFS pre-order with `parents` (-1 for the first row, which
        sits at the state's length). Nothing in `st` changes; `commit` keeps a path."""
        M = len(tokens)
        if M > ROWS_MAX + 1:
            raise ValueError(f"a verify block of {M} rows: the row kernels take {ROWS_MAX + 1} at most")
        rb = self._rows_buffers(M, key)
        rb["copied"].synchronize()
        h, L = rb["h"], st.length
        paths = []
        for i, p in enumerate(parents):
            if p >= i:
                raise ValueError(f"row {i}: parent {p} is not before it (DFS pre-order)")
            paths.append([i] if p < 0 else paths[p] + [i])
        d = np.asarray([len(pa) - 1 for pa in paths], dtype=np.int64)
        h["tok"].numpy()[:] = tokens
        h["pos"].numpy()[:] = L + d
        h["slot"].numpy()[:] = L + np.arange(M)
        h["rslot"].numpy()[:] = (L + np.arange(M)) % max(self.ring, 1)
        h["ctl"].numpy()[:, 0] = L + d + 1
        h["ctl"].numpy()[:, 1] = L
        anc = h["anc"].numpy()
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = L + np.asarray(pa)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()
        st.block = (list(tokens), list(parents), paths, M if key is None else key)

    def rows_step(self, st: FlintState, M: int, ns: int | None = None, key=None) -> torch.Tensor:
        """M rows from the device buffers (`stage` / `stage_rows` first) through every layer and the head: the
        rows' K and V land at their slots (past the length for a block); `st.length` is the caller's. `ns`: the
        full layers' attention class (kernels.class_of). Returns logits [M, vocab] fp32 (a static buffer) and
        leaves the greedy tokens in rb["next"] and its pinned host copy."""
        rb = self._rows_buffers(M, key)
        self._rows([(st, rb, M, ns or K.class_of(st.length + M, st.cap))])
        return rb["logits"]

    def _rows(self, parts: list) -> None:
        """The rows of one or more states [(state, its row buffers, rows, attention class)] in one pass: the
        projections, the experts and the head over every row at once (their weights read once), attention per
        state on its own rows and caches. Every kernel is row-invariant, so a state's rows get the bits of its
        lone call."""
        c = self.cfg
        eps = c.rms_norm_eps
        cuts = [0]
        for _, _, M, _ in parts:
            cuts.append(cuts[-1] + M)
        R = cuts[-1]
        sl = [slice(cuts[j], cuts[j + 1]) for j in range(len(parts))]
        one = len(parts) == 1
        tok = parts[0][1]["d"]["tok"] if one else torch.cat([rb["d"]["tok"] for _, rb, _, _ in parts])
        h = self.W["embed_tokens.weight"].index_select(0, tok)
        h, n = LK.add_rms(h, None, self.W["layers.0.input_layernorm.weight"], eps)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            a = p + "self_attn."
            sliding, rope = c.is_sliding(li), c.has_rope(li)
            qkv = self._fused(n, a, "qkv", ("q_proj", "k_proj", "v_proj"))
            o = torch.empty(R, c.num_attention_heads, c.head_dim, dtype=self.dtype, device=self.device)
            for (st, rb, M, ns), r in zip(parts, sl):
                d = rb["d"]
                q = LK.qk_prep(qkv[r], d["pos"] if rope else d["zero"], d["rslot"] if sliding else d["slot"],
                               self.W[a + "q_norm.weight"], self.W[a + "k_norm.weight"],
                               self.rope_cos if rope else self.id_cos, self.rope_sin if rope else self.id_sin,
                               st.k[li], st.v[li], c.num_attention_heads, c.num_key_value_heads, eps)
                K.attn_rows(q, st.k[li], st.v[li], d["ctl"], d["anc"], ns, self._sink[li],
                            window=c.sliding_window if sliding else 0, ring=self.ring,
                            bufs=rb["attn"].setdefault("sw" if sliding else ns, {}), out=o[r])
            out = self.lin(o.view(R, -1), self.W[a + "o_proj.weight"])
            h, n = LK.add_rms(h, out, self.W[p + "post_attention_layernorm.weight"], eps)
            m = self.moe_rows(n, li) if c.is_moe(li) else self.glu_rows(n, p + "mlp.", c.intermediate_size)
            nxt = self.W[self.pre(li + 1) + "input_layernorm.weight"] if li + 1 < c.num_hidden_layers \
                else self.W["norm.weight"]
            h, n = K.add2_rms(h, m[0], m[1], nxt, eps) if isinstance(m, tuple) else LK.add_rms(h, m, nxt, eps)
        if one:
            rb = parts[0][1]
            rb["hidden"].copy_(n)
            self.head(n, out=rb["logits"])
        else:
            logits = self.head(n)
            for (_, rb, _, _), r in zip(parts, sl):
                rb["hidden"].copy_(n[r])
                rb["logits"].copy_(logits[r])
        for _, rb, _, _ in parts:
            torch.argmax(rb["logits"], dim=-1, out=rb["next"])
            rb["next_h"].copy_(rb["next"], non_blocking=True)

    def step(self, st: FlintState, ns: int | None = None, slot=None) -> torch.Tensor:
        """One decode token from the device buffers (`stage` first): its KV rows land at the length;
        `st.length` is the caller's. Logits [1, vocab] fp32 (static)."""
        return self.rows_step(st, 1, ns=ns, key=self._key(slot, 1))

    @property
    def g(self) -> dict:
        return self._rb.get(1, {})

    @torch.no_grad()
    def verify(self, tokens: list[int], parents: list[int], st: FlintState, graphs: dict | None = None,
               slot=None) -> list[int]:
        """The greedy token at every row of a draft block (DFS pre-order), eager or from `graphs`
        ((M, class) -> VerifyGraph). Nothing in `st` moves until `commit`."""
        M = len(tokens)
        key = self._key(slot, M)
        ns = K.class_of(st.length + M, st.cap)
        gr = None
        if graphs is not None:
            gr = graphs.get((M, ns))
            if gr is None:
                gr = graphs[(M, ns)] = VerifyGraph(self, st, M, ns, key)
                gr.capture()
        t0 = time.perf_counter()
        self.stage_rows(tokens, parents, st, key=key)
        self.host_ms["stage_rows"] = self.host_ms.get("stage_rows", 0.0) + 1e3 * (time.perf_counter() - t0)
        if gr is not None:
            gr.replay()
        else:
            self.rows_step(st, M, ns=ns, key=key)
        self._width = M
        _sync(self.device)
        return self._rb[key]["next_h"].tolist()

    @torch.no_grad()
    def verify_multi(self, items: list) -> list:
        """The verify blocks of several states [(tokens, parents, state)] in one pass of at most 32 rows: each
        state's greedy tokens are its lone verify's, bit for bit. Eager; each state commits its block."""
        if sum(len(t) for t, _, _ in items) > ROWS_MAX:
            raise ValueError(f"a joint pass of {sum(len(t) for t, _, _ in items)} rows: at most {ROWS_MAX}")
        parts = []
        for j, (tokens, parents, st) in enumerate(items):
            M = len(tokens)
            key = ("joint", j, M)
            self.stage_rows(tokens, parents, st, key=key)
            parts.append((st, self._rows_buffers(M, key), M, K.class_of(st.length + M, st.cap)))
        self._rows(parts)
        _sync(self.device)
        return [rb for _, rb, _, _ in parts]

    @torch.no_grad()
    def decode_multi(self, items: list, graphs: dict | None = None) -> list:
        """One decode token of each of several states [(token, state, slot)] in one pass, each row the bits of
        that state's lone decode step. From `graphs` (engine/models/graph_lru.py), else eager."""
        parts = []
        for token, st, slot in items:
            self.stage(token, st, slot)
            parts.append((st, self._rows_buffers(1, self._key(slot, 1)), 1, K.class_of(st.length + 1, st.cap)))
        if graphs is not None:
            key = tuple((slot, ns) for (_, _, slot), (_, _, _, ns) in zip(items, parts))
            g = graphs.take(key, lambda: JointDecodeGraph(self, parts).captured())
            g.graph.replay()
        else:
            self._rows(parts)
        _sync(self.device)
        return [rb for _, rb, _, _ in parts]

    @torch.no_grad()
    def commit(self, st: FlintState, path: list[int], graphs: dict | None = None) -> None:
        """Keep the rows `path` (row 0 first, each the next one's parent) of the last verify block: the path's
        KV rows moved to their positions (the full layers' slots and the rings'), the length moved."""
        tokens, parents, _, rbkey = st.block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        key = self.stage_commit(st, path)
        if rbkey != len(tokens):
            graphs = None
        if graphs is not None and path and key[2]:
            g = graphs.get(key)
            if g is None:
                g = graphs[key] = CommitGraph(self, st, key)
                g.capture()
            g.graph.replay()
        elif path and key[2]:
            self.commit_dev(st, key)
        st.length += len(path)

    def _commit_bufs(self, key) -> dict:
        cb = self._rb.get(("commit",) + key)
        if cb is not None:
            return cb
        dev = self.device
        n = key[1]
        sizes = (("src", n), ("dst", n), ("rsrc", n), ("rdst", n))
        tot = sum(w for _, w in sizes)
        hbuf = torch.zeros(tot, dtype=torch.int64, pin_memory=dev.type == "cuda")
        dbuf = torch.zeros(tot, dtype=torch.int64, device=dev)

        def views(b):
            out, k = {}, 0
            for name, w in sizes:
                out[name] = b[k:k + w]
                k += w
            return out
        copied = _event(dev)
        copied.record()
        cb = {"hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied}
        self._rb[("commit",) + key] = cb
        return cb

    def stage_commit(self, st: FlintState, path: list[int]) -> tuple:
        M, n = len(st.block[0]), len(path)
        moved = any(path[j] != j for j in range(n))
        key = (M, n, moved)
        if not path or not moved:
            return key
        cb = self._commit_bufs(key)
        cb["copied"].synchronize()
        h, L = cb["h"], st.length
        R = max(self.ring, 1)
        src = np.asarray([L + r for r in path], dtype=np.int64)
        dst = L + np.arange(n, dtype=np.int64)
        h["src"].numpy()[:] = src
        h["dst"].numpy()[:] = dst
        h["rsrc"].numpy()[:] = src % R
        h["rdst"].numpy()[:] = dst % R
        cb["dbuf"].copy_(cb["hbuf"], non_blocking=True)
        cb["copied"].record()
        return key

    def commit_dev(self, st: FlintState, key) -> None:
        """The device half of `commit` (capturable): the path's rows moved down in every layer."""
        c = self.cfg
        d = self._commit_bufs(key)["d"]
        for li in range(c.num_hidden_layers):
            s, t = (d["rsrc"], d["rdst"]) if c.is_sliding(li) else (d["src"], d["dst"])
            st.k[li].index_copy_(1, t, st.k[li].index_select(1, s))
            st.v[li].index_copy_(1, t, st.v[li].index_select(1, s))

    # ------------------------------------------------------------------------- prompt chunks
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, st: FlintState, head: str = "last") -> torch.Tensor | None:
        """A prompt (any length) in chunks of PREFILL_CHUNK: the state advances. Logits of the last row
        [1, vocab] fp32 (`head="all"`: of every row of the last chunk; "none": None)."""
        tokens = tokens.view(-1).to(self.device)
        out, rows = None, []
        for a in range(0, tokens.numel(), PREFILL_CHUNK):
            chunk = tokens[a:a + PREFILL_CHUNK]
            last = a + PREFILL_CHUNK >= tokens.numel()
            out = self.forward_chunk(chunk, st, head=head if last else "none")
            rows.append(self.prompt_hidden)
        self.prompt_hidden = rows[0] if len(rows) == 1 else torch.cat(rows)
        return out

    def forward_chunk(self, tokens: torch.Tensor, st: FlintState, head: str = "last") -> torch.Tensor | None:
        from tools import fn_kernels as FK
        c = self.cfg
        T, L, eps = tokens.numel(), st.length, c.rms_norm_eps
        if L + T > st.cap:
            raise ValueError(f"{L + T} rows exceed the state's capacity {st.cap}")
        dev = self.device
        pos = torch.arange(L, L + T, device=dev)
        h = self.W["embed_tokens.weight"].index_select(0, tokens)
        h, n = LK.add_rms(h, None, self.W["layers.0.input_layernorm.weight"], eps) if dev.type == "cuda" \
            else LK.add_rms_reference(h, None, self.W["layers.0.input_layernorm.weight"], eps)
        add_rms = LK.add_rms if dev.type == "cuda" else LK.add_rms_reference
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            out = self._attn_chunk(n, li, st, pos)
            h, n = add_rms(h, out, self.W[p + "post_attention_layernorm.weight"], eps)
            if c.is_moe(li):
                logits = F.linear(n, self._gate[li]).float()
                if dev.type == "cuda":
                    ids, wts = K.router(logits, self._bias.get(li), c.num_experts_per_tok, c.route_eps,
                                        scale=c.route_scale)
                else:
                    ids, wts = K.router_reference(logits, self._bias.get(li), c.num_experts_per_tok, c.route_eps,
                                                  c.route_scale)
                if li in self.banks and _is_nvfp4_bank(self.banks[li]):
                    yd = self._moe_nvfp4_chunk(n, ids, self.banks[li])
                elif li in self.banks:
                    from tools import fn_prefill_kernels as PK
                    yd = PK.moe_grouped(n, ids, self.banks[li])
                else:
                    yd = self._moe_bf16_chunk(n, ids, li)
                m = FK.moe_combine(yd, ids, wts) if dev.type == "cuda" else LK.combine_reference(yd, ids, wts)
                if c.shared_expert_intermediate_size:
                    m = m + self._glu_chunk(n, p + "mlp.shared_expert.", c.shared_expert_intermediate_size)
            else:
                m = self._glu_chunk(n, p + "mlp.", c.intermediate_size)
            nxt = self.W[self.pre(li + 1) + "input_layernorm.weight"] if li + 1 < c.num_hidden_layers \
                else self.W["norm.weight"]
            h, n = add_rms(h, m, nxt, eps)
        st.length = L + T
        self.prompt_hidden = n
        if head == "none":
            return None
        return self.head(n[-1:] if head == "last" else n)

    def _glu_chunk(self, n, p, I):
        gu = self._fused(n, p, "gate_up", ("gate_proj", "up_proj"))
        return self.lin(F.silu(gu[:, :I]) * gu[:, I:], self.W[p + "down_proj.weight"])

    def _attn_chunk(self, n, li, st, pos):
        """Attention of a prompt chunk: q|k|v, the prep kernel (norms, rotary on a sliding layer), then
        kernels.prompt_attention: a full layer over the cache's rows [0, L + T) (the chunk's K and V written at
        their positions first); a sliding layer over its window's earlier rows from the ring and the chunk's
        own, after which the ring keeps the last rows."""
        c = self.cfg
        a = self.pre(li) + "self_attn."
        T, L = n.shape[0], int(pos[0])
        Hq, Hk, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        qkv = self._fused(n, a, "qkv", ("q_proj", "k_proj", "v_proj"))
        rope = c.has_rope(li)
        cos, sin = (self.rope_cos, self.rope_sin) if rope else (self.id_cos, self.id_sin)
        qpos = pos if rope else torch.zeros_like(pos)
        prep = LK.qk_prep if qkv.is_cuda else _qk_prep_cpu
        if not c.is_sliding(li):
            q = prep(qkv, qpos, pos, self.W[a + "q_norm.weight"], self.W[a + "k_norm.weight"], cos, sin,
                     st.k[li], st.v[li], Hq, Hk, c.rms_norm_eps)
            o = K.prompt_attention(q, st.k[li][:, :L + T], st.v[li][:, :L + T], 0, self._sink[li], 0)
        else:
            W, R = c.sliding_window, self.ring
            lo = max(0, L - (W - 1))
            w0 = L - lo
            kt = torch.empty(Hk, w0 + T, D, dtype=self.dtype, device=n.device)
            vt = torch.empty_like(kt)
            if w0:
                old = torch.arange(lo, L, device=n.device) % R
                kt[:, :w0] = st.k[li].index_select(1, old)
                vt[:, :w0] = st.v[li].index_select(1, old)
            q = prep(qkv, qpos, torch.arange(w0, w0 + T, device=n.device), self.W[a + "q_norm.weight"],
                     self.W[a + "k_norm.weight"], cos, sin, kt, vt, Hq, Hk, c.rms_norm_eps)
            o = K.prompt_attention(q, kt, vt, lo, self._sink[li], W)
            keep = min(T, R - CAPTURE_ROWS)
            rows = torch.arange(L + T - keep, L + T, device=n.device)
            st.k[li].index_copy_(1, rows % R, kt[:, w0 + T - keep:])
            st.v[li].index_copy_(1, rows % R, vt[:, w0 + T - keep:])
        return self.lin(o.reshape(T, -1), self.W[a + "o_proj.weight"])

    def _moe_bf16_chunk(self, n, ids, li):
        """The BF16 experts of a chunk: each expert once over its rows -> yd [T S, H] fp32."""
        e = self.experts[li]
        if n.is_cuda and GROUPED:
            return K.moe_grouped_glu(n, ids, e.w13, e.w2, e.I2 // 2, False, e.E)
        T, S = ids.shape
        I = e.I2 // 2
        flat = ids.reshape(-1).long()
        yd = torch.empty(T * S, n.shape[1], dtype=torch.float32, device=n.device)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=e.E).tolist()
        o = 0
        for ex, cnt in enumerate(counts):
            if not cnt:
                continue
            pairs = order[o:o + cnt]
            o += cnt
            x = n.index_select(0, pairs // S)
            gu = F.linear(x, e.w13[ex])
            yd[pairs] = F.linear(F.silu(gu[:, :I]) * gu[:, I:], e.w2[ex]).float()
        return yd

    def _moe_nvfp4_chunk(self, n, ids, bank):
        if n.is_cuda and GROUPED:
            return K.moe_grouped_glu(n, ids, (bank.gw, bank.gs, bank.g2), (bank.dw, bank.ds, bank.d2), bank.I, True,
                                     bank.E)
        return self._moe_nvfp4_chunk_loop(n, ids, bank)

    def _moe_nvfp4_chunk_loop(self, n, ids, bank):
        """An NVFP4 bank's experts over a chunk: each routed expert's rows through cuBLAS on its weights unpacked
        to bf16 (a few rows, the pair kernel) -> yd [T S, H] fp32."""
        T, S = ids.shape
        I = bank.I
        flat = ids.reshape(-1).long()
        yd = torch.empty(T * S, n.shape[1], dtype=torch.float32, device=n.device)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=bank.E).tolist()
        o = 0
        for ex, cnt in enumerate(counts):
            if not cnt:
                continue
            pairs = order[o:o + cnt]
            o += cnt
            x = n.index_select(0, pairs // S)
            gu_w, d_w = bank.expert_bf16(ex)
            gu = F.linear(x, gu_w)
            yd[pairs] = F.linear(F.silu(gu[:, :I]) * gu[:, I:], d_w).float()
        return yd

    # ------------------------------------------------------------------------- the load-time check
    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 24, log=print) -> dict:
        """The load-time check that a verify block's rows are the decode step's: decode `steps` tokens one at a
        time from a prefill, then verify the same tokens as one chain block from the prefill's state, and
        compare every row's logits bit for bit. Any difference leaves `spec_ok` False (speculation off)."""
        st = self.new_state(min(self.max_len, len(prompt) + steps + CAPTURE_ROWS))
        lg = self.prefill(torch.tensor(prompt), st)
        L = st.length
        toks, rows = [int(lg[0].argmax())], []
        for _ in range(min(steps, ROWS_MAX)):
            self.stage(toks[-1], st)
            out = self.step(st)
            st.length += 1
            rows.append(out[0].clone())
            toks.append(int(out[0].argmax()))
        st.length = L
        M = len(rows)
        self.stage_rows(toks[:M], list(range(-1, M - 1)), st)
        got = self.rows_step(st, M)
        bad = next((i for i in range(M) if not torch.equal(got[i], rows[i])), None)
        self.spec_ok = bad is None
        res = {"ok": self.spec_ok, "rows": M, "first_diff": bad}
        if log:
            log(f"[flint-selfcheck] {'PASS' if self.spec_ok else 'FAIL: speculation stays off'}: {res}")
        return res

    spec_ok = False


def _qk_prep_cpu(qkv, pos, slot, qw, kw, cos, sin, kc, vc, hq, hk, eps):
    """LFM2's qk_prep on the CPU (its torch twin, the K and V written at `slot`)."""
    D = kc.shape[2]
    q, k, v = LK.qk_prep_reference(qkv, pos, qw, kw, cos, sin, hq, hk, D, eps)
    kc.index_copy_(1, slot, k.transpose(0, 1))
    vc.index_copy_(1, slot, v.transpose(0, 1))
    return q


class DecodeGraph:
    """`FlintEngine.step` captured once for a state and an attention class; `__call__(token)` stages the token,
    replays the class's graph (captured when the length first enters it) and moves the length."""

    def __init__(self, eng: FlintEngine, st: FlintState, slot=None):
        self.eng, self.st, self.slot = eng, st, slot
        self.rb = eng._rows_buffers(1, eng._key(slot, 1))
        self.graphs: dict = {}
        self.graph = None
        self.captures = 0

    def capture(self, ns: int) -> None:
        eng, st = self.eng, self.st
        saved = _state_saved(eng, st)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.stage(0, st, self.slot)
                eng.step(st, ns, self.slot)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g):
            eng.step(st, ns, self.slot)
        torch.cuda.synchronize()
        _state_restore(saved)
        self.graphs[ns] = g

    def _ready(self) -> None:
        ns = K.class_of(self.st.length + 1, self.st.cap)
        if ns not in self.graphs:
            self.capture(ns)
            self.captures += 1
        self.graph = self.graphs[ns]

    @torch.no_grad()
    def __call__(self, token: int) -> torch.Tensor:
        eng, st = self.eng, self.st
        self._ready()
        eng.stage(token, st, self.slot)
        self.graph.replay()
        st.length += 1
        return self.rb["logits"]

    @torch.no_grad()
    def greedy(self, token: int) -> int:
        self(token)
        torch.cuda.current_stream().synchronize()
        return int(self.rb["next_h"][0])


class JointDecodeGraph:
    """`FlintEngine._rows` of several states' decode tokens captured once for that set of states and classes;
    the warm-up runs write KV rows, so each state is saved and restored around them."""

    def __init__(self, eng: FlintEngine, parts: list):
        self.eng, self.parts = eng, parts
        self.graph = None

    def capture(self) -> None:
        eng, parts = self.eng, self.parts
        saved = [x for st, _, _, _ in parts for x in _state_saved(eng, st)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng._rows(parts)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng._rows(parts)
        torch.cuda.synchronize()
        _state_restore(saved)

    def captured(self) -> "JointDecodeGraph":
        self.capture()
        return self


class VerifyGraph:
    """`FlintEngine.rows_step` of an M-row verify block captured once for a state and a class."""

    def __init__(self, eng: FlintEngine, st: FlintState, M: int, ns: int, key=None):
        self.eng, self.st, self.M, self.ns, self.key = eng, st, M, ns, M if key is None else key
        self.graph = None

    def capture(self) -> None:
        eng, st, M = self.eng, self.st, self.M
        saved = _state_saved(eng, st)
        eng.stage_rows([0] * M, list(range(-1, M - 1)), st, key=self.key)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.rows_step(st, M, ns=self.ns, key=self.key)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.rows_step(st, M, ns=self.ns, key=self.key)
        torch.cuda.synchronize()
        _state_restore(saved)

    def replay(self) -> None:
        self.graph.replay()


class CommitGraph:
    """`FlintEngine.commit_dev` for a verify block of M rows and a moved path of n rows; the warm-up runs write
    the state, which is saved and restored around them."""

    def __init__(self, eng: FlintEngine, st: FlintState, key: tuple):
        self.eng, self.st, self.key = eng, st, key
        self.graph = None

    def capture(self) -> None:
        eng, st = self.eng, self.st
        saved = _state_saved(eng, st)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            eng.commit_dev(st, self.key)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.commit_dev(st, self.key)
        torch.cuda.synchronize()
        _state_restore(saved)


def _state_saved(eng: FlintEngine, st: FlintState) -> list:
    """What a capture's warm-up may write, cloned: the full layers' K and V rows from the length on
    (`CAPTURE_ROWS`), the rings whole (the rows past the length wrap)."""
    L = st.length
    out = []
    for d in (st.k, st.v):
        for li, t in d.items():
            if li in st.ring:
                out.append((t, 1, 0, t.shape[1]))
            else:
                out.append((t, 1, L, min(CAPTURE_ROWS, t.shape[1] - L)))
    return [(t, ax, lo, n, t.narrow(ax, lo, n).clone()) for t, ax, lo, n in out if n > 0]


def _state_restore(saved: list) -> None:
    for t, ax, lo, n, x in saved:
        t.narrow(ax, lo, n).copy_(x)
