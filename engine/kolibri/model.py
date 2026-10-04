"""Kolibri-1's forward on the engine: the layer loop, prefill, decode, and the decode step as a CUDA graph.

Per layer, as `tools/kolibri_ref.layer_forward` (the plugin's forward written out):

    r += post_attn_norm(attn(input_layernorm(r)))
    r += post_ffn_norm(routed experts + shared expert (post_attention_layernorm(r)))

The residual stream is fp32; every projection reads a bf16 normed input; the MoE sum and the
logits are fp32. Attention and its KV cache come from `engine.kolibri.attn` when it is
there and from `TorchAttention` below otherwise (the reference: plain torch, eager only).
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from engine.kolibri.config import KolibriConfig
from engine.kolibri import kernels as KK
from engine.kolibri.kernels import add_rms2, moe_experts, rms, rms_fused, route, route_fused

#: the FP8 shared expert as GEMM, SwiGLU kernel, GEMM, its output added in the MoE combine (0: torch)
FUSED_SHARED = os.environ.get("KOLIBRI_FUSED_SHARED", "1") != "0"


# ------------------------------------------------------------------------------ reference attention
class TorchKV:
    """Full rows for every layer, bf16 [max_len, nkv, hd]; `length` is managed by the engine."""

    def __init__(self, cfg: KolibriConfig, max_len: int, device, dtype=torch.bfloat16):
        self.cfg, self.max_len, self.length = cfg, max_len, 0
        self.k = [torch.zeros(max_len, cfg.nkv, cfg.hd, dtype=dtype, device=device) for _ in range(cfg.layers)]
        self.v = [torch.zeros(max_len, cfg.nkv, cfg.hd, dtype=dtype, device=device) for _ in range(cfg.layers)]

    def reset(self):
        self.length = 0

    def truncate(self, n: int):
        assert 0 <= n <= self.length, (n, self.length)
        self.length = n


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """NeoX rotary over the whole head, fp32 (`tools/kolibri_ref.rope`)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d))
    f = pos.float()[:, None] * inv[None, :]
    cos, sin = torch.cat([f.cos()] * 2, -1), torch.cat([f.sin()] * 2, -1)
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
    return xf * cos[:, None, :] + torch.cat([-x2, x1], -1) * sin[:, None, :]


class TorchAttention:
    """The attention block in plain torch over `TorchKV`; the same interface as `engine.kolibri.attn`."""

    graphable = False

    def __init__(self, cfg: KolibriConfig):
        self.cfg = cfg

    def make_kv(self, max_len, device):
        return TorchKV(self.cfg, max_len, device)

    def _qkv(self, x, lw, pos0):
        c = self.cfg
        T = x.shape[0]
        y = lw.qkv.matmul(x)
        q = rms(y[:, : c.q_size].view(T, c.nq, c.hd), lw.q_norm, c.eps)
        k = rms(y[:, c.q_size: c.q_size + c.kv_size].view(T, c.nkv, c.hd), lw.k_norm, c.eps)
        v = y[:, c.q_size + c.kv_size:].reshape(T, c.nkv, c.hd)
        if lw.sliding:
            pos = torch.arange(pos0, pos0 + T, device=x.device)
            q = rope(q, pos, c.rope_theta).to(torch.bfloat16)
            k = rope(k, pos, c.rope_theta).to(torch.bfloat16)
        return q, k, v

    def attn_prefill(self, x, lw, kv, pos0: int):
        c = self.cfg
        T = x.shape[0]
        q, k, v = self._qkv(x, lw, pos0)
        L = lw.index
        kv.k[L][pos0:pos0 + T] = k
        kv.v[L][pos0:pos0 + T] = v
        end = pos0 + T
        k0 = max(0, pos0 - (c.window - 1)) if lw.sliding else 0
        kk = kv.k[L][k0:end].float().transpose(0, 1).repeat_interleave(c.nq // c.nkv, 0)
        vv = kv.v[L][k0:end].float().transpose(0, 1).repeat_interleave(c.nq // c.nkv, 0)
        out = torch.empty(T, c.nq, c.hd, dtype=torch.float32, device=x.device)
        for i0 in range(0, T, 512):
            i1 = min(T, i0 + 512)
            pq = torch.arange(pos0 + i0, pos0 + i1, device=x.device)[:, None]
            pk = torch.arange(k0, end, device=x.device)[None, :]
            allowed = pk <= pq
            if lw.sliding:
                allowed &= pk >= pq - (c.window - 1)
            o = F.scaled_dot_product_attention(q[i0:i1].float().transpose(0, 1), kk, vv,
                                               attn_mask=allowed[None], scale=c.hd ** -0.5)
            out[i0:i1] = o.transpose(0, 1)
        return lw.o.matmul(out.reshape(T, c.q_size).to(torch.bfloat16))

    def attn_decode(self, x, lw, kv, pos):
        p = int(pos) if not torch.is_tensor(pos) else int(pos.item())
        return self.attn_prefill(x, lw, kv, p)


def attention_impl(cfg: KolibriConfig, prefer: str = "auto"):
    """`engine.kolibri.attn` when present (and not `prefer="torch"`), else the reference."""
    if prefer != "torch":
        try:
            from engine.kolibri import attn as A
            return A.KolibriAttention(cfg)
        except (ImportError, AttributeError):
            if prefer == "kernel":
                raise
    return TorchAttention(cfg)


# ------------------------------------------------------------------------------ the engine
class KolibriEngine:
    def __init__(self, cfg: KolibriConfig, layers, emb, final_norm, head, device, max_len: int = 65536,
                 attn=None, info: dict | None = None, chunk: int = 2048):
        self.cfg, self.layers, self.emb, self.final_norm, self.head = cfg, layers, emb, final_norm, head
        self.device = torch.device(device)
        self.attn = attn or attention_impl(cfg)
        self.kv = self.attn.make_kv(max_len, self.device)
        self.max_len = max_len
        self.info = info or {}
        self.chunk = chunk
        self.shared_id = cfg.experts
        self._graph = None
        # residual-stream taps for a block drafter (`set_taps`); none by default
        self.tap_layers: tuple = ()
        self._tap_at: dict = {}
        self.dec_taps = None
        self.on_taps = None

    @classmethod
    def load(cls, set_dir: str, fp8_dir: str | None = None, device="cuda", max_len: int = 65536,
             attn: str | None = None, attention: str = "auto", graphs: bool = True, log=print):
        from engine.kolibri.weights import load_all
        cfg, layers, emb, norm, head, info = load_all(set_dir, fp8_dir, device, attn=attn, log=log)
        eng = cls(cfg, layers, emb, norm, head, device, max_len, attn=attention_impl(cfg, attention), info=info)
        eng.use_graphs = graphs and getattr(eng.attn, "graphable", False) and eng.device.type == "cuda"
        log(f"[kolibri] attention {type(eng.attn).__module__}.{type(eng.attn).__name__}, "
            f"max_len {max_len}, decode graph {'on' if eng.use_graphs else 'off'}")
        return eng

    # -- one layer
    def _moe(self, x, lw):
        sid = self.shared_id if lw.shared is None else None
        ids, w = route_fused(x, lw.gate, lw.bias, self.cfg.topk, sid, self.cfg.norm_topk_prob)
        # one decode row: the combine rides in the next add_rms2 launch (KOLIBRI_FUSE_COMBINE)
        parts = x.is_cuda and x.shape[0] == 1 and KK.FUSE_COMBINE
        if lw.shared is not None and x.is_cuda and FUSED_SHARED:
            # the shared expert's output rides into the routed combine (no .float() and add launches)
            return moe_experts(x, ids, w, lw.G, lw.U, lw.D, extra=lw.shared.act(x), parts=parts)
        y = moe_experts(x, ids, w, lw.G, lw.U, lw.D, parts=parts and lw.shared is None)
        return y if lw.shared is None else y + lw.shared(x)

    def _hidden(self, ids: torch.Tensor, pos, decode: bool) -> torch.Tensor:
        """The final-normed hidden state, fp32 [T, H]. Per layer:
        r += post_attn_norm(attn(x)); x2 = post_attention_layernorm(r);
        r += post_ffn_norm(moe(x2)); x = the next layer's input_layernorm(r) (or the final norm),
        each `r += norm(..); x = norm(r)` pair one fused kernel on the GPU."""
        c = self.cfg
        r = self.emb[ids].float()
        x = rms_fused(r, self.layers[0].n_in, c.eps)
        n = len(self.layers)
        for i, lw in enumerate(self.layers):
            a = self.attn.attn_decode(x, lw, self.kv, pos) if decode else self.attn.attn_prefill(x, lw, self.kv, pos)
            r, x2 = add_rms2(r, a, lw.n_pa, lw.n_pal, c.eps)
            y = self._moe(x2, lw)
            if i + 1 < n:
                r, x = add_rms2(r, y, lw.n_pf, self.layers[i + 1].n_in, c.eps)
            else:
                r, x = add_rms2(r, y, lw.n_pf, self.final_norm, c.eps, torch.float32)
            j = self._tap_at.get(i)
            if j is not None:
                self._tap(j, r, pos, decode)
        return x

    # -- taps for a block drafter
    def set_taps(self, layers) -> None:
        """Keep the residual stream after these layers (fp32): the decode step writes them to
        `dec_taps` [len(layers), H] (in the captured graph; it is recaptured), a prefill hands each
        chunk's to `on_taps(j, rows [T, H], start)` when that is set, and a verify
        (`engine/kolibri/verify.py`) writes `Verifier.taps`. Empty: none (the default)."""
        self.tap_layers = tuple(int(L) for L in layers)
        self._tap_at = {L: j for j, L in enumerate(self.tap_layers)}
        self.dec_taps = (torch.zeros(len(self.tap_layers), self.cfg.hidden, dtype=torch.float32,
                                     device=self.device) if self.tap_layers else None)
        self._graph = None

    def _tap(self, j: int, r: torch.Tensor, pos, decode: bool) -> None:
        if decode:
            self.dec_taps[j].copy_(r[0])
        elif self.on_taps is not None:
            self.on_taps(j, r, pos)

    # -- public API
    def _ids(self, ids) -> torch.Tensor:
        if not torch.is_tensor(ids):
            ids = torch.tensor(list(ids), dtype=torch.long)
        return ids.to(self.device, torch.long).reshape(-1)

    @torch.inference_mode()
    def forward(self, ids, start: int = 0) -> torch.Tensor:
        """Every row's fp32 logits [T, V]; writes KV rows start..start+T-1; kv.length = start + T."""
        ids = self._ids(ids)
        T = ids.numel()
        assert start + T <= self.max_len, (start, T, self.max_len)
        out = []
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            h = self._hidden(ids[c0:c1], start + c0, False)
            out.append(self.head.logits(h))
        self.kv.length = start + T
        return torch.cat(out)

    @torch.inference_mode()
    def prefill(self, ids, start: int = 0, on_chunk=None) -> torch.Tensor:
        """fp32 logits [V] of the last token; kv.length = start + len(ids)."""
        ids = self._ids(ids)
        T = ids.numel()
        assert T > 0 and start + T <= self.max_len, (start, T, self.max_len)
        assert start <= self.kv.length, (start, self.kv.length)
        h = None
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            h = self._hidden(ids[c0:c1], start + c0, False)
            if c1 < T and on_chunk is not None:
                on_chunk(start + c1, start + T)
        self.kv.length = start + T
        return self.head.logits(h[-1:])[0]

    @torch.inference_mode()
    def decode(self, tok: int) -> torch.Tensor:
        """fp32 logits [V] for the next position after `tok` at position kv.length; kv.length += 1."""
        pos = self.kv.length
        assert pos < self.max_len, (pos, self.max_len)
        if getattr(self, "use_graphs", False):
            out = self._graph_step(tok, pos)
        else:
            h = self._hidden(torch.tensor([tok], device=self.device), pos, True)
            out = self.head.logits(h)[0]
        self.kv.length = pos + 1
        return out

    def truncate(self, n: int) -> None:
        self.kv.truncate(n)

    def reset(self) -> None:
        self.kv.reset()

    # -- the decode step as one CUDA graph (needs a graph-capturable attention)
    def _graph_step(self, tok: int, pos: int) -> torch.Tensor:
        if self._graph is None:
            self._capture(tok, pos)
        self._tok.fill_(tok)
        self._pos.fill_(pos)
        self._graph.replay()
        return self._logits[0]

    def _capture(self, tok: int = 0, pos: int = 0):
        """Warm up and capture at the step about to run (token `tok` at row `pos`): the warm-up and
        the capture write that row's K/V, and the replay right after rewrites it with the same
        values. Capturing at row 0 would overwrite the prompt's first row."""
        dev = self.device
        self._tok = torch.full((1,), tok, dtype=torch.long, device=dev)
        self._pos = torch.full((1,), pos, dtype=torch.int32, device=dev)
        saved = self.kv.length
        # warm-up on a side stream (Triton compiles, allocator settles), then capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                h = self._hidden(self._tok, self._pos, True)
                self.head.logits(h)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            h = self._hidden(self._tok, self._pos, True)
            self._logits = self.head.logits(h)
        self._graph = g
        self.kv.length = saved

    def decode_bytes(self) -> int:
        """Weight bytes one decode token reads (no KV): attention, 7 of 385 experts, router, head."""
        n = 0
        for lw in self.layers:
            per = (lw.G.nbytes + lw.U.nbytes + lw.D.nbytes) / lw.G.E
            sh = lw.shared.nbytes if lw.shared is not None else per
            n += lw.qkv.nbytes + lw.o.nbytes + per * self.cfg.topk + sh + lw.gate.numel() * 2
        return int(n + self.head.nbytes + self.cfg.hidden * 2)
