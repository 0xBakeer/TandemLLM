"""Kolibri-1 attention and its KV cache.

Kolibri-1 has 50 layers, every one with attention: 40 sliding-window layers and 10 full-attention
layers, in the pattern 4 sliding then 1 full. GQA with 48 query heads and 4 KV heads of 128. Each
head's q and k get an RMSNorm (`x * rsqrt(mean(x^2) + eps) * w`, NOT Qwen's `(1 + w)`) before
anything else. Sliding layers then apply NeoX RoPE (base 10,000) over the whole head and see the
512 rows before the query plus the query itself (`sliding_window` 513). Full layers have NO position
encoding at all. There is no output gate. The reference is `tools/kolibri_ref.py`, the forward of
Aleph Alpha's vLLM plugin written out in tensors.

THE CACHE (`KolibriKV`).

* Full layers: one row per token, the layout of Qwen's `engine.model.KVCache`
  (`k`, `v` [10, 1, Hkv, max_len, D]), so `engine/cache.py` snapshots and restores them as it
  does Qwen's. 2 KB a token a layer in BF16: 5.37 GB for the 10 layers at 262,144 tokens.
* Sliding layers: a RING of `R` rows a layer (default 640). The row with sequence index `i` lives
  in slot `i % R`, and `ring.idx[slot]` says which index the slot holds (-1 empty). Attention masks
  by that index, never by the slot, so the order of the slots does not matter and a rejected row
  past the length is invisible until it is overwritten. 52 MB for all 40 layers in BF16.

How far back the cache can be cut. After rows up to L - 1, the ring holds indices L - R .. L - 1. A
query at n needs n - 512 .. n - 1, so `truncate(n)` is exact for n >= L - (R - 512) = L - 128 at
R = 640, and raises `RingLost` below that. A cut further back needs the ring as it was at n: the
serving layer keeps ring snapshots ("anchors") at prefill-chunk boundaries (`kv.snapshot()`,
`kv.restore(snap, n)`), the same idea as the resident prefix of Qwen's engine, where the GDN state
plays the ring's part. `kv.state` is the ring, and it offers `tensors()` and `nbytes`, so
`engine/cache.py` can anchor and restore it unchanged.

THE CALLS.

* `attn_prefill(x, lw, kv, pos0)`: T rows from a host position. A chunk of `PREFILL_ROWS` rows or
  more runs SDPA: full layers over the cache rows in place with the causal mask aligned lower right
  (the fused backends take it), sliding layers over the previous 512 rows plus the chunk in query
  blocks of 512. Fewer rows (a prompt's short tail, a verify block) run the decode kernels on the
  GPU. Never changes `kv.length`.
* `attn_decode(x, lw, kv, pos)`: one row, `pos` an int32 device tensor; CUDA-graph capturable (no
  host read of `pos`, every allocation of fixed size): full layers through
  `tools/attn_kernels.decode_attention_dev`, sliding layers through the ring kernel with the start
  read on the device.
* `attn_block(x, lw, kv, start, positions, block_mask)`: a verify chain or tree (rows at `start`,
  RoPE at `positions`, `block_mask` the ancestor relation), for speculation later.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: A forward with this many rows or more is a prefill chunk and takes the SDPA path.
PREFILL_ROWS = int(os.environ.get("KOLIBRI_PREFILL_ROWS", "64"))
#: Query block of the sliding layers' prefill: each block reads at most QBLOCK + 512 keys.
QBLOCK = 512
#: Rows of the sliding ring a layer.
RING_ROWS = int(os.environ.get("KOLIBRI_RING_ROWS", "640"))
#: q/k norm and RoPE in one Triton launch on the GPU (0: the torch ops).
FUSED_QK = os.environ.get("KOLIBRI_FUSED_QK", "1") != "0"
#: the decode step's attention in three launches a layer (tools/kolibri_attn_kernels.decode_fused;
#: BF16 KV only, FP8 KV keeps the older path). 0: the older path.
DEC_FUSED = os.environ.get("KOLIBRI_DEC_FUSED", "1") != "0"
FP8_MAX = 448.0


class RingLost(RuntimeError):
    """A cut further back than the ring remembers: restore an anchor or prefill from 0."""


@dataclass
class AttnSpec:
    hidden: int
    nq: int
    nkv: int
    hd: int
    window: int
    rope_theta: float
    eps: float
    layer_types: list

    @classmethod
    def of(cls, cfg) -> "AttnSpec":
        """From `KolibriConfig`, or from a checkpoint's config dict."""
        if isinstance(cfg, AttnSpec):
            return cfg
        if isinstance(cfg, dict):
            return cls(hidden=int(cfg["hidden_size"]), nq=int(cfg["num_attention_heads"]),
                       nkv=int(cfg["num_key_value_heads"]), hd=int(cfg["head_dim"]),
                       window=int(cfg["sliding_window"]),
                       rope_theta=float(cfg.get("rope_theta", 10000.0)),
                       eps=float(cfg["rms_norm_eps"]), layer_types=list(cfg["layer_types"]))
        return cls(hidden=cfg.hidden, nq=cfg.nq, nkv=cfg.nkv, hd=cfg.hd, window=cfg.window,
                   rope_theta=cfg.rope_theta, eps=cfg.eps, layer_types=list(cfg.layer_types))

    def sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    @property
    def full_layers(self) -> list:
        return [i for i, t in enumerate(self.layer_types) if t != "sliding_attention"]

    @property
    def sliding_layers(self) -> list:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    @property
    def q_size(self) -> int:
        return self.nq * self.hd

    @property
    def kv_size(self) -> int:
        return self.nkv * self.hd


def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., D] -> e4m3 codes and one fp32 scale per vector (amax / 448); the arithmetic of
    `tools.attn_kernels.quantize_kv`, here so the CPU tests need no Triton."""
    amax = x.float().abs().amax(-1).clamp_min(1e-12)
    s = amax / FP8_MAX
    codes = (x.float() / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, s


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """An e4m3 tensor as its bytes (index_copy/index_select have no e4m3 kernels on every
    backend); any other tensor as it is."""
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


class SlidingRing:
    """The 40 sliding layers' last R rows, one ring a layer, and the index each slot holds."""

    def __init__(self, spec: AttnSpec, rows: int, device, dtype=torch.bfloat16, fp8: bool = False):
        if rows < spec.window + 1:
            raise ValueError(f"a ring of {rows} rows cannot hold the {spec.window}-row window "
                             f"and one more row")
        n = len(spec.sliding_layers)
        self.R = int(rows)
        self.window = spec.window
        self.slot = {l: i for i, l in enumerate(spec.sliding_layers)}
        self.fp8 = bool(fp8)
        cdt = torch.float8_e4m3fn if fp8 else dtype
        self.k = torch.zeros(n, 1, spec.nkv, self.R, spec.hd, dtype=cdt, device=device)
        self.v = torch.zeros_like(self.k)
        self.ks = (torch.zeros(n, 1, spec.nkv, self.R, dtype=torch.float32, device=device)
                   if fp8 else None)
        self.vs = torch.zeros_like(self.ks) if fp8 else None
        self.idx = torch.full((self.R,), -1, dtype=torch.int32, device=device)
        self.primed = False

    # what engine/cache.py snapshots: every buffer that is part of the state
    def tensors(self) -> list:
        t = [self.k, self.v, self.idx]
        if self.fp8:
            t += [self.ks, self.vs]
        return t

    @property
    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in self.tensors())

    def reset(self) -> None:
        self.idx.fill_(-1)
        self.primed = False

    def slots(self, first: int, n: int, device) -> torch.Tensor:
        return torch.arange(first, first + n, device=device) % self.R

    def mark(self, start: int, T: int) -> None:
        """The slots a forward of T rows at `start` writes now hold those indices. Every layer
        of a forward calls it with the same values; the table is shared by all 40 rings."""
        n = min(T, self.R)
        first = start + T - n
        ar = torch.arange(first, first + n, device=self.idx.device, dtype=torch.int32)
        self.idx[(ar % self.R).long()] = ar

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """k, v [Hkv, T, D]: rows start..start+T-1 (only the last R are kept)."""
        i = self.slot[layer]
        T = k.shape[1]
        n = min(T, self.R)
        k, v = k[:, T - n:], v[:, T - n:]
        sl = self.slots(start + T - n, n, k.device)
        self._put(i, sl, k, v)

    def _put(self, i, sl, k, v) -> None:
        if self.fp8:
            kc, ksc = quantize_kv(k)
            vc, vsc = quantize_kv(v)
            _bytes(self.k[i, 0]).index_copy_(1, sl, _bytes(kc))
            _bytes(self.v[i, 0]).index_copy_(1, sl, _bytes(vc))
            self.ks[i, 0].index_copy_(1, sl, ksc)
            self.vs[i, 0].index_copy_(1, sl, vsc)
        else:
            self.k[i, 0].index_copy_(1, sl, k.to(self.k.dtype))
            self.v[i, 0].index_copy_(1, sl, v.to(self.v.dtype))

    def rows(self, layer: int, lo: int, hi: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows lo..hi-1 (all within the last R) in index order, [Hkv, hi - lo, D], dequantised."""
        i = self.slot[layer]
        sl = self.slots(lo, hi - lo, self.k.device)
        k = _bytes(self.k[i, 0]).index_select(1, sl)
        v = _bytes(self.v[i, 0]).index_select(1, sl)
        if self.fp8:
            k = k.view(torch.float8_e4m3fn).float() * self.ks[i, 0].index_select(1, sl)[..., None]
            v = v.view(torch.float8_e4m3fn).float() * self.vs[i, 0].index_select(1, sl)[..., None]
        return k.to(dtype), v.to(dtype)


class KolibriKV:
    """The whole KV of one sequence: full-layer rows plus the sliding ring.

    `k`, `v`, `ks`, `vs`, `slot`, `length`, `max_len`, `fp8` mean what they mean on Qwen's
    `KVCache`; `ring` (also `state`) is the part a cut cannot recover.
    """

    def __init__(self, cfg, max_len: int, device, dtype=torch.bfloat16, fp8: bool = False,
                 ring_rows: int = RING_ROWS):
        spec = AttnSpec.of(cfg)
        n = len(spec.full_layers)
        self.spec = spec
        self.slot = {l: i for i, l in enumerate(spec.full_layers)}
        self.fp8 = bool(fp8)
        self.dtype = dtype
        cdt = torch.float8_e4m3fn if fp8 else dtype
        self.k = torch.zeros(n, 1, spec.nkv, max_len, spec.hd, dtype=cdt, device=device)
        self.v = torch.zeros_like(self.k)
        self.ks = (torch.zeros(n, 1, spec.nkv, max_len, dtype=torch.float32, device=device)
                   if fp8 else None)
        self.vs = torch.zeros_like(self.ks) if fp8 else None
        self.ring = SlidingRing(spec, ring_rows, device, dtype, fp8)
        self.length = 0
        self.max_len = int(max_len)

    @property
    def state(self) -> SlidingRing:
        return self.ring

    @property
    def back(self) -> int:
        """How many rows `truncate` can take back exactly (128 at R = 640)."""
        return self.ring.R - (self.spec.window - 1)

    @property
    def bytes_per_token(self) -> int:
        """Bytes one more token of context costs (the full layers only)."""
        per = 2 * len(self.slot) * self.spec.nkv * self.spec.hd * self.k.element_size()
        if self.fp8:
            per += 2 * len(self.slot) * self.spec.nkv * 4
        return per

    def nbytes(self) -> int:
        t = [self.k, self.v] + ([self.ks, self.vs] if self.fp8 else [])
        return sum(x.numel() * x.element_size() for x in t) + self.ring.nbytes

    # --- the sequence -----------------------------------------------------------------------
    def reset(self) -> None:
        self.length = 0
        self.ring.reset()

    def truncate(self, n: int) -> None:
        """Keep the first n rows. Exact for n >= length - `back`; below that the ring no longer
        holds the window before n and this raises `RingLost` (the caller restores an anchor)."""
        if not 0 <= n <= self.length:
            raise ValueError(f"truncate to {n} with {self.length} rows")
        if n and n < self.length - self.back:
            raise RingLost(f"truncate to {n} from {self.length}: the ring keeps {self.back} rows "
                           f"back")
        self.length = int(n)
        self.ring.primed = n > 0

    rollback = truncate

    def snapshot(self) -> tuple:
        """The ring as it is now (an anchor at `length`): 52 MB in BF16."""
        return (self.length,) + tuple(t.clone() for t in self.ring.tensors())

    def snapshot_into(self, bufs: tuple) -> tuple:
        """`snapshot()` into the buffers of an older one (no allocation)."""
        for dst, src in zip(bufs[1:], self.ring.tensors()):
            dst.copy_(src)
        return (self.length,) + tuple(bufs[1:])

    def restore(self, snap: tuple, n: int | None = None) -> None:
        """Back to the anchor `snap` (taken at length L): the ring as it was, length L. The
        full-layer rows below L are the ones written then, as long as nothing has been written
        below L since (a cut below L invalidates the anchor; the caller's bookkeeping)."""
        for dst, src in zip(self.ring.tensors(), snap[1:]):
            dst.copy_(src)
        self.length = int(snap[0])
        self.ring.primed = self.length > 0
        if n is not None:
            self.truncate(n)

    def commit_path(self, start: int, path: list) -> None:
        """A tree verify wrote node j at index start + j; keep the accepted nodes `path`
        (ascending, path[0] = 0 the anchor) at start .. start + len(path) - 1."""
        n = len(path)
        if list(path) != list(range(n)):
            dev = self.k.device
            src = torch.tensor([start + p for p in path], device=dev)
            dst = torch.arange(start, start + n, device=dev)
            for t in ([self.k, self.v] + ([self.ks, self.vs] if self.fp8 else [])):
                t = _bytes(t)
                t.index_copy_(3, dst, t.index_select(3, src))
            r = self.ring
            rs, rd = src % r.R, dst % r.R
            for t in r.tensors():
                if t is r.idx:
                    continue
                t = _bytes(t)
                t.index_copy_(3, rd, t.index_select(3, rs))
            r.idx[rd] = dst.to(torch.int32)
        self.length = start + n
        self.ring.primed = True

    # --- full-layer rows --------------------------------------------------------------------
    def write_full(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """k, v [Hkv, T, D] at rows start..start+T-1."""
        i = self.slot[layer]
        T = k.shape[1]
        if self.fp8:
            kc, ksc = quantize_kv(k)
            vc, vsc = quantize_kv(v)
            self.k[i, 0, :, start:start + T] = kc
            self.v[i, 0, :, start:start + T] = vc
            self.ks[i, 0, :, start:start + T] = ksc
            self.vs[i, 0, :, start:start + T] = vsc
        else:
            self.k[i, 0, :, start:start + T] = k.to(self.k.dtype)
            self.v[i, 0, :, start:start + T] = v.to(self.v.dtype)

    def full_rows(self, layer: int, n: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The first n rows of a full layer, [Hkv, n, D]: a view of the cache in BF16,
        dequantised when e4m3."""
        i = self.slot[layer]
        k, v = self.k[i, 0, :, :n], self.v[i, 0, :, :n]
        if self.fp8:
            return ((k.float() * self.ks[i, 0, :, :n, None]).to(dtype),
                    (v.float() * self.vs[i, 0, :, :n, None]).to(dtype))
        return (k, v) if k.dtype == dtype else (k.to(dtype), v.to(dtype))


def rms_heads(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Kolibri's RMSNorm over the last axis: `x * rsqrt(mean(x^2) + eps) * w`, in fp32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """NeoX rotary over the whole head, in fp32, the arithmetic of `tools/kolibri_ref.rope`.
    x [T, H, D], pos [T] (a device tensor is fine: nothing here reads it on the host)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d))
    f = pos.float()[:, None] * inv[None, :]
    cos, sin = torch.cat([f.cos()] * 2, -1), torch.cat([f.sin()] * 2, -1)
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
    rot = torch.cat([-x2, x1], -1)
    return (xf * cos[:, None, :] + rot * sin[:, None, :]).to(x.dtype)


def _mm(x: torch.Tensor, w) -> torch.Tensor:
    """`x @ w^T` for a plain tensor or anything with `.matmul` (engine/linear.py, engine/kolibri/kernels.py)."""
    if isinstance(w, torch.Tensor):
        return F.linear(x, w if w.dtype == x.dtype else w.to(x.dtype))
    return w.matmul(x)


def _sdpa(q, k, v, mask, scale):
    """q [Hq, Tq, D], k/v [Hkv, Tk, D], mask [Tq, Tk] bool -> [Hq, Tq, D]."""
    rep = q.shape[0] // k.shape[0]
    if rep > 1:
        k = k.repeat_interleave(rep, 0)
        v = v.repeat_interleave(rep, 0)
    return F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask[None, None],
                                          scale=scale)[0]


def _have_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:                                              # pragma: no cover
        return False
    return True


def _next_pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


class KolibriAttention:
    """One object for all 50 layers; the weights come in per call as `engine.kolibri.weights.KLayer`
    (`index`, `sliding`, `qkv`, `o`, `q_norm`, `k_norm`)."""

    def __init__(self, cfg, fp8_kv: bool | None = None, ring_rows: int = RING_ROWS):
        self.spec = AttnSpec.of(cfg)
        self.scale = 1.0 / math.sqrt(self.spec.hd)
        self.fp8_kv = (os.environ.get("KOLIBRI_KV_FP8") == "1") if fp8_kv is None else bool(fp8_kv)
        self.ring_rows = int(ring_rows)
        self._tri: dict = {}
        self._one: dict = {}

    @property
    def graphable(self) -> bool:
        """The decode step can be captured: the device-length kernels exist for a BF16 cache."""
        return _have_triton() and os.environ.get("KOLIBRI_ATTN_KERNEL", "1") != "0"

    def make_kv(self, max_len: int, device, dtype=torch.bfloat16) -> KolibriKV:
        return KolibriKV(self.spec, max_len, device, dtype=dtype, fp8=self.fp8_kv,
                         ring_rows=self.ring_rows)

    # --- projections ------------------------------------------------------------------------
    def _qkv(self, x: torch.Tensor, lw, positions: torch.Tensor):
        """x [T, H] -> q [T, Hq, D], k, v [T, Hkv, D]: normed, rotated on sliding layers."""
        s = self.spec
        T = x.shape[0]
        y = _mm(x, lw.qkv)
        if y.is_cuda and y.dtype == torch.bfloat16 and FUSED_QK and _have_triton():
            # norm + RoPE of q and k in one launch (tools/kolibri_attn_kernels.qk_prep); the
            # torch form below is about fifteen elementwise passes
            from tools.kolibri_attn_kernels import qk_prep
            q, k = qk_prep(y, lw.q_norm, lw.k_norm, positions, s.nq, s.nkv, s.hd, s.rope_theta,
                           s.eps, bool(lw.sliding))
            return q, k, y[:, s.q_size + s.kv_size:].reshape(T, s.nkv, s.hd)
        q = y[:, :s.q_size].reshape(T, s.nq, s.hd)
        k = y[:, s.q_size:s.q_size + s.kv_size].reshape(T, s.nkv, s.hd)
        v = y[:, s.q_size + s.kv_size:].reshape(T, s.nkv, s.hd)
        q = rms_heads(q, lw.q_norm, s.eps)
        k = rms_heads(k, lw.k_norm, s.eps)
        if lw.sliding:
            q, k = rope(q, positions, s.rope_theta), rope(k, positions, s.rope_theta)
        return q, k, v

    def _out(self, o: torch.Tensor, lw) -> torch.Tensor:
        """o [T, Hq, D] -> o_proj, [T, H]."""
        return _mm(o.reshape(o.shape[0], -1), lw.o)

    # --- the three calls --------------------------------------------------------------------
    def attn_prefill(self, x: torch.Tensor, lw, kv: KolibriKV, pos0: int) -> torch.Tensor:
        """T rows at host position `pos0`; writes their K/V; does not change `kv.length`."""
        return self.attn_block(x, lw, kv, int(pos0))

    def attn_block(self, x, lw, kv: KolibriKV, start: int, positions=None, block_mask=None):
        T = x.shape[0]
        if positions is None:
            positions = torch.arange(start, start + T, device=x.device)
        q, k, v = self._qkv(x, lw, positions)
        o = self.attend(lw.index, q, k, v, kv, start, positions, block_mask)
        return self._out(o, lw)

    def attn_decode(self, x: torch.Tensor, lw, kv: KolibriKV, pos) -> torch.Tensor:
        """One row; `pos` an int32 device tensor [1] (an int works too, off the graph)."""
        if not torch.is_tensor(pos) or not x.is_cuda or not self.graphable:
            p = int(pos.item()) if torch.is_tensor(pos) else int(pos)
            return self.attn_block(x, lw, kv, p)
        s = self.spec
        if DEC_FUSED and FUSED_QK and not kv.fp8 and x.dtype == torch.bfloat16:
            from tools.kolibri_attn_kernels import decode_fused
            y = _mm(x, lw.qkv)
            p32 = pos.reshape(1)
            if p32.dtype != torch.int32:
                p32 = p32.to(torch.int32)
            if lw.sliding:
                r = kv.ring
                i = r.slot[lw.index]
                kc, vc, idx = r.k[i], r.v[i], r.idx
            else:
                i = kv.slot[lw.index]
                kc, vc, idx = kv.k[i], kv.v[i], None
            o = decode_fused(y, lw.q_norm, lw.k_norm, p32, kc, vc, idx, nq=s.nq, nk=s.nkv,
                             d=s.hd, theta=s.rope_theta, eps=s.eps, rope=bool(lw.sliding),
                             window=s.window, scale=self.scale)
            return _mm(o, lw.o)
        posl = pos.reshape(1).long()
        q, k, v = self._qkv(x, lw, posl)
        qt = q.transpose(0, 1)[None]                                 # [1, Hq, 1, D]
        L = lw.index
        one = self._one.get(x.device)
        if one is None:
            # int8, the form the kernels read: no conversion launch per layer
            one = self._one[x.device] = torch.ones(1, 1, dtype=torch.int8, device=x.device)
        kt, vt = k.transpose(0, 1), v.transpose(0, 1)               # [Hkv, 1, D]
        if lw.sliding:
            from tools.kolibri_attn_kernels import ring_attention
            r = kv.ring
            i = r.slot[L]
            sl = posl % r.R
            r.idx.index_copy_(0, sl, pos.reshape(1).to(torch.int32))
            r._put(i, sl, kt, vt)                    # BF16, or e4m3 codes and their scales
            qlo = (pos.reshape(1) - (s.window - 1)).to(torch.int32)
            o = ring_attention(qt, r.k[i], r.v[i], r.idx, 0, qlo, one, scale=self.scale,
                               startp=pos.reshape(1).to(torch.int32),
                               ks=r.ks[i] if r.fp8 else None, vs=r.vs[i] if r.fp8 else None)
        else:
            from tools.attn_kernels import decode_attention_dev
            i = kv.slot[L]
            if kv.fp8:
                kc, ksc = quantize_kv(kt)
                vc, vsc = quantize_kv(vt)
                _bytes(kv.k[i, 0]).index_copy_(1, posl, _bytes(kc))
                _bytes(kv.v[i, 0]).index_copy_(1, posl, _bytes(vc))
                kv.ks[i, 0].index_copy_(1, posl, ksc)
                kv.vs[i, 0].index_copy_(1, posl, vsc)
            else:
                kv.k[i, 0].index_copy_(1, posl, kt.to(kv.k.dtype))
                kv.v[i, 0].index_copy_(1, posl, vt.to(kv.v.dtype))
            p32 = pos.reshape(1).to(torch.int32)
            lenp = torch.cat([p32, p32 + 1])
            o = decode_attention_dev(qt, kv.k[i], kv.v[i], lenp, one,
                                     max(1024, _next_pow2(kv.max_len)), scale=self.scale,
                                     ks=kv.ks[i] if kv.fp8 else None,
                                     vs=kv.vs[i] if kv.fp8 else None)
        return self._out(o[0].transpose(0, 1), lw)

    # --- the attention itself, after the projections ---------------------------------------
    def attend(self, layer, q, k, v, kv: KolibriKV, start, positions, block_mask):
        """q [T, Hq, D], k/v [T, Hkv, D] -> [T, Hq, D]. Writes k, v into the cache first."""
        s = self.spec
        T = q.shape[0]
        kt, vt = k.transpose(0, 1), v.transpose(0, 1)               # [Hkv, T, D]
        qt = q.transpose(0, 1)                                       # [Hq, T, D]
        kernels = q.is_cuda and T < PREFILL_ROWS and _have_triton() \
            and os.environ.get("KOLIBRI_ATTN_KERNEL", "1") != "0"
        if s.sliding(layer):
            kv.ring.mark(start, T)
            lo = max(0, start - (s.window - 1))
            if not kernels:
                # the previous rows BEFORE this chunk's write can overwrite them
                pk, pv = kv.ring.rows(layer, lo, start, q.dtype) if start > lo else (None, None)
                kv.ring.write(layer, kt, vt, start)
                o = self._sliding_sdpa(qt, kt, vt, pk, pv, lo, start, positions, block_mask)
            else:
                kv.ring.write(layer, kt, vt, start)
                o = self._sliding_kernel(layer, qt, kv, start, positions, block_mask)
        else:
            kv.write_full(layer, kt, vt, start)
            if not kernels:
                o = self._full_sdpa(layer, qt, kv, start, block_mask)
            else:
                o = self._full_kernel(layer, qt, kv, start, block_mask)
        return o.transpose(0, 1)

    def _block_vis(self, T, block_mask, device):
        if block_mask is not None:
            return block_mask.to(device=device, dtype=torch.bool)
        m = self._tri.get((T, device))
        if m is None:
            m = self._tri[(T, device)] = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        return m

    def _full_sdpa(self, layer, qt, kv, start, block_mask):
        T = qt.shape[1]
        kk, vv = kv.full_rows(layer, start + T, qt.dtype)            # the rows in place
        if block_mask is None and qt.is_cuda:
            # the causal mask aligned lower right, which the fused backends take as it is
            if start == 0:
                return F.scaled_dot_product_attention(qt[None], kk[None], vv[None], is_causal=True,
                                                      scale=self.scale, enable_gqa=True)[0]
            from torch.nn.attention.bias import causal_lower_right
            mask = causal_lower_right(T, kk.shape[1])
            return F.scaled_dot_product_attention(qt[None], kk[None], vv[None], attn_mask=mask,
                                                  scale=self.scale, enable_gqa=True)[0]
        vis = torch.ones(T, start + T, dtype=torch.bool, device=qt.device)
        vis[:, start:] = self._block_vis(T, block_mask, qt.device)
        return _sdpa(qt, kk, vv, vis, self.scale)

    def _sliding_sdpa(self, qt, kt, vt, pk, pv, lo, start, positions, block_mask):
        """Window attention over the previous rows (indices lo..start-1) and the block's own."""
        W = self.spec.window
        T = qt.shape[1]
        dev = qt.device
        kt, vt = kt.to(qt.dtype), vt.to(qt.dtype)
        if pk is not None:
            kk, vv = torch.cat([pk, kt], 1), torch.cat([pv, vt], 1)
            kpos = torch.cat([torch.arange(lo, start, device=dev), positions.to(dev)])
        else:
            kk, vv = kt, vt
            kpos = positions.to(dev)
        P = start - lo                                               # previous rows
        bvis = self._block_vis(T, block_mask, dev) if block_mask is not None else None
        out = torch.empty_like(qt)
        for q0 in range(0, T, QBLOCK):
            q1 = min(T, q0 + QBLOCK)
            qpos = positions[q0:q1].to(dev)
            # keys a row of this block may reach; for a chain key index = position - lo
            k0 = max(0, int(start + q0 - (W - 1) - lo)) if block_mask is None else 0
            k1 = P + q1
            kp = kpos[k0:k1]
            vis = (kp[None, :] >= qpos[:, None] - (W - 1)) & (kp[None, :] <= qpos[:, None])
            # the block's own columns: causal for a chain, the ancestors for a tree
            jb0 = max(0, k0 - P)
            nb = k1 - P - jb0
            if nb > 0:
                cols = slice(max(0, P - k0), k1 - k0)
                if bvis is not None:
                    vis[:, cols] &= bvis[q0:q1, jb0:jb0 + nb]
                else:
                    j = torch.arange(jb0, jb0 + nb, device=dev)
                    t = torch.arange(q0, q1, device=dev)
                    vis[:, cols] &= j[None, :] <= t[:, None]
            out[:, q0:q1] = _sdpa(qt[:, q0:q1], kk[:, k0:k1], vv[:, k0:k1], vis, self.scale)
        return out

    def _full_kernel(self, layer, qt, kv, start, block_mask):
        from tools.attn_kernels import decode_attention
        i = kv.slot[layer]
        T = qt.shape[1]
        n = start + T
        ks = kv.ks[i, :, :, :n] if kv.fp8 else None
        vs = kv.vs[i, :, :, :n] if kv.fp8 else None
        o = decode_attention(qt[None], kv.k[i, :, :, :n], kv.v[i, :, :, :n], start,
                             self._block_vis(T, block_mask, qt.device), ks=ks, vs=vs,
                             scale=self.scale)
        return o[0]                                                  # [Hq, T, D]

    def _sliding_kernel(self, layer, qt, kv, start, positions, block_mask):
        from tools.kolibri_attn_kernels import ring_attention
        r = kv.ring
        i = r.slot[layer]
        # the lowest committed index a row sees: its position - (window - 1); a tree row's
        # position is start + its depth
        qlo = (positions.to(qt.device) - (self.spec.window - 1)).to(torch.int32)
        o = ring_attention(qt[None], r.k[i], r.v[i], r.idx, start, qlo,
                           self._block_vis(qt.shape[1], block_mask, qt.device),
                           ks=r.ks[i] if r.fp8 else None, vs=r.vs[i] if r.fp8 else None,
                           scale=self.scale)
        return o[0]
