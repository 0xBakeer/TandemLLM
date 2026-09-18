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
from tools.head_gemv import FP8Head  # noqa: E402
from tools.nvfp4_linear import NVFP4Block, nvfp4_matmul  # noqa: E402
from tools.nvfp4_linear_v2 import nvfp4_matmul_group  # noqa: E402
from tools import nvfp4_verify_tiles as _verify_tiles  # noqa: E402,F401  (registers on import)


# Fused kernels replacing parts of the reference forward. Off unless asked for: each one is a
# different arithmetic ORDER for the same quantity, so each has to earn its place against the gates
# (argmax agreement, greedy losslessness) and not only against a stopwatch. `tools/*_kernels.py`
# and `tools/head_gemv.py` each carry a `check()` against the function they replace.
# Four of the five are ON as of 11:55: each one is faster in the engine's own step or verify curve,
# and the whole set passes the losslessness gate of 10:07. `attn` is off because indexing the KV
# groups saves 0.6 ms at one token and costs 2.4 ms at eight. Set any of these to 0 to compare.
#: Skip the recurrent-state clone on a TREE verify, where nothing advances it; see `forward_tree`.
#:
#: OFF in the release candidate, and not because it failed anything. It was predicted at 1.7 ms
#: of a 134 ms block and it is not: measured at
#: 134.613 -> 134.344 ms, about 0.27 ms, which is inside the spread of repeated runs of the same
#: configuration. What it does have is the gates -- the losslessness gate PASSES with it on, and
#: `test_forward_tree`, `test_tree` and `test_cache` are identical either way -- and it removes
#: 151 MB of allocation and 302 MB of traffic per block that the tree path had no use for. It
#: would ship on being unnecessary work rather than on a stopwatch reading. It went into the RC
#: together with `tree_mode = nodes`, the row read differently, and both were reverted -- and then
#: re-running the reverted configuration unchanged read differently again, by more than either
#: change was worth. The row's run-to-run spread is 10 % of its mean when the policy is adaptive,
#: so it cannot resolve a change this size and did not. This is off in the RC because the RC was
#: soaked without it, not because anything caught it, and it goes to the `speed` branch where it
#: can be measured against several rows rather than one.
TREE_ALIAS_STATE = os.environ.get("QWEN38_TREE_ALIAS_STATE", "0") == "1"

FUSED = {
    "norm": os.environ.get("QWEN38_FUSED_NORM", "1") == "1",
    "gdn": os.environ.get("QWEN38_FUSED_GDN", "1") == "1",
    "head": os.environ.get("QWEN38_FUSED_HEAD", "1") == "1",
    "attn": os.environ.get("QWEN38_FUSED_ATTN", "0") == "1",
    "gdnblock": os.environ.get("QWEN38_FUSED_GDNBLOCK", "1") == "1",
    "gdnpre": os.environ.get("QWEN38_FUSED_GDNPRE", "1") == "1",
    # The tree's counterpart to gdnblock. Same bet, same shape of kernel, one file over:
    # tools/gdn_tree_kernels.py walks the DFS pre-order carrying one factor per DEPTH, which is the
    # node's own ancestry, so the entry tile is read once as it is for a chain.
    "gdntree": os.environ.get("QWEN38_FUSED_GDNTREE", "1") == "1",
}

# "1" rank-k rollback, "0" the replay it replaces, "check" both with the difference recorded.
RANKK = os.environ.get("QWEN38_RANKK", "1")

# A chain-shaped tree is a chain, and the chain is 12.5 ms cheaper because it has a kernel the tree
# cannot use. So `forward_tree` hands one to `forward_block` and a drafter takes the cheaper price
# by proposing a line. Off only in the tests that have to exercise the tree path on a chain shape,
# where the delegation would make the comparison a tautology.
TREE_CHAIN_DELEGATE = os.environ.get("QWEN38_TREE_CHAIN_DELEGATE", "1") == "1"

# The chunked delta rule's blocking on a prefill. The reference uses 64. It is a blocking choice,
# not a semantic one: the chunk loop is serial in Tp/chunk, and the intra-chunk work grows with the
# square of the chunk, so the best value is a measurement. See tools/profile_prefill.py --gdn-chunk.
GDN_PREFILL_CHUNK = int(os.environ.get("QWEN38_GDN_CHUNK", "64"))

# Index the KV groups instead of materialising them from this many rows up.
GQA_FROM = int(os.environ.get("QWEN38_GQA_FROM", "64"))

# Tell SDPA a prefill is causal instead of handing it a [T, T] boolean. Set to 0 for the
# materialised mask the engine used until phase 4, which is the control this is measured against.
PREFILL_CAUSAL = os.environ.get("QWEN38_PREFILL_CAUSAL", "1") == "1"

# The same argument one step further, for a prefill CHUNK. `engine/cache.py` splits a prefill so it
# can checkpoint, and every chunk after the first starts at `start > 0`, where the mask wanted is
# the causal triangle offset by `start` -- the BOTTOM-RIGHT alignment, which `is_causal` is not.
# Handing SDPA a materialised [T, ctx] boolean instead takes it off its fused backend and it was
# measured at 1.7-1.9x on a cold prefill (SPEED-LEDGER, track D, 18:27). `causal_lower_right` says
# the same thing in a form the kernel keeps its backend for.
#
# From this many rows up, and no lower: a speculative verify block is 8 or 16 rows and it keeps the
# materialised mask it has been measured with all day, so no other track's number moves. 0 turns
# this off and restores the boolean everywhere, which is the control.
CHUNK_LOWER_RIGHT_FROM = int(os.environ.get("QWEN38_LOWER_RIGHT_FROM", "64"))
ROLLBACK_DIFF: list = []


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if FUSED["norm"]:
        from tools.norm_kernels import rms_norm as fused
        return fused(x, weight, eps)
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    if FUSED["norm"]:
        from tools.norm_kernels import rms_norm_gated as fused
        return fused(x, gate, weight, eps)
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dt)
    h = h * F.silu(gate.float())
    return h.to(dt)


def head_logits(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The vocabulary projection. The largest single read in the step -- see tools/head_gemv.py.

    ONE ROW ONLY. The kernel is a GEMV: it broadcasts the activation against the weight tile and
    accumulates in fp32 registers, which is the right shape at M = 1 and the wrong one at M = 8,
    where it does eight times the scalar arithmetic and never touches a tensor core. Measured at
    11:26: with this routed at M = 8 the verify pass went from 184 ms a block to 451. The library's
    GEMM owns everything above one row.
    """
    m = h.shape[-2] if h.dim() > 1 else 1
    if isinstance(weight, FP8Head):
        # e4m3 codes with a scale per vocabulary row: half the bytes, fp32 logits, and its own
        # kernel at both row counts -- a GEMV at one row and a GEMM above it, for the reason in the
        # docstring above.
        from tools.head_gemv import head_matmul_fp8
        return head_matmul_fp8(h, weight).view(*h.shape[:-1], weight.N)
    if FUSED["head"] and m == 1 and not isinstance(weight, (FP8Block, NVFP4Block)):
        from tools.head_gemv import head_matmul
        return head_matmul(h, weight, bm=1).view(*h.shape[:-1], weight.shape[0])
    return linear(h, weight)


def linear(x: torch.Tensor, w: FP8Block | NVFP4Block | torch.Tensor) -> torch.Tensor:
    """`x @ w^T` for a stored fp8 block weight, an NVFP4 one, or a plain bf16 one."""
    if isinstance(w, FP8Block):
        flat = x.reshape(-1, x.shape[-1])
        return fp8_matmul(flat, w).view(*x.shape[:-1], w.N)
    if isinstance(w, NVFP4Block):
        flat = x.reshape(-1, x.shape[-1])
        return nvfp4_matmul(flat, w).view(*x.shape[:-1], w.N)
    if isinstance(w, FP8Head):
        return head_logits(x, w)
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
        # ENG-16: a verify block that starts inside the room clamp and crosses `max_len` used to
        # die here as a torch shape mismatch (2026-09-18, twice: 16 rows into 12, 8 into 6),
        # because the loops clamp on OUTPUT tokens while this write counts ROWS -- a chain block
        # is `1 + len(draft)` rows (the anchor's own), a tree is all its nodes, and a tree's
        # rejected rows are written past `length` and merely never read. The loops now clamp on
        # rows; this guard is the second line of defence, so a path that misses the clamp fails
        # with a sentence rather than a traceback the log cannot attribute.
        if start + t > self.max_len:
            raise RuntimeError(
                f"verify block overruns the KV window: rows [{start}, {start + t}) "
                f"into a {self.max_len}-row buffer (ENG-16)")
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


class BlockTrace:
    """What a speculative block has to remember so a partial accept can be undone.

    The recurrent state after k of B verified tokens was never materialised -- the chunked form
    produces outputs for every position and one final state. Snapshotting every prefix would move
    151 MB per position; recomputing the layer would re-read every weight. What is kept instead is
    the entry state and the per-token projections the recurrence consumes, which are three orders of
    magnitude smaller, and the replay then touches no weights at all.

    When every drafted token is accepted -- the case a good drafter produces most of the time --
    the final state the verify pass already wrote is correct and none of this is read.
    """

    def __init__(self):
        self.S_entry: torch.Tensor | None = None
        self.conv_entry: torch.Tensor | None = None
        self.layers: dict[int, tuple] = {}
        # The rank-k form: per layer, the normalised keys, the chunk's pseudo-values and its
        # cumulative gate. With these a partial accept is a weighted sum, not a replay -- see
        # `gdn.chunk_gated_delta_rule(return_factors=True)` for why they do not depend on how much
        # of the block is kept. `layers` stays as the fallback for a block that spans two chunks.
        self.factors: dict[int, tuple] = {}

    @property
    def nbytes(self) -> int:
        n = 0
        for t in self.layers.values():
            n += sum(x.numel() * x.element_size() for x in t)
        for x in (self.S_entry, self.conv_entry):
            if x is not None:
                n += x.numel() * x.element_size()
        return n


class TreeCtx:
    """Everything a tree verify needs that depends only on the tree's SHAPE, built once and cached.

    A draft tree changes three things in the forward pass and nothing else: which nodes a node may
    attend to (`anc_incl`), which columns its convolution reads (`conv_idx`), and what position it
    occupies for RoPE (`depths`, since a node's position is its depth, not its index). All three are
    functions of the parent array alone, so two steps that happen to draft the same shape -- which
    with a fixed node budget is most steps -- reuse them.
    """

    _cache: dict = {}

    def __init__(self, parents: tuple[int, ...], device: str, width: int):
        from engine.tree import DraftTree
        t = DraftTree(tokens=[0] * len(parents), parents=list(parents))
        n = len(parents)
        self.parents = parents
        self.n = n
        m = t.ancestor_mask()
        self.anc_incl = torch.tensor(m, dtype=torch.bool, device=device)
        self.anc_strict = self.anc_incl & ~torch.eye(n, dtype=torch.bool, device=device)
        self.conv_idx = torch.tensor(t.conv_windows(width), dtype=torch.long, device=device)
        self.depth_list = t.depths()
        self.depths = torch.tensor(self.depth_list, dtype=torch.long, device=device)
        self.is_chain = list(parents) == [-1] + list(range(n - 1))

    @classmethod
    def get(cls, parents, device: str, width: int) -> "TreeCtx":
        key = (tuple(parents), device, width)
        ctx = cls._cache.get(key)
        if ctx is None:
            ctx = cls(tuple(parents), device, width)
            if len(cls._cache) > 256:
                cls._cache.clear()
            cls._cache[key] = ctx
        return ctx


class Qwen38Engine:
    def __init__(self, cfg: TextConfig, w: Weights, max_len: int = 8192, device: str = "cuda"):
        self.cfg = cfg
        self.w = w
        self.device = device
        self.max_len = max_len
        self.kv = KVCache(cfg, max_len, device)
        self.state = GDNState(cfg, device)
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._trace: "BlockTrace | None" = None
        self.hidden_pre_norm: torch.Tensor | None = None
        self.hidden_post_norm: torch.Tensor | None = None
        self.tap = None  # set to a callable to receive every layer's hidden state
        self.trace: BlockTrace | None = None  # set during a speculative block verify
        self.tree: TreeCtx | None = None      # set during a speculative TREE verify
        self._tree: TreeCtx | None = None     # the one the last forward_tree ran
        self._tree_start = 0
        self._tree_is_chain = False

    # ---------------------------------------------------------------- rotary
    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Partial rotary over the first `rotary_dim` of each head.

        The published model carries an interleaved 3-axis mRoPE for image and video grids. For text
        the three position rows are identical, so the interleave rewrites each row with a copy of
        itself and the result is ordinary RoPE. This engine is text only, so that is what is built.
        """
        if self._rope_cache is None:
            # Built once for every position the engine can reach, then gathered. The table was
            # being rebuilt from scratch on every attention layer of every step -- eight small
            # kernels, sixteen times a token, over numbers that never change. The values are
            # identical: the same product, computed for all rows at once instead of one row.
            dim = self.cfg.rotary_dim
            inv = 1.0 / (self.cfg.rope_theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
            t = torch.arange(self.max_len + 64, dtype=torch.float32, device=self.device)
            freqs = t[:, None] * inv[None, :]
            emb = torch.cat([freqs, freqs], dim=-1)
            self._rope_cache = (emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16))
        cos, sin = self._rope_cache
        return cos[positions], sin[positions]

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
        g = self.w.group(f"{p}.mlp.gate_up")
        if g is not None:
            # One launch for both halves. They read the same activation and neither reads the
            # other's output, so the only thing that made them two kernels was that they are two
            # names -- and two kernels is 272 programs apiece on 48 SMs, twice, with the board
            # draining between them.
            y = nvfp4_matmul_group(h.reshape(-1, h.shape[-1]), g)
            gate, up = y.split(g.sizes, dim=-1)
            act = F.silu(gate) * up
            return linear(act, self.w.proj(f"{p}.mlp.down_proj")).view(*h.shape[:-1], -1)
        gate = linear(h, self.w.proj(f"{p}.mlp.gate_proj"))
        up = linear(h, self.w.proj(f"{p}.mlp.up_proj"))
        return linear(F.silu(gate) * up, self.w.proj(f"{p}.mlp.down_proj"))

    def _attn_mask(self, T: int, ctx: int, start: int, device) -> torch.Tensor:
        """What each of the T rows may attend to across the whole `ctx`-long cache.

        Straight-line decoding wants the causal triangle offset by `start`. A tree wants the
        committed prefix for everybody and, inside the block, each node's own ancestors -- the same
        relation the linear layers get, expressed as an attention mask. The KV cache is written in
        DFS order, so column `start + j` is node j and the block half of the mask is exactly
        `ancestor_incl`.
        """
        if self.tree is None:
            return torch.ones(T, ctx, dtype=torch.bool, device=device).tril(start)
        m = torch.zeros(T, ctx, dtype=torch.bool, device=device)
        if start:
            m[:, :start] = True
        m[:, start:start + self.tree.n] = self.tree.anc_incl
        return m

    def attention(self, h: torch.Tensor, p: str, layer: int, start: int,
                  positions: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h.shape
        grp = self.w.group(f"{p}.self_attn.qkv")
        if grp is not None:
            # q is twelve times the size of k or v, and k and v were each putting sixteen programs
            # on the board -- a launch that cannot fill it at any tiling. Together they are one.
            y = nvfp4_matmul_group(h.reshape(-1, h.shape[-1]), grp)
            qy, ky, vy = y.split(grp.sizes, dim=-1)
            # a column slice is not contiguous, and `reshape` is where the copy is paid: 459 kB a
            # layer at the block's row count, which is 0.04 ms a step across all sixteen
            qg = qy.reshape(B, T, cfg.num_attention_heads, cfg.head_dim * 2)
            kk_ = ky.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim)
            vv_ = vy.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim)
        else:
            qg = linear(h, self.w.proj(f"{p}.self_attn.q_proj")).view(
                B, T, cfg.num_attention_heads, cfg.head_dim * 2)
            kk_ = linear(h, self.w.proj(f"{p}.self_attn.k_proj")).view(
                B, T, cfg.num_key_value_heads, cfg.head_dim)
            vv_ = linear(h, self.w.proj(f"{p}.self_attn.v_proj")).view(
                B, T, cfg.num_key_value_heads, cfg.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = rms_norm(q, self.w.norm(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        k = rms_norm(kk_, self.w.norm(f"{p}.self_attn.k_norm.weight"),
                     cfg.rms_norm_eps).transpose(1, 2)
        v = vv_.transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = self.apply_rope(q, k, cos, sin)
        kk, vv = self.kv.append(layer, k, v, start)
        rep = cfg.num_attention_heads // cfg.num_key_value_heads
        # A prefill is the one case where the mask is the plain causal triangle over the whole
        # context, and saying so instead of handing the kernel a [T, T] boolean is not a
        # micro-optimisation: a materialised mask takes SDPA off its fused backend, and the
        # fallback writes a [24, T, T] score matrix -- 3.2 GB at T = 8,192, per layer, sixteen
        # times. `is_causal` aligns to the top-left, which is what `start == 0` means.
        causal = PREFILL_CAUSAL and T > 1 and self.tree is None and start == 0
        if causal or T == 1:
            mask = None
        elif (self.tree is None and CHUNK_LOWER_RIGHT_FROM
                and T >= CHUNK_LOWER_RIGHT_FROM):
            from torch.nn.attention.bias import causal_lower_right
            mask = causal_lower_right(T, kk.shape[2])
        else:
            mask = self._attn_mask(T, kk.shape[2], start, h.device)
        if FUSED["attn"] or T >= GQA_FROM:
            # `repeat_interleave` materialises the whole context six times over, once per query
            # group: at 4k of context that is 200 MB written and read again per token across the
            # sixteen attention layers, for data the kernel can index instead. `enable_gqa` lets
            # it index. It loses 2.4 ms on a verify block of eight (11:26) and wins on a long
            # sequence, where the copy is 100 MB a layer, so the row count decides.
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=causal,
                                               enable_gqa=True)
        else:
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=causal)
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return linear(o, self.w.proj(f"{p}.self_attn.o_proj"))

    def linear_attention(self, h: torch.Tensor, p: str, layer: int,
                         use_state: bool) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h.shape
        i = self.state.slot[layer]
        if (T == 1 and use_state and self.tree is None and self.trace is None
                and FUSED["gdn"] and FUSED["gdnpre"]):
            return self._linear_attention_decode(h, p, i)
        grp = self.w.group(f"{p}.linear_attn.qkvz")
        z_pre = None
        if grp is not None:
            y = nvfp4_matmul_group(h.reshape(-1, h.shape[-1]), grp)
            qkv_y, z_y = y.split(grp.sizes, dim=-1)
            mixed = qkv_y.reshape(B, T, -1).transpose(1, 2)
            z_pre = z_y.reshape(B, T, -1)
        else:
            mixed = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_qkv")).transpose(1, 2)
        raw = mixed if self.trace is not None else None
        cw = self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1)
        if self.tree is not None:
            # A tree has no single successor convolution state, so this does NOT advance the one in
            # `self.state`; `commit_tree` writes the accepted path's tail instead.
            mixed = gdn.conv_tree(mixed, self.state.conv[i], cw, self.tree.conv_idx)
        elif use_state:
            mixed = gdn.conv_update(mixed, self.state.conv[i], cw)
        else:
            mixed, new_conv = gdn.conv_prefill(mixed, cw)
            self.state.conv[i].copy_(new_conv)
        mixed = mixed.transpose(1, 2)
        q, k, v = mixed.split([cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1)
        q = q.view(B, T, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.view(B, T, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.view(B, T, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        z = (z_pre if z_pre is not None
             else linear(h, self.w.proj(f"{p}.linear_attn.in_proj_z"))).view(
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
        if self.trace is not None:
            # q, k and v are post-convolution and post repeat-interleave, so a prefix of them is
            # exactly what the recurrence for that prefix consumes; `raw` is what the convolution
            # state has to be rebuilt from.
            self.trace.layers[layer] = (raw[0].clone(), q.clone(), k.clone(), v.clone(),
                                        g.clone(), beta.clone())
        if self.tree is not None and FUSED["gdntree"] and use_state and T <= 64:
            # The state tile is loaded once and the node's ancestry is carried in registers, one
            # factor per depth. Everything the chunked path returns, the same three buffers.
            from tools.gdn_tree_kernels import fused_tree_step
            o, delta, gc = fused_tree_step(q, k, v, g, beta, self.tree.depths,
                                           self.state.S[i])
            self.trace.factors[layer] = (
                gdn.l2norm(k.float(), dim=-1).transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                gc.transpose(1, 2).contiguous())
        elif self.tree is not None:
            # The state is deliberately left at its entry value: the chunked form's final state sums
            # over every node, which on a tree mixes branches that never coexist. The accepted
            # path's state is reconstructed from the factors in `commit_tree`.
            o, _, fac = gdn.chunk_gated_delta_rule(
                q, k, v, g, beta, self.state.S[i], chunk_size=T, return_factors=True,
                tree=(self.tree.anc_incl, self.tree.anc_strict))
            if fac is None:
                raise RuntimeError("tree verify did not get its factors back")
            self.trace.factors[layer] = fac
        elif use_state and T == 1 and FUSED["gdn"]:
            from tools.gdn_kernels import fused_decode_step
            o = fused_decode_step(q, k, v, g, beta, self.state.S[i])
        elif use_state and T == 1:
            o, _ = gdn.recurrent_gated_delta_rule(q, k, v, g, beta, self.state.S[i])
        elif self.trace is not None and FUSED["gdnblock"] and use_state and T <= 16:
            # A verify block is eight tokens, and eight tokens of this recurrence fit in registers.
            # The chunked form exists for sequences that do not fit: it turns the recurrence into
            # matrix work, pays a serial matrix inverse for the intra-chunk term, and reads the
            # state several times. Walking the state forward eight times in one kernel reads it
            # once. The per-token update vectors come back with it, so the rank-k rollback needs
            # nothing extra.
            from tools.gdn_kernels import fused_block_step
            o, delta = fused_block_step(q, k, v, g, beta, self.state.S[i])
            self.trace.factors[layer] = (
                gdn.l2norm(k.float(), dim=-1).transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                g.float().cumsum(dim=1).transpose(1, 2).contiguous())
        elif self.trace is not None:
            o, S, fac = gdn.chunk_gated_delta_rule(
                q, k, v, g, beta, self.state.S[i] if use_state else None,
                chunk_size=GDN_PREFILL_CHUNK if T > GDN_PREFILL_CHUNK else max(2, T),
                return_factors=True)
            self.state.S[i].copy_(S)
            if fac is not None:
                self.trace.factors[layer] = fac
        else:
            o, S = gdn.chunk_gated_delta_rule(
                q, k, v, g, beta, self.state.S[i] if use_state else None,
                chunk_size=GDN_PREFILL_CHUNK if T > GDN_PREFILL_CHUNK else max(2, T))
            self.state.S[i].copy_(S)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        o = o.view(B, T, -1)
        return linear(o, self.w.proj(f"{p}.linear_attn.out_proj"))

    def _linear_attention_decode(self, h: torch.Tensor, p: str, i: int) -> torch.Tensor:
        """The T = 1 mixer with the glue in two kernels instead of a dozen.

        Same arithmetic as the general path above, in the same order; what is gone is the launches.
        The convolution, its SiLU and the state shift are one kernel, `g` and `beta` are one more,
        and the key side is not widened to forty-eight heads at all -- the recurrence kernel indexes
        it. Nine launches where the general path issues about twenty, on a mixer whose useful work
        is a 64.9 MB weight read.

        It runs only when there is no trace and no tree: a verify pass needs the pre-convolution
        projections kept for the rollback, and a tree has no single convolution successor.
        """
        from tools.gdn_kernels import decode_pre, fused_decode_step
        cfg = self.cfg
        mixed = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_qkv")).reshape(-1)
        z = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_z")).view(
            1, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        b = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_b.weight")).reshape(-1)
        a = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_a.weight")).reshape(-1)
        qkv, g, beta = decode_pre(
            mixed, self.state.conv[i],
            self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1),
            a, b, self.w.norm(f"{p}.linear_attn.A_log"),
            self.w.norm(f"{p}.linear_attn.dt_bias"))
        kd, vd = cfg.key_dim, cfg.value_dim
        q = qkv[:kd].view(1, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = qkv[kd:2 * kd].view(1, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = qkv[2 * kd:2 * kd + vd].view(1, 1, cfg.linear_num_value_heads,
                                         cfg.linear_value_head_dim)
        o = fused_decode_step(q, k, v, g, beta, self.state.S[i], rep=cfg.num_v_per_k)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        return linear(o.view(1, 1, -1), self.w.proj(f"{p}.linear_attn.out_proj"))

    # ---------------------------------------------------------------- forward
    def forward(self, tokens: torch.Tensor, start: int = 0, *,
                last_only: bool = False) -> torch.Tensor:
        cfg = self.cfg
        T = tokens.numel()
        h = F.embedding(tokens.view(1, T), self.w.norm("embed_tokens.weight"))
        if self.tap is not None:
            self.tap(h[0].detach())
        # A tree node's POSITION is its depth, not its index: two siblings are both the token
        # after their parent and both carry that position, which is what makes a tree a set of
        # alternative continuations rather than a longer sequence.
        positions = (start + self.tree.depths) if self.tree is not None else \
            torch.arange(start, start + T, device=self.device)
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
        self.hidden_pre_norm = h
        h = rms_norm(h, self.w.norm("norm.weight"), cfg.rms_norm_eps)
        self.hidden_post_norm = h
        if last_only:
            h = h[:, -1:]
        return head_logits(h, self.w.norm("lm_head.weight"))

    def forward_block(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        """Verify a block of tokens in one pass, keeping what a rollback would need.

        Returns logits for every position: `logits[i]` is the distribution for the token that
        follows `tokens[i]`.
        """
        self.trace = BlockTrace()
        self.trace.S_entry = self.state.S.clone()
        self.trace.conv_entry = self.state.conv.clone()
        try:
            logits = self.forward(tokens, start=start)
        finally:
            trace, self.trace = self.trace, None
        self._trace = trace
        return logits[0]

    def forward_tree(self, tokens: torch.Tensor, parents, start: int) -> torch.Tensor:
        """Verify a whole draft TREE in one forward pass.

        `tokens` and `parents` are `engine.tree.DraftTree` in DFS pre-order: node 0 is the anchor,
        every other node is a drafted continuation of its parent, and every parent index is lower
        than its child's. `logits[i]` is what the target would write after the path root..i.

        The step costs what a chain of the same length costs -- the weights are read once either way
        -- so on this board the tree is nearly free width: the measured NVFP4 curve is 126 ms at one
        row, 145 ms from two to four, 153 ms at eight and 175 ms at sixteen, while a chain of the
        same sixteen nodes can only ever accept along one line.

        Nothing is committed here. `commit_tree` takes the path the caller accepted.
        """
        n = len(parents)
        if tokens.numel() != n:
            raise ValueError(f"{tokens.numel()} tokens for {n} parents")
        if n < 2:
            raise ValueError("a tree verify needs an anchor and at least one draft")
        ctx = TreeCtx.get(parents, self.device, self.cfg.linear_conv_kernel_dim)
        if ctx.is_chain and TREE_CHAIN_DELEGATE:
            # A chain-shaped tree IS a chain, and the chain has a kernel this path cannot use:
            # `fused_block_step` walks the recurrence in registers where the tree carries a factor
            # per depth, and the difference was measured at 12.5 ms a block (SPEED-LEDGER 13:49).
            # So a drafter that proposes a line gets the line's price, and the router can fall back
            # to a chain by proposing one rather than by knowing anything about kernels.
            self._tree = ctx
            self._tree_start = start
            self._tree_is_chain = True
            return self.forward_block(tokens, start)
        self._tree_is_chain = False
        self.trace = BlockTrace()
        # A TREE verify never advances the recurrent state, so there is nothing for the clone to
        # protect against. Both tree paths say so in their own words -- `fused_tree_step`: "the
        # entry state is NOT advanced: a tree has as many final states as it has leaves", and the
        # chunked one leaves it deliberately because its final state sums over nodes that never
        # coexist -- and `commit_tree` is what reconstructs the accepted path's state afterwards.
        # The clone is 302 MB a block, about 1.7 ms of 134.
        #
        # What it costs to skip: `S_entry` then ALIASES `state.S`. `commit_tree` reads `S_entry[i]`
        # and writes `state.S[i]` in the same iteration, which is safe because the right-hand side
        # is evaluated into a new tensor before the copy and no iteration reads another's index;
        # `rollback` becomes a self-copy, which is a no-op and is the right answer for a state that
        # was never changed. The chain path keeps its clone, where `fused_block_step` really does
        # walk the state forward in place.
        self.trace.S_entry = (self.state.S if TREE_ALIAS_STATE else self.state.S.clone())
        self.trace.conv_entry = self.state.conv.clone()
        self.tree = ctx
        try:
            logits = self.forward(tokens, start=start)
        finally:
            self.tree = None
            trace, self.trace = self.trace, None
        self._trace = trace
        self._tree = ctx
        self._tree_start = start
        return logits[0]

    def commit_tree(self, path: list[int]) -> None:
        """Make the engine's state what verifying `path` alone as a chain would have left.

        `path` is node indices from the anchor down, ascending, starting at 0. For the 48 linear
        layers this is the SpecLA identity: within one chunk the pseudo-values `u_t` depend only on
        the tokens above t on its own branch, so the state after any path is

            S_path = exp(gc[last]) . S_entry + SUM_{t in path} exp(gc[last] - gc[t]) . k_t (x) u_t

        which is one state read and a [Dk, L] x [L, Dv] product per layer, over factors the verify
        pass already computed. It is the rank-k rollback of `rollback_to` with the prefix replaced
        by a gather, and it costs the same whether the path is the whole tree or one node of it.

        For the 16 attention layers the cache holds every node at its DFS slot, so committing is a
        gather of the accepted rows down to `start .. start + L - 1`; everything past that is
        overwritten by the next block and never read, because `kv.length` says so.
        """
        trace, ctx = self._trace, self._tree
        if trace is None or ctx is None:
            raise RuntimeError("commit_tree without a preceding forward_tree")
        if self._tree_is_chain:
            # the path of a chain is a prefix, so the chain rollback is the whole of the commit
            self.rollback_to(len(path))
            self._tree = None
            return
        if not path or path[0] != 0 or any(b <= a for a, b in zip(path, path[1:])):
            raise ValueError(f"path must start at the anchor and ascend: {path}")
        start, width, L = self._tree_start, self.cfg.linear_conv_kernel_dim, len(path)
        idx = torch.tensor(path, dtype=torch.long, device=self.device)
        last = path[-1]
        for layer, (kk, u, gc) in trace.factors.items():
            i = self.state.slot[layer]
            gl = gc[:, :, last]                                        # [B, H]
            w = torch.exp(gl[..., None] - gc[:, :, idx])               # [B, H, L]
            kw = (kk[:, :, idx] * w[..., None]).transpose(-1, -2)      # [B, H, Dk, L]
            self.state.S[i].copy_(gl[..., None, None].exp() * trace.S_entry[i]
                                  + kw @ u[:, :, idx])
            raw = trace.layers[layer][0]                               # [C, n], pre-convolution
            self.state.conv[i].copy_(gdn.conv_tail(raw[None], trace.conv_entry[i], idx, width))
        # A path that is already a DFS prefix needs no gather: its rows are the rows they would be
        # copied to. That is every block of a chain-shaped tree, accepted in full or not.
        if path != list(range(L)):
            sel = start + idx
            self.kv.k[..., start:start + L, :] = self.kv.k[..., sel, :]
            self.kv.v[..., start:start + L, :] = self.kv.v[..., sel, :]
        self.kv.length = start + L
        self._trace = None
        self._tree = None

    def accept_tree(self, tree, picks: list[int]) -> tuple[list[int], list[int]]:
        """Walk the tree along what the target actually wrote. Returns (path, new tokens).

        Greedy verification of a tree is a walk, not a comparison: at each node take the target's
        own argmax and descend into the child carrying that token, stopping where there is none.
        The tokens gained are the path's own plus the target's token at the node it stopped at, so a
        tree yields at least one token exactly as a chain does.
        """
        node, path = 0, [0]
        while True:
            want = picks[node]
            nxt = next((c for c in range(node + 1, len(tree.tokens))
                        if tree.parents[c] == node and tree.tokens[c] == want), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        new = [tree.tokens[i] for i in path[1:]] + [picks[node]]
        return path, new

    def rollback_to(self, keep: int) -> None:
        """Make the state what it would have been after only the first `keep` tokens of the block.

        No weight is read: the entry state and the projections cached during the verify pass are
        everything the recurrence needs. The convolution state is rebuilt from the same prefix of
        the pre-convolution projections.
        """
        trace = self._trace
        if trace is None:
            raise RuntimeError("rollback_to without a preceding forward_block")
        width = self.cfg.linear_conv_kernel_dim
        self.state.S.copy_(trace.S_entry)
        if trace.factors and len(trace.factors) == len(trace.layers) and RANKK != "0":
            # Rank-k: one state read and one [Dk, keep] x [keep, Dv] product per layer. The replay
            # below re-runs the chunked recurrence, whose intra-chunk inverse is a serial loop, and
            # the ledger charges it 22-23 ms on every block that takes a rollback.
            for layer, (kk, u, gc) in trace.factors.items():
                i = self.state.slot[layer]
                gk = gc[:, :, keep - 1]                                   # [B, H]
                w = torch.exp(gk[..., None] - gc[:, :, :keep])            # [B, H, keep]
                kw = (kk[:, :, :keep] * w[..., None]).transpose(-1, -2)   # [B, H, Dk, keep]
                S_rank = gk[..., None, None].exp() * trace.S_entry[i] + kw @ u[:, :, :keep]
                if RANKK == "check":
                    q, k2, v2, g2, b2 = trace.layers[layer][1:]
                    _, S_replay = gdn.chunk_gated_delta_rule(
                        q[:, :keep], k2[:, :keep], v2[:, :keep], g2[:, :keep], b2[:, :keep],
                        trace.S_entry[i], chunk_size=max(2, min(64, keep)))
                    d = (S_rank - S_replay).abs().max().item()
                    ROLLBACK_DIFF.append((layer, keep, d,
                                          S_replay.abs().max().item()))
                self.state.S[i].copy_(S_rank)
                raw = trace.layers[layer][0]
                joined = torch.cat([trace.conv_entry[i], raw[None, :, :keep]], dim=-1)
                self.state.conv[i].copy_(joined[:, :, -(width - 1):])
            return
        # The chunk size is a blocking choice, not a semantic one, and the chunked form pays a
        # serial loop of `chunk_size - 1` small operations for its intra-chunk inverse. A verified
        # block is a dozen tokens, so blocking it at 64 runs that loop 63 times per layer, 3,000
        # times per rollback, for nothing. Block it at the prefix length instead.
        chunk = max(2, min(64, keep))
        for layer, (raw, q, k, v, g, beta) in trace.layers.items():
            i = self.state.slot[layer]
            _, S = gdn.chunk_gated_delta_rule(q[:, :keep], k[:, :keep], v[:, :keep],
                                              g[:, :keep], beta[:, :keep],
                                              trace.S_entry[i], chunk_size=chunk)
            self.state.S[i].copy_(S)
            joined = torch.cat([trace.conv_entry[i], raw[None, :, :keep]], dim=-1)
            self.state.conv[i].copy_(joined[:, :, -(width - 1):])

    def reset(self) -> None:
        self.state.S.zero_()
        self.state.conv.zero_()
        self.state.primed = False
        self.kv.length = 0
