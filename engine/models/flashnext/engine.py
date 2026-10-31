"""Flash-Next on the board (ENG-221, ENG-227): the reference forward with its weight reads on the
kernels, and a row step -- one decode token, or a verify block of up to 32 draft rows -- that reads
its inputs from device buffers, so a CUDA graph can hold it.

`FlashNextRef` (ref.py) is the model, operation for operation, checked against `transformers`. This
class keeps its structure and routes what reads weights to kernels over the pack's formats:

  EXL3 projections and the head     tools/fn_kernels.dense (up to 32 rows), tools/exl3_linear above
  mixers (int8 rows or fp16)        tools/fn_kernels.gemv (down and inject in one weight) + mix_up
  router, routed and shared experts tools/fn_kernels.router + moe (one bank, the shared expert last)
  GDN a/b, PLE projections (fp16)   tools/fn_kernels.gemv
  GDN conv, gates and recurrence    tools/fn_seq_kernels.gdn_pre + gdn_rows (a row's path)
  PLE gate and dilated conv         tools/fn_seq_kernels.ple_gate + ple_conv (a row's path)
  attention                         tools/fn_seq_kernels.attn_rows (a row's visible keys)

`forward()` (the reference's) serves a prompt chunk of any length. `rows_step()` is M rows whose
tokens, positions, cache slots, ancestors and PLE rows sit in device buffers (`stage` for a decode
token, `stage_rows` for a draft block in DFS pre-order); `DecodeGraph` and `VerifyGraph` capture it.
Every part of a row is computed with an instruction sequence that does not depend on the other rows
or on where its predecessors sit, so the row of a verify block is the decode step's row at that
position bit for bit: a greedy speculative run writes the plain greedy text (ENG-227). A decode step
advances the state in place; a verify block writes nothing but KV rows past the length and keeps what
`commit` needs, which then walks the accepted path's GDN rows again from the entry state (the decode
arithmetic), takes the conv windows from the path and moves the path's KV rows into place.

The row step covers the dense attention regime: a context of up to `indexer_budget + 3` tokens
(2,051), where QSA keeps every token. Past it a decode step takes the reference's attention (eager,
slow, exact) and a verify block is refused; the selected-key path is ENG-147's work.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from engine import capture_lock
from engine.settings import SETTINGS as _S
from engine.models.flashnext.ref import MTP, FlashNextRef, FNState
from engine.models.flashnext.ple import decode_rows


# rows up to which an EXL3 projection runs on tools/fn_kernels.dense (the kernel's limit);
# QWEN38_FN_DENSE=0: the library path of tools/exl3_linear for every row count (an A/B)
DENSE_MAX = 32 if _S.get("FN_DENSE") == "1" else 0
#: SPD-215: the joint pass runs its row-wise parts once over every state's rows (`_verify_multi_rowwise`;
#: 8 states x 4 rows: 159 -> 150 ms, every state's logits its lone verify's bits)
ROWWISE = _S.get("FN_JOINT_ROWWISE") == "1"
#: SPD-215: the joint pass's experts through the grouped kernels (a block's pairs of one expert share a CTA;
#: tools/fn_kernels.py MOE_GROUP); a lone verify keeps its kernels
JOINT_MOE_GROUP = _S.get("FN_JOINT_MOE_GROUP") == "1"
#: SPD-239: every N joint passes their GPU ms by part in the log (`_timed`); 0: off
JOINT_TIMING = int(_S.get("FN_JOINT_TIMING") or 0)
#: SPD-239: a joint pass reads every state's PLE rows in one batch a layer (`_ple_batch`); 0: a block at a time
PLE_BATCH = _S.get("FN_PLE_BATCH") == "1"
#: SPD-239: a joint round's commits pool every lane's indexer blocks of every layer in one launch; 0: a launch each
POOL_MULTI = _S.get("FN_POOL_MULTI") == "1"
#: SPD-239: a joint pass reads its PLE rows on a thread while the layers before the first PLE layer run; 0: before the pass
PLE_LATE = _S.get("FN_PLE_LATE") == "1"
#: SPD-239: a joint pass runs every state's GDN delta rule of a layer in one launch (`_gdn_multi`; 8 lanes: 17.4 -> 13.5 ms
#: a pass); 0: a state at a time
GDN_MULTI = _S.get("FN_GDN_MULTI") == "1"
#: SPD-249: the mixers' down GEMV writes the mix_up's A in its own launch (`fn_kernels.gemv_mix`); 0: mix_a apart
MIX_FUSED = _S.get("FN_MIX_FUSED") == "1"
#: SPD-249 P2: a prompt chunk's mixers in two launches a site (`_mix_pre`, tools/fn_pf_hc.py), the same bits; 0: the
#: library GEMMs with silu and mix_tail apart
PF_HC_FUSED = _S.get("FN_PF_HC_FUSED") == "1"
#: SPD-249 P2: a prompt chunk's injects in a CTA a row (`_inject_pre`, tools/fn_pf_hc.py), the same bits; 0: the
#: decode path's inject_norm
PF_INJ_FUSED = _S.get("FN_PF_INJ_FUSED") == "1"
#: SPD-249 P6: the pipeline stages of a prompt chunk's attention split launch (`_attn_pre`), the same bits; 2: the
#: decode path's launch
PF_ATTN_STAGES = int(_S.get("FN_PF_ATTN_STAGES") or 1)
#: SPD-249 P4: a prompt chunk's copies that a view or the reading kernel's own stride serves are not launched (the GDN
#: values and z in place, the PLE states from the rows' words, the indexer scores' fill past the pooled blocks only, a
#: window's mixer outputs written side by side), the same bits; 0: the copies
PF_NOCOPY = _S.get("FN_PF_NOCOPY") == "1"
#: SPD-249: a joint pass's GDN conv and gates of every state in one launch a layer (`gdn_pre_multi`); 0: a launch a state
GDN_PRE_MULTI = _S.get("FN_GDN_PRE_MULTI") == "1"
#: SPD-249: a joint pass leaves up to N parked GDN rows parked (no state write-back that round; the next call walks
#: them again, bit for bit); 0: every call writes the state back
GDN_DEFER = min(int(_S.get("FN_GDN_DEFER") or 0), 32)
#: the parked-row buffers' rows (`_pend_bufs`; GDN_PEND_ROWS in tools/csrc/fn_seq_kernels/seq.cu)
PEND_ROWS = 33
#: SPD-249: every N-th joint pass counts its distinct routed experts a layer (`union.ExpertUnion`); 0: off
UNION_EVERY = int(_S.get("FN_UNION_EVERY") or 0)
#: SPD-249: a joint pass's attention layer preps and attends every lane's rows in one launch each (`_attn_multi`: lane
#: entry tables on the device, each row's own arithmetic) and gates them once; 0: a lane at a time
ATTN_MULTI = _S.get("FN_ATTN_MULTI") == "1"
#: SPD-249: a joint pass's PLE layer decodes, projects, norms and gates every lane's rows at once (row-wise), only the
#: dilated conv over each lane's own state a lane at a time; 0: the whole PLE step a lane at a time
PLE_MULTI = _S.get("FN_PLE_MULTI") == "1"
#: SPD-249: a dense joint pass (every lane's context within the dense class) from a CUDA graph keyed by (lanes, the
#: rows padded to JG_ROWS, class 0, the lanes' KV strides): the lanes' rows contiguous in one rows buffer, every lane's state reached through
#: the graph's entry tables, the pad rows routed to no expert; a key the server has not let capture yet runs eager.
#: 0: every joint pass eager
JOINT_GRAPH = _S.get("FN_JOINT_GRAPH") == "1"
# SPD-249: stage_rows' tree arrays kept by tree shape, a joint graph's lane views kept (the same host rows)
TREE_PLAN = _S.get("FN_TREE_PLAN") == "1"
# SPD-249: a joint graph pass reads its PLE rows on the reader thread while the segments before the PLE layer run
JG_PLE_LATE = _S.get("FN_JG_PLE_LATE") == "1"
#: SPD-249: a lane's tree is handed over as soon as its tokens are known (`ple_prefetch`, the round runner after its
#: drafts' collect): its PLE rows are read into the table's row cache on the reader thread, so the verify's own read
#: (the same ids, the same rows) finds them there; 0: off
PLE_PREFETCH = _S.get("FN_PLE_PREFETCH") == "1"
#: SPD-249: the joint graphs kept (the least recently replayed go first)
JOINT_GRAPH_MAX = int(_S.get("FN_JOINT_GRAPH_MAX") or 12)
#: SPD-249: a key is captured at its n-th pass (the eager ones before it compile every kernel it runs)
JOINT_GRAPH_AFTER = int(_S.get("FN_JOINT_GRAPH_AFTER") or 2)
#: SPD-249: the padded row counts of the joint graphs (a pass's rows go up to the next one)
JG_ROWS = (16, 24, 32)
#: SPD-249: the most lanes a joint graph holds (its entry tables)
JG_LANES = 8
#: SPD-249: the layers where a joint graph's pass is cut into graphs of its own: the host writes a segment's entry rows
#: while the GPU runs the segment before it (only the first segment's rows wait for the host)
JG_CUTS = (2, 10, 22, 34)
#: SPD-249 (T1): a packed MTP draft's depth loop (each lane's draft head and top-2, then the one-row steps after the feed)
#: replayed from a CUDA graph a lane count (`PackGraph`, key: lanes, depth, draft head, the caches' strides): every lane's
#: MTP cache and staged rows reached through the graph's attention entry tables (`attn_prep_multi` / `attn_rows_multi`,
#: each row's arithmetic its lane-at-a-time launch's), the lanes' picks in one pinned copy; the feed of the committed rows
#: stays eager. A dense draft only (every lane's class 0). 0: the whole packed draft eager
PACK_GRAPH = _S.get("FN_PACK_GRAPH") == "1"
#: SPD-249 (T1): a key's draft graph is captured at its n-th packed draft (the eager drafts before it load what it runs)
PACK_GRAPH_AFTER = int(_S.get("FN_PACK_GRAPH_AFTER") or 2)
#: SPD-249 (T1): the packed draft graphs kept; one more resets them all (their capture stream's workspace goes with them)
PACK_GRAPH_MAX = 16
#: SPD-249 (T1): the pinned host copies of a draft graph's picks, used in turn: a lane's draft is read before its graph's
#: PACK_RING-th next replay (a round reads its drafts before the next one is queued; a lane that sits rounds out reads
#: later, and the same lane count replays at most once a round meanwhile)
PACK_RING = 8
# ENG-225: a prompt chunk through the prefill kernels (tools/fn_prefill_kernels.py: experts grouped by
# expert, mixers and attention on many rows) or the reference forward ("ref"), and the chunk length
PREFILL = _S.get("FN_PREFILL")
PREFILL_CHUNK = int(_S.get("FN_PREFILL_CHUNK"))
#: the prefill's rows a tile of the indexer's scores and select (`_attn_pre`): a multiple of the scores' 1,024-row
#: GEMM block, so a tile's products are the whole chunk's; and the tokens a tile of the PLE decode (`_ple_pre`)
QSA_TILE = 1024
PLE_DECODE_TOKENS = 512


class ClassError(RuntimeError):
    """SPD-249: an attention launch at a context class below its rows' own (see `class_check`)."""


class FlashNextEngine(FlashNextRef):

    #: rows an item of the prefill MoE group GEMM takes; None = `fn_prefill_kernels.GROUP_ROWS`
    #: (a family whose bits were not checked at 64 keeps 32)
    moe_group_rows: int | None = None

    def __init__(self, cfg, W: dict, tables: dict, banks: dict, device="cuda", max_len: int = 8192):
        super().__init__(cfg, W, tables, dtype=torch.bfloat16, device=device)
        self.banks = banks
        self.max_len = max_len
        cos, sin = FlashNextRef.rope(self, torch.arange(max_len))
        self.rope_cos, self.rope_sin = cos, sin                          # [max_len, rotary] bf16
        self.dense_limit = (cfg.indexer_budget + cfg.indexer_compress_ratio - 1) if cfg.has_indexer else 1 << 60
        self._rb: dict = {}                                              # rows buffers by row count
        self._order1 = torch.empty(0, dtype=torch.int32, device=device)    # one row: launch order
        self.rec: dict = {}                                              # what a commit needs
        self.host_ms: dict = {}                                          # host staging time by part
        # the commit folded into the next call (QWEN38_FN_FOLD_COMMIT): the accepted path's GDN rows,
        # parked a layer, applied by the next verify block or decode step to each state tile as it
        # loads it (bit for bit the commit's own walk). SPD-229: each state parks in its own buffers
        # (`_pend_of`), so a joint pass's lanes each fold their own rows into their walk; one set of
        # buffers made every lane's commit land the lane before's rows first, a walk of its own a layer
        self.fold = _S.get("FN_FOLD_COMMIT") == "1"
        self._lin_idx = {li: i for i, li in enumerate(cfg.linear_layers)}
        self._pending: dict = {}         # id(state) -> the state, while it has parked rows
        self._joint: dict = {}          # id(state) -> (block, rec, class) of its block in the last joint pass
        self._mtp_key = 0               # whose MTP chain buffers the next draft uses (a head's `key`)
        self._multi = None              # SPD-235: [(rows buffers, MTP state, rows, class)] of a packed MTP step
        self.joint_timing = {} if JOINT_TIMING else None  # a dict: `verify_multi` adds each part's GPU ms (a measurement)
        from engine.models.flashnext.union import ExpertUnion
        self.union = ExpertUnion(cfg.num_hidden_layers, cfg.num_experts, UNION_EVERY, device)   # SPD-249
        # the sequence kernels' weight forms: GDN conv taps [C, 4] bf16, A_log and dt_bias fp32, the
        # PLE conv taps rounded to bf16 (the reference casts them to the activations' dtype)
        self._wc: dict = {}
        for li in cfg.linear_layers:
            a = self.pre(li) + "linear_attn."
            if a + "conv1d.weight" not in W:
                continue
            self._wc[li] = W[a + "conv1d.weight"].reshape(cfg.conv_dim, -1).to(torch.bfloat16).contiguous()
            self._wc[("alog", li)] = W[a + "A_log"].float().contiguous()
            self._wc[("dtb", li)] = W[a + "dt_bias"].float().contiguous()
        for li in cfg.ple_layers:
            w = self.pre(li) + "ple.conv1d.weight"
            if w in W:
                self._wc[("ple", li)] = W[w].reshape(W[w].shape[0], -1).to(torch.bfloat16).contiguous()

    # ------------------------------------------------------------------------- state
    def new_state(self, cap: int | None = None, paged: bool = False) -> FNState:
        """A state with every tensor the step writes allocated up front (a graph holds them). `paged`
        with the KV pool on (engine/kvpool.py, ENG-241): the K, V and indexer-key buffers are reserved
        as addresses and committed as rows are written -- by the caller, before every write
        (`FlashNextServing._ensure`) -- with the same shapes, strides and addresses for every kernel and
        graph, so the bits are the static buffers'."""
        from engine import kvpool as _kvp
        c = self.cfg
        cap = cap or self.max_len
        st = FNState(cap=cap)
        if (paged and _kvp.POOL is not None and cap >= _kvp.PAGED_MIN_ROWS
                and str(self.device).startswith("cuda")):
            st.pages = _kvp.PagedSet(_kvp.OWNER)
        dev = self.device
        for li in range(c.num_hidden_layers):
            if c.is_linear(li):
                st.S[li] = torch.zeros(1, c.linear_num_value_heads, c.linear_key_head_dim,
                                       c.linear_value_head_dim, dtype=torch.float32, device=dev)
                st.conv[li] = torch.zeros(1, c.conv_dim, c.linear_conv_kernel_dim - 1, dtype=self.dtype, device=dev)
            else:
                st.k[li] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, f"k.{li}")
                st.v[li] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, f"v.{li}")
                if c.has_indexer:
                    st.ik[li] = self._rows_tensor(st, (cap, c.indexer_head_dim), 0, f"ik.{li}")
                    st.pool[li] = self._pool_tensor(cap)
        for li in c.ple_layers:
            st.ple_conv[li] = torch.zeros(1, c.hc_hidden, (c.ple_conv_kernel_size - 1) * c.ngram_size,
                                          dtype=self.dtype, device=dev)
        return st

    def _rows_tensor(self, st: FNState, shape: tuple, axis: int, name: str) -> torch.Tensor:
        """A buffer indexed by position on `axis`: paged in the state's set, else zeros."""
        if st.pages is None:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)
        from engine import kvpool as _kvp
        return st.pages.add(_kvp.PagedBuffer(shape, axis, self.dtype, self.device, owner=st.pages.owner,
                                             name=f"fn.{name}")).tensor

    def _pool_tensor(self, cap: int) -> torch.Tensor:
        """The pooled indexer keys of a cache of `cap` tokens: a row a block of 4, kept up to date by
        every decode step and commit (ENG-147), plus the rows a commit may write past the last block."""
        from tools import fn_seq_kernels as SK
        return torch.zeros(cap // self.cfg.indexer_compress_ratio + SK.OWN + 1, self.cfg.indexer_head_dim,
                           dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def build_pool(self, st: FNState, layers=None) -> None:
        """Pool every complete block of the state's indexer keys (after a prefill, which the reference
        forward runs without the pool). The same kernel as a step's, so the bits are the step's."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        nb = st.length // c.indexer_compress_ratio
        if not c.has_indexer or nb == 0:
            return
        dev = self.device
        b = torch.arange(nb, dtype=torch.int32, device=dev)
        slots = (b[:, None] * 4 + torch.arange(4, dtype=torch.int32, device=dev)).contiguous()
        cos, sin, off = self._rot(st, 0, nb * 4)
        for li in (layers if layers is not None else c.attention_layers):
            p = self.pre(li) + "self_attn.indexer."
            SK.qsa_pool(st.ik[li], slots, b * 4 + off, self.W[p + "k_layernorm.weight"], cos, sin,
                        c.rms_norm_eps, out=st.pool[li][:nb])

    def reset(self, st: FNState) -> None:
        self.drop_parked(st)
        st.reset_inplace()
        for d in (st.k, st.v, st.ik, st.pool):
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

    def _rot(self, st: FNState, a: int, b: int):
        """The rotary rows of cache slots [a, b) of `st` (VIS-97): (cos, sin, off), slot s reads row
        s + off. A text row reads the 1-D table at its position, slot + delta: 0 before any image, the
        prompt's `pos3` after one, `st.delta` past the prompt (transformers' `get_rope_index`). A range
        that holds image rows gets rows of its own: engine/vision.py `mrope_cos_sin` of the three
        coordinates, on the host as the table is made, so a text row among them is the table's row bit
        for bit."""
        mm = st.mm
        if mm is None:
            return self.rope_cos, self.rope_sin, st.delta
        if not any(s.start < b and s.start + s.n > a for s in mm.spans):
            return self.rope_cos, self.rope_sin, (int(mm.pos3[0, a]) - a if a < mm.n else st.delta)
        from engine.vision import mrope_axes, mrope_cos_sin
        c = self.cfg
        p3 = mm.pos3[:, a:b]
        if b > mm.n:
            tail = torch.arange(max(a, mm.n), b) + st.delta
            p3 = torch.cat([p3, tail.view(1, -1).expand(3, -1)], dim=1)
        axes = torch.tensor(mrope_axes(c.rotary_dim, list(c.mrope_section)))
        cos, sin = mrope_cos_sin(p3, self.inv_freq, axes, dtype=self.dtype)
        return cos.to(self.device), sin.to(self.device), -a

    def _image_rows(self, e: torch.Tensor, st: FNState, L0: int) -> torch.Tensor:
        """A chunk's embeddings [T, H] at slots L0.. with the images' rows over their placeholders'
        (transformers' `masked_scatter` of the image features into `inputs_embeds`); an image is encoded
        when the first chunk that needs it runs (engine/vision.py `MMContext.embeds`)."""
        mm, T = st.mm, e.shape[0]
        for s in mm.spans:
            a, b = max(s.start, L0), min(s.start + s.n, L0 + T)
            if a < b:
                e[a - L0:b - L0] = mm.embeds(s.image)[a - s.start:b - s.start].to(e.dtype)
        return e

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

    def moe_parts(self, x, li, pad=None):
        """The router and the experts of every (row, slot) pair, without the combine (the trunk
        folds it into the next inject): (yd [M S, H] fp32, ids, wts). `pad` (SPD-249, a joint graph): bool [M, 1]
        on the device, its rows' ids set to -1 (the grouped kernels skip them: they pull no expert)."""
        from tools import fn_kernels as K
        c = self.cfg
        logits = K.gemv(x, self.W[self.pre(li) + "mlp.router"])
        ids, wts = K.router(logits, c.num_experts, c.num_experts_per_tok)
        if pad is not None:
            ids.masked_fill_(pad, -1)
        if self.route_trace is not None:
            self.route_trace.append((li, ids[:, :-1].long()))
        yd = K.moe_experts(x, ids, self.banks[li], order=self._order1 if x.shape[0] == 1 else None)
        if self._moe_record is not None:
            self._moe_record.append((li, x.clone(), ids.clone(), wts.clone(), yd.clone()))
        return yd, ids, wts

    _moe_record = None
    spec_ok = False            # the load-time self-check passed (speculation may run)
    _mtp_pg = None             # SPD-249 (T1): a packed draft graph's step (its tokens, lanes, graph)
    pack_capture_ok = False    # SPD-249 (T1): a packed draft graph may be captured now (the server's word)
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
        """The mixer from its normed streams: (x [M, H], d [M, R + hc] fp32 = down and inject). SPD-249
        (QWEN38_FN_MIX_FUSED): int8 mixers write the mix_up's A in the down GEMV's launch (`gemv_mix`), one launch
        a mixer fewer, the same bits."""
        from tools import fn_kernels as K
        wd, wu = self.W[p + ".down_inject"], self.W[p + ".input_mix_weight_up.weight"]
        if MIX_FUSED and hasattr(wd, "q") and hasattr(wu, "q") and wd.q.dtype == torch.int8 and wu.q.dtype == torch.int8:
            di, a = K.gemv_mix(n, wd, wu.q.shape[1], self.cfg.hc_count)
            return K.mix_up_a(a, n, wu, self.cfg.hc_count), di
        di = K.gemv(n, wd)
        return K.mix_up(di, n, wu, self.cfg.hc_count), di

    def _site_norm(self, li: int):
        """The norm weight of layer li's first mixer (the final mixer's past the last layer)."""
        if li >= self.cfg.num_hidden_layers:
            return self.W["hyper_connection_mixer.hc_norm.weight"]
        return self.W[self.pre(li) + "attn_hyper_connection.hc_norm.weight"]

    def trunk(self, h, tokens, st, positions, step: bool = False, pre: bool = False):
        """Every layer: each inject writes the streams and, in the same launch, the next site's norm
        (not in front of a PLE layer, which changes the streams first). Returns the final mixer's
        normed streams n [M, hc H]; the streams themselves are `self.stack`. `pre`: a prompt chunk
        through the prefill kernels (ENG-225)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        mix = self._mix_pre if pre else self._mix
        n = None
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            if li in c.ple_layers:
                if step:
                    h = h + self._ple_step(h, li, st)
                else:
                    h = h + (self._ple_pre(h, tokens, li, st) if pre else self.ple(h, tokens, li, st))
                n = None
            if n is None:
                n = K.hc_norm(h, self._site_norm(li), H, eps)
            x, di = mix(n, p + "attn_hyper_connection")
            if c.is_linear(li):
                out = self._gdn1(x, li, st) if step else (self._gdn_pre(x, li, st) if pre else self.gdn(x, li, st))
            else:
                out = self._attn1(x, li, st) if step else (self._attn_pre(x, li, st) if pre else
                                                           self.attention(x, li, st, positions))
            h, n = (self._inject_pre if pre else K.inject_norm)(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                                                                self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
            x, di = mix(n, p + "mlp_hyper_connection")
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            R = self.W[p + "mlp_hyper_connection.rank"]
            if pre:
                h, n = self._inject_pre(h, self._moe_pre(x, li), di, R, nxt, hc, eps)
            elif x.shape[0] <= self.MOE_ROWS:
                yd, ids, wts = self.moe_parts(x, li)
                if step and not self._write:
                    self.rec[("ids", li)] = ids            # the experts a verify block reads (ENG-155 prices)
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

    # ------------------------------------------------------------------------- prompt chunks (ENG-225)
    @torch.no_grad()
    def forward_pre(self, tokens, st, head="all"):
        """A prompt chunk through the prefill kernels: the trunk's arithmetic shared across rows (experts
        grouped by expert, mixers and the router on the library GEMM, attention over each row's visible
        keys with the QSA selection on the kernels, the GDN conv and gates in one launch, the chunked
        delta rule). Agrees with `forward` on the next token, not bit for bit (tools/fn_prefill_check.py).
        Advances `st` as `forward` does: caches, recurrent and conv states, the pooled indexer keys."""
        tokens = tokens.view(-1).to(self.device)
        T = tokens.shape[0]
        self._pre = self._pre_ctl(st.length, T, st)
        e = self.embed(tokens)
        if st.mm is not None:
            e = self._image_rows(e, st, st.length)
        h = e.repeat(1, self.cfg.hc_count)
        n = self.trunk(h, tokens, st, None, pre=True)
        self._n32_clear()
        st.length += T
        self._pre = None
        if head == "none":
            return None
        return self.final(n[-1:] if head == "last" else n)

    def _pre_ctl(self, L0: int, T: int, st: FNState) -> dict:
        """What every layer of a chunk of `st` at positions L0..L0+T-1 shares: the positions, their
        rotary rows (`cos`, `sin` read at `rpos` by the rows and at `pool_pos` by the blocks the chunk
        completes, `_rot`), the GDN conv and PLE conv taps along the chain (`gdn_pre` / `ple_conv`
        sources), each row's attention control (`attn_rows`: visible keys, no ancestors, dense flag),
        and past 2,051 tokens the QSA inputs (`qsa_scores_rows` / `qsa_select`: complete blocks, all in
        the shared pool; the tail)."""
        from tools import fn_seq_kernels as SK
        c, dev = self.cfg, self.device
        pos = torch.arange(L0, L0 + T, device=dev)
        m = torch.arange(T, device=dev)[:, None]

        def chain(dists, slen):
            d = torch.tensor(dists, device=dev)[None, :]
            return torch.where(m - d >= 0, m - d, -1 - (slen - d + m)).to(torch.int32).contiguous()
        out = {"pos": pos, "csrc": chain((3, 2, 1), c.linear_conv_kernel_dim - 1),
               "psrc": chain((9, 6, 3), (c.ple_conv_kernel_size - 1) * c.ngram_size), "bufs": {}}
        r = c.indexer_compress_ratio if c.has_indexer else 4
        cos, sin, off = self._rot(st, L0 - L0 % r, L0 + T)
        out.update(cos=cos, sin=sin, rpos=pos + off)
        nblk = (pos + 1) // r
        dense = (nblk <= SK.TOPK) | (not c.has_indexer)
        ctl = torch.zeros(T, 4, dtype=torch.int32, device=dev)
        ctl[:, 0] = torch.where(dense, pos + 1, 4 * SK.TOPK + pos + 1 - r * nblk)
        ctl[:, 1] = 1 << 30                      # no row's keys come from ancestors: every key is cached
        ctl[:, 2] = dense
        out.update(ctl=ctl, anc=torch.zeros(T, 1, dtype=torch.int32, device=dev),
                   sel=torch.zeros(T, SK.NSEL, dtype=torch.int32, device=dev), cls=self.qsa_class(L0 + T - 1))
        if c.has_indexer:
            b = torch.arange(L0 // r, (L0 + T) // r, dtype=torch.int32, device=dev)
            out.update(nblk=nblk, tail=(r * nblk[:, None] + torch.arange(3, device=dev)[None, :]).to(torch.int32),
                       pool=(L0 // r, (L0 + T) // r), pool_slots=(b[:, None] * r + torch.arange(r, dtype=torch.int32,
                                                                                               device=dev)).contiguous(),
                       pool_pos=(b * r + off).contiguous())
        return out

    def _mix_pre(self, n, p, out=None):
        """The mixer of a chunk: down and inject on `mm_rows` (the library GEMM in fp32 by default), the up
        projection of a = bf16(silu(bf16(d) / hc)) likewise, the stream mean with mix_up's roundings (into `out`
        [M, H] bf16 when given; `mix_up_tail`: one launch for both under QWEN38_FN_PREFILL_MM=tcn). SPD-249 P2
        (QWEN38_FN_PF_HC_FUSED): the same GEMMs of the `mm_rows` mode that runs on the same values, a from one launch
        and, where the GEMM reads fp32, n's fp32 form from the inject's launch when it wrote one (`_inject_pre`;
        tools/fn_pf_hc.py `mix`)."""
        from tools import fn_prefill_kernels as PK
        c = self.cfg
        wd, wu = self.W[p + ".down_inject"], self.W[p + ".input_mix_weight_up.weight"]
        R = self.W[p + ".rank"]
        st = getattr(self, "_n32", None)
        ent = st.pop(id(n), None) if st else None
        if PF_HC_FUSED:
            from tools import fn_pf_hc as HC
            if HC.mix_ok(n, c.hc_count):
                return HC.mix(n, ent[1] if ent is not None and ent[0] is n else None, wd, wu, R, c.hc_count, out)
        di = PK.mm_rows(n, wd)
        a = torch.nn.functional.silu(di[:, :R].to(self.dtype).float() / c.hc_count).to(self.dtype)
        return PK.mix_up_tail(a, wu, n, c.hc_count, out=out), di

    def _inject_pre(self, h, out, di, R, w, hc, eps):
        """A chunk's inject and the next site's norm: `fn_kernels.inject_norm`, or with QWEN38_FN_PF_INJ_FUSED its bits
        in a CTA a row (tools/fn_pf_hc.py); with QWEN38_FN_PF_HC_FUSED too and a mixer GEMM that reads fp32, the same
        launch writes n's fp32 form for the next `_mix_pre` (kept by n's identity until that call takes it, or
        `_n32_clear`)."""
        from tools import fn_kernels as K
        if PF_INJ_FUSED:
            from tools import fn_pf_hc as HC
            if HC.inject_ok(h, out, hc):
                want = w is not None and PF_HC_FUSED and HC.wants_n32()
                hn, nn, n32 = HC.inject_norm(h, out, di, R, w, hc, eps, n32=want)
                if n32 is not None:
                    if getattr(self, "_n32", None) is None:
                        self._n32 = {}
                    self._n32[id(nn)] = (nn, n32)
                return hn, nn
        return K.inject_norm(h, out, di, R, w, hc, eps)

    def _n32_clear(self) -> None:
        """Drop the fp32 forms no mixer took (the last layer's, which feeds the final mixer's own kernels)."""
        if getattr(self, "_n32", None):
            self._n32.clear()

    def _moe_pre(self, x, li):
        """The MoE of a chunk: the router (library GEMM, the router kernel), every pair's expert output
        grouped by expert (`moe_grouped`), the combine (`moe_combine`: one CTA a row's columns reads
        each pair's output once; the decode path folds it into the inject, four times a row) -> [M, H]."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        c = self.cfg
        ids, wts = K.router(PK.mm_rows(x, self.W[self.pre(li) + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
        if self.route_trace is not None:
            self.route_trace.append((li, ids[:, :-1].long()))
        rows = self.moe_group_rows or PK.GROUP_ROWS
        return K.moe_combine(PK.moe_grouped(x, ids, self.banks[li], rows=rows), ids, wts)

    def _attn_pre(self, x, li, st):
        """Attention of a chunk: q|k|v and the indexer projection, the prep kernel (norms, rope, the
        chunk's keys, values and indexer keys into the caches at their positions), the chunk's complete
        blocks pooled into the shared pool, past 2,051 tokens each row's block scores and its top 512,
        attention over each row's visible keys (`attn_rows`), the output gate, o_proj."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        from tools import fn_seq_kernels as SK
        c, P = self.cfg, self._pre
        p = self.pre(li) + "self_attn."
        ip = p + "indexer."
        qkv = self.lin(x, self.W[p + "qkv"])
        iqk = self.lin(x, self.W[ip + "index_qk_proj.weight"]) if c.has_indexer else None
        cls = P["cls"] if c.has_indexer else 0
        iq = None
        if cls:
            iq = P["bufs"].get("iq")
            if iq is None:
                iq = P["bufs"]["iq"] = torch.empty(x.shape[0], c.indexer_n_heads, c.indexer_head_dim,
                                                   dtype=self.dtype, device=x.device)
        q = K.attn_prep(qkv, iqk, P["rpos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        P["cos"], P["sin"], st.k[li], st.v[li], st.ik.get(li), c.num_attention_heads,
                        c.num_key_value_heads, c.indexer_n_heads * c.indexer_head_dim if c.has_indexer else 0,
                        c.rms_norm_eps, slot=P["pos"], qln=self.W[ip + "q_layernorm.weight"] if cls else None, iq=iq)
        if c.has_indexer and P["pool"][1] > P["pool"][0]:
            SK.qsa_pool(st.ik[li], P["pool_slots"], P["pool_pos"], self.W[ip + "k_layernorm.weight"], P["cos"],
                        P["sin"], c.rms_norm_eps, out=st.pool[li][P["pool"][0]:P["pool"][1]])
        if cls:
            pool = st.pool[li][:P["pool"][1]] if _S.get("FN_QSA_LIVE") == "1" else st.pool[li]
            # in tiles of QSA_TILE rows: a chunk's scores are [rows, class] fp32 (1.07 GiB for 4,096 rows at 65,536
            # blocks, beside the products of each 1,024-row block), a tile's a quarter of it. The scores' GEMMs run
            # in blocks of 1,024 rows either way (`qsa_scores_rows`) and the select is a row's own, so the tiles'
            # arithmetic is the whole call's (measured 2026-11-06 at 200k: the chunk's peak 4.2 -> 2.7 GiB)
            for r0 in range(0, x.shape[0], QSA_TILE):
                r1 = min(x.shape[0], r0 + QSA_TILE)
                sc = PK.qsa_scores_rows(iq[r0:r1], pool, P["nblk"][r0:r1], cls, lean=PF_NOCOPY)
                SK.qsa_select(sc, P["tail"][r0:r1], P["sel"][r0:r1])
                del sc
        o = SK.attn_rows(q, st.k[li], st.v[li], P["ctl"], P["anc"], P["sel"], bufs=P["bufs"], stages=PF_ATTN_STAGES)
        y = K.gate_mul(o, qkv, c.head_dim)
        return self.lin(y, self.W[p + "o_proj.weight"])

    def _gdn_pre(self, x, li, st):
        """The GDN layer of a chunk: qkv|z (EXL3), a|b on the library GEMM, the conv over the chain and
        the gates in one launch (`gdn_pre`), the chunked delta rule (`delta_rule`), the gated norm,
        out_proj. The conv state keeps the chunk's last three inputs."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        from tools import fn_seq_kernels as SK
        from tools.gdn_prefill_kernels import CHUNK as GDN_CHUNK, fused_chunk_prefill, fused_prefill_refusal
        c = self.cfg
        a = self.pre(li) + "linear_attn."
        T = x.shape[0]
        C = c.conv_dim
        qkvz = self.lin(x, self.W[a + "qkvz"])
        ab = PK.mm_rows(x, self.W[a + "ab"]).to(self.dtype)
        y, g, beta = SK.gdn_pre(qkvz, st.conv[li], self._wc[li], self._pre["csrc"], ab, self._wc[("alog", li)],
                                self._wc[("dtb", li)])
        w = c.linear_conv_kernel_dim - 1
        hist = qkvz[-w:, :C].T if T >= w else torch.cat([st.conv[li][0], qkvz[:, :C].T], 1)[:, -w:]
        st.conv[li][0].copy_(hist)
        kd = c.key_dim
        q = y[:, :kd].view(1, T, -1, c.linear_key_head_dim)
        k = y[:, kd:2 * kd].view(1, T, -1, c.linear_key_head_dim)
        v = y[:, 2 * kd:].view(1, T, -1, c.linear_value_head_dim)
        H = c.linear_num_value_heads
        g, beta = g.view(1, T, H), beta.view(1, T, H)
        if T > 64 and not fused_prefill_refusal(GDN_CHUNK):
            # the key heads as they are (the kernel maps value heads to them) and value blocks of 16:
            # 10.2 vs 15.4 ms a layer at 4,096 rows (2026-10-11), the output within one bf16 step
            o, S = fused_chunk_prefill(q, k, v, g, beta, st.S[li], chunk_size=GDN_CHUNK, bv=16, view=PF_NOCOPY)
        else:
            if c.num_v_per_k > 1:
                q = q.repeat_interleave(c.num_v_per_k, dim=2)
                k = k.repeat_interleave(c.num_v_per_k, dim=2)
            o, S = self.delta_rule(q, k, v, g, beta, st.S[li])
        st.S[li].copy_(S)
        z = qkvz[:, C:]
        o = o.reshape(-1, c.linear_value_head_dim).to(self.dtype)
        if PF_NOCOPY and PK.gated_norm_ok(o, z):
            hh = PK.gated_norm_rows(o, z, self.W[a + "norm.weight"], c.rms_norm_eps, c.output_gate_type == "sigmoid")
        else:
            hh = K.gated_norm(o, z.reshape(-1, c.linear_value_head_dim),
                              self.W[a + "norm.weight"], c.rms_norm_eps, c.output_gate_type == "sigmoid")
        return self.lin(hh.view(T, -1), self.W[a + "out_proj.weight"])

    def _ple_pre(self, h, tokens, li, st):
        """PLE of a chunk: the n-gram rows hashed on the host (the reference's `ple_ids`, which advances
        the context), read from the table's page cache as the decode step reads them and decoded on the
        device, the projections on the library GEMM, then the step's kernels (gate, norms, the dilated
        conv over the chain); the conv state keeps the last 9."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        from tools import fn_seq_kernels as SK
        c = self.cfg
        p = self.pre(li) + "ple."
        H, eps = c.hidden_size, c.rms_norm_eps
        T = h.shape[0]
        tab, pd = self.tables[li], self._ple_dev()
        ids = self.ple_ids(tokens, li, st).reshape(-1).numpy()
        packed = torch.from_numpy(tab.rows_of(ids)).to(self.device)
        heads = torch.from_numpy(np.searchsorted(tab.offsets_np, ids, side="right") - 1).to(self.device)
        # the decode in tiles of PLE_DECODE_TOKENS tokens' rows: its bit streams are int32 [rows, 160 K, 16]
        # (2.2 GiB for a 4,096-token chunk, the chunk's largest allocation below 131k of context). A row decodes
        # alone (integer ops, one multiply and add an element), so the tiles' rows are the whole call's
        step = max(1, PLE_DECODE_TOKENS * (packed.shape[0] // max(1, T)))
        emb = torch.cat([decode_rows(packed[r0:r0 + step], heads[r0:r0 + step], pd["bias"], tab.K,
                                     pd["codebook"], words=PF_NOCOPY).to(self.dtype)
                         for r0 in range(0, packed.shape[0], step)]).reshape(T, -1)
        key = K.hc_norm(PK.mm_rows(emb, self.W[p + "key_proj.weight"]).to(self.dtype), self.W[p + "norm_key.weight"], H, eps)
        val = PK.mm_rows(emb, self.W[p + "value_proj.weight"]).to(self.dtype)
        qry = K.hc_norm(h, self.W[p + "norm_query.weight"], H, eps)
        gv = SK.ple_gate(key, qry, val, c.hc_count)
        gvn = K.hc_norm(gv, self.W[p + "norm_conv.weight"], H, eps)
        out = SK.ple_conv(gvn, gv, st.ple_conv[li], self._wc[("ple", li)], self._pre["psrc"])
        ps = st.ple_conv[li]
        w = ps.shape[-1]
        ps[0].copy_(gvn[-w:].T if T >= w else torch.cat([ps[0], gvn.T], 1)[:, -w:])
        return out

    # ------------------------------------------------------------------------- rows: decode and verify
    def _rows_buffers(self, M: int, key=None) -> dict:
        """The device inputs of an M-row call and their pinned host image (one H2D copy a call):

          tok, pos, slot      int64 [M]: the token, its position (block start + depth) and its cache
                              slot (block start + row; equal to the position in a chain)
          heads, packed       the PLE rows: hash head int64 [M nh], packed table row int16 [M nh, words]
          actl                int32 [M, 4]: visible keys, block start, dense flag (attn_rows)
          anc                 int32 [M, ANC]: the cache slot of the row's ancestor at each depth
          gctl                int32 [M, 2]: the GDN tile's (load, save) (gdn_rows)
          csrc, psrc          int32 [M, 3]: where the GDN conv taps (distance 3, 2, 1) and the PLE
                              conv taps (9, 6, 3) live: a row of the block or a state column
          qctl                int32 [M, 4]: QSA complete blocks, first own block (ENG-147)
          oslot, opos         int32 [M, OWN, 4], [M, OWN]: the key slots and first position of the
                              row's own blocks (those holding positions of the round)
          tail                int32 [M, 3]: the positions past the row's last complete block
          pidx                int64 [1]: the pool row a decode step writes (its own block)
        """
        key = M if key is None else key
        rb = self._rb.get(key)
        if rb is not None:
            return rb
        from tools import fn_seq_kernels as SK
        c, dev = self.cfg, self.device
        tab = next(iter(self.tables.values())) if self.tables else None
        nh = c.ngram_heads
        words = tab.words if tab is not None else 1
        n64 = M * (3 + nh) + 1
        w32 = (("actl", 4), ("anc", SK.ANC), ("gctl", 2), ("csrc", 3), ("psrc", 3), ("qctl", 4),
               ("oslot", SK.OWN * 4), ("opos", SK.OWN), ("tail", 3))
        n32 = M * sum(w for _, w in w32)
        n32 += n32 % 2                                          # the int16 rows start 8-byte aligned
        nb = 8 * n64 + 4 * n32 + 2 * M * nh * words

        def views(b):
            i64 = b[: 8 * n64].view(torch.int64)
            i32 = b[8 * n64: 8 * n64 + 4 * n32].view(torch.int32)
            v = {"tok": i64[:M], "pos": i64[M:2 * M], "slot": i64[2 * M:3 * M], "heads": i64[3 * M:n64 - 1],
                 "pidx": i64[n64 - 1:]}
            k = 0
            for name, w in w32:
                v[name] = i32[k:k + M * w].view(M, w)
                k += M * w
            v["packed"] = b[8 * n64 + 4 * n32:].view(torch.int16).view(M * nh, words)
            return v
        hbuf = torch.zeros(nb, dtype=torch.uint8, pin_memory=True)
        dbuf = torch.zeros(nb, dtype=torch.uint8, device=dev)
        copied = torch.cuda.Event()
        copied.record()
        rb = {"M": M, "hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied,
              "logits": torch.zeros(M, c.vocab_size, dtype=torch.float32, device=dev),
              "next": torch.zeros(M, dtype=torch.long, device=dev),
              "next_h": torch.zeros(M, dtype=torch.long, pin_memory=True), "attn": {}}
        rb["d"]["sel"] = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=dev)
        if c.has_indexer:
            rb["d"]["iq"] = torch.zeros(M, c.indexer_n_heads, c.indexer_head_dim, dtype=self.dtype, device=dev)
        if M == 1:                                   # a decode step: everything but the position is fixed
            h = rb["h"]
            h["gctl"].numpy()[:] = (-2, -1)
            h["csrc"].numpy()[:] = (-1, -2, -3)
            h["psrc"].numpy()[:] = (-1, -4, -7)
        self._rb[key] = rb
        return rb

    def _ple_dev(self) -> dict:
        if not hasattr(self, "_pled"):
            tab = next(iter(self.tables.values()))
            self._pled = {"bias": tab.bias.to(self.device), "codebook": tab.codebook(self.device)}
        return self._pled

    def stage(self, token: int, st: FNState) -> None:
        """Host work of one decode step (before the step or its graph runs): the token, its position,
        its PLE rows (hash with the state's context, which advances, and gather from the table)."""
        rb = self._rows_buffers(1)
        rb["copied"].synchronize()             # the image is the previous call's copy source until then
        h, L = rb["h"], st.length
        h["tok"].numpy()[0] = token
        rb["last"] = L                         # SPD-249: the row's position for `class_check`
        h["pos"].numpy()[0] = L + st.delta
        h["slot"].numpy()[0] = L
        h["anc"].numpy()[0, 0] = L
        self._stage_qsa(h, L, L // self.cfg.indexer_compress_ratio if self.cfg.has_indexer else 0,
                        np.array([L]), h["anc"].numpy(), 1, st.delta)
        for li in self.cfg.ple_layers:
            hsh = self.hash[li]
            ctx = self._ple_ctx(st, li)
            self._ple_fill(h, 0, li, hsh.ids_one(token, ctx))
            st.ple_ctx[li] = torch.tensor([(ctx + [token])[-hsh.context_len:]], dtype=torch.long)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()

    def _stage_qsa(self, h: dict, L: int, own0: int, pos: np.ndarray, anc: np.ndarray, n: int,
                   delta: int = 0) -> None:
        """The attention control of n rows at positions `pos`: positions below L are cache slots,
        from L on the row's ancestor at that depth (`anc`). Rows whose complete blocks fit the budget
        see every position (dense); the others see their selected blocks and their tail (QSA). The
        row's own blocks start at block own0 (the shared pool holds the blocks before it); each is
        pooled with the rotary row of its first slot + `delta` (the state's, VIS-97: a round's own
        blocks start at most 3 rows before it, in text after any image -- the chat template ends a
        prompt with the assistant's header)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        r = c.indexer_compress_ratio if c.has_indexer else 4
        nblk = (pos + 1) // r
        dense = (nblk <= SK.TOPK) | (not c.has_indexer)
        actl = h["actl"].numpy()
        actl[:n, 0] = np.where(dense, pos + 1, 4 * SK.TOPK + pos + 1 - r * nblk)
        actl[:n, 1] = L
        actl[:n, 2] = dense
        if not c.has_indexer:
            return
        q = (r * (own0 + np.arange(SK.OWN)))[:, None] + np.arange(r)[None, :]          # [OWN, 4] positions
        ai = anc[:n][:, np.clip(q - L, 0, anc.shape[1] - 1)]                         # [n, OWN, 4]
        slots = np.where(q[None] < L, q[None], np.where(q[None] <= pos[:, None, None], ai, 0))
        h["oslot"].numpy()[:n] = slots.reshape(n, -1)
        h["opos"].numpy()[:n] = r * (own0 + np.arange(SK.OWN)) + delta
        qc = h["qctl"].numpy()
        qc[:n, 0] = nblk
        qc[:n, 1] = own0
        h["tail"].numpy()[:n] = r * nblk[:, None] + np.arange(3)[None, :]
        h["pidx"].numpy()[0] = own0

    def qsa_class(self, last: int) -> int:
        """The context class of a call whose last position is `last`: 0 while every row is dense
        (no indexer work), else the score width in blocks (a power of two, at least 1,024)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        if not c.has_indexer or (last + 1) // c.indexer_compress_ratio <= SK.TOPK:
            return 0
        nb = 1024
        while nb * c.indexer_compress_ratio < last + 1:
            nb *= 2
        return nb

    def class_check(self, last: int, cls: int, where: str) -> None:
        """SPD-249: a launch's class may not be below the class of its last row's position. Below it the
        rows past the dense budget are marked sparse (`_stage_qsa`) while the launch selects no blocks:
        `sel` keeps what an earlier launch left and the attention reads K/V rows there (an idle capture
        of class 0 on a 2,268-token state: an illegal memory access in `attn_rows`). A class above it is
        fine (a dense row reads up to its own position, a sparse one only its complete blocks)."""
        if self.cfg.has_indexer and cls < self.qsa_class(int(last)):
            raise ClassError(f"{where}: class {cls} below {self.qsa_class(int(last))}, the class of position {int(last)}")

    @staticmethod
    def path_ple_ids(hsh, ctx0: list[int], tokens: list[int], paths: list[list[int]]) -> list[list[int]]:
        """The n-gram row ids of every row of a block: each row hashed after the tokens on its own
        path (its ancestors, then the state's context `ctx0` before the block)."""
        n = hsh.context_len
        return [hsh.ids_one(tokens[pa[-1]], (ctx0 + [tokens[a] for a in pa[:-1]])[-n:]) for pa in paths]

    def _ple_ctx(self, st: FNState, li: int) -> list[int]:
        ctx = st.ple_ctx.get(li)
        return [self.cfg.eos_token_id] * self.hash[li].context_len if ctx is None else ctx.view(-1).tolist()

    def _ple_fill(self, h: dict, row: int, li: int, ids, rows: np.ndarray | None = None) -> None:
        """The PLE rows of consecutive block rows from `row` on: `ids` their n-gram ids (a row's heads, then the
        next row's), `rows` their packed rows when read already (`_ple_batch`), else read here in one batch."""
        tab = self.tables[li]
        nh = self.cfg.ngram_heads
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        h["packed"].numpy()[row * nh:row * nh + ids.size] = tab.rows_of(ids) if rows is None else rows
        h["heads"].numpy()[row * nh:row * nh + ids.size] = np.searchsorted(tab.offsets_np, ids, side="right") - 1

    #: SPD-249: tree plans kept (`_tree_plan`); a joint round's trees repeat a few dozen shapes
    TREE_PLANS_MAX = 4096

    def _tree_plan(self, parents: list[int]) -> dict:
        """SPD-249: the host arrays `stage_rows` derives from a block's `parents` alone, computed once a shape: depth,
        paths, the ancestor cells (rows, columns, path rows), the GDN tree control and slots, the conv and PLE window
        sources. The values are the ones the per-call code computes, so the staged rows are the same bits."""
        from tools import fn_seq_kernels as SK
        key = tuple(parents)
        plans = self.__dict__.setdefault("_tplans", {})
        tp = plans.get(key)
        if tp is not None:
            return tp
        if len(plans) >= self.TREE_PLANS_MAX:
            plans.clear()
        depth, paths = self._tree_paths(parents)
        ri = np.concatenate([np.full(len(pa), i, dtype=np.int64) for i, pa in enumerate(paths)])
        ci = np.concatenate([np.arange(len(pa), dtype=np.int64) for pa in paths])
        pv = np.concatenate([np.asarray(pa, dtype=np.int64) for pa in paths])
        ctl, slots = SK.tree_ctl(parents)
        tp = plans[key] = {"depth": np.asarray(depth, dtype=np.int64), "dmax": int(max(depth)), "paths": paths,
                           "ri": ri, "ci": ci, "pv": pv, "ctl": np.asarray(ctl), "slots": slots,
                           "csrc": np.asarray(SK.window_src(parents, (3, 2, 1), 3)),
                           "psrc": np.asarray(SK.window_src(parents, (9, 6, 3), 9))}
        return tp

    @staticmethod
    def _tree_paths(parents: list[int]) -> tuple[list[int], list[list[int]]]:
        """Each row's depth and its root-to-row path (DFS pre-order, `parents`)."""
        depth, paths = [], []
        for i, p in enumerate(parents):
            depth.append(0 if p < 0 else depth[p] + 1)
            paths.append([i] if p < 0 else paths[p] + [i])
        return depth, paths

    def _ple_batch(self, items: list) -> list:
        """SPD-239: the PLE rows of several blocks [(tokens, parents, st)] read in one batch a PLE layer: per block
        {layer: (ids, packed rows)} for `stage_rows`. A cold table reads its misses from disk; one batch for every
        state's rows is one round of reads instead of one a block row (8 states x 3 rows: 29-31 ms -> one)."""
        out: list = [{} for _ in items]
        for li in self.cfg.ple_layers:
            hsh = self.hash[li]
            ids = [np.asarray(self.path_ple_ids(hsh, self._ple_ctx(st, li), tokens,
                                                self._tree_plan(parents)["paths"] if TREE_PLAN else
                                                self._tree_paths(parents)[1]),
                              dtype=np.int64).reshape(-1) for tokens, parents, st in items]
            rows = self.tables[li].rows_of(np.concatenate(ids)) if ids else None
            k = 0
            for j, x in enumerate(ids):
                out[j][li] = (x, rows[k:k + x.size])
                k += x.size
        return out

    def ple_prefetch(self, items: list) -> None:
        """SPD-249: [(tokens, parents, st)] of trees a verify will stage: their PLE rows read on the reader thread
        (`_ple_batch`, the result dropped), so the rows sit in the table's cache when the verify reads them. Values
        never come from here: the verify reads its own rows (a tree its round cuts reads a subset of these ids, a
        state that moves before the verify reads other ids and misses)."""
        if not (PLE_PREFETCH and items and self.cfg.ple_layers):
            return
        self._ple_reader().submit(self._ple_warm, items)

    def _ple_warm(self, items: list) -> None:
        t0 = time.perf_counter()
        try:
            self._ple_batch(items)
        except Exception:                                          # noqa: BLE001
            # a warm-up only: the verify reads (and raises) on its own
            self._pf_stats()["errors"] += 1
            return
        st = self._pf_stats()
        st["batches"] += 1
        st["ms"] += 1e3 * (time.perf_counter() - t0)

    def _pf_stats(self) -> dict:
        st = self.__dict__.get("_pfs")
        if st is None:
            st = self._pfs = {"batches": 0, "ms": 0.0, "errors": 0}
        return st

    def _ple_reader(self):
        """The one thread a joint pass reads its PLE rows on (`verify_multi` with QWEN38_FN_PLE_LATE)."""
        ex = getattr(self, "_plex", None)
        if ex is None:
            from concurrent.futures import ThreadPoolExecutor
            ex = self._plex = ThreadPoolExecutor(1, thread_name_prefix="fn-ple")
        return ex

    def _ple_late(self, cx: list, ples: list) -> None:
        """SPD-239: the joint pass's PLE rows, read meanwhile (`_ple_batch`), into each state's rows buffer and to the
        device in stream order before the PLE layer's kernels."""
        for x, ple in zip(cx, ples):
            rb = x["rb"]
            h, d = rb["h"], rb["d"]
            for li in self.cfg.ple_layers:
                self._ple_fill(h, 0, li, *ple[li])
            d["packed"].copy_(h["packed"], non_blocking=True)
            d["heads"].copy_(h["heads"], non_blocking=True)
            rb["copied"].record()

    def stage_rows(self, tokens: list[int], parents: list[int], st: FNState, key=None, ple: dict | None = None,
                   rb: dict | None = None) -> None:
        """Host work of a verify block: `tokens` in DFS pre-order with `parents` (-1 for the first
        row, which sits at the state's length). Nothing in `st` changes; `commit` keeps a path. `key`:
        the rows buffers' own key (a joint pass stages each state's block in its own, `verify_multi`).
        `ple`: the block's PLE ids and rows read already ({layer: (ids, rows)}, `_ple_batch`). `rb` (SPD-249): a lane's
        rows of a joint graph's buffer (`_jg_lane`): written there, the caller copies the whole buffer once."""
        from tools import fn_seq_kernels as SK
        M = len(tokens)
        hm, t0 = self.host_ms, time.perf_counter()
        lane = rb is not None
        if not lane:
            rb = self._rows_buffers(M, key)
            rb["copied"].synchronize()
        t1 = time.perf_counter()
        hm["stage_wait"] = hm.get("stage_wait", 0.0) + 1e3 * (t1 - t0)
        h, L = rb["h"], st.length
        tp = self._tree_plan(parents) if TREE_PLAN else None
        if tp is not None:
            d, paths = tp["depth"], tp["paths"]
            rb["last"] = L + tp["dmax"]
        else:
            depth, paths = self._tree_paths(parents)
            d = np.asarray(depth, dtype=np.int64)
            rb["last"] = L + int(d.max())          # SPD-249: the block's last position for `class_check`
        h["tok"].numpy()[:] = tokens
        h["pos"].numpy()[:] = L + d + st.delta
        h["slot"].numpy()[:] = L + np.arange(M)
        anc = h["anc"].numpy()
        if tp is not None:
            anc[tp["ri"], tp["ci"]] = L + tp["pv"]          # the cells the loop below writes, the same values
        else:
            for i, pa in enumerate(paths):
                anc[i, :len(pa)] = L + np.asarray(pa)
        t2 = time.perf_counter()
        self._stage_qsa(h, L, L // self.cfg.indexer_compress_ratio if self.cfg.has_indexer else 0, L + d, anc, M,
                        st.delta)
        t3 = time.perf_counter()
        if tp is not None:
            self._ctl_slots = tp["slots"]
            h["gctl"].numpy()[:] = tp["ctl"]
            h["csrc"].numpy()[:] = tp["csrc"]
            h["psrc"].numpy()[:] = tp["psrc"]
        else:
            ctl, self._ctl_slots = SK.tree_ctl(parents)
            h["gctl"].numpy()[:] = ctl
            h["csrc"].numpy()[:] = SK.window_src(parents, (3, 2, 1), 3)
            h["psrc"].numpy()[:] = SK.window_src(parents, (9, 6, 3), 9)
        t4 = time.perf_counter()
        for li in self.cfg.ple_layers:
            if ple == "late":
                break                               # filled by `_ple_late` before the first PLE layer runs
            if ple is not None:
                self._ple_fill(h, 0, li, *ple[li])
            else:
                self._ple_fill(h, 0, li, self.path_ple_ids(self.hash[li], self._ple_ctx(st, li), tokens, paths))
        t5 = time.perf_counter()
        # SPD-239: the staging's host ms by part (the copy buffer's wait, the rows, the attention control,
        # the tree control, the PLE rows)
        hm["stage_rows_ctl"] = hm.get("stage_rows_ctl", 0.0) + 1e3 * (t2 - t1)
        hm["stage_qsa"] = hm.get("stage_qsa", 0.0) + 1e3 * (t3 - t2)
        hm["stage_tree"] = hm.get("stage_tree", 0.0) + 1e3 * (t4 - t3)
        hm["stage_ple"] = hm.get("stage_ple", 0.0) + 1e3 * (t5 - t4)
        if not lane:
            rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
            rb["copied"].record()
        self._block = (list(tokens), list(parents), paths)

    def rows_step(self, st: FNState, M: int, write: bool = False, cls: int = 0) -> torch.Tensor:
        """M rows from the device buffers (`stage` / `stage_rows` first) through the trunk and the
        head. `write` (one row, a decode step): the recurrent and conv states and the pooled indexer
        keys advance in place. Otherwise nothing but the KV rows at the block's slots is written, and
        what a commit needs is kept in `self.rec`. `cls` (`qsa_class`): 0 = every row dense, else the
        indexer selects each sparse row's blocks over that many block scores. Returns logits [M, vocab]
        fp32 (a static buffer) and leaves the greedy tokens in rows["next"] and its pinned host copy.
        `st.length` is the caller's."""
        from tools import fn_kernels as K
        c = self.cfg
        rb = self._rows_buffers(M)
        self.class_check(rb.get("last", st.length + M - 1), cls, "rows_step")
        self._cur, self._write, self._cls = rb, write, cls
        self.rec = {}
        d = rb["d"]
        h = self.W["embed_tokens.weight"].index_select(0, d["tok"]).repeat(1, c.hc_count)
        n = self.trunk(h, None, st, None, step=True)
        dd = K.gemv(n, self.W["hyper_connection_mixer.down_inject"])
        x = K.mix_up(dd, n, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], c.hc_count)
        if DENSE_MAX:
            K.dense(x, self.W["lm_head.weight"], out=rb["logits"])
        else:
            from tools.exl3_linear import exl3_matmul
            exl3_matmul(x, self.W["lm_head.weight"], out=rb["logits"])
        # the greedy tokens, and their copy into pinned host memory: the host reads them after a sync
        torch.argmax(rb["logits"], dim=-1, out=rb["next"])
        rb["next_h"].copy_(rb["next"], non_blocking=True)
        if self.fold:
            self._pend_of(st)[1].zero_()           # every GDN layer applied its parked rows
        self.rec["stack"] = self.stack
        return rb["logits"]

    def step(self, st: FNState, cls: int | None = None) -> torch.Tensor:
        """One decode token from the device buffers (`stage` first); the state advances in place except
        `st.length`, which the caller moves. Returns logits [1, vocab] fp32 (a static buffer)."""
        return self.rows_step(st, 1, write=True, cls=self.qsa_class(st.length) if cls is None else cls)

    @property
    def g(self) -> dict:
        """The decode step's buffers (the one-row call's)."""
        return self._rb.get(1, {})

    @torch.no_grad()
    def verify(self, tokens: list[int], parents: list[int], st: FNState, graphs: dict | None = None) -> list[int]:
        """The target's greedy token at every row of a draft block (DFS pre-order, `parents`), eager or
        from `graphs` (M -> VerifyGraph). Each row's token is the decode step's at that position bit
        for bit. Nothing in `st` moves until `commit`."""
        M = len(tokens)
        depth = [0] * M
        for i, p in enumerate(parents):
            depth[i] = 0 if p < 0 else depth[p] + 1
        cls = self.qsa_class(st.length + max(depth))
        self.settle(st)
        self._joint.pop(id(st), None)
        gr = None
        if graphs is not None:
            gr = graphs.get((M, cls))
            if gr is None:
                gr = graphs[(M, cls)] = VerifyGraph(self, st, M, cls)
                gr.capture()
        t0 = time.perf_counter()
        self.stage_rows(tokens, parents, st)
        self.host_ms["stage_rows"] = self.host_ms.get("stage_rows", 0.0) + 1e3 * (time.perf_counter() - t0)
        self._vcls = cls
        if gr is not None:
            gr.replay()
        else:
            self.rows_step(st, M, cls=cls)
        self._pending.pop(id(st), None)
        torch.cuda.current_stream().synchronize()
        return self._rb[M]["next_h"].tolist()

    @torch.no_grad()
    def verify_multi(self, items: list) -> list:
        """ENG-241 step 3: the verify blocks of several states (sequence slots over these weights) in
        one pass: [(tokens, parents, st)] -> per state (logits [M, vocab] fp32, its greedy tokens on the
        device, its streams [M, hc H]). The input and output projections of every layer, the experts and
        the head run once over all the states' rows (their weights read once); the PLE, the norms and
        mixers, the GDN recurrence and the attention run per state on its own rows and caches, as a
        lone verify runs them. The projections and the experts give a row the bits it gets alone up to
        32 rows (the self-check's widths), so each state's tokens are its lone verify's; the caller
        keeps a pass at or under 32 rows. Parked GDN rows are applied first (`flush`): each state's
        walk then starts from its own state (SPD-229: each state's walk folds in its own parked rows, as its
        lone verify does). Eager; each state commits its block (`commit`)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        S = c.num_experts_per_tok + 1
        if JOINT_GRAPH:
            out = self._joint_graph_pass(items)
            if out is not None:
                return out
        jt = self.joint_timing
        marks = []
        t_h0 = time.perf_counter()

        def mark(name):
            # QWEN38 joint-pass timing (tools/fn_joint_check.py --timing): an event at each part's start
            if jt is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                marks.append((name, ev))
        mark("stage")
        cx = []
        late = None
        if PLE_BATCH and PLE_LATE and ROWWISE and c.ple_layers and min(c.ple_layers) > 0:
            # SPD-239: the PLE rows read while the layers before the first PLE layer run (`_ple_late`)
            late = self._ple_reader().submit(self._ple_batch, items)
            ples = ["late"] * len(items)
        else:
            ples = self._ple_batch(items) if PLE_BATCH else [None] * len(items)
        for j, (tokens, parents, st) in enumerate(items):
            M = len(tokens)
            depth = [0] * M
            for i, p in enumerate(parents):
                depth[i] = 0 if p < 0 else depth[p] + 1
            self.stage_rows(tokens, parents, st, key=("joint", j, M), ple=ples[j])
            cx.append({"st": st, "M": M, "cls": self.qsa_class(st.length + max(depth)), "block": self._block,
                       "rb": self._rows_buffers(M, ("joint", j, M)), "rec": {}, "slots": self._ctl_slots})
        if jt is not None:
            jt["host_stage"] = jt.get("host_stage", 0.0) + 1e3 * (time.perf_counter() - t_h0)
            self._t_h0 = t_h0
        cuts = [0]
        for x in cx:
            cuts.append(cuts[-1] + x["M"])
        rows = [slice(cuts[j], cuts[j + 1]) for j in range(len(cx))]

        def use(x):
            self._cur, self._write, self._cls, self.rec = x["rb"], False, x["cls"], x["rec"]
        if ROWWISE:
            return self._verify_multi_rowwise(cx, rows, use, mark, marks, jt, K, H, hc, eps, S, late)
        mark("mix")
        hs = [self.W["embed_tokens.weight"].index_select(0, x["rb"]["d"]["tok"]).repeat(1, hc) for x in cx]
        ns = [None] * len(cx)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            mark("mix")
            xs, dis = [], []
            for j, x in enumerate(cx):
                use(x)
                if li in c.ple_layers:
                    hs[j] = hs[j] + self._ple_step(hs[j], li, x["st"])
                    ns[j] = None
                if ns[j] is None:
                    ns[j] = K.hc_norm(hs[j], self._site_norm(li), H, eps)
                xa, di = self._mix(ns[j], p + "attn_hyper_connection")
                xs.append(xa)
                dis.append(di)
            X = torch.cat(xs)
            mark("proj")
            if c.is_linear(li):
                a = p + "linear_attn."
                qkvz, ab = self.lin(X, self.W[a + "qkvz"]), K.gemv_bf16(X, self.W[a + "ab"])
                mark("gdn")
                parts = []
                for j, x in enumerate(cx):
                    use(x)
                    parts.append(self._gdn1_core(qkvz[rows[j]], ab[rows[j]], li, x["st"]))
                mark("proj")
                out = self.lin(torch.cat(parts), self.W[a + "out_proj.weight"])
            else:
                a = p + "self_attn."
                qkv = self.lin(X, self.W[a + "qkv"])
                iqk = self.lin(X, self.W[a + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
                mark("attn")
                parts = []
                for j, x in enumerate(cx):
                    use(x)
                    parts.append(self._attn1_core(qkv[rows[j]], iqk[rows[j]] if iqk is not None else None, li,
                                                  x["st"]))
                mark("proj")
                out = self.lin(torch.cat(parts), self.W[a + "o_proj.weight"])
            mark("mix")
            xs, dis2 = [], []
            for j in range(len(cx)):
                hs[j], ns[j] = K.inject_norm(hs[j], out[rows[j]], dis[j], self.W[p + "attn_hyper_connection.rank"],
                                             self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
                xm, di = self._mix(ns[j], p + "mlp_hyper_connection")
                xs.append(xm)
                dis2.append(di)
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            R = self.W[p + "mlp_hyper_connection.rank"]
            mark("moe")
            yd, ids, wts = self.moe_parts(torch.cat(xs), li)
            mark("mix")
            for j, x in enumerate(cx):
                r = rows[j]
                x["rec"][("ids", li)] = ids[r]
                hs[j], ns[j] = K.combine_inject_norm(hs[j], yd[r.start * S:r.stop * S], ids[r], wts[r], dis2[j], R,
                                                     nxt, hc, eps)
        mark("head")
        heads = []
        for j, x in enumerate(cx):
            if ns[j] is None:
                ns[j] = K.hc_norm(hs[j], self._site_norm(c.num_hidden_layers), H, eps)
            dd = K.gemv(ns[j], self.W["hyper_connection_mixer.down_inject"])
            heads.append(K.mix_up(dd, ns[j], self.W["hyper_connection_mixer.input_mix_weight_up.weight"], hc))
            x["rec"]["stack"] = hs[j]
        Xh = torch.cat(heads)
        if DENSE_MAX:
            logits = K.dense(Xh, self.W["lm_head.weight"])
        else:
            from tools.exl3_linear import exl3_matmul
            logits = exl3_matmul(Xh, self.W["lm_head.weight"])
        out = []
        for j, x in enumerate(cx):
            rb = x["rb"]
            rb["logits"][:x["M"]].copy_(logits[rows[j]])
            torch.argmax(rb["logits"][:x["M"]], dim=-1, out=rb["next"][:x["M"]])
            self._joint[id(x["st"])] = (x["block"], x["rec"], x["cls"])
            out.append((rb["logits"][:x["M"]], rb["next"][:x["M"]], hs[j]))
        self._applied([x["st"] for x in cx])
        if jt is not None:
            mark("end")
            self._timed(jt, marks, cx)
        return out

    # ------------------------------------------------------------------ SPD-249: joint passes from CUDA graphs
    def _jg_state(self) -> dict:
        """The joint graphs' state: kept graphs by key (LRU order), passes seen a key, the shared pool and capture
        stream, and the counts for /health (replays, captures, eager passes by reason)."""
        g = self.__dict__.get("_jgs")
        if g is None:
            # setdefault, one C call: /health's thread and the runner's thread both get the first dict made
            g = self.__dict__.setdefault("_jgs", {
                "graphs": {}, "seen": {}, "pool": None, "stream": None, "bufs": {},
                "stats": {"replays": 0, "captures": 0, "capture_ms": 0.0, "evicted": 0, "eager": {},
                          "stage_ms": 0.0, "plan_ms": 0.0, "stage_parts_ms": {}, "stage_cpu_ms": 0.0,
                          "plan_cpu_ms": 0.0}})
        return g

    def joint_graph_stats(self) -> dict:
        """SPD-249: the joint graphs' counts (kept keys, replays, captures and their ms, eager passes by reason). /health
        calls it from its own thread while the runner reorders the kept graphs on every replay: the keys and the counts
        are copied first, each in one C call, and only the copies are walked."""
        g = self._jg_state()
        keys = list(g["graphs"])
        stats = dict(g["stats"])
        out = {**{k: dict(v) if isinstance(v, dict) else v for k, v in stats.items()},
               "kept": [[*k[:3], list(k[3])] if len(k) > 3 else list(k) for k in keys]}
        if PLE_PREFETCH:
            out["ple_prefetch"] = dict(self._pf_stats())
        return out

    def _jg_eager(self, why: str) -> None:
        e = self._jg_state()["stats"]["eager"]
        e[why] = e.get(why, 0) + 1

    def _jg_why(self, items: list, rows: int) -> str | None:
        """Why this pass runs eager instead of from a graph (None: it may run from one)."""
        from tools import fn_kernels as K
        c = self.cfg
        if not (ROWWISE and GDN_MULTI and GDN_PRE_MULTI and ATTN_MULTI and PLE_MULTI and JOINT_MOE_GROUP and PLE_BATCH):
            return "switches"
        if K.MOE_STAGE or self.joint_timing is not None or self.route_trace is not None or self._moe_record is not None:
            return "measuring"
        if len(items) > JG_LANES:
            return "lanes"
        if rows > JG_ROWS[-1]:
            return "rows"
        k0 = None
        for tokens, parents, st in items:
            depth = [0] * len(parents)
            for i, p in enumerate(parents):
                depth[i] = 0 if p < 0 else depth[p] + 1
            if self.qsa_class(st.length + max(depth)):
                return "sparse"
            for li in c.attention_layers:
                kc, vc = st.k[li], st.v[li]
                if not (kc.is_contiguous() and vc.is_contiguous()):
                    return "layout"
                ks = (kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1))
                if k0 is None:
                    k0 = ks
                elif ks != k0:
                    return "layout"
        return None

    def _jg_key(self, items: list, R: int) -> tuple:
        """A joint graph's key: (lanes, padded rows, class 0, the lanes' K/V strides). `attn_rows_multi` takes the
        strides as scalars, so a captured graph holds them: slot 0's caches and a side slot's of another cap have other
        strides, and a graph of one read the other's keys at the wrong rows (`_jg_why` checked that a pass's lanes
        agree)."""
        c = self.cfg
        kv = ()
        if c.attention_layers:
            st, li = items[0][2], c.attention_layers[0]
            kc, vc = st.k[li], st.v[li]
            kv = (kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1))
        return (len(items), R, 0, kv)

    def _jg_buffers(self, R: int) -> dict:
        """The rows buffer of an R-row joint graph (every lane's rows contiguous, `_rows_buffers`' layout) and its pad
        mask (bool [R, 1], pinned host and device)."""
        rb = self._rows_buffers(R, ("jg", R))
        if "pad_h" not in rb:
            rb["pad_h"] = torch.zeros(R, 1, dtype=torch.bool, pin_memory=True)
            rb["d"]["pad"] = torch.zeros(R, 1, dtype=torch.bool, device=self.device)
        return rb

    def _jg_lane(self, rb: dict, s: int, M: int) -> dict:
        """Lane rows s..s+M-1 of a joint rows buffer as a rows buffer of its own (views: `stage_rows` writes the lane's
        host rows there, the entry tables point the kernels at its device rows). SPD-249: kept by (s, M) on the
        buffer, so a pass makes no views it made before."""
        if TREE_PLAN:
            lanes = rb.setdefault("lanes", {})
            lv = lanes.get((s, M))
            if lv is None:
                lv = lanes[(s, M)] = self._jg_lane_new(rb, s, M)
            return lv
        return self._jg_lane_new(rb, s, M)

    def _jg_lane_new(self, rb: dict, s: int, M: int) -> dict:
        nh = self.cfg.ngram_heads

        def cut(v):
            out = {}
            for name, t in v.items():
                if name == "pad":
                    continue
                if name == "pidx":
                    out[name] = t                   # a decode step's pool row: shared, no verify reads it
                    continue
                out[name] = t[s * nh:(s + M) * nh] if name in ("heads", "packed") else t[s:s + M]
            return out
        return {"M": M, "h": cut(rb["h"]), "d": cut(rb["d"]), "attn": {}}

    def _jg_pad(self, rb: dict, n: int, R: int, ple: bool = True) -> None:
        """The pad rows n..R-1 of a joint rows buffer's host image: row 0's token and PLE rows (valid table rows), no
        visible key, position and slot 0; the pad mask set past n. `ple` False: the PLE rows later (`_jg_ple_late`)."""
        h, nh = rb["h"], self.cfg.ngram_heads
        rb["pad_h"].numpy()[:n] = False
        rb["pad_h"].numpy()[n:] = True
        if n >= R:
            return
        h["tok"].numpy()[n:] = h["tok"].numpy()[0]
        h["pos"].numpy()[n:] = 0
        h["slot"].numpy()[n:] = 0
        h["actl"].numpy()[n:] = 0
        if ple:
            self._jg_pad_ple(h, n, R)

    def _jg_pad_ple(self, h: dict, n: int, R: int) -> None:
        nh = self.cfg.ngram_heads
        h["heads"].numpy()[n * nh:] = np.resize(h["heads"].numpy()[:nh], (R - n) * nh)
        h["packed"].numpy()[n * nh:] = np.resize(h["packed"].numpy()[:nh], ((R - n) * nh, h["packed"].shape[1]))

    def _jg_ple_late(self, rb: dict, cx: list, ples: list, n: int, R: int, stats: dict) -> None:
        """SPD-249: the PLE rows read meanwhile (`_ple_batch` on the reader thread) into each lane's host rows of a
        joint graph's buffer, the pad rows' after them, and the buffer's PLE rows to the device (stream order: before
        the segment of the first PLE layer replays; the whole-buffer copy before the first segment carried stale PLE
        rows that no layer before it reads)."""
        t0 = time.perf_counter()
        for x, ple in zip(cx, ples):
            h = x["rb"]["h"]
            for li in self.cfg.ple_layers:
                self._ple_fill(h, 0, li, *ple[li])
        if n < R:
            self._jg_pad_ple(rb["h"], n, R)
        rb["d"]["packed"].copy_(rb["h"]["packed"], non_blocking=True)
        rb["d"]["heads"].copy_(rb["h"]["heads"], non_blocking=True)
        sp = stats["stage_parts_ms"]
        sp["ple_late"] = sp.get("ple_late", 0.0) + 1e3 * (time.perf_counter() - t0)

    def _jg_lanes(self, G, rows: list, R: int) -> None:
        """The attention's row-to-lane table of a graph's pass (the pad rows on the last lane: no key visible)."""
        lh = G.aent["lane_h"].numpy()
        for j, r in enumerate(rows):
            lh[r.start:r.stop] = j
        lh[rows[-1].stop:R] = len(rows) - 1

    def _ple_conv_fill(self, cx, rows, li, use, gvn, gv, y, ent, k) -> None:
        """The PLE conv's lane entries of layer li (`ent` row k) and each lane's normed rows for its commit."""
        from tools import fn_seq_kernels as SK
        hrow = ent["h"].numpy()[k]
        for j, x in enumerate(cx):
            use(x)
            r = rows[j]
            hrow[j] = SK.ple_conv_entry(gvn[r], gv[r], x["st"].ple_conv[li], self._wc[("ple", li)], self._cur["d"]["psrc"],
                                        y[r])
            self.rec[("ple", li)] = gvn[r]

    def _ple_multi_jg(self, cx, rows, h, li, use, G, jd):
        """`_ple_multi` in a joint graph: the decode, projections, norms and gate over the buffer's R rows, the conv of
        every lane in one launch (`ple_conv_multi`, each lane's state from its entry) -> [R, hc H] (pad rows: no
        value)."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        p = self.pre(li) + "ple."
        H, eps = c.hidden_size, c.rms_norm_eps
        tab, pd = self.tables[li], self._ple_dev()
        R = h.shape[0]
        emb = decode_rows(jd["packed"], jd["heads"], pd["bias"], tab.K, pd["codebook"]).to(self.dtype).reshape(R, -1)
        key = K.hc_norm(K.gemv_bf16(emb, self.W[p + "key_proj.weight"]), self.W[p + "norm_key.weight"], H, eps)
        val = K.gemv_bf16(emb, self.W[p + "value_proj.weight"])
        qry = K.hc_norm(h, self.W[p + "norm_query.weight"], H, eps)
        gv = SK.ple_gate(key, qry, val, c.hc_count)
        gvn = K.hc_norm(gv, self.W[p + "norm_conv.weight"], H, eps)
        y = torch.empty_like(gv)
        G.t[("ple", li)] = (gvn, gv, y)
        k = G.pent["k"]
        self._ple_conv_fill(cx, rows, li, use, gvn, gv, y, G.pent, k)
        dev = G.pent["d"][k, :len(cx)]
        dev.copy_(G.pent["h"][k, :len(cx)], non_blocking=True)
        G.pent["k"] += 1
        SK.ple_conv_multi(dev, gv.shape[1], R)
        return y

    def _jg_body(self, cx, rows, G, jd, use, a: int, b: int) -> None:
        """Layers a..b-1 of a joint graph's pass (one segment, captured): `_verify_multi_rowwise`'s layers over the R
        rows of the buffer `jd`, every lane-level call through the graph's entry tables, the pad rows' expert ids -1;
        its tensors kept in G.t (the streams carried to the next segment there)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        G.gent["k"], G.aent["k"], G.pent["k"] = G.kidx[a]
        if a == 0:
            G.aent["lane_dd"].copy_(G.aent["lane_h"], non_blocking=True)
            Hs = self.W["embed_tokens.weight"].index_select(0, jd["tok"]).repeat(1, hc)
            Ns = None
        else:
            Hs, Ns = G.t.pop("carry")
        for li in range(a, b):
            p = self.pre(li)
            if li in c.ple_layers:
                Hs, Ns = Hs + self._ple_multi_jg(cx, rows, Hs, li, use, G, jd), None
            if Ns is None:
                Ns = K.hc_norm(Hs, self._site_norm(li), H, eps)
            X, DI = self._mix(Ns, p + "attn_hyper_connection")
            if c.is_linear(li):
                a = p + "linear_attn."
                qkvz, ab = self.lin(X, self.W[a + "qkvz"]), K.gemv_bf16(X, self.W[a + "ab"])
                out = self.lin(self._gdn_multi(cx, rows, qkvz, ab, li, use, jg=G), self.W[a + "out_proj.weight"])
            else:
                a = p + "self_attn."
                qkv = self.lin(X, self.W[a + "qkv"])
                iqk = self.lin(X, self.W[a + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
                out = self.lin(self._attn_multi(cx, rows, qkv, iqk, li, use, jg=G), self.W[a + "o_proj.weight"])
            Hs, Ns = K.inject_norm(Hs, out, DI, self.W[p + "attn_hyper_connection.rank"],
                                   self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
            XM, DI2 = self._mix(Ns, p + "mlp_hyper_connection")
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            grp, K.MOE_GROUP = K.MOE_GROUP, True
            try:
                yd, ids, wts = self.moe_parts(XM, li, pad=jd["pad"])
            finally:
                K.MOE_GROUP = grp
            G.t[("ids", li)] = ids
            Hs, Ns = K.combine_inject_norm(Hs, yd, ids, wts, DI2, self.W[p + "mlp_hyper_connection.rank"], nxt, hc, eps)
        if b < c.num_hidden_layers:
            G.t["carry"] = (Hs, Ns)
            return
        if Ns is None:
            Ns = K.hc_norm(Hs, self._site_norm(c.num_hidden_layers), H, eps)
        dd = K.gemv(Ns, self.W["hyper_connection_mixer.down_inject"])
        Xh = K.mix_up(dd, Ns, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], hc)
        if DENSE_MAX:
            logits = K.dense(Xh, self.W["lm_head.weight"])
        else:
            from tools.exl3_linear import exl3_matmul
            logits = exl3_matmul(Xh, self.W["lm_head.weight"])
        G.t["logits"], G.t["next"], G.t["stack"] = logits, torch.argmax(logits, dim=-1), Hs

    def _jg_plan(self, G, cx, rows, use, a: int, b: int) -> None:
        """Before a segment's replay (layers a..b-1): its entry tables' host rows for this pass's lanes, over the graph's
        own tensors (the `_fill` parts the capture ran, layer by layer in the same order); each lane's commit records."""
        c = self.cfg
        kg, ka, kp = G.kidx[a]
        for li in range(a, b):
            if li in c.ple_layers:
                gvn, gv, y = G.t[("ple", li)]
                self._ple_conv_fill(cx, rows, li, use, gvn, gv, y, G.pent, kp)
                kp += 1
            if c.is_linear(li):
                qkvz, ab, o, Y, Gg, Bt = G.t[("gdn", li)]
                self._gdn_multi_fill(cx, rows, qkvz, ab, li, use, o, Y, Gg, Bt, G.gent, kg)
                kg += 1
            else:
                self._attn_multi_fill(cx, rows, li, use, G.aent, ka)
                ka += 1

    def _jg_capture(self, key: tuple, cx, rows, jd, use):
        """Capture key's graph over this pass's lanes (on the runner's thread, under the capture lock; the caller
        replays it for the pass). The lanes' parked-row buffers and the scratch exist before it."""
        js = self._jg_state()
        t0 = time.perf_counter()
        c = self.cfg
        self._scratch()
        self._ple_dev()
        for x in cx:
            for li in c.linear_layers:
                self._parked(li, x["st"])
        if js["pool"] is None:
            js["pool"] = torch.cuda.graph_pool_handle()
            js["stream"] = capture_lock.lease_stream()
        G = JointGraph(self, key)
        self._jg_lanes(G, rows, key[1])
        s = js["stream"]
        s.wait_stream(torch.cuda.current_stream())
        for (a, b), g in zip(G.segs, G.graphs):
            with capture_lock.LOCK, torch.cuda.graph(g, pool=js["pool"], stream=s):
                self._jg_body(cx, rows, G, jd, use, a, b)
        torch.cuda.current_stream().wait_stream(s)
        graphs = js["graphs"]
        while len(graphs) >= JOINT_GRAPH_MAX:
            old = graphs.pop(next(iter(graphs)))
            old.drop()
            js["stats"]["evicted"] += 1
        graphs[key] = G
        ms = 1e3 * (time.perf_counter() - t0)
        js["stats"]["captures"] += 1
        js["stats"]["capture_ms"] += ms
        print(f"[fn-jgraph] captured lanes {key[0]} rows {key[1]} class {key[2]}: {ms:.0f} ms ({len(graphs)} kept)",
              flush=True)
        return G

    def _joint_graph_pass(self, items: list):
        """`verify_multi` from a joint graph: the outputs as the eager pass leaves them, or None (run it eager): a pass
        that does not qualify (`_jg_why`), or whose key has no graph yet and may not capture now (the server lets a
        capture happen only on its round runner while no request waits: `joint_capture_ok`)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        E = len(items)
        n = sum(len(t) for t, _, _ in items)
        why = self._jg_why(items, n)
        if why is not None:
            self._jg_eager(why)
            return None
        R = next(r for r in JG_ROWS if r >= n)
        key = self._jg_key(items, R)
        js = self._jg_state()
        G = js["graphs"].get(key)
        if G is None:
            seen = js["seen"][key] = js["seen"].get(key, 0) + 1
            if seen < JOINT_GRAPH_AFTER or not getattr(self, "joint_capture_ok", False):
                self._jg_eager("no graph")
                return None
        slots = 0
        for tokens, parents, st in items:
            slots += SK.tree_ctl(parents)[1]
        if slots > self._scratch().shape[0]:
            self._jg_eager("scratch")
            return None
        t0, c0 = time.perf_counter(), time.thread_time()
        rb = self._jg_buffers(R)
        rb["copied"].synchronize()
        ta = time.perf_counter()
        # SPD-249: with a graph to replay, the PLE rows are read on the reader thread while the lanes are staged and
        # the segments before the first PLE layer run; they land in the host rows (and on the device, in stream order)
        # before that layer's segment replays (`_jg_ple_late`): the same rows
        late = None
        if G is not None and JG_PLE_LATE and c.ple_layers and min(c.ple_layers) > 0:
            late = self._ple_reader().submit(self._ple_batch, items)
            ples = ["late"] * len(items)
        else:
            ples = self._ple_batch(items)
        tb = time.perf_counter()
        cx, s0 = [], 0
        for j, (tokens, parents, st) in enumerate(items):
            M = len(tokens)
            lv = self._jg_lane(rb, s0, M)
            self.stage_rows(tokens, parents, st, ple=ples[j], rb=lv)
            cx.append({"st": st, "M": M, "cls": 0, "block": self._block, "rb": lv, "rec": {}, "slots": self._ctl_slots})
            s0 += M
        tc = time.perf_counter()
        self._jg_pad(rb, n, R, ple=late is None)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["d"]["pad"].copy_(rb["pad_h"], non_blocking=True)
        rows, s0 = [], 0
        for x in cx:
            rows.append(slice(s0, s0 + x["M"]))
            s0 += x["M"]

        def use(x):
            self._cur, self._write, self._cls, self.rec = x["rb"], False, x["cls"], x["rec"]
        t1, c1 = time.perf_counter(), time.thread_time()
        # SPD-249: the staging's host ms by part, and its thread CPU ms (wall above CPU: the thread waited, e.g. on the GIL)
        sp = js["stats"]["stage_parts_ms"]
        for name, v in (("wait", ta - t0), ("ple", tb - ta), ("lanes", tc - tb), ("pad_copy", t1 - tc)):
            sp[name] = sp.get(name, 0.0) + 1e3 * v
        js["stats"]["stage_cpu_ms"] += 1e3 * (c1 - c0)
        if G is None:
            G = self._jg_capture(key, cx, rows, rb["d"], use)
            for g in G.graphs:
                g.replay()
            t2 = t1
        else:
            js["graphs"][key] = js["graphs"].pop(key)          # most recently used last
            self._jg_lanes(G, rows, R)
            t2, c2 = 0.0, 0.0
            p0 = min(c.ple_layers) if late is not None else None
            for (a, b), g in zip(G.segs, G.graphs):
                # a segment's rows are written while the GPU runs the segments before it
                t, cpu = time.perf_counter(), time.thread_time()
                if late is not None and a <= p0 < b:
                    self._jg_ple_late(rb, cx, late.result(), n, R, js["stats"])
                    late = None
                self._jg_plan(G, cx, rows, use, a, b)
                t2 += time.perf_counter() - t
                c2 += time.thread_time() - cpu
                g.replay()
            js["stats"]["plan_cpu_ms"] += 1e3 * c2
            t2 += t1
            js["stats"]["plan_ms"] += 1e3 * (t2 - t1)
        rb["copied"].record()
        js["stats"]["stage_ms"] += 1e3 * (t1 - t0)
        js["stats"]["replays"] += 1
        self._gdn_multi_on = True
        out = []
        for j, x in enumerate(cx):
            r = rows[j]
            for li in range(c.num_hidden_layers):
                x["rec"][("ids", li)] = G.t[("ids", li)][r]
            x["rec"]["stack"] = G.t["stack"][r]
            self._joint[id(x["st"])] = (x["block"], x["rec"], 0)
            out.append((G.t["logits"][r], G.t["next"][r], G.t["stack"][r]))
        if GDN_DEFER and self.fold:
            self._applied_deferred(cx)
        else:
            self._applied([x["st"] for x in cx])
        return out

    def _timed(self, jt: dict, marks: list, cx: list) -> None:
        """A timed joint pass's parts into `jt` (GPU ms summed, its passes counted); with QWEN38_FN_JOINT_TIMING
        (SPD-239) every N passes the averages a pass (with its states and rows) are logged and `jt` starts again."""
        jt["host_issue"] = jt.get("host_issue", 0.0) + 1e3 * (time.perf_counter() - self._t_h0)
        torch.cuda.synchronize()
        for (name, a), (_, b) in zip(marks, marks[1:]):
            jt[name] = jt.get(name, 0.0) + a.elapsed_time(b)
        jt["passes"] = jt.get("passes", 0) + 1
        if JOINT_TIMING:
            jt["states"] = jt.get("states", 0) + len(cx)
            jt["rows"] = jt.get("rows", 0) + sum(x["M"] for x in cx)
            n = jt["passes"]
            if n >= JOINT_TIMING:
                parts = {k: round(v / n, 2) for k, v in jt.items() if k != "passes"}
                gpu = round(sum(v for k, v in parts.items() if not k.startswith("host_") and k not in ("states", "rows")), 2)
                hm, prev = self.host_ms, getattr(self, "_hm_prev", {})
                stg = {k: round((v - prev.get(k, 0.0)) / n, 2) for k, v in hm.items() if k.startswith("stage_")}
                self._hm_prev = dict(hm)
                print(f"[fn-joint] {n} passes, a pass: {parts}, gpu ms {gpu}, staging host ms {stg}",
                      flush=True)
                jt.clear()

    def _verify_multi_rowwise(self, cx, rows, use, mark, marks, jt, K, H, hc, eps, S, late=None):
        """SPD-215: `verify_multi`'s layers with the row-wise parts (the mixers' norms, down/up mixes, the
        injects, the MoE combine, the head's mixer) run once over every state's rows instead of a state at
        a time; the PLE, the GDN recurrence and the attention still run per state on its own rows and
        caches. Each row's bits are its own (the kernels are row-wise; tools/fn_joint_check.py compares
        every state with its lone verify)."""
        c = self.cfg
        self._joint_tables_begin()
        Hs = torch.cat([self.W["embed_tokens.weight"].index_select(0, x["rb"]["d"]["tok"]).repeat(1, hc)
                        for x in cx])
        Ns = None
        un = getattr(self, "union", None)
        um = un.begin() if un is not None else None    # SPD-249: a sampled pass's expert mask, else None
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            mark("mix")
            if li in c.ple_layers:
                if late is not None:
                    self._ple_late(cx, late.result())
                    late = None
                if PLE_MULTI:
                    Hs, Ns = Hs + self._ple_multi(cx, rows, Hs, li, use), None
                else:
                    parts = []
                    for j, x in enumerate(cx):
                        use(x)
                        parts.append(Hs[rows[j]] + self._ple_step(Hs[rows[j]], li, x["st"]))
                    Hs, Ns = torch.cat(parts), None
            if Ns is None:
                Ns = K.hc_norm(Hs, self._site_norm(li), H, eps)
            X, DI = self._mix(Ns, p + "attn_hyper_connection")
            mark("proj")
            if c.is_linear(li):
                a = p + "linear_attn."
                qkvz, ab = self.lin(X, self.W[a + "qkvz"]), K.gemv_bf16(X, self.W[a + "ab"])
                mark("gdn")
                g = self._gdn_multi(cx, rows, qkvz, ab, li, use) if GDN_MULTI else None
                if g is None:
                    parts = []
                    for j, x in enumerate(cx):
                        use(x)
                        parts.append(self._gdn1_core(qkvz[rows[j]], ab[rows[j]], li, x["st"]))
                    g = torch.cat(parts)
                mark("proj")
                out = self.lin(g, self.W[a + "out_proj.weight"])
            else:
                a = p + "self_attn."
                qkv = self.lin(X, self.W[a + "qkv"])
                iqk = self.lin(X, self.W[a + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
                mark("attn")
                y = self._attn_multi(cx, rows, qkv, iqk, li, use) if ATTN_MULTI else None
                if y is None:
                    parts = []
                    for j, x in enumerate(cx):
                        use(x)
                        parts.append(self._attn1_core(qkv[rows[j]], iqk[rows[j]] if iqk is not None else None, li,
                                                      x["st"]))
                    y = torch.cat(parts)
                mark("proj")
                out = self.lin(y, self.W[a + "o_proj.weight"])
            mark("mix")
            Hs, Ns = K.inject_norm(Hs, out, DI, self.W[p + "attn_hyper_connection.rank"],
                                   self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
            XM, DI2 = self._mix(Ns, p + "mlp_hyper_connection")
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            R = self.W[p + "mlp_hyper_connection.rank"]
            mark("moe")
            grp, K.MOE_GROUP = K.MOE_GROUP, K.MOE_GROUP or JOINT_MOE_GROUP
            try:
                yd, ids, wts = self.moe_parts(XM, li)
            finally:
                K.MOE_GROUP = grp
            if um is not None:
                un.note(um, li, ids)
            mark("mix")
            for j, x in enumerate(cx):
                x["rec"][("ids", li)] = ids[rows[j]]
            Hs, Ns = K.combine_inject_norm(Hs, yd, ids, wts, DI2, R, nxt, hc, eps)
        mark("head")
        self._gdn_multi_end()
        self._attn_multi_end()
        if um is not None:
            un.end(um, sum(x["M"] for x in cx), len(cx))
        if Ns is None:
            Ns = K.hc_norm(Hs, self._site_norm(c.num_hidden_layers), H, eps)
        dd = K.gemv(Ns, self.W["hyper_connection_mixer.down_inject"])
        Xh = K.mix_up(dd, Ns, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], hc)
        for j, x in enumerate(cx):
            x["rec"]["stack"] = Hs[rows[j]]
        if DENSE_MAX:
            logits = K.dense(Xh, self.W["lm_head.weight"])
        else:
            from tools.exl3_linear import exl3_matmul
            logits = exl3_matmul(Xh, self.W["lm_head.weight"])
        out = []
        for j, x in enumerate(cx):
            rb = x["rb"]
            rb["logits"][:x["M"]].copy_(logits[rows[j]])
            torch.argmax(rb["logits"][:x["M"]], dim=-1, out=rb["next"][:x["M"]])
            self._joint[id(x["st"])] = (x["block"], x["rec"], x["cls"])
            out.append((rb["logits"][:x["M"]], rb["next"][:x["M"]], Hs[rows[j]]))
        if GDN_DEFER and self.fold and getattr(self, "_gdn_multi_on", False):
            self._applied_deferred(cx)
        else:
            self._applied([x["st"] for x in cx])
        if jt is not None:
            mark("end")
            self._timed(jt, marks, cx)
        return out

    @torch.no_grad()
    def commit(self, st: FNState, path: list[int], graphs: dict | None = None) -> torch.Tensor:
        """Keep the rows `path` (a root-to-node path of the last verify block, row 0 first): the GDN
        states walked again over the path's rows from the round's entry state (the decode step's
        arithmetic, written in place), the conv windows and the PLE conv state taken from the path,
        the path's KV rows moved down to their positions, the PLE context advanced, the length moved.
        `graphs` ((rows, path length, moved) -> CommitGraph): from a captured graph, which reads the
        verify graph's own intermediates. Returns the path's streams [len(path), hc H] (the MTP
        layer's input; a static buffer with graphs)."""
        joint = self._joint.pop(id(st), None)
        if joint is not None:
            # a block of a joint pass (`verify_multi`): its own block and intermediates, committed
            # eagerly (a commit graph reads the lone verify graph's intermediates)
            self._block, self.rec, self._vcls = joint
            graphs = None
            if GDN_DEFER and self.fold and self.parked(st):
                self.flush(st)              # SPD-249: rows the pass left parked land first; this path parks at row 0
        tokens = self._block[0]
        self.settle(st)
        t0 = time.perf_counter()
        key = self.stage_commit(st, path)
        self.host_ms["stage_commit"] = self.host_ms.get("stage_commit", 0.0) + 1e3 * (time.perf_counter() - t0)
        if graphs is not None:
            g = graphs.get(key)
            if g is None:
                g = graphs[key] = CommitGraph(self, st, key)
                g.capture()
            g.graph.replay()
        else:
            self.commit_dev(st, key)
        for li in self.cfg.ple_layers:
            hsh = self.hash[li]
            st.ple_ctx[li] = torch.tensor([(self._ple_ctx(st, li) + [tokens[i] for i in path])[-hsh.context_len:]],
                                          dtype=torch.long)
        st.length += len(path)
        if self.fold:
            self._pending[id(st)] = st
        return self._commit_bufs(key)["stack"]

    @torch.no_grad()
    def commit_multi(self, items: list) -> list:
        """SPD-239: `commit` for the states of one joint pass, [(st, path)] -> each path's streams. Each state's
        copies -- its GDN rows parked a layer, its conv windows, its PLE window, its streams -- go in one launch for
        all of them (`fn_seq_kernels.commit_multi`: the same gathers as `commit_dev`, so the same bits); the KV
        moves of a branch, the indexer's pooled blocks and the host's context follow a state at a time. A state
        not of a joint pass, or the commit without the fold, takes `commit`."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        out: list = [None] * len(items)
        rows, done = [], []
        for j, (st, path) in enumerate(items):
            joint = self._joint.get(id(st))
            if joint is None or not self.fold:
                continue
            self._joint.pop(id(st))
            self._block, self.rec, self._vcls = joint
            self.settle(st)
            t0 = time.perf_counter()
            cb, moved = self._stage_lane(j, st, path)
            self.host_ms["stage_commit"] = self.host_ms.get("stage_commit", 0.0) + 1e3 * (time.perf_counter() - t0)
            rec, n = self.rec, len(path)
            idx = cb["d"]["idx"]
            c3, c9 = self._tail_cols(path, c.linear_conv_kernel_dim - 1), None
            for li in c.linear_layers:
                qkvz, y, gg, beta = rec[li]
                py, pg, pb = self._pend_bufs(li, st)
                # SPD-249: with GDN_DEFER the path parks after the rows a deferred pass left (the count on the device)
                off = self._pend_of(st)[1][self._lin_idx[li]:self._lin_idx[li] + 1] if GDN_DEFER else None
                rows += [SK.rows_entry(y, py, idx, n, off=off), SK.rows_entry(gg, pg, idx, n, off=off),
                         SK.rows_entry(beta, pb, idx, n, off=off), SK.window_entry(st.conv[li], qkvz, c3)]
            for li in c.ple_layers:
                c9 = c9 or self._tail_cols(path, st.ple_conv[li].shape[-1])
                rows.append(SK.window_entry(st.ple_conv[li], rec[("ple", li)], c9))
            rows.append(SK.rows_entry(rec["stack"], cb["stack"], idx, n))
            done.append((j, st, path, cb, moved, self._block[0], rec))
        if rows:
            ent = self._commit_entries(len(rows))
            ent["h"].numpy()[:len(rows)] = np.asarray(rows, dtype=np.int64)
            ent["d"][:len(rows)].copy_(ent["h"][:len(rows)], non_blocking=True)
            SK.commit_multi(ent["d"][:len(rows)])
            ent["copied"].record()
        pools = []
        for j, st, path, cb, moved, tokens, rec in done:
            d, n = cb["d"], len(path)
            if moved:
                for li in c.attention_layers:
                    src, dst = d["src"][:n], d["dst"][:n]
                    st.k[li].index_copy_(1, dst, st.k[li].index_select(1, src))
                    st.v[li].index_copy_(1, dst, st.v[li].index_select(1, src))
                    if li in st.ik:
                        st.ik[li].index_copy_(0, dst, st.ik[li].index_select(0, src))
            if c.has_indexer:
                slots, posb = cb["d32"][:SK.OWN * 4].view(SK.OWN, 4), cb["d32"][SK.OWN * 4:]
                for li in c.attention_layers:
                    kln = self.W[self.pre(li) + "self_attn.indexer.k_layernorm.weight"]
                    if POOL_MULTI:
                        pools.append(SK.qsa_pool_entry(st.ik[li], slots, posb, kln, st.pool[li], cb["dpi"]))
                        continue
                    own = SK.qsa_pool(st.ik[li], slots, posb, kln, self.rope_cos, self.rope_sin, c.rms_norm_eps)
                    st.pool[li].index_copy_(0, cb["dpi"], own)
            if GDN_DEFER:
                self._pend_of(st)[1].add_(n)            # SPD-249: after the rows still parked (0 where written back)
            else:
                self._pend_of(st)[1].fill_(n)
            self._commit_lane(st, rec, d, n)
            for li in c.ple_layers:
                hsh = self.hash[li]
                st.ple_ctx[li] = torch.tensor([(self._ple_ctx(st, li) + [tokens[i] for i in path])[-hsh.context_len:]],
                                              dtype=torch.long)
            st.length += len(path)
            self._pending[id(st)] = st
            out[j] = cb["stack"][:n]
        if pools:
            # SPD-239: every lane's pooled indexer blocks of every attention layer in one launch (after its KV moves)
            ent = self._commit_entries(len(pools), "_pent")
            ent["h"].numpy()[:len(pools)] = np.asarray(pools, dtype=np.int64)
            ent["d"][:len(pools)].copy_(ent["h"][:len(pools)], non_blocking=True)
            SK.qsa_pool_multi(ent["d"][:len(pools)], SK.OWN, self.rope_cos, self.rope_sin, c.rms_norm_eps)
            ent["copied"].record()
        for j, (st, path) in enumerate(items):
            if out[j] is None:
                out[j] = self.commit(st, path)
        return out

    def _commit_lane(self, st: FNState, rec: dict, d: dict, n: int) -> None:
        """A family's own part of a lane's commit in `commit_multi` (`commit_dev`'s past the shared copies): `rec`
        the lane's joint intermediates, `d` its device path rows / KV slots (the first n). Flash-Next: none."""
        return

    #: the most rows a joint pass's path keeps (a block's rows; `commit_multi`'s lane buffers)
    LANE_ROWS = 33

    def _stage_lane(self, j: int, st: FNState, path: list[int]) -> tuple:
        """`stage_commit` into lane j's own buffers (`commit_multi`: every lane's copies run in one launch after
        all are staged, so two lanes of one shape may not share a buffer): the path's rows, its KV source and
        destination slots and the indexer's pooled blocks, copied to the device; returns (buffers, moved)."""
        from tools import fn_seq_kernels as SK
        c, dev, R = self.cfg, self.device, self.LANE_ROWS
        lanes = self.__dict__.setdefault("_lane_bufs", {})
        cb = lanes.get(j)
        if cb is None:
            hbuf = torch.zeros(3 * R, dtype=torch.int64, pin_memory=True)
            dbuf = torch.zeros(3 * R, dtype=torch.int64, device=dev)
            cb = lanes[j] = {"hbuf": hbuf, "dbuf": dbuf, "copied": torch.cuda.Event(),
                             "h": {"idx": hbuf[:R], "src": hbuf[R:2 * R], "dst": hbuf[2 * R:]},
                             "d": {"idx": dbuf[:R], "src": dbuf[R:2 * R], "dst": dbuf[2 * R:]},
                             "h32": torch.zeros(SK.OWN * 5, dtype=torch.int32, pin_memory=True),
                             "d32": torch.zeros(SK.OWN * 5, dtype=torch.int32, device=dev),
                             "hpi": torch.zeros(SK.OWN, dtype=torch.int64, pin_memory=True),
                             "dpi": torch.zeros(SK.OWN, dtype=torch.int64, device=dev),
                             "stack": torch.zeros(R, c.hc_count * c.hidden_size, dtype=self.dtype, device=dev)}
            cb["copied"].record()
        n = len(path)
        assert n <= R, f"a joint path of {n} rows"
        cb["copied"].synchronize()
        h, L = cb["h"], st.length
        h["idx"].numpy()[:n] = path
        h["src"].numpy()[:n] = [L + r for r in path]
        h["dst"].numpy()[:n] = L + np.arange(n)
        if c.has_indexer:
            r = c.indexer_compress_ratio
            b = L // r + np.arange(SK.OWN)
            h32 = cb["h32"].numpy()
            h32[:SK.OWN * 4] = np.clip((r * b)[:, None] + np.arange(r)[None, :], 0, st.cap - 1).reshape(-1)
            h32[SK.OWN * 4:] = r * b + st.delta
            cb["hpi"].numpy()[:] = np.clip(b, 0, next(iter(st.pool.values())).shape[0] - 1)
            cb["d32"].copy_(cb["h32"], non_blocking=True)
            cb["dpi"].copy_(cb["hpi"], non_blocking=True)
        cb["dbuf"].copy_(cb["hbuf"], non_blocking=True)
        cb["copied"].record()
        return cb, any(path[k] != k for k in range(n))

    def _commit_entries(self, n: int, name: str = "_cent") -> dict:
        """`commit_multi`'s entry tables (`name`: the copies', the pools'): pinned host rows and their device copy,
        grown to fit; the host rows are rewritten only after the last copy from them is done."""
        from tools import fn_seq_kernels as SK
        ent = getattr(self, name, None)
        if ent is None or ent["h"].shape[0] < n:
            m = max(n, 1024)
            ent = {"h": torch.zeros(m, SK.COMMIT_FIELDS, dtype=torch.int64, pin_memory=True),
                   "d": torch.zeros(m, SK.COMMIT_FIELDS, dtype=torch.int64, device=self.device),
                   "copied": torch.cuda.Event()}
            setattr(self, name, ent)
            ent["copied"].record()
        ent["copied"].synchronize()
        return ent

    def _commit_bufs(self, key) -> dict:
        """A commit's device inputs (path rows, KV source and destination slots, the conv tails) and
        its pinned host image; the path's streams as the output."""
        cb = self._rb.get(("commit",) + key)
        if cb is not None:
            return cb
        from tools import fn_seq_kernels as SK
        c, dev = self.cfg, self.device
        M, n = key[0], key[1]
        w9 = (c.ple_conv_kernel_size - 1) * c.ngram_size
        sizes = (("idx", n), ("src", n), ("dst", n), ("c3", c.linear_conv_kernel_dim - 1), ("c9", w9))
        tot = sum(w for _, w in sizes)
        hbuf = torch.zeros(tot, dtype=torch.int64, pin_memory=True)
        dbuf = torch.zeros(tot, dtype=torch.int64, device=dev)

        def views(b):
            out, k = {}, 0
            for name, w in sizes:
                out[name] = b[k:k + w]
                k += w
            return out
        copied = torch.cuda.Event()
        copied.record()
        cb = {"hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied,
              "stack": torch.zeros(n, c.hc_count * c.hidden_size, dtype=self.dtype, device=dev)}
        # the pool rows a commit refreshes: the blocks from the round's first block on (int32 slots
        # and first positions, int64 pool rows), from the committed cache after the moves
        cb["h32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, pin_memory=True)
        cb["d32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, device=dev)
        cb["hpi"] = torch.zeros(SK.OWN, dtype=torch.int64, pin_memory=True)
        cb["dpi"] = torch.zeros(SK.OWN, dtype=torch.int64, device=dev)
        self._rb[("commit",) + key] = cb
        return cb

    def stage_commit(self, st: FNState, path: list[int]) -> tuple:
        from tools import fn_seq_kernels as SK
        M, n = len(self._block[0]), len(path)
        moved = any(path[j] != j for j in range(n))
        key = (M, n, moved, getattr(self, "_vcls", 0))
        cb = self._commit_bufs(key)
        cb["copied"].synchronize()
        h, L = cb["h"], st.length
        h["idx"].numpy()[:] = path
        h["src"].numpy()[:] = [L + r for r in path]
        h["dst"].numpy()[:] = L + np.arange(n)
        h["c3"].numpy()[:] = self._tail_cols(path, h["c3"].numel())
        h["c9"].numpy()[:] = self._tail_cols(path, h["c9"].numel())
        if self.cfg.has_indexer:
            r = self.cfg.indexer_compress_ratio
            b = L // r + np.arange(SK.OWN)
            h32 = cb["h32"].numpy()
            h32[:SK.OWN * 4] = np.clip((r * b)[:, None] + np.arange(r)[None, :], 0, st.cap - 1).reshape(-1)
            h32[SK.OWN * 4:] = r * b + st.delta
            cb["hpi"].numpy()[:] = np.clip(b, 0, next(iter(st.pool.values())).shape[0] - 1)
            cb["d32"].copy_(cb["h32"], non_blocking=True)
            cb["dpi"].copy_(cb["hpi"], non_blocking=True)
        cb["dbuf"].copy_(cb["hbuf"], non_blocking=True)
        cb["copied"].record()
        return key

    def commit_dev(self, st: FNState, key) -> None:
        """The device half of `commit` (capturable): every input from `_commit_bufs(key)`."""
        from tools import fn_seq_kernels as SK
        c, rec = self.cfg, self.rec
        cb = self._commit_bufs(key)
        d = cb["d"]
        idx = d["idx"]
        chain = self._chain_ctl(key[1])
        for li in c.linear_layers:
            qkvz, y, gg, beta = rec[li]
            if self.fold:
                py, pg, pb = self._pend_bufs(li, st)
                torch.index_select(y, 0, idx, out=py[:key[1]])
                torch.index_select(gg, 0, idx, out=pg[:key[1]])
                torch.index_select(beta, 0, idx, out=pb[:key[1]])
            else:
                SK.gdn_rows(y.index_select(0, idx), c.key_dim, gg.index_select(0, idx), beta.index_select(0, idx),
                            st.S[li], self._scratch(), chain, c.num_v_per_k, True)
            hist = torch.cat([st.conv[li][0], qkvz[:, :c.conv_dim].T], 1)
            st.conv[li][0].copy_(hist.index_select(1, d["c3"]))
        if key[2]:
            for li in c.attention_layers:
                st.k[li].index_copy_(1, d["dst"], st.k[li].index_select(1, d["src"]))
                st.v[li].index_copy_(1, d["dst"], st.v[li].index_select(1, d["src"]))
                if li in st.ik:
                    st.ik[li].index_copy_(0, d["dst"], st.ik[li].index_select(0, d["src"]))
        if c.has_indexer:
            # the blocks the round touched, pooled again from the committed keys (ENG-147)
            slots, posb = cb["d32"][:SK.OWN * 4].view(SK.OWN, 4), cb["d32"][SK.OWN * 4:]
            for li in c.attention_layers:
                own = SK.qsa_pool(st.ik[li], slots, posb, self.W[self.pre(li) + "self_attn.indexer.k_layernorm.weight"],
                                  self.rope_cos, self.rope_sin, c.rms_norm_eps)
                st.pool[li].index_copy_(0, cb["dpi"], own)
        for li in c.ple_layers:
            hist = torch.cat([st.ple_conv[li][0], rec[("ple", li)].T], 1)
            st.ple_conv[li][0].copy_(hist.index_select(1, d["c9"]))
        torch.index_select(rec["stack"], 0, idx, out=cb["stack"])
        if self.fold:
            self._pend_of(st)[1].fill_(key[1])

    def _pend_of(self, st: FNState) -> tuple:
        """`st`'s parked GDN rows (SPD-229): ({layer: (y [33, conv], g, beta [33, Hv])}, the rows a layer int32
        [linear layers] on the device), made on first use and kept with the state (a graph holds them)."""
        p = getattr(st, "pend", None)
        if p is None:
            p = st.pend = ({}, torch.zeros(max(len(self.cfg.linear_layers), 1), dtype=torch.int32,
                                           device=self.device))
        return p

    def _pend_bufs(self, li: int, st: FNState):
        d = self._pend_of(st)[0]
        b = d.get(li)
        if b is None:
            c = self.cfg
            b = d[li] = (torch.zeros(33, c.conv_dim, dtype=self.dtype, device=self.device),
                         torch.zeros(33, c.linear_num_value_heads, dtype=torch.float32, device=self.device),
                         torch.zeros(33, c.linear_num_value_heads, dtype=torch.float32, device=self.device))
        return b

    def _parked(self, li: int, st: FNState):
        """`st`'s parked rows of GDN layer li for its next kernel call (None without the fold)."""
        if not self.fold:
            return None
        i = self._lin_idx[li]
        return self._pend_bufs(li, st) + (self._pend_of(st)[1][i:i + 1],)

    def parked(self, st: FNState) -> bool:
        """`st` has GDN rows parked for its next call."""
        return id(st) in self._pending

    def drop_parked(self, st: FNState) -> None:
        """Forget `st`'s parked rows (its state was replaced or reset)."""
        if self._pending.pop(id(st), None) is not None:
            self._pend_of(st)[1].zero_()

    def _applied_deferred(self, cx) -> None:
        """SPD-249: after a joint pass whose GDN calls could leave parked rows parked (QWEN38_FN_GDN_DEFER): each
        state's count is zeroed where the kernel wrote its state back (more than GDN_DEFER rows parked, or no room
        after them for this block's path) and kept otherwise, the kernel's own rule on the same values. The state
        stays among the parked ones (`flush` before any other reader; a count of 0 walks nothing)."""
        for x in cx:
            cnt = self._pend_of(x["st"])[1]
            cnt.masked_fill_((cnt > GDN_DEFER) | (cnt + int(x["M"]) > PEND_ROWS), 0)
            self._pending[id(x["st"])] = x["st"]

    def _applied(self, states) -> None:
        """A call on these states walked their parked rows into their state (verify_multi): none left."""
        for st in states:
            if self._pending.pop(id(st), None) is not None and self.fold:
                self._pend_of(st)[1].zero_()

    @torch.no_grad()
    def flush(self, st: FNState | None = None) -> None:
        """Apply the parked GDN rows to their state now (before anything that reads the state other
        than a verify block or a decode step: the reference forward, a snapshot): `st`'s, or every
        state's with parked rows."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        z = self._chain_ctl(1)[:0]
        for s in ([st] if st is not None else list(self._pending.values())):
            if id(s) not in self._pending:
                continue
            for li in c.linear_layers:
                py, pg, pb = self._pend_bufs(li, s)
                SK.gdn_rows(py[:0], c.key_dim, pg[:0], pb[:0], s.S[li], self._scratch(), z, c.num_v_per_k, False,
                            out=torch.empty(0, c.linear_num_value_heads, 128, dtype=torch.bfloat16,
                                            device=self.device),
                            pre=self._parked(li, s))
            self._pend_of(s)[1].zero_()
            del self._pending[id(s)]

    def settle(self, st: FNState) -> None:
        """Before a call on `st`. SPD-229: every state keeps its own parked rows, which its next call
        folds in, so nothing of another state's lands first."""
        return

    @staticmethod
    def _tail_cols(path: list[int], w: int) -> list[int]:
        """The last `w` positions along `path` as columns of [state (w columns) | the block's rows]."""
        return (list(range(w)) + [w + r for r in path])[-w:]

    def _chain_ctl(self, n: int) -> torch.Tensor:
        if not hasattr(self, "_chains"):
            self._chains = {}
        if n not in self._chains:
            self._chains[n] = torch.tensor([[-2, -1]] + [[-1, -1]] * (n - 1), dtype=torch.int32, device=self.device)
        return self._chains[n]

    def _scratch(self) -> torch.Tensor:
        """The GDN tiles a tree block parks (one slot a branching row, at most 31); allocated once."""
        if not hasattr(self, "_stk"):
            c = self.cfg
            self._stk = torch.empty(31, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                                    dtype=torch.float32, device=self.device)
        return self._stk

    def _ple_step(self, h, li, st):
        """PLE of the rows: the gathered n-gram rows decoded, key/value projections, the gate, the norm,
        the dilated conv over the row's path (`psrc`), all row-invariant."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "ple."
        H, eps = c.hidden_size, c.rms_norm_eps
        tab, pd = self.tables[li], self._ple_dev()
        M = h.shape[0]
        emb = decode_rows(d["packed"], d["heads"], pd["bias"], tab.K, pd["codebook"]).to(self.dtype).reshape(M, -1)
        key = K.hc_norm(K.gemv_bf16(emb, self.W[p + "key_proj.weight"]), self.W[p + "norm_key.weight"], H, eps)
        val = K.gemv_bf16(emb, self.W[p + "value_proj.weight"])
        qry = K.hc_norm(h, self.W[p + "norm_query.weight"], H, eps)
        gv = SK.ple_gate(key, qry, val, c.hc_count)
        gvn = K.hc_norm(gv, self.W[p + "norm_conv.weight"], H, eps)
        out = SK.ple_conv(gvn, gv, st.ple_conv[li], self._wc[("ple", li)], d["psrc"])
        if self._write:
            ps = st.ple_conv[li]
            ps.copy_(torch.cat([ps[..., 1:], gvn.T.unsqueeze(0)], -1))
        else:
            self.rec[("ple", li)] = gvn
        return out

    def _ple_multi(self, cx, rows, h, li, use):
        """SPD-249: `_ple_step` for every lane's rows of a joint pass: the n-gram rows decoded, the projections, the norms
        and the gate once over all rows (row-wise), the dilated conv a lane at a time over its own conv state; each
        lane's normed rows recorded for its commit. Each row's arithmetic is `_ple_step`'s -> [M, hc H]."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        p = self.pre(li) + "ple."
        H, eps = c.hidden_size, c.rms_norm_eps
        tab, pd = self.tables[li], self._ple_dev()
        ds = [x["rb"]["d"] for x in cx]
        packed = torch.cat([d["packed"] for d in ds])
        heads = torch.cat([d["heads"] for d in ds])
        M = h.shape[0]
        emb = decode_rows(packed, heads, pd["bias"], tab.K, pd["codebook"]).to(self.dtype).reshape(M, -1)
        key = K.hc_norm(K.gemv_bf16(emb, self.W[p + "key_proj.weight"]), self.W[p + "norm_key.weight"], H, eps)
        val = K.gemv_bf16(emb, self.W[p + "value_proj.weight"])
        qry = K.hc_norm(h, self.W[p + "norm_query.weight"], H, eps)
        gv = SK.ple_gate(key, qry, val, c.hc_count)
        gvn = K.hc_norm(gv, self.W[p + "norm_conv.weight"], H, eps)
        outs = []
        for j, x in enumerate(cx):
            use(x)
            r = rows[j]
            outs.append(SK.ple_conv(gvn[r], gv[r], x["st"].ple_conv[li], self._wc[("ple", li)], self._cur["d"]["psrc"]))
            self.rec[("ple", li)] = gvn[r]
        return torch.cat(outs)

    def _gdn_multi(self, cx, rows, qkvz, ab, li, use, jg=None):
        """SPD-239: a joint pass's GDN layer li past its projections for every state: each state's conv and gates
        (`gdn_pre`, its rows recorded for its commit as `_gdn1_core` records them), the delta rule of every state in
        one launch (`fn_seq_kernels.gdn_rows_multi`: each state's own call, its tree's scratch slots its own), the
        gated norm once over every row (row-wise). None (the state-at-a-time path) when the trees' slots exceed
        the scratch. `jg` (SPD-249): a joint graph's capture: its own entry tables, its tensors kept for replays."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        stk = self._scratch()
        if sum(x["slots"] for x in cx) > stk.shape[0]:
            self._gdn_multi_on = False
            return None
        self._gdn_multi_on = True
        R, H = qkvz.shape[0], c.linear_num_value_heads
        o = torch.empty(R, H, c.linear_value_head_dim, dtype=torch.bfloat16, device=self.device)
        pre_multi = GDN_PRE_MULTI and not self._write
        Y = Gg = Bt = None
        if pre_multi:
            # SPD-249: every state's conv and gates in one launch (`gdn_pre_multi`), each row's own arithmetic
            C = self._wc[li].shape[0]
            Y = torch.empty(R, C, dtype=torch.bfloat16, device=self.device)
            Gg = torch.empty(R, H, dtype=torch.float32, device=self.device)
            Bt = torch.empty(R, H, dtype=torch.float32, device=self.device)
        ent = self._gdn_entries(len(cx)) if jg is None else jg.gent
        k = ent["k"]
        self._gdn_multi_fill(cx, rows, qkvz, ab, li, use, o, Y, Gg, Bt, ent, k)
        if jg is not None:
            jg.t[("gdn", li)] = (qkvz, ab, o, Y, Gg, Bt)
        if pre_multi:
            pdev = ent["dp"][k, :len(cx)]
            pdev.copy_(ent["hp"][k, :len(cx)], non_blocking=True)
            SK.gdn_pre_multi(pdev, self._wc[li], self._wc[("alog", li)], self._wc[("dtb", li)],
                             R if jg is not None else max(x["M"] for x in cx))
        dev = ent["d"][k, :len(cx)]
        dev.copy_(ent["h"][k, :len(cx)], non_blocking=True)
        ent["k"] += 1
        SK.gdn_rows_multi(dev, c.key_dim, H, c.num_v_per_k)
        p = self.pre(li) + "linear_attn."
        hh = K.gated_norm(o.view(-1, c.linear_value_head_dim), qkvz[:, c.conv_dim:].reshape(-1, c.linear_value_head_dim),
                          self.W[p + "norm.weight"], c.rms_norm_eps, c.output_gate_type == "sigmoid")
        return hh.view(R, -1)

    def _gdn_multi_fill(self, cx, rows, qkvz, ab, li, use, o, Y, Gg, Bt, ent, k) -> None:
        """`_gdn_multi`'s host part: each state's entry rows of layer li (`ent` row k; Y None: its `gdn_pre` launched
        here, a state at a time) and its rows recorded for its commit. A joint graph's replay runs only this, over
        the graph's own tensors (`_jg_plan`)."""
        from tools import fn_seq_kernels as SK
        stk = self._scratch()
        hrow, slot = ent["h"].numpy()[k], 0
        prow = ent["hp"].numpy()[k] if Y is not None else None
        for j, x in enumerate(cx):
            use(x)
            st, d = x["st"], self._cur["d"]
            q = qkvz[rows[j]]
            if Y is not None:
                y, gg, beta = Y[rows[j]], Gg[rows[j]], Bt[rows[j]]
                prow[j] = SK.gdn_pre_entry(q, st.conv[li], d["csrc"], y, ab[rows[j]], gg, beta)
            else:
                y, gg, beta = SK.gdn_pre(q, st.conv[li], self._wc[li], d["csrc"], ab[rows[j]], self._wc[("alog", li)],
                                         self._wc[("dtb", li)], shift=self._write)
            self.rec[li] = (q, y, gg, beta)
            hrow[j] = SK.gdn_rows_entry(y, gg, beta, st.S[li], stk[slot:slot + max(x["slots"], 1)], d["gctl"],
                                        o[rows[j]], self._write, pre=self._parked(li, st),
                                        limit=GDN_DEFER if (self.fold and not self._write) else 0)
            slot += x["slots"]

    def _attn_multi(self, cx, rows, qkv, iqk, li, use, jg=None):
        """SPD-249: a joint pass's attention layer li past its projections for every lane at once: one `attn_prep_multi`
        (each lane's caches, positions and slots from its entry), a sparse lane's indexer selection on its own as
        `_attn1_core` runs it, one `attn_rows_multi` and one gate over every row. Each row's arithmetic is its lone
        verify's. None (the lane-at-a-time path) when the lanes' caches differ in layout."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        ks = [x["st"].k[li] for x in cx]
        vs = [x["st"].v[li] for x in cx]
        strides = (ks[0].stride(0), ks[0].stride(1), vs[0].stride(0), vs[0].stride(1))
        if any((k.stride(0), k.stride(1), v.stride(0), v.stride(1)) != strides or not k.is_contiguous()
               or not v.is_contiguous() for k, v in zip(ks, vs)):
            return None
        ent = self._attn_entries(len(cx), sum(x["M"] for x in cx), rows) if jg is None else jg.aent
        k = ent["k"]
        p = self.pre(li) + "self_attn."
        ip = p + "indexer."
        ihd = c.indexer_head_dim if c.has_indexer else 0
        nhi, sparse = self._attn_multi_fill(cx, rows, li, use, ent, k)
        n = len(cx)
        apd, ard = ent["ap_d"][k, :n], ent["ar_d"][k, :n]
        apd.copy_(ent["ap_h"][k, :n], non_blocking=True)
        ard.copy_(ent["ar_h"][k, :n], non_blocking=True)
        ent["k"] += 1
        q = K.attn_prep_multi(qkv, iqk, apd, self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"], self.rope_cos,
                              self.rope_sin, Hq, Hk, c.indexer_n_heads * c.indexer_head_dim if c.has_indexer else 0,
                              ihd if any(x["st"].ik.get(li) is not None for x in cx) else 0, nhi, c.rms_norm_eps,
                              qln=self.W[ip + "q_layernorm.weight"] if nhi else None)
        for j in sparse:
            # the lane's own blocks pooled and its blocks selected (`_attn1_core`), after the prep wrote its keys
            x = cx[j]
            use(x)
            st, d, cls = x["st"], self._cur["d"], self._cls
            M = x["M"]
            own = SK.qsa_pool(st.ik[li], d["oslot"].view(-1, 4), d["opos"].view(-1), self.W[ip + "k_layernorm.weight"],
                              self.rope_cos, self.rope_sin, c.rms_norm_eps)
            sc = self._cur.get(("sc", cls))
            if sc is None:
                sc = self._cur[("sc", cls)] = torch.empty(M, cls, dtype=torch.float32, device=qkv.device)
            SK.qsa_scores(d["iq"], st.pool[li], own.view(M, SK.OWN, -1), d["qctl"], cls, out=sc)
            SK.qsa_select(sc, d["tail"], d["sel"])
        o = SK.attn_rows_multi(q, ard, ent["lane_d"], strides, Hk, bufs=ent.setdefault(("attn", q.shape[0]), {}))
        return K.gate_mul(o, qkv, D)

    def _attn_multi_fill(self, cx, rows, li, use, ent, k) -> tuple:
        """`_attn_multi`'s host part: each lane's prep and rows entries of layer li (`ent` row k) -> (the most indexer
        heads of a sparse lane, the sparse lanes). A joint graph's replay runs only this (`_jg_plan`)."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        ap, ar = ent["ap_h"].numpy()[k], ent["ar_h"].numpy()[k]
        nhi, sparse = 0, []
        for j, x in enumerate(cx):
            use(x)
            st, d = x["st"], self._cur["d"]
            cls = self._cls if c.has_indexer else 0
            iq = d["iq"] if cls else None
            if cls:
                sparse.append(j)
                nhi = max(nhi, iq.shape[1])
            ap[j] = K.attn_prep_entry(rows[j].start, x["M"], st.k[li], st.v[li], st.ik.get(li), d["pos"], d["slot"], iq)
            ar[j] = SK.attn_rows_entry(rows[j].start, st.k[li], st.v[li], d["actl"], d["anc"], d["sel"])
        return nhi, sparse

    def _attn_entries(self, n: int, M: int, rows) -> dict:
        """`_attn_multi`'s entry tables: pinned host rows and their device copies for every attention layer of a pass
        (`k` the next), and the joint rows' lanes (int32, written once a pass); rewritten only after the last pass's
        copies are done (`_attn_multi_end`)."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        ent = getattr(self, "_aent", None)
        L = sum(1 for li in range(self.cfg.num_hidden_layers) if not self.cfg.is_linear(li))
        if ent is None or ent["ap_h"].shape[1] < n or ent["lane_h"].numel() < M:
            m, rmax = max(n, 8), max(M, 32)
            ent = self._aent = {
                "ap_h": torch.zeros(L, m, K.APM_FIELDS, dtype=torch.int64, pin_memory=True),
                "ap_d": torch.zeros(L, m, K.APM_FIELDS, dtype=torch.int64, device=self.device),
                "ar_h": torch.zeros(L, m, SK.ARM_FIELDS, dtype=torch.int64, pin_memory=True),
                "ar_d": torch.zeros(L, m, SK.ARM_FIELDS, dtype=torch.int64, device=self.device),
                "lane_h": torch.zeros(rmax, dtype=torch.int32, pin_memory=True),
                "lane_dd": torch.zeros(rmax, dtype=torch.int32, device=self.device),
                "copied": torch.cuda.Event(), "k": L}
            ent["copied"].record()
        if ent["k"] >= L:                   # a new pass: its first attention layer; the lanes of its rows
            ent["copied"].synchronize()
            ent["k"] = 0
            lh = ent["lane_h"].numpy()
            for j, r in enumerate(rows):
                lh[r.start:r.stop] = j
            ent["lane_dd"][:M].copy_(ent["lane_h"][:M], non_blocking=True)
        ent["lane_d"] = ent["lane_dd"][:M]
        return ent

    def _joint_tables_begin(self) -> None:
        """An eager joint pass starts on clean entry tables. Only a pass that ended puts `k` back at the tables' end;
        one that raised after its first attention or GDN layer (an allocation, the PLE reader, an entry's check) left
        it short, and the next pass then kept that pass's row-to-lane map for its first attention layers: its rows
        attended over another lane's caches. Such tables are closed here as `_attn_multi_end` / `_gdn_multi_end`
        close them (the copies the failed pass queued recorded, the next pass waits for them before it rewrites the
        host rows); after a pass that ended nothing is recorded or waited for."""
        La = sum(1 for li in range(self.cfg.num_hidden_layers) if not self.cfg.is_linear(li))
        for name, n in (("_aent", La), ("_gent", len(self.cfg.linear_layers))):
            ent = self.__dict__.get(name)
            if ent is not None and ent["k"] < n:
                ent["copied"].record()
                ent["k"] = n

    def _attn_multi_end(self) -> None:
        """After a joint pass's last layer: the attention entry tables' copies are recorded, the next pass starts over."""
        ent = getattr(self, "_aent", None)
        if ent is not None and ent["k"]:
            ent["copied"].record()
            ent["k"] = sum(1 for li in range(self.cfg.num_hidden_layers) if not self.cfg.is_linear(li))

    def _gdn_multi_end(self) -> None:
        """After a joint pass's last GDN layer: its entry tables' copies are recorded, the next pass starts over."""
        ent = getattr(self, "_gent", None)
        if ent is not None:
            ent["copied"].record()
            ent["k"] = len(self.cfg.linear_layers)

    def _gdn_entries(self, n: int) -> dict:
        """`_gdn_multi`'s entry tables: a pinned host row and its device copy for every GDN layer of a pass (`k`
        the next), the host rows rewritten only after the last pass's copies are done."""
        from tools import fn_seq_kernels as SK
        ent = getattr(self, "_gent", None)
        L = len(self.cfg.linear_layers)
        if ent is None or ent["h"].shape[1] < n:
            m = max(n, 8)
            ent = self._gent = {"h": torch.zeros(L, m, SK.COMMIT_FIELDS, dtype=torch.int64, pin_memory=True),
                                "d": torch.zeros(L, m, SK.COMMIT_FIELDS, dtype=torch.int64, device=self.device),
                                "hp": torch.zeros(L, m, SK.COMMIT_FIELDS, dtype=torch.int64, pin_memory=True),
                                "dp": torch.zeros(L, m, SK.COMMIT_FIELDS, dtype=torch.int64, device=self.device),
                                "copied": torch.cuda.Event(), "k": L}
            ent["copied"].record()
        if ent["k"] >= L:                   # a new pass: its first layer
            ent["copied"].synchronize()
            ent["k"] = 0
        return ent

    def _gdn1(self, x, li, st):
        """The GDN layer of the rows: qkv|z and a|b, the conv over each row's window and the gates
        (`csrc`), the delta rule over the block's tree (`gctl`), the gated norm, out_proj."""
        from tools import fn_kernels as K
        p = self.pre(li) + "linear_attn."
        qkvz = self.lin(x, self.W[p + "qkvz"])
        ab = K.gemv_bf16(x, self.W[p + "ab"])
        return self.lin(self._gdn1_core(qkvz, ab, li, st), self.W[p + "out_proj.weight"])

    def _gdn1_core(self, qkvz, ab, li, st):
        """`_gdn1` past its input projections and before out_proj: the rows of one state (a joint
        pass runs the projections over every state's rows at once, `verify_multi`)."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "linear_attn."
        M = qkvz.shape[0]
        y, gg, beta = SK.gdn_pre(qkvz, st.conv[li], self._wc[li], d["csrc"], ab, self._wc[("alog", li)],
                                 self._wc[("dtb", li)], shift=self._write)
        o = SK.gdn_rows(y, c.key_dim, gg, beta, st.S[li], self._scratch(), d["gctl"], c.num_v_per_k, self._write,
                        pre=self._parked(li, st))
        if not self._write:
            self.rec[li] = (qkvz, y, gg, beta)
        z = qkvz[:, c.conv_dim:]
        hh = K.gated_norm(o.view(-1, c.linear_value_head_dim), z.reshape(-1, c.linear_value_head_dim),
                          self.W[p + "norm.weight"], c.rms_norm_eps, c.output_gate_type == "sigmoid")
        return hh.view(M, -1)

    def _attn1(self, x, li, st):
        """Attention of the rows: q|k|v and the indexer projection, one prep launch (norms, rope at
        each row's position, cache writes at each row's slot), attention over each row's visible keys
        (`actl`, `anc`), the output gate, o_proj."""
        c = self.cfg
        p = self.pre(li) + "self_attn."
        qkv = self.lin(x, self.W[p + "qkv"])
        iqk = self.lin(x, self.W[p + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
        if self._mtp_pg is not None:
            # SPD-249 (T1): a packed draft graph's step: every lane's row through the graph's entry tables
            pg = self._mtp_pg
            o = self._attn_multi(pg["cx"], pg["G"].rows, qkv, iqk, li, self._pack_use, jg=pg["G"])
            assert o is not None, "a packed draft graph over caches of different layouts"
            return self.lin(o, self.W[p + "o_proj.weight"])
        if self._multi is not None:
            # SPD-235: a packed MTP step: the projections once over every lane's rows, the core per lane
            parts, r0 = [], 0
            for rb, st_i, m, cls in self._multi:
                self._cur, self._cls = rb, cls
                parts.append(self._attn1_core(qkv[r0:r0 + m], iqk[r0:r0 + m] if iqk is not None else None, li, st_i))
                r0 += m
            return self.lin(torch.cat(parts), self.W[p + "o_proj.weight"])
        return self.lin(self._attn1_core(qkv, iqk, li, st), self.W[p + "o_proj.weight"])

    def _attn1_core(self, qkv, iqk, li, st):
        """`_attn1` past its input projections and before o_proj: the rows of one state."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "self_attn."
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        M = qkv.shape[0]
        cls = self._cls if c.has_indexer else 0
        ip = p + "indexer."
        q = K.attn_prep(qkv, iqk, d["pos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        self.rope_cos, self.rope_sin, st.k[li], st.v[li], st.ik.get(li), Hq, Hk,
                        c.indexer_n_heads * c.indexer_head_dim if c.has_indexer else 0, c.rms_norm_eps,
                        slot=d["slot"], qln=self.W[ip + "q_layernorm.weight"] if cls else None,
                        iq=d["iq"] if cls else None)
        if c.has_indexer and (cls or self._write):
            # the pooled keys of the row's own blocks (from its own path); a decode step also keeps
            # its block in the shared pool (garbage until the block is complete, then its value)
            own = SK.qsa_pool(st.ik[li], d["oslot"].view(-1, 4), d["opos"].view(-1), self.W[ip + "k_layernorm.weight"],
                              self.rope_cos, self.rope_sin, c.rms_norm_eps)
            if self._write:
                st.pool[li].index_copy_(0, d["pidx"], own[:1])
            if cls:
                sc = self._cur.get(("sc", cls))
                if sc is None:
                    sc = self._cur[("sc", cls)] = torch.empty(M, cls, dtype=torch.float32, device=qkv.device)
                SK.qsa_scores(d["iq"], st.pool[li], own.view(M, SK.OWN, -1), d["qctl"], cls, out=sc)
                SK.qsa_select(sc, d["tail"], d["sel"])
        o = SK.attn_rows(q, st.k[li], st.v[li], d["actl"], d["anc"], d["sel"], bufs=self._cur["attn"])
        return K.gate_mul(o, qkv, D)

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

    def new_mtp_state(self, cap: int | None = None, pages=None) -> FNState:
        """The MTP layer's cache (KV and indexer keys of its one attention layer), allocated up front;
        with `pages` (the trunk state's paged set) its rows are paged with the trunk's: the head never
        holds a row past the trunk's."""
        c, cap = self.cfg, cap or self.max_len
        st = FNState(cap=cap, pages=pages)
        st.k[MTP] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, "k.mtp")
        st.v[MTP] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, "v.mtp")
        if c.has_indexer:
            st.ik[MTP] = self._rows_tensor(st, (cap, c.indexer_head_dim), 0, "ik.mtp")
            st.pool[MTP] = self._pool_tensor(cap)
        return st

    def mtp_rows(self, st_m: FNState, key, stack_in: torch.Tensor, cls: int = 0):
        """The MTP layer over the rows of buffers `key` (tokens, positions, slots, ancestors staged by
        `stage_mtp`), from the streams `stack_in` [M, hc H]: the reference's input fusion (rms of the
        token's embedding -> fc_embedding, rms of all hc H streams -> fc_hidden per stream, added),
        its attention layer through the trunk's row kernels on the MTP cache, the MoE (bank -1), its
        own final mixer. Returns (x [M, H], the layer's streams [M, hc H]); the head is the caller's."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        M = stack_in.shape[0]
        tok = self._mtp_begin(M, key, cls)
        e = self.rms(self.W["embed_tokens.weight"].index_select(0, tok).to(self.dtype),
                     self.W["mtp.pre_fc_norm_embedding.weight"], eps)
        e = self.lin(e, self.W["mtp.fc_embedding.weight"])
        hs = self.rms(stack_in, self.W["mtp.pre_fc_norm_hidden.weight"], eps)
        hs = self.lin(hs.reshape(M * hc, H), self.W["mtp.fc_hidden.weight"]).view(M, hc, H)
        h = (hs + e.unsqueeze(1)).flatten(1)
        p = self.pre(MTP)
        n = K.hc_norm(h, self.W[p + "attn_hyper_connection.hc_norm.weight"], H, eps)
        x, di = self._mix(n, p + "attn_hyper_connection")
        out = self._attn1(x, MTP, st_m)
        h, n = K.inject_norm(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                             self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
        x, di = self._mix(n, p + "mlp_hyper_connection")
        yd, ids, wts = self.moe_parts(x, MTP)
        h, n = K.combine_inject_norm(h, yd, ids, wts, di, self.W[p + "mlp_hyper_connection.rank"],
                                     self.W["mtp.hyper_connection_mixer.hc_norm.weight"], hc, eps)
        dd = K.gemv(n, self.W["mtp.hyper_connection_mixer.down_inject"])
        x = K.mix_up(dd, n, self.W["mtp.hyper_connection_mixer.input_mix_weight_up.weight"], hc)
        return x, h

    def _mtp_begin(self, M: int, key, cls: int) -> torch.Tensor:
        """The rows' tokens for an MTP step: the staged buffers of `key` (the attention reads them), or under
        a packed step (`_multi`, SPD-235) every lane's tokens in lane order. The trunk's records (`rec`) are
        left alone: they are the last verify block's until its commit, and the MTP layer writes none. The
        round runner issues the next draft before the commit (QSE_RUNNER_PIPE), so an MTP graph captured
        there (its warm-up runs this step) emptied them and the commit graph captured after it failed on
        its first layer (SRV-713: a raw 4,024-token request, HTTP 500, KeyError: 0 in `commit_dev`)."""
        self._write = False
        if self._mtp_pg is not None:
            return self._mtp_pg["tok"]          # SPD-249 (T1): every lane's pick of the step before, in lane order
        if self._multi is not None:
            return torch.cat([rb["d"]["tok"] for rb, _, _, _ in self._multi])
        rb = self._rows_buffers(M, key)
        self._cur, self._cls = rb, cls
        return rb["d"]["tok"]

    def _mtp_pick_multi(self, x: torch.Tensor, j: int, kp: tuple | None = None) -> tuple:
        """A packed draft step's head over every lane's row x [lanes, H] at depth j: (prob [lanes, 2] fp32, tok
        [lanes, 2]) of each lane's top-2 under the draft head, then the coupled pick of the keyed lanes where `kp` =
        (keyed head, its ids, keys [lanes, 32], parameters [lanes, NPAR], the keyed head is its own, every lane keyed);
        the eager pass and the draft graph (T1) run this one sequence."""
        from tools import fn_kernels as K
        c = self.cfg
        if kp is not None and kp[5]:
            head, hids = kp[0], kp[1]
        else:
            head, hids = self.draft_head, self.draft_ids
        if head is not None:
            lg = K.gemv(x, head)
        else:
            lg = K.dense(x, self.W["lm_head.weight"]).float()
        vals, ids = torch.topk(lg, 2, dim=-1)
        z = torch.exp(lg - vals[:, :1]).sum(-1, keepdim=True)
        prob = torch.exp(vals - vals[:, :1]) / z
        tok = hids.index_select(0, ids.view(-1)).view(ids.shape) if head is not None else ids
        if kp is not None:
            from engine.models.flashnext import coupled as CP
            kh, kids, KEYS, PAR, chead, all_k = kp
            if chead and not all_k:
                # lanes of both kinds: the keyed lanes' pick over their head's own rows
                lgc = K.gemv(x, kh)
                zc = torch.exp(lgc - lgc.amax(-1, keepdim=True)).sum(-1, keepdim=True)
                CP.pick(lgc, kids, KEYS[:, j], PAR, zc, tok, prob, c.vocab_size, x=x, depth=j)
            else:
                CP.pick(lg, hids if head is not None else None, KEYS[:, j], PAR, z, tok, prob, c.vocab_size,
                        x=x, depth=j)
        return prob, tok

    @torch.no_grad()
    def mtp_chain_multi(self, items: list, keyed: list | None = None):
        """SPD-235: `mtp_chain_dev` of several heads over these weights at once: [(head key, MTP state, n fed
        rows, depth, class)], each staged (`stage_mtp`, its streams in its `_mtp_bufs()["stack"]`). Every step
        runs the MTP layer once over all lanes' rows (the attention on each lane's own cache), the draft head
        once over each lane's last row; each lane's top-2 tokens and probabilities land in its own buffers.
        SPD-249: `keyed` (a flag a lane, staged by `stage_mtp(keyed=...)`): those lanes' pairs are the coupled
        pick's (engine/models/flashnext/coupled.py), over the keyed head of this many lanes (`keyed_head`: one lane
        the lone head, more the coupled head) where it has its own; with every lane keyed only that head is read,
        else the others keep the served head's argmax pairs bit for bit.
        SPD-249 (T1, QWEN38_FN_PACK_GRAPH): the steps after the feed from the lane count's draft graph; then the
        picks land in that graph's pinned copy instead and the call returns its `PackOut` (None: the lanes' own
        buffers hold them)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        bs = []
        for key, st_m, n, depth, cls in items:
            self._mtp_key = key
            bs.append(self._mtp_bufs(depth))
        depth = items[0][3]
        kk = keyed is not None and any(keyed)
        kp = None
        if kk:
            kh, kids = self.keyed_head(len(items))
            kp = (kh, kids, torch.stack([b["ckeys"] for b in bs]),         # [lanes, 32]
                  torch.stack([b["cpar"] for b in bs]),                    # [lanes, NPAR]: `on` 0 for a lane not keyed
                  kh is not self.draft_head,                               # the keyed lanes read a head of their own
                  all(keyed))
        cuts = np.cumsum([it[2] for it in items])
        last = torch.as_tensor(cuts - 1, dtype=torch.long, device=self.device)
        self._multi = [(self._rows_buffers(n, ("mtp", key, 0, n)), st_m, n, cls) for key, st_m, n, _, cls in items]
        try:
            x, h = self.mtp_rows(None, None, torch.cat([b["stack"][:it[2]] for b, it in zip(bs, items)]))
        finally:
            self._multi = None
        if c.has_indexer:                     # the blocks each feed completed, into its MTP pool
            ip = self.pre(MTP) + "self_attn.indexer."
            for (_, st_m, _, _, _), b in zip(items, bs):
                own = SK.qsa_pool(st_m.ik[MTP], b["d32"][:SK.OWN * 4].view(SK.OWN, 4), b["d32"][SK.OWN * 4:],
                                  self.W[ip + "k_layernorm.weight"], self.rope_cos, self.rope_sin, c.rms_norm_eps)
                st_m.pool[MTP].index_copy_(0, b["dpi"], own)
        if PACK_GRAPH:
            out = self._pack_graph_run(items, last, x, h, kp)
            if out is not None:
                return out
        x, h = x.index_select(0, last), h.index_select(0, last)
        for j in range(depth):
            prob, tok = self._mtp_pick_multi(x, j, kp)
            for i, b in enumerate(bs):
                b["prob"][j].copy_(prob[i])
                b["tok"][j].copy_(tok[i])
            if j + 1 == depth:
                break
            self._multi = []
            for i, (key, st_m, _, _, cls) in enumerate(items):
                rb = self._rows_buffers(1, ("mtp", key, j + 1, 1))
                rb["d"]["tok"].copy_(tok[i, :1])
                self._multi.append((rb, st_m, 1, cls))
            try:
                x, h = self.mtp_rows(None, None, h)
            finally:
                self._multi = None
        for b in bs:
            b["tok_h"].copy_(b["tok"], non_blocking=True)
            b["prob_h"].copy_(b["prob"], non_blocking=True)
        return None

    # ------------------------------------------------------------------------- SPD-249 (T1): the packed draft graph
    def _pg_state(self) -> dict:
        ps = self.__dict__.get("_pgs")
        if ps is None:
            ps = self._pgs = {"graphs": {}, "seen": {}, "stream": None, "eager": {},
                              "stats": {"captures": 0, "capture_ms": 0.0, "replays": 0, "refills": 0, "flushes": 0}}
        return ps

    def pack_graph_stats(self) -> dict:
        """SPD-249 (T1): the packed draft graphs for /health: captures, replays, entry refills (a new set or order of
        lanes on a kept graph), the eager drafts by reason, the kept keys (lanes, depth), late reads."""
        ps = self._pg_state()
        return {**ps["stats"], "eager": dict(ps["eager"]), "kept": [list(k[:2]) for k in ps["graphs"]],
                "stale": sum(g.stale for g in ps["graphs"].values())}

    def _pack_use(self, x: dict) -> None:
        self._cur, self._cls = x["rb"], 0

    def _pack_why(self, items: list) -> str | None:
        """Why this packed draft's depth loop runs eager (None: it may run from a graph)."""
        from tools import fn_kernels as K
        if K.MOE_STAGE or self.route_trace is not None or self._moe_record is not None:
            return "measuring"
        if len(items) < 2:
            return "lanes"
        depth = items[0][3]
        k0, v0 = items[0][1].k[MTP], items[0][1].v[MTP]
        s0 = (k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1))
        for _, st_m, _, d, cls in items:
            if d != depth:
                return "depth"
            if cls:
                return "sparse"
            k, v = st_m.k[MTP], st_m.v[MTP]
            if not (k.is_contiguous() and v.is_contiguous()) or (k.stride(0), k.stride(1), v.stride(0),
                                                                  v.stride(1)) != s0:
                return "layout"
        return None

    def _pack_lanes(self, items: list) -> list:
        """Per draft step after the feed (1 .. depth - 1), every lane's one row: its MTP state and its staged rows
        buffer (`stage_mtp`'s), in lane order."""
        depth = items[0][3]
        return [[{"st": st_m, "M": 1, "rb": self._rows_buffers(1, ("mtp", key, j, 1))} for key, st_m, _, _, _ in items]
                for j in range(1, depth)]

    def _pack_body(self, G, cxs: list) -> None:
        """The depth loop of a packed draft over the graph's inputs G.x / G.h (each lane's last fed row): the head and
        top-2 of every lane, then the MTP layer over every lane's draft row, `depth` times; the picks into G.tok /
        G.prob. `mtp_chain_multi`'s eager loop step for step (the coupled pick of keyed lanes over G.keys / G.par),
        the attention through G's entry tables."""
        G.aent["k"] = 0
        x, h = G.x, G.h
        for j in range(G.depth):
            prob, tok = self._mtp_pick_multi(x, j, G.kp)
            G.tok[j].copy_(tok)
            G.prob[j].copy_(prob)
            if j + 1 == G.depth:
                break
            self._mtp_pg = {"tok": tok[:, 0].contiguous(), "cx": cxs[j], "G": G}
            try:
                x, h = self.mtp_rows(None, None, h)
            finally:
                self._mtp_pg = None

    def _pack_capture(self, G, cxs: list) -> None:
        """Capture G's depth loop (on the runner's thread, under the capture lock, on the engine's leased capture
        stream): one eager run there first (it loads every kernel and makes the scratch outside the graph's pool;
        it computes what the replay after it computes again), then the capture."""
        ps = self._pg_state()
        t0 = time.perf_counter()
        graphs = ps["graphs"]
        if len(graphs) >= PACK_GRAPH_MAX:
            for old in graphs.values():
                old.drop()
            graphs.clear()
            ps["stats"]["flushes"] += 1
        if ps["stream"] is None:
            ps["stream"] = capture_lock.lease_stream()
        s = ps["stream"]
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._pack_body(G, cxs)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g, stream=s):
            self._pack_body(G, cxs)
        torch.cuda.current_stream().wait_stream(s)
        G.graph = g
        graphs[G.key] = G
        ms = 1e3 * (time.perf_counter() - t0)
        ps["stats"]["captures"] += 1
        ps["stats"]["capture_ms"] += ms
        print(f"[fn-pgraph] captured lanes {G.L} depth {G.depth}: {ms:.0f} ms ({len(graphs)} kept)", flush=True)

    def _pack_graph_run(self, items: list, last: torch.Tensor, x: torch.Tensor, h: torch.Tensor, kp: tuple | None = None):
        """`mtp_chain_multi` past its feed from the lane count's graph: each lane's last fed row (x, h at `last`) and,
        with keyed lanes (`kp`), their keys and parameters into the graph's inputs, the replay, the picks to the host ->
        `PackOut`; None: eager (`_pack_why`, or no graph for the key yet and none may be captured now: the
        PACK_GRAPH_AFTER-th draft of a key, while the server allows)."""
        from tools import fn_kernels as K
        ps = self._pg_state()
        why = self._pack_why(items)
        if why is not None:
            ps["eager"][why] = ps["eager"].get(why, 0) + 1
            return None
        L, depth = len(items), items[0][3]
        k0, v0 = items[0][1].k[MTP], items[0][1].v[MTP]
        key = (L, depth, id(self.draft_head) if self.draft_head is not None else None,
               (k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1)), bool(K.MOE_GROUP), x.dtype, h.dtype,
               None if kp is None else (id(kp[0]) if kp[0] is not None else None, kp[4], kp[5]))
        graphs = ps["graphs"]
        G = graphs.get(key)
        cxs = None
        if G is None:
            seen = ps["seen"][key] = ps["seen"].get(key, 0) + 1
            if seen < PACK_GRAPH_AFTER or not self.pack_capture_ok:
                ps["eager"]["no graph"] = ps["eager"].get("no graph", 0) + 1
                return None
            G = PackGraph(self, key, x.shape[1], h.shape[1], x.dtype, h.dtype, kp)
        else:
            graphs[key] = graphs.pop(key)               # most recently used last
        sig = tuple((it[0], id(it[1])) for it in items)
        torch.index_select(x, 0, last, out=G.x)
        torch.index_select(h, 0, last, out=G.h)
        if kp is not None:
            G.keys.copy_(kp[2])
            G.par.copy_(kp[3])
        captured = G.graph is None
        if captured:
            cxs = self._pack_lanes(items)
            self._pack_capture(G, cxs)                  # its own run wrote the entry rows of these lanes
            G.sig = sig
        elif G.sig != sig:
            # other lanes (or another order) on a kept graph: their entry rows, once the last replay's copies are done
            cxs = self._pack_lanes(items)
            G.done.synchronize()
            for j in range(depth - 1):
                self._attn_multi_fill(cxs[j], G.rows, MTP, self._pack_use, G.aent, j)
            G.sig = sig
            ps["stats"]["refills"] += 1
        G.graph.replay()
        out = G.publish(captured)
        ps["stats"]["replays"] += 1
        return out

    def stage_mtp(self, st_m: FNState, nexts: list[int], depth: int, keyed: tuple | None = None) -> int:
        """Host work of a greedy chain draft at st_m.length: the feed of n = len(nexts) committed rows
        (their streams copied into the static input on the device by the caller), then depth - 1
        one-row draft steps after them, each at its own position (cache slot = position), and the
        pool rows the feed completes. Returns the context class of the draft (`qsa_class`).
        SPD-249: `keyed` (engine/models/flashnext/coupled.py `keyed_for`: the noise keys a depth and the row's
        parameters) into the head's coupled buffers; with coupled drafting on and no `keyed`, the row's `on` flag
        0 (a packed chain that keys other lanes leaves this one's pairs alone)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        n, L = len(nexts), st_m.length
        r = c.indexer_compress_ratio if c.has_indexer else 4
        cls = self.qsa_class(L + n + depth - 1)
        for j in range(depth):
            M = n if j == 0 else 1
            rb = self._rows_buffers(M, ("mtp", self._mtp_key, j, M))
            rb["copied"].synchronize()
            h = rb["h"]
            p0 = L if j == 0 else L + n + j - 1
            pos = p0 + np.arange(M)
            if j == 0:
                h["tok"].numpy()[:] = nexts
            h["pos"].numpy()[:] = pos + st_m.delta
            h["slot"].numpy()[:] = pos
            anc = h["anc"].numpy()
            for i in range(M):
                anc[i, :i + 1] = p0 + np.arange(i + 1)
            # the feed's rows own the blocks from its first; a draft step's shared pool ends where the
            # feed's committed rows end (the pool is refreshed after the feed)
            self._stage_qsa(h, p0, (L if j == 0 else L + n) // r, pos, anc, M, st_m.delta)
            rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
            rb["copied"].record()
        if c.has_indexer:
            b = self._mtp_bufs(depth)
            blk = L // r + np.arange(SK.OWN)
            h32 = b["h32"].numpy()
            h32[:SK.OWN * 4] = np.clip((r * blk)[:, None] + np.arange(r)[None, :], 0, st_m.cap - 1).reshape(-1)
            h32[SK.OWN * 4:] = r * blk + st_m.delta
            b["hpi"].numpy()[:] = np.clip(blk, 0, st_m.pool[MTP].shape[0] - 1)
            b["d32"].copy_(b["h32"], non_blocking=True)
            b["dpi"].copy_(b["hpi"], non_blocking=True)
        if keyed is not None or self._coupled_on():
            b = self._mtp_bufs(depth)
            if keyed is not None:
                b["ckeys_h"].numpy()[:depth] = np.asarray(keyed[0], dtype=np.uint64).view(np.int64)
                b["cpar_h"].numpy()[:] = keyed[1]
            else:
                b["cpar_h"].numpy()[4] = 0.0
            b["ckeys"].copy_(b["ckeys_h"], non_blocking=True)
            b["cpar"].copy_(b["cpar_h"], non_blocking=True)
        return cls

    @staticmethod
    def _coupled_on() -> bool:
        from engine.models.flashnext import coupled as CP
        return CP.ON

    def _mtp_bufs(self, depth: int) -> dict:
        """The MTP chain's static buffers: the committed rows' streams (input), and per draft step the
        top-2 tokens and their probabilities under the draft head (output, with pinned copies); one set
        a head (`_mtp_key`): a side slot's draft must not land in another slot's buffers between its
        launch and its read (ENG-241)."""
        bs = self.__dict__.setdefault("_mtpbs", {})
        b = bs.get(self._mtp_key)
        if b is None:
            c = self.cfg
            b = bs[self._mtp_key] = {"stack": torch.zeros(33, c.hc_count * c.hidden_size, dtype=self.dtype, device=self.device),
                              "tok": torch.zeros(32, 2, dtype=torch.long, device=self.device),
                              "prob": torch.zeros(32, 2, dtype=torch.float32, device=self.device),
                              "tok_h": torch.zeros(32, 2, dtype=torch.long, pin_memory=True),
                              "prob_h": torch.zeros(32, 2, dtype=torch.float32, pin_memory=True),
                              "logits": torch.zeros(1, c.vocab_size, dtype=torch.float32, device=self.device)}
            from tools import fn_seq_kernels as SK
            b["h32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, pin_memory=True)
            b["d32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, device=self.device)
            b["hpi"] = torch.zeros(SK.OWN, dtype=torch.int64, pin_memory=True)
            b["dpi"] = torch.zeros(SK.OWN, dtype=torch.int64, device=self.device)
            # SPD-249: a keyed chain's noise keys a depth and its row parameters (coupled.py NPAR)
            b["ckeys_h"] = torch.zeros(32, dtype=torch.int64, pin_memory=True)
            b["ckeys"] = torch.zeros(32, dtype=torch.int64, device=self.device)
            b["cpar_h"] = torch.zeros(8, dtype=torch.float32, pin_memory=True)
            b["cpar"] = torch.zeros(8, dtype=torch.float32, device=self.device)
        assert depth <= 32
        for hids in (self.draft_ids, self.coupled_ids, self.lone_ids):
            if hids is not None and ("hot", hids.numel()) not in b:
                b[("hot", hids.numel())] = torch.zeros(1, hids.numel(), dtype=torch.float32, device=self.device)
        return b

    draft_head = None          # Int8Rows over a hot vocabulary (`set_draft_head`), else the full head
    draft_ids = None
    coupled_head = None        # SPD-249: the keyed drafts' own hot head (`set_coupled_head`), else the draft head
    coupled_ids = None
    lone_head = None           # SPD-249: a one-lane keyed chain's head (the coupled head's first rows), else
    lone_ids = None            # the coupled head

    def set_draft_head(self, ids) -> None:
        """Draft over a hot vocabulary `ids` (token ids; None: the full head): the head's rows,
        dequantised and stored as int8 rows with an fp32 scale (`tools.fn_kernels.gemv`), read in place
        of the full EXL3 head by the MTP drafts. Only the drafts change; the verify reads the served
        head. A head once built is kept for the process (captured draft graphs point into it)."""
        if ids is None:
            self.draft_head = self.draft_ids = None
            return
        self.draft_head, self.draft_ids = self._hot_head(ids)

    @staticmethod
    def _hot_key(ids) -> int:
        return hash(bytes(memoryview(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)))))

    def _hot_head(self, ids) -> tuple:
        from engine.models.flashnext.pack import Int8Rows
        heads = self.__dict__.setdefault("_draft_heads", {})
        key = self._hot_key(ids)
        if key not in heads:
            idx = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=self.device)
            w = self.W["lm_head.weight"].dequant()
            heads[key] = (Int8Rows.of(w.index_select(0, idx).half().contiguous()), idx)
            del w
            torch.cuda.empty_cache()
        return heads[key]

    def set_coupled_head(self, ids, lone: int = 0) -> None:
        """SPD-249: the keyed (coupled) drafts' own hot head over `ids` (None, or the draft head's ids: they
        read the draft head). Greedy and unseeded drafts keep the draft head; a keyed chain reads only this one.
        `lone` (FN_MTP_COUPLED_HOT_LONE, below the coupled head's size): a keyed chain of one lane reads the first
        `lone` rows of the coupled head instead (`keyed_head`), the hot ids being one ranked list
        (ops/fn-hot-vocab.npy), so a smaller head is a prefix of a larger. One int8 scale a row: a head over a
        prefix of the ids is the same bytes as the larger head's first rows, so it is a view of them, and the
        served draft head, where its ids are such a prefix, becomes one too (its own matrix freed): one matrix."""
        self.lone_head = self.lone_ids = None
        if ids is None:
            self.coupled_head = self.coupled_ids = None
            return
        h, i = self._hot_head(ids)
        if self.draft_head is not None and h is self.draft_head:
            self.coupled_head = self.coupled_ids = None
            return
        self.coupled_head, self.coupled_ids = h, i
        n = int(i.numel())
        nd = int(self.draft_ids.numel()) if self.draft_ids is not None else 0
        if 0 < nd < n and torch.equal(i[:nd], self.draft_ids):
            heads = self.__dict__.setdefault("_draft_heads", {})
            heads.pop(self._hot_key(self.draft_ids.cpu().numpy()), None)
            self.draft_head, self.draft_ids = self._head_rows(h, nd), i[:nd]
            torch.cuda.empty_cache()
        if 0 < lone < n:
            if lone == nd and self.draft_head is not None and torch.equal(i[:lone], self.draft_ids):
                self.lone_head, self.lone_ids = self.draft_head, self.draft_ids
            else:
                self.lone_head, self.lone_ids = self._head_rows(h, lone), i[:lone]

    @staticmethod
    def _head_rows(h, n: int):
        """The first n rows of an int8 hot head (one scale a row) as a view: the same bytes a head over those
        n ids alone holds (`quant_int8_rows` scales each row by its own max)."""
        from engine.models.flashnext.pack import Int8Rows
        assert h.s.dim() == 1, "a hot head has one scale a row"
        return Int8Rows(h.q[:n], h.s[:n])

    def keyed_head(self, lanes: int) -> tuple:
        """SPD-249: (head, ids) a keyed chain of `lanes` lanes reads: one lane the lone head where set, else
        the coupled head where set, else the draft head (None, None: the full head)."""
        if lanes <= 1 and self.lone_head is not None:
            return self.lone_head, self.lone_ids
        if self.coupled_head is not None:
            return self.coupled_head, self.coupled_ids
        return self.draft_head, self.draft_ids

    def mtp_chain_dev(self, st_m: FNState, n: int, depth: int, cls: int = 0, keyed: bool = False) -> None:
        """A greedy chain of `depth` drafts on the device (`stage_mtp` first, the n committed rows'
        streams in `_mtp_bufs()["stack"]`): the feed, then each draft's token and streams into the
        next step without the host. Per step the top-2 tokens and their probabilities under the draft
        head land in `tok` / `prob` and their pinned copies (the second token is a sibling candidate).
        SPD-249 `keyed`: each step's pair is the coupled pick (engine/models/flashnext/coupled.py) over the
        one-lane keyed head (`keyed_head(1)`: the lone head, else the coupled head, else the draft head), with the
        keys and parameters `stage_mtp` staged."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        self.class_check(st_m.length + n + depth - 1, cls, "mtp_chain_dev")
        b = self._mtp_bufs(depth)
        x, h = self.mtp_rows(st_m, ("mtp", self._mtp_key, 0, n), b["stack"][:n], cls)
        if c.has_indexer:                     # the blocks the feed completed, into the MTP pool
            ip = self.pre(MTP) + "self_attn.indexer."
            own = SK.qsa_pool(st_m.ik[MTP], b["d32"][:SK.OWN * 4].view(SK.OWN, 4), b["d32"][SK.OWN * 4:],
                              self.W[ip + "k_layernorm.weight"], self.rope_cos, self.rope_sin, c.rms_norm_eps)
            st_m.pool[MTP].index_copy_(0, b["dpi"], own)
        head, hids = self.keyed_head(1) if keyed else (self.draft_head, self.draft_ids)
        if keyed:
            from engine.models.flashnext import coupled as CP
        for j in range(depth):
            if head is not None:
                lg = b[("hot", hids.numel())]
                K.gemv(x[-1:], head, out=lg)
            else:
                lg = b["logits"]
                K.dense(x[-1:], self.W["lm_head.weight"], out=lg)
            vals, ids = torch.topk(lg[0], 2)
            z = torch.exp(lg[0] - vals[0]).sum()
            b["prob"][j].copy_(torch.exp(vals - vals[0]) / z)
            b["tok"][j].copy_(hids.index_select(0, ids) if head is not None else ids)
            if keyed:
                CP.pick(lg, hids if head is not None else None, b["ckeys"][j:j + 1], b["cpar"].view(1, -1), z,
                        b["tok"][j:j + 1], b["prob"][j:j + 1], c.vocab_size, x=x[-1:], depth=j)
            if j + 1 == depth:
                break
            self._rows_buffers(1, ("mtp", self._mtp_key, j + 1, 1))["d"]["tok"].copy_(b["tok"][j, :1])
            x, h = self.mtp_rows(st_m, ("mtp", self._mtp_key, j + 1, 1), h[-1:], cls)
        b["tok_h"].copy_(b["tok"], non_blocking=True)
        b["prob_h"].copy_(b["prob"], non_blocking=True)

    # ------------------------------------------------------------------------- generation
    prefill_path = PREFILL           # "kernels" (ENG-225) or "ref"

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, st: FNState, chunk: int | None = None, keep_stack: bool = False,
                path: str | None = None) -> torch.Tensor:
        """The prompt in chunks; logits of the last row. `path`: "kernels" (`forward_pre`, ENG-225) or
        "ref" (the reference's batched forward); default `prefill_path`. `keep_stack`: every row's
        pre-final-mixer streams in `self.prompt_stack` [T, hc H] (the MTP layer's input; `self.stack`
        holds the last chunk's only)."""
        self.flush(st)
        path = path or self.prefill_path
        if path != "kernels" and (st.mm is not None or st.delta):
            raise NotImplementedError("images on the reference prefill path (QWEN38_FN_PREFILL=ref): the "
                                      "prefill kernels take them")
        fwd = self.forward_pre if path == "kernels" else self.forward
        chunk = chunk or (PREFILL_CHUNK if path == "kernels" else 2048)
        out = None
        stacks = []
        from engine.models.flashnext import layer_major as LM
        starts = list(range(0, ids.numel(), chunk))
        wins = ([[i] for i in range(len(starts))] if not (LM.ROWS > chunk and len(starts) > 1 and LM.ok(self, st))
                else LM.groups([min(chunk, ids.numel() - a) for a in starts], LM.rows_now(chunk, LM.ROWS)))
        for win in wins:
            last = win[-1] == len(starts) - 1
            if len(win) > 1:
                # SPD-249 P1 (QWEN38_FN_PF_WINDOW): the window's chunks layer-major, each chunk's own bits
                out, hs = LM.window(self, [ids[starts[j]:starts[j] + chunk].to(self.device) for j in win], st,
                                   head=last)
                if keep_stack:
                    stacks += hs
                del hs
            else:
                i = starts[win[0]]
                out = fwd(ids[i:i + chunk].to(self.device), st, head="last" if last else "none")
                if keep_stack:
                    stacks.append(self.stack)
            if st.length > 4096:
                # the reference attention's score blocks grow with the context; cached blocks of one
                # chunk fit no request of the next (a 32k prompt held tens of GB, 2026-10-11)
                torch.cuda.empty_cache()
        self.prompt_stack = torch.cat(stacks) if keep_stack else None
        if st.pool and path != "kernels":            # the kernel path pools every chunk's blocks itself
            self.build_pool(st)
        return out[-1:]

    @torch.no_grad()
    def mtp_prime(self, stack: torch.Tensor, next_tokens: list[int], st_m: FNState, path: str | None = None) -> None:
        """The MTP layer's cache over committed rows: `stack` [T, hc H] (the trunk's streams at
        positions st_m.length..) and the tokens that follow them, in chunks; on the prefill kernels
        (`mtp_pre`) or the reference's `mtp_feed` with the pool built after."""
        path = path or self.prefill_path
        T = len(next_tokens)
        # the trunk's chunk is QWEN38_FN_PREFILL_CHUNK, but the MTP layer reshapes its stack slice to
        # T*hc rows for mtp.fc_hidden (a 16,384-row chunk crashes exl3_matmul with CUDA error:
        # invalid argument, 2026-10-30): the catch-up caps its own chunk where that matmul holds
        chunk = min(PREFILL_CHUNK, 8192) if path == "kernels" else 2048
        for i in range(0, T, chunk):
            j = min(i + chunk, T)
            nt = torch.tensor(next_tokens[i:j])
            if path == "kernels":
                self.mtp_pre(stack[i:j], nt, st_m)
            else:
                self.mtp_feed(stack[i:j], nt, st_m)
            if st_m.length > 4096:
                torch.cuda.empty_cache()           # as `prefill`
        if path != "kernels":
            self.build_pool(st_m, layers=[MTP])

    @torch.no_grad()
    def mtp_pre(self, stack: torch.Tensor, next_tokens: torch.Tensor, st_m: FNState) -> torch.Tensor:
        """The MTP layer over a chunk of committed rows on the prefill kernels: the reference's input
        fusion, its attention and MoE through `_attn_pre` / `_moe_pre`; the cache advances. Returns the
        layer's streams [T, hc H] (the final mixer and the head are the drafts' business)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        T = next_tokens.numel()
        self._pre = self._pre_ctl(st_m.length, T, st_m)
        e = self.rms(self.embed(next_tokens.view(-1)), self.W["mtp.pre_fc_norm_embedding.weight"], eps)
        e = self.lin(e, self.W["mtp.fc_embedding.weight"])
        hs = self.rms(stack, self.W["mtp.pre_fc_norm_hidden.weight"], eps)
        hs = self.lin(hs.reshape(T * hc, H), self.W["mtp.fc_hidden.weight"]).view(T, hc, H)
        h = (hs + e.unsqueeze(1)).flatten(1)
        p = self.pre(MTP)
        n = K.hc_norm(h, self.W[p + "attn_hyper_connection.hc_norm.weight"], H, eps)
        x, di = self._mix_pre(n, p + "attn_hyper_connection")
        out = self._attn_pre(x, MTP, st_m)
        h, n = self._inject_pre(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                                self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
        x, di = self._mix_pre(n, p + "mlp_hyper_connection")
        h, _ = self._inject_pre(h, self._moe_pre(x, MTP), di, self.W[p + "mlp_hyper_connection.rank"], None, hc, eps)
        self._n32_clear()
        st_m.length += T
        self._pre = None
        return h

    @torch.no_grad()
    def decode_eager(self, token: int, st: FNState) -> torch.Tensor:
        """One token through `step` (no graph)."""
        self.settle(st)
        self.stage(token, st)
        out = self.step(st)
        self._pending.pop(id(st), None)
        st.length += 1
        return out


class DecodeGraph:
    """`FlashNextEngine.step` captured once for a state and a context class (`qsa_class`: dense, or the
    width of the indexer's block scores); `__call__(token)` stages the token (host), replays the
    class's graph (captured when the length first enters it), and moves the length. `max_lc` is kept
    for the callers; the step reads the length from the device."""

    def __init__(self, eng: FlashNextEngine, st: FNState, max_lc: int = 2048):
        self.eng, self.st, self.max_lc = eng, st, max_lc
        eng._rows_buffers(1)
        self.graphs: dict = {}
        self.stacks: dict = {}                 # class -> the graph's streams output [1, hc H]
        self.graph = None
        self.captures = 0                      # classes captured lazily by a step
        self.dead: list = []                   # SRV-795: graphs whose capture raised

    def capture(self, cls: int | None = None) -> None:
        eng, st = self.eng, self.st
        cls = eng.qsa_class(st.length) if cls is None else cls
        eng.settle(st)
        saved = _state_saved(eng, st)
        saved_n = eng._pend_of(st)[1].clone()
        pctx = dict(st.ple_ctx)
        length = st.length
        g = torch.cuda.CUDAGraph()
        cap = capture_stream(st)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    eng.stage(0, st)
                    eng.step(st, cls)
            torch.cuda.current_stream().wait_stream(s)
            with capture_lock.LOCK, torch.cuda.graph(g, stream=cap):
                eng.step(st, cls)
            self.stacks[cls] = eng.stack
        except BaseException:
            # SRV-795: kept to be reset with its owner's graphs, never alone (SPD-223); a capture whose end raised
            # left the thread on the capture stream, which it leaves here (the state is put back below either way)
            self.dead.append(g)
            cap = cap if cap is not None else getattr(torch.cuda.graph, "default_capture_stream", None)
            if cap is not None and torch.cuda.current_stream() == cap:
                torch.cuda.set_stream(torch.cuda.default_stream())
            raise
        finally:
            torch.cuda.synchronize()
            _state_restore(saved)                       # the warm-up steps wrote the state: undo
            eng._pend_of(st)[1].copy_(saved_n)
            st.ple_ctx.clear()
            st.ple_ctx.update(pctx)
            st.length = length
        self.graphs[cls] = self.graph = g

    def _ready(self) -> None:
        cls = self.eng.qsa_class(self.st.length)
        if cls not in self.graphs:
            self.capture(cls)
            self.captures += 1              # this step's time is the capture's (a price skips it)
        self.graph = self.graphs[cls]

    @torch.no_grad()
    def greedy(self, token: int) -> int:
        """One greedy step: the next token, read from the graph's own argmax (no extra launch)."""
        eng, st = self.eng, self.st
        self._ready()
        eng.settle(st)
        t0 = time.perf_counter()
        eng.stage(token, st)
        eng.host_ms["stage"] = eng.host_ms.get("stage", 0.0) + 1e3 * (time.perf_counter() - t0)
        self.graph.replay()
        eng._pending.pop(id(st), None)
        st.length += 1
        torch.cuda.current_stream().synchronize()
        return int(eng.g["next_h"][0])

    @torch.no_grad()
    def __call__(self, token: int) -> torch.Tensor:
        eng, st = self.eng, self.st
        self._ready()
        eng.settle(st)
        eng.stage(token, st)
        self.graph.replay()
        eng._pending.pop(id(st), None)
        st.length += 1
        return eng.g["logits"]


class JointGraph:
    """SPD-249: one key's joint verify pass captured (`FlashNextEngine._joint_graph_pass`): the graph, its entry
    tables (pinned host rows a replay rewrites, device rows the graph copies them to) and the tensors its replays
    leave (logits, picks, streams, each layer's commit records)."""

    def __init__(self, eng: FlashNextEngine, key: tuple):
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, dev = eng.cfg, eng.device
        E, R = key[0], key[1]
        self.key = key
        L = c.num_hidden_layers
        cuts = [0] + [x for x in JG_CUTS if 0 < x < L] + [L]
        self.segs = list(zip(cuts, cuts[1:]))
        self.graphs = [torch.cuda.CUDAGraph() for _ in self.segs]
        # each layer's first entry rows: (GDN, attention, PLE) rows of the layers before it
        self.kidx, kg, ka, kp = {}, 0, 0, 0
        for li in range(L + 1):
            self.kidx[li] = (kg, ka, kp)
            if li < L:
                kp += li in c.ple_layers
                kg += c.is_linear(li)
                ka += not c.is_linear(li)
        self.t: dict = {}
        nl = max(len(c.linear_layers), 1)
        na = max(sum(1 for li in range(c.num_hidden_layers) if not c.is_linear(li)), 1)
        npl = max(len(c.ple_layers), 1)

        def pair(L, F):
            return (torch.zeros(L, E, F, dtype=torch.int64, pin_memory=True),
                    torch.zeros(L, E, F, dtype=torch.int64, device=dev))
        h, d = pair(nl, SK.COMMIT_FIELDS)
        hp, dp = pair(nl, SK.COMMIT_FIELDS)
        self.gent = {"h": h, "d": d, "hp": hp, "dp": dp, "k": 0}
        aph, apd = pair(na, K.APM_FIELDS)
        arh, ard = pair(na, SK.ARM_FIELDS)
        lane_dd = torch.zeros(R, dtype=torch.int32, device=dev)
        self.aent = {"ap_h": aph, "ap_d": apd, "ar_h": arh, "ar_d": ard, "k": 0, "lane_dd": lane_dd, "lane_d": lane_dd,
                     "lane_h": torch.zeros(R, dtype=torch.int32, pin_memory=True)}
        ph, pd = pair(npl, 16)
        self.pent = {"h": ph, "d": pd, "k": 0}

    def drop(self) -> None:
        for g in self.graphs:
            g.reset()
        self.t.clear()


class VerifyGraph:
    """`FlashNextEngine.rows_step` of an M-row verify block captured once for a state: every block
    of M rows (chain or tree; positions, slots, ancestors and the GDN tile's walk are device inputs)
    replays it. `replay()` after `stage_rows`; the commit then reads the graph's own intermediates."""

    def __init__(self, eng: FlashNextEngine, st: FNState, M: int, cls: int = 0):
        self.eng, self.st, self.M, self.cls = eng, st, M, cls
        self.graph = None
        self.rec: dict = {}

    def capture(self) -> None:
        eng, st, M = self.eng, self.st, self.M
        eng.stage_rows([0] * M, list(range(-1, M - 1)), st)   # writes only KV rows past the length
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.rows_step(st, M, cls=self.cls)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph, stream=capture_stream(st)):
            eng.rows_step(st, M, cls=self.cls)
        torch.cuda.synchronize()
        self.rec = eng.rec

    def replay(self) -> None:
        eng = self.eng
        eng.class_check(eng._rows_buffers(self.M).get("last", self.st.length + self.M - 1), self.cls, "verify graph")
        self.graph.replay()
        eng.rec = self.rec


class CommitGraph:
    """`FlashNextEngine.commit_dev` for a verify block of M rows and a path of n rows (moved or not),
    captured once after that block's VerifyGraph (it reads the verify graph's intermediates). The
    warm-up runs write the state: it is saved and restored around them."""

    def __init__(self, eng: FlashNextEngine, st: FNState, key: tuple):
        self.eng, self.st, self.key = eng, st, key
        self.graph = None

    def capture(self) -> None:
        eng, st = self.eng, self.st
        saved = _state_saved(eng, st)
        saved_n = eng._pend_of(st)[1].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            eng.commit_dev(st, self.key)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph, stream=capture_stream(st)):
            eng.commit_dev(st, self.key)
        torch.cuda.synchronize()
        _state_restore(saved)
        eng._pend_of(st)[1].copy_(saved_n)


class MTPChainGraph:
    """`FlashNextEngine.mtp_chain_dev` for a feed of n rows and a chain depth, captured once for an MTP
    cache: the whole draft (feed, depth - 1 deeper steps, every head and argmax) is one replay. SPD-249:
    `keyed`, the coupled chain (its keys and parameters staged before each replay)."""

    def __init__(self, eng: FlashNextEngine, st_m: FNState, n: int, depth: int, cls: int = 0, keyed: bool = False):
        self.eng, self.st_m, self.n, self.depth, self.cls, self.keyed = eng, st_m, n, depth, cls, keyed
        self.graph = None

    def capture(self) -> None:
        eng, st_m = self.eng, self.st_m
        # writes only cache rows past the length, and pool rows the next feed writes again
        kw = {"keyed": True} if self.keyed else {}
        if self.keyed:
            eng.stage_mtp(st_m, [0] * self.n, self.depth, keyed=([0] * self.depth, [1.0, 20.0, 0.95, -1.0, 1.0, 0.0, 0.0, 0.0]))
        else:
            eng.stage_mtp(st_m, [0] * self.n, self.depth)
        saved = {k: v.clone() for k, v in st_m.pool.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.mtp_chain_dev(st_m, self.n, self.depth, self.cls, **kw)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph, stream=capture_stream(st_m)):
            eng.mtp_chain_dev(st_m, self.n, self.depth, self.cls, **kw)
        torch.cuda.synchronize()
        for k, v in st_m.pool.items():
            v.copy_(saved[k])


class PackGraph:
    """SPD-249 (T1): the depth loop of a packed MTP draft over `L` lanes captured once (`FlashNextEngine._pack_capture`):
    its inputs (each lane's last fed row: x [L, H], streams h [L, hc H]; keyed lanes' keys and parameters), its picks (tok, prob [depth, L, 2]) and their
    pinned copies (`PACK_RING` of them, one a replay), the attention entry tables of its depth - 1 one-row steps (pinned
    host rows the graph copies to the device at each replay, rewritten only when the lanes change: `sig`)."""

    def __init__(self, eng: FlashNextEngine, key: tuple, xw: int, hw: int, xdt, hdt, kp: tuple | None = None):
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        dev = eng.device
        L, depth = key[0], key[1]
        self.key, self.L, self.depth = key, L, depth
        self.x = torch.zeros(L, xw, dtype=xdt, device=dev)
        self.h = torch.zeros(L, hw, dtype=hdt, device=dev)
        self.tok = torch.zeros(depth, L, 2, dtype=torch.long, device=dev)
        self.prob = torch.zeros(depth, L, 2, dtype=torch.float32, device=dev)
        # keyed lanes (coupled drafts): their keys and parameters, copied in before each replay; the pick's heads
        self.kp = self.keys = self.par = None
        if kp is not None:
            self.keys, self.par = torch.zeros_like(kp[2]), torch.zeros_like(kp[3])
            self.kp = (kp[0], kp[1], self.keys, self.par, kp[4], kp[5])
        self.ring = [(torch.zeros(depth, L, 2, dtype=torch.long, pin_memory=True),
                      torch.zeros(depth, L, 2, dtype=torch.float32, pin_memory=True)) for _ in range(PACK_RING)]
        self.gen = [0] * PACK_RING          # the replay whose picks each ring slot holds
        self.n = 0                          # replays so far
        self.stale = 0                      # reads of a ring slot a later replay took (none expected)
        n = max(depth - 1, 1)
        self.aent = {"ap_h": torch.zeros(n, L, K.APM_FIELDS, dtype=torch.int64, pin_memory=True),
                     "ap_d": torch.zeros(n, L, K.APM_FIELDS, dtype=torch.int64, device=dev),
                     "ar_h": torch.zeros(n, L, SK.ARM_FIELDS, dtype=torch.int64, pin_memory=True),
                     "ar_d": torch.zeros(n, L, SK.ARM_FIELDS, dtype=torch.int64, device=dev),
                     "k": 0, "lane_d": torch.arange(L, dtype=torch.int32, device=dev)}
        self.rows = [slice(i, i + 1) for i in range(L)]
        self.graph = None
        self.sig = None                     # the lanes (head key, MTP state) the entry rows hold
        self.done = torch.cuda.Event()      # after the last replay's copies

    def publish(self, captured: bool) -> "PackOut":
        """After a replay: its picks into the next ring slot (one copy each), on the stream after the graph."""
        s = self.n % PACK_RING
        self.n += 1
        self.gen[s] = self.n
        tok_h, prob_h = self.ring[s]
        tok_h.copy_(self.tok, non_blocking=True)
        prob_h.copy_(self.prob, non_blocking=True)
        self.done.record()
        return PackOut(self, s, self.n, captured)

    def drop(self) -> None:
        if self.graph is not None:
            self.graph.reset()
            self.graph = None


class PackOut:
    """SPD-249 (T1): one replay's picks: `read(i, depth)` is lane i's (top-2 tokens, probabilities) a depth, as its
    head's own pinned buffers would hold them, once the stream is past the replay (the head's read syncs first)."""

    def __init__(self, G: PackGraph, slot: int, gen: int, captured: bool):
        self.G, self.slot, self.gen_, self.captured = G, slot, gen, captured

    def read(self, i: int, depth: int) -> tuple[list, list]:
        G = self.G
        if G.gen[self.slot] != self.gen_:
            G.stale += 1                    # a draft only proposes: the verify decides the text; counted, never seen
            print(f"[fn-pgraph] a late read: lanes {G.L}, replay {self.gen_} read after {G.n}", flush=True)
        tok_h, prob_h = G.ring[self.slot]
        return tok_h[:depth, i].tolist(), prob_h[:depth, i].tolist()


#: the rows past the length a captured step, verify block or commit may write (a block is at most 33 rows)
CAPTURE_ROWS = 64


def capture_stream(st: FNState):
    """SRV-795: the stream a graph of state `st` is captured on: its owner's own (`st.graphs`, a slot adapter's
    engine/models/graph_set.py `GraphSet`), else torch's shared capture stream (None)."""
    gs = getattr(st, "graphs", None)
    return gs.stream() if gs is not None else None


def _state_saved(eng: FlashNextEngine, st: FNState) -> list:
    """What a capture's warm-up may write, cloned: the recurrent sections whole, the KV and indexer-key
    rows from the length on (`CAPTURE_ROWS`) and the pooled-key blocks from the length's block on.
    Cloning the whole buffers took 8 GB at a 262,144-token window, and with the KV pool the rows past
    the committed ones are not mapped at all (ENG-241)."""
    L = st.length
    r = eng.cfg.indexer_compress_ratio if eng.cfg.has_indexer else 4
    out = []
    for name, t in _state_tensors(st).items():
        kind = name.split(".", 1)[0]
        if kind in ("k", "v"):
            out.append((t, 1, L, min(CAPTURE_ROWS, t.shape[1] - L)))
        elif kind == "ik":
            out.append((t, 0, L, min(CAPTURE_ROWS, t.shape[0] - L)))
        elif kind == "pool":
            b = L // r
            out.append((t, 0, b, min(CAPTURE_ROWS // r + 2, t.shape[0] - b)))
        else:
            out.append((t, 0, 0, t.shape[0]))
    return [(t, ax, lo, n, t.narrow(ax, lo, n).clone()) for t, ax, lo, n in out if n > 0]


def _state_restore(saved: list) -> None:
    for t, ax, lo, n, x in saved:
        t.narrow(ax, lo, n).copy_(x)


def _state_tensors(st: FNState) -> dict:
    out = {}
    for name, d in (("S", st.S), ("conv", st.conv), ("k", st.k), ("v", st.v), ("ik", st.ik), ("pconv", st.ple_conv),
                    ("pool", st.pool)):
        for li, t in d.items():
            out[f"{name}.{li}"] = t
    return out
