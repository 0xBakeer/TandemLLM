"""The block drafter's draft call served from a CUDA graph.

A draft call is ~500 launches -- five decoder layers at sixteen rows, the vocabulary head, the
top-k and the selector's lattice -- and the block budget charged ~2.3 ms a block of GPU idle
to the gaps between them. The call has a fixed shape except for one thing: its context, the
drafter's own KV cache over the sliding window behind the anchor, which is as long as the text so
far up to 2,048 positions. So the graph always attends over exactly the window:

  * the window's cache rows are GATHERED at device indices `lo .. lo + 2047`, `lo` the window's
    first position, and every row past the committed context is masked out;
  * the anchor token, the block's first position and the context length live in device scalars,
    filled before a replay;
  * the selector's lattice takes the anchor as a device tensor.

Masked columns contribute exact zeros to the softmax, so a replay computes what the eager call
computes over the unpadded context, up to the summation order of the attention's own kernel. The
walk over the lattice stays on the host, as `walk_host` (one copy). The drafter only proposes: its
arithmetic cannot change the engine's output, only how much of a draft is accepted.

The gathered span is a context CLASS, not always the whole window: the next power of two of the
context from 512 up to the 2,048 window. The first version gathered 2,048 every call and lost -- on
the row's ~500-token contexts the padded attention cost 2.5 ms of device time a block against 1.3 ms
of idle saved (K4). One graph per class per drafter (the eight- and the sixteen-wide arm are two
drafters), captured on first use after an eager run of the same body. Greedy drafting with one
block and the target's head only; anything else takes the eager call.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine import capture_lock



def dflash2_head(eng):
    """The head the draft reads: the target's, or QWEN38_DRAFT_HEAD_EXL3's (engine/drafters/dflash2.py)."""
    from engine.drafters import dflash2
    if dflash2.DRAFT_HEAD_EXL3:
        return dflash2.exl3_draft_head(eng)
    return eng.w.norm("lm_head.weight")


class DraftGraph:
    def __init__(self, drafter):
        self.d = drafter
        eng = drafter.eng
        dev = eng.device
        cfg = drafter.cfg
        self.bs = cfg.block_size
        self.win = int(cfg.sliding_window)
        self.scal = torch.zeros(3, dtype=torch.long, device=dev)      # anchor, pos0, ctx_len
        self.host = torch.zeros(3, dtype=torch.long).pin_memory()
        self.graphs: dict = {}                                        # class -> (graph, out)
        # the cache the graph gathers from; a drafter that reallocates it gets a new graph
        self.ck_ptr = drafter._ck.data_ptr()
        self.stream = torch.cuda.Stream()
        self.out = None
        self.stats = {"captured": 0, "replayed": 0}

    @staticmethod
    def eligible(drafter) -> bool:  # noqa: D401
        from engine.drafters import dflash2
        return (drafter.blocks == 1 and drafter.head is None and not dflash2.DRAFT_HEAD_NVFP4
                and drafter.use_selector and drafter.path == "greedy"
                and drafter.cfg.sliding_window is not None
                and all(t == "sliding_attention" for t in drafter.cfg.layer_types)
                and not (drafter.sampler is not None and getattr(drafter.sampler, "on", False)))

    def span(self, pos0: int) -> int:
        """The gathered span for a block at `pos0`: next power of two of the context, 512..window."""
        c = 512
        while c < min(pos0 + 1, self.win):
            c *= 2
        return min(c, self.win)

    def _body(self, span: int, temp: float | None = None):
        d = self.d
        m = d.module
        cfg = d.cfg
        eng = d.eng
        dev = eng.device
        bs, W = self.bs, self.win
        anchor, pos0, ctx_len = self.scal[0:1], self.scal[1:2], self.scal[2:3]
        tail = torch.full((bs - 1,), cfg.mask_token_id, dtype=torch.long, device=dev)
        ids = torch.cat([anchor, tail])
        noise = F.embedding(ids, eng.w.norm("embed_tokens.weight")).to(m.dtype)
        positions = pos0 + torch.arange(bs, device=dev)
        lo = torch.clamp(pos0 - W + 1, min=0)
        idx = lo + torch.arange(span, device=dev)
        valid = idx < ctx_len
        idx = torch.clamp(idx, max=d.max_len - 1)
        if getattr(d, "_cs", None) is not None:
            valid = valid & (d._cs.index_select(0, idx) > 0)      # SPD-177: rows that are real
        ctx_kv = [(d._ck[i].index_select(1, idx), d._cv[i].index_select(1, idx))
                  for i in range(cfg.num_hidden_layers)]
        allpos = torch.cat([idx, positions])
        seen = torch.cat([valid, torch.ones(bs, dtype=torch.bool, device=dev)])
        mask = ((positions[:, None] - allpos[None, :]) < W) & seen[None, :]
        hidden = m.forward_block(noise, positions, ctx_kv, idx, masks=(mask, mask))
        pred = hidden[1:]
        from engine.model import head_logits
        logits = head_logits(pred, dflash2_head(eng))
        cand, unary = m.unary_candidates(logits)
        scores = m.lattice(pred, cand, unary, anchor)
        if temp is None:
            return cand, scores
        # the tree's log-probabilities in the graph, read back in one copy with the walk
        return cand, scores, torch.log_softmax(scores.float() / temp, dim=-1)

    def run(self, anchor: int, pos0: int):
        """(candidates [L, k], lattice scores[, log-probabilities]) for the block at `pos0` after
        `anchor`; the third with QWEN38_HOST_ASYNC, at the drafter's tree temperature."""
        from engine import model as M
        self.host[0], self.host[1], self.host[2] = anchor, pos0, self.d.ctx_len
        self.scal.copy_(self.host, non_blocking=True)
        span = self.span(pos0)
        temp = float(self.d.tree_temp) if M.HOST_ASYNC else None
        # the kernel flags too, as the verify graphs key on them (an in-process A/B flips
        # them; a served process never does, so this is one key a span)
        from engine.verify_graph import VerifyGraphs
        key = (span, temp, VerifyGraphs.signature())
        got = self.graphs.get(key)
        if got is None:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._body(span, temp)                          # compile everything first
            torch.cuda.current_stream().wait_stream(self.stream)
            g = torch.cuda.CUDAGraph()
            # no other thread's CUDA call may run inside a capture (engine/capture_lock.py)
            with capture_lock.LOCK, torch.cuda.graph(g, stream=self.stream):
                out = self._body(span, temp)
            got = self.graphs[key] = (g, out)
            self.stats["captured"] += 1
        got[0].replay()
        self.stats["replayed"] += 1
        return got[1]


def _span(pos0: int, win: int) -> int:
    c = 512
    while c < min(pos0 + 1, win):
        c *= 2
    return min(c, win)


def _packed_body(ds: list, scals: list, spans: list, temp, projs: list | None = None):
    """`DraftGraph._body` for several drafters over one module, each block's inputs from its own
    device scalars (anchor, pos0, ctx_len), the projections and the head once for all rows. `projs`
    (QWEN38_ROUND_LOG_LATTICE): each block's selector-projected hidden [L, r] is appended to it."""
    from engine.drafters.dflash2 import forward_block_multi
    from engine.model import head_logits
    d0 = ds[0]
    m, cfg, eng = d0.module, d0.cfg, d0.eng
    dev = eng.device
    bs, W = cfg.block_size, int(cfg.sliding_window)
    items, anchors = [], []
    for d, sc, span in zip(ds, scals, spans):
        anchor, pos0, ctx_len = sc[0:1], sc[1:2], sc[2:3]
        tail = torch.full((bs - 1,), cfg.mask_token_id, dtype=torch.long, device=dev)
        ids = torch.cat([anchor, tail])
        noise = F.embedding(ids, eng.w.norm("embed_tokens.weight")).to(m.dtype)
        positions = pos0 + torch.arange(bs, device=dev)
        lo = torch.clamp(pos0 - W + 1, min=0)
        idx = lo + torch.arange(span, device=dev)
        valid = idx < ctx_len
        idx = torch.clamp(idx, max=d.max_len - 1)
        if getattr(d, "_cs", None) is not None:
            valid = valid & (d._cs.index_select(0, idx) > 0)      # SPD-177: rows that are real
        from engine.drafters.dflash2 import PACK_TAIL
        if PACK_TAIL:
            # SPD-203: the window gathered with bs rows to spare at its tail, where the block's own keys
            # and values are written (`_forward_block_steps`): one copy of the window a layer, not two
            idx_t = torch.cat([idx, torch.zeros(bs, dtype=idx.dtype, device=dev)])
            ctx_kv = [(d._ck[i].index_select(1, idx_t), d._cv[i].index_select(1, idx_t), span)
                      for i in range(cfg.num_hidden_layers)]
        else:
            ctx_kv = [(d._ck[i].index_select(1, idx), d._cv[i].index_select(1, idx))
                      for i in range(cfg.num_hidden_layers)]
        allpos = torch.cat([idx, positions])
        seen = torch.cat([valid, torch.ones(bs, dtype=torch.bool, device=dev)])
        mask = ((positions[:, None] - allpos[None, :]) < W) & seen[None, :]
        items.append((noise, positions, ctx_kv, idx, (mask, mask)))
        anchors.append(anchor)
    hidden = forward_block_multi(m, items)
    preds = [h[1:] for h in hidden]
    if projs is not None:
        wp = m.w["candidate_selector.hidden_projection.weight"]
        projs.extend(F.linear(p, wp) for p in preds)
    logits = head_logits(torch.cat(preds, dim=0), dflash2_head(eng))
    out = []
    sizes = [p.shape[0] for p in preds]
    from engine.drafters.dflash2 import PACK_TOPK
    if PACK_TOPK:
        # SPD-161: the head's top-k once over every block's rows (a row's top-k is its own)
        c_all, u_all = m.unary_candidates(logits, exact=False)
        cu = list(zip(c_all.split(sizes, dim=0), u_all.split(sizes, dim=0)))
    for j, (a_t, lg, pred) in enumerate(zip(anchors, logits.split(sizes, dim=0), preds)):
        cand, unary = cu[j] if PACK_TOPK else m.unary_candidates(lg)
        scores = m.lattice(pred, cand, unary, a_t)
        out.append((cand, scores) if temp is None
                   else (cand, scores, torch.log_softmax(scores.float() / temp, dim=-1)))
    return out


def _packed_body_batched(ds: list, scals: list, L: int, temp, projs: list | None = None):
    """`_packed_body` as one batched block pass (dflash2 PACK_BATCH): every block's window gathered
    to the longest span `L` (rows past a block's own context masked, so each block attends over the
    rows it attends alone), the inputs, masks and top-k built once for all blocks."""
    from engine.drafters.dflash2 import forward_block_batched
    from engine.model import head_logits
    d0 = ds[0]
    m, cfg, eng = d0.module, d0.cfg, d0.eng
    dev = eng.device
    bs, W, E = cfg.block_size, int(cfg.sliding_window), len(ds)
    sc = torch.stack(scals)                                               # [E, 3]
    pos0, ctx_len = sc[:, 1:2], sc[:, 2:3]
    ids = torch.cat([sc[:, 0:1], torch.full((E, bs - 1), cfg.mask_token_id, dtype=torch.long,
                                            device=dev)], dim=1)
    noise = F.embedding(ids.reshape(-1), eng.w.norm("embed_tokens.weight")).to(m.dtype)
    positions = pos0 + torch.arange(bs, device=dev)                       # [E, bs]
    idx = torch.clamp(pos0 - W + 1, min=0) + torch.arange(L, device=dev)  # [E, L]
    valid = idx < ctx_len
    idx = torch.clamp(idx, max=min(d.max_len for d in ds) - 1)
    if getattr(d0, "_cs", None) is not None:
        valid = valid & torch.stack([d._cs.index_select(0, idx[e]) > 0 for e, d in enumerate(ds)])
    # each window gathered straight into its block's row of one buffer, bs rows to spare at the tail
    # for the block's own keys and values (no stacked or concatenated copy of the windows)
    idx_t = torch.cat([idx, torch.zeros(E, bs, dtype=idx.dtype, device=dev)], dim=1)
    ctx_k, ctx_v = [], []
    for i in range(cfg.num_hidden_layers):
        ck0 = d0._ck[i]
        kb = torch.empty((E, ck0.shape[0], L + bs, ck0.shape[2]), dtype=ck0.dtype, device=dev)
        vb = torch.empty_like(kb)
        for e, d in enumerate(ds):
            torch.index_select(d._ck[i], 1, idx_t[e], out=kb[e])
            torch.index_select(d._cv[i], 1, idx_t[e], out=vb[e])
        ctx_k.append(kb)
        ctx_v.append(vb)
    allpos = torch.cat([idx, positions], dim=1)                           # [E, L + bs]
    seen = torch.cat([valid, torch.ones(E, bs, dtype=torch.bool, device=dev)], dim=1)
    mask = ((positions[:, :, None] - allpos[:, None, :]) < W) & seen[:, None, :]
    hidden = forward_block_batched(m, noise, positions.reshape(-1), ctx_k, ctx_v, mask)
    preds = hidden.view(E, bs, -1)[:, 1:]                                 # [E, bs - 1, H]
    if projs is not None:
        wp = m.w["candidate_selector.hidden_projection.weight"]
        projs.extend(F.linear(preds[e], wp) for e in range(E))
    logits = head_logits(preds.reshape(E * (bs - 1), -1), dflash2_head(eng))
    c_all, u_all = m.unary_candidates(logits)
    out = []
    for e in range(E):
        r = slice(e * (bs - 1), (e + 1) * (bs - 1))
        scores = m.lattice(preds[e], c_all[r], u_all[r], scals[e][0:1])
        out.append((c_all[r], scores) if temp is None
                   else (c_all[r], scores, torch.log_softmax(scores.float() / temp, dim=-1)))
    return out


class PackedDraftGraph:
    """The packed draft of one set of drafters served from CUDA graphs, one per (spans, temp,
    kernel flags), as `DraftGraph` serves the lone call. Static device scalars per drafter."""

    def __init__(self, ds: tuple):
        self.ds = ds
        dev = ds[0].eng.device
        self.scal = [torch.zeros(3, dtype=torch.long, device=dev) for _ in ds]
        self.host = torch.zeros(len(ds), 3, dtype=torch.long).pin_memory()
        self.ptrs = tuple(d._ck.data_ptr() for d in ds)
        self.graphs: dict = {}
        self.stream = torch.cuda.Stream()
        # one memory pool for this set's graphs (one per span combination): they never run at once,
        # and each replay's outputs are read before the set's next packed call
        self.pool = torch.cuda.graph_pool_handle()
        self.stats = {"captured": 0, "replayed": 0}

    def run(self, reqs: list, temp):
        from engine.verify_graph import VerifyGraphs
        W = int(self.ds[0].cfg.sliding_window)
        for j, (d, anchor, pos0) in enumerate(reqs):
            self.host[j, 0], self.host[j, 1], self.host[j, 2] = anchor, pos0, d.ctx_len
        for j in range(len(reqs)):
            self.scal[j].copy_(self.host[j], non_blocking=True)
        spans = tuple(_span(pos0, W) for _, _, pos0 in reqs)
        from engine.drafters.dflash2 import PACK_BATCH
        if PACK_BATCH:
            spans = (max(spans),)                   # one padded width for every block
            if HOST_ONE and temp is None:
                # the tree's log-probabilities in the graph too (what each block's tree computes alone
                # after its walk: the same op on the same tensor), so the host's one copy carries them
                temp = float(self.ds[0].tree_temp)
        key = (spans, temp, VerifyGraphs.signature())
        got = self.graphs.get(key)
        if got is None:
            if len(self.graphs) > 32:
                self.graphs.clear()
            self.stream.wait_stream(torch.cuda.current_stream())
            projs = [] if LOG_PROJ else None
            inner = ((lambda pj: _packed_body_batched(list(self.ds), self.scal, spans[0], temp, pj))
                     if PACK_BATCH else
                     (lambda pj: _packed_body(list(self.ds), self.scal, list(spans), temp, pj)))
            one = PACK_BATCH and temp is not None and HOST_ONE

            def body(pj):
                res = inner(pj)
                if not one:
                    return res, None
                # every block's walk blob and log-probabilities in two tensors, for one copy a round
                from engine.drafters.dflash2 import DFlash2Module
                return res, (torch.stack([DFlash2Module._blob(c, sc) for c, sc, _ in res]),
                             torch.stack([lp for _, _, lp in res]))
            with torch.cuda.stream(self.stream):
                body([] if LOG_PROJ else None)
            torch.cuda.current_stream().wait_stream(self.stream)
            g = torch.cuda.CUDAGraph()
            # no other thread's CUDA call may run inside a capture (engine/capture_lock.py)
            with capture_lock.LOCK, torch.cuda.graph(g, pool=self.pool, stream=self.stream):
                out, both = body(projs)
            got = self.graphs[key] = (g, out, projs, _HostBlobs(*both) if both is not None else None)
            self.stats["captured"] += 1
        got[0].replay()
        self.stats["replayed"] += 1
        if got[2]:
            for d, pj in zip(self.ds, got[2]):
                d._pred_proj = pj               # this replay's buffers (the round logger reads them)
        hb = got[3]
        if hb is None:
            return got[1]
        hb.fetch()
        return [r + ((hb, e),) for e, r in enumerate(got[1])]


class _HostBlobs:
    """SPD-201: a packed draft's walk blobs and log-probabilities brought to the host in one copy behind
    the replay; the first block that reads waits once, the others read what is there. The values are the
    ones each block's own `walk_host_logp` copies."""

    def __init__(self, blob: torch.Tensor, logp: torch.Tensor):
        self.blob, self.logp = blob, logp
        self.hb = torch.empty(blob.shape, dtype=blob.dtype, pin_memory=True)
        self.hl = torch.empty(logp.shape, dtype=logp.dtype, pin_memory=True)
        self.ev = torch.cuda.Event()
        self.lists = None

    def fetch(self) -> None:
        self.hb.copy_(self.blob, non_blocking=True)
        self.hl.copy_(self.logp, non_blocking=True)
        self.ev.record()
        self.lists = None

    def get(self, e: int) -> tuple[list, list]:
        if self.lists is None:
            self.ev.synchronize()
            self.lists = (self.hb.tolist(), self.hl.tolist())
        return self.lists[0][e], self.lists[1][e]


_PACKED: dict = {}
#: SPD-157: packed-draft graph sets kept at most (one per ordered set of drafters); the least recently used
#: goes first. With slots packed in slot order a box sees few sets, but subsets of 8 slots are 247
_PACKED_MAX = 24


#: the packed draft from CUDA graphs (False: eager, for the board check's comparison)
PACKED_GRAPH = True
#: SPD-201 (QWEN38_ROUND_LOG_LATTICE): each packed block's selector-projected hidden kept on its drafter
#: (`_pred_proj`, [L, r]) for the round logger
from engine.settings import SETTINGS as _S  # noqa: E402
LOG_PROJ = _S.get("ROUND_LOG_LATTICE") == "1"
#: SPD-201: the batched packed draft's walk blobs and log-probabilities for every block in one copy and one
#: wait (`_HostBlobs`), not a copy, a sync and two launches a block. 0: each block's own
HOST_ONE = os.environ.get("QWEN38_DF2_PACK_HOST1", "1") == "1"


def packed_draft(reqs: list, graph: bool | None = None) -> list:
    """Parallel requests: the draft call of several slots' drafters over ONE module, as `DraftGraph._body`
    computes it for each alone (the same gathered span, mask, head, top-k and lattice), with the
    module's projections and the vocabulary head run once over every block's rows.
    `reqs` = [(drafter, anchor, pos0)]; returns each one's `(cand, scores[, logp])` -- static
    graph buffers with `graph` (read them before the same set's next packed call)."""
    from engine import model as M
    from engine.drafters.dflash2 import flush_syncs
    ds = [r[0] for r in reqs]
    flush_syncs(ds)                    # SPD-161: every slot's pending sync in one call, before the read
    # each drafter builds its own module object over the one cached weight dict (dflash2
    # `load_weights`); a module holds nothing but those weights, so one set of weights is one module
    if any(d.module.w is not ds[0].module.w for d in ds):
        raise ValueError("a packed draft runs one drafter checkpoint")
    temp = float(ds[0].tree_temp) if M.HOST_ASYNC else None
    graph = PACKED_GRAPH if graph is None else graph
    if graph and torch.cuda.is_available():
        key = tuple(id(d) for d in ds)
        pg = _PACKED.pop(key, None)
        if pg is None or pg.ptrs != tuple(d._ck.data_ptr() for d in ds):
            pg = PackedDraftGraph(tuple(ds))
        _PACKED[key] = pg                          # most recently used last
        while len(_PACKED) > _PACKED_MAX:
            _PACKED.pop(next(iter(_PACKED)))
        return pg.run(reqs, temp)
    dev = ds[0].eng.device
    scals = [torch.tensor([a, p, d.ctx_len], dtype=torch.long, device=dev) for d, a, p in reqs]
    spans = [_span(p, int(ds[0].cfg.sliding_window)) for _, _, p in reqs]
    projs = [] if LOG_PROJ else None
    from engine.drafters.dflash2 import PACK_BATCH
    out = (_packed_body_batched(ds, scals, max(spans), temp, projs) if PACK_BATCH
           else _packed_body(ds, scals, spans, temp, projs))
    for d, pj in zip(ds, projs or []):
        d._pred_proj = pj
    return out
