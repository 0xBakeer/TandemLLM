"""The block drafter's draft call served from a CUDA graph (SPD-32).

A draft call is ~500 launches -- five decoder layers at sixteen rows, the vocabulary head, the
top-k and the selector's lattice -- and the block budget (SPD-19) charged ~2.3 ms a block of GPU idle
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

One graph per drafter (the eight- and the sixteen-wide arm are two drafters), captured on first use
after an eager run of the same body. Greedy drafting with one block and the target's head only;
anything else takes the eager call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
        self.graph = None
        # the cache the graph gathers from; a drafter that reallocates it gets a new graph
        self.ck_ptr = drafter._ck.data_ptr()
        self.stream = torch.cuda.Stream()
        self.out = None
        self.stats = {"captured": 0, "replayed": 0}

    @staticmethod
    def eligible(drafter) -> bool:
        from engine.drafters import dflash2
        return (drafter.blocks == 1 and drafter.head is None and not dflash2.DRAFT_HEAD_NVFP4
                and drafter.use_selector and drafter.path == "greedy"
                and drafter.cfg.sliding_window is not None
                and all(t == "sliding_attention" for t in drafter.cfg.layer_types)
                and not (drafter.sampler is not None and getattr(drafter.sampler, "on", False)))

    def _body(self):
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
        idx = lo + torch.arange(W, device=dev)
        valid = idx < ctx_len
        idx = torch.clamp(idx, max=d.max_len - 1)
        ctx_kv = [(d._ck[i].index_select(1, idx), d._cv[i].index_select(1, idx))
                  for i in range(cfg.num_hidden_layers)]
        allpos = torch.cat([idx, positions])
        seen = torch.cat([valid, torch.ones(bs, dtype=torch.bool, device=dev)])
        mask = ((positions[:, None] - allpos[None, :]) < W) & seen[None, :]
        hidden = m.forward_block(noise, positions, ctx_kv, idx, masks=(mask, mask))
        pred = hidden[1:]
        from engine.model import head_logits
        logits = head_logits(pred, eng.w.norm("lm_head.weight"))
        cand, unary = m.unary_candidates(logits)
        scores = m.lattice(pred, cand, unary, anchor)
        return cand, scores

    def run(self, anchor: int, pos0: int):
        """(candidates [L, k], lattice scores) for the block at `pos0` after `anchor`."""
        self.host[0], self.host[1], self.host[2] = anchor, pos0, self.d.ctx_len
        self.scal.copy_(self.host, non_blocking=True)
        if self.graph is None:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._body()                                    # compile everything first
            torch.cuda.current_stream().wait_stream(self.stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream):
                self.out = self._body()
            self.graph = g
            self.stats["captured"] += 1
        self.graph.replay()
        self.stats["replayed"] += 1
        return self.out
