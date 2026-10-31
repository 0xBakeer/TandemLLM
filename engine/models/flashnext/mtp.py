"""Flash-Next's MTP layer as a draft head (ENG-153, ENG-245): what the shared MTP drafter
(engine/methods/mtp_drafter.py) asks of a model's head, over `FlashNextEngine`'s MTP forward.

The head keeps its own cache (the MTP layer's KV and state, `FlashNextEngine.new_mtp_state`). A
round's draft -- the feed of the rows committed since the last one, `depth` - 1 deeper steps, every
head and its top-2 -- is one replay of a captured graph for that feed's row count and context class
(`MTPChainGraph`); the host reads the drafts once. A round's lanes draft together (`run_packed`): the
feed eager, and with QWEN38_FN_PACK_GRAPH (SPD-249, T1) the steps after it from the lane count's graph.

SPD-249: a seeded sampled request's rounds draft coupled (engine/models/flashnext/coupled.py): the drafter sets
`keyed` from `keyed_for(sampler, start)` before each launch; the round's chain is then the keyed graph of its feed
(or, packed, its lane's rows keyed), and `cut_inputs` gives the cut the measured inputs of such a round.
With FN_MTP_COUPLED_HOT_LONE a keyed chain of one lane (the lone graph, or a packed pass of one head) reads the
coupled head's first rows and a packed chain of more lanes the whole coupled head (`FlashNextEngine.keyed_head`);
the round's calibration follows the head its chain read (`keyed_hot`).
"""

from __future__ import annotations

import time

import torch

from engine.methods import draft_meet
from engine.models.flashnext import coupled as CP
from engine.models.flashnext.engine import FlashNextEngine, MTPChainGraph
from engine.models.flashnext.ref import MTP


class FlashNextMTPHead:
    def __init__(self, fn: FlashNextEngine, depth: int, cap: int, pages=None):
        self.fn, self.depth = fn, depth
        self.st = fn.new_mtp_state(cap, pages=pages)
        self.graphs: dict = {}
        self.captures = 0               # graphs captured so far (a round that captured is not a price)
        self.fresh: set = set()         # graphs captured ahead (`capture_class`) not replayed yet
        # its own chain buffers and staging (FlashNextEngine `_mtp_key`): the engine's first head 0
        self.key = getattr(fn, "_mtp_heads", 0)
        fn._mtp_heads = self.key + 1
        self._n = 0
        self.keyed = None               # SPD-249: this round's keys and parameters (`keyed_for`), or None
        self._keyed = False             # the staged draft is keyed
        self._khot = 0                  # the vocabulary size of the head the last keyed chain read (0: full)
        self._pk = None                 # SPD-249 (T1): (the packed draft graph's replay, this head's lane) to read

    # --- SPD-249 ---------------------------------------------------------------------
    def keyed_for(self, sampler, start: int):
        """What the next launch stages when the request drafts coupled (None: argmax)."""
        return CP.keyed_for(sampler, start, self.depth)

    @property
    def keyed_hot(self) -> int:
        """The vocabulary size of the head this round's keyed chain read (0: the full head): the lone head for a
        chain of one lane, the coupled head for a packed chain of more (`FlashNextEngine.keyed_head`); the
        coupled calibration is by this size."""
        return self._khot

    def _hot_of(self, lanes: int) -> int:
        _, ids = self.fn.keyed_head(lanes)
        return int(ids.numel()) if ids is not None else 0

    def cut_inputs(self, keyed: bool, lanes: int, cls: int = 0):
        """A keyed round's cut inputs (calibration, prices, draft ms, any None) and whether the cut may keep a
        depth-1 sibling; None: the cut's own inputs and no sibling."""
        if not keyed:
            return None
        ci = CP.CalInputs.get()
        cal, prices, draft = ci.inputs(self.keyed_hot, lanes, cls) if ci is not None else (None, None, None)
        if self.keyed is not None and len(self.keyed[1]) > 5 and self.keyed[1][5]:
            # SPD-249: the corrector picked this round's drafts: its own calibration where its file has one
            from engine.models.flashnext import corrector as CR
            cr = CR.get()
            cc = cr.calibration(self.keyed_hot) if cr is not None else None
            if cc is not None:
                cal = cc
        return {"cal": cal, "prices": prices, "draft_ms": draft, "sib1": CP.SIB}

    def reset(self) -> None:
        self.st.length = 0
        self.st.pool_n = {}

    def views(self) -> list:
        """The head's position-indexed cache as (tensor, token axis): the MTP layer's K, V and indexer
        keys (the caches keep its rows with the trunk's, ENG-157)."""
        st = self.st
        return [(st.k[MTP], 1), (st.v[MTP], 1)] + ([(st.ik[MTP], 0)] if MTP in st.ik else [])

    def resume(self, n: int) -> None:
        """The head's cache is valid for its first `n` rows (a cache wrote them back, or they are in
        place): the length, and the pooled indexer keys built again from those rows."""
        self.st.length = int(n)
        self.st.pool_n = {}
        if n > 0:
            self.fn.build_pool(self.st, layers=[MTP])

    def prime(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """Rows [n, hc H] and the token after each into the head's cache, in one call (a backlog)."""
        self.fn.mtp_prime(rows, nexts, self.st)

    def capture_all(self) -> None:
        """The graphs of every feed a round can bring (1 .. depth + 1 rows) at the current context class."""
        self.fn._mtp_key = self.key
        cls = self.fn.qsa_class(self.st.length + self.depth + 1)
        for n in range(1, self.depth + 2):
            if (n, cls) not in self.graphs:
                self.graphs[(n, cls)] = MTPChainGraph(self.fn, self.st, n, self.depth, cls)
                self.graphs[(n, cls)].capture()
                self.captures += 1
            # SPD-249: with coupled drafting on, the keyed chain of each feed too (a request's first keyed
            # round would pay its capture)
            if CP.ON and (n, cls, "k") not in self.graphs:
                self.graphs[(n, cls, "k")] = MTPChainGraph(self.fn, self.st, n, self.depth, cls, keyed=True)
                self.graphs[(n, cls, "k")].capture()
                self.captures += 1

    def capture_class(self, n: int, cls: int) -> None:
        """The graph of an n-row feed at context class `cls`, ahead of the round that needs it (the
        server's idle captures, engine/models/flashnext/serving.py); the cache as it was."""
        self.fn._mtp_key = self.key
        if (n, cls) not in self.graphs:
            g = MTPChainGraph(self.fn, self.st, n, self.depth, cls)
            g.capture()
            self.graphs[(n, cls)] = g
            self.fresh.add((n, cls))

    def launch(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """Feed rows [n, hc H] with their next tokens and draft `depth` tokens from the last: the graph
        is launched, not waited for (`read`)."""
        self.fn._mtp_key = self.key
        fn = self.fn
        n = len(nexts)
        self._pk = None
        cls = fn.qsa_class(self.st.length + n + self.depth - 1)
        kd = self.keyed if CP.ON else None
        self._keyed = kd is not None
        if draft_meet.MEET is not None and draft_meet.MEET(self):
            # SPD-235: the round runner runs this draft with its other lanes' (`run_packed`), before the read
            fn._mtp_bufs(self.depth)["stack"][:n].copy_(rows)
            assert (fn.stage_mtp(self.st, nexts, self.depth, keyed=kd) if kd is not None
                    else fn.stage_mtp(self.st, nexts, self.depth)) == cls
            self._n, self._cls = n, cls
            return
        gk = (n, cls, "k") if kd is not None else (n, cls)
        g = self.graphs.get(gk)
        if gk in self.fresh:
            self.fresh.discard(gk)
            self.captures += 1          # an idle capture's first replay uploads it: the drafter skips it
        if g is None:
            g = self.graphs[gk] = MTPChainGraph(fn, self.st, n, self.depth, cls, keyed=kd is not None)
            g.capture()
            self.captures += 1          # this round's draft time is the capture's (the drafter skips it)
        b = fn._mtp_bufs(self.depth)
        b["stack"][:n].copy_(rows)
        t0 = time.perf_counter()
        assert (fn.stage_mtp(self.st, nexts, self.depth, keyed=kd) if kd is not None
                else fn.stage_mtp(self.st, nexts, self.depth)) == cls
        fn.host_ms["stage_mtp"] = fn.host_ms.get("stage_mtp", 0.0) + 1e3 * (time.perf_counter() - t0)
        g.graph.replay()
        self._khot = self._hot_of(1)
        self._n = n

    @staticmethod
    def run_packed(heads: list) -> None:
        """SPD-235: the staged drafts of these heads (one engine) in one packed chain. SPD-249 (T1): from the lane
        count's draft graph, each head reads its lane of that replay's picks."""
        keyed = [h._keyed for h in heads]
        items = [(h.key, h.st, h._n, h.depth, h._cls) for h in heads]
        if any(keyed):
            out = heads[0].fn.mtp_chain_multi(items, keyed=keyed)
            hot = heads[0]._hot_of(len(heads))
            for h in heads:
                h._khot = hot
        else:
            out = heads[0].fn.mtp_chain_multi(items)
        for i, h in enumerate(heads):
            h._pk = None if out is None else (out, i)
            if out is not None and out.captured:
                h.captures += 1         # this round's draft time is the capture's (a price skips it)

    def read(self) -> tuple[list, list]:
        """After `launch`: per depth the (first, second) token and their probabilities; the fed rows
        join the head's cache."""
        self.fn._mtp_key = self.key
        torch.cuda.current_stream().synchronize()
        self.st.length += self._n
        pk, self._pk = self._pk, None
        if pk is not None:
            return pk[0].read(pk[1], self.depth)
        b = self.fn._mtp_bufs(self.depth)
        return b["tok_h"][:self.depth].tolist(), b["prob_h"][:self.depth].tolist()
