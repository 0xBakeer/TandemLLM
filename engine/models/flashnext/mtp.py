"""Flash-Next's MTP layer as a draft head (ENG-153, ENG-245): what the shared MTP drafter
(engine/methods/mtp_drafter.py) asks of a model's head, over `FlashNextEngine`'s MTP forward.

The head keeps its own cache (the MTP layer's KV and state, `FlashNextEngine.new_mtp_state`). A
round's draft -- the feed of the rows committed since the last one, `depth` - 1 deeper steps, every
head and its top-2 -- is one replay of a captured graph for that feed's row count and context class
(`MTPChainGraph`); the host reads the drafts once.
"""

from __future__ import annotations

import time

import torch

from engine.methods import draft_meet
from engine.methods.mtp_drafter import MTPTreeDrafter
from engine.models.flashnext.engine import FlashNextEngine, MTPChainGraph
from engine.models.flashnext.ref import MTP


class FlashNextMTPHead:
    """`depths` (QWEN38_FN_DEPTH_LADDER): shorter chains the drafter may ask for a round (`cur`), each with its own
    graphs; `depth` stays the deepest (the buffers, the cut's ceiling). Empty: every round drafts `depth`."""

    def __init__(self, fn: FlashNextEngine, depth: int, cap: int, pages=None, depths: tuple = ()):
        self.fn, self.depth = fn, depth
        self.depths = tuple(sorted({d for d in depths if 0 < d < depth} | {depth}))
        self.cur = depth
        self.st = fn.new_mtp_state(cap, pages=pages)
        self.graphs: dict = {}
        self.captures = 0               # graphs captured so far (a round that captured is not a price)
        self.fresh: set = set()         # graphs captured ahead (`capture_class`) not replayed yet
        # its own chain buffers and staging (FlashNextEngine `_mtp_key`): the engine's first head 0
        self.key = getattr(fn, "_mtp_heads", 0)
        fn._mtp_heads = self.key + 1
        self._n = 0

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
        for d in self.depths:
            cls = self.fn.qsa_class(self.st.length + d + 1)
            for n in range(1, self.depth + 2):
                k = self._key(n, cls, d)
                if k not in self.graphs:
                    self.graphs[k] = MTPChainGraph(self.fn, self.st, n, d, cls)
                    self.graphs[k].capture()
                    self.captures += 1

    def _key(self, n: int, cls: int, d: int):
        """A graph's key: (feed rows, class) at the full depth (as the idle captures name it), with the depth below."""
        return (n, cls) if d == self.depth else (n, cls, d)

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
        d = self.depth if draft_meet.MEET is not None else self.cur
        self._d = d
        cls = fn.qsa_class(self.st.length + n + d - 1)
        if draft_meet.MEET is not None and draft_meet.MEET(self):
            # SPD-235: the round runner runs this draft with its other lanes' (`run_packed`), before the read
            fn._mtp_bufs(self.depth)["stack"][:n].copy_(rows)
            assert fn.stage_mtp(self.st, nexts, self.depth) == cls
            self._n, self._cls = n, cls
            return
        key = self._key(n, cls, d)
        g = self.graphs.get(key)
        if key in self.fresh:
            self.fresh.discard(key)
            self.captures += 1          # an idle capture's first replay uploads it: the drafter skips it
        if g is None:
            g = self.graphs[key] = MTPChainGraph(fn, self.st, n, d, cls)
            g.capture()
            self.captures += 1          # this round's draft time is the capture's (the drafter skips it)
        b = fn._mtp_bufs(d)
        b["stack"][:n].copy_(rows)
        t0 = time.perf_counter()
        assert fn.stage_mtp(self.st, nexts, d) == cls
        fn.host_ms["stage_mtp"] = fn.host_ms.get("stage_mtp", 0.0) + 1e3 * (time.perf_counter() - t0)
        g.graph.replay()
        self._n = n

    @staticmethod
    def run_packed(heads: list) -> None:
        """SPD-235: the staged drafts of these heads (one engine) in one packed chain."""
        heads[0].fn.mtp_chain_multi([(h.key, h.st, h._n, h.depth, h._cls) for h in heads])

    def read(self) -> tuple[list, list]:
        """After `launch`: per depth the (first, second) token and their probabilities; the fed rows
        join the head's cache."""
        self.fn._mtp_key = self.key
        torch.cuda.current_stream().synchronize()
        self.st.length += self._n
        d = getattr(self, "_d", self.depth)
        b = self.fn._mtp_bufs(d)
        return b["tok_h"][:d].tolist(), b["prob_h"][:d].tolist()


class FlashNextMTPDrafter(MTPTreeDrafter):
    """The shared tree drafter over a Flash-Next MTP head with a depth ladder (`FlashNextMTPHead.depths`,
    QWEN38_FN_DEPTH_LADDER): after each drafted round the head's next chain is the smallest rung at or above
    DEPTH_A x the recent accepted drafts (an EMA) + DEPTH_B, and a round that took every draft goes a rung up at once.
    Without a ladder it is the shared drafter."""

    #: the EMA weight of a drafted round's accepted drafts, and the depth asked for over it (a x EMA + b: prose at
    #: ~1.2 accepted drafts 3, code at ~4 drafts 7-8)
    DEPTH_EMA, DEPTH_A, DEPTH_B = 0.25, 1.5, 1.0

    def _pick_depth(self, accepted: int) -> None:
        h = self.head
        ladder = getattr(h, "depths", ())
        if len(ladder) < 2:
            return
        e = getattr(self, "_acc_ema", None)
        e = self._acc_ema = accepted if e is None else e + self.DEPTH_EMA * (accepted - e)
        want = self.DEPTH_A * e + self.DEPTH_B
        d = next((r for r in ladder if r >= want), ladder[-1])
        if accepted >= h.cur:
            d = max(d, next((r for r in ladder if r > h.cur), ladder[-1]))
        h.cur = d

    def observe(self, tokens: list[int]) -> None:
        if self.round is not None and self._drafted:
            self._pick_depth(len(tokens) - 1)
        super().observe(tokens)

