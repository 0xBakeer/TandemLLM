"""A model's MTP head as the server loop's tree drafter, written once (ENG-245).

The loop syncs the rows it committed (`sync(tokens, streams, pos)`); a row's MTP input pairs its
streams with the token after it, known when the next row arrives or, for the last row, when the next
proposal names its anchor. The backlog goes into the head's cache in one call (`head.prime`) except
the last `depth` + 1 rows, which ride the round's own feed (`head.launch`); the round's draft is cut by
StairCut (engine/methods/staircut.py), which learns from the verify the loop timed (`on_verify`).

What it asks of the head (Flash-Next's: engine/models/flashnext/mtp.py): `depth`, `reset()`,
`prime(rows, nexts)`, `capture_all()`, `launch(rows, nexts)` (asynchronous), `read()` -> per depth
the top-2 tokens and their probabilities, `views()` and `resume(n)`. Of the engine: `captured` (the
last block captured a verify or commit graph), `commit_ms` and `last_streams()` (the last committed
row's streams, a section of its cache state).

A lookup source (`ngram`, engine/drafters/ngram.py: the request's own text, the corpus and the server's
suffix store, which the server hands to it through `add_store`) adds its best continuation to the
round's cut (ENG-229): the head and the lookup in one tree, one verify. It is asked while the draft
runs, as deep as its match allows.

The caches (ENG-157) keep the head's cache with the engine's rows: `kv_views` (the resident prefix,
the disk tier, the spill), `state_snapshot` / `state_restore` (the state store) and `state_resume(n)`
(a resume in place). After a resume at n the head holds rows below n - 1 and the row at n - 1 waits for
its next token, as it did when the rows were written. A row the head had not fed when a cache took
it is stale; it changes a draft, never the text (the verify decides).
"""

from __future__ import annotations

import time

import torch

from engine.drafters import run_steps
from engine.methods.staircut import StairCut
from engine.tree import DraftTree


class MTPTreeDrafter:
    name = "mtp"
    last_q = None
    wants_rows = False

    def __init__(self, eng, head, cut: StairCut, ngram=None, look_depth: int = 16):
        self.s, self.head, self.cut, self.depth = eng, head, cut, head.depth
        self.ngram, self.look_depth = ngram, look_depth
        self.pending = None                 # (streams, next tokens) of the round's own feed
        self.held = None                    # (tokens, streams) of the last rows synced
        self.queue: list = []               # (streams, next tokens) not fed yet
        self.round = None                   # the draft ms of the block awaiting its verify
        self.verify_ms = None
        self.stats = {"rounds": 0, "draft_ms": 0.0, "fed_rows": 0, "looked": 0}

    def report(self) -> str:
        """The verbose server's line at each request: rounds, draft ms, StairCut's prices and offset."""
        c = self.cut
        V, delta = c.prices.learned(c.kind)
        return (f"mtp rounds={self.stats['rounds']} fed={self.stats['fed_rows']} looked={self.stats['looked']} "
                f"draft_ms={c.draft_ms if c.draft_ms is None else round(c.draft_ms, 2)} delta={round(delta, 2)} "
                f"V={ {r: round(v, 1) for r, v in sorted(V.items())} } "
                f"spine_cal={[round(c.cal.q('spine', (b + 0.5) / 10), 2) for b in range(10)]}")

    def capture_all(self) -> None:
        self.head.capture_all()

    # --- the loop's drafter interface -------------------------------------------------------------
    def reset(self) -> None:
        self.head.reset()
        if self.ngram is not None:
            self.ngram.reset()
        self.pending = None
        self.held, self.queue, self.round, self.verify_ms = None, [], None, None

    def prime(self, tokens: list[int]) -> None:
        # the prompt's rows come through `sync`; the lookup indexes its tokens
        if self.ngram is not None:
            self.ngram.prime(tokens)

    def set_sampling(self, sampler) -> None:
        pass                                # the draft is the greedy tree; a sampled walk draws on it

    def sync(self, tokens: list[int], streams: torch.Tensor, pos: int, rows=None) -> None:
        toks = [int(t) for t in tokens]
        h = streams.reshape(len(toks), -1)
        if h.shape[0] <= 64:
            h = h.clone()                   # a static buffer the next forward writes again
        if self.held is not None:
            ht, hs = self.held
            self.queue.append((hs, ht[1:] + [toks[0]]))
        self.held = (toks, h)

    def observe(self, tokens: list[int]) -> None:
        if self.ngram is not None:
            self.ngram.observe(tokens)
        # a round that captured a graph took the capture's time: StairCut learns nothing from it
        # (its price for that row count would start at ~170 ms and the cut would never ask again)
        if self.round is not None and self.verify_ms is not None and not self.s.captured:
            self.cut.observe(list(range(len(tokens))), self.verify_ms + self.s.commit_ms, self.round)
        self.round, self.verify_ms = None, None

    def on_verify(self, rows: int, ms: float) -> None:
        self.verify_ms = ms

    def propose_tree(self, context: list[int], k: int) -> DraftTree | None:
        return run_steps(self.propose_tree_steps(context, k))

    def propose_tree_steps(self, context: list[int], k: int):
        """The block for the round anchored at context[-1]: the backlog fed, the draft launched (the
        generator stops there so the loop streams the last block meanwhile), the cut."""
        t0 = time.perf_counter()
        if not self._feed(int(context[-1])) or k < 1:
            return None
        rows, nexts = self.pending
        self.pending = None
        self.head.launch(rows, nexts)
        host = time.perf_counter() - t0
        yield                               # the loop streams the last block while the draft runs
        t1 = time.perf_counter()
        look = self._look(context)
        tok, prob = self.head.read()
        toks, parents = self.cut.cut(int(context[-1]), tok, prob, look)
        tree = DraftTree(tokens=toks, parents=parents).truncate(int(k) + 1)
        # the draft's own cost: its host work on both sides of the stop, not the streaming between
        # (StairCut priced the gap as drafting and stopped drafting, 2026-10-11)
        self.round = (host + time.perf_counter() - t1) * 1e3
        self.stats["rounds"] += 1
        self.stats["draft_ms"] += self.round
        return tree

    def _look(self, context: list[int]) -> tuple | None:
        """The lookup's best continuation and its share of the vote (smoothed as the lookup's own
        tree is), or None when nothing matches."""
        if self.ngram is None:
            return None
        _, cands = self.ngram.candidates(context, self.look_depth)
        if not cands:
            return None
        self.stats["looked"] += 1
        total = sum(w for _, w in cands)
        return cands[0][0], cands[0][1] / (total + self.ngram.alpha)

    # --- the caches (ENG-157) -------------------------------------------------------------------
    def kv_views(self) -> list:
        return self.head.views()

    def snapshot_bytes_per_token(self) -> int:
        return sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in self.head.views())

    def state_resume(self, n: int) -> None:
        n = int(n)
        self.head.resume(max(0, n - 1))
        self.pending, self.queue, self.round, self.verify_ms = None, [], None, None
        self.held = ([0], self.s.last_streams().clone()) if n > 0 else None

    def state_snapshot(self) -> dict:
        """The head's cache after every synced row that knows its next token went in."""
        self._flush()
        L = int(self.head.st.length)
        return {"len": L, "views": tuple(t.narrow(ax, 0, L).clone() for t, ax in self.head.views())}

    def state_restore(self, snap: dict) -> None:
        L = int(snap["len"])
        for (t, ax), x in zip(self.head.views(), snap["views"]):
            t.narrow(ax, 0, L).copy_(x)
        self.head.resume(L)
        self.pending, self.queue, self.round, self.verify_ms = None, [], None, None
        self.held = ([0], self.s.last_streams().clone())

    # --- internals -------------------------------------------------------------------------------
    def _flush(self) -> None:
        """Feed the queue and the round's own feed now (every row there knows its next token)."""
        rows = [q[0] for q in self.queue] + ([self.pending[0]] if self.pending is not None else [])
        nexts = [t for q in self.queue for t in q[1]] + (list(self.pending[1]) if self.pending is not None else [])
        self.queue, self.pending = [], None
        if nexts:
            self.head.prime(torch.cat(rows) if len(rows) > 1 else rows[0], nexts)

    def _feed(self, anchor: int) -> bool:
        """Every synced row paired with its next token (the last with the anchor); all but the last
        depth + 1 into the head's cache now, those into the round's own feed. False: nothing synced."""
        if self.held is not None:
            ht, hs = self.held
            self.queue.append((hs, ht[1:] + [anchor]))
            self.held = None
        if not self.queue:
            return self.pending is not None
        rows = torch.cat([q[0] for q in self.queue]) if len(self.queue) > 1 else self.queue[0][0]
        nexts = [t for q in self.queue for t in q[1]]
        self.queue = []
        n, keep = len(nexts), min(len(nexts), self.depth + 1)
        if n > keep:
            self.head.prime(rows[:n - keep], nexts[:n - keep])
        self.pending = (rows[n - keep:], nexts[n - keep:])
        self.stats["fed_rows"] += n
        return True
