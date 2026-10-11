"""Greedy speculative decoding on Flash-Next (ENG-227, ENG-153): draft, verify, walk, commit.

A round starts at the anchor -- the last token the target chose, not yet run through the trunk -- at
the state's length L. A drafter proposes a block in DFS pre-order with the anchor as row 0
(`engine/tree.py`'s convention: `tokens`, `parents`). One verify (`FlashNextEngine.verify`) gives the
target's greedy token at every row; the walk follows the rows whose token is the target's choice at
their parent; the commit keeps that path. The round emits the path's drafted tokens and the target's
token after the path's last row, which is the next anchor.

Every verified row is the decode step's row at that position bit for bit (the row-invariant kernels
and the path-aware sequence kernels), and the path's state is the decode walk's, so the emitted text
is the plain greedy text token for token whatever the drafter proposes. `tools/fn_spec_check.py`
proves it on the board.
"""

from __future__ import annotations

import time

import torch

from engine.models.flashnext.engine import FlashNextEngine, MTPChainGraph, VerifyGraph
from engine.models.flashnext.ref import FNState


def walk(tokens: list[int], parents: list[int], target: list[int]) -> list[int]:
    """The accepted path: from row 0, move to the child that carries the target's token at the
    current row (the first such child; a block holds a token once among siblings)."""
    children: dict = {}
    for i, p in enumerate(parents):
        if p >= 0:
            children.setdefault(p, []).append(i)
    path = [0]
    while True:
        nxt = [c for c in children.get(path[-1], ()) if tokens[c] == target[path[-1]]]
        if not nxt:
            return path
        path.append(nxt[0])


class Drafter:
    """What the loop asks of a draft source."""

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        """A block for the round: (tokens, parents), tokens[0] = the anchor, DFS pre-order."""
        raise NotImplementedError

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        """The round's committed rows: their streams [n, hc H] (positions L..L+n-1) and the token that
        follows each of them (the path's drafted tokens, then the new anchor)."""


class SpecDecoder:
    """The round loop over one sequence. `graphs`: verify blocks from captured graphs (one per row
    count, captured on first use)."""

    def __init__(self, eng: FlashNextEngine, st: FNState, drafter: Drafter, graphs: bool = True):
        self.eng, self.st, self.drafter = eng, st, drafter
        self.graphs: dict | None = {} if graphs else None
        self.commit_graphs: dict | None = {} if graphs else None
        self.stats = {"rounds": 0, "rows": 0, "accepted": 0, "moved": 0, "draft_s": 0.0, "verify_s": 0.0,
                      "commit_s": 0.0}

    def _graph(self, M: int):
        if self.graphs is not None and M not in self.graphs:
            self.graphs[M] = VerifyGraph(self.eng, self.st, M)
        return self.graphs

    @torch.no_grad()
    def generate(self, first: int, new: int, eos: set, round_ms: list | None = None) -> list[int]:
        """Tokens from `first` (the anchor after the prompt) until `new` tokens or an EOS, first included."""
        eng, st, s = self.eng, self.st, self.stats
        out = [first]
        while len(out) < new and out[-1] not in eos:
            t0 = time.perf_counter()
            tokens, parents = self.drafter.propose(out[-1])
            t1 = time.perf_counter()
            target = eng.verify(tokens, parents, st, self._graph(len(tokens)))
            t2 = time.perf_counter()
            path = walk(tokens, parents, target)
            stack = eng.commit(st, path, self.commit_graphs)
            emitted = [tokens[r] for r in path[1:]] + [target[path[-1]]]
            self.drafter.accepted(stack, emitted)
            torch.cuda.current_stream().synchronize()
            t3 = time.perf_counter()
            s["rounds"] += 1
            s["rows"] += len(tokens)
            s["accepted"] += len(path) - 1
            s["moved"] += int(any(path[j] != j for j in range(len(path))))
            s["draft_s"] += t1 - t0
            s["verify_s"] += t2 - t1
            s["commit_s"] += t3 - t2
            if round_ms is not None:
                round_ms.append(((t3 - t0) * 1e3, len(emitted)))
            for t in emitted:
                out.append(t)
                if t in eos or len(out) >= new:
                    break
        return out


class MTPChainDrafter(Drafter):
    """The pack's MTP layer as a greedy chain drafter (eager): each round feeds the committed rows
    (kept in its own cache) and drafts `depth` tokens from the last row's own streams, the deeper
    feeds rolled back."""

    def __init__(self, eng: FlashNextEngine, depth: int, cap: int):
        self.eng, self.depth = eng, depth
        self.st = FNState(cap=cap)
        self.pending = None

    def prime(self, stack: torch.Tensor, ids: list[int], first: int) -> None:
        """After the prompt: its streams [L, hc H] and tokens; `first` follows the last one."""
        L = len(ids)
        if L > 1:
            self.eng.mtp_feed(stack[: L - 1], torch.tensor(ids[1:L]), self.st)
        self.pending = (stack[L - 1:L], [first])

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        eng = self.eng
        rows, nexts = self.pending
        logits, h = eng.mtp_feed(rows, torch.tensor(nexts), self.st)
        keep = self.st.length
        out = [int(logits[-1].argmax())]
        for _ in range(self.depth - 1):
            logits, h = eng.mtp_feed(h[-1:], torch.tensor([out[-1]]), self.st)
            out.append(int(logits[-1].argmax()))
        self.st.length = keep
        self.st.clamp_pool()
        return [anchor] + out, list(range(-1, self.depth))

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        self.pending = (stack, nexts)


class MTPGraphDrafter(Drafter):
    """The MTP layer as a greedy chain drafter from captured graphs (ENG-153): a round's draft --
    the feed of the committed rows, `depth` - 1 deeper steps, every head and argmax -- is one replay
    of the graph for that feed's row count; the host reads the drafts once."""

    def __init__(self, eng: FlashNextEngine, depth: int, cap: int):
        self.eng, self.depth = eng, depth
        self.st = eng.new_mtp_state(cap)
        self.graphs: dict = {}
        self.pending = None

    def prime(self, stack: torch.Tensor, ids: list[int], first: int) -> None:
        L = len(ids)
        if L > 1:
            self.eng.mtp_feed(stack[: L - 1], torch.tensor(ids[1:L]), self.st)
        self.pending = (stack[L - 1:L], [first])

    def capture_all(self) -> None:
        """The graphs of every feed a round can bring (1 .. depth + 1 committed rows)."""
        for n in range(1, self.depth + 2):
            if n not in self.graphs:
                self.graphs[n] = MTPChainGraph(self.eng, self.st, n, self.depth)
                self.graphs[n].capture()

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        eng = self.eng
        rows, nexts = self.pending
        n = len(nexts)
        g = self.graphs.get(n)
        if g is None:
            g = self.graphs[n] = MTPChainGraph(eng, self.st, n, self.depth)
            g.capture()
        b = eng._mtp_bufs(self.depth)
        b["stack"][:n].copy_(rows)
        eng.stage_mtp(self.st, nexts, self.depth)
        g.graph.replay()
        torch.cuda.current_stream().synchronize()
        self.st.length += n
        return [anchor] + b["drafts_h"][:self.depth].tolist(), list(range(-1, self.depth))

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        self.pending = (stack, nexts)
