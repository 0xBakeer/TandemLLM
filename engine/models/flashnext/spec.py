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

from engine.models.flashnext.engine import FlashNextEngine, MTPChainGraph
from engine.models.flashnext.ref import MTP, FNState


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

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        """The round's accepted path (rows of the block it proposed) and what the round cost."""


class SpecDecoder:
    """The round loop over one sequence. `graphs`: verify blocks from captured graphs (one per row
    count, captured on first use)."""

    def __init__(self, eng: FlashNextEngine, st: FNState, drafter: Drafter, graphs: bool = True,
                 log_union: bool = False):
        self.eng, self.st, self.drafter = eng, st, drafter
        self.log_union = log_union
        self.union: dict = {}          # rows -> [rounds, sum over layers of distinct experts read]
        self.graphs: dict | None = {} if graphs else None
        self.commit_graphs: dict | None = {} if graphs else None
        self.stats = {"rounds": 0, "rows": 0, "accepted": 0, "moved": 0, "draft_s": 0.0, "verify_s": 0.0,
                      "commit_s": 0.0}


    @torch.no_grad()
    def generate(self, first: int, new: int, eos: set, round_ms: list | None = None, emit=None) -> list[int]:
        """Tokens from `first` (the anchor after the prompt) until `new` tokens or an EOS, first included.
        `emit(tokens)` is called with each round's new tokens (after the cut at `new` or an EOS); a
        False return stops after that round. `self.full` keeps every emitted token uncut: the state
        has run (prompt + full)[:length]."""
        eng, st, s = self.eng, self.st, self.stats
        out = [first]
        self.full = [first]
        if emit is not None and emit([first]) is False:
            return out
        while len(out) < new and out[-1] not in eos:
            t0 = time.perf_counter()
            tokens, parents = self.drafter.propose(out[-1])
            t1 = time.perf_counter()
            target = eng.verify(tokens, parents, st, self.graphs)
            t2 = time.perf_counter()
            if self.log_union:
                self._union(len(tokens))
            path = walk(tokens, parents, target)
            stack = eng.commit(st, path, self.commit_graphs)
            emitted = [tokens[r] for r in path[1:]] + [target[path[-1]]]
            self.drafter.accepted(stack, emitted)
            torch.cuda.current_stream().synchronize()
            t3 = time.perf_counter()
            self.drafter.observe(path, (t3 - t1) * 1e3, (t1 - t0) * 1e3)
            s["rounds"] += 1
            s["rows"] += len(tokens)
            s["accepted"] += len(path) - 1
            s["moved"] += int(any(path[j] != j for j in range(len(path))))
            s["draft_s"] += t1 - t0
            s["verify_s"] += t2 - t1
            s["commit_s"] += t3 - t2
            if round_ms is not None:
                round_ms.append(((t3 - t0) * 1e3, len(emitted)))
            self.full += emitted
            n0 = len(out)
            for t in emitted:
                out.append(t)
                if t in eos or len(out) >= new:
                    break
            if emit is not None and emit(out[n0:]) is False:
                break
        return out


    def _union(self, M: int) -> None:
        """The distinct routed experts the verify block's rows read, summed over the MoE layers."""
        E = self.eng.cfg.num_experts
        ids = [v for k, v in self.eng.rec.items() if isinstance(k, tuple) and k[0] == "ids"]
        if not ids:
            return
        a = torch.stack([i[:M] for i in ids]).cpu()                    # [layers, M, k + 1]
        n = sum(len(set(x.tolist()) - {E}) for x in a.view(a.shape[0], -1))
        u = self.union.setdefault(M, [0, 0])
        u[0] += 1
        u[1] += n


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


class StairCut:
    """StairCut on Flash-Next's MTP drafts: each round keeps the part of the draft that maximises
    expected committed tokens per millisecond,

        (1 + sum of q over the kept nodes) / (V(rows) + draft),

    over a spine (the MTP's greedy chain) and, with `siblings`, one leaf per depth (the MTP's second
    token at that depth, a child of the spine node above it). q of a spine node is the chance the walk
    reaches and takes it: the product of the calibrated first-token probabilities down to it; q of a
    sibling at depth k is the chance the walk reaches depth k, rejects the spine token there and takes
    the sibling. Calibration maps the draft head's probability to the measured rate of acceptance, per
    tenth of probability, starting from the probability itself (a prior of 4 rounds a bin) and learned
    online from every verified round. V(rows) is the measured verify-and-commit time a row count costs,
    started from the 2026-10-11 bench (Flash-Next 3.05, 1k) and followed online (EMA 0.1); the counts
    not measured yet follow the prior shifted by the learned offset (a long context moves every
    count), and the prices never fall with more rows. A verify of N rows reads the union of the
    experts its rows route to, so the price is a ramp, not a flat stair.
    """

    V0 = {1: 24.0, 2: 27.9, 3: 30.6, 4: 34.6, 5: 37.4, 6: 41.3}   # verify + commit, ms (bench 2026-10-11)
    STEP = 3.7                                                    # ms a row past the table

    def __init__(self, siblings: bool = True, max_rows: int = 32):
        self.siblings, self.max_rows = siblings, max_rows
        self.V: dict = {}              # learned verify + commit ms by row count (EMA 0.1)
        self.delta = 0.0               # learned shift of the prior table (the context moves it)
        self.draft_ms = None
        self.cal = {"spine": [[0.0, 0.0] for _ in range(10)], "sib": [[0.0, 0.0] for _ in range(10)]}
        self.last = None
        self.frozen = False            # observe nothing (a warm-up run of a measurement)

    def prior(self, rows: int) -> float:
        top = max(self.V0)
        return self.V0[rows] if rows in self.V0 else self.V0[top] + self.STEP * (rows - top)

    def prices(self, n: int) -> list[float]:
        """The verify + commit price of 1..n rows: learned where measured, else the prior table
        shifted by the learned offset; never cheaper for more rows (a cut must not buy a row for
        nothing because one count was measured at a busier moment than the next)."""
        out, p = [0.0], 0.0
        for r in range(1, n + 1):
            p = max(p, self.V.get(r, self.prior(r) + self.delta))
            out.append(p)
        return out

    def price(self, rows: int) -> float:
        return self.prices(rows)[rows]

    def c(self, kind: str, p: float) -> float:
        b = min(int(p * 10), 9)
        hits, n = self.cal[kind][b]
        return (hits + 4 * p) / (n + 4)

    def cut(self, anchor: int, tok: list, prob: list) -> tuple[list[int], list[int]]:
        """tok / prob: per depth the (first, second) draft token and its probability."""
        D = len(tok)
        reach, spine_q, sib_q = 1.0, [], []
        for k in range(D):
            c1 = self.c("spine", prob[k][0])
            sib_q.append(reach * (1 - c1) * self.c("sib", prob[k][1]) if self.siblings else 0.0)
            reach *= c1
            spine_q.append(reach)
        draft = self.draft_ms or 0.0
        price = self.prices(min(self.max_rows, 2 * D + 1))
        best = (1.0 / (price[1] + draft), 0, ())
        for n in range(1, D + 1):
            gain = 1.0 + sum(spine_q[:n])
            rows = 1 + n
            if rows > self.max_rows:
                break
            score = gain / (price[rows] + draft)
            if score > best[0]:
                best = (score, n, ())
            # siblings at depths 1..n + 1 (a sibling's parent is the spine node above it), best first
            sibs = []
            for k in sorted(range(min(n + 1, D)), key=lambda k: -sib_q[k]):
                if rows + 1 > self.max_rows or sib_q[k] <= 0.0:
                    break
                g2, r2 = gain + sib_q[k], rows + 1
                s2 = g2 / (price[r2] + draft)
                if s2 <= score:
                    break
                gain, rows, score = g2, r2, s2
                sibs.append(k)
                if score > best[0]:
                    best = (score, n, tuple(sibs))
        _, n, sibs = best
        # DFS pre-order: a depth's sibling leaf before the spine child that continues the chain
        tokens, parents, kind = [anchor], [-1], [None]
        spine_row = 0
        for k in range(n + 1):
            if k in sibs:
                tokens.append(tok[k][1])
                parents.append(spine_row)
                kind.append(("sib", k))
            if k < n:
                tokens.append(tok[k][0])
                parents.append(spine_row)
                kind.append(("spine", k))
                spine_row = len(tokens) - 1
        self.last = (kind, prob, len(tokens))
        return tokens, parents

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        if self.frozen:
            return
        kind, prob, rows = self.last
        on = {kind[r] for r in path[1:]}
        for r in range(1, rows):
            t, k = kind[r]
            reached = all(("spine", i) in on for i in range(k))
            if not reached:
                continue
            if t == "spine":
                self._learn("spine", prob[k][0], ("spine", k) in on)
            elif ("spine", k) not in on:
                self._learn("sib", prob[k][1], ("sib", k) in on)
        verify = round_ms
        self.V[rows] = verify if rows not in self.V else 0.9 * self.V[rows] + 0.1 * verify
        self.delta = 0.9 * self.delta + 0.1 * (verify - self.prior(rows))
        self.draft_ms = draft_ms if self.draft_ms is None else 0.9 * self.draft_ms + 0.1 * draft_ms

    def _learn(self, kind: str, p: float, hit: bool) -> None:
        b = self.cal[kind][min(int(p * 10), 9)]
        b[0] += float(hit)
        b[1] += 1.0
        if b[1] > 400:                         # follow the text: halve old evidence
            b[0] *= 0.5
            b[1] *= 0.5


class MTPGraphDrafter(Drafter):
    """The MTP layer as a drafter from captured graphs (ENG-153): a round's draft -- the feed of the
    committed rows, `depth` - 1 deeper steps, every head and top-2 -- is one replay of the graph for
    that feed's row count; the host reads the drafts once. Without a `cut` the block is the greedy
    chain of `depth`; with one (`StairCut`) the block is the chain and sibling leaves it keeps."""

    def __init__(self, eng: FlashNextEngine, depth: int, cap: int, cut: StairCut | None = None):
        self.eng, self.depth, self.cut = eng, depth, cut
        self.st = eng.new_mtp_state(cap)
        self.graphs: dict = {}
        self.pending = None

    def prime(self, stack: torch.Tensor, ids: list[int], first: int) -> None:
        L = len(ids)
        if L > 1:
            for i in range(0, L - 1, 2048):            # the reference's forward, in prompt-sized chunks
                j = min(i + 2048, L - 1)
                self.eng.mtp_feed(stack[i:j], torch.tensor(ids[i + 1:j + 1]), self.st)
                if j > 4096:
                    torch.cuda.empty_cache()           # as FlashNextEngine.prefill
            self.eng.build_pool(self.st, layers=[MTP])
        self.pending = (stack[L - 1:L], [first])

    def capture_all(self) -> None:
        """The graphs of every feed a round can bring (1 .. depth + 1 committed rows) at the current
        context class."""
        cls = self.eng.qsa_class(self.st.length + self.depth + 1)
        for n in range(1, self.depth + 2):
            if (n, cls) not in self.graphs:
                self.graphs[(n, cls)] = MTPChainGraph(self.eng, self.st, n, self.depth, cls)
                self.graphs[(n, cls)].capture()

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        eng = self.eng
        rows, nexts = self.pending
        n = len(nexts)
        cls = eng.qsa_class(self.st.length + n + self.depth - 1)
        g = self.graphs.get((n, cls))
        if g is None:
            g = self.graphs[(n, cls)] = MTPChainGraph(eng, self.st, n, self.depth, cls)
            g.capture()
        b = eng._mtp_bufs(self.depth)
        b["stack"][:n].copy_(rows)
        t0 = time.perf_counter()
        assert eng.stage_mtp(self.st, nexts, self.depth) == cls
        eng.host_ms["stage_mtp"] = eng.host_ms.get("stage_mtp", 0.0) + 1e3 * (time.perf_counter() - t0)
        g.graph.replay()
        torch.cuda.current_stream().synchronize()
        self.st.length += n
        tok = b["tok_h"][:self.depth].tolist()
        if self.cut is None:
            return [anchor] + [t[0] for t in tok], list(range(-1, self.depth))
        return self.cut.cut(anchor, tok, b["prob_h"][:self.depth].tolist())

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        self.pending = (stack, nexts)

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        if self.cut is not None:
            self.cut.observe(path, round_ms, draft_ms)
