"""Deterministic anti-repetition penalties, exact under speculative verification.

The engine decodes greedily and refuses to sample, so when the model's argmax settles on one
token there is nothing in the decoder that can break the loop -- measured 2026-09-18, a "Super
Jump Bros" prompt whose pasted level rows are literal `" C C C …"` strings looped ~7,000 `C`
tokens inside the reasoning block (ENG-17, the diagnosis note in Memo).

The fix is the one vLLM and SGLang ship: penalties on the TARGET's logits, before the argmax.
The property that makes that safe here is the one the whole engine already rests on --
acceptance is everywhere "draft token == target argmax" -- so any deterministic function of the
committed history applied to the row before argmax keeps the speculative path's output identical
to the unspeculated one, *under the same rule*. vLLM applies penalties to the target logits before
the rejection argmax (`v1/sample/rejection_sampler.py`); SGLang does the same in "a relaxed
version" because its verifier is not argmax-exact. This one is exact:

  * chain blocks -- vLLM's per-position history: row i of `[anchor] + draft` is conditioned on
    the committed history plus `draft[:i]`, so rows are penalized left to right, one draft token
    entering the working counts after the row that precedes it;
  * tree blocks -- ancestor history: a node is conditioned on its ANCESTOR PATH, not its
    siblings, and the rows arrive in DFS pre-order, so a depth stack adds each node's token on
    the way down and removes it on the way back up;
  * rollback costs nothing: only accepted tokens are ever committed to the base counts, so a
    rejected block leaves no penalty state behind.

Defaults are the identities -- rep 1.0, presence 0, frequency 0 -- and an off spec short-circuits
every entry point, so a run that does not ask for penalties pays nothing and produces the bytes
it produced before this file existed. The response cache key gains the three values, because
greedy-under-penalties is a different function.

The one pass costs a handful of elementwise kernels over a 248,320-wide row (the count vector is
1 MiB of int32), one per verify row, against a ~99 ms verify; the budget is < 1 ms a block and
`tools/profile_cycle.py` is the gate that checks it.
"""

from __future__ import annotations

import torch


class PenaltySpec:
    """The three standard penalties. Defaults are today's output."""

    __slots__ = ("rep", "presence", "freq")

    def __init__(self, rep: float = 1.0, presence: float = 0.0, freq: float = 0.0):
        if rep <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {rep}")
        if not -2.0 <= presence <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {presence}")
        if not -2.0 <= freq <= 2.0:
            raise ValueError(f"frequency_penalty must be in [-2, 2], got {freq}")
        self.rep = float(rep)
        self.presence = float(presence)
        self.freq = float(freq)

    @property
    def on(self) -> bool:
        return self.rep != 1.0 or self.presence != 0.0 or self.freq != 0.0

    def key(self) -> tuple:
        """The identity that joins the response-cache key: two runs with the same key answer the
        same question, and two runs with different keys do not."""
        return (round(self.rep, 6), round(self.presence, 6), round(self.freq, 6))

    def __repr__(self) -> str:
        return (f"PenaltySpec(rep={self.rep:g}, presence={self.presence:g}, "
                f"freq={self.freq:g})")


class PenaltyState:
    """The count vector behind the penalties, one per request, seeded from the prompt.

    `seed` and `commit` take token ids (python ints); `apply_*` take the verify's logits rows and
    penalize them IN PLACE, so the caller computes picks/argmax afterwards exactly as it did
    before. An off spec makes every method a no-op and allocates nothing.
    """

    def __init__(self, spec: PenaltySpec, vocab_size: int, device: str):
        self.spec = spec
        self.vocab = int(vocab_size)
        self.device = device
        self.counts: torch.Tensor | None = None      # committed history, int32 [vocab]
        self._work: torch.Tensor | None = None        # per-block scratch, a copy of counts

    # --- history ------------------------------------------------------------------------------
    def seed(self, ids) -> None:
        """Start a request: RESET, then add. `seed` is what a fresh request begins with, so a
        reused state (the gate tools run plain and speculative back to back) starts clean."""
        if not self.spec.on:
            return
        if self.counts is None:
            self.counts = torch.zeros(self.vocab, dtype=torch.int32, device=self.device)
        else:
            self.counts.zero_()
        self._add(ids)

    def commit(self, ids) -> None:
        self._add(ids)

    @torch.no_grad()
    def _add(self, ids) -> None:
        if not self.spec.on or not ids:
            return
        if self.counts is None:
            self.counts = torch.zeros(self.vocab, dtype=torch.int32, device=self.device)
        self.counts.index_add_(0, torch.tensor(list(ids), dtype=torch.long, device=self.device),
                               torch.ones(len(ids), dtype=torch.int32, device=self.device))

    # --- rows ---------------------------------------------------------------------------------
    @torch.no_grad()
    def apply_single(self, row: torch.Tensor) -> None:
        """A one-row decision: the prefill's first token, a declined block's step, the token
        after a forced reasoning close. History is the committed counts, nothing else."""
        if not self.spec.on:
            return
        assert self.counts is not None, "apply before seed"
        self._apply(row, self.counts)

    @torch.no_grad()
    def apply_chain(self, lg: torch.Tensor, draft: list[int]) -> None:
        """Rows of `forward_block([anchor] + draft)`, in place, left to right.

        Row i is conditioned on the committed history plus `draft[:i]` (the anchor is committed
        and already in the counts), so each draft token joins the working copy after the row
        above it has been penalized. `lg` has `1 + len(draft)` rows.
        """
        if not self.spec.on:
            return
        assert self.counts is not None, "apply before seed"
        if self._work is None:
            self._work = torch.empty_like(self.counts)
        self._work.copy_(self.counts)
        for i in range(lg.shape[0]):
            self._apply(lg[i], self._work)
            if i < len(draft):
                self._work[int(draft[i])] += 1

    @torch.no_grad()
    def apply_tree(self, lg: torch.Tensor, tree) -> None:
        """Rows of `forward_tree`, in place, each with its ANCESTOR history.

        DFS pre-order means a node's ancestors are the stack's contents after popping back to
        its depth; siblings leave nothing behind. Node 0 is the anchor and is already committed,
        so it is in the base counts and is not pushed again.
        """
        if not self.spec.on:
            return
        assert self.counts is not None, "apply before seed"
        depths = tree.depths()
        if self._work is None:
            self._work = torch.empty_like(self.counts)
        self._work.copy_(self.counts)
        path: list[int] = []
        for j in range(lg.shape[0]):
            d = depths[j]
            while len(path) > d:
                self._work[path.pop()] -= 1
            self._apply(lg[j], self._work)
            if j > 0:
                t = int(tree.tokens[j])
                self._work[t] += 1
                path.append(t)

    @torch.no_grad()
    def _apply(self, row: torch.Tensor, counts: torch.Tensor) -> None:
        """The three penalties, on one row, against one count vector.

        rep is multiplicative (the CTRL/HF form: a negative logit is multiplied by it, a
        positive one divided by it), presence is a flat subtraction on tokens that occur at all,
        frequency subtracts per occurrence. Applied only to tokens the history contains, which is
        what makes presence and frequency differ and what makes rep a no-op on fresh vocabulary.
        """
        s = self.spec
        m = counts > 0
        if s.rep != 1.0:
            neg = row < 0
            row[m & neg] *= s.rep
            row[m & ~neg] /= s.rep
        if s.presence != 0.0:
            row[m] -= s.presence
        if s.freq != 0.0:
            row[m] -= s.freq * counts[m].to(row.dtype)
