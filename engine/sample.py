"""Sampling: the decoder the greedy fixed point cannot be (ENG-19).

The engine's exact path is greedy, and under greedy a repeated token is a stable point no
penalty can break (measured; see the knowledge card on greedy loops). The model is designed to
run with sampling -- temperature, top-p and the presence penalty are its own loop-breakers -- so
a request that asks for sampling must get real sampling, not a 400.

`temperature`, `top-k` and `top-p` are applied to a row AFTER the penalties and the no-repeat
rule (deterministic transforms, still meaningful under sampling), with a per-request
`torch.Generator` so a `seed` reproduces a request exactly.

Both halves are here. The single-token path samples one row (`__call__`). The speculative path
verifies a draft chain or a draft tree by rejection sampling (`chain_pick`, `tree_walk`): the
target's OWN token is drawn at each node, and a drafted token survives exactly while the draw
lands on it -- if the draw lands elsewhere, that draw IS the rejection's residual sample. This
is the textbook sampler specialised to a deterministic drafter: with proposal q = delta_d,

    accept with min(1, p(d)/q(d)) = p(d), and the residual (p - q)+ is p restricted to x != d,

which is precisely what "draw x ~ p; if x == d accept, else x is the token" computes -- the same
distribution, with no q to estimate and no second draw. The output therefore follows the
target's own sampled distribution, speculation or not, which is the property `verify_lossless`
checks for greedy and `test_sample.py` checks statistically here.

Greedy is untouched: `temperature = 0` is the argmax it always was, byte for byte.
"""

from __future__ import annotations

import os

import torch


class Sampler:
    """Temperature / top-k / top-p over rows of target logits, one per request."""

    __slots__ = ("temperature", "top_p", "top_k", "seed", "generator", "draft_temperature")

    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                 seed: int | None = None, draft_temperature: float | None = None):
        if temperature < 0.0 or not temperature == temperature:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = None if seed is None else int(seed)
        # ENG-102: how sharply the DRAFTER's proposal distribution is tempered, relative to the
        # request. A cooler draft is sharper, and `min(1, p(d)/q(d))` then approaches `p(d)`.
        # None -> the drafter's own default (QWEN38_DRAFT_TEMP).
        self.draft_temperature = None if draft_temperature is None else float(draft_temperature)
        if self.draft_temperature is not None and not 0.02 <= self.draft_temperature <= 1.0:
            raise ValueError(f"draft_temperature must be in [0.02, 1], got {draft_temperature}")
        self.generator: torch.Generator | None = None

    @property
    def on(self) -> bool:
        # `temperature = 0` is greedy whatever top-p and top-k say -- OpenAI's reading and vLLM's,
        # and the only one that makes sense: both filters shrink a candidate set the argmax is
        # already the maximum of. Reading a lone top-p as "sampling on" made
        # `{"temperature": 0, "top_p": 0.95}` sample the RAW logits, because `_filter` divides by
        # the temperature only when there is one, and it also took the request out of the response
        # cache, which the server consults only for a greedy answer.
        return self.temperature > 0.0

    def key(self) -> tuple:
        """Joins the response-cache key: sampled answers are not memoised at all today (the
        server skips the cache when sampling is on), but the identity is kept for completeness."""
        return (round(self.temperature, 6), round(self.top_p, 6), self.top_k, self.seed)

    def _rng(self, device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            seed = self.seed if self.seed is not None else int.from_bytes(os.urandom(8), "little")
            self.generator.manual_seed(seed)
        return self.generator

    @torch.no_grad()
    def _filter(self, logits: torch.Tensor) -> torch.Tensor:
        """The requested distribution over the last dim of a `[..., V]` float tensor.

        The top-p filter is the HF one: sort descending, keep the shortest prefix whose cumulative
        mass exceeds `top_p`, always keep at least the first token.
        """
        if self.temperature > 0.0:
            logits = logits / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, logits.shape[-1])
            cutoff = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        if self.top_p < 1.0:
            sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            drop = cumulative > self.top_p
            drop[..., 1:] = drop[..., :-1].clone()   # keep the first token that crosses
            drop[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, order, sorted_logits)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def probs_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """`[n, V]` distributions for a verified block's rows (penalties already applied)."""
        return self._filter(rows.float())

    def pick(self, probs_row: torch.Tensor) -> int:
        """One token from one distribution row, in generation order on the request's RNG."""
        return int(torch.multinomial(probs_row, 1, generator=self._rng(probs_row.device)))

    @torch.no_grad()
    def chain_accept(self, dists: torch.Tensor, draft: list[int],
                     qrows: list[torch.Tensor | None] | None = None) -> tuple[int, int]:
        """Accept a draft chain, q-aware where the drafter sampled its proposal (ENG-102).

        Each position either carries a real proposal distribution `q` -- the drafter sampled the
        token from it, so the textbook rule applies: accept with `min(1, p(d)/q(d))`, and on
        rejection draw from the residual `(p - q)+` renormalised -- or none, which is a
        deterministic drafter whose point mass `q = delta_d` makes the accept `p(d)` and the
        residual the draw itself (the `chain_pick` shortcut). Mixing the two per position is
        exact: every position's acceptance uses the distribution its token was proposed from.
        Returns `(accepted, first new token)`, the same contract as `chain_pick`.
        """
        for i, d in enumerate(draft):
            if qrows is None or qrows[i] is None:
                x = self.pick(dists[i])
                if x != d:
                    return i, x
                continue
            p_row, q_row = dists[i], qrows[i]
            pd, qd = float(p_row[d]), float(q_row[d])
            u = float(torch.rand(1, generator=self._rng(p_row.device),
                               device=p_row.device))
            if qd > 0.0 and u * qd < pd:
                continue
            r = torch.clamp(p_row - q_row, min=0.0)
            total = float(r.sum())
            x = self.pick(r / total) if total > 0.0 else self.pick(p_row)
            return i, x
        return len(draft), self.pick(dists[len(draft)])

    @torch.no_grad()
    def __call__(self, row: torch.Tensor) -> int:
        """One token from one logits row. `temperature = 0` is the argmax, exactly as before."""
        if not self.on:
            return int(row.argmax())
        return self.pick(self._filter(row.float()))

    def chain_pick(self, dists: torch.Tensor, draft: list[int]) -> tuple[int, int]:
        """Accept a draft chain by rejection sampling; returns `(accepted, first new token)`.

        `dists` is `[len(draft) + 1, V]`: every draft position's row plus the bonus row after the
        last draft. Accepted drafts keep the block's later rows valid; the first draw that lands
        off the draft ends the block there, and that draw is the token.
        """
        for i, d in enumerate(draft):
            x = self.pick(dists[i])
            if x != d:
                return i, x
        return len(draft), self.pick(dists[len(draft)])

    def tree_walk(self, dists: torch.Tensor, tokens: list[int], parents: list[int],
                  ) -> tuple[list[int], list[int]]:
        """The same accept, down a draft tree; returns `(node path, new tokens)`.

        At each node the target's own token is drawn and the walk follows the child carrying it.
        Where no child carries it, the draw is the token and the walk stops -- byte-for-byte the
        greedy `accept_tree` walk with the sample in place of the argmax.
        """
        path, node = [0], 0
        while True:
            x = self.pick(dists[node])
            nxt = next((c for c in range(node + 1, len(tokens))
                        if parents[c] == node and tokens[c] == x), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        new = [tokens[i] for i in path[1:]] + [x]
        return path, new
