"""Sampling: the decoder the greedy fixed point cannot be (ENG-19, half one).

The engine's exact path is greedy, and under greedy a repeated token is a stable point no
penalty can break (measured; see the knowledge card on greedy loops). The model is designed to
run with sampling -- temperature, top-p and the presence penalty are its own loop-breakers -- so
a request that asks for sampling must get real sampling, not a 400.

This first half samples the target's own distribution on the single-token path: `temperature`,
`top-k` and `top-p`, applied to the row AFTER the penalties and no-repeat rule (which are
deterministic transforms and stay meaningful under sampling), with a per-request `torch.Generator`
so a `seed` reproduces a request exactly.

The second half -- rejection sampling under the speculative verify, where a draft is accepted
with `min(1, p/q)` and a rejection is corrected from `(p - q)+` -- is not built yet, so a sampled
request decodes WITHOUT a drafter and runs at the no-drafter rate (~9-25 tok/s). Greedy is
untouched: `temperature = 0` is the argmax it always was, byte for byte.

The top-p filter is the HF one: sort descending, keep the shortest prefix whose cumulative mass
exceeds `top_p`, always keep at least the first token.
"""

from __future__ import annotations

import os

import torch


class Sampler:
    """Temperature / top-k / top-p over one row of target logits, one per request."""

    __slots__ = ("temperature", "top_p", "top_k", "seed", "generator")

    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                 seed: int | None = None):
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
        self.generator: torch.Generator | None = None

    @property
    def on(self) -> bool:
        return self.temperature > 0.0 or self.top_p < 1.0 or self.top_k > 0

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
    def __call__(self, row: torch.Tensor) -> int:
        """One token from one row. `temperature = 0` is the argmax, exactly as before."""
        if not self.on:
            return int(row.argmax())
        logits = row.float()
        if self.temperature > 0.0:
            logits = logits / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, logits.numel())
            cutoff = torch.topk(logits, k).values[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        if self.top_p < 1.0:
            sorted_logits, order = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            drop = cumulative > self.top_p
            drop[1:] = drop[:-1].clone()       # keep the first token that crosses the threshold
            drop[0] = False
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter_(0, order, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self._rng(row.device)))
