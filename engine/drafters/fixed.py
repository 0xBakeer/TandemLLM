"""A drafter that proposes deliberately wrong tokens.

Its only purpose is to force the verify path down every branch it has: a block where nothing is
accepted, a block where some prefix is accepted, a block where everything is. A speculative decoder
that is correct must produce exactly the same greedy output with this drafter attached as with no
drafter at all, and it must do so at every rejection point, which a good drafter would never reach.
"""

from __future__ import annotations

import random

from . import Drafter


class AdversarialDrafter(Drafter):
    name = "adversarial"

    def __init__(self, vocab: int, seed: int = 0, truth: list[int] | None = None,
                 accept_prefix: int = 0):
        self.rng = random.Random(seed)
        self.vocab = vocab
        self.truth = truth or []
        self.accept_prefix = accept_prefix
        self.n = 0

    def propose(self, context: list[int], k: int) -> list[int]:
        """The first `accept_prefix` tokens are the true continuation, the rest are noise."""
        out = []
        for i in range(k):
            j = self.n + i
            if i < self.accept_prefix and j < len(self.truth):
                out.append(self.truth[j])
            else:
                out.append(self.rng.randrange(1000, min(self.vocab, 200000)))
        return out

    def observe(self, tokens: list[int]) -> None:
        self.n += len(tokens)
