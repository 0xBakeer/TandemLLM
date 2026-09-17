"""Drafters. A drafter is one method: given the tokens so far, propose the next few.

The verify step corrects whatever a drafter gets wrong, so a drafter is free to be approximate; it
is not free to be expensive. On this board a proposal that reads a gigabyte of weights has spent
4 ms of a ~150 ms budget before the verify step starts, and it has to earn that back in accepted
tokens. The cost of each drafter, in bytes read per proposed block, is the number to compare them
by -- see notes/ARCHITECTURE.md section 1.4.
"""

from __future__ import annotations


class Drafter:
    name = "none"

    def propose(self, context: list[int], k: int) -> list[int]:
        """Up to `k` tokens continuing `context`. May return fewer, including none."""
        raise NotImplementedError

    def observe(self, tokens: list[int]) -> None:
        """Tokens that were actually accepted, in order. Cheap: this runs every step."""

    def reset(self) -> None:
        pass
