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
    last_q = None                  # ENG-102: per-token q rows when this drafter samples

    def propose(self, context: list[int], k: int) -> list[int]:
        """Up to `k` tokens continuing `context`. May return fewer, including none."""
        raise NotImplementedError

    def observe(self, tokens: list[int]) -> None:
        """Tokens that were actually accepted, in order. Cheap: this runs every step."""

    def reset(self) -> None:
        pass

    def set_sampling(self, sampler) -> None:
        """Hand the request's sampling profile to a drafter that can use one.

        Drafters that can sample their own proposals (the block drafter's head) override this and
        carry `last_q`; the default is a no-op, so the loop can hand it to any arm. ENG-102.
        """


def run_steps(steps):
    """Drive a `*_steps` generator to its end and return what it returns.

    SPD-49: a proposal that launches a draft on the device is written as a generator that stops
    once, right after the launch, so the serving loop can stream the last block's tokens while the
    draft runs and only then wait for it. Everything else calls the plain method, which is this
    over the same generator, so the two orders run the same code and decide the same things.
    """
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def tree_steps(drafter, context: list[int], k: int):
    """`drafter.propose_tree_steps`, or its plain `propose_tree` for a drafter without one (a
    lookup drafter launches nothing, so it has nothing to stop for)."""
    steps = getattr(drafter, "propose_tree_steps", None)
    if steps is not None:
        return (yield from steps(context, k))
    return drafter.propose_tree(context, k)
