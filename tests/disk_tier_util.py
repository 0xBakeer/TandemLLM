"""A disk tier whose captures wait for a staging slot (#113).

A prefill that admits three anchors captures them into the tier's two anchor slots. The server's tier
never waits for one: an anchor that finds both still being written is dropped (`dropped.staging`), so a
request is never blocked. A test that counts the lines written would then count the writer thread's
speed: on a loaded runner test_disk_cache got [64, 80] for [64, 80, 96]. `waiting(tier)` makes a capture
wait for the slot the writer gives back.
"""

from __future__ import annotations

WAIT_S = 60.0


def waiting(tier, at_least: float = WAIT_S):
    take = tier._take_anchor_slot
    tier._take_anchor_slot = lambda wait=0.0: take(wait=max(wait, at_least))
    return tier
