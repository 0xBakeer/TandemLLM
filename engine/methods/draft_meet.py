"""SPD-235: a round's MTP drafts in one packed pass. While the round runner collects its lanes' proposals,
`MEET(head)` (the runner's) takes a draft head's staged launch instead of the head running it; the runner then
runs every collected head of one engine at once (`type(head).run_packed(heads)`), before any proposal reads its
draft. A draft only proposes tokens -- the verify decides what is written -- so the packed pass's arithmetic is
speed alone. None: every head runs its own draft (the lone path)."""

MEET = None
