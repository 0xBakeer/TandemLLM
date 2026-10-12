"""The context guard and compaction (SRV-351; plan note Chat 7).

Before every model call chatd estimates the request in tokens: the request's characters times the
chat's own tokens-per-character, learned from the last usage frame (the engine's real count over
the characters chatd sent), 0.3 until there is one. Then:

1. over 80 % of the window (`compact_at`, from the model card's context length), the conversation
   is compacted first: one call writes a summary of everything before the last two user turns;
   from then on the model gets the summary in its system block plus those turns. The database
   keeps every message; the snapshot lists the compaction (`Conversation.compactions`) and a
   `compaction` event tells every device. The request carries `tandem.compaction: true`, so our
   engine does not guess (SRV-351) and serves it as a continuation of the conversation it holds;
2. still too long (one huge turn, nothing before the last two to summarize): the reasoning of
   earlier, finished turns is left out of the request. Templates such as Qwen's drop it anyway;
   others keep it, and 28,000 tokens of thinking then fill a 40,960 window (seen live on
   2026-10-12). Inside the running turn it is never dropped;
3. still too long: the run fails before the engine is asked, with the numbers, instead of the
   engine's 400 ("prompt is 40,982 tokens and the context is 40,960");
4. `max_tokens` is set to what the window has left (and the model card's output limit), so a
   long thought ends at the window instead of past it.
"""

from __future__ import annotations

import json

COMPACT_SHARE = 0.8
DEFAULT_RATIO = 0.3                 # tokens per character until the engine has told us
MIN_ANSWER = 1024                   # the room an answer needs, at least
KEEP_USER_TURNS = 2
SUMMARY_TOKENS = 2048
INSTRUCTIONS = (
    "Write a summary of the conversation above, for yourself, so that you can continue it without "
    "the earlier messages. Keep: what the user wants and every decision they made; facts, names, "
    "numbers, file paths and commands that matter; what was done, what worked and what did not; "
    "the open questions and the next steps. Leave out pleasantries. Plain prose and short lists, "
    "at most 600 words. Do not continue the conversation and do not call tools.")


def estimate(body: dict, ratio: float | None) -> int:
    chars = len(json.dumps(body.get("messages") or [], ensure_ascii=False))
    chars += len(json.dumps(body.get("tools") or [], ensure_ascii=False))
    return int(chars * (ratio or DEFAULT_RATIO)) + 16


def learn(prompt_tokens: int, request_chars: int) -> float | None:
    if prompt_tokens > 0 and request_chars > 200:
        return max(0.15, min(0.8, prompt_tokens / request_chars))
    return None


def boundary(branch: list[dict]) -> int | None:
    """The index of the first message kept after a compaction (the second-to-last user turn),
    or None when there is nothing before it to summarize."""
    users = [i for i, m in enumerate(branch) if m["role"] == "user"]
    if len(users) < KEEP_USER_TURNS + 1:
        return None
    i = users[-KEEP_USER_TURNS]
    return i if i > 0 else None


def shrink(messages: list[dict], budget_chars: int) -> list[dict]:
    """A compaction request's history that fits: reasoning dropped, then each long message cut
    in its middle, oldest first."""
    out = [dict(m) for m in messages]
    for m in out:
        m.pop("reasoning_content", None)
    total = len(json.dumps(out, ensure_ascii=False))
    for m in out:
        if total <= budget_chars:
            break
        c = m.get("content")
        if isinstance(c, str) and len(c) > 2000:
            keep = max(1000, len(c) - (total - budget_chars))
            cut = c[:keep // 2] + "\n[... cut for the summary ...]\n" + c[-keep // 2:]
            total -= len(c) - len(cut)
            m["content"] = cut
    return out
