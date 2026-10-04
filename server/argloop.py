"""The tool-argument loop guard (0.3.26-test): a tool call whose arguments repeat one unit.

On 2026-10-04 at 16:18 Qwen opened a `write` of a 3.5 KB game, wrote 3 KB of it, then the same
97-character row of a tile map 12,866 times: 65,536 tokens, 1.26 MB, 327 s, until the token
limit. The lookup drafter copied the loop (31 of 31 accepted in 2,007 of 2,209 blocks), the call
never closed, nothing was written, and the cut block went back into the next prompt as text. The
reasoning-loop guard (D1) watches the thinking only; nothing watched a call's arguments.

The guard reads the text of the open tool-call block (what follows `<tool_call>`), the way the
stream's `ToolCallBuffer` holds it, every `every` tokens once the block is `min_tokens` long. Two
rules, either fires:

    periodic   the block's last `span_tokens` tokens are one unit of at most `max_period`
               characters repeated, exactly, at least `min_repeats` times. The check takes the
               tail's last `probe` characters, finds where they occurred before (the nearest is the
               smallest period that fits them), and compares the whole span against itself shifted
               by that period: one string compare, at C speed.
    size       the block is longer than `size_tokens` and `size_context` times the longest
               message the request carries (in tokens, estimated at `chars_per_token`), and its
               last `window_chars` characters compress `min_ratio` times or more (zlib): a call far
               bigger than anything the model read, that is mostly repetition.

Thresholds come from `tools/argloop_fp.py` over every write and edit in the opencode database
(the legit set) and the loops found there; the numbers are in docs/server.md, "The tool-argument
loop guard". The guard acts only where nothing of the call has reached the client (the server
holds the calls for the SP3 checks, `ToolCallBuffer.hold`); elsewhere it is logged.

No torch, no server state: text and counts in, a verdict out.
"""

from __future__ import annotations

import zlib
from bisect import bisect_right
from dataclasses import dataclass, field

DEFAULTS = {
    "every": 64,             # tokens between checks inside an open block
    "min_tokens": 1024,      # no check before the block is this long
    "span_tokens": 1024,     # periodic: the repeated stretch, in tokens
    "max_period": 4096,      # periodic: the longest unit, in characters
    "min_repeats": 6,        # periodic: the unit at least this many times in the stretch
    "probe": 256,            # periodic: characters of the tail looked up to find the period
    "size_tokens": 32768,    # size: never below this many tokens in the block
    "size_context": 3.0,     # size: and this many times the longest message the request carries
    "chars_per_token": 3.5,  # size: a message's tokens, estimated
    "window_chars": 16384,   # size: the window whose compression is measured
    "min_ratio": 16.0,       # size: zlib ratio of that window
}


def periodic_tail(text: str, max_period: int, probe: int = 256) -> tuple[int, int]:
    """`(period, span)`: the smallest period the tail's last `probe` characters repeat at, within
    `max_period`, and how many characters at the end of `text` are periodic with it (unit
    included). `(0, 0)` when the tail does not repeat."""
    n = len(text)
    if n < probe * 2:
        return 0, 0
    tail = text[n - probe:]
    j = text.rfind(tail, max(0, n - probe - max_period), n - 1)
    if j < 0:
        return 0, 0
    p = n - probe - j
    # the longest L with text[n-L-p : n-p] == text[n-L : n]; monotone in L, so a binary search
    lo, hi = probe, n - p
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text[n - mid - p:n - p] == text[n - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return p, lo + p


def ratio(text: str) -> float:
    """zlib's compression ratio of `text` (UTF-8, level 1: the guard's cost, not the best)."""
    raw = text.encode("utf-8", "surrogatepass")
    if not raw:
        return 1.0
    return len(raw) / max(1, len(zlib.compress(raw, 1)))


def longest_message_chars(messages) -> int:
    """The longest text a request's messages carry: a file read, a pasted log, the system."""
    best = 0
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            best = max(best, len(c))
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    best = max(best, len(part["text"]))
        for call in m.get("tool_calls") or ():
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                best = max(best, len(fn["arguments"]))
    return best


@dataclass
class Verdict:
    rule: str                 # periodic | size
    tokens: int               # tokens in the block when it fired
    chars: int                # characters in the block
    period: int = 0           # periodic: the unit's characters
    repeats: float = 0.0      # periodic: span / period
    span_tokens: int = 0      # periodic: the repeated stretch
    ratio: float = 0.0        # size: the window's compression
    detail: dict = field(default_factory=dict)
    text: str = ""                         # the sentence the client got, once it acted

    def as_dict(self) -> dict:
        d = {"rule": self.rule, "tokens": self.tokens, "chars": self.chars}
        if self.rule == "periodic":
            d.update(period=self.period, repeats=round(self.repeats, 1),
                     span_tokens=self.span_tokens)
        else:
            d.update(ratio=round(self.ratio, 1))
        d.update(self.detail)
        return d


class ArgLoopGuard:
    """One request's guard. `feed(n_tokens, length, text)` after each token the stream decided:
    `length` is the open block's length in characters (None outside a block) and `text()` gives
    its text, called only at a check. Returns a `Verdict` once, when a rule fires; else None."""

    def __init__(self, cfg: dict | None = None, context_chars: int = 0):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.context_chars = int(context_chars)
        self.fired: Verdict | None = None
        self.checks = 0
        self._start: int | None = None   # the stream's token count when the block opened
        self._offs: list[int] = []       # the block's length in characters after each token
        self._next = 0

    def size_limit(self) -> int:
        c = self.cfg
        return int(max(c["size_tokens"],
                       c["size_context"] * self.context_chars / c["chars_per_token"]))

    def feed(self, n_tokens: int, length: int | None, text) -> Verdict | None:
        if self.fired is not None:
            return None
        if length is None:
            self._start, self._offs = None, []
            return None
        if self._start is None:
            self._start, self._offs = n_tokens - 1, []
            self._next = self.cfg["min_tokens"]
        self._offs.append(int(length))
        k = n_tokens - self._start
        if k < self._next:
            return None
        self._next = k + self.cfg["every"]
        return self.check(text(), k)

    def _tokens_from(self, char_pos: int) -> int:
        """Tokens of the block that end after `char_pos` (the stretch from there to the end)."""
        i = bisect_right(self._offs, char_pos)
        return len(self._offs) - i

    def check(self, text: str, k: int) -> Verdict | None:
        c = self.cfg
        self.checks += 1
        p, span = periodic_tail(text, c["max_period"], c["probe"])
        if p and span >= c["min_repeats"] * p:
            st = self._tokens_from(len(text) - span) if self._offs else k
            if st >= c["span_tokens"]:
                self.fired = Verdict("periodic", k, len(text), period=p, repeats=span / p,
                                     span_tokens=st)
                return self.fired
        if k >= self.size_limit():
            r = ratio(text[-c["window_chars"]:])
            if r >= c["min_ratio"]:
                self.fired = Verdict("size", k, len(text), ratio=r,
                                     detail={"limit": self.size_limit()})
                return self.fired
        return None


def stop_text(v: Verdict, name: str | None) -> str:
    """The visible sentence the client gets in place of the call."""
    call = f"`{name}` call" if name else "tool call"
    if v.rule == "periodic":
        why = (f"its arguments repeated the same {v.period}-character piece "
               f"{int(v.repeats)} times in a row")
    else:
        why = (f"its arguments grew past {v.tokens:,} tokens, far larger than anything in the "
               f"conversation, and were mostly repetition")
    return (f"[TandemLLM stopped the {call} after {v.tokens:,} tokens: {why}. Nothing was run "
            f"or written.]")


NOTE = ("Runtime note: your last tool call was stopped and not run. Its arguments fell into a "
        "loop: {why}. Write the file again without the repetition. Generate repeated structure "
        "(map rows, tables, arrays) with code such as a loop or String.repeat instead of writing "
        "every copy, and split a large file into parts: write the first part, then add the rest "
        "with edits.")


def note_text(v: dict) -> str:
    """The hidden note for the session's next request, from a verdict's `as_dict()`."""
    if v.get("rule") == "periodic":
        why = (f"the same {v.get('period')}-character piece {int(v.get('repeats', 0))} times "
               f"after {v.get('tokens', 0):,} tokens")
    else:
        why = f"{v.get('tokens', 0):,} tokens of mostly repeated text"
    return NOTE.replace("{why}", why)
