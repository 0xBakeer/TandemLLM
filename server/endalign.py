"""The end anchor for a copy whose whitespace is not the rows'.

THE MEASUREMENT. Within an opencode session some tool turns resumed at the previous prompt's last
1,024 line instead of the end of the last generation (2026-10-03 09:33: 73,374 prompt tokens,
68,608 from the resident rows, 4,766 re-read; 09:39: 88,300 / 83,968 / 4,332; TTFT 4 to 7 s). In
the box's suffix store (the last 4,000 documents, token ids only) the follow-up's copy of the
previous output held it token for token in 2,723 turns and parted inside it in 673; of the turns
that part inside a generation, the place is a tool call's parameter (220) or the `</think>` line
(34), and in 221 of those 254 the copy differs from what the engine wrote in whitespace alone:

  * a parameter value: the model writes `<parameter=oldString>\\n      x = 1\\n\\n</parameter>`;
    server/toolcall.py hands the client the value trimmed (`x = 1`), and the template renders it
    back as `<parameter=oldString>\\nx = 1\\n</parameter>` -- the indentation and one newline
    are gone from the copy;
  * the reasoning: the model writes `... done</think>` without a newline; the template writes
    `reasoning|trim + "\\n</think>"` -- one newline more in the copy.

The rest (33) were a boolean the parser typed (`True` -> `true`) and calls of one turn sent back in
another order: not whitespace, served as sent.

THE FIX. `EndAlign.replay` comes after `ResidentPrefix.replay` (the hidden spans): when
the client's copy parts from the rows inside a generation the engine wrote -- the last one, or an
earlier one this alignment already served -- and the two are the same text once every ASCII
whitespace byte is taken out of both, the prompt is served with the rows' tokens up to the end
anchor and the client's after it, and resumes there. That is D2's rule for the whitespace next to
a hidden span, widened to the whole of the model's own output: what differs is whitespace our own
parser and the template took out of the model's tokens, never a character the client wrote. Every
non-whitespace byte must be the rows' own, in order, and everything outside a generation must be
the rows' token for token; anything else is served as the client sent it. `usage.prompt_tokens`
stays the client's count; the journal row says `extra.end_aligned` (the tokens served from the
rows in the regions).

A generation's region is `[its prompt's length, the end anchor)` in the rows; the regions travel
with the end anchor (`kept`, from `_remember`): the regions of the served prompt (still where they
were, the prompt starting with the old rows) and the new generation's. A generation the copy
re-renders the same way every turn is aligned on every turn after it, as the hidden spans are
replayed.

THE REASONING (SRV-239, 2026-10-07). Six of six Open WebUI chats whose first answer thought
re-read that answer on their first follow-up (cached 6,144 of 6,882 to 8,621 tokens, TTFT 2.5 to
5.0 s), while the chats whose first answer did not think resumed at its end. Open WebUI does not
send an answer's reasoning back, and the Qwen3.8 template (`preserve_thinking` on by default)
renders such an assistant message as `<think>\n\n</think>\n\n` + the answer: the copy holds an
empty block where the rows hold the reasoning, it parts from them right after `<think>`, and the
resume falls back to the grid line below (the head's 6,144). A region therefore remembers where
its reasoning ends (its first `</think>`, `_think_end`), and when the copy does not align with
the reasoning in it, a second pass aligns it with every region's reasoning taken out (the
whitespace before it included): every other non-whitespace byte must still be the rows' own, in
order. The prompt is then served with the rows' tokens up to the end anchor, the reasoning the
engine wrote included -- the template's own default, which renders an earlier answer's reasoning
when the client sends it -- and the turn resumes at the end anchor. A client that sends the
reasoning back (opencode) aligns on the first pass as before. `QWEN38_END_ALIGN_REASONING=off`:
the first pass alone.

Exactness: what is served up to the end anchor is the context the engine had, the session cache's
resume, held to the greedy gate as every end-anchor resume is (docs/caches.md). The alignment
reads token ids and the tokenizer's bytes; it writes nothing the client sees. Off (`--end-align
off`): `ResidentPrefix.replay` alone, as before.
"""

from __future__ import annotations

WS = b" \t\n\r"
THINK_END = b"</think>"


def _byte_decoder() -> dict:
    """GPT-2's byte-level alphabet backwards: the character a token string uses for each byte."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok):
    """`id -> bytes` for a byte-level BPE tokenizer (Qwen's): the token's own bytes, a special
    token's text; memoised. A tokenizer without `convert_ids_to_tokens` decodes one id."""
    dec, memo = _byte_decoder(), {}
    conv = getattr(tok, "convert_ids_to_tokens", None)

    def get(i: int) -> bytes:
        b = memo.get(i)
        if b is None:
            s = conv(int(i)) if conv is not None else None
            if isinstance(s, str) and s and all(ch in dec for ch in s):
                b = bytes(dec[ch] for ch in s)
            elif isinstance(s, str):
                b = s.encode()
            else:
                b = tok.decode([int(i)], skip_special_tokens=False).encode()
            memo[i] = b
        return b
    return get


def _bare(b: bytes) -> bytes:
    return b.translate(None, WS)


def _common(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i, step = 0, 4096
    while i < n:
        j = min(n, i + step)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return n


class EndAlign:
    """The whitespace alignment of a copy against the end anchor's rows, and the regions it may
    use. One per server; called under the engine lock, as the replay is."""

    def __init__(self, tbytes, on: bool = True, reasoning: bool = True):
        self.tbytes = tbytes
        self.on = bool(on)
        self.reasoning = bool(reasoning)  # SRV-239: the second pass, without the reasoning
        self._for = None             # the end anchor (the `ResidentPrefix.end` dict) of `_regions`
        # (start, end, the position of the region's first `</think>` or -1): rows positions
        self._regions: list[tuple[int, int, int]] = []
        self._carry: list[tuple[int, int, int]] = []    # the served prompt's, for the next `kept`
        self.stats = {"aligned": 0, "aligned_tokens": 0, "refused": 0, "regions": 0,
                      "aligned_reasoning": 0, "reasoning_tokens": 0}
        # the rows ranges the last `replay` served that the client's copy left out (SRV-233)
        self.last_omitted: list[tuple[int, int]] = []

    def regions(self, res) -> list[tuple[int, int, int]]:
        """The end anchor's generation regions (rows positions), when they are its."""
        end = getattr(res, "end", None)
        return list(self._regions) if end is not None and end is self._for else []

    def _think_end(self, rows: list, a: int, b: int) -> int:
        """The position of the first `</think>` in rows [a, b), -1 when the region has none (a turn
        that did not think: its prompt closed the block)."""
        tb = self.tbytes
        for p in range(a, min(b, len(rows))):
            if tb(rows[p]) == THINK_END:
                return p
        return -1

    def replay(self, res, ids: list[int]) -> tuple[list[int], list, int]:
        """`(ids, hidden spans put back, tokens aligned)`: `res.replay`'s answer when it has one
        (or when off), else the copy aligned to the rows where only whitespace differs."""
        out, spans = res.replay(ids)
        aligned = 0
        self.last_omitted = []
        if not spans and self.on and res.end is not None:
            got = self._align(res, ids)
            if got is not None:
                out, aligned = got
                spans = list(res.end["hidden"])
        self._carry = self._carried(res, out)
        return out, spans, aligned

    def _carried(self, res, served: list[int]) -> list[tuple[int, int]]:
        regs = self.regions(res)
        if not regs or res.end is None:
            return []
        e = res.end["at"]
        return regs if len(served) >= e and served[:e] == res.tokens[:e] else []

    def kept(self, res, n_prompt: int, why: str) -> None:
        """After `ResidentPrefix.keep_end`: the new end anchor's regions are the served prompt's
        and its own generation's. Anything but `kept` leaves the old end anchor and its regions."""
        if why != "kept" or res.end is None:
            self._carry = []
            return
        e = int(res.end["at"])
        regs = [r for r in self._carry if r[1] <= int(n_prompt)]
        if self.on and e > int(n_prompt):
            regs.append((int(n_prompt), e, self._think_end(res.tokens, int(n_prompt), e)))
        self._for, self._regions, self._carry = res.end, regs, []
        self.stats["regions"] = len(regs)

    def adopt(self, res, regions) -> None:
        """A conversation read back from the NVMe (engine/spill.py): its end anchor's regions, as
        they were when it was written."""
        if res.end is None:
            return
        e = int(res.end["at"])
        self._for = res.end
        regs = []
        for r in regions:
            a, b = int(r[0]), int(r[1])
            if b <= e:
                # an entry written before SRV-239 has no reasoning end: found again in the rows
                t = int(r[2]) if len(r) > 2 else self._think_end(res.tokens, a, b)
                regs.append((a, b, t))
        self._regions = regs
        self._carry = []
        self.stats["regions"] = len(self._regions)

    # ------------------------------------------------------------------ the walk
    def _align(self, res, copy: list[int]):
        """The copy aligned to the rows: first as the rows are, then (SRV-239) with every
        region's reasoning left out of what the copy must hold."""
        refused = self.stats["refused"]
        self.last_omitted = []
        got = self._walk(res, copy, False)
        if got is None and self.reasoning and any(r[2] > r[0] for r in self.regions(res)):
            got = self._walk(res, copy, True)
            if got is not None:
                self.stats["aligned_reasoning"] += 1
                self.stats["reasoning_tokens"] += got[2]
                self.last_omitted = [(a, t) for a, _, t in self.regions(res) if t > a]
        # one refusal a copy, whichever pass said it
        self.stats["refused"] = refused + (0 if got is not None else
                                           int(self.stats["refused"] > refused))
        return None if got is None else got[:2]

    def _walk(self, res, copy: list[int], omit: bool):
        end = res.end
        e, rows = int(end["at"]), res.tokens
        regs = self.regions(res)
        if not regs or len(rows) < e:
            return None
        c = _common(copy, rows[:e])
        if c >= e:
            return None                                # the copy holds the rows: nothing to align
        hidden = [(int(p), list(h)) for p, h in end["hidden"]]
        hid = set()
        zones = [(r[0], r[1]) for r in regs]
        left = 0
        if omit:
            # each region's reasoning (from its start to its `</think>`) is not in the copy; the
            # whitespace right before it (the prompt's `\n` after `<think>`, which the copy
            # writes as one `\n\n` token with the empty block's) may differ too
            for a, b, t in regs:
                if t <= a:
                    continue
                hid.update(range(a, t))
                left += t - a
                z = a
                while z > 0 and not _bare(self.tbytes(rows[z - 1])):
                    z -= 1
                zones.append((z, b))
        for p, h in hidden:
            hid.update(range(p, p + len(h)))
            a, b = p, p + len(h)
            while a > 0 and not _bare(self.tbytes(rows[a - 1])):
                a -= 1
            while b < e and not _bare(self.tbytes(rows[b])):
                b += 1
            zones.append((a, b))
        zones = _merge(sorted(zones))
        first = next((z for z in zones if z[0] <= c < z[1] or c == z[1]), None)
        if first is None:
            self.stats["refused"] += 1
            return None
        i = j = first[0]
        aligned = 0
        tb = self.tbytes
        for a, b in zones:
            if b <= i:
                continue
            if copy[j:j + a - i] != rows[i:a]:
                self.stats["refused"] += 1
                return None
            j += a - i
            want = b"".join(_bare(tb(rows[p])) for p in range(a, b) if p not in hid)
            pos = 0
            while pos < len(want):
                piece = _bare(tb(copy[j])) if j < len(copy) else None
                if piece is None or want[pos:pos + len(piece)] != piece:
                    self.stats["refused"] += 1
                    return None
                pos += len(piece)
                j += 1
            i = b
            aligned += b - a
            # whitespace the copy has where the rows go on with the next token
            while i < e and j < len(copy) and copy[j] != rows[i] and not _bare(tb(copy[j])):
                j += 1
        if len(copy) - j <= e - i or copy[j:j + e - i] != rows[i:e]:
            self.stats["refused"] += 1
            return None
        self.stats["aligned"] += 1
        self.stats["aligned_tokens"] += aligned
        return rows[:e] + copy[j + e - i:], aligned, left

    def report(self) -> dict:
        return {"on": self.on, "reasoning": self.reasoning, **self.stats}


def _merge(zones: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in zones:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
