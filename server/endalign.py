"""The end anchor for a copy whose whitespace is not the rows' (0.3.21-test, agent D3).

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

THE FIX. `EndAlign.replay` comes after `ResidentPrefix.replay` (the hidden spans, 0.3.15-test): when
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

Exactness: what is served up to the end anchor is the context the engine had, the session cache's
resume, held to the greedy gate as every end-anchor resume is (docs/caches.md). The alignment
reads token ids and the tokenizer's bytes; it writes nothing the client sees. Off (`--end-align
off`): `ResidentPrefix.replay` alone, as 0.3.20-test.
"""

from __future__ import annotations

WS = b" \t\n\r"


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

    def __init__(self, tbytes, on: bool = True):
        self.tbytes = tbytes
        self.on = bool(on)
        self._for = None             # the end anchor (the `ResidentPrefix.end` dict) of `_regions`
        self._regions: list[tuple[int, int]] = []
        self._carry: list[tuple[int, int]] = []    # the served prompt's, for the next `kept`
        self.stats = {"aligned": 0, "aligned_tokens": 0, "refused": 0, "regions": 0}

    def regions(self, res) -> list[tuple[int, int]]:
        """The end anchor's generation regions (rows positions), when they are its."""
        end = getattr(res, "end", None)
        return list(self._regions) if end is not None and end is self._for else []

    def replay(self, res, ids: list[int]) -> tuple[list[int], list, int]:
        """`(ids, hidden spans put back, tokens aligned)`: `res.replay`'s answer when it has one
        (or when off), else the copy aligned to the rows where only whitespace differs."""
        out, spans = res.replay(ids)
        aligned = 0
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
            regs.append((int(n_prompt), e))
        self._for, self._regions, self._carry = res.end, regs, []
        self.stats["regions"] = len(regs)

    # ------------------------------------------------------------------ the walk
    def _align(self, res, copy: list[int]):
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
        zones = [tuple(r) for r in regs]
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
        return rows[:e] + copy[j + e - i:], aligned

    def report(self) -> dict:
        return {"on": self.on, **self.stats}


def _merge(zones: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in zones:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
