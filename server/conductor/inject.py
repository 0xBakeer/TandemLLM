"""Where hidden text goes into a prompt, and how it stays there across turns.

The Qwen3.8 chat template renders `reasoning_effort` as the first system tokens (xhigh and low add
a sentence, medium adds none), and `enable_thinking: false` drops that sentence too, so changing
either per turn would change the head of a 100k-token prompt and cost a cold prefill. Everything the
router adds therefore goes at the other end: thinking off replaces the generation prompt's
`<think>\n` with the template's own closed block, and a note goes into
the last turn, right before its `<|im_end|>`, after the newest content and before the generation
prompt (`<|im_start|>assistant\\n<think>\\n`). The prefix up to that point is the one the engine
already holds, and `ThinkBudget.start` still sees the prompt's own tail.

The next turn's prompt does not contain the note (the client never saw it), so it would part from
the resident prefix at the note and the engine would re-read the whole previous answer. So a note
is REPLAYED: the session remembers `(pos, hash of the tokens before pos, the note's ids)`, with
`pos` counted in the client's prompt (without notes), and every later prompt whose first `pos`
tokens hash the same gets the same ids at the same place. The prefix then matches the previous
turn's KV through the answer, and the model sees one conversation that does not change under it.
After a compaction the hash no longer matches and the note is gone for good.

Token work only: lists of ints in, lists of ints out. The router decides what; this decides where.
"""

from __future__ import annotations

import hashlib
from array import array

IM_END = "<|im_end|>"
# the generation prompt with thinking on, and the template's own closed block for thinking off
THINK_OPEN = "<think>\n"
THINK_OFF = "<think>\n\n</think>\n\n"


def special_id(tok, text: str) -> int | None:
    try:
        v = tok.convert_tokens_to_ids(text)
    except Exception:                                              # noqa: BLE001
        return None
    unk = getattr(tok, "unk_token_id", None)
    return v if isinstance(v, int) and v >= 0 and v != unk else None


def insertion_point(ids: list[int], im_end: int | None) -> int | None:
    """The index of the prompt's last `<|im_end|>`: the end of the last message. None when the
    prompt has none (a completion request, or a tokenizer without the token)."""
    if im_end is None:
        return None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_end:
            return i
    return None


def hashes_at(ids: list[int], positions) -> dict[int, str]:
    """sha256 of `ids[:p]` for every `p`, in one pass over the prompt."""
    buf = array("q", ids).tobytes()
    width = array("q").itemsize
    h = hashlib.sha256()
    last, out = 0, {}
    for p in sorted(set(int(x) for x in positions)):
        if p < 0 or p > len(ids):
            continue
        h.update(buf[last * width:p * width])
        last = p
        out[p] = h.copy().hexdigest()[:32]
    return out


def splice(ids: list[int], notes: list[tuple[int, list[int]]]) -> tuple[list[int], list[int]]:
    """`ids` with each `(pos, note_ids)` inserted before original position `pos`; also the final
    start index of each note, in the order given. Notes at one position keep their order."""
    order = sorted(range(len(notes)), key=lambda i: (notes[i][0], i))
    out: list[int] = []
    starts = [0] * len(notes)
    last = 0
    for i in order:
        pos, nid = notes[i]
        out.extend(ids[last:pos])
        starts[i] = len(out)
        out.extend(nid)
        last = pos
    out.extend(ids[last:])
    return out, starts
