"""The model's write channel to its memory: one hidden block in its answer (2026-10-05).

The design: Memo "Memory for TandemLLM: design advice" (2026-10-05), section 2. The model writes,
in its answer and usually after its text:

    <tandem_memory op="add" scope="project" kind="gotcha">
    title: the summer scene differs because offerings.js recolours by season
    tags: pagoda, checker

    Gate (a) wants the summer scene within 1.0 of night 1; ...
    </tandem_memory>

`op` is add, update (with `id:`), archive (with `id:` and `reason:`) or propose (P-M3: a change
to a package, with `name:`). An attribute in the opener and a `key: value` line at the head of
the block say the same; the body is what follows the first blank line. The engine holds the block
back the way it holds a `<tool_call>` (`MemFilter`, beside server/toolcall.py's buffer), so the
client never sees it: the stream is the answer alone, plus one state line `› remembered: ...`.

A CLOSED BLOCK IS NEVER SHOWN (SRV-277, 2026-10-08). Every closed block is held back from the
client, whether it was written or not: one over 2,000 characters (`too_long`), the fifth of a turn
(`too_many`), one the schema or the store refused (`schema`, `cap`, `secret`, ...). Before, a
refused block went out as the text it was, so an Open WebUI chat (no working directory, so no
project scope) showed the model's `<tandem_memory scope="project">` markup to the user. The journal
and the log say why a block was not written; `hidden` lists every block kept from the client (its
tokens are hidden spans, as a written one's). Only text that is not a closed block goes out as
written: one that never closed (`unclosed`), one still open past `HOLD_CHARS` (`too_long`: the
model is not writing a note any more), `<tandem_memoryx`. Only the answer writes: a block inside
the reasoning is a thought about remembering, as a call inside the reasoning is a thought about
calling, and stays as the model wrote it.

THE CALL FORM (2026-11-03). With tools in view the model may write the note as a tool call,
`<tool_call><function=tandem_memory><parameter=op>add</parameter>...</function></tool_call>` (or
the JSON form, `{"name": "tandem_memory", "arguments": {...}}`). No client offers that tool, so
the call is the block: `MemFilter(calls=True)` holds it back, `call_block()` turns it into the
block it means (the same function the background writer reads its calls with), and it is written
and hidden as the block is. Every other call passes as it came, once its function is named. A
client that offers a tool named `tandem_memory` gets its calls (the server then passes
`calls=False`).

THE CACHE. The rows hold the block; the client's copy of the answer does not. `token_spans()`
finds the block's tokens in the generation, and the server records them as hidden spans, which
the end anchor's replay puts back on the next turn (engine/cache.py `ResidentPrefix.replay`), as
it does for the monitor's hidden close. A block whose edges fall inside a token that also holds
visible text is not recorded (`unaligned`): the next turn then prefills from there, correct and
a little slower.

Standard library only.
"""

from __future__ import annotations

import json
import re

OPEN = "<tandem_memory"
CLOSE = "</tandem_memory>"
MAX_BLOCKS = 4
MAX_CHARS = 2000
HOLD_CHARS = 2 * MAX_CHARS       # how long an open block is held back before it is text after all
OPS = ("add", "update", "archive", "propose")
FIELDS = ("op", "scope", "kind", "title", "tags", "id", "reason", "verify_by", "name", "mode")
_ATTR = re.compile(r"""([a-z_]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))""")
_HEAD = re.compile(r"^(" + "|".join(FIELDS) + r")\s*:\s*(.*)$")
BLOCK = re.compile(re.escape(OPEN) + r"(?=[\s>])[\s\S]*?" + re.escape(CLOSE))


class BlockError(ValueError):
    def __init__(self, why: str, message: str):
        super().__init__(message)
        self.why = why


def parse(raw: str) -> dict:
    """One closed block's op: `{op, scope, kind, title, tags, id, reason, verify_by, name,
    body}`; `BlockError` when it is not one."""
    if not raw.startswith(OPEN) or not raw.endswith(CLOSE):
        raise BlockError("schema", "not a closed <tandem_memory> block")
    gt = raw.find(">", len(OPEN))
    if gt < 0 or gt > len(raw) - len(CLOSE):
        raise BlockError("schema", "the opener has no '>'")
    op: dict = {}
    for m in _ATTR.finditer(raw[len(OPEN):gt]):
        k = m.group(1)
        if k in FIELDS:
            op[k] = next(g for g in m.groups()[1:] if g is not None)
    inner = raw[gt + 1:len(raw) - len(CLOSE)].strip("\n")
    lines = inner.split("\n")
    i = 0
    while i < len(lines):
        m = _HEAD.match(lines[i].strip())
        if not m:
            break
        op.setdefault(m.group(1), m.group(2).strip())
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    op["body"] = "\n".join(lines[i:]).strip()
    op["op"] = (op.get("op") or "add").strip()
    if op["op"] not in OPS:
        raise BlockError("schema", f"op: one of {', '.join(OPS)}")
    if op["op"] in ("update", "archive") and not op.get("id"):
        raise BlockError("schema", f"{op['op']} needs id: the note's id")
    if op["op"] == "propose" and not op.get("name"):
        raise BlockError("schema", "propose needs name: the package or skill it changes")
    return op


def strip_blocks(text: str) -> tuple[str, list[str]]:
    """`text` without its closed blocks, and the blocks (the non-streamed answer)."""
    found = BLOCK.findall(text or "")
    return (BLOCK.sub("", text) if found else text), found


# ------------------------------------------------------------------------ the call form
TOOL = "tandem_memory"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
CALL_DECIDE = 64                 # characters after `<tool_call>` within which a call names its function
_FUNC_XML = re.compile(r"<function=" + TOOL + r">([\s\S]*?)</function>")
_PARAM = re.compile(r"<parameter=([a-z_]+)>\n?([\s\S]*?)\n?</parameter>")
_NAME = re.compile(r'\s*(?:<function=([^>\s]*)>|\{\s*"name"\s*:\s*"([^"]*)")')
_ATTRS = ("op", "scope", "kind")
_HEADS = ("title", "tags", "id", "reason", "verify_by", "name", "mode")
_BODIES = ("body", "content", "text")
_ANY_CALL = re.compile(re.escape(CALL_OPEN) + r"[\s\S]*?" + re.escape(CALL_CLOSE) + r"|"
                       + _FUNC_XML.pattern)


def _value(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def params_block(params: dict) -> str:
    """A call's parameters as the `<tandem_memory>` block they mean: op, scope and kind in the
    opener; title, tags, id, reason, verify_by (and a proposal's name, mode) as head lines; the
    `body` (or `content`, `text`) as the body, and any other parameter (a `why`, an `evidence`)
    after it as `name: value` lines, so nothing the model wrote is lost. `parse` checks it."""
    p = {k: _value(v).strip() for k, v in params.items() if isinstance(k, str) and v is not None}
    attrs = " ".join(f'{k}="{p[k]}"' for k in _ATTRS if p.get(k) and '"' not in p[k])
    heads = "".join(f"{k}: {' '.join(p[k].split())}\n" for k in _HEADS if p.get(k))
    body = next((p[k] for k in _BODIES if p.get(k)), "")
    extra = "\n".join(f"{k}: {p[k]}" for k in p
                      if p[k] and k not in _ATTRS + _HEADS + _BODIES)
    body = "\n\n".join(x for x in (body, extra) if x)
    return f"{OPEN} {attrs}>\n{heads}\n{body}\n{CLOSE}"


def call_block(call: str) -> str | None:
    """One `tandem_memory` call (`<tool_call>...</tool_call>`, or the bare
    `<function=tandem_memory>...</function>` a tokenizer that drops `<tool_call>` leaves) as its
    block; None when `call` is not one."""
    m = _FUNC_XML.search(call or "")
    if m is not None:
        return params_block(dict(_PARAM.findall(m.group(1))))
    inner = (call or "").strip()
    if inner.startswith(CALL_OPEN):
        inner = inner[len(CALL_OPEN):]
    if inner.endswith(CALL_CLOSE):
        inner = inner[:-len(CALL_CLOSE)]
    try:
        doc = json.loads(inner)
    except ValueError:
        return None
    if not isinstance(doc, dict) or doc.get("name") != TOOL:
        return None
    args = doc.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return params_block(args) if isinstance(args, dict) else None


def tool_blocks(text: str) -> list[str]:
    """Every `tandem_memory` call in `text` as its block, in order: the XML form wherever its
    function appears (with or without `<tool_call>`), the JSON form inside `<tool_call>`."""
    out = []
    for m in _ANY_CALL.finditer(text or ""):
        xs = _FUNC_XML.findall(m.group(0))
        if xs:
            out += [params_block(dict(_PARAM.findall(x))) for x in xs]
            continue
        b = call_block(m.group(0))
        if b is not None:
            out.append(b)
    return out


def without_notes(text: str, calls: bool = True) -> tuple[str, int, int]:
    """`text` as the client gets it from `MemFilter`: (the text, closed blocks taken out, closed
    `tandem_memory` calls taken out). The supervisor reads the answer this way (an answer of
    notes alone is an empty answer)."""
    f = MemFilter(lambda raw: None, calls=calls)
    out = f.feed(text or "") + f.finish()
    return out, len(f.hidden) - f.called, f.called


def _partial(s: str, opener: str) -> int:
    """The length of the longest end of `s` that `opener` could still begin with."""
    for n in range(min(len(s), len(opener)), 0, -1):
        if opener.startswith(s[-n:]):
            return n
    return 0


def _partial_open(s: str) -> int:
    return _partial(s, OPEN)


class MemFilter:
    """The answer's text with the closed blocks taken out, one piece at a time.

    `on_block(raw)` is called once a block closes; it returns None when the block was taken
    (written), else the reason it was not. Either way a closed block is hidden (SRV-277).
    `feed()` returns the text the client gets now; `finish()` the rest (an unclosed block as
    text). With `calls`, a closed `tandem_memory` tool call is a block too (THE CALL FORM):
    `on_block` gets the block it means, `hidden` and `taken` the call as the model wrote it (its
    tokens are the hidden span). Any other call is held only until its function is named."""

    def __init__(self, on_block, calls: bool = False):
        self.on_block = on_block
        self.calls = bool(calls)
        self.pending = ""
        self.inside = False
        self.in_call = False
        self.blocks = 0                  # blocks seen closed this turn (calls included)
        self.called = 0                  # of them, `tandem_memory` calls
        self.taken: list[str] = []       # the raw text of the blocks that were written
        self.hidden: list[str] = []      # every closed block kept from the client (taken or not)
        self.failed: list[str] = []      # why each block that was not written was not
        self.shown: list[str] = []       # why each piece that looked like a block went out as text

    def _close(self, raw: str, block: str | None) -> None:
        self.blocks += 1
        why = ("schema" if block is None else
               "too_long" if len(block) > MAX_CHARS else
               "too_many" if self.blocks > MAX_BLOCKS else self.on_block(block))
        self.hidden.append(raw)
        if why is None:
            self.taken.append(raw)
        else:
            self.failed.append(why)

    def _call_opens(self, out: list) -> bool:
        """`pending` starts with `<tool_call>`: True once it is ours (`in_call` set), else the
        opener goes out (False: go on), or None while the function is not named yet."""
        rest = self.pending[len(CALL_OPEN):]
        m = _NAME.match(rest)
        if m is None:
            if len(rest) < CALL_DECIDE and rest.lstrip()[:1] in ("", "<", "{"):
                return None                      # the function's name decides
        elif (m.group(1) if m.group(1) is not None else m.group(2)) == TOOL:
            self.in_call = True
            return True
        out.append(CALL_OPEN)                    # another tool's call: the client's
        self.pending = rest
        return False

    def feed(self, piece: str) -> str:
        if not piece and not self.pending:
            return ""
        self.pending += piece
        out = []
        while self.pending:
            if not self.inside and not self.in_call:
                i = self.pending.find(OPEN)
                c = self.pending.find(CALL_OPEN) if self.calls else -1
                if c >= 0 and (i < 0 or c < i):
                    out.append(self.pending[:c])
                    self.pending = self.pending[c:]
                    got = self._call_opens(out)
                    if got is None:
                        break
                    if not got:
                        continue
                elif i < 0:
                    keep = max(_partial(self.pending, OPEN),
                               _partial(self.pending, CALL_OPEN) if self.calls else 0)
                    out.append(self.pending[:len(self.pending) - keep])
                    self.pending = self.pending[len(self.pending) - keep:]
                    break
                else:
                    after = self.pending[i + len(OPEN):i + len(OPEN) + 1]
                    if not after:
                        out.append(self.pending[:i])
                        self.pending = self.pending[i:]
                        break                       # the character after the opener decides
                    if not (after.isspace() or after == ">"):
                        out.append(self.pending[:i + 1])
                        self.pending = self.pending[i + 1:]
                        continue                    # `<tandem_memoryx`: not ours
                    out.append(self.pending[:i])
                    self.pending = self.pending[i:]
                    self.inside = True
            close = CALL_CLOSE if self.in_call else CLOSE
            j = self.pending.find(close)
            if j < 0:
                if len(self.pending) > HOLD_CHARS + len(close) + (CALL_DECIDE if self.in_call
                                                                    else 0):
                    self.failed.append("too_long")
                    self.shown.append("too_long")
                    out.append(self.pending)
                    self.pending, self.inside, self.in_call = "", False, False
                break
            raw = self.pending[:j + len(close)]
            self.pending = self.pending[j + len(close):]
            if self.in_call:
                self.in_call = False
                self.called += 1
                self._close(raw, call_block(raw))
            else:
                self.inside = False
                self._close(raw, raw)
        return "".join(out)

    def finish(self) -> str:
        rest, self.pending = self.pending, ""
        if self.in_call and "</function>" in rest:
            # the end token right after `</function>`: the call is whole without its closer
            k = rest.index("</function>", max(0, rest.find("<function="))) + len("</function>")
            raw, rest = rest[:k], rest[k:]
            self.in_call = False
            self.called += 1
            self._close(raw, call_block(raw))
            return rest
        if self.inside or self.in_call:
            self.failed.append("unclosed")
            self.shown.append("unclosed")
        self.inside = self.in_call = False
        return rest


def token_spans(ids: list[int], decode, raws: list[str]) -> tuple[list[tuple[int, int]], int]:
    """The token ranges `[k0, k1)` of `ids` that hold exactly the taken blocks (with whitespace
    around them at most), in order; and how many blocks could not be aligned to whole tokens.
    `decode(ids)` is the tokenizer's decode without special tokens."""
    if not raws or not ids:
        return [], 0
    text = decode(ids)
    lens: dict[int, int] = {}

    def at(k: int) -> int:
        if k not in lens:
            lens[k] = len(decode(ids[:k]))
        return lens[k]

    def first_k(c: int, lo: int) -> int:
        """The smallest k >= lo whose prefix decodes to at least `c` characters."""
        hi = len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if at(mid) >= c:
                hi = mid
            else:
                lo = mid + 1
        return lo

    out, bad, pos, k_lo = [], 0, 0, 0
    for raw in raws:
        c0 = text.find(raw, pos)
        if c0 < 0:
            bad += 1
            continue
        c1 = c0 + len(raw)
        pos = c1
        k0 = first_k(c0 + 1, k_lo) - 1            # the token the block's first character is in
        k1 = first_k(c1, max(k0, k_lo))
        if k0 < 0 or k1 > len(ids) or k1 <= k0:
            bad += 1
            continue
        got = decode(ids[k0:k1])
        if got.strip() != raw.strip() or not _only_space_around(got, raw):
            bad += 1
            continue
        out.append((k0, k1))
        k_lo = k1
    return out, bad


def _only_space_around(got: str, raw: str) -> bool:
    i = got.find(raw)
    return i >= 0 and not got[:i].strip() and not got[i + len(raw):].strip()
