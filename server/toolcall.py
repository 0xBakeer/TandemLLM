"""Qwen XML tool calls -> the OpenAI `tool_calls` shape.

The model writes tool calls as text:

    <tool_call>
    <function=write_file>
    <parameter=path>
    /app/out.html
    </parameter>
    <parameter=content>
    <!DOCTYPE html> ...
    </parameter>
    </function>
    </tool_call>

Without a parser the block is literal content: the client receives no `tool_calls`, cannot act,
the user reads raw XML, and the payload inside the parameter eats the token budget (chat
efa916ed: `write_file` was attempted, the file arrived as a parameter, and the 16,384 cap cut it
mid-code).

Two pieces:

* `parse_tool_calls(text)` finds every COMPLETE block in a string and returns
  `(content_without_blocks, calls)` -- the non-streaming path, and the tests;
* `ToolCallBuffer` is the streaming counterpart: pieces are fed as they are detokenized, content
  is released immediately, and the text from an opener to its closer is held back until the block
  completes. A block that never closes or does not parse is released AS CONTENT -- an honest
  fallback: showing the text beats losing it.

Inside an opener, the buffer streams the call as OpenAI argument deltas WHILE the model writes it
(`drain_deltas()`): as soon as `<function=NAME>` is complete, one delta carries the id and the
name; each parameter's value then goes out as JSON-escaped fragments as it arrives, with the same
whitespace trimming the collector applies. A whole-file `write_file` argument is thousands of
tokens, and holding it silently made the client show NOTHING for minutes and then the entire call
at once (measured 2026-09-19, the rolex_svg case: a request looked hung for minutes, then every
chunk landed at the end). Calls streamed this way are listed in `streamed_ids` and are skipped by
the end-of-stream sweep; a block the machine cannot recognise is never partially streamed -- it
waits for the closer and follows the fallback above.

Two things a stream cannot do are what the rest of the machinery is about. It cannot take a delta
back, so a function whose name repeats the call before it -- the echo `_add` drops -- is not
streamed at all and goes out at closure if it survives the drop. And it cannot be re-read, so a
call is credited to `streamed_ids` only while what went out IS what `_parse_one` makes of the
closed block; anything else is corrected by the sweep. `finish()` is the end of a generation: its
content goes to the wire as it is, never back through `feed()`.

Nothing here is content-specific: any tool name and any parameters work; the arguments are the
JSON object of the parameter map, which is what OpenAI clients expect.
"""

from __future__ import annotations

import json
import re
import uuid

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
_BLOCK = re.compile(re.escape(OPEN) + r"(.*?)" + re.escape(CLOSE), re.S)
_FUNC = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def _call(name: str, params: dict) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(params)}}


def _add(calls: list[dict], new: dict) -> None:
    """Append unless it repeats the previous call exactly.

    The model sometimes echoes a tool call twice in one answer (measured 2026-09-19: the same
    write_file call emitted twice); a client would execute both. Consecutive exact duplicates are
    artefacts and are dropped.
    """
    if calls and calls[-1]["function"] == new["function"]:
        return
    calls.append(new)


def _parse_one(inner: str) -> list[dict] | None:
    calls = []
    for m in _FUNC.finditer(inner):
        # A function with no parameters is a call: `get_time` takes none, and refusing it left the
        # block in the content as raw XML while the streamer had already sent half of one.
        params = {pm.group(1): pm.group(2) for pm in _PARAM.finditer(m.group(2))}
        _add(calls, _call(m.group(1), params))
    return calls or None


def _same_args(streamed: str, parsed: str) -> bool:
    """Do the fragments that went out spell the argument object the parser read?"""
    try:
        return json.loads(streamed) == json.loads(parsed)
    except ValueError:
        return False


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split every complete block out of `text`; returns (content, calls)."""
    calls: list[dict] = []
    out, i = [], 0
    for m in _BLOCK.finditer(text):
        parsed = _parse_one(m.group(1))
        if parsed is None:
            continue                         # leave it in the content: honest fallback
        out.append(text[i:m.start()])
        i = m.end()
        for call in parsed:
            _add(calls, call)
    out.append(text[i:])
    return "".join(out), calls


_WS = " \t\r\n"


def _esc(s: str) -> str:
    """JSON string escaping, fragment-safe: every character escapes independently."""
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _is_prefix_of(text: str, tag: str) -> bool:
    return bool(text) and len(text) < len(tag) and tag.startswith(text)


class ToolCallBuffer:
    """Streaming: feed detokenized pieces, get content to emit; tool calls collect at closure.

    `drain_deltas()` additionally yields the OpenAI `tool_calls` delta fragments vended live while
    a recognised parameter-XML call is being written.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.streamed_ids: set[str] = set()
        self._buf = ""
        self._open = False
        self._deltas: list[dict] = []
        self._next_index = 0
        self._pos = 0              # how far into the open block the stream machine has read
        self._block_streamed = False
        self._streamed: list[dict] = []   # {id, name, args} per function streamed in this block
        self._hold = False         # stop streaming this block; it goes out at closure instead
        self._stream_name: str | None = None
        self._stream_index: int | None = None
        self._in_param = False
        self._value_started = False
        self._pending_ws = ""
        self._first_param = True
        self._done_params = False

    def drain_deltas(self) -> list[dict]:
        d, self._deltas = self._deltas, []
        return d

    def take_index(self) -> int:
        i = self._next_index
        self._next_index += 1
        return i

    def _delta_args(self, fragment: str) -> None:
        self._streamed[-1]["args"] += fragment
        self._deltas.append({"index": self._stream_index,
                             "function": {"arguments": fragment}})

    def _previous_name(self) -> str | None:
        """The name of the call before this one -- in this block, or the last one collected."""
        if self._streamed:
            return self._streamed[-1]["name"]
        return self.calls[-1]["function"]["name"] if self.calls else None

    def _end_function(self) -> None:
        """`</function>`: the next `<function=` in the same block starts a new call."""
        self._stream_name = None
        self._in_param = False
        self._value_started = False
        self._pending_ws = ""
        self._first_param = True

    def _stream_open(self, limit: int) -> None:
        """Consume the open block as far as it is safe, vending argument deltas."""
        while True:
            if self._hold:
                return
            rest = self._buf[self._pos:limit]
            if self._stream_name is None:
                if self._done_params:
                    return
                stripped = rest.lstrip(_WS)
                lead = len(rest) - len(stripped)
                if not stripped:
                    self._pos += lead
                    return
                if not stripped.startswith("<function="):
                    if _is_prefix_of(stripped, "<function="):
                        return                                   # hold the partial tag
                    self._done_params = True                     # unknown shape: fallback
                    return
                k = stripped.find(">")
                if k < 0:
                    return
                name = stripped[len("<function="):k].strip()
                self._pos += lead + k + 1
                if not name:
                    self._done_params = True
                    return
                if name == self._previous_name():
                    # The model echoes a call twice (measured 2026-09-19) and `_add` drops the
                    # repeat -- but a repeat already streamed cannot be taken back, so streaming
                    # delivered two and JSON one. A function whose name repeats the call before it
                    # is not streamed live: it goes out at closure, through the end-of-stream
                    # sweep, if `_add` keeps it at all.
                    self._hold = True
                    return
                self._stream_name = name
                self._stream_index = self.take_index()
                self._block_streamed = True
                self._streamed.append({"id": "call_" + uuid.uuid4().hex[:24],
                                       "name": name, "args": ""})
                self._deltas.append({"index": self._stream_index,
                                     "id": self._streamed[-1]["id"], "type": "function",
                                     "function": {"name": name, "arguments": ""}})
                continue
            if self._done_params:
                return
            if not self._in_param:
                stripped = rest.lstrip(_WS)
                lead = len(rest) - len(stripped)
                if not stripped:
                    self._pos += lead
                    return
                if stripped.startswith("<parameter="):
                    k = stripped.find(">")
                    if k < 0:
                        return
                    key = stripped[len("<parameter="):k].strip()
                    self._pos += lead + k + 1
                    self._in_param = True
                    self._value_started = False
                    self._pending_ws = ""
                    lead_sep = "{" if self._first_param else ", "
                    self._first_param = False
                    self._delta_args(lead_sep + json.dumps(key) + ': "')
                    continue
                if stripped.startswith("</function>"):
                    self._pos += lead + len("</function>")
                    # `{}` when the function took no parameters: `}` on its own is not JSON.
                    self._delta_args("}" if not self._first_param else "{}")
                    self._end_function()
                    continue
                if stripped[0] == "<":
                    if _is_prefix_of(stripped, "</function>") or _is_prefix_of(stripped, "<parameter="):
                        return
                    k = stripped.find(">")
                    if k < 0:
                        return                                   # the tag is not complete yet
                    # An unknown tag between parameters. `_parse_one` ignores it, so the stream
                    # does too: stopping here truncated the arguments to invalid JSON while the
                    # parser went on to read the whole block.
                    self._pos += lead + k + 1
                    continue
                self._pos += lead + 1                            # stray text between parameters
                continue
            lt = rest.find("<")
            if lt < 0:
                self._pos += len(rest)
                self._emit_value(rest)
                return
            tail = rest[lt:]
            if "</parameter>".startswith(tail):
                self._pos += lt
                self._emit_value(rest[:lt])
                return
            if tail.startswith("</parameter>"):
                self._emit_value(rest[:lt])
                self._pending_ws = ""
                self._delta_args('"')
                self._in_param = False
                self._pos += lt + len("</parameter>")
                continue
            self._pos += lt + 1
            self._emit_value(rest[:lt + 1])

    def _emit_value(self, text: str) -> None:
        if not text:
            return
        out = []
        for ch in text:
            if not self._value_started:
                if ch in _WS:
                    continue                                  # leading whitespace is trimmed
                self._value_started = True
            if ch in _WS:
                self._pending_ws += ch
                continue
            if self._pending_ws:
                out.append(_esc(self._pending_ws))
                self._pending_ws = ""
            out.append(_esc(ch))
        if out:
            self._delta_args("".join(out))

    def feed(self, piece: str) -> list[str]:
        """Content pieces to emit now. Text inside an opener..closer is withheld."""
        out: list[str] = []
        self._buf += piece
        while True:
            if not self._open:
                i = self._buf.find(OPEN)
                if i < 0:
                    # Hold back ONLY the longest tail that could be the start of an opener --
                    # never a fixed window. Holding 9 chars unconditionally delayed the first
                    # content delta and moved the row's TTFT by +14 % (measured 2026-09-19, the
                    # certification run that caught it).
                    keep = 0
                    for k in range(min(len(self._buf), len(OPEN) - 1), 0, -1):
                        if OPEN.startswith(self._buf[-k:]):
                            keep = k
                            break
                    if len(self._buf) > keep:
                        out.append(self._buf[:len(self._buf) - keep])
                        self._buf = self._buf[-keep:] if keep else ""
                    return out
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(OPEN):]
                self._open = True
                self._pos = 0
                self._block_streamed = False
                self._streamed = []
                self._hold = False
                self._stream_name = None
                self._stream_index = None
                self._in_param = False
                self._value_started = False
                self._pending_ws = ""
                self._first_param = True
                self._done_params = False
            j = self._buf.find(CLOSE)
            self._stream_open(j if j >= 0 else len(self._buf))
            if j < 0:
                return out
            inner = self._buf[:j]
            self._buf = self._buf[j + len(CLOSE):]
            self._open = False
            parsed = _parse_one(inner)
            if parsed:
                self._credit_streamed(parsed)
                for call in parsed:
                    _add(self.calls, call)
            elif not self._block_streamed:
                out.append(OPEN + inner + CLOSE)     # honest fallback

    def _credit_streamed(self, parsed: list[dict]) -> None:
        """Give the calls that WERE streamed live their streamed id, so the end-of-stream sweep
        does not send them a second time.

        The n-th parsed call is the n-th function of the block -- a function is streamed only up to
        the first repeated name, which is the only place `_add` can drop one -- so the two lists
        line up from the front. A call is credited only while what went out over the wire IS what
        the parser read: if a tag the machine skipped ate a parameter, the client is holding half
        an argument object, and then the complete call has to be sent by the sweep rather than
        suppressed as "already streamed".
        """
        for call, rec in zip(parsed, self._streamed):
            fn = call["function"]
            if rec["name"] != fn["name"] or not _same_args(rec["args"], fn["arguments"]):
                return
            call["id"] = rec["id"]
            self.streamed_ids.add(rec["id"])

    def flush(self) -> str:
        """At the end of a generation: anything still held goes out as content."""
        s = (OPEN + self._buf) if self._open else self._buf
        self._buf, self._open = "", False
        return s

    def finish(self) -> tuple[str, list[dict]]:
        """End of the generation: (content still held, the deltas for calls not streamed live).

        The content goes to the wire AS IS. Feeding it back through `feed()` -- which is what
        routing it through the server's own `send()` did -- re-opens the block it still contains,
        streams its arguments a second time at a new index, and swallows the fallback text; with
        nothing but a partial opener held ("hello <tool") it swallowed all of it.
        """
        left = self.flush()
        deltas = []
        for call in self.calls:
            if call["id"] in self.streamed_ids:
                continue
            deltas.append({"index": self.take_index(), "id": call["id"], "type": "function",
                           "function": {"name": call["function"]["name"],
                                        "arguments": call["function"]["arguments"]}})
        return left, deltas
