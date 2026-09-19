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
        params = {pm.group(1): pm.group(2) for pm in _PARAM.finditer(m.group(2))}
        if not params:
            return None                      # a function with no parameters is not a tool call
        _add(calls, _call(m.group(1), params))
    return calls or None


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


class ToolCallBuffer:
    """Streaming: feed detokenized pieces, get content to emit; tool calls collect at closure."""

    def __init__(self):
        self.calls: list[dict] = []
        self._buf = ""
        self._open = False

    def feed(self, piece: str) -> list[str]:
        """Content pieces to emit now. Text inside an opener..closer is withheld."""
        out: list[str] = []
        self._buf += piece
        while True:
            if not self._open:
                i = self._buf.find(OPEN)
                if i < 0:
                    keep = len(OPEN) - 1     # a partial opener may be at the tail
                    if len(self._buf) > keep:
                        out.append(self._buf[:-keep])
                        self._buf = self._buf[-keep:]
                    return out
                out.append(self._buf[:i])
                self._buf = self._buf[i:]
                self._open = True
            j = self._buf.find(CLOSE)
            if j < 0:
                return out
            block = self._buf[:j + len(CLOSE)]
            self._buf = self._buf[j + len(CLOSE):]
            self._open = False
            inner = block[len(OPEN):-len(CLOSE)]
            parsed = _parse_one(inner)
            if parsed:
                for call in parsed:
                    _add(self.calls, call)
            else:
                out.append(block)            # honest fallback

    def flush(self) -> str:
        """At the end of a generation: anything still held goes out as content."""
        s, self._buf, self._open = self._buf, "", False
        return s
