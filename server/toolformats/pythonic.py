"""The Pythonic form (SRV-425, ENG-407): a model that writes its calls as a Python list between two tokens of
its own, rewritten into server/toolcall.py's JSON tier as the text arrives.
"""

from __future__ import annotations

import json

from server.toolcall import CLOSE, OPEN
from server.toolformats import ToolFormat, register

# LiquidAI's LFM2 models write their calls as a Python list of calls between two tokens of their own:
#
#     <|tool_call_start|>[get_weather(city='Berlin', days=3), search(q="a, b")]<|tool_call_end|>
#
# `PythonicCalls` turns each such block into this module's own JSON tier, `<tool_call>[{"name": ...,
# "arguments": {...}}, ...]</tool_call>`, as the text arrives, so everything above (the stream machine,
# the holds of the conductor's checks, `answer_items`, `parse_tool_calls`) reads it as it reads any call.
# A value is a Python literal (`ast.literal_eval`: strings, numbers, lists, dicts, tuples, True / False /
# None, and the JSON spellings true / false / null), never evaluated code; a block that is not a list of
# calls with keyword arguments only goes out as the text it was (the honest fallback, as for XML).
PY_OPEN = "<|tool_call_start|>"
PY_CLOSE = "<|tool_call_end|>"
_PY_NAMES = {"true": True, "false": False, "null": None, "none": None}


def _py_value(node):
    import ast
    if isinstance(node, ast.Name) and node.id.lower() in _PY_NAMES:
        return _PY_NAMES[node.id.lower()]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_py_value(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("a ** inside a dict")
        return {_py_value(k): _py_value(v) for k, v in zip(node.keys, node.values)}
    v = ast.literal_eval(node)
    if isinstance(v, (set, bytes, complex)):
        raise ValueError(f"{type(v).__name__} is not a JSON value")
    return v


def _py_name(node) -> str:
    import ast
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _py_name(node.value) + "." + node.attr
    raise ValueError("a call's name must be a name")


def pythonic_calls(inner: str) -> list[dict] | None:
    """The calls of one block's inner text, [{"name", "arguments"}], or None if it is not a list of
    calls (or one call) with keyword arguments only."""
    import ast
    s = inner.strip()
    if not s:
        return None
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError:
        return None
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    out = []
    try:
        for it in items:
            if not isinstance(it, ast.Call) or it.args or any(k.arg is None for k in it.keywords):
                return None
            out.append({"name": _py_name(it.func), "arguments": {k.arg: _py_value(k.value) for k in it.keywords}})
    except (ValueError, TypeError, SyntaxError):
        return None
    return out or None


def pythonic_block(inner: str) -> str | None:
    """One Pythonic block as this module's JSON tier, or None when it does not parse."""
    calls = pythonic_calls(inner)
    if calls is None:
        return None
    return OPEN + json.dumps(calls, ensure_ascii=False) + CLOSE


class PythonicCalls:
    """The stream's rewrite of Pythonic blocks (above): `feed(piece)` returns the text to pass on, a
    block's opener and what follows it held until its closer; `finish(eos)` the rest -- an open block
    that parses is a call when the generation ended (EOS), and text otherwise (a block the token limit
    or a stop string cut stays content, as for XML)."""

    def __init__(self):
        self._buf = ""
        self._open = False
        self.blocks = 0          # blocks rewritten
        self.bad = 0             # blocks released as text

    def feed(self, piece: str) -> str:
        self._buf += piece
        out = ""
        while True:
            if not self._open:
                i = self._buf.find(PY_OPEN)
                if i < 0:
                    k = len(self._buf)
                    # keep a tail that could still become the opener
                    for n in range(min(len(PY_OPEN) - 1, len(self._buf)), 0, -1):
                        if PY_OPEN.startswith(self._buf[-n:]):
                            k = len(self._buf) - n
                            break
                    out += self._buf[:k]
                    self._buf = self._buf[k:]
                    return out
                out += self._buf[:i]
                self._buf = self._buf[i + len(PY_OPEN):]
                self._open = True
            j = self._buf.find(PY_CLOSE)
            if j < 0:
                return out
            out += self._close(self._buf[:j], closed=True)
            self._buf = self._buf[j + len(PY_CLOSE):]
            self._open = False

    def _close(self, inner: str, closed: bool) -> str:
        got = pythonic_block(inner)
        if got is None:
            self.bad += 1
            return PY_OPEN + inner + (PY_CLOSE if closed else "")
        self.blocks += 1
        return got

    def finish(self, eos: bool = False) -> str:
        buf, self._buf = self._buf, ""
        if not self._open:
            return buf
        self._open = False
        if eos:
            return self._close(buf, closed=False)
        self.bad += 1
        return PY_OPEN + buf


def rewrite_pythonic(text: str, eos: bool = False) -> str:
    """A whole answer with its Pythonic blocks rewritten (the non-streamed path)."""
    p = PythonicCalls()
    return p.feed(text) + p.finish(eos=eos)


class PythonicFormat(ToolFormat):
    name = "pythonic"
    joins_documents = False

    def rewrite(self, text: str, eos: bool = False) -> str:
        return rewrite_pythonic(text, eos=eos)

    def stream_rewriter(self):
        return PythonicCalls()

    def render(self, name: str, args: dict) -> str:
        return PY_OPEN + "[" + name + "(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")]" + PY_CLOSE

    def example(self, name: str, key: str | None = None) -> str:
        args = (f", each argument a keyword argument such as {key}= (with that argument's own "
                f"name) with its value as a Python literal" if key else "")
        return f"{PY_OPEN}, a Python list holding the call to {name}{args}, then {PY_CLOSE}"


FORMAT = register(PythonicFormat())
