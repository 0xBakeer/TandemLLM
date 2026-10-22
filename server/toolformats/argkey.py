"""The arg-key form (ENG-407, from SRV-683/687/689): `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE
</arg_value>...</tool_call>`, a string value exactly as written, any other value as JSON.
"""

from __future__ import annotations

import json
import re

from server.toolcall import _WS, CLOSE, OPEN, _call, _is_prefix_of, _typed
from server.toolformats import BlockReader, ToolFormat, register
from server.toolformats.pythonic import PythonicCalls

# The arg-key form of a call: `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...
# </tool_call>`, a string value as written and any other value as JSON. A block is rewritten into this module's
# JSON tier with every value as the text the model wrote, so `_typed` turns the values a schema types into JSON
# values and a string stays exactly as written (no trimming); a block that does not read as one call goes out as
# the text it was.
_ARGKEY_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def argkey_calls(inner: str) -> list[dict] | None:
    """The call of one arg-key block's inner text, [{"name", "arguments"}], or None if it is not one."""
    i = inner.find("<arg_key>")
    name = (inner if i < 0 else inner[:i]).strip()
    if not name or "<" in name or "\n" in name:
        return None
    rest = "" if i < 0 else inner[i:]
    args: dict = {}
    pos = 0
    for m in _ARGKEY_ARG.finditer(rest):
        if rest[pos:m.start()].strip():
            return None
        args[m.group(1).strip()] = m.group(2)
        pos = m.end()
    if rest[pos:].strip():
        return None
    return [{"name": name, "arguments": args}]


def argkey_block(inner: str) -> str | None:
    calls = argkey_calls(inner)
    if calls is None:
        return None
    return OPEN + json.dumps(calls, ensure_ascii=False) + CLOSE


class ArgKeyCalls(PythonicCalls):
    """The stream's rewrite of arg-key blocks (above): PythonicCalls' machine on the `<tool_call>` tags."""

    OPEN_TAG, CLOSE_TAG = OPEN, CLOSE

    def feed(self, piece: str) -> str:
        self._buf += piece
        out = ""
        while True:
            if not self._open:
                i = self._buf.find(OPEN)
                if i < 0:
                    k = len(self._buf)
                    for n in range(min(len(OPEN) - 1, len(self._buf)), 0, -1):
                        if OPEN.startswith(self._buf[-n:]):
                            k = len(self._buf) - n
                            break
                    out += self._buf[:k]
                    self._buf = self._buf[k:]
                    return out
                out += self._buf[:i]
                self._buf = self._buf[i + len(OPEN):]
                self._open = True
            j = self._buf.find(CLOSE)
            if j < 0:
                return out
            out += self._close(self._buf[:j], closed=True)
            self._buf = self._buf[j + len(CLOSE):]
            self._open = False

    def _close(self, inner: str, closed: bool) -> str:
        got = argkey_block(inner)
        if got is None:
            self.bad += 1
            return OPEN + inner + (CLOSE if closed else "")
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
        return OPEN + buf




def _argkey_diagnose(inner: str) -> str:
    """`diagnose` for an arg-key block `argkey_calls` refused."""
    i = inner.find("<arg_key>")
    name = (inner if i < 0 else inner[:i]).strip()
    if not name:
        return "the block does not begin with the tool's name"
    if "<" in name or "\n" in name:
        return "the tool's name is not followed by its first <arg_key>"
    rest = inner[i:]
    pos = 0
    for m in re.finditer(r"<arg_key>", rest):
        if m.start() < pos:
            continue
        k = rest.find("</arg_key>", m.end())
        if k < 0:
            return "an <arg_key> has no </arg_key>"
        key = rest[m.end():k].strip()
        v = rest.find("<arg_value>", k)
        if v < 0 or rest[k + len("</arg_key>"):v].strip():
            return f"argument '{key}' has no <arg_value> after its </arg_key>"
        e = rest.find("</arg_value>", v)
        if e < 0:
            return f"argument '{key}' has no </arg_value>"
        pos = e + len("</arg_value>")
    return "there is text between the arguments that is not an argument"


def rewrite_argkey(text: str, eos: bool = False) -> str:
    """A whole answer with its arg-key blocks rewritten (the non-streamed path)."""
    p = ArgKeyCalls()
    return p.feed(text) + p.finish(eos=eos)


class ArgKeyReader(BlockReader):
    """The stream's reader of an arg-key block (ENG-407, from SRV-687's buffer machine)."""

    def __init__(self):
        self.name: str | None = None     # the block's tool, until its first argument's opener

    def reset(self) -> None:
        self.name = None

    def takes(self, head: str) -> bool:
        # the form begins with the tool's name; a `<function=` or JSON block is the shared parser's
        return head[0] not in "<{["

    def parse(self, inner: str, types: dict | None) -> list[dict] | None:
        if inner.lstrip()[:1] in ("{", "["):
            return None
        got = argkey_calls(inner)
        if got is None:
            return None
        name, args = got[0]["name"], got[0]["arguments"]
        return [_call(name, _typed(name, args, types))]

    def stream(self, buf, limit: int) -> None:
        """`_stream_open` for a block in the arg-key form: the tool's name, then
        `<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>` per argument, read as `argkey_calls`
        reads them. A string value streams exactly as written (no trimming: the parse keeps it
        so); a typed one is held to its closer. Anything the form does not allow stops the
        stream, and the closer's parse decides what the client gets (`_credit_streamed`)."""
        ak, av, ak_end, av_end = "<arg_key>", "<arg_value>", "</arg_key>", "</arg_value>"
        buf._exact = True
        while True:
            if buf._hold or buf._done_params:
                return
            rest = buf._buf[buf._pos:limit]
            if buf._stream_name is None and self.name is None:
                if buf._streamed:
                    buf._done_params = True                 # one call a block
                    return
                lt = rest.find("<")
                if lt < 0:
                    return                                   # the name is still being written
                tail = rest[lt:]
                if _is_prefix_of(tail, ak) or _is_prefix_of(tail, CLOSE):
                    return                                   # a tag still being written
                name = rest[:lt].strip()
                if not tail.startswith(ak) or not name or "\n" in name:
                    buf._done_params = True                 # not this form: the parse decides
                    return
                buf._pos += lt
                # the name goes out with the first argument's opener, once the block has shown
                # it is the form: a block that is not goes out whole as the text it was
                self.name = name
                continue
            if not buf._in_param:
                stripped = rest.lstrip(_WS)
                lead = len(rest) - len(stripped)
                if not stripped:
                    buf._pos += lead
                    return
                if not stripped.startswith(ak):
                    # the next key or the block's closer may still be on its way
                    if not (_is_prefix_of(stripped, ak) or _is_prefix_of(stripped, CLOSE)):
                        buf._done_params = True
                    return
                k = stripped.find(ak_end)
                if k < 0:
                    return                                   # the key is still being written
                after = stripped[k + len(ak_end):]
                val = after.lstrip(_WS)
                if not val.startswith(av):
                    if val and not _is_prefix_of(val, av):
                        buf._done_params = True
                    return
                if buf._stream_name is None:
                    name, self.name = self.name, None
                    if not buf._start_function(name):
                        return
                buf._pos += lead + len(stripped) - len(val) + len(av)
                buf._open_param(stripped[len(ak):k].strip())
                continue
            lt = rest.find("<")
            if lt < 0:
                buf._pos += len(rest)
                buf._value(rest)
                return
            tail = rest[lt:]
            if _is_prefix_of(tail, av_end):
                buf._pos += lt
                buf._value(rest[:lt])
                return
            if tail.startswith(av_end):
                buf._value(rest[:lt])
                buf._close_param(exact=True)
                buf._pos += lt + len(av_end)
                continue
            buf._pos += lt + 1
            buf._value(rest[:lt + 1])


#: the form's tags, which a forced-call constraint lets the model write as its own token ids
MARKERS = (OPEN, CLOSE, "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")


def forced_regex(tools: list[dict], allowed: list[str], many: bool = True) -> tuple[str, dict[str, set[str]]]:
    """tool_choice required/named as a constraint in the arg-key form (SRV-689): the name, then the schema's
    parameters in declared order (the required ones present, no others), a string value any text without
    its closer (exact, as the parser reads it), an enum one of its values, anything else the schema's JSON;
    whitespace between the tags bounded as in the XML form's. Returns the regex and, per function, the
    parameters whose values are JSON literals."""
    from engine import grammar
    text = grammar.avoiding("</arg_value>")
    ws = grammar._S
    by_name = {}
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            by_name[fn["name"]] = fn.get("parameters") or {}
    calls, typed = [], {}
    for name in allowed:
        params = by_name.get(name, {})
        props = params.get("properties") if isinstance(params, dict) else None
        if not props:
            body = rf"(?:{ws}<arg_key>[^<>\n]+</arg_key>{ws}<arg_value>{text}</arg_value>)*"
        else:
            required = set(params.get("required", []))
            members = []
            for k in props:
                value, is_json = _value(props[k], text)
                if is_json:
                    typed.setdefault(name, set()).add(k)
                m = rf"{ws}<arg_key>{grammar.literal(k)}</arg_key>{ws}<arg_value>{value}</arg_value>"
                members.append(m if k in required else f"(?:{m})?")
            body = "".join(members)
        calls.append(rf"<tool_call>{ws}{grammar.literal(name)}{body}{ws}</tool_call>")
    one = "(?:" + "|".join(calls) + ")"
    return (f"{ws}(?:{one}{ws})+" if many else f"{ws}{one}{ws}"), typed


def _value(schema: dict, text: str) -> tuple[str, bool]:
    """(a value's regex, whether it is a JSON literal), as the XML form's (engine/grammar.py `_xml_value`)
    with this form's closer."""
    from engine import grammar
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(schema, dict) and "enum" in schema and all(isinstance(v, str) for v in schema["enum"]):
        return "(?:" + "|".join(grammar.literal(v) for v in schema["enum"]) + ")", False
    if t == "string" or not isinstance(schema, dict) or not (t or "enum" in schema or "anyOf" in schema
                                                              or "oneOf" in schema or "const" in schema):
        if isinstance(schema, dict) and t == "string" and "pattern" in schema:
            return f"(?:{schema['pattern']})", False
        return text, False
    return grammar.schema_regex(schema), True


class ArgKeyFormat(ToolFormat):
    name = "argkey"
    markers = MARKERS
    joins_documents = False

    def rewrite(self, text: str, eos: bool = False) -> str:
        return rewrite_argkey(text, eos=eos)

    def buffer_reader(self) -> BlockReader:
        return ArgKeyReader()

    def render(self, name: str, args: dict) -> str:
        return OPEN + name + "".join(f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v)}"
                                     f"</arg_value>" for k, v in args.items()) + CLOSE

    def forced_render(self, name: str, args: dict) -> str:
        return self.render(name, args)

    def diagnose(self, inner: str) -> str | None:
        return _argkey_diagnose(inner)

    def example(self, name: str, key: str | None = None) -> str:
        args = (f"then each argument as <arg_key>{key}</arg_key> (with that argument's own "
                f"name), <arg_value>, its value, </arg_value>, " if key else "")
        return f"<tool_call>{name}, {args}then </tool_call>"

    def forced_regex(self, tools: list[dict], allowed: list[str], many: bool):
        return forced_regex(tools, allowed, many=many)

    def open_call_name(self, open_text: str) -> str | None:
        got = super().open_call_name(open_text)
        if got:
            return got
        m = re.match(r"\s*([^<\s]+)\s*<arg_key>", open_text or "")
        return m.group(1) if m else None


FORMAT = register(ArgKeyFormat())
