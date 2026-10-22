"""The forms a model writes its tool calls in (ENG-407): one `ToolFormat` a form, named by the family's
`ServeSpec.tool_format` (engine/models/adapter.py), discovered from this package's modules.

Shared code (server/toolcall.py, server/compat.py, the conductor's checks, the request loop) calls a
format through this interface and never names one. A format does what is particular to its form:

  rewrite(text, eos)        a whole answer with its blocks in toolcall.py's JSON tier, which every
                            parser above reads (the identity for the XML form)
  render(name, args)        one call as the model writes it; forced_render as forced_regex writes it
  stream_rewriter()         the same as a stream (`feed(piece) -> str`, `finish(eos) -> str`), or None
  buffer_reader()           a reader the stream's ToolCallBuffer hands a block it can read in place,
                            so a string argument streams as it is written (see `BlockReader`), or None
  diagnose(inner)           why a block of this form does not parse, or None (the XML reasons)
  example(name, key)        how one call is written, with an offered tool and one of its arguments: the
                            line C4's note shows, never a placeholder a model could copy
  forced_regex(...)         tool_choice required/named as a constraint in this form, and `markers`, the
                            form's tags the constraint lets the model write as its own token ids
  joins_documents           whether calls and a document can share one constraint (SRV-619)
  open_call_name(text)      the tool an open block is calling (the argument loop guard's line)

The default form is `xml.py`'s, the one the shared parser reads natively.
"""

from __future__ import annotations

import importlib
import json
import os
import re

FORMATS: dict = {}
DEFAULT = "xml"
_found = []                     # the package's modules were imported (`_discover`)


class BlockReader:
    """A format's reader of one `<tool_call>` block inside the stream's ToolCallBuffer (one reader a
    buffer, `reset()` at each block). `takes(head)`: the block's first characters say it is this form.
    `stream(buf, limit)` consumes `buf._buf[buf._pos:limit]` through the buffer's own steps
    (`_start_function`, `_open_param`, `_value`, `_close_param`, `_done_params`), the values exact
    when it sets `buf._exact`. `parse(inner, types)`: the calls of a closed block of this form, or None
    (the shared parser then reads it)."""

    def reset(self) -> None:
        pass

    def takes(self, head: str) -> bool:
        return False

    def stream(self, buf, limit: int) -> None:
        pass

    def parse(self, inner: str, types: dict | None) -> list[dict] | None:
        return None


class ToolFormat:
    """The XML form's behaviour; another form overrides what differs."""

    name = DEFAULT
    markers: tuple = ()
    joins_documents = True

    def rewrite(self, text: str, eos: bool = False) -> str:
        return text

    def render(self, name: str, args: dict) -> str:
        """One call as the model writes it in this form (what a chat template writes back; the
        contract tests read it through every parser)."""
        params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                         for k, v in args.items())
        return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"

    def forced_render(self, name: str, args: dict) -> str:
        """A call as `forced_regex` writes it (the XML form unless the form has its own)."""
        return ToolFormat.render(self, name, args)

    def stream_rewriter(self):
        return None

    def buffer_reader(self) -> BlockReader | None:
        return None

    def diagnose(self, inner: str) -> str | None:
        return None

    def example(self, name: str, key: str | None = None) -> str:
        args = (f"each argument as <parameter={key}> (with that argument's own name), its value, "
                f"</parameter>, " if key else "")
        return f"<tool_call>, <function={name}>, {args}</function>, </tool_call>"

    def forced_regex(self, tools: list[dict], allowed: list[str], many: bool):
        from engine import grammar
        return grammar.tool_call_regex(tools, allowed, many=many)

    def open_call_name(self, open_text: str) -> str | None:
        """The tool an open block (the text after its `<tool_call>`) is calling, once its name is
        written, else None: what the argument loop guard names (server/argloop.py)."""
        m = re.match(r"\s*<function=([^>\s]+)", open_text or "")
        return m.group(1) if m else None


def register(fmt: ToolFormat) -> ToolFormat:
    FORMATS[fmt.name] = fmt
    return fmt


def _discover() -> None:
    _found.append(True)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(os.listdir(here)):
        if name.endswith(".py") and not name.startswith("_"):
            importlib.import_module(f"{__name__}.{name[:-3]}")


def get(name: str | None) -> ToolFormat:
    """The format a `ServeSpec.tool_format` names (the default for None or empty)."""
    if not _found:
        _discover()
    key = name or DEFAULT
    if key not in FORMATS:
        raise KeyError(f"no tool-call format {key!r} (server/toolformats: {sorted(FORMATS)})")
    return FORMATS[key]


def resolve(fmt) -> ToolFormat:
    """A format, its name or None (the default) as a format."""
    return fmt if isinstance(fmt, ToolFormat) else get(fmt)
