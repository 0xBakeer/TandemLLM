"""The XML form (`<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`),
the one server/toolcall.py parses and streams natively: the default `ToolFormat` as it is."""

from __future__ import annotations

from server.toolformats import ToolFormat, register

FORMAT = register(ToolFormat())
