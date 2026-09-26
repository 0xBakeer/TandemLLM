"""The real-client smoke set's own logic, on a CPU: what a validating client refuses.

Run: python tests/test_client_smoke.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.client_smoke import OC_TOOLS, WEATHER, filler, validate  # noqa: E402


def _oc():
    with open(OC_TOOLS) as f:
        return json.load(f)["tools"]


def test_validate_refuses_what_opencode_refused():
    # the exact 2026-09-26 shapes: a string where the schema asks for an integer / an array
    tools = _oc()
    bad = validate([(0, "c", "read", json.dumps({"filePath": "/a", "offset": "150"}), 1)], tools)
    assert bad and "offset" in bad[0], bad
    bad = validate([(0, "c", "todowrite", json.dumps({"todos": "[]"}), 1)], tools)
    assert bad and "todos" in bad[0], bad
    assert validate([(0, "c", "read", json.dumps({"filePath": "/a", "offset": 150}), 1)],
                    tools) == []


def test_validate_names_json_and_unknown_tools():
    assert "not JSON" in validate([(0, "c", "get_time", '{"city": "Ro', 1)], WEATHER)[0]
    assert "not a tool" in validate([(0, "c", "rm", "{}", 1)], WEATHER)[0]
    assert "required" in validate([(0, "c", "get_weather", '{"city": "Oslo"}', 1)], WEATHER)[0]


def test_filler_is_roughly_the_asked_size_and_never_repeats_a_line():
    text = filler(12000)
    lines = text.splitlines()
    assert len(set(lines)) == len(lines) and 400 < len(lines) < 500


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:62s} ok")
            passed += 1
    print(f"{passed} passed")
