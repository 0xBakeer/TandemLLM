"""A class's packages fit the tools the request offers (SRV-243, server/conductor/router.py `_fit`).

UX2's normal-user run (2026-10-07, 0.3.48-test): a plain Open WebUI chat whose only tools were
search_web and fetch_url asked for a landing page and got the testing and Laya packages. They tell
the model to run a shell and write files: it called `terminal` and `write_file` ('Tool not found'),
tried to fetch the library's address with fetch_url, wrote the page into a failed call and printed
it again, 518 s for one HTML file (the snake game 423 s, the 3D garden 522 s).

  1  the capabilities read off the tools' schemas: opencode's bash/write/edit, the Open WebUI
     terminal's run_command/write_file, search_web/fetch_url (none), `tool_choice: none`
  2  what a package needs: its own `requires_tools`, else a shell when it calls TandemLLM through
     `{tandem_url}`, else nothing
  3  no tools at all: the shell and file packages stay out, `coding` gives way to its fallback, the
     note says to deliver the files in the answer, and no `{tandem_url}` address is in the prompt
  4  Open WebUI's search tools: the same, and the note names them
  5  an agent chat with a terminal and a file tool: every package as before, no note
  6  `QWEN38_TOOL_FIT=off` and the router's `tool_fit.mode: shadow`: the packages as before (the
     shadow row says what would have gone)
  7  the decision row lists the tools offered, their capabilities and what was dropped

Run: python tests/test_tool_fit.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_conductor_router as T  # noqa: E402
from server.conductor import packages as packages_mod  # noqa: E402
from server.conductor import router as router_mod  # noqa: E402
from server.conductor import signals  # noqa: E402


def _fn(name, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string"} for p in props},
                       "required": list(required)}}}


OPENCODE = [_fn("bash", ["command", "description"], ["command"]),
            _fn("write", ["filePath", "content"]), _fn("edit", ["filePath", "oldString",
                                                                 "newString"]),
            _fn("read", ["filePath"])]
TERMINAL = [_fn("run_command", ["command", "cwd"]), _fn("write_file", ["path", "content"]),
            _fn("read_file", ["path"])]
SEARCH = [_fn("search_web", ["query"]), _fn("fetch_url", ["url"])]

PKGS = {
    "files-care": ("Never overwrite the user's files.", {"requires_tools": ["write"],
                                                        "general": True}),
    "coding": ("Run after every step; play the game yourself.",
               {"requires_tools": ["shell"], "fallback": "coding-answer", "general": True}),
    "coding-answer": ("Write each file complete; check it by reading.", {}),
    "testing": ("curl {tandem_url}/v1/laya/kit/laya_drive.py and run it.", {"general": True}),
    "ui-design": ("Spacing, type, colour and states.", {"general": True}),
}


def _setup(mode=None):
    cond, fakes = T.setup(decider=lambda t, o: ("none", 0.9) if "none" in o else
                          ("design_ui", 0.95))
    _, p = cond.store.get("packages")
    for n, (body, extra) in PKGS.items():
        p["packages"][n] = {"title": n, "kind": "package", "source": "inline",
                            "description": f"The {n} package's description.", "body": body,
                            "max_tokens": 3000, "enabled": True, **extra}
    cond.store.put("packages", None, p, note="fit", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "design_ui":
            c["policy"]["packages"] = ["files-care", "coding", "testing", "ui-design"]
    if mode is not None:
        r["tool_fit"] = {"mode": mode}
    cond.store.put("router", None, r, note="design packages", author="t")
    return cond


def _turn(cond, tools, sid):
    b = T.body([{"role": "user", "content": "Make a landing page for a coffee shop."}],
               tools=False)
    if tools is not None:
        b["tools"] = tools
    plan, ids, _ = T.turn(cond, b, headers={"x-session-id": sid}, kind="open-webui")
    return plan, T.text_of(ids)


def _parts(plan):
    return [p["name"] for p in (plan.block or {}).get("parts", [])]


# ------------------------------------------------------------------ 1-2 the reading of the tools

def test_1_capabilities_from_the_schemas():
    assert signals.tool_caps({"tools": OPENCODE})[0] == {"shell", "write"}
    assert signals.tool_caps({"tools": TERMINAL}) == (frozenset({"shell", "write"}),
                                                      ["run_command", "write_file", "read_file"])
    assert signals.tool_caps({"tools": SEARCH})[0] == frozenset()
    assert signals.tool_caps({"tools": OPENCODE, "tool_choice": "none"}) == (frozenset(), [])
    assert signals.tool_caps({}) == (frozenset(), [])
    flat = [{"name": "Bash", "parameters": {"type": "object",
                                            "properties": {"command": {"type": "string"}}}}]
    assert signals.tool_caps({"tools": flat})[0] == {"shell"}
    # a `command` that is not a string (an enum of actions) is no shell
    odd = [{"type": "function", "function": {"name": "x", "parameters": {
        "type": "object", "properties": {"command": {"type": "integer"}}}}}]
    assert signals.tool_caps({"tools": odd})[0] == frozenset()


def test_2_what_a_package_needs():
    lib = packages_mod.Library()
    doc = {"packages": {n: {"title": n, "kind": "package", "source": "inline", "body": b,
                            "description": "d" * 8, "max_tokens": 100, "enabled": True, **x}
                        for n, (b, x) in PKGS.items()}}
    assert lib.requires(doc, "files-care") == ("write",)
    assert lib.requires(doc, "coding") == ("shell",)
    assert lib.requires(doc, "testing") == ("shell",)          # `{tandem_url}` in its text
    assert lib.requires(doc, "ui-design") == ()
    assert lib.fallback(doc, "coding") == "coding-answer" and lib.fallback(doc, "testing") is None


# ------------------------------------------------------------------ 3-5 the turn

def test_3_no_tools_the_shell_and_file_packages_stay_out():
    cond = _setup()
    plan, text = _turn(cond, None, "fit-none")
    assert _parts(plan) == ["high", "coding-answer", "ui-design", "fit"], _parts(plan)
    assert "no shell and no file tools." in text and "(its tools" not in text
    assert "/v1/laya" not in text and "127.0.0.1" not in text and "/s/" not in text
    t = plan.row["extra"]["tools"]
    assert t == {"offered": [], "caps": [], "dropped": ["files-care", "testing"],
                 "fallback": {"coding": "coding-answer"}}, t


def test_4_open_webui_search_tools_are_named():
    cond = _setup()
    plan, text = _turn(cond, SEARCH, "fit-search")
    assert _parts(plan) == ["high", "coding-answer", "ui-design", "fit"], _parts(plan)
    assert "(its tools: search_web, fetch_url)" in text
    assert plan.row["extra"]["tools"]["offered"] == ["search_web", "fetch_url"]


def test_5_an_agent_chat_keeps_every_package():
    cond = _setup()
    plan, text = _turn(cond, TERMINAL, "fit-agent")
    assert _parts(plan) == ["high", "files-care", "coding", "testing", "ui-design"], _parts(plan)
    assert "no shell" not in text and "/v1/laya/kit" in text
    t = plan.row["extra"]["tools"]
    assert t == {"offered": ["run_command", "write_file", "read_file"],
                 "caps": ["shell", "write"]}, t


def test_5b_a_file_tool_without_a_shell():
    cond = _setup()
    plan, text = _turn(cond, [_fn("write_file", ["path", "content"])], "fit-write")
    assert _parts(plan) == ["high", "files-care", "coding-answer", "ui-design", "fit"], _parts(plan)
    assert "can write files but has no shell (its tools: write_file)" in text


# ------------------------------------------------------------------ 6 off and shadow

def test_6_off_and_shadow_keep_the_packages():
    was = router_mod.TOOL_FIT
    try:
        router_mod.TOOL_FIT = "off"
        plan, _ = _turn(_setup(), None, "fit-off")
        assert _parts(plan) == ["high", "files-care", "coding", "testing", "ui-design"], _parts(plan)
    finally:
        router_mod.TOOL_FIT = was
    plan, text = _turn(_setup("shadow"), None, "fit-shadow")
    assert _parts(plan) == ["high", "files-care", "coding", "testing", "ui-design"], _parts(plan)
    assert plan.row["shadow"]["tools"] == {"dropped": ["files-care", "testing"],
                                           "fallback": {"coding": "coding-answer"}}
    assert "no shell" not in text


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
