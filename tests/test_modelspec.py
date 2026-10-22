"""ENG-406 (ENG-E17): what a family tells the server about its model, as data on its registry entry.

  1  every registered family has a ServeSpec; the families that differ from the default say so in their
     own arch.py (the arg-key form and a model that opens its own reasoning block; Pythonic call lists)
  2  the server reads the served family's spec (`app._spec()`), the default when none was loaded
  3  thinking off: a model that opens its own block gets an empty block after the generation prompt; a
     template that opened the block gets it closed; anything else is the template's own prompt
  4  one default family (`default_arch`) serves a `--model` that names no checkpoint directory

CPU, the fake engine in this process. Run: python tests/test_modelspec.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine.models import REGISTRY, default_arch  # noqa: E402
from engine.models.adapter import ReasoningSpec, ServeSpec  # noqa: E402
from server import app, fake_engine  # noqa: E402
from server.stream import CLOSE_THINK, OPEN_THINK  # noqa: E402


def _serve(spec: ServeSpec | None) -> None:
    a = app.parser().parse_args(["--fake-engine", "--fake-tps", "200000", "--max-len", "65536"])
    app.STATE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        fake_engine.load(app, a)
    if spec is not None:
        app.STATE["spec"] = spec


def test_every_family_has_a_spec():
    for name, a in REGISTRY.items():
        assert isinstance(a.spec, ServeSpec) and isinstance(a.spec.reasoning, ReasoningSpec), name
        assert a.spec.tool_format and a.spec.tool_format == a.spec.tool_format.strip(), name
    own = {n: (a.spec.tool_format, a.spec.reasoning.opens_itself, a.spec.prompt.system_first, a.spec.side_slots)
           for n, a in REGISTRY.items() if a.spec != ServeSpec()}
    assert own == {"flint": ("argkey", True, False, True), "lfm2_moe": ("pythonic", False, True, True),
                   "qwen3_5_moe": ("xml", False, True, False)}, own
    return f"{len(REGISTRY)} families, {len(own)} with their own spec"


def test_the_spec_says_what_the_engine_builds():
    """#104: a family's ServeSpec.side_slots (what the Setup Guide reads) is its engine's
    Capabilities.side_slots (what the server reads at load)."""
    import importlib
    import inspect
    from engine.models.adapter import Capabilities
    seen = {}
    for name, a in REGISTRY.items():
        pkg = a.serve.split(":")[0].rsplit(".", 1)[0] if a.serve else "engine.models.qwen3_5"
        mod = importlib.import_module(pkg + ".serving")
        caps = [c.caps for _, c in inspect.getmembers(mod, inspect.isclass)
                if c.__module__ == mod.__name__ and isinstance(getattr(c, "caps", None), Capabilities)]
        if not caps and name == "qwen3_5":       # the 27B's engine carries its own (engine/models/qwen3_5/adapter.py)
            importlib.import_module("engine.models.qwen3_5.adapter")
            caps = [a.engine_class().caps]
        assert caps, (name, "no engine class with Capabilities")
        assert all(c.side_slots == a.spec.side_slots for c in caps), (name, a.spec.side_slots, caps)
        seen[name] = a.spec.side_slots
    assert seen["qwen3_5_moe"] is False and sum(seen.values()) == len(seen) - 1, seen
    return seen


def test_health_names_the_slot_it_runs():
    """#104: `--parallel 2` on an engine without side slots runs one slot, and /health says so."""
    _serve(None)
    assert app._par_status() == {}
    app.STATE["parallel_unbuilt"] = 2
    p = app._par_status()["parallel"]
    assert p["requested"] == 2 and p["max_running"] == 1 and p["n"] == 1 and "not built" in p["max_running_reason"]
    app.STATE.pop("parallel_unbuilt")


def test_the_server_reads_the_served_spec():
    _serve(None)
    assert app._spec() is app.DEFAULT_SPEC and app._spec() == ServeSpec()
    s = ServeSpec(tool_format="argkey", reasoning=ReasoningSpec(opens_itself=True))
    _serve(s)
    assert app._spec() is s


def _prompt(spec: ServeSpec | None, thinking: bool) -> str:
    _serve(spec)
    body = {"messages": [{"role": "user", "content": "Say hi."}],
            "chat_template_kwargs": {"enable_thinking": thinking}}
    client = app.thinking_mod.read(body, app._effort_caps())
    ids = app.build_prompt(body, client, count="ids")
    return app.STATE["tok"].decode(ids, skip_special_tokens=False)


def test_thinking_off_renders_by_the_spec():
    plain_on, plain_off = _prompt(None, True), _prompt(None, False)
    own = ServeSpec(reasoning=ReasoningSpec(opens_itself=True))
    own_on, own_off = _prompt(own, True), _prompt(own, False)
    assert own_on == plain_on, "thinking on: the spec changes nothing"
    # 0.14.0's render (SRV-685): a model that opens its own block gets an empty one after the generation
    # prompt, unless the template left a block open, which the server closes as for any model
    assert not plain_off.rstrip().endswith(OPEN_THINK), "the fake template closes its own block"
    assert own_off == plain_off + OPEN_THINK + CLOSE_THINK, own_off[-80:]
    return repr(own_off[-40:])


def test_thinking_off_adds_the_spec_line_before_the_conversation():
    """SRV-697: a spec with an off line has it before the conversation (after the client's system messages)
    with thinking off, and nowhere with thinking on; a spec without one adds nothing."""
    note = "Answer directly. Do not show your reasoning."
    spec = ServeSpec(reasoning=ReasoningSpec(opens_itself=True, off_note=note, off_role="system"))
    off, on = _prompt(spec, False), _prompt(spec, True)
    assert note in off and note not in on, off[-200:]
    assert off.index(note) < off.index("Say hi."), "the line comes before the user's message"
    assert note not in _prompt(None, False)
    flint = REGISTRY["flint"].spec.reasoning
    assert flint.off_note and flint.off_role == "developer"


def test_where_system_messages_may_stand():
    """A family whose template reads one system prompt first (the default, the 27B's): developer read as
    system, a leading run joined, one later refused. A family whose template takes them where they stand
    (Flint): kept in place with their role."""
    from server import compat
    msgs = [{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
            {"role": "user", "content": "hi"}, {"role": "developer", "content": "C"}]
    try:
        compat.messages({"messages": msgs})
    except compat.Refusal as exc:
        assert "must come before the conversation" in str(exc.body())
    else:
        raise AssertionError("the default refuses a developer message after the conversation began")
    assert compat.messages({"messages": msgs[:3]}) == [{"role": "system", "content": "A\n\nB"},
                                                       {"role": "user", "content": "hi"}]
    assert compat.messages({"messages": msgs}, system_first=False) == msgs


def test_one_default_family():
    d = default_arch()
    assert sum(a.default for a in REGISTRY.values()) == 1 and app._arch_of(None) is d
    assert app._arch_of("/nonexistent/dir") is d


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
