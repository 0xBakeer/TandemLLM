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
    own = {n: (a.spec.tool_format, a.spec.reasoning.opens_itself) for n, a in REGISTRY.items()
           if a.spec != ServeSpec()}
    assert own == {"flint": ("argkey", True), "lfm2_moe": ("pythonic", False)}, own
    return f"{len(REGISTRY)} families, {len(own)} with their own spec"


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


def test_one_default_family():
    d = default_arch()
    assert sum(a.default for a in REGISTRY.values()) == 1 and app._arch_of(None) is d
    assert app._arch_of("/nonexistent/dir") is d


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
