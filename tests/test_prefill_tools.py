"""SPD-183: the prefill tools run on main. tools/prefill_profile.py builds the bare 27B engine, so it
imports the family's adapter for its cold verbs (`cache_state`); `prompt_ids` refuses an empty source
instead of doubling an empty list forever (a run from outside the repository did that on 2026-10-23)."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _Tok:
    def encode(self, text):
        return [ord(c) % 97 for c in text]


def test_prompt_ids_refuses_an_empty_source():
    from tools.prefill_window_check import prompt_ids
    with tempfile.TemporaryDirectory() as d:
        try:
            prompt_ids(_Tok(), Path(d), 64)
        except SystemExit as exc:
            assert "no prompt text" in str(exc)
        else:
            raise AssertionError("an empty source was accepted")


def test_prompt_ids_repeats_a_short_source():
    from tools.prefill_window_check import prompt_ids
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "bench").mkdir()
        (Path(d) / "bench" / "calib.txt").write_text("abc")
        assert prompt_ids(_Tok(), Path(d), 10) == [ord(c) % 97 for c in "abcabcabca"]


def test_the_profiler_imports_the_27b_adapter():
    tree = ast.parse((ROOT / "tools" / "prefill_profile.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "engine.models.qwen3_5.adapter" in names


def test_the_adapter_gives_the_engine_cache_state():
    import engine.models.qwen3_5.adapter  # noqa: F401
    from engine.model import Qwen38Engine
    assert callable(getattr(Qwen38Engine, "cache_state", None))


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
