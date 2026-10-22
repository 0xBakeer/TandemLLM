"""The flint family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register
from engine.models.adapter import EngineHooks, PromptSpec, ReasoningSpec, ServeSpec

ARCH = register(Arch(
    name="flint",
    architectures=("FlintForCausalLM",),
    model_types=("flint",),
    config="engine.models.flint.config:load_flint_config",
    engine="engine.models.flint.engine:FlintEngine",
    package="models/flint",
    serve="engine.models.flint.serving:serve",
    manifest="engine.models.flint.serving:manifest",
    rank=5,
    # the arg-key call form; the template opens no reasoning block, the model writes `<think>` when it reasons
    # SRV-697: with thinking off and only the empty block, 7 of 10 hard asks (greedy) and 13 of 20 (the card's
    # sampling) opened the answer with reasoning, median 582 and 600 tokens (of at most 600); with this line, 0 of
    # 10 and 0 of 20, median 153 and 116. "Answer directly." in its place made the model say it had no tools on 13
    # of tool-eval-bench's 69 scenarios (score 67); this line, on none (75; 77 without a line). 2026-10-22, 4.0, raw
    # its template renders system and developer messages where they stand, with their own role
    spec=ServeSpec(tool_format="argkey", reasoning=ReasoningSpec(
        opens_itself=True,
        off_note="Do not show your reasoning. Reply with the answer, or call a tool when one helps."),
        prompt=PromptSpec(system_first=False)),
    uses=("engine/models/lfm2_moe/kernels.py", "engine/models/lfm2_moe/load.py", "engine/models/lfm2_moe/ref.py"),
    # SPD-145: the decode step's kernels as programmatic dependents, the shared expert on a second stream
    hooks=EngineHooks(pdl=True, side_stream=True),
))
