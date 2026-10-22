"""The flint family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register
from engine.models.adapter import EngineHooks, ReasoningSpec, ServeSpec

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
    # SRV-697: with thinking off and only the empty block, 8 of 10 hard asks (greedy) and 15 of 20 (the card's
    # sampling) opened the answer with reasoning, median 582 and 534 tokens; with this line first, 1 of 10 and
    # 1 of 20, median 145 and 128 (2026-10-22, live 4.0)
    spec=ServeSpec(tool_format="argkey", reasoning=ReasoningSpec(
        opens_itself=True, off_note="Answer directly. Do not show your reasoning.")),
    uses=("engine/models/lfm2_moe/kernels.py", "engine/models/lfm2_moe/load.py", "engine/models/lfm2_moe/ref.py"),
    # SPD-145: the decode step's kernels as programmatic dependents, the shared expert on a second stream
    hooks=EngineHooks(pdl=True, side_stream=True),
))
