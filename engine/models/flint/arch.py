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
    spec=ServeSpec(tool_format="argkey", reasoning=ReasoningSpec(opens_itself=True)),
    uses=("engine/models/lfm2_moe/kernels.py", "engine/models/lfm2_moe/load.py", "engine/models/lfm2_moe/ref.py"),
    # SPD-145: the decode step's kernels as programmatic dependents, the shared expert on a second stream
    hooks=EngineHooks(pdl=True, side_stream=True),
))
