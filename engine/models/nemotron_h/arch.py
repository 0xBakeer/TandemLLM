"""The nemotron_h family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register

ARCH = register(Arch(
    name="nemotron_h",
    architectures=("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM", "NemotronH_Nano_Omni_Reasoning_V3",
                   "NemotronH_Omni_Reasoning_V3"),
    model_types=("nemotron_h", "nemotron_h_puzzle"),
    config="engine.models.nemotron_h.config:load_nemotron_h_config",
    engine="engine.models.nemotron_h.engine:NemotronHEngine",
    package="models/nemotron_h",
    serve="engine.models.nemotron_h.serving:serve",
    manifest="engine.models.nemotron_h.serving:manifest",
    rank=2,
    uses=("engine/models/lfm2_moe/kernels.py", "engine/models/lfm2_moe/pack.py"),
))
