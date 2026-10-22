"""The lfm2_moe family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register
from engine.models.adapter import ServeSpec

ARCH = register(Arch(
    name="lfm2_moe",
    architectures=("Lfm2MoeForCausalLM",),
    model_types=("lfm2_moe",),
    config="engine.models.lfm2_moe.config:load_lfm2_moe_config",
    engine="engine.models.lfm2_moe.engine:Lfm2MoeEngine",
    package="models/lfm2_moe",
    serve="engine.models.lfm2_moe.serving:serve",
    manifest="engine.models.lfm2_moe.serving:manifest",
    rank=3,
    spec=ServeSpec(tool_format="pythonic"),      # LFM2's Pythonic call lists
    # the row kernels, the packs, the expert loaders and the reference other families build on
    shares=("engine/models/lfm2_moe/kernels.py", "engine/models/lfm2_moe/pack.py", "engine/models/lfm2_moe/load.py",
            "engine/models/lfm2_moe/ref.py"),
))
