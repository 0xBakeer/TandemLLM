"""The flint family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register

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
))
