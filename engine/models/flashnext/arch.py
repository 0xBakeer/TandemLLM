"""The qwen4_exp family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register

ARCH = register(Arch(
    name="qwen4_exp",
    architectures=("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"),
    model_types=("qwen4_exp", "qwen4_exp_text"),
    config="engine.models.flashnext.config:load_flashnext_config",
    engine="engine.models.flashnext.engine:FlashNextEngine",
    package="models/flashnext",
    serve="engine.models.flashnext.serving:serve",
    manifest="engine.models.flashnext.serving:manifest",
    rank=1,
))
