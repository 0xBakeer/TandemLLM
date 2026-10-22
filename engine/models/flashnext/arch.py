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
    # the trunk the Qwen3.5 MoE family builds on: the engine, loader, reference and tower (in its rows), the
    # config, pack index and adapter it extends
    shares=("engine/models/flashnext/engine.py", "engine/models/flashnext/load.py", "engine/models/flashnext/ref.py",
            "engine/models/flashnext/vision.py", "engine/models/flashnext/config.py", "engine/models/flashnext/pack.py",
            "engine/models/flashnext/ple.py", "engine/models/flashnext/serving.py"),
))
