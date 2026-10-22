"""The qwen3_5_moe family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register

ARCH = register(Arch(
    name="qwen3_5_moe",
    architectures=("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"),
    model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
    config="engine.models.qwen3_5_moe.config:load_qwen35_moe_config",
    engine="engine.models.qwen3_5_moe.engine:Qwen35MoeEngine",
    package="models/qwen3_5_moe",
    serve="engine.models.qwen3_5_moe.serving:serve",
    manifest="engine.models.qwen3_5_moe.serving:manifest",
    rank=4,
))
