"""The qwen3_5 family's entry in the model registry (engine/models/__init__.py discovers it, ENG-405)."""

from engine.models import Arch, register

ARCH = register(Arch(
    name="qwen3_5",
    architectures=("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"),
    model_types=("qwen3_5", "qwen3_5_text"),
    config="engine.config:load_config",
    engine="engine.models.qwen3_5.adapter:Qwen38Engine",     # engine/model.py's, with the cold verbs
    manifest="engine.models.qwen3_5.adapter:manifest",
    rank=0,
))
