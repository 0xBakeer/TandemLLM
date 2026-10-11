"""The model layer: which architecture a checkpoint is, and the code that runs it (ENG-221).

The engine served one architecture, the Qwen3.8-27B (`engine/model.py`, `Qwen38Engine`), and every
caller knew it. A second one, Qwen3.8-Flash-Next (`qwen4_exp`, `engine/models/flashnext/`), is
selected here by the checkpoint's own config, never by its file name:

    arch = architecture(raw_config)       # an Arch, or ValueError naming the registered ones
    cfg = arch.load_config(snapshot)      # that model's config object

Each entry names its config reader and its engine by import path, so registering a model imports
nothing; `Arch.load_config()` / `Arch.engine_class()` import on use.

The package sits outside the disk tier's fingerprint (`engine/disk_cache._engine_files` leaves it
out like `decision/` and `generative/`): the 27B's fingerprint is the same bytes with or without
it, and a model under `engine/models/<pkg>/` adds its own package when it opts into the tier
(`disk_cache.fingerprint(..., packages=("models/<pkg>",))`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str                   # the registry key: the text config's `model_type` family
    architectures: tuple        # the checkpoint's `architectures` strings that select it
    model_types: tuple          # `model_type` values (top level or `text_config`) that select it
    config: str                 # "module:function" -> the model's config from a snapshot path
    engine: str                 # "module:Class" -> the model's engine
    package: str | None = None  # `engine/<package>` holds the model's own code (fingerprint)

    def _resolve(self, ref: str):
        mod, attr = ref.split(":")
        return getattr(importlib.import_module(mod), attr)

    def load_config(self, path: str):
        return self._resolve(self.config)(path)

    def engine_class(self):
        return self._resolve(self.engine)


REGISTRY: dict[str, Arch] = {}


def register(arch: Arch) -> Arch:
    REGISTRY[arch.name] = arch
    return arch


register(Arch(
    name="qwen3_5",
    architectures=("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"),
    model_types=("qwen3_5", "qwen3_5_text"),
    config="engine.config:load_config",
    engine="engine.models.qwen3_5.adapter:Qwen38Engine",     # engine/model.py's, with the cold verbs
))
register(Arch(
    name="qwen4_exp",
    architectures=("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"),
    model_types=("qwen4_exp", "qwen4_exp_text"),
    config="engine.models.flashnext.config:load_flashnext_config",
    engine="engine.models.flashnext.engine:FlashNextEngine",
    package="models/flashnext",
))


def architecture(raw: dict) -> Arch:
    """The registered architecture of a checkpoint's config.json, by its `architectures` list
    first, then by `model_type` (top level, then `text_config`)."""
    names = [str(a) for a in (raw.get("architectures") or [])]
    types = [str(raw.get("model_type", "")), str((raw.get("text_config") or {}).get("model_type", ""))]
    for arch in REGISTRY.values():
        if any(a in arch.architectures for a in names) or any(t in arch.model_types for t in types if t):
            return arch
    what = names[0] if names else (types[0] or types[1] or "<no architectures, no model_type>")
    known = ", ".join(f"{a.name} ({', '.join(a.architectures)})" for a in REGISTRY.values())
    raise ValueError(f"unknown architecture {what!r}: this engine runs {known}")
