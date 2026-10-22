"""The model layer: which architecture a checkpoint is, and the code that runs it (ENG-221, ENG-405).

Each family package declares itself in its own `arch.py` (`register(Arch(...))`), and this module
discovers them (`_discover`): adding a model edits no file here. A checkpoint's family is chosen by its
own config, never by its file name:

    arch = architecture(raw_config)       # an Arch, or ValueError naming the registered ones
    cfg = arch.load_config(snapshot)      # that model's config object

Each entry names its config reader, its engine, its server builder and its manifest by import path, so
registering a model imports nothing heavy; `Arch.load_config()` / `Arch.engine_class()` import on use.

The package sits outside the disk tier's fingerprint v2 (`engine/disk_cache._engine_files` leaves it
out like `decision/` and `generative/`). A model's fingerprint v3 (ENG-243) is what its adapter's
manifest lists (engine/methods/manifest.py); tests/test_adapter_guards.py holds every adapter's to its
golden and checks that a change to one package moves only its own.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field

from engine.models.adapter import ServeSpec


@dataclass(frozen=True)
class Arch:
    name: str                   # the registry key: the text config's `model_type` family
    architectures: tuple        # the checkpoint's `architectures` strings that select it
    model_types: tuple          # `model_type` values (top level or `text_config`) that select it
    config: str                 # "module:function" -> the model's config from a snapshot path
    engine: str                 # "module:Class" -> the model's engine
    package: str | None = None  # `engine/<package>` holds the model's own code (fingerprint)
    # "module:function" -> the model built for the server, `serve(a, ctx) -> dict` (server/app.py
    # `_load`, SRV-341); None: the server's own builder, the 27B's (`_build_qwen3_5`)
    serve: str | None = None
    # "module:function" -> the adapter's fingerprint manifest (engine/methods/manifest.py `Manifest`)
    manifest: str | None = None
    # the registry's order (architecture()'s message lists them so); a new family takes the default
    rank: int = 100
    # ENG-406: what the server needs to know about serving the family's model: its tool-call form and
    # how it reasons (engine/models/adapter.py)
    spec: ServeSpec = field(default_factory=ServeSpec)
    # the family `--model` serves when it names no checkpoint directory (one family is)
    default: bool = False

    def _resolve(self, ref: str):
        mod, attr = ref.split(":")
        return getattr(importlib.import_module(mod), attr)

    def load_config(self, path: str):
        return self._resolve(self.config)(path)

    def engine_class(self):
        return self._resolve(self.engine)

    def serve_builder(self):
        return self._resolve(self.serve) if self.serve else None

    def manifest_of(self):
        """The adapter's manifest (imports its module)."""
        return self._resolve(self.manifest)()


REGISTRY: dict[str, Arch] = {}


def register(arch: Arch) -> Arch:
    REGISTRY[arch.name] = arch
    return arch


def _discover() -> None:
    """ENG-405: every family package that has an `arch.py` registers itself there, so adding a model
    edits no shared file. Imported in name order, then kept in `rank` order."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(os.listdir(here)):
        if os.path.isfile(os.path.join(here, name, "arch.py")):
            importlib.import_module(f"{__name__}.{name}.arch")
    ordered = sorted(REGISTRY.values(), key=lambda a: (a.rank, a.name))
    REGISTRY.clear()
    REGISTRY.update((a.name, a) for a in ordered)


def default_arch() -> Arch:
    """The family a server without a checkpoint directory serves (its arch.py says `default=True`)."""
    found = [a for a in REGISTRY.values() if a.default]
    if len(found) != 1:
        raise ValueError(f"exactly one registered family is the default, found {[a.name for a in found]}")
    return found[0]


def architecture(raw: dict) -> Arch:
    """The registered architecture of a checkpoint's config.json, by its `architectures` list
    first, then by `model_type` (top level, then `text_config`, then an Omni wrapper's `llm_config`)."""
    names = [str(a) for a in (raw.get("architectures") or [])]
    types = [str(raw.get("model_type", "")), str((raw.get("text_config") or {}).get("model_type", "")),
             str((raw.get("llm_config") or {}).get("model_type", ""))]
    for arch in REGISTRY.values():
        if any(a in arch.architectures for a in names) or any(t in arch.model_types for t in types if t):
            return arch
    what = names[0] if names else (types[0] or types[1] or "<no architectures, no model_type>")
    known = ", ".join(f"{a.name} ({', '.join(a.architectures)})" for a in REGISTRY.values())
    raise ValueError(f"unknown architecture {what!r}: this engine runs {known}")


_discover()
