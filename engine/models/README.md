# engine/models: one adapter per model family

Each model family the engine serves is one package here. The package holds the family's own code: its
config reader, its engine, its loader, its knobs and its server builder. The shared methods (the
server's loop, the caches, StairCut, the router) read what a family tells them as data. They never
branch on the model, so work on one family never moves another family's rows.

```
engine/models/
  __init__.py     the registry: discovers each package's arch.py, picks a checkpoint's family
  adapter.py      the data a family hands the shared methods (Capabilities, CostModel, ServeSpec, hooks)
  <family>/
    arch.py       register(Arch(...)): what selects the family, and its config, engine, server, manifest
    knobs.py      the family's own QWEN38_* knobs (ENG-405)
    serving.py    the server builder and the fingerprint manifest
    ...           the family's engine, loader and kernels' Python sides
```

## What an adapter answers

The `Arch` entry says which checkpoints the family serves (`architectures`, `model_types`), how to
read their config, which engine runs them, how the server builds them (`serve`), and what their stored
rows depend on (`manifest`, the fingerprint v3, engine/methods/manifest.py). It also gives what the
server needs to know (`spec`: the tool-call form and the reasoning), the launch options the family
turns on in the shared kernels (`hooks`), and the files it shares with other families or uses from
them (`shares`, `uses`). The engine object gives the loop's verbs, its `caps` and its `cost()`
(engine/models/adapter.py).

## How one is picked

A checkpoint's own config picks its family, never its file name: `architecture(raw_config)` returns
the first registered `Arch` whose `architectures` or `model_types` match, by `rank`.

## Adding a family

Add a package with an `arch.py` that registers its `Arch`. The registry discovers it, and no shared
file changes. Its knobs go in its own `knobs.py`, and its manifest lists the files that compute its
rows.

## Rules

- Shared code (server/, proxy/, engine/*.py, engine/methods/, engine/drafters/) never names a model.
- A family imports another family's files only when it declares them in `uses`.
- A family's knobs live in its package.
- `tests/test_adapter_guards.py` pins each family's fingerprint to its golden. It checks that a change
  to one package moves only the families that hold the file, holds the lint and the imports, and
  checks the contract.

Per-GPU choices are not a family's. They live in the GPU adapters (`tools/gpu/README.md`), and the
gate's sizes for a box live in `tools/gate/hw/README.md`.
