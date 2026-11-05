# `tools/gate/hw`: how the gate sizes a box

The quality gate (docs/quality-gate.md) asks one profile per box how hard to load it. The profile
answers how many requests run at once, the prompt depths, the memory ceilings, the reference pack,
and the rounds a speed row takes. Each GPU has one file. A size that holds on one GPU never moves
another GPU's sizes.

```
tools/gate/hw/
  __init__.py   detect() the box (nvidia-smi and the engine's torch), pick() its profile
  base.py       Profile: every question, with conservative answers
  gb10.py       the DGX Spark: 8 lanes, 66 / 77 GiB ceilings, Flash-Next 3.05
  ada.py        Ada cards: sizes from the card's memory, LFM2.5 EXL3
  generic.py    a GPU without a profile: small sizes from its memory, `--pack` names the pack
  fake.py       the fake engine's box, for the gate's own tests
```

## How a profile is picked

`detect()` reads the fingerprint: the name, memory, driver and compute capability from nvidia-smi, and
what torch sees from the probe that `tools/gpu` owns (`tools.gpu.PROBE`). Which GPU the box has is
`tools/gpu`'s answer (`tools/gpu/README.md`). A profile names the `tools/gpu` adapter it sizes for
(`gpu_adapter`), and `pick()` takes the first one whose adapter claims the fingerprint. When none
does, `generic.py` answers. `--hw NAME` picks a profile by its id.

## Adding a profile

Write `tools/gate/hw/<gpu>.py`, a `Profile` subclass with its `id`, its `gpu_adapter` and the sizes
it changes, and add it to `ADAPTERS` in `__init__.py`. The GPU itself (capability, name, arch flags,
kernel choices) belongs in `tools/gpu`, not here.

## Rules

- No GPU-specific size of the gate lives outside these files.
- A profile never reads the capability. It asks `tools/gpu` through `gpu_adapter`.
- `tests/test_gate_hw.py` and `tests/test_gpu_adapters.py` pin the picking and the sizes.

The engine's own per-GPU choices are in `tools/gpu`, and its per-model choices are in the model adapters
(`engine/models/README.md`).
