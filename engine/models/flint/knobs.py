"""Flint's knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import. This file
is in Flint's manifest, so its defaults are part of Flint's fingerprint and of no other model's."""

KNOBS: dict[str, tuple] = {
    # Flint: the lookup cut by StairCut (0 = plain decode); rounds after an anchor-only cut that take the plain
    # step (1, 2, 4 .. up to this; 0 = off)
    'FLINT_LOOKUP': ('1', 'engine/models/flint/serving.py'),
    'FLINT_ANCHOR_BACKOFF': ('0', 'engine/models/flint/serving.py'),
    # Flint: rounds after a lookup miss (or an anchor-only cut) that take the plain step without looking, 1, 2, 4 ..
    'FLINT_LOOK_BACKOFF': ('0', 'engine/models/flint/serving.py'),
    # Flint: the shared expert on a second stream beside the routed experts (SPD-145)
    'FLINT_SIDE_STREAM': ('1', 'engine/models/flint/engine.py'),
    # Flint: the decode step's Triton kernels as programmatic dependent launches (SPD-145)
    'FLINT_PDL': ('1', 'engine/models/flint/engine.py'),
}
