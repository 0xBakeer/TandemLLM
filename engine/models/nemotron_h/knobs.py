"""NemotronH's knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import. This file
is in NemotronH's manifest, so its defaults are part of NemotronH's fingerprint and of no other model's."""

KNOBS: dict[str, tuple] = {
    # NemotronH: the attention layers' K and V as e4m3 codes with one fp32 scale a (head, token); off (the exact
    # bf16 cache). The MTP head's own cache stays bf16
    'NH_KV_FP8': ('0', 'engine/models/nemotron_h/engine.py'),
}
