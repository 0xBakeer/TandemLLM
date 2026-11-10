"""LFM2's knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import. This file
is in LFM2's manifest, so its defaults are part of LFM2's fingerprint and of no other model's."""

KNOBS: dict[str, tuple] = {
    # LFM2: the attention layers' K and V as e4m3 codes with one fp32 scale a (head, token); off (the exact bf16 cache)
    'LFM2_KV_FP8': ('0', 'engine/models/lfm2_moe/engine.py'),
}
