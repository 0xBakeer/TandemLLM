"""Qwen3.5 MoE's own knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import. The
family's older knobs stay in the shared table (tests/test_adapter_guards.py LEGACY_FAMILY_KNOBS)."""

KNOBS: dict[str, tuple] = {
    # SPD-150: the attention layers keep K and V as e4m3 codes with one fp32 scale a (head, token) (the MTP layer's
    # cache stays bf16); 0: the exact bf16 cache
    'Q35M_KV_FP8': ('0', 'engine/models/qwen3_5_moe/engine.py'),
}
