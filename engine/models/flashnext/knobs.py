"""Flash-Next's own knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import."""

KNOBS: dict[str, tuple] = {
    # SPD-215: the joint pass runs its row-wise parts (the mixers' norms and mixes, the injects, the MoE
    # combine, the head's mixer) once over every state's rows; 0: a state at a time
    'FN_JOINT_ROWWISE': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-215: the joint pass's experts through the grouped kernels (tools/fn_kernels.py MOE_GROUP); 0: a CTA a pair
    'FN_JOINT_MOE_GROUP': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-233: the rows an 8-lane joint pass keeps (24: the best of 8/16/24/32 at C8); 0: the cap, 32
    'FN_JOINT_ROWS8': ('24', 'engine/models/flashnext/serving.py'),
}
