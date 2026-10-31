"""Flash-Next's own knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import."""

KNOBS: dict[str, tuple] = {
    # SPD-215: the joint pass runs its row-wise parts (the mixers' norms and mixes, the injects, the MoE
    # combine, the head's mixer) once over every state's rows; 0: a state at a time
    'FN_JOINT_ROWWISE': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-215: the joint pass's experts through the grouped kernels (tools/fn_kernels.py MOE_GROUP); 0: a CTA a pair
    'FN_JOINT_MOE_GROUP': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-233: the rows an 8-lane joint pass keeps (24: the best of 8/16/24/32 at C8); 0: the cap, 32
    'FN_JOINT_ROWS8': ('24', 'engine/models/flashnext/serving.py'),
    # the rows an 8-lane joint pass keeps on NVIDIA FP4-Local (its kernels are row-invariant to 64): served C8
    # 2026-10-31, code / prose tok/s at 24 / 32 / 40 / 48 / 64 rows: 141.6 / 162.7 / 183.8 / 199.5 / 205.8 and
    # 125.4 / 120.4 / 138.0 / 135.3 / 140.3; 0: the cap, 64
    'FN_NV_JOINT_ROWS8': ('64', 'engine/models/flashnext/serving.py'),
    # SPD-239: every N joint passes, log their GPU ms by part (`verify_multi`'s marks; a measurement: each
    # pass then syncs at its end); 0: off
    'FN_JOINT_TIMING': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-239: a joint pass reads every state's PLE rows in one batch a layer (a cold table: one round of disk
    # reads, not one a block row); 0: each block's own (a block's rows are always one batch)
    'FN_PLE_BATCH': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-239: the cold n-gram table's reads in flight a miss batch (its thread pool)
    'FN_PLE_THREADS': ('16', 'engine/models/flashnext/ple.py'),
    # SPD-239: a joint pass reads its PLE rows on a thread while the layers before the first PLE layer run (the
    # same rows; served steady rounds 78.4 -> 77.5 ms at 4 lanes, 114.2 -> 112.4 ms at 8); 0: before the pass
    'FN_PLE_LATE': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-239: a joint round's commits pool every lane's indexer blocks of every attention layer in one launch (the
    # pooling kernel's own arithmetic); 0: a launch and a copy a lane and layer
    'FN_POOL_MULTI': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-239: the joint pass's GDN delta rule of a layer for every state in one launch (each state's own
    # arithmetic, its tree's scratch slots its own; served C8 93.2 -> 94.2, GDN 17.4 -> 13.5 ms a pass); 0: a launch a state
    'FN_GDN_MULTI': ('1', 'engine/models/flashnext/engine.py'),
    # load-time quantisation rules `scope:pattern=from>to` (engine/models/flashnext/quant.py): draft:lm_head=bf16>nvfp4
    # builds the MTP drafter's head in NVFP4 (the verify reads the served head: the text is unchanged); empty = none
    'QUANT': ('', 'engine/models/flashnext/quant.py'),
    # the MTP head's shorter chains, a round drafting the smallest at or above 1.5 x its recent accepted drafts + 1
    # (engine/methods/mtp_drafter.py `_pick_depth`; the deepest is --depth); empty = every round drafts --depth
    'FN_DEPTH_LADDER': ('', 'engine/models/flashnext/mtp.py'),
}
