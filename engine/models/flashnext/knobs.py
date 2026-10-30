"""Flash-Next's own knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at import."""

KNOBS: dict[str, tuple] = {
    # SPD-215: the joint pass runs its row-wise parts (the mixers' norms and mixes, the injects, the MoE
    # combine, the head's mixer) once over every state's rows; 0: a state at a time
    'FN_JOINT_ROWWISE': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-215: the joint pass's experts through the grouped kernels (tools/fn_kernels.py MOE_GROUP); 0: a CTA a pair
    'FN_JOINT_MOE_GROUP': ('1', 'engine/models/flashnext/engine.py'),
    # SPD-233: the rows an 8-lane joint pass keeps (24: the best of 8/16/24/32 at C8); 0: the cap, 32
    'FN_JOINT_ROWS8': ('24', 'engine/models/flashnext/serving.py'),
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
    # SPD-249: the mixers' down|inject GEMV writes the mix_up's A in its launch (one launch a mixer fewer, the same
    # bits; int8 mixers on the tensor-core path); 0: mix_a in a launch of its own
    'FN_MIX_FUSED': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: a joint pass's GDN conv and gates (`gdn_pre`) of every state in one launch a layer (each row's own
    # arithmetic; 36 launches a pass instead of 36 a state); 0: a launch a state
    'FN_GDN_PRE_MULTI': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: a joint pass leaves up to N parked GDN rows parked instead of writing every state back (the write-back
    # every few rounds; each row's bits the same, an fp32 store and load being exact); 0: written back every call
    'FN_GDN_DEFER': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: every N-th joint pass counts its distinct routed experts a layer (the pass's expert bytes are that
    # union times one expert's), in /health (parallel.joint_stats) and the runner's line; 0: off (a diagnosis, set
    # it only to measure)
    'FN_UNION_EVERY': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: the MTP drafters (slot 0's and the side slots') keep their adaptive state at its start: the cut's
    # calibration and prices from the price file, no turn ageing, the usual answer length (A/B arms on fresh servers
    # draft alike for the same tokens); 0: they learn as served
    'FN_DRAFT_FREEZE': ('0', 'engine/models/flashnext/serving.py'),
    # SPD-249: a joint pass's attention layer: every lane's prep in one launch, its rows in one split and one combine
    # launch from lane entry tables, one gate over every row (each row's own arithmetic); 0: a lane at a time
    'FN_ATTN_MULTI': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: a joint pass's PLE layer: decode, projections, norms and gate once over every lane's rows, the conv a lane
    # at a time (each row's own arithmetic); 0: the PLE step a lane at a time
    'FN_PLE_MULTI': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: a dense joint pass from a CUDA graph keyed by (lanes, rows padded to 16/24/32, class 0), the lanes'
    # states through its entry tables, the pad rows routed to no expert; captured on the round runner only while no
    # request waits; 0: every joint pass eager
    'FN_JOINT_GRAPH': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: stage_rows' tree arrays (depth, paths, ancestor cells, GDN tree control, window sources) kept by tree
    # shape, and a joint graph's lane views kept by (start, rows): the same host rows with less host work; 0: per call
    'FN_TREE_PLAN': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: a joint graph pass reads its PLE rows on the reader thread while the lanes are staged and the segments
    # before the first PLE layer run (the eager pass's FN_PLE_LATE); the rows land before that layer's segment; 0: read
    # before the first segment
    'FN_JG_PLE_LATE': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: the round runner hands a lane's tree over when its tokens are known (before its cuts and staging): its PLE
    # rows are read into the table's row cache on the reader thread, the verify's own read finds them there; 0: off
    'FN_PLE_PREFETCH': ('0', 'engine/models/flashnext/engine.py'),
    # SPD-249: the joint graphs kept (the least recently replayed go first)
    'FN_JOINT_GRAPH_MAX': ('12', 'engine/models/flashnext/engine.py'),
    # SPD-249: a key is captured at its n-th pass (the eager passes before it compile what it runs)
    'FN_JOINT_GRAPH_AFTER': ('2', 'engine/models/flashnext/engine.py'),
}
