"""The 27B family's own knobs (ENG-405): the same shape as engine/settings.py's `KNOBS`, joined to it at
import. Only knobs owned by this package's files go here; the 27B's older knobs stay in the shared table."""

KNOBS: dict[str, tuple] = {
    # SPD-153: a chain-shaped tree block shares the joint verify pass of a parallel round (0: it takes a pass
    # of its own, its chain kernel, while the other slots wait)
    'JOINT_CHAINS': ('1', 'engine/models/qwen3_5/adapter.py'),
    # SPD-163: the joint verify pass from CUDA graphs (engine/models/qwen3_5/joint_graphs.py); 0: eager
    'JOINT_GRAPHS': ('0', 'engine/models/qwen3_5/joint_graphs.py'),
    # SPD-163: joint graphs kept, the least recently replayed dropped first
    'JOINT_GRAPHS_MAX': ('12', 'engine/models/qwen3_5/joint_graphs.py'),
}
