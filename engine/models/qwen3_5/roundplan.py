"""The 27B's round planner (SPD-187): the planner of engine/methods/roundplan.py over this family's joint pass.

QWEN38_ROUND_PLAN: off (the fixed cut), shadow (the plan is computed and logged, the round keeps the fixed
cut), on (the round takes the plan's bucket: each tree cut to T rows, the joint graph at T rows a block).
The buckets are the joint graphs' row counts (4, 8, 16; a block is graphed up to the grouped WY tile), the
row cap the family's joint_policy. Prices: QWEN38_ROUND_PLAN_PRICES (a PriceModel file with kinds
"joint{E}", rows E x T, by context class: tools/verify_curve.py --joint), followed online from the round's
CUDA events (server/app.py `_run_round`).
"""

from __future__ import annotations

import os

from engine.methods.roundplan import RoundPlanner
from engine.methods.staircut import PriceModel
from engine.settings import SETTINGS as _S

MODE = (_S.get("ROUND_PLAN") or "off").strip().lower()
PRICES = _S.get("ROUND_PLAN_PRICES") or ""
#: the rows a block may take in a planned round (the joint graphs' tile caps them at 16)
BUCKETS = tuple(int(x) for x in (_S.get("ROUND_PLAN_BUCKETS") or "4,8,16").split(","))
#: SPD-187: where the planner's learned state is kept ('' : ~/.tandemllm/roundplan/<pack>.<host>.json)
STATE_PATH = _S.get("ROUND_PLAN_STATE") or ""
#: SPD-187: rounds between two saves of the learned state
SAVE_EVERY = int(_S.get("ROUND_PLAN_SAVE_EVERY") or 512)

_PLANNERS: dict = {}


def ctx_class(n: int) -> int:
    """The planner's context class: the next power of two from 1,024 (the verify graphs' classes)."""
    c = 1024
    while c < n:
        c *= 2
    return c


def _state_path(eng) -> str:
    """The planner's learned state, one file per weights pack and box (QWEN38_ROUND_PLAN_STATE: another)."""
    if STATE_PATH:
        return os.path.expanduser(STATE_PATH)
    import socket
    w = getattr(eng, "w", None)
    pack = os.path.basename(os.path.normpath(getattr(w, "exl3_source", None) or getattr(w, "snapshot", "") or "pack"))
    return os.path.join(os.path.expanduser("~/.tandemllm/roundplan"), f"{pack}.{socket.gethostname()}.json")


def planner_for(eng, max_rows: int) -> RoundPlanner | None:
    """The planner of the engines over `eng`'s weights, `max_rows` the family's row cap (None while
    QWEN38_ROUND_PLAN is off)."""
    if MODE not in ("shadow", "on"):
        return None
    key = id(getattr(eng, "w", eng))
    rp = _PLANNERS.get(key)
    if rp is None:
        path = PRICES or os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "prices",
                                      "qwen3.8-27b", "joint.dgx-spark.json")
        path = os.path.normpath(path)
        pm = PriceModel.load(path)
        cap = int(max_rows)
        rp = _PLANNERS[key] = RoundPlanner(pm, buckets=BUCKETS, cap=lambda E, c=cap: c)
        rp.mode = MODE
        # SPD-187, learned online: planning on, every allowed bucket's round period is measured first (a short
        # warm-up); the payoff from the acceptance model (learned in shadow too); the reference cut is the
        # fixed rule's bucket (the joint graphs' t_pad); the state is kept per pack and box and warm-started
        from engine.methods.accept_model import AcceptModel
        from engine.models.qwen3_5 import joint_graphs as JG
        rp.explore = MODE == "on"
        rp.accept = AcceptModel()
        rp.ref_T = lambda E, c=cap: JG.t_pad(E, c)
        rp.state_path = _state_path(eng)
        rp.save_every = max(1, SAVE_EVERY)
        rp.warm = False
        try:
            import json
            with open(rp.state_path) as f:
                rp.load(json.load(f))
            rp.warm = True                      # warm-started from the saved state (the report says so)
        except (OSError, ValueError):
            pass
        rp.snap = rp._freeze()
        # the rest of a round by E (the packed draft and the slots' tails) from the file, followed online
        try:
            import json
            with open(path) as f:
                over = json.load(f).get("overhead_ms") or {}
        except (OSError, ValueError):
            over = {}
        rp.tails.update({int(k): float(v) for k, v in over.items()})
        rp.snap = rp._freeze()
        from engine.models.qwen3_5 import joint_graphs as JG

        def force(T):
            JG.FORCE_T = T                  # the joint graphs take the round's bucket (None: their own)
        rp.force = force
    return rp
