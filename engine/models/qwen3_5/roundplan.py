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

_PLANNERS: dict = {}


def ctx_class(n: int) -> int:
    """The planner's context class: the next power of two from 1,024 (the verify graphs' classes)."""
    c = 1024
    while c < n:
        c *= 2
    return c


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
        pm = PriceModel.load(os.path.normpath(path))
        cap = int(max_rows)
        rp = _PLANNERS[key] = RoundPlanner(pm, buckets=BUCKETS, cap=lambda E, c=cap: c)
        rp.mode = MODE
        from engine.models.qwen3_5 import joint_graphs as JG

        def force(T):
            JG.FORCE_T = T                  # the joint graphs take the round's bucket (None: their own)
        rp.force = force
    return rp
