"""SRV-309: the parallel window cap holds while another slot decodes, also a slot the round
runner decodes (phase "runner", SPD-201) -- the phase every 0.15 decode is in."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

from server import app  # noqa: E402
from server import slots as S  # noqa: E402


def _cap(phases, rows=1024):
    pool = S.Pool([S.Slot(i, 65_536) for i in range(len(phases))])
    for s, ph in zip(pool.slots, phases):
        s.phase = ph
    saved = (dict.get(app.STATE, "pool"), app.PAR_WINDOW_ROWS)
    app.STATE["pool"], app.PAR_WINDOW_ROWS = pool, rows
    try:
        return app._par_window_cap()
    finally:
        app.STATE["pool"], app.PAR_WINDOW_ROWS = saved


def test_a_side_slot_the_runner_decodes_caps_the_prefill():
    assert _cap(["prefill", "runner"]) == 1024


def test_a_side_slot_decoding_on_its_own_caps_the_prefill():
    assert _cap(["prefill", "decode"]) == 1024


def test_no_other_slot_decoding_leaves_the_prefill_whole():
    assert _cap(["prefill", "idle"]) is None
    assert _cap(["runner", "idle"]) is None          # the prefilling slot's own phase is not a turn


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
