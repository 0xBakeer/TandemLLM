"""An untagged item finds its row through the (bid, p, c) index, not a full scan.

tick() matched every loose item by scanning every row in `done`. Both stores hold
up to ITEMS_MAX entries and the scan ran under the lock ingest() takes, so a busy
proxy could hold the request path for seconds per tick. The rows are now indexed
by (bid, prompt_tokens, completion_tokens) — the same candidates, without the
sweep. This test counts the sweeps: a dict subclass records every .items() call
on `done` while 200 loose items wait next to 5000 rows.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy import collect_api  # noqa: E402


class _Counting(dict):
    def __init__(self):
        super().__init__()
        self.sweeps = 0

    def items(self):
        self.sweeps += 1                     # the one from _expire's housekeeping
        return super().items()


def test_no_per_item_sweep_of_done():
    col = collect_api.Collector.__new__(collect_api.Collector)
    col._lock = __import__("threading").Lock()
    col.done = _Counting()
    col.done_by = {}
    col.items = {}
    col.loose = {}
    col.expect = {}
    col.backends = {}
    col._n = 0
    col.svc = None
    col.log = lambda *a: None
    t = [1000.0]
    col.clock = lambda: t[0]
    col._b = lambda bid: collect_api._Backend()

    # 5000 rows near the cap, 200 loose items of the same token counts (ambiguous on
    # purpose: the answer is 'no guess' either way — only the work per item is the point)
    for i in range(5000):
        dk = ("v1", f"row:{i}")
        col.done[dk] = {"request_id": i, "ts_ms": 0, "t": 1000.0, "end": 1000.0,
                        "prompt_tokens": 7, "completion_tokens": 9, "tagged": True}
        col.done_by.setdefault(("v1", 7, 9), set()).add(dk)
    for n in range(200):
        col.loose[("v1", n)] = ({"prompt_tokens": 7, "completion_tokens": 9,
                                "t_end": 1000.0}, 1000.0)

    col.tick()                              # one round of the old code = 200 sweeps
    assert len(col.loose) == 200, "the ambiguous case still declines to guess"
    assert col.done.sweeps <= 1, f"done was swept {col.done.sweeps} times in one tick"


if __name__ == "__main__":
    test_no_per_item_sweep_of_done()
    print("1 passed")
