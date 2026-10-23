"""SPD-201, QSE_ROUND_RUNNER: the runner-owned round on the CPU, with the random 4-layer model.

Two and three requests decoding at once through the round runner (server/round_runner.py) write, each,
the tokens they write alone; the runner ran joint passes (the requests verified together); every slot
leaves with its phase cleared and the turn free.

Run: python tests/test_round_runner.py
"""

from __future__ import annotations

import os
import sys

os.environ["QSE_ROUND_RUNNER"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_parallel_verify as T  # noqa: E402
from server import round_runner as R  # noqa: E402


def _check(k):
    cfg = T.tiny_config()
    w = T.FakeWeights(cfg, seed=5)
    n = 40
    prompts = [[(3 + 7 * i + 11 * j) % 96 + 1 for j in range(10 + 3 * i)] for i in range(k)]
    refs = [T._greedy(cfg, w, p, n) for p in prompts]
    fulls = [p + r for p, r in zip(prompts, refs)]
    alone = [T._alone(cfg, w, p, f, n) for p, f in zip(prompts, fulls)]
    R.RUNNER = None                          # a runner of this test's pool
    together, stats, phases = T._together(cfg, w, prompts, fulls, n)
    j0 = 0
    for i in range(k):
        assert together[i] == alone[i], (i, alone[i], together[i])
    assert R.RUNNER is not None and R.RUNNER.stats["joint"] > j0, R.RUNNER.stats if R.RUNNER else None
    assert all(p == "" for p in phases), phases
    assert R.RUNNER.stats["errors"] == 0
    return f"{k} requests: each its alone text; {R.RUNNER.stats['joint'] - j0} joint passes by the runner"


def test_two_requests_through_the_runner_write_what_they_write_alone():
    return _check(2)


def test_three_requests_through_the_runner_write_what_they_write_alone():
    return _check(3)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
