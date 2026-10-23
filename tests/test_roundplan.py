"""SPD-187: the round planner (engine/methods/roundplan.py) on the CPU.

  1. a plan is a pure function of the key and the snapshot: 10,000 random rounds planned by two planners
     on the same snapshot hash the same;
  2. monotone: more q never takes fewer rows a block, a dearer large bucket never more;
  3. the hold zone: a challenger 4 % better than the incumbent bucket does not move it, 6 % does; a price
     move of 9 % never cuts a new snapshot, one of 11 % does, at the snapshot boundary only;
  4. `split` keeps at most T nodes a slot and an ancestor-closed set; `gain_curve` is the best-first
     cumulative q, flat past the tree.
"""

from __future__ import annotations

import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.methods.roundplan import RoundPlanner, gain_curve, split  # noqa: E402
from engine.methods.staircut import PriceModel  # noqa: E402


def prices(scale=1.0, big=1.0):
    t = {}
    for E in (2, 4, 8):
        rows = {E * T: (60.0 + 0.5 * E * T + (15.0 if E * T > 64 else 0.0)) * scale * (big if E * T > 32 else 1.0)
                for T in (4, 8, 16, 32)}
        t[f"joint{E}"] = rows
    return PriceModel({1024: t})


def rand_tree(rng, n):
    parents, path = [-1], [0]
    for i in range(1, n):
        path = path[:rng.randint(1, len(path))]
        parents.append(path[-1])
        path.append(i)
    scores = [1.0] + [rng.random() ** 2 for _ in range(n - 1)]
    return parents, scores


def curves(rng, E, k=1.0):
    out = []
    for _ in range(E):
        parents, scores = rand_tree(rng, rng.randint(2, 32))
        out.append([g * k for g in gain_curve(None, parents, scores, 32)])
    return out


def test_a_plan_is_a_function_of_the_key_and_the_snapshot():
    digests = []
    for _ in range(2):
        rp = RoundPlanner(prices(), cap=lambda E: 128)
        rp.D, rp.tails = {2: 13.0, 4: 18.0, 8: 33.0}, {2: 10.0, 4: 25.0, 8: 55.0}
        rp.snap = rp._freeze()
        rng = random.Random(1)
        h = hashlib.sha256()
        for _ in range(10_000):
            E = rng.choice((2, 4, 8))
            p = rp.plan(E, 1024, curves(rng, E))
            h.update(repr((p["T"], p["key"], p["version"])).encode())
        digests.append(h.hexdigest())
    assert digests[0] == digests[1]
    return f"10,000 rounds, two planners on one snapshot: the same plans ({digests[0][:12]})"


def test_more_q_never_fewer_rows_and_a_dearer_bucket_never_more():
    rng = random.Random(2)
    for _ in range(300):
        E = rng.choice((2, 4, 8))
        seed = rng.random()
        Ts = []
        for k in (0.25, 0.5, 1.0, 2.0, 4.0):
            rp = RoundPlanner(prices(), cap=lambda E: 128)
            Ts.append(rp.plan(E, 1024, curves(random.Random(seed), E, k))["T"])
        assert Ts == sorted(Ts), Ts
        Tp = []
        for big in (1.0, 1.2, 1.5, 2.0):
            rp = RoundPlanner(prices(big=big), cap=lambda E: 128)
            Tp.append(rp.plan(E, 1024, curves(random.Random(seed), E))["T"])
        assert Tp == sorted(Tp, reverse=True), Tp
    return "300 groups: the bucket never falls with more q, never rises with a dearer large pass"


def test_the_hold_zone_and_the_snapshots():
    rp = RoundPlanner(prices(), buckets=(8, 16), cap=lambda E: 128, every=4)
    c = [[0.0] + [1.0] * 31 for _ in range(4)]
    rp.incumbent[(0, rp.key(4, 1024, c))] = 8
    cost8, cost16 = rp.cost(4, 8, 1024), rp.cost(4, 16, 1024)

    def with_gain(r16):
        # slot curves whose 16-row gain makes the 16 bucket score r16 times the 8 bucket's
        g8 = 3.0
        s8 = 4 * (1 + g8) / cost8
        g16 = (r16 * s8 * cost16) / 4 - 1
        return [[0.0] * 7 + [g8] * 8 + [g16] * 17 for _ in range(4)]
    k = rp.key(4, 1024, with_gain(1.04))
    rp.incumbent[(0, k)] = 8
    assert rp.plan(4, 1024, with_gain(1.04))["T"] == 8, "4 % better: the incumbent holds"
    rp.incumbent[(0, rp.key(4, 1024, with_gain(1.06)))] = 8
    assert rp.plan(4, 1024, with_gain(1.06))["T"] == 16, "6 % better: it moves"
    # snapshots: a 9 % move does not cut one, an 11 % move does (at the 4-round boundary)
    for f, cuts in ((1.09, False), (1.11, True)):
        # a group whose 16 bucket wins by 8 %: either move turns the decision, only 11 % is a price move
        rp = RoundPlanner(prices(), buckets=(8, 16), cap=lambda E: 128, every=4)
        cost8, cost16 = rp.cost(4, 8, 1024), rp.cost(4, 16, 1024)
        g16 = (1.08 * 4 * 4.0 / cost8 * cost16) / 4 - 1
        cv = [[0.0] * 7 + [3.0] * 8 + [g16] * 17 for _ in range(4)]
        keys = [(4, 1024, cv)]
        assert rp.plan(4, 1024, cv)["T"] == 16
        rp.live.table[1024]["joint4"][64] *= f
        moved = [rp.tick(keys) for _ in range(4)]
        assert moved[:3] == [False] * 3, "only at the boundary"
        assert moved[3] == cuts, (f, moved)
        assert rp.version == (1 if cuts else 0)
        assert rp.plan(4, 1024, cv)["T"] == (8 if cuts else 16)
    return "4 % holds, 6 % moves; 9 % price move: same snapshot, 11 %: a new one at the boundary"


def test_split_and_the_gain_curve():
    rng = random.Random(4)
    for _ in range(200):
        nodes = [rand_tree(rng, rng.randint(2, 32)) for _ in range(rng.randint(2, 8))]
        T = rng.choice((4, 8, 16))
        for (parents, _), keep in zip(nodes, split(nodes, T)):
            assert len(keep) <= T and 0 in keep
            assert all(parents[i] in keep for i in keep if i)
    g = gain_curve([None, 0.9, 0.5, 0.4, 0.2], [-1, 0, 1, 0, 3], [1, 0.9, 0.5, 0.4, 0.2], 6)
    assert [round(x, 6) for x in g] == [0.0, 0.9, 1.4, 1.8, 2.0, 2.0]
    return "200 groups: within T a slot, ancestor-closed; the curve is best-first, flat past the tree"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:64s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
