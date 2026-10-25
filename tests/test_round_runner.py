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


class _Tok:
    """A tokenizer of two specials for `ThinkBudget`: `<think>` and `</think>` are the ids given."""

    def __init__(self, open_id, end_id):
        self.ids = {"<think>": open_id, "</think>": end_id}

    def convert_tokens_to_ids(self, text):
        return self.ids.get(text, -1)

    def __call__(self, text, add_special_tokens=False):
        class _E:
            pass
        e = _E()
        e.input_ids = [self.ids[text]] if text in self.ids else [1, 2, 3]
        return e


def test_a_passive_reasoning_block_goes_with_the_runner_and_follows_the_slot_loop():
    from engine.spec import ThinkBudget
    cfg = T.tiny_config()
    w = T.FakeWeights(cfg, seed=5)
    n, k = 40, 2
    prompts = [[(3 + 7 * i + 11 * j) % 96 + 1 for j in range(10 + 3 * i)] for i in range(k)]
    refs = [T._greedy(cfg, w, p, n) for p in prompts]
    fulls = [p + r for p, r in zip(prompts, refs)]
    # each request opens its block at its 4th token and closes it at its 20th (when those differ)
    toks = [_Tok(r[3], r[19] if r[19] != r[3] else -5) for r in refs]
    alone_t = [ThinkBudget(tk) for tk in toks]
    alone = [T._alone(cfg, w, p, f, n, think=th) for p, f, th in zip(prompts, fulls, alone_t)]
    R.RUNNER = None
    R.SKIPS.clear()
    tog_t = [ThinkBudget(tk) for tk in toks]
    together, _, phases = T._together(cfg, w, prompts, fulls, n, thinks=tog_t)
    for i in range(k):
        assert together[i] == alone[i], (i, alone[i], together[i])
        a, b = alone_t[i], tog_t[i]
        assert (a.inside, a.done, a.n, a._seen) == (b.inside, b.done, b.n, b._seen), (i, (a.inside, a.done, a.n, a._seen), (b.inside, b.done, b.n, b._seen))
    assert R.RUNNER is not None and R.RUNNER.stats["tasks"] >= k - 1, R.RUNNER.stats
    assert "think_budget" not in R.SKIPS, R.SKIPS
    assert all(p == "" for p in phases), phases
    seen = sum(int(t.done or t.inside) for t in tog_t)
    return f"{k} requests with a passive block: alone texts, block states equal ({seen} opened)"


def test_seeded_requests_through_the_runner_write_what_they_write_alone():
    """Seeded sampled requests (the det sampled tree, keyed draws) through the runner: each its alone text."""
    from engine.sample import Sampler
    cfg = T.tiny_config()
    w = T.FakeWeights(cfg, seed=5)
    n, k = 40, 3
    prompts = [[(3 + 7 * i + 11 * j) % 96 + 1 for j in range(10 + 3 * i)] for i in range(k)]
    refs = [T._greedy(cfg, w, p, n) for p in prompts]
    fulls = [p + r for p, r in zip(prompts, refs)]
    mk = lambda i: Sampler(temperature=1.0, top_p=0.95, top_k=20, seed=1000 + i)
    alone = [T._alone(cfg, w, p, f, n, sampler=mk(i), sampled_tree="det")
             for i, (p, f) in enumerate(zip(prompts, fulls))]
    R.RUNNER = None
    together, _, phases = T._together(cfg, w, prompts, fulls, n, samplers=[mk(i) for i in range(k)],
                                      sampled_tree="det")
    bad = [i for i in range(k) if together[i] != alone[i]]
    assert not bad, [(i, alone[i], together[i]) for i in bad]
    assert R.RUNNER is not None and R.RUNNER.stats["tasks"] >= k - 1, R.RUNNER.stats
    return f"{k} seeded requests: each its alone text ({R.RUNNER.stats['joint']} joint passes)"


class _Lane:
    """Slot 0's side lane with one waiting job: `serve` runs it (counted) at the holder's safe point."""

    def __init__(self):
        self.jobs, self.served = [object()], []      # a job always waits: every safe point serves one

    def serve(self, where):
        self.served.append(where)
        return 1


def test_slot_0_goes_with_the_runner_and_its_lane_is_served_between_rounds():
    from server import app
    lane = _Lane()
    real = T._setup

    def setup(*a, **kw):
        pool = real(*a, **kw)
        app.STATE["lane"] = lane
        return pool
    T._setup = setup
    try:
        out = _check(2)
    finally:
        T._setup = real
    st = R.RUNNER.stats
    assert set(lane.served) == {"decode"}, lane.served
    assert st["lane_jobs"] >= st["rounds"] - 1 and "lane" not in R.SKIPS, (st, R.SKIPS)
    assert 2 in R.RUNNER.stats["joint_hist"], R.RUNNER.stats["joint_hist"]
    return out + f"; slot 0's lane served at {st['lane_jobs']} of {st['rounds']} round starts"


def test_a_block_that_can_be_closed_keeps_the_slot_loop():
    from engine.spec import ThinkBudget
    R.SKIPS.clear()
    kw = dict(sup=None, probe=None, lpr=None, ahead=None, drafter=object(), lane=None, turn=object())
    assert R.why_not(think=ThinkBudget(_Tok(5, 6), 64), **kw) == "think_budget"
    assert R.why_not(think=ThinkBudget(_Tok(5, 6), 0, arms={"loop": None}), **kw) == "think_budget"
    assert R.why_not(think=ThinkBudget(_Tok(5, 6), 0), **kw) is None
    assert R.why_not(think=None, **kw) is None
    assert R.SKIPS == {"think_budget": 2}, R.SKIPS
    return "a budget or stall arms: the slot's own loop (counted); a passive block or none: the runner"


def test_a_stream_read_half_way_and_closed_after_its_slot_went_back_leaves_the_turn_free():
    """SRV-709: a stream that ends at a stop string stops reading its generation half way. Its slot is
    given back (`_par_end`: the turn released, the slot's phase and owner cleared), and the generation is
    closed only later (when the handler's frame goes away). The runner's generator then must not take the
    slot's turn back: held for good, every lane waited (Ornith, 8 lanes, 2026-10-24)."""
    import threading
    import torch
    from server import app, slots
    cfg = T.tiny_config()
    w = T.FakeWeights(cfg, seed=5)
    n = 40
    prompts = [[(3 + 7 * i + 11 * j) % 96 + 1 for j in range(10 + 3 * i)] for i in range(2)]
    fulls = [p + T._greedy(cfg, w, p, n) for p in prompts]
    R.RUNNER = None
    pool = T._setup([T._engine(cfg, w) for _ in prompts], [T.OracleTree(f) for f in fulls])
    out0, errs = [], []
    go = threading.Barrier(2)
    left = threading.Event()

    def slot0():
        try:
            go.wait()
            with app.LOCK:
                with torch.no_grad():
                    for t in app.generate_stream(torch.tensor(prompts[0]), n, set()):
                        out0.append(t)
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    def slot1():
        s = pool.slots[1]
        try:
            go.wait()
            slots.bind(s)
            s.owner = "req-1"
            pool.turn.reset(1)
            pool.turn.acquire(1)
            gen = app.generate_stream(torch.tensor(prompts[1]), n, set())
            with torch.no_grad():
                for j, _ in enumerate(gen):
                    if j == 4:
                        break                                     # a stop string
            if pool.turn.held_by(1):                              # `_par_end`
                pool.turn.release(1)
            s.phase, s.owner = "", None
            slots.bind(None)
            left.set()
            gen.close()                                           # the frame went away
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    ths = [threading.Thread(target=slot0, daemon=True), threading.Thread(target=slot1, daemon=True)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    owner = pool.turn.owner
    alive = [th.is_alive() for th in ths]
    T._teardown()
    assert not errs, errs
    assert left.is_set() and alive == [False, False], alive
    assert owner is None, f"the turn held by {owner!r} after both requests ended"
    assert len(out0) == n, len(out0)
    return f"slot 1 read 5 of {n} tokens and closed late; slot 0 wrote its {len(out0)}; the turn free"


def test_the_familys_rows_at_a_lane_count_bound_the_joint_cut():
    """SPD-233: `joint_policy` "rows_at" (the meeting's `rows_at`): a joint round of that many lanes cuts its
    trees to those rows (never above the cap, never under 2 a lane), the texts stay the lanes' alone texts."""
    from server import slots
    real_cut, real_setup, seen = slots.joint_cut, T._setup, []

    def cut(trees, max_rows, cap=None):
        seen.append((len(trees), max_rows))
        return real_cut(trees, max_rows, cap)

    def setup(*a, **kw):
        pool = real_setup(*a, **kw)
        pool.meeting.rows_at = {2: 4}
        return pool
    slots.joint_cut, T._setup = cut, setup
    try:
        out = _check(2)
    finally:
        slots.joint_cut, T._setup = real_cut, real_setup
    assert seen and all(r == 4 for e, r in seen if e == 2), seen
    return f"{out}; {len(seen)} cuts at 2 lanes to 4 rows"


def test_the_report_gives_each_phase_by_the_slots_in_the_round():
    # SPD-239: slot counts with fewer than 8 rounds are left out; the rest are ms a round
    d = {8: {"n": 10, "verify": 1180.0, "tails": 52.0}, 2: {"n": 3, "verify": 90.0}}
    assert R.Runner._by_e(d) == {8: {"n": 10, "tails": 5.2, "verify": 118.0}}


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
