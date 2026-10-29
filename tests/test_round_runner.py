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


def test_a_block_that_can_be_closed_or_a_supervisor_goes_with_the_runner():
    """SPD-249: a budget, stall arms or a supervisor go with the runner (handed back at SP1, SP2 and the
    budget); QSE_RUNNER_SUPERVISED=0 keeps them on the slot loop, counted, as before."""
    from engine.spec import ThinkBudget
    kw = dict(probe=None, lpr=None, ahead=None, drafter=object(), lane=None, turn=object())
    saved = R.SUPERVISED
    try:
        for on in (True, False):
            R.SUPERVISED = on
            R.SKIPS.clear()
            got = [R.why_not(sup=None, think=ThinkBudget(_Tok(5, 6), 64), **kw),
                   R.why_not(sup=None, think=ThinkBudget(_Tok(5, 6), 0, arms={"loop": None}), **kw),
                   R.why_not(sup=object(), think=ThinkBudget(_Tok(5, 6), 0), **kw),
                   R.why_not(sup=None, think=ThinkBudget(_Tok(5, 6), 0), **kw),
                   R.why_not(sup=None, think=None, **kw)]
            if on:
                assert got == [None] * 5 and R.SKIPS == {}, (got, R.SKIPS)
            else:
                assert got == ["think_budget", "think_budget", "supervisor", None, None], got
                assert R.SKIPS == {"think_budget": 2, "supervisor": 1}, R.SKIPS
        R.SUPERVISED = True
        R.SKIPS.clear()
        assert R.why_not(sup=object(), think=None, probe=None, lpr=object(), ahead=None, drafter=object(),
                         lane=None, turn=object()) == "logprobs"
    finally:
        R.SUPERVISED = saved
    return "budget, arms, supervisor: the runner; QSE_RUNNER_SUPERVISED=0: the slot loop (counted)"


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



def test_lanes_take_the_mode_that_makes_more_tokens_a_ms():
    """SPD-233, QSE_RUNNER_LANE_PLAIN: at E lanes of joint-plain families the runner learns the drafted rounds'
    rate (4 rounds), then the plain rounds' (2), takes the better one with a probe of the other every
    LANE_PROBE-th round, and goes back when the other mode's rate passes it; a lane of a family that cannot join
    a plain step to the pass keeps drafting, and under "auto" so does a round with an engine that did not opt in."""
    from types import SimpleNamespace as NS
    saved = R.LANE_PLAIN, R.LANE_PROBE, R.LANE_MODE
    R.LANE_PLAIN, R.LANE_PROBE, R.LANE_MODE = True, 8, "1"
    try:
        r = R.Runner.__new__(R.Runner)
        r._lane = {}
        joint = NS(joint_plain=True)
        r.tasks = [NS(done=False, eng=NS(caps=joint), drafter=NS(lane_skip=False)) for _ in range(4)]
        modes = []
        for _ in range(40):
            plain = r._lane_plain()
            modes.append(plain)
            r._lane_note(4, plain, 4 / 40.0 if plain else 6 / 80.0)      # plain 0.1 tokens a ms, drafted 0.075
        assert modes[:4] == [False] * 4 and modes[4:6] == [True, True], modes
        assert all(m == ((i + 1) % 8 != 0) for i, m in enumerate(modes[6:], 6)), modes
        for _ in range(80):                                              # the drafts get better: 0.2 tokens a ms
            plain = r._lane_plain()
            modes.append(plain)
            r._lane_note(4, plain, 4 / 40.0 if plain else 16 / 80.0)
        tail = modes[-24:]
        assert tail.count(False) > tail.count(True) and r._lane[4]["plain"] is False, (tail, r._lane)
        r.tasks[1].eng = NS(caps=NS(joint_plain=False))
        assert r._lane_plain() is False and 4 in r._lane and 3 not in r._lane
        # "auto": only engines that opt in (`lane_plain`)
        R.LANE_MODE = "auto"
        r.tasks[1].eng = NS(caps=joint)
        assert r._lane_plain() is False
        for t in r.tasks:
            t.eng.lane_plain = True
        r._lane_plain()
        assert r._lane[4]["n"] > 120
    finally:
        R.LANE_PLAIN, R.LANE_PROBE, R.LANE_MODE = saved
    return f"plain after {modes.index(True)} drafted rounds; back to drafting when the drafts made more"


def test_the_lane_mode_is_decided_on_live_rounds_and_a_capture_sets_no_rate():
    """SPD-233: a round's `_propose` calls alternate as served (the round start's call of the tasks not yet drafted,
    empty, then the tail's for the next round): the mode is decided once a round, in the live call, so in plain
    mode the 16th decided round drafts (before, the counter stepped on both calls and the probe could land on the
    empty one every time); a round that captured a graph enters no rate, the next one does."""
    from types import SimpleNamespace as NS

    from engine.models import graph_lru
    saved = R.LANE_PLAIN, R.LANE_PROBE, R.LANE_MODE
    R.LANE_PLAIN, R.LANE_PROBE, R.LANE_MODE = True, 16, "1"

    def gen():
        return
        yield

    class Drafter:
        lane_skip = False

        def propose_tree_steps(self, ctx, k):
            return gen()
    try:
        r = R.Runner.__new__(R.Runner)
        r._lane, r._lane_at, r.stats = {}, (None, False), {"rounds": 0}
        r.ms = {k: 0.0 for k in ("prop", "packed")}
        r._ev, r._collect, r._mtp_collect = [], None, None
        r.pool = NS()
        r.app = NS(_row_cap=lambda pool, d: None, _launch=lambda steps: (None, None))
        joint = NS(joint_plain=True)
        r.tasks = [NS(done=False, pre=False, eng=NS(caps=joint), drafter=Drafter(), slot=NS(index=i), deadline=None,
                      ctx=[1, 2, 3], k=4, max_new=512, n_out=0) for i in range(8)]
        modes = []
        for _ in range(60):
            r.stats["rounds"] += 1
            r._propose([], 0.0)                                         # the round start: nothing left to draft
            r._propose(r.tasks, 0.0, deadline=False)                   # the tail: the next round's drafts
            plain = r.tasks[0].drafter.lane_skip
            modes.append(plain)
            r._lane_note(8, plain, 8 / 64.0 if plain else 11 / 114.0)
        st = r._lane[8]
        assert st["n"] == 60 and st["plain"] is True, st
        probes = [i for i, m in enumerate(modes) if not m and i >= 6]
        assert probes and all((i + 1) % 16 == 0 for i in probes), (probes, modes)
        assert all(m for i, m in enumerate(modes[6:], 6) if (i + 1) % 16), modes

        # a capture round's period is the capture's: no rate from it
        class Ev:
            def __init__(self, t):
                self.t = t

            def query(self):
                return True

            def elapsed_time(self, other):
                return other.t - self.t
        r._steady, r._gpu_e, r.gpu, r.gpu_n = {}, {}, {"verify": 0.0, "between": 0.0, "period": 0.0}, 0
        np0, p0 = st["np"], st["plain_rate"]
        r._ev_done = [([("start", Ev(0.0)), ("verify", Ev(300.0))], Ev(-1.0), 8, 8, 0, True, True)]
        r._gpu_read()
        assert st["np"] == np0 and st["plain_rate"] == p0, st
        r._ev_done = [([("start", Ev(0.0)), ("verify", Ev(64.0))], Ev(0.0), 8, 8, 0, True, False)]
        r._gpu_read()
        assert st["np"] == np0 + 1, st
        # what tells a round captured: the joint graph caches' count, a head's, an engine's flag
        for t in r.tasks:
            t.eng.fn = fn = NS(joint_graphs=graph_lru.GraphLRU())
        c0 = r._captures(r.tasks)
        fn.joint_graphs.captures += 1
        assert r._captures(r.tasks) != c0
        c0 = r._captures(r.tasks)
        r.tasks[3].drafter.head = NS(captures=1)
        assert r._captures(r.tasks) != c0
        c0 = r._captures(r.tasks)
        r.tasks[3].eng.captured = True
        assert r._captures(r.tasks) != c0
    finally:
        R.LANE_PLAIN, R.LANE_PROBE, R.LANE_MODE = saved
    return f"probes at decided rounds {[i + 1 for i in probes]}; a capture round left the plain rate as it was"


def test_a_family_that_commits_a_round_at_once_gets_one_call():
    # SPD-239: an engine with `commit_tree_multi` gets each joint round's paths in one call (here the engine's
    # own commit_tree a slot at a time inside it), and no tail commits again: each request writes its alone text
    from engine.model import Qwen38Engine
    calls = []

    def multi(items):
        calls.append(len(items))
        for eng, path in items:
            eng.commit_tree(path)
    Qwen38Engine.commit_tree_multi = staticmethod(multi)
    try:
        out = _check(3)
    finally:
        del Qwen38Engine.commit_tree_multi
    assert calls and all(n > 1 for n in calls), calls
    return f"{out}; {len(calls)} rounds committed in one call each"


# ------------------------------------------------------------------ SPD-249: hand-backs, runner vs slot loop

class _GapTree(T.OracleTree):
    """The oracle tree, with no draft at every third context length: the plain step (with `joint_plain`, the
    anchor alone in the joint pass)."""

    def propose_tree(self, ctx, count):
        if len(ctx) % 3 == 0:
            from engine.tree import DraftTree
            return DraftTree(tokens=[int(ctx[-1])], parents=[-1])
        return super().propose_tree(ctx, count)


_JP_ANCHORS = [0]


def _jp_engine(cfg, w):
    """The CPU engine as a family that joins a plain step to the joint pass (`joint_plain`): a one-row block
    is the plain step (its row written at once, nothing left to commit), every other block the tree verify."""
    import dataclasses
    from engine.model import forward_tree_multi as ftm
    from engine.models.qwen3_5.adapter import Qwen38Engine

    class JointPlain(Qwen38Engine):
        caps = dataclasses.replace(Qwen38Engine.caps, joint_plain=True)
        _anchor = False

        def forward_tree(self, block, parents, start=0):
            if len(parents) == 1:
                _JP_ANCHORS[0] += 1
                self._anchor, self.picks = True, None
                return self.forward(block, start=start, last_only=True)[0]
            self._anchor = False
            return super().forward_tree(block, parents, start=start)

        def commit_tree(self, path):
            if self._anchor:
                self._anchor = False
                return None
            return super().commit_tree(path)

        @staticmethod
        def forward_tree_multi(items):
            out = [None] * len(items)
            rest = [(k, it) for k, it in enumerate(items) if len(it[2]) > 1]
            for k, (eng, tokens, parents, start) in enumerate(items):
                if len(parents) == 1:
                    out[k] = eng.forward_tree(tokens, parents, start=start)
            if rest:
                for (k, it), lg in zip(rest, ftm([it for _, it in rest])):
                    it[0]._anchor = False
                    out[k] = lg
            return out

    eng = JointPlain(cfg, w, max_len=256, device="cpu")
    eng.kv.k = eng.kv.k.to(T.DT)
    eng.kv.v = eng.kv.v.to(T.DT)
    eng.state.conv = eng.state.conv.to(T.DT)
    return eng


def _serve(cfg, w, prompts, fulls, n, mk, *, runner, gap=False, joint_plain=None, consume=None):
    """The requests decoding at once on slot 0 and side slots, through the runner (`runner`) or each slot's
    own loop. `mk(i)` gives request i's generate_stream keywords (eos, think, sup, ...), fresh for each run;
    `consume(i, gen, pool)` reads its stream (all of it by default). Returns per request (tokens, hidden
    spans, keywords), and the runner's stats."""
    import threading
    import torch
    from server import app, slots
    import test_supervisor_frame as F
    saved_on = R.ON
    R.ON, R.RUNNER = runner, None
    engines = [(_jp_engine if joint_plain else T._engine)(cfg, w) for _ in prompts]
    mkd = (lambda f: _GapTree(f)) if gap else (lambda f: T.OracleTree(f))
    pool = T._setup(engines, [mkd(f) for f in fulls])
    app.STATE["tok"] = F.MonTok()
    kws = [mk(i) for i in range(len(prompts))]
    outs, spans, errs = {}, {}, []
    go = threading.Barrier(len(prompts))
    read = consume or (lambda i, gen, pool: list(gen))

    def run(i):
        kw = dict(kws[i])
        eos = kw.pop("eos", set())
        try:
            go.wait()
            if i == 0:
                with app.LOCK:
                    with torch.no_grad():
                        outs[i] = read(i, app.generate_stream(torch.tensor(prompts[i]), n, eos, **kw), pool)
                    spans[i] = [(a, list(b)) for a, b in app.STATE.get("hidden_spans") or []]
                return
            s = pool.slots[i]
            slots.bind(s)
            s.owner = f"req-{i}"
            pool.turn.reset(i)
            pool.turn.acquire(i)
            try:
                with torch.no_grad():
                    outs[i] = read(i, app.generate_stream(torch.tensor(prompts[i]), n, eos, **kw), pool)
                spans[i] = [(a, list(b)) for a, b in app.STATE.get("hidden_spans") or []]
            finally:
                if pool.turn.held_by(i):
                    pool.turn.release(i)
                s.phase, s.owner = "", None
                slots.bind(None)
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    ths = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(prompts))]
    try:
        with F.quiet():                         # the [think] lines; sys.stdout is the process's, never a thread's
            for th in ths:
                th.start()
            for th in ths:
                th.join(120)
        alive = [th.is_alive() for th in ths]
        owner = pool.turn.owner
        stats = dict(R.RUNNER.stats) if R.RUNNER is not None else None
        T._teardown()
    finally:
        R.ON = saved_on
    if errs:
        raise errs[0]
    assert not any(alive), f"a request never ended (deadlock?): {alive}"
    assert owner is None, f"the turn held by {owner!r} after every request ended"
    return [(outs[i], spans.get(i), kws[i]) for i in range(len(prompts))], stats


def _prompts(k, think=True):
    from test_supervisor_frame import OPEN
    base = [[(3 + 7 * i + 11 * j) % 88 + 1 for j in range(10 + 3 * i)] for i in range(k)]
    return [p + [OPEN] for p in base] if think else base


def _pair(mk, k=2, n=40, eos_at=None, gap=False, joint_plain=None, think=True, consume=None):
    """Runner vs slot loop over the same requests: (slot loop results, runner results, runner stats)."""
    cfg = T.tiny_config()
    w = T.FakeWeights(cfg, seed=18)          # a seed whose greedy texts do not fall into a 2-cycle at once
    prompts = _prompts(k, think)
    refs = [T._greedy(cfg, w, p, n) for p in prompts]
    fulls = [p + r for p, r in zip(prompts, refs)]

    def mk2(i):
        kw = mk(i)
        if eos_at is not None and "eos" not in kw:
            # an end token the request writes at `eos_at` (its first time), so SP2 is asked there
            at = next(j for j in range(eos_at, n) if refs[i][j] not in refs[i][:j])
            kw["eos"] = {refs[i][at]}
        return kw
    R.SKIPS.clear()
    loop, _ = _serve(cfg, w, prompts, fulls, n, mk2, runner=False, gap=gap, joint_plain=joint_plain)
    skips0 = dict(R.SKIPS)
    run, stats = _serve(cfg, w, prompts, fulls, n, mk2, runner=True, gap=gap, joint_plain=joint_plain,
                        consume=consume)
    assert "supervisor" not in R.SKIPS and "think_budget" not in R.SKIPS, R.SKIPS
    assert stats is not None and stats["tasks"] >= k and stats["errors"] == 0, stats
    del skips0
    return loop, run, stats


def _same(loop, run, what=""):
    for i, ((a, sa, ka), (b, sb, kb)) in enumerate(zip(loop, run)):
        assert a == b, (what, i, "texts", a, b)
        assert sa == sb, (what, i, "hidden spans", sa, sb)
        sup_a, sup_b = ka.get("sup"), kb.get("sup")
        if sup_a is not None:
            assert (sup_a.acted, sup_a.gave_up, list(sup_a.cuts), sup_a.hidden_tokens) == \
                (sup_b.acted, sup_b.gave_up, list(sup_b.cuts), sup_b.hidden_tokens), (what, i, "supervisor")
        th_a, th_b = ka.get("think"), kb.get("think")
        if th_a is not None:
            assert (th_a.inside, th_a.done, th_a.n, th_a.reason) == (th_b.inside, th_b.done, th_b.n, th_b.reason), \
                (what, i, "think")


def _sup_kw(tasks, budget=0, arms=None, tools=(), max_tokens=64):
    """Request keywords: a fresh supervisor on `tasks` and its ThinkBudget (arms: the supervisor's own, or
    `arms`)."""
    from engine.spec import ThinkBudget
    import test_supervisor_frame as F
    mon = F.monitor("on", tasks)
    sup = F.begin(mon, tools=tools, max_tokens=max_tokens)
    return {"sup": sup, "think": ThinkBudget(F.MonTok(), budget=budget, arms=arms or sup.arms)}


def test_spd249_a_supervisor_that_never_acts_writes_the_slot_loops_text():
    """Every SP1 marker answered None and every SP2 ask passed: the slot loop's texts, nothing skipped, SP1's
    markers waited for (and the slots sat out while they waited)."""
    import test_supervisor_frame as F
    with F.rule("t249_pass1", "SP1", lambda ev, task, sup: None), \
            F.rule("t249_pass2", "SP2", lambda ev, task, sup: None):
        tasks = {"P1": {"rule": "t249_pass1", "mode": "act"}, "P2": {"rule": "t249_pass2", "mode": "act"}}
        loop, run, st = _pair(lambda i: _sup_kw(tasks), eos_at=20)
    _same(loop, run, "pass")
    assert all(len(o) < 40 and o[-1] in kw["eos"] for o, _, kw in run), [len(o) for o, _, _ in run]
    assert st["back"].get("sp2", 0) + st["back"].get("sp2_plain", 0) >= 2, st["back"]
    assert "sp1" not in st["back"] and st["sp1_marks"] > 0 and st["joint"] > 0, st
    return (f"no act: the slot loop's texts; {st['sp1_marks']} SP1 verdicts None, hand-backs {st['back']}, "
            f"{st['joint']} joint passes, {st['sat_out']} rounds sat out")


def test_spd249_sp2_hidden_continue_keeps_texts_and_hidden_spans():
    import test_supervisor_frame as F
    with F.rule("t249_cont", "SP2", F._continue_once):
        tasks = {"T": {"rule": "t249_cont", "mode": "act"}}
        loop, run, st = _pair(lambda i: _sup_kw(tasks), eos_at=12)
    _same(loop, run, "sp2")
    assert all(kw["sup"].acted and sp for _, sp, kw in run), [(sp, kw["sup"].acted) for _, sp, kw in run]
    assert st["back"].get("sp2", 0) + st["back"].get("sp2_plain", 0) >= 2, st["back"]
    return f"SP2 continued once a request: texts and hidden spans equal; hand-backs {st['back']}"


def test_spd249_sp1_closes_the_reasoning_at_a_block():
    import test_supervisor_frame as F
    loop, run, st = _pair(lambda i: _sup_kw(F.backstop(9 + 4 * i)))
    _same(loop, run, "sp1")
    assert all(kw["sup"].acted and sp for _, sp, kw in run), [sp for _, sp, _ in run]
    assert st["back"].get("sp1", 0) == 2, st["back"]
    return f"SP1 closed each reasoning block at its backstop; hand-backs {st['back']}"


def test_spd249_give_up_at_sp1_ends_the_stream():
    import test_supervisor_frame as F
    from server.conductor.monitor.policy import GIVE_UP
    from server.conductor.monitor import Action

    def give_up(ev, task, sup):
        return Action(GIVE_UP, reason="t") if ev.reasoning_tokens >= 7 else None
    with F.rule("t249_give", "SP1", give_up):
        loop, run, st = _pair(lambda i: _sup_kw({"G": {"rule": "t249_give", "mode": "act"}}))
    _same(loop, run, "give_up")
    assert all(kw["sup"].gave_up and len(o) < 40 for o, _, kw in run), [len(o) for o, _, _ in run]
    assert st["back"].get("sp1", 0) == 2, st["back"]
    return f"GIVE_UP at SP1: the stream ended where the slot loop ends it ({[len(o) for o, _, _ in run]})"


def test_spd249_a_d1_arm_stall_closes_through_sp1():
    """D1: the ThinkBudget's stall arms set the reason, the supervisor's reasoning_loop closes at SP1."""
    import test_supervisor_frame as F
    arms = {"loop": {"min_size": 1, "max_size": 8, "min_count": 2, "min_span": 6},
            "novelty": {"window": 8, "gram": 2, "min_tokens": 8, "below": 0.9}}
    tasks = {"D1": {"rule": "reasoning_loop", "mode": "act", "close": F.CLOSE_TEXT, "arms": arms}}
    loop, run, st = _pair(lambda i: _sup_kw(tasks), n=48)
    _same(loop, run, "d1")
    fired = [kw["think"].reason for _, _, kw in run]
    assert all(fired) and all(kw["sup"].acted for _, _, kw in run), fired
    assert st["back"].get("sp1", 0) == 2, st["back"]
    return f"the arms fired ({fired}) and SP1 closed: texts equal"


def test_spd249_a_spent_budget_is_closed_on_the_slots_thread():
    """A client's reasoning budget, with and without a supervisor: the forced close (its phrase streamed) is
    the slot loop's."""
    from engine.spec import ThinkBudget
    import test_supervisor_frame as F
    for with_sup in (False, True):
        def mk(i):
            if with_sup:
                return _sup_kw({}, budget=6 + 5 * i)
            return {"think": ThinkBudget(F.MonTok(), budget=6 + 5 * i)}
        loop, run, st = _pair(mk)
        _same(loop, run, f"budget sup={with_sup}")
        assert all(kw["think"].done for _, _, kw in run), [kw["think"].n for _, _, kw in run]
        assert st["back"].get("budget", 0) == 2, st["back"]
    return f"the budget's forced close on the slot's thread, with and without a supervisor ({st['back']})"


def test_spd249_sp3_regenerate_records_the_same_cuts():
    import test_supervisor_frame as F
    from server.conductor.monitor import Action
    from server.conductor.monitor.policy import REGENERATE_CALL

    def regen(ev, task, sup):
        if sup.memo.get("regen"):
            return None
        sup.memo["regen"] = True
        a = Action(REGENERATE_CALL, reason="t", text="again")
        a.ids = sup.monitor.ids(sup.policy.templates["regen_turn"].replace("{text}", "again"))
        a.visible = []
        return a
    with F.rule("t249_regen", "SP2", regen):
        loop, run, st = _pair(lambda i: _sup_kw({"R": {"rule": "t249_regen", "mode": "act"}}), eos_at=10)
    _same(loop, run, "sp3")
    cuts = [list(kw["sup"].cuts) for _, _, kw in run]
    assert all(cuts), cuts
    return f"REGENERATE_CALL at SP2: the cuts {cuts} and texts equal"


def test_spd249_plain_steps_with_and_without_joint_plain():
    """No draft at every third block: the plain step alone (`_plain`, SP2 there too), or with `joint_plain`
    the anchor alone in the joint pass (the 27B's and LFM2.5's families): the slot loop's texts."""
    import test_supervisor_frame as F
    with F.rule("t249_cont", "SP2", F._continue_once):
        tasks = {"T": {"rule": "t249_cont", "mode": "act"}}
        for jp in (False, True):
            _JP_ANCHORS[0] = 0
            loop, run, st = _pair(lambda i: _sup_kw(tasks, budget=14 + 3 * i), eos_at=8, gap=True,
                                  joint_plain=jp)
            _same(loop, run, f"joint_plain={jp}")
            assert st["plain" if not jp else "joint"] > 0, st
            assert (_JP_ANCHORS[0] > 0) == jp, _JP_ANCHORS
            assert sum(st["back"].values()) >= 2, st["back"]
    return f"plain steps and joint anchors with hand-backs: texts equal ({st['back']})"


def test_spd249_a_slow_consumer_waits_alone():
    """SP1 is asked when the stream has read up to its marker: a slow reader keeps its task out of the
    rounds (no deadlock), and the other lane writes the slot loop's text."""
    import time
    import test_supervisor_frame as F
    with F.rule("t249_pass1", "SP1", lambda ev, task, sup: None):
        tasks = {"P1": {"rule": "t249_pass1", "mode": "act"}}

        def slow(i, gen, pool):
            out = []
            for t in gen:
                out.append(t)
                if i == 1:
                    time.sleep(0.02)
            return out
        loop, run, st = _pair(lambda i: _sup_kw(tasks), consume=slow, n=36)
    _same(loop, run, "slow")
    assert st["sat_out"] > 0, st
    return f"a slow reader: {st['sat_out']} rounds sat out, texts equal"


def test_spd249_a_client_that_leaves_during_a_hand_back_frees_the_turn():
    """The client of slot 1 stops reading inside a hand-back (the tokens before a dropped end token, after a
    hidden continue); its slot is given back and the stream closed later: the turn is free at the end, and
    slot 0 writes the slot loop's text."""
    import test_supervisor_frame as F

    def leave(i, gen, pool):
        out = []
        if i == 0:
            return list(gen)
        for t in gen:
            out.append(t)
            if len(out) >= 3 and _acted[0]:
                break
        if pool.turn.held_by(i):                              # `_par_end`
            pool.turn.release(i)
        s = pool.slots[i]
        s.phase, s.owner = "", None
        from server import slots
        slots.bind(None)
        gen.close()
        return out
    _acted = [False]

    def mk(i):
        kw = _sup_kw({"T": {"rule": "t249_cont", "mode": "act"}})
        if i == 1:
            real = kw["sup"].at_finish

            def at_finish(ev, room):
                a = real(ev, room)
                _acted[0] = _acted[0] or a is not None
                return a
            kw["sup"].at_finish = at_finish
        return kw
    with F.rule("t249_cont", "SP2", F._continue_once):
        loop, run, st = _pair(mk, eos_at=12, consume=leave)
    assert loop[0][0] == run[0][0], (loop[0][0], run[0][0])
    assert _acted[0] and len(run[1][0]) < len(loop[1][0]), (len(run[1][0]), len(loop[1][0]))
    return f"slot 1 left after {len(run[1][0])} tokens inside a hand-back; the turn free, slot 0's text kept"

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
