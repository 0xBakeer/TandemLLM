"""The length router's policy, driven by simulated acceptance instead of a board.

What matters is not that it picks a length but that the picks follow the measured curve and the
evidence the loop can actually give it. Each test fixes an acceptance behaviour -- the acceptance a
workload would produce -- and checks the choice that behaviour implies, including the two properties
the design turns on: coming DOWN from the wide block needs no experiment, and going UP needs one.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.lenrouter import LengthRouter, verify_ms_b  # noqa: E402


class FakeCfg:
    def __init__(self, block):
        self.block_size = block


class FakeEng:
    def __init__(self):
        self.tap = None


class FakeDrafter:
    """A block drafter of a fixed width that always proposes, and records what it was asked."""

    wants_rows = True
    tree_temp = 1.0

    def __init__(self, eng, block, base=9000):
        self.eng = eng
        self.cfg = FakeCfg(block)
        self.base = base
        self.calls = 0
        self.taps = 0
        self.synced = []
        self._lattice = None

    def propose(self, context, k):
        self.calls += 1
        return [self.base + i for i in range(min(k, self.cfg.block_size - 1))]

    def _on_tap(self, h):
        self.taps += 1

    def reset(self):
        pass

    def prime(self, tokens):
        pass

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.synced.append((len(tokens), first_pos))


def build(**kw):
    """A router over two instant drafters.

    `learn_cost=False` matters: these fakes return in microseconds, so a router that timed them
    would price the draft at zero and the wide block would look free. The costs under test are the
    measured ones.
    """
    eng = FakeEng()
    small = FakeDrafter(eng, 8, base=9000)
    large = FakeDrafter(eng, 16, base=9000)
    kw.setdefault("learn_cost", False)
    return LengthRouter(small, large, **kw), small, large


def run(router, blocks, runs):
    """Drive `blocks` steps against a fixed acceptance behaviour.

    `runs` maps a drafter key -- "s" or "l" -- to a cycle of RUN LENGTHS: how many of that
    drafter's proposals the target would agree with before the block width truncates them. Stating
    the fixture this way rather than as an accepted count per width is what makes the truncation
    consistent: a narrow block is a prefix of a wide one, so the same run length has to produce
    `min(run, 7)` at eight and `min(run, 15)` at sixteen. A fixture that let those two be chosen
    independently would let a test assert something the engine cannot do.
    """
    idx = {"s": 0, "l": 0}
    widths = []
    for _ in range(blocks):
        draft = router.propose(list(range(50)), 15)
        key, width = router.last_key, router.last_width
        widths.append(width)
        seq = runs[key]
        r = seq[idx[key] % len(seq)]
        idx[key] += 1
        n = min(r, max(width - 1, 0), len(draft))
        router.observe(draft[:n] + [12345])
    return widths


# --- the curve ---------------------------------------------------------------------------------

def test_verify_curve_is_read_at_the_measured_points():
    assert verify_ms_b(8) == 116.22
    assert verify_ms_b(16) == 132.38
    # below the first point and above the last, flat and linear respectively, never a negative cost
    assert verify_ms_b(1) == 95.56
    assert verify_ms_b(32) > verify_ms_b(16)


def test_expected_prefix_is_the_running_product():
    p = [0.9, 0.8, 0.5]
    got = LengthRouter._expected_prefix(p, 3)
    assert abs(got - (0.9 + 0.72 + 0.36)) < 1e-9
    # a prefix expectation never exceeds the number of slots it is taken over
    assert LengthRouter._expected_prefix([1.0] * 15, 7) == 7.0


def test_break_even_matches_the_measured_ratio():
    """The wide block has to commit about 15.6 % more, and the router's own pricing says so."""
    r, _, _ = build()
    narrow = r._cost_ms("s", 8, 7.0)
    wide = r._cost_ms("l", 16, 15.0)
    assert 1.13 < wide / narrow < 1.19


# --- the tap ------------------------------------------------------------------------------------

def test_one_tap_feeds_both_drafters():
    """One bound callback, held: `self._on_tap is self._on_tap` is False in CPython, so a detach
    that compares them with `is` silently leaves the tap installed."""
    r, small, large = build()
    assert r.eng.tap is r._tap_cb
    for _ in range(5):
        r.eng.tap(object())
    assert small.taps == 5 and large.taps == 5


def test_both_caches_are_kept_current():
    """The drafter that did not propose still has to be synced or its cache falls behind."""
    r, small, large = build()
    r.sync([1, 2, 3], None, 10)
    assert small.synced == [(3, 10)] and large.synced == [(3, 10)]


# --- the regimes ---------------------------------------------------------------------------------

def test_reproduction_text_converges_on_the_wide_block():
    """`quote`: the narrow block accepts every slot it has, so its own number is censored."""
    r, _, _ = build(explore_period=32)
    widths = run(r, 40, {"s": [15], "l": [15]})
    tail = widths[-20:]
    assert sum(1 for w in tail if w == 16) >= 18, widths


def test_fresh_prose_stays_narrow_and_explores_rarely():
    """`prose`: the wide block buys slots the drafter cannot fill, so the narrow one wins."""
    r, _, _ = build(explore_period=32)
    widths = run(r, 96, {"s": [2], "l": [2]})
    wide = sum(1 for w in widths if w == 16)
    # the first block, which always goes wide, plus the forced probes the schedule bounds
    assert wide <= 96 // 32 + 3, widths
    assert widths[-1] == 8


def test_the_first_block_goes_wide():
    """A wide block prices both options and a narrow one prices only itself, so the first block --
    the one with no evidence behind it at all -- is taken at the width that learns the most."""
    r, small, large = build()
    run(r, 1, {"s": [3], "l": [3]})
    assert large.calls == 1 and small.calls == 0
    t, _, _, _ = build_tree()
    run_tree(t, 1, {"s": [3], "l": [3]})
    assert t.stats["large"] == 1 and t.stats["small"] == 0


def test_code_like_gain_below_break_even_stays_narrow():
    """Measured `code`: 5.45 committed a block at eight against 6.10 at sixteen -- a 12 % gain
    against a 15.6 % price, so the wide block loses and the router has to notice that it loses.

    The two means are not integers, so the fixture cycles the accepted counts to reach them: 4.45
    accepted at eight, 5.10 at sixteen.
    """
    r, _, _ = build(explore_period=16)
    # E[min(run, 7)] = 4.4 for the narrow drafter; the wide one has the longer tail -- 4.9 over
    # fifteen slots -- but only 4.1 over seven, which is the shape that makes a wide block worth
    # 12 % more and not the 15.6 % it costs.
    widths = run(r, 80, {"s": [4, 5, 4, 5, 4, 4, 5, 4, 5, 4],
                         "l": [2, 3, 4, 5, 6, 15, 2, 3, 4, 5]})
    assert sum(1 for w in widths[-30:] if w == 8) >= 24, widths


# --- the asymmetry ---------------------------------------------------------------------------------

def test_the_narrow_option_is_priced_from_wide_blocks_for_free():
    """A wide block that accepted 3 says exactly what a narrow one would have committed: 4."""
    r, _, _ = build()
    r.fixed = 16
    run(r, 6, {"s": [3], "l": [3]})
    assert abs(r.acc[("l", 8)].value - 4.0) < 1e-9


def test_coming_back_down_needs_no_exploration():
    """Pinned wide, then released: the router leaves the wide block without ever having to run a
    narrow one to find out, because the truncation of its own wide blocks already priced it."""
    r, small, _ = build(explore_period=1000)
    r.fixed = 16
    run(r, 12, {"s": [2], "l": [2]})                 # wide blocks, badly accepted
    calls_before = small.calls
    r.fixed = 0
    widths = run(r, 12, {"s": [2], "l": [2]})
    assert small.calls > calls_before       # it did come back down
    assert widths[-1] == 8, widths


def test_going_up_is_forced_because_the_narrow_number_is_censored():
    """With the ceiling signal suppressed the router still probes, on the slow schedule: a policy
    that only learns about the option it took keeps taking it."""
    r, _, large = build(explore_period=8, ceiling_trigger=2.0)
    widths = run(r, 40, {"s": [2], "l": [2]})
    assert 16 in widths
    assert large.calls >= 4


def test_a_saturating_narrow_block_shortens_the_probe_schedule():
    """Once the router is on the narrow block, the ceiling rate is what buys the next wide probe.

    Both routers are given a bad wide history first, so the expected-value rule keeps them narrow,
    and then a narrow stretch that accepts every slot it has. That is the censored observation: the
    truth is "at least seven" and what sixteen would have committed is not in the data. The router
    that reads the ceiling rate buys the answer within four blocks; the one with the trigger
    switched off waits for its slow schedule and never finds out.
    """
    slow, _, _ = build(explore_period=64, ceiling_trigger=2.0)   # never triggered
    fast, _, _ = build(explore_period=64, ceiling_period=4, ceiling_trigger=0.25)
    for r in (slow, fast):
        r.fixed = 16
        run(r, 4, {"s": [2], "l": [2]})          # the wide arm looks bad
        r.fixed = 8
        run(r, 8, {"s": [15], "l": [15]})        # and the narrow one saturates
        r.fixed = 0
    w_slow = run(slow, 16, {"s": [15], "l": [15]})
    w_fast = run(fast, 16, {"s": [15], "l": [15]})
    assert sum(1 for w in w_fast if w == 16) > sum(1 for w in w_slow if w == 16), (w_slow, w_fast)
    assert slow.ceiling.value == 1.0 and fast.ceiling.value > 0.0


# --- the tree arms -------------------------------------------------------------------------------

class FakeTree:
    def __init__(self, anchor, toks):
        self.tokens = [anchor] + list(toks)
        self.parents = [-1] + list(range(len(toks)))

    @property
    def n_draft(self):
        return len(self.tokens) - 1


class FakeNgram:
    """The lookup drafter both arms share. It counts how often it is told about a block."""

    def __init__(self):
        self.primed = 0
        self.observed = 0

    def prime(self, tokens):
        self.primed += 1

    def observe(self, tokens):
        self.observed += 1

    def reset(self):
        pass


class FakeArm:
    """A MergedRouter stand-in: it wraps a block drafter as `.mtp` and extends the shared index."""

    wants_rows = True

    def __init__(self, head, ngram, budget):
        self.mtp = head
        self.ngram = ngram
        self.budget = budget
        self.calls = 0

    def propose_tree(self, context, k):
        self.calls += 1
        n = min(k, self.budget)
        return FakeTree(context[-1], [9000 + i for i in range(n)]) if n else None

    def observe(self, tokens):
        self.ngram.observe(tokens)

    def reset(self):
        pass

    def prime(self, tokens):
        self.ngram.prime(tokens)

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.mtp.sync(tokens, hidden, first_pos, rows=rows)


def build_tree(**kw):
    eng = FakeEng()
    small = FakeDrafter(eng, 8)
    large = FakeDrafter(eng, 16)
    ng = FakeNgram()
    kw.setdefault("learn_cost", False)
    r = LengthRouter(FakeArm(small, ng, 7), FakeArm(large, ng, 15),
                     tree=True, ngram=ng, **kw)
    return r, small, large, ng


def run_tree(router, blocks, runs):
    idx = {"s": 0, "l": 0}
    widths = []
    for _ in range(blocks):
        tree = router.propose_tree(list(range(50)), 15)
        key, width = router.last_key, router.last_width
        widths.append(width)
        seq = runs[key]
        n = min(seq[idx[key] % len(seq)], max(width - 1, 0))
        idx[key] += 1
        router.observe([9000 + i for i in range(n)] + [12345])
    return widths


def test_the_tree_arms_share_one_lookup_index():
    """Two arms, one suffix memory, one update a block. Indexing every token twice would make the
    store disagree with the text it is a memory of."""
    r, _, _, ng = build_tree()
    r.prime(list(range(20)))
    assert ng.primed == 1
    run_tree(r, 10, {"s": [3], "l": [3]})
    assert ng.observed == 10


def test_the_tree_path_prices_on_the_tree_curve():
    """A tree pays its commit on every block and a chain pays a rollback only on a rejection."""
    chain, _, _ = build()
    tree, _, _, _ = build_tree()
    assert tree._cost_ms("l", 16, 2.0) > tree.vms[16].value + tree.dms["l"].value
    # the commit is flat in the accepted length; the chain's rollback is not
    assert abs(tree._cost_ms("l", 16, 2.0) - tree._cost_ms("l", 16, 15.0)) < 1e-9
    assert chain._cost_ms("l", 16, 2.0) > chain._cost_ms("l", 16, 15.0)


def test_the_tree_router_still_finds_the_wide_budget():
    r, _, _, _ = build_tree(explore_period=32)
    widths = run_tree(r, 40, {"s": [15], "l": [15]})
    assert sum(1 for w in widths[-20:] if w == 16) >= 18, widths


# --- the pins ------------------------------------------------------------------------------------

def test_fixed_pins_the_width_through_the_same_code():
    r, small, large = build(fixed=8)
    assert set(run(r, 10, {"s": [3], "l": [3]})) == {8}
    assert large.calls == 0
    r2, small2, large2 = build(fixed=16)
    assert set(run(r2, 10, {"s": [3], "l": [3]})) == {16}
    assert small2.calls == 0


def test_a_short_budget_never_asks_for_a_wide_block():
    """Near the end of a generation the loop asks for fewer tokens than a wide block proposes.

    The short block's evidence still has to land somewhere: it belongs to the narrow arm, not to a
    width of five that the router could never choose on purpose.
    """
    r, _, large = build()
    draft = r.propose(list(range(50)), 4)
    assert len(draft) <= 4 and r.last_width <= 5 and large.calls == 0
    r.observe(draft[:2] + [1])
    assert r.acc[("s", 8)].n == 1 and (("s", 5) not in r.acc)


def test_the_loop_teaches_the_router_what_a_block_cost():
    r, _, _ = build(learn_cost=True)
    before = r.vms[16].value
    for _ in range(8):
        r.on_verify(16, 200.0)
    assert r.vms[16].value > before


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
