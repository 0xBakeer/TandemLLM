"""Kolibri speculation on a tiny random Kolibri, CPU only.

The GPU verifier's bit-equality is checked on the GPU (`tools/kolibri_spec_check.py`,
`Verifier.selfcheck` at every load). Here: the semantics every verifier must have
(`ReplayVerifier`), the loop that walks a verified tree (`SpecRows`), StairCut and the prices,
the calibration, and the serving loop with speculation on giving the tokens it gives with it off.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine.kolibri import spec as S  # noqa: E402
from engine.kolibri.verify import ReplayVerifier, tree_tables  # noqa: E402
from engine.tree import DraftTree  # noqa: E402


def _engine(d, max_len=512, ring=64):
    from engine.kolibri.model import KolibriEngine
    from tools import kolibri_tiny as tiny
    st, rel = tiny.write(d)
    eng = KolibriEngine.load(st, rel, device="cpu", max_len=max_len, attention="kernel",
                             log=lambda s: None)
    eng.attn.ring_rows = ring
    eng.kv = eng.attn.make_kv(max_len, "cpu")
    return eng


def _cheap_prices(scale=1.0) -> S.PriceTable:
    """A verify of R rows costs a little more than one step: the cut fires on any decent tree."""
    rows = {str(r): 10.0 + 0.2 * r * scale for r in (1, 2, 4, 8, 16, 32)}
    return S.PriceTable({"decode": {"1024": 10.0}, "chain": {"1024": rows}, "overhead_ms": 0.0})


def test_tree_tables():
    depth, anc = tree_tables([-1, 0, 1, 0, 3])
    assert depth == [0, 1, 2, 1, 2]
    assert anc[2][:3] == [0, 1, 2]
    assert anc[4][:3] == [0, 3, 4]
    assert anc[3][2] == -1


def test_replay_verifier_is_the_decode_step():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        prompt = [int(x) for x in torch.randint(0, 90, (40,))]
        eng.prefill(prompt[:-1])
        p = eng.kv.length
        # plain decode of a chain
        toks = [prompt[-1], 5, 9, 13, 2]
        ref = [eng.decode(t).clone() for t in toks]
        eng.truncate(p)
        v = ReplayVerifier(eng)
        tree = [prompt[-1], 5, 9, 13, 2, 7, 8]
        parents = [-1, 0, 1, 2, 3, 2, 0]           # DFS pre-order: a sibling of node 3, one of node 1
        lg = v.verify(tree, parents)
        assert eng.kv.length == p                  # a verify keeps no row
        for i in range(5):
            assert torch.equal(lg[i], ref[i]), i
        # node 5 (token 7 after 5, 9) is the decode of 7 at depth 2
        eng.truncate(p)
        eng.decode(prompt[-1])
        eng.decode(5)
        eng.decode(9)
        alt = eng.decode(7).clone()
        eng.truncate(p)
        lg = v.verify(tree, parents)
        assert torch.equal(lg[5], alt)
        v.commit([0, 1, 2])
        assert eng.kv.length == p + 3
        assert torch.equal(eng.decode(13), ref[3])


def test_price_table_interpolates():
    pt = S.PriceTable({"decode": {"1024": 14.0, "8192": 15.0},
                       "chain": {"1024": {"1": 14.0, "2": 16.0, "4": 20.0},
                                 "8192": {"1": 15.0, "2": 17.0, "4": 21.0}}})
    assert pt.decode_ms(1024) == 14.0
    assert abs(pt.decode_ms(4608) - 14.5) < 1e-9
    assert pt.verify_ms(3, 1024) == 18.0
    assert pt.verify_ms(6, 1024) == 24.0                    # past the last row: the last slope
    assert pt.verify_ms(2, 100000) == 17.0                  # past the last context: the last class
    assert pt.verify_ms(2, 1024, tree=True) == 16.0         # no tree curve: the chain's


def test_staircut_fires_on_a_likely_chain_and_declines_a_guess():
    ramp = S.PriceTable({"decode": {"1024": 14.0},
                         "chain": {"1024": {"1": 14.0, "2": 16.8, "4": 21.4, "8": 28.8,
                                            "16": 40.3, "32": 56.6}}, "overhead_ms": 0.3})
    cut = S.StairCut(ramp)
    sure = DraftTree.chain(1, list(range(2, 18)), scores=[0.95 ** (i + 1) for i in range(16)])
    best, v, plain = cut.cut(sure, 1024)
    assert best is not None and v > plain
    assert best.n_draft + 1 in S.SIZES
    guess = DraftTree.chain(1, [2, 3, 4], scores=[0.2, 0.04, 0.008])
    best, v, plain = cut.cut(guess, 1024)
    assert best is None and v < plain
    # a long sure chain on the ramp stops before 32 rows: the last rows cost more than they bring
    long = DraftTree.chain(1, list(range(2, 40)), scores=[0.9 ** (i + 1) for i in range(38)])
    best, _, _ = cut.cut(long, 1024)
    assert best is not None and best.n_draft + 1 < 32


def test_rho_learns_from_the_verify():
    r = S.Rho()
    assert abs(r("local", 5) - 0.75) < 1e-9
    assert r("corpus", 7) < 0.5
    for _ in range(50):
        r.update("local", 5, accepted=10, stopped=True)
    assert r("local", 5) > 0.85
    for _ in range(200):
        r.update("corpus", 7, accepted=0, stopped=True)
    assert r("corpus", 7) < 0.1


def _plain(eng, prompt, n):
    eng.reset()
    row = eng.prefill(prompt)
    out = []
    for _ in range(n):
        t = int(row.argmax())
        out.append(t)
        row = eng.decode(t)
    return out


def _spec(eng, prompt, n, rows: S.SpecRows):
    eng.reset()
    row = eng.prefill(prompt)
    ctx = list(prompt)
    rows.start(ctx)
    out = []
    for i in range(n):
        t = int(row.argmax())
        out.append(t)
        ctx.append(t)
        if i + 1 == n:
            break
        row = rows.next(t, ctx)
    rows.settle()
    assert eng.kv.length == len(ctx) - 1
    return out


def test_spec_rows_greedy_equals_plain_and_fires():
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        # a prompt that repeats itself, so the model's own continuation shows up in it
        base = [int(x) for x in torch.randint(0, 90, (24,))]
        plain_first = _plain(eng, base, 40)
        prompt = base + plain_first + base         # the answer is in the prompt: a copy
        ref = _plain(eng, prompt, 40)
        v = ReplayVerifier(eng)
        look = S.LookupSource(corpus="", min_order=2)
        spec = S.KolibriSpec([look], S.StairCut(_cheap_prices(), max_rows=8), max_rows=8)
        rows = S.SpecRows(eng, v, spec)
        got = _spec(eng, prompt, 40, rows)
        assert got == ref
        assert rows.stats["from_verify"] > 0, rows.report()
        assert v.stats["commits"] >= 1
        # off: the same tokens, every row a decode step
        off = S.SpecRows(eng, None, None)
        assert _spec(eng, prompt, 40, off) == ref
        assert off.stats["from_verify"] == 0


def test_serving_loop_with_speculation_gives_the_same_tokens():
    from server.kolibri_serve import RingPrefix, Served, make_generate_stream
    from server import app as real

    class _App:
        def __init__(self):
            self.STATE = {}
            self.BlockStats = real.BlockStats

    torch.manual_seed(2)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        base = [int(x) for x in torch.randint(0, 90, (20,))]
        prompt = base + _plain(eng, base, 30) + base
        eng.reset()
        outs = {}
        for mode in ("off", "on"):
            app = _App()
            app.STATE["engine"] = Served(eng, RingPrefix(chunk=8))
            if mode == "on":
                look = S.LookupSource(corpus="", min_order=2)
                app.STATE["kolibri_verifier"] = ReplayVerifier(eng)
                app.STATE["kolibri_spec"] = S.KolibriSpec(
                    [look], S.StairCut(_cheap_prices(), max_rows=8), max_rows=8)
            eng.reset()
            g = make_generate_stream(app)
            outs[mode] = list(g(torch.tensor(prompt), 32, set()))
            if mode == "on":
                assert app.STATE["kolibri_rows"].stats["from_verify"] > 0
            # rows [0, kv.length) hold the prompt and the answer but its last token
            assert eng.kv.length == len(prompt) + len(outs[mode]) - 1
        assert outs["on"] == outs["off"]


def test_lookup_lines_are_scored_whether_fired_or_not():
    look = S.LookupSource(corpus="", min_order=2)
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ctx = seq + [1, 2, 3]
    look.prime(ctx[:-1])
    assert look.propose(ctx, 6) is not None             # the line 4 5 6 7 8 9
    ctx += [4, 5, 6, 40]                                 # three right, then another token
    look.observe_ctx(ctx)
    assert look.rho.c[("local", 3)] == [3.0, 1.0]


class _FakeBlock:
    """A block drafter's shape: proposes its guess of the next tokens with path probabilities."""

    name = "block"
    cost_ms = 2.0

    def __init__(self, guess):
        self.guess = guess
        self.calls = 0

    def prime(self, ctx):
        pass

    def propose(self, ctx, budget):
        self.calls += 1
        g = self.guess(ctx)[:budget]
        return DraftTree.chain(ctx[-1], g, scores=[0.8 ** (i + 1) for i in range(len(g))],
                               source="block") if g else None

    def feedback(self, tree, path):
        pass


def test_block_source_merges_with_the_lookup_and_stays_lossless():
    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        base = [int(x) for x in torch.randint(0, 90, (20,))]
        ref = _plain(eng, base, 60)
        truth = base + ref

        def oracle(ctx):                         # right for 3 tokens, then wrong
            i = len(ctx)
            return truth[i:i + 3] + [(t + 1) % 90 for t in truth[i + 3:i + 6]]

        blk = _FakeBlock(oracle)
        look = S.LookupSource(corpus="", min_order=2)
        spec = S.KolibriSpec([look, blk], S.StairCut(_cheap_prices(), max_rows=8), max_rows=8)
        rows = S.SpecRows(eng, ReplayVerifier(eng), spec)
        got = _spec(eng, base, 60, rows)
        assert got == ref
        assert blk.calls > 0 and rows.stats["from_verify"] >= 30, rows.report()


def test_taps_follow_the_residual_stream():
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as d:
        eng = _engine(d)
        ids = [int(x) for x in torch.randint(0, 90, (12,))]
        eng.set_taps([0, 2])
        got = {}
        eng.on_taps = lambda j, r, start: got.setdefault(j, []).append((start, r.clone()))
        eng.prefill(ids)
        assert sorted(got) == [0, 1] and got[0][0][1].shape == (12, eng.cfg.hidden)
        eng.truncate(11)
        eng.decode(ids[11])
        # the decode row's taps are the prefill's last row (another arithmetic, so close)
        for j in (0, 1):
            assert torch.allclose(eng.dec_taps[j], got[j][0][1][11], atol=5e-2, rtol=0), j
        eng.set_taps([])
        assert eng.dec_taps is None


def test_price_file_round_trip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "stair.json")
        json.dump({"decode": {"1024": 14.0}, "chain": {"1024": {"2": 17.0, "4": 21.0}},
                   "overhead_ms": 0.2, "measured": "test"}, open(p, "w"))
        pt = S.PriceTable.load(p)
        assert pt.overhead_ms == 0.2 and pt.verify_ms(3, 1024) == 19.0
