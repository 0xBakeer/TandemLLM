"""SPD-249: Flash-Next's coupled (keyed) MTP drafts (engine/models/flashnext/coupled.py).

CPU:
  * a keyed round's noise keys are the walk's own: `_key(seed, start + d)` for draft depth d + 1, start = len(context);
    greedy (temperature 0), unseeded and knob-off requests stage nothing (the argmax draft);
  * the shared drafter hands the head each round's start, clears the request's sampler on reset, and leaves a head
    that cannot key alone;
  * the split head (FN_MTP_COUPLED_HOT_LONE): a keyed chain of one lane reads the lone head, of more lanes the coupled
    head; a round's calibration size is the head its chain read (lone graph, packed pass of one head or of several);
  * a keyed round's cut inputs are swapped in for that cut alone; the depth-1 sibling is offered only with a
    probability above 0; `DepthDeciles` reads its table by depth and falls back to the probability itself;
  * a calibration file that does not read (not JSON, not a map, a row that is not 10 bins, a rate above 1, another
    statistic, a price that is not a number): one log line and the cut's own inputs; the shipped file reads.
GPU (skipped without CUDA and Triton):
  * the kernel's pick equals the torch reference (`pick_ref`: the noise from `torch.rand(V)` after
    `manual_seed(key)`, the filter as `Sampler._filter`) token for token, both statistics, over random rows and
    parameters (tau, top-k, top-p, sibling margin);
  * a flat row picks the argmax of the target's noise at its candidates (the noise at the hot ids is the target's);
  * tau -> 0 is the argmax draft; top-k 1 keeps one candidate (no sibling); a row with `on` 0 keeps its pair bit
    for bit;
  * one matrix: the lone head and the served draft head are views of the coupled head's first rows, the same bytes
    as their own matrices, the gemv over them bit for bit theirs, and the served head's own matrix freed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead  # noqa: E402
from engine.methods.staircut import DepthDeciles, PriceModel, ProbDeciles, StairCut  # noqa: E402
from engine.models.flashnext import coupled as CP  # noqa: E402
from engine.sample import Sampler, _key  # noqa: E402


def _skip(why: str) -> Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scriptmode import Skip
    return Skip(why)


class _On:
    """The module's knobs for a test (the registry reads them at import)."""

    def __init__(self, **kw):
        self.kw = {"ON": True, **kw}

    def __enter__(self):
        self.old = {k: getattr(CP, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(CP, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(CP, k, v)


def test_keys_are_the_walks_keys():
    sm = Sampler(temperature=1.0, top_k=20, top_p=0.95, seed=1000)
    with _On(TAU=0.85, FILTER=True, SIB=False):
        keys, par = CP.keyed_for(sm, 717, 4)
    assert keys == [_key(1000, 717 + d) for d in range(4)]
    assert par[:5] == [0.85, 20.0, 0.95, -1.0, 1.0] and len(par) == CP.NPAR
    with _On(TAU=0.85, FILTER=False, SIB=True, SIB_MARGIN=1.3):
        _, par = CP.keyed_for(Sampler(temperature=0.5, top_k=20, top_p=0.95, seed=7), 10, 4)
    assert par[:5] == [0.425, 0.0, 1.0, 1.3, 1.0]
    return "keys _key(seed, start + d); tau relative to the request's temperature"


def test_greedy_and_unseeded_requests_draft_argmax():
    with _On():
        assert CP.keyed_for(Sampler(temperature=0.0, seed=5), 10, 4) is None
        assert CP.keyed_for(Sampler(temperature=1.0, top_k=20, seed=None), 10, 4) is None
        assert CP.keyed_for(None, 10, 4) is None
    with _On(ON=False):
        assert CP.keyed_for(Sampler(temperature=1.0, top_k=20, seed=5), 10, 4) is None
    return "greedy, unseeded and knob-off: no keys"


class _KeyHead:
    """A head that drafts tokens 10 + k (p 0.6, second 20 + k p 0.3) and records what each launch was keyed with."""

    depth = 3

    def __init__(self, inputs=None):
        self.st = types.SimpleNamespace(length=0)
        self.keyed = None
        self.launched = []
        self.inputs = inputs

    def reset(self):
        self.st.length = 0

    def prime(self, rows, nexts):
        self.st.length += len(nexts)

    def launch(self, rows, nexts):
        self.launched.append(self.keyed)
        self.st.length += len(nexts)

    def read(self):
        return [[10 + k, 20 + k] for k in range(self.depth)], [[0.6, 0.3]] * self.depth

    def keyed_for(self, sampler, start):
        return ("keys", sampler.seed, start) if sampler is not None and sampler.coupled else None

    def cut_inputs(self, keyed, lanes, cls=0):
        return self.inputs if keyed else None


def _round(d, ctx):
    d.sync(ctx[-3:-1], torch.zeros(2, 3), len(ctx) - 3)
    gen = d.propose_tree_steps(ctx, 8)
    next(gen)
    try:
        next(gen)
    except StopIteration as e:
        return e.value


def test_the_drafter_hands_the_head_the_rounds_start():
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    h = _KeyHead()
    d = MTPTreeDrafter(eng, h, StairCut(PriceModel()), freeze=True)
    d._turn = True
    d.set_keyed(Sampler(temperature=1.0, top_k=20, top_p=0.95, seed=1000))
    ctx = list(range(100, 140))
    _round(d, ctx)
    assert h.launched[-1] == ("keys", 1000, 40), h.launched
    assert d.keyed_rounds == 1
    d.set_keyed(None)                                   # a greedy request
    _round(d, ctx + [7, 8])
    assert h.launched[-1] is None
    d.set_keyed(Sampler(temperature=1.0, top_k=20, seed=3))
    d.reset()                                           # the next request starts unkeyed
    assert d.keyed_sm is None
    plain = MTPTreeDrafter(eng, NoHead(), StairCut(PriceModel()))
    plain.set_keyed(Sampler(temperature=1.0, top_k=20, seed=3))
    assert plain.keyed_sm is None
    return "start = len(context); None for greedy; reset clears; a head without keyed_for stays argmax"


def test_a_keyed_rounds_cut_inputs_are_its_own():
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    tab = {"spine": [[0.05] * 10] * 4, "sib": [[0.9] * 10] * 4}
    cal = DepthDeciles(tab)
    prices = PriceModel({0: {"chain": {1: 20.0, 2: 21.0, 3: 22.0, 4: 23.0, 5: 24.0}}}, None)
    h = _KeyHead({"cal": cal, "prices": prices, "draft_ms": 4.3, "sib1": True})
    base = StairCut(PriceModel({0: {"chain": {1: 20.0, 2: 20.5, 3: 21.0, 4: 21.5, 5: 22.0}}}, None), siblings=False)
    d = MTPTreeDrafter(eng, h, base, freeze=True)
    d._turn = True
    before = (base.cal, base.prices, dict(base.draft), base.sib1)
    tree = _round(d, list(range(100, 140)))             # greedy round: the cut's own inputs, no sibling
    assert len(tree.tokens) == 4 and tree.parents == [-1, 0, 1, 2], (tree.tokens, tree.parents)
    d.set_keyed(Sampler(temperature=1.0, top_k=20, seed=1))
    tree = _round(d, list(range(100, 150)))
    # the measured calibration (spine 0.05 everywhere) keeps one spine row and the depth-1 sibling (0.95 x 0.9)
    assert tree.tokens[1:] == [20, 10] and tree.parents == [-1, 0, 0], (tree.tokens, tree.parents)
    assert (base.cal, base.prices, dict(base.draft), base.sib1) == before
    return "the calibration inputs and the sibling for the keyed cut alone; the cut's own inputs restored"


def test_the_depth_one_sibling_needs_a_probability():
    c = StairCut(PriceModel({0: {"chain": {1: 20.0, 2: 20.2, 3: 20.4, 4: 20.6}}}, None), siblings=False)
    c.frozen = True
    c.sib1 = True
    toks, parents = c.cut(1, [[10, 20], [11, 21]], [[0.4, 0.5], [0.9, 0.1]], None, 0)
    assert 20 in toks and parents[toks.index(20)] == 0
    toks, parents = c.cut(1, [[10, 20], [11, 21]], [[0.4, 0.0], [0.9, 0.1]], None, 0)
    assert 20 not in toks and 21 not in toks
    c.sib1 = False
    toks, _ = c.cut(1, [[10, 20], [11, 21]], [[0.4, 0.5], [0.9, 0.1]], None, 0)
    assert 20 not in toks
    return "sibling at depth 1 only with sib1 and p > 0; never deeper"


def test_depth_deciles():
    cal = DepthDeciles({"spine": [[0.1 * b for b in range(10)], [None] * 9 + [0.5]]})
    assert abs(cal.q("spine", 0.35, 0) - 0.3) < 1e-9
    assert abs(cal.q("spine", 0.95, 1) - 0.5) < 1e-9
    assert abs(cal.q("spine", 0.95, 3) - 0.5) < 1e-9          # past the table: its last depth
    base = ProbDeciles()
    assert cal.q("spine", 0.35, 1) == base.q("spine", 0.35)   # no measurement: the probability itself
    assert cal.q("sib", 0.42, 0) == base.q("sib", 0.42)
    assert cal.q("spine", 0.35) == base.q("spine", 0.35)      # no depth: the base
    return "by depth; None and missing kinds fall back"


def test_the_calibration_file_gives_one_lane_class_0_its_prices():
    doc = {"stat": "margin", "cal": {"65536": {"spine": [[0.2] * 10, [0.3] * 10], "sib": [[0.4] * 10]}},
           "lone": {"1": 23.1, "2": 27.9, "3": 29.6}, "draft_ms": {"32768": 4.3, "65536": 5.7}}
    ci = CP.CalInputs(doc, "x.json")
    assert ci.stat == CP.STAT_MARGIN
    cal, prices, draft = ci.inputs(65536, 1, 0)
    assert abs(cal.q("spine", 0.5, 1) - 0.3) < 1e-9 and abs(cal.q("sib", 0.5, 0) - 0.4) < 1e-9
    assert prices is not None and abs(prices.price(2) - 27.9) < 1e-9 and draft == 5.7
    for lanes, cls in ((8, 0), (1, 32768)):
        cal2, p2, d2 = ci.inputs(65536, lanes, cls)
        assert p2 is None and d2 is None and abs(cal2.q("spine", 0.5, 0) - 0.2) < 1e-9
    cal3, _, d3 = ci.inputs(32768, 1, 0)                      # no table for 32k: the probability itself
    assert cal3.q("spine", 0.55, 0) == ProbDeciles().q("spine", 0.55) and d3 == 4.3
    return "calibration by hot size; prices and draft cost at one lane, class 0 only"


def test_a_calibration_file_that_does_not_read_keeps_the_cuts_own_inputs():
    import contextlib
    import io
    import tempfile
    d = tempfile.mkdtemp(prefix="coupled-cal-")
    good = {"stat": "q", "cal": {"65536": {"spine": [[0.2] * 10] * 4, "sib": [[0.4] * 10]}},
            "lone": {"1": 23.1}, "draft_ms": {"65536": 5.7}}
    bad = {"not json": "{\"stat\": \"q\", \"cal\": {", "a list": "[]", "a short row":
           json.dumps(dict(good, cal={"65536": {"spine": [[0.2] * 9]}})), "a rate above 1":
           json.dumps(dict(good, cal={"65536": {"spine": [[1.5] * 10]}})), "another statistic":
           json.dumps(dict(good, stat="c")), "a price that is not a number": json.dumps(dict(good, lone={"1": "x"}))}
    files = {k: os.path.join(d, f"{i}.json") for i, k in enumerate(list(bad) + ["good"])}
    for k, text in dict(bad, good=json.dumps(good)).items():
        open(files[k], "w").write(text)
    old = (CP.CAL, CP.CalInputs.path, dict(CP.CalInputs._cache))
    got = {}
    try:
        CP.CAL = True
        for k, f in files.items():
            CP.CalInputs.path = classmethod(lambda cls, pack=None, f=f: f)
            CP.CalInputs._cache.clear()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ci = CP.CalInputs.get()
                assert CP.CalInputs.get() is ci
            got[k] = (ci, [ln for ln in out.getvalue().splitlines() if ln.startswith("[coupled]")])
    finally:
        CP.CAL, CP.CalInputs.path = old[0], old[1]
        CP.CalInputs._cache.clear()
        CP.CalInputs._cache.update(old[2])
    for k in bad:
        ci, lines = got[k]
        assert ci is None and len(lines) == 1 and "unusable" in lines[0], (k, lines)
    ci, lines = got["good"]
    assert ci is not None and lines == [] and abs(ci.inputs(65536, 1, 0)[0].q("spine", 0.5, 2) - 0.2) < 1e-9
    shipped = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops", "prices",
                           "qwen3.8-flash-next", "flashnext-3.05.dgx-spark.coupled.json")
    CP._check_doc(json.load(open(shipped)))
    return f"{len(bad)} files that do not read: one line each, the cut's own inputs; a good one and the shipped one read"


# ---------------------------------------------------------------- the split head (FN_MTP_COUPLED_HOT_LONE)

def _heads(draft=32, coupled=64, lone=32):
    """A stand-in engine's heads: (head, ids) tags of the given sizes (lone None: not set)."""
    def h(n):
        return (f"h{n}", torch.arange(n)) if n else (None, None)
    ns = types.SimpleNamespace()
    ns.draft_head, ns.draft_ids = h(draft)
    ns.coupled_head, ns.coupled_ids = h(coupled)
    ns.lone_head, ns.lone_ids = h(lone)
    if lone and lone == draft:
        ns.lone_head, ns.lone_ids = ns.draft_head, ns.draft_ids
    return ns


def test_the_keyed_head_follows_the_lane_count():
    from engine.models.flashnext.engine import FlashNextEngine as E
    ns = _heads(32, 64, 32)
    assert E.keyed_head(ns, 1)[0] == "h32" and E.keyed_head(ns, 2)[0] == "h64" and E.keyed_head(ns, 8)[0] == "h64"
    ns = _heads(32, 64, 0)                                    # no lone head: the coupled head for every lane count
    assert E.keyed_head(ns, 1)[0] == "h64" and E.keyed_head(ns, 8)[0] == "h64"
    ns = _heads(32, 0, 0)                                     # no coupled head: the draft head
    assert E.keyed_head(ns, 1)[0] == "h32" and E.keyed_head(ns, 8)[0] == "h32"
    ns = _heads(0, 0, 0)                                      # the full head
    assert E.keyed_head(ns, 1) == (None, None)
    return "one lane: lone; more: coupled; unset: coupled, then the draft head, then the full head"


def test_a_rounds_calibration_size_is_the_head_its_chain_read():
    from engine.models.flashnext.mtp import FlashNextMTPHead as H
    from engine.models.flashnext.engine import FlashNextEngine as E
    eng = _heads(32768, 65536, 32768)
    eng.keyed_head = types.MethodType(E.keyed_head, eng)
    calls = []
    eng.mtp_chain_multi = lambda items, keyed=None: calls.append((len(items), keyed))

    def head(k):
        h = types.SimpleNamespace(fn=eng, key=k, st=None, _n=1, depth=4, _cls=0, _keyed=True, _khot=0)
        h._hot_of = types.MethodType(H._hot_of, h)
        return h
    many = [head(k) for k in range(3)]
    H.run_packed(many)
    assert [h._khot for h in many] == [65536] * 3 and calls[-1] == (3, [True] * 3)
    one = [head(9)]
    H.run_packed(one)                                         # a packed pass of one head is a one-lane chain
    assert one[0]._khot == 32768
    plain = [head(5)]
    plain[0]._keyed = False
    H.run_packed(plain)                                       # no keyed lane: nothing to calibrate
    assert plain[0]._khot == 0 and calls[-1] == (1, None)
    lone = head(7)
    assert H.keyed_hot.fget(lone) == 0 and H._hot_of(lone, 1) == 32768
    return "packed of 3: 65536; packed of 1 and the lone graph: 32768; the calibration table follows"


# ---------------------------------------------------------------- GPU

def _gpu():
    if not torch.cuda.is_available() or not CP.HAVE_TRITON:
        raise _skip("needs CUDA and Triton")
    return torch.device("cuda")


def _rows(dev, R, Vh, V, seed):
    g = torch.Generator().manual_seed(seed)
    lg = (torch.randn(R, Vh, generator=g) * 3.0).to(dev)
    hids = torch.randperm(V, generator=g)[:Vh].to(dev)
    keys = [int(torch.randint(0, 2 ** 62, (1,), generator=g)) for _ in range(R)]
    return lg, hids, keys


def _par(rows: list, dev):
    return torch.tensor([list(r) + [0.0] * (CP.NPAR - len(r)) for r in rows], dtype=torch.float32, device=dev)


def _run(lg, hids, keys, par, V, statistic):
    dev = lg.device
    R = lg.shape[0]
    z = torch.exp(lg - lg.amax(-1, keepdim=True)).sum(-1, keepdim=True)
    tok = torch.full((R, 2), -7, dtype=torch.long, device=dev)
    prob = torch.full((R, 2), -7.0, device=dev)
    kt = torch.tensor(keys, dtype=torch.int64, device=dev)
    old = CP.stat
    CP.stat = lambda: statistic
    try:
        CP.pick(lg, hids, kt, par, z, tok, prob, V)
    finally:
        CP.stat = old
    torch.cuda.synchronize()
    return tok.cpu(), prob.cpu()


def test_the_kernel_is_the_reference():
    dev = _gpu()
    V, Vh = 248320, 32768
    shapes = [(0.85, 20, 0.95, 1.3, 1), (1.0, 0, 1.0, -1.0, 1), (0.7, 5, 0.5, 9.0, 1), (0.85, 48, 0.99, 0.5, 1),
              (1.0, 20, 0.3, 2.0, 1), (0.6, 64, 1.0, 1.0, 1), (0.85, 20, 0.95, 1.3, 1), (2.0, 3, 0.95, 5.0, 1)]
    n = 0
    for seed in range(12):
        lg, hids, keys = _rows(dev, len(shapes), Vh, V, seed)
        par = _par(shapes, dev)
        for statistic in (CP.STAT_Q, CP.STAT_MARGIN):
            tok, prob = _run(lg, hids, keys, par, V, statistic)
            rt, rp, _ = CP.pick_ref(lg, hids, keys, par.cpu(), V, statistic)
            assert torch.equal(tok[:, 0], rt[:, 0]), (seed, statistic, tok[:, 0], rt[:, 0])
            has2 = rp[:, 1] > 0
            assert torch.equal(tok[has2, 1], rt[has2, 1]), (seed, statistic)
            assert torch.allclose(prob, rp, atol=2e-5, rtol=2e-4), (seed, statistic, prob, rp)
            n += len(shapes)
    return f"{n} rows equal (tokens, statistics), both statistics"


def test_a_flat_row_picks_the_targets_noise():
    dev = _gpu()
    V, Vh = 248320, 65536
    for seed in range(8):
        lg, hids, keys = _rows(dev, 4, Vh, V, 100 + seed)
        lg = lg * 1e-6                                   # q flat over the candidates: the noise decides
        par = _par([(1.0, 0, 1.0, -1.0, 1)] * 4, dev)
        tok, _ = _run(lg, hids, keys, par, V, CP.STAT_Q)
        v, i = torch.topk(lg, CP.KP, dim=-1)
        for r in range(4):
            gen = torch.Generator(device=dev)
            gen.manual_seed(keys[r])
            u = torch.rand(V, generator=gen, device=dev)
            cand = hids[i[r]]
            want = int(cand[torch.argmax(-torch.log(-torch.log(u[cand].clamp_min(1e-20))))])
            assert int(tok[r, 0]) == want, (seed, r)
    return "32 flat rows: the pick is the argmax of torch.rand(V) at the candidates (the target's noise)"


def test_edges_tau_topk_and_off_rows():
    dev = _gpu()
    V, Vh = 248320, 32768
    lg, hids, keys = _rows(dev, 6, Vh, V, 7)
    lg[:, 0] += 50.0                                     # a clear argmax
    par = _par([(1e-4, 20, 0.95, 1.3, 1), (1e-4, 0, 1.0, -1.0, 1), (1.0, 1, 1.0, 9.0, 1), (0.85, 20, 0.95, 9.0, 1),
                (0.85, 20, 0.95, 1.3, 0), (0.85, 20, 0.95, 1.3, 0)], dev)
    for statistic in (CP.STAT_Q, CP.STAT_MARGIN):
        tok, prob = _run(lg, hids, keys, par, V, statistic)
        am = hids[lg.argmax(-1)].cpu()
        assert int(tok[0, 0]) == int(am[0]) and int(tok[1, 0]) == int(am[1])     # tau -> 0: the argmax draft
        assert int(tok[2, 0]) == int(am[2]) and float(prob[2, 1]) == 0.0          # top-k 1: one candidate, no sibling
        if statistic == CP.STAT_MARGIN:
            assert float(prob[2, 0]) == 1.0
        assert tok[4:].tolist() == [[-7, -7]] * 2 and prob[4:].tolist() == [[-7.0, -7.0]] * 2   # `on` 0: untouched
    return "tau -> 0 = argmax; top-k 1 = one candidate; on 0 rows bit for bit"


def test_one_matrix_the_lone_and_served_heads_are_its_first_rows():
    dev = _gpu()
    import numpy as np
    from engine.models.flashnext.engine import FlashNextEngine as E
    from engine.models.flashnext.pack import Int8Rows
    from tools import fn_kernels as K
    V, D = 8192, 2048
    g = torch.Generator().manual_seed(11)
    w = (torch.randn(V, D, generator=g) * 0.02).half()
    hv = torch.randperm(V, generator=g).numpy().astype(np.int64)        # one ranked list of hot ids
    eng = types.SimpleNamespace(device=dev, W={"lm_head.weight": types.SimpleNamespace(dequant=lambda: w.to(dev).float())},
                                draft_head=None, draft_ids=None, coupled_head=None, coupled_ids=None, lone_head=None,
                                lone_ids=None)
    for n in ("_hot_key", "_hot_head", "set_draft_head", "set_coupled_head", "_head_rows", "keyed_head"):
        f = getattr(E, n)
        setattr(eng, n, f if isinstance(E.__dict__[n], staticmethod) else types.MethodType(f, eng))
    eng.set_draft_head(hv[:1024])
    own = eng.draft_head                                      # the served head's own matrix
    own_q, own_s = own.q.clone(), own.s.clone()
    eng.set_coupled_head(hv[:4096], lone=1024)
    c = eng.coupled_head
    assert eng.draft_head is not own and eng.lone_head is eng.draft_head
    assert eng.draft_head.q.data_ptr() == c.q.data_ptr() and eng.draft_head.s.data_ptr() == c.s.data_ptr()
    assert torch.equal(eng.draft_head.q, own_q) and torch.equal(eng.draft_head.s, own_s)
    assert torch.equal(eng.draft_ids, torch.as_tensor(hv[:1024], device=dev))
    assert len(eng._draft_heads) == 1, "the served head's own matrix is freed"
    assert eng.keyed_head(1)[0] is eng.draft_head and eng.keyed_head(8)[0] is c
    ref = Int8Rows.of(w.to(dev)[torch.as_tensor(hv[:1024], device=dev)].contiguous())
    x = (torch.randn(8, D, generator=g) * 0.5).to(dev, torch.bfloat16)
    for rows in (1, 8):
        assert torch.equal(K.gemv(x[:rows], eng.draft_head), K.gemv(x[:rows], ref)), rows
        assert torch.equal(K.gemv(x[:rows], c)[:, :1024], K.gemv(x[:rows], ref)), rows
    eng.set_coupled_head(hv[:4096], lone=2048)                # a lone size of its own: a view too
    assert eng.lone_head.q.data_ptr() == c.q.data_ptr() and eng.lone_ids.numel() == 2048
    ref2 = Int8Rows.of(w.to(dev)[torch.as_tensor(hv[:2048], device=dev)].contiguous())
    assert torch.equal(eng.lone_head.q, ref2.q) and torch.equal(K.gemv(x[:1], eng.lone_head), K.gemv(x[:1], ref2))
    eng.set_coupled_head(hv[:4096], lone=4096)                # not below the coupled head: no lone head
    assert eng.lone_head is None and eng.keyed_head(1)[0] is c
    return "lone and served heads = views of the coupled head's first rows; bytes and gemv bit for bit; one matrix"


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scriptmode import Skip
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except Skip as e:
            print(f"  {name:<56} skip {e}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
