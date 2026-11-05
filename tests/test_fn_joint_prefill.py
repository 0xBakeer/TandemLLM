"""SPD-249: the host side of Flash-Next's joint prefill (engine/models/flashnext/joint_prefill.py), no GPU.

  * a fake engine with Flash-Next's own `prefill` / `forward_pre` / `trunk` / `_mix_pre` / `_moe_pre` and row-wise
    stand-ins for the kernels, but GEMMs whose result depends on the call's row count (as the library GEMMs pick
    their algorithm by M): the joint prefill leaves every prompt's logits, streams, state and length as its own
    prefill does, with one expert call a layer over every prompt's rows, each other part a prompt at a time with its
    own `_pre`, the slab and fp32-weight caches on only inside the layers;
  * the jobs are cut into runs of at most FN_PREFILL_CHUNK rows, a lone run through `prefill` itself;
  * `ok`: the kernels path, one chunk, no decode step, no images;
  * `mm_rows`' fp32 weight cache converts a weight once while on and gives the uncached bits;
  * the adapter's verbs: off by default, never for a family built on the adapter (Ornith), each slot left as its
    `forward` leaves it;
  * the server's burst hook runs an adapter's own joint prefill for the members it takes;
  * a member's own steps before the forward (`prefill_joint_prepare`: its chain, its start, its KV pages) run on its own
    thread: a member whose pool says no (`PoolFull`) or whose start is off its state fails alone, first or last to
    come, and the others still get their logits from one joint forward without it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext import engine as E  # noqa: E402
from engine.models.flashnext import joint_prefill as JP  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from engine.kvpool import PoolFull  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

H, HC, NL, NE, TOPK, R = 4, 2, 4, 5, 2, 3


class _Cfg:
    hidden_size, hc_count, rms_norm_eps = H, HC, 1e-6
    num_hidden_layers, num_experts, num_experts_per_tok = NL, NE, TOPK
    ple_layers = {1: 0}

    @staticmethod
    def is_linear(li):
        return li != 2


class _Bank:
    def __init__(self, li):
        self.scale = 1.0 + 0.1 * li


def _which(run: list, objs: list) -> list:
    """The indices in `objs` of a joint run's members, sorted: a run lists them in the order they joined, which two
    threads started one after the other do not fix."""
    return sorted(next(i for i, o in enumerate(objs) if o is x) for x in run)


def _mm(x, w):
    """A library GEMM whose bits depend on the call's rows (its algorithm by M)."""
    return x.float() @ w.float().T + 1e-3 * x.shape[0]


class _Eng:
    """What the joint prefill reads of FlashNextEngine; the trunk and the prefill are FlashNextEngine's own."""

    prefill = E.FlashNextEngine.prefill
    forward_pre = E.FlashNextEngine.forward_pre
    trunk = E.FlashNextEngine.trunk
    _mix_pre = E.FlashNextEngine._mix_pre
    _moe_pre = E.FlashNextEngine._moe_pre
    moe_group_rows = None                # the grouped GEMM's default item rows, as FlashNextEngine
    prefill_path = "kernels"
    device, dtype = "cpu", torch.float32

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        r = lambda *s: torch.randn(*s, generator=g) * 0.3          # noqa: E731
        self.cfg = _Cfg()
        self.W = {"emb": r(50, H), "head": r(11, HC * H)}
        for li in range(NL + 1):
            self.W[f"norm.{li}"] = r(H)
        for li in range(NL):
            p = f"layers.{li}."
            for site in ("attn_hyper_connection", "mlp_hyper_connection"):
                self.W[p + site + ".down_inject"] = r(R + HC, HC * H)
                self.W[p + site + ".rank"] = R
                self.W[p + site + ".input_mix_weight_up.weight"] = r(HC * H, R)
            self.W[p + "mlp_hyper_connection.hc_norm.weight"] = r(H)
            self.W[p + "attn_hyper_connection.hc_norm.weight"] = r(H)
            self.W[p + "mlp.router"] = r(NE + 1, H)
        self.banks = {li: _Bank(li) for li in range(NL)}
        self.route_trace = None
        self.flushed, self.parts, self.moe_rows = [], [], []

    def pre(self, li):
        return f"layers.{li}."

    def flush(self, st=None):
        self.flushed.append(id(st))

    def _pre_ctl(self, L0, T, st):
        return {"L0": L0, "T": T, "st": id(st)}

    def embed(self, t):
        return self.W["emb"][t.long()]

    def _site_norm(self, li):
        return self.W[f"norm.{li}"]

    def _seen(self, name, x, st):
        assert self._pre["st"] == id(st) and self._pre["T"] == x.shape[0], (name, self._pre, x.shape)
        self.parts.append((name, id(st), x.shape[0], PK._WF_CACHE is not None, X._SLAB_CACHE is not None))

    def _ple_pre(self, h, tokens, li, st):
        self._seen("ple", h, st)
        st.ple_ctx[li] = torch.cat([st.ple_ctx.get(li, torch.zeros(1, 0, dtype=torch.long)), tokens.view(1, -1)], 1)
        return h * 0.5 + tokens[:, None].float() * 0.01 + 1e-3 * h.shape[0]

    def _gdn_pre(self, x, li, st):
        self._seen("gdn", x, st)
        s0 = st.S.setdefault(li, torch.zeros(H))
        y = torch.tanh(x + s0) + 1e-3 * x.shape[0]
        st.S[li] = s0 + y.sum(0)
        return y

    def _attn_pre(self, x, li, st):
        self._seen("attn", x, st)
        k = st.k.setdefault(li, torch.zeros(64, H))
        k[self._pre["L0"]:self._pre["L0"] + x.shape[0]] = x
        return torch.cumsum(x, 0) / (torch.arange(x.shape[0])[:, None] + 1 + self._pre["L0"]) + 1e-3 * x.shape[0]

    def final(self, n):
        return n.float() @ self.W["head"].T


def _hc_norm(x, w, group, eps, out=None):
    v = x.float().view(x.shape[0], -1, group)
    return (v * torch.rsqrt((v * v).mean(-1, keepdim=True) + eps) * (1 + w.float())).view(x.shape).to(x.dtype)


def _inject_norm(h0, out, d, Rk, w, hc, eps):
    inj = 2 * torch.sigmoid(d[:, Rk:Rk + hc].float() / hc)
    h = h0 + (out.float()[:, None, :] * inj[:, :, None]).reshape(h0.shape)
    return h, (_hc_norm(h, w, H, eps) if w is not None else None)


def _router(logits, E_, k):
    v, i = torch.topk(logits[:, :E_], k, dim=-1)
    ids = torch.cat([i, torch.full_like(i[:, :1], E_)], 1).to(torch.int32)
    return ids, torch.cat([torch.softmax(v, -1), torch.sigmoid(logits[:, E_:])], 1)


def _moe_grouped(x, ids, bank, rows=32):
    S = ids.shape[1]
    _moe_grouped.calls.append(x.shape[0])
    p = torch.arange(ids.numel())
    return x[p // S].float() * (ids.reshape(-1)[:, None].float() + 1) * bank.scale


def _moe_combine(yd, ids, wts, out=None):
    M, S = ids.shape
    return (yd.view(M, S, -1) * wts[:, :, None]).sum(1)


def _mix_tail(u, n, hc):
    return (torch.sigmoid(u) * n.float()).view(n.shape[0], hc, -1).mean(1)


def _kernels(monkeypatch):
    monkeypatch.setattr(K, "hc_norm", _hc_norm)
    monkeypatch.setattr(K, "inject_norm", _inject_norm)
    monkeypatch.setattr(K, "router", _router)
    monkeypatch.setattr(K, "moe_combine", _moe_combine)
    monkeypatch.setattr(PK, "mm_rows", _mm)
    monkeypatch.setattr(PK, "mix_tail", _mix_tail)
    _moe_grouped.calls = []
    monkeypatch.setattr(PK, "moe_grouped", _moe_grouped)


def _state(length=0):
    st = FNState(cap=64)
    st.length = length
    return st


def _same_state(a, b):
    assert a.length == b.length
    assert sorted(a.S) == sorted(b.S) and all(torch.equal(a.S[li], b.S[li]) for li in a.S)
    assert sorted(a.k) == sorted(b.k) and all(torch.equal(a.k[li], b.k[li]) for li in a.k)
    assert sorted(a.ple_ctx) == sorted(b.ple_ctx) and all(torch.equal(a.ple_ctx[li], b.ple_ctx[li]) for li in a.ple_ctx)


def test_the_joint_prefill_leaves_every_prompt_as_its_own_prefill(monkeypatch):
    _kernels(monkeypatch)
    fn = _Eng()
    toks = [torch.tensor([3, 9, 1, 4, 7]), torch.tensor([2, 8, 5]), torch.tensor([6, 1, 1, 2, 9, 4, 3])]
    starts = [0, 0, 4]                                  # the third resumes past a 4-row prefix
    lone = [_state(s) for s in starts]
    joint = [_state(s) for s in starts]
    for st in lone + joint:
        st.S[0] = torch.full((H,), 0.25 * id(st) % 3)  # differing entry states would show a mix-up
    for a, b in zip(lone, joint):
        b.S[0] = a.S[0].clone()
    want = []
    for t, st in zip(toks, lone):
        lg = fn.prefill(t, st, keep_stack=True)
        want.append((lg, fn.prompt_stack))
    lone_moe = list(_moe_grouped.calls)
    fn.parts.clear()
    fn.flushed.clear()
    _moe_grouped.calls = []
    got = JP.prefill_joint(fn, list(zip(toks, joint)))
    for (lg, stack), (lg2, stack2), a, b in zip(want, got, lone, joint):
        assert torch.equal(lg, lg2) and lg2.shape == (1, 11)
        assert torch.equal(stack, stack2)
        _same_state(a, b)
    assert lone_moe == [5, 5, 5, 5, 3, 3, 3, 3, 7, 7, 7, 7]
    assert _moe_grouped.calls == [15] * NL                       # one expert call a layer, every prompt's rows
    assert fn.flushed == [id(st) for st in joint]
    assert [(n, r) for n, _, r, _, _ in fn.parts[:3]] == [("gdn", 5), ("gdn", 3), ("gdn", 7)]
    assert [n for n, _, _, _, _ in fn.parts if n == "ple"] == ["ple"] * 3
    assert all(wf and slab for _, _, _, wf, slab in fn.parts)    # the caches on inside the layers
    assert PK._WF_CACHE is None and X._SLAB_CACHE is None and fn._pre is None
    assert fn.prompt_stack is got[-1][1]
    return "logits, streams, states equal to each lone prefill; one expert call of 15 rows a layer"


def test_a_lone_run_and_the_runs_cut(monkeypatch):
    _kernels(monkeypatch)
    assert JP.runs([5, 3, 7], 8) == [[0, 1], [2]]
    assert JP.runs([9, 1], 8) == [[0], [1]]
    assert JP.runs([2, 2, 2, 2], 8) == [[0, 1, 2, 3]]
    assert JP.runs([], 8) == []
    monkeypatch.setattr(E, "PREFILL_CHUNK", 8)
    fn = _Eng()
    toks = [torch.tensor([3, 9, 1, 4, 7]), torch.tensor([2, 8, 5]), torch.tensor([6, 1, 1, 2, 9, 4, 3])]
    lone, joint = [_state() for _ in toks], [_state() for _ in toks]
    want = [(fn.prefill(t, st, keep_stack=True), fn.prompt_stack) for t, st in zip(toks, lone)]
    _moe_grouped.calls = []
    got = JP.prefill_joint(fn, list(zip(toks, joint)))
    assert _moe_grouped.calls == [8] * NL + [7] * NL             # a run of two, then the third alone
    for (lg, stack), (lg2, stack2), a, b in zip(want, got, lone, joint):
        assert torch.equal(lg, lg2) and torch.equal(stack, stack2)
        _same_state(a, b)
    st = _state()
    refused = False
    try:
        JP.prefill_joint(fn, [(toks[0], st), (toks[1], st)])
    except AssertionError as e:
        refused = "a state twice" in str(e)
    assert refused
    return "runs of at most the chunk; a lone job through prefill itself"


def test_which_prompts_join():
    fn = types.SimpleNamespace(prefill_path="kernels")
    st = _state()
    assert JP.ok(fn, 205, 0, st) and JP.ok(fn, 1, 0, st) and JP.ok(fn, E.PREFILL_CHUNK, 0, st)
    assert not JP.ok(fn, E.PREFILL_CHUNK + 1, 0, st)              # more than one chunk
    assert not JP.ok(fn, 1, 9, st)                                # a decode step
    assert JP.ok(fn, 7, 9, st)                                    # a chunk past a prefix
    assert not JP.ok(fn, 0, 0, st)
    st.mm = object()
    assert not JP.ok(fn, 205, 0, st)                              # images
    st.mm, st.delta = None, 3
    assert not JP.ok(fn, 205, 0, st)
    assert not JP.ok(types.SimpleNamespace(prefill_path="ref"), 205, 0, _state())
    return "one kernels chunk, no decode step, no images"


def test_the_weight_cache_converts_once_and_keeps_the_bits():
    class W8:
        def __init__(self):
            g = torch.Generator().manual_seed(3)
            self.q = torch.randint(-127, 127, (6, 8), generator=g, dtype=torch.int8)
            self.s = torch.rand(6, 1, generator=g)
            self.n = 0

        def dequant(self):
            self.n += 1
            return self.q.float() * self.s
    w = W8()
    x = torch.randn(5, 8).to(torch.bfloat16)
    base, base3 = PK.mm_rows(x, w), PK.mm_rows(x[:3], w)
    assert w.n == 2 and PK._WF_CACHE is None
    PK.wf_cache_on()
    try:
        a, b = PK.mm_rows(x, w), PK.mm_rows(x[:3], w)
        assert w.n == 3 and torch.equal(a, base) and torch.equal(b, base3)
        PK.wf_cache_clear()
        PK.mm_rows(x, w)
        assert w.n == 4
    finally:
        PK.wf_cache_off()
    PK.mm_rows(x, w)
    assert w.n == 5 and PK._WF_CACHE is None
    return "one conversion a weight while on, the same bits"


class _SFN:
    """What FlashNextServing reads of the engine on the host."""

    device, dtype = "cpu", torch.float32

    def __init__(self):
        self.cfg = types.SimpleNamespace(ple_layers={}, eos_token_id=0, hc_count=1, hidden_size=3)
        self.hash = {}
        self._pending = {}

    def new_state(self, cap=None, paged=False):
        return _state()

    def parked(self, st):
        return False

    def drop_parked(self, st):
        pass


def test_the_adapter_verbs(monkeypatch):
    fn = _SFN()
    a, b = FlashNextServing(fn, 64), FlashNextServing(fn, 64)
    b.st.length = 4
    fn.prefill_path = "kernels"
    assert not JP.ON and not a.prefill_joint_ok(205, 0)          # off by default
    monkeypatch.setattr(JP, "ON", True)
    assert a.prefill_joint_ok(205, 0) and not a.prefill_joint_ok(1, 4) and not a.prefill_joint_ok(E.PREFILL_CHUNK + 1, 0)

    class Sub(FlashNextServing):
        pass
    assert not Sub(fn, 64).prefill_joint_ok(205, 0)               # a family built on the adapter: its own trunk
    seen = []

    def fake(fn_, jobs):
        seen.append((fn_, [(t.tolist(), st) for t, st in jobs]))
        return [(torch.full((1, 5), float(i)), torch.arange(t.numel() * 3, dtype=torch.float32).view(-1, 3) + i)
                for i, (t, st) in enumerate(jobs)]
    monkeypatch.setattr(JP, "prefill_joint", fake)
    out = FlashNextServing.prefill_joint([(a, torch.tensor([1, 2]), 0), (b, torch.tensor([3, 4, 5]), 4)])
    assert seen == [(fn, [([1, 2], a.st), ([3, 4, 5], b.st)])]
    assert out[0].shape == (1, 1, 5) and float(out[1].view(-1)[0]) == 1.0
    assert a.hidden_post_norm.shape == (1, 2, 3) and b.hidden_post_norm.shape == (1, 3, 3)
    assert a._h_last.tolist() == [[3.0, 4.0, 5.0]] and b._h_last.tolist() == [[7.0, 8.0, 9.0]]
    assert a.stats["prefill_rows"] == 2 and b.stats["prefill_rows"] == 3
    try:
        FlashNextServing.prefill_joint([(a, torch.tensor([1]), 3)])
        raise AssertionError("a start off the state's length")
    except ValueError:
        pass
    return "off by default, not for a subclass; each slot's streams, last row and counts its own"


def test_the_burst_hook_runs_an_adapters_own_joint_prefill(monkeypatch):
    from server import app, prefill_burst
    from server import slots as slots_mod
    runs = []

    class Eng:
        def __init__(self, take):
            self.take = take

        def prefill_joint_ok(self, T, start):
            return self.take

        @staticmethod
        def prefill_joint(items):
            runs.append([(e, t.tolist(), s) for e, t, s in items])
            return [f"logits {t.tolist()}" for e, t, s in items]
    burst = prefill_burst.Burst(wait_s=5.0)
    monkeypatch.setitem(app.STATE, "pool", types.SimpleNamespace(burst=burst, turn=None))
    engs = [Eng(True), Eng(True), Eng(False)]
    for i in range(3):
        assert burst.arrive(i, idle=True)
    got = {}

    def member(i):
        slots_mod.bind(types.SimpleNamespace(index=i) if i else None)
        got[i] = app._joint_forward(engs[i], torch.tensor([i, i + 1]), 0, True)
    ts = [threading.Thread(target=member, args=(i,)) for i in (2, 0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert got == {0: "logits [0, 1]", 1: "logits [1, 2]", 2: None}
    assert len(runs) == 1 and _which([x[0] for x in runs[0]], engs) == [0, 1]
    assert burst.stats["joint"] == 1 and burst.stats["joint_rows"] == 4
    return "the two that join get the adapter's joint logits; the one it refuses prefills alone"



class _Pages:
    """A paged KV set (engine/kvpool.py `PagedSet`) whose pool says no past `cap` rows."""

    def __init__(self, cap):
        self.cap, self.rows, self.calls = cap, 0, 0

    def margin(self):
        return 0

    def on_device(self):
        return False

    def ensure(self, rows):
        self.calls += 1
        if rows > self.cap:
            raise PoolFull(f"KV pool: {rows} rows past {self.cap}")
        self.rows = max(self.rows, int(rows))
        return 0


def test_a_members_own_failure_fails_it_alone(monkeypatch):
    from server import app, prefill_burst
    from server import slots as slots_mod
    fn = _SFN()
    fn.prefill_path = "kernels"
    monkeypatch.setattr(JP, "ON", True)
    runs = []

    def fake(fn_, jobs):
        runs.append([st for _, st in jobs])
        for t, st in jobs:
            st.length += t.numel()
        return [(torch.full((1, 5), float(t[0])), torch.zeros(t.numel(), 3)) for t, st in jobs]
    monkeypatch.setattr(JP, "prefill_joint", fake)
    for failing_last in (False, True):
        runs.clear()
        burst = prefill_burst.Burst(wait_s=5.0)
        monkeypatch.setitem(app.STATE, "pool", types.SimpleNamespace(burst=burst, turn=None))
        ss = [FlashNextServing(fn, 64) for _ in range(4)]
        for s in ss:
            s.st.pages = _Pages(cap=100)
        ss[2].st.pages = _Pages(cap=4)                 # its pool says no to its 6 rows
        asks = [(torch.tensor([10, 11]), 0), (torch.tensor([20, 21]), 0), (torch.arange(30, 36), 0),
                (torch.tensor([40, 41]), 4)]           # member 3: a start its state (0 rows) does not hold
        for i in range(4):
            assert burst.arrive(i, idle=True)
        got = {}

        def member(i):
            slots_mod.bind(types.SimpleNamespace(index=i) if i else None)
            try:
                got[i] = app._joint_forward(ss[i], asks[i][0], asks[i][1], True)
            except Exception as e:                      # noqa: BLE001
                got[i] = e

        def go(idx):
            ts = [threading.Thread(target=member, args=(i,)) for i in idx]
            for t in ts:
                t.start()
            return ts
        if failing_last:                               # the healthy two wait; member 2 comes last and leads them
            ts = go((0, 1))
            t_end = time.monotonic() + 5
            while time.monotonic() < t_end:
                with burst.cv:
                    if len(burst.jobs) == 2:
                        break
                time.sleep(0.005)
            for i in (3, 2):
                ts += go((i,))
                ts[-1].join(10)
        else:                                          # the failing two first, then the healthy two
            ts = go((2, 3))
            for t in ts:
                t.join(10)
            ts += go((0, 1))
        for t in ts:
            t.join(10)
        assert isinstance(got[2], PoolFull) and isinstance(got[3], ValueError), got
        assert float(got[0].view(-1)[0]) == 10.0 and float(got[1].view(-1)[0]) == 20.0
        # one joint forward, without the failing members
        assert len(runs) == 1 and _which(runs[0], [s.st for s in ss]) == [0, 1], runs
        assert burst.stats["joint"] == 1 and burst.stats["joint_rows"] == 4 and not burst.is_open()
        assert ss[2].st.length == 0 and ss[3].st.length == 0 and ss[2].stats["prefill_rows"] == 0
        assert ss[0].st.length == 2 and ss[0].stats["prefill_rows"] == 2
        assert [s.st.pages.calls for s in ss] == [1, 1, 1, 0]  # each slot's pages on its own thread, once
    return "PoolFull and a bad start fail their member alone (first or last); the others share one forward"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
