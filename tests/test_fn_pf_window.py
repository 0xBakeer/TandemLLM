"""SPD-249 P1: the host side of Flash-Next's layer-major prefill windows (engine/models/flashnext/layer_major.py), no
GPU.

  * a fake engine with Flash-Next's own `prefill` / `forward_pre` / `trunk` / `_mix_pre` / `_moe_pre`, row-wise
    stand-ins for the kernels, GEMMs whose bits depend on the call's row count and order-sensitive GDN and PLE states:
    a prompt prefilled in windows ends with the chunk-major prefill's logits, streams, states and length, with one
    expert call a layer over a window's rows, every other part a chunk at a time at its own start and rows, the slab
    and fp32-weight caches on only inside the layers; a resumed prompt too; the route trace in chunk-major order;
  * the windows' cut (`groups`), what may run one (`ok`), the knob 8,192 by default;
  * the slot verbs engine/cache.py's chunk loop calls (`window_ok`, `prefill_window`, `replay_taps`): never for a
    family built on the adapter; with a state buffer a line, each line replayed as the loop replays it holds the
    chunk-major prefill's sections, length and streams at that line, and the slot ends as `forward` leaves it.
"""

from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.models.flashnext import engine as E  # noqa: E402
from engine.models.flashnext import layer_major as LM  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402
from test_fn_joint_prefill import H, HC, NL, _Cfg, _Eng, _kernels, _moe_grouped  # noqa: E402

LINEAR = [li for li in range(NL) if _Cfg.is_linear(li)]


class _WCfg(_Cfg):
    eos_token_id = 0


class _WEng(_Eng):
    """`_Eng` with the state a slot holds (GDN state and conv tail, PLE conv and context, K rows) updated in place
    and in an order-sensitive way, and what FlashNextServing reads on the host."""

    def __init__(self):
        super().__init__()
        self.cfg = _WCfg()
        self.hash = {1: types.SimpleNamespace(context_len=3)}
        self._pending = {}
        self.route_trace = None

    def new_state(self, cap=None, paged=False):
        st = FNState(cap=64)
        for li in LINEAR:
            st.S[li] = torch.zeros(H)
            st.conv[li] = torch.zeros(1, H)
        st.ple_conv[1] = torch.zeros(1, HC * H)
        for li in range(NL):
            if not _Cfg.is_linear(li):
                st.k[li] = torch.zeros(64, H)
        return st

    def parked(self, st):
        return False

    def drop_parked(self, st):
        pass

    def _seen(self, name, x, st):
        super()._seen(name, x, st)
        assert self._pre["L0"] == st.length, (name, self._pre, st.length)      # the chunk's own start

    def _ple_pre(self, h, tokens, li, st):
        self._seen("ple", h, st)
        old = st.ple_ctx.get(li, torch.zeros(1, 0, dtype=torch.long))
        st.ple_ctx[li] = torch.cat([old, tokens.view(1, -1)], 1)[:, -3:]
        st.ple_conv[li].mul_(0.5).add_(h[-1:])
        return h * 0.5 + st.ple_conv[li] * 0.1 + 1e-3 * h.shape[0]

    def _gdn_pre(self, x, li, st):
        self._seen("gdn", x, st)
        y = torch.tanh(x + st.S[li]) + 1e-3 * x.shape[0]
        st.S[li].add_(y.sum(0))
        st.conv[li].copy_(x[-1:])
        return y

    def _attn_pre(self, x, li, st):
        self._seen("attn", x, st)
        k = st.k[li]
        L0 = self._pre["L0"]
        k[L0:L0 + x.shape[0]] = x
        return (torch.cumsum(k[:L0 + x.shape[0]], 0)[L0:] / (torch.arange(x.shape[0])[:, None] + 1 + L0)
                + 1e-3 * x.shape[0])


def _same(a, b):
    assert a.length == b.length, (a.length, b.length)
    for name in ("S", "conv", "ple_conv", "k"):
        da, db = getattr(a, name), getattr(b, name)
        assert sorted(da) == sorted(db) and all(torch.equal(da[li], db[li]) for li in da), name
    assert sorted(a.ple_ctx) == sorted(b.ple_ctx) and all(torch.equal(a.ple_ctx[li], b.ple_ctx[li]) for li in a.ple_ctx)


def _prefill(fn, toks, rows, st=None, monkeypatch=None):
    monkeypatch.setattr(LM, "ROWS", rows)
    st = st or fn.new_state()
    _moe_grouped.calls = []
    fn.parts.clear()
    lg = fn.prefill(toks, st, keep_stack=True)
    return lg, fn.prompt_stack, st, list(_moe_grouped.calls)


def test_windows_leave_the_prompt_as_the_chunk_major_prefill(monkeypatch):
    _kernels(monkeypatch)
    monkeypatch.setattr(E, "PREFILL_CHUNK", 4)
    fn = _WEng()
    toks = torch.tensor([3, 9, 1, 4, 7, 2, 8, 5, 6, 1, 1])        # chunks of 4, 4, 3
    lg0, s0, st0, calls0 = _prefill(fn, toks, 0, monkeypatch=monkeypatch)
    assert calls0 == [4] * NL + [4] * NL + [3] * NL
    for rows, want in ((8, [8] * NL + [3] * NL), (12, [11] * NL), (64, [11] * NL)):
        lg, s, st, calls = _prefill(fn, toks, rows, monkeypatch=monkeypatch)
        assert torch.equal(lg, lg0) and lg.shape == (1, 11), rows
        assert torch.equal(s, s0) and s.shape == (11, HC * H), rows
        _same(st, st0)
        assert calls == want, (rows, calls)
        if rows >= 11:                                             # one window: the caches on in every layer
            assert fn.parts and all(wf and slab for *_, wf, slab in fn.parts)
        assert PK._WF_CACHE is None and X._SLAB_CACHE is None and fn._pre is None
    # a window runs layer l for every chunk before layer l + 1: the first layer's parts are the window's chunks
    lg, s, st, calls = _prefill(fn, toks, 12, monkeypatch=monkeypatch)
    assert [(n, r) for n, _, r, _, _ in fn.parts[:3]] == [("gdn", 4), ("gdn", 4), ("gdn", 3)]
    return "windows of 8 and 12+ rows: logits, streams, states, length = chunk-major; one expert call a window and layer"


def test_nocopy_windows_write_the_experts_input_side_by_side(monkeypatch):
    """SPD-249 P4 (QWEN38_FN_PF_NOCOPY): a window's MLP mixers write one buffer of the window's rows, which the
    experts read in every layer; the prompt is the chunk-major prefill's."""
    _kernels(monkeypatch)
    monkeypatch.setattr(E, "PREFILL_CHUNK", 4)
    fn = _WEng()
    toks = torch.tensor([3, 9, 1, 4, 7, 2, 8, 5, 6, 1, 1])        # chunks of 4, 4, 3
    lg0, s0, st0, _ = _prefill(fn, toks, 0, monkeypatch=monkeypatch)
    assert E.PF_NOCOPY is (os.environ.get("QWEN38_FN_PF_NOCOPY", "1") == "1")
    seen, outs, grouped, tail = [], [], PK.moe_grouped, PK.mix_tail

    def rec(x, ids, bank, rows=32):
        seen.append((x.data_ptr(), x.shape[0]))
        return grouped(x, ids, bank, rows=rows)

    def mt(u, n, hc, out=None):
        outs.append(None if out is None else out.data_ptr())
        return tail(u, n, hc, out)
    monkeypatch.setattr(PK, "moe_grouped", rec)
    monkeypatch.setattr(PK, "mix_tail", mt)
    for nocopy in (False, True):
        monkeypatch.setattr(E, "PF_NOCOPY", nocopy)
        for rows, want, chunks in ((8, [8] * NL + [3] * NL, 2), (12, [11] * NL, 3)):
            seen.clear()
            outs.clear()
            lg, s, st, calls = _prefill(fn, toks, rows, monkeypatch=monkeypatch)
            assert torch.equal(lg, lg0) and torch.equal(s, s0), (nocopy, rows)
            _same(st, st0)
            assert calls == want and [n for _, n in seen] == want, (nocopy, rows, calls)
            given = [p for p in outs if p is not None]
            assert len(outs) == 2 * 3 * NL, len(outs)              # two mixers a chunk and layer
            if not nocopy:
                assert not given, (rows, len(given))
                continue
            assert len(given) == chunks * NL, (rows, len(given))   # the MLP mixers of the window's chunks
            assert {p for p, n in seen if n > 4} == {min(given)}, rows   # the experts read that buffer, every layer
    return "the window's MLP mixers write one buffer the experts read; without the knob none; the same prompt"


def test_a_resumed_prompt_and_the_route_trace(monkeypatch):
    _kernels(monkeypatch)
    monkeypatch.setattr(E, "PREFILL_CHUNK", 4)
    fn = _WEng()
    head, toks = torch.tensor([5, 2, 7]), torch.tensor([3, 9, 1, 4, 7, 2, 8, 5, 6])
    outs = []
    for rows in (0, 12):
        st = fn.new_state()
        monkeypatch.setattr(LM, "ROWS", 0)
        fn.prefill(head, st)
        fn.route_trace = []
        lg, s, st, calls = _prefill(fn, toks, rows, st=st, monkeypatch=monkeypatch)
        outs.append((lg, s, st, [(li, t.clone()) for li, t in fn.route_trace]))
        fn.route_trace = None
    (lg0, s0, st0, tr0), (lg1, s1, st1, tr1) = outs
    assert torch.equal(lg0, lg1) and torch.equal(s0, s1) and st1.length == 12
    _same(st0, st1)
    assert [li for li, _ in tr0] == [li for li, _ in tr1] == list(range(NL)) * 3
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(tr0, tr1))
    return "a prompt resumed at 3: the chunk-major bits; the route trace in chunk-major order"


def test_a_window_takes_the_engine_inject(monkeypatch):
    """The injects of a window are `_inject_pre`'s (QWEN38_FN_PF_INJ_FUSED reaches them), two a chunk and layer, and
    the fp32 forms no mixer took are dropped at the window's end."""
    _kernels(monkeypatch)
    monkeypatch.setattr(E, "PREFILL_CHUNK", 4)
    fn = _WEng()
    seen, cleared = [], []
    real = E.FlashNextEngine._inject_pre

    def inj(h, out, di, R, w, hc, eps):
        seen.append(h.shape[0])
        return real(fn, h, out, di, R, w, hc, eps)
    monkeypatch.setattr(fn, "_inject_pre", inj, raising=False)
    monkeypatch.setattr(fn, "_n32_clear", lambda: cleared.append(1), raising=False)
    toks = torch.tensor([3, 9, 1, 4, 7, 2, 8, 5, 6, 1, 1])        # chunks of 4, 4, 3
    _prefill(fn, toks, 12, monkeypatch=monkeypatch)
    assert sorted(seen) == sorted([4, 4, 3] * 2 * NL), seen
    assert cleared == [1]
    return "a window of 3 chunks: 2 engine injects a chunk and layer, one clear"


def test_the_cut_and_what_runs_a_window(monkeypatch):
    assert LM.groups([4, 4, 3], 8) == [[0, 1], [2]]
    assert LM.groups([4, 4, 4, 4], 8) == [[0, 1], [2, 3]]
    assert LM.groups([4, 4, 3], 4) == [[0], [1], [2]]
    assert LM.groups([4, 4, 3], 64) == [[0, 1, 2]]
    assert LM.groups([], 8) == []
    assert LM.ROWS == int(os.environ.get("QWEN38_FN_PF_WINDOW") or 8192)          # 8,192 by default
    fn = _WEng()
    st = fn.new_state()
    assert LM.ok(fn, st)
    st.delta = 3
    assert not LM.ok(fn, st)
    st.delta, st.mm = 0, object()
    assert not LM.ok(fn, st)
    st.mm = None
    fn.prefill_path = "ref"
    assert not LM.ok(fn, st)
    return "windows of at most the knob's rows; text on the prefill kernels only; 8,192 by default"


def test_the_window_takes_the_memory_that_is_free(monkeypatch):
    assert LM.rows_now(4096, 0) == 4096 and LM.rows_now(4096, 4096) == 4096       # off, or one chunk: chunk-major
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    G = 1 << 30
    for free, cached, want in ((LM.MARGIN + 64 * G, 0, 32768),                     # room for the whole window
                               (LM.MARGIN + 2 * G, 0, 4096 + (2 * G // LM.ROW_BYTES) // 4096 * 4096),
                               (LM.MARGIN + 1 * G, 1 * G, 4096 + (2 * G // LM.ROW_BYTES) // 4096 * 4096),
                               (LM.MARGIN // 2, 0, 4096)):                         # short of the margin: chunk-major
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda f=free: (f, 128 * G))
        nested = {"reserved_bytes": {"all": {"current": 3 * G + cached}}, "allocated_bytes": {"all": {"current": 3 * G}}}
        monkeypatch.setattr(torch.cuda, "memory_stats_as_nested_dict", lambda n=nested: n)   # cache.cuda_stats' source
        got = LM.rows_now(4096, 32768)
        assert got == min(32768, want) and got % 4096 == 0, (free, cached, got, want)
    return "free + cached memory past the margin buys rows at ROW_BYTES each, in whole chunks, at most the knob"


def _replay(s, toks, starts, monkeypatch):
    """engine/cache.py's window loop around the slot verb, with a state buffer at every line."""
    bufs = [cache.state_empty(s) for _ in toks]
    lg, hid, taps = s.prefill_window(toks, starts, bufs)
    cache.state_save(bufs[-1], s)
    lines = []
    for c, (a, t) in enumerate(zip(starts, toks)):
        cache.state_load(s, bufs[c])
        s.kv.length = a + t.numel()
        s.replay_taps(taps[c])
        lines.append(({k: v.clone() for k, v in s.state.sections().items()}, s.st.length,
                      s.hidden_post_norm.clone(), {li: v.clone() for li, v in s.st.ple_ctx.items()}))
    return lg, hid, lines


def test_the_slot_verbs_hold_every_line(monkeypatch):
    _kernels(monkeypatch)
    monkeypatch.setattr(E, "PREFILL_CHUNK", 4)
    from engine.models.flashnext import serving as SV
    monkeypatch.setattr(SV, "PREFILL_CHUNK", 4)
    fn = _WEng()
    toks = torch.tensor([3, 9, 1, 4, 7, 2, 8, 5, 6, 1, 1, 2, 9, 4])           # chunks of 4, 4, 4, 2
    starts = [0, 4, 8, 12]
    chunks = [toks[a:a + 4] for a in starts]
    a = FlashNextServing(fn, 64)
    monkeypatch.setattr(LM, "ROWS", 0)
    assert not a.window_ok(0)                                                  # QWEN38_FN_PF_WINDOW=0: chunk-major
    monkeypatch.setattr(LM, "ROWS", 8)
    assert a.window_ok(0)

    class Sub(FlashNextServing):
        pass
    assert not Sub(fn, 64).window_ok(0)                                       # a family built on the adapter
    want = []
    monkeypatch.setattr(LM, "ROWS", 0)
    for t, s0 in zip(chunks, starts):                                          # chunk-major: one forward a chunk
        lg0 = a.forward(t, s0)
        want.append(({k: v.clone() for k, v in a.state.sections().items()}, a.st.length,
                     a.hidden_post_norm.clone(), {li: v.clone() for li, v in a.st.ple_ctx.items()}))
    for rows in (8, 64):                                                       # windows [[0, 1], [2, 3]] and one
        monkeypatch.setattr(LM, "ROWS", rows)
        b = FlashNextServing(fn, 64)
        lg, hid, lines = _replay(b, chunks, starts, monkeypatch)
        assert torch.equal(lg, lg0) and lg.shape == (1, 1, 11)
        assert [h.shape[1] for h in hid] == [4, 4, 4, 2]
        for c, (got, ref) in enumerate(zip(lines, want)):
            assert sorted(got[0]) == sorted(ref[0]), c
            bad = [k for k in ref[0] if not torch.equal(ref[0][k], got[0][k])]
            assert not bad, (rows, c, bad)
            assert got[1] == ref[1] and torch.equal(got[2], ref[2]), (rows, c)
            assert sorted(got[3]) == sorted(ref[3]) and all(torch.equal(got[3][li], ref[3][li]) for li in ref[3])
        _same(a.st, b.st)
        assert torch.equal(a._h_last, b._h_last) and b.stats["prefill_rows"] == 14
    return "off at 0, not for a subclass; every line's sections, length, streams and PLE context = chunk-major"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
