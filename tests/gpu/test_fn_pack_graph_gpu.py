"""On the board (SPD-249, T1): the attention of a packed MTP draft graph's one-row steps.

A packed draft's step after the feed is one row a lane, each on its own MTP cache. The graph (QWEN38_FN_PACK_GRAPH)
runs it as `attn_prep_multi` + `attn_rows_multi` + one gate over every lane's row, the lanes reached through entry
tables, where the eager pass calls `attn_prep` + `attn_rows` + the gate a lane at a time. Here, at the MTP layer's
shapes (dense rows, the indexer key written, no indexer query):

  * 8 lanes of one row (different lengths, caches, slots): the queries, the K / V / indexer-key cache rows written and
    the gated outputs equal the lane-at-a-time calls bit for bit;
  * the same in a CUDA graph (the entry copies inside it) captured once and replayed after the pinned entry rows were
    rewritten for other caches in another lane order, and for 5 lanes of a 5-lane graph: each replay equals the eager
    lane-at-a-time calls on the same inputs.

The draft's own bits (every pick of a real trace, eager pass against graph) are tools/fn_pack_graph_check.py's.

Run on the box: python tests/gpu/test_fn_pack_graph_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import fn_kernels as K  # noqa: E402
from tools import fn_seq_kernels as SK  # noqa: E402

DEV = "cuda"
HQ, HK, D, ROT, CAP = 16, 2, 256, 64, 2048
IHD, NHI = 128, 4
IOFF = NHI * IHD


def _consts(g):
    qw = torch.randn(D, generator=g, device=DEV).bfloat16() * 0.1
    kw = torch.randn(D, generator=g, device=DEV).bfloat16() * 0.1
    ang = torch.arange(CAP, device=DEV).float()[:, None] * torch.linspace(0.001, 0.5, ROT, device=DEV)[None, :]
    return qw, kw, ang.cos().bfloat16(), ang.sin().bfloat16()


def _lane(g, L):
    """One lane's MTP cache and its staged draft row at position L (a draft step: the row sees L + 1 keys)."""
    kc = torch.randn(HK, CAP, D, generator=g, device=DEV).bfloat16()
    vc = torch.randn(HK, CAP, D, generator=g, device=DEV).bfloat16()
    ikc = torch.randn(CAP, IHD, generator=g, device=DEV).bfloat16()
    pos = torch.tensor([L], dtype=torch.int64, device=DEV)
    ctl = torch.tensor([[L + 1, L, 1, 0]], dtype=torch.int32, device=DEV)
    anc = torch.zeros(1, SK.ANC, dtype=torch.int32, device=DEV)
    anc[0, 0] = L
    sel = torch.zeros(1, SK.NSEL, dtype=torch.int32, device=DEV)
    return {"kc": kc, "vc": vc, "ikc": ikc, "pos": pos, "slot": pos.clone(), "ctl": ctl, "anc": anc, "sel": sel}


def _clone(lanes):
    return [{k: v.clone() for k, v in x.items()} for x in lanes]


def _eager(lanes, qkv, iqk, c):
    """The eager pass: a lane at a time (`_attn1_core` of a packed step), its gate, the rows joined."""
    qw, kw, cos, sin = c
    ys = []
    for j, x in enumerate(lanes):
        q = K.attn_prep(qkv[j:j + 1], iqk[j:j + 1], x["pos"], qw, kw, cos, sin, x["kc"], x["vc"], x["ikc"], HQ, HK, IOFF,
                        1e-6, slot=x["slot"])
        o = SK.attn_rows(q, x["kc"], x["vc"], x["ctl"], x["anc"], x["sel"])
        ys.append(K.gate_mul(o, qkv[j:j + 1], D))
    return torch.cat(ys)


def _fill(ap_h, ar_h, lanes):
    for j, x in enumerate(lanes):
        ap_h[j] = torch.tensor(K.attn_prep_entry(j, 1, x["kc"], x["vc"], x["ikc"], x["pos"], x["slot"]), dtype=torch.int64)
        ar_h[j] = torch.tensor(SK.attn_rows_entry(j, x["kc"], x["vc"], x["ctl"], x["anc"], x["sel"]), dtype=torch.int64)


class _Multi:
    """The graph's part: entry rows copied in, one prep, one attention, one gate over the lanes' rows."""

    def __init__(self, n, qkv, iqk, c, strides):
        self.ap_h = torch.zeros(n, K.APM_FIELDS, dtype=torch.int64).pin_memory()
        self.ar_h = torch.zeros(n, SK.ARM_FIELDS, dtype=torch.int64).pin_memory()
        self.ap_d = torch.zeros(n, K.APM_FIELDS, dtype=torch.int64, device=DEV)
        self.ar_d = torch.zeros(n, SK.ARM_FIELDS, dtype=torch.int64, device=DEV)
        self.lane = torch.arange(n, dtype=torch.int32, device=DEV)
        self.qkv, self.iqk, self.c, self.strides, self.bufs = qkv, iqk, c, strides, {}

    def __call__(self):
        qw, kw, cos, sin = self.c
        self.ap_d.copy_(self.ap_h, non_blocking=True)
        self.ar_d.copy_(self.ar_h, non_blocking=True)
        q = K.attn_prep_multi(self.qkv, self.iqk, self.ap_d, qw, kw, cos, sin, HQ, HK, IOFF, IHD, 0, 1e-6)
        o = SK.attn_rows_multi(q, self.ar_d, self.lane, self.strides, HK, bufs=self.bufs)
        return K.gate_mul(o, self.qkv, D)


def _same(a, b, what):
    for x, y in zip(a, b):
        for k in ("kc", "vc", "ikc"):
            assert torch.equal(x[k], y[k]), f"{what}: cache {k}"


def test_one_row_lanes_equal_a_lane_at_a_time():
    g = torch.Generator(device=DEV).manual_seed(41)
    c = _consts(g)
    Ls = [205, 211, 233, 190, 640, 1201, 999, 300]
    lanes = [_lane(g, L) for L in Ls]
    n = len(lanes)
    qkv = torch.randn(n, HQ * 2 * D + 2 * HK * D, generator=g, device=DEV).bfloat16()
    iqk = torch.randn(n, IOFF + IHD, generator=g, device=DEV).bfloat16()
    lone = _clone(lanes)
    y0 = _eager(lone, qkv, iqk, c)
    kc = lanes[0]["kc"]
    m = _Multi(n, qkv, iqk, c, (kc.stride(0), kc.stride(1), lanes[0]["vc"].stride(0), lanes[0]["vc"].stride(1)))
    _fill(m.ap_h, m.ar_h, lanes)
    y1 = m()
    torch.cuda.synchronize()
    assert torch.equal(y1, y0), "gated outputs"
    _same(lanes, lone, "one launch each")
    return f"{n} one-row lanes: outputs and K / V / indexer-key rows bit-equal to a lane at a time"


def test_a_graph_replays_other_lanes_after_new_entries():
    g = torch.Generator(device=DEV).manual_seed(43)
    c = _consts(g)
    out = []
    for n in (8, 5):
        qkv = torch.empty(n, HQ * 2 * D + 2 * HK * D, dtype=torch.bfloat16, device=DEV)
        iqk = torch.empty(n, IOFF + IHD, dtype=torch.bfloat16, device=DEV)
        pool = [_lane(g, L) for L in (205, 211, 233, 190, 640, 1201, 999, 300, 410, 77)[:n + 2]]
        kc0, vc0 = pool[0]["kc"], pool[0]["vc"]
        m = _Multi(n, qkv, iqk, c, (kc0.stride(0), kc0.stride(1), vc0.stride(0), vc0.stride(1)))
        order = list(range(n))
        _fill(m.ap_h, m.ar_h, [pool[i] for i in order])
        qkv.copy_(torch.randn(n, qkv.shape[1], generator=g, device=DEV).bfloat16())
        iqk.copy_(torch.randn(n, iqk.shape[1], generator=g, device=DEV).bfloat16())
        saved = _clone(pool)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m()                                       # the warm-up (the capture's first run)
        torch.cuda.current_stream().wait_stream(s)
        for x, y in zip(pool, saved):                 # the warm-up wrote cache rows: start the check from the saved
            for k in ("kc", "vc", "ikc"):
                x[k].copy_(y[k])
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            y = m()
        perms = [list(range(n)), [n + 1] + list(range(1, n)), list(reversed(range(2, n + 2)))]
        for r, order in enumerate(perms):
            lanes = [pool[i] for i in order]
            qkv.copy_(torch.randn(n, qkv.shape[1], generator=g, device=DEV).bfloat16())
            iqk.copy_(torch.randn(n, iqk.shape[1], generator=g, device=DEV).bfloat16())
            torch.cuda.synchronize()
            lone = _clone(lanes)
            y0 = _eager(lone, qkv, iqk, c)
            _fill(m.ap_h, m.ar_h, lanes)
            gr.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, y0), (n, r, "gated outputs")
            _same(lanes, lone, f"{n} lanes, replay {r}")
        out.append(f"{n} lanes x {len(perms)} lane sets")
    return "graph replays after new entry rows = eager lane at a time: " + ", ".join(out)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}")
    sys.exit(1 if fails else 0)
