"""On the board: the lane-grouped attention of a graphed joint pass (engine/models/qwen3_5/attn_group.py,
QWEN38_JOINT_ATTN_GROUP) gives every lane the bits of its lone launch (`decode_attention_dev` after the
lane's own scatter), over lanes of different lengths, context classes and tree masks, at 8 and 16 rows,
and the scatter writes the same rows. Run on the box under the lock: python tests/gpu/test_attn_group_gpu.py
"""

from __future__ import annotations

import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.qwen3_5 import attn_group as AG  # noqa: E402
from tools.attn_kernels import decode_attention_dev  # noqa: E402

HQ, HKV, D = 24, 4, 128                      # the 27B's attention heads
MAXLEN = 6144


def _lane(g, T: int, start: int, dev):
    """A stand-in engine: its cache, static slots and lengths, its tree mask."""
    k = (torch.randn(1, HKV, MAXLEN, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    v = (torch.randn(1, HKV, MAXLEN, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    kv = types.SimpleNamespace(k=k.unsqueeze(0), v=v.unsqueeze(0), slot=[0])   # [layers, 1, Hkv, L, D]
    slots = torch.arange(start, start + T, device=dev)
    lenp = torch.tensor([start, start + T], dtype=torch.int32, device=dev)
    lc = start + T
    c = 1024
    while c < lc:
        c *= 2
    jg = types.SimpleNamespace(slots=slots, lenp=lenp, max_lc=c)
    # a tree: row t sees its ancestors; here a chain with a fork at row 2
    anc = torch.zeros(T, T, dtype=torch.bool, device=dev)
    for t in range(T):
        anc[t, t] = True
        parent = t - 1 if t != 2 else 0
        while parent >= 0:
            anc[t, parent] = True
            parent = parent - 1 if parent != 2 else 0
            if parent == 0 and t != 0:
                anc[t, 0] = True
                break
    tree = types.SimpleNamespace(anc_incl=anc)
    eng = types.SimpleNamespace(kv=kv, _jg=jg, tree=tree)
    return eng


def _case(T: int, starts: list, seed: int):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    payloads, refs = [], []
    for s in starts:
        eng = _lane(g, T, s, dev)
        q = (torch.randn(1, HQ, T, D, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        k = (torch.randn(1, HKV, T, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        # v as the engine hands it: a transpose of [1, T, Hkv, D] (not contiguous)
        v = (torch.randn(1, T, HKV, D, device=dev, generator=g) * 0.5).to(torch.bfloat16).transpose(1, 2)
        # the lone launch on a copy of the cache
        kc, vc = eng.kv.k[0].clone(), eng.kv.v[0].clone()
        kc.index_copy_(2, eng._jg.slots, k)
        vc.index_copy_(2, eng._jg.slots, v)
        refs.append((decode_attention_dev(q, kc, vc, eng._jg.lenp, eng.tree.anc_incl, eng._jg.max_lc), kc, vc))
        payloads.append((eng, q, k, v))
    AG.ON = True
    outs = AG.run_attn("0", payloads)
    for (eng, *_), o, (ref, kc, vc) in zip(payloads, outs, refs):
        assert o.shape == ref.shape, (o.shape, ref.shape)
        assert torch.equal(o, ref), (T, starts, (o != ref).sum().item())
        assert torch.equal(eng.kv.k[0], kc) and torch.equal(eng.kv.v[0], vc), "the scatter's rows"


def test_grouped_bits():
    _case(8, [300, 700, 1500, 40, 2100, 999, 5000, 1023], 0)
    _case(16, [120, 3000, 640, 2047], 1)
    _case(8, [512], 2)                                      # one lane: the lone path
    _case(16, [1024, 1025], 3)                              # the class boundary


def _refs(T: int, starts: list, seed: int):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    payloads, refs = [], []
    for s in starts:
        eng = _lane(g, T, s, dev)
        q = (torch.randn(1, HQ, T, D, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        k = (torch.randn(1, HKV, T, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        v = (torch.randn(1, T, HKV, D, device=dev, generator=g) * 0.5).to(torch.bfloat16).transpose(1, 2)
        kc, vc = eng.kv.k[0].clone(), eng.kv.v[0].clone()
        kc.index_copy_(2, eng._jg.slots, k)
        vc.index_copy_(2, eng._jg.slots, v)
        refs.append(decode_attention_dev(q, kc, vc, eng._jg.lenp, eng.tree.anc_incl, eng._jg.max_lc))
        payloads.append((eng, q, k, v))
    return payloads, refs


def test_every_launch_keeps_its_table():
    """Two graphs of different shapes over the same lane objects, replayed in turn, each give the lone bits:
    a graph's table is its own (a table shared by the layer gave the last capture's pointers to every
    graph's replay). Eager calls back to back without a synchronisation keep their rows too."""
    AG.ON = True
    pa, ra = _refs(8, [300, 700, 1500, 40], 10)
    pb, rb = _refs(16, [120, 3000, 640, 2047], 11)
    graphs = []
    for payloads in (pa, pb):
        keep: list = []
        AG.CAPTURE_KEEP = keep
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            AG.run_attn("0", payloads)                          # warm (compiles)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = AG.run_attn("0", payloads)
        AG.CAPTURE_KEEP = None
        assert len(keep) == 1, len(keep)
        graphs.append((g, outs, keep))
    for _ in range(3):
        for (g, outs, _), refs in zip(graphs, (ra, rb)):
            g.replay()
            torch.cuda.synchronize()
            for o, ref in zip(outs, refs):
                assert torch.equal(o, ref), "a graph's replay lost its table"
    # eager, back to back
    oa = AG.run_attn("0", pa)
    ob = AG.run_attn("0", pb)
    torch.cuda.synchronize()
    assert all(torch.equal(o, r) for o, r in zip(oa, ra)) and all(torch.equal(o, r) for o, r in zip(ob, rb))


if __name__ == "__main__":
    test_grouped_bits()
    test_every_launch_keeps_its_table()
    print("ok", AG.STATS)
