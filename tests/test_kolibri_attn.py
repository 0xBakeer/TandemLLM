"""Kolibri-1 attention and KV (engine/kolibri/attn.py) against the reference forward
(tools/kolibri_ref.py), on a tiny random Kolibri on the CPU in float32.

The tiny model keeps everything that makes the layer hard: GQA (6 query heads on 2 KV heads), the
q/k RMSNorm, RoPE on sliding layers only, a full layer with no positions, a 9-row window, and a ring
of 16 rows, so a 60-token sequence wraps the ring several times. Each test feeds the same inputs
through the cache in a different way (one prefill, chunks, single tokens, verify blocks with a
rollback, a tree with a commit, a snapshot restore) and compares with the reference's whole-sequence
attention.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri import attn as ka  # noqa: E402
from tools import kolibri_ref as ref  # noqa: E402

TOL = 2e-5


def tiny_config(window=9):
    return {"hidden_size": 64, "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 16,
            "sliding_window": window, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
            "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
            "max_position_embeddings": 4096, "num_experts_per_tok": 2, "num_experts": 4,
            "norm_topk_prob": False}


class LW:
    """The engine's KLayer, the attention half: index, sliding, fused qkv, o, q_norm, k_norm."""

    def __init__(self, **t):
        self.__dict__.update(t)


class W:
    """Random attention weights per layer, as KLayers and as the reference's LayerW."""

    def __init__(self, c, seed=0):
        g = torch.Generator().manual_seed(seed)
        H, nq, nkv, hd = (c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"],
                          c["head_dim"])
        self.lw = {}
        self.k = {}
        for L in range(len(c["layer_types"])):
            p = f"model.layers.{L}.self_attn."
            t = {"q": torch.randn(nq * hd, H, generator=g) / H ** 0.5,
                 "k": torch.randn(nkv * hd, H, generator=g) / H ** 0.5,
                 "v": torch.randn(nkv * hd, H, generator=g) / H ** 0.5,
                 "o": torch.randn(H, nq * hd, generator=g) / (nq * hd) ** 0.5,
                 "qn": 1 + 0.1 * torch.randn(hd, generator=g),
                 "kn": 1 + 0.1 * torch.randn(hd, generator=g)}
            self.lw[L] = ref.LayerW(**t)
            self.k[L] = LW(index=L, sliding=c["layer_types"][L] == "sliding_attention",
                           qkv=torch.cat([t["q"], t["k"], t["v"]]), o=t["o"],
                           q_norm=t["qn"], k_norm=t["kn"])


def setup(window=9, ring=16, max_len=256, seed=0, fp8=False, dtype=torch.float32):
    c = tiny_config(window)
    spec = ka.AttnSpec.of(c)
    w = W(c, seed)
    attn = Bound(ka.KolibriAttention(c, fp8_kv=fp8, ring_rows=ring), w)
    kv = attn.a.make_kv(max_len, "cpu", dtype=dtype)
    attn.kv = kv
    return c, spec, w, kv, attn


class Bound:
    """attn(L, x, start, positions, block_mask) over one KV, for the tests."""

    def __init__(self, a, w):
        self.a, self.w, self.kv = a, w, None

    def __call__(self, L, x, start, positions=None, block_mask=None):
        return self.a.attn_block(x, self.w.k[L], self.kv, start, positions, block_mask)


def inputs(c, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, c["hidden_size"], generator=g) for _ in c["layer_types"]]


def reference(c, w, xs):
    return [ref.attn_block(x, w.lw[L], c, c["layer_types"][L] == "sliding_attention")
            for L, x in enumerate(xs)]


def run(kv, attn, xs, start, T, positions=None, block_mask=None):
    out = [attn(L, x[start:start + T] if positions is None else x, start, positions=positions,
                block_mask=block_mask) for L, x in enumerate(xs)]
    kv.length = start + T
    return out


def feed(kv, attn, xs, sizes):
    outs, s = [], 0
    for T in sizes:
        outs.append(run(kv, attn, xs, s, T))
        s += T
    return [torch.cat([o[L] for o in outs]) for L in range(len(xs))]


def maxdiff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def test_one_prefill_equals_reference():
    c, spec, w, kv, attn = setup()
    xs = inputs(c, 60)
    assert maxdiff(feed(kv, attn, xs, [60]), reference(c, w, xs)) < TOL


@pytest.mark.parametrize("sizes", [[13, 7, 1, 1, 1, 30, 7], [1] * 60, [5, 4, 3, 2, 1, 45],
                                   [64, 1, 1, 1]])
def test_chunks_and_decode_wrap_the_ring(sizes):
    c, spec, w, kv, attn = setup()
    n = sum(sizes)
    xs = inputs(c, n)
    assert maxdiff(feed(kv, attn, xs, sizes), reference(c, w, xs)) < TOL


def test_prefill_rows_threshold_both_sides(monkeypatch):
    # the same chunking through the prefill path (rows >= PREFILL_ROWS) and the block path
    c, spec, w, kv, attn = setup()
    xs = inputs(c, 50)
    monkeypatch.setattr(ka, "PREFILL_ROWS", 4)
    a = feed(kv, attn, xs, [10, 3, 37])
    kv.reset()
    monkeypatch.setattr(ka, "PREFILL_ROWS", 1000)
    b = feed(kv, attn, xs, [10, 3, 37])
    assert maxdiff(a, b) < TOL
    assert maxdiff(a, reference(c, w, xs)) < TOL


def test_verify_block_with_rollback():
    c, spec, w, kv, attn = setup()
    n = 48
    xs = inputs(c, n)
    junk = inputs(c, n, seed=7)
    want = reference(c, w, xs)
    got = [[] for _ in xs]
    s = 0
    out = run(kv, attn, xs, 0, 20)
    for L in range(len(xs)):
        got[L].append(out[L])
    s = 20
    while s < n:
        T = min(5, n - s)
        acc = min(2, T)                      # accept 2 rows, the rest of the block is wrong
        blk = [torch.cat([x[s:s + acc], j[s + acc:s + T]]) for x, j in zip(xs, junk)]
        o = [attn(L, b, s) for L, b in enumerate(blk)]
        kv.length = s + T
        kv.rollback(s + acc)
        for L in range(len(xs)):
            got[L].append(o[L][:acc])
        s += acc
    got = [torch.cat(g) for g in got]
    assert maxdiff(got, want) < TOL


def test_tree_verify_and_commit():
    c, spec, w, kv, attn = setup()
    n = 40
    xs = inputs(c, n)
    junk = inputs(c, n, seed=9)
    want = reference(c, w, xs)
    feed(kv, attn, xs, [30])
    s = 30
    # a tree in DFS order: 0 (anchor) -> 1 (wrong) ; 0 -> 2 -> 3 (right), 3 -> 4 (wrong)
    parents = [-1, 0, 0, 2, 3]
    depth = [0, 1, 1, 2, 3]
    right = {0: 0, 2: 1, 3: 2}               # node -> offset in the true sequence
    nodes = []
    for L in range(len(xs)):
        rows = []
        for j in range(5):
            rows.append(xs[L][s + right[j]] if j in right else junk[L][s + j])
        nodes.append(torch.stack(rows))
    anc = torch.zeros(5, 5, dtype=torch.bool)
    for j in range(5):
        a = j
        while a >= 0:
            anc[j, a] = True
            a = parents[a]
    pos = torch.tensor([s + d for d in depth])
    out = [attn(L, nodes[L], s, positions=pos, block_mask=anc) for L in range(len(xs))]
    kv.length = s + 5
    for L in range(len(xs)):
        for j, off in right.items():
            assert (out[L][j] - want[L][s + off]).abs().max().item() < TOL
    kv.commit_path(s, [0, 2, 3])
    assert kv.length == s + 3
    rest = feed_from(kv, attn, xs, s + 3, [1] * (n - s - 3))
    for L in range(len(xs)):
        assert (rest[L] - want[L][s + 3:]).abs().max().item() < TOL


def feed_from(kv, attn, xs, s, sizes):
    outs = []
    for T in sizes:
        outs.append(run(kv, attn, xs, s, T))
        s += T
    return [torch.cat([o[L] for o in outs]) for L in range(len(xs))]


class _Eng:
    """What engine/cache.py reads from an engine."""

    def __init__(self, kv):
        self.kv = kv
        self.state = kv.ring


def test_snapshot_restore_through_cache():
    from engine import cache
    c, spec, w, kv, attn = setup()
    xs = inputs(c, 60)
    junk = inputs(c, 60, seed=5)
    want = reference(c, w, xs)
    eng = _Eng(kv)
    feed(kv, attn, xs, [16, 8])              # 24 rows
    snap = cache.capture(eng)
    assert snap.length == 24 and snap.conv is None
    feed_from(kv, attn, junk, 24, [16, 8, 8])  # the ring wraps twice with the wrong rows
    cache.restore(eng, snap)
    assert kv.length == 24
    rest = feed_from(kv, attn, xs, 24, [8, 1, 1, 26])
    for L in range(len(xs)):
        assert (rest[L] - want[L][24:]).abs().max().item() < TOL


def test_resident_anchor_resume():
    from engine import cache
    c, spec, w, kv, attn = setup()
    xs = inputs(c, 60)
    junk = inputs(c, 60, seed=5)
    want = reference(c, w, xs)
    eng = _Eng(kv)
    res = cache.ResidentPrefix(1 << 20, 8)
    res.capturing = True
    feed(kv, attn, xs, [8])
    res.anchor(eng, 8)
    feed_from(kv, attn, xs, 8, [8])
    res.anchor(eng, 16)
    feed_from(kv, attn, junk, 16, [8, 8, 8])
    res.resume(eng, None, 16)
    assert kv.length == 16
    rest = feed_from(kv, attn, xs, 16, [8, 8, 28])
    for L in range(len(xs)):
        assert (rest[L] - want[L][16:]).abs().max().item() < TOL


def test_fp8_kv_is_close():
    c, spec, w, kv, attn = setup(fp8=True)
    xs = inputs(c, 40)
    got = feed(kv, attn, xs, [20, 1, 1, 18])
    want = reference(c, w, xs)
    for g, r in zip(got, want):
        rel = ((g - r).norm() / r.norm()).item()
        assert rel < 0.05, rel


def test_ring_reference_matches_window_attention():
    from tools.kolibri_attn_kernels import ring_reference
    torch.manual_seed(0)
    R, W, T, start = 16, 9, 3, 30
    total = start + T
    idx = torch.full((R,), -1, dtype=torch.int32)
    ar = torch.arange(total - R, total)
    idx[ar % R] = ar.to(torch.int32)
    q = torch.randn(1, 6, T, 16)
    k = torch.randn(1, 2, R, 16)
    v = torch.randn(1, 2, R, 16)
    qlo = (torch.arange(start, start + T) - (W - 1)).to(torch.int32)
    tri = torch.ones(T, T, dtype=torch.bool).tril()
    got = ring_reference(q, k, v, idx, start, qlo, tri)
    # the same by gathering rows in index order
    for t in range(T):
        p = start + t
        keep = [s for s in range(R) if p - (W - 1) <= int(idx[s]) <= p]
        kk = k[0][:, keep].repeat_interleave(3, 0)
        vv = v[0][:, keep].repeat_interleave(3, 0)
        a = torch.softmax(torch.einsum("hd,hnd->hn", q[0, :, t], kk) / 4.0, -1)
        o = torch.einsum("hn,hnd->hd", a, vv)
        assert (o - got[0, :, t]).abs().max().item() < 1e-5


def test_memory_at_256k():
    c = tiny_config()
    c.update(hidden_size=2560, num_attention_heads=48, num_key_value_heads=4, head_dim=128,
             sliding_window=513, layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 10)
    spec = ka.AttnSpec.of(c)
    # bytes only, no allocation: 10 full layers, 2 KB a token each in BF16
    per = 2 * len(spec.full_layers) * 4 * 128 * 2
    assert per * 262144 == 5_368_709_120
    ring = 2 * len(spec.sliding_layers) * 4 * ka.RING_ROWS * 128 * 2 + 4 * ka.RING_ROWS
    assert ring < 60_000_000


def test_truncate_back_limit_and_anchor():
    c, spec, w, kv, attn = setup()
    xs = inputs(c, 60)
    junk = inputs(c, 60, seed=3)
    want = reference(c, w, xs)
    feed(kv, attn, xs, [24])
    snap = kv.snapshot()
    feed_from(kv, attn, xs, 24, [16])           # 40 rows; the ring (16) keeps 8 rows back
    assert kv.back == 8
    kv.truncate(33)                              # within reach: exact
    rest = feed_from(kv, attn, xs, 33, [1, 1, 5])
    for L in range(len(xs)):
        assert (rest[L] - want[L][33:40]).abs().max().item() < TOL
    with pytest.raises(ka.RingLost):
        kv.truncate(20)
    feed_from(kv, attn, junk, 40, [8])
    kv.restore(snap, 24)                        # the anchor at 24, then the true rows again
    rest = feed_from(kv, attn, xs, 24, [36])
    for L in range(len(xs)):
        assert (rest[L] - want[L][24:]).abs().max().item() < TOL


def test_in_the_engine_equals_torch_attention():
    """The engine on the tiny written model: this attention against its torch reference."""
    import tempfile
    from engine.kolibri.model import KolibriEngine
    from tools import kolibri_tiny as tiny
    ids = [5, 17, 3, 88, 41, 41, 2, 60, 7, 19, 33, 71, 12, 9, 50, 51, 52, 1, 2, 3]
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d)
        a = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="torch",
                               log=lambda s: None)
        b = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="kernel",
                               log=lambda s: None)
        assert type(b.attn).__module__ == "engine.kolibri.attn"
        fa, fb = a.forward(ids), b.forward(ids)
        # both run bf16 activations; each is as far from the other as from the fp32 reference,
        # so the bar is bf16 noise
        assert (fa - fb).abs().mean().item() < 0.02
        assert (fa.argmax(-1) == fb.argmax(-1)).float().mean().item() >= 0.9
        b.reset()
        b.chunk = 3
        assert (b.prefill(ids[:7]) - fb[6]).abs().max().item() < 0.1
        for i, t in enumerate(ids[7:], start=7):
            assert (b.decode(t) - fb[i]).abs().max().item() < 0.1, i
        b.truncate(15)
        assert (b.prefill(ids[15:18], start=15) - fb[17]).abs().max().item() < 0.1
