"""The resident prefix's end anchor: the next turn resumes at the previous prompt's last GDN-chunk
line instead of its last prefix-chunk line, on the random 4-layer model.

Two claims, and they are separate.

  1. Taking the anchor changes nothing the prefill computes. The state is copied out of the
     prefill's last forward as it passes the line; the forward is the same call, so its logits,
     S, conv and KV are bit-equal with the anchor on and off. Checked with the model as it is.

  2. Resuming from it is exact PROVIDED a forward's rows do not depend on where the forward
     started. On the GPU that is a property of the kernels and tools/resident_check.py measures
     it on the real engine. On this CPU it is not true of `scaled_dot_product_attention` (its
     blocking depends on the query count), so these tests swap in an attention that computes
     every query row on its own. What they then check is the bookkeeping: the right state at the
     right row, the right anchor chosen, and the fallbacks.

Here the prefix chunk is 64 and the GDN chunk 16 (the served stack: 1024 and 64).

Run: python tests/test_resident_end.py
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine.model as M  # noqa: E402
from engine import cache, gdn  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402

CH = 64
SUB = 16
BIG = 1 << 30
M.GDN_PREFILL_CHUNK = SUB

_SDPA = F.scaled_dot_product_attention


def rowwise_sdpa(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False, **_):
    """Attention with every query row computed by itself: its result cannot depend on how many
    other rows share the call."""
    from torch.nn.attention.bias import CausalBias
    B, H, T, D = q.shape
    N = k.shape[2]
    if k.shape[1] != H:
        rep = H // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    rows = []
    for r in range(T):
        if isinstance(attn_mask, torch.Tensor) and not isinstance(attn_mask, CausalBias):
            cols = attn_mask[r].nonzero().flatten()
            kk, vv = k[:, :, cols], v[:, :, cols]
        else:
            lim = r + 1 if is_causal else (N - T + r + 1 if isinstance(attn_mask, CausalBias)
                                          else N)
            kk, vv = k[:, :, :lim], v[:, :, :lim]
        s = (q[:, :, r:r + 1] @ kk.transpose(-1, -2)) * D ** -0.5
        rows.append(torch.softmax(s, -1) @ vv)
    return torch.cat(rows, 2)


class rowwise:
    def __enter__(self):
        F.scaled_dot_product_attention = rowwise_sdpa

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = _SDPA
        return False


def resident(sub=SUB, **kw):
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(kw.pop("budget", BIG), CH, sub=sub, **kw)


def cold(ids):
    eng = fresh()
    lg, _, _ = cache.prefill(eng, None, ids, DEV, store=None, chunk=CH)
    return eng, lg


def same_state(a, b, n):
    assert torch.equal(a.state.S, b.state.S), (a.state.S - b.state.S).abs().max().item()
    assert torch.equal(a.state.conv, b.state.conv)
    assert torch.equal(a.kv.k[:, :, :, :n], b.kv.k[:, :, :, :n])
    assert torch.equal(a.kv.v[:, :, :, :n], b.kv.v[:, :, :, :n])
    assert a.kv.length == b.kv.length == n


def decode(eng, n_prompt, k=9, seed=99):
    gen = tokens(k, seed=seed)
    eng.forward(torch.tensor(gen), start=n_prompt, last_only=True)
    return gen


# ------------------------------------------------------------------ 1. taking it changes nothing

def test_the_anchor_sits_at_the_last_gdn_line_and_the_prefill_is_unchanged():
    ids = tokens(300, seed=1)
    res_on, eng_on = resident(), fresh()
    lg_on, _, _ = cache.prefill(eng_on, None, ids, DEV, chunk=CH, resident=res_on)
    res_off, eng_off = resident(sub=0), fresh()
    lg_off, _, _ = cache.prefill(eng_off, None, ids, DEV, chunk=CH, resident=res_off)
    assert sorted(res_on.anchors) == [64, 128, 192, 256, 288], sorted(res_on.anchors)
    assert sorted(res_off.anchors) == [64, 128, 192, 256]
    assert res_on.stats["tail_anchors"] == 1
    assert torch.equal(lg_on, lg_off)
    same_state(eng_on, eng_off, 300)


def test_the_anchor_is_the_state_a_forward_ending_there_leaves():
    """The captured S and conv against a forward that stops at the line: the recurrence and the
    convolution tail are per chunk and per row, so on this CPU they agree bit for bit too."""
    ids = tokens(300, seed=2)
    res, eng = resident(), fresh()
    cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    S, conv = res.anchors[288]
    with rowwise():
        ref = fresh()
        for s, e in ((0, 64), (64, 128), (128, 192), (192, 256), (256, 288)):
            ref.forward(torch.tensor(ids[s:e]), start=s, last_only=True)
    assert torch.equal(S, ref.state.S), (S - ref.state.S).abs().max().item()
    assert torch.equal(conv, ref.state.conv)


def test_the_reference_recurrence_hands_back_the_state_at_a_chunk_line():
    g = torch.Generator().manual_seed(3)
    B, T, H, D = 1, 80, 2, 8
    q, k, v = (torch.randn(B, T, H, D, generator=g) for _ in range(3))
    gg = -torch.rand(B, T, H, generator=g)
    beta = torch.rand(B, T, H, generator=g)
    S0 = torch.randn(B, H, D, D, generator=g)
    out = torch.empty(B, H, D, D)
    o_full, S_full = gdn.chunk_gated_delta_rule(q, k, v, gg, beta, S0, chunk_size=16,
                                                state_at=3, state_out=out)
    o_plain, S_plain = gdn.chunk_gated_delta_rule(q, k, v, gg, beta, S0, chunk_size=16)
    assert torch.equal(o_full, o_plain) and torch.equal(S_full, S_plain)
    _, S_48 = gdn.chunk_gated_delta_rule(q[:, :48], k[:, :48], v[:, :48], gg[:, :48],
                                         beta[:, :48], S0, chunk_size=16)
    assert torch.equal(out, S_48), (out - S_48).abs().max().item()


# ------------------------------------------------------------------ 2. the next turn

def test_the_next_turn_resumes_at_the_end_anchor_bit_for_bit():
    with rowwise():
        turn1 = tokens(300, seed=4)
        res, eng = resident(), fresh()
        cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
        turn2 = turn1 + decode(eng, 300) + tokens(30, seed=5)
        info = {}
        warm, reused, fwd = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res,
                                          info=info)
        assert (reused, fwd, info["kind"]) == (288, len(turn2) - 288, "resident")
        assert res.stats["tail_hits"] == 1
        ref, lg = cold(turn2)
        assert torch.equal(warm, lg), (warm - lg).abs().max().item()
        same_state(eng, ref, len(turn2))


def test_a_growing_conversation_reads_only_the_new_tokens_turn_after_turn():
    """Eight turns of an agent loop: each forwards at most the new tokens plus 15, and the last
    one equals a cold prefill of the whole conversation."""
    with rowwise():
        conv = tokens(130, seed=6)
        res, eng = resident(), fresh()
        cache.prefill(eng, None, conv, DEV, chunk=CH, resident=res)
        fwd_total, new_total = 0, 0
        for turn in range(8):
            before = len(conv)
            conv = conv + decode(eng, before, k=5, seed=40 + turn) + tokens(20 + 2 * turn,
                                                                            seed=60 + turn)
            warm, reused, fwd = cache.prefill(eng, None, conv, DEV, chunk=CH, resident=res)
            assert reused >= before - (SUB - 1), (turn, reused, before)
            fwd_total += fwd
            new_total += len(conv) - before
        assert fwd_total - new_total <= 8 * (SUB - 1), (fwd_total, new_total)
        ref, lg = cold(conv)
        assert torch.equal(warm, lg)
        same_state(eng, ref, len(conv))


def test_a_first_forward_shorter_than_a_gdn_chunk_falls_back_to_the_grid():
    """Resuming at 288 for a 300-token prompt would make the first forward 12 rows: fewer than a
    GDN chunk, which the served stack runs through other kernels than the cold forward's."""
    with rowwise():
        turn1 = tokens(300, seed=7)
        res, eng = resident(), fresh()
        cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
        turn2 = turn1[:298] + [5, 6]
        warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
        assert reused == 256, reused
        ref, lg = cold(turn2)
        assert torch.equal(warm, lg)
        same_state(eng, ref, 300)


def test_a_turn_that_diverges_inside_the_previous_prompt():
    with rowwise():
        turn1 = tokens(300, seed=8)
        for cut, want in ((292, 288), (280, 256)):
            res, eng = resident(), fresh()
            cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
            decode(eng, 300)
            turn2 = turn1[:cut] + tokens(40, seed=9)
            warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
            assert reused == want, (cut, reused)
            ref, lg = cold(turn2)
            assert torch.equal(warm, lg), cut
            same_state(eng, ref, len(turn2))


def test_no_end_anchor_below_the_first_grid_line():
    """Under the first grid line the cold forward starts the sequence (causal from row 0, a fresh
    convolution); a forward resumed there would not."""
    res, eng = resident(), fresh()
    cache.prefill(eng, None, tokens(50, seed=10), DEV, chunk=CH, resident=res)
    assert res.anchors == {} and res.stats["tail_anchors"] == 0


def test_a_prompt_on_a_gdn_line_keeps_its_final_state():
    with rowwise():
        turn1 = tokens(304, seed=11)
        res, eng = resident(), fresh()
        cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
        assert 304 in res.anchors
        turn2 = turn1 + decode(eng, 304) + tokens(25, seed=12)
        warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
        assert reused == 304
        ref, lg = cold(turn2)
        assert torch.equal(warm, lg)
        same_state(eng, ref, len(turn2))


def test_a_guest_between_two_turns_keeps_the_end_anchor():
    with rowwise():
        turn1 = tokens(300, seed=13)
        res, eng = resident(), fresh()
        cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
        answer = decode(eng, 300)
        cache.prefill(eng, None, tokens(20, seed=14), DEV, chunk=CH, resident=res)
        decode(eng, 20, k=12, seed=15)
        assert res.stats["guests"] == 1
        turn2 = turn1 + answer + tokens(11, seed=16)
        warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
        assert reused == 288, reused
        ref, lg = cold(turn2)
        assert torch.equal(warm, lg)
        same_state(eng, ref, len(turn2))


# ------------------------------------------------------------------ 3. failure and budget

def test_a_capture_a_layer_cannot_honour_is_never_used():
    ids = tokens(300, seed=17)
    res, eng = resident(), fresh()
    old = M.GDN_PREFILL_CHUNK
    M.GDN_PREFILL_CHUNK = 24               # 288 - 256 = 32 is not on a 24-row chunk grid
    try:
        cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    finally:
        M.GDN_PREFILL_CHUNK = old
    assert 288 not in res.anchors and res.stats["tail_spoiled"] == 1
    assert len(res._free) == 1, "the buffers go back to the pool"


def test_a_forward_that_raises_returns_the_capture_buffers():
    ids = tokens(300, seed=18)
    res, eng = resident(), fresh()
    real = eng.forward

    def boom(blk, start=0, last_only=False):
        if eng.capture is not None:
            raise RuntimeError("forward failed")
        return real(blk, start=start, last_only=last_only)

    eng.forward = boom
    try:
        cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert eng.capture is None and 288 not in res.anchors and res.stats["tail_spoiled"] == 1


def test_the_anchor_budget_is_not_exceeded():
    eng = fresh()
    per = eng.state.nbytes
    res = resident(budget=5 * per, tail=2)
    cache.prefill(eng, None, tokens(500, seed=19), DEV, chunk=CH, resident=res)
    assert len(res.anchors) <= 5 and res._allocated <= 5, (sorted(res.anchors), res._allocated)
    assert 496 in res.anchors


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:70s} ok", flush=True)
            passed += 1
    print(f"{passed} passed")
