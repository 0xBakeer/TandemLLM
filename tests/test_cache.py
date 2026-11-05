"""The serving-time caches, on a random 4-layer model a laptop can run in fp32.

What is worth testing here is one claim and three properties.

THE CLAIM: a restore is exact. A request that resumes at a checkpoint must compute the same logits,
to the last bit, as one that forwarded the whole prefix under the same chunking. Not "the same
argmax", not "close" -- the same tensor, because the snapshot is the same bytes and the arithmetic
after it sees the same inputs. If that holds on this model it holds on the 27 B one, because the
thing being tested is the state layout and not the size: the same 48-layer recurrence, the same
convolution window, the same KV, the same partial rotary.

The properties: the store must never answer with a state that a DIFFERENT prefix produced (a hash
is a hint, the tokens are the answer); the byte budget must actually bound the memory; and the
suffix store must survive a restart and find what it was told.

Run: python tests/test_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Triton kernels are a different arithmetic order for the same quantities and this file runs on a
# CPU. Same convention as tests/test_forward_tree.py.
for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from engine.models.qwen3_5.adapter import Qwen38Engine  # noqa: E402  (the served class: its cold verbs)
from test_forward_tree import FakeWeights, tiny_config  # noqa: E402

DEV = "cpu"
DT = torch.float32


def fresh(seed: int = 3):
    cfg = tiny_config()
    eng = Qwen38Engine(cfg, FakeWeights(cfg, seed), max_len=512, device=DEV)
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    return eng


def tokens(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 96, (n,), generator=g).tolist()


# ------------------------------------------------------------------ 1. the hash is a hash

def test_prefix_hashes_are_prefixes():
    ids = tokens(64)
    h = cache.prefix_hashes(ids)
    assert len(h) == len(ids) + 1
    for i in (0, 1, 17, 63, 64):
        assert h[i] == cache.prefix_hashes(ids[:i])[i]
    # a different token at the end is a different hash for that length and the same for shorter
    other = list(ids)
    other[-1] ^= 1
    h2 = cache.prefix_hashes(other)
    assert h2[-1] != h[-1] and h2[-2] == h[-2]


def test_the_hash_memo_is_a_fresh_pass_for_every_prompt():
    memo = cache._PrefixHashMemo()
    a = tokens(300, 5)
    seqs = [a, a + tokens(20, 6), a[:150] + tokens(40, 7), [], tokens(9, 8), a]
    for ids in seqs:
        assert memo(ids) == cache.prefix_hashes(ids)


def test_extend_hashes_matches_a_fresh_pass():
    a, b = tokens(20, 1), tokens(9, 2)
    assert cache.extend_hashes(cache.prefix_hashes(a), b) == cache.prefix_hashes(a + b)


# ------------------------------------------------------------------ 2. a restore is exact

def test_restore_reproduces_the_state_bit_for_bit():
    eng = fresh()
    ids = tokens(40)
    with torch.no_grad():
        eng.forward(torch.tensor(ids), start=0, last_only=True)
    snap = cache.capture(eng)
    S, conv = eng.state.S.clone(), eng.state.conv.clone()
    k, v, L = eng.kv.k.clone(), eng.kv.v.clone(), eng.kv.length
    # scribble over everything, the way another request would
    with torch.no_grad():
        eng.reset()
        eng.forward(torch.tensor(tokens(33, seed=9)), start=0, last_only=True)
    cache.restore(eng, snap)
    assert torch.equal(eng.state.S, S)
    assert torch.equal(eng.state.conv, conv)
    assert torch.equal(eng.kv.k[:, :, :, :L, :], k[:, :, :, :L, :])
    assert torch.equal(eng.kv.v[:, :, :, :L, :], v[:, :, :, :L, :])
    assert eng.kv.length == L and eng.state.primed


def test_resumed_prefill_equals_a_cold_one_bit_for_bit():
    """THE claim. Same chunking on both sides, so the two are the same arithmetic."""
    ids = tokens(100, seed=4)
    chunk = 16
    eng = fresh()
    store = cache.StateStore(1 << 30, chunk=chunk)
    cold, reused, fwd = cache.prefill(eng, None, ids, DEV, store=store, chunk=chunk,
                                      checkpoint=True)
    assert (reused, fwd) == (0, len(ids))
    assert store.report()["entries"] >= 5

    # a second request with the same prompt: everything up to the last checkpoint is restored
    eng2 = fresh()
    warm, reused2, fwd2 = cache.prefill(eng2, None, ids, DEV, store=store, chunk=chunk,
                                        checkpoint=True)
    assert reused2 == 96, reused2          # 100 tokens, boundaries at 16..96
    assert fwd2 == 4
    assert torch.equal(cold, warm), (cold - warm).abs().max().item()
    assert torch.equal(eng.state.S, eng2.state.S)
    assert torch.equal(eng.kv.k[:, :, :, :100, :], eng2.kv.k[:, :, :, :100, :])


def test_a_longer_prompt_sharing_a_prefix_resumes_at_the_last_shared_boundary():
    """The shared-system-prompt case: same 64 tokens, different tail."""
    chunk = 16
    shared = tokens(64, seed=5)
    a = shared + tokens(20, seed=6)
    b = shared + tokens(25, seed=7)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, a, DEV, store=store, chunk=chunk, checkpoint=True)

    eng2 = fresh()
    lg_warm, reused, fwd = cache.prefill(eng2, None, b, DEV, store=store, chunk=chunk,
                                         checkpoint=True)
    assert reused == 64 and fwd == 25
    eng3 = fresh()
    lg_cold, _, _ = cache.prefill(eng3, None, b, DEV, store=None, chunk=chunk)
    assert torch.equal(lg_cold, lg_warm)


def test_a_session_resumes_from_the_end_of_the_previous_turn():
    """Turn two forwards the new tokens and nothing else.

    And it is NOT bit-identical to a cold prefill of turn two, which is the one place in this file
    where that sentence is true. A session boundary is wherever the previous turn happened to stop
    -- 50 here -- and a cold prefill's chunk grid is anchored at 0, so the cold run computes
    positions 48..56 in one forward and the warm run computes 50..56 in one forward. The chunked
    delta rule blocks a 9-row call differently from a 7-row one and the two arithmetics part in the
    last bits. On the real engine there is a second, larger reason: the state a turn ENDS in was
    written by a speculative verify block, not by a prefill chunk, so it is the state that really
    produced the previous turn rather than the state a re-read of it would produce.

    So the gate for the session cache is the gate the whole engine is held to -- the same argmax --
    and not bitwise equality. `tools/cache_gate.py` measures it on the 27 B model.
    """
    chunk = 16
    turn1 = tokens(50, seed=11)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=chunk, checkpoint=True)
    # what the server does at the end of a turn
    store.put(turn1, cache.capture(eng), conv_id="c1")

    turn2 = turn1 + tokens(7, seed=12)
    eng2 = fresh()
    warm, reused, fwd = cache.prefill(eng2, None, turn2, DEV, store=store, chunk=chunk)
    assert reused == 50 and fwd == 7
    eng3 = fresh()
    cold, _, _ = cache.prefill(eng3, None, turn2, DEV, store=None, chunk=chunk)
    assert int(cold.argmax(-1)) == int(warm.argmax(-1))
    rel = (cold - warm).abs().max().item() / cold.abs().max().item()
    assert rel < 1e-5, rel


def test_a_session_resume_at_a_grid_boundary_is_bit_exact():
    """The same case with the boundary on the chunk grid: then the tail splits identically."""
    chunk = 16
    turn1 = tokens(48, seed=11)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=chunk, checkpoint=True)
    store.put(turn1, cache.capture(eng), conv_id="c1")
    turn2 = turn1 + tokens(7, seed=12)
    eng2 = fresh()
    warm, reused, fwd = cache.prefill(eng2, None, turn2, DEV, store=store, chunk=chunk)
    assert reused == 48 and fwd == 7
    eng3 = fresh()
    cold, _, _ = cache.prefill(eng3, None, turn2, DEV, store=None, chunk=chunk)
    assert torch.equal(cold, warm)


def test_a_full_prompt_hit_still_forwards_its_last_token():
    """A snapshot holds no logits, so the answer always costs at least one forward."""
    chunk = 8
    ids = tokens(32, seed=13)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    store.put(ids, cache.capture(eng))
    eng2 = fresh()
    _, reused, fwd = cache.prefill(eng2, None, ids, DEV, store=store, chunk=chunk)
    # the entry at 32 is a complete answer to the prompt and is still not usable: the logits the
    # request needs are the ones the 32nd token's own forward produces, and no snapshot holds
    # them. So the resume is the last boundary BELOW the prompt, which is 24.
    assert (reused, fwd) == (24, 8)


def test_the_lower_right_bias_computes_the_same_thing_as_the_boolean_mask():
    """The fused form of a chunk's mask has to BE the mask, not merely be faster.

    `engine/model.py` hands a prefill chunk at `start > 0` a `causal_lower_right` bias instead of a
    materialised boolean, because the boolean costs 1.7-1.9x on the board. The two are the same
    relation -- `ones(T, ctx).tril(ctx - T)` -- and this asserts it end to end rather than on the
    mask alone, at a threshold low enough for a 16-token chunk to reach it.
    """
    import importlib
    import engine.model as M
    ids = tokens(80, seed=81)
    prev = M.CHUNK_LOWER_RIGHT_FROM
    try:
        M.CHUNK_LOWER_RIGHT_FROM = 0            # the boolean everywhere
        a, _, _ = cache.prefill(fresh(), None, ids, DEV, chunk=16)
        M.CHUNK_LOWER_RIGHT_FROM = 4            # the fused bias from 4 rows up
        b, _, _ = cache.prefill(fresh(), None, ids, DEV, chunk=16)
    finally:
        M.CHUNK_LOWER_RIGHT_FROM = prev
        importlib.invalidate_caches()
    assert int(a.argmax(-1)) == int(b.argmax(-1))
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel < 1e-5, rel


# ------------------------------------------------------------------ 3. the store's own rules

def test_the_chunk_grid_is_anchored_at_zero_not_at_the_resume():
    """A session resume is off the grid; everything it checkpoints must be back on it.

    Otherwise the misalignment propagates: the next request resumes from an off-grid checkpoint,
    chunks from there, stores more off-grid boundaries, and the bit-exactness the prefix cache is
    supposed to have is gone for good after one conversation.
    """
    chunk = 16
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    turn1 = tokens(50, seed=71)
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=chunk, checkpoint=True)
    store.put(turn1, cache.capture(eng), conv_id="c")          # the off-grid session boundary
    turn2 = turn1 + tokens(60, seed=72)
    eng2 = fresh()
    _, reused, _ = cache.prefill(eng2, None, turn2, DEV, store=store, chunk=chunk, checkpoint=True)
    assert reused == 50
    grid = [L for L in store.report()["boundaries"] if L % chunk]
    assert grid == [50], grid     # the session put is the only off-grid entry in the store


def test_a_collision_is_rejected_rather_than_answered():
    store = cache.StateStore(1 << 30, chunk=8)
    eng = fresh()
    a = tokens(24, seed=21)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
    store.put(a, cache.capture(eng))
    # forge a lookup for a DIFFERENT prefix whose hash is claimed to be a's
    b = tokens(24, seed=22)
    forged = cache.prefix_hashes(a)
    assert store.find(b, forged, max_len=24) is None
    assert store.stats["rejected_collisions"] >= 1


def test_the_budget_bounds_the_memory():
    eng = fresh()
    with torch.no_grad():
        eng.forward(torch.tensor(tokens(40, seed=31)), start=0, last_only=True)
    one = cache.capture(eng).nbytes
    store = cache.StateStore(int(one * 2.5), chunk=8)
    for i in range(10):
        ids = tokens(40, seed=100 + i)
        store.put(ids, cache.capture(eng))
    r = store.report()
    assert r["bytes"] <= store.budget
    assert r["entries"] <= 3 and r["evictions"] >= 7


def test_capture_refuses_before_cloning():
    eng = fresh()
    a = tokens(40, seed=61)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
    one = cache.capture(eng).nbytes
    assert cache.capture(eng, max_bytes=one // 4) is None, "the cap must refuse before cloning"
    assert cache.capture(eng, max_bytes=one * 3) is not None


def test_a_rolled_back_block_snapshots_as_the_prefix_it_kept():
    """`rollback_to` restored the recurrent state to the kept prefix and left
    `kv.length` at the end of the REJECTED block, and `capture` snapshots `kv.length` rows. A
    generation that ended on a chain block rejected at its last slot -- `kv.length == len(ctx)`,
    so `put` accepts it -- stored a snapshot keyed by `len(ctx)` tokens whose recurrent state had
    seen one fewer and whose last KV row was the rejected draft's. The next turn resuming from it
    was conditioned on a context nobody wrote. The snapshot after a rollback must be the kept
    prefix, exactly: resuming it must equal forwarding that prefix cold."""
    eng = fresh()
    a, blk = tokens(20, seed=71), tokens(6, seed=72)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
        eng.forward_block(torch.tensor(blk), start=20)
        eng.rollback_to(3)
    assert eng.kv.length == 23, f"kv.length {eng.kv.length}: the kept prefix is 20 + 3"
    snap = cache.capture(eng)
    assert snap.length == 23
    warm, cold = fresh(), fresh()
    cache.restore(warm, snap)
    with torch.no_grad():
        x = warm.forward(torch.tensor([7]), start=23, last_only=True)
        cold.forward(torch.tensor(a + blk[:3]), start=0, last_only=True)
        y = cold.forward(torch.tensor([7]), start=23, last_only=True)
    assert torch.allclose(x, y, atol=1e-4, rtol=1e-4), (x - y).abs().max()

class _Arm:
    """A snapshottable drafter arm that says what its snapshot costs per token."""

    def __init__(self, per_token, block=8):
        self.per_token = per_token
        self.eng = type("E", (), {"tap": None})()
        self.cfg = type("C", (), {"block_size": block})()

    def state_snapshot(self):
        return ("arm",)

    def snapshot_bytes_per_token(self):
        return self.per_token


def test_the_snapshot_estimate_counts_every_arm_the_snapshot_carries():
    """The estimate added ONE arm's 20 kB a token, and the served length router snapshots
    two until the latch releases one. The drafter is asked, so 1-arm and 2-arm stacks both count."""
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.lenrouter import LengthRouter
    eng = fresh()
    kv = 2 * len(eng.cfg.attention_layers) * eng.cfg.num_key_value_heads * eng.cfg.head_dim * 2
    L = 1000

    def est(d):
        return cache._snapshot_estimate(eng, L, d)

    base = est(None)
    assert est(_Arm(21_000)) - base == int(1.25 * 21_000 * L), "one arm"
    router = LengthRouter(_Arm(21_000), _Arm(21_000, 16), latch=True, drop_idle=True,
                          learn_cost=False)
    assert est(router) - base == int(1.25 * 42_000 * L), "two arms until the latch releases one"
    router.idle = "s"
    assert est(router) - base == int(1.25 * 21_000 * L), "a released arm is not in the snapshot"
    # the real arm's figure is its own draft K and V per position
    arm = DFlash2Drafter.__new__(DFlash2Drafter)
    arm._ck = torch.zeros(5, 4, 64, 128, dtype=torch.bfloat16)
    assert arm.snapshot_bytes_per_token() == 2 * 5 * 4 * 128 * 2
    arm._ck = None
    assert arm.snapshot_bytes_per_token() == 0
    assert kv > 0

def test_a_single_snapshot_over_the_entry_cap_is_declined():
    """The 2026-09-19 wedge: a multi-GB boundary snapshot cloned before evictions run pushed the
    board to the memory edge. `put` must refuse an entry bigger than the cap the server sets."""
    eng = fresh()
    a = tokens(40, seed=51)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
    one = cache.capture(eng).nbytes
    store = cache.StateStore(int(one * 2.5), chunk=8, max_entry_bytes=one - 1)
    store.put(a, cache.capture(eng))
    r = store.report()
    assert r["entries"] == 0 and r["declined_big"] == 1
    ok = cache.StateStore(int(one * 2.5), chunk=8, max_entry_bytes=one)
    ok.put(a, cache.capture(eng))
    assert ok.report()["entries"] == 1, "an entry within the cap is accepted"


def test_the_recurrent_state_is_the_bulk_of_a_short_snapshot():
    """The sizing argument this cache is budgeted by, asserted rather than assumed."""
    eng = fresh()
    with torch.no_grad():
        eng.forward(torch.tensor(tokens(16, seed=41)), start=0, last_only=True)
    parts = cache.capture(eng).parts()
    assert parts["recurrent"] > parts["kv"]


# ------------------------------------------------------------------ 4. drafters

class _FakeDrafter:
    """A drafter with a position-indexed cache, which is the thing that must not get a hole."""

    def __init__(self):
        self.buf = torch.zeros(256, 4)
        self.ctx_len = 0

    def sync(self, tokens, hidden, first_pos, rows=None):
        n = len(tokens)
        self.buf[first_pos:first_pos + n] = torch.tensor(
            [[float(t)] * 4 for t in tokens])
        self.ctx_len = first_pos + n

    def state_snapshot(self):
        return ("fake", self.ctx_len, self.buf[:self.ctx_len].clone())

    def state_restore(self, snap):
        _, n, buf = snap
        self.buf[:n] = buf
        self.ctx_len = n


def test_the_drafter_cache_crosses_a_restore_without_a_hole():
    chunk = 8
    ids = tokens(40, seed=51)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng, d = fresh(), _FakeDrafter()
    cache.prefill(eng, d, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    want = d.buf[:40].clone()

    eng2, d2 = fresh(), _FakeDrafter()
    _, reused, _ = cache.prefill(eng2, d2, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    assert reused > 0
    assert d2.ctx_len == 40
    assert torch.equal(d2.buf[:40], want)


def test_a_drafter_that_cannot_snapshot_turns_the_cache_off():
    class NeedsHidden:
        def sync(self, *a, **k):
            pass

    class Lookup:
        def prime(self, toks):
            pass

    assert cache.drafter_is_cacheable(None)
    assert cache.drafter_is_cacheable(Lookup())
    assert cache.drafter_is_cacheable(_FakeDrafter())
    assert not cache.drafter_is_cacheable(NeedsHidden())


# ------------------------------------------------------------------ 5. response cache

def test_the_response_cache_keys_on_prompt_and_params():
    rc = cache.ResponseCache(1 << 20, ttl_s=60)
    ids = tokens(12, seed=61)
    k1 = cache.ResponseCache.key(ids, max_new=64, stops=())
    k2 = cache.ResponseCache.key(ids, max_new=128, stops=())
    k3 = cache.ResponseCache.key(ids[:-1] + [ids[-1] ^ 1], max_new=64, stops=())
    assert rc.get(k1) is None
    rc.put(k1, [1, 2, 3], ids)
    assert rc.get(k1) == [1, 2, 3]
    assert rc.get(k2) is None and rc.get(k3) is None
    assert rc.report()["hits"] == 1


def test_the_response_cache_expires_and_is_bounded():
    rc = cache.ResponseCache(1 << 20, ttl_s=1e-6)
    k = cache.ResponseCache.key([1, 2], max_new=8)
    rc.put(k, [9], [1, 2])
    # past the TTL before the read: on a fast CPU the read came within the microsecond and hit
    time.sleep(0.002)
    assert rc.get(k) is None and rc.report()["expired"] == 1

    rc = cache.ResponseCache(200, ttl_s=0)
    for i in range(20):
        rc.put(cache.ResponseCache.key([i], max_new=8), [i] * 4, [i])
    assert rc.bytes <= 200 and rc.report()["evictions"] > 0


# ------------------------------------------------------------------ 6. the persistent store

def test_the_suffix_store_finds_what_it_was_told_and_survives_a_restart():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "suffix")
        st = cache.PersistentSuffixStore(path, rebuild_every=1 << 30).open()
        seq = list(range(200, 260))
        st.append(seq * 3)
        st.rebuild(background=False)
        n, pos = st.lookup([200, 201, 202, 203, 204, 205, 206, 207], min_order=4)
        assert n == 8 and pos
        assert st.continuation(pos[0], 3) == [208, 209, 210]

        # a restart reads the log back and reindexes
        st2 = cache.PersistentSuffixStore(path, rebuild_every=1 << 30).open()
        n2, pos2 = st2.lookup([200, 201, 202, 203, 204, 205, 206, 207], min_order=4)
        assert n2 == 8 and pos2
        assert st2.report()["tokens"] == st.report()["tokens"]


def test_a_prefill_without_the_prefix_cache_is_still_chunked():
    """With the prefix cache off a long prompt used to be ONE forward of all its rows."""
    assert cache.prefill_chunk(False, 1024, 8192) == 8192
    assert cache.prefill_chunk(True, 1024, 8192) == 1024          # the cache's grid, unchanged
    assert cache.prefill_chunk(False, 1024, 0) == 0               # the old single call, on request


def test_a_readonly_store_reads_what_is_there_and_writes_nothing():
    """A benchmark measured against real traffic's store must not write itself into it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "suffix")
        st = cache.PersistentSuffixStore(path, rebuild_every=1 << 30).open()
        st.append(list(range(300, 320)))
        st.rebuild(background=False)
        size = os.path.getsize(os.path.join(path, "tokens.bin"))
        ro = cache.PersistentSuffixStore(path, rebuild_every=1, readonly=True).open()
        n, pos = ro.lookup(list(range(300, 308)), min_order=4)
        assert n == 8 and pos
        ro.append(list(range(900, 950)))
        ro.rebuild(background=False)
        assert os.path.getsize(os.path.join(path, "tokens.bin")) == size
        assert ro.lookup(list(range(900, 908)), min_order=4)[0] < 4


def test_a_continuation_stops_at_the_document_boundary():
    with tempfile.TemporaryDirectory() as d:
        st = cache.PersistentSuffixStore(os.path.join(d, "s"), rebuild_every=1 << 30).open()
        st.append([11, 12, 13, 14, 15, 16, 17, 18])
        st.append([50, 51, 52])
        st.rebuild(background=False)
        n, pos = st.lookup([11, 12, 13, 14, 15, 16, 17, 18], min_order=4)
        assert n == 8
        # the match ends the first document; nothing from the second may follow it
        assert st.continuation(pos[0], 4) == []


def test_the_store_set_asks_everyone_and_keeps_the_positions_apart():
    with tempfile.TemporaryDirectory() as d:
        a = cache.PersistentSuffixStore(os.path.join(d, "a"), rebuild_every=1 << 30).open()
        b = cache.PersistentSuffixStore(os.path.join(d, "b"), rebuild_every=1 << 30).open()
        a.append([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b.append([1, 2, 3, 4, 5, 6, 7, 8, 90, 91])
        a.rebuild(background=False)
        b.rebuild(background=False)
        s = cache.SuffixStoreSet([a, b])
        n, pos = s.lookup([1, 2, 3, 4, 5, 6, 7, 8], min_order=4)
        assert n == 8 and len(pos) == 2
        conts = sorted(s.continuation(p, 2) for p in pos)
        assert conts == [[9], [90, 91]], conts


def test_the_lookup_drafter_reads_the_persistent_store():
    from engine.drafters.ngram import NgramDrafter
    with tempfile.TemporaryDirectory() as d:
        st = cache.PersistentSuffixStore(os.path.join(d, "s"), rebuild_every=1 << 30).open()
        body = list(range(300, 340))
        st.append(body * 4)
        st.rebuild(background=False)
        drafter = NgramDrafter(corpus_path="", min_order=4, min_corpus_order=4, min_expected=0.2)
        drafter.add_store(st)
        drafter.prime([1, 2, 3])                       # nothing in the LOCAL index matches
        got = drafter.propose([300, 301, 302, 303, 304, 305, 306, 307], 4)
        assert got == [308, 309, 310, 311], got


def test_the_size_cap_forgets_the_oldest():
    with tempfile.TemporaryDirectory() as d:
        st = cache.PersistentSuffixStore(os.path.join(d, "s"), max_tokens=200,
                                         rebuild_every=1 << 30).open()
        for i in range(20):
            st.append(list(range(1000 + 40 * i, 1040 + 40 * i)))
        st.rebuild(background=False)
        assert st.report()["tokens"] <= 200
        assert st.stats["trims"] >= 1
        # the newest document is still findable
        n, pos = st.lookup(list(range(1760, 1768)), min_order=4)
        assert n == 8 and pos


# ------------------------------------------------------------------ SPD-249: a marked grid

def test_spd249_the_marked_grid_cuts_at_the_mark_then_every_chunk_from_it():
    g = cache.Grid(16, 23)
    assert g.mark == 23 and g.lines(0, 80) == [16, 23, 39, 55, 71], g.lines(0, 80)
    assert [g.next(i) for i in (0, 15, 16, 22, 23, 38, 39)] == [16, 16, 23, 23, 39, 39, 55]
    assert [i for i in range(80) if g.is_line(i)] == [0, 16, 23, 39, 55, 71]
    assert cache.Grid(16, 32).mark is None and cache.Grid(16, None).lines(0, 50) == [16, 32, 48]
    assert cache.Grid(16).next(17) == 32 and cache.Grid(16).is_line(48)
    # what a snapshot serves: its grid's lines, and a turn's end whatever the grid
    assert g.accepts(23, 23) and g.accepts(16, None) and not g.accepts(32, None)
    assert g.accepts(50, None, kind="session")
    plain = cache.Grid(16)
    assert plain.accepts(32, None) and plain.accepts(50, None) and not plain.accepts(23, 23)
    assert plain.accepts(32, 23)


def _cold_marked(ids, mark, chunk=16):
    eng = fresh()
    lg, reused, fwd = cache.prefill(eng, None, ids, DEV, store=None, chunk=chunk, grid_mark=mark)
    assert (reused, fwd) == (0, len(ids))
    return eng, lg


def test_spd249_a_chat_resumes_at_another_chats_head_bit_for_bit():
    """Two chats share a head of 23 rows (off the 16-row grid), then part. The first's prefill cuts at
    23 and snapshots it; the second resumes there and computes what its own cold prefill on the same
    grid computes, to the bit: logits, recurrent state, every KV row."""
    chunk, head = 16, tokens(23, seed=40)
    a, b = head + tokens(30, seed=41), head + tokens(45, seed=42)
    store = cache.StateStore(1 << 30, chunk=chunk)
    cache.prefill(fresh(), None, a, DEV, store=store, chunk=chunk, checkpoint=True, grid_mark=23)
    assert 23 in store.report()["boundaries"], store.report()["boundaries"]
    eng = fresh()
    warm, reused, fwd = cache.prefill(eng, None, b, DEV, store=store, chunk=chunk, checkpoint=True,
                                      grid_mark=23)
    assert (reused, fwd) == (23, 45), (reused, fwd)
    ref, cold = _cold_marked(b, 23)
    assert torch.equal(warm, cold), (warm - cold).abs().max().item()
    assert torch.equal(eng.state.S, ref.state.S) and torch.equal(eng.state.conv, ref.state.conv)
    n = len(b)
    assert torch.equal(eng.kv.k[:, :, :, :n, :], ref.kv.k[:, :, :, :n, :])
    assert torch.equal(eng.kv.v[:, :, :, :n, :], ref.kv.v[:, :, :, :n, :])
    # the same prompt again resumes at its last marked line (23 + 16 * 2 = 55), still exact
    eng2 = fresh()
    again, reused2, _ = cache.prefill(eng2, None, b, DEV, store=store, chunk=chunk, checkpoint=True,
                                      grid_mark=23)
    assert reused2 == 55 and torch.equal(again, cold), reused2


def test_spd249_grids_never_resume_from_each_others_lines():
    """A snapshot at the mark is no line of the 0-anchored grid, and a 0-grid snapshot past the mark
    is no line of the marked one: neither is taken, and each prefill stays its own cold one's bits.
    Below the mark both grids are the same lines and share."""
    chunk, ids = 16, tokens(60, seed=50)
    store = cache.StateStore(1 << 30, chunk=chunk)
    cache.prefill(fresh(), None, ids, DEV, store=store, chunk=chunk, checkpoint=True, grid_mark=23)
    eng = fresh()
    lg, reused, _ = cache.prefill(eng, None, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    assert reused == 16, reused                   # not 23, 39 or 55: the 0-grid's own line below
    _, ref = (lambda e: (e, cache.prefill(e, None, ids, DEV, chunk=chunk)[0]))(fresh())
    assert torch.equal(lg, ref)
    # the 0-grid prefill put 32 and 48: the marked one resumes at its own 55 and never at 48 or 32
    assert {32, 48} <= set(store.report()["boundaries"])
    eng2 = fresh()
    lg2, reused2, _ = cache.prefill(eng2, None, ids, DEV, store=store, chunk=chunk, checkpoint=True,
                                    grid_mark=23)
    _, cold = _cold_marked(ids, 23)
    assert reused2 == 55 and torch.equal(lg2, cold), reused2
    # a fresh store with 0-grid snapshots only: the marked prefill takes 16 (a line of both)
    s2 = cache.StateStore(1 << 30, chunk=chunk)
    cache.prefill(fresh(), None, ids, DEV, store=s2, chunk=chunk, checkpoint=True)
    lg3, reused3, _ = cache.prefill(fresh(), None, ids, DEV, store=s2, chunk=chunk, checkpoint=True,
                                    grid_mark=23)
    assert reused3 == 16 and torch.equal(lg3, cold), reused3


def test_spd249_the_resident_anchors_follow_the_marked_grid():
    """The rows in place: a marked prefill anchors at its lines; the next prompt on the 0-grid resumes
    only from an anchor on its own lines, and its logits are its cold prefill's."""
    chunk, ids = 16, tokens(60, seed=60)
    res = cache.ResidentPrefix(1 << 30, chunk, stash_bytes=1 << 30)
    eng = fresh()
    cache.prefill(eng, None, ids, DEV, chunk=chunk, resident=res, grid_mark=23)
    assert sorted(res.anchors) == [16, 23, 39, 55], sorted(res.anchors)
    assert res.grid is None, "the grid is the running prefill's only"
    longer = ids + tokens(10, seed=61)
    lg, reused, _ = cache.prefill(eng, None, longer, DEV, chunk=chunk, resident=res)
    assert reused == 16, reused
    ref = cache.prefill(fresh(), None, longer, DEV, chunk=chunk)[0]
    assert torch.equal(lg, ref)


def test_spd249_a_window_plan_cuts_on_the_marked_grid():
    class Eng:
        def prefill_window(self):
            pass

        def window_ok(self, i):
            return True
    plan = cache._window_plan(Eng(), 0, 70, 16, 64, cache.Grid(16, 23))
    assert plan == [(0, 16), (16, 7), (23, 16), (39, 16), (55, 15)][:len(plan)] and len(plan) >= 4, plan
    assert cache._window_plan(Eng(), 0, 70, 16, 64) == [(0, 16), (16, 16), (32, 16), (48, 16)]


def test_spd249_a_head_snapshot_is_evicted_last():
    """The store under pressure: turn ends put after a head's snapshot push out the oldest entries but
    the head's (the burst's next chats resume there); with nothing else left it goes too."""
    chunk, head = 16, tokens(23, seed=80)
    store = cache.StateStore(1 << 30, chunk=chunk)
    cache.prefill(fresh(), None, head + tokens(5, seed=81), DEV, store=store, chunk=chunk, checkpoint=True,
                  grid_mark=23)
    assert 23 in store.report()["boundaries"]
    snaps = []
    for k in range(3):
        e = fresh()
        ids = tokens(30 + k, seed=90 + k)
        cache.prefill(e, None, ids, DEV, chunk=chunk)
        snaps.append((ids, cache.capture(e)))
    one = snaps[0][1].nbytes
    store.budget = store.bytes + 2 * one                   # room for two more, not three
    for ids, snap in snaps:
        store.put(ids, snap, conv_id="s", kind="session")
    lines = store.report()["boundaries"]
    assert lines == [23, 31, 32], lines                     # the oldest went (16, then turn 30); the head stayed
    assert store.stats["evictions"] >= 1 and not store.stats.get("evictions_head")
    store.budget = one // 2                                 # nothing fits: the head goes as well
    store.reserve(0)
    assert 23 not in store.report()["boundaries"] and store.stats.get("evictions_head") == 1


def _head_snap(store, ids, mark, chunk=16):
    cache.prefill(fresh(), None, ids, DEV, store=store, chunk=chunk, checkpoint=True, grid_mark=mark)


def test_spd249_a_store_of_heads_alone_evicts_the_oldest_head_and_never_refuses():
    """Only heads in the store: room is made by the oldest head, every insert goes in, nothing stalls."""
    chunk = 16
    store = cache.StateStore(1 << 30, chunk=chunk)
    heads = [tokens(23, seed=100 + k) for k in range(4)]
    _head_snap(store, heads[0] + [5], 23)
    one = store.bytes // max(1, store.report()["entries"])
    store.budget = 2 * one + one // 2                  # two snapshots fit
    for h in heads[1:]:
        assert store.reserve(one)                      # never refused: the oldest head makes room
        _head_snap(store, h + [5], 23)
    keep = {tuple(e.tokens) for e in store._d.values() if e.mark is not None and len(e.tokens) == e.mark}
    assert keep <= {tuple(h) for h in heads[2:]} and keep, keep
    assert store.bytes <= store.budget and store.stats.get("evictions_head", 0) >= 2, store.stats


def test_spd249_a_replaced_head_loses_its_place():
    """The profile's notes changed: its new head shares the old one's first rows, the old one is stale
    and goes like any old entry (before a newer turn's end); the new head stays."""
    chunk = 16
    store = cache.StateStore(1 << 30, chunk=chunk)
    store.same_head_rows = 10
    base = tokens(12, seed=110)
    old, new = base + tokens(11, seed=111), base + tokens(11, seed=112)
    _head_snap(store, old + [5], 23)
    _head_snap(store, new + [5], 23)
    stale = [e for e in store._d.values() if len(e.tokens) == 23 and e.stale]
    assert len(stale) == 1 and stale[0].tokens == tuple(old) and store.stats["heads_stale"] == 1
    other = cache.StateStore(1 << 30, chunk=chunk)       # a head of another prompt: no shared rows
    _head_snap(other, tokens(23, seed=120) + [5], 23)
    _head_snap(other, tokens(23, seed=121) + [5], 23)
    assert not any(e.stale for e in other._d.values())
    e = fresh()
    turn = tokens(30, seed=113)
    cache.prefill(e, None, turn, DEV, chunk=chunk)
    one = cache.capture(e).nbytes
    head_new = next(x for x in store._d.values() if x.tokens == tuple(new))
    store.budget = head_new.snap.nbytes + one + one // 4   # room for the new head and the turn's end
    store.put(turn, cache.capture(e), conv_id="s", kind="session")
    left = {x.tokens for x in store._d.values()}
    assert tuple(old) not in left and tuple(new) in left and tuple(turn) in left, [len(t) for t in left]
    assert not store.stats.get("evictions_head"), "the stale head went as an old entry, not as a head"

def _plain_entries(store, n, seed=140):
    for k in range(n):
        e = fresh()
        ids = tokens(30 + k, seed=seed + k)
        cache.prefill(e, None, ids, DEV, chunk=16)
        store.put(ids, cache.capture(e), conv_id="s", kind="session")


def _heads(store):
    return [k for k, e in store._d.items() if store._head(k, e)]


def test_spd249_a_partial_shed_frees_what_is_needed_and_keeps_the_head():
    """The memory gate's store shed in part (`shed_to`): a need of X gives back at least X and at most X + the
    margin + one entry, in the store's order (plain entries oldest first): the head stays while plain ones are left."""
    store = cache.StateStore(1 << 30, chunk=16)
    _head_snap(store, tokens(23, seed=130) + [5], 23)
    _plain_entries(store, 3)
    sizes = [e.snap.nbytes for e in store._d.values()]
    store.shed_margin = min(sizes) // 2
    need = min(sizes) + 1
    before, heads = store.bytes, _heads(store)
    got = store.shed_to(False, need)
    assert need <= got["freed"] <= need + store.shed_margin + max(sizes), (got, need)
    assert before - store.bytes == got["freed"] and _heads(store) == heads and got["heads_kept"] == 1, got
    assert got["dropped"] >= 1 and not got["whole"] and not got["heads_dropped"], got
    return f"need {need}: {got['freed']} bytes in {got['dropped']} entries, the head kept"


def test_spd249_a_shed_past_the_plain_entries_drops_the_oldest_head():
    """More need than every plain entry holds: they all go, then the oldest head; the newer head stays."""
    store = cache.StateStore(1 << 30, chunk=16)
    _head_snap(store, tokens(23, seed=131) + [5], 23)
    _plain_entries(store, 2, seed=150)
    _head_snap(store, tokens(23, seed=132) + [5], 23)
    heads = _heads(store)
    assert len(heads) == 2, heads
    store.shed_margin = 0
    plain = sum(e.snap.nbytes for k, e in store._d.items() if k not in heads)
    got = store.shed_to(False, plain + 1)
    assert _heads(store) == heads[1:] and got["heads_dropped"] == 1 and got["heads_kept"] == 1, (got, _heads(store))
    assert all(k in heads for k in store._d), list(store._d)
    return "every plain entry, then the oldest head; the newer head kept"


def test_spd249_a_plain_only_chunk_never_takes_a_head():
    """The gate's first chunk (`heads=False`): plain entries only, however large the need; the heads stay."""
    store = cache.StateStore(1 << 30, chunk=16)
    _head_snap(store, tokens(23, seed=134) + [5], 23)
    _plain_entries(store, 2, seed=170)
    heads = _heads(store)
    got = store.shed_to(False, store.bytes * 4, heads=False)
    assert _heads(store) == heads and all(k in heads for k in store._d) and got["heads_dropped"] == 0, got
    assert not got["whole"] and got["heads_kept"] == len(heads), got
    return "every plain entry, no head"


class _StoreAlloc:
    """A reserve that is `base` + the store's bytes + `other` (what the next level holds): a store shed lowers it."""

    def __init__(self, store, base, other):
        self.store, self.base, self.other = store, base, other

    def probe(self):
        r = self.base + self.store.bytes + self.other
        return r, r


def _gate_over(store, over, other):
    """A memory gate whose reserve is `over` bytes past its ceiling, the store and an 'anchors' level of
    `other` bytes registered as the server registers them."""
    fa = _StoreAlloc(store, 1 << 30, other)
    ceiling = fa.base + store.bytes + other - over
    m = cache.Budget(ceiling, 1 << 50, 0, probe=fa.probe, empty=lambda: None, peak_probe=lambda reset=False: 0)
    m.register("store", store.shed)
    m.register_partial("store", store.shed_to)
    gone = []

    def anchors(safe=False):
        gone.append(fa.other)
        b, fa.other = fa.other, 0
        return b
    m.register("anchors", anchors)
    return m, gone


def test_spd249_a_head_the_store_cannot_close_the_need_with_stays_and_the_next_level_goes():
    """After the plain entries, the gate still needs 4 x the head's bytes: the head stays, the next level is shed
    (as a live gate once kept 0.20 GB of head against 0.81 GB still needed)."""
    store = cache.StateStore(1 << 30, chunk=16)
    _plain_entries(store, 2, seed=180)
    _head_snap(store, tokens(23, seed=181) + [5], 23)
    store.shed_margin = 0
    heads = _heads(store)
    head_b = sum(store._d[k].snap.nbytes for k in heads)
    plain_b = store.bytes - head_b
    m, gone = _gate_over(store, plain_b + 4 * head_b, other=8 * head_b)
    g = m.gate(10)
    assert _heads(store) == heads and store.bytes == head_b, (store.report(), heads)
    assert gone and g["shed"] == ["store", "anchors"], (gone, g)
    chunks = [e for e in m.events if e["kind"] == "shed_chunk"]
    assert [c["decision"] for c in chunks] == ["continue"], chunks
    return f"head {head_b} B kept against {4 * head_b} B still needed; the anchors went"


def test_spd249_heads_the_store_can_close_the_need_with_go_oldest_first():
    """Two heads; after the plain entries the gate needs half a head: the older head goes, the newer stays."""
    store = cache.StateStore(1 << 30, chunk=16)
    _head_snap(store, tokens(23, seed=182) + [5], 23)
    _plain_entries(store, 2, seed=183)
    _head_snap(store, tokens(23, seed=184) + [5], 23)
    store.shed_margin = 0
    heads = _heads(store)
    assert len(heads) == 2, heads
    head_b = store._d[heads[0]].snap.nbytes
    plain_b = store.bytes - sum(store._d[k].snap.nbytes for k in heads)
    m, gone = _gate_over(store, plain_b + head_b // 2, other=8 * head_b)
    g = m.gate(10)
    assert _heads(store) == heads[1:] and not gone and g["shed"] == ["store"], (_heads(store), gone, g)
    return "the older head went, the newer one stayed; no other level shed"


def test_spd249_a_store_of_heads_alone_is_shed_before_the_next_level():
    """Only heads in the store, the gate over by less than they hold: the plain-only first chunk frees nothing,
    so the heads are asked next; the oldest goes and no other level is shed (before: the store was skipped and
    the anchors went whole while the heads stayed)."""
    store = cache.StateStore(1 << 30, chunk=32)              # a 32 grid: the head's mark is its only line
    _head_snap(store, tokens(23, seed=185) + [5], 23, chunk=32)
    _head_snap(store, tokens(23, seed=186) + [5], 23, chunk=32)
    store.shed_margin = 0
    heads = _heads(store)
    assert len(heads) == 2 and store.bytes == sum(store._d[k].snap.nbytes for k in heads), heads
    head_b = store._d[heads[0]].snap.nbytes
    m, gone = _gate_over(store, head_b // 2, other=8 * head_b)
    g = m.gate(10)
    assert _heads(store) == heads[1:] and not gone and g["shed"] == ["store"], (_heads(store), gone, g)
    # a need no head can close: the heads stay and the next level goes, as before
    store2 = cache.StateStore(1 << 30, chunk=32)
    _head_snap(store2, tokens(23, seed=187) + [5], 23, chunk=32)
    store2.shed_margin = 0
    b2 = store2.bytes
    m2, gone2 = _gate_over(store2, 4 * b2, other=8 * b2)
    g2 = m2.gate(10)
    assert store2.bytes == b2 and gone2 and g2["shed"] == ["anchors"], (store2.report(), gone2, g2)
    return "the oldest head went, not the anchors; a head that cannot close the need stays"


class _StickyAlloc(_StoreAlloc):
    """A reserve that keeps the plain entries' freed bytes (the allocator holds their segments): only a head's
    free lowers it."""

    def __init__(self, store, base, other):
        super().__init__(store, base, other)
        self.plain0 = self._plain()

    def _plain(self):
        return sum(e.snap.nbytes for k, e in self.store._d.items() if not self.store._head(k, e))

    def probe(self):
        r = self.base + self.store.bytes + self.other + (self.plain0 - self._plain())
        return r, r


def test_spd249_a_chunk_too_small_to_tell_does_not_keep_the_heads():
    """A tiny plain entry and two heads; the plain chunk does not lower the reserve. Under PROOF_BYTES that says
    nothing, so the heads are asked next and the oldest goes; at PROOF_BYTES or more the stop holds: the heads stay
    and the next level goes."""
    def run(proof):
        store = cache.StateStore(1 << 30, chunk=32)          # one plain entry beside the two heads
        _head_snap(store, tokens(23, seed=188) + [5], 23, chunk=32)
        _plain_entries(store, 1, seed=189)
        _head_snap(store, tokens(23, seed=190) + [5], 23, chunk=32)
        store.shed_margin = 0
        heads = _heads(store)
        assert len(heads) == 2 and len(store._d) == 3, list(store._d)
        head_b = store._d[heads[0]].snap.nbytes
        plain_b = store.bytes - sum(store._d[k].snap.nbytes for k in heads)
        fa = _StickyAlloc(store, 1 << 30, 8 * head_b)
        m = cache.Budget(fa.base + store.bytes + fa.other - (head_b // 2 + 1), 1 << 50, 0, probe=fa.probe,
                         empty=lambda: None, peak_probe=lambda reset=False: 0)
        m.PROOF_BYTES = proof
        m.register("store", store.shed)
        m.register_partial("store", store.shed_to)
        gone = []

        def anchors(safe=False):
            gone.append(fa.other)
            b, fa.other = fa.other, 0
            return b
        m.register("anchors", anchors)
        g = m.gate(10)
        chunks = [e["decision"] for e in m.events if e["kind"] == "shed_chunk"]
        return store, heads, gone, g, chunks, plain_b
    store, heads, gone, g, chunks, plain_b = run(proof=1 << 40)
    assert _heads(store) == heads[1:] and not gone and g["shed"] == ["store"], (_heads(store), gone, g)
    assert chunks[0] == "continue: too small to tell", chunks
    store, heads, gone, g, chunks, plain_b = run(proof=1)
    assert _heads(store) == heads and gone and g["shed"] == ["store", "anchors"], (_heads(store), gone, g)
    assert chunks == ["stop: the reserve kept the bytes"], chunks
    return f"a {plain_b} B chunk: the heads asked under PROOF_BYTES, kept at or over it"


def test_spd249_a_shed_needing_the_whole_store_empties_it():
    """A need of the store's bytes or more: the store empties, as the full shed does."""
    store = cache.StateStore(1 << 30, chunk=16)
    _head_snap(store, tokens(23, seed=133) + [5], 23)
    _plain_entries(store, 2, seed=160)
    n, b = store.report()["entries"], store.bytes
    got = store.shed_to(False, b)
    assert got["whole"] and got["freed"] == b and got["dropped"] == n and store.bytes == 0 and not store._d, got
    assert store.shed(False) == 0
    return "the whole store, as before"


def test_srv773_resume_estimate_matches_the_prefills_choice():
    """SRV-773: the memory gate charges the rows a prefill forwards from where it resumes in place, estimated before
    any lock or restore (`ResidentPrefix.resume_estimate`): the end anchor over the last prompt's end over the anchors,
    with `_prefill`'s own calls (`end_offer`, `pend_offer`, `best` at `n - 1`), 0 with nothing shared; an end after
    images only when asked (`mm_ok`, which leaves the prompt's end out as `end_mm` does). Each estimate is checked
    against the prefill that follows it, and changes nothing of the prefix."""
    was = cache.PROMPT_END
    cache.PROMPT_END = True
    try:
        eng = fresh()
        res = cache.ResidentPrefix(1 << 30, 16, end=True)

        def state():
            return (list(res.tokens), res.valid, sorted(res.anchors), dict(res.end or {}), dict(res.pend or {}),
                    dict(res.stats), res.end_hold)

        def check(ids):
            before = state()
            est = res.resume_estimate(ids, len(ids))
            assert state() == before, "the estimate changes nothing"
            info: dict = {}
            cache.prefill(eng, None, ids, DEV, chunk=16, resident=res, info=info)
            assert est == (info["start"], info["kind"]), (est, info["start"], info["kind"])
            return est
        a = tokens(100, seed=21)
        assert check(a) == (0, None)                                   # cold: anchors 16..96, the end of the prompt
        assert sorted(res.anchors) == list(range(16, 97, 16)) and res.pend["at"] == 100, (res.anchors, res.pend)
        assert check(a + tokens(20, seed=22)) == (100, "pend")         # the prompt's end beats the anchor at 96
        b = list(res.tokens)
        assert check(b[:70] + [b[70] % 95 + 1] + tokens(29, seed=23)) == (64, "resident")   # parts at 70
        d = [b[0] % 95 + 1] + tokens(39, seed=24)
        assert check(d) == (0, None)                                   # nothing shared
        assert res.keep_end(eng, None, d, len(d)) == "kept" and res.end["at"] == 40
        nxt = d + tokens(9, seed=25)
        res.end["mm"] = True                                           # an end kept after images
        assert res.resume_estimate(nxt, len(nxt)) == (40, "pend"), "no end for a text prompt"
        assert res.resume_estimate(nxt, len(nxt), mm_ok=True) == (40, "end")
        res.end["mm"] = False
        assert check(nxt) == (40, "end")                               # the end over the prompt's end
    finally:
        cache.PROMPT_END = was
    return "pend 100, anchor 64, 0, end 40: each the prefill's own start and kind"


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
