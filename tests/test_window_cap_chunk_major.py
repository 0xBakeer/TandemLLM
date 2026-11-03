"""SRV-309 on a chunk-major prefill (a family with no layer-major window): the cap splits a chunk
into forwards of at most `cap` rows, and every anchor, disk capture and snapshot still fires on
the chunk grid only. A fake engine on CPU records the forwards' rows and where each line fires."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache as C  # noqa: E402

CHUNK, N = 4096, 10000


class _KV:
    length = 0


class _Eng:
    """Chunk-major: no `prefill_window`, so `_window_plan` gives no window."""
    def __init__(self):
        self.kv, self.rows = _KV(), []
        self.hidden_post_norm = torch.zeros(1, 1, 4)

    def reset(self):
        self.kv.length = 0

    def forward(self, blk, start, last_only=True):
        assert start == self.kv.length
        self.rows.append(len(blk))
        self.kv.length = start + len(blk)
        return torch.zeros(1, 8)


class _Res:
    def __init__(self):
        self.anchors = []

    def advanced(self, i):
        pass

    def anchor(self, eng, i):
        assert eng.kv.length == i
        self.anchors.append(i)


class _Store:
    max_entry = 1 << 40

    def __init__(self):
        self.snaps, self.stats = [], {}

    def reserve(self, est):
        return True

    def put(self, keys, snap, conv_id, hashes, mark=None):
        self.snaps.append(snap)


def _run(cap):
    eng, res, store, caps = _Eng(), _Res(), _Store(), []
    saved = (C.WINDOW_CAP, C._disk, C.capture, C._snapshot_estimate)

    def disk(d, op, *a, default=None):
        if op == "capture":
            caps.append(a[1].kv.length)
        return default
    C.WINDOW_CAP = (lambda: cap) if cap is not None else None
    C._disk, C.capture = disk, (lambda e, d, max_bytes=None: e.kv.length)
    C._snapshot_estimate = lambda e, i, d: 0
    try:
        ids = list(range(N))
        real = C._after_chunk

        def after(eng_, drafter, _res, _store, _disk, _job, ckpt, checkpoint, chunk, i, n, lim,
                  keys, hashes, conv_id, mem, grid=None):
            real(eng_, drafter, res, store, object(), "job", None, True, chunk, i, n, n,
                 keys, hashes, conv_id, None, grid)
        C._after_chunk = after
        try:
            C.prefill(eng, None, ids, "cpu", chunk=CHUNK)
        finally:
            C._after_chunk = real
    finally:
        C.WINDOW_CAP, C._disk, C.capture, C._snapshot_estimate = saved
    return eng.rows, res.anchors, caps, store.snaps


def test_no_cap_forwards_whole_chunks():
    rows, anchors, caps, snaps = _run(None)
    assert rows == [4096, 4096, 1808], rows
    assert anchors == caps == snaps == [4096, 8192], (anchors, caps, snaps)


def test_a_cap_splits_each_chunk_and_the_lines_stay_on_the_grid():
    rows, anchors, caps, snaps = _run(1024)
    assert rows == [1024] * 9 + [784], rows
    assert anchors == caps == snaps == [4096, 8192], (anchors, caps, snaps)


def test_a_cap_off_the_chunk_size_still_lands_on_every_line():
    # 1,500 does not divide 4,096: a step stops at the grid line before it stops at the cap
    rows, anchors, caps, snaps = _run(1500)
    assert sum(rows) == N and max(rows) <= 1500, rows
    ends = [sum(rows[:k + 1]) for k in range(len(rows))]
    assert 4096 in ends and 8192 in ends, ends
    assert anchors == caps == snaps == [4096, 8192], (anchors, caps, snaps)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
