"""SPD-239: a joint pass reads every state's PLE rows in one batch a layer (`FlashNextEngine._ple_batch`).

The staging's PLE part on the tiny random Flash-Next's n-gram hashes (tests/fn_tiny.py) over a table that
counts its reads: the rows and head indices a block stages from the joint batch are the ones it stages
reading its own rows, and those are the ones a row-by-row read gives; one batch a layer reads every state's
rows, where a block at a time read once a block.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Table:
    """Packed rows that are a function of the id, its reads counted."""

    WORDS = 4

    def __init__(self, sizes):
        self.offsets_np = np.r_[0, np.cumsum(sizes)][:-1].astype(np.int64)
        self.calls = 0

    def rows_of(self, ids):
        self.calls += 1
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        return ((ids[:, None] * 7 + np.arange(self.WORDS)[None, :]) % 30011).astype(np.int16)


def _engine():
    from fn_tiny import tiny
    from engine.models.flashnext.engine import FlashNextEngine
    from engine.models.flashnext.ref import FlashNextRef
    cfg, W, tables = tiny(seed=5)
    ref = FlashNextRef(cfg, W, tables, dtype=torch.float32)
    eng = FlashNextEngine.__new__(FlashNextEngine)
    eng.cfg, eng.hash = cfg, ref.hash
    eng.tables = {li: _Table([1 << 20] * cfg.ngram_heads) for li in cfg.ple_layers}
    return eng


def _bufs(eng, M):
    nh = eng.cfg.ngram_heads
    return {"packed": torch.zeros(M * nh, _Table.WORDS, dtype=torch.int16),
            "heads": torch.zeros(M * nh, dtype=torch.int64)}


def _blocks(eng):
    g = torch.Generator().manual_seed(9)
    out = []
    for j, parents in enumerate(([-1, 0, 1], [-1, 0, 0, 2], [-1], [-1, 0, 1, 2, 3])):
        toks = torch.randint(0, eng.cfg.vocab_size, (len(parents),), generator=g).tolist()
        ctx = {} if j == 2 else {li: torch.randint(0, eng.cfg.vocab_size, (1, eng.hash[li].context_len), generator=g)
                                 for li in eng.cfg.ple_layers}
        out.append((toks, parents, types.SimpleNamespace(ple_ctx=ctx)))
    return out


def test_the_joint_batch_stages_each_blocks_own_rows():
    eng = _engine()
    items = _blocks(eng)
    ples = eng._ple_batch(items)
    calls = {li: t.calls for li, t in eng.tables.items()}
    assert all(c == 1 for c in calls.values()), calls             # one read a layer for every state
    for (toks, parents, st), ple in zip(items, ples):
        _, paths = eng._tree_paths(parents)
        for li in eng.cfg.ple_layers:
            a, b, c = _bufs(eng, len(toks)), _bufs(eng, len(toks)), _bufs(eng, len(toks))
            eng._ple_fill(a, 0, li, *ple[li])
            ids = eng.path_ple_ids(eng.hash[li], eng._ple_ctx(st, li), toks, paths)
            eng._ple_fill(b, 0, li, ids)                          # the block's own read
            for i, row in enumerate(ids):
                eng._ple_fill(c, i, li, row)                      # a read a row (before SPD-239)
            for k in ("packed", "heads"):
                assert torch.equal(a[k], b[k]) and torch.equal(b[k], c[k]), (li, k, parents)
    return f"{len(items)} blocks of 1-5 rows: one read a PLE layer, each block's rows and heads as read alone"


def test_tree_paths_are_the_stagings():
    from engine.models.flashnext.engine import FlashNextEngine
    depth, paths = FlashNextEngine._tree_paths([-1, 0, 0, 2, 3, 1])
    assert depth == [0, 1, 1, 2, 3, 2] and paths == [[0], [0, 1], [0, 2], [0, 2, 3], [0, 2, 3, 4], [0, 1, 5]]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
