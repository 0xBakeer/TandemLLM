"""The skinny CUDA kernel's routing and its tile table, without a board.

The kernel's own arithmetic is checked on the board by `tools/nvfp4_skinny.py --check` (against the
exact product and against v2, and row independence at every row count). What can be checked here is
the part that would be a silent correctness bug rather than a slow kernel: the losslessness gate
compares a greedy run decoding at M = 1 with a speculative one verifying 8 or 16 rows, so every
decode-side row count must land on the same kernel with the same K split, and the split must be a
function of the weight's shape alone.
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
          (12288, 5120), (1024, 5120), (34816, 5120), (14336, 5120), (16384, 5120)]


def _reload(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from tools import nvfp4_skinny
        return importlib.reload(nvfp4_skinny)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_off_by_default_and_the_switch_switches():
    sk = _reload()
    assert not any(sk.use_skinny(m) for m in (1, 8, 16, 32)), "on without the flag"
    sk = _reload(QWEN38_NVFP4_SKINNY="1")
    assert all(sk.use_skinny(m) for m in (1, 2, 8, 15, 16, 17, 32)), "flag on, a row count off"
    assert not sk.use_skinny(33) and not sk.use_skinny(0), "outside 1..32"
    _reload()
    return "off by default; on for 1..32 with QWEN38_NVFP4_SKINNY=1, never above"


def test_one_row_and_a_verified_block_take_the_same_kernel():
    from tools import nvfp4_linear
    sk = _reload(QWEN38_NVFP4_SKINNY="1")
    try:
        routed = {m: nvfp4_linear.use_skinny(m) for m in (1, 2, 4, 8, 11, 14, 15, 16, 24, 32)}
        assert set(routed.values()) == {True}, routed
    finally:
        _reload()
    return "M in 1..32 all on the skinny kernel when it is on"


def test_the_tile_is_a_function_of_the_shape_only():
    sk = _reload()
    import inspect
    assert list(inspect.signature(sk.pick).parameters) == ["N", "K"], \
        "pick() takes a row count: the K split would depend on M and break row independence"
    for (n, k) in SHAPES:
        cfg = sk.pick(n, k)
        assert set(cfg) == {"nt", "wk", "pf"}, cfg
        assert cfg["nt"] in (4, 8, 16) and cfg["wk"] in (1, 2, 4, 8) and cfg["pf"] in (0, 1), cfg
    return f"{len(SHAPES)} shapes, complete configs, no row-count argument"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
