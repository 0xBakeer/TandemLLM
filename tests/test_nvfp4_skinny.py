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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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


def test_programmatic_dependent_launch_is_off_by_default():
    """SPD-30: QWEN38_SKINNY_PDL, and the norms release their dependents only when it is on."""
    sk = _reload()
    from tools import norm_kernels
    assert not sk.PDL and not norm_kernels._pdl()
    sk = _reload(QWEN38_NVFP4_SKINNY="1", QWEN38_SKINNY_PDL="1")
    try:
        assert sk.PDL and norm_kernels._pdl()
    finally:
        _reload()
    return "off by default; on only with the skinny kernel on"


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
    for table in (None, os.path.join(ROOT, "ops/skinny-tiles.json")):
        sk = _reload(**({"QWEN38_SKINNY_TILES": table} if table else {}))
        for (n, k) in SHAPES:
            cfg = sk.pick(n, k)
            assert set(cfg) - {"il"} == {"nt", "wk", "pf", "minb"}, cfg
            assert cfg["nt"] in (1, 2, 4, 8, 16) and cfg["wk"] in (1, 2, 4, 8, 16), cfg
            assert cfg["pf"] in (0, 1, 2) and cfg["minb"] in (1, 2), cfg
            # 512 threads at two CTAs an SM exists only for the 8-row tile (SPD-33)
            assert not (cfg["wk"] == 16 and cfg["minb"] == 2 and cfg["nt"] != 1), cfg
            assert cfg.get("il", 0) in (0, 1), cfg
    _reload()
    return f"{len(SHAPES)} shapes, complete configs, no row-count argument, served table included"


def test_the_second_table_is_off_until_the_switch_and_names_only_its_shapes():
    """SPD-33's in-process A/B: QWEN38_SKINNY_TILES_B loads a second table and `ALT` routes the
    shapes it names through it; every other shape, and every shape with the switch off, keeps the
    first table's tile."""
    import json
    import tempfile
    alt = os.path.join(tempfile.mkdtemp(), "alt.json")
    json.dump({"5120x17408": {"nt": 1, "wk": 16, "pf": 2, "il": 1}}, open(alt, "w"))
    first = os.path.join(ROOT, "ops/skinny-tiles.json")
    sk = _reload(QWEN38_SKINNY_TILES=first, QWEN38_SKINNY_TILES_B=alt)
    try:
        before = {shp: sk.pick(*shp) for shp in SHAPES}
        assert not sk.ALT
        sk.ALT = True
        assert sk.pick(5120, 17408) == {"nt": 1, "wk": 16, "pf": 2, "minb": 1, "il": 1}
        assert all(sk.pick(*shp) == before[shp] for shp in SHAPES if shp != (5120, 17408))
        sk.ALT = False
        assert all(sk.pick(*shp) == before[shp] for shp in SHAPES)
    finally:
        _reload()
    return "ALT off = the first table; on = the second table's shapes only"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
