"""The allocator's counters without cyclic garbage (decode 46 -> 23 t/s in 25 minutes on 0.15.1).

torch 2.13's `torch.cuda.memory_stats` flattens the allocator's nested dict through a closure that refers
to itself, so every call (and `memory_allocated`, `memory_reserved`, `max_memory_allocated`, which call it)
leaves its (key, value) tuples, ~4K with the graphs' pools, in a reference cycle. The server read them
about twelve times a decode round; with QSE_RUNNER_GC the young collection at a round's end found 50-80K
unreachable objects (40-80 ms a collection). `engine.cache.cuda_stats` reads the nested dict instead:

  1  a dotted name resolves to its counter, a missing one to 0       test_cuda_stats_reads_the_nested_dict
  2  torch's flattening call leaves cyclic garbage; `cuda_stats` none
     (the flat call measured, so a torch that stops leaking says so)  test_no_cyclic_garbage
  3  engine/ and server/ never call the flattening readers            test_no_flattening_reader_in_the_tree
"""
import gc
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from engine import cache

NESTED = {"allocated_bytes": {"all": {"current": 7, "peak": 11}, "small_pool": {"current": 1}},
          "reserved_bytes": {"all": {"current": 13}}, "num_ooms": 0}


def _flat_like_torch(stats: dict) -> dict:
    """torch.cuda.memory_stats' own flattening (torch 2.13), over a given nested dict."""
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if prefix:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + str(k), v)
        else:
            result.append((prefix, obj))

    _recurse_add_to_result("", stats)
    result.sort()
    return dict(result)


def test_cuda_stats_reads_the_nested_dict():
    saved = torch.cuda.memory_stats_as_nested_dict
    torch.cuda.memory_stats_as_nested_dict = lambda *a, **k: NESTED
    try:
        assert cache.cuda_stats("allocated_bytes.all.current", "reserved_bytes.all.current") == (7, 13)
        assert cache.cuda_stats("allocated_bytes.all.peak") == (11,)
        assert cache.cuda_stats("num_ooms", "nope.all.current", "allocated_bytes.all") == (0, 0, 0)
    finally:
        torch.cuda.memory_stats_as_nested_dict = saved


def test_no_cyclic_garbage():
    saved = torch.cuda.memory_stats_as_nested_dict
    torch.cuda.memory_stats_as_nested_dict = lambda *a, **k: {"a": dict(NESTED), "b": dict(NESTED)}
    was = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        for _ in range(50):
            _flat_like_torch(torch.cuda.memory_stats_as_nested_dict())
        flat = gc.collect()
        for _ in range(50):
            cache.cuda_stats("a.allocated_bytes.all.current", "b.reserved_bytes.all.current")
        ours = gc.collect()
    finally:
        torch.cuda.memory_stats_as_nested_dict = saved
        if was:
            gc.enable()
    assert flat > 50, flat
    assert ours == 0, ours


READERS = re.compile(r"torch\.cuda\.(memory_stats|memory_allocated|memory_reserved|max_memory_allocated|"
                     r"max_memory_reserved)\(")


def test_no_flattening_reader_in_the_tree():
    bad = []
    for top in ("engine", "server"):
        for p in (ROOT / top).rglob("*.py"):
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                code = line.split("#", 1)[0]
                if READERS.search(code) and "reset_peak" not in code:
                    bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
