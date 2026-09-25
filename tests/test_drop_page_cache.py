"""`tools/drop_page_cache.py` (SPD-18): which files it advises, and that it survives the odd one.

Run: python tests/test_drop_page_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.drop_page_cache import cached_gb, default_paths, drop  # noqa: E402


def _tree() -> str:
    d = tempfile.mkdtemp(prefix="dropcache-")
    os.makedirs(os.path.join(d, "hub/blobs"))
    os.makedirs(os.path.join(d, "hub/snapshots/abc"))
    open(os.path.join(d, "hub/blobs/1111"), "wb").write(b"x" * 1000)
    # an HF snapshot is symlinks into blobs: the blob is advised once, by its real path
    os.symlink("../../blobs/1111", os.path.join(d, "hub/snapshots/abc/model.safetensors"))
    open(os.path.join(d, "one.safetensors"), "wb").write(b"y" * 500)
    return d


def test_every_file_once_through_the_symlinks():
    d = _tree()
    seen = []
    r = drop([os.path.join(d, "hub"), os.path.join(d, "one.safetensors"), os.path.join(d, "missing")],
             advise=lambda fd: seen.append(os.readlink(f"/proc/self/fd/{fd}") if os.path.exists("/proc/self/fd") else fd))
    assert r["files"] == 2 and r["failed"] == 0, r
    assert abs(r["gb"] - 1500 / 1e9) < 1e-12, r
    assert len(seen) == 2, seen


def test_a_failing_advice_is_counted_not_raised():
    d = _tree()

    def bad(fd):
        raise OSError("EINVAL")
    r = drop([d], advise=bad)
    assert r["files"] == 0 and r["failed"] == 2, r


def test_the_defaults_leave_the_suffix_corpus_alone():
    names = [str(p) for p in default_paths()]
    assert any("Qwen3.8-27B-FP8" in n for n in names) and any(n.endswith("nvfp4") for n in names)
    assert not any("corpus" in n or "suffix" in n for n in names), names


def test_cached_is_read_in_gb():
    assert abs(cached_gb("MemTotal: 1 kB\nCached:  1048576 kB\n") - 1048576 * 1024 / 1e9) < 1e-9


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
