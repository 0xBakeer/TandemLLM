"""Drop the page cache of the engine's weight files once they are on the GPU.

The 2026-09-25 memory curve found what the 131k wedge's first NVRM lines were: with the engine
loaded, ~52 GB of the board's memory is the page cache of the weight files it read (the FP8
checkpoint, 29 GB, and the NVFP4 sets, 23 GB), MemFree is ~10 GB, and MemAvailable counts the
cache as free. An 8,192-row prefill chunk wants ~9 GB more from the allocator; MemFree went to 1.0 GB
while MemAvailable read 56 GB, and the driver logged `NVRM ... Out of memory [NV_ERR_NO_MEMORY]
... _memdescAllocInternal` -- the GPU's allocations do not wait for the kernel to reclaim that
cache. row3's MemGuard watches MemAvailable and cannot see it.

Nothing reads those files again once the weights are on the device, so the cache is dropped with
`posix_fadvise(DONTNEED)`, which any user may ask for on a file they can open (no root, unlike
`drop_caches`). Pages still mapped into a process are not dropped; a clean page is re-read from disk
if anything wants it later (the next load, which then reads the NVMe instead of RAM).

    python tools/drop_page_cache.py            # the default weight paths, prints GB dropped
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO = Path(__file__).resolve().parent.parent


def default_paths() -> list[Path]:
    """The files the engine reads at load and never again: the target checkpoint, the NVFP4 sets,
    the drafters. Not the suffix corpus, which the lookup drafter keeps reading."""
    return [HOME / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8", HOME / "nvfp4",
            REPO / "train/ft-b8-v2", REPO / "train/ft-b16"]


def _files(paths) -> list[str]:
    out = []
    for p in paths:
        p = str(p)
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for root, _dirs, names in os.walk(p, followlinks=True):
                out += [os.path.join(root, n) for n in names if os.path.isfile(os.path.join(root, n))]
    return sorted(set(os.path.realpath(f) for f in out))


def cached_gb(meminfo: str | None = None) -> float:
    text = meminfo if meminfo is not None else open("/proc/meminfo").read()
    for line in text.splitlines():
        if line.startswith("Cached:"):
            return int(line.split()[1]) * 1024 / 1e9
    return float("nan")


def drop(paths=None, advise=None) -> dict:
    """Fadvise(DONTNEED) every file under `paths`; returns the file count and their bytes."""
    advise = advise or (lambda fd: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED))
    files = _files(default_paths() if paths is None else paths)
    n, size, failed = 0, 0, 0
    for f in files:
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            failed += 1
            continue
        try:
            advise(fd)
            n += 1
            size += os.fstat(fd).st_size
        except OSError:
            failed += 1
        finally:
            os.close(fd)
    return {"files": n, "gb": size / 1e9, "failed": failed}


def main() -> None:
    before = cached_gb()
    r = drop([Path(p) for p in sys.argv[1:]] or None)
    print(f"[dropcache] {r['files']} files, {r['gb']:.1f} GB advised DONTNEED ({r['failed']} failed); "
          f"page cache {before:.1f} -> {cached_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
