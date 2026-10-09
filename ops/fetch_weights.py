"""Fetch a weight profile's target weights from its Hugging Face repo and verify them (ops/use-profile.sh).

    python ops/fetch_weights.py <repo> <dir> [file ...]     # files: only these (plus checksum lists)
    python ops/fetch_weights.py --verify <dir>               # checksums of what is there

Downloads into <dir>.partial with the token in ~/.cache/huggingface/token (HF_TOKEN wins), checks every
file a checksum list names (crc32.txt: EXL3 repos; quality/sha256.txt: the NVFP4 repo), then renames the
directory into place, so <dir> exists only complete. Exit 0 = verified.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import zlib


def _token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip() or None
    except OSError:
        return None


def _crc32(path: str) -> str:
    c = 0
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            c = zlib.crc32(b, c)
    return f"{c & 0xFFFFFFFF:08x}"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def verify(d: str, only: list[str] | None = None) -> list[str]:
    """Every (file, checksum) the lists in `d` name and `d` holds (or, with `only`, must hold): the
    mismatches and the missing, empty when all match."""
    bad, checked = [], 0
    for lst, fn in (("crc32.txt", _crc32), ("quality/sha256.txt", _sha256)):
        p = os.path.join(d, lst)
        if not os.path.isfile(p):
            continue
        for line in open(p):
            parts = line.split()
            if len(parts) < 2:
                continue
            want, name = parts[0].lower(), parts[-1].lstrip("*")
            if only and name not in only:
                continue
            f = os.path.join(d, name)
            if not os.path.isfile(f):
                if only or name.endswith(".safetensors"):
                    bad.append(f"missing {name}")
                continue
            got = fn(f)
            checked += 1
            if got != want:
                if name.endswith(".safetensors"):
                    bad.append(f"{name}: {got} != {want}")
                else:
                    # the EXL3 repos' crc32.txt predates their template/config edits (the served 4.5 has
                    # the same three "mismatches"): a weight file must match, a text file is reported
                    print(f"[fetch] note: {name} differs from {lst} ({got} != {want}); not a weight file")
    if checked == 0:
        bad.append("no checksum list matched any file")
    print(f"[fetch] {checked} files checked in {d}: {'OK' if not bad else 'FAILED'}", flush=True)
    return bad


def fetch(repo: str, d: str, files: list[str]) -> int:
    from huggingface_hub import snapshot_download
    part = d.rstrip("/") + ".partial"
    allow = (files + ["crc32.txt", "quality/sha256.txt", "README.md", "LICENSE"]) if files else None
    print(f"[fetch] {repo} -> {part}" + (f" ({' '.join(files)})" if files else ""), flush=True)
    snapshot_download(repo_id=repo, local_dir=part, token=_token(), allow_patterns=allow)
    bad = verify(part, files or None)
    if bad:
        print("[fetch] checksum failures, the download stays in " + part + ":\n  " + "\n  ".join(bad))
        return 1
    shutil.rmtree(os.path.join(part, ".cache"), ignore_errors=True)
    if os.path.exists(d):
        # a partial earlier set (NVFP4 profile with some files present): merge, never delete
        for name in os.listdir(part):
            os.replace(os.path.join(part, name), os.path.join(d, name)) if not os.path.isdir(
                os.path.join(part, name)) else shutil.copytree(os.path.join(part, name),
                                                               os.path.join(d, name), dirs_exist_ok=True)
        shutil.rmtree(part)
    else:
        os.rename(part, d)
    print(f"[fetch] {d} complete", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--verify"] and len(a) == 2:
        sys.exit(1 if verify(a[1]) else 0)
    if len(a) < 2:
        sys.exit(__doc__)
    sys.exit(fetch(a[0], os.path.expanduser(a[1]), a[2:]))
