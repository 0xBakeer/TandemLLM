"""Fetch a weight profile's target weights from its Hugging Face repo and verify them (ops/use-profile.sh).

    python ops/fetch_weights.py [--revision R] <repo> <dir> [file ...]   # files: only these (plus checksum lists)
    python ops/fetch_weights.py --verify <dir>                            # checksums of what is there

Downloads into <dir>.partial with the token in ~/.cache/huggingface/token (HF_TOKEN wins), checks every
file a checksum list names (crc32.txt: EXL3 repos; quality/sha256.txt: the NVFP4 repo), then renames the
directory into place, so <dir> exists only complete. Exit 0 = verified. --revision: a branch of the repo
(turboderp's Flash-Next packs, one branch a size, OPS-97), pinned to its commit at the start; such a
repo has no checksum list, so the hub's own (the sha256 of every LFS file, written to _hub_manifest.json
before the download) is the list, and a file a sibling pack beside <dir> already holds with the same
sha256 is hard-linked instead of downloaded (the n-gram table of the 4.05, 5.05 and 6.05 packs is one
file of 36 GiB).
"""
from __future__ import annotations

import glob
import hashlib
import json
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


MANIFEST = "_hub_manifest.json"


def verify(d: str, only: list[str] | None = None) -> list[str]:
    """Every (file, checksum) the lists in `d` name and `d` holds (or, with `only`, must hold): the
    mismatches and the missing, empty when all match."""
    bad, checked = [], 0
    mp = os.path.join(d, MANIFEST)
    if os.path.isfile(mp):
        for name, m in sorted(json.load(open(mp))["files"].items()):
            if not m.get("sha256") or (only and name not in only):
                continue
            f = os.path.join(d, name)
            if not os.path.isfile(f):
                bad.append(f"missing {name}")
                continue
            got = _sha256(f)
            checked += 1
            if got != m["sha256"]:
                bad.append(f"{name}: {got} != {m['sha256']}")
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


def manifest(repo: str, revision: str, part: str) -> str:
    """The hub's sizes and LFS sha256 of `revision`, written to `part`/_hub_manifest.json; its commit."""
    from huggingface_hub import HfApi
    info = HfApi(token=_token()).repo_info(repo, revision=revision, files_metadata=True)
    files = {s.rfilename: {"size": s.size, "sha256": s.lfs.sha256 if s.lfs else None} for s in info.siblings}
    os.makedirs(part, exist_ok=True)
    with open(os.path.join(part, MANIFEST), "w") as fh:
        json.dump({"repo": repo, "revision": revision, "commit": info.sha, "files": files}, fh, indent=1)
    return info.sha


def share(part: str, d: str) -> list[str]:
    """The files of `part`'s manifest that a sibling of `d` holds with the same sha256 and size (its own
    manifest says so), hard-linked into `part`; their names. A link that fails is downloaded."""
    mine = json.load(open(os.path.join(part, MANIFEST)))["files"]
    skip = {os.path.realpath(d), os.path.realpath(part)}
    out = []
    for sib in sorted(glob.glob(os.path.join(os.path.dirname(d.rstrip("/")), "*", MANIFEST))):
        sd = os.path.dirname(sib)
        if os.path.realpath(sd) in skip:
            continue
        theirs = json.load(open(sib))["files"]
        for name, m in mine.items():
            t, f, dst = theirs.get(name) or {}, os.path.join(sd, name), os.path.join(part, name)
            if name in out or not m.get("sha256") or t.get("sha256") != m["sha256"]:
                continue
            if not os.path.isfile(f) or os.path.getsize(f) != m["size"]:
                continue
            try:
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.link(f, dst)
            except OSError as exc:
                print(f"[fetch] {name}: no link from {sd} ({exc}); downloading it", flush=True)
                continue
            print(f"[fetch] {name}: the same file as {sd}'s, linked ({m['size'] / 1e9:.1f} GB not downloaded)",
                  flush=True)
            out.append(name)
    return out


def fetch(repo: str, d: str, files: list[str], revision: str | None = None) -> int:
    from huggingface_hub import snapshot_download
    part = d.rstrip("/") + ".partial"
    allow = (files + ["crc32.txt", "quality/sha256.txt", "README.md", "LICENSE"]) if files else None
    commit = manifest(repo, revision, part) if revision else None
    linked = share(part, d) if revision else []
    print(f"[fetch] {repo}" + (f"@{revision} ({commit})" if revision else "") + f" -> {part}"
          + (f" ({' '.join(files)})" if files else ""), flush=True)
    snapshot_download(repo_id=repo, revision=commit, local_dir=part, token=_token(), allow_patterns=allow,
                      ignore_patterns=linked or None)
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
    rev = None
    if a[:1] == ["--revision"] and len(a) >= 2:
        rev, a = a[1], a[2:]
    if len(a) < 2:
        sys.exit(__doc__)
    sys.exit(fetch(a[0], os.path.expanduser(a[1]), a[2:], rev))
