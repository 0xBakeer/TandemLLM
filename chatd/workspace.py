"""A chat's files: `<data>/workspaces/<conversation id>/`, mounted as `/workspace` in its sandbox.

chatd reads and writes them on the host, so a read, a write, a search or the files pane needs no
running container. The tree is the agent's, so nothing here trusts it:

- every path is resolved under the workspace root; `..`, an absolute path outside `/workspace`
  and a symlink anywhere on the way out of the root are refused (`PathError`, a 400);
- the last component is opened with O_NOFOLLOW, so a symlink swapped in between the check and the
  open is refused too;
- nothing here executes anything from the tree. git runs only inside the container: a `.git/config`
  the agent wrote (core.fsmonitor, hooks) would otherwise run on the host as chatd's user.

The `etag` of a file is its git blob hash (sha1 of `blob <size>\\0<bytes>`), so the editor's
If-Match and git agree on what "this version" is.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import stat

MOUNT = "/workspace"
MAX_READ = 2 * 1024 * 1024          # a read the tools hand the model at once (the pane streams)
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".tandem"}


class PathError(ValueError):
    pass


def etag(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Workspace:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)

    def ensure(self) -> None:
        os.makedirs(self.root, mode=0o755, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def rel(self, path: str) -> str:
        """`path` (relative, or absolute under /workspace) as a clean relative path; PathError
        for anything that leaves the root."""
        p = (path or "").strip()
        if p.startswith(MOUNT):
            p = p[len(MOUNT):]
        elif p.startswith("/"):
            raise PathError(f"{path}: outside {MOUNT}; the file tools read and write {MOUNT} only "
                            f"(copy the file there with bash first)")
        parts = []
        for seg in p.replace("\\", "/").split("/"):
            if seg in ("", "."):
                continue
            if seg == ".." or "\0" in seg:
                raise PathError(f"{path}: `..` is not allowed")
            parts.append(seg)
        return "/".join(parts)

    def resolve(self, path: str, must_exist: bool = False) -> str:
        """The host path; every existing component on the way must be a real directory or file,
        not a symlink (a symlink inside the tree may still point outside it)."""
        rel = self.rel(path)
        cur = self.root
        for seg in rel.split("/") if rel else []:
            cur = os.path.join(cur, seg)
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                if must_exist:
                    raise PathError(f"{path}: no such file") from None
                break
            if stat.S_ISLNK(st.st_mode):
                raise PathError(f"{path}: a symlink on the way is not followed")
        full = os.path.join(self.root, rel) if rel else self.root
        real = os.path.realpath(full)
        if real != self.root and not real.startswith(self.root + os.sep):
            raise PathError(f"{path}: outside {MOUNT}; the file tools read and write {MOUNT} only "
                            f"(copy the file there with bash first)")
        return full

    def show(self, full: str) -> str:
        """A host path as the agent sees it."""
        rel = os.path.relpath(full, self.root)
        return MOUNT if rel == "." else f"{MOUNT}/{rel}"

    # ------------------------------------------------------------------ files
    def read(self, path: str, limit: int | None = None) -> bytes:
        full = self.resolve(path, must_exist=True)
        fd = os.open(full, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as fh:
            return fh.read(limit) if limit else fh.read()

    def stat(self, path: str) -> os.stat_result:
        return os.lstat(self.resolve(path, must_exist=True))

    def write(self, path: str, data: bytes, if_match: str | None = None) -> str:
        """Create or replace; `if_match`: the etag the caller read (a 412 when it moved)."""
        full = self.resolve(path)
        if os.path.isdir(full):
            raise PathError(f"{path}: is a directory")
        if if_match is not None and os.path.exists(full):
            if etag(self.read(path)) != if_match:
                raise ChangedError(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        self.resolve(os.path.dirname(self.rel(path)) or ".")      # the parents made are ours
        tmp = full + f".tandem-{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, full)
        return etag(data)

    def delete(self, path: str) -> None:
        full = self.resolve(path, must_exist=True)
        if full == self.root:
            raise PathError("the workspace itself is not deleted")
        if os.path.isdir(full) and not os.path.islink(full):
            import shutil
            shutil.rmtree(full)
        else:
            os.unlink(full)

    def list(self, path: str = MOUNT) -> list[dict]:
        full = self.resolve(path, must_exist=True)
        out = []
        with os.scandir(full) as it:
            for e in it:
                st = e.stat(follow_symlinks=False)
                kind = "dir" if stat.S_ISDIR(st.st_mode) else "file"
                if stat.S_ISLNK(st.st_mode):
                    kind = "file"
                out.append({"path": self.show(e.path), "kind": kind,
                            "bytes": 0 if kind == "dir" else int(st.st_size),
                            "modified_at": _iso(st.st_mtime)})
        return sorted(out, key=lambda f: (f["kind"] != "dir", f["path"]))

    def size(self) -> int:
        total = 0
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            for f in filenames:
                try:
                    total += os.lstat(os.path.join(dirpath, f)).st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------ search
    def walk(self, start: str = MOUNT):
        base = self.resolve(start, must_exist=True)
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS
                                 and not os.path.islink(os.path.join(dirpath, d)))
            for f in sorted(filenames):
                p = os.path.join(dirpath, f)
                if not os.path.islink(p):
                    yield p

    def glob(self, pattern: str, start: str = MOUNT, limit: int = 500) -> list[str]:
        pat = pattern.lstrip("/")
        if pat.startswith("workspace/"):
            pat = pat[len("workspace/"):]
        rx = _glob_rx(pat)
        out = []
        for p in self.walk(start):
            rel = os.path.relpath(p, self.root)
            if rx.fullmatch(rel) or ("/" not in pat and fnmatch.fnmatch(os.path.basename(p), pat)):
                out.append(self.show(p))
                if len(out) >= limit:
                    break
        return out

    def grep(self, pattern: str, start: str = MOUNT, include: str | None = None,
             ignore_case: bool = False, limit: int = 200) -> list[str]:
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as exc:
            raise PathError(f"bad pattern: {exc}") from None
        out = []
        for p in self.walk(start):
            if include and not fnmatch.fnmatch(os.path.basename(p), include):
                continue
            try:
                if os.lstat(p).st_size > MAX_READ:
                    continue
                with open(p, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if b"\0" in data[:4096]:
                continue
            for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{self.show(p)}:{n}: {line[:300]}")
                    if len(out) >= limit:
                        return out
        return out


class ChangedError(Exception):
    """The file moved on since the caller read it (the editor's 412)."""


def _glob_rx(pat: str) -> re.Pattern:
    """`**` crosses directories, `*` and `?` do not."""
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pat.startswith("**", i):
            out += ".*"
            i += 2
        elif pat[i] == "*":
            out += "[^/]*"
            i += 1
        elif pat[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pat[i])
            i += 1
    return re.compile(out)


def _iso(t: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
