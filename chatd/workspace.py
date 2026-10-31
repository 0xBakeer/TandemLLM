"""A chat's files: `<data>/workspaces/<conversation id>/`, mounted as `/workspace` in its sandbox.

chatd reads and writes them on the host, so a read, a write, a search or the files pane needs no
running container. The tree is the agent's, and the agent changes it while chatd works in it, so
nothing here trusts it (SEC-1):

- every path is resolved under the workspace root; `..` and an absolute path outside `/workspace`
  are refused (`PathError`, a 400);
- every access walks from the root one directory at a time, each opened with O_NOFOLLOW relative
  to the one before (`SandboxDir`). A symlink anywhere on the way is refused, and a directory the
  sandbox swaps for a symlink after a check cannot redirect the access: there is no check apart
  from the open itself. A file is opened with O_NOFOLLOW and O_NONBLOCK and read only when its
  fstat says it is a regular file (a FIFO would block chatd);
- a write goes to a new temporary file (O_EXCL) in the verified directory and is renamed over the
  name, so it never writes into a file that is already there;
- nothing here executes anything from the tree. git runs only inside the container: a `.git/config`
  the agent wrote (core.fsmonitor, hooks) would otherwise run on the host as chatd's user.

The sandbox's home (`<data>/homes/<id>`, /home/agent) is a `SandboxDir` too (chatd/owui.py).

The `etag` of a file is its git blob hash (sha1 of `blob <size>\\0<bytes>`), so the editor's
If-Match and git agree on what "this version" is.
"""

from __future__ import annotations

import codecs
import errno
import fnmatch
import hashlib
import os
import re
import secrets
import shutil
import stat
import time

MOUNT = "/workspace"
MAX_READ = 2 * 1024 * 1024          # a read the tools hand the model at once (the pane streams)
GREP_LINE_MAX = 64 * 1024           # grep searches a line's first 64 K chars, drops the rest
GREP_FILE_MAX = 16 * 1024 * 1024    # grep scans at most 16 MiB of a file (8x MAX_READ)
GREP_CHUNK = 1024 * 1024            # grep reads a MiB at a time: one line pass per chunk
_GREP_EOL = re.compile(r"\r\n|\r|\n|\x0b|\x0c|\x1c|\x1d|\x1e|\x85|\u2028|\u2029")
# a pure-'\n' chunk -- nothing but '\n' among the ASCII control separators, and none
# of the three Unicode boundaries either -- partitions identically under str.split('\n')
# (C-fast, like main's splitlines) and the general _GREP_EOL.split. Six memchr finds
# answer the ASCII half on the raw bytes; the Unicode half is probed on the DECODED
# head, because in the bytes E2/80/A8 can sit across the chunk seam. An exotic chunk
# takes the general path; a chunk boundary cannot fabricate an ASCII control byte.
_GREP_CTL = (b"\r", b"\x0b", b"\x0c", b"\x1c", b"\x1d", b"\x1e")
_GREP_HIGH = ("\u0085", "\u2028", "\u2029")   # _GREP_EOL's other boundaries
GREP_BUDGET_S = 10                  # one grep call gets 10 s across all files; the
                                    # budget is checked BETWEEN chunks, so a single
                                    # catastrophic search can run past it (^(a+)+$ on
                                    # 26 a's and a ! took 1.7 s, and every extra a
                                    # doubles that) -- the overrun is one search
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".tandem"}

_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class PathError(ValueError):
    pass


def etag(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class SandboxDir:
    """A folder the sandbox writes into, seen from the host: `rel` paths under `root` ("a/b.txt"),
    every one reached through directory fds opened with O_NOFOLLOW. `mount` is where the sandbox
    sees the root; messages name paths that way."""

    def __init__(self, root: str, mount: str):
        self.root = os.path.realpath(root)
        self.mount = mount

    def ensure(self) -> None:
        os.makedirs(self.root, mode=0o755, exist_ok=True)

    def show(self, rel: str) -> str:
        return f"{self.mount}/{rel}" if rel else self.mount

    def _parts(self, rel: str) -> list[str]:
        parts = [s for s in rel.split("/") if s not in ("", ".")]
        if any(s == ".." or "\0" in s for s in parts):
            raise PathError(f"{self.show(rel)}: `..` is not allowed")
        return parts

    def _refused(self, dfd: int, seg: str, label: str) -> PathError:
        try:
            link = stat.S_ISLNK(os.stat(seg, dir_fd=dfd, follow_symlinks=False).st_mode)
        except OSError:
            link = False
        return PathError(f"{label}: a symlink on the way is not followed" if link else
                         f"{label}: {seg} is not a directory")

    def _dir(self, parts: list[str], label: str, make: bool = False) -> int:
        """An fd of the folder `parts`, each step opened from the one before with O_NOFOLLOW (and
        made when `make`); the caller closes it."""
        fd = os.open(self.root, _DIR)
        try:
            for seg in parts:
                try:
                    try:
                        nxt = os.open(seg, _DIR, dir_fd=fd)
                    except FileNotFoundError:
                        if not make:
                            raise PathError(f"{label}: no such file") from None
                        try:
                            os.mkdir(seg, 0o755, dir_fd=fd)
                        except FileExistsError:
                            pass
                        nxt = os.open(seg, _DIR, dir_fd=fd)
                except OSError as exc:
                    if exc.errno in (errno.ENOTDIR, errno.ELOOP, errno.EMLINK):
                        raise self._refused(fd, seg, label) from None
                    raise
                os.close(fd)
                fd = nxt
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _open_in(self, dfd: int, name: str, label: str):
        """(fd, fstat) of the regular file `name` in the folder `dfd`."""
        try:
            fd = os.open(name, _FILE, dir_fd=dfd)
        except FileNotFoundError:
            raise PathError(f"{label}: no such file") from None
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise PathError(f"{label}: a symlink on the way is not followed") from None
            raise
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise PathError(f"{label}: is a directory" if stat.S_ISDIR(st.st_mode) else
                            f"{label}: not a file")
        return fd, st

    def open(self, rel: str, label: str | None = None):
        """(binary file object, its fstat) of a regular file; the caller closes it. The stat is
        the open file's, so a file replaced after the open is read as it was."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: is a directory")
        dfd = self._dir(parts[:-1], label)
        try:
            fd, st = self._open_in(dfd, parts[-1], label)
        finally:
            os.close(dfd)
        return os.fdopen(fd, "rb"), st

    def read(self, rel: str, limit: int | None = None, label: str | None = None) -> bytes:
        fh, _ = self.open(rel, label)
        with fh:
            return fh.read(limit) if limit else fh.read()

    def stat(self, rel: str, label: str | None = None) -> os.stat_result:
        """The lstat of `rel` (a symlink at the end is itself)."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            return os.stat(self.root)
        dfd = self._dir(parts[:-1], label)
        try:
            return os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
        except FileNotFoundError:
            raise PathError(f"{label}: no such file") from None
        finally:
            os.close(dfd)

    def write(self, rel: str, data: bytes, mode: int = 0o644, label: str | None = None,
              over_link: bool = False, if_etag: str | None = None) -> None:
        """Create or replace `rel` atomically (its folders made as needed): the bytes go to a new
        temporary file in the verified folder, renamed over the name. A symlink at the name is
        refused, or replaced itself with `over_link`; `if_etag`: the etag the file must still have
        (ChangedError)."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: is a directory")
        name = parts[-1]
        dfd = self._dir(parts[:-1], label, make=True)
        try:
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                st = None
            if st is not None and stat.S_ISDIR(st.st_mode):
                raise PathError(f"{label}: is a directory")
            if st is not None and stat.S_ISLNK(st.st_mode) and not over_link:
                raise PathError(f"{label}: a symlink on the way is not followed")
            if if_etag is not None and st is not None:
                fd, _ = self._open_in(dfd, name, label)
                with os.fdopen(fd, "rb") as fh:
                    if etag(fh.read()) != if_etag:
                        raise ChangedError(label)
            tmp = f".tandem-{secrets.token_hex(6)}.tmp"
            fd = os.open(tmp, _NEW, mode, dir_fd=dfd)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
            except BaseException:
                try:
                    os.unlink(tmp, dir_fd=dfd)
                except OSError:
                    pass
                raise
        finally:
            os.close(dfd)

    def create(self, rel: str, data: bytes, mode: int = 0o644, label: str | None = None) -> None:
        """A new file at `rel` (its folders made as needed); FileExistsError when the name is
        taken by anything, a dangling symlink included."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: is a directory")
        dfd = self._dir(parts[:-1], label, make=True)
        try:
            fd = os.open(parts[-1], _NEW, mode, dir_fd=dfd)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except BaseException:
                try:
                    os.unlink(parts[-1], dir_fd=dfd)
                except OSError:
                    pass
                raise
        finally:
            os.close(dfd)

    def delete(self, rel: str, label: str | None = None) -> None:
        """A file, a symlink (itself) or a folder with everything in it."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: the workspace itself is not deleted")
        dfd = self._dir(parts[:-1], label)
        try:
            try:
                st = os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                raise PathError(f"{label}: no such file") from None
            if stat.S_ISDIR(st.st_mode):
                shutil.rmtree(parts[-1], dir_fd=dfd)
            else:
                os.unlink(parts[-1], dir_fd=dfd)
        finally:
            os.close(dfd)

    def scan(self, rel: str, label: str | None = None) -> list[tuple[str, os.stat_result]]:
        """(name, lstat) of each entry of the folder `rel`."""
        dfd = self._dir(self._parts(rel), label or self.show(rel))
        try:
            with os.scandir(dfd) as it:
                return [(e.name, e.stat(follow_symlinks=False)) for e in it]
        finally:
            os.close(dfd)

    def files(self, rel: str = "", skip=()):
        """The rel path of every regular file under `rel` (sorted, folders in `skip` left out), by
        os.fwalk from a verified fd: a symlinked folder is not entered, a symlink not listed."""
        parts = self._parts(rel)
        dfd = self._dir(parts, self.show(rel))
        try:
            for dirpath, dirnames, filenames, dirfd in os.fwalk(".", dir_fd=dfd,
                                                                 follow_symlinks=False):
                dirnames[:] = sorted(d for d in dirnames if d not in skip)
                base = parts + [s for s in dirpath.split("/") if s not in ("", ".")]
                for f in sorted(filenames):
                    try:
                        st = os.stat(f, dir_fd=dirfd, follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISREG(st.st_mode):
                        yield "/".join(base + [f])
        finally:
            os.close(dfd)


class Workspace:
    def __init__(self, root: str):
        self.dir = SandboxDir(root, MOUNT)
        self.root = self.dir.root

    def ensure(self) -> None:
        self.dir.ensure()

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
        """The host path, for show() and checks only: every existing component on the way must be
        a real directory or file, not a symlink. Nothing opens what it returns; the reads and
        writes below go through SandboxDir."""
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
        return self.dir.read(self.rel(path), limit, label=path)

    def open(self, path: str):
        """(binary file object, its fstat) for a streamed read; the caller closes it. The stat is
        the open file's, so a file replaced after the open is read as it was."""
        return self.dir.open(self.rel(path), label=path)

    def stat(self, path: str) -> os.stat_result:
        return self.dir.stat(self.rel(path), label=path)

    def write(self, path: str, data: bytes, if_match: str | None = None) -> str:
        """Create or replace; `if_match`: the etag the caller read (a 412 when it moved)."""
        self.dir.write(self.rel(path), data, label=path, if_etag=if_match)
        return etag(data)

    def delete(self, path: str) -> None:
        self.dir.delete(self.rel(path), label=path)

    def list(self, path: str = MOUNT) -> list[dict]:
        rel = self.rel(path)
        out = []
        for name, st in self.dir.scan(rel, label=path):
            kind = "dir" if stat.S_ISDIR(st.st_mode) else "file"
            out.append({"path": self.dir.show(f"{rel}/{name}" if rel else name), "kind": kind,
                        "bytes": 0 if kind == "dir" else int(st.st_size),
                        "modified_at": _iso(st.st_mtime)})
        return sorted(out, key=lambda f: (f["kind"] != "dir", f["path"]))

    def size(self) -> int:
        total = 0
        for dirpath, dirnames, filenames, dirfd in os.fwalk(self.root, follow_symlinks=False):
            for f in filenames:
                try:
                    total += os.stat(f, dir_fd=dirfd, follow_symlinks=False).st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------ search
    def walk(self, start: str = MOUNT):
        """The host path of every regular file under `start` (SandboxDir.files: no symlink, no
        SKIP_DIRS); open them through read() or open(), never by the path."""
        for rel in self.dir.files(self.rel(start), SKIP_DIRS):
            yield os.path.join(self.root, rel)

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
             ignore_case: bool = False, limit: int = 200,
             skipped: list | None = None) -> list[str]:
        """`file:line: text` of the lines that match, at most `limit`. `start` is a folder or one
        file: both are streamed to GREP_FILE_MAX with the bounds in the streaming pass below, so
        no file is skipped for its size and `skipped` (the caller's list, kept for the tool's
        call shape) stays empty."""
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as exc:
            raise PathError(f"bad pattern: {exc}") from None
        out: list[str] = []
        deadline = time.monotonic() + GREP_BUDGET_S
        # `start` may be a FILE — the tool's own contract ("path: file or directory"). Streaming
        # per line removes the old bulk-read cliff: past MAX_READ, files were skipped with no word.
        try:
            tgt = self.resolve(start, must_exist=True)
            single = stat.S_ISREG(os.lstat(tgt).st_mode)
        except Exception:
            single = False
        targets = [tgt] if single else list(self.walk(start))
        for p in targets:
            if include and not fnmatch.fnmatch(os.path.basename(p), include):
                continue
            try:
                fh, st = self.dir.open(os.path.relpath(p, self.root))
                with fh:
                    head = fh.read(4096)
                    if b"\0" in head:
                        continue                      # binary
                    fh.seek(0)
                    # streamed with a bounded window: readline() has no cap and neither did the
                    # loop, so one newline-less multi-GB file OOM'd the thread (the engine's
                    # request dumps can be exactly that). The cliffs, chosen on purpose:
                    #   - a line is searched only in its first GREP_LINE_MAX bytes (64 KiB) —
                    #     a minified bundle or a request dump past that is read, not copied;
                    #     the line number stays right: a boundary advances it once, whatever
                    #     it is — the set is exactly str.splitlines()'s, so grep and read()
                    #     number every line of every file the same (a form feed advances both)
                    #   - a file is searched only in its first GREP_FILE_MAX bytes (16 MiB,
                    #     8x MAX_READ: the read() tool hands the model 2 MiB at once, and the
                    #     old skip-silently cliff at exactly MAX_READ is why this streams at
                    #     all — a grep that says "no matches" about a 2.2 MB file lies).
                    #     Above 16 MiB grep is the wrong tool anyway: find the shape with
                    #     glob(), page the bytes with read().
                    n = 0                       # 1-based; a giant line keeps its own number
                    dec = codecs.getincrementaldecoder("utf-8")("replace")
                    buf = ""                    # carry-over between chunks, at most one line
                    skip = False                # inside a line past its window: drop to its EOL
                    held = False                # a \r parked at the previous chunk's end: it
                                                # sits INSIDE this head — a pure chunk with a
                                                # held \r may not take the fast path
                    total = 0
                    while True:
                        raw = fh.read(GREP_CHUNK)
                        if not raw:
                            break               # eof; a trailing line is handled after the loop
                        total += len(raw)
                        cliff = total >= GREP_FILE_MAX   # this chunk is the last: it gets
                                                        # searched, then the walk goes on
                        if b"\0" in raw:
                            break               # the 4 KiB head sniff missed a binary tail
                        if cliff and not fh.read(1):
                            cliff = False       # a file of exactly 16 MiB has no rest to
                                                # call unread: peek before announcing one
                        # one decode per chunk, incremental so a char cut by the chunk
                        # seam is never mangled: the decoder holds the partial char and
                        # resolves it against the next chunk
                        # a chunk holding none of the 6 ascii control separators other
                        # than '\n' partitions identically under str.split('\n');
                        # six memchr finds answer it, allocation-free. A chunk edge
                        # cannot fabricate a control byte: a lone \r would already
                        # have triggered the general path and been held via `held`.
                        exotic = any(raw.find(c) >= 0 for c in _GREP_CTL)
                        buf += dec.decode(raw)
                        if time.monotonic() > deadline:
                            out.append(f"{self.show(p)}: <grep stopped at the {GREP_BUDGET_S}s "
                                       f"budget — partial results>")
                            return out          # the budget is checked per chunk: a wedged
                                                    # regex can overrun it by one chunk at most
                        if skip:                # the rest of an over-window line: drop it,
                            m = _GREP_EOL.search(buf)   # but the count keeps walking: the long
                            if not m:                   # line ends at the FIRST boundary in the
                                buf = ""                # buffer — lines past it belong to the
                                held = False            # walk, not to the skip, so the rest of
                                continue                # the chunk is never swallowed
                            n += 1
                            buf = buf[m.end():]
                            skip = False
                            held = False        # the whole head was scanned: no held \r left
                        # one line pass per chunk: split the chunk once — the shipped loop
                        # re-split(b"\n", 1) the whole buffer per line and re-copied it every
                        # time, 9x main's speed on 40 files of 1.8 MiB. A \r parked at the
                        # chunk end is held back: \r\n across the seam is ONE boundary, and
                        # a split now would count it twice.
                        head = buf[:-1] if buf.endswith("\r") else buf
                        tail_s = buf[len(head):]        # the held \r when there is one
                        # pure-'\n' chunks take the str.split fast path; the rare exotic
                        # head pays for the general partition instead. The Unicode halves
                        # of _GREP_EOL are probed HERE, on the decoded head -- in the raw
                        # bytes they can straddle the chunk seam and hide from a per-chunk
                        # scan -- so a chunk of '\n' and U+0085/2028/2029 cannot take the
                        # fast path and under-count lines splitlines() counts.
                        lines = (head.split("\n") if not exotic and not held
                                 and not any(u in head for u in _GREP_HIGH)
                                 else _GREP_EOL.split(head))
                        held = tail_s == "\r"
                        buf = lines.pop() + tail_s      # the last line has no boundary yet
                        for line in lines:
                            n += 1
                            if len(line) > GREP_LINE_MAX:   # the window: only pay the slice
                                line = line[:GREP_LINE_MAX]  # where a line actually crosses it
                            if rx.search(line):
                                out.append(f"{self.show(p)}:{n}: {line[:300]}")
                                if len(out) >= limit:
                                    return out
                        if len(buf) > GREP_LINE_MAX:
                            # the line goes on past the window: search the window under its
                            # number (n+1 — the boundary has not come yet), drop the rest and
                            # wait for the boundary; the window is 64 K chars of the line
                            if rx.search(buf[:GREP_LINE_MAX]):
                                out.append(f"{self.show(p)}:{n + 1}: {buf[:300]}")
                                if len(out) >= limit:
                                    return out
                            buf = ""
                            skip = True
                            held = False
                        if cliff:               # the last chunk is searched, the rest is not:
                            out.append(f"{self.show(p)}: <grep searched the first 16 MiB of "
                                       f"this file — the rest of it is unread>")
                            break               # say so, the budget's way; the walk goes on
                    tail = dec.decode(b"", True)   # a char cut by the last read resolves here
                    if tail:
                        buf += tail
                    if buf and not skip:        # a final line with no newline
                        n += 1
                        if rx.search(buf[:GREP_LINE_MAX]):
                            out.append(f"{self.show(p)}:{n}: {buf[:300]}")
            except (PathError, OSError):
                continue
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
