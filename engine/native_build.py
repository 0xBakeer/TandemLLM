"""The C++ and CUDA sources of a module that compiles native code, in files beside it (ENG-411).

A module `<dir>/<name>.py` that builds an extension with torch's `load_inline` keeps its sources in
`<dir>/csrc/<name>/`: a CUDA body as `.cu`, the binding declarations or host C++ as `.cpp`, declarations
as `.h`, and the text rewrites a variant applies to a shared kernel as `patches/<what>.old.cuh` (the text
replaced) beside `patches/<what>.new.cuh` (its replacement). The module reads each file once at import
(`source`) and assembles what it compiles in Python as before: the `#define` prefixes of its knobs, the
slices and rewrites of another module's source. A file holds the exact text the module compiles (no
newline translation on read), so a build's name tag, a hash of that text (`tag`), names the same build
directory under ~/.cache/torch_extensions as long as the text is the same.

The fingerprint of a model reads these files with the module (engine/methods/manifest.py `files`): an
edit to a kernel moves the fingerprint of every model whose manifest reaches the module that builds it.
"""

from __future__ import annotations

import hashlib
import os
import sys

#: the folder beside a module that holds its sources, one subfolder a module
CSRC = "csrc"


def folder(module_file: str) -> str:
    """`<dir>/csrc/<name>/` for the module `<dir>/<name>.py`."""
    d, f = os.path.split(os.path.abspath(module_file))
    return os.path.join(d, CSRC, os.path.splitext(f)[0])


def source(module_file: str, name: str) -> str:
    """The text of `name` (a path under the module's source folder) as the file holds it."""
    with open(os.path.join(folder(module_file), name), encoding="utf-8", newline="") as f:
        return f.read()


def sources(module_file: str) -> list[str]:
    """Every file under the module's source folder (absolute paths, by relative path); [] without one."""
    base = folder(module_file)
    out = []
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.join(dirpath, f) for f in files]
    return sorted(out, key=lambda p: os.path.relpath(p, base))


def tag(*texts: str) -> str:
    """A build's name tag: the first 10 hex digits of the sha1 of the texts joined."""
    return hashlib.sha1("".join(texts).encode()).hexdigest()[:10]


def load(name: str, **kw):
    """torch's `load_inline(name=name, **kw)` with the interpreter's bin folder on PATH (ninja lives there)."""
    from torch.utils.cpp_extension import load_inline
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    return load_inline(name=name, **kw)
