"""ENG-411: the C++ and CUDA sources live in files beside the modules that compile them. CPU only.

A module `<dir>/<name>.py` that builds an extension keeps its sources in `<dir>/csrc/<name>/` and reads them
with engine/native_build.py (`source`); the fingerprint reads the same folder (engine/methods/manifest.py
`files`). Checked here:

  * layout      every csrc folder has its module; every file a module reads is there, and every file there
                is read by its module (no orphan, no stray build output)
  * no strings  no module under engine/ or tools/ calls load_inline itself or keeps C++/CUDA in a string
  * exact text  `source` returns the bytes as written (no newline translation); `tag` is the sha1 scheme
                the build names had
  * fingerprint a kernel file is in the files of every model whose manifest reaches its module, an edit to
                it moves exactly those models' fp4, and a source no manifest reaches moves none

Run: python tests/test_native_sources.py
"""

from __future__ import annotations

import ast
import glob
import hashlib
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from engine import native_build as nb  # noqa: E402
from engine.methods import manifest as mf  # noqa: E402

#: the trees whose modules may compile native code
TREES = ("engine", "tools")


def _modules() -> list[str]:
    out = []
    for t in TREES:
        out += glob.glob(os.path.join(ROOT, t, "**", "*.py"), recursive=True)
    return sorted(p for p in out if "__pycache__" not in p)


def _reads(path: str) -> set:
    """The files a module reads with `nb.source(__file__, "<file>")` (literal names)."""
    out = set()
    for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "source"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "nb"):
            assert len(n.args) == 2 and isinstance(n.args[0], ast.Name) and n.args[0].id == "__file__", (
                path, ast.unparse(n))
            assert isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str), (path, ast.unparse(n))
            out.add(n.args[1].value)
    return out


def _folders() -> list[str]:
    out = []
    for t in TREES:
        for d in glob.glob(os.path.join(ROOT, t, "**", nb.CSRC), recursive=True):
            out += [os.path.join(d, x) for x in sorted(os.listdir(d))]
    return sorted(out)


def test_every_source_folder_belongs_to_its_module_and_holds_what_it_reads():
    folders = _folders()
    assert folders, "no csrc folder"
    read = 0
    for f in folders:
        assert os.path.isdir(f), f"{os.path.relpath(f, ROOT)}: a csrc entry is a folder a module"
        mod = os.path.join(os.path.dirname(os.path.dirname(f)), os.path.basename(f) + ".py")
        assert os.path.isfile(mod), f"{os.path.relpath(f, ROOT)} has no module {os.path.relpath(mod, ROOT)}"
        have = {os.path.relpath(p, f) for p in nb.sources(mod)}
        want = _reads(mod)
        assert want, f"{os.path.relpath(mod, ROOT)} reads nothing from {os.path.relpath(f, ROOT)}"
        assert have == want, (os.path.relpath(mod, ROOT), "missing", sorted(want - have), "unread", sorted(have - want))
        read += len(want)
    for mod in _modules():
        if _reads(mod):
            assert os.path.isdir(nb.folder(mod)), f"{os.path.relpath(mod, ROOT)} reads from a missing folder"
    return f"{len(folders)} modules, {read} source files"


def _code_strings(path: str) -> list[tuple[int, str]]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


#: what only C++ or CUDA source holds
NATIVE = ("#include <", "__global__", "__device__", "torch::Tensor", "TORCH_CHECK(", "PYBIND11_MODULE")


def test_no_module_compiles_a_string():
    bad = []
    for mod in _modules():
        rel = os.path.relpath(mod, ROOT)
        if rel == "engine/native_build.py":
            continue
        text = open(mod, encoding="utf-8").read()
        if "load_inline(" in text.replace("`torch.utils.cpp_extension.load_inline`", ""):
            for ln, line in enumerate(text.splitlines(), 1):
                if "load_inline(" in line and not line.lstrip().startswith("#"):
                    bad.append(f"{rel}:{ln} calls load_inline (engine/native_build.py `load`)")
        for ln, s in _code_strings(mod):
            if any(k in s for k in NATIVE) and (";" in s or "{" in s):     # a search key is no source
                bad.append(f"{rel}:{ln} holds C++/CUDA in a string ({s.strip()[:40]!r})")
    assert not bad, bad
    return f"{len(_modules())} modules"


def test_source_reads_the_bytes_as_written_and_tag_is_the_sha1_scheme():
    with tempfile.TemporaryDirectory() as d:
        mod = os.path.join(d, "kern.py")
        open(mod, "w").close()
        os.makedirs(os.path.join(d, "csrc", "kern", "patches"))
        text = "\n// a kernel\r\nint x;\t \n\n"
        with open(os.path.join(d, "csrc", "kern", "k.cu"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
        with open(os.path.join(d, "csrc", "kern", "patches", "a.old.cuh"), "w", encoding="utf-8", newline="") as f:
            f.write("    int y;")                                   # no newline at the end
        assert nb.source(mod, "k.cu") == text
        assert nb.source(mod, "patches/a.old.cuh") == "    int y;"
        assert [os.path.relpath(p, nb.folder(mod)) for p in nb.sources(mod)] == ["k.cu", os.path.join("patches", "a.old.cuh")]
        assert nb.sources(os.path.join(d, "other.py")) == []
        try:
            nb.source(mod, "gone.cu")
            raise AssertionError("a missing source read as text")
        except FileNotFoundError:
            pass
    a, b = "void f();\n", "#define X 1\n__global__ void k() {}\n"
    assert nb.tag(a, b) == hashlib.sha1((a + b).encode()).hexdigest()[:10]
    assert nb.tag(a) == hashlib.sha1(a.encode()).hexdigest()[:10]
    return "bytes kept (CRLF, tabs, no final newline); tag = sha1[:10]"


def _torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _skip(why: str) -> Exception:
    from scriptmode import Skip
    return Skip(why)


def test_a_kernel_file_moves_exactly_the_models_that_reach_it():
    if not _torch():
        raise _skip("the manifests import torch")
    from engine.models import REGISTRY
    mans = {n: REGISTRY[n].manifest_of() for n in REGISTRY}
    holders: dict = {}
    for n, m in mans.items():
        for p in mf.files(ROOT, m):
            holders.setdefault(os.path.relpath(p, ROOT), set()).add(n)
    # every source of a module in a model's files is in them too, and the reader with them
    for rel, who in sorted(holders.items()):
        if rel.endswith(".py"):
            for s in nb.sources(os.path.join(ROOT, rel)):
                assert holders.get(os.path.relpath(s, ROOT), set()) >= who, (os.path.relpath(s, ROOT), who)
    native = {rel for rel in holders if f"/{nb.CSRC}/" in rel}
    assert "tools/csrc/exl3_linear/exl3.cu" in native and "tools/csrc/fn_kernels/kernels.cu" in native, sorted(native)
    for rel in native:
        assert holders[rel] <= holders["engine/native_build.py"], rel
    unreached = [os.path.relpath(p, ROOT) for f in _folders() for p in nb.sources(
        os.path.join(os.path.dirname(os.path.dirname(f)), os.path.basename(f) + ".py"))]
    unreached = sorted(r for r in unreached if r not in holders)
    assert "tools/csrc/l2_prefetch_bench/prefetch.cu" in unreached, unreached
    probes = ["tools/csrc/exl3_linear/exl3.cu", "tools/csrc/fn_kernels/kernels.cu",
              "tools/csrc/nvfp4_skinny/skinny.cu", "engine/native_build.py", unreached[0]]
    with tempfile.TemporaryDirectory() as d:
        for sub in ("engine", "tools", "ops"):
            shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pt", "*.safetensors", "*.npy"))
        before = {n: mf.fingerprint(d, m, {}) for n, m in mans.items()}
        for rel in probes:
            with open(os.path.join(d, rel), "a", encoding="utf-8") as f:
                f.write("\n// a probe\n" if not rel.endswith(".py") else "\n# a probe\n")
            after = {n: mf.fingerprint(d, m, {}) for n, m in mans.items()}
            moved = {n for n in mans if after[n] != before[n]}
            assert moved == holders.get(rel, set()), (rel, sorted(moved), sorted(holders.get(rel, set())))
            before = after
    return (f"{len(native)} sources in {len(mans)} models' files; probes moved exactly their holders "
            f"({', '.join(f'{os.path.basename(p)}: {len(holders.get(p, ()))}' for p in probes)})")


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
