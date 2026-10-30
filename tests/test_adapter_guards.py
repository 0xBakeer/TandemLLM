"""ENG-404 (ENG-E17): the guards of the model adapters. Work on one model never moves another model's rows,
and shared code never names a model.

  1  goldens       every registered adapter's fingerprint v3 (fp4) and the 27B's v2 equal v0.14.0's, raw
                   bytes, extra {}; only the adapters in MAY_MOVE may differ (Flint, once, ENG-E17)
  2  SAME_ROWS     the table is v0.14.0's: the refactor adds no entry (a release that needs one updates
                   SAME_ROWS_SHA256 here on purpose, with the entry's reason in engine/methods/manifest.py)
  3  isolation     in a copy of the tree, a changed file of one package moves exactly the adapters whose
                   manifest holds it: its own, and for a shared library its declared dependents
  4  lint          the shared modules name no model, family or tool format in code (strings and names;
                   comments and docstrings are free): a ratchet, each file's count may only go down
  5  contract      every registered adapter has the fields and files the server relies on

The goldens and isolation import every adapter's manifest (torch): they run where torch is (the box);
elsewhere they are reported as skipped. Run: python tests/test_adapter_guards.py
"""

from __future__ import annotations

import collections
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.methods import manifest as mf  # noqa: E402
from engine.models import REGISTRY  # noqa: E402

#: v0.14.0 (7c682ab0), extra {}: the 27B's v2 (skip NOT_ARITHMETIC, then the tier's) read through SAME_ROWS
#: (the bytes every release since 0.7.2 adopts under; engine/grammar.py's 0.14.0 and pre-0.14.0 bytes both
#: count as 0.9's), and each adapter's fp4 on its raw bytes
#: SPD-E5 (2026-10-23): the 27B's v2 and fp4 move once for the branch (engine/model.py's joint row-wise
#: steps and window-MLP knob, engine/drafters/draft_graph.py's packed-graph cap, two knobs in engine/settings.py;
#: the 27B's prefill knobs SPD-169/189/191/193 and their modules tools/exl3_slab_gemm.py, tools/gdn_prep_kernels.py)
#: SPD-201 (2026-10-24): and again once: the round host cut (engine/rowcopy.py, engine/native_tree.py, the drafters'
#: row copies, two-pass top-k and span step), the sequential GDN verify (tools/gdn_seq_kernels.py) and the
#: invariant verify attention, every one behind a knob that is off
#: SPD-201 (2026-10-24, later): the measured speed setup defaults on (engine/settings.py, the 27B's knobs.py) and
#: SPD-211's draft window fix (engine/drafters/dflash2.py, engine/kvpool.py): v2 and every adapter's fp4 move;
#: SPD-215 the resident prefix's prompt end (engine/cache.py): v2 again
#: ENG-411: the kernel source strings leave engine/native_tree.py and the tools/ kernels (csrc/ files): v2 again
GOLDEN_V2 = ("faf028096854cce746c42f570e94f7e04a62775902d5d3b1f75640582c427b5a",
             "fe2e7ba3ee4fc933eff399243aa4d7d506e142be73fd4910d6611dff950c396f")
GOLDEN_FP4 = {
    # SPD-201 (2026-10-24): the speed setup's knobs default on, so every family's fingerprint moved once;
    # SPD-215: Flash-Next's joint pass row-wise and grouped experts by default (its two adapters again);
    # SPD-217: Ornith's joint verify pass (qwen3_5_moe once more)
    # SPD-229: each state parks its own GDN rows; Ornith's joint pass on the grouped experts (Flash-Next and Ornith)
    # SPD-227: the prefill's int8 mixer GEMM on the tensor cores as an option (Flash-Next's kernels)
    # SPD-231: the register-blocked mixer kernels behind QWEN38_FN_GEMV_NB (off)
    # SPD-227: mm_i8_tc compiles once a weight shape
    # SPD-233/235: Flash-Next's 8-lane row count; the packed MTP draft (Flash-Next, Ornith)
    # SPD-231: QWEN38_FN_GEMV_NB in the shared table beside FN_GEMV_SPLIT
    # SRV-717: the allocator snapshot reused 5 s
    # SPD-223: draft graphs check their window pages and every pointer they bake
    # SPD-239: FN_JOINT_TIMING knob, the joint pass's part timing in _timed
    # SPD-239: the joint timing's host parts
    # SPD-239: staging parts, FN_GEMV_MC
    # SPD-239: PLE rows in one batch
    # SPD-239: FN_PLE_THREADS
    # SPD-239: commit_multi
    # SPD-239: FN_GDN_MULTI
    # SPD-239: FN_GDN_MULTI default
    # SPD-241: mixer MMA kernels
    # SPD-239: Ornith grouped GDN, lane-aware skip
    # SPD-241: FN_MIXER_MMA default
    # SPD-239: FN_POOL_MULTI
    # SPD-245: QWEN38_EXL3_PDL (off): the skinny GEMM's launch under programmatic serialization, no arithmetic moved
    # SPD-245: GDN_PREP scan 16 / 4 by default (the 27B's prefill)
    # SPD-245: the GDN gate read in place (gate2d, gate_rows) and no normed-row store before the slab MLP
    # SPD-245: lane-grouped joint attention, the commit's path rows pinned
    # SPD-245: one attention table a launch, kept by the capturing graph
    # SPD-223: a capture stream for each graph owner
    # SPD-223: a new pool after a joint flush
    # SPD-245: GDN_CHUNKED knob (off), the gate tool's --flag
    # SPD-245: GDN_CHUNKED on (the GDN prefill walk value-block fastest, the UT transform in fewer registers)
    # SPD-245: prefill_joint and the skinny-order slab product
    # SRV-713: a Flash-Next MTP step leaves the verify block's records for its commit (engine/models/flashnext/engine.py,
    # no row changes); Flash-Next and Ornith
    # ENG-411: the kernel sources move into csrc/ files beside their modules and count in the fingerprint (the
    # compiled text unchanged); every adapter
    "qwen3_5": "445cd028ced11b62ce367778495aec746b56d77ff8f23e765f2a9801f961a2d4",
    "qwen4_exp": "25b36ced9a41d1873adba9d63f487c0a46932fb69c93ca48856c07a663c3cdc0",
    "nemotron_h": "5a36031e5acb49888a9b0760bea12c46d6b10f7bc19052df91c014f717539c7e",
    "lfm2_moe": "2f5201ec99d275dfae9efe00796a250a51525ce2e451f502ec5dc02f35a463c6",
    "qwen3_5_moe": "84fd123427d375e901915fbc274c29b52dd743641a8408926f4b7005c14105eb",
    "flint": "be4aa2e667130b9b9454c8cccc97f1807466b8b029d521b3b857df5009d6c011",
}
#: adapters the refactor may move once (the coordinator's OK, 2026-10-22): Flint's knobs leave the shared
#: table and its kernels take the generic launch option
MAY_MOVE = {"flint"}
#: sha256 of SAME_ROWS' entries at v0.14.0 (sorted [path, now, replaced] triples, JSON)
SAME_ROWS_SHA256 = "8f5071c81945e487ddfe54f9a2891df42f9eda89b7f9499937be113a946177b0"   # 30 entries


def _manifest(name: str):
    """The registered adapter's manifest (imports its serving module, torch)."""
    return REGISTRY[name].manifest_of()


def _torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _skip(why: str) -> Exception:
    sys.path.insert(0, HERE)
    from scriptmode import Skip
    return Skip(why)


def _same_rows_sha() -> str:
    rows = sorted([k[0], k[1], v] for k, v in mf.SAME_ROWS.items())
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


# ------------------------------------------------------------------ 1: goldens

def test_goldens_hold():
    if not _torch():
        raise _skip("the manifests import torch")
    from engine import disk_cache as dc
    from engine import spill as sp
    v2 = (dc.fingerprint(ROOT, {}, skip=sp.NOT_ARITHMETIC, same=mf.SAME_ROWS),
          dc.fingerprint(ROOT, {}, same=mf.SAME_ROWS))
    assert v2 == GOLDEN_V2, f"the 27B's v2 moved: {v2}"
    moved = {}
    for name in REGISTRY:
        got = mf.fingerprint(ROOT, _manifest(name), {})
        if got != GOLDEN_FP4.get(name):
            moved[name] = got
    assert set(moved) <= MAY_MOVE, f"fingerprints moved: {moved}"
    assert set(GOLDEN_FP4) <= set(REGISTRY), "an adapter of v0.14.0 is gone"
    return f"v2 and {len(REGISTRY)} adapters; moved: {sorted(moved) or 'none'}"


def test_same_rows_is_frozen():
    assert _same_rows_sha() == SAME_ROWS_SHA256, (
        f"SAME_ROWS changed ({len(mf.SAME_ROWS)} entries, sha {_same_rows_sha()}): the refactor adds none")


# ------------------------------------------------------------------ 3: isolation

def _shared() -> dict:
    """ENG-408: each shared library file (a provider's `shares`) and the families whose fingerprint it is
    part of: the provider and every family that `uses` it."""
    out = {}
    for name, a in REGISTRY.items():
        for f in a.shares:
            out.setdefault(f, set()).add(name)
    for name, a in REGISTRY.items():
        for f in a.uses:
            assert f in out, f"{name} uses {f}, which no family shares"
            out[f].add(name)
    return out


def _imported_files(path: str) -> set:
    """The files of engine/models/<pkg>/ a module imports (`from engine.models.X import y` names X/y.py
    when y is a module, else X/__init__.py)."""
    import ast
    out = set()
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("engine.models."):
            mods = [(node.module, [a.name for a in node.names])]
        elif isinstance(node, ast.Import):
            mods = [(a.name, []) for a in node.names if a.name.startswith("engine.models.")]
        for mod, names in mods:
            parts = mod.split(".")[2:]
            if len(parts) >= 2:
                out.add(f"engine/models/{parts[0]}/{parts[1]}.py")
            elif len(parts) == 1:
                for n in names or ["__init__"]:
                    f = f"engine/models/{parts[0]}/{n}.py"
                    out.add(f if os.path.isfile(os.path.join(ROOT, f)) else f"engine/models/{parts[0]}/__init__.py")
    return out


def test_a_family_imports_only_what_it_declares():
    pkg_of = {a.package.split("/")[-1]: n for n, a in REGISTRY.items() if a.package}
    pkg_of.update({"qwen3_5": "qwen3_5"})
    bad = []
    for pkg, name in pkg_of.items():
        for path in glob.glob(os.path.join(ROOT, "engine", "models", pkg, "*.py")):
            for f in sorted(_imported_files(path)):
                other = f.split("/")[2]
                if other == pkg or other not in pkg_of:
                    continue                      # its own package, or the shared types (adapter, state)
                if f not in REGISTRY[name].uses:
                    bad.append(f"{os.path.relpath(path, ROOT)} imports {f}, not in {name}'s uses")
    assert not bad, bad
    shared = _shared()
    for f, holders in shared.items():
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            words = {w for _, w in lint_hits(fh.read())}
        others = {REGISTRY[h].package.split("/")[-1] for h in holders if REGISTRY[h].package} - {f.split("/")[2]}
        assert not words & {"flint"} and not any(o in words for o in others), (f, words)
    return f"{sum(len(a.uses) for a in REGISTRY.values())} declared uses, {len(shared)} shared files"


def test_one_package_moves_only_its_dependents():
    if not _torch():
        raise _skip("the manifests import torch")
    mans = {n: _manifest(n) for n in REGISTRY}
    holders = collections.defaultdict(set)
    for n, m in mans.items():
        for p in mf.files(ROOT, m):
            holders[os.path.relpath(p, ROOT)].add(n)
    pkg_files = {p: h for p, h in holders.items() if p.startswith("engine/models/")}
    cross = {p: h for p, h in pkg_files.items() if len(h) > 1}
    shared = _shared()
    undeclared = {p: h for p, h in cross.items() if shared.get(p) != h}
    assert not undeclared, f"a manifest holds another package's file its entry does not declare: {undeclared}"
    with tempfile.TemporaryDirectory() as d:
        for sub in ("engine", "tools", "ops"):
            shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pt", "*.safetensors"))
        before = {n: mf.fingerprint(d, m, {}) for n, m in mans.items()}
        checked = 0
        own = {min(p for p, h in pkg_files.items() if h == {n}) for n in mans
               if any(h == {n} for h in pkg_files.values())}
        for path in sorted(own | set(cross)):
            full = os.path.join(d, path)
            with open(full, "ab") as f:
                f.write(b"\n# isolation probe\n")
            after = {n: mf.fingerprint(d, m, {}) for n, m in mans.items()}
            moved = {n for n in mans if after[n] != before[n]}
            assert moved == pkg_files[path], f"{path}: moved {moved}, expected {pkg_files[path]}"
            before = after
            checked += 1
    return f"{checked} files probed; each moved exactly its holders"


# ------------------------------------------------------------------ 4: lint

#: what a shared module may not name in code: models, families, tool formats, model-only switches
WORDS = re.compile(r"flint|lfm|liquid|nemotron|flashnext|flash_next|ornith|qwen3_5|qwen4_exp|qwen"
                   r"|argkey|pythonic|self_think", re.I)
#: the product's own namespace, which is not a model: QWEN38_* settings, ~/.qwen38-spark-engine
NAMESPACE = re.compile(r"qwen38", re.I)
SHARED_GLOBS = ("server/**/*.py", "proxy/**/*.py", "engine/*.py", "engine/methods/*.py", "engine/drafters/*.py")
#: the format modules and the model packages are where those names belong
OWN_PLACES = ("server/toolformats/",)

#: the ratchet: v0.14.0's counts a file may keep (ENG-E17 lowers them; a count may only go down, and a file
#: at 0 leaves the table). Each line says why the rest stays.
BASELINE = {
    # the conductor's router stage that asks the served model ("qwen" kind) and its prompts: a stored
    # document's key, not a model check
    "server/conductor/classify.py": 6, "server/conductor/decisions.py": 2, "server/conductor/fake_decider.py": 1,
    "server/conductor/hook.py": 3, "server/conductor/libsearch.py": 1, "server/conductor/memory.py": 3,
    "server/conductor/memwriter.py": 1, "server/conductor/narrator.py": 2, "server/conductor/notes.py": 2,
    "server/conductor/proposals.py": 1, "server/conductor/router.py": 15, "server/conductor/schema.py": 18,
    "server/conductor/shadow.py": 5, "server/metrics.py": 2, "proxy/harness.py": 5, "engine/shadow.py": 1,
    "server/setup/wren.py": 17,
    # the 27B's own config and drafters (engine/*.py is its package by history, ENG-221)
    "engine/config.py": 2, "engine/drafters/dflash2.py": 1, "engine/drafters/dspark.py": 1,
    # SAME_ROWS names files by path; the legacy FN_*/Q35M_* knob owners (frozen, nothing new)
    "engine/methods/manifest.py": 13, "engine/settings.py": 16,
    # the default --served-model name, the API's own since 0.1 (clients address it)
    "server/app.py": 1,
    # the fake engine emulates each format's output for the tests
    "server/fake_engine.py": 4,
}


def _shared_files() -> list[str]:
    out = set()
    for p in SHARED_GLOBS:
        out |= set(glob.glob(os.path.join(ROOT, p), recursive=True))
    return sorted(os.path.relpath(p, ROOT) for p in out
                  if not os.path.relpath(p, ROOT).startswith(OWN_PLACES))


def lint_hits(src: str) -> list[tuple[int, str]]:
    """(line, word) of every model word in `src`'s code: string literals that are not a statement of
    their own (a docstring), and names. Comments never count."""
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out, prev = [], None
    skip = (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.NL)
    fmid = getattr(tokenize, "FSTRING_MIDDLE", None)       # 3.12+: an f-string's text is its own token
    for i, t in enumerate(toks):
        if fmid is not None and t.type == fmid:
            prev = t
        elif t.type == tokenize.STRING:
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if (prev is None or prev.type in skip) and (nxt is None or nxt.type in skip + (tokenize.ENDMARKER,)):
                prev = t
                continue                                   # a docstring or a bare string statement
        elif t.type != tokenize.NAME:
            if t.type not in (tokenize.COMMENT, tokenize.NL):
                prev = t
            continue
        prev = t
        for m in WORDS.finditer(t.string):
            if NAMESPACE.search(t.string[m.start():m.end() + 2]):
                continue
            out.append((t.start[0], m.group(0).lower()))
    return out


def test_lint_shared_modules_name_no_model():
    counts, bad = {}, []
    for rel in _shared_files():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            hits = lint_hits(f.read())
        if hits:
            counts[rel] = len(hits)
        if len(hits) > BASELINE.get(rel, 0):
            bad.append(f"{rel}: {len(hits)} > {BASELINE.get(rel, 0)} ({sorted({w for _, w in hits})}, "
                       f"lines {[ln for ln, _ in hits][:8]})")
    stale = sorted(f for f, n in BASELINE.items() if counts.get(f, 0) < n)
    assert not bad, "shared modules name a model: " + "; ".join(bad)
    assert not stale, f"the ratchet went down: lower BASELINE for {stale} to {[counts.get(f, 0) for f in stale]}"
    return f"{len(counts)} files with legacy names, {sum(counts.values())} names"


def test_the_lint_reads_code_not_comments():
    src = ('"""A docstring about Flint."""\n# argkey in a comment\nx = 1\n'
           'def f():\n    """lfm in a docstring"""\n    return "argkey"\n'
           'y = QWEN38_FOO = "~/.qwen38-spark-engine"\nflint_thing = 2\n')
    assert lint_hits(src) == [(6, "argkey"), (8, "flint")], lint_hits(src)


# ------------------------------------------------------------------ 5: contract

#: family knobs engine/settings.py kept from before ENG-405 (frozen: a new family knob goes in its package's
#: knobs.py)
LEGACY_FAMILY_KNOBS = {
    "FN_DENSE", "FN_DRAFT_HOT", "FN_DROP_SHARD_CACHE", "FN_FOLD_COMMIT", "FN_LOOKUP", "FN_MIX", "FN_PREFILL",
    "FN_PREFILL_CHUNK", "FN_TABLE", "FN_TABLE_CACHE_GB", "FN_WARM_TABLE", "Q35M_ANCHOR_BACKOFF",
    "Q35M_DFLASH2_HOT", "Q35M_DFLASH2_NVFP4", "Q35M_LOOKUP", "Q35M_SIBLINGS"}


def test_every_adapter_has_the_contract():
    pkgs = sorted(d for d in os.listdir(os.path.join(ROOT, "engine", "models"))
                  if os.path.isfile(os.path.join(ROOT, "engine", "models", d, "arch.py")))
    assert len(pkgs) == len(REGISTRY), "a package's arch.py registers no adapter, or two"
    for name, a in REGISTRY.items():
        assert a.name == name and a.architectures and a.model_types, name
        for ref in (a.config, a.engine, a.manifest) + ((a.serve,) if a.serve else ()):
            assert ref and ref.count(":") == 1, (name, ref)
        if a.package:
            assert os.path.isdir(os.path.join(ROOT, "engine", a.package)), name
    names = [m for a in REGISTRY.values() for m in a.model_types]
    assert len(names) == len(set(names)), "a model_type selects two adapters"
    return f"{len(REGISTRY)} adapters from {len(pkgs)} packages"


def test_a_knob_registered_twice_is_refused():
    import importlib
    from engine import settings
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "fam"))
        with open(os.path.join(d, "fam", "knobs.py"), "w") as f:
            f.write("KNOBS = {'PREFILL_CHUNK': ('1', 'engine/models/fam/x.py')}\n")
        name = next(iter(settings.KNOBS))
        with open(os.path.join(d, "fam", "knobs.py"), "w") as f:
            f.write(f"KNOBS = {{{name!r}: ('1', 'engine/models/fam/x.py')}}\n")
        try:
            settings._family_knobs(d)
        except ValueError as exc:
            assert "registered twice" in str(exc)
        else:
            raise AssertionError("a knob in two tables must be refused")
    for path in glob.glob(os.path.join(ROOT, "engine", "models", "*", "knobs.py")):
        pkg = os.path.basename(os.path.dirname(path))
        mod = importlib.import_module(f"engine.models.{pkg}.knobs")
        assert all(settings.KNOBS[k] == v for k, v in mod.KNOBS.items()), pkg


def test_community_bases_are_their_familys():
    """ENG-409: a model's community-pack rules are in its model.yaml, its profile and pack check in its family's
    package (`hook`); the setup registry names no model."""
    import importlib
    from server.setup import registry
    bases = registry.community_bases()
    assert bases, "no community base"
    pkgs = {a.package.split("/")[-1] if a.package else n for n, a in REGISTRY.items()}
    for mid, rules in bases.items():
        mod = importlib.import_module(rules["hook"])
        assert callable(mod.env) and callable(mod.check_pack), mid
        assert rules["hook"].split(".")[2] in pkgs, (mid, rules["hook"])
        assert rules["serves"] in ("loader", "profile") and rules["drafters"], mid
    return f"{len(bases)} bases: {sorted(bases)}"


def test_family_knobs_live_in_their_package():
    from engine import settings
    from engine.methods.manifest import _knob_table
    shared = _knob_table(os.path.join(ROOT, "engine", "settings.py"))
    fam = {k for k, v in shared.items() if len(v) > 1 and str(v[1]).startswith("engine/models/")}
    assert fam == LEGACY_FAMILY_KNOBS, f"a family knob in engine/settings.py: {sorted(fam - LEGACY_FAMILY_KNOBS)}"
    for path in glob.glob(os.path.join(ROOT, "engine", "models", "*", "knobs.py")):
        pkg = os.path.relpath(os.path.dirname(path), ROOT) + "/"
        for k, v in _knob_table(path).items():
            assert str(v[1]).startswith(pkg), (path, k, v)
            assert settings.KNOBS[k] == v and settings.SETTINGS.get(k) is not None, k
    if _torch():
        for name, a in REGISTRY.items():
            own = os.path.join(ROOT, "engine", a.package or "", "knobs.py")
            if a.package and os.path.isfile(own):
                assert os.path.relpath(own, ROOT) in a.manifest_of().modules, f"{name}: knobs.py not in its manifest"
    return f"{len(LEGACY_FAMILY_KNOBS)} legacy family knobs in the shared table, none new"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
