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
#: SPD-249: the prefill's marked grid (engine/cache.py `Grid`, off unless a prompt has a head mark) and the
#: store keeping a head's snapshot last, and engine/capture_lock.py's guard for CUDA work off the engine's
#: thread, and the gate's store shed in chunks: v2's second value again (the first skips engine/cache.py)
#: #105: engine/drafters/ngram.py skips a lookup corpus of another tokenizer instead of refusing it: v2 again
#: SPD-249: the gate's store shed asks the heads before the next level when only heads are left or the chunks before
#: them were too small to tell (engine/cache.py): v2's second value again
#: SPD-257: each resident prefix's reserved end anchor, the policy's read of the allocator's pools and the graph
#: captures on plain segments (engine/cache.py, engine/capture_lock.py, engine/disk_cache.py): v2's second value again
#: SRV-773/SRV-775: the memory gate's charge by forwarded rows, the prefills in flight by thread and the resident
#: prefix's resume estimate (engine/cache.py): v2's second value again
#: SRV-775: the admission's count of a prefill in flight by its rows (`Budget.workspace_rows(line=False)`,
#: engine/cache.py): v2's second value again
#: SRV-775: a side slot's gate runs the peak stage only and `ws_now` is the largest prefill in flight
#: (engine/cache.py): v2's second value again
#: SRV-775: a side slot's gate far under the peak admits without the policy's lock (engine/cache.py): v2's second
#: value again
#: SRV-775: the policy's free default-pool bytes a prefill admitted over the peak takes first
#: (`Budget.workspace_spare`, engine/cache.py): v2's second value again
#: SRV-775: that credit only on a live pools reading (engine/cache.py): v2's second value again
#: the allocator's counters read off torch's nested dict, which leaves no cyclic garbage (`cache.cuda_stats`,
#: engine/cache.py): v2's second value again
#: the spill codec's import answer remembered (engine/spill.py): v2's second value again
#: the GPU adapters (tools/gpu) count with the kernels that ask them, a tools/ package read whole
#: (engine/disk_cache.py), and two MoE knobs owned by tools/gpu/gb10.py; no row changes on the GB10: v2 again
#: #144: the Ada (sm_89) Triton kernels behind the tools/gpu adapter; no row changes on the GB10: v2 again
#: #159: the QWEN38_MTP_BACKLOG_CAP default in engine/settings.py (v2 counts its knobs; the drafts only, no text)
#: SRV-E6: the spill's idle backfill, its device steps under the capture guard and the resident prefix's row
#: generation (engine/spill.py, engine/cache.py, engine/disk_cache.py's write guard): v2's second value again
#: SRV-797 (2026-11-08): the policy holds the caches' room a turn was let in on (`Budget.hold_growth`,
#: engine/cache.py): v2's second value again
#: SPD-249: tools/gdn_prefill_kernels.py's `view` (off unless Flash-Next's QWEN38_FN_PF_NOCOPY asks) and the
#: fn_pf_hc extension in tools/gpu/base.py's inline list; no row changes on the GB10: v2 again
GOLDEN_V2 = ("fc7ccaed9feffdbe43ec4147f552a6052829438d0f2b2c138eb1c63946af6fb9",
             "6a6d431e289bf70a0a9b26aeabdc898518d47be78ea63c9812ef44757cdb1a0d")
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
    # SPD-249: Flash-Next's launch class checked on the host, no class below the state's in an idle capture (no row
    # changes); Flash-Next and Ornith
    # SPD-249: the expert-union count (off by default), the mixer GEMV writing the mix_up's A,
    # the joint pass's GDN pre in one launch and the deferred GDN write-back, the grouped MoE grid rows, staged rows
    # and group cap (all off, the bits the same); Flash-Next, Ornith and the adapters that read its kernel sources
    # SPD-249: FN_DRAFT_FREEZE (off), the drafters' adaptive state kept at its start for A/B arms; Flash-Next, Ornith
    # SPD-249: FN_ATTN_MULTI and FN_PLE_MULTI (off), a joint pass's attention and PLE for every lane at once;
    # the grouped MoE's pre-lever kernels at the lever defaults; Flash-Next's kernels and the adapters that read them
    # SPD-249: FN_JOINT_GRAPH (off), a dense joint pass from CUDA graphs; the pad rows' expert id -1 skipped by the
    # grouped kernels and ranked by slot in the combine (no real row changes); Flash-Next and the adapters that read
    # its kernel sources
    # SPD-249: the round runner's CPU ms in /health and QSE_RUNNER_HOST_BLOCKS (off); the adapters that read the runner
    # SPD-249: FN_PLE_PREFETCH (the round runner hands a lane's tree to the family before its verify) and the grouped
    # expert kernels with their rows streamed and their groups from one block's scans (all off, the bits the same);
    # every adapter but qwen3_5
    # SPD-255: NemotronH's 33-row verify block on the row kernels (engine/models/nemotron_h/engine.py): nemotron_h
    # SPD-249: comments only (engine/settings.py, the round runner, Flash-Next's engine and serving, the fn_kernels
    # sources); every adapter but qwen3_5
    # SPD-249: an eager joint pass after one that raised starts on clean entry tables, a joint graph is kept by its
    # lanes' KV strides too, /health copies the kept keys (engine/models/flashnext/engine.py): Flash-Next and Ornith
    # SPD-249: Flash-Next's coupled MTP drafts for seeded sampled requests, their one-lane head, the cut calibration
    # file and the corrector (all off by default; Flash-Next and Ornith, which reads Flash-Next's engine and knobs);
    # the joint prefill and the MTP lanes' lookup backoff (off), the round runner's host counters, busy work and
    # collector control (off; engine/methods/busy.py, mtp_drafter.py, staircut.py, the runner): every adapter but
    # qwen3_5 and nemotron_h
    # the MTP catch-up's own prefill chunk cap (engine/models/flashnext/engine.py): qwen3_5_moe, qwen4_exp
    # #142: IQ2, Q8 and FP8 weights keep a 33-row verify block on their row kernels (tools/iq2_linear.py,
    # tools/nvfp4_relu2.py): nemotron_h (flint reads nvfp4_relu2.py and may move)
    # #136: the dense kernels take 7-bit EXL3 tensors (tools/csrc/fn_kernels/kernels.cu, the LFM2.5, NemotronH and
    # Flint engines' dense rates): qwen4_exp, nemotron_h, lfm2_moe, qwen3_5_moe (flint may move)
    # 64-row items in the MoE group GEMM (tools/fn_prefill_kernels.py): qwen4_exp, flint; lfm2_moe and
    # qwen3_5_moe stay on 32 rows and move only because their engine files pin the 32
    # the GPU adapters (tools/gpu): every kernel module asks them for its arch flags and launch shapes, so
    # their files count; the GB10's answers are the ones before (no row changes): every adapter
    # #144: the Ada (sm_89) Triton kernels behind the tools/gpu adapter (tools/gpu, nvfp4_linear*.py,
    # nvfp4_relu2.py, exl3_linear.py, fn_kernels.py); the GB10's answers unchanged: every adapter
    # #143: the ggml/GGUF quant types on NemotronH's GGUF loader (engine/models/nemotron_h/gguf*.py,
    # tools/gguf_quants.py, tools/gguf_grids.py): nemotron_h
    # #158: the indexer's radix select past 8,192 blocks, its fused score epilogue and live pool slice (the bits
    # the same; tools/fn_seq_kernels, tools/fn_prefill_kernels.py, knobs): qwen4_exp, nemotron_h, lfm2_moe, qwen3_5_moe, flint
    # SPD-249: the layer-major prefill windows, the fused hyper-connection inputs, the attention stages and fewer
    # copies (QWEN38_FN_PF_*, all off, the bits the same), the tensor-core mixer modes (QWEN38_FN_PREFILL_MM, fp32 by
    # default); tools/gdn_prefill_kernels.py's `view` and tools/gpu/base.py's fn_pf_hc: every adapter
    "qwen3_5": "8cda55ed3a16bc084685cdf040f31151800c4dab822497c3452dbab4bae89a1d",
    # the prefill's indexer scores and PLE decode in row tiles (engine/models/flashnext/engine.py): qwen4_exp, qwen3_5_moe
    # SRV-795: each graph captured on its slot's own stream (engine/models/flashnext/engine.py; the same rows, its
    # SAME_ROWS entry): qwen4_exp, qwen3_5_moe
    "qwen4_exp": "9c1f6a5dc12f3f39881123be4ee793cceefb4a1e46012588d81519d5b374f46a",
    "nemotron_h": "0da1cfeb821d1d657f78455f7b325a9b691704efa1202c322a510a58e1788846",
    "lfm2_moe": "e6b9d858da7627b878662c97485e408e639b40ac269e6aef1001ef6bcda97ecd",
    "qwen3_5_moe": "dc5e81cf346b1f9343cab67efaf31e8eebceb325bb127f4397f81524a8803e8d",
    "flint": "a2d781443f467bb6d36583810a148fefedd173a8addb91a1c196b99352989d3b",
}
#: adapters the refactor may move once (the coordinator's OK, 2026-10-22): Flint's knobs leave the shared
#: table and its kernels take the generic launch option
MAY_MOVE = {"flint"}
#: sha256 of SAME_ROWS' entries at v0.14.0 (sorted [path, now, replaced] triples, JSON); SRV-795 adds Flash-Next's
#: engine.py on purpose (its reason in engine/methods/manifest.py)
SAME_ROWS_SHA256 = "4d7ac63adfdf6dc4dbc2cb026fe6752505d68dfefb96a1300fd274edff18fd5b"   # 31 entries


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
    # SAME_ROWS names files by path; the legacy FN_*/Q35M_* knob owners (frozen, nothing new). The FN_* knobs of the
    # shared kernel libraries (tools/fn_kernels.py, tools/fn_prefill_kernels.py: FN_MOE_*, FN_GEMV_*, FN_PDL, ...)
    # live in this table by their library, which Flash-Next and Ornith both read; their owner is that library, never a
    # model package (test_family_knobs_live_in_their_package holds both). SRV-795's SAME_ROWS entry names one more
    "engine/methods/manifest.py": 14, "engine/settings.py": 16,
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
    # a model-prefixed knob in the shared table that is not legacy belongs to a shared kernel library under tools/
    lib = {k: str(v[1]) for k, v in shared.items() if k.startswith(("FN_", "Q35M_")) and k not in LEGACY_FAMILY_KNOBS}
    bad = {k: o for k, o in lib.items() if not (o.startswith("tools/") and o.endswith(".py"))}
    assert not bad, f"a model's knob in engine/settings.py whose owner is no tools/ library: {bad}"
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
    return f"{len(LEGACY_FAMILY_KNOBS)} legacy family knobs in the shared table, none new; {len(lib)} library knobs"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
