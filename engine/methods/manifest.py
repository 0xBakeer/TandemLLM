"""Fingerprint v3 (ENG-243): what a model's stored rows depend on, as the model's adapter lists it.

The disk tier's fingerprint v2 (`engine/disk_cache.fingerprint`) hashes every file under `engine/`
but a few, so moving method code -- StairCut, the router, sampling, the caches, the server's loop --
sweeps every saved conversation, and it misses the tile tables that set the kernels' summation order
(a knob names the file; v2 hashed the name, never the table). v3 hashes the adapter's `manifest()`:

  modules   the files whose code computes stored rows (forward, verify and commit bodies, the
            drafters whose state is cached), and the `tools/` modules they import, transitively;
            with each, the C++ and CUDA sources it compiles (`csrc/`, engine/native_build.py)
  knobs     the defaults of the knobs those files own (engine/settings.py `KNOBS`); a value set in
            the environment is in the caller's `extra`, as under v2
  data      the tables those files read that set the arithmetic (tile tables), by content: the
            manifest's own, every .json a knob of theirs names, a module's default table
  extra     the weights, the environment, the grid, the libraries (server/diskcache.py)

Method code is not in it: it chooses which rows are computed, never how.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
from dataclasses import dataclass

from engine import disk_cache as dc
from engine import native_build

#: fp4 (rc-0.8.0): fp3's hash over the server's extra by content (server/diskcache.py
#: `fingerprint_extra`), where fp3's named files and tables by path; fp3 is still computed, by path,
#: to adopt the files the builds before wrote (`server/diskcache.py legacy_fingerprints`)
FORMULA = "fp4"

#: 0.8.2: module bytes that changed without changing a stored row on any profile the bytes they
#: replace served: (repo-relative path, sha256 of the bytes now) -> sha256 of the bytes replaced.
#: The fingerprints read the bytes as they are; the legacy ones (server/diskcache.py
#: `legacy_fingerprints`) also read them with the replaced bytes, and the first start under
#: `adoption()` adopts the files written under those, once (engine/disk_cache.py `equivalents`).
#: #1 (0xWhiteMage, a6f2142c): engine/loader.py loads a standalone EXL3 pack as --model (its
#: config.json says exl3 and no QWEN38_EXL3 overlay is set); a 27B profile sets QWEN38_EXL3 or serves
#: NVFP4 and takes the path it took before, reading the same tensors (the shards now opened at once)
#: ENG-285 (LFM2.5): the shared EXL3 kernels decode half-integer rates (1.5, 2.5, 3.5 bits) beside the whole
#: ones; a whole rate's code is the same (its tile width 8 K words, no new branch taken), and on the Spark
#: every dense, skinny-GEMM, slab, MoE and prefill-MoE output at rates 2..8 equalled the tree before's bit
#: for bit (the same random blocks and inputs through both trees)
#: SPD-125: tools/fn_kernels.py adds `dense_rows` (a verify block's EXL3 projections, opt-in) beside the
#: kernels it had; their code is the same, and on the Spark `dense` at 1..32 rows at Bolt's, the 27B's and
#: Flash-Next's shapes equalled 0.9.0-0.9.2's byte for byte (tools/fn_dense_rows_check.py same)
#: ENG-381 (ENG-363, 0.11.2): engine/models/lfm2_moe/engine.py keeps at most 32 joint decode graphs (which
#: graphs are kept, never what one computes) and kernels.py's `attn_rows` sizes its split partials for a
#: class's largest row count and reads the first rows' share (the same kernel and grid); on the Spark
#: `attn_rows` at 1..32 rows, two classes, lone and with shared buffers, equalled 0.11.1's bit for bit
#: (tools/lfm_attn_rows_check.py)
#: OPS-215 (ENG-383): tools/fn_kernels.py's router takes up to 512 experts, with a narrower instantiation for
#: up to 256; before it refused any count but 512, and at 512 it launches the same instantiation with the
#: same arguments, so no row a released profile stored changes (by the code; not run on the GPU)
#: OPS-219: engine/models/flashnext/ref.py's module docstring loses half a sentence; no code changes
#: SRV-673: engine/models/qwen3_5_moe/engine.py's prompt chunks and MTP primes call kernels.py's new
#: `prompt_attention`: SDPA over the KV heads as they are (`enable_gqa`) on the flash kernel in place of a
#: copy repeated per query head (still the path where flash refuses); on the Spark its output equalled the
#: repeated copy's bit for bit (tests/test_q35m_kernels.py), and a 179,678-token request wrote the same
#: text plain and drafted, before and after
#: SRV-681: SRV-673's `prompt_attention` gave other bits for a block of one query row (a chunk of one row,
#: or one of 1,025, 2,049 or 3,073 rows): with one row the flash kernel folds a KV head's query heads into
#: rows. Such a block now runs as two rows, the row twice, and keeps the second; every block equals 0.13.0's
#: repeated copy bit for bit on the Spark (tests/test_prompt_gqa.py, blocks of 1 to 128 rows, 0 to 3,000
#: cached rows). The entry names SRV-673's bytes, which name 0.13.0's: a start adopts both trees' files
#: (SRV-673's differ from 0.13.0's only after a one-row block). engine/models/lfm2_moe/engine.py's prompt
#: chunks attend over the KV heads as they are the same way (`chunk_attention`); its output equalled the
#: copy's bit for bit at Hq 32, Hk 8, D 64 for chunks of 1 to 600 rows after 0 to 5,000 cached rows
#: OPS-225: comments and docstrings that named other inference engines reworded, no code change (each
#: file's AST without its docstrings equals the replaced bytes'), in engine/drafters/dflash2.py,
#: engine/penalty.py, engine/sample.py, engine/models/flashnext/ref.py, engine/models/nemotron_h/{engine,
#: gguf, gguf_load, ref}.py and tools/nvfp4_linear_v2.py; tools/iq2_linear.py's refusal message for a scale
#: byte 0xFF reworded; dflash2.py's `_grouped_conv` written again from its formula, whose outputs equal the
#: replaced function's bit for bit on the CPU and the GPU (fp32, bf16, fp16; 1 to 4 taps; one and two blocks)
#: Entries stay when their bytes are replaced again: a key may be another entry's replaced bytes, and
#: `same_variants` follows the chain (ENG-341)
#: 0.15.0 (SPD-201 to SPD-245): no entry across 0.15.0. The speed setup's knob defaults, the joint passes,
#: the 27B's prefill orders and Flash-Next's tensor-core mixers moved every family's stored rows or their
#: fingerprint, so no model adopts a file from before 0.15.0; the entries above only reach older trees
#: (tests/test_fp_same_rows.py)
#: ENG-411: no entry for the kernel sources' move into csrc/ files (the compiled text is 0.15.0's byte for
#: byte). An entry reads a module's bytes as the bytes they replaced; it cannot count a file that a tree did
#: not have as absent; every family whose manifest reaches a module with moved sources reads new files, so
#: its fingerprint moves and 0.15.0's cache files are not adopted
SAME_ROWS = {("engine/loader.py", "823f9dd4c230ec45d133b52e0a5edc0343098184526d9dfcad5ee4271f933d97"):
             "fa9277e004043d172d29b2dc2b8b7958b4b7b2834bc26a73e6130bb0f0d05441",
             ("tools/exl3_linear.py", "a75b6a3f1fd36ff9ecafe79c09cb797106ce48f53edc73b10d4ef19a0fe3c40f"):
             "5078b33c3d03eac42706d3b05650b0604f2ae96d267db2c996f59409929a3a48",
             ("tools/fn_kernels.py", "152025a29380f1bc9c0a1d93b2382531260b443cbe542549f332fa245ac23b7f"):
             "26139deda6a564355375158270ddedd8ba7acbbf16c97aaf2b5646c6c8678e93",
             ("tools/fn_kernels.py", "26139deda6a564355375158270ddedd8ba7acbbf16c97aaf2b5646c6c8678e93"):
             "630236e58c8230010d683b22604bfb97083ad932bfa36847ab048ef0ec5af55a",
             ("tools/fn_kernels.py", "85e31c84a0709bfaec8f40c195354f29c225ed72defe4df85a5ad0a175a985e8"):
             "152025a29380f1bc9c0a1d93b2382531260b443cbe542549f332fa245ac23b7f",
             ("engine/models/lfm2_moe/engine.py", "76bfb598fee3b1d731a51bd4f4ceb4afc8bb30ee8388ed4ca9551de82d7d388b"):
             "2360b32729713510020bd47ed86e6c185a40f54b22c622fc33c4fa7e6646783e",
             ("engine/models/lfm2_moe/kernels.py", "8f2e42e559a134a9992729497524b6fa5d57f6cb08378e56ed3d0b4fa8f02dea"):
             "dbfbfd3aa2cc532ecca1d2ce0b5f23d6d48d68c7bfea6463298113fe8a49d754",
             ("tools/fn_prefill_kernels.py", "653b447087545778f29eb856e12681d0d1229847ada9ec7291477aeda8c54106"):
             "13fd5c1579579b11f94af618835798548216881173821cbb6447b3e6b21ec987",
             # SRV-619: the answer constraint's union and the call markers; a mask on the sampled
             # token, never a stored row
             ("engine/grammar.py", "ddcb9cf828c5e89c80a29c4ff25b20fac4a5d6230bfa4248a70893a94a97778b"):
             "c7fbee9ad0f1526b7c30df798e8c178c3a2185244ab7a2b29e44eed5ffe50749",
             # SRV-647: the masks keyed by a serial of the grammar, not its id(); no row changes
             ("engine/grammar.py", "9c6617a799bd64b1fdbc3cfc23a2eba5b2ee471b64d1d21b4da7dc23adf59c65"):
             "c7fbee9ad0f1526b7c30df798e8c178c3a2185244ab7a2b29e44eed5ffe50749",
             # SRV-689: tool_choice's calls in the arg-key form for a model that writes it; a mask
             # on the sampled token, never a stored row
             ("engine/grammar.py", "7d08c32d4841f5bbad40e4d55c22936c4b357dc854b3f848538d8efa634cfc8a"):
             "c7fbee9ad0f1526b7c30df798e8c178c3a2185244ab7a2b29e44eed5ffe50749",
             ("engine/models/flashnext/ref.py", "497c681a0b2e8d33e45a3bb2d4f565b5983ca2edc9da021619048bdb258267c4"):
             "f814ce162695f81e1ff8d7b80eb8c90619837f99e6cebb35f10bcb39983bf3f4",
             # OPS-225: the files' code unchanged (comments, docstrings, one message); dflash2.py's
             # grouped convolution the same bits
             ("engine/drafters/dflash2.py", "62e399ac5e844be6aff9ea49a00da435e379547333ed1e8d02df9c29e9b46f30"):
             "a371ec2d68005675666fe7613365f9e0b93ae5e1ac5f5e7c8fa78bf8abdcf095",
             ("engine/penalty.py", "afc9c329b1598b40e091a225eceb6123b2284bf4388cdcd5285c7731da9735d7"):
             "963b17f583fb80639345702142fe4dac14f9b3dfbe20274e0908d69a20dfebf2",
             ("engine/sample.py", "8996932fd6652b60a1e7688a478b99d089592e764889e98affc96f1a53e127f3"):
             "8b14156ed6f3fd858ed6949e69ffcc51830efd03d7291370f3f17a3ce9f0b1bf",
             ("tools/nvfp4_linear_v2.py", "9b63f4295a0de60c1b1bb49398ef4a2212da93621ae55da9e50b2771000fdf2d"):
             "6d1277865066910afa61f4ef34d1312677fc6eeaa2e1612d0866c498d1752c5e",
             ("engine/models/flashnext/ref.py", "78207bf3ffa9bac8dc3e59d3dab3b52999c39b54210cd389a103c98ae66e69ea"):
             "497c681a0b2e8d33e45a3bb2d4f565b5983ca2edc9da021619048bdb258267c4",
             ("engine/models/nemotron_h/engine.py", "d124dfdad22a4fdba21385a480d64cc64aeeebc1b95afad49ea189098d5119b0"):
             "fc05d2ea1ebaa234e26b4385f115e590de7404b68ffc17d0b1d212066b826277",
             ("engine/models/nemotron_h/gguf.py", "4dde9b6bb626cccc0521533313f31a848b31894430d4f84fe1b5abfb02705d96"):
             "0af3421c894d88be3b1dc657ec629a4c2853586e51ef71c33f51c02e6f87a91b",
             # ENG-399, SPD-142/145: Flint's edits to shared files leave every other model's rows as they were:
             # LFM2's add_rms and qk_prep take an opt-in programmatic launch (off unless a caller passes it), the
             # tree drafter an opt-in lookup back-off (0 unless set), the settings and the arch table new keys
             ("engine/models/lfm2_moe/kernels.py", "d2f45d58cecc6b99a0dfa8fc6f86c914544644b345bc51a0e861dc2d45665afd"):
             "8f2e42e559a134a9992729497524b6fa5d57f6cb08378e56ed3d0b4fa8f02dea",
             ("engine/methods/mtp_drafter.py", "8033612434e1e2576247bff40def9056fb570eef082515c005eaeaa19e78afad"):
             "4f577dc55c73388034a9563c6d0596b01213a79d0b863768094762dcdd93a546",
             ("engine/settings.py", "dbe32ab2408eb7c93fbadfec657ff86529b1b5d1e9430d69ba9f1ad21aeb5442"):
             "eb49205d754b25cc481e6bb4dca13164d60bed85e2c5dae187b9b56af8a0224b",
             ("engine/models/__init__.py", "9ba29c2a5e7f55baca11bfab71433b27371d8dbdb06dae40aee9e529421c2926"):
             "c5fa4cb7441e29bff12eff553cb80874b945d8ac0f32da2bcb828fe7a7fea3ab",
             ("engine/models/nemotron_h/gguf_load.py", "cd515152dfbb7216fecae20d2f38b3bc3c21cb789a755d04d052be9d3f0136af"):
             "2e6b8c98a0cc24c93a1a6f11b343590502120b50318064bc392733637f470805",
             ("engine/models/nemotron_h/ref.py", "9543b02c3e1ba3d0e1292b3742d3b738c145f91ea361e99e1155f18cfeb3d864"):
             "c1cceb15d0a89d59892a7e1373e232895fd1507dee87139c2bf04e39d066f294",
             ("tools/iq2_linear.py", "70a766dbf8775e6292ca7aa05051264101e9560ba5b197f87ac1d74cfad761c1"):
             "863ea0a0fbf36d26d621b5f3467cb6463f26459ee221b6085f59d5acff1e22c0",
             ("engine/models/qwen3_5_moe/engine.py", "77fa0e939417f8a7a3f4402b9e65f9804544f91adbcf752e19c4a055e28b5520"):
             "78fcdad2522b216c6d7f07c7cca8a6a065a79104231544eb01a8e60c839044fd",
             ("engine/models/qwen3_5_moe/kernels.py", "2bc18f6e41b6f483360c5e0215e5df6f30da5ea254bfed026d212164dfc66331"):
             "d6fa493c51d8746cc8b5a974bbc5e636c034894ffbd86a8e2ef8b4caaf5487d7",
             ("engine/models/qwen3_5_moe/kernels.py", "2ecc9dcb27f12c9a9e4738e3ff9c58016e82039854680304adb22193321167cc"):
             "2bc18f6e41b6f483360c5e0215e5df6f30da5ea254bfed026d212164dfc66331",
             ("engine/models/lfm2_moe/engine.py", "ab1a6b3f3d1f73cf8b2ad336c5c486582a0683d94f35d456e9538f61bcc14eac"):
             "76bfb598fee3b1d731a51bd4f4ceb4afc8bb30ee8388ed4ca9551de82d7d388b"}


def adoption() -> str:
    """The key `equivalents` records an adoption under: the formula, and `SAME_ROWS`' entries, so
    a tree that adds one adopts once more on a directory that already adopted under the formula."""
    if not SAME_ROWS:
        return FORMULA
    rows = sorted([k[0], k[1], v] for k, v in SAME_ROWS.items())
    return f"{FORMULA}+same-{hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class Manifest:
    arch: str                   # the registry name (engine/models: qwen3_5, qwen4_exp)
    modules: tuple              # repo-relative files that compute stored rows
    data: tuple = ()            # tables they read that set the arithmetic (repo-relative or absolute)


def files(root: str, m: Manifest) -> list[str]:
    """The manifest's modules and the `tools/` modules they import, each followed by the C++ and CUDA
    sources it compiles (engine/native_build.py: `<dir>/csrc/<name>/`), then the reader of those sources
    when there are any (absolute paths, in order; ENG-411)."""
    mods = [os.path.join(root, p) for p in m.modules]
    out, native = [], False
    for p in mods + dc._tools_modules(root, mods):
        src = native_build.sources(p)
        out += [p] + src
        native = native or bool(src)
    reader = os.path.join(root, "engine", "native_build.py")
    if native and reader not in out:
        out.append(reader)
    return out


def same_variants(root: str, m: Manifest) -> list[dict]:
    """`same` tables, one a combination of the bytes `m`'s files had (ENG-341): each file whose bytes
    `SAME_ROWS` names as its own bytes or as any bytes they replaced, down the chain, the others as
    they are. `SAME_ROWS` itself reads every named file one step back, which is no release's tree once
    a release replaced one file's bytes and kept another's (0.9.x: fn_kernels.py's 0.9 bytes beside
    exl3_linear.py's, which replaced 0.8.2's); one of these tables is that tree's. Without the one
    that changes nothing."""
    choices = []
    for p in files(root, m):
        rel = os.path.relpath(p, root)
        if rel == dc.SETTINGS_FILE:
            continue
        with open(p, "rb") as f:
            now = hashlib.sha256(f.read()).hexdigest()
        olds, cur = [], now
        while (rel, cur) in SAME_ROWS and SAME_ROWS[(rel, cur)] not in olds:
            cur = SAME_ROWS[(rel, cur)]
            olds.append(cur)
        if olds:
            choices.append([None] + [((rel, now), old) for old in olds])
    return [dict(c for c in combo if c) for combo in itertools.product(*choices)][1:]


#: tables a module reads when its knob names none (the knob's value, when set, is read instead)
DEFAULT_TABLES = {"tools/exl3_linear.py": ("EXL3_TILES", os.path.join("ops", "exl3-tiles-fused.json"))}


def _knob_table(settings_path: str) -> dict:
    with open(settings_path, "rb") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == "KNOBS" and node.value is not None:
            return ast.literal_eval(node.value)
    raise ValueError(f"{settings_path} has no literal KNOBS table")


def knobs_of(settings_path: str, owners: set) -> bytes:
    """(name, default) of every knob in `settings_path` whose owner is one of `owners`, in name order."""
    knobs = _knob_table(settings_path)
    return json.dumps(sorted((str(k), v[0]) for k, v in knobs.items()
                             if len(v) > 1 and str(v[1]) in owners), default=str).encode()


def tables(root: str, owners: set, env=None) -> list[str]:
    """The tables the owners read: a knob of theirs whose value (the environment's, else its default)
    names a .json file, and a module's own default table when its knob is unset (`DEFAULT_TABLES`)."""
    env = os.environ if env is None else env
    knobs = dict(_knob_table(os.path.join(root, dc.SETTINGS_FILE)))
    for own in sorted(owners):
        if own.endswith("/knobs.py"):
            knobs.update(_knob_table(os.path.join(root, own)))   # a family's own table (ENG-405)
    out = set()
    for k, v in knobs.items():
        if len(v) > 1 and str(v[1]) in owners:
            val = env.get("QWEN38_" + k, v[0])
            if isinstance(val, str) and val.endswith(".json"):
                out.add(val)
    for mod, (k, path) in DEFAULT_TABLES.items():
        if mod in owners and not env.get("QWEN38_" + k, knobs.get(k, (None,))[0]):
            out.add(path)
    return sorted(out)


def fingerprint(root: str, m: Manifest, extra: dict, formula: str = FORMULA,
                same: dict | None = None) -> str:
    """`same` (`SAME_ROWS`): a module whose bytes it names counts as the bytes it replaced."""
    h = hashlib.sha256()
    h.update(formula.encode())
    h.update(m.arch.encode())
    paths = files(root, m)
    owners = {os.path.relpath(p, root) for p in paths}
    for p in paths:
        rel = os.path.relpath(p, root)
        h.update(rel.encode())
        if rel == dc.SETTINGS_FILE:
            data = knobs_of(p, owners)
        else:
            with open(p, "rb") as f:
                data = f.read()
        h.update(dc.module_digest(rel, data, same))
    for d in sorted(set(m.data) | set(tables(root, owners))):
        full = os.path.expanduser(d) if os.path.isabs(os.path.expanduser(d)) else os.path.join(root, d)
        rel = os.path.relpath(full, root)
        h.update((full if rel.startswith("..") else rel).encode())
        try:
            with open(full, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"missing")
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()
