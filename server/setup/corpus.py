"""The lookup corpus of each tokenizer (#97): a store of public text for the model the box serves.

A corpus holds token ids (tools/build_corpus.py), so it serves only the tokenizer that wrote it, and the engine
refuses a store whose meta.json records another one (engine/tokfp.py). One `CORPUS` folder therefore holds one
store a tokenizer:

    CORPUS/                                  the store install.sh builds for the 27B (or a store built by hand)
    CORPUS/tokenizers/<16 hex>/              the store of one tokenizer: the first 16 hex digits of the sha256
                                             of its tokenizer.json, the fingerprint the engine checks

`pick` names the store the engine reads: CORPUS itself when its meta.json records this tokenizer, else this
tokenizer's folder when it is built, else CORPUS as before (a store that records no tokenizer is read as this
one's, with a warning). ops/start.sh asks it only when CORPUS/tokenizers exists.

`build` writes this tokenizer's folder when neither CORPUS nor the folder serves it: the installer's recipe, one
English and one German Wikipedia shard (CC BY-SA 4.0) and the Python sources of transformers, torch, triton and
this repository, about 31 million tokens. ops/use-profile.sh runs it after the weights are complete, so the Setup
Guide's install builds the corpus of the model it installs. The shards come from the Hugging Face cache, where
`install.sh --setup` put them (`fetch`); a shard that is not there is fetched, and without the network the store
is built from the local sources alone. A CORPUS that records no tokenizer is left alone: it is read as every
model's, as before this module.

    python -m server.setup.corpus pick  --corpus DIR [--engine-flags FLAGS]
    python -m server.setup.corpus build --corpus DIR [--engine-flags FLAGS] [--no-fetch] [--dry-run]
    python -m server.setup.corpus fetch [--dry-run]

FLAGS is the profile's QSE_ENGINE_FLAGS: its `--model` is the checkpoint whose tokenizer the engine reads; without
one, the 27B's snapshot as engine/config.py resolves it (QWEN38_MODEL).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import sysconfig
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBDIR = "tokenizers"
KEY = "tokenizer_sha256"            # engine/tokfp.py's
# install.sh's recipe (tests/test_lookup_corpus.py holds the two equal)
WIKI_REPO = "wikimedia/wikipedia"
WIKI = (("20231101.en/train-00000-of-00041.parquet", 12_000_000),
        ("20231101.de/train-00000-of-00020.parquet", 7_000_000))
SITE_SOURCES = (("transformers", 6_000_000), ("torch", 5_000_000), ("triton", 1_500_000))
MAX_TOKENS = 40_000_000


def model_dir(engine_flags: str | None) -> str | None:
    """The checkpoint folder whose tokenizer.json the engine fingerprints: `--model` of the profile's engine
    flags, else the 27B's snapshot (engine/config.py resolve_snapshot); None when there is none."""
    try:
        toks = shlex.split(engine_flags or "")
    except ValueError:
        toks = []
    if "--model" in toks:
        i = toks.index("--model")
        return os.path.expanduser(toks[i + 1]) if i + 1 < len(toks) else None
    try:
        from engine.config import resolve_snapshot
        return resolve_snapshot(None)
    except Exception:                                              # noqa: BLE001
        return None


def fingerprint(model: str | None) -> str | None:
    from engine.tokfp import fingerprint as fp
    return fp(model)


def store_dir(corpus: str, fp: str) -> str:
    return os.path.join(corpus, SUBDIR, fp[:16])


def _meta(path: str) -> dict | None:
    """A store's meta.json, or None when the folder holds no complete store."""
    try:
        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not all(os.path.isfile(os.path.join(path, f)) for f in ("tokens.npy", "sa.npy")):
        return None
    return meta


def pick(corpus: str | None, fp: str | None) -> str:
    """The store the engine reads for tokenizer `fp`: CORPUS when it records `fp`, else `fp`'s own folder when
    it is built, else CORPUS unchanged."""
    if not corpus or not fp:
        return corpus or ""
    root = _meta(corpus)
    if root is not None and root.get(KEY) == fp:
        return corpus
    own = store_dir(corpus, fp)
    mine = _meta(own)
    if mine is not None and mine.get(KEY) == fp:
        return own
    return corpus


def needs_build(corpus: str | None, fp: str | None) -> tuple[bool, str]:
    """Whether `build` writes `fp`'s folder, and why (one line for the log)."""
    if not corpus:
        return False, "CORPUS is empty: the lookup reads the conversation only"
    if not fp:
        return False, "the checkpoint has no tokenizer.json: no store can name its tokenizer"
    root = _meta(corpus)
    if root is not None and root.get(KEY) == fp:
        return False, f"{corpus} is this tokenizer's store"
    if root is not None and root.get(KEY) is None:
        return False, f"{corpus} records no tokenizer and is read as this one's, as before"
    own = store_dir(corpus, fp)
    mine = _meta(own)
    if mine is not None and mine.get(KEY) == fp:
        return False, f"{own} is built ({mine.get('n_tokens', 0):,} tokens)"
    return True, f"{own} is not built"


def shards(fetch: bool = True, log=print) -> list[tuple[str, int]]:
    """The Wikipedia shards in the Hugging Face cache, (path, token budget) each; a missing one is fetched when
    `fetch`, and one that cannot be had is left out."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log("[corpus] huggingface_hub is not installed: the store goes without the Wikipedia shards")
        return []
    out = []
    for name, budget in WIKI:
        try:
            out.append((hf_hub_download(WIKI_REPO, name, repo_type="dataset", local_files_only=True), budget))
            continue
        except Exception:                                          # noqa: BLE001
            pass
        if not fetch:
            log(f"[corpus] {WIKI_REPO} {name} is not in the Hugging Face cache: left out")
            continue
        try:
            log(f"[corpus] fetching {WIKI_REPO} {name} (CC BY-SA 4.0)")
            out.append((hf_hub_download(WIKI_REPO, name, repo_type="dataset"), budget))
        except Exception as exc:                                   # noqa: BLE001
            log(f"[corpus] {WIKI_REPO} {name} could not be fetched ({type(exc).__name__}): left out")
    return out


def build_args(out: str, parquets: list[tuple[str, int]], site: str | None = None) -> list[str]:
    """tools/build_corpus.py's arguments for the installer's recipe into `out`."""
    site = site or sysconfig.get_paths()["purelib"]
    args = ["--out", out]
    for path, budget in parquets:
        args += ["--parquet", f"{path}:text::{budget}"]
    for pkg, budget in SITE_SOURCES:
        args += ["--src", f"{os.path.join(site, pkg)}:{budget}"]
    args += ["--src", os.path.join(ROOT, "engine"), "--src", os.path.join(ROOT, "tools"),
             "--max-tokens", str(MAX_TOKENS)]
    return args


def _run_builder(model: str, args: list[str]) -> None:
    """tools/build_corpus.py over `args`, tokenising with `model`'s tokenizer.json. The tool reads the
    checkpoint's config through the 27B's loader only to find its folder, so another family's checkpoint is
    passed as that folder; the store records the same fingerprint the engine computes for it."""
    from types import SimpleNamespace
    sys.path.insert(0, ROOT)
    import tools.build_corpus as bc
    real, argv = bc.load_config, sys.argv
    bc.load_config = lambda _path=None: SimpleNamespace(path=model)
    sys.argv = ["tools/build_corpus.py", *args]
    try:
        bc.main()
    finally:
        bc.load_config, sys.argv = real, argv


def build(corpus: str | None, model: str | None, *, fetch: bool = True, dry_run: bool = False,
          log=print, runner=_run_builder) -> int:
    """Build `model`'s tokenizer's store under `corpus` when it needs one (`needs_build`); 0 when it is there
    or was built, 1 when the build failed (nothing is left half-written)."""
    fp = fingerprint(model)
    if dry_run and corpus and fp is None and model and not os.path.isdir(model):
        log(f"[corpus] dry run: would build the store of {model}'s tokenizer once its files are on the box")
        return 0
    todo, why = needs_build(corpus, fp)
    if not todo:
        log(f"[corpus] {why}")
        return 0
    out = store_dir(corpus, fp)
    if dry_run:
        names = ", ".join(n for n, _ in WIKI)
        log(f"[corpus] dry run: would build {out} for {model}: {WIKI_REPO} {names} (from the Hugging Face cache, "
            f"else fetched), the Python sources of {', '.join(p for p, _ in SITE_SOURCES)}, engine/ and tools/")
        return 0
    log(f"[corpus] building {out} for {model} (CPU only, a few minutes)")
    t0 = time.time()
    parts = shards(fetch, log)
    partial = out + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    try:
        runner(model, build_args(partial, parts))
        meta = _meta(partial)
        if meta is None or meta.get(KEY) != fp:
            raise RuntimeError(f"the build wrote no store for this tokenizer in {partial}")
        shutil.rmtree(out, ignore_errors=True)
        os.replace(partial, out)
    except (Exception, SystemExit) as exc:                         # noqa: BLE001
        shutil.rmtree(partial, ignore_errors=True)
        log(f"[corpus] the build failed ({type(exc).__name__}: {exc}); the lookup reads the conversation only")
        return 1
    log(f"[corpus] built {out}: {meta.get('n_tokens', 0):,} tokens in {time.time() - t0:.0f} s"
        f"{'' if len(parts) == len(WIKI) else ', without the Wikipedia shards it could not get'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m server.setup.corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="print the store the engine reads")
    p.add_argument("--corpus", default="")
    p.add_argument("--engine-flags", default="")
    b = sub.add_parser("build", help="build this tokenizer's store when it has none")
    b.add_argument("--corpus", default="")
    b.add_argument("--engine-flags", default="")
    b.add_argument("--no-fetch", action="store_true", help="only the shards in the Hugging Face cache")
    b.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fetch", help="the Wikipedia shards into the Hugging Face cache")
    f.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "pick":
        try:
            print(pick(a.corpus, fingerprint(model_dir(a.engine_flags))))
        except Exception:                                          # noqa: BLE001
            print(a.corpus)                                        # never stops a start
        return 0
    if a.cmd == "build":
        return build(a.corpus, model_dir(a.engine_flags), fetch=not a.no_fetch, dry_run=a.dry_run)
    if a.dry_run:
        print(f"[corpus] dry run: would fetch {WIKI_REPO} " + ", ".join(n for n, _ in WIKI) + " (1.2 GB)")
        return 0
    return 0 if len(shards(True)) == len(WIKI) else 1


if __name__ == "__main__":
    sys.exit(main())
