"""The lookup corpus of each tokenizer (#97, server/setup/corpus.py): one store a tokenizer under
CORPUS/tokenizers/<fingerprint>, which ops/use-profile.sh builds after the weights and ops/start.sh hands the
engine.

  1  pick: CORPUS when it records this tokenizer, else this tokenizer's folder once built, else CORPUS as
     before (a store that records none, another tokenizer's, or no store at all)
                                                        test_pick_and_when_to_build
  2  build: the store goes into place whole, records the fingerprint the engine checks, and a failed build
     leaves nothing behind                              test_build_is_whole_or_nothing
  3  a checkpoint of another family (no 27B config): the real builder tokenises with its tokenizer.json, and
     the engine's store loader takes the picked store under that checkpoint's fingerprint
                                                        test_a_store_for_another_family_loads_in_the_engine
  4  the profile's checkpoint: --model of QSE_ENGINE_FLAGS, else the 27B's snapshot
                                                        test_the_model_folder_of_a_profile
  5  install.sh's recipe and this module's are one     test_the_installer_and_the_module_share_the_recipe
  6  ops/use-profile.sh --dry-run names the store it would build, and a built one it leaves alone
                                                        test_use_profile_builds_the_profiles_store

Run: python tests/test_lookup_corpus.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from server.setup import corpus as lc  # noqa: E402
from engine.tokfp import fingerprint  # noqa: E402


def _tokenizer(folder: str, words: list[str]) -> str:
    """A small word-level tokenizer.json in `folder`; its fingerprint."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    os.makedirs(folder, exist_ok=True)
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.save(os.path.join(folder, "tokenizer.json"))
    return fingerprint(folder)


def _store(path: str, fp: str | None, n: int = 8) -> None:
    from tools.build_corpus import build_suffix_array, write_store
    toks = np.arange(1, n + 1, dtype=np.int32)
    meta = {"max_order": 8}
    if fp is not None:
        meta[lc.KEY] = fp
    write_store(path, toks, build_suffix_array(toks), meta)


def test_pick_and_when_to_build():
    with tempfile.TemporaryDirectory() as d:
        a = _tokenizer(os.path.join(d, "a"), ["x", "y"])
        b = _tokenizer(os.path.join(d, "b"), ["p", "q", "r"])
        assert a != b
        corpus = os.path.join(d, "corpus")
        # no store at all (install.sh --setup): CORPUS as it is, and a build is due
        assert lc.pick(corpus, a) == corpus and lc.needs_build(corpus, a)[0]
        assert lc.pick("", a) == "" and not lc.needs_build("", a)[0]
        assert not lc.needs_build(corpus, None)[0]
        # install.sh's store, recorded for tokenizer a: a reads it and builds nothing; b needs its own
        _store(corpus, a)
        assert lc.pick(corpus, a) == corpus and not lc.needs_build(corpus, a)[0]
        assert lc.pick(corpus, b) == corpus and lc.needs_build(corpus, b)[0]
        _store(lc.store_dir(corpus, b), b)
        assert lc.store_dir(corpus, b) == os.path.join(corpus, "tokenizers", b[:16])
        assert lc.pick(corpus, b) == lc.store_dir(corpus, b) and not lc.needs_build(corpus, b)[0]
        assert lc.pick(corpus, a) == corpus
        # a folder whose store records another tokenizer, or is half-written, is not taken
        os.remove(os.path.join(lc.store_dir(corpus, b), "sa.npy"))
        assert lc.pick(corpus, b) == corpus and lc.needs_build(corpus, b)[0]
        # a store that records no tokenizer (built before the fingerprint) is read as every model's, as before:
        # nothing is built beside it
        shutil.rmtree(corpus)
        _store(corpus, None)
        assert lc.pick(corpus, b) == corpus and not lc.needs_build(corpus, b)[0]
    return "6 states"


def test_build_is_whole_or_nothing():
    with tempfile.TemporaryDirectory() as d:
        fp = _tokenizer(os.path.join(d, "m"), ["x"])
        corpus = os.path.join(d, "corpus")
        lines = []
        seen = {}

        def runner(model, args):
            out = args[args.index("--out") + 1]
            seen.update(model=model, out=out, args=args)
            _store(out, fp)

        assert lc.build(corpus, os.path.join(d, "m"), fetch=False, log=lines.append, runner=runner) == 0
        own = lc.store_dir(corpus, fp)
        assert seen["out"] == own + ".partial" and seen["model"] == os.path.join(d, "m")
        assert lc.pick(corpus, fp) == own and not os.path.exists(own + ".partial"), os.listdir(corpus)
        # the installer's recipe: the site packages' and the repository's sources, the token cap
        a = seen["args"]
        assert a[a.index("--max-tokens") + 1] == str(lc.MAX_TOKENS)
        assert [s.rsplit(":", 1)[-1] for s in a if s.endswith(("transformers:6000000", "torch:5000000",
                                                               "triton:1500000"))] == ["6000000", "5000000", "1500000"]
        assert os.path.join(ROOT, "engine") in a and os.path.join(ROOT, "tools") in a
        # built: the next call does nothing
        assert lc.build(corpus, os.path.join(d, "m"), fetch=False, log=lines.append, runner=runner) == 0
        assert "is built" in lines[-1], lines
        # a failing build, or one that records another tokenizer, leaves no folder behind
        shutil.rmtree(own)

        def broken(model, args):
            _store(args[args.index("--out") + 1], "f" * 64)

        def raising(model, args):
            os.makedirs(args[args.index("--out") + 1])
            raise SystemExit("nothing to build from; pass --src")

        for bad in (broken, raising):
            assert lc.build(corpus, os.path.join(d, "m"), fetch=False, log=lines.append, runner=bad) == 1
            assert not os.path.exists(own) and not os.path.exists(own + ".partial")
            assert "the build failed" in lines[-1], lines
        # a dry run names the store and builds nothing
        assert lc.build(corpus, os.path.join(d, "m"), dry_run=True, log=lines.append, runner=raising) == 0
        assert f"would build {own}" in lines[-1] and not os.path.exists(own), lines
    return "built, kept, refused twice"


def test_a_store_for_another_family_loads_in_the_engine():
    from engine.drafters.ngram import CorpusSuffixStore
    with tempfile.TemporaryDirectory() as d:
        model = os.path.join(d, "flint-like")           # a tokenizer.json and no 27B config.json
        words = "the lookup copies text it has seen before".split()
        fp = _tokenizer(model, words)
        src = os.path.join(d, "src")
        os.makedirs(src)
        with open(os.path.join(src, "a.md"), "w") as fh:
            fh.write(" ".join(words * 20))
        corpus = os.path.join(d, "corpus")
        _store(corpus, "0" * 64)                         # the 27B's store, another tokenizer
        lc._run_builder(model, ["--out", lc.store_dir(corpus, fp), "--src", src])
        got = lc.pick(corpus, fp)
        assert got == lc.store_dir(corpus, fp), got
        store = CorpusSuffixStore.load(got, fp)
        assert store is not None and store.n == 20 * len(words) + 1, store and store.n
        assert store.meta[lc.KEY] == fp and int(store.tokens[0]) == 1
        assert CorpusSuffixStore.load(corpus, fp) is None   # another tokenizer's root store is skipped (#105)
    return f"{20 * len(words) + 1} tokens"


def test_the_model_folder_of_a_profile():
    assert lc.model_dir("--model /p/flint-exl3-4.0 --drafter mtp") == "/p/flint-exl3-4.0"
    assert lc.model_dir("--model ~/m --drafter mtp").startswith(os.path.expanduser("~"))
    assert lc.model_dir("--drafter mtp --model") is None
    with tempfile.TemporaryDirectory() as d:
        snap = os.path.join(d, "snapshots", "abc")
        os.makedirs(snap)
        open(os.path.join(snap, "config.json"), "w").write("{}")
        out = subprocess.run([sys.executable, "-c", "from server.setup import corpus; print(corpus.model_dir(''))"],
                             cwd=ROOT, capture_output=True, text=True,
                             env=dict(os.environ, QWEN38_MODEL=os.path.join(d, "snapshots")))
        assert out.stdout.strip() == snap, out.stdout + out.stderr
    out = subprocess.run([sys.executable, "-m", "server.setup.corpus", "pick", "--corpus", "/nowhere/corpus",
                          "--engine-flags", "--model /nowhere/model"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "/nowhere/corpus", out.stdout + out.stderr


def test_the_installer_and_the_module_share_the_recipe():
    text = open(os.path.join(ROOT, "install.sh")).read()
    vals = dict(re.findall(r'^(WIKI_REPO|WIKI_EN|WIKI_DE)="([^"]+)"$', text, re.M))
    assert vals == {"WIKI_REPO": lc.WIKI_REPO, "WIKI_EN": lc.WIKI[0][0], "WIKI_DE": lc.WIKI[1][0]}, vals
    line = re.search(r"tools/build_corpus\.py --model .*?--max-tokens (\d+)\)", text, re.S)
    assert line and int(line.group(1)) == lc.MAX_TOKENS
    body = line.group(0)
    assert f'"$EN_PQ:text::{lc.WIKI[0][1]}"' in body and f'"$DE_PQ:text::{lc.WIKI[1][1]}"' in body, body
    for pkg, budget in lc.SITE_SOURCES:
        assert f'"$SITE/{pkg}:{budget}"' in body, (pkg, body)
    # --setup fetches the shards and leaves CORPUS to the profile's build
    assert "server.setup.corpus fetch" in text and "CORPUS_MODE=shards" in text


def test_use_profile_builds_the_profiles_store():
    with tempfile.TemporaryDirectory() as home:
        repo = os.path.join(home, "repo")
        os.makedirs(os.path.join(repo, "ops"))
        shutil.copy(os.path.join(ROOT, "ops/use-profile.sh"), os.path.join(repo, "ops"))
        shutil.copy(os.path.join(ROOT, "ops/fetch_weights.py"), os.path.join(repo, "ops"))
        shutil.copytree(os.path.join(ROOT, "ops/profiles"), os.path.join(repo, "ops/profiles"))
        for pkg in ("server", "engine", "tools"):
            os.symlink(os.path.join(ROOT, pkg), os.path.join(repo, pkg))
        corpus = os.path.join(home, "corpus")
        tail = open(os.path.join(ROOT, "ops/serve.env")).read().split("# Weight profile (ops/use-profile.sh)")[1]
        with open(os.path.join(repo, "ops/serve.env"), "w") as fh:
            fh.write(f"REPO={repo}\nPY={sys.executable}\nPORT=18995\nCORPUS={corpus}\n# Weight profile{tail}")
        pack = os.path.join(home, "flashnext-packs/3.05")     # a complete pack on disk, with its tokenizer
        os.makedirs(pack)
        for n, body in (("config.json", "{}"), ("model.safetensors.index.json", '{"weight_map": {}}'),
                        ("ngram_embedding.safetensors", "t")):
            open(os.path.join(pack, n), "w").write(body)
        fp = _tokenizer(pack, ["x", "y"])
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "MEM_NEED_GB": "1"}

        def run():
            r = subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "flashnext-3.05", "--dry-run"],
                               env=env, capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stdout + r.stderr
            return r.stdout

        out = run()
        assert f"[corpus] dry run: would build {lc.store_dir(corpus, fp)} for {pack}" in out, out
        _store(lc.store_dir(corpus, fp), fp)
        out = run()
        assert f"{lc.store_dir(corpus, fp)} is built" in out, out
        env["QSE_CORPUS_BUILD"] = "off"
        assert "[corpus]" not in run()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:60s} ok{'  ' + got if got else ''}")
