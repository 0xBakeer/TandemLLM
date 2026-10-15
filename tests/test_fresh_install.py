"""A fresh box (OPS-143): fetch_weights.py checks a fetch entry's globs. No GPU and no network: the
downloads run against tests/test_weight_profiles.py's fake hub.

The test matrix, the Gherkin of OPS-143:

  1  a fetch entry's names and globs (`*.safetensors`, `encoder/*`) select the files verify() checks
     against the hub's sha256: a matching file is checked, a changed one fails, a missing one fails, a
     file no pattern matches is left alone                       test_verify_reads_globs
  2  a tool model whose entry is all globs (Qwen3.5-0.8B, the shadow reader's) fetches and verifies;
     a changed file under a glob (Laya's encoder/*) fails the fetch and stays out of the folder
                                                                 test_a_tool_model_of_globs_fetches

Run: python tests/test_fresh_install.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weight_profiles as W  # noqa: E402


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def test_verify_reads_globs():
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights as fw
    with tempfile.TemporaryDirectory() as d:
        body = {"model-00001-of-00002.safetensors": b"a" * 64, "model-00002-of-00002.safetensors": b"b" * 64,
                "encoder/model.safetensors": b"e" * 64, "tokenizer.json": b"{}", "mtp.safetensors": b"m" * 64}
        for n, b in body.items():
            _write(os.path.join(d, n), b)
        files = {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in body.items()}
        files["config.json"] = {"size": 2, "sha256": None}
        json.dump({"repo": "r", "revision": None, "commit": "c", "files": files},
                  open(os.path.join(d, fw.MANIFEST), "w"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.verify(d, ["*.json", "model-*.safetensors", "encoder/*"]) == []
            assert fw.verify(d, ["tokenizer.json"]) == []          # a name, as before
        assert "[fetch] 4 files checked" in out.getvalue() and "[fetch] 1 files checked" in out.getvalue(), \
            out.getvalue()
        # a file outside the patterns is not checked, one under a glob is
        open(os.path.join(d, "mtp.safetensors"), "ab").write(b"x")
        open(os.path.join(d, "encoder/model.safetensors"), "ab").write(b"x")
        with contextlib.redirect_stdout(io.StringIO()):
            bad = fw.verify(d, ["model-*.safetensors", "encoder/*"])
        assert [b.split(":")[0] for b in bad] == ["encoder/model.safetensors"], bad
        os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
        with contextlib.redirect_stdout(io.StringIO()):
            bad = fw.verify(d, ["model-*.safetensors"])
        assert bad == ["missing model-00002-of-00002.safetensors"], bad


def test_a_tool_model_of_globs_fetches():
    fw, tmp, home, repo, old = W._fetch_box()
    hub = W._Hub(fw, os.path.join(home, ".cache", "huggingface", "hub"))
    for r, names in {"Qwen/Qwen3.5-0.8B": ["config.json", "tokenizer.json", "tokenizer_config.json",
                                           "chat_template.jinja", "merges.txt", "LICENSE",
                                           "model.safetensors-00001-of-00001.safetensors", "README.md"],
                     "convaiinnovations/laya": ["config.json", "rl_agent_config.json", "model.safetensors",
                                                "encoder/model.safetensors", "encoder/config.json",
                                                "tokenizer/tokenizer.json", "multilingual/model.safetensors",
                                                "typed-decisions/model.safetensors", "laya.py"]}.items():
        for n in names:
            hub.body[(r, n)] = f"{r}/{n}".encode() * 3
            hub.sha[(r, n)] = hashlib.sha256(hub.body[(r, n)]).hexdigest()
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = fw.assets("exl3-4.5", repo, margin_gb=0, also=["qwen3.5-0.8b"])
        assert rc == 0, out.getvalue()
        small = os.path.join(home, ".qwen38-spark-engine/models/qwen3.5-0.8b")
        assert set(os.listdir(small)) == {"LICENSE", "README.md", fw.MANIFEST, "chat_template.jinja", "config.json",
                                          "merges.txt", "model.safetensors-00001-of-00001.safetensors",
                                          "tokenizer.json", "tokenizer_config.json"}, os.listdir(small)
        assert f"[fetch] 2 files checked in {small}.partial-qwen3.5-0.8b: OK" in out.getvalue(), out.getvalue()
        # Laya: a changed file under encoder/* fails the fetch, and the folder the engine reads stays absent
        hub.bad = {"encoder/model.safetensors"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fw.assets("exl3-4.5", repo, margin_gb=0, also=["laya"]) == 5, out.getvalue()
        laya = os.path.join(home, ".qwen38-spark-engine/deciders/laya")
        assert "encoder/model.safetensors: " in out.getvalue() and not os.path.exists(laya), out.getvalue()
        hub.bad = set()
        with contextlib.redirect_stdout(io.StringIO()):
            assert fw.assets("exl3-4.5", repo, margin_gb=0, also=["laya"]) == 0
        assert os.path.isfile(os.path.join(laya, "encoder/model.safetensors"))
        assert not os.path.exists(os.path.join(laya, "laya.py"))
    finally:
        W._fetch_box_close(tmp, old, hub)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name:60s} ok")
