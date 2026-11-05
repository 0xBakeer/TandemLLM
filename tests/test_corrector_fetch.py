"""SPD-249: ops/use-profile.sh fetches flashnext-3.05's drafts' corrector (ops/fetch_weights.py --file).

The test matrix:

  1  the profile: flashnext-3.05 names the file's repository, commit, sha256 and the path the engine reads it from
     (engine/models/flashnext/corrector.py `default_path`); the other packs name none
                                                    test_the_profile_names_the_pinned_file
  2  --file, the download faked: a file that matches its sha256 lands at the path and nothing else is left; one with
     another sha256 is not taken (exit 1); a hub that does not answer is exit 2 with its error in one line; a file
     already there with the sha256 is not downloaded again; one with another sha256 is replaced, the old one kept
     as .old                                        test_fetch_file
  3  use-profile.sh flashnext-3.05 --fetch-only on a box with the pack (the hub faked by a huggingface_hub module on
     PYTHONPATH): the file lands where the engine reads it; with the hub down, or a download of another sha256, the
     switch goes on (exit 0) with one warning that the drafts stay plain coupled and no file at the path; --dry-run
     names the URL at the pinned commit and downloads nothing
                                                    test_use_profile_fetches_the_corrector

Run: python tests/test_corrector_fetch.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ops"))

import fetch_weights  # noqa: E402
import test_weight_profiles as W  # noqa: E402

REL = "ops/correctors/qwen3.8-flash-next/flashnext-3.05.corrector.safetensors"
KEYS = ("QSE_CORRECTOR_HF", "QSE_CORRECTOR_REVISION", "QSE_CORRECTOR_SHA256", "QSE_CORRECTOR_PATH")


def _profile_env(name: str) -> dict:
    """The corrector keys as serve.env resolves them for the profile `name`."""
    old = W.KEYS
    W.KEYS = KEYS
    try:
        return W._env(name)
    finally:
        W.KEYS = old


def test_the_profile_names_the_pinned_file():
    e = _profile_env("flashnext-3.05")
    assert e["QSE_CORRECTOR_HF"] == "0xBakeer/TandemLLM-Qwen3.8-Flash-Next-EXL3-3.05-corrector", e
    assert len(e["QSE_CORRECTOR_REVISION"]) == 40 and int(e["QSE_CORRECTOR_REVISION"], 16) >= 0, e   # a commit
    assert len(e["QSE_CORRECTOR_SHA256"]) == 64 and int(e["QSE_CORRECTOR_SHA256"], 16) >= 0, e
    # serve.env's REPO is the checkout (W._env links it under the temporary home)
    assert e["QSE_CORRECTOR_PATH"].endswith("/" + REL), e
    # the engine's auto path for this profile (corrector.default_path: ops/correctors/<model>/<profile>.corrector...)
    assert REL == os.path.join("ops", "correctors", "qwen3.8-flash-next", "flashnext-3.05" + ".corrector.safetensors")
    for other in ("flashnext-2.05", "flashnext-4.05", "flashnext-5.05", "flashnext-6.05", "exl3-4.5", "nvfp4"):
        o = _profile_env(other)
        assert all(o[k] == "<unset>" for k in KEYS), (other, o)
    return f"{e['QSE_CORRECTOR_HF']}@{e['QSE_CORRECTOR_REVISION'][:12]} -> {REL}"


def _fake(body: bytes | None, calls: list):
    """A fetch_weights._hub_download that writes `body` (None: raises as a hub that does not answer)."""
    def get(repo, commit, name, to):
        calls.append((repo, commit, name))
        if body is None:
            raise OSError("[Errno 101] Network is unreachable")
        os.makedirs(to, exist_ok=True)
        with open(os.path.join(to, name), "wb") as fh:
            fh.write(body)
        return os.path.join(to, name)
    return get


def test_fetch_file():
    body = b"corrector bytes" * 1000
    sha = hashlib.sha256(body).hexdigest()
    old = fetch_weights._hub_download
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ops", "correctors", "m", "p.corrector.safetensors")
            folder = os.path.dirname(path)
            calls: list = []
            fetch_weights._hub_download = _fake(body, calls)
            assert fetch_weights.fetch_file("o/r", "c" * 40, path, sha) == 0
            assert open(path, "rb").read() == body and os.listdir(folder) == ["p.corrector.safetensors"]
            assert calls == [("o/r", "c" * 40, "p.corrector.safetensors")]
            # there with the sha256: no download
            assert fetch_weights.fetch_file("o/r", "c" * 40, path, sha.upper()) == 0 and len(calls) == 1
            # another sha256 on the hub: not taken, the file at the path stays, no partial left
            fetch_weights._hub_download = _fake(b"other bytes", calls)
            assert fetch_weights.fetch_file("o/r", "c" * 40, path + ".x", sha) == 1
            assert not os.path.exists(path + ".x") and sorted(os.listdir(folder)) == ["p.corrector.safetensors"]
            # the hub does not answer: exit 2, nothing at the path
            fetch_weights._hub_download = _fake(None, calls)
            assert fetch_weights.fetch_file("o/r", "c" * 40, path + ".y", sha) == 2
            assert sorted(os.listdir(folder)) == ["p.corrector.safetensors"]
            # a file of another sha256 at the path is replaced; the old one is kept as .old
            open(path, "wb").write(b"an older corrector")
            fetch_weights._hub_download = _fake(body, calls)
            assert fetch_weights.fetch_file("o/r", "c" * 40, path, sha) == 0
            assert open(path, "rb").read() == body and open(path + ".old", "rb").read() == b"an older corrector"
    finally:
        fetch_weights._hub_download = old
    return "fetched and checked; another sha256 refused (1); no hub (2); kept when there; replaced, old kept"


FAKE_HUB = '''import os
def hf_hub_download(repo_id, filename, revision=None, local_dir=None, token=None, **kw):
    with open(os.environ["FAKE_HUB_LOG"], "a") as fh:
        fh.write(f"{repo_id} {revision} {filename}\\n")
    if os.environ.get("FAKE_HUB_DOWN"):
        raise ConnectionError("huggingface.co did not answer")
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, filename), "wb") as fh:
        fh.write(os.environ["FAKE_HUB_BODY"].encode())
    return os.path.join(local_dir, filename)
'''


def test_use_profile_fetches_the_corrector():
    body = "the corrector's bytes\n" * 200
    sha = hashlib.sha256(body.encode()).hexdigest()
    with tempfile.TemporaryDirectory() as home:
        repo, _marks, env = W._fake_box(home, fails="none")
        shutil.copy(os.path.join(ROOT, "ops/fetch_weights.py"), os.path.join(repo, "ops"))
        prof = os.path.join(repo, "ops/profiles/flashnext-3.05.env")
        text = open(prof).read()
        pinned = next(ln.split("=", 1)[1] for ln in text.splitlines() if ln.startswith("QSE_CORRECTOR_SHA256="))
        commit = next(ln.split("=", 1)[1] for ln in text.splitlines() if ln.startswith("QSE_CORRECTOR_REVISION="))
        with open(prof, "w") as fh:                                  # the fake hub's bytes stand for the file
            fh.write(text.replace(pinned, sha))
        fake = os.path.join(home, "fakehub")
        os.makedirs(os.path.join(fake, "huggingface_hub"))
        with open(os.path.join(fake, "huggingface_hub", "__init__.py"), "w") as fh:
            fh.write(FAKE_HUB)
        log = os.path.join(home, "hub.log")
        env.update({"PYTHONPATH": fake, "FAKE_HUB_LOG": log, "FAKE_HUB_BODY": body})
        path = os.path.join(repo, REL)

        def run(*args, **extra):
            r = subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "flashnext-3.05", *args],
                               env={**env, **extra}, capture_output=True, text=True, timeout=120)
            return r, r.stdout + r.stderr

        r, out = run("--dry-run")
        assert r.returncode == 0, out
        url = f"https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-Flash-Next-EXL3-3.05-corrector/resolve/{commit}/"
        assert "dry run: the drafts' corrector " + path in out and url + os.path.basename(REL) in out, out
        assert not os.path.exists(log) and not os.path.exists(path), out
        # the hub does not answer: one warning, the switch goes on, nothing at the path
        r, out = run("--fetch-only", FAKE_HUB_DOWN="1")
        assert r.returncode == 0 and "fetch-only: done" in out, out
        assert out.count("WARNING: the drafts' corrector did not fetch") == 1 and "plain coupled" in out, out
        assert "did not give it (ConnectionError" in out and not os.path.exists(path), out
        # a download of another sha256 than the profile's: refused, the same warning
        r, out = run("--fetch-only", FAKE_HUB_BODY="tampered")
        assert r.returncode == 0 and "not taken" in out and "did not fetch" in out, out
        assert not os.path.exists(path) and os.listdir(os.path.dirname(path)) == [], out
        # the hub answers: the file lands where the engine reads it, at the pinned commit
        r, out = run("--fetch-only")
        assert r.returncode == 0 and "did not fetch" not in out, out
        assert open(path).read() == body and f"{path} complete" in out, out
        assert open(log).read().splitlines()[-1] == \
            f"0xBakeer/TandemLLM-Qwen3.8-Flash-Next-EXL3-3.05-corrector {commit} {os.path.basename(REL)}"
        # and once there, a switch checks it and downloads nothing
        n = len(open(log).read().splitlines())
        r, out = run("--fetch-only")
        assert r.returncode == 0 and "on this box (sha256" in out and len(open(log).read().splitlines()) == n, out
    return "fetched at the pinned commit and checked; hub down or another sha256: a warning, plain coupled drafts"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
