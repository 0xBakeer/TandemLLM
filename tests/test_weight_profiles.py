"""ops/profiles: serve.env's overlay sets only the weight-specific values, and every profile resolves."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("QSE_WEIGHTS", "QWEN38_EXL3", "QWEN38_DRAFT_EXL3", "QWEN38_DRAFT_HEAD_EXL3", "QWEN38_SAMPLE_NUCLEUS",
        "QSE_DISK_CACHE", "NV", "HEAD", "QSE_WEIGHTS_HF", "PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG",
        "QSE_WEIGHTS_DIR", "QSE_WEIGHTS_REVISION", "QSE_ENGINE_FLAGS", "QSE_ASSETS", "QSE_MEM_NEED_GIB",
        "QWEN38_FN_TABLE", "QSE_PAGE_CACHE_DROP", "QSE_PAGE_CACHE_KEEP", "VISION", "MAX_LEN", "QWEN38_FN_LOOKUP",
        "RESIDENT_STASH_GB", "QSE_PARALLEL", "QSE_KV_POOL", "QSE_SIDE_MAX_LEN")


def _env(profile: str | None, extra: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory() as home:
        # serve.env's REPO (${HOME}/<dir>: the repository's, or a deploy tree's own name)
        m = re.search(r"^REPO=\$\{HOME\}/(\S+)$", open(os.path.join(ROOT, "ops/serve.env")).read(), re.M)
        os.symlink(ROOT, os.path.join(home, m.group(1) if m else "qwen38-spark-engine"))
        script = (f"set -a; . {ROOT}/ops/serve.env; "
                  + "; ".join(f'echo "{k}=${{{k}-<unset>}}"' for k in KEYS))
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(extra or {})
        if profile is not None:
            env["QSE_PROFILE_OVERRIDE"] = profile
        out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout
        return {k.replace(home, "~"): v.replace(home, "~") for k, v in (l.split("=", 1) for l in out.splitlines())}


def test_no_profile_keeps_serve_env():
    e = _env(None)
    assert e["QSE_WEIGHTS"] == "<unset>" and e["QWEN38_EXL3"] == "<unset>", e


def test_exl3_profile_sets_only_the_weights():
    base, e = _env(None), _env("exl3-4.5")
    assert e["QSE_WEIGHTS"] == "exl3-4.5" and e["QWEN38_EXL3"] == "~/exl3-quants/4.5", e
    assert e["QWEN38_DRAFT_HEAD_EXL3"].endswith("exl3-draft-head-k4") and e["QWEN38_SAMPLE_NUCLEUS"] == "1"
    assert e["QSE_WEIGHTS_HF"] == "0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.5"
    assert e["QSE_DISK_CACHE"].endswith("disk-exl3-4.5")
    for k in ("PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG", "NV", "HEAD"):
        assert e[k] == base[k], k


def test_nvfp4_profile_unsets_exl3():
    e = _env("nvfp4", {"QWEN38_EXL3": "/leak", "QWEN38_DRAFT_EXL3": "/leak"})
    assert e["QSE_WEIGHTS"] == "nvfp4" and e["QWEN38_EXL3"] == "<unset>" and e["QWEN38_SAMPLE_NUCLEUS"] == "0", e


def test_flashnext_profiles_name_their_pack_and_replace_the_27b_flags():
    """OPS-97: a Flash-Next profile is a branch of turboderp's repo, its engine flags replace the 27B's
    (start.sh), it needs no 27B asset, keeps a warm table out of the page-cache drop, and leaves the
    shared values alone."""
    base = _env(None)
    for bpw, table in (("2.05", "warm"), ("3.05", "cold"), ("4.05", "cold"), ("5.05", "cold"), ("6.05", "cold")):
        e = _env(f"flashnext-{bpw}", {"QWEN38_EXL3": "/leak", "QWEN38_DRAFT_EXL3": "/leak"})
        d = f"~/flashnext-packs/{bpw}"
        assert e["QSE_WEIGHTS"] == f"flashnext-{bpw}" and e["QSE_WEIGHTS_DIR"] == d, e
        assert e["QSE_WEIGHTS_HF"] == "turboderp/Qwen3.8-Flash-Next-exl3", e
        assert e["QSE_WEIGHTS_REVISION"].startswith(f"{bpw}bpw_"), e
        assert e["QSE_ENGINE_FLAGS"] == f"--model {d} --drafter mtp", e
        assert e["QWEN38_EXL3"] == e["QWEN38_DRAFT_EXL3"] == e["QWEN38_DRAFT_HEAD_EXL3"] == "<unset>", e
        assert e["QSE_ASSETS"] == "" and int(e["QSE_MEM_NEED_GIB"]) > 0, e
        assert e["QWEN38_FN_TABLE"] == table and e["QSE_PAGE_CACHE_DROP"] == d, e
        assert e["QSE_PAGE_CACHE_KEEP"] == ("ngram_embedding.safetensors" if table == "warm" else ""), e
        assert e["QSE_DISK_CACHE"].endswith(f"disk-flashnext-{bpw}") and e["VISION"] == "on", e
        assert e["QWEN38_FN_LOOKUP"] == "1", e
        # ENG-241: three slots on the KV pool; the side window serve.env's
        assert e["QSE_PARALLEL"] == "3" and e["QSE_KV_POOL"] == "on", e
        # the guest stash as serve.env sets it (ENG-157 fixed: it was off on Flash-Next)
        for k in ("PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG", "MAX_LEN", "RESIDENT_STASH_GB", "QSE_SIDE_MAX_LEN"):
            assert e[k] == base[k], k
    # a 27B profile sets none of them: start.sh's 27B flags and use-profile.sh's 27B assets stay
    e = _env("exl3-4.5")
    assert all(e[k] == "<unset>" for k in ("QSE_ENGINE_FLAGS", "QSE_ASSETS", "QSE_MEM_NEED_GIB",
                                           "QSE_WEIGHTS_REVISION", "QSE_PAGE_CACHE_KEEP")), e


def _use_profile(*args: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as home:
        m = re.search(r"^REPO=\$\{HOME\}/(\S+)$", open(os.path.join(ROOT, "ops/serve.env")).read(), re.M)
        os.symlink(ROOT, os.path.join(home, m.group(1) if m else "qwen38-spark-engine"))
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        return subprocess.run(["bash", os.path.join(ROOT, "ops/use-profile.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=60)


def test_use_profile_fetches_a_flashnext_branch_and_needs_no_27b_asset():
    """OPS-97: a Flash-Next switch on a box with none of the 27B's drafters, draft head or base snapshot
    is not refused; it would fetch the pack's branch, and the memory it waits for is the profile's. A
    27B profile on the same box is still refused for its missing assets."""
    if sys.platform == "darwin":
        print("    (skipped: GNU df -- run it on the box)")
        return
    r = _use_profile("flashnext-3.05", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "turboderp/Qwen3.8-Flash-Next-exl3@3.05bpw_h5_ng5 (~85.1 GB)" in r.stdout, r.stdout
    assert "dry run: would fetch" in r.stdout and "MemAvailable >= 70 GB" in r.stdout, r.stdout
    r = _use_profile("exl3-4.5", "--dry-run")
    assert r.returncode == 3 and "REFUSING" in r.stdout and "27B-FP8 base snapshot" in r.stdout, r.stdout
    r = _use_profile("--list")
    assert r.returncode == 0 and "flashnext-6.05" in r.stdout, r.stdout


def test_a_branch_without_a_checksum_list_is_verified_against_the_hubs():
    """fetch_weights.py: a Flash-Next branch has no crc32.txt; the hub's LFS sha256 (its
    _hub_manifest.json) is the list -- a changed or missing weight file fails, a non-LFS file is not
    checked."""
    import hashlib
    import json
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "model-00001-of-00001.safetensors"), "wb").write(b"w" * 4096)
        open(os.path.join(d, "ngram_embedding.safetensors"), "wb").write(b"t" * 1024)
        open(os.path.join(d, "config.json"), "w").write("{}")
        files = {n: {"size": os.path.getsize(os.path.join(d, n)),
                     "sha256": hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest()}
                 for n in ("model-00001-of-00001.safetensors", "ngram_embedding.safetensors")}
        files["config.json"] = {"size": 2, "sha256": None}
        json.dump({"repo": "r", "revision": "b", "commit": "c", "files": files},
                  open(os.path.join(d, fetch_weights.MANIFEST), "w"))
        assert fetch_weights.verify(d) == []
        open(os.path.join(d, "ngram_embedding.safetensors"), "ab").write(b"x")
        assert [b.split(":")[0] for b in fetch_weights.verify(d)] == ["ngram_embedding.safetensors"]
        os.remove(os.path.join(d, "ngram_embedding.safetensors"))
        assert fetch_weights.verify(d) == ["missing ngram_embedding.safetensors"]


def test_a_pack_links_the_table_a_sibling_pack_already_holds():
    """OPS-97: 4.05, 5.05 and 6.05 share one n-gram table (same sha256): a fetch of one next to another
    hard-links it instead of downloading 39 GB; a file with another sha256 is downloaded."""
    import json
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    import fetch_weights
    with tempfile.TemporaryDirectory() as parent:
        have, part = os.path.join(parent, "4.05"), os.path.join(parent, "5.05.partial")
        os.makedirs(have)
        os.makedirs(part)
        open(os.path.join(have, "ngram_embedding.safetensors"), "wb").write(b"t" * 64)
        open(os.path.join(have, "model-00001-of-00001.safetensors"), "wb").write(b"a" * 64)
        files = {"ngram_embedding.safetensors": {"size": 64, "sha256": "t" * 64},
                 "model-00001-of-00001.safetensors": {"size": 64, "sha256": "a" * 64}}
        json.dump({"files": files}, open(os.path.join(have, fetch_weights.MANIFEST), "w"))
        mine = dict(files, **{"model-00001-of-00001.safetensors": {"size": 64, "sha256": "b" * 64}})
        json.dump({"files": mine}, open(os.path.join(part, fetch_weights.MANIFEST), "w"))
        assert fetch_weights.share(part, os.path.join(parent, "5.05")) == ["ngram_embedding.safetensors"]
        a, b = (os.stat(os.path.join(x, "ngram_embedding.safetensors")) for x in (have, part))
        assert a.st_ino == b.st_ino and not os.path.exists(os.path.join(part, "model-00001-of-00001.safetensors"))


def test_a_failed_switch_to_flashnext_rolls_back_to_the_27b():
    """OPS-99: a switch from a 27B profile to Flash-Next whose start fails stops, starts Flash-Next,
    stops again and starts the 27B profile -- one engine at a time, in that order -- and says so. The
    scripts it calls are fakes; only use-profile.sh and the profiles are real."""
    if not os.path.isfile("/proc/meminfo") or not shutil.which("flock"):
        print("    (skipped: needs /proc/meminfo and flock -- run it on the box)")
        return
    with tempfile.TemporaryDirectory() as home:
        repo = os.path.join(home, "repo")
        os.makedirs(os.path.join(repo, "ops"))
        shutil.copy(os.path.join(ROOT, "ops/use-profile.sh"), os.path.join(repo, "ops"))
        shutil.copytree(os.path.join(ROOT, "ops/profiles"), os.path.join(repo, "ops/profiles"))
        tail = open(os.path.join(ROOT, "ops/serve.env")).read().split("# Weight profile (ops/use-profile.sh)")[1]
        with open(os.path.join(repo, "ops/serve.env"), "w") as fh:
            fh.write(f"REPO={repo}\nPY={sys.executable}\nPORT=18996\n# Weight profile{tail}")
        marks = os.path.join(home, "marks")
        fakes = {"stop.sh": f'echo "stop" >> {marks}\n',
                 # the Flash-Next start fails; anything else comes up
                 "start.sh": f'echo "start $(cat {repo}/ops/active-profile)" >> {marks}\n'
                             f'[ "$(cat {repo}/ops/active-profile)" != flashnext-3.05 ]\n'}
        for name, body in fakes.items():
            with open(os.path.join(repo, "ops", name), "w") as fh:
                fh.write("#!/bin/bash\n" + body)
            os.chmod(os.path.join(repo, "ops", name), 0o755)
        binary = os.path.join(home, "bin")
        os.makedirs(binary)
        with open(os.path.join(binary, "curl"), "w") as fh:         # /health names the active profile
            fh.write(f'#!/bin/bash\ncase "$*" in *health*) echo "{{\\"weights\\": {{\\"name\\": '
                     f'\\"$(cat {repo}/ops/active-profile)\\"}}}}";; '
                     '*) echo \'{"choices": [{"message": {"content": "ready"}}]}\';; esac\n')
        os.chmod(os.path.join(binary, "curl"), 0o755)
        pack = os.path.join(home, "flashnext-packs/3.05")            # a complete pack on disk
        os.makedirs(pack)
        for n, body in (("config.json", "{}"), ("model.safetensors.index.json", '{"weight_map": {}}'),
                        ("ngram_embedding.safetensors", "t")):
            open(os.path.join(pack, n), "w").write(body)
        open(os.path.join(repo, "ops/active-profile"), "w").write("exl3-4.5\n")
        env = {"HOME": home, "PATH": binary + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
               "BOX_LOCK": os.path.join(home, "box.flock"), "MEM_NEED_GB": "1"}
        r = subprocess.run(["bash", os.path.join(repo, "ops/use-profile.sh"), "flashnext-3.05"], env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 1 and "rolled back: exl3-4.5 live" in r.stdout, r.stdout + r.stderr
        assert open(marks).read().split("\n")[:-1] == ["stop", "start flashnext-3.05", "stop", "start exl3-4.5"]
        assert open(os.path.join(repo, "ops/active-profile")).read().strip() == "exl3-4.5"


def test_every_profile_resolves():
    names = [f[:-4] for f in os.listdir(os.path.join(ROOT, "ops/profiles")) if f.endswith(".env") and not f.startswith("_")]
    assert {"nvfp4", "exl3-2.5", "exl3-3.0", "exl3-3.5", "exl3-4.0", "exl3-4.5", "exl3-6.0",
            "flashnext-2.05", "flashnext-3.05", "flashnext-4.05", "flashnext-5.05", "flashnext-6.05"} <= set(names), names
    for n in names:
        assert _env(n)["QSE_WEIGHTS"] == n, n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name:60s} ok")
