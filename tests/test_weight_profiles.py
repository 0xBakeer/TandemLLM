"""ops/profiles: serve.env's overlay sets only the weight-specific values, and every profile resolves."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("QSE_WEIGHTS", "QWEN38_EXL3", "QWEN38_DRAFT_EXL3", "QWEN38_DRAFT_HEAD_EXL3", "QWEN38_SAMPLE_NUCLEUS",
        "QSE_DISK_CACHE", "NV", "HEAD", "QSE_WEIGHTS_HF", "PORT", "SERVED_MODEL", "QSE_CONDUCTOR_CONFIG")


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


def test_every_profile_resolves():
    names = [f[:-4] for f in os.listdir(os.path.join(ROOT, "ops/profiles")) if f.endswith(".env") and not f.startswith("_")]
    assert {"nvfp4", "exl3-2.5", "exl3-3.0", "exl3-3.5", "exl3-4.0", "exl3-4.5", "exl3-6.0"} <= set(names), names
    for n in names:
        assert _env(n)["QSE_WEIGHTS"] == n, n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name:60s} ok")
