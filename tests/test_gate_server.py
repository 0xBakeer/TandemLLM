"""tools/gate/server.py and the runner's plan, on a CPU with no server: the arm trees, the port rule,
the arm's setup.env and serve.env, the knobs a report records, the order of the arms, the packs.

Run: python tests/test_gate_server.py
"""

from __future__ import annotations

import argparse
import difflib
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.gate import server  # noqa: E402
from tools.gate.hw.fake import Fake  # noqa: E402
from tools.gate.hw.generic import Generic  # noqa: E402
from tools.gate.hw.base import Fingerprint  # noqa: E402
from tools.gate.runner import arm_env, arms_of, load_packs, plan_of  # noqa: E402
from tools.row3 import code_hash  # noqa: E402

HEALTH = {
    "status": "ok", "version": "0.15.2+flashnext-3.05", "max_len": 262144, "max_queue": 8,
    "weights": {"name": "flashnext-3.05", "weights": "/home/x/flashnext-packs/3.05", "format": "exl3",
                "self_check": {"ok": True}, "corrector": {"loaded": True}},
    "memory": {"reserved_gb": 57.8, "ceiling_gb": 64.82, "peak_gb": 75.82},
    "cache": {"prefix_cache": True, "prefix_chunk": 4096, "suffix_store": None, "response_cache": None,
              "state_store": {"budget": 2147483648}},
    "parallel": {"n": 8, "requested": 8, "max_running": 8, "max_running_reason": None,
                 "slots": [{"max_len": 262144}] * 8, "admission": {"floor_gib": 10.0}},
}


def _git_repo() -> str | None:
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def test_tree_is_the_commit_and_a_patch_moves_its_hash():
    sha = _git_repo()
    if not sha:
        print("skipped: not a git checkout")
        return
    d = Path(tempfile.mkdtemp(prefix="gate-tree-"))
    h = server.make_tree(ROOT, sha, d / "a")
    assert h == code_hash(d / "a") and (d / "a" / "tools" / "row3.py").exists()
    p = d / "x.patch"
    p.write_text("--- a/VERSION\n+++ b/VERSION\n@@ -1 +1 @@\n-"
                 + (d / "a" / "VERSION").read_text() + "+9.9.9\n")
    h2 = server.make_tree(ROOT, sha, d / "b", str(p))
    assert (d / "b" / "VERSION").read_text() == "9.9.9\n"
    assert h2 == h                       # VERSION is outside the code folders the hash covers
    src = (d / "a" / "tools/row3.py").read_text().splitlines(True)
    p.write_text("".join(difflib.unified_diff(src, ["# planted\n"] + src, "a/tools/row3.py", "b/tools/row3.py")))
    assert server.make_tree(ROOT, sha, d / "c", str(p)) != h


def test_ports_stay_at_or_above_the_floor_and_skip_busy_ones():
    try:
        server.free_port(8000, set())
        raise AssertionError("a port under 8100 was accepted")
    except SystemExit as exc:
        assert "never beside" in str(exc)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    try:
        got = server.free_port(busy, set(), floor=1024)
        assert got != busy and got > busy
        assert server.free_port(busy, {busy + 1}, floor=1024) not in (busy, busy + 1)
    finally:
        s.close()


def test_the_arm_setup_env_and_serve_env():
    d = Path(tempfile.mkdtemp(prefix="gate-arm-"))
    tree = d / "tree"
    (tree / "ops").mkdir(parents=True)
    (tree / "ops" / "serve.env").write_text("HOME_DIR=${HOME}\nREPO=${HOME}/qwen38-spark-engine\nPORT=8000\n")
    box = d / "box-setup.env"
    box.write_text("QSE_PARALLEL='8'\nPORT='8000'\nRESPONSE_CACHE='on'\n")
    local = d / "local.env"
    local.write_text("PY=/opt/py\n")
    src = d / "corrector.bin"
    src.write_bytes(b"x")
    (d / "community").mkdir()
    (d / "community" / "a.env").write_text("A=1\n")
    cond = d / "conductor"
    cond.mkdir()
    (cond / "config.json").write_text("{}")
    (cond / "decisions-1.jsonl").write_text("secret")
    (cond / "x.token").write_text("secret")
    site = {"setup_env": str(box), "local_env": str(local), "conductor_from": str(cond),
            "copies": {"ops/correctors/c.bin": str(src), "run/community": str(d / "community")},
            "links": {"corpus": str(d)}}
    pack = {"launcher": "ops", "profile": "flashnext-3.05"}
    ol = server.OpsLauncher(Generic(Fingerprint(memory_gib=48)), pack, site)
    ol.prepare(tree)
    serve = (tree / "ops" / "serve.env").read_text()
    assert f"REPO={tree}\n" in serve and "qwen38-spark-engine" not in serve
    assert (tree / "ops" / "active-profile").read_text() == "flashnext-3.05\n"
    assert (tree / "run" / "local.env").read_text() == "PY=/opt/py\n"
    assert oct((tree / "run").stat().st_mode)[-3:] == "700"
    assert (tree / "ops/correctors/c.bin").read_bytes() == b"x" and (tree / "corpus").is_symlink()
    assert (tree / "run/community/a.env").read_text() == "A=1\n"
    arm = server.Arm("head", "plain", tree, 8105, d / "home",
                     {"QSE_PARALLEL": "4", "QSE_ENGINE_FLAGS": "--model ${QSE_WEIGHTS_DIR} --drafter none"})
    ol.arm_state(arm)
    got = sorted(os.listdir(d / "home" / "conductor"))
    assert got == ["config.json"], got
    assert (d / "home" / "secrets.env").read_text() == ""
    env = ol.setup_env(arm)
    assert env.index("RESPONSE_CACHE='on'") < env.index("RESPONSE_CACHE='off'"), env
    assert "PORT='8105'" in env and "CONDUCTOR='raw'" in env and "QSE_USAGE_LEDGER='off'" in env
    assert f"QSE_CONDUCTOR_CONFIG='{d / 'home' / 'conductor'}'" in env
    assert 'QSE_ENGINE_FLAGS="--model ${QSE_WEIGHTS_DIR} --drafter none"' in env
    assert env.rindex("QSE_PARALLEL='4'") > env.index("QSE_PARALLEL='8'")
    # the overlay, sourced by bash, gives the arm's values last
    out = subprocess.run(["bash", "-c", f"set -a; QSE_WEIGHTS_DIR=/w; . /dev/stdin; echo $PORT $QSE_PARALLEL "
                          "$RESPONSE_CACHE \"$QSE_ENGINE_FLAGS\""], input=env, capture_output=True, text=True)
    assert out.stdout.strip() == "8105 4 off --model /w --drafter none", out.stdout + out.stderr


def test_knobs_record_every_limit_and_no_paths():
    k = server.knobs(HEALTH, {"QSE_PARALLEL": "8", "QWEN38_FN_PREFILL_CHUNK": "16384"},
                     ["app.py", "--port", "8100", "--model", "/home/x/flashnext-packs/3.05", "--drafter", "mtp",
                      "--response-cache"])
    for key in ("lanes", "max_running", "max_queue", "max_len", "prefill_chunk", "prefix_cache",
                "prefix_chunk", "cache_gb", "ceiling_gb", "peak_gb", "suffix_store", "response_cache",
                "drafter", "admission_floor_gib"):
        assert key in k, key
    assert k["lanes"] == 8 and k["prefill_chunk"] == "16384" and k["cache_gb"] == 2.0
    assert k["drafter"] == "mtp" and k["flags"]["--model"] == "3.05" and k["flags"]["--response-cache"] is True
    assert "/home" not in repr(k)


def test_secret_named_env_is_never_a_knob():
    assert server.KNOB_ENV.match("QSE_ADMIN_TOKEN") and server.SECRET.search("QSE_ADMIN_TOKEN")
    assert server.SECRET.search("QSE_DASHBOARD_LOGIN") and not server.SECRET.search("QSE_PARALLEL")


def plan(suite, files, prof=None, pack=None, pairs=0):
    a = argparse.Namespace(suite=suite, only=None, pairs=pairs)
    return plan_of(a, prof or Fake(), pack or {"launcher": "fake", "id": "fake"}, files)


def test_arm_order_and_variants():
    p = plan("auto", ["engine/spec.py"])
    assert p.level == "quick" and [(s, v) for s, v, _ in arms_of(p)] == [("base", "main"), ("head", "main")]
    flash = dict(load_packs()["flashnext-3.05"], id="flashnext-3.05")
    g = Generic(Fingerprint(memory_gib=48))
    p = plan("auto", ["engine/spec.py"], g, flash)
    assert [(s, v) for s, v, _ in arms_of(p)][-1] == ("head", "plain")
    p = plan("auto", ["server/slots.py"], g, flash)
    assert p.level == "quick+long"
    assert [(s, v) for s, v, _ in arms_of(p)][-2:] == [("base", "wide"), ("head", "wide")]
    p = plan("full", [], g, flash)
    assert p.pairs == 2 and [s for s, v, _ in arms_of(p) if v == "main"] == ["base", "head", "base", "head"]
    assert "stability" in arms_of(p)[0][2]
    assert plan("auto", ["docs/a.md"]).level == "none"


def test_arm_env_takes_the_variant_and_the_planted_head_settings():
    flash = dict(load_packs()["flashnext-3.05"], id="flashnext-3.05")
    g = Generic(Fingerprint(memory_gib=48))
    plant = {"REP_PENALTY": "1.05"}
    assert arm_env(g, flash, "base", "main", plant) == g.arm_env()
    assert arm_env(g, flash, "head", "main", plant) == {**g.arm_env(), **plant}
    assert "--drafter none" in arm_env(g, flash, "head", "plain")["QSE_ENGINE_FLAGS"]
    assert arm_env(g, flash, "base", "wide")["QWEN38_FN_PREFILL_CHUNK"] == "16384"


def test_packs_name_real_profiles():
    for pid, p in load_packs().items():
        assert p["launcher"] in ("ops", "fake"), pid
        if p["launcher"] == "ops":
            assert os.path.isfile(os.path.join(ROOT, "ops", "profiles", p["profile"] + ".env")), pid


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
