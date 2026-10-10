"""The setup guide's reset levels (SRV-329, server/setup/reset.py and POST /v1/setup/reset), on a
CPU in a temporary home and checkout: never the box's own caches, weights or history.

The test matrix:

  1  GET /v1/setup/reset: levels 2 to 5 with their items, sizes, words and restart flags; level 4
     lists every profile's weights but the active one's and their HF cache copies; nothing that
     is never removed is listed                          test_the_levels
  2  the confirmation: level 1 is 400 browser_only, an unknown level 422, a wrong or missing word
     400 confirm_required with the plan, and nothing is deleted     test_the_confirmation
  3  dry_run lists the same items and deletes nothing; busy and job_running are 409
                                                         test_dry_run_busy_and_job_running
  4  level 3 on a box with no engine: the disk caches of every profile, the shadow's disk and the
     lookup store go; the history, the conductor, the notes, the models, the weights, secrets.env
     and the HF token stay (hashes before and after)    test_level_3
  5  level 3 beside a running engine: the running profile's disk tier and the RAM caches are
     cleared by the engine's own functions under its lock; the shadow's disk is kept while the
     shadow reader runs and reported; the lookup store goes and asks for a restart
                                                         test_level_3_beside_a_running_engine
  6  level 4: the other profiles' weights and HF copies go; the active profile, the protected
     assets and Wren's model stay; a symlink goes as a link and its target stays
                                                         test_level_4
  7  level 5: the history, the conductor's data and the notes go, and a restart is asked for
                                                         test_level_5
  8  level 2: the overlay and its backups are listed; the reset is a job of the apply runner that
     removes them and starts the recommended profile; a failed start puts the overlay back
                                                         test_level_2
  9  a delete that fails is reported in `failed` and the rest still goes; every call is one audit
     line without paths                                  test_partial_failure_and_the_audit

Run: python tests/test_setup_reset.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_setup_apply import Box, FakeWren, Recorder, _api, _sel, _smoke_ok  # noqa: E402

from server import app, auth  # noqa: E402
from server.setup import apply, registry, reset  # noqa: E402

M = registry.get("qwen3.8-27b")


def _put(path: str, size: int = 1000) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)


def _home(box: Box) -> dict:
    """The state of a used box under the temporary home; returns the paths by name."""
    h, sd = box.home, os.path.join(box.home, ".qwen38-spark-engine")
    paths = {
        "disk45": os.path.join(sd, "cache", "disk-exl3-4.5", "heads", "a.bin"),
        "disk60": os.path.join(sd, "cache", "disk-exl3-6.0", "sessions", "b.bin"),
        "shadow": os.path.join(sd, "cache", "shadow", "s.bin"),
        "suffix": os.path.join(sd, "suffix", "log.bin"),
        "ledger": os.path.join(sd, "usage", "ledger.sqlite3"),
        "conductor": os.path.join(sd, "conductor", "router", "v1.json"),
        "notes": os.path.join(sd, "memory", "notes", "owner", "n.md"),
        "secrets": os.path.join(sd, "secrets.env"),
        "narrator": os.path.join(sd, "models", "qwen3.5-2b", "config.json"),
        "laya": os.path.join(sd, "deciders", "laya", "model.safetensors"),
        "token": os.path.join(h, ".cache", "huggingface", "token"),
        "hf25": os.path.join(h, ".cache", "huggingface", "hub",
                             "models--0xBakeer--TandemLLM-Qwen3.8-27B-EXL3-2.5", "blobs", "w"),
        "hf60": os.path.join(h, ".cache", "huggingface", "hub",
                             "models--0xBakeer--TandemLLM-Qwen3.8-27B-EXL3-6.0", "blobs", "w"),
    }
    for p in paths.values():
        _put(p, 2000)
    return paths


def _hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


KEEP = ("ledger", "conductor", "notes", "secrets", "narrator", "laya", "token")


def _setup_app(**state):
    app.STATE.clear()
    app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=FakeWren(), **state)
    app.INFLIGHT.update(running=0, waiting=0)


def test_the_levels():
    with Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        p = _home(box)
        _setup_app()
        code, b = _api("GET", "/v1/setup/reset")
        assert code == 200 and [lv["level"] for lv in b["levels"]] == [2, 3, 4, 5]
        lv = {x["level"]: x for x in b["levels"]}
        assert lv[2]["confirm_word"] is None and lv[3]["confirm_word"] == "delete caches"
        assert lv[4]["confirm_word"] == "delete weights" and lv[5]["confirm_word"] == "delete history"
        assert lv[2]["restart"] and lv[5]["restart"] and not lv[4]["restart"]
        l3 = {i["path"]: i for i in lv[3]["items"]}
        assert "~/.qwen38-spark-engine/cache/disk-exl3-4.5" in l3 and \
            "~/.qwen38-spark-engine/cache/shadow" in l3 and "~/.qwen38-spark-engine/suffix" in l3
        assert l3["~/.qwen38-spark-engine/cache/disk-exl3-4.5"]["bytes"] == 2000
        l4 = [i["path"] for i in lv[4]["items"]]
        assert "~/exl3-quants/4.5" in l4 and "~/exl3-quants/6.0" not in l4, l4   # 6.0 is active
        assert any("EXL3-2.5" in x for x in l4) and not any("EXL3-6.0" in x for x in l4)
        l5 = [i["path"] for i in lv[5]["items"]]
        assert l5 == ["~/.qwen38-spark-engine/usage", "~/.qwen38-spark-engine/conductor",
                      "~/.qwen38-spark-engine/memory"], l5
        listed = json.dumps([lv[n]["items"] for n in (2, 3, 4)])
        for k in ("secrets", "token", "narrator", "laya", "usage", "conductor", "memory",
                  "exl3-drafters-k3", "draft-head-k4", "Qwen3.8-27B-FP8"):
            assert k not in listed, k
        assert b["busy"] is False and lv[4]["bytes"] > 0


def test_the_confirmation():
    with Box() as box:
        p = _home(box)
        _setup_app()
        before = _hash(p.values())
        code, b = _api("POST", "/v1/setup/reset", {"level": 1, "confirm": True})
        assert code == 400 and b["error"]["code"] == "browser_only"
        code, b = _api("POST", "/v1/setup/reset", {"level": 7, "confirm": True})
        assert code == 422
        for conf in (None, True, "delete cache", "DELETE", "", "delete caches please"):
            code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": conf})
            assert code == 400 and b["error"]["code"] == "confirm_required", (conf, b)
            assert b["level"]["confirm_word"] == "delete caches" and b["level"]["items"]
        code, b = _api("POST", "/v1/setup/reset", {"level": 2, "confirm": "reset"})
        assert code == 400 and b["error"]["code"] == "confirm_required"
        code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": "x", "path": "/"})
        assert code == 422 and b["error"]["code"] == "unknown_field"
        assert _hash(p.values()) == before


def test_dry_run_busy_and_job_running():
    with Box() as box:
        p = _home(box)
        _setup_app()
        before = _hash(p.values())
        code, b = _api("POST", "/v1/setup/reset", {"level": 4, "dry_run": True})
        assert code == 200 and b["dry_run"] is True and b["items"] and b["level"] == 4
        assert _hash(p.values()) == before
        app.INFLIGHT["waiting"] = 2
        code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": "delete caches"})
        assert code == 409 and b["error"]["code"] == "busy" and b["error"]["waiting"] == 2
        app.INFLIGHT["waiting"] = 0
        apply.new_job(M, _sel(), apply.plan(M, _sel(), setup_mode=False, password_set=False),
                      setup_mode=False)
        code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": "delete caches"})
        assert code == 409 and b["error"]["code"] == "job_running"
        assert _hash(p.values()) == before


def test_level_3():
    with Box() as box:
        p = _home(box)
        _setup_app()
        keep = _hash([p[k] for k in KEEP] + [p["hf25"]])
        code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": " Delete Caches "})
        assert code == 200 and b["ok"] and b["failed"] == [], b
        for k in ("disk45", "disk60", "shadow", "suffix"):
            assert not os.path.exists(p[k]), k
        assert _hash([p[k] for k in KEEP] + [p["hf25"]]) == keep
        assert os.path.isdir(os.path.join(box.home, "exl3-quants", "4.5"))
        assert b["freed_gb"] >= 0 and b["restart_needed"] is False
        assert len(b["removed"]) == 4, b["removed"]


class Tier:
    def __init__(self, root):
        self.root, self.cleared = root, 0

    def clear(self):
        self.cleared += 1
        return {"files": 1}


class Cache:
    def __init__(self):
        self.cleared, self.parked, self.locked_when = 0, 0, None

    def clear(self):
        self.cleared += 1
        self.locked_when = app.LOCK.locked()

    def shed_park(self):
        self.parked += 1


def test_level_3_beside_a_running_engine():
    with Box() as box:
        p = _home(box)
        tier = Tier(os.path.join(box.home, ".qwen38-spark-engine", "cache", "disk-exl3-6.0"))
        res, store, spill = Cache(), Cache(), Cache()
        _setup_app(disk_cache=tier, resident=res, state_store=store, spill=spill,
                   shadow=object(), suffix_store=object())
        code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": "delete caches"})
        assert code == 200, b
        assert tier.cleared == 1 and spill.cleared == 1 and spill.locked_when is True
        assert res.cleared == 1 and res.parked == 1 and store.cleared == 1
        assert os.path.exists(p["disk60"])                  # the engine cleared it, not a delete
        assert not os.path.exists(p["disk45"])              # another profile's: deleted
        assert os.path.exists(p["shadow"])
        assert [f["path"] for f in b["failed"]] == ["~/.qwen38-spark-engine/cache/shadow"], b
        assert not b["ok"] and not os.path.exists(p["suffix"]) and b["restart_needed"] is True
        assert not app.LOCK.locked()


def test_level_4():
    with Box(on_disk=("exl3-4.5", "exl3-6.0")) as box:
        p = _home(box)
        outside = os.path.join(box.tmp, "outside", "big.bin")
        _put(outside)
        os.symlink(os.path.dirname(outside), os.path.join(box.home, "exl3-quants", "3.0"))
        _setup_app()
        assets = [os.path.join(box.home, ".qwen38-spark-engine", "weights", "exl3-drafters-k3",
                               "ft-b8-v2.safetensors"),
                  os.path.join(box.home, ".qwen38-spark-engine", "weights", "exl3-draft-head-k4",
                               "model.safetensors")]
        keep = _hash([p[k] for k in KEEP] + assets + [p["hf60"]])
        code, b = _api("POST", "/v1/setup/reset", {"level": 4, "confirm": "delete weights"})
        assert code == 200 and b["ok"], b
        assert not os.path.exists(os.path.join(box.home, "exl3-quants", "4.5"))
        assert not os.path.lexists(os.path.join(box.home, "exl3-quants", "3.0"))
        assert os.path.exists(outside)                       # the link went, not its target
        assert os.path.isdir(os.path.join(box.home, "exl3-quants", "6.0"))   # active
        assert not os.path.exists(p["hf25"])
        assert _hash([p[k] for k in KEEP] + assets + [p["hf60"]]) == keep
        assert os.path.exists(p["disk45"]) and os.path.exists(p["suffix"])


def test_level_5():
    with Box() as box:
        p = _home(box)
        _setup_app()
        keep = _hash([p["secrets"], p["narrator"], p["laya"], p["token"], p["disk45"]])
        code, b = _api("POST", "/v1/setup/reset", {"level": 5, "confirm": "delete history"})
        assert code == 200 and b["ok"] and b["restart_needed"] is True, b
        for k in ("ledger", "conductor", "notes"):
            assert not os.path.exists(p[k]), k
        assert _hash([p["secrets"], p["narrator"], p["laya"], p["token"], p["disk45"]]) == keep


def test_level_2():
    with Box() as box:
        _home(box)
        _setup_app()
        apply.write_overlay(M, _sel(slots=2))
        apply.write_overlay(M, _sel(slots=3), clock=lambda: 1_700_000_000)
        code, b = _api("POST", "/v1/setup/reset", {"level": 2, "dry_run": True})
        assert code == 200 and len(b["items"]) == 2 and b["restart"] is True, b
        spawned = []
        real = apply.spawn
        apply.spawn = lambda jid: spawned.append(jid) or 1
        try:
            code, b = _api("POST", "/v1/setup/reset", {"level": 2, "confirm": True})
        finally:
            apply.spawn = real
        assert code == 202 and b["job_id"] == spawned[0] and b["job"]["kind"] == "reset", b
        assert app.STATE["wren"].unloads == ["reset"]
        rec = Recorder()
        assert apply.Runner(b["job_id"], cmd=rec, http=_smoke_ok("exl3-4.5"),
                            ops=box.ops).main() == 0, apply.read_job(b["job_id"])
        assert not [f for f in os.listdir(apply.run_dir()) if f.startswith("setup.env")]
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["use-profile.sh exl3-4.5"], seq
        assert rec.calls[0]["env"]["SMOKE_MODEL"] == "TandemLLM"
        # a failed start: the overlay comes back
        apply.write_overlay(M, _sel(slots=2))
        with open(apply.overlay_path()) as fh:
            before = fh.read()
        p = apply.reset_plan(M, setup_mode=False)
        p["profile"]["current"] = "exl3-6.0"
        jid = apply.new_job(M, p["selection"], p, setup_mode=False, kind="reset")["id"]
        rec = Recorder(rc={"use-profile.sh exl3-4.5": 1})
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 1
        assert apply.read_job(jid)["state"] == "rolled_back"
        with open(apply.overlay_path()) as fh:
            assert fh.read() == before
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["use-profile.sh exl3-4.5", "use-profile.sh exl3-6.0"], seq


def test_partial_failure_and_the_audit():
    with Box() as box:
        p = _home(box)
        _setup_app()
        locked = os.path.dirname(p["shadow"])
        os.chmod(locked, 0o500)
        try:
            code, b = _api("POST", "/v1/setup/reset", {"level": 3, "confirm": "delete caches"})
        finally:
            os.chmod(locked, 0o700)
        assert code == 200 and not b["ok"], b
        assert [f["path"] for f in b["failed"]] == ["~/.qwen38-spark-engine/cache/shadow"], b
        assert not os.path.exists(p["disk45"]) and "could not be removed" in b["message"]
        _api("POST", "/v1/setup/reset", {"level": 4, "dry_run": True})
        _api("POST", "/v1/setup/reset", {"level": 5, "confirm": "nope"})
        with open(os.path.join(apply.run_dir(), "setup-audit.log")) as fh:
            lines = [json.loads(x) for x in fh]
        assert [(x["level"], x["dry_run"], x["result"]) for x in lines] == [
            (3, False, "1 failed"), (4, True, "plan"), (5, False, "confirm_required")], lines
        assert "/" not in json.dumps(lines), lines


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
