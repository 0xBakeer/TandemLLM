"""The setup guide's apply job (SRV-327, server/setup/apply.py and its routes), on a CPU against a
checkout in a temporary folder: never the box's own serve.env, run/ or weights.

The test matrix:

  1  the overlay: the registry's keys with the selection's values, ${HOME} kept for bash, every
     value one plain line; it parses (bash -n) and serve.env reads it after the profile; a value
     with a command substitution in it is refused         test_the_overlay
  2  the plan: what changes against serve.env now, the commands it will run, the warnings (no
     password, the 4 x 262,144 worst case); and each refusal with its code: over_budget,
     missing_assets, missing_token, disk_full, no_overlay_hook, secrets_exist, port_in_use,
     repo_mismatch
                                                         test_the_plan
  3  the allowlist, with a recording runner over a corpus of selections: every command is bash
     and one of ops/use-profile.sh, start.sh, stop.sh, make-secrets.sh, every argument a
     registry id, a number or one of the scripts' own flags, the environment a short fixed list,
     nothing written outside the checkout's run/        test_nothing_outside_the_allowlist
  4  the steps: on disk the fetch is skipped; a download runs --fetch-only first; a port change
     stops the old engine before the overlay; generate runs make-secrets.sh --admin
                                                         test_the_steps
  5  rollback: a failed switch and a failed smoke put the old run/setup.env back and start the
     previous profile (setup mode: the marker back and start.sh --setup); the job says rolled_back
                                                         test_rollback
  6  the routes: a body that names a step, a command, an env or a path is 422 unknown_field; a
     hostile profile or name is refused; dry_run is the plan and no job; busy and job_running are
     409, a missing asset 412; a good apply is 202, unloads Wren and starts the runner
                                                         test_the_routes
  7  the job route: ?format=json the job; SSE replays the events with their ids, and Last-Event-ID
     resumes after one; an unknown or malformed id is 404   test_the_job_route
  8  no secret in the log: the HF token, an hf_ token and a bearer token are cut out of every
     line the runner writes                              test_secrets_are_redacted
  9  the real runner, detached (its own session) with stub scripts and a local engine for the
     smoke: the job runs to done on its own and the status responder serves its state
                                                         test_the_detached_runner

Run: python tests/test_setup_apply.py
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, auth  # noqa: E402
from server.setup import apply, device as device_mod, estimate, registry  # noqa: E402

REPO = registry.ROOT
M = registry.get("qwen3.8-27b")
HF = "hf_" + "Zq9" * 12


class Box:
    """A checkout in a temporary folder: ops/ with serve.env (REPO pointing here), the profiles and
    stub scripts; run/; a HOME with the assets and the weights of the profiles named on disk."""

    def __init__(self, on_disk=("exl3-4.5", "exl3-6.0"), assets=True, hook=True, port=8000):
        self.tmp = tempfile.mkdtemp(prefix="setup-box-")
        self.root = os.path.join(self.tmp, "repo")
        self.home = os.path.join(self.tmp, "home")
        ops = self.ops = os.path.join(self.root, "ops")
        shutil.copytree(os.path.join(REPO, "ops", "profiles"), os.path.join(ops, "profiles"))
        with open(os.path.join(REPO, "ops", "serve.env")) as fh:
            env = fh.read()
        env = env.replace("REPO=${HOME}/qwen38-spark-engine", f"REPO={self.root}")
        env = env.replace("PORT=8000", f"PORT={port}")
        if not hook:
            env = env[:env.index("# The setup guide's overlay")]
        with open(os.path.join(ops, "serve.env"), "w") as fh:
            fh.write(env)
        for s in apply.SCRIPTS:
            with open(os.path.join(ops, s), "w") as fh:
                fh.write(f'#!/bin/bash\necho "[{s}] $*"\nexit ${{STUB_RC:-0}}\n')
        os.makedirs(os.path.join(self.root, "run"))
        os.makedirs(self.home)
        if assets:
            for a in M["assets"]:
                for p in [a["path"], *a["also"]]:
                    p = p.replace("${HOME}", self.home).replace("${REPO}", self.root).replace(
                        "${HF_HUB}", os.path.join(self.home, ".cache", "huggingface", "hub")
                    ).replace("*", "snap")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as fh:
                        fh.write("x")
        for pid in on_disk:
            d = os.path.join(self.home, "exl3-quants", pid.split("-")[1])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "config.json"), "w") as fh:
                fh.write("{}")
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
                json.dump({"weight_map": {"a": "m-1.safetensors"}}, fh)
            with open(os.path.join(d, "m-1.safetensors"), "w") as fh:
                fh.write("w")
        self.old = {k: os.environ.get(k) for k in ("HOME", "QSE_OPS_DIR", "HF_TOKEN",
                                                    "HF_HUB_CACHE", "QSE_WEIGHTS", "QSE_SECRETS")}
        os.environ.update(HOME=self.home, QSE_OPS_DIR=ops, QSE_SECRETS=os.path.join(
            self.home, ".qwen38-spark-engine", "secrets.env"))
        for k in ("HF_TOKEN", "HF_HUB_CACHE", "QSE_WEIGHTS"):
            os.environ.pop(k, None)
        with open(os.path.join(ops, "active-profile"), "w") as fh:
            fh.write("exl3-6.0\n")
        apply.ROOT = self.root
        apply.DISK_MARGIN_GB = 0              # this machine's free disk is not the test's
        device_mod.clear()

    def close(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        apply.ROOT = REPO
        apply.DISK_MARGIN_GB = 60
        device_mod.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def files(self) -> set:
        out = set()
        for d, _, fs in os.walk(self.tmp):
            for f in fs:
                out.add(os.path.relpath(os.path.join(d, f), self.tmp))
        return out


def _sel(**kw):
    return estimate.check(M, dict({"model": M["id"]}, **kw))


def test_the_overlay():
    with Box() as box:
        sel = _sel(slots=2, window=65_536, components={"vision": False},
                   advanced={"served_model": "my-model"})
        text = apply.render_overlay(M, sel)
        assert "QSE_PARALLEL=2\n" in text and "MAX_LEN=65536\n" in text
        assert "QSE_SIDE_MAX_LEN=65536\n" in text and "QSE_KV_POOL=on\n" in text
        assert 'QSE_CONDUCTOR_CONFIG="${HOME}/.qwen38-spark-engine/conductor"' in text
        assert "VISION=off\n" in text and "SERVED_MODEL=my-model\n" in text
        assert "QSE_DISK_CACHE" not in text                  # on: the profile's folder stands
        apply.write_overlay(M, sel)
        got = apply.resolve_env(["QSE_PARALLEL", "VISION", "QSE_DISK_CACHE", "SERVED_MODEL",
                                 "QWEN38_EXL3", "QSE_CONDUCTOR_CONFIG"])
        assert got["QSE_PARALLEL"] == "2" and got["VISION"] == "off", got
        assert got["SERVED_MODEL"] == "my-model"
        assert got["QSE_DISK_CACHE"].endswith("/cache/disk-exl3-6.0"), got   # the profile's
        assert got["QWEN38_EXL3"] == os.path.join(box.home, "exl3-quants", "6.0")
        assert got["QSE_CONDUCTOR_CONFIG"] == os.path.join(box.home, ".qwen38-spark-engine",
                                                           "conductor")
        bad = json.loads(json.dumps(M))
        bad["components"][6]["knobs"]["VISION"]["off"] = "$(reboot)"
        try:
            apply.overlay_values(bad, sel)
        except ValueError as exc:
            assert "VISION" in str(exc)
        else:
            raise AssertionError("a command substitution went into the overlay")
        for v in ("a b", "x'y", 'x"y', "`id`", "a\nb", "${PATH}", "$HOME/x"):
            assert not apply._value_ok(v) or v == "a b", v


def _plan(sel=None, **kw):
    return apply.plan(M, sel or _sel(), setup_mode=kw.pop("setup_mode", False),
                      password_set=kw.pop("password_set", False), **kw)


def test_the_plan():
    with Box():
        p = _plan()
        assert p["ok"], p["errors"]
        keys = {c["key"]: c for c in p["changes"]}
        assert keys["QSE_PARALLEL"]["current"] == "1" and keys["QSE_PARALLEL"]["next"] == "4"
        assert keys["QSE_PARALLEL"]["from"] == "parallelism.slots"
        assert keys["QWEN38_NARRATOR_MODEL"]["next"] == "qwen3.5-2b"
        assert "PORT" not in keys                              # unchanged keys are not listed
        assert p["profile"] == {"current": "exl3-6.0", "next": "exl3-4.5", "on_disk": True,
                                "download_gb": 0.0}, p["profile"]
        assert any("use-profile.sh exl3-4.5" in c for c in p["commands"])
        assert not any("--fetch-only" in c for c in p["commands"])
        assert any("No admin password" in w for w in p["warnings"])
        assert any("full window" in w for w in p["warnings"])
        assert p["estimate"]["fits"] is True and p["restart"] is True
        p = _plan(_sel(kv_pool=False))
        assert not p["ok"] and p["error_codes"][0] == "over_budget", p
        p = _plan(_sel(profile="exl3-3.0"))
        assert "missing_token" in p["error_codes"] and p["profile"]["download_gb"] == 13.5
        assert any("--fetch-only" in c for c in p["commands"])
        os.environ["HF_TOKEN"] = HF
        p = _plan(_sel(profile="exl3-3.0"))
        assert p["ok"], p["errors"]
        assert HF not in json.dumps(p)
        apply.DISK_MARGIN_GB = 10 ** 6
        p = _plan(_sel(profile="exl3-3.0"))
        assert p["error_codes"] == ["disk_full"] and "GB free" in p["errors"][0], p["errors"]
        apply.DISK_MARGIN_GB = 0
        os.environ.pop("HF_TOKEN")
        p = _plan(_sel(advanced={"admin_password": "generate"}))
        assert p["ok"] and any("make-secrets.sh --admin" in c for c in p["commands"])
        os.makedirs(os.path.dirname(os.environ["QSE_SECRETS"]), exist_ok=True)
        with open(os.environ["QSE_SECRETS"], "w") as fh:
            fh.write("QSE_METRICS_TOKEN=x\n")
        assert "secrets_exist" in _plan(_sel(advanced={"admin_password": "generate"}))[
            "error_codes"]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            p = _plan(_sel(advanced={"port": port}))
            assert "port_in_use" in p["error_codes"], p["errors"]
    with Box(assets=False):
        p = _plan()
        assert p["error_codes"][0] == "missing_assets" and "OPS-95" in p["errors"][0], p
        assert any("Draft models (EXL3 K3)" in e for e in p["errors"])
        assert any("Narrator is on, but its model" in w for w in p["warnings"])
    with Box(hook=False):
        assert "no_overlay_hook" in _plan()["error_codes"]
    with Box() as box:
        apply.ROOT = os.path.join(box.tmp, "elsewhere")
        assert "repo_mismatch" in _plan()["error_codes"]


class Recorder:
    def __init__(self, rc=None, out=()):
        self.calls, self.rc, self.out = [], rc or {}, out

    def __call__(self, argv, *, timeout, env, cwd, line):
        self.calls.append({"argv": list(argv), "env": dict(env), "timeout": timeout, "cwd": cwd})
        for ln in self.out:
            line(ln)
        key = " ".join(os.path.basename(a) for a in argv[1:])
        return next((rc for k, rc in self.rc.items() if key.startswith(k)), 0)


def _smoke_ok(profile="exl3-4.5"):
    def http(url, body=None, timeout=10):
        if url.endswith("/health"):
            return {"status": "ok", "weights": {"name": profile}}
        return {"choices": [{"message": {"content": "ready"}}]}
    return http


def _job(sel, setup_mode=False, previous="exl3-6.0"):
    p = _plan(sel, setup_mode=setup_mode)
    assert p["ok"], p["errors"]
    p["profile"]["current"] = previous
    job = apply.new_job(M, sel, p, setup_mode=setup_mode)
    return job["id"]


ALLOWED_ARGS = {"--fetch-only", "--setup", "--admin", "120"} | {p["id"] for p in M["profiles"]}
ALLOWED_ENV = {"PATH", "HOME", "LANG", "LC_ALL", "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE",
               "HF_TOKEN_PATH", "BOX_LOCK", "TZ", "CUDA_HOME", "SMOKE_URL", "SMOKE_MODEL"}


def test_nothing_outside_the_allowlist():
    corpus = [{}, {"profile": "exl3-2.5"}, {"profile": "nvfp4", "slots": 1},
              {"components": {"conductor": False, "narrator": False, "laya": False,
                              "shadow": False}, "advanced": {"served_model": "a.b-c:d"}},
              {"advanced": {"port": 8123, "admin_password": "generate"}},
              {"window": 8192, "kv_pool": False, "slots": 1}]
    hostile = [{"profile": "../../etc/passwd"}, {"advanced": {"served_model": "x; rm -rf ~"}},
               {"advanced": {"served_model": "$(reboot)"}}, {"steps": ["rm"]},
               {"env": {"LD_PRELOAD": "x"}}, {"path": "/etc"}, {"script": "curl x|sh"},
               {"methods": {"reasoning_effort": "low; reboot"}}, {"cmd": "id"}]
    for h in hostile:
        try:
            _sel(**h)
        except estimate.SelectionError:
            continue
        raise AssertionError(f"accepted {h}")
    for extra in corpus:
        with Box(on_disk=("exl3-6.0",)) as box:
            os.environ["HF_TOKEN"] = HF
            before = box.files()
            rec = Recorder()
            sel = _sel(**extra)
            jid = _job(sel)
            r = apply.Runner(jid, cmd=rec, http=_smoke_ok(sel["profile"]), ops=box.ops)
            assert r.main() == 0, (extra, apply.read_job(jid)["message"])
            assert rec.calls, extra
            for c in rec.calls:
                argv = c["argv"]
                assert argv[0] == "bash", argv
                assert os.path.dirname(argv[1]) == box.ops and \
                    os.path.basename(argv[1]) in apply.SCRIPTS, argv
                assert set(argv[2:]) <= ALLOWED_ARGS, argv
                assert set(c["env"]) <= ALLOWED_ENV, set(c["env"]) - ALLOWED_ENV
                assert c["cwd"] == box.root and c["timeout"] > 0
            new = {f for f in box.files() - before}
            assert all(f.startswith("repo/run/") for f in new), new
            assert "repo/run/setup.env" in new
            os.environ.pop("HF_TOKEN", None)


def _steps(jid):
    return {s["id"]: s["state"] for s in apply.read_job(jid)["steps"]}


def test_the_steps():
    with Box() as box:
        rec = Recorder()
        jid = _job(_sel())
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 0
        st = _steps(jid)
        assert st == {"check": "done", "unload_assistant": "done", "fetch": "skipped",
                      "write_config": "done", "switch": "done", "smoke": "done",
                      "rollback": "skipped"}, st
        assert [os.path.basename(a) for a in rec.calls[0]["argv"][1:]] == ["use-profile.sh",
                                                                           "exl3-4.5"]
        assert rec.calls[0]["env"]["SMOKE_URL"] == "http://127.0.0.1:8000/v1"
        assert rec.calls[0]["env"]["SMOKE_MODEL"] == "TandemLLM"
        job = apply.read_job(jid)
        assert job["state"] == "done" and "exl3-4.5 is live" in job["message"]
        assert os.path.exists(os.path.join(box.root, "run", "setup.env"))
        # again: the overlay before is kept as a backup
        jid = _job(_sel(slots=2))
        apply.Runner(jid, cmd=Recorder(), http=_smoke_ok(), ops=box.ops).main()
        assert [f for f in os.listdir(os.path.join(box.root, "run")) if ".bak-" in f]
    with Box(on_disk=()) as box:
        os.environ["HF_TOKEN"] = HF
        rec = Recorder()
        jid = _job(_sel(advanced={"port": 8124, "admin_password": "generate"}))
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 0
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["use-profile.sh exl3-4.5 --fetch-only", "stop.sh 120",
                       "make-secrets.sh --admin", "use-profile.sh exl3-4.5"], seq
        assert rec.calls[-1]["env"]["SMOKE_URL"] == "http://127.0.0.1:8124/v1"
        rec = Recorder(rc={"use-profile.sh exl3-4.5 --fetch-only": 5})
        jid = _job(_sel())
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 1
        job = apply.read_job(jid)
        assert job["failed_step"] == "fetch" and len(rec.calls) == 1, job
        assert _steps(jid)["write_config"] == "skipped"
        os.environ.pop("HF_TOKEN", None)


def test_rollback():
    with Box() as box:
        apply.write_overlay(M, _sel(slots=3))
        with open(os.path.join(box.root, "run", "setup.env")) as fh:
            old = fh.read()
        rec = Recorder(rc={"use-profile.sh exl3-4.5": 1})
        jid = _job(_sel())
        assert apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main() == 1
        job = apply.read_job(jid)
        assert job["state"] == "rolled_back" and job["rolled_back"] is True, job
        assert job["failed_step"] == "switch" and _steps(jid)["rollback"] == "done"
        with open(os.path.join(box.root, "run", "setup.env")) as fh:
            assert fh.read() == old
        seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
        assert seq == ["use-profile.sh exl3-4.5", "use-profile.sh exl3-6.0"], seq
        # a smoke that fails rolls back too; with no overlay before, the new one goes
        os.unlink(os.path.join(box.root, "run", "setup.env"))
        jid = _job(_sel())
        bad = _smoke_ok("exl3-6.0")
        assert apply.Runner(jid, cmd=Recorder(), http=bad, ops=box.ops).main() == 1
        assert apply.read_job(jid)["failed_step"] == "smoke"
        assert not os.path.exists(os.path.join(box.root, "run", "setup.env"))
        # setup mode: the marker goes before the switch and comes back with start.sh --setup
        pending = os.path.join(box.root, "run", "setup-pending")
        with open(pending, "w") as fh:
            fh.write("x")
        seen = []
        rec = Recorder(rc={"use-profile.sh exl3-4.5": 1})
        orig = rec.__call__

        def look(argv, **kw):
            seen.append((os.path.basename(argv[1]), os.path.exists(pending)))
            return orig(argv, **kw)

        jid = _job(_sel(), setup_mode=True, previous=None)
        assert apply.Runner(jid, cmd=look, http=_smoke_ok(), ops=box.ops).main() == 1
        assert seen == [("use-profile.sh", False), ("stop.sh", True), ("start.sh", True)], seen
        assert rec.calls[-1]["argv"][2:] == ["--setup"]
        assert apply.read_job(jid)["state"] == "rolled_back"


class Req(app.Handler):
    def __init__(self, method, path, body=None, headers=None):
        raw = json.dumps(body if body is not None else {}).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = dict({"Content-Length": str(len(raw))}, **(headers or {}))
        self.path, self.command, self.request_version = path, method, "HTTP/1.1"
        self.requestline, self.client_address = f"{method} {path}", ("127.0.0.1", 0)
        self.close_connection = True

    def go(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            getattr(self, f"do_{self.command}")()
        head, _, raw = self.wfile.getvalue().partition(b"\r\n\r\n")
        return int(head.decode().split(" ", 2)[1]), head.decode(), raw.decode()


def _api(method, path, body=None, headers=None):
    code, head, raw = Req(method, path, body, headers).go()
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, raw


class FakeWren:
    def __init__(self):
        self.unloads = []

    def unload(self, why="asked"):
        self.unloads.append(why)

    def status(self):
        return {"name": "Wren", "state": "not_loaded"}


def test_the_routes():
    with Box():
        app.STATE.clear()
        w = FakeWren()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=w)
        app.INFLIGHT.update(running=0, waiting=0)
        spawned = []
        real = apply.spawn
        apply.spawn = lambda jid: spawned.append(jid) or 4242
        try:
            sel = {"model": M["id"]}
            for extra in ({"steps": []}, {"cmd": "id"}, {"script": "x"}, {"env": {}},
                          {"path": "/tmp"}):
                code, b = _api("POST", "/v1/setup/apply", dict({"selection": sel}, **extra))
                assert code == 422 and b["error"]["code"] == "unknown_field", (extra, b)
            for bad in ({"profile": "../../etc"}, {"advanced": {"served_model": "a;b"}}):
                code, b = _api("POST", "/v1/setup/apply", {"selection": dict(sel, **bad)})
                assert code in (400, 422), (bad, code, b)
            code, b = _api("POST", "/v1/setup/apply", {"selection": sel, "dry_run": True})
            assert code == 200 and b["dry_run"] is True and b["ok"], b
            assert not os.path.exists(apply.jobs_dir()) or not os.listdir(apply.jobs_dir())
            code, b = _api("POST", "/v1/setup/plan", sel)
            assert code == 200 and b["dry_run"] is False and b["changes"]
            app.INFLIGHT["running"] = 1
            code, b = _api("POST", "/v1/setup/apply", {"selection": sel})
            assert code == 409 and b["error"]["code"] == "busy" and b["error"]["running"] == 1
            app.INFLIGHT["running"] = 0
            code, b = _api("POST", "/v1/setup/apply", {"selection": dict(sel, kv_pool=False)})
            assert code == 409 and b["error"]["code"] == "over_budget" and "plan" in b
            code, b = _api("POST", "/v1/setup/apply", {"selection": sel})
            assert code == 202 and b["job"]["state"] == "queued", b
            assert spawned == [b["job"]["id"]] and w.unloads == ["apply"]
            code, b2 = _api("POST", "/v1/setup/apply", {"selection": sel})
            assert code == 409 and b2["error"]["code"] == "job_running", b2
        finally:
            apply.spawn = real
    with Box(assets=False):
        app.STATE.update(wren=FakeWren())
        code, b = _api("POST", "/v1/setup/apply", {"selection": {"model": M["id"]}})
        assert code == 412 and b["error"]["code"] == "missing_assets", b


def test_the_job_route():
    with Box() as box:
        app.STATE.clear()
        app.STATE.update(auth=auth.Auth(None, None), started=time.time(), wren=FakeWren())
        jid = _job(_sel())
        apply.Runner(jid, cmd=Recorder(out=["a line"]), http=_smoke_ok(), ops=box.ops).main()
        code, j = _api("GET", f"/v1/setup/jobs/{jid}?format=json")
        assert code == 200 and j["state"] == "done" and j["id"] == jid
        code, head, raw = Req("GET", f"/v1/setup/jobs/{jid}", None,
                              {"Accept": "text/event-stream"}).go()
        assert code == 200 and "text/event-stream" in head
        frames = [f for f in raw.split("\n\n") if f.strip()]
        evs = [json.loads(f.split("data: ", 1)[1]) for f in frames]
        ids = [int(f.split("\n")[0][4:]) for f in frames if f.startswith("id: ")]
        assert evs[0]["type"] == "job" and evs[-1]["type"] == "done", evs[-1]
        assert ids == sorted(ids) and ids[0] == 1
        assert any(e["type"] == "log" and e["line"] == "a line" for e in evs)
        assert any(e["type"] == "step" and e["step"]["id"] == "smoke" for e in evs)
        code, _, raw = Req("GET", f"/v1/setup/jobs/{jid}", None,
                           {"Accept": "text/event-stream", "Last-Event-ID": str(ids[-2])}).go()
        again = [f for f in raw.split("\n\n") if f.strip()]
        assert len(again) == 1 and again[0].startswith(f"id: {ids[-1]}"), again
        for bad in ("0123456789abcdef", "../../etc", "nope"):
            code, _ = _api("GET", f"/v1/setup/jobs/{bad}?format=json")
            assert code == 404, bad


def test_secrets_are_redacted():
    with Box() as box:
        os.environ["HF_TOKEN"] = HF
        jid = _job(_sel())
        rec = Recorder(out=[f"token {HF} here", "Authorization: Bearer abcdefabcdefabcdefabcdef12",
                            "hf_" + "A1b2C3" * 6])
        apply.Runner(jid, cmd=rec, http=_smoke_ok(), ops=box.ops).main()
        os.environ.pop("HF_TOKEN")
        blob = open(os.path.join(apply.job_dir(jid), "events.jsonl")).read() + \
            open(os.path.join(apply.job_dir(jid), "state.json")).read()
        for s in (HF, "abcdefabcdefabcdefabcdef12", "A1b2C3A1b2C3"):
            assert s not in blob, s
        assert "[redacted" in blob


class Engine(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._send({"status": "ok", "weights": {"name": "exl3-4.5"}})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send({"choices": [{"message": {"content": "ready"}}]})

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def test_the_detached_runner():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old_pp = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = REPO
    try:
        with Box(port=port) as box:
            jid = _job(_sel(advanced={"port": port}))
            pid = apply.spawn(jid)
            t0 = time.time()
            while time.time() - t0 < 60:
                j = apply.read_job(jid)
                if j["state"] not in ("queued", "running"):
                    break
                time.sleep(0.2)
            out = open(os.path.join(apply.job_dir(jid), "runner.out")).read()
            assert j["state"] == "done", (j, out)
            assert j["pid"] == pid
            assert os.getsid(0) != _sid(pid, os.getsid(0))
            evs = [json.loads(x)["event"] for x in open(os.path.join(
                apply.job_dir(jid), "events.jsonl"))]
            assert any(e.get("line", "").startswith("[use-profile.sh] exl3-4.5") for e in evs), evs
            # the status responder answers for the job while the engine is away
            r = subprocess.run([sys.executable, "-c", (
                "import sys, json; sys.path.insert(0, sys.argv[1]); import importlib.util as u; "
                "s = u.spec_from_file_location('es', sys.argv[1] + '/ops/engine_status.py'); "
                "m = u.module_from_spec(s); s.loader.exec_module(m); m.REPO = sys.argv[2]; "
                "import io; h = m.Handler.__new__(m.Handler); h.command = 'GET'; "
                "h.path = '/v1/setup/jobs/' + sys.argv[3]; h.wfile = io.BytesIO(); "
                "h.request_version = 'HTTP/1.1'; h.requestline = 'GET'; h.client_address = ('x', 0); "
                "h.headers = {}; h._send = lambda c, b, t, r: print(c, b.decode()); h._answer()"),
                REPO, box.root, jid], capture_output=True, text=True, timeout=30)
            code, _, body = r.stdout.partition(" ")
            assert code == "200" and json.loads(body)["id"] == jid, r.stdout + r.stderr
    finally:
        srv.shutdown()
        if old_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pp


def _sid(pid, default):
    try:
        return os.getsid(pid)
    except ProcessLookupError:
        return default + 1          # gone already: it ran in a session of its own


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
