"""The apply job (SRV-327): a checked selection becomes the engine's configuration, through a fixed
list of steps in this file and the repository's own scripts. Nothing a client sends is run.

    POST /v1/setup/plan {selection}             the review step's facts; nothing changes
    POST /v1/setup/apply {selection, dry_run?}  dry_run: the plan; else 202 and a job
    GET  /v1/setup/jobs/{id}                    SSE of the job's events (`id:` = sequence,
                                                Last-Event-ID resumes); ?format=json: the job

THE STEPS (STEPS below; the argv of every command is built here from the registry's ids and the
selection's checked numbers, `shell=False`, a minimal environment, a timeout each):

    check             the selection again (the runner checks it itself), the estimate, the disk,
                      the assets, the HF token, the overlay hook in serve.env, the box lock
    unload_assistant  Wren unloaded (the server does it before it starts the runner)
    fetch             bash ops/use-profile.sh <profile> --fetch-only    (skipped when on disk)
    write_config      run/setup.env from the registry's keys and the selection, the old one kept
                      as run/setup.env.bak-<utc>, written whole and renamed into place; with a
                      port change the engine on the old port is stopped first (bash ops/stop.sh);
                      with `admin_password: generate`, bash ops/make-secrets.sh --admin
    switch            bash ops/use-profile.sh <profile>: the box lock, stop, start, its smoke,
                      its own rollback of the profile (in setup mode the marker run/setup-pending
                      goes first, so start.sh starts the configured engine)
    smoke             /health names the profile; one short chat answers
    rollback          on a failed switch or smoke: the old run/setup.env back, then
                      bash ops/use-profile.sh <previous profile> (setup mode: the marker back and
                      bash ops/start.sh --setup)

ops/serve.env is never edited: it reads run/setup.env after the weight profile. The runner is a
process of its own (`python -m server.setup.apply run <job folder>`, its own session), because
the switch replaces the server that started it; it writes run/setup-jobs/<id>/state.json and events.jsonl,
which the restarted server (and ops/engine_status.py's responder meanwhile) reads. Log lines pass
the conductor's secret filter, and the HF and admin tokens are cut out of them by value.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request

from server.setup import device as device_mod
from server.setup import estimate as estimate_mod
from server.setup import registry

# the checkout this server runs from; serve.env's REPO must be the same folder (the plan checks),
# since serve.env reads ${REPO}/run/setup.env and start.sh ${REPO}/run/setup-pending
ROOT = registry.ROOT


def run_dir() -> str:
    return os.path.join(ROOT, "run")


def overlay_path() -> str:
    return os.path.join(run_dir(), "setup.env")


def pending_path() -> str:
    return os.path.join(run_dir(), "setup-pending")


def jobs_dir() -> str:
    return os.path.join(run_dir(), "setup-jobs")


DISK_MARGIN_GB = 60          # ops/use-profile.sh's: the disk tier and the spill grow into it
SCRIPTS = ("use-profile.sh", "start.sh", "stop.sh", "make-secrets.sh")
TIMEOUTS = {"fetch": 7200, "switch": 1800, "stop": 300, "secrets": 60, "start": 900}
JOB_ID = re.compile(r"^[a-f0-9]{16}$")
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
STEPS = (("check", "Check the selection, the disk and the box"),
         ("unload_assistant", "Unload Wren"),
         ("fetch", "Download the weights"),
         ("write_config", "Write the configuration (run/setup.env)"),
         ("switch", "Restart the engine with the new setup"),
         ("smoke", "Check that the new engine answers"),
         ("rollback", "Roll back"))
LABELS = dict(STEPS)


class PlanError(ValueError):
    """apply refused before anything changed: `status` 409 or 412, `code`, the plan."""

    def __init__(self, status: int, code: str, message: str, plan: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.plan = status, code, plan

    def body(self) -> dict:
        return {"error": {"type": self.code, "code": self.code, "message": str(self)},
                **({"plan": self.plan} if self.plan is not None else {})}


# ---------------------------------------------------------------------- the overlay
def _value_ok(v: str) -> bool:
    """A value that goes into run/setup.env: one line, no quote, backslash, backtick or command
    substitution (`${HOME}` from the registry is the one expansion it keeps)."""
    return isinstance(v, str) and not re.search(r"[\n\r\"'`\\]|\$\(", v) and \
        all(x == "${HOME}" for x in re.findall(r"\$\{?[A-Za-z_]*\}?", v))


def overlay_values(m: dict, sel: dict) -> dict:
    """serve.env key -> value for a checked selection, in the order they are written; a tool's
    `null` state writes nothing (the profile's value stands)."""
    out = {}
    p = m["parallelism"]
    k = p["knobs"]
    out[k["slots"]] = str(sel["slots"])
    out[k["window"]] = str(sel["window"])
    out[k["side_window"]] = str(sel["side_window"])
    out[k["kv_pool"]] = "on" if sel["kv_pool"] else "off"
    out[k["floor_gb"]] = f"{sel['floor_gb']:g}"
    out.update(p.get("fixed") or {})
    for c in m["components"]:
        on = sel["components"][c["id"]]
        for key, v in c["knobs"].items():
            val = v["on" if on else "off"]
            if val is not None:
                out[key] = val
    for group in ("methods", "advanced"):
        for f in m[group]:
            if f["knob"]:
                v = sel[group][f["id"]]
                out[f["knob"]] = ("true" if v else "false") if isinstance(v, bool) else \
                    (f"{v:g}" if isinstance(v, float) else str(v))
    bad = [key for key, v in out.items() if not re.match(registry.ENV_KEY, key) or not _value_ok(v)]
    if bad:
        raise ValueError(f"values that cannot go into run/setup.env: {', '.join(bad)}")
    return out


def render_overlay(m: dict, sel: dict, when: str | None = None) -> str:
    when = when or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [f"# Written by the setup guide (server/setup/apply.py) on {when}: {m['id']}, profile "
             f"{sel['profile']}.",
             "# Generated: the next apply replaces it and reset level 2 removes it. ops/serve.env",
             "# reads it after the weight profile, so it never sets the weights."]
    for key, v in overlay_values(m, sel).items():
        lines.append(f'{key}="{v}"' if "$" in v else f"{key}={v}")
    return "\n".join(lines) + "\n"


def hook_present(ops: str | None = None) -> bool:
    """ops/serve.env reads the overlay (a box whose serve.env predates the setup guide does not)."""
    try:
        with open(os.path.join(ops or device_mod.ops_dir(), "serve.env")) as fh:
            return "run/setup.env" in fh.read()
    except OSError:
        return False


def resolve_env(keys: list[str], *, profile: str | None = None, overlay: str | None = None,
                ops: str | None = None, timeout: float = 10.0) -> dict:
    """key -> value as ops/serve.env resolves them (QSE_PROFILE_OVERRIDE `profile`, QSE_SETUP_ENV
    `overlay`), in a bash with nothing of this process's environment but PATH and HOME."""
    env_file = os.path.join(ops or device_mod.ops_dir(), "serve.env")
    if not os.path.isfile(env_file):
        return {}
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
           "ENVF": env_file}
    if profile:
        env["QSE_PROFILE_OVERRIDE"] = profile
    if overlay:
        env["QSE_SETUP_ENV"] = overlay
    script = ('set -a; . "$ENVF" >/dev/null 2>&1; for k in "$@"; do '
              'if [ -n "${!k+x}" ]; then printf "%s\\t%s\\n" "$k" "${!k}"; fi; done')
    try:
        r = subprocess.run(["bash", "-c", script, "-", *keys], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("\t")
        if k in keys:
            out[k] = v
    return out


# ---------------------------------------------------------------------- the plan
def secrets_file() -> str:
    return os.environ.get("QSE_SECRETS") or os.path.expanduser("~/.qwen38-spark-engine/secrets.env")


def plan(m: dict, sel: dict, *, setup_mode: bool, password_set: bool,
         available_gb: float | None = None, mem_total_gb: float | None = None) -> dict:
    """The review step's facts (SetupPlan): what is wrong, what changes, what will run."""
    est = estimate_mod.estimate(m, sel, available_gb=available_gb)
    prof = registry.profile(m, sel["profile"])
    disk = device_mod.on_disk(m, fresh=True).get(prof["id"], {})
    errors, codes, warnings = [], [], []

    def error(code: str, text: str) -> None:
        errors.append(text)
        codes.append(code)

    if not est["fits"]:
        error("over_budget", f"The selection needs {est['total_gb']:.2f} GiB, {est['total_gb'] - est['budget_gb']:.2f}"
                      f" GiB more than the {est['budget_gb']:.2f} GiB there is. "
                      + (est["notes"][-1] if est["notes"] else ""))
    missing = [registry.asset(m, a) for a in disk.get("assets_missing", [])]
    if missing:
        error("missing_assets", "This profile needs files that are not on this box and that nothing here "
                      "can download yet: " + "; ".join(
                          f"{a['label']} ({registry.expand(a['path'], disk.get('repo'))})" for a in missing)
                      + ". See OPS-95 and docs/setup.md.")
    on_disk = bool(disk.get("weights"))
    if not on_disk and prof["hf_private"] and not device_mod.hf_token_present():
        error("missing_token", f"{prof['hf_repo']} is private and no Hugging Face token is set. Add one "
                      f"with `hf auth login` on the box, or set HF_TOKEN, then try again.")
    if not on_disk:
        try:
            free = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
        except OSError:
            free = None
        if free is not None and free < prof["disk_gb"] + DISK_MARGIN_GB:
            error("disk_full", f"The download needs {prof['disk_gb']:g} GB and {DISK_MARGIN_GB} GB "
                          f"left after it; the disk has {free:.0f} GB free.")
    if not hook_present():
        error("no_overlay_hook", "ops/serve.env on this box does not read run/setup.env. Add the setup "
                      "overlay lines from the repository's ops/serve.env (the end of the file) "
                      "first.")
    if sel["advanced"].get("admin_password") == "generate" and os.path.exists(secrets_file()):
        error("secrets_exist", "secrets.env already exists. ops/make-secrets.sh --rotate --admin would "
                      "replace the metrics token too, so run it on the box yourself.")
    keys = list(overlay_values(m, sel))
    current = resolve_env(keys + ["REPO"])
    repo = current.pop("REPO", None)
    if repo and os.path.realpath(repo) != os.path.realpath(ROOT):
        error("repo_mismatch", f"ops/serve.env's REPO is {repo}, but this server runs from "
                               f"{ROOT}; the overlay would not be read.")
    with_overlay = _preview(m, sel, keys)
    changes = []
    for k in keys:
        cur, nxt = current.get(k), with_overlay.get(k)
        if cur != nxt:
            changes.append({"key": k, "current": cur, "next": nxt, "from": _source(m, k)})
    old_port = int(current.get("PORT") or 8000) if (current.get("PORT") or "").isdigit() else 8000
    new_port = int(sel["advanced"].get("port") or old_port)
    if new_port != old_port and _port_taken(new_port):
        error("port_in_use", f"Port {new_port} is in use by another program on this box.")
    if not password_set:
        warnings.append("No admin password is set: anyone who can reach this port can run the "
                        "setup, apply it and reset the box. Generate one in the advanced step.")
    if est["fits"] and not est["fits_worst"]:
        warnings.append(est["notes"][0])
    for c in m["components"]:
        if sel["components"][c["id"]]:
            for a in c["assets"]:
                if not registry.asset_present(registry.asset(m, a), disk.get("repo")):
                    warnings.append(f"{c['label']} is on, but its model is not on this box "
                                    f"({registry.expand(registry.asset(m, a)['path'], disk.get('repo'))});"
                                    f" the engine starts without it.")
    if mem_total_gb is not None and device_mod.preset(m, mem_total_gb, None) is None:
        warnings.append("These numbers were measured on a DGX Spark, not on this machine.")
    active = device_mod.active_profile()
    commands = []
    if not on_disk:
        commands.append(f"bash ops/use-profile.sh {prof['id']} --fetch-only   "
                        f"(about {prof['disk_gb']:g} GB from {prof['hf_repo']})")
    if new_port != old_port:
        commands.append(f"bash ops/stop.sh 120   (the engine on port {old_port})")
    commands.append(f"write run/setup.env ({len(keys)} keys; the old one kept as a .bak)")
    if sel["advanced"].get("admin_password") == "generate":
        commands.append("bash ops/make-secrets.sh --admin")
    commands.append(f"bash ops/use-profile.sh {prof['id']}   (stop, start, smoke, roll back on "
                    f"failure)")
    commands.append(f"smoke: GET http://127.0.0.1:{new_port}/health and one short chat")
    return {"contract_version": "1.0", "ok": not errors, "errors": errors, "error_codes": codes,
            "warnings": warnings,
            "profile": {"current": None if setup_mode and not active else active,
                        "next": prof["id"], "on_disk": on_disk,
                        "download_gb": 0.0 if on_disk else prof["disk_gb"]},
            "changes": changes, "restart": True, "commands": commands, "estimate": est,
            "ports": {"current": old_port, "next": new_port}}


def _preview(m: dict, sel: dict, keys: list[str]) -> dict:
    import tempfile
    fd, path = tempfile.mkstemp(prefix="setup-preview-", suffix=".env")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(render_overlay(m, sel))
        return resolve_env(keys, profile=sel["profile"], overlay=path)
    finally:
        os.unlink(path)


def _source(m: dict, key: str) -> str:
    return registry.env_keys(m).get(key, "")


def _port_taken(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------------- jobs
def job_dir(jid: str) -> str:
    if not JOB_ID.match(jid or ""):
        raise ValueError("not a job id")
    return os.path.join(jobs_dir(), jid)


def _write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def read_job(jid: str) -> dict | None:
    """The job's state (SetupJob), with a runner that died marked failed (interrupted)."""
    try:
        with open(os.path.join(job_dir(jid), "state.json")) as fh:
            job = json.load(fh)
    except (OSError, ValueError):
        return None
    starting = job["state"] == "queued" and time.time() - job.get("created_at", 0) < 30
    if job["state"] in ("queued", "running") and not starting and not _alive(job.get("pid")):
        job.update(state="failed", message="interrupted: the job's process is gone",
                   ended_at=job.get("ended_at") or time.time())
    return job


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                  # a zombie child of this server
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def running_job() -> dict | None:
    try:
        ids = sorted(os.listdir(jobs_dir()))
    except OSError:
        return None
    for jid in ids:
        if JOB_ID.match(jid):
            j = read_job(jid)
            if j is not None and j["state"] in ("queued", "running"):
                return j
    return None


def events(jid: str, after: int = 0) -> list[dict]:
    """The job's events after sequence `after`, each {seq, event}."""
    out = []
    try:
        with open(os.path.join(job_dir(jid), "events.jsonl")) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("seq", 0) > after:
                    out.append(rec)
    except OSError:
        pass
    return out


def new_job(m: dict, sel: dict, p: dict, *, setup_mode: bool) -> dict:
    """The job's folder and its job.json (what the runner reads), state queued."""
    jid = secrets.token_hex(8)
    d = job_dir(jid)
    os.makedirs(d, mode=0o700)
    now = time.time()
    steps = [{"id": sid, "label": label, "state": "pending", "started_at": None, "ended_at": None,
              "detail": None} for sid, label in STEPS]
    job = {"id": jid, "state": "queued", "created_at": now, "ended_at": None,
           "profile": sel["profile"], "steps": steps, "message": None, "changes": p["changes"],
           "rolled_back": False, "failed_step": None, "pid": None,
           "url": f"http://<this box>:{p['ports']['next']}/dashboard/"}
    _write_json(os.path.join(d, "job.json"),
                {"model": m["id"], "selection": sel, "setup_mode": setup_mode,
                 "previous": p["profile"]["current"], "ports": p["ports"],
                 "on_disk": p["profile"]["on_disk"]})
    _write_json(os.path.join(d, "state.json"), job)
    return job


def spawn(jid: str) -> int:
    """The runner, detached: its own session, so the engine's restart does not take it along."""
    d = job_dir(jid)
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "HF_TOKEN", "HF_HOME",
                    "HF_HUB_CACHE", "HF_TOKEN_PATH", "QSE_OPS_DIR", "BOX_LOCK", "TZ",
                    "CUDA_HOME")}
    with open(os.path.join(d, "runner.out"), "ab") as out:
        p = subprocess.Popen([sys.executable, "-m", "server.setup.apply", "run", d], cwd=ROOT,
                             env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                             start_new_session=True, close_fds=True)
    return p.pid


# ---------------------------------------------------------------------- the runner
def _run_cmd(argv: list[str], *, timeout: float, env: dict, cwd: str, line) -> int:
    """One allowlisted command, its output line by line to `line`; the exit status (124 after
    the timeout)."""
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    t_end = time.monotonic() + timeout
    import selectors
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    while True:
        if time.monotonic() > t_end:
            p.kill()
            p.wait()
            line(f"[setup] {os.path.basename(argv[1])}: no end after {timeout:.0f} s, stopped")
            return 124
        if sel.select(timeout=1.0):
            ln = p.stdout.readline()
            if ln:
                line(ln.rstrip("\n"))
                continue
        if p.poll() is not None:
            for ln in p.stdout:
                line(ln.rstrip("\n"))
            return p.returncode


class Runner:
    """Runs one job's steps. `cmd(argv, timeout=, env=, cwd=, line=) -> rc` and
    `http(url, body=None, timeout=) -> dict` are injectable (tests/test_setup_apply.py records
    every argv)."""

    def __init__(self, jid: str, *, cmd=_run_cmd, http=None, ops: str | None = None,
                 clock=time.time):
        self.jid, self.dir = jid, job_dir(jid)
        self.cmd, self.http = cmd, http or _http
        self.ops = ops or device_mod.ops_dir()
        self.clock = clock
        with open(os.path.join(self.dir, "job.json")) as fh:
            self.spec = json.load(fh)
        with open(os.path.join(self.dir, "state.json")) as fh:
            self.job = json.load(fh)
        self.seq = len(events(jid))
        self.secrets = [v for v in (os.environ.get("HF_TOKEN"), os.environ.get("QSE_ADMIN_TOKEN"))
                        if v]

    # -- bookkeeping
    def _emit(self, ev: dict) -> None:
        self.seq += 1
        with open(os.path.join(self.dir, "events.jsonl"), "a") as fh:
            fh.write(json.dumps({"seq": self.seq, "event": ev}) + "\n")

    def _save(self) -> None:
        _write_json(os.path.join(self.dir, "state.json"), self.job)

    def log(self, text: str) -> None:
        from server.conductor.redact import redact
        for s in self.secrets:
            text = text.replace(s, "[redacted]")
        text, _ = redact(text)
        self._emit({"type": "log", "line": text[:2000], "ts": round(self.clock(), 3)})

    def step(self, sid: str, state: str, detail: str | None = None) -> None:
        st = next(s for s in self.job["steps"] if s["id"] == sid)
        now = self.clock()
        if state == "running":
            st["started_at"] = now
        elif state in ("done", "failed", "skipped"):
            st["ended_at"] = now
            st["started_at"] = st["started_at"] or now
        st["state"], st["detail"] = state, detail
        self._save()
        self._emit({"type": "step", "step": dict(st)})

    def argv(self, script: str, *args: str) -> list[str]:
        """An allowlisted script of ops/ and its arguments (registry ids and numbers only)."""
        if script not in SCRIPTS:
            raise ValueError(f"{script} is not one of the setup job's scripts")
        for a in args:
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", a):
                raise ValueError(f"argument {a!r} is not an id or a number")
        return ["bash", os.path.join(self.ops, script), *args]

    def env(self, **extra) -> dict:
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE",
                "HF_TOKEN_PATH", "BOX_LOCK", "TZ", "CUDA_HOME")
        e = {k: v for k, v in os.environ.items() if k in keep}
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def run(self, argv: list[str], timeout: float, **env) -> int:
        self.log(f"$ {' '.join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in argv)}")
        return self.cmd(argv, timeout=timeout, env=self.env(**env), cwd=ROOT, line=self.log)

    # -- the job
    def main(self) -> int:
        self.job.update(state="running", pid=os.getpid())
        self._save()
        self._emit({"type": "job", "job": dict(self.job)})
        self.step("unload_assistant", "done", "Wren was unloaded before the job started")
        ok = False
        try:
            ok = self._steps()
        except Exception as exc:                                  # noqa: BLE001
            self.log(f"[setup] the job stopped: {type(exc).__name__}: {exc}")
            self.job["message"] = f"{type(exc).__name__}: {exc}"[:400]
            ok = False
        end = self.job["state"] if self.job["state"] in ("done", "rolled_back") else (
            "done" if ok else "failed")
        self.job.update(state=end, ended_at=self.clock())
        for s in self.job["steps"]:
            if s["state"] == "pending":
                s["state"] = "skipped"
        self._save()
        self._emit({"type": "done", "state": end, "message": self.job["message"]})
        return 0 if end == "done" else 1

    def _fail(self, sid: str, why: str) -> bool:
        self.step(sid, "failed", why)
        self.job.update(failed_step=sid, message=why)
        return False

    def _steps(self) -> bool:
        spec = self.spec
        self.step("check", "running")
        m = registry.get(spec["model"])
        if m is None:
            return self._fail("check", f"no model {spec['model']!r} in models/")
        try:
            sel = estimate_mod.check(m, spec["selection"])
        except estimate_mod.SelectionError as exc:
            return self._fail("check", f"the selection does not check: {exc}")
        if not PROFILE_ID.match(sel["profile"]) or not os.path.isfile(
                os.path.join(self.ops, "profiles", sel["profile"] + ".env")):
            return self._fail("check", f"no ops/profiles/{sel['profile']}.env on this box")
        prev = spec.get("previous")
        if prev is not None and (not PROFILE_ID.match(prev) or not os.path.isfile(
                os.path.join(self.ops, "profiles", prev + ".env"))):
            prev = None
        p = plan(m, sel, setup_mode=spec["setup_mode"], password_set=True)
        if p["errors"]:
            return self._fail("check", " ".join(p["errors"]))
        if _box_lock_held():
            return self._fail("check", "the box lock (~/.qwen38-box.flock) is held: a hold or "
                                       "another job owns the board; try again later")
        self.step("check", "done", f"{len(p['changes'])} settings change")
        # fetch
        if p["profile"]["on_disk"]:
            self.step("fetch", "skipped", "the weights are on this box")
        else:
            self.step("fetch", "running", f"about {p['profile']['download_gb']:g} GB")
            rc = self.run(self.argv("use-profile.sh", sel["profile"], "--fetch-only"),
                          TIMEOUTS["fetch"])
            if rc != 0:
                return self._fail("fetch", f"the download did not finish (exit {rc}); nothing "
                                           f"else changed")
            self.step("fetch", "done")
        # write_config
        self.step("write_config", "running")
        old_port, new_port = spec["ports"]["current"], spec["ports"]["next"]
        if new_port != old_port:
            self.run(self.argv("stop.sh", "120"), TIMEOUTS["stop"])
        backup = write_overlay(m, sel, self.clock)
        self.job["backup"] = backup
        if sel["advanced"].get("admin_password") == "generate":
            rc = self.run(self.argv("make-secrets.sh", "--admin"), TIMEOUTS["secrets"])
            if rc != 0:
                restore_overlay(backup)
                return self._fail("write_config", f"ops/make-secrets.sh failed (exit {rc}); the "
                                                  f"old configuration is back")
        self.step("write_config", "done", f"run/setup.env, {len(overlay_values(m, sel))} keys"
                                          + (f"; the old one is {os.path.basename(backup)}"
                                             if backup else ""))
        # switch
        self.step("switch", "running", "about 4 minutes; the dashboard is away meanwhile")
        if spec["setup_mode"] and os.path.exists(pending_path()):
            os.unlink(pending_path())
        smoke_model = "TandemLLM" if sel["components"].get("conductor") else \
            sel["advanced"].get("served_model", "qwen38-spark-engine")
        rc = self.run(self.argv("use-profile.sh", sel["profile"]), TIMEOUTS["switch"],
                      SMOKE_URL=f"http://127.0.0.1:{new_port}/v1", SMOKE_MODEL=smoke_model)
        if rc != 0:
            self._fail("switch", f"the new setup did not start (ops/use-profile.sh exit {rc})")
            return self._rollback(m, spec, backup, prev, old_port)
        self.step("switch", "done")
        # smoke
        self.step("smoke", "running")
        why = self._smoke(sel["profile"], new_port, smoke_model)
        if why:
            self._fail("smoke", why)
            return self._rollback(m, spec, backup, prev, old_port)
        self.step("smoke", "done", f"{sel['profile']} answers on port {new_port}")
        self.job["message"] = f"{sel['profile']} is live on port {new_port}."
        return True

    def _smoke(self, profile: str, port: int, model: str) -> str | None:
        base = f"http://127.0.0.1:{port}"
        try:
            h = self.http(base + "/health", timeout=10)
        except Exception as exc:                                  # noqa: BLE001
            return f"/health did not answer ({type(exc).__name__})"
        if ((h or {}).get("weights") or {}).get("name") != profile:
            return f"/health does not name {profile}"
        body = {"model": model, "max_tokens": 16, "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = self.http(base + "/v1/chat/completions", body=body, timeout=300)
            text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        except Exception as exc:                                  # noqa: BLE001
            return f"the chat did not answer ({type(exc).__name__})"
        if not text.strip():
            return "the chat answered with no text"
        self.log(f"[smoke] {text.strip()[:60]}")
        return None

    def _rollback(self, m, spec, backup, prev, old_port) -> bool:
        self.step("rollback", "running")
        restore_overlay(backup)
        if spec["setup_mode"]:
            os.makedirs(run_dir(), exist_ok=True)
            with open(pending_path(), "w") as fh:
                fh.write("setup mode: the apply job rolled back\n")
            self.run(self.argv("stop.sh", "120"), TIMEOUTS["stop"])
            rc = self.run(self.argv("start.sh", "--setup"), TIMEOUTS["start"])
        elif prev is not None:
            rc = self.run(self.argv("use-profile.sh", prev), TIMEOUTS["switch"],
                          SMOKE_URL=f"http://127.0.0.1:{old_port}/v1")
        else:
            rc = self.run(self.argv("start.sh"), TIMEOUTS["start"])
        if rc == 0:
            self.step("rollback", "done", "the previous setup runs again")
            self.job.update(state="rolled_back", rolled_back=True)
        else:
            self.step("rollback", "failed", f"the previous setup did not start either (exit {rc}); "
                                            f"see logs/engine.log on the box")
            self.job["rolled_back"] = False
        return False


def write_overlay(m: dict, sel: dict, clock=time.time) -> str | None:
    """run/setup.env, written whole and renamed into place after `bash -n`; the one it replaces
    kept as run/setup.env.bak-<utc>. Returns that backup's path (None: there was none)."""
    os.makedirs(run_dir(), exist_ok=True)
    text = render_overlay(m, sel)
    overlay = overlay_path()
    tmp = overlay + f".tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        os.unlink(tmp)
        raise ValueError(f"the generated overlay does not parse: {r.stderr.strip()[:200]}")
    backup = None
    if os.path.exists(overlay):
        backup = overlay + time.strftime(".bak-%Y%m%dT%H%M%SZ", time.gmtime(clock()))
        shutil.copy2(overlay, backup)
    os.replace(tmp, overlay)
    return backup


def restore_overlay(backup: str | None) -> None:
    overlay = overlay_path()
    if backup and os.path.exists(backup):
        shutil.copy2(backup, overlay)
    elif os.path.exists(overlay):
        os.unlink(overlay)


def _box_lock_held() -> bool:
    from server.decider_api import held
    return held(service_start=False, hold_restart=False)


def _http(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "X-Tandem-Memory": "off"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def main(argv: list[str] | None = None) -> int:
    """`run <checkout>/run/setup-jobs/<id>`: the job the server wrote, in the checkout it serves."""
    global ROOT
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "run":
        d = os.path.realpath(a[1])
        jobs = os.path.dirname(d)
        if JOB_ID.match(os.path.basename(d)) and os.path.basename(jobs) == "setup-jobs" and \
                os.path.basename(os.path.dirname(jobs)) == "run":
            ROOT = os.path.dirname(os.path.dirname(jobs))
            return Runner(os.path.basename(d)).main()
    print("usage: python -m server.setup.apply run <checkout>/run/setup-jobs/<id>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
