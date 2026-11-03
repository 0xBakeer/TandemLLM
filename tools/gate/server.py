"""One arm's tree and server: a `git archive` of the commit, its own config copy and state directory,
a fresh server on a free port of its own, the limit knobs read from the running process, and a stop
that leaves nothing behind. Never touches a server it did not start.

Two launchers (tools/gate/packs.yaml `launcher`):

  ops    the tree's own ops/start.sh with HOLD_RESTART=1, its ops/serve.env REPO set to the tree and
         run/setup.env written for the arm: the port, the conductor raw, no ledger, no suffix store,
         no response cache, no disk tier, no spill, no small models, its own conductor and backends
         copies, the hardware profile's environment and the variant's. ops/stop.sh stops it.
  fake   server/app.py --fake-engine (no model, no GPU), for the gate's own tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tools import parallel_card as pc
from tools.row3 import MemGuard, code_hash

#: a gate server never takes a port below this one (the box's own services live there)
MIN_PORT = 8100
#: the environment an arm's setup.env sets on top of the box's (the bare harness's overrides)
ARM_OVERRIDES = {
    "HOST": "127.0.0.1", "CONDUCTOR": "raw", "RESPONSE_CACHE": "off", "QSE_USAGE_LEDGER": "off",
    "QWEN38_DECIDER_MODELS": "", "QWEN38_NARRATOR_MODEL": "", "QWEN38_SHADOW_MODEL": "",
    "ROLES_AUTOLOAD": "", "QWEN38_SUFFIX_STORE": "", "QSE_SUFFIX_STORE": "", "QWEN38_NOTES_HEAD": "off",
    "QSE_STATUS_PORT": "", "QSE_DISK_CACHE": "off", "SESSION_SPILL": "off", "QSE_REQ_PEAK": "1",
    "QSE_CHATD": "off", "QSE_DASHBOARD_LOGIN": "off",
}
#: knobs read from the running server's environment (no secret-named key ever)
KNOB_ENV = re.compile(r"^(QSE_[A-Z0-9_]+|QWEN38_[A-Z0-9_]+|CONDUCTOR|RESPONSE_CACHE|SESSION_SPILL|MAX_QUEUE|"
                      r"MAX_LEN|PYTORCH_CUDA_ALLOC_CONF|MEM_[A-Z_]+|CUDA_VISIBLE_DEVICES)$")
SECRET = re.compile(r"TOKEN|SECRET|PASS|KEY|AUTH|LOGIN|CRED")
FAKE_LAUNCH = ("import sys, runpy; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
               "sys.argv = ['server/app.py'] + sys.argv[1:]; runpy.run_path('server/app.py', "
               "run_name='__main__')")


def git(repo: str, *args) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def make_tree(repo: str, sha: str, dest: Path, patch: str | None = None) -> str:
    """`git archive` of `sha` into `dest` (and `patch` applied: a dry run's planted change); the
    tree's code hash (tools/row3.py `code_hash`: engine/, server/, tools/, ops/)."""
    dest.mkdir(parents=True, exist_ok=False)
    arch = subprocess.Popen(["git", "-C", repo, "archive", sha], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=arch.stdout, check=True)
    if arch.wait() != 0:
        raise RuntimeError(f"git archive {sha} failed")
    if patch:
        subprocess.run(["git", "apply", "--unsafe-paths", os.path.abspath(patch)], cwd=dest, check=True)
    return code_hash(dest)


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket() as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def free_port(start: int, taken: set, floor: int = MIN_PORT) -> int:
    if start < floor:
        raise SystemExit(f"[gate] port {start} is under {floor}: the gate starts its servers on "
                         f"{floor} or above and never beside the box's own")
    for p in range(start, start + 50):
        if p not in taken and not port_busy(p):
            return p
    raise SystemExit(f"[gate] no free port in {start}-{start + 49}")


def env_lines(env: dict) -> str:
    """KEY='value' lines; a value with ${...} is double-quoted so the profile's variables expand."""
    out = []
    for k, v in env.items():
        v = str(v)
        out.append(f'{k}="{v}"' if "${" in v else f"{k}='{v}'")
    return "\n".join(out) + "\n"


class PidProc:
    """A started server known by its pid only (ops/start.sh detaches it): what MemGuard needs."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            return 1


def environ_of(pid: int) -> dict:
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        return {}
    out = {}
    for e in raw:
        k, _, v = e.decode(errors="replace").partition("=")
        if KNOB_ENV.match(k) and not SECRET.search(k):
            out[k] = v
    return out


def argv_of(pid: int) -> list[str]:
    try:
        return [x.decode(errors="replace") for x in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if x]
    except OSError:
        return []


def flags(argv: list[str]) -> dict:
    out = {}
    for i, x in enumerate(argv):
        if x.startswith("--"):
            k, eq, v = x.partition("=")
            if eq:
                out[k] = v
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[k] = argv[i + 1]
            else:
                out[k] = True
    for k in ("--model", "--corpus", "--usage-ledger", "--conductor-config", "--backends"):
        if isinstance(out.get(k), str):
            out[k] = os.path.basename(out[k].rstrip("/"))         # no paths in a report
    return out


def knobs(health: dict, env: dict, argv: list[str]) -> dict:
    """Every limit knob of a running server, defaults included: from /health (what it runs) and its
    process (what it was given)."""
    h = pc.health_record(health)
    m = health.get("memory") or {}
    c = health.get("cache") or {}
    st = c.get("state_store") or {}
    f = flags(argv)
    return {"version": h["version"], "weights": h["weights"], "lanes": h["lanes"],
            "lanes_requested": h["lanes_requested"], "max_running": h["max_running"],
            "max_running_reason": h["max_running_reason"], "max_queue": h["max_queue"],
            "max_len": h["max_len"], "windows": h["windows"],
            "prefill_chunk": env.get("QWEN38_FN_PREFILL_CHUNK", "default"),
            "prefix_cache": h["prefix_cache"], "prefix_chunk": h["prefix_chunk"],
            "cache_gb": round(st["budget"] / 2**30, 2) if st.get("budget") else None,
            "ceiling_gb": m.get("ceiling_gb"), "peak_gb": m.get("peak_gb"),
            "suffix_store": h["suffix_store"], "response_cache": h["response_cache"],
            "drafter": f.get("--drafter"), "corrector_loaded": (h.get("corrector") or {}).get("loaded")
            if isinstance(h.get("corrector"), dict) else None,
            "self_check": ((health.get("weights") or {}).get("self_check") or {}).get("ok"),
            "admission_floor_gib": ((health.get("parallel") or {}).get("admission") or {}).get("floor_gib"),
            "env": dict(sorted(env.items())), "flags": f}


def weights_sha256(path: str | None) -> str | None:
    """A fingerprint of a weights directory without reading the tensors: every file's name and size,
    and the bytes of its small JSON files (config, index, tokenizer config)."""
    if not path or not os.path.isdir(path):
        return None
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, path)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            h.update(f"{rel}\0{size}\0".encode())
            if fn.endswith(".json") and size <= 4 << 20:
                h.update(open(p, "rb").read())
    return h.hexdigest()


def wait_health(base: str, alive, deadline_s: float) -> dict | None:
    t = time.monotonic()
    while time.monotonic() - t < deadline_s:
        h = pc.health(base)
        if h.get("status") == "ok":
            return h
        if not alive():
            return None
        time.sleep(1.0)
    return None


class Arm:
    """One server of one side and variant: its tree, port, state directory, log and process."""

    def __init__(self, side: str, variant: str, tree: Path, port: int, home: Path, env: dict,
                 log=print):
        self.side, self.variant, self.tree, self.port = side, variant, tree, port
        self.home, self.env, self.log = home, env, log
        self.base = f"http://127.0.0.1:{port}"
        self.pid: int | None = None
        self.proc: subprocess.Popen | None = None
        self.guard: MemGuard | None = None
        self.started_s: float | None = None

    @property
    def key(self) -> str:
        return self.side if self.variant == "main" else f"{self.side}:{self.variant}"


class FakeLauncher:
    """server/app.py --fake-engine in the arm's tree."""

    def __init__(self, python: str, profile, served: str = "TandemLLM", tps: float = 400.0):
        self.python, self.profile, self.served, self.tps = python, profile, served, tps

    def start(self, arm: Arm) -> dict:
        arm.home.mkdir(parents=True, exist_ok=True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(arm.home),
               "PYTHONPATH": str(arm.tree), "CUDA_VISIBLE_DEVICES": "", "TZ": "Europe/Berlin",
               "QSE_ADMIN_TOKEN": "gate-" + "0" * 40, **{k: str(v) for k, v in arm.env.items()}}
        argv = ["--fake-engine", "--port", str(arm.port), "--served-model", self.served,
                "--usage-ledger", str(arm.home / "ledger.sqlite3"), "--fake-tps", f"{self.tps:g}",
                "--parallel", str(self.profile.lanes()), "--host", "127.0.0.1"]
        logf = open(arm.home / "server.log", "w")
        t0 = time.monotonic()
        arm.proc = subprocess.Popen([self.python, "-c", FAKE_LAUNCH, *argv], cwd=arm.tree, env=env,
                                    stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        arm.pid = arm.proc.pid
        h = wait_health(arm.base, lambda: arm.proc.poll() is None, 120)
        if h is None:
            self.stop(arm)
            raise RuntimeError(f"the {arm.key} fake server did not answer /health: "
                               + (arm.home / "server.log").read_text()[-600:])
        arm.started_s = round(time.monotonic() - t0, 1)
        return knobs(h, {k: v for k, v in env.items() if KNOB_ENV.match(k) and not SECRET.search(k)},
                     ["app.py", *argv])

    def stop(self, arm: Arm) -> None:
        if arm.proc is None or arm.proc.poll() is not None:
            return
        os.killpg(arm.proc.pid, signal.SIGTERM)
        try:
            arm.proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(arm.proc.pid, signal.SIGKILL)
            arm.proc.wait(10)


class OpsLauncher:
    """The tree's own ops/start.sh and ops/stop.sh, with the arm's setup.env."""

    def __init__(self, profile, pack: dict, site: dict, start_deadline_s: float = 900.0):
        self.profile, self.pack, self.site = profile, pack, site
        self.deadline = start_deadline_s

    def prepare(self, tree: Path) -> None:
        """What a box adds beside a checkout (as a box's server tree): its serve.env with REPO set to
        the tree, the profile, run/ and logs/, run/local.env, and the site's links and copies."""
        src = self.site.get("serve_env")
        serve = tree / "ops" / "serve.env"
        text = Path(src).read_text() if src else serve.read_text()
        text, n = re.subn(r"(?m)^REPO=.*$", f"REPO={tree}", text)
        if n != 1:
            raise RuntimeError(f"{src or serve}: expected one REPO= line, found {n}")
        serve.write_text(text)
        (tree / "ops" / "active-profile").write_text(self.pack["profile"] + "\n")
        (tree / "run").mkdir(mode=0o700, exist_ok=True)
        (tree / "logs").mkdir(exist_ok=True)
        if self.site.get("local_env"):
            shutil.copy(self.site["local_env"], tree / "run" / "local.env")
        for dst, srcp in (self.site.get("links") or {}).items():
            d = tree / dst
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                os.symlink(os.path.realpath(os.path.expanduser(srcp)), d)
        for dst, srcp in (self.site.get("copies") or {}).items():
            d, srcp = tree / dst, os.path.expanduser(srcp)
            d.parent.mkdir(parents=True, exist_ok=True)
            if os.path.isdir(srcp):
                shutil.copytree(srcp, d, dirs_exist_ok=True)
            else:
                shutil.copy2(srcp, d)

    def setup_env(self, arm: Arm) -> str:
        """The arm's run/setup.env: the box's own (site `setup_env`), then the arm's overrides."""
        base = Path(self.site["setup_env"]).read_text() if self.site.get("setup_env") else ""
        own = {"PORT": str(arm.port), **ARM_OVERRIDES,
               "QSE_CONDUCTOR_CONFIG": str(arm.home / "conductor"),
               "QSE_BACKENDS": str(arm.home / "backends.sqlite3"),
               "QWEN38_PROBE_DIR": str(arm.home / "probe"),
               "QSE_SECRETS": str(arm.home / "secrets.env"),
               **arm.env}
        return base.rstrip("\n") + "\n# tools/gate: this arm's own settings\n" + env_lines(own)

    def arm_state(self, arm: Arm) -> None:
        """The arm's own state: a conductor config (the site's copy, never its decisions or tokens),
        a probe folder and an empty secrets file."""
        src = self.site.get("conductor_from")
        if src and not (arm.home / "conductor").exists():
            shutil.copytree(os.path.expanduser(src), arm.home / "conductor",
                            ignore=shutil.ignore_patterns("decisions*", "*.token", "session-tokens.json"))
        for d in ("conductor", "probe"):
            (arm.home / d).mkdir(parents=True, mode=0o700, exist_ok=True)
        (arm.home / "secrets.env").touch(mode=0o600)

    def start(self, arm: Arm) -> dict:
        self.arm_state(arm)
        (arm.tree / "run" / "setup.env").write_text(self.setup_env(arm))
        env = {k: v for k, v in os.environ.items()
               if not (k.startswith(("QSE_", "QWEN38_")) or k in ("REPO", "PORT"))}
        env.update(HOLD_RESTART="1", QSE_STATUS_KEEP="1", START_REASON="a gate arm")
        t0 = time.monotonic()
        with open(arm.home / "start.log", "w") as fh:
            r = subprocess.run(["bash", "ops/start.sh"], cwd=arm.tree, env=env, stdout=fh,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=self.deadline)
        pidf = arm.tree / "logs" / "engine.pid"
        arm.pid = int(pidf.read_text().strip()) if pidf.exists() else None
        h = pc.health(arm.base)
        if r.returncode != 0 or h.get("status") != "ok":
            self.stop(arm)
            raise RuntimeError(f"the {arm.key} server did not start (exit {r.returncode}): "
                               + (arm.home / "start.log").read_text()[-800:])
        arm.started_s = round(time.monotonic() - t0, 1)
        floor = self.profile.mem_floor_gib()
        if floor and arm.pid:
            arm.guard = MemGuard(PidProc(arm.pid), floor_gb=floor, period=0.5).start()
        return knobs(h, environ_of(arm.pid) if arm.pid else {}, argv_of(arm.pid) if arm.pid else [])

    def stop(self, arm: Arm) -> None:
        if arm.guard is not None:
            arm.guard.stop()
        subprocess.run(["bash", "ops/stop.sh", "120"], cwd=arm.tree, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=300)
        log = arm.tree / "logs" / "engine.log"
        if log.exists():
            shutil.copy(os.path.realpath(log), arm.home / "engine.log")
        (arm.tree / "run" / "setup.env").unlink(missing_ok=True)
        if port_busy(arm.port):
            raise RuntimeError(f":{arm.port} still answers after the {arm.key} server's stop")


def launcher(pack: dict, profile, python: str, site: dict):
    if pack["launcher"] == "fake":
        return FakeLauncher(python, profile)
    if pack["launcher"] == "ops":
        return OpsLauncher(profile, pack, site)
    raise SystemExit(f"[gate] unknown launcher {pack['launcher']!r}")


def load_tokenizer(health: dict):
    """The served weights' tokenizer (transformers), or None (the fake engine, or no transformers)."""
    path = (health.get("weights") or {}).get("weights")
    if not path or not os.path.isdir(path):
        return None
    try:
        return pc.load_tokenizer(path)
    except Exception as exc:                                         # noqa: BLE001
        print(f"[gate] no tokenizer from the weights ({type(exc).__name__}): prompts as plain text",
              file=sys.stderr)
        return None


def dumps(x) -> str:
    return json.dumps(x, indent=1, sort_keys=True, default=str)
