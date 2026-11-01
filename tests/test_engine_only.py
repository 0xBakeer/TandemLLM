"""Engine only by default, the chat (beta) as an opt-in (OPS-260, QA OPS-261).

One setting, QSE_MODE (engine | chat), applied by the last block of ops/serve.env after the weight
profile, run/setup.env and an install's run/local.env. engine: no conductor config directory,
`--conductor raw`, no decision models, narrator, shadow reader or roles, no chatd. chat: the conductor
with its directory. Empty: chat when the conductor's directory exists (a box that ran it keeps its
mode on an upgrade), else engine.

The test matrix, the Gherkin of OPS-260:

  1  the template on a new box (no conductor directory) resolves engine and turns everything beside
     the engine off, the Flash-Next profile's deciders included; an unknown value is engine, said so
                                                              test_a_new_box_resolves_engine
  2  the upgrade rule on copied setups: a live-like run/setup.env (CONDUCTOR on, the conductor's
     directory, Laya) with the directory on disk stays chat with its values; QSE_CONDUCTOR_CONFIG=off
     set by hand stays engine; QSE_MODE set explicitly wins; chat with no directory gets the default;
     a quoted ~/ path counts as the server reads it
                                                              test_the_upgrade_keeps_the_mode
  3  ops/start.sh per mode, against a fake engine: engine passes --conductor raw and
     --conductor-config off, the engine's environment has no decision models, no autoload or
     narrator call, no role and no chatd; chat passes the conductor, asks the autoload, starts chatd;
     the switch to engine stops a running chatd of this tree (not for a hold's restart), and
     ops/chatd.sh stop --this-tree leaves another tree's chatd running (Linux only)
                                                              test_start_flags_per_mode
                                                              test_chatd_stop_this_tree_only
  4  /health and /v1/models of a fake-engine server report the mode and what is loaded, chatd up or
     down by a connect to QSE_CHAT_URL                         test_health_reports_the_mode
     a QSE_CHAT_URL whose port is no port is chatd down, a report that raises is an error in
     `loaded`, and /v1/models builds its copy at most once a TTL
                                                              test_the_mode_report_reads_what_is_loaded
  5  install.sh: a new install writes engine, --with-chat chat, a re-run keeps the mode, a config
     from before the key keeps what the box ran, --engine-only turns the chat off, and run/local.env
     is read before the mode block (Linux only)                test_install_mode
  6  the Setup Guide's switch: every model's Chat (beta) component writes QSE_MODE, is off by
     default with its dependents; switched on it brings Laya (and the 27B's shadow reader) as before
     this change, and Flash-Next's and Ornith's decision models are the same as --with-chat's
                                                              test_the_setup_guide_switch
  7  the live box through the guide: its seed (current_selection) on the live overlay reads Chat
     (beta) and Laya on, nothing unmatched, and a re-apply from that seed keeps QSE_MODE=chat with
     the same decision models, so the guide never turns live's chat off by itself
                                                              test_the_live_seed_keeps_the_chat

The dashboard's off state (no chat request on an engine-only box) is dashboard/tests/chat-backend.test.ts.

Run: python tests/test_engine_only.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

TEMPLATE = os.path.join(ROOT, "ops", "serve.env")
MARK = "# OPS-260: the mode (QSE_MODE above), applied last"
KEYS = ("QSE_MODE", "CONDUCTOR", "QSE_CONDUCTOR_CONFIG", "QWEN38_DECIDER_MODELS", "QWEN38_NARRATOR_MODEL",
        "QWEN38_SHADOW_MODEL", "ROLES_AUTOLOAD", "QSE_CHATD")
LAYA = "laya-english,laya-multilingual,laya-typed-decisions"
# the live box's run/setup.env, its conductor keys as the Setup Guide wrote them (values quoted)
LIVE_OVERLAY = ("CONDUCTOR='on'\nQSE_CONDUCTOR_CONFIG=\"${HOME}/.qwen38-spark-engine/conductor\"\n"
                f"QWEN38_DECIDER_MODELS='{LAYA}'\nQSE_BACKENDS='off'\n")


def _box(profile: str | None = "flashnext-3.05") -> tuple[str, str, dict]:
    """The release's template and profiles in a repo of their own, in a HOME of its own."""
    d = tempfile.mkdtemp(prefix="engonly-")
    repo = os.path.join(d, "repo")
    shutil.copytree(os.path.join(ROOT, "ops", "profiles"), os.path.join(repo, "ops", "profiles"))
    env_file = os.path.join(repo, "ops", "serve.env")
    with open(env_file, "w") as fh:
        fh.write(open(TEMPLATE).read().replace("REPO=${HOME}/qwen38-spark-engine", f"REPO={repo}"))
    if profile:
        with open(os.path.join(repo, "ops", "active-profile"), "w") as fh:
            fh.write(profile + "\n")
    home = os.path.join(d, "home")
    os.makedirs(home)
    return repo, env_file, {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home}


def _resolve(env_file: str, env: dict, overlay: str | None = None) -> tuple[dict, str]:
    """KEYS as the template resolves them under set -euo pipefail (bin/tandem's way), and its stderr."""
    e = dict(env)
    if overlay is not None:
        p = os.path.join(os.path.dirname(env_file), "setup.env")
        with open(p, "w") as fh:
            fh.write(overlay)
        e["QSE_SETUP_ENV"] = p
    script = 'set -euo pipefail; set -a; . "$1"; set +a; ' + "; ".join(f'echo "{k}=${{{k}-<unset>}}"' for k in KEYS)
    r = subprocess.run(["bash", "-c", script, "-", env_file], capture_output=True, text=True, env=e, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(line.split("=", 1) for line in r.stdout.splitlines()), r.stderr


ENGINE = {"QSE_MODE": "engine", "CONDUCTOR": "raw", "QSE_CONDUCTOR_CONFIG": "off", "QWEN38_DECIDER_MODELS": "",
          "QWEN38_NARRATOR_MODEL": "", "QWEN38_SHADOW_MODEL": "", "ROLES_AUTOLOAD": "", "QSE_CHATD": "off"}


def test_a_new_box_resolves_engine():
    for profile in ("flashnext-3.05", "nvfp4", None):
        _, env_file, env = _box(profile)
        got, err = _resolve(env_file, env)
        assert got == ENGINE, (profile, got)
        assert not err.strip(), err
    # the template itself: the key is empty, the chat's values stay the chat's, the block is last
    text = open(TEMPLATE).read()
    assert "\nQSE_MODE=\n" in text and "\nCONDUCTOR=on\n" in text
    assert text.index(MARK) > text.index('. "${QSE_SETUP_ENV_FILE}"')
    # a typo is engine, and the template says so
    _, env_file, env = _box()
    got, err = _resolve(env_file, env, "QSE_MODE='Chat'\n")
    assert got == ENGINE and "QSE_MODE=Chat is neither engine nor chat: engine" in err, (got, err)
    return "engine, Flash-Next's Laya included; a typo is engine"


def test_the_upgrade_keeps_the_mode():
    _, env_file, env = _box()
    cdir = os.path.join(env["HOME"], ".qwen38-spark-engine", "conductor")
    # a box that never ran the conductor, with the live overlay copied over: engine (no directory)
    got, _ = _resolve(env_file, env, LIVE_OVERLAY)
    assert got["QSE_MODE"] == "engine", got
    os.makedirs(cdir)
    # the live box: the overlay and the directory -> chat, every value as it was
    got, _ = _resolve(env_file, env, LIVE_OVERLAY)
    assert got == {"QSE_MODE": "chat", "CONDUCTOR": "on", "QSE_CONDUCTOR_CONFIG": cdir,
                   "QWEN38_DECIDER_MODELS": LAYA, "QWEN38_NARRATOR_MODEL": "", "QWEN38_SHADOW_MODEL": "",
                   "ROLES_AUTOLOAD": "", "QSE_CHATD": "<unset>"}, got
    # a box on the template's own values (no overlay) with the directory: chat, the profile's Laya
    got, _ = _resolve(env_file, env)
    assert (got["QSE_MODE"], got["CONDUCTOR"], got["QSE_CONDUCTOR_CONFIG"], got["QWEN38_DECIDER_MODELS"]) == \
        ("chat", "on", cdir, LAYA), got
    # a quoted ~ path (local.env) is the directory the server opens (expanduser): chat
    got, _ = _resolve(env_file, env, "QSE_CONDUCTOR_CONFIG='~/.qwen38-spark-engine/conductor'\n")
    assert got["QSE_MODE"] == "chat", got
    # the conductor turned off by hand (the Setup Guide's off, or an operator) stays engine
    got, _ = _resolve(env_file, env, "CONDUCTOR='raw'\nQSE_CONDUCTOR_CONFIG='off'\n")
    assert got == ENGINE, got
    # an explicit setting wins over the directory, both ways
    got, _ = _resolve(env_file, env, LIVE_OVERLAY + "QSE_MODE='engine'\n")
    assert got == ENGINE, got
    got, _ = _resolve(env_file, env, "QSE_MODE='chat'\nQSE_CONDUCTOR_CONFIG='off'\n")
    assert (got["QSE_MODE"], got["QSE_CONDUCTOR_CONFIG"]) == ("chat", cdir), got
    return "the live overlay stays chat; config off stays engine; explicit wins"


# --------------------------------------------------------------------------- start.sh per mode

FAKE_CHATD = """#!/bin/bash
echo "chatd.sh $*" >> "$MARKERS"
case "$1" in start) touch "$CHAT_STATE";; stop) rm -f "$CHAT_STATE";; status) [ -f "$CHAT_STATE" ];; esac
"""


def _start(mode: str, chatd_running: bool = False, **extra) -> tuple[subprocess.CompletedProcess, list[str], str]:
    import test_ops_scripts as T
    repo = T._box(["start.sh"])
    if chatd_running:
        open(os.path.join(repo, "chat-up"), "w").close()
    cdir = os.path.join(repo, "conductor")
    os.makedirs(cdir)
    engine = os.path.join(repo, "fake-engine.sh")
    with open(engine, "w") as fh:
        fh.write(f"#!/bin/bash\necho \"engine $*\" >> {repo}/markers\n"
                 f"echo \"env deciders=${{QWEN38_DECIDER_MODELS-unset}} narrator=${{QWEN38_NARRATOR_MODEL-unset}}"
                 f" mode=${{QSE_MODE-unset}}\" >> {repo}/markers\ntouch {repo}/up\nsleep 30\n")
    os.chmod(engine, 0o755)
    for name, body in (("roles.sh", f"#!/bin/bash\necho \"roles ${{ROLES_AUTOLOAD:-none}}\" >> {repo}/markers\n"),
                       ("chatd.sh", FAKE_CHATD)):
        with open(os.path.join(repo, "ops", name), "w") as fh:
            fh.write(body)
        os.chmod(os.path.join(repo, "ops", name), 0o755)
    with open(os.path.join(repo, "bin", "curl"), "w") as fh:      # every call recorded; healthy once up
        fh.write(f"#!/bin/bash\necho \"curl ${{@: -1}}\" >> {repo}/markers\n[ -f {repo}/up ] || exit 22\n"
                 "case \" $* \" in *\" -w \"*) printf 200;; esac\nexit 0\n")
    os.chmod(os.path.join(repo, "bin", "curl"), 0o755)
    if not shutil.which("setsid"):                 # macOS has none; start.sh launches with it
        with open(os.path.join(repo, "bin", "setsid"), "w") as fh:
            fh.write('#!/bin/bash\nexec "$@"\n')
        os.chmod(os.path.join(repo, "bin", "setsid"), 0o755)
    text = open(TEMPLATE).read()
    block = text[text.index(MARK):]
    env_file = os.path.join(repo, "ops", "serve.env")
    with open(env_file) as fh:
        body = fh.read()
    with open(env_file, "w") as fh:
        fh.write(body.replace("PY=/bin/false", f"PY={engine}").replace("PORT=8000", "PORT=18995"))
        fh.write("SERVED_MODEL=t\nMAX_LEN=64\nDEFAULT_MAX_TOKENS=8\nREASONING_FORMAT=tags\n"
                 "REASONING_EFFORT=medium\nLEN_FIXED=0\nLEN_LATCH=1\nBUDGET=16\nCORPUS=x\n"
                 "CKPT8=x\nCKPT16=x\nNV=x\nHEAD=x\nCACHE_GB=0\nREQUEST_TIMEOUT=1\n"
                 "MAX_QUEUE=1\nQUEUE_TIMEOUT=1\nFUSE_PROJ=0\nQWEN38_DF2_TREE_MODE=paths\n"
                 f"QWEN38_TREE_ALIAS_STATE=0\nDROP_PAGE_CACHE=0\nCONDUCTOR=on\nQSE_CONDUCTOR_CONFIG={cdir}\n"
                 f"QWEN38_DECIDER_MODELS={LAYA}\nQWEN38_NARRATOR_MODEL=qwen3.5-2b\nROLES_AUTOLOAD=decider\n"
                 f"QSE_MODE={mode}\n{block}")
    try:
        r = subprocess.run(["bash", os.path.join(repo, "ops", "start.sh")], capture_output=True, text=True,
                           env=T._env(repo, CHAT_STATE=os.path.join(repo, "chat-up"),
                                      QSE_SECRETS=os.path.join(repo, "no-secrets.env"), **extra), timeout=120)
    finally:
        subprocess.run(["pkill", "-f", engine])
    marks = open(os.path.join(repo, "markers")).read().splitlines()
    return r, marks, cdir


def test_start_flags_per_mode():
    r, marks, _ = _start("engine")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = next(m for m in marks if m.startswith("engine ")).split()
    assert argv[argv.index("--conductor") + 1] == "raw" and argv[argv.index("--conductor-config") + 1] == "off", argv
    assert "env deciders= narrator= mode=engine" in marks, marks
    assert not any("deciders/autoload" in m or "narrator/load" in m for m in marks), marks
    assert "roles none" in marks and not any(m.startswith("chatd.sh start") for m in marks), marks
    assert "[start] mode: engine" in r.stdout and "[start] chatd: off (engine only" in r.stdout, r.stdout
    r, marks, cdir = _start("chat")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = next(m for m in marks if m.startswith("engine ")).split()
    assert argv[argv.index("--conductor") + 1] == "on" and argv[argv.index("--conductor-config") + 1] == cdir, argv
    assert f"env deciders={LAYA} narrator=qwen3.5-2b mode=chat" in marks, marks
    assert any("deciders/autoload" in m for m in marks) and any("narrator/load" in m for m in marks), marks
    assert "roles decider" in marks and "chatd.sh start" in marks, marks
    assert "[start] mode: chat" in r.stdout, r.stdout
    # the switch to engine (the guide's off, local.env and a restart): the chat's chatd stops too,
    # this tree's only; a hold's restart leaves it
    r, marks, _ = _start("engine", chatd_running=True)
    assert r.returncode == 0 and "chatd.sh stop --this-tree" in marks, marks
    assert not any(m.startswith("chatd.sh start") for m in marks), marks
    r, marks, _ = _start("engine", chatd_running=True, HOLD_RESTART="1")
    assert not any(m.startswith("chatd.sh stop") for m in marks), marks
    r, marks, _ = _start("chat", chatd_running=True)
    assert not any(m.startswith(("chatd.sh stop", "chatd.sh start")) for m in marks), marks
    return "engine: raw, off, nothing beside it, a running chatd stopped; chat: the conductor, the autoload, chatd"


def test_chatd_stop_this_tree_only():
    if not os.path.isdir("/proc/self"):
        print("    (skipped: the tree check reads /proc -- run it on the box)")
        return "skipped off Linux"
    d = tempfile.mkdtemp(prefix="engonly-chatd-")
    data, other = os.path.join(d, "data"), os.path.join(d, "other")
    os.makedirs(data)
    os.makedirs(other)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": d, "REPO": ROOT, "QSE_CHAT_DATA": data,
           "QSE_CHATD_ENV": os.path.join(d, "none.env"), "QSE_CHAT_URL": "http://127.0.0.1:9"}
    procs = []
    try:
        for cwd, kept in ((other, True), (ROOT, False)):
            p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "chatd"], cwd=cwd)
            procs.append(p)
            with open(os.path.join(data, "chatd.pid"), "w") as fh:
                fh.write(f"{p.pid}\n")
            r = subprocess.run(["bash", os.path.join(ROOT, "ops", "chatd.sh"), "stop", "--this-tree"], env=env,
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, r.stdout + r.stderr
            if kept:
                assert p.poll() is None and "another tree's chatd" in r.stdout, r.stdout
            else:
                assert p.wait(timeout=30) is not None and "[chatd] stopped" in r.stdout, r.stdout
    finally:
        for p in procs:
            p.kill()
        shutil.rmtree(d, ignore_errors=True)
    return "another tree's chatd stays; this tree's stops"


# --------------------------------------------------------------------------- /health and /v1/models

def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def _listener() -> tuple[socket.socket, int]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)

    def accept():
        while True:
            try:
                c, _ = s.accept()
                c.close()
            except OSError:
                return
    threading.Thread(target=accept, daemon=True).start()
    return s, s.getsockname()[1]


def test_health_reports_the_mode():
    from test_conductor_e2e import LAUNCH, _port, _wait
    home = tempfile.mkdtemp(prefix="engonly-health-")
    fake_chatd, chat_port = _listener()
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_port = closed.getsockname()[1]
    closed.close()
    try:
        for mode, chat in (("engine", dead_port), ("chat", chat_port)):
            port = _port()
            args = [sys.executable, "-c", LAUNCH, "--fake-engine", "--port", str(port), "--served-model", "t-engine"]
            if mode == "chat":
                args += ["--conductor", "on", "--conductor-config", os.path.join(home, "conductor"),
                         "--conductor-defaults", "shipped"]
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "PYTHONPATH": ROOT,
                   "CUDA_VISIBLE_DEVICES": "", "QSE_CHAT_URL": f"http://127.0.0.1:{chat}"}
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                _wait(port, proc)
                h = _get(port, "/health")
                m = _get(port, "/v1/models")
            finally:
                proc.terminate()
                proc.wait(timeout=30)
            assert h["mode"] == mode and m["mode"] == mode, (h.get("mode"), m.get("mode"))
            loaded = h["loaded"]
            assert loaded == m["loaded"], (loaded, m["loaded"])
            assert loaded["decision_models"] == [] and loaded["narrator"] is None and loaded["roles"] == [], loaded
            if mode == "engine":
                assert loaded["conductor"] is False and loaded["monitor"] == "raw" and loaded["chatd"] == "down", loaded
                assert [d["id"] for d in m["data"]] == ["t-engine"], m["data"]
            else:
                assert loaded["conductor"] is True and loaded["monitor"] == "on" and loaded["chatd"] == "up", loaded
                assert [d["id"] for d in m["data"]] == ["TandemLLM"], m["data"]
    finally:
        fake_chatd.close()
        shutil.rmtree(home, ignore_errors=True)
    return "engine: no conductor, chatd down; chat: the conductor on, chatd up"


def test_the_mode_report_reads_what_is_loaded():
    from server import mode as M

    class Slot:
        def __init__(self, state):
            self.state, self.model = state, type("Mo", (), {"name": "qwen3.5-2b"})()

    class Reg:
        def loaded(self):
            return ["laya-english"]
    state = {"conductor": object(), "monitor": type("Mon", (), {"mode": "shadow"})(), "deciders": Reg(),
             "narrator": type("N", (), {"slot": Slot("loaded")})()}
    roles = {"roles": {"laya": {"state": "loaded"}, "narrator": {"state": "loaded"}, "shadow": {"state": "not_loaded"}}}
    assert M.report(state, roles, chatd=lambda: False) == {
        "mode": "chat", "loaded": {"conductor": True, "monitor": "shadow", "decision_models": ["laya-english"],
                                   "narrator": "qwen3.5-2b", "roles": ["laya", "narrator"], "chatd": "down"}}
    state["narrator"].slot.state = "loading"
    assert M.report(state, None, chatd=lambda: True)["loaded"]["narrator"] is None
    assert M.report({}, None, chatd=lambda: False) == {
        "mode": "engine", "loaded": {"conductor": False, "monitor": "raw", "decision_models": [], "narrator": None,
                                     "roles": [], "chatd": "down"}}
    # the chatd connect: at most once a TTL
    calls = []
    t = [100.0]
    M._chatd.update(at=None, up=False)
    up = lambda addr: calls.append(addr) or True   # noqa: E731
    assert M.chatd_up(clock=lambda: t[0], connect=up) and M.chatd_up(clock=lambda: t[0] + 1, connect=up)
    assert len(calls) == 1
    t[0] += M.CHATD_TTL_S + 1
    M.chatd_up(clock=lambda: t[0], connect=up)
    assert len(calls) == 2
    M._chatd.update(at=None, up=False)
    # a QSE_CHAT_URL whose port is no port is chatd down, never an exception in /health
    old = os.environ.get("QSE_CHAT_URL")
    os.environ["QSE_CHAT_URL"] = "http://127.0.0.1:x"
    try:
        assert M.chatd_up() is False
    finally:
        M._chatd.update(at=None, up=False)
        if old is None:
            os.environ.pop("QSE_CHAT_URL")
        else:
            os.environ["QSE_CHAT_URL"] = old
    # the server's copy: a report that raises is an error in `loaded`; /v1/models's at most once a TTL
    from server import app
    real = M.report
    try:
        M.report = lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
        got = app._mode_report({"roles": {}})
        assert got["mode"] in ("engine", "chat") and "ValueError: bad" in got["loaded"]["error"], got
    finally:
        M.report = real
    n = []
    real_r = app._mode_report
    try:
        app._mode_report = lambda roles=None: n.append(1) or {"mode": "engine", "loaded": {}}
        app._modes_last.update(at=None, body=None)
        t = [50.0]
        for dt in (0.0, 1.0, app.MODES_TTL_S + 0.5):
            app._models_mode(clock=lambda: t[0] + dt)
        assert len(n) == 2, n
    finally:
        app._mode_report = real_r
        app._modes_last.update(at=None, body=None)
    return "the conductor, its monitor, Laya, the narrator, the roles, chatd once a TTL; a bad URL is down"


# --------------------------------------------------------------------------- install.sh

def _mode_of(box: dict) -> str:
    serve = os.path.join(box["dir"], "run", "ops", "serve.env")
    r = subprocess.run(["bash", "-c", 'set -a; . "$1"; echo "$QSE_MODE"', "-", serve], env=box["env"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _line(box: dict) -> str:
    text = open(os.path.join(box["dir"], "run", "ops", "serve.env")).read()
    return next(ln for ln in text.splitlines() if ln.startswith("QSE_MODE="))


def _rerun(box: dict, *extra: str) -> None:
    args = [a for a in box["args"] if a not in ("--with-chat", "--engine-only")] + list(extra)
    r = subprocess.run(args, env=box["env"], capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_install_mode():
    import test_install_control as I
    box = I.install("engonly-new", True, I.free_port())
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return "skipped off Linux"
    serve = os.path.join(box["dir"], "run", "ops", "serve.env")
    text = open(serve).read()
    assert _line(box) == "QSE_MODE=engine" and _mode_of(box) == "engine", _line(box)
    local = os.path.join(box["dir"], "run", "local.env")
    assert text.index(f"if [ -f '{local}' ]") < text.index(MARK) and text.count(f". '{local}'") == 1
    # run/local.env counts: it is read before the mode block
    with open(local, "a") as fh:
        fh.write("QSE_MODE=chat\n")
    assert _mode_of(box) == "chat"
    with open(local, "w") as fh:
        fh.write("QSE_STATUS_PORT=\n")
    # --with-chat; a re-run keeps it; --engine-only turns it off
    _rerun(box, "--with-chat")
    assert _line(box) == "QSE_MODE=chat" and _mode_of(box) == "chat"
    _rerun(box)
    assert _line(box) == "QSE_MODE=chat"
    _rerun(box, "--engine-only")
    assert _line(box) == "QSE_MODE=engine" and _mode_of(box) == "engine"
    # a config from before the key: the re-run leaves the mode empty, and the box keeps what it ran
    with open(serve) as fh:
        old = "".join(ln for ln in fh if not ln.startswith("QSE_MODE="))
    with open(serve, "w") as fh:
        fh.write(old)
    _rerun(box)
    assert _line(box) == "QSE_MODE=" and _mode_of(box) == "engine"
    os.makedirs(os.path.join(box["home"], ".qwen38-spark-engine", "conductor"), exist_ok=True)
    assert _mode_of(box) == "chat"
    # a new install with --with-chat
    box2 = I.install("engonly-chat", True, I.free_port(), ("--with-chat",))
    assert _line(box2) == "QSE_MODE=chat" and _mode_of(box2) == "chat"
    return "new: engine; --with-chat: chat; re-run keeps; old config keeps what it ran; local.env counts"


# --------------------------------------------------------------------------- the Setup Guide

def test_the_setup_guide_switch():
    from server.setup import registry
    models, errors = registry.load_all()
    assert not errors, errors
    for mid, m in models.items():
        comps = {c["id"]: c for c in m["components"]}
        c = comps["conductor"]
        assert c["label"] == "Chat (beta)" and c["default"] is False, (mid, c["label"], c["default"])
        assert c["knobs"]["QSE_MODE"] == {"on": "chat", "off": "engine"}, (mid, c["knobs"])
        assert c["knobs"]["QSE_CONDUCTOR_CONFIG"]["off"] == "off" and c["knobs"]["CONDUCTOR"]["off"] == "raw"
        for d in m["components"]:
            if "conductor" in d["depends_on"]:
                assert d["default"] is False, (mid, d["id"])
        assert not registry.unregistered_keys(m), registry.unregistered_keys(m)
        # switched on, it brings what it shipped with before OPS-260: Laya everywhere (the 27B: the
        # shadow reader too); a selection that names Laya off keeps it off
        from server.setup import estimate
        sel = estimate.check(m, {"model": mid, "components": {"conductor": True}})
        assert sel["components"]["laya"] is True, mid
        if "shadow" in comps:
            assert sel["components"]["shadow"] is (mid == "qwen3.8-27b"), mid
        assert sel["components"]["narrator"] is False, mid
        assert estimate.check(m, {"model": mid, "components": {"conductor": True, "laya": False}})[
            "components"]["laya"] is False
    # with_parent is for an optional tool, off by default, under one parent
    import copy
    fn = copy.deepcopy(models["qwen3.8-flash-next"])
    laya = next(c for c in fn["components"] if c["id"] == "laya")
    laya["default"] = True
    assert any("with_parent" in e for e in registry._cross(fn, registry.ROOT)), "a default-on with_parent"
    # the guide's chat and --with-chat load the same decision models where the profile names them
    from server.setup import apply, estimate
    for mid, profile in (("qwen3.8-flash-next", "flashnext-3.05"), ("ornith-1.5-122b", None)):
        m = models[mid]
        profile = profile or registry.recommended_profile(m)["id"]
        sel = estimate.check(m, {"model": mid, "profile": profile, "components": {"conductor": True}})
        _, env_file, env = _box(profile)
        os.makedirs(os.path.join(env["HOME"], ".qwen38-spark-engine", "conductor"))
        guide, _ = _resolve(env_file, env, apply.render_overlay(m, sel))
        flag, _ = _resolve(env_file, env, "QSE_MODE='chat'\n")
        assert guide["QSE_MODE"] == flag["QSE_MODE"] == "chat", (mid, guide, flag)
        assert guide["QWEN38_DECIDER_MODELS"] == flag["QWEN38_DECIDER_MODELS"] == LAYA, (mid, guide, flag)
    return f"{len(models)} models: Chat (beta) off by default, QSE_MODE written, on with Laya"


def test_the_live_seed_keeps_the_chat():
    import test_setup_apply as SA
    from server.setup import apply, device as device_mod, estimate, registry
    fn = registry.get("qwen3.8-flash-next")
    with SA.Box() as box:
        with open(os.path.join(box.ops, "active-profile"), "w") as fh:
            fh.write("flashnext-3.05\n")
        device_mod.clear()
        with open(apply.overlay_path(), "w") as fh:
            fh.write(LIVE_OVERLAY)
        b = apply.current_selection(fn)
        comps = b["selection"]["components"]
        assert comps["conductor"] is True and comps["laya"] is True and comps["shadow"] is False, comps
        chat = ("components.conductor", "components.laya", "components.shadow", "components.narrator")
        assert not [u for u in b["unmatched"] if u["field"] in chat], b["unmatched"]
        sel = estimate.check(fn, b["selection"])
        vals = apply.overlay_values(fn, sel)
        assert (vals["QSE_MODE"], vals["CONDUCTOR"], vals["QWEN38_DECIDER_MODELS"]) == ("chat", "on", LAYA), vals
        apply.write_overlay(fn, sel)
        got = apply.resolve_env(["QSE_MODE", "QSE_CONDUCTOR_CONFIG", "QWEN38_DECIDER_MODELS"])
        assert got == {"QSE_MODE": "chat", "QWEN38_DECIDER_MODELS": LAYA,
                       "QSE_CONDUCTOR_CONFIG": os.path.join(box.home, ".qwen38-spark-engine", "conductor")}, got
    return "live's seed: Chat (beta) and Laya on; a re-apply keeps chat and Laya"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            out = fn()
            print(f"  {name:50s} ok  {out or ''}")
            passed += 1
    print(f"{passed} passed")
