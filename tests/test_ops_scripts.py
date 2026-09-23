"""The two supervisor scripts, run for real against fake `curl`/`pgrep` (OPS, the pre-merge review).

Both faults here are the same shape: a file that outlives the thing it describes. `logs/loading.since`
outlived the loader whose load it was timing, and `.watchdog.off` outlived the hold that armed it --
and in both cases the survivor condemns whatever comes next, which on this box means a restart in
the middle of a 29 GB load, or no service at all until a timeout that is measured in tens of minutes.

`ops/watchdog.sh` and `ops/start.sh` are copied into a temporary directory beside a `serve.env` that
points at it, so the real scripts run with nothing real behind them: `curl`, `pgrep` and (on macOS,
where `stat -c` does not exist) `stat` come from a fake bin on the PATH, and `stop.sh`/`start.sh`
leave a marker instead of touching a board.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

OPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops")

SERVE_ENV = """HOME_DIR={repo}
REPO={repo}
PY=/bin/false
PYTHONPATH={repo}
HOST=127.0.0.1
PORT=8000
WATCHDOG_FAILS=3
LOADING_MAX=900
WATCHDOG_PAUSE_MAX=2400
"""

FAKE_CURL = """#!/bin/bash
# `-w '%{http_code}'` asks for the code on stdout; `-sf` asks for an exit status.
case " $* " in
  *" -w "*) echo "${FAKE_HTTP_CODE:-000}";;
  *) [ "${FAKE_HEALTHY:-0}" = "1" ] || exit 22;;
esac
exit 0
"""

FAKE_PGREP = """#!/bin/bash
[ -n "${FAKE_PID:-}" ] || exit 1
echo "$FAKE_PID"
"""

FAKE_STAT = """#!/bin/bash
# GNU `stat -c %Y` on a box that only has the BSD one.
[ "$1" = "-c" ] && exec /usr/bin/stat -f %m "$3"
exec /usr/bin/stat "$@"
"""

MARKER = """#!/bin/bash
echo "$(basename "$0") $*" >> "$MARKERS"
"""


def _box(scripts: list[str]) -> str:
    """A temporary repo with the real scripts, a serve.env pointing at it, and fakes on the PATH."""
    repo = tempfile.mkdtemp(prefix="opsbox-")
    ops = os.path.join(repo, "ops")
    os.makedirs(os.path.join(repo, "logs"))
    os.makedirs(ops)
    for name in scripts:
        shutil.copy(os.path.join(OPS, name), os.path.join(ops, name))
    with open(os.path.join(ops, "serve.env"), "w") as fh:
        fh.write(SERVE_ENV.format(repo=repo))
    binary = os.path.join(repo, "bin")
    os.makedirs(binary)
    fakes = {"curl": FAKE_CURL, "pgrep": FAKE_PGREP, "stop.sh": MARKER, "start.sh": MARKER}
    if sys.platform == "darwin":
        fakes["stat"] = FAKE_STAT
    for name, body in fakes.items():
        path = os.path.join(ops if name.endswith(".sh") else binary, name)
        if os.path.exists(path):
            continue                       # the script under test, not a fake
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    return repo


def _run(repo: str, script: str, **env) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e["PATH"] = os.path.join(repo, "bin") + os.pathsep + e["PATH"]
    e["MARKERS"] = os.path.join(repo, "markers")
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", os.path.join(repo, "ops", script)],
                          capture_output=True, text=True, env=e)


def _since(repo: str) -> str:
    path = os.path.join(repo, "logs", "loading.since")
    return open(path).read().strip() if os.path.exists(path) else ""


def _wd_log(repo: str) -> str:
    path = os.path.join(repo, "logs", "watchdog.log")
    return open(path).read() if os.path.exists(path) else ""


def test_a_dead_loaders_timestamp_does_not_condemn_the_next_one():
    # The fault: `loading.since` was removed only on a 200. A loader that died before it ever
    # answered left its start time behind, the next one inherited an age past LOADING_MAX on its
    # very first check, and three strikes later the watchdog restarted a server mid-load.
    repo = _box(["watchdog.sh"])
    stale = int(time.time()) - 5000
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"111 {stale}\n")
    r = _run(repo, "watchdog.sh", FAKE_PID="222")
    assert r.returncode == 0, r.stderr
    assert "loading" in _wd_log(repo) and "treating as hung" not in _wd_log(repo)
    pid, since = _since(repo).split()
    assert pid == "222" and int(since) > stale, "the new loader times its own load"
    assert not os.path.exists(os.path.join(repo, "markers")), "nothing was restarted"


def test_the_same_loader_keeps_its_first_attempt():
    # The reset is on the pid, not on every check: a loader that has been silent for ten minutes
    # must still be counted from when IT started, or LOADING_MAX never elapses.
    repo = _box(["watchdog.sh"])
    started = int(time.time()) - 300
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"222 {started}\n")
    assert _run(repo, "watchdog.sh", FAKE_PID="222").returncode == 0
    assert _since(repo) == f"222 {started}"
    assert "loading (code 000, 3" in _wd_log(repo), _wd_log(repo)


def test_the_restart_path_clears_the_timestamp():
    repo = _box(["watchdog.sh"])
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"222 {int(time.time()) - 5000}\n")
    with open(os.path.join(repo, "logs", "watchdog.fails"), "w") as fh:
        fh.write("2\n")                                   # two strikes already
    r = _run(repo, "watchdog.sh", FAKE_PID="222")
    assert r.returncode == 0, r.stderr
    markers = open(os.path.join(repo, "markers")).read()
    assert "stop.sh" in markers and "start.sh" in markers
    assert _since(repo) == "", "the restarted engine must not inherit the hung one's clock"


def test_a_pause_file_that_outlived_its_hold_does_not_block_the_reboot_start():
    # `.watchdog.off` cannot survive a hold, but it survives a reboot, and the @reboot line then
    # refused to start the service for up to WATCHDOG_PAUSE_MAX with nobody holding anything.
    repo = _box(["start.sh"])
    pause = os.path.join(repo, ".watchdog.off")
    with open(pause, "w") as fh:
        fh.write("bench: row3\n")
    old = time.time() - 3000
    os.utime(pause, (old, old))
    r = _run(repo, "start.sh", FAKE_HEALTHY="1")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "older than the longest hold" in r.stdout, r.stdout
    assert not os.path.exists(pause), "the stale pause file is removed, not stepped around"


def test_a_live_hold_still_refuses_and_says_so_with_a_non_zero_status():
    repo = _box(["start.sh"])
    pause = os.path.join(repo, ".watchdog.off")
    with open(pause, "w") as fh:
        fh.write("bench: row3\n")
    r = _run(repo, "start.sh", FAKE_HEALTHY="1")
    assert r.returncode != 0, "a refusal cron cannot see is a refusal nobody notices"
    assert "a hold owns the board" in r.stdout
    assert os.path.exists(pause), "a live hold's file is left alone"


def test_a_hold_longer_than_the_pause_limit_keeps_the_watchdog_out():
    """OPS-10. The watchdog ignores a pause file older than WATCHDOG_PAUSE_MAX (2400 s), and on
    2026-09-18 a hand-made hold outlived it: :8000 came back beside a test server mid-gate.
    `ops/hold.sh`'s keeper refreshes the file for the life of the hold, so the limit never bites
    a live hold. The same scripts on a compressed clock: pause limit 3 s, refresh 1 s, a held
    command that runs 6 s and asks the watchdog -- with the service silent and one strike to
    restart -- at the end of it. The watchdog must stay out."""
    repo = _box(["hold.sh", "watchdog.sh"])
    env_file = os.path.join(repo, "ops", "serve.env")
    with open(env_file) as fh:
        body = fh.read()
    with open(env_file, "w") as fh:          # serve.env is sourced, so the knobs live there
        fh.write(body.replace("WATCHDOG_PAUSE_MAX=2400", "WATCHDOG_PAUSE_MAX=3")
                     .replace("WATCHDOG_FAILS=3", "WATCHDOG_FAILS=1"))
    probe = os.path.join(repo, "probe.sh")
    with open(probe, "w") as fh:
        fh.write(f"sleep 6\nbash {repo}/ops/watchdog.sh\n")
    r = subprocess.run(
        ["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--", "bash", probe],
        capture_output=True, text=True,
        env={**os.environ, "PATH": os.path.join(repo, "bin") + os.pathsep + os.environ["PATH"],
             "MARKERS": os.path.join(repo, "markers"), "HOLD_REFRESH": "1",
             "FAKE_HTTP_CODE": "000"})
    assert r.returncode == 0, r.stdout + r.stderr
    log = _wd_log(repo)
    assert "ignoring it" not in log and "restarting" not in log, log
    markers = open(os.path.join(repo, "markers")).read().split("\n")
    assert [m.split()[0] for m in markers if m] == ["stop.sh", "start.sh"], markers
    assert not os.path.exists(os.path.join(repo, ".watchdog.off")), "the hold removes its pause"


def test_a_pause_nobody_refreshes_still_ages_out():
    """OPS-10's other half: the limit is the safety for a pause whose hold is gone, and it stays."""
    repo = _box(["watchdog.sh"])
    pause = os.path.join(repo, ".watchdog.off")
    with open(pause, "w") as fh:
        fh.write("hold\n")
    old = time.time() - 3000
    os.utime(pause, (old, old))
    with open(os.path.join(repo, "logs", "watchdog.fails"), "w") as fh:
        fh.write("2\n")                                   # two strikes already
    r = _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000")
    assert r.returncode == 0, r.stderr
    log = _wd_log(repo)
    assert "ignoring it" in log and "restarting" in log, log

def test_the_engine_does_not_inherit_the_box_lock():
    """OPS-15. Holds run as `flock ~/.qwen38-box.flock bash ops/hold.sh ...`, and flock hands its
    lock descriptor to the command unless told `-o`. It was inherited all the way down -- hold.sh,
    start.sh, `setsid nohup` -- into the RESTARTED :8000 engine, which then held the box lock for
    its whole life, and the next agent's flock waited on a server that never exits. start.sh must
    launch the engine with nothing open but stdin/stdout/stderr, whatever its caller held."""
    if not os.path.isdir("/proc/self/fd") or shutil.which("flock") is None:
        print("    (skipped: needs /proc and flock -- run it on the box)")
        return
    repo = _box(["start.sh"])
    env_file = os.path.join(repo, "ops", "serve.env")
    engine = os.path.join(repo, "fake-engine.sh")
    with open(engine, "w") as fh:
        fh.write(f"#!/bin/bash\nls -l /proc/$$/fd > {repo}/engine-fds\ntouch {repo}/up\nsleep 30\n")
    os.chmod(engine, 0o755)
    with open(env_file) as fh:
        body = fh.read()
    with open(env_file, "w") as fh:
        fh.write(body.replace("PY=/bin/false", f"PY={engine}").replace("PORT=8000", "PORT=18999"))
        fh.write("SERVED_MODEL=t\nMAX_LEN=64\nDEFAULT_MAX_TOKENS=8\nREASONING_FORMAT=tags\n"
                 "REASONING_EFFORT=medium\nLEN_FIXED=0\nLEN_LATCH=1\nBUDGET=16\nCORPUS=x\n"
                 "CKPT8=x\nCKPT16=x\nNV=x\nHEAD=x\nCACHE_GB=0\nREQUEST_TIMEOUT=1\n"
                 "MAX_QUEUE=1\nQUEUE_TIMEOUT=1\nFUSE_PROJ=0\nQWEN38_DF2_TREE_MODE=paths\n"
                 "QWEN38_TREE_ALIAS_STATE=0\n")
    with open(os.path.join(repo, "bin", "curl"), "w") as fh:     # healthy once the engine is up
        fh.write(f"#!/bin/bash\n[ -f {repo}/up ] || exit 22\nexit 0\n")
    lock = os.path.join(repo, "box.flock")
    e = dict(os.environ, PATH=os.path.join(repo, "bin") + os.pathsep + os.environ["PATH"])
    r = subprocess.run(["flock", lock, "bash", os.path.join(repo, "ops", "start.sh")],
                       capture_output=True, text=True, env=e, timeout=120)
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        fds = open(os.path.join(repo, "engine-fds")).read()
        assert "box.flock" not in fds, f"the engine holds the box lock:\n{fds}"
        free = subprocess.run(["flock", "-n", lock, "true"], timeout=10)
        assert free.returncode == 0, "the lock must be free once start.sh has returned"
    finally:
        subprocess.run(["pkill", "-f", engine])

def test_a_killed_hold_stops_its_command_before_it_restarts_the_service():
    """OPS-16. A hold that is itself signalled -- an ssh session dropped, a tool timeout, a
    `kill` -- ran its EXIT trap and restarted :8000 while the command it was holding for kept
    running: a row3 with its own engine on :8001, beside the restarted service, which is the
    two-engine state that wedges this box (OPS-11). The hold must stop the command's whole process
    group first, and only then bring the service back."""
    if not os.path.isdir("/proc/self"):
        print("    (skipped: needs /proc -- run it on the box)")
        return
    import signal
    repo = _box(["hold.sh"])
    tag = f"{os.getpid()}.25"                              # a sleep only this test runs
    with open(os.path.join(repo, "ops", "start.sh"), "w") as fh:
        fh.write(f"#!/bin/bash\n/usr/bin/pgrep -f 'sleep {tag}' >/dev/null "
                 f"&& echo 'start.sh BESIDE-THE-COMMAND' >> $MARKERS "
                 f"|| echo 'start.sh alone' >> $MARKERS\n")
    held = os.path.join(repo, "held.sh")
    with open(held, "w") as fh:                            # a command with a child, like row3
        fh.write(f"#!/bin/bash\nsleep {tag} &\nwait\n")
    e = dict(os.environ, PATH=os.path.join(repo, "bin") + os.pathsep + os.environ["PATH"],
             MARKERS=os.path.join(repo, "markers"))
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if os.path.exists(e["MARKERS"]):
            os.remove(e["MARKERS"])
        hold = subprocess.Popen(["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--",
                                 "bash", held], env=e, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(1.5)
        hold.send_signal(sig)
        hold.wait(timeout=30)
        time.sleep(0.5)
        left = subprocess.run(["/usr/bin/pgrep", "-f", f"sleep {tag}"], capture_output=True)
        subprocess.run(["/usr/bin/pkill", "-f", f"sleep {tag}"])
        markers = open(e["MARKERS"]).read()
        assert "start.sh alone" in markers, f"{sig.name}: {markers}"
        assert left.returncode != 0, f"{sig.name}: the held command outlived its hold"
        assert not os.path.exists(os.path.join(repo, ".watchdog.off"))


def test_the_restarted_service_does_not_inherit_the_holds_descriptors():
    """A hold run as `flock LOCKFILE hold.sh ...` has the box lock on an open descriptor, and the
    service it restarts used to inherit it: on 2026-09-23 at 09:24 the restarted :8000 engine held
    the lock, and every later `flock` waited on the operator's engine. The restart runs with nothing
    above stdio open. Here the lock is fd 9 and fd 5, and the fake start.sh reports what it got."""
    repo = _box(["hold.sh"])
    start = os.path.join(repo, "ops", "start.sh")
    with open(start, "w") as fh:
        fh.write('#!/bin/bash\nfor fd in 5 9; do { true >&$fd; } 2>/dev/null && '
                 'echo "start.sh inherited fd $fd" >> "$MARKERS"; done\n'
                 'echo "start.sh ran" >> "$MARKERS"\n')
    os.chmod(start, 0o755)
    lock = os.path.join(repo, "box.flock")
    e = dict(os.environ)
    e["PATH"] = os.path.join(repo, "bin") + os.pathsep + e["PATH"]
    e["MARKERS"] = os.path.join(repo, "markers")
    out = open(os.path.join(repo, "hold.out"), "w")
    r = subprocess.run(["bash", "-c", f'exec 9>"{lock}" 5>"{lock}.2"; '
                        f'bash "{repo}/ops/hold.sh" 1 -- true'],
                       stdout=out, stderr=subprocess.STDOUT, env=e, timeout=60)
    # to a file, not a pipe: the hold's keeper loop leaves a `sleep 60` behind it holding any pipe
    assert r.returncode == 0, open(out.name).read()
    marks = open(e["MARKERS"]).read()
    assert "start.sh ran" in marks, marks
    assert "inherited" not in marks, marks


def test_a_signalled_hold_under_flock_o_restores_alone_and_inside_the_lock():
    """OPS-16 under OPS-15's `flock -o`. With -o the lock lives in flock's own process and nowhere
    below it, so the order that matters is: the held command is gone before start.sh runs, and the
    lock is still taken while start.sh runs -- flock waits on hold.sh, so a signalled hold's stop
    and restore both happen inside it. The signal goes to hold.sh, as a `kill` of the hold does."""
    if not os.path.isdir("/proc/self") or shutil.which("flock") is None:
        print("    (skipped: needs /proc and flock -- run it on the box)")
        return
    import signal
    repo = _box(["hold.sh"])
    lock = os.path.join(repo, "box.flock")
    tag = f"{os.getpid()}.35"
    with open(os.path.join(repo, "ops", "start.sh"), "w") as fh:
        fh.write(f"#!/bin/bash\n/usr/bin/pgrep -f 'sleep {tag}' >/dev/null "
                 f"&& echo 'start.sh BESIDE-THE-COMMAND' >> $MARKERS "
                 f"|| echo 'start.sh alone' >> $MARKERS\n"
                 f"flock -n {lock} true && echo 'lock FREE during the restore' >> $MARKERS "
                 f"|| echo 'lock held during the restore' >> $MARKERS\n")
    held = os.path.join(repo, "held.sh")
    with open(held, "w") as fh:
        fh.write(f"#!/bin/bash\nsleep {tag} &\nwait\n")
    e = dict(os.environ, PATH=os.path.join(repo, "bin") + os.pathsep + os.environ["PATH"],
             MARKERS=os.path.join(repo, "markers"))
    fl = subprocess.Popen(["flock", "-o", lock, "bash", os.path.join(repo, "ops", "hold.sh"), "1",
                           "--", "bash", held], env=e, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        time.sleep(1.5)
        hold = subprocess.run(["/usr/bin/pgrep", "-P", str(fl.pid)], capture_output=True, text=True)
        assert hold.stdout.split(), "hold.sh is not running under flock"
        os.kill(int(hold.stdout.split()[0]), signal.SIGTERM)
        fl.wait(timeout=30)
        time.sleep(0.5)
        left = subprocess.run(["/usr/bin/pgrep", "-f", f"sleep {tag}"], capture_output=True)
        markers = open(e["MARKERS"]).read()
        assert "start.sh alone" in markers, markers
        assert "lock held during the restore" in markers, markers
        assert left.returncode != 0, "the held command outlived its hold"
        assert subprocess.run(["flock", "-n", lock, "true"], timeout=10).returncode == 0
    finally:
        subprocess.run(["/usr/bin/pkill", "-f", f"sleep {tag}"])


def test_a_plain_flock_hold_frees_the_lock_when_it_returns():
    """The plain form, `flock LOCK hold.sh ...`, puts the lock on a descriptor every child of the
    hold inherits. The engine no longer gets it (OPS-15, both scripts), and neither may the keeper:
    `kill $KEEPER` ends the loop but not the `sleep 60` inside it, which held the lock for up to a
    minute after the hold had returned, and the next agent's hold waited on a sleep."""
    if shutil.which("flock") is None:
        print("    (skipped: needs flock -- run it on the box)")
        return
    repo = _box(["hold.sh"])
    lock = os.path.join(repo, "box.flock")
    e = dict(os.environ, PATH=os.path.join(repo, "bin") + os.pathsep + os.environ["PATH"],
             MARKERS=os.path.join(repo, "markers"))
    out = open(os.path.join(repo, "hold.out"), "w")
    r = subprocess.run(["flock", lock, "bash", os.path.join(repo, "ops", "hold.sh"), "1", "--",
                        "true"], stdout=out, stderr=subprocess.STDOUT, env=e, timeout=60)
    assert r.returncode == 0, open(out.name).read()
    free = subprocess.run(["flock", "-n", lock, "true"], timeout=10)
    assert free.returncode == 0, "the lock must be free as soon as the hold has returned"


def test_the_scripts_parse():
    for name in ("watchdog.sh", "start.sh", "stop.sh", "hold.sh"):
        r = subprocess.run(["bash", "-n", os.path.join(OPS, name)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
