"""#131: the kit's close ends a browser's process group at once, no Chrome needed.

Given the group's SIGTERM, Chrome's helpers died at once and its browser process, waiting for them in
its shutdown, sometimes stayed up until the SIGKILL 5 s later. FAKE is a browser in that shape: a
leader in its own group with two helpers, which on SIGTERM asks each helper to finish and waits for its
answer. A helper killed first never answers, so the leader waits out its own 30 s.

  * the group's SIGTERM (`_end_group` as before) keeps the leader up until the grace runs out;
  * `leader_first` ends the group well inside the grace, nothing of it left;
  * a helper that outlives its leader and ignores SIGTERM still goes, by SIGKILL after the grace;
  * Browser.close() closes the Chrome it started with `leader_first`.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "server", "laya_kit"))

import laya_drive as ld  # noqa: E402

FAKE = textwrap.dedent('''
    import os, select, signal, subprocess, sys, time
    HELPER = """
    import os, signal, sys, time
    if len(sys.argv) > 1:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)      # a helper that outlives its leader
    sys.stdout.write("up\\\\n"); sys.stdout.flush()
    sys.stdin.readline()                                   # the leader's 'finish'
    sys.stdout.write("bye\\\\n"); sys.stdout.flush()
    if len(sys.argv) > 1:
        time.sleep(60)
    """
    stubborn = sys.argv[1:] == ["stubborn"]
    kids = [subprocess.Popen([sys.executable, "-c", HELPER] + (["x"] if stubborn and i == 1 else []),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for i in range(2)]
    for k in kids:
        k.stdout.readline()
    def term(_n, _f):
        for k in kids:
            try:
                k.stdin.write("finish\\n"); k.stdin.flush()
            except OSError:
                pass
        end = time.monotonic() + 30          # Chrome waits for its helpers; a dead one never answers
        for k in kids:
            while time.monotonic() < end:
                r, _, _ = select.select([k.stdout], [], [], 0.1)
                if r and k.stdout.readline().strip() == "bye":
                    break
        os._exit(0)
    signal.signal(signal.SIGTERM, term)
    print("ready", flush=True)
    while True:
        time.sleep(1)
''')


def _start(*args):
    p = subprocess.Popen([sys.executable, "-c", FAKE, *args], stdout=subprocess.PIPE, text=True,
                         **ld._spawn_kw())
    assert p.stdout.readline().strip() == "ready"
    return p


def _gone(pg, wait: float = 3.0) -> bool:
    """The group is gone; a SIGKILLed helper that outlived its leader is reaped by init, not by us,
    so its zombie can stand for a moment after _end_group returns."""
    end = time.monotonic() + wait
    while True:
        try:
            os.killpg(pg, 0)
        except OSError:
            return True
        if time.monotonic() > end:
            return False
        time.sleep(0.02)


def test_the_groups_sigterm_waits_out_the_grace():
    p = _start()
    t0 = time.monotonic()
    ld._end_group(p, grace=1.5)
    took = time.monotonic() - t0
    assert took >= 1.5, f"{took:.2f} s: the fake browser no longer waits for its helpers"
    assert _gone(p.pid)
    return f"{took:.2f} s, ended by SIGKILL"


def test_leader_first_ends_the_group_at_once():
    times = []
    for _ in range(3):
        p = _start()
        t0 = time.monotonic()
        ld._end_group(p, grace=5.0, leader_first=True)
        times.append(time.monotonic() - t0)
        assert _gone(p.pid), "a process of the group is left"
        assert p.returncode == 0, p.returncode           # the leader ended on its SIGTERM
    assert max(times) < 1.0, times
    return ", ".join(f"{t:.2f}" for t in times) + " s"


def test_a_helper_that_outlives_its_leader_is_killed():
    p = _start("stubborn")
    t0 = time.monotonic()
    ld._end_group(p, grace=1.0, leader_first=True)
    took = time.monotonic() - t0
    assert _gone(p.pid), "the stubborn helper is left"
    assert 1.0 <= took < 4.0, took
    return f"{took:.2f} s, the helper by SIGKILL"


def test_a_group_already_gone_is_left_alone():
    p = subprocess.Popen([sys.executable, "-c", "pass"], **ld._spawn_kw())
    p.wait()
    t0 = time.monotonic()
    ld._end_group(p, leader_first=True)
    ld._end_group(None)
    assert time.monotonic() - t0 < 0.5


def test_browser_close_signals_chrome_first():
    src = inspect.getsource(ld.Browser.close)
    assert "_end_group(self.proc, leader_first=True)" in src, src
    # a game under `proc` runs behind a shell, which passes no signal on: it keeps the group's SIGTERM
    assert "leader_first" not in inspect.getsource(ld.Proc.close)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<50} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<50} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
